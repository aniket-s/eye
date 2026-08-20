"""Train the continuous fingerspelling model.

.. code-block:: bash

    # Validate the pipeline on procedural sequences (NOT a usable model):
    python -m mudra_train.train_ctc --synthetic --out ../models/fingerspell-ctc

Scored with **character error rate**, not per-frame accuracy. CER is edit distance
against the reference spelling, so it penalises misrecognition, double-firing and
misses in a single number. Per-frame accuracy penalises none of them — a decoder that
emits every letter twice can still look excellent frame by frame.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

# Before anything third-party, so a missing dependency reports itself clearly
# rather than as a ModuleNotFoundError from the middle of this file.
from .preflight import require_dependencies

require_dependencies()

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from .eval.metrics import split_by_signer
from .export.ctc_pack import build_ctc_pack
from .features.augment import AugmentConfig, augment_batch
from .features.normalise import FEATURE_LENGTH, normalise_batch
from .ingest import sequences as sequence_data
from .ingest.sequences import SequenceDataset
from .models.temporal import FingerspellCtc, ctc_loss, greedy_decode


def levenshtein(reference: list[int], hypothesis: list[int]) -> int:
    """Edit distance between two label sequences."""
    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)

    previous = list(range(len(hypothesis) + 1))
    for i, ref in enumerate(reference, start=1):
        current = [i]
        for j, hyp in enumerate(hypothesis, start=1):
            current.append(
                min(previous[j - 1] + (ref != hyp), previous[j] + 1, current[j - 1] + 1)
            )
        previous = current
    return previous[-1]


def encode(dataset: SequenceDataset) -> tuple[list[list[int]], list[str]]:
    """Map letters to class indices, with blank reserved at 0."""
    classes = dataset.classes
    lookup = {label: index for index, label in enumerate(classes)}
    return [[lookup[letter] for letter in label] for label in dataset.labels], classes


def make_batch(
    dataset: SequenceDataset,
    targets: list[list[int]],
    indices: np.ndarray,
    rng: np.random.Generator,
    augment: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a padded batch.

    Padding is zeros and ``input_lengths`` records the true length, so CTC never scores
    the padding. Getting this wrong produces a model that learns to emit blanks.
    """
    frames = [len(dataset.sequences[i]) for i in indices]
    longest = max(frames)

    features = np.zeros((len(indices), longest, FEATURE_LENGTH), dtype=np.float32)
    for slot, index in enumerate(indices):
        sequence = dataset.sequences[index]
        hands = np.array([dataset.hands[index]] * len(sequence))

        if augment:
            # Augment the whole sequence with one draw, so a hand does not appear to
            # teleport between consecutive frames.
            sequence, hands = augment_batch(sequence, hands, rng, _SEQUENCE_AUGMENT)

        features[slot, : len(sequence)] = normalise_batch(sequence, hands)

    flat_targets = torch.tensor([t for index in indices for t in targets[index]], dtype=torch.long)
    target_lengths = torch.tensor([len(targets[index]) for index in indices], dtype=torch.long)
    return (
        torch.from_numpy(features),
        flat_targets,
        torch.tensor(frames, dtype=torch.long),
        target_lengths,
    )


#: Gentler than the static config: strong per-frame jitter would break the temporal
#: continuity the model is trying to learn.
_SEQUENCE_AUGMENT = AugmentConfig(
    mirror_probability=0.5,
    max_rotation_degrees=20.0,
    rotation_probability=0.7,
    scale_range=(0.9, 1.1),
    jitter_std=0.01,
    dropout_probability=0.1,
    max_dropout=1,
)


def evaluate_cer(
    model: FingerspellCtc,
    dataset: SequenceDataset,
    targets: list[list[int]],
    indices: np.ndarray,
    device: torch.device,
    batch_size: int = 32,
) -> tuple[float, float]:
    """Return ``(character_error_rate, exact_match_rate)``."""
    model.eval()
    total_distance = 0
    total_length = 0
    exact = 0

    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch = indices[start : start + batch_size]
            features, _, input_lengths, _ = make_batch(
                dataset, targets, batch, np.random.default_rng(0), augment=False
            )
            log_probabilities = model(features.to(device))

            for slot, index in enumerate(batch):
                length = int(input_lengths[slot])
                decoded = greedy_decode(log_probabilities[slot : slot + 1, :length])[0]
                reference = targets[index]
                total_distance += levenshtein(reference, decoded)
                total_length += len(reference)
                exact += int(decoded == reference)

    cer = total_distance / max(total_length, 1)
    return cer, exact / max(len(indices), 1)


def train(
    dataset: SequenceDataset,
    output: Path,
    *,
    epochs: int = 60,
    batch_size: int = 32,
    learning_rate: float = 2e-3,
    seed: int = 20260820,
    synthetic_data: bool = False,
    quiet: bool = False,
) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    targets, classes = encode(dataset)
    train_indices, test_indices, held_out = split_by_signer(dataset.signers, seed=seed)

    if not quiet:
        print(dataset.summary())
        print(f"  train {len(train_indices)} / test {len(test_indices)}")
        print(f"  held-out signers: {', '.join(held_out)}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FingerspellCtc(classes=len(classes), feature_length=FEATURE_LENGTH).to(device)
    if not quiet:
        print(f"Model: {model.parameter_count:,} parameters on {device}\n")

    optimiser = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    steps_per_epoch = max(1, len(train_indices) // batch_size)
    scheduler = OneCycleLR(
        optimiser, max_lr=learning_rate, total_steps=epochs * steps_per_epoch, pct_start=0.3
    )

    started = time.time()
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(train_indices)
        total_loss = 0.0
        batches = 0

        for start in range(0, len(order) - batch_size + 1, batch_size):
            batch = order[start : start + batch_size]
            features, flat_targets, input_lengths, target_lengths = make_batch(
                dataset, targets, batch, rng, augment=True
            )

            log_probabilities = model(features.to(device))
            loss = ctc_loss(
                log_probabilities, flat_targets.to(device), input_lengths, target_lengths
            )

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            scheduler.step()

            total_loss += float(loss.item())
            batches += 1

        if not quiet and (epoch % 10 == 0 or epoch == epochs - 1):
            cer, exact = evaluate_cer(model, dataset, targets, test_indices, device)
            print(
                f"  epoch {epoch + 1:>3}/{epochs}  loss {total_loss / max(batches, 1):.4f}"
                f"  CER {cer:.3f}  exact {exact:.1%}"
            )

    cer, exact = evaluate_cer(model, dataset, targets, test_indices, device)
    if not quiet:
        print(f"\nTrained in {time.time() - started:.1f}s")
        print("═" * 60)
        print("EVALUATION — signer-independent, continuous decoding")
        print("═" * 60)
        print(f"  character error rate  {cer:.4f}")
        print(f"  exact-match rate      {exact:.1%}")
        print(f"  held out              {', '.join(held_out)}")
        print("═" * 60)

    pack = build_ctc_pack(
        model=model,
        classes=classes,
        cer=cer,
        exact_match=exact,
        test_signers=held_out,
        output=output,
        synthetic=synthetic_data,
    )
    if not quiet:
        print(f"\nWrote model pack to {output}")
        print(f"  model.onnx   {pack['sizeBytes'] / 1024:.0f} KB int8")
        print(f"  export delta {pack['onnxDelta']:.2e}")
    return pack


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--synthetic",
        action="store_true",
        help="Procedural sequences to validate the pipeline (NOT a usable model)",
    )
    source.add_argument(
        "--fsboard",
        type=Path,
        help="Extracted FSboard directory — 3M+ characters from 147 Deaf signers",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Use only this many clips. Worth trying before committing to a full run.",
    )
    args = parser.parse_args()

    if args.fsboard is not None:
        from .ingest.fsboard import load_fsboard

        dataset = load_fsboard(args.fsboard, limit=args.limit)
        characters = sum(len(label) for label in dataset.labels)
        signers = len(set(dataset.signers.tolist()))
        print(
            f"FSboard: {len(dataset):,} clips, {characters:,} characters, {signers} signers.\n"
            "  Real recordings of real people — the numbers this produces mean something.\n"
        )
        synthetic = False
    else:
        print("⚠  SYNTHETIC SEQUENCES — validates the pipeline, produces an unusable model.\n")
        dataset = sequence_data.generate()
        synthetic = True

    train(
        dataset,
        args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        synthetic_data=synthetic,
    )


if __name__ == "__main__":
    main()
