"""Train the isolated word-sign classifier.

.. code-block:: bash

    # Validate the pipeline on procedural clips (NOT a usable model):
    python -m mudra_train.train_words --synthetic --out ../models/asl-words

    # Train on PopSign or GISLR landmark parquet:
    python -m mudra_train.train_words \\
        --index /data/popsign/train.csv --data-root /data/popsign --out ../models/asl-words

Scored with top-1 and top-5 accuracy on held-out signers. Top-5 matters here in a way
it does not for fingerspelling: with 250 visually similar classes, a UI that offers
runners-up is genuinely useful, so knowing the correct sign is usually *somewhere* in
the shortlist is worth measuring.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from .eval.metrics import evaluate, format_report, split_by_signer
from .export.isolated_pack import build_isolated_pack
from .features.normalise_body import BODY_FEATURE_LENGTH, normalise_body_sequence
from .ingest import words as word_data
from .ingest.words import WordDataset
from .models.isolated import IsolatedSignClassifier

#: Frames each clip is resampled to. Must match `maxFrames` in the browser recogniser.
CLIP_FRAMES = 96


def resample(features: np.ndarray, frames: int = CLIP_FRAMES) -> np.ndarray:
    """Resample a clip to a fixed length by nearest-frame indexing.

    Uniform resampling rather than truncation: cutting a clip short would discard its
    ending, and in many signs the final movement is what distinguishes it.
    """
    if len(features) == frames:
        return features
    indices = np.round(np.linspace(0, len(features) - 1, frames)).astype(int)
    return features[indices]


def prepare(dataset: WordDataset) -> np.ndarray:
    """Normalise and resample every clip to ``(n, CLIP_FRAMES, BODY_FEATURE_LENGTH)``."""
    out = np.zeros((len(dataset), CLIP_FRAMES, BODY_FEATURE_LENGTH), dtype=np.float32)
    for i in range(len(dataset)):
        features = normalise_body_sequence(
            dataset.dominant_hands[i],
            dataset.other_hands[i],
            dataset.poses[i],
            dataset.dominants[i],
        )
        out[i] = resample(features)
    return out


def augment(batch: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Augment normalised clips.

    Applied after normalisation, unlike the fingerspelling pipeline, because these
    features are already in a body-relative frame where the meaningful perturbations
    are simple: scale (signer build), translation (camera framing), and time warp
    (signing pace). Mirroring is *not* applied here — handedness is canonicalised
    during normalisation, and mirroring afterwards would swap the hand slots without
    swapping the labels.
    """
    out = batch.copy()
    count = len(out)

    # Coordinate channels only; the two presence flags at the end must stay 0 or 1.
    spatial = slice(0, BODY_FEATURE_LENGTH - 2)

    scales = rng.uniform(0.88, 1.12, size=(count, 1, 1))
    out[:, :, spatial] *= scales

    shifts = rng.normal(0.0, 0.05, size=(count, 1, 1))
    out[:, :, spatial] += shifts

    noise = rng.normal(0.0, 0.012, size=out[:, :, spatial].shape)
    out[:, :, spatial] += noise

    # Time warp: resample a random sub-window, standing in for a faster or slower signer.
    for i in range(count):
        if rng.random() < 0.5:
            span = rng.uniform(0.75, 1.0)
            start = rng.uniform(0, 1 - span)
            source = np.linspace(start, start + span, CLIP_FRAMES) * (CLIP_FRAMES - 1)
            out[i] = out[i][np.round(source).astype(int)]

    return out


def train(
    dataset: WordDataset,
    output: Path,
    *,
    epochs: int = 80,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    label_smoothing: float = 0.1,
    seed: int = 20260820,
    synthetic_data: bool = False,
    quiet: bool = False,
) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    classes = dataset.classes
    lookup = {label: index for index, label in enumerate(classes)}
    targets = np.array([lookup[label] for label in dataset.labels], dtype=np.int64)

    if not quiet:
        print(dataset.summary())
    features = prepare(dataset)

    train_indices, test_indices, held_out = split_by_signer(dataset.signers, seed=seed)
    if not quiet:
        print(f"  train {len(train_indices)} / test {len(test_indices)}")
        print(f"  held-out signers: {', '.join(held_out)}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IsolatedSignClassifier(
        classes=len(classes), feature_length=BODY_FEATURE_LENGTH
    ).to(device)
    if not quiet:
        print(f"Model: {model.parameter_count:,} parameters on {device}\n")

    optimiser = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.05)
    steps_per_epoch = max(1, len(train_indices) // batch_size)
    scheduler = OneCycleLR(
        optimiser, max_lr=learning_rate, total_steps=epochs * steps_per_epoch, pct_start=0.25
    )

    started = time.time()
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(train_indices)
        total_loss = 0.0
        batches = 0

        for start in range(0, len(order) - batch_size + 1, batch_size):
            batch = order[start : start + batch_size]
            inputs = torch.from_numpy(augment(features[batch], rng)).to(device)
            labels = torch.from_numpy(targets[batch]).to(device)

            logits = model(inputs)
            loss = F.cross_entropy(logits, labels, label_smoothing=label_smoothing)

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            scheduler.step()

            total_loss += float(loss.item())
            batches += 1

        if not quiet and (epoch % 20 == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch + 1:>3}/{epochs}  loss {total_loss / max(batches, 1):.4f}")

    if not quiet:
        print(f"\nTrained in {time.time() - started:.1f}s\n")

    # ── Evaluate ──
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(features[test_indices]).to(device))
        probabilities = F.softmax(logits, dim=-1).cpu().numpy()

    predicted = probabilities.argmax(axis=1)
    truth = targets[test_indices]

    top5 = float(
        np.mean(
            [
                truth[i] in np.argsort(probabilities[i])[-5:]
                for i in range(len(truth))
            ]
        )
    )

    slices = {
        f"dominant={hand}": dataset.dominants[test_indices] == hand
        for hand in sorted(set(dataset.dominants[test_indices].tolist()))
    }
    report = evaluate(truth, predicted, probabilities, classes, slices, held_out)

    if not quiet:
        print(format_report(report))
        print(f"\n  top-5 accuracy  {top5:.4f}")

    pack = build_isolated_pack(
        model=model,
        classes=classes,
        report=report,
        top5=top5,
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
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--index", type=Path, help="GISLR/PopSign train.csv")
    parser.add_argument("--data-root", type=Path, help="Directory the index paths are relative to")
    parser.add_argument("--limit", type=int, help="Read at most this many clips")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    if args.synthetic:
        print("⚠  SYNTHETIC CLIPS — validates the pipeline, produces an unusable model.\n")
        dataset = word_data.generate_synthetic()
    elif args.index is not None and args.data_root is not None:
        dataset = word_data.load_gislr(args.index, args.data_root, limit=args.limit)
    else:
        parser.error("provide either --synthetic, or both --index and --data-root")

    train(
        dataset,
        args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        synthetic_data=bool(args.synthetic),
    )


if __name__ == "__main__":
    main()
