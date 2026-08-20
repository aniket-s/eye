"""Train a static handshape classifier and write a model pack.

Usage
-----

.. code-block:: bash

    # Prove the pipeline works, using procedural fixtures (NOT a usable model):
    python -m mudra_train.train --synthetic --out ../models/synthetic

    # Train on your own recordings:
    python -m mudra_train.train --data ./recordings --out ../models/asl-fingerspell

The split is always signer-independent, and training refuses to start on a
single-signer dataset. That is not pedantry: a model evaluated on the person it was
trained on reports a number that means nothing, which is the flaw the v1 app shipped.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from .eval.metrics import evaluate, format_report, split_by_signer
from .export.pack import build_pack
from .features.augment import AugmentConfig, augment_batch
from .features.normalise import FEATURE_LENGTH, normalise_batch
from .ingest import synthetic
from .ingest.recorder import Dataset, load_directory
from .models.classifier import HandshapeClassifier


def encode_labels(labels: np.ndarray, classes: list[str]) -> np.ndarray:
    lookup = {label: index for index, label in enumerate(classes)}
    return np.array([lookup[label] for label in labels], dtype=np.int64)


def build_slices(dataset: Dataset, indices: np.ndarray) -> dict[str, np.ndarray]:
    """Boolean masks for per-slice reporting.

    Handedness and recording condition are the two axes where a model most often
    looks fine overall and fails completely for a subgroup.
    """
    slices: dict[str, np.ndarray] = {}
    for hand in sorted(set(dataset.hands[indices].tolist())):
        slices[f"hand={hand}"] = dataset.hands[indices] == hand
    for condition in sorted(set(dataset.conditions[indices].tolist())):
        slices[f"condition={condition}"] = dataset.conditions[indices] == condition
    return slices


def train(
    dataset: Dataset,
    output: Path,
    *,
    epochs: int = 60,
    batch_size: int = 256,
    learning_rate: float = 3e-3,
    weight_decay: float = 0.05,
    label_smoothing: float = 0.1,
    seed: int = 20260820,
    synthetic_data: bool = False,
    quiet: bool = False,
) -> dict:
    """Train, evaluate, and write a model pack. Returns the metrics summary."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    classes = dataset.classes
    targets = encode_labels(dataset.labels, classes)

    train_indices, test_indices, held_out = split_by_signer(dataset.signers, seed=seed)
    if not quiet:
        print(dataset.summary())
        print(f"  train {len(train_indices)} / test {len(test_indices)}")
        print(f"  held-out signers: {', '.join(held_out)}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HandshapeClassifier(classes=len(classes), feature_length=FEATURE_LENGTH).to(device)
    if not quiet:
        print(f"Model: {model.parameter_count:,} parameters on {device}\n")

    optimiser = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    steps_per_epoch = max(1, len(train_indices) // batch_size)
    scheduler = OneCycleLR(
        optimiser, max_lr=learning_rate, total_steps=epochs * steps_per_epoch, pct_start=0.25
    )
    augment_config = AugmentConfig()

    train_landmarks = dataset.landmarks[train_indices]
    train_hands = dataset.hands[train_indices]
    train_targets = targets[train_indices]

    started = time.time()
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(train_indices))
        total_loss = 0.0
        batches = 0

        for start in range(0, len(order) - batch_size + 1, batch_size):
            batch = order[start : start + batch_size]

            # Augment raw landmarks, then normalise — the same order inference uses.
            points, hands = augment_batch(
                train_landmarks[batch], train_hands[batch], rng, augment_config
            )
            features = torch.from_numpy(normalise_batch(points, hands)).to(device)
            labels = torch.from_numpy(train_targets[batch]).to(device)

            logits = model(features, labels)
            loss = F.cross_entropy(logits, labels, label_smoothing=label_smoothing)

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            scheduler.step()

            total_loss += float(loss.item())
            batches += 1

        if not quiet and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch + 1:>3}/{epochs}  loss {total_loss / max(batches, 1):.4f}")

    if not quiet:
        print(f"\nTrained in {time.time() - started:.1f}s\n")

    # ── Evaluate on held-out signers, without augmentation ──
    model.eval()
    test_features = torch.from_numpy(
        normalise_batch(dataset.landmarks[test_indices], dataset.hands[test_indices])
    ).to(device)

    with torch.no_grad():
        logits = model(test_features)
        probabilities = F.softmax(logits, dim=-1).cpu().numpy()

    predicted = probabilities.argmax(axis=1)
    report = evaluate(
        true_indices=targets[test_indices],
        predicted_indices=predicted,
        probabilities=probabilities,
        labels=classes,
        slices=build_slices(dataset, test_indices),
        test_signers=held_out,
    )

    if not quiet:
        print(format_report(report))

    pack = build_pack(
        model=model,
        classes=classes,
        report=report,
        output=output,
        synthetic=synthetic_data,
    )
    if not quiet:
        size_kb = pack["sizeBytes"] / 1024
        saved = 1 - pack["sizeBytes"] / pack["floatSizeBytes"]
        print(f"\nWrote model pack to {output}")
        print(f"  model.onnx   {size_kb:.0f} KB int8 ({saved:.0%} smaller than fp32)")
        print(f"  export delta {pack['onnxDelta']:.2e} fp32 / {pack['quantisedDelta']:.2e} int8")

    return pack


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", type=Path, help="Directory of recorder JSONL files")
    source.add_argument(
        "--synthetic",
        action="store_true",
        help="Use procedural fixtures to validate the pipeline (NOT a usable model)",
    )
    parser.add_argument("--out", type=Path, required=True, help="Model pack output directory")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260820)
    args = parser.parse_args()

    if args.synthetic:
        print("⚠  SYNTHETIC DATA — validates the pipeline, produces an unusable model.\n")
        dataset = synthetic.generate()
    else:
        dataset = load_directory(args.data)

    train(
        dataset,
        args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        synthetic_data=bool(args.synthetic),
    )


if __name__ == "__main__":
    main()
