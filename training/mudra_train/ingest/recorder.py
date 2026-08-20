"""Load JSONL captured by the in-browser landmark recorder.

The recorder (``packages/web/recorder.html``) writes one JSON object per line:

.. code-block:: json

    {"label": "A", "signer": "aniket-01", "hand": "right",
     "condition": "normal", "timestampMs": 1234, "landmarks": [x0, y0, z0, ...]}

The ``signer`` and ``condition`` fields are not metadata trivia. ``signer`` drives the
signer-independent split without which every accuracy number is inflated; ``condition``
drives the per-slice reporting that stops a healthy aggregate score hiding a subgroup
the model fails completely.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

LANDMARKS_PER_HAND = 21


@dataclass(frozen=True, slots=True)
class Sample:
    """One recorded hand pose."""

    label: str
    signer: str
    hand: str
    condition: str
    landmarks: np.ndarray  # (21, 3)

    def __post_init__(self) -> None:
        if self.landmarks.shape != (LANDMARKS_PER_HAND, 3):
            raise ValueError(f"landmarks must be (21, 3), got {self.landmarks.shape}")


@dataclass(frozen=True, slots=True)
class Dataset:
    """A collection of samples with the arrays training needs."""

    landmarks: np.ndarray  # (n, 21, 3)
    labels: np.ndarray  # (n,) str
    signers: np.ndarray  # (n,) str
    hands: np.ndarray  # (n,) str
    conditions: np.ndarray  # (n,) str

    def __len__(self) -> int:
        return int(self.landmarks.shape[0])

    @property
    def classes(self) -> list[str]:
        """Sorted unique labels, with ``none`` forced last if present.

        Keeping ``none`` at a stable index means a pack manifest's label order
        survives adding a new letter.
        """
        unique = sorted(set(self.labels.tolist()))
        if "none" in unique:
            unique.remove("none")
            unique.append("none")
        return unique

    def summary(self) -> str:
        lines = [f"{len(self)} samples, {len(self.classes)} classes"]
        lines.append(f"  signers:    {sorted(set(self.signers.tolist()))}")
        lines.append(f"  conditions: {sorted(set(self.conditions.tolist()))}")
        lines.append(f"  hands:      {sorted(set(self.hands.tolist()))}")
        return "\n".join(lines)


def read_jsonl(path: Path) -> Iterator[Sample]:
    """Yield samples from one JSONL file, skipping blank lines."""
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from error

            flat = raw.get("landmarks")
            if not isinstance(flat, list) or len(flat) != LANDMARKS_PER_HAND * 3:
                raise ValueError(
                    f"{path}:{line_number} expects {LANDMARKS_PER_HAND * 3} landmark "
                    f"values, got {len(flat) if isinstance(flat, list) else 'none'}"
                )

            yield Sample(
                label=str(raw["label"]),
                signer=str(raw.get("signer", "unknown")),
                hand=str(raw.get("hand", "right")),
                condition=str(raw.get("condition", "unknown")),
                landmarks=np.asarray(flat, dtype=np.float64).reshape(LANDMARKS_PER_HAND, 3),
            )


def load(paths: Iterable[Path]) -> Dataset:
    """Load and concatenate every JSONL file in ``paths``."""
    samples: list[Sample] = []
    for path in paths:
        samples.extend(read_jsonl(path))

    if not samples:
        raise ValueError("No samples found. Record some with packages/web/recorder.html.")

    return from_samples(samples)


def load_directory(directory: Path) -> Dataset:
    """Load every ``*.jsonl`` under ``directory``, recursively."""
    return load(sorted(directory.rglob("*.jsonl")))


def from_samples(samples: list[Sample]) -> Dataset:
    """Build a :class:`Dataset` from a list of samples."""
    return Dataset(
        landmarks=np.stack([s.landmarks for s in samples]),
        labels=np.array([s.label for s in samples]),
        signers=np.array([s.signer for s in samples]),
        hands=np.array([s.hand for s in samples]),
        conditions=np.array([s.condition for s in samples]),
    )
