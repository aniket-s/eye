"""Fingerspelled sequences — **for pipeline validation only**.

.. warning::

   Like :mod:`mudra_train.ingest.synthetic`, this produces procedurally generated
   geometry, not recordings of real hands. It exists so the CTC pipeline can be
   trained, evaluated, exported and streamed end to end without waiting on a dataset.
   A model fitted on it will not read real fingerspelling.

Real sequence data comes from **FSboard** (3M+ characters, 147 Deaf signers, CC BY 4.0)
or from recording continuous spelling yourself.

What makes this useful as a fixture is that it reproduces the two properties that make
continuous fingerspelling hard, and that a static classifier cannot handle:

* **Transitions.** Between two letters the hand passes through shapes belonging to
  neither. Those frames are labelled blank, which is what teaches the model to emit
  nothing rather than guess.
* **Variable dwell.** Real signers hold letters for different lengths, and a doubled
  letter is two brief holds separated by a transition — the case a dwell timer cannot
  express at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .synthetic import _HANDSHAPES, _build_hand
from .recorder import LANDMARKS_PER_HAND


@dataclass(frozen=True, slots=True)
class SequenceDataset:
    """Variable-length landmark sequences with their spelled labels."""

    sequences: list[np.ndarray]  # each (frames, 21, 3)
    labels: list[list[str]]  # each the letters spelled
    signers: np.ndarray
    hands: np.ndarray

    def __len__(self) -> int:
        return len(self.sequences)

    @property
    def classes(self) -> list[str]:
        """Alphabet with the CTC blank first, matching ``BLANK_INDEX = 0``."""
        letters = sorted({letter for label in self.labels for letter in label})
        return ["", *letters]

    def summary(self) -> str:
        lengths = [len(s) for s in self.sequences]
        spelled = [len(label) for label in self.labels]
        return (
            f"{len(self)} sequences, {len(self.classes) - 1} letters\n"
            f"  frames:  {min(lengths)}–{max(lengths)} (mean {np.mean(lengths):.0f})\n"
            f"  letters: {min(spelled)}–{max(spelled)} per sequence\n"
            f"  signers: {sorted(set(self.signers.tolist()))}"
        )


#: Words worth spelling: doubled letters, and letters that differ only by orientation.
_WORDS = [
    "HELLO",
    "YES",
    "NO",
    "PLEASE",
    "SORRY",
    "GOOD",
    "NAME",
    "HELP",
    "WATER",
    "BOOK",
    "COFFEE",  # doubled F
    "LETTER",  # doubled T
    "PUPPY",  # P and Y
    "QUIET",  # Q, which differs from G only by rotation
    "KIND",  # K, which differs from P only by rotation
]


def _interpolate(start: np.ndarray, end: np.ndarray, steps: int) -> list[np.ndarray]:
    """Frames between two handshapes.

    Linear interpolation is crude but captures the property that matters: transition
    frames belong to no letter, and are labelled blank.
    """
    return [start * (1 - t) + end * t for t in np.linspace(0, 1, steps + 2)[1:-1]]


def generate(
    sequences_per_signer: int = 40,
    signers: int = 6,
    seed: int = 20260820,
    jitter: float = 0.012,
) -> SequenceDataset:
    """Generate fingerspelled sequences."""
    rng = np.random.default_rng(seed)

    all_sequences: list[np.ndarray] = []
    all_labels: list[list[str]] = []
    signer_ids: list[str] = []
    hand_ids: list[str] = []

    for signer in range(signers):
        signer_scale = float(rng.uniform(0.85, 1.15))
        signer_tilt = float(rng.uniform(-12.0, 12.0))
        signer_jitter = jitter * float(rng.uniform(0.6, 1.6))
        signer_name = f"synthetic-{signer:02d}"
        # Signers spell at different speeds.
        signer_pace = float(rng.uniform(0.7, 1.4))

        for _ in range(sequences_per_signer):
            word = _WORDS[int(rng.integers(len(_WORDS)))]
            letters = [c for c in word if c in _HANDSHAPES]
            if not letters:
                continue

            hand = "left" if rng.random() < 0.3 else "right"
            frames: list[np.ndarray] = []
            previous_shape: np.ndarray | None = None

            for letter in letters:
                curls, thumb, rotation = _HANDSHAPES[letter]
                shape = _build_hand(
                    curls,
                    thumb,
                    rotation + signer_tilt + float(rng.normal(0, 5.0)),
                    rng,
                    0.0,
                )

                if previous_shape is not None:
                    transition = max(2, int(round(rng.integers(3, 7) * signer_pace)))
                    frames.extend(_interpolate(previous_shape, shape, transition))

                hold = max(2, int(round(rng.integers(4, 10) * signer_pace)))
                frames.extend(shape.copy() for _ in range(hold))
                previous_shape = shape

            sequence = np.stack(frames)
            sequence[:, :, :2] *= signer_scale
            sequence[:, :, :2] += rng.normal(
                0.0, signer_jitter, size=(len(sequence), LANDMARKS_PER_HAND, 2)
            )
            if hand == "left":
                sequence[:, :, 0] = -sequence[:, :, 0]

            all_sequences.append(sequence)
            all_labels.append(letters)
            signer_ids.append(signer_name)
            hand_ids.append(hand)

    return SequenceDataset(
        sequences=all_sequences,
        labels=all_labels,
        signers=np.array(signer_ids),
        hands=np.array(hand_ids),
    )
