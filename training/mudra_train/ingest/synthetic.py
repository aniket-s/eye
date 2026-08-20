"""Procedural handshapes — **for pipeline validation only**.

.. warning::

   This is **not training data** and must never be shipped as a model pack. It is a
   kinematic approximation of ASL handshapes used to prove the pipeline runs
   end-to-end — ingest, augment, train, evaluate, export, quantise, and load in the
   browser — without waiting on a dataset download.

   A model fitted on this will score well here and fail completely on a real webcam,
   because it has learned an idealised geometry no hand actually produces. Every
   artefact built from it is tagged ``synthetic`` so it cannot be mistaken for a real
   pack.

Real training data comes from two places:

* **FSboard** — 3M+ characters, 147 Deaf signers, CC BY 4.0 (needs a Kaggle account).
* **Your own recordings** — ``packages/web/recorder.html``, which closes the gap
  between public datasets and your actual users' cameras.

The value of this module is that the pipeline is *tested*, so the first real training
run exercises code that has already been proven to work.
"""

from __future__ import annotations

import numpy as np

from .recorder import Dataset, LANDMARKS_PER_HAND

#: Per-letter finger curl, 0.0 = fully extended, 1.0 = fully curled.
#: Order: index, middle, ring, pinky. Plus thumb tuck and hand rotation in degrees.
#:
#: Approximated from the standard ASL manual alphabet. Note the pairs that differ
#: only by rotation — K/P, G/Q — which is exactly why the normaliser preserves
#: orientation rather than canonicalising it away.
_HANDSHAPES: dict[str, tuple[tuple[float, float, float, float], float, float]] = {
    "A": ((1.0, 1.0, 1.0, 1.0), 0.0, 0.0),
    "B": ((0.0, 0.0, 0.0, 0.0), 1.0, 0.0),
    "C": ((0.5, 0.5, 0.5, 0.5), 0.3, 0.0),
    "D": ((0.0, 1.0, 1.0, 1.0), 0.5, 0.0),
    "E": ((0.8, 0.8, 0.8, 0.8), 0.9, 0.0),
    "F": ((0.9, 0.0, 0.0, 0.0), 0.9, 0.0),
    "G": ((0.0, 1.0, 1.0, 1.0), 0.0, 90.0),
    "H": ((0.0, 0.0, 1.0, 1.0), 0.5, 90.0),
    "I": ((1.0, 1.0, 1.0, 0.0), 0.5, 0.0),
    "K": ((0.0, 0.0, 1.0, 1.0), 0.0, 0.0),
    "L": ((0.0, 1.0, 1.0, 1.0), 0.0, 0.0),
    "M": ((1.0, 1.0, 1.0, 1.0), 0.7, 0.0),
    "N": ((1.0, 1.0, 1.0, 1.0), 0.6, 0.0),
    "O": ((0.7, 0.7, 0.7, 0.7), 0.7, 0.0),
    "P": ((0.0, 0.0, 1.0, 1.0), 0.0, 180.0),
    "Q": ((0.0, 1.0, 1.0, 1.0), 0.0, 180.0),
    "R": ((0.0, 0.0, 1.0, 1.0), 0.8, 0.0),
    "S": ((1.0, 1.0, 1.0, 1.0), 0.5, 0.0),
    "T": ((1.0, 1.0, 1.0, 1.0), 0.4, 0.0),
    "U": ((0.0, 0.0, 1.0, 1.0), 0.9, 0.0),
    "V": ((0.0, 0.0, 1.0, 1.0), 0.9, 15.0),
    "W": ((0.0, 0.0, 0.0, 1.0), 0.9, 0.0),
    "X": ((0.5, 1.0, 1.0, 1.0), 0.6, 0.0),
    "Y": ((1.0, 1.0, 1.0, 0.0), 0.0, 0.0),
}

#: Fraction of generated samples that are ``none`` — hands doing nothing in
#: particular. Without a negative class every transitional pose is force-classified.
NONE_FRACTION = 0.15

# Finger geometry in palm units: MCP position along the knuckle line, and segment
# lengths. Deliberately crude — this is a fixture generator, not a hand model.
_FINGER_BASE_X = (-0.30, -0.10, 0.10, 0.28)
_FINGER_LENGTHS = ((0.34, 0.22, 0.16), (0.38, 0.24, 0.17), (0.34, 0.22, 0.16), (0.26, 0.17, 0.13))


def _build_hand(
    curls: tuple[float, float, float, float],
    thumb_tuck: float,
    rotation_degrees: float,
    rng: np.random.Generator,
    jitter: float,
) -> np.ndarray:
    """Build 21 landmarks from finger curls."""
    points = np.zeros((LANDMARKS_PER_HAND, 3), dtype=np.float64)

    # Wrist at the origin; the palm extends upward (negative y, matching image space).
    points[0] = (0.0, 0.0, 0.0)

    # Thumb: swings from out to the side (extended) to across the palm (tucked).
    thumb_angle = np.deg2rad(200.0 + 60.0 * thumb_tuck)
    for segment in range(4):
        reach = 0.16 * segment
        points[segment] = (
            reach * np.cos(thumb_angle) - 0.18 * min(segment, 1),
            reach * np.sin(thumb_angle) - 0.22,
            0.0,
        )

    # Four fingers: index 5-8, middle 9-12, ring 13-16, pinky 17-20.
    for finger in range(4):
        base_index = 5 + finger * 4
        base_x = _FINGER_BASE_X[finger]
        base_y = -0.55
        points[base_index] = (base_x, base_y, 0.0)

        curl = curls[finger]
        x, y = base_x, base_y
        # Each successive joint bends further, so a curled finger folds toward the palm.
        angle = -np.pi / 2
        for segment, length in enumerate(_FINGER_LENGTHS[finger]):
            angle += curl * (np.pi / 2.6)
            x += length * np.cos(angle - np.pi / 2) * 0.35
            y += length * np.sin(angle)
            points[base_index + segment + 1] = (x, y, 0.0)

    if rotation_degrees:
        theta = np.deg2rad(rotation_degrees)
        rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        points[:, :2] = points[:, :2] @ rotation.T

    if jitter:
        points[:, :2] += rng.normal(0.0, jitter, size=(LANDMARKS_PER_HAND, 2))

    return points


def generate(
    samples_per_class: int = 120,
    signers: int = 6,
    seed: int = 20260820,
    jitter: float = 0.012,
) -> Dataset:
    """Generate a synthetic dataset.

    Each simulated signer gets a slightly different hand geometry and jitter level, so
    a signer-independent split is a genuine test of generalisation rather than a
    formality.
    """
    rng = np.random.default_rng(seed)
    conditions = ("normal", "near", "far", "angled", "dim")

    landmarks: list[np.ndarray] = []
    labels: list[str] = []
    signer_ids: list[str] = []
    hands: list[str] = []
    condition_ids: list[str] = []

    for signer in range(signers):
        # Per-signer variation: hand proportions, baseline tilt, and noise level.
        signer_scale = float(rng.uniform(0.85, 1.15))
        signer_tilt = float(rng.uniform(-12.0, 12.0))
        signer_jitter = jitter * float(rng.uniform(0.6, 1.6))
        signer_name = f"synthetic-{signer:02d}"

        for label, (curls, thumb, rotation) in _HANDSHAPES.items():
            for _ in range(samples_per_class // signers):
                hand = "left" if rng.random() < 0.3 else "right"
                condition = conditions[int(rng.integers(len(conditions)))]

                wobble = rng.normal(0.0, 0.05, size=4)
                noisy_curls = tuple(float(np.clip(c + w, 0.0, 1.0)) for c, w in zip(curls, wobble))

                points = _build_hand(
                    noisy_curls,  # type: ignore[arg-type]
                    float(np.clip(thumb + rng.normal(0, 0.06), 0.0, 1.0)),
                    rotation + signer_tilt + float(rng.normal(0, 6.0)),
                    rng,
                    signer_jitter,
                )
                points[:, :2] *= signer_scale
                if hand == "left":
                    points[:, 0] = -points[:, 0]

                landmarks.append(points)
                labels.append(label)
                signer_ids.append(signer_name)
                hands.append(hand)
                condition_ids.append(condition)

        # Negatives: random curls that do not correspond to any letter, plus
        # interpolations between two letters — which is geometrically what a
        # transition between signs looks like, and the case v1 could not reject.
        none_count = int(len(_HANDSHAPES) * (samples_per_class // signers) * NONE_FRACTION)
        shape_list = list(_HANDSHAPES.values())
        for _ in range(none_count):
            first = shape_list[int(rng.integers(len(shape_list)))]
            second = shape_list[int(rng.integers(len(shape_list)))]
            blend = float(rng.uniform(0.35, 0.65))
            mixed = tuple(
                float(a * (1 - blend) + b * blend) for a, b in zip(first[0], second[0])
            )
            points = _build_hand(
                mixed,  # type: ignore[arg-type]
                float(first[1] * (1 - blend) + second[1] * blend),
                float(rng.uniform(-180, 180)),
                rng,
                signer_jitter * 1.5,
            )
            points[:, :2] *= signer_scale
            hand = "left" if rng.random() < 0.3 else "right"
            if hand == "left":
                points[:, 0] = -points[:, 0]

            landmarks.append(points)
            labels.append("none")
            signer_ids.append(signer_name)
            hands.append(hand)
            condition_ids.append("transition")

    return Dataset(
        landmarks=np.stack(landmarks),
        labels=np.array(labels),
        signers=np.array(signer_ids),
        hands=np.array(hands),
        conditions=np.array(condition_ids),
    )
