"""Feature normalisation — the Python mirror of ``packages/core/src/features/normalise.ts``.

These two implementations **must** agree. If training normalises differently from
inference, the model sees a different distribution in production than it was fitted
on, and the symptom is a model that scores well offline and behaves badly on a real
webcam. That failure is silent, which is what makes it dangerous.

The guard is ``tests/test_parity.py``: it loads the shared fixtures in
``training/fixtures/normalisation.json`` — generated from the TypeScript
implementation — and asserts this module reproduces them to 1e-6. Change one side
without the other and the build fails.

See the TypeScript module for the full rationale, in particular why rotation is
deliberately *not* normalised (K/P, G/Q and H/U differ only by orientation, so
canonicalising rotation would merge those classes).
"""

from __future__ import annotations

import numpy as np

#: Landmarks MediaPipe reports per hand.
LANDMARKS_PER_HAND = 21

#: Output length: 21 landmarks x (x, y).
FEATURE_LENGTH = LANDMARKS_PER_HAND * 2

#: Smallest permitted scale divisor, guarding a degenerate hand.
MIN_SPREAD = 1e-6

#: Identifier recorded in a pack manifest. Must match ``NORMALISATION_SCHEME`` in
#: ``packages/core/src/registry/manifest.ts``; the loader refuses a mismatch.
NORMALISATION_SCHEME = "centroid-rms-v2"


def normalise_hand(landmarks: np.ndarray, handedness: str = "right") -> np.ndarray:
    """Turn one hand into the model's input features.

    Parameters
    ----------
    landmarks:
        Array of shape ``(21, 2)`` or ``(21, 3)``. A third column is ignored — ``z``
        is weakly calibrated from a single view and is dropped deliberately.
    handedness:
        ``"left"`` or ``"right"``. Left hands are mirrored so the model only ever
        sees right-handed geometry, which is what makes left-handed signers work.

    Returns
    -------
    Array of shape ``(42,)``, ordered ``[x0, y0, x1, y1, ...]``.
    """
    points = np.asarray(landmarks, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] != LANDMARKS_PER_HAND or points.shape[1] < 2:
        raise ValueError(
            f"Expected landmarks of shape ({LANDMARKS_PER_HAND}, 2+), got {points.shape}"
        )

    xy = points[:, :2].copy()

    # 1. Handedness. Mirroring x is sufficient for a single hand: landmark indices are
    #    anatomically defined, so a mirrored left hand *is* a right hand.
    if handedness == "left":
        xy[:, 0] = -xy[:, 0]
    elif handedness != "right":
        raise ValueError(f'handedness must be "left" or "right", got "{handedness}"')

    # 2. Centre on the centroid — more stable than the wrist, which sits at the edge
    #    of the hand where MediaPipe jitters most.
    xy -= xy.mean(axis=0)

    # 3. Scale by RMS spread. A single bone length foreshortens as the palm tilts, so
    #    it confounds pose with distance; RMS spread does not.
    spread = max(float(np.sqrt((xy**2).sum() / LANDMARKS_PER_HAND)), MIN_SPREAD)
    xy /= spread

    return xy.reshape(-1).astype(np.float32)


def normalise_batch(landmarks: np.ndarray, handedness: np.ndarray) -> np.ndarray:
    """Vectorised :func:`normalise_hand` over a batch.

    Parameters
    ----------
    landmarks:
        Array of shape ``(n, 21, 2+)``.
    handedness:
        Array of ``n`` strings, each ``"left"`` or ``"right"``.

    Returns
    -------
    Array of shape ``(n, 42)``.
    """
    points = np.asarray(landmarks, dtype=np.float64)
    if points.ndim != 3 or points.shape[1] != LANDMARKS_PER_HAND:
        raise ValueError(f"Expected shape (n, {LANDMARKS_PER_HAND}, 2+), got {points.shape}")
    if len(handedness) != points.shape[0]:
        raise ValueError("handedness must have one entry per sample")

    xy = points[:, :, :2].copy()

    flip = np.where(np.asarray(handedness) == "left", -1.0, 1.0).reshape(-1, 1)
    xy[:, :, 0] *= flip

    xy -= xy.mean(axis=1, keepdims=True)

    spread = np.sqrt((xy**2).sum(axis=(1, 2)) / LANDMARKS_PER_HAND)
    spread = np.maximum(spread, MIN_SPREAD).reshape(-1, 1, 1)
    xy /= spread

    return xy.reshape(points.shape[0], -1).astype(np.float32)


def mirror_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Mirror horizontally, for the handedness augmentation."""
    out = np.asarray(landmarks, dtype=np.float64).copy()
    out[..., 0] = -out[..., 0]
    return out
