"""Training-time augmentation.

Augmentation is where robustness comes from. The normaliser removes translation,
scale and handedness exactly; everything else — tilt, camera angle, tracking noise,
partial occlusion — is bought here.

Priorities follow what actually moved the needle in Google's sign-recognition
competitions, in order:

1. **Mirror with handedness swap** — the highest-value augmentation. Combined with
   canonicalisation in the normaliser it makes left- and right-handed signers
   equivalent.
2. **Rotation**, bounded to ±25°. Bounded deliberately: K/P and G/Q differ by
   orientation, so unbounded rotation would relabel the data.
3. **Anisotropic scale and shear** — stands in for camera angle and perspective.
4. **Landmark jitter and dropout** — MediaPipe is not exact, and fingers occlude.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class AugmentConfig:
    """Augmentation strengths. Every probability is per-sample, per-epoch."""

    mirror_probability: float = 0.5
    #: Bounded to stay well clear of the ~90° that separates K from P and G from Q.
    max_rotation_degrees: float = 25.0
    rotation_probability: float = 0.8
    scale_range: tuple[float, float] = (0.85, 1.15)
    scale_probability: float = 0.7
    max_shear: float = 0.12
    shear_probability: float = 0.5
    #: Standard deviation of per-landmark noise, in normalised hand units.
    jitter_std: float = 0.02
    jitter_probability: float = 0.8
    #: Landmarks replaced by the hand centroid, mimicking an occluded finger.
    max_dropout: int = 2
    dropout_probability: float = 0.25


def augment_batch(
    landmarks: np.ndarray,
    handedness: np.ndarray,
    rng: np.random.Generator,
    config: AugmentConfig = AugmentConfig(),
) -> tuple[np.ndarray, np.ndarray]:
    """Augment a batch of raw landmarks, before normalisation.

    Parameters
    ----------
    landmarks:
        ``(n, 21, 3)`` raw landmarks.
    handedness:
        ``(n,)`` array of ``"left"`` / ``"right"``.

    Returns
    -------
    ``(augmented_landmarks, augmented_handedness)``. Handedness changes when a sample
    is mirrored, and both must travel together — mirroring the geometry without
    flipping the label would teach the model the exact opposite of the intended
    invariance.
    """
    points = np.array(landmarks, dtype=np.float64, copy=True)
    hands = np.array(handedness, copy=True)
    count = points.shape[0]

    # 1. Mirror, flipping the handedness label with it.
    if config.mirror_probability > 0:
        selected = rng.random(count) < config.mirror_probability
        points[selected, :, 0] = -points[selected, :, 0]
        hands[selected] = np.where(hands[selected] == "left", "right", "left")

    centroids = points[:, :, :2].mean(axis=1, keepdims=True)
    centred = points[:, :, :2] - centroids

    # 2. Rotation about the hand centre.
    if config.rotation_probability > 0 and config.max_rotation_degrees > 0:
        selected = rng.random(count) < config.rotation_probability
        angles = np.deg2rad(
            rng.uniform(-config.max_rotation_degrees, config.max_rotation_degrees, size=count)
        )
        angles[~selected] = 0.0
        cos, sin = np.cos(angles), np.sin(angles)
        x = centred[:, :, 0].copy()
        y = centred[:, :, 1].copy()
        centred[:, :, 0] = cos[:, None] * x - sin[:, None] * y
        centred[:, :, 1] = sin[:, None] * x + cos[:, None] * y

    # 3. Anisotropic scale. Independent per axis, so it also covers the apparent
    #    squash of a hand seen off-axis.
    if config.scale_probability > 0:
        selected = rng.random(count) < config.scale_probability
        factors = rng.uniform(config.scale_range[0], config.scale_range[1], size=(count, 1, 2))
        factors[~selected] = 1.0
        centred *= factors

    # 4. Shear.
    if config.shear_probability > 0 and config.max_shear > 0:
        selected = rng.random(count) < config.shear_probability
        shear = rng.uniform(-config.max_shear, config.max_shear, size=count)
        shear[~selected] = 0.0
        centred[:, :, 0] += shear[:, None] * centred[:, :, 1]

    # 5. Jitter, scaled to each hand's size so it means the same at any distance.
    if config.jitter_probability > 0 and config.jitter_std > 0:
        selected = rng.random(count) < config.jitter_probability
        spread = np.sqrt((centred**2).sum(axis=(1, 2)) / centred.shape[1])
        noise = rng.normal(0.0, config.jitter_std, size=centred.shape)
        noise *= spread[:, None, None]
        noise[~selected] = 0.0
        centred += noise

    points[:, :, :2] = centred + centroids

    # 6. Dropout: collapse a few landmarks onto the hand centre, as MediaPipe does
    #    when a finger is hidden behind the palm.
    if config.dropout_probability > 0 and config.max_dropout > 0:
        selected = np.flatnonzero(rng.random(count) < config.dropout_probability)
        for index in selected:
            how_many = int(rng.integers(1, config.max_dropout + 1))
            which = rng.choice(points.shape[1], size=how_many, replace=False)
            points[index, which, :2] = points[index, :, :2].mean(axis=0)

    return points, hands
