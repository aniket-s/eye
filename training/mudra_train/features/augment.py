"""Training-time augmentation.

Augmentation is where robustness comes from. The normaliser removes translation,
scale and handedness exactly; everything else — tilt, camera angle, tracking noise,
partial occlusion — is bought here.

Priorities follow what actually moved the needle in Google's sign-recognition
competitions, in order:

1. **Mirror with handedness swap** — the highest-value augmentation. Combined with
   canonicalisation in the normaliser it makes left- and right-handed signers
   equivalent.
2. **Rotation**, bounded hard. Bounded deliberately: H/U, K/P and G/Q differ by
   orientation alone, so rotation past a letter's declared band relabels the data.
   The bound is small — the *coverage* comes from each letter's declared orientation
   band in ``asl_alphabet.py``, where it can be stated per letter and checked against
   its twin, rather than from a blanket ±25° that had to be a compromise between
   "enough tilt for B" and "not enough to turn a U into an H".
3. **Anisotropic scale and shear** — stands in for camera angle and perspective.
4. **Landmark jitter and dropout** — MediaPipe is not exact, and fingers occlude.
5. **Targeted occlusion** — the two ways MediaPipe specifically fails on this
   vocabulary: a thumb buried under the fingers, and crossed fingers it cannot
   separate. Applied per letter rather than to everything, because an occlusion that
   cannot physically happen to a letter is not robustness, it is a wrong label.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class AugmentConfig:
    """Augmentation strengths. Every probability is per-sample, per-epoch."""

    mirror_probability: float = 0.5
    #: Small on purpose — see the module docstring. ``assert_bands_are_separable``
    #: rejects any value that would let two rotation-only letters meet.
    max_rotation_degrees: float = 8.0
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
    #: Chance of collapsing a hidden thumb toward the palm, for the letters that hide one.
    thumb_occlusion_probability: float = 0.35
    #: How far toward the palm centre a collapsed thumb is pulled, as a range.
    thumb_occlusion_strength: tuple[float, float] = (0.25, 0.85)
    #: Chance of merging crossed fingertips, for the letters that cross them.
    finger_merge_probability: float = 0.4
    #: How far the crossed tips are pulled toward each other, as a range.
    finger_merge_strength: tuple[float, float] = (0.3, 0.95)


#: Thumb chain past the wrist: MCP, IP, tip. Landmark 1 is the base and stays put — a
#: hidden thumb still has a visible root.
THUMB_CHAIN = (2, 3, 4)
#: Palm landmarks a collapsing thumb is pulled toward.
PALM = (0, 5, 9, 13, 17)
#: The two fingers R winds round each other: their DIPs and tips, which is where they
#: overlap. The knuckles stay where they are — those are never in doubt.
CROSSED_TIPS = ((7, 8), (11, 12))


def augment_batch(
    landmarks: np.ndarray,
    handedness: np.ndarray,
    rng: np.random.Generator,
    config: AugmentConfig = AugmentConfig(),
    labels: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Augment a batch of raw landmarks, before normalisation.

    Parameters
    ----------
    landmarks:
        ``(n, 21, 3)`` raw landmarks.
    handedness:
        ``(n,)`` array of ``"left"`` / ``"right"``.
    labels:
        ``(n,)`` array of class labels. Optional, and only the targeted occlusions in
        step 7 use it — they are the one augmentation that is not valid for every
        letter, because a thumb held out in the open cannot be occluded and pretending
        otherwise relabels the sample. Omitting it skips step 7 entirely rather than
        applying it blindly.

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

    # 7. Targeted occlusion: the two ways MediaPipe fails on *this* vocabulary.
    if labels is not None:
        _occlude(points, labels, rng, config)

    return points, hands


def _occlude(
    points: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
    config: AugmentConfig,
) -> None:
    """Degrade landmarks the way a camera does, for the letters it happens to.

    Modelled rather than randomised, because MediaPipe's failures are structured and
    Gaussian noise does not imitate them. A hidden thumb does not go missing — the
    network still emits one, regressed toward where a thumb usually is, which is
    inward toward the palm. Crossed fingers do not scatter — their tips converge,
    because the model cannot tell which of two overlapping fingers it is looking at.

    Both are applied in place, after normalisation-independent geometry is settled, and
    both leave the landmarks that stay visible alone: a buried thumb still shows its
    base, and crossed fingers still show their knuckles.
    """
    from ..ingest.asl_alphabet import MERGED_FINGERS, OCCLUDED_THUMB

    if config.thumb_occlusion_probability > 0:
        eligible = np.isin(labels, list(OCCLUDED_THUMB))
        chosen = np.flatnonzero(eligible & (rng.random(len(labels)) < config.thumb_occlusion_probability))
        for index in chosen:
            palm = points[index, list(PALM), :2].mean(axis=0)
            strength = float(rng.uniform(*config.thumb_occlusion_strength))
            for joint in THUMB_CHAIN:
                points[index, joint, :2] += (palm - points[index, joint, :2]) * strength

    if config.finger_merge_probability > 0:
        eligible = np.isin(labels, list(MERGED_FINGERS))
        chosen = np.flatnonzero(eligible & (rng.random(len(labels)) < config.finger_merge_probability))
        for index in chosen:
            strength = float(rng.uniform(*config.finger_merge_strength))
            for first, second in zip(*CROSSED_TIPS, strict=True):
                middle = (points[index, first, :2] + points[index, second, :2]) / 2.0
                points[index, first, :2] += (middle - points[index, first, :2]) * strength
                points[index, second, :2] += (middle - points[index, second, :2]) * strength
