"""Two-handed body-relative normalisation — the Python mirror of
``packages/core/src/features/normaliseBody.ts``.

Same contract as the fingerspelling normaliser: these two implementations must agree,
and ``tests/test_parity.py`` enforces it against shared fixtures. See the TypeScript
module for the full rationale — in particular why both hands share one shoulder-anchored
reference frame rather than being normalised independently.
"""

from __future__ import annotations

import numpy as np

LANDMARKS_PER_HAND = 21

#: Pose landmarks retained: nose, shoulders, elbows, wrists.
POSE_POINTS = 7

#: Index pairs swapped when mirroring. The nose has no counterpart.
POSE_MIRROR_PAIRS = ((1, 2), (3, 4), (5, 6))

_HAND_VALUES = LANDMARKS_PER_HAND * 2

#: Offsets into the feature vector, matching ``BODY_LAYOUT`` in TypeScript.
DOMINANT_HAND = 0
OTHER_HAND = _HAND_VALUES
POSE = _HAND_VALUES * 2
DOMINANT_PRESENT = _HAND_VALUES * 2 + POSE_POINTS * 2
OTHER_PRESENT = DOMINANT_PRESENT + 1

#: 42 + 42 + 14 + 2.
BODY_FEATURE_LENGTH = OTHER_PRESENT + 1

#: Identifier recorded in a pack manifest. Must match the TypeScript constant.
BODY_NORMALISATION_SCHEME = "shoulder-frame-v1"

MIN_SCALE = 1e-6


def normalise_body_frame(
    dominant_hand: np.ndarray | None,
    other_hand: np.ndarray | None,
    pose: np.ndarray | None,
    dominant: str = "right",
) -> np.ndarray:
    """Build one frame's feature vector.

    Parameters
    ----------
    dominant_hand, other_hand:
        ``(21, 2+)`` arrays, or ``None`` when the hand was not detected.
    pose:
        ``(7, 2+)`` array of upper-body landmarks, or ``None``.
    dominant:
        ``"left"`` or ``"right"``. Left-dominant signers are mirrored.

    Returns
    -------
    ``(BODY_FEATURE_LENGTH,)`` float32.
    """
    if dominant not in ("left", "right"):
        raise ValueError(f'dominant must be "left" or "right", got "{dominant}"')

    features = np.zeros(BODY_FEATURE_LENGTH, dtype=np.float64)
    flip = -1.0 if dominant == "left" else 1.0

    # ── Reference frame ──
    if pose is not None and len(pose) >= 3:
        pose_xy = np.asarray(pose, dtype=np.float64)[:, :2]
        left_shoulder, right_shoulder = pose_xy[1], pose_xy[2]
        origin_x = float((left_shoulder[0] + right_shoulder[0]) / 2 * flip)
        origin_y = float((left_shoulder[1] + right_shoulder[1]) / 2)
        scale = max(float(np.hypot(*(left_shoulder - right_shoulder))), MIN_SCALE)
    else:
        origin_x, origin_y, scale = _hands_reference(dominant_hand, other_hand, flip)

    def write(offset: int, points: np.ndarray) -> None:
        xy = np.asarray(points, dtype=np.float64)[:, :2].copy()
        xy[:, 0] = xy[:, 0] * flip - origin_x
        xy[:, 1] = xy[:, 1] - origin_y
        xy /= scale
        features[offset : offset + xy.size] = xy.reshape(-1)

    # ── Hands ──
    if dominant_hand is not None and len(dominant_hand) == LANDMARKS_PER_HAND:
        write(DOMINANT_HAND, dominant_hand)
        features[DOMINANT_PRESENT] = 1.0
    if other_hand is not None and len(other_hand) == LANDMARKS_PER_HAND:
        write(OTHER_HAND, other_hand)
        features[OTHER_PRESENT] = 1.0

    # ── Pose ──
    if pose is not None:
        source = np.asarray(pose, dtype=np.float64)[:POSE_POINTS]
        if flip == -1.0:
            # Mirroring swaps the body's own left and right, so the groups swap too.
            source = source.copy()
            for a, b in POSE_MIRROR_PAIRS:
                if a < len(source) and b < len(source):
                    source[[a, b]] = source[[b, a]]
        write(POSE, source)

    return features.astype(np.float32)


def _hands_reference(
    dominant_hand: np.ndarray | None, other_hand: np.ndarray | None, flip: float
) -> tuple[float, float, float]:
    """Origin and scale from the hands, when pose is unavailable."""
    available = [h for h in (dominant_hand, other_hand) if h is not None]
    if not available:
        return 0.0, 0.0, 1.0

    points = np.concatenate([np.asarray(h, dtype=np.float64)[:, :2] for h in available])
    points = points.copy()
    points[:, 0] *= flip

    centre = points.mean(axis=0)
    spread = float(np.sqrt(((points - centre) ** 2).sum(axis=1).mean()))
    # x2.5 so the scale is comparable to shoulder width, keeping feature magnitudes in
    # the same range whichever path produced them.
    return float(centre[0]), float(centre[1]), max(spread * 2.5, MIN_SCALE)


def normalise_body_sequence(
    dominant_hands: np.ndarray,
    other_hands: np.ndarray,
    poses: np.ndarray,
    dominant: str = "right",
) -> np.ndarray:
    """Normalise a whole sequence.

    Arrays are ``(frames, 21, 2+)`` for hands and ``(frames, 7, 2+)`` for pose. Use
    ``NaN`` to mark a missing hand on a given frame; those frames get a zero vector and
    a cleared presence flag.

    Returns ``(frames, BODY_FEATURE_LENGTH)``.
    """
    frames = len(dominant_hands)
    out = np.zeros((frames, BODY_FEATURE_LENGTH), dtype=np.float32)

    for i in range(frames):
        dominant_frame = dominant_hands[i]
        other_frame = other_hands[i]
        pose_frame = poses[i] if poses is not None else None

        out[i] = normalise_body_frame(
            None if dominant_frame is None or np.isnan(dominant_frame).all() else dominant_frame,
            None if other_frame is None or np.isnan(other_frame).all() else other_frame,
            None if pose_frame is None or np.isnan(pose_frame).all() else pose_frame,
            dominant,
        )

    return out
