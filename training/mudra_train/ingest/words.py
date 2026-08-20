"""Word-level sign data: a real-format adapter and a synthetic fixture generator.

Two sources:

**GISLR / PopSign parquet** (:func:`load_gislr`). Google's isolated-sign competition
and the PopSign release both ship pre-extracted MediaPipe Holistic landmarks in the same
long-format parquet: one row per landmark per frame, with columns ``frame``, ``type``,
``landmark_index``, ``x``, ``y``, ``z``. That is the real path to a 250-sign model.

Prefer **PopSign ASL v1.0** over the Kaggle competition data. Same 250-sign vocabulary,
47 consenting Deaf signers, and — unlike the competition set — a **CC BY 4.0 licence**,
so landmarks derived from it can be published rather than merely trained on.

**Synthetic two-handed sequences** (:func:`generate_synthetic`). Same caveat as
everywhere else in ``ingest``: procedural geometry that validates the pipeline and
produces an unusable model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .synthetic import _HANDSHAPES, _build_hand

LANDMARKS_PER_HAND = 21
POSE_POINTS = 7

#: Indices into MediaPipe's 33-point pose that the feature extractor keeps:
#: nose, shoulders, elbows, wrists.
GISLR_POSE_INDICES = (0, 11, 12, 13, 14, 15, 16)


@dataclass(frozen=True, slots=True)
class WordDataset:
    """Variable-length two-handed sequences, one label each."""

    #: Each ``(frames, 21, 3)``. NaN marks a hand absent on that frame.
    dominant_hands: list[np.ndarray]
    other_hands: list[np.ndarray]
    #: Each ``(frames, 7, 3)``, or None when pose was unavailable.
    poses: list[np.ndarray | None]
    labels: np.ndarray
    signers: np.ndarray
    dominants: np.ndarray

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def classes(self) -> list[str]:
        return sorted(set(self.labels.tolist()))

    def summary(self) -> str:
        lengths = [len(s) for s in self.dominant_hands]
        return (
            f"{len(self)} clips, {len(self.classes)} signs\n"
            f"  frames:  {min(lengths)}–{max(lengths)} (mean {np.mean(lengths):.0f})\n"
            f"  signers: {len(set(self.signers.tolist()))}"
        )


def load_gislr(
    index_csv: Path,
    data_root: Path,
    *,
    label_column: str = "sign",
    signer_column: str = "participant_id",
    path_column: str = "path",
    limit: int | None = None,
) -> WordDataset:
    """Load competition-format landmark parquet.

    Parameters
    ----------
    index_csv:
        ``train.csv``-style index listing one row per clip.
    data_root:
        Directory the index's ``path`` column is relative to.
    limit:
        Read at most this many clips. Useful for a smoke run before committing to the
        full 40 GB.

    Notes
    -----
    Handedness is inferred per clip from which hand has more detected frames. The
    competition data does not record which hand a signer favours, and assuming
    right-dominance would mirror every left-dominant signer's clips into the wrong slot.
    """
    import pandas as pd

    index = pd.read_csv(index_csv)
    if limit is not None:
        index = index.head(limit)

    dominant_hands: list[np.ndarray] = []
    other_hands: list[np.ndarray] = []
    poses: list[np.ndarray | None] = []
    labels: list[str] = []
    signers: list[str] = []
    dominants: list[str] = []

    for row in index.itertuples(index=False):
        frame = pd.read_parquet(data_root / getattr(row, path_column))
        left, right, pose = _split_parquet(frame)

        # More detected frames = the working hand.
        left_frames = int((~np.isnan(left).all(axis=(1, 2))).sum())
        right_frames = int((~np.isnan(right).all(axis=(1, 2))).sum())
        dominant = "left" if left_frames > right_frames else "right"

        dominant_hands.append(left if dominant == "left" else right)
        other_hands.append(right if dominant == "left" else left)
        poses.append(pose)
        labels.append(str(getattr(row, label_column)))
        signers.append(str(getattr(row, signer_column)))
        dominants.append(dominant)

    return WordDataset(
        dominant_hands=dominant_hands,
        other_hands=other_hands,
        poses=poses,
        labels=np.array(labels),
        signers=np.array(signers),
        dominants=np.array(dominants),
    )


def _split_parquet(frame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pivot long-format landmarks into ``(frames, points, 3)`` arrays."""
    frame_ids = np.sort(frame["frame"].unique())
    position = {value: index for index, value in enumerate(frame_ids)}
    count = len(frame_ids)

    left = np.full((count, LANDMARKS_PER_HAND, 3), np.nan)
    right = np.full((count, LANDMARKS_PER_HAND, 3), np.nan)
    pose = np.full((count, POSE_POINTS, 3), np.nan)

    pose_slot = {source: slot for slot, source in enumerate(GISLR_POSE_INDICES)}

    for kind, group in frame.groupby("type", observed=True):
        if kind == "left_hand":
            target, valid = left, None
        elif kind == "right_hand":
            target, valid = right, None
        elif kind == "pose":
            target, valid = pose, pose_slot
        else:
            continue  # face landmarks are unused here

        for record in group.itertuples(index=False):
            slot = record.landmark_index if valid is None else valid.get(record.landmark_index)
            if slot is None:
                continue
            target[position[record.frame], slot] = (record.x, record.y, record.z)

    return left, right, pose


# ── Synthetic fixtures ────────────────────────────────────────────────────────

#: Two-handed sign templates: (name, handshape per hand, motion path, one/two handed).
#: Paths are (dx, dy) offsets traced over the clip, in normalised body units.
_SIGN_TEMPLATES: list[tuple[str, str, str, tuple[float, float], bool]] = [
    ("hello", "B", "B", (0.35, -0.15), False),
    ("thank_you", "B", "B", (0.0, 0.30), False),
    ("please", "B", "B", (0.15, 0.15), False),
    ("sorry", "A", "A", (0.10, 0.10), False),
    ("yes", "S", "S", (0.0, 0.20), False),
    ("no", "D", "D", (0.25, 0.0), False),
    ("help", "A", "B", (0.0, -0.30), True),
    ("more", "O", "O", (0.20, 0.0), True),
    ("book", "B", "B", (0.30, 0.0), True),
    ("water", "W", "W", (0.0, 0.18), False),
    ("eat", "O", "O", (0.0, -0.25), False),
    ("name", "U", "U", (0.15, 0.12), True),
    ("home", "O", "O", (0.28, -0.10), False),
    ("friend", "X", "X", (0.12, 0.12), True),
    ("family", "F", "F", (0.35, 0.0), True),
]


def generate_synthetic(
    clips_per_sign: int = 24,
    signers: int = 6,
    seed: int = 20260820,
) -> WordDataset:
    """Generate two-handed word-sign clips.

    **Not training data.** Procedural geometry for pipeline validation only; see
    :mod:`mudra_train.ingest.synthetic`.

    Each sign gets a distinct handshape pair *and* a distinct motion path, because
    handshape alone does not identify a word sign — movement and location carry as much
    meaning, which is precisely why this model consumes sequences rather than frames.
    """
    rng = np.random.default_rng(seed)

    dominant_hands: list[np.ndarray] = []
    other_hands: list[np.ndarray] = []
    poses: list[np.ndarray | None] = []
    labels: list[str] = []
    signer_ids: list[str] = []
    dominants: list[str] = []

    for signer in range(signers):
        signer_scale = float(rng.uniform(0.88, 1.12))
        signer_pace = float(rng.uniform(0.75, 1.35))
        shoulder_width = 0.24 * signer_scale
        signer_name = f"synthetic-{signer:02d}"

        for name, dominant_shape, other_shape, (path_x, path_y), two_handed in _SIGN_TEMPLATES:
            for _ in range(clips_per_sign // signers):
                frames = max(12, int(round(rng.integers(20, 40) * signer_pace)))
                dominant = "left" if rng.random() < 0.25 else "right"

                dominant_track = np.full((frames, LANDMARKS_PER_HAND, 3), np.nan)
                other_track = np.full((frames, LANDMARKS_PER_HAND, 3), np.nan)
                pose_track = np.full((frames, POSE_POINTS, 3), np.nan)

                start = np.array([0.5 - path_x / 2, 0.55 - path_y / 2])
                jitter = float(rng.uniform(0.004, 0.014))

                for t in range(frames):
                    progress = t / max(frames - 1, 1)
                    offset = start + np.array([path_x, path_y]) * progress

                    shape = _build_hand(*_HANDSHAPES[dominant_shape], rng, jitter)
                    shape[:, :2] = shape[:, :2] * 0.11 * signer_scale + offset
                    dominant_track[t] = shape

                    if two_handed:
                        mirror_shape = _build_hand(*_HANDSHAPES[other_shape], rng, jitter)
                        mirror_shape[:, :2] = (
                            mirror_shape[:, :2] * 0.11 * signer_scale
                            + offset
                            + np.array([-0.16 * signer_scale, 0.03])
                        )
                        other_track[t] = mirror_shape

                    # Shoulders fixed; the signer's body does not travel with the sign.
                    pose_track[t] = np.array(
                        [
                            [0.5, 0.22, 0.0],  # nose
                            [0.5 - shoulder_width / 2, 0.38, 0.0],  # left shoulder
                            [0.5 + shoulder_width / 2, 0.38, 0.0],  # right shoulder
                            [0.5 - shoulder_width, 0.52, 0.0],  # left elbow
                            [0.5 + shoulder_width, 0.52, 0.0],  # right elbow
                            [0.5 - shoulder_width, 0.64, 0.0],  # left wrist
                            [0.5 + shoulder_width, 0.64, 0.0],  # right wrist
                        ]
                    )

                if dominant == "left":
                    for track in (dominant_track, other_track, pose_track):
                        track[:, :, 0] = 1.0 - track[:, :, 0]

                dominant_hands.append(dominant_track)
                other_hands.append(other_track)
                poses.append(pose_track)
                labels.append(name)
                signer_ids.append(signer_name)
                dominants.append(dominant)

    return WordDataset(
        dominant_hands=dominant_hands,
        other_hands=other_hands,
        poses=poses,
        labels=np.array(labels),
        signers=np.array(signer_ids),
        dominants=np.array(dominants),
    )
