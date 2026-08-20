"""Extract landmarks from video, for datasets that ship clips rather than coordinates.

FSboard and the GISLR competition publish pre-extracted landmarks. Most other sign
corpora — **INCLUDE** for Indian Sign Language, **PopSign** in its raw form — ship
video, so the landmarks have to be produced locally.

That is a one-off cost with a lasting benefit: the derived landmarks are far smaller
than the video (kilobytes against megabytes), they are what the model actually consumes,
and for a CC BY 4.0 source like INCLUDE or PopSign they can be **published**, which is
how this project can contribute something back rather than only consuming datasets.

Extraction must use the *same* landmark topology the browser produces, or the model
trains on one distribution and is served another. This module therefore uses MediaPipe's
**Tasks** API with the very same ``.task`` bundles `packages/web/src/vision/landmarker.ts`
loads — not the legacy ``mediapipe.solutions`` graphs, which are a different set of
weights, are absent from recent wheels, and reached end of life in March 2023.

Requires ``mediapipe`` and ``opencv-python``, plus the model bundles that
``npm run setup:assets`` downloads::

    pip install mediapipe opencv-python
    npm run setup:assets
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

LANDMARKS_PER_HAND = 21

#: Pose landmarks kept, matching `UPPER_BODY_POSE_INDICES` in
#: `packages/web/src/vision/landmarker.ts`: nose, shoulders, elbows, wrists.
POSE_INDICES = (0, 11, 12, 13, 14, 15, 16)

#: Where `npm run setup:assets` puts the bundles the browser also serves.
_DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[3] / "packages/web/public/models"

HAND_MODEL_NAME = "hand_landmarker.task"
POSE_MODEL_NAME = "pose_landmarker_lite.task"

#: Override the bundle directory, e.g. on a training box with no checkout.
MODEL_DIR_ENV = "MUDRA_MEDIAPIPE_MODELS"


def model_dir() -> Path:
    """Directory holding the ``.task`` bundles."""
    override = os.environ.get(MODEL_DIR_ENV)
    return Path(override) if override else _DEFAULT_MODEL_DIR


def require_model(name: str) -> Path:
    """Locate a bundle, failing with instructions rather than a bare ``FileNotFoundError``.

    Silently skipping a missing pose model would be worse than stopping: extraction would
    finish, write plausible-looking files, and every clip would carry an all-NaN pose.
    """
    path = model_dir() / name
    if not path.is_file():
        raise FileNotFoundError(
            f"MediaPipe bundle {name} not found at {path}.\n"
            f"Run `npm run setup:assets` from the repository root to download it, "
            f"or set {MODEL_DIR_ENV} to a directory that already has it."
        )
    return path


@dataclass(frozen=True, slots=True)
class ExtractedClip:
    """Landmarks for one video. NaN marks a hand or pose absent on that frame."""

    left_hand: np.ndarray  # (frames, 21, 3)
    right_hand: np.ndarray  # (frames, 21, 3)
    pose: np.ndarray  # (frames, 7, 3)

    @property
    def frames(self) -> int:
        return int(self.left_hand.shape[0])

    @property
    def detection_rate(self) -> float:
        """Fraction of frames with at least one hand.

        A low rate means the clip is unusable — the signer out of frame, the video too
        dark, the crop too tight. Worth checking before training on it.
        """
        left = ~np.isnan(self.left_hand).all(axis=(1, 2))
        right = ~np.isnan(self.right_hand).all(axis=(1, 2))
        return float((left | right).mean()) if self.frames else 0.0


# --------------------------------------------------------------------------------------
# Result conversion
#
# Kept as free functions over structural types so the topology and the handedness mapping
# — the two things that can silently desynchronise from the browser — are unit-testable
# without a GPU, a video file, or the model bundles.
# --------------------------------------------------------------------------------------


class _Point(Protocol):
    x: float
    y: float
    z: float


def hands_to_arrays(
    hand_landmarks: list[list[_Point]],
    handedness: list[list[Any]],
) -> tuple[np.ndarray, np.ndarray]:
    """Sort a ``HandLandmarkerResult`` into the signer's left and right slots.

    MediaPipe labels handedness from the *camera's* point of view. The browser preview is
    mirrored, so MediaPipe's "Left" is the signer's right hand — the same inversion
    `readHandedness` applies in the TypeScript landmarker. Getting this backwards trains
    a mirror image of the intended model, which fails quietly: accuracy stays plausible
    on symmetric signs and collapses on the rest.

    Returns ``(left, right)``, each ``(21, 3)`` and NaN-filled when absent.
    """
    left = np.full((LANDMARKS_PER_HAND, 3), np.nan)
    right = np.full((LANDMARKS_PER_HAND, 3), np.nan)

    for index, landmarks in enumerate(hand_landmarks):
        if len(landmarks) != LANDMARKS_PER_HAND:
            continue  # not a topology we recognise; better dropped than mis-slotted
        points = np.array([(p.x, p.y, p.z) for p in landmarks], dtype=np.float64)

        categories = handedness[index] if index < len(handedness) else []
        label = categories[0].category_name if categories else "Right"
        if label == "Left":
            right = points
        else:
            left = points

    return left, right


def pose_to_array(pose_landmarks: list[list[_Point]]) -> np.ndarray:
    """Take the upper-body subset from a ``PoseLandmarkerResult``. ``(7, 3)``, NaN if absent."""
    points = np.full((len(POSE_INDICES), 3), np.nan)
    if not pose_landmarks:
        return points

    body = pose_landmarks[0]
    for slot, source in enumerate(POSE_INDICES):
        if source < len(body):
            point = body[source]
            points[slot] = (point.x, point.y, point.z)
    return points


def _create_landmarkers(*, with_pose: bool) -> tuple[Any, Any]:
    """Build video-mode landmarkers. Imported lazily so the package works without MediaPipe."""
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    hand = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(require_model(HAND_MODEL_NAME))
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )

    pose = None
    if with_pose:
        pose = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=str(require_model(POSE_MODEL_NAME))
                ),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.5,
            )
        )

    return hand, pose


def extract_video(
    path: Path,
    *,
    max_frames: int | None = None,
    stride: int = 1,
    with_pose: bool = True,
) -> ExtractedClip:
    """Run MediaPipe over a video file.

    Parameters
    ----------
    max_frames:
        Stop after this many *retained* frames.
    stride:
        Keep every Nth frame. Useful when source footage is 60 fps and the model was
        trained around 30 — sampling down costs nothing and halves extraction time.
    with_pose:
        Extract upper-body pose as well. Needed for the two-handed `shoulder-frame-v1`
        scheme; skippable for single-hand fingerspelling, where it is pure overhead.
    """
    import cv2  # imported lazily so the rest of the package works without OpenCV
    import mediapipe as mp

    if stride < 1:
        raise ValueError(f"stride must be at least 1, received {stride}")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open {path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    # A container with no usable frame rate still has to yield increasing timestamps, or
    # video mode rejects the frame. 30 fps is a safe stand-in: the value only affects
    # MediaPipe's internal tracking, not the coordinates.
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0

    hand_landmarker, pose_landmarker = _create_landmarkers(with_pose=with_pose)

    left_frames: list[np.ndarray] = []
    right_frames: list[np.ndarray] = []
    pose_frames: list[np.ndarray] = []

    try:
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % stride != 0:
                index += 1
                continue

            # Timestamps come from the source position, not a counter, so a strided run
            # keeps the real spacing between frames.
            timestamp_ms = int(round(index * 1000.0 / fps))
            index += 1

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            hand_result = hand_landmarker.detect_for_video(image, timestamp_ms)
            left, right = hands_to_arrays(hand_result.hand_landmarks, hand_result.handedness)

            if pose_landmarker is not None:
                pose_result = pose_landmarker.detect_for_video(image, timestamp_ms)
                pose_points = pose_to_array(pose_result.pose_landmarks)
            else:
                pose_points = np.full((len(POSE_INDICES), 3), np.nan)

            left_frames.append(left)
            right_frames.append(right)
            pose_frames.append(pose_points)

            if max_frames is not None and len(left_frames) >= max_frames:
                break
    finally:
        capture.release()
        hand_landmarker.close()
        if pose_landmarker is not None:
            pose_landmarker.close()

    if not left_frames:
        raise ValueError(f"{path} produced no frames")

    return ExtractedClip(
        left_hand=np.stack(left_frames),
        right_hand=np.stack(right_frames),
        pose=np.stack(pose_frames),
    )


def dominant_from_clip(clip: ExtractedClip) -> str:
    """Infer the signer's dominant hand from which is detected more often.

    Assuming right-dominance would mirror every left-dominant signer's clips into the
    wrong slot — the model would still train, just badly.
    """
    left = int((~np.isnan(clip.left_hand).all(axis=(1, 2))).sum())
    right = int((~np.isnan(clip.right_hand).all(axis=(1, 2))).sum())
    return "left" if left > right else "right"


def extract_directory(
    root: Path,
    output: Path,
    *,
    pattern: str = "**/*.mp4",
    min_detection_rate: float = 0.5,
    stride: int = 1,
    with_pose: bool = True,
) -> dict[str, int]:
    """Extract every video under ``root`` into compressed ``.npz`` files.

    The label is taken from the parent directory name, which is how INCLUDE and most
    video corpora are organised.

    Clips whose hand-detection rate falls below ``min_detection_rate`` are **skipped and
    counted**, not silently included. Training on clips where the hands were rarely
    found teaches the model mostly about missing data.
    """
    output.mkdir(parents=True, exist_ok=True)
    counts = {"written": 0, "skipped_low_detection": 0, "failed": 0}

    for video in sorted(root.glob(pattern)):
        try:
            clip = extract_video(video, stride=stride, with_pose=with_pose)
        except FileNotFoundError:
            raise  # a missing model bundle affects every file; stop rather than log 5000 times
        except Exception as error:  # noqa: BLE001 — one bad file must not stop the run
            print(f"  ! {video.name}: {error}")
            counts["failed"] += 1
            continue

        if clip.detection_rate < min_detection_rate:
            print(f"  - {video.name}: hands found in only {clip.detection_rate:.0%} of frames")
            counts["skipped_low_detection"] += 1
            continue

        label = video.parent.name
        destination = output / label
        destination.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            destination / f"{video.stem}.npz",
            left_hand=clip.left_hand.astype(np.float32),
            right_hand=clip.right_hand.astype(np.float32),
            pose=clip.pose.astype(np.float32),
            label=label,
            dominant=dominant_from_clip(clip),
        )
        counts["written"] += 1

    return counts


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, required=True, help="Directory of video files")
    parser.add_argument("--out", type=Path, required=True, help="Where to write .npz landmarks")
    parser.add_argument("--pattern", default="**/*.mp4")
    parser.add_argument("--stride", type=int, default=1, help="Keep every Nth frame")
    parser.add_argument(
        "--no-pose",
        action="store_true",
        help="Skip pose extraction (single-hand fingerspelling does not use it)",
    )
    args = parser.parse_args()

    counts = extract_directory(
        args.videos,
        args.out,
        pattern=args.pattern,
        stride=args.stride,
        with_pose=not args.no_pose,
    )
    print(
        f"\nWrote {counts['written']} clips. "
        f"Skipped {counts['skipped_low_detection']} for low detection, "
        f"{counts['failed']} failed."
    )


if __name__ == "__main__":
    main()
