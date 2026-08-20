"""Video landmark extraction.

Two layers are covered. First, the parts that can silently desynchronise from the browser
— the handedness inversion and the pose subset — since those produce a model that trains
cleanly and serves wrongly. Second, the extraction loop itself, run over videos written
here with OpenCV, with MediaPipe's landmarkers faked: its `.task` bundles are a separate
download, so CI cannot depend on them, but everything this module actually owns (frame
decoding, striding, timestamps, cleanup, the skip-and-count policy) is real.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from mudra_train.ingest import video as video_module
from mudra_train.ingest.video import (
    HAND_MODEL_NAME,
    LANDMARKS_PER_HAND,
    MODEL_DIR_ENV,
    POSE_INDICES,
    ExtractedClip,
    dominant_from_clip,
    extract_directory,
    extract_video,
    hands_to_arrays,
    pose_to_array,
    require_model,
)

cv2 = pytest.importorskip("cv2")


@dataclass
class FakePoint:
    x: float
    y: float
    z: float


@dataclass
class FakeCategory:
    category_name: str


def hand(value: float) -> list[FakePoint]:
    """21 landmarks, all at the same spot, so the slot is identifiable by value."""
    return [FakePoint(value, value, value) for _ in range(LANDMARKS_PER_HAND)]


class TestHandsToArrays:
    def test_absent_hands_are_nan(self) -> None:
        left, right = hands_to_arrays([], [])
        assert np.isnan(left).all()
        assert np.isnan(right).all()
        assert left.shape == (LANDMARKS_PER_HAND, 3)

    def test_mediapipe_left_is_the_signers_right(self) -> None:
        """The inversion that `readHandedness` applies in the browser.

        MediaPipe labels from the camera's point of view; the preview is mirrored. Get
        this backwards and the model learns a mirror image of the intended one.
        """
        left, right = hands_to_arrays([hand(0.25)], [[FakeCategory("Left")]])
        assert np.allclose(right, 0.25)
        assert np.isnan(left).all()

    def test_mediapipe_right_is_the_signers_left(self) -> None:
        left, right = hands_to_arrays([hand(0.75)], [[FakeCategory("Right")]])
        assert np.allclose(left, 0.75)
        assert np.isnan(right).all()

    def test_both_hands_land_in_separate_slots(self) -> None:
        left, right = hands_to_arrays(
            [hand(0.1), hand(0.9)],
            [[FakeCategory("Left")], [FakeCategory("Right")]],
        )
        assert np.allclose(right, 0.1)
        assert np.allclose(left, 0.9)

    def test_missing_handedness_defaults_rather_than_crashing(self) -> None:
        """A truncated result should cost one hand, not the whole extraction run."""
        left, right = hands_to_arrays([hand(0.5)], [])
        assert np.allclose(left, 0.5)

    def test_unexpected_topology_is_dropped(self) -> None:
        """Better an absent hand than 21 slots filled from a different landmark set."""
        left, right = hands_to_arrays([[FakePoint(0.5, 0.5, 0.5)]], [[FakeCategory("Right")]])
        assert np.isnan(left).all()
        assert np.isnan(right).all()

    def test_coordinates_are_preserved_in_order(self) -> None:
        points = [FakePoint(i / 100, i / 200, i / 400) for i in range(LANDMARKS_PER_HAND)]
        left, _ = hands_to_arrays([points], [[FakeCategory("Right")]])
        assert left[7] == pytest.approx([0.07, 0.035, 0.0175])


class TestPoseToArray:
    def test_absent_pose_is_nan(self) -> None:
        assert np.isnan(pose_to_array([])).all()

    def test_takes_the_browsers_subset_in_the_browsers_order(self) -> None:
        """Same indices as `UPPER_BODY_POSE_INDICES`, in the same order.

        The order is load-bearing: `normalise_body` addresses shoulders by position, so a
        reordering here shifts every body-relative feature.
        """
        body = [FakePoint(i, i, i) for i in range(33)]
        points = pose_to_array([body])
        assert points.shape == (len(POSE_INDICES), 3)
        assert [int(p[0]) for p in points] == list(POSE_INDICES)

    def test_short_pose_yields_nan_rather_than_indexerror(self) -> None:
        points = pose_to_array([[FakePoint(1, 1, 1)]])
        assert not np.isnan(points[0]).any()  # index 0 present
        assert np.isnan(points[1:]).all()  # 11..16 absent

    def test_only_the_first_person_is_used(self) -> None:
        first = [FakePoint(0.1, 0.1, 0.1) for _ in range(33)]
        second = [FakePoint(0.9, 0.9, 0.9) for _ in range(33)]
        assert np.allclose(pose_to_array([first, second]), 0.1)


class TestClip:
    def make(self, left_present: int, right_present: int, frames: int = 10) -> ExtractedClip:
        left = np.full((frames, LANDMARKS_PER_HAND, 3), np.nan)
        right = np.full((frames, LANDMARKS_PER_HAND, 3), np.nan)
        left[:left_present] = 0.5
        right[:right_present] = 0.5
        return ExtractedClip(left, right, np.full((frames, len(POSE_INDICES), 3), np.nan))

    def test_detection_rate_counts_frames_with_either_hand(self) -> None:
        assert self.make(3, 3).detection_rate == pytest.approx(0.3)
        # Overlapping frames must not be double-counted.
        assert self.make(4, 2).detection_rate == pytest.approx(0.4)

    def test_detection_rate_of_an_empty_clip_is_zero(self) -> None:
        assert self.make(0, 0).detection_rate == 0.0

    def test_dominant_hand_follows_the_more_frequent_one(self) -> None:
        assert dominant_from_clip(self.make(2, 8)) == "right"
        assert dominant_from_clip(self.make(8, 2)) == "left"

    def test_ties_assume_right_dominance(self) -> None:
        assert dominant_from_clip(self.make(5, 5)) == "right"


class TestModelDiscovery:
    def test_a_missing_bundle_says_how_to_get_it(self, tmp_path, monkeypatch) -> None:
        """Extraction must stop loudly. Continuing would write thousands of empty clips."""
        monkeypatch.setenv(MODEL_DIR_ENV, str(tmp_path))
        with pytest.raises(FileNotFoundError, match="setup:assets"):
            require_model(HAND_MODEL_NAME)

    def test_finds_a_bundle_that_is_present(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv(MODEL_DIR_ENV, str(tmp_path))
        (tmp_path / HAND_MODEL_NAME).write_bytes(b"not a real bundle")
        assert require_model(HAND_MODEL_NAME) == tmp_path / HAND_MODEL_NAME


# --------------------------------------------------------------------------------------
# The extraction loop
#
# MediaPipe's `.task` bundles are a separate download, so CI cannot rely on them. Faking
# the landmarkers still exercises everything this module owns: frame decoding, striding,
# timestamp derivation, resource cleanup, and the skip-and-count policy in
# `extract_directory`. Only MediaPipe's own inference is stubbed.
# --------------------------------------------------------------------------------------


@dataclass
class FakeHandResult:
    hand_landmarks: list
    handedness: list


@dataclass
class FakePoseResult:
    pose_landmarks: list


class FakeLandmarker:
    """Records the timestamps it was given, so striding can be asserted."""

    def __init__(self, result_for) -> None:
        self._result_for = result_for
        self.timestamps: list[int] = []
        self.closed = False

    def detect_for_video(self, image, timestamp_ms: int):
        self.timestamps.append(timestamp_ms)
        return self._result_for(len(self.timestamps) - 1)

    def close(self) -> None:
        self.closed = True


def write_video(path, frames: int = 12, fps: float = 30.0) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 64))
    assert writer.isOpened(), "OpenCV has no mp4v encoder in this environment"
    for i in range(frames):
        writer.write(np.full((64, 64, 3), i * 5 % 256, dtype=np.uint8))
    writer.release()


@pytest.fixture
def fake_landmarkers(monkeypatch):
    """Every frame reports one right-camera hand and a full pose."""
    made: dict[str, FakeLandmarker] = {}

    def create(*, with_pose: bool):
        made["hand"] = FakeLandmarker(
            lambda _: FakeHandResult([hand(0.5)], [[FakeCategory("Right")]])
        )
        made["pose"] = (
            FakeLandmarker(lambda _: FakePoseResult([[FakePoint(0.2, 0.2, 0.2)] * 33]))
            if with_pose
            else None
        )
        return made["hand"], made["pose"]

    monkeypatch.setattr(video_module, "_create_landmarkers", create)
    return made


class TestExtractionLoop:
    def test_reads_every_frame_of_a_real_video(self, tmp_path, fake_landmarkers) -> None:
        write_video(tmp_path / "clip.mp4", frames=12)
        clip = extract_video(tmp_path / "clip.mp4")

        assert clip.frames == 12
        assert clip.detection_rate == 1.0
        assert np.allclose(clip.left_hand, 0.5)  # camera "Right" → signer's left
        assert np.allclose(clip.pose, 0.2)

    def test_stride_keeps_every_nth_frame(self, tmp_path, fake_landmarkers) -> None:
        write_video(tmp_path / "clip.mp4", frames=12)
        assert extract_video(tmp_path / "clip.mp4", stride=3).frames == 4

    def test_stride_preserves_real_frame_spacing(self, tmp_path, fake_landmarkers) -> None:
        """Timestamps track source position, not a counter.

        Feeding 0, 33, 66… for frames that are actually 100 ms apart would tell MediaPipe
        the signer moves three times faster than they do, degrading its tracking.
        """
        write_video(tmp_path / "clip.mp4", frames=12, fps=30.0)
        extract_video(tmp_path / "clip.mp4", stride=3)
        assert fake_landmarkers["hand"].timestamps == [0, 100, 200, 300]

    def test_timestamps_strictly_increase(self, tmp_path, fake_landmarkers) -> None:
        """MediaPipe's video mode rejects a frame that does not advance the clock."""
        write_video(tmp_path / "clip.mp4", frames=12)
        extract_video(tmp_path / "clip.mp4")
        stamps = fake_landmarkers["hand"].timestamps
        assert all(b > a for a, b in zip(stamps, stamps[1:]))

    def test_max_frames_stops_early(self, tmp_path, fake_landmarkers) -> None:
        write_video(tmp_path / "clip.mp4", frames=12)
        assert extract_video(tmp_path / "clip.mp4", max_frames=5).frames == 5

    def test_landmarkers_are_closed_even_when_the_run_fails(
        self, tmp_path, fake_landmarkers
    ) -> None:
        """Leaking a landmarker per clip exhausts memory partway through a large corpus."""
        write_video(tmp_path / "clip.mp4", frames=4)
        extract_video(tmp_path / "clip.mp4")
        assert fake_landmarkers["hand"].closed
        assert fake_landmarkers["pose"].closed

    def test_no_pose_skips_the_pose_landmarker(self, tmp_path, fake_landmarkers) -> None:
        write_video(tmp_path / "clip.mp4", frames=4)
        clip = extract_video(tmp_path / "clip.mp4", with_pose=False)
        assert fake_landmarkers["pose"] is None
        assert np.isnan(clip.pose).all()

    def test_rejects_a_nonsense_stride(self, tmp_path, fake_landmarkers) -> None:
        write_video(tmp_path / "clip.mp4", frames=4)
        with pytest.raises(ValueError, match="stride"):
            extract_video(tmp_path / "clip.mp4", stride=0)

    def test_unopenable_file_raises(self, tmp_path, fake_landmarkers) -> None:
        (tmp_path / "broken.mp4").write_bytes(b"not a video")
        with pytest.raises(ValueError, match="Could not open"):
            extract_video(tmp_path / "broken.mp4")


class TestExtractDirectory:
    def test_labels_come_from_the_parent_directory(self, tmp_path, fake_landmarkers) -> None:
        (tmp_path / "videos/hello").mkdir(parents=True)
        write_video(tmp_path / "videos/hello/a.mp4", frames=4)

        counts = extract_directory(tmp_path / "videos", tmp_path / "out")

        assert counts["written"] == 1
        stored = np.load(tmp_path / "out/hello/a.npz")
        assert str(stored["label"]) == "hello"
        assert str(stored["dominant"]) == "left"
        assert stored["left_hand"].dtype == np.float32

    def test_low_detection_clips_are_skipped_and_counted(
        self, tmp_path, monkeypatch, fake_landmarkers
    ) -> None:
        """Silently including them would teach the model mostly about missing data."""
        (tmp_path / "videos/hello").mkdir(parents=True)
        write_video(tmp_path / "videos/hello/a.mp4", frames=4)

        def create(*, with_pose: bool):
            return FakeLandmarker(lambda _: FakeHandResult([], [])), None

        monkeypatch.setattr(video_module, "_create_landmarkers", create)
        counts = extract_directory(tmp_path / "videos", tmp_path / "out", with_pose=False)

        assert counts == {"written": 0, "skipped_low_detection": 1, "failed": 0}

    def test_one_bad_file_does_not_stop_the_run(self, tmp_path, fake_landmarkers) -> None:
        (tmp_path / "videos/hello").mkdir(parents=True)
        (tmp_path / "videos/hello/broken.mp4").write_bytes(b"not a video")
        write_video(tmp_path / "videos/hello/good.mp4", frames=4)

        counts = extract_directory(tmp_path / "videos", tmp_path / "out")

        assert counts["written"] == 1
        assert counts["failed"] == 1

    def test_a_missing_bundle_stops_the_whole_run(self, tmp_path, monkeypatch) -> None:
        """Every file would fail identically; 5000 identical log lines help nobody."""
        (tmp_path / "videos/hello").mkdir(parents=True)
        write_video(tmp_path / "videos/hello/a.mp4", frames=4)
        monkeypatch.setenv(MODEL_DIR_ENV, str(tmp_path / "empty"))

        with pytest.raises(FileNotFoundError, match="setup:assets"):
            extract_directory(tmp_path / "videos", tmp_path / "out")
