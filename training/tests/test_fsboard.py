"""The FSboard adapter.

FSboard is a large Kaggle download and cannot be fetched in CI, so these build a small
dataset in the same shape and read it back. That covers everything the adapter is
responsible for — schema discovery, dominant-hand selection, phrase filtering, quality
gates — and leaves only "does the real release match this shape" untested, which is
exactly what `--inspect` exists to answer before anyone spends an afternoon on a training
run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

import pandas as pd  # noqa: E402

from mudra_train.ingest.fsboard import (  # noqa: E402
    ALPHABET,
    discover_schema,
    iter_sequences,
    load_fsboard,
    spellable,
)

HAND_GROUPS = ("right_hand", "left_hand")


def landmark_columns() -> list[str]:
    return [
        f"{axis}_{group}_{index}"
        for group in HAND_GROUPS
        for index in range(21)
        for axis in "xyz"
    ]


def write_dataset(
    root: Path,
    clips: list[dict],
    *,
    sequence_column: str = "sequence_id",
    phrase_column: str = "phrase",
    participant_column: str = "participant_id",
) -> None:
    """Build a miniature FSboard: one parquet of landmarks plus a metadata CSV."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "train_landmarks").mkdir(exist_ok=True)

    rows = []
    metadata = []
    for clip in clips:
        frames = clip["frames"]
        for frame in range(frames):
            row = {sequence_column: clip["id"], "frame": frame}
            for group in HAND_GROUPS:
                present = group in clip["hands"] and frame not in clip.get("missing", ())
                for index in range(21):
                    for axis_name, offset in zip("xyz", range(3), strict=True):
                        row[f"{axis_name}_{group}_{index}"] = (
                            0.1 * index + 0.01 * frame + offset if present else np.nan
                        )
            rows.append(row)

        metadata.append(
            {
                "path": "train_landmarks/data.parquet",
                "file_id": "data",
                sequence_column: clip["id"],
                participant_column: clip["signer"],
                phrase_column: clip["phrase"],
            }
        )

    pd.DataFrame(rows).to_parquet(root / "train_landmarks/data.parquet", index=False)
    pd.DataFrame(metadata).to_csv(root / "train.csv", index=False)


@pytest.fixture
def dataset_root(tmp_path) -> Path:
    root = tmp_path / "fsboard"
    write_dataset(
        root,
        [
            {"id": 1, "frames": 30, "hands": ("right_hand",), "signer": "p01", "phrase": "hello"},
            {"id": 2, "frames": 24, "hands": ("left_hand",), "signer": "p02", "phrase": "world"},
            {"id": 3, "frames": 20, "hands": ("right_hand",), "signer": "p03", "phrase": "1600"},
        ],
    )
    return root


class TestSpellable:
    def test_keeps_letters_and_upper_cases_them(self) -> None:
        assert spellable("hello") == list("HELLO")

    def test_drops_digits_and_punctuation(self) -> None:
        """FSboard prompts are addresses, phone numbers and URLs.

        Mapping a digit onto a nearby letter would quietly teach the model something
        false, so unspellable characters are removed rather than approximated.
        """
        assert spellable("1600 Penn. Ave") == list("PENNAVE")

    def test_a_phrase_with_no_letters_yields_nothing(self) -> None:
        assert spellable("555-0100") == []

    def test_covers_the_whole_alphabet_including_the_motion_letters(self) -> None:
        """J and Z belong here. They are impossible for a static classifier and ordinary
        for a sequence model, which is why FSboard trains a CTC pack."""
        assert spellable(ALPHABET.lower()) == list(ALPHABET)
        assert "J" in ALPHABET and "Z" in ALPHABET


class TestSchemaDiscovery:
    def test_finds_the_metadata_and_landmark_columns(self, dataset_root) -> None:
        schema = discover_schema(dataset_root)
        assert schema.participant_column == "participant_id"
        assert schema.phrase_column == "phrase"
        assert schema.sequence_column == "sequence_id"
        assert schema.groups == {"right_hand": 21, "left_hand": 21}
        assert schema.has_hands

    def test_accepts_alternative_column_names(self, tmp_path) -> None:
        """The adapter cannot be tested against the real release, so it discovers rather
        than assumes."""
        root = tmp_path / "variant"
        write_dataset(
            root,
            [{"id": 1, "frames": 20, "hands": ("right_hand",), "signer": "s1", "phrase": "abc"}],
            sequence_column="sequence",
            phrase_column="text",
            participant_column="signer_id",
        )
        schema = discover_schema(root)
        assert schema.participant_column == "signer_id"
        assert schema.phrase_column == "text"

    def test_names_the_columns_it_saw_when_it_cannot_find_one(self, tmp_path) -> None:
        """A dataset adapter that dies with `KeyError` after ten minutes of loading is
        worse than useless."""
        root = tmp_path / "wrong"
        write_dataset(
            root,
            [{"id": 1, "frames": 20, "hands": ("right_hand",), "signer": "s", "phrase": "a"}],
            participant_column="who",
        )
        with pytest.raises(ValueError, match="signer column"):
            discover_schema(root)

    def test_explains_a_missing_download(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match="download"):
            discover_schema(tmp_path / "not-here")

    def test_explains_a_directory_with_no_metadata(self, tmp_path) -> None:
        (tmp_path / "empty").mkdir()
        with pytest.raises(FileNotFoundError, match="CSV"):
            discover_schema(tmp_path / "empty")

    def test_describe_mentions_every_detected_field(self, dataset_root) -> None:
        described = discover_schema(dataset_root).describe()
        for expected in ("participant_id", "phrase", "sequence_id", "right_hand"):
            assert expected in described


class TestLoading:
    def test_reads_clips_with_their_letters_and_signers(self, dataset_root) -> None:
        clips = list(iter_sequences(dataset_root))
        assert len(clips) == 2  # the numeric phrase has no spellable letters

        by_signer = {signer: (points, letters, hand) for points, letters, signer, hand in clips}
        assert set(by_signer) == {"p01", "p02"}
        assert by_signer["p01"][1] == list("HELLO")

    def test_picks_whichever_hand_the_signer_used(self, dataset_root) -> None:
        """Assuming right-dominance would mirror every left-handed signer into the wrong
        slot — the model still trains, just badly."""
        hands = {signer: hand for _, _, signer, hand in iter_sequences(dataset_root)}
        assert hands["p01"] == "right"
        assert hands["p02"] == "left"

    def test_landmarks_come_back_in_mediapipe_shape(self, dataset_root) -> None:
        points, _, _, _ = next(iter_sequences(dataset_root))
        assert points.ndim == 3
        assert points.shape[1:] == (21, 3)
        assert np.isfinite(points).all(), "NaN must not reach the trainer"

    def test_drops_frames_where_the_hand_vanished(self, tmp_path) -> None:
        """A gap is not a trajectory. Interpolating across it would invent motion that
        never happened."""
        root = tmp_path / "gaps"
        write_dataset(
            root,
            [
                {
                    "id": 1,
                    "frames": 30,
                    "hands": ("right_hand",),
                    "signer": "p01",
                    "phrase": "hello",
                    "missing": tuple(range(5)),
                }
            ],
        )
        points, _, _, _ = next(iter_sequences(root))
        assert len(points) == 25

    def test_skips_clips_that_are_mostly_untracked(self, tmp_path) -> None:
        """Training on them teaches the model mostly about missing data."""
        root = tmp_path / "sparse"
        write_dataset(
            root,
            [
                {
                    "id": 1,
                    "frames": 30,
                    "hands": ("right_hand",),
                    "signer": "p01",
                    "phrase": "hello",
                    "missing": tuple(range(25)),
                }
            ],
        )
        assert list(iter_sequences(root)) == []

    def test_skips_clips_too_short_to_hold_their_phrase(self, tmp_path) -> None:
        root = tmp_path / "short"
        write_dataset(
            root,
            [{"id": 1, "frames": 3, "hands": ("right_hand",), "signer": "p", "phrase": "hello"}],
        )
        assert list(iter_sequences(root)) == []

    def test_limit_stops_early(self, dataset_root) -> None:
        """FSboard is large; trying something on a subset should not cost an afternoon."""
        assert len(list(iter_sequences(dataset_root, limit=1))) == 1


class TestSequenceDataset:
    def test_builds_a_dataset_the_ctc_trainer_accepts(self, dataset_root) -> None:
        dataset = load_fsboard(dataset_root)
        assert len(dataset) == 2
        assert len(dataset.labels) == 2
        assert set(dataset.signers.tolist()) == {"p01", "p02"}
        assert dataset.sequences[0].dtype == np.float32

    def test_keeps_real_signer_identities(self, dataset_root) -> None:
        """The signer-independent split is only meaningful if these are real people."""
        dataset = load_fsboard(dataset_root)
        assert len(set(dataset.signers.tolist())) == 2

    def test_refuses_to_return_an_empty_dataset(self, tmp_path) -> None:
        """Which would train happily and produce nonsense."""
        root = tmp_path / "unusable"
        write_dataset(
            root,
            [{"id": 1, "frames": 20, "hands": ("right_hand",), "signer": "p", "phrase": "1234"}],
        )
        with pytest.raises(ValueError, match="No usable clips"):
            load_fsboard(root)
