"""Pipeline tests: augmentation, splits, metrics, and export.

These run on synthetic fixtures so they are fast and deterministic. They check the
machinery, not model quality — quality is a data question and is reported by the
evaluation harness itself.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mudra_train.eval.metrics import (
    evaluate,
    expected_calibration_error,
    rejection_auroc,
    split_by_signer,
)
from mudra_train.features.augment import AugmentConfig, augment_batch
from mudra_train.features.normalise import normalise_batch
from mudra_train.ingest import synthetic
from mudra_train.ingest.recorder import Sample, from_samples, read_jsonl


# ── Ingest ────────────────────────────────────────────────────────────────────


def test_reads_recorder_jsonl(tmp_path: Path) -> None:
    record = {
        "label": "A",
        "signer": "aniket-01",
        "hand": "right",
        "condition": "normal",
        "timestampMs": 100,
        "landmarks": [0.1] * 63,
    }
    path = tmp_path / "sample.jsonl"
    path.write_text(json.dumps(record) + "\n\n", encoding="utf-8")

    samples = list(read_jsonl(path))
    assert len(samples) == 1
    assert samples[0].label == "A"
    assert samples[0].landmarks.shape == (21, 3)


def test_rejects_truncated_landmarks(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"label": "A", "landmarks": [0.1] * 10}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="landmark values"):
        list(read_jsonl(path))


def test_none_sorts_last_so_label_indices_are_stable() -> None:
    """A new letter must not renumber the `none` class a pack already refers to."""
    samples = [
        Sample("B", "s1", "right", "normal", np.zeros((21, 3))),
        Sample("none", "s1", "right", "normal", np.zeros((21, 3))),
        Sample("A", "s1", "right", "normal", np.zeros((21, 3))),
    ]
    assert from_samples(samples).classes == ["A", "B", "none"]


def test_synthetic_dataset_has_the_shape_training_expects() -> None:
    dataset = synthetic.generate(samples_per_class=24, signers=4)
    assert len(dataset) > 0
    assert dataset.landmarks.shape[1:] == (21, 3)
    assert "none" in dataset.classes
    assert len(set(dataset.signers.tolist())) == 4
    # Both hands must appear, or the handedness slice cannot be evaluated.
    assert {"left", "right"} <= set(dataset.hands.tolist())


# ── Splits ────────────────────────────────────────────────────────────────────


def test_split_is_signer_independent() -> None:
    signers = np.array([f"s{i // 10}" for i in range(80)])
    train, test, held_out = split_by_signer(signers, test_fraction=0.25)

    assert set(signers[train]).isdisjoint(set(signers[test]))
    assert set(held_out) == set(signers[test].tolist())
    assert len(train) + len(test) == len(signers)


def test_split_refuses_a_single_signer() -> None:
    """The check that stops the v1 mistake being repeated."""
    with pytest.raises(ValueError, match="at least 2 signers"):
        split_by_signer(np.array(["only-me"] * 50))


def test_split_is_deterministic() -> None:
    signers = np.array([f"s{i // 5}" for i in range(50)])
    first = split_by_signer(signers, seed=1)[2]
    second = split_by_signer(signers, seed=1)[2]
    assert first == second


# ── Augmentation ──────────────────────────────────────────────────────────────


def test_mirroring_flips_the_handedness_label_with_the_geometry() -> None:
    """Mirroring without flipping the label teaches the exact opposite invariance."""
    rng = np.random.default_rng(3)
    landmarks = rng.uniform(0.2, 0.8, size=(200, 21, 3))
    hands = np.array(["right"] * 200)

    config = AugmentConfig(
        mirror_probability=1.0,
        rotation_probability=0.0,
        scale_probability=0.0,
        shear_probability=0.0,
        jitter_probability=0.0,
        dropout_probability=0.0,
    )
    augmented, augmented_hands = augment_batch(landmarks, hands, rng, config)

    assert (augmented_hands == "left").all()
    # After normalisation the mirrored-and-relabelled hand is identical to the original.
    before = normalise_batch(landmarks, hands)
    after = normalise_batch(augmented, augmented_hands)
    assert np.max(np.abs(before - after)) < 1e-5


def test_augmentation_preserves_shape_and_stays_finite() -> None:
    rng = np.random.default_rng(5)
    landmarks = rng.uniform(0.2, 0.8, size=(64, 21, 3))
    hands = np.array(["right"] * 64)

    augmented, augmented_hands = augment_batch(landmarks, hands, rng)
    assert augmented.shape == landmarks.shape
    assert len(augmented_hands) == 64
    assert np.isfinite(augmented).all()


def test_rotation_stays_within_bounds() -> None:
    """Unbounded rotation would relabel the data: K becomes P, G becomes Q."""
    assert AugmentConfig().max_rotation_degrees <= 45


def test_augmentation_actually_changes_the_data() -> None:
    rng = np.random.default_rng(9)
    landmarks = rng.uniform(0.2, 0.8, size=(64, 21, 3))
    hands = np.array(["right"] * 64)
    augmented, _ = augment_batch(landmarks, hands, rng)
    assert np.max(np.abs(augmented - landmarks)) > 1e-3


# ── Metrics ───────────────────────────────────────────────────────────────────


def test_perfect_predictions_score_one() -> None:
    true = np.array([0, 1, 2, 0, 1, 2])
    probabilities = np.eye(3)[true]
    report = evaluate(true, true, probabilities, ["A", "B", "none"])
    assert report.macro_f1 == pytest.approx(1.0)
    assert report.accuracy == pytest.approx(1.0)


def test_macro_f1_exposes_a_class_that_never_fires() -> None:
    """Accuracy would look healthy here; macro-F1 is what reveals the dead class."""
    true = np.array([0] * 90 + [1] * 10)
    predicted = np.zeros(100, dtype=int)  # class 1 never predicted
    probabilities = np.eye(2)[predicted]

    report = evaluate(true, predicted, probabilities, ["A", "B"])
    assert report.accuracy == pytest.approx(0.90)
    assert report.macro_f1 < 0.55


def test_slices_are_reported_separately() -> None:
    true = np.array([0, 0, 1, 1])
    predicted = np.array([0, 0, 0, 0])
    probabilities = np.eye(2)[predicted]
    left = np.array([True, True, False, False])

    report = evaluate(
        true, predicted, probabilities, ["A", "B"], slices={"hand=left": left, "hand=right": ~left}
    )
    assert {s.name for s in report.slices} == {"hand=left", "hand=right"}
    worst = report.worst_slice
    assert worst is not None and worst.name == "hand=right"


def test_calibration_error_detects_overconfidence() -> None:
    # Claims 99% confidence, right half the time.
    probabilities = np.tile(np.array([0.99, 0.01]), (100, 1))
    correct = np.array([1.0] * 50 + [0.0] * 50)
    assert expected_calibration_error(probabilities, correct) > 0.4


def test_calibration_error_is_low_when_honest() -> None:
    probabilities = np.tile(np.array([0.6, 0.4]), (100, 1))
    correct = np.array([1.0] * 60 + [0.0] * 40)
    assert expected_calibration_error(probabilities, correct) < 0.05


def test_rejection_auroc_ranks_known_above_unknown() -> None:
    scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    is_known = np.array([True, True, True, False, False])
    assert rejection_auroc(scores, is_known) == pytest.approx(1.0)


def test_rejection_auroc_is_half_for_random_scores() -> None:
    rng = np.random.default_rng(21)
    scores = rng.uniform(size=2000)
    is_known = rng.random(2000) < 0.5
    assert rejection_auroc(scores, is_known) == pytest.approx(0.5, abs=0.05)


def test_confusion_matrix_counts_mistakes() -> None:
    true = np.array([0, 0, 1])
    predicted = np.array([0, 1, 1])
    report = evaluate(true, predicted, np.eye(2)[predicted], ["A", "B"])
    assert report.top_confusions()[0] == ("A", "B", 1)
