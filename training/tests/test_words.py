"""Tests for the word-level pipeline: model, synthetic clips, and export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from mudra_train.eval.metrics import evaluate
from mudra_train.export.isolated_pack import CLIP_FRAMES, build_isolated_pack, verify_isolated_onnx
from mudra_train.features.normalise_body import BODY_FEATURE_LENGTH
from mudra_train.ingest import words
from mudra_train.models.isolated import IsolatedSignClassifier, motion_features
from mudra_train.train_words import resample


@pytest.fixture(scope="module")
def model() -> IsolatedSignClassifier:
    torch.manual_seed(0)
    return IsolatedSignClassifier(classes=15, feature_length=BODY_FEATURE_LENGTH).eval()


def test_motion_features_triple_the_width() -> None:
    x = torch.randn(2, 30, BODY_FEATURE_LENGTH)
    assert motion_features(x).shape == (2, 30, BODY_FEATURE_LENGTH * 3)


def test_motion_features_capture_velocity() -> None:
    """Position alone cannot separate signs sharing a handshape but differing in movement."""
    still = torch.zeros(1, 10, 4)
    moving = torch.arange(10, dtype=torch.float32).reshape(1, 10, 1).repeat(1, 1, 4)

    still_deltas = motion_features(still)[:, :, 4:8]
    moving_deltas = motion_features(moving)[:, :, 4:8]

    assert torch.allclose(still_deltas, torch.zeros_like(still_deltas))
    assert moving_deltas[0, 5, 0] == pytest.approx(1.0)


def test_motion_features_do_not_change_the_frame_count() -> None:
    assert motion_features(torch.randn(1, 7, 3)).shape[1] == 7


def test_model_size_is_reasonable(model: IsolatedSignClassifier) -> None:
    """Runs once per sign rather than per frame, so the budget is looser than the
    fingerspelling models — but it still ships to a browser."""
    assert model.parameter_count < 5_000_000


def test_output_shape_is_one_row_per_clip(model: IsolatedSignClassifier) -> None:
    assert model(torch.randn(4, CLIP_FRAMES, BODY_FEATURE_LENGTH)).shape == (4, 15)


def test_accepts_clips_of_different_lengths(model: IsolatedSignClassifier) -> None:
    """The browser sends short signs as-is and downsamples long ones."""
    for frames in (24, 48, CLIP_FRAMES):
        assert model(torch.randn(1, frames, BODY_FEATURE_LENGTH)).shape == (1, 15)


def test_reversing_a_clip_changes_the_prediction(model: IsolatedSignClassifier) -> None:
    """Movement direction carries meaning, so the model must not be time-symmetric."""
    torch.manual_seed(3)
    clip = torch.randn(1, 40, BODY_FEATURE_LENGTH)
    with torch.no_grad():
        original = model(clip)
        reversed_logits = model(torch.flip(clip, dims=[1]))
    assert not torch.allclose(original, reversed_logits, atol=1e-3)


def test_synthetic_clips_have_the_expected_structure() -> None:
    dataset = words.generate_synthetic(clips_per_sign=6, signers=3)
    assert len(dataset) > 0
    assert len(dataset.classes) == 15
    assert dataset.dominant_hands[0].shape[1:] == (21, 3)
    assert dataset.poses[0] is not None


def test_synthetic_clips_cover_both_dominances() -> None:
    dataset = words.generate_synthetic(clips_per_sign=12, signers=3)
    assert {"left", "right"} <= set(dataset.dominants.tolist())


def test_two_handed_signs_populate_the_second_hand() -> None:
    """One-handed signs leave the other hand NaN; two-handed ones must not."""
    dataset = words.generate_synthetic(clips_per_sign=6, signers=3)
    populated = [not np.isnan(other).all() for other in dataset.other_hands]
    assert any(populated), "no two-handed clips generated"
    assert not all(populated), "no one-handed clips generated"


def test_clips_vary_in_length() -> None:
    dataset = words.generate_synthetic(clips_per_sign=18, signers=3)
    assert len({len(c) for c in dataset.dominant_hands}) > 4


def test_resample_hits_the_target_length() -> None:
    for source in (20, 96, 250):
        assert len(resample(np.zeros((source, BODY_FEATURE_LENGTH)), 96)) == 96


def test_resample_keeps_the_first_and_last_frame() -> None:
    """Truncation would discard a sign's ending, which often carries the meaning."""
    features = np.arange(200, dtype=np.float32).reshape(200, 1)
    out = resample(features, 96)
    assert out[0, 0] == pytest.approx(0.0)
    assert out[-1, 0] == pytest.approx(199.0)


def test_export_handles_short_and_full_clips(model: IsolatedSignClassifier, tmp_path: Path) -> None:
    from mudra_train.export.isolated_pack import export_isolated_onnx

    path = tmp_path / "m.onnx"
    export_isolated_onnx(model, path, BODY_FEATURE_LENGTH)
    assert verify_isolated_onnx(model, path, BODY_FEATURE_LENGTH) < 5e-3


def test_word_pack_declares_the_two_handed_scheme(
    model: IsolatedSignClassifier, tmp_path: Path
) -> None:
    classes = [f"sign{i}" for i in range(15)]
    truth = np.arange(15).repeat(2)
    report = evaluate(truth, truth, np.eye(15)[truth], classes, test_signers=["a", "b"])

    result = build_isolated_pack(
        model=model, classes=classes, report=report, top5=1.0, output=tmp_path, synthetic=True
    )
    manifest = result["manifest"]

    assert manifest["task"] == "temporal-isolated"
    assert manifest["input"]["hands"] == 2
    assert manifest["input"]["normalisation"] == "shoulder-frame-v1"
    assert manifest["input"]["featureLength"] == BODY_FEATURE_LENGTH
    assert manifest["input"]["windowFrames"] == CLIP_FRAMES
    assert not list(tmp_path.glob("*.onnx.data"))

    card = (tmp_path / "card.md").read_text()
    assert "top-5 accuracy" in card
    assert "NOT A USABLE MODEL" in card
