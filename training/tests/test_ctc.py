"""Tests for the temporal model, CTC decoding, and sequence data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from mudra_train.export.ctc_pack import WINDOW_FRAMES, build_ctc_pack, export_ctc_onnx, verify_ctc_onnx
from mudra_train.features.normalise import FEATURE_LENGTH
from mudra_train.ingest import sequences
from mudra_train.models.temporal import BLANK_INDEX, FingerspellCtc, ctc_loss, greedy_decode
from mudra_train.train_ctc import levenshtein


@pytest.fixture(scope="module")
def model() -> FingerspellCtc:
    torch.manual_seed(0)
    return FingerspellCtc(classes=8, feature_length=FEATURE_LENGTH).eval()


def test_model_fits_the_frame_budget(model: FingerspellCtc) -> None:
    """~10 ms is what remains after MediaPipe; that caps the size."""
    assert model.parameter_count < 2_000_000


def test_output_shape_follows_the_input(model: FingerspellCtc) -> None:
    for frames in (16, 64, 100):
        out = model(torch.randn(2, frames, FEATURE_LENGTH))
        assert out.shape == (2, frames, 8)


def test_output_is_log_probabilities(model: FingerspellCtc) -> None:
    out = model(torch.randn(1, 32, FEATURE_LENGTH))
    assert torch.allclose(out.exp().sum(dim=-1), torch.ones(1, 32), atol=1e-5)


def test_encoder_is_causal(model: FingerspellCtc) -> None:
    """Output at frame t must not depend on frames after t.

    This is what lets the same weights serve offline evaluation and live streaming. A
    non-causal model would score well offline and behave differently on a camera.
    """
    torch.manual_seed(1)
    base = torch.randn(1, 48, FEATURE_LENGTH)
    altered = base.clone()
    altered[0, 30:] = torch.randn(18, FEATURE_LENGTH)  # change only the future

    with torch.no_grad():
        first = model(base)
        second = model(altered)

    # Frames before the change must be identical.
    assert torch.allclose(first[0, :30], second[0, :30], atol=1e-5)
    # And the changed region must actually differ, or the test proves nothing.
    assert not torch.allclose(first[0, 30:], second[0, 30:], atol=1e-4)


def test_ctc_loss_is_finite(model: FingerspellCtc) -> None:
    log_probabilities = model(torch.randn(2, 40, FEATURE_LENGTH))
    loss = ctc_loss(
        log_probabilities,
        torch.tensor([1, 2, 3, 4], dtype=torch.long),
        torch.tensor([40, 40]),
        torch.tensor([2, 2]),
    )
    assert torch.isfinite(loss)


def test_greedy_decode_collapses_and_strips() -> None:
    # frames: A A blank A  ->  ["A", "A"]
    log_probabilities = torch.full((1, 4, 3), -10.0)
    log_probabilities[0, 0, 1] = 0.0
    log_probabilities[0, 1, 1] = 0.0
    log_probabilities[0, 2, BLANK_INDEX] = 0.0
    log_probabilities[0, 3, 1] = 0.0
    assert greedy_decode(log_probabilities) == [[1, 1]]


def test_levenshtein_matches_expectations() -> None:
    assert levenshtein([1, 2, 3], [1, 2, 3]) == 0
    assert levenshtein([1, 2, 3], [1, 9, 3]) == 1
    assert levenshtein([], [1, 2]) == 2
    assert levenshtein([1], []) == 1


# ── Sequence data ─────────────────────────────────────────────────────────────


def test_sequences_have_blank_first() -> None:
    """CTC requires blank at index 0, matching BLANK_INDEX and the TypeScript decoder."""
    dataset = sequences.generate(sequences_per_signer=4, signers=2)
    assert dataset.classes[0] == ""


def test_sequences_are_longer_than_their_spelling() -> None:
    """Every letter is held for several frames, plus transitions between them.

    If frames ever fell below label count, CTC could not align and the loss would be
    infinite for those samples.
    """
    dataset = sequences.generate(sequences_per_signer=6, signers=2)
    for sequence, label in zip(dataset.sequences, dataset.labels):
        assert len(sequence) > len(label)


def test_sequences_vary_in_length() -> None:
    """Real signers do not spell at a fixed rate; fixed-length data would not teach that."""
    dataset = sequences.generate(sequences_per_signer=20, signers=3)
    lengths = {len(s) for s in dataset.sequences}
    assert len(lengths) > 5


def test_sequences_include_doubled_letters() -> None:
    """The case a dwell timer cannot express — COFFEE, LETTER."""
    dataset = sequences.generate(sequences_per_signer=40, signers=2)
    assert any(
        any(a == b for a, b in zip(label, label[1:])) for label in dataset.labels
    )


def test_sequences_cover_both_hands() -> None:
    dataset = sequences.generate(sequences_per_signer=20, signers=3)
    assert {"left", "right"} <= set(dataset.hands.tolist())


# ── Export ────────────────────────────────────────────────────────────────────


def test_export_keeps_the_frame_axis_dynamic(model: FingerspellCtc, tmp_path: Path) -> None:
    """The browser feeds a partial window while its buffer fills.

    A graph that baked in the training window would work after two seconds and fail
    before that — verify at two lengths so that cannot ship.
    """
    path = tmp_path / "m.onnx"
    export_ctc_onnx(model, path, FEATURE_LENGTH)
    assert verify_ctc_onnx(model, path, FEATURE_LENGTH) < 5e-3


def test_ctc_pack_round_trips(model: FingerspellCtc, tmp_path: Path) -> None:
    classes = ["", "A", "B", "C", "D", "E", "F", "G"]
    result = build_ctc_pack(
        model=model,
        classes=classes,
        cer=0.08,
        exact_match=0.75,
        test_signers=["a", "b"],
        output=tmp_path,
        synthetic=True,
    )

    assert (tmp_path / "model.onnx").exists()
    assert not list(tmp_path.glob("*.onnx.data"))
    assert not (tmp_path / "model.fp32.onnx").exists()

    manifest = result["manifest"]
    assert manifest["task"] == "temporal-ctc"
    assert manifest["input"]["windowFrames"] == WINDOW_FRAMES
    assert manifest["labels"][0] == ""
    # CER 0.08 maps to a quality score of 0.92.
    assert manifest["metrics"]["macroF1"] == pytest.approx(0.92)
    assert result["sizeBytes"] < result["floatSizeBytes"] * 0.6


def test_ctc_card_reports_cer(model: FingerspellCtc, tmp_path: Path) -> None:
    build_ctc_pack(
        model=model,
        classes=["", "A", "B", "C", "D", "E", "F", "G"],
        cer=0.0812,
        exact_match=0.5,
        test_signers=["a"],
        output=tmp_path,
        synthetic=True,
    )
    card = (tmp_path / "card.md").read_text()
    assert "Character error rate" in card
    assert "0.0812" in card
    assert "NOT A USABLE MODEL" in card
