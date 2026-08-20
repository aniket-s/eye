"""Export tests: ONNX round-trip and quantisation.

Export is not a no-op. Operator differences, constant folding and quantisation can all
change results, and an unverified export is a silent accuracy regression. These tests
run the real toolchain, so the build fails rather than shipping a changed model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from mudra_train.eval.metrics import evaluate
from mudra_train.export.pack import build_pack, choose_thresholds, verify_onnx
from mudra_train.features.normalise import FEATURE_LENGTH, NORMALISATION_SCHEME
from mudra_train.models.classifier import HandshapeClassifier


@pytest.fixture(scope="module")
def model() -> HandshapeClassifier:
    torch.manual_seed(0)
    return HandshapeClassifier(classes=5, feature_length=FEATURE_LENGTH).eval()


def test_model_is_small_enough_for_the_frame_budget(model: HandshapeClassifier) -> None:
    """Under 10 ms is the budget left after MediaPipe; that caps the parameter count."""
    assert model.parameter_count < 500_000


def test_embeddings_lie_on_the_unit_sphere(model: HandshapeClassifier) -> None:
    """Required for cosine similarity to a class centre to be meaningful — and for the
    few-shot custom-sign feature that depends on it."""
    with torch.no_grad():
        embeddings = model.embed(torch.randn(32, FEATURE_LENGTH))
    norms = embeddings.norm(dim=-1).numpy()
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_margin_applies_only_during_training(model: HandshapeClassifier) -> None:
    """The exported graph must not contain the training-only ArcFace margin."""
    features = torch.randn(8, FEATURE_LENGTH)
    targets = torch.randint(0, 5, (8,))
    with torch.no_grad():
        inference = model(features)
        training = model(features, targets)
    assert not torch.allclose(inference, training)


def report_for(classes: list[str]) -> object:
    count = len(classes)
    true = np.arange(count).repeat(4)
    probabilities = np.eye(count)[true]
    return evaluate(true, true, probabilities, classes, test_signers=["a", "b"])


def test_pack_round_trips_through_onnx_and_quantisation(
    model: HandshapeClassifier, tmp_path: Path
) -> None:
    classes = ["A", "B", "C", "D", "none"]
    result = build_pack(
        model=model, classes=classes, report=report_for(classes), output=tmp_path, synthetic=True
    )

    assert (tmp_path / "model.onnx").exists()
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "card.md").exists()
    # A pack must be exactly the files its manifest lists — no external weight sidecar.
    assert not list(tmp_path.glob("*.onnx.data"))
    assert not (tmp_path / "model.fp32.onnx").exists()

    # fp32 export must be near-exact; int8 is lossy but bounded.
    assert result["onnxDelta"] < 2e-4
    assert result["quantisedDelta"] < 2.0
    # Quantisation should be a real saving, not a rounding error.
    assert result["sizeBytes"] < result["floatSizeBytes"] * 0.5


def test_manifest_matches_what_the_typescript_loader_requires(
    model: HandshapeClassifier, tmp_path: Path
) -> None:
    classes = ["A", "B", "none"]
    build_pack(
        model=model, classes=classes, report=report_for(classes), output=tmp_path, synthetic=True
    )
    manifest = json.loads((tmp_path / "manifest.json").read_text())

    assert manifest["input"]["normalisation"] == NORMALISATION_SCHEME
    assert manifest["input"]["featureLength"] == FEATURE_LENGTH
    assert manifest["labels"] == classes
    assert manifest["metrics"]["testSigners"] >= 1
    assert set(manifest["thresholds"]) == {"minProbability", "minMargin", "maxEnergy"}
    assert len(manifest["sha256"]) == 64


def test_synthetic_card_is_labelled_unusable(model: HandshapeClassifier, tmp_path: Path) -> None:
    classes = ["A", "none"]
    build_pack(
        model=model, classes=classes, report=report_for(classes), output=tmp_path, synthetic=True
    )
    assert "NOT A USABLE MODEL" in (tmp_path / "card.md").read_text()


def test_thresholds_tighten_when_the_model_is_poorly_calibrated() -> None:
    class Report:
        ece = 0.2

    class Honest:
        ece = 0.01

    assert choose_thresholds(Report()).get("minProbability") > choose_thresholds(Honest()).get(
        "minProbability"
    )


def test_verify_rejects_a_mismatched_graph(model: HandshapeClassifier, tmp_path: Path) -> None:
    """The guard itself must work, or it protects nothing."""
    from mudra_train.export.pack import export_onnx

    path = tmp_path / "m.onnx"
    export_onnx(model, path, FEATURE_LENGTH)

    other = HandshapeClassifier(classes=5, feature_length=FEATURE_LENGTH).eval()
    with pytest.raises(RuntimeError, match="does not match PyTorch"):
        verify_onnx(other, path, FEATURE_LENGTH)
