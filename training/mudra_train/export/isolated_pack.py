"""Export the isolated word-sign model as a model pack.

Same contract as the other two pack builders. The differences: task is
``temporal-isolated``, input carries a frame axis, normalisation is the two-handed
``shoulder-frame-v1`` scheme, and the card reports top-1 and top-5.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import torch

from ..eval.metrics import EvaluationReport
from ..features.normalise_body import BODY_FEATURE_LENGTH, BODY_NORMALISATION_SCHEME
from ..models.isolated import IsolatedSignClassifier
from .pack import (
    OPSET,
    _inline_external_data,
    _strip_initializer_value_info,
    quantise_int8,
    sha256_of,
)

#: Frames per clip. Must match `CLIP_FRAMES` in train_words and `maxFrames` in the
#: browser recogniser — a mismatch would feed the model a clip length it never saw.
CLIP_FRAMES = 96


def export_isolated_onnx(model: IsolatedSignClassifier, path: Path, feature_length: int) -> None:
    """Export with batch and frame axes dynamic."""
    model.eval()
    example = torch.randn(1, CLIP_FRAMES, feature_length)

    torch.onnx.export(
        model,
        (example,),
        str(path),
        input_names=["clip"],
        output_names=["logits"],
        dynamic_shapes={"x": {0: torch.export.Dim.AUTO, 1: torch.export.Dim.AUTO}},
        opset_version=OPSET,
        do_constant_folding=True,
    )

    _inline_external_data(path)
    _strip_initializer_value_info(path)


def verify_isolated_onnx(
    model: IsolatedSignClassifier, path: Path, feature_length: int, tolerance: float = 5e-3
) -> float:
    """Check the export at two clip lengths.

    The browser downsamples long signs and sends short ones as-is, so the graph must
    handle a clip shorter than the training length.
    """
    import onnxruntime as ort

    model.eval()
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(99)
    worst = 0.0

    for frames in (CLIP_FRAMES, 40):
        sample = rng.normal(0, 1, size=(1, frames, feature_length)).astype(np.float32)
        with torch.no_grad():
            expected = model(torch.from_numpy(sample)).numpy()
        actual = session.run(["logits"], {"clip": sample})[0]
        worst = max(worst, float(np.max(np.abs(expected - actual))))

    if worst > tolerance:
        raise RuntimeError(
            f"Isolated ONNX export does not match PyTorch: max delta {worst:.2e} "
            f"exceeds {tolerance:.0e}. Refusing to ship a model that changed on export."
        )
    return worst


def build_isolated_pack(
    *,
    model: IsolatedSignClassifier,
    classes: list[str],
    report: EvaluationReport,
    top5: float,
    output: Path,
    synthetic: bool = False,
    language: str = "ase",
) -> dict:
    """Write manifest, weights and card for a word-level pack."""
    output.mkdir(parents=True, exist_ok=True)
    float_path = output / "model.fp32.onnx"
    model_path = output / "model.onnx"

    export_isolated_onnx(model, float_path, BODY_FEATURE_LENGTH)
    delta = verify_isolated_onnx(model, float_path, BODY_FEATURE_LENGTH)

    quantise_int8(float_path, model_path)
    verify_isolated_onnx(model, model_path, BODY_FEATURE_LENGTH, tolerance=6.0)
    float_size = float_path.stat().st_size
    float_path.unlink()

    pack_id = "synthetic-words" if synthetic else "asl-words"
    worst = report.worst_slice

    manifest = {
        "id": pack_id,
        "version": "1.0.0",
        "name": "Synthetic Words (pipeline test)" if synthetic else "ASL Word Signs",
        "language": language,
        "task": "temporal-isolated",
        "labels": classes,
        "input": {
            "featureLength": BODY_FEATURE_LENGTH,
            "hands": 2,
            "normalisation": BODY_NORMALISATION_SCHEME,
            "windowFrames": CLIP_FRAMES,
        },
        # A 250-class model is confidently wrong on gestures that are not signs, so the
        # probability floor does real work here. The segmenter handles timing.
        "thresholds": {"minProbability": 0.35, "minMargin": 0.0, "maxEnergy": None},
        "metrics": {
            "macroF1": round(report.macro_f1, 6),
            "worstSliceF1": round(worst.macro_f1 if worst else report.macro_f1, 6),
            "testSigners": len(report.test_signers),
        },
        "modelFile": "model.onnx",
        "sha256": sha256_of(model_path),
        "licence": "synthetic (no real data)" if synthetic else "see attribution",
    }
    if not synthetic:
        manifest["attribution"] = "PopSign ASL v1.0 (CC BY 4.0)"

    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    lines = [f"# Model card — {pack_id}", "", f"Built {date.today().isoformat()}.", ""]
    if synthetic:
        lines += [
            "> ## ⚠ NOT A USABLE MODEL",
            ">",
            "> Trained on procedurally generated clips, not recordings of signers.",
            "> It exists to prove the word-level pipeline runs end to end.",
            "",
        ]
    lines += [
        "## Intended use",
        "",
        "Recognising isolated ASL word signs from two-handed MediaPipe landmarks. The app",
        "segments on motion and classifies each completed sign once.",
        "",
        "## Architecture",
        "",
        f"- Conv1D + self-attention encoder, {model.parameter_count:,} parameters.",
        f"- Input: {CLIP_FRAMES}-frame clip × {BODY_FEATURE_LENGTH} features,",
        f"  `{BODY_NORMALISATION_SCHEME}` — both hands and upper-body pose in one",
        "  shoulder-anchored reference frame, so hand position relative to the body is",
        "  preserved. That matters: signs can share a handshape and differ only in location.",
        "- Motion features (first and second temporal differences) are computed inside the",
        "  graph, so the browser cannot compute them differently from training.",
        "",
        "## Evaluation",
        "",
        "Signer-independent.",
        "",
        f"- Held-out signers: {', '.join(report.test_signers) or 'none'}",
        f"- top-1 accuracy: **{report.accuracy:.4f}**",
        f"- top-5 accuracy: {top5:.4f}",
        f"- macro-F1: {report.macro_f1:.4f}",
    ]
    if worst is not None:
        lines.append(f"- Worst slice: {worst.name} at {worst.macro_f1:.4f}")

    confusions = report.top_confusions(8)
    if confusions:
        lines += ["", "### Most frequent confusions", ""]
        lines += [f"- {t} → {p} (×{n})" for t, p, n in confusions]

    lines += [
        "",
        "## Limitations",
        "",
        "- Isolated signs only. Continuous signing is not segmented linguistically; the",
        "  segmenter splits on motion, which is a proxy.",
        "- Facial expression is not modelled. ASL non-manual markers carry grammar, so",
        "  signs distinguished only by expression are not separable here.",
        "- Vocabulary is fixed at training time. Users can add their own signs through the",
        "  few-shot path instead.",
        "- Requires visible shoulders for the body frame; a tight crop falls back to a",
        "  hand-derived frame and loses location information.",
        "",
    ]
    (output / "card.md").write_text("\n".join(lines), encoding="utf-8")

    return {
        "manifest": manifest,
        "onnxDelta": delta,
        "sizeBytes": model_path.stat().st_size,
        "floatSizeBytes": float_size,
    }
