"""Export the continuous fingerspelling model as a model pack.

Same contract as the static pack (:mod:`mudra_train.export.pack`): a manifest carrying
the operating point and metrics, a single self-contained ONNX file, and a model card.
The differences are that the task is ``temporal-ctc``, the input has a frame axis, and
the metric is character error rate rather than macro-F1.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import torch

from ..features.normalise import FEATURE_LENGTH, NORMALISATION_SCHEME
from ..models.temporal import BLANK_INDEX, FingerspellCtc
from .pack import OPSET, _inline_external_data, _strip_initializer_value_info, quantise_int8, sha256_of

#: Frames the browser buffers before decoding. About two seconds at 30 fps — long
#: enough to hold a short word, short enough to keep inference near 10 ms.
WINDOW_FRAMES = 64


def export_ctc_onnx(model: FingerspellCtc, path: Path, feature_length: int) -> None:
    """Export with both batch and frame axes dynamic.

    The frame axis must be dynamic because the browser's window is not full until a
    couple of seconds in, and a fixed-length graph would force zero-padding that the
    model was never trained to ignore.
    """
    model.eval()
    example = torch.randn(1, WINDOW_FRAMES, feature_length)

    # `Dim.AUTO` rather than explicit bounds: the channel-attention block pools across
    # frames, and hand-written ranges make the exporter derive a contradictory
    # constraint from that reduction ("Invalid ranges [2:1]"). Letting torch infer the
    # bounds keeps both axes dynamic, which `verify_ctc_onnx` then checks at two
    # different window lengths.
    torch.onnx.export(
        model,
        (example,),
        str(path),
        input_names=["features"],
        output_names=["log_probabilities"],
        dynamic_shapes={"x": {0: torch.export.Dim.AUTO, 1: torch.export.Dim.AUTO}},
        opset_version=OPSET,
        do_constant_folding=True,
    )

    _inline_external_data(path)
    _strip_initializer_value_info(path)


def verify_ctc_onnx(
    model: FingerspellCtc, path: Path, feature_length: int, tolerance: float = 5e-3
) -> float:
    """Check the exported graph reproduces PyTorch, at two different window lengths.

    Testing a second length is the point: a graph that silently baked in the example
    window would pass at 64 frames and fail at 40, which is exactly what the browser
    sends while its buffer is filling.
    """
    import onnxruntime as ort

    model.eval()
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(1234)
    worst = 0.0

    for frames in (WINDOW_FRAMES, 40):
        sample = rng.normal(0, 1, size=(1, frames, feature_length)).astype(np.float32)
        with torch.no_grad():
            expected = model(torch.from_numpy(sample)).numpy()
        actual = session.run(["log_probabilities"], {"features": sample})[0]
        worst = max(worst, float(np.max(np.abs(expected - actual))))

    if worst > tolerance:
        raise RuntimeError(
            f"CTC ONNX export does not match PyTorch: max delta {worst:.2e} "
            f"exceeds {tolerance:.0e}. Refusing to ship a model that changed on export."
        )
    return worst


def build_ctc_pack(
    *,
    model: FingerspellCtc,
    classes: list[str],
    cer: float,
    exact_match: float,
    test_signers: list[str],
    output: Path,
    synthetic: bool = False,
    language: str = "ase",
) -> dict:
    """Write manifest, weights and card for a continuous fingerspelling pack."""
    output.mkdir(parents=True, exist_ok=True)
    float_path = output / "model.fp32.onnx"
    model_path = output / "model.onnx"

    export_ctc_onnx(model, float_path, FEATURE_LENGTH)
    delta = verify_ctc_onnx(model, float_path, FEATURE_LENGTH)

    quantise_int8(float_path, model_path)
    verify_ctc_onnx(model, model_path, FEATURE_LENGTH, tolerance=3.0)
    float_size = float_path.stat().st_size
    float_path.unlink()

    pack_id = "synthetic-fingerspell-ctc" if synthetic else "asl-fingerspell-ctc"

    # CER maps to the manifest's macro-F1 slot as (1 - CER) so a single gate can
    # compare packs of either task type. The card reports CER directly.
    quality = max(0.0, 1.0 - cer)

    manifest = {
        "id": pack_id,
        "version": "1.0.0",
        "name": ("Synthetic CTC (pipeline test)" if synthetic else "ASL Continuous Fingerspelling"),
        "language": language,
        "task": "temporal-ctc",
        "labels": classes,
        "input": {
            "featureLength": FEATURE_LENGTH,
            "hands": 1,
            "normalisation": NORMALISATION_SCHEME,
            "windowFrames": WINDOW_FRAMES,
        },
        # CTC commits on blank transitions and LocalAgreement, not on a probability
        # threshold, so the per-frame gates are permissive by design.
        "thresholds": {"minProbability": 0.0, "minMargin": 0.0, "maxEnergy": None},
        "metrics": {
            "macroF1": round(quality, 6),
            "worstSliceF1": round(quality, 6),
            "testSigners": len(test_signers),
        },
        "modelFile": "model.onnx",
        "sha256": sha256_of(model_path),
        "licence": "synthetic (no real data)" if synthetic else "see attribution",
    }

    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    lines = [f"# Model card — {pack_id}", "", f"Built {date.today().isoformat()}.", ""]
    if synthetic:
        lines += [
            "> ## ⚠ NOT A USABLE MODEL",
            ">",
            "> Trained on procedurally generated sequences, not real fingerspelling.",
            "> It exists to prove the CTC pipeline runs end to end.",
            "",
        ]
    lines += [
        "## Intended use",
        "",
        "Continuous ASL fingerspelling from MediaPipe hand landmarks, decoded with CTC.",
        "Unlike the static pack, this handles J and Z natively — they are motion-defined,",
        "and a temporal model sees motion — and can spell doubled letters.",
        "",
        "## Architecture",
        "",
        f"- Causal depthwise-separable Conv1D encoder, {model.parameter_count:,} parameters.",
        f"- Input: {WINDOW_FRAMES}-frame window × {FEATURE_LENGTH} features, `{NORMALISATION_SCHEME}`.",
        f"- Output: {len(classes)} classes (blank at index {BLANK_INDEX}).",
        "- Causal throughout, so the same weights serve offline and streaming decoding.",
        "",
        "## Evaluation",
        "",
        "Signer-independent. Scored with character error rate, which penalises",
        "misrecognition, double-firing and misses together — per-frame accuracy",
        "penalises none of them.",
        "",
        f"- Held-out signers: {', '.join(test_signers) or 'none'}",
        f"- Character error rate: **{cer:.4f}**",
        f"- Exact-match rate: {exact_match:.1%}",
        "",
        "## Limitations",
        "",
        "- Fingerspelling only. Word-level signs are Phase 4.",
        "- Latency is bounded by the decode stride and the LocalAgreement window; text",
        "  settles a few hundred milliseconds behind the hand.",
        "- A very long pause mid-word flushes the buffer and starts a new word.",
        "",
    ]
    (output / "card.md").write_text("\n".join(lines), encoding="utf-8")

    return {
        "manifest": manifest,
        "onnxDelta": delta,
        "sizeBytes": model_path.stat().st_size,
        "floatSizeBytes": float_size,
    }
