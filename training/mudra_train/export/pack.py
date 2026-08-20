"""Export a trained model as a versioned model pack.

A pack is self-describing: weights, the operating point chosen during evaluation, the
metrics behind that choice, and a model card stating the limitations. Nothing about
how to run the model is left in application code, which is what stops a retrain from
silently invalidating thresholds tuned for different weights.

See ``docs/adr/0003-model-packs.md`` and ``packages/core/src/registry/manifest.ts``.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np
import torch

from ..eval.metrics import EvaluationReport
from ..features.normalise import FEATURE_LENGTH, NORMALISATION_SCHEME
from ..models.classifier import HandshapeClassifier

#: Bumped when the pack layout changes in a way the loader must know about.
PACK_VERSION = "1.0.0"


def choose_thresholds(report: EvaluationReport) -> dict[str, float | None]:
    """Pick an operating point from the evaluation.

    A well-calibrated model can use a lower bar than a poorly calibrated one, so the
    probability threshold is nudged by the measured ECE rather than fixed by habit.
    The margin requirement guards the case max-probability misses entirely: a model
    torn between two letters reports high confidence in the winner while the gap
    between the top two is near zero.
    """
    base = 0.60
    if report.ece is not None and report.ece > 0.10:
        base = 0.75  # confidence is unreliable, so demand more of it
    elif report.ece is not None and report.ece < 0.03:
        base = 0.50

    return {"minProbability": base, "minMargin": 0.15, "maxEnergy": None}


#: ONNX opset. 18 is what PyTorch's exporter emits natively and is comfortably within
#: what ONNX Runtime Web 1.27 supports; pinning it avoids a lossy down-conversion pass.
OPSET = 18


class _ExportWrapper(torch.nn.Module):
    """Expose both logits and the embedding as graph outputs.

    The embedding is what makes **user-defined signs** possible without training: the
    app averages a handful of examples into a class centre and classifies by cosine
    similarity. That only works if the browser can reach the embedding, so it is an
    output of the exported graph rather than an internal activation.

    Costs nothing — the encoder already computes it, and the head consumes it.
    """

    def __init__(self, model: HandshapeClassifier) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.model.encoder(x)
        return self.model.head(embedding), embedding


def export_onnx(model: HandshapeClassifier, path: Path, feature_length: int) -> None:
    """Export inference-only ONNX as a single self-contained file.

    ``target`` is omitted so the ArcFace margin — a training-only path — never enters
    the graph. The batch axis is dynamic so the browser can classify one frame while
    offline evaluation pushes thousands.

    PyTorch may write weights to a sibling ``.data`` file. That is fine on a server and
    wrong for a model pack: the pack's ``sha256`` would cover only the graph, and the
    browser would need a second fetch nobody declared. The weights are folded back in
    below so a pack is always exactly the files its manifest lists.
    """
    model.eval()
    wrapper = _ExportWrapper(model).eval()
    example = torch.randn(1, feature_length)

    torch.onnx.export(
        wrapper,
        (example,),
        str(path),
        input_names=["features"],
        output_names=["logits", "embedding"],
        dynamic_shapes={"x": {0: torch.export.Dim.AUTO}},
        opset_version=OPSET,
        do_constant_folding=True,
    )

    _inline_external_data(path)
    _strip_initializer_value_info(path)


def _inline_external_data(path: Path) -> None:
    """Fold any external weight files back into the single ``.onnx`` file."""
    import onnx

    sidecars = list(path.parent.glob(f"{path.name}.data")) + list(path.parent.glob("*.onnx.data"))
    if not sidecars:
        return

    graph = onnx.load(str(path), load_external_data=True)
    onnx.save(graph, str(path), save_as_external_data=False)
    for sidecar in sidecars:
        sidecar.unlink(missing_ok=True)


def _strip_initializer_value_info(path: Path) -> None:
    """Remove `value_info` entries that describe weights rather than intermediates.

    PyTorch's exporter annotates initializers in ``graph.value_info``. ONNX shape
    inference then reads those as claims about intermediate tensors and reports a
    contradiction — for this model, "inferred (42) vs existing (192)" from the stem
    layer's weight matrix. The practical effect is that quantisation refuses to run,
    so the model can only ship at 4x its necessary size.

    Dropping the annotations and re-inferring is lossless: everything removed here is
    recomputable, and shape inference does exactly that.
    """
    import onnx

    graph = onnx.load(str(path))
    initializers = {tensor.name for tensor in graph.graph.initializer}
    keep = [vi for vi in graph.graph.value_info if vi.name not in initializers]

    if len(keep) != len(graph.graph.value_info):
        del graph.graph.value_info[:]
        graph.graph.value_info.extend(keep)
        graph = onnx.shape_inference.infer_shapes(graph, strict_mode=False)
        onnx.save(graph, str(path))


def quantise_int8(source: Path, destination: Path) -> None:
    """Dynamic int8 quantisation.

    Roughly a 4× file-size reduction with no calibration dataset required. Safe for
    this model: the input is low-dimensional, bounded, and well-conditioned by the
    normaliser, which is exactly the regime dynamic quantisation handles well. It is
    also well supported on ONNX Runtime Web's WASM backend, which is the one we use.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        model_input=str(source),
        model_output=str(destination),
        weight_type=QuantType.QInt8,
    )


def verify_onnx(
    model: HandshapeClassifier, path: Path, feature_length: int, tolerance: float = 2e-4
) -> float:
    """Check the exported graph reproduces PyTorch.

    Export is not a no-op — operator differences and constant folding can change
    results. An unverified export is a silent accuracy regression waiting to happen,
    so this runs on every build.

    :returns: the maximum absolute difference in logits.
    """
    import onnxruntime as ort

    model.eval()
    rng = np.random.default_rng(4242)
    sample = rng.normal(0, 1, size=(16, feature_length)).astype(np.float32)

    with torch.no_grad():
        expected = model(torch.from_numpy(sample)).numpy()

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    logits, embedding = session.run(["logits", "embedding"], {"features": sample})

    # Embeddings must stay on the unit hypersphere, or cosine similarity to a class
    # centre — the basis of the custom-sign feature — stops being meaningful.
    norms = np.linalg.norm(embedding, axis=-1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise RuntimeError(
            f"Exported embeddings are not unit-norm (min {norms.min():.4f}, "
            f"max {norms.max():.4f}). Custom-sign matching would be unreliable."
        )

    delta = float(np.max(np.abs(expected - logits)))
    if delta > tolerance:
        raise RuntimeError(
            f"ONNX export does not match PyTorch: max logit delta {delta:.2e} "
            f"exceeds {tolerance:.0e}. Refusing to ship a model that changed on export."
        )
    return delta


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_card(
    path: Path,
    *,
    pack_id: str,
    report: EvaluationReport,
    classes: list[str],
    parameters: int,
    synthetic: bool,
    simulated: bool = False,
) -> None:
    """Write the model card.

    Every pack carries one, so limitations are documented rather than discovered.
    """
    worst = report.worst_slice
    lines = [
        f"# Model card — {pack_id}",
        "",
        f"Built {date.today().isoformat()}.",
        "",
    ]

    if synthetic:
        lines += [
            "> ## ⚠ NOT A USABLE MODEL",
            ">",
            "> Trained on procedurally generated handshapes, not real hands. It exists to",
            "> prove the pipeline runs end to end. It will fail on a real webcam.",
            "> Train on FSboard or your own recordings before shipping anything.",
            "",
        ]

    if simulated:
        lines += [
            "> ## ⚠ TRAINED ON SIMULATED HANDS",
            ">",
            "> No real hands were recorded, and none were seen during training. Every",
            "> sample comes from a kinematic hand model — a bone skeleton with",
            "> anthropometric proportions, articulated to the joint configurations that",
            "> define the manual alphabet, then projected through a camera model with",
            "> viewpoint, anatomy and tracking noise randomised.",
            ">",
            "> **What this means in practice.** It works on real hands, and less well than",
            "> a model trained on real ones. The scores below are measured on held-out",
            "> *simulated* signers, so they describe how well the model generalises across",
            "> simulated anatomy — **not** how well it will do on your camera. Treat them",
            "> as an upper bound and expect the real figure to be lower.",
            ">",
            "> **Known gaps.** MediaPipe's errors on a real hand are structured — it",
            "> confuses occluded fingers in specific, repeatable ways — and Gaussian noise",
            "> is a poor imitation of that. There is no skin, no motion blur, and no",
            "> self-occlusion beyond what the geometry implies. M, N, T and the",
            "> thumb-tucked letters are the least faithful, because how far the thumb",
            "> disappears under the fingers is exactly what a geometric model guesses at.",
            ">",
            "> **J and Z are absent.** Both are motion letters and cannot be represented by",
            "> a single frame. Use a `temporal-ctc` pack for those.",
            ">",
            "> Recording thirty minutes of your own data with `packages/web/recorder.html`",
            "> and retraining will beat this. That is the intended upgrade path.",
            "",
        ]

    lines += [
        "## Intended use",
        "",
        "Recognising static ASL fingerspelling handshapes from MediaPipe hand landmarks,",
        "in a browser, on the signer's own device. Not a medical, legal, or safety-critical",
        "tool, and not a substitute for a qualified interpreter.",
        "",
        "## Architecture",
        "",
        f"- Residual MLP encoder with an ArcFace head, {parameters:,} parameters.",
        f"- Input: {FEATURE_LENGTH} features (21 landmarks × x, y), `{NORMALISATION_SCHEME}`.",
        f"- Output: {len(classes)} classes — {', '.join(classes)}.",
        "",
        "## Evaluation",
        "",
        "Signer-independent: no signer appears in both training and test.",
        "",
        f"- Held-out signers: {', '.join(report.test_signers) or 'none'}",
        f"- macro-F1: **{report.macro_f1:.4f}**",
        f"- accuracy: {report.accuracy:.4f}",
    ]
    if report.ece is not None:
        lines.append(f"- Expected calibration error: {report.ece:.4f}")
    if report.rejection_auroc is not None and not np.isnan(report.rejection_auroc):
        lines.append(f"- Rejection AUROC (known vs `none`): {report.rejection_auroc:.4f}")

    if report.slices:
        lines += ["", "### Per slice", "", "| Slice | Support | macro-F1 |", "| --- | --- | --- |"]
        for entry in sorted(report.slices, key=lambda s: s.macro_f1):
            lines.append(f"| {entry.name} | {entry.support} | {entry.macro_f1:.4f} |")
        if worst is not None:
            lines += ["", f"**Worst slice: {worst.name} at {worst.macro_f1:.4f}.**"]

    confusions = report.top_confusions(8)
    if confusions:
        lines += ["", "### Most frequent confusions", ""]
        lines += [f"- {t} → {p} (×{n})" for t, p, n in confusions]

    lines += [
        "",
        "## Limitations",
        "",
        "- Static handshapes only. J and Z require motion and are not covered here.",
        "- Trained on the conditions present in the data. Lighting, camera angles, or",
        "  skin tones outside that range are untested.",
        "- Rotation is preserved rather than normalised, because K/P and G/Q differ only",
        "  by orientation. Tilt beyond the ±25° augmentation range will degrade.",
        "- The `none` class rejects transitions, but no rejector is perfect; spurious",
        "  letters remain possible during fast fingerspelling.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def confusion_profile(report: EvaluationReport, *, minimum: float = 0.01) -> dict:
    """Which letters this model mistakes for which, as conditional probabilities.

    Carried in the manifest so the application can correct spelling using the model's
    *own* error profile rather than generic edit distance. The difference is large: a K
    read by this model is overwhelmingly likely to be a V, and vanishingly likely to be a
    W, so "kes" should suggest "yes" far behind "ves" — and a spell-checker that treats
    every substitution as equally likely cannot know that.

    It also keeps the operating point with the weights, exactly as the thresholds are
    (ADR 0003). A retrain ships a new error profile automatically; nothing in the
    application needs to be told the model changed.

    :param minimum: Drop pairs rarer than this. Below about 1% the estimate is a couple of
        samples of noise, and including it would let a fluke drive a suggestion.
    :returns: ``{true_letter: {predicted_letter: probability}}``, probabilities being
        P(predicted | true) on the held-out signers.
    """
    profile: dict[str, dict[str, float]] = {}
    for index, label in enumerate(report.labels):
        if label == "none":
            continue
        total = int(report.confusion[index].sum())
        if total == 0:
            continue

        row = {}
        for other, predicted in enumerate(report.labels):
            if other == index or predicted == "none":
                continue
            share = float(report.confusion[index, other]) / total
            if share >= minimum:
                row[predicted] = round(share, 4)

        if row:
            profile[label] = dict(sorted(row.items(), key=lambda item: -item[1]))

    return profile


def build_vocabularies(classes: list[str]) -> dict:
    """Group the trained handshapes into the vocabularies a reader can switch between.

    The model predicts *handshapes*; a vocabulary decides what a handshape means. This is
    how ASL itself works — 2 and V are one shape, and a signer tells them apart from
    context — and modelling it any other way makes the classifier worse. Asking one head
    to output 26 letters and 10 digits would force it to split classes carrying no
    distinguishing signal, dividing probability mass that belongs to V between V and 2.

    :returns: ``{vocabulary: {trained_label: displayed_text}}``, or ``{}`` when the pack
        has no digits and there is nothing to switch between.
    """
    from ..ingest.asl_numbers import NUMBER_VOCABULARY

    trained = set(classes)
    numbers = {
        label: digit for label, digit in NUMBER_VOCABULARY.items() if label in trained
    }
    if len(numbers) < 10:
        return {}

    letters = {label: label for label in classes if label.isalpha() and label != "none"}
    return {"letters": letters, "numbers": numbers}


def build_pack(
    *,
    model: HandshapeClassifier,
    classes: list[str],
    report: EvaluationReport,
    output: Path,
    synthetic: bool = False,
    simulated: bool = False,
    language: str = "ase",
) -> dict:
    """Write manifest, ONNX weights, and model card into ``output``."""
    output.mkdir(parents=True, exist_ok=True)
    float_path = output / "model.fp32.onnx"
    model_path = output / "model.onnx"

    export_onnx(model, float_path, FEATURE_LENGTH)
    delta = verify_onnx(model, float_path, FEATURE_LENGTH)

    # Quantise, then confirm the int8 graph still agrees with PyTorch. Quantisation is
    # lossy by design, so the tolerance is looser — but an unchecked quantised model is
    # an accuracy regression that would only surface in production.
    quantise_int8(float_path, model_path)
    quantised_delta = verify_onnx(model, model_path, FEATURE_LENGTH, tolerance=2.0)
    float_size = float_path.stat().st_size
    float_path.unlink()

    pack_id = "synthetic-fingerspell" if synthetic else "asl-fingerspell"
    worst = report.worst_slice

    manifest = {
        "id": pack_id,
        "version": PACK_VERSION,
        "name": (
            "Synthetic (pipeline test)"
            if synthetic
            else "ASL Fingerspelling (simulated)"
            if simulated
            else "ASL Fingerspelling"
        ),
        "language": language,
        "task": "static-handshape",
        "labels": classes,
        "input": {
            "featureLength": FEATURE_LENGTH,
            "hands": 1,
            "normalisation": NORMALISATION_SCHEME,
        },
        "thresholds": choose_thresholds(report),
        "metrics": {
            "macroF1": round(report.macro_f1, 6),
            "worstSliceF1": round(worst.macro_f1 if worst else report.macro_f1, 6),
            "ece": round(report.ece, 6) if report.ece is not None else None,
            "testSigners": len(report.test_signers),
        },
        "confusions": confusion_profile(report),
        **(
            {"vocabularies": vocabularies}
            if (vocabularies := build_vocabularies(classes))
            else {}
        ),
        "modelFile": "model.onnx",
        "sha256": sha256_of(model_path),
        "licence": (
            "synthetic (no real data)"
            if synthetic
            else "simulated (no real data)"
            if simulated
            else "see attribution"
        ),
    }
    if manifest["metrics"]["ece"] is None:
        del manifest["metrics"]["ece"]

    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_card(
        output / "card.md",
        pack_id=pack_id,
        report=report,
        classes=classes,
        parameters=model.parameter_count,
        synthetic=synthetic,
        simulated=simulated,
    )

    return {
        "manifest": manifest,
        "onnxDelta": delta,
        "quantisedDelta": quantised_delta,
        "sizeBytes": model_path.stat().st_size,
        "floatSizeBytes": float_size,
    }
