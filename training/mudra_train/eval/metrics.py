"""Evaluation that does not flatter the model.

The v1 app shipped an in-browser "accuracy test" in which the same person, on the same
camera, in the same session, signed each letter and the result was reported as a
percentage. That number could not predict performance for anyone else, and it was the
most dangerous item in the audit precisely because it *looked* like validation.

Everything here is built to avoid repeating that:

* **Signer-independent splits.** No signer appears in both train and test. Without
  this every score is inflated, because the model memorises one person's hands.
* **Macro-F1, not accuracy.** Classes are imbalanced, and accuracy hides a letter that
  never fires.
* **Per-slice reporting.** A model at 94% overall and 61% for left-handed signers has
  not passed. Aggregates hide exactly the failures that matter.
* **Calibration.** A confidence threshold is only meaningful if 0.9 means 0.9.
* **Rejection quality.** Measured as AUROC for known-vs-``none``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True, slots=True)
class ClassMetrics:
    label: str
    support: int
    precision: float
    recall: float
    f1: float


@dataclass(slots=True)
class SliceMetrics:
    name: str
    support: int
    macro_f1: float
    accuracy: float


@dataclass(slots=True)
class EvaluationReport:
    macro_f1: float
    accuracy: float
    per_class: list[ClassMetrics]
    confusion: np.ndarray
    labels: list[str]
    slices: list[SliceMetrics] = field(default_factory=list)
    ece: float | None = None
    rejection_auroc: float | None = None
    test_signers: list[str] = field(default_factory=list)

    @property
    def worst_slice(self) -> SliceMetrics | None:
        """The slice that performs worst. The number that actually gates a release."""
        return min(self.slices, key=lambda s: s.macro_f1) if self.slices else None

    def top_confusions(self, limit: int = 10) -> list[tuple[str, str, int]]:
        """Most frequent (true, predicted) mistakes."""
        pairs: list[tuple[str, str, int]] = []
        for i, true_label in enumerate(self.labels):
            for j, predicted_label in enumerate(self.labels):
                if i != j and self.confusion[i, j] > 0:
                    pairs.append((true_label, predicted_label, int(self.confusion[i, j])))
        pairs.sort(key=lambda item: item[2], reverse=True)
        return pairs[:limit]


def split_by_signer(
    signers: np.ndarray,
    test_fraction: float = 0.25,
    seed: int = 20260820,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Partition indices so no signer appears on both sides.

    Returns ``(train_indices, test_indices, held_out_signers)``.

    :raises ValueError: if there are fewer than two signers. A single-signer dataset
        cannot support an honest evaluation, and silently proceeding would reproduce
        exactly the flaw this module exists to prevent.
    """
    unique = np.array(sorted(set(signers.tolist())))
    if len(unique) < 2:
        raise ValueError(
            f"Signer-independent evaluation needs at least 2 signers, found {len(unique)}. "
            "Record samples from another person, or you cannot know whether the model "
            "generalises beyond one pair of hands."
        )

    rng = np.random.default_rng(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)

    held_out_count = max(1, int(round(len(shuffled) * test_fraction)))
    held_out = set(shuffled[:held_out_count].tolist())

    is_test = np.array([signer in held_out for signer in signers])
    return np.flatnonzero(~is_test), np.flatnonzero(is_test), sorted(held_out)


def confusion_matrix(true: np.ndarray, predicted: np.ndarray, classes: int) -> np.ndarray:
    matrix = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(matrix, (true, predicted), 1)
    return matrix


def per_class_metrics(matrix: np.ndarray, labels: list[str]) -> list[ClassMetrics]:
    results: list[ClassMetrics] = []
    for index, label in enumerate(labels):
        true_positive = int(matrix[index, index])
        predicted_positive = int(matrix[:, index].sum())
        actual_positive = int(matrix[index, :].sum())

        precision = true_positive / predicted_positive if predicted_positive else 0.0
        recall = true_positive / actual_positive if actual_positive else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        results.append(
            ClassMetrics(
                label=label,
                support=actual_positive,
                precision=precision,
                recall=recall,
                f1=f1,
            )
        )
    return results


def expected_calibration_error(
    probabilities: np.ndarray, correct: np.ndarray, bins: int = 15
) -> float:
    """Weighted gap between confidence and accuracy across confidence bins.

    A well-calibrated model has ECE near 0: among predictions it makes at 90%
    confidence, about 90% are right. Without this a probability threshold is arbitrary.
    """
    confidence = probabilities.max(axis=1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(confidence)
    error = 0.0

    for lower, upper in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > lower) & (confidence <= upper)
        count = int(in_bin.sum())
        if count == 0:
            continue
        error += (count / total) * abs(float(correct[in_bin].mean()) - float(confidence[in_bin].mean()))

    return error


def rejection_auroc(scores: np.ndarray, is_known: np.ndarray) -> float:
    """AUROC for separating real signs from ``none``, computed by rank statistics.

    1.0 means perfect separation; 0.5 means the score carries no information.
    """
    known = scores[is_known]
    unknown = scores[~is_known]
    if len(known) == 0 or len(unknown) == 0:
        return float("nan")

    order = np.argsort(np.concatenate([known, unknown]), kind="mergesort")
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)

    known_rank_sum = ranks[: len(known)].sum()
    u_statistic = known_rank_sum - len(known) * (len(known) + 1) / 2
    return float(u_statistic / (len(known) * len(unknown)))


def evaluate(
    true_indices: np.ndarray,
    predicted_indices: np.ndarray,
    probabilities: np.ndarray,
    labels: list[str],
    slices: dict[str, np.ndarray] | None = None,
    test_signers: list[str] | None = None,
) -> EvaluationReport:
    """Build a full report.

    Parameters
    ----------
    slices:
        Named boolean masks over the test set, e.g. ``{"hand=left": mask}``. Each is
        reported separately so a subgroup failure cannot hide inside the aggregate.
    """
    matrix = confusion_matrix(true_indices, predicted_indices, len(labels))
    per_class = per_class_metrics(matrix, labels)

    scored = [c for c in per_class if c.support > 0]
    macro_f1 = float(np.mean([c.f1 for c in scored])) if scored else 0.0
    accuracy = float((true_indices == predicted_indices).mean()) if len(true_indices) else 0.0

    slice_metrics: list[SliceMetrics] = []
    for name, mask in (slices or {}).items():
        if int(mask.sum()) == 0:
            continue
        sub_matrix = confusion_matrix(true_indices[mask], predicted_indices[mask], len(labels))
        sub_per_class = [c for c in per_class_metrics(sub_matrix, labels) if c.support > 0]
        slice_metrics.append(
            SliceMetrics(
                name=name,
                support=int(mask.sum()),
                macro_f1=float(np.mean([c.f1 for c in sub_per_class])) if sub_per_class else 0.0,
                accuracy=float((true_indices[mask] == predicted_indices[mask]).mean()),
            )
        )

    correct = (true_indices == predicted_indices).astype(np.float64)
    ece = expected_calibration_error(probabilities, correct)

    auroc: float | None = None
    if "none" in labels:
        none_index = labels.index("none")
        is_known = true_indices != none_index
        # Score = probability the sample is a real sign.
        auroc = rejection_auroc(1.0 - probabilities[:, none_index], is_known)

    return EvaluationReport(
        macro_f1=macro_f1,
        accuracy=accuracy,
        per_class=per_class,
        confusion=matrix,
        labels=labels,
        slices=slice_metrics,
        ece=ece,
        rejection_auroc=auroc,
        test_signers=test_signers or [],
    )


def format_report(report: EvaluationReport) -> str:
    """Render a report for a terminal or a CI log."""
    lines = [
        "═" * 66,
        "EVALUATION — signer-independent",
        "═" * 66,
        f"  macro-F1   {report.macro_f1:.4f}",
        f"  accuracy   {report.accuracy:.4f}",
    ]
    if report.ece is not None:
        lines.append(f"  ECE        {report.ece:.4f}  (lower is better)")
    if report.rejection_auroc is not None and not np.isnan(report.rejection_auroc):
        lines.append(f"  reject AUC {report.rejection_auroc:.4f}  (known vs none)")
    if report.test_signers:
        lines.append(f"  held out   {', '.join(report.test_signers)}")

    lines += ["", "Per class:", f"  {'label':<10}{'support':>9}{'prec':>9}{'recall':>9}{'F1':>9}"]
    for entry in sorted(report.per_class, key=lambda c: c.f1):
        if entry.support == 0:
            continue
        lines.append(
            f"  {entry.label:<10}{entry.support:>9}{entry.precision:>9.3f}"
            f"{entry.recall:>9.3f}{entry.f1:>9.3f}"
        )

    if report.slices:
        lines += ["", "Per slice:", f"  {'slice':<24}{'support':>9}{'macro-F1':>11}"]
        for entry in sorted(report.slices, key=lambda s: s.macro_f1):
            lines.append(f"  {entry.name:<24}{entry.support:>9}{entry.macro_f1:>11.4f}")
        worst = report.worst_slice
        if worst is not None:
            lines.append(f"\n  Worst slice: {worst.name} at {worst.macro_f1:.4f}")

    confusions = report.top_confusions(8)
    if confusions:
        lines += ["", "Most frequent confusions:"]
        lines += [f"  {t} → {p}  ×{n}" for t, p, n in confusions]

    lines.append("═" * 66)
    return "\n".join(lines)
