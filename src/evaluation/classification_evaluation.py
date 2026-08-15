"""Compute form-class metrics for matched enhanced repetitions only."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import fmean
from typing import Sequence

SUPPORTED_FORM_CLASSES: tuple[str, ...] = (
    "correct",
    "insufficient_depth",
    "incomplete_extension",
    "alignment_deviation",
    "unscorable",
)


@dataclass(frozen=True)
class PerClassMetrics:
    """Metrics for one ground-truth class in the reporting order.

    ``support`` is the number of matched repetitions whose GT label is this
    class. Precision, recall and F1 are zero when their denominator is zero.
    """

    label: str
    true_positives: int
    false_positives: int
    false_negatives: int
    support: int
    precision: float
    recall: float
    f1: float

    def to_dict(self) -> dict[str, object]:
        """Return the metric fields in a serialization-ready mapping."""
        return asdict(self)


@dataclass(frozen=True)
class ClassificationEvaluation:
    """Classification metrics derived from matched enhanced events.

    Confusion-matrix rows are GT classes and columns are predicted classes in
    ``labels`` order. The standard formal report uses
    ``SUPPORTED_FORM_CLASSES`` as its fixed order.
    """

    evaluated_matched_repetitions: int
    labels: tuple[str, ...]
    confusion_matrix: tuple[tuple[int, ...], ...]
    per_class: tuple[PerClassMetrics, ...]
    accuracy: float | None
    macro_f1: float | None

    def to_dict(self) -> dict[str, object]:
        """Return lists and mappings suitable for JSON serialization."""
        return {
            "evaluated_matched_repetitions": (self.evaluated_matched_repetitions),
            "labels": list(self.labels),
            "confusion_matrix": [list(row) for row in self.confusion_matrix],
            "per_class": [metrics.to_dict() for metrics in self.per_class],
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
        }


def _safe_divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def _validate_reporting_labels(
    labels: Sequence[str],
) -> tuple[str, ...]:
    if isinstance(labels, (str, bytes)):
        raise ValueError(
            "Configured reporting labels must be a sequence of non-empty strings"
        )

    configured_labels = tuple(labels)

    if not configured_labels:
        raise ValueError("Configured reporting labels must not be empty")

    for label in configured_labels:
        if not isinstance(label, str) or not label.strip():
            raise ValueError("Configured reporting labels must be non-empty strings")

        if label not in SUPPORTED_FORM_CLASSES:
            raise ValueError(f"Unsupported configured reporting label: {label!r}")

    if len(set(configured_labels)) != len(configured_labels):
        raise ValueError("Configured reporting labels must be unique")

    return configured_labels


def _validate_input_labels(
    values: tuple[str, ...],
    *,
    name: str,
    configured_labels: tuple[str, ...],
) -> None:
    configured_set = set(configured_labels)

    for index, label in enumerate(values):
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"{name}[{index}] must be a non-empty string")

        if label not in configured_set:
            raise ValueError(
                f"{name}[{index}] has unsupported label "
                f"{label!r}; expected one of "
                f"{configured_labels!r}"
            )


def _validated_confusion_matrix(
    confusion_matrix: Sequence[Sequence[int]],
    *,
    class_count: int,
) -> tuple[tuple[int, ...], ...]:
    if isinstance(confusion_matrix, (str, bytes)):
        raise ValueError("Confusion matrix must be a square sequence of rows")

    try:
        rows = tuple(tuple(row) for row in confusion_matrix)
    except TypeError as error:
        raise ValueError(
            "Confusion matrix must be a square sequence of rows"
        ) from error

    if len(rows) != class_count or any(len(row) != class_count for row in rows):
        raise ValueError(
            "Confusion matrix dimensions must match the configured reporting labels"
        )

    for row_index, row in enumerate(rows):
        for column_index, value in enumerate(row):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    "Confusion matrix counts must be "
                    "non-negative integers; invalid cell "
                    f"({row_index}, {column_index})"
                )

    return rows


def evaluate_classification_from_confusion_matrix(
    confusion_matrix: Sequence[Sequence[int]],
    labels: Sequence[str] = SUPPORTED_FORM_CLASSES,
) -> ClassificationEvaluation:
    """Recompute matched-repetition metrics from raw matrix counts.

    Rows are ground-truth classes and columns are predicted classes.
    Class support is the corresponding row total. Zero-denominator precision,
    recall and F1 are reported as ``0.0``. Macro F1 averages only classes with
    positive GT support, so it is ``None`` when the entire matrix is empty;
    accuracy is also ``None`` for an empty evaluation.
    """
    configured_labels = _validate_reporting_labels(labels)
    matrix = _validated_confusion_matrix(
        confusion_matrix,
        class_count=len(configured_labels),
    )
    per_class = []

    for class_index, label in enumerate(configured_labels):
        true_positives = matrix[class_index][class_index]
        support = sum(matrix[class_index])
        false_negatives = support - true_positives
        false_positives = sum(
            matrix[row_index][class_index]
            for row_index in range(len(configured_labels))
            if row_index != class_index
        )
        precision = _safe_divide(
            true_positives,
            true_positives + false_positives,
        )
        recall = _safe_divide(
            true_positives,
            true_positives + false_negatives,
        )
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0.0
            else 0.0
        )
        per_class.append(
            PerClassMetrics(
                label=label,
                true_positives=true_positives,
                false_positives=false_positives,
                false_negatives=false_negatives,
                support=support,
                precision=precision,
                recall=recall,
                f1=f1,
            )
        )

    evaluated_count = sum(sum(row) for row in matrix)
    supported_class_f1_values = [
        metrics.f1 for metrics in per_class if metrics.support > 0
    ]

    return ClassificationEvaluation(
        evaluated_matched_repetitions=evaluated_count,
        labels=configured_labels,
        confusion_matrix=matrix,
        per_class=tuple(per_class),
        accuracy=(
            sum(matrix[index][index] for index in range(len(configured_labels)))
            / evaluated_count
            if evaluated_count > 0
            else None
        ),
        macro_f1=(
            fmean(supported_class_f1_values) if supported_class_f1_values else None
        ),
    )


def evaluate_classification(
    ground_truth_labels: Sequence[str],
    predicted_labels: Sequence[str],
    labels: Sequence[str] = SUPPORTED_FORM_CLASSES,
) -> ClassificationEvaluation:
    """Evaluate enhanced classes for already matched repetitions.

    Confusion-matrix rows are ground-truth classes and columns are
    enhanced predicted classes. Detection misses and extras must not be
    passed to this function, and baseline frame warnings are not formal class
    predictions. Empty matched sequences produce an empty evaluation under the
    same zero-denominator rules as the matrix-based function.
    """
    configured_labels = _validate_reporting_labels(labels)
    ground_truth = tuple(ground_truth_labels)
    predictions = tuple(predicted_labels)

    if len(ground_truth) != len(predictions):
        raise ValueError(
            "Ground-truth and predicted label sequences must have equal lengths"
        )

    _validate_input_labels(
        ground_truth,
        name="ground_truth_labels",
        configured_labels=configured_labels,
    )
    _validate_input_labels(
        predictions,
        name="predicted_labels",
        configured_labels=configured_labels,
    )

    label_indices = {label: index for index, label in enumerate(configured_labels)}
    matrix = [[0 for _ in configured_labels] for _ in configured_labels]

    for ground_truth_label, predicted_label in zip(
        ground_truth,
        predictions,
    ):
        matrix[label_indices[ground_truth_label]][label_indices[predicted_label]] += 1

    return evaluate_classification_from_confusion_matrix(
        matrix,
        labels=configured_labels,
    )
