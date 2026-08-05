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
    label: str
    true_positives: int
    false_positives: int
    false_negatives: int
    support: int
    precision: float
    recall: float
    f1: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ClassificationEvaluation:
    evaluated_matched_repetitions: int
    labels: tuple[str, ...]
    confusion_matrix: tuple[tuple[int, ...], ...]
    per_class: tuple[PerClassMetrics, ...]
    accuracy: float | None
    macro_f1: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "evaluated_matched_repetitions": (
                self.evaluated_matched_repetitions
            ),
            "labels": list(self.labels),
            "confusion_matrix": [
                list(row) for row in self.confusion_matrix
            ],
            "per_class": [
                metrics.to_dict()
                for metrics in self.per_class
            ],
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
        }


def _safe_divide(numerator: int, denominator: int) -> float:
    return (
        numerator / denominator
        if denominator > 0
        else 0.0
    )


def _validate_reporting_labels(
    labels: Sequence[str],
) -> tuple[str, ...]:
    if isinstance(labels, (str, bytes)):
        raise ValueError(
            "Configured reporting labels must be a sequence "
            "of non-empty strings"
        )

    configured_labels = tuple(labels)

    if not configured_labels:
        raise ValueError(
            "Configured reporting labels must not be empty"
        )

    for label in configured_labels:
        if not isinstance(label, str) or not label.strip():
            raise ValueError(
                "Configured reporting labels must be "
                "non-empty strings"
            )

        if label not in SUPPORTED_FORM_CLASSES:
            raise ValueError(
                f"Unsupported configured reporting label: "
                f"{label!r}"
            )

    if len(set(configured_labels)) != len(configured_labels):
        raise ValueError(
            "Configured reporting labels must be unique"
        )

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
            raise ValueError(
                f"{name}[{index}] must be a non-empty string"
            )

        if label not in configured_set:
            raise ValueError(
                f"{name}[{index}] has unsupported label "
                f"{label!r}; expected one of "
                f"{configured_labels!r}"
            )


def evaluate_classification(
    ground_truth_labels: Sequence[str],
    predicted_labels: Sequence[str],
    labels: Sequence[str] = SUPPORTED_FORM_CLASSES,
) -> ClassificationEvaluation:
    """Evaluate enhanced classes for already matched repetitions.

    Confusion-matrix rows are ground-truth classes and columns are
    enhanced predicted classes. Detection misses and extras must not be
    passed to this function.
    """
    configured_labels = _validate_reporting_labels(labels)
    ground_truth = tuple(ground_truth_labels)
    predictions = tuple(predicted_labels)

    if len(ground_truth) != len(predictions):
        raise ValueError(
            "Ground-truth and predicted label sequences must "
            "have equal lengths"
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

    label_indices = {
        label: index
        for index, label in enumerate(configured_labels)
    }
    matrix = [
        [0 for _ in configured_labels]
        for _ in configured_labels
    ]

    for ground_truth_label, predicted_label in zip(
        ground_truth,
        predictions,
    ):
        matrix[label_indices[ground_truth_label]][
            label_indices[predicted_label]
        ] += 1

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

    evaluated_count = len(ground_truth)
    supported_class_f1_values = [
        metrics.f1
        for metrics in per_class
        if metrics.support > 0
    ]

    return ClassificationEvaluation(
        evaluated_matched_repetitions=evaluated_count,
        labels=configured_labels,
        confusion_matrix=tuple(
            tuple(row) for row in matrix
        ),
        per_class=tuple(per_class),
        accuracy=(
            sum(
                matrix[index][index]
                for index in range(len(configured_labels))
            )
            / evaluated_count
            if evaluated_count > 0
            else None
        ),
        macro_f1=(
            fmean(supported_class_f1_values)
            if supported_class_f1_values
            else None
        ),
    )
