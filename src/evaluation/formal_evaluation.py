from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Sequence

from evaluation.classification_evaluation import (
    SUPPORTED_FORM_CLASSES,
    ClassificationEvaluation,
    evaluate_classification,
)
from evaluation.detection_evaluation import (
    DetectionSummary,
    evaluate_detection_for_clip,
)
from evaluation.event_matching import (
    DEFAULT_EVENT_TOLERANCE_SECONDS,
    EventMatchResult,
)
from evaluation.repetition_events import (
    EnhancedRepetitionEvent,
    GroundTruthRepetitionEvent,
)


@dataclass(frozen=True)
class MatchedClassificationPair:
    ground_truth_attempt_id: str
    predicted_rep_id: int
    ground_truth_completion_frame: int
    predicted_completion_frame: int
    ground_truth_class: str
    predicted_class: str
    signed_frame_error: int
    signed_timing_error_seconds: float
    absolute_timing_error_seconds: float
    matching_basis: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class GroundTruthClassDetectionRecall:
    label: str
    ground_truth_support: int
    matched_ground_truth_repetitions: int
    missed_ground_truth_repetitions: int
    recall: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EnhancedClipEvaluation:
    detection: DetectionSummary
    matched_pairs: tuple[MatchedClassificationPair, ...]
    unmatched_prediction_ids: tuple[int, ...]
    unmatched_ground_truth_attempt_ids: tuple[str, ...]
    classification: ClassificationEvaluation
    detection_recall_by_ground_truth_class: tuple[
        GroundTruthClassDetectionRecall, ...
    ]

    def to_dict(self) -> dict[str, object]:
        return {
            "detection": self.detection.to_dict(),
            "matched_pairs": [
                pair.to_dict()
                for pair in self.matched_pairs
            ],
            "unmatched_prediction_ids": list(
                self.unmatched_prediction_ids
            ),
            "unmatched_ground_truth_attempt_ids": list(
                self.unmatched_ground_truth_attempt_ids
            ),
            "classification": self.classification.to_dict(),
            "detection_recall_by_ground_truth_class": [
                class_recall.to_dict()
                for class_recall in (
                    self.detection_recall_by_ground_truth_class
                )
            ],
        }


def _validate_label(
    label: object,
    *,
    source_name: str,
    identifier: object,
) -> None:
    if not isinstance(label, str) or not label.strip():
        raise ValueError(
            f"{source_name} {identifier!r} must have a "
            "non-blank form class"
        )

    if label not in SUPPORTED_FORM_CLASSES:
        raise ValueError(
            f"{source_name} {identifier!r} has unsupported "
            f"form class {label!r}; expected one of "
            f"{SUPPORTED_FORM_CLASSES!r}"
        )


def _validate_events(
    predictions: tuple[EnhancedRepetitionEvent, ...],
    annotations: tuple[GroundTruthRepetitionEvent, ...],
    *,
    clip_id: str,
) -> None:
    if not isinstance(clip_id, str) or not clip_id.strip():
        raise ValueError("Clip ID must be a non-blank string")

    for event in annotations:
        if not isinstance(event, GroundTruthRepetitionEvent):
            raise ValueError(
                "Ground-truth inputs must be "
                "GroundTruthRepetitionEvent instances"
            )

        _validate_label(
            event.ground_truth_class,
            source_name="Ground-truth attempt",
            identifier=event.ground_truth_attempt_id,
        )

    for event in predictions:
        if not isinstance(event, EnhancedRepetitionEvent):
            raise ValueError(
                "Enhanced predictions must be "
                "EnhancedRepetitionEvent instances"
            )

        _validate_label(
            event.predicted_class,
            source_name="Enhanced prediction",
            identifier=event.predicted_rep_id,
        )


def _validate_matched_references(
    match_result: EventMatchResult,
    predictions: tuple[EnhancedRepetitionEvent, ...],
    annotations: tuple[GroundTruthRepetitionEvent, ...],
) -> None:
    predictions_by_id = {
        event.predicted_rep_id: event
        for event in predictions
    }
    annotations_by_id = {
        event.ground_truth_attempt_id: event
        for event in annotations
    }
    matched_prediction_ids = []
    matched_annotation_ids = []

    for pair in match_result.matched_pairs:
        prediction_id = pair.prediction.predicted_rep_id
        annotation_id = (
            pair.annotation.ground_truth_attempt_id
        )

        if (
            predictions_by_id.get(prediction_id)
            is not pair.prediction
        ):
            raise ValueError(
                "A matched prediction does not resolve to "
                "exactly one integration input"
            )

        if (
            annotations_by_id.get(annotation_id)
            is not pair.annotation
        ):
            raise ValueError(
                "A matched annotation does not resolve to "
                "exactly one integration input"
            )

        matched_prediction_ids.append(prediction_id)
        matched_annotation_ids.append(annotation_id)

    if len(matched_prediction_ids) != len(
        set(matched_prediction_ids)
    ):
        raise ValueError(
            "A prediction appears in more than one matched pair"
        )

    if len(matched_annotation_ids) != len(
        set(matched_annotation_ids)
    ):
        raise ValueError(
            "An annotation appears in more than one matched pair"
        )


def _summarise_detection_recall_by_class(
    annotations: tuple[GroundTruthRepetitionEvent, ...],
    match_result: EventMatchResult,
    detection: DetectionSummary,
) -> tuple[GroundTruthClassDetectionRecall, ...]:
    support_counts = Counter(
        event.ground_truth_class
        for event in annotations
    )
    matched_counts = Counter(
        pair.annotation.ground_truth_class
        for pair in match_result.matched_pairs
    )
    missed_counts = Counter(
        event.ground_truth_class
        for event in match_result.unmatched_annotations
    )
    rows = tuple(
        GroundTruthClassDetectionRecall(
            label=label,
            ground_truth_support=support_counts[label],
            matched_ground_truth_repetitions=(
                matched_counts[label]
            ),
            missed_ground_truth_repetitions=(
                missed_counts[label]
            ),
            recall=(
                matched_counts[label] / support_counts[label]
                if support_counts[label] > 0
                else None
            ),
        )
        for label in SUPPORTED_FORM_CLASSES
    )

    if any(
        row.ground_truth_support
        != (
            row.matched_ground_truth_repetitions
            + row.missed_ground_truth_repetitions
        )
        for row in rows
    ):
        raise ValueError(
            "Per-class ground-truth support does not equal "
            "matched plus missed repetitions"
        )

    if sum(
        row.ground_truth_support for row in rows
    ) != detection.ground_truth_event_count:
        raise ValueError(
            "Per-class support does not equal the overall "
            "ground-truth event count"
        )

    if sum(
        row.matched_ground_truth_repetitions
        for row in rows
    ) != detection.matched_events:
        raise ValueError(
            "Per-class matched counts do not equal the overall "
            "matched event count"
        )

    if sum(
        row.missed_ground_truth_repetitions
        for row in rows
    ) != detection.missed_annotations:
        raise ValueError(
            "Per-class missed counts do not equal the overall "
            "detection miss count"
        )

    return rows


def evaluate_enhanced_clip(
    predictions: Sequence[EnhancedRepetitionEvent],
    annotations: Sequence[GroundTruthRepetitionEvent],
    *,
    clip_id: str,
    source_fps: float,
    tolerance_seconds: float = (
        DEFAULT_EVENT_TOLERANCE_SECONDS
    ),
) -> EnhancedClipEvaluation:
    """Evaluate detection and matched classification for one clip."""
    enhanced_predictions = tuple(predictions)
    ground_truth_annotations = tuple(annotations)
    _validate_events(
        enhanced_predictions,
        ground_truth_annotations,
        clip_id=clip_id,
    )
    match_result, detection = evaluate_detection_for_clip(
        enhanced_predictions,
        ground_truth_annotations,
        clip_id=clip_id,
        method="enhanced",
        source_fps=source_fps,
        tolerance_seconds=tolerance_seconds,
    )
    _validate_matched_references(
        match_result,
        enhanced_predictions,
        ground_truth_annotations,
    )
    matched_ground_truth_labels = tuple(
        pair.annotation.ground_truth_class
        for pair in match_result.matched_pairs
    )
    matched_predicted_labels = tuple(
        pair.prediction.predicted_class
        for pair in match_result.matched_pairs
    )
    classification = evaluate_classification(
        matched_ground_truth_labels,
        matched_predicted_labels,
        labels=SUPPORTED_FORM_CLASSES,
    )
    class_detection_recall = (
        _summarise_detection_recall_by_class(
            ground_truth_annotations,
            match_result,
            detection,
        )
    )

    return EnhancedClipEvaluation(
        detection=detection,
        matched_pairs=tuple(
            MatchedClassificationPair(
                ground_truth_attempt_id=(
                    pair.annotation
                    .ground_truth_attempt_id
                ),
                predicted_rep_id=(
                    pair.prediction.predicted_rep_id
                ),
                ground_truth_completion_frame=(
                    pair.annotation.completion_frame
                ),
                predicted_completion_frame=(
                    pair.prediction.completion_frame
                ),
                ground_truth_class=(
                    pair.annotation.ground_truth_class
                ),
                predicted_class=(
                    pair.prediction.predicted_class
                ),
                signed_frame_error=pair.signed_frame_error,
                signed_timing_error_seconds=(
                    pair.signed_timing_error_seconds
                ),
                absolute_timing_error_seconds=(
                    pair.absolute_timing_error_seconds
                ),
                matching_basis=pair.matching_basis,
            )
            for pair in match_result.matched_pairs
        ),
        unmatched_prediction_ids=tuple(
            event.predicted_rep_id
            for event in match_result.unmatched_predictions
        ),
        unmatched_ground_truth_attempt_ids=tuple(
            event.ground_truth_attempt_id
            for event in match_result.unmatched_annotations
        ),
        classification=classification,
        detection_recall_by_ground_truth_class=(
            class_detection_recall
        ),
    )
