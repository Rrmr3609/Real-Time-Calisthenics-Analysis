"""Aggregate evaluated clips and write formal metric/provenance outputs.

This module does not load videos, extract events or perform event matching. It
validates already evaluated clip results, builds deterministic metric content
and writes a separate evaluation-run metadata record containing provenance.
"""

from __future__ import annotations

import csv
import json
import math
import os
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from config.runtime import ALLOWED_SPLITS
from evaluation.classification_evaluation import (
    SUPPORTED_FORM_CLASSES,
    ClassificationEvaluation,
    evaluate_classification_from_confusion_matrix,
)
from evaluation.detection_evaluation import DetectionSummary
from evaluation.event_matching import (
    DEFAULT_EVENT_TOLERANCE_SECONDS,
)
from evaluation.formal_evaluation import (
    EnhancedClipEvaluation,
    GroundTruthClassDetectionRecall,
)
from evaluation.formal_evidence_metrics import (
    ClipEvidenceMetrics,
    FormalEvidenceMetrics,
    aggregate_formal_evidence,
    unavailable_clip_evidence,
)
from utils.csv_logger import prepare_output_paths
from utils.run_provenance import (
    collect_git_state,
    collect_software_versions,
    utc_timestamp,
    write_json_atomically,
)

FORMAL_EVALUATION_REPORT_SCHEMA_VERSION = 2

PER_CLIP_COLUMNS = (
    "clip_id",
    "source_fps",
    "ground_truth_repetition_count",
    "baseline_predicted_count",
    "baseline_signed_count_error",
    "baseline_absolute_count_error",
    "baseline_matches",
    "baseline_misses",
    "baseline_extras",
    "baseline_event_precision",
    "baseline_event_recall",
    "baseline_event_f1",
    "enhanced_predicted_count",
    "enhanced_signed_count_error",
    "enhanced_absolute_count_error",
    "enhanced_matches",
    "enhanced_misses",
    "enhanced_extras",
    "enhanced_event_precision",
    "enhanced_event_recall",
    "enhanced_event_f1",
    "enhanced_matched_classification_count",
    "enhanced_classification_accuracy",
    "enhanced_classification_macro_f1",
    "baseline_analyzed_frame_count",
    "baseline_mean_measured_processing_time_ms",
    "baseline_median_measured_processing_time_ms",
    "baseline_measured_analysis_throughput_fps",
    "baseline_pose_available_frames",
    "baseline_pose_denominator_frames",
    "baseline_pose_availability_rate",
    "baseline_elbow_available_frames",
    "baseline_elbow_denominator_frames",
    "baseline_elbow_availability_rate",
    "baseline_alignment_available_frames",
    "baseline_alignment_denominator_frames",
    "baseline_alignment_availability_rate",
    "baseline_selected_side_available_frames",
    "baseline_selected_side_denominator_frames",
    "baseline_selected_side_availability_rate",
    "baseline_instantaneous_side_change_count",
    "enhanced_analyzed_frame_count",
    "enhanced_mean_measured_processing_time_ms",
    "enhanced_median_measured_processing_time_ms",
    "enhanced_measured_analysis_throughput_fps",
    "enhanced_pose_available_frames",
    "enhanced_pose_denominator_frames",
    "enhanced_pose_availability_rate",
    "enhanced_elbow_valid_frames",
    "enhanced_elbow_denominator_frames",
    "enhanced_elbow_availability_rate",
    "enhanced_alignment_valid_frames",
    "enhanced_alignment_denominator_frames",
    "enhanced_alignment_availability_rate",
    "enhanced_selected_side_available_frames",
    "enhanced_selected_side_denominator_frames",
    "enhanced_selected_side_availability_rate",
    "enhanced_stable_side_change_count",
    "enhanced_predicted_unscorable_count",
    "enhanced_predicted_unscorable_rate",
    "enhanced_alignment_coverage_observation_count",
    "enhanced_mean_alignment_valid_ratio",
    "human_evaluable_attempt_count",
    "human_alignment_evidence_adequate_count",
    "human_alignment_evidence_inadequate_count",
    "human_alignment_evidence_adequate_rate",
)

CLASSIFICATION_PER_CLASS_COLUMNS = (
    "class_label",
    "true_positives",
    "false_positives",
    "false_negatives",
    "support",
    "precision",
    "recall",
    "f1",
)

DETECTION_RECALL_COLUMNS = (
    "class_label",
    "support",
    "matched",
    "missed",
    "recall",
)


@dataclass(frozen=True)
class EvaluationClipContext:
    """Manifest identity, split and source FPS for one evaluated clip."""

    clip_id: str
    split: str
    source_fps: float
    evidence_metrics: ClipEvidenceMetrics | None = None


@dataclass(frozen=True)
class EvaluationEvidenceProvenance:
    """Privacy-safe identities and hashes for the exact formal evidence set."""

    manifest_path: str
    manifest_sha256: str
    annotations_path: str
    annotations_sha256: str
    frozen_annotation_sha256: str
    review_metadata_path: str
    review_metadata_sha256: str
    review_status: str

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_manifest": {
                "path": self.manifest_path,
                "sha256": self.manifest_sha256,
            },
            "repetition_annotations": {
                "path": self.annotations_path,
                "sha256": self.annotations_sha256,
                "frozen_sha256": self.frozen_annotation_sha256,
            },
            "annotation_review": {
                "path": self.review_metadata_path,
                "sha256": self.review_metadata_sha256,
                "status": self.review_status,
            },
        }


@dataclass(frozen=True)
class SourceRunProvenance:
    """Identity and hashes for one consumed baseline or enhanced source run.

    Source file paths are privacy-safe relative identifiers; hashes bind the
    exact metadata, consumed CSV, input video and resolved configuration used
    by the evaluation.
    """

    clip_id: str
    method: str
    source_run_id: str
    split: str
    source_metadata_path: str
    source_metadata_sha256: str
    consumed_output_name: str
    consumed_output_path: str
    consumed_output_sha256: str
    descriptive_frame_csv_path: str
    descriptive_frame_csv_sha256: str
    source_input_video_sha256: str
    source_git_commit: str | None
    source_git_dirty: bool | None
    resolved_configuration_sha256: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "clip_id": self.clip_id,
            "method": self.method,
            "source_run_id": self.source_run_id,
            "split": self.split,
            "source_metadata_file": {
                "path": self.source_metadata_path,
                "sha256": self.source_metadata_sha256,
            },
            "consumed_output_csv": {
                "output_name": self.consumed_output_name,
                "path": self.consumed_output_path,
                "sha256": self.consumed_output_sha256,
            },
            "descriptive_frame_csv": {
                "path": self.descriptive_frame_csv_path,
                "sha256": self.descriptive_frame_csv_sha256,
            },
            "source_input_video_sha256": (self.source_input_video_sha256),
            "source_git": {
                "commit": self.source_git_commit,
                "dirty": self.source_git_dirty,
            },
            "resolved_configuration_sha256": (self.resolved_configuration_sha256),
        }


@dataclass(frozen=True)
class DetectionAggregate:
    """Cross-clip pooled detection totals and derived summary metrics.

    Event rates pool matches, misses and extras across clips. Count-error means
    average clip-level errors, while completion-timing means are weighted by
    matched events and expressed in seconds.
    """

    evaluated_clips: int
    total_ground_truth_repetitions: int
    total_predicted_repetitions: int
    total_matched_events: int
    total_misses: int
    total_extras: int
    pooled_event_precision: float
    pooled_event_recall: float
    pooled_event_f1: float
    total_signed_count_error: int
    mean_signed_count_error: float | None
    mean_absolute_count_error: float | None
    exact_count_clip_count: int
    exact_count_clip_accuracy: float | None
    total_matched_timing_observations: int
    mean_signed_completion_timing_error_seconds: float | None
    mean_absolute_completion_timing_error_seconds: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PerClipFormalMetrics:
    """Side-by-side baseline/enhanced metrics for one manifest clip."""

    clip_id: str
    source_fps: float
    ground_truth_repetition_count: int
    baseline_predicted_count: int
    baseline_signed_count_error: int
    baseline_absolute_count_error: int
    baseline_matches: int
    baseline_misses: int
    baseline_extras: int
    baseline_event_precision: float
    baseline_event_recall: float
    baseline_event_f1: float
    enhanced_predicted_count: int
    enhanced_signed_count_error: int
    enhanced_absolute_count_error: int
    enhanced_matches: int
    enhanced_misses: int
    enhanced_extras: int
    enhanced_event_precision: float
    enhanced_event_recall: float
    enhanced_event_f1: float
    enhanced_matched_classification_count: int
    enhanced_classification_accuracy: float | None
    enhanced_classification_macro_f1: float | None
    baseline_analyzed_frame_count: int
    baseline_mean_measured_processing_time_ms: float | None
    baseline_median_measured_processing_time_ms: float | None
    baseline_measured_analysis_throughput_fps: float | None
    baseline_pose_available_frames: int | None
    baseline_pose_denominator_frames: int
    baseline_pose_availability_rate: float | None
    baseline_elbow_available_frames: int | None
    baseline_elbow_denominator_frames: int
    baseline_elbow_availability_rate: float | None
    baseline_alignment_available_frames: int | None
    baseline_alignment_denominator_frames: int
    baseline_alignment_availability_rate: float | None
    baseline_selected_side_available_frames: int | None
    baseline_selected_side_denominator_frames: int
    baseline_selected_side_availability_rate: float | None
    baseline_instantaneous_side_change_count: int | None
    enhanced_analyzed_frame_count: int
    enhanced_mean_measured_processing_time_ms: float | None
    enhanced_median_measured_processing_time_ms: float | None
    enhanced_measured_analysis_throughput_fps: float | None
    enhanced_pose_available_frames: int | None
    enhanced_pose_denominator_frames: int
    enhanced_pose_availability_rate: float | None
    enhanced_elbow_valid_frames: int | None
    enhanced_elbow_denominator_frames: int
    enhanced_elbow_availability_rate: float | None
    enhanced_alignment_valid_frames: int | None
    enhanced_alignment_denominator_frames: int
    enhanced_alignment_availability_rate: float | None
    enhanced_selected_side_available_frames: int | None
    enhanced_selected_side_denominator_frames: int
    enhanced_selected_side_availability_rate: float | None
    enhanced_stable_side_change_count: int | None
    enhanced_predicted_unscorable_count: int
    enhanced_predicted_unscorable_rate: float | None
    enhanced_alignment_coverage_observation_count: int
    enhanced_mean_alignment_valid_ratio: float | None
    human_evaluable_attempt_count: int
    human_alignment_evidence_adequate_count: int
    human_alignment_evidence_inadequate_count: int
    human_alignment_evidence_adequate_rate: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FormalEvaluationReport:
    """Deterministic scientific content for one split and shared tolerance.

    Clip IDs are sorted deterministically rather than by input sequence;
    enhanced class rows use ``SUPPORTED_FORM_CLASSES`` order. Runtime metadata,
    timestamps, software and Git state are deliberately outside this value.
    """

    report_schema_version: int
    split: str
    event_tolerance_seconds: float
    ordered_clip_ids: tuple[str, ...]
    baseline_detection: DetectionAggregate
    enhanced_detection: DetectionAggregate
    enhanced_classification: ClassificationEvaluation
    enhanced_detection_recall_by_ground_truth_class: tuple[
        GroundTruthClassDetectionRecall, ...
    ]
    per_clip_metrics: tuple[PerClipFormalMetrics, ...]
    descriptive_evidence: FormalEvidenceMetrics

    def to_dict(self) -> dict[str, object]:
        return {
            "report_schema_version": self.report_schema_version,
            "split": self.split,
            "event_tolerance_seconds": (self.event_tolerance_seconds),
            "ordered_clip_ids": list(self.ordered_clip_ids),
            "baseline_detection": (self.baseline_detection.to_dict()),
            "enhanced_detection": (self.enhanced_detection.to_dict()),
            "enhanced_classification": (self.enhanced_classification.to_dict()),
            "enhanced_detection_recall_by_ground_truth_class": [
                row.to_dict()
                for row in (self.enhanced_detection_recall_by_ground_truth_class)
            ],
            "per_clip_metrics": [row.to_dict() for row in self.per_clip_metrics],
            "descriptive_evidence": self.descriptive_evidence.to_dict(),
        }


@dataclass(frozen=True)
class FormalEvaluationOutputPaths:
    """The complete JSON, CSV and metadata output set for one report run.

    Metric outputs comprise the report JSON, per-clip table, enhanced confusion
    matrix, enhanced per-class metrics and GT-class detection recall. The final
    path is the separate evaluation-run metadata/provenance document.
    """

    report_json: Path
    per_clip_csv: Path
    confusion_matrix_csv: Path
    classification_per_class_csv: Path
    detection_recall_by_class_csv: Path
    metadata_json: Path

    def named_paths(self) -> dict[str, Path]:
        return {
            "formal_evaluation_json": self.report_json,
            "per_clip_metrics_csv": self.per_clip_csv,
            "classification_confusion_matrix_csv": (self.confusion_matrix_csv),
            "classification_per_class_csv": (self.classification_per_class_csv),
            "detection_recall_by_class_csv": (self.detection_recall_by_class_csv),
            "evaluation_metadata_json": self.metadata_json,
        }

    def all_paths(self) -> tuple[Path, ...]:
        return tuple(self.named_paths().values())


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def _validate_nonnegative_integer(
    value: object,
    *,
    field_name: str,
) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")


def _validated_contexts(
    contexts: Sequence[EvaluationClipContext],
    *,
    split: str,
) -> dict[str, EvaluationClipContext]:
    if split not in ALLOWED_SPLITS:
        raise ValueError(
            f"Unsupported evaluation split {split!r}; expected "
            f"one of {ALLOWED_SPLITS!r}"
        )

    indexed = {}

    for context in contexts:
        if not isinstance(context, EvaluationClipContext):
            raise ValueError("Clip contexts must be EvaluationClipContext instances")

        if not isinstance(context.clip_id, str) or not context.clip_id.strip():
            raise ValueError("Each evaluation clip must have a non-blank clip ID")

        if context.clip_id in indexed:
            raise ValueError(f"Duplicate evaluation clip ID: {context.clip_id!r}")

        if context.split not in ALLOWED_SPLITS:
            raise ValueError(
                f"Clip {context.clip_id!r} has unsupported split {context.split!r}"
            )

        if context.split != split:
            raise ValueError(
                "Mixed evaluation splits are not allowed within "
                f"one report; clip {context.clip_id!r} uses "
                f"{context.split!r}, report uses {split!r}"
            )

        try:
            source_fps = float(context.source_fps)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Clip {context.clip_id!r} source FPS must be a positive finite number"
            ) from error

        if not math.isfinite(source_fps) or source_fps <= 0.0:
            raise ValueError(
                f"Clip {context.clip_id!r} source FPS must be a positive finite number"
            )

        if context.evidence_metrics is not None and not isinstance(
            context.evidence_metrics,
            ClipEvidenceMetrics,
        ):
            raise ValueError(
                f"Clip {context.clip_id!r} evidence must be ClipEvidenceMetrics"
            )

        indexed[context.clip_id] = context

    return indexed


def _index_detection_results(
    results: Sequence[DetectionSummary],
    *,
    method: str,
) -> dict[str, DetectionSummary]:
    indexed = {}

    for result in results:
        if not isinstance(result, DetectionSummary):
            raise ValueError(
                f"{method.title()} results must be DetectionSummary instances"
            )

        if not result.clip_id.strip():
            raise ValueError(f"{method.title()} results require non-blank clip IDs")

        if result.clip_id in indexed:
            raise ValueError(f"Duplicate {method} clip ID: {result.clip_id!r}")

        if result.method != method:
            raise ValueError(
                f"Clip {result.clip_id!r} must use method "
                f"{method!r}, found {result.method!r}"
            )

        indexed[result.clip_id] = result

    return indexed


def _index_enhanced_results(
    results: Sequence[EnhancedClipEvaluation],
) -> dict[str, EnhancedClipEvaluation]:
    indexed = {}

    for result in results:
        if not isinstance(result, EnhancedClipEvaluation):
            raise ValueError(
                "Enhanced results must be EnhancedClipEvaluation instances"
            )

        clip_id = result.detection.clip_id

        if not clip_id.strip():
            raise ValueError("Enhanced results require non-blank clip IDs")

        if clip_id in indexed:
            raise ValueError(f"Duplicate enhanced clip ID: {clip_id!r}")

        indexed[clip_id] = result

    return indexed


def _validate_detection_summary(
    result: DetectionSummary,
    *,
    context: EvaluationClipContext,
    method: str,
    tolerance_seconds: float,
) -> None:
    if result.clip_id != context.clip_id:
        raise ValueError("Detection result clip ID does not match its context")

    if result.method != method:
        raise ValueError(
            f"Clip {context.clip_id!r} has inconsistent method {result.method!r}"
        )

    count_fields = (
        "ground_truth_event_count",
        "predicted_event_count",
        "matched_events",
        "missed_annotations",
        "extra_predictions",
        "absolute_count_error",
        "tolerance_frames",
    )

    for field_name in count_fields:
        _validate_nonnegative_integer(
            getattr(result, field_name),
            field_name=(f"{method} clip {context.clip_id!r} {field_name}"),
        )

    if (
        result.ground_truth_event_count
        != result.matched_events + result.missed_annotations
    ):
        raise ValueError(
            f"{method} clip {context.clip_id!r} ground-truth "
            "count is inconsistent with matches and misses"
        )

    if result.predicted_event_count != result.matched_events + result.extra_predictions:
        raise ValueError(
            f"{method} clip {context.clip_id!r} predicted "
            "count is inconsistent with matches and extras"
        )

    expected_signed_error = (
        result.predicted_event_count - result.ground_truth_event_count
    )

    if result.signed_count_error != expected_signed_error:
        raise ValueError(
            f"{method} clip {context.clip_id!r} signed count error is inconsistent"
        )

    if result.absolute_count_error != abs(expected_signed_error):
        raise ValueError(
            f"{method} clip {context.clip_id!r} absolute count error is inconsistent"
        )

    if not math.isclose(
        result.tolerance_seconds,
        tolerance_seconds,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"{method} clip {context.clip_id!r} uses event "
            f"tolerance {result.tolerance_seconds}, expected "
            f"{tolerance_seconds}"
        )

    expected_tolerance_frames = math.ceil(tolerance_seconds * float(context.source_fps))

    if result.tolerance_frames != expected_tolerance_frames:
        raise ValueError(
            f"{method} clip {context.clip_id!r} tolerance "
            "frames are inconsistent with the shared source FPS"
        )

    timing_means = (
        result.mean_signed_completion_timing_error_seconds,
        result.mean_absolute_completion_timing_error_seconds,
    )

    if result.matched_events == 0:
        if any(value is not None for value in timing_means):
            raise ValueError(
                f"{method} clip {context.clip_id!r} cannot have "
                "timing means without matched events"
            )
    elif any(value is None for value in timing_means):
        raise ValueError(
            f"{method} clip {context.clip_id!r} matched events require timing means"
        )


def _validate_enhanced_result(
    result: EnhancedClipEvaluation,
) -> None:
    detection = result.detection
    classification = result.classification

    if detection.method != "enhanced":
        raise ValueError(
            f"Enhanced clip {detection.clip_id!r} has method {detection.method!r}"
        )

    if classification.labels != SUPPORTED_FORM_CLASSES:
        raise ValueError(
            f"Enhanced clip {detection.clip_id!r} does not use "
            "the supported deterministic class order"
        )

    recomputed = evaluate_classification_from_confusion_matrix(
        classification.confusion_matrix,
        labels=SUPPORTED_FORM_CLASSES,
    )

    if recomputed != classification:
        raise ValueError(
            f"Enhanced clip {detection.clip_id!r} contains "
            "inconsistent classification metrics"
        )

    if classification.evaluated_matched_repetitions != detection.matched_events:
        raise ValueError(
            f"Enhanced clip {detection.clip_id!r} matched "
            "classification count must equal matched events"
        )

    if len(result.matched_pairs) != detection.matched_events:
        raise ValueError(
            f"Enhanced clip {detection.clip_id!r} matched-pair count is inconsistent"
        )

    if len(result.unmatched_prediction_ids) != detection.extra_predictions:
        raise ValueError(
            f"Enhanced clip {detection.clip_id!r} unmatched "
            "prediction count is inconsistent"
        )

    if len(result.unmatched_ground_truth_attempt_ids) != detection.missed_annotations:
        raise ValueError(
            f"Enhanced clip {detection.clip_id!r} unmatched "
            "annotation count is inconsistent"
        )

    recall_rows = result.detection_recall_by_ground_truth_class

    if tuple(row.label for row in recall_rows) != (SUPPORTED_FORM_CLASSES):
        raise ValueError(
            f"Enhanced clip {detection.clip_id!r} detection "
            "recall does not use the supported class order"
        )

    if (
        sum(row.ground_truth_support for row in recall_rows)
        != detection.ground_truth_event_count
    ):
        raise ValueError(
            f"Enhanced clip {detection.clip_id!r} stratified "
            "support does not equal ground-truth count"
        )

    if (
        sum(row.matched_ground_truth_repetitions for row in recall_rows)
        != detection.matched_events
    ):
        raise ValueError(
            f"Enhanced clip {detection.clip_id!r} stratified "
            "matches do not equal matched events"
        )

    if (
        sum(row.missed_ground_truth_repetitions for row in recall_rows)
        != detection.missed_annotations
    ):
        raise ValueError(
            f"Enhanced clip {detection.clip_id!r} stratified "
            "misses do not equal detection misses"
        )


def _aggregate_detection(
    results: Sequence[DetectionSummary],
) -> DetectionAggregate:
    """Pool event counts and matched-event timing across evaluated clips."""
    evaluated_clips = len(results)
    total_ground_truth = sum(result.ground_truth_event_count for result in results)
    total_predictions = sum(result.predicted_event_count for result in results)
    total_matches = sum(result.matched_events for result in results)
    total_misses = sum(result.missed_annotations for result in results)
    total_extras = sum(result.extra_predictions for result in results)
    precision = _safe_ratio(
        total_matches,
        total_matches + total_extras,
    )
    recall = _safe_ratio(
        total_matches,
        total_matches + total_misses,
    )
    pooled_f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0.0
        else 0.0
    )
    total_signed_error = sum(result.signed_count_error for result in results)
    total_absolute_error = sum(result.absolute_count_error for result in results)
    exact_count_clips = sum(result.signed_count_error == 0 for result in results)
    weighted_signed_timing = sum(
        (result.mean_signed_completion_timing_error_seconds or 0.0)
        * result.matched_events
        for result in results
    )
    weighted_absolute_timing = sum(
        (result.mean_absolute_completion_timing_error_seconds or 0.0)
        * result.matched_events
        for result in results
    )

    return DetectionAggregate(
        evaluated_clips=evaluated_clips,
        total_ground_truth_repetitions=total_ground_truth,
        total_predicted_repetitions=total_predictions,
        total_matched_events=total_matches,
        total_misses=total_misses,
        total_extras=total_extras,
        pooled_event_precision=precision,
        pooled_event_recall=recall,
        pooled_event_f1=pooled_f1,
        total_signed_count_error=total_signed_error,
        mean_signed_count_error=(
            total_signed_error / evaluated_clips if evaluated_clips > 0 else None
        ),
        mean_absolute_count_error=(
            total_absolute_error / evaluated_clips if evaluated_clips > 0 else None
        ),
        exact_count_clip_count=exact_count_clips,
        exact_count_clip_accuracy=(
            exact_count_clips / evaluated_clips if evaluated_clips > 0 else None
        ),
        total_matched_timing_observations=total_matches,
        mean_signed_completion_timing_error_seconds=(
            weighted_signed_timing / total_matches if total_matches > 0 else None
        ),
        mean_absolute_completion_timing_error_seconds=(
            weighted_absolute_timing / total_matches if total_matches > 0 else None
        ),
    )


def _aggregate_classification(
    enhanced_results: Sequence[EnhancedClipEvaluation],
) -> ClassificationEvaluation:
    """Pool enhanced matched classifications through summed matrix cells."""
    class_count = len(SUPPORTED_FORM_CLASSES)
    pooled_matrix = [[0 for _ in range(class_count)] for _ in range(class_count)]

    for result in enhanced_results:
        for row_index, row in enumerate(result.classification.confusion_matrix):
            for column_index, count in enumerate(row):
                pooled_matrix[row_index][column_index] += count

    return evaluate_classification_from_confusion_matrix(
        pooled_matrix,
        labels=SUPPORTED_FORM_CLASSES,
    )


def _aggregate_detection_recall_by_class(
    enhanced_results: Sequence[EnhancedClipEvaluation],
    enhanced_detection: DetectionAggregate,
) -> tuple[GroundTruthClassDetectionRecall, ...]:
    """Pool enhanced GT support, matches and misses in fixed class order."""
    rows = []

    for class_index, label in enumerate(SUPPORTED_FORM_CLASSES):
        support = sum(
            result.detection_recall_by_ground_truth_class[
                class_index
            ].ground_truth_support
            for result in enhanced_results
        )
        matched = sum(
            result.detection_recall_by_ground_truth_class[
                class_index
            ].matched_ground_truth_repetitions
            for result in enhanced_results
        )
        missed = sum(
            result.detection_recall_by_ground_truth_class[
                class_index
            ].missed_ground_truth_repetitions
            for result in enhanced_results
        )
        rows.append(
            GroundTruthClassDetectionRecall(
                label=label,
                ground_truth_support=support,
                matched_ground_truth_repetitions=matched,
                missed_ground_truth_repetitions=missed,
                recall=(matched / support if support > 0 else None),
            )
        )

    if (
        sum(row.ground_truth_support for row in rows)
        != enhanced_detection.total_ground_truth_repetitions
    ):
        raise ValueError(
            "Aggregated class support does not equal enhanced ground-truth count"
        )

    if (
        sum(row.matched_ground_truth_repetitions for row in rows)
        != enhanced_detection.total_matched_events
    ):
        raise ValueError(
            "Aggregated class matches do not equal enhanced matched events"
        )

    if (
        sum(row.missed_ground_truth_repetitions for row in rows)
        != enhanced_detection.total_misses
    ):
        raise ValueError(
            "Aggregated class misses do not equal enhanced detection misses"
        )

    return tuple(rows)


def _per_clip_evidence_fields(
    context: EvaluationClipContext,
) -> dict[str, object]:
    """Flatten one clip's descriptive evidence for the per-clip CSV."""
    evidence = context.evidence_metrics or unavailable_clip_evidence()
    baseline = evidence.baseline_frames
    enhanced = evidence.enhanced_frames
    repetitions = evidence.enhanced_repetitions
    human = evidence.human_alignment_evidence
    return {
        "source_fps": context.source_fps,
        "baseline_analyzed_frame_count": baseline.analyzed_frame_count,
        "baseline_mean_measured_processing_time_ms": (
            baseline.mean_measured_processing_time_ms
        ),
        "baseline_median_measured_processing_time_ms": (
            baseline.median_measured_processing_time_ms
        ),
        "baseline_measured_analysis_throughput_fps": (
            baseline.measured_analysis_throughput_fps
        ),
        "baseline_pose_available_frames": (baseline.pose_availability.available_frames),
        "baseline_pose_denominator_frames": (
            baseline.pose_availability.denominator_frames
        ),
        "baseline_pose_availability_rate": baseline.pose_availability.rate,
        "baseline_elbow_available_frames": (
            baseline.elbow_availability.available_frames
        ),
        "baseline_elbow_denominator_frames": (
            baseline.elbow_availability.denominator_frames
        ),
        "baseline_elbow_availability_rate": baseline.elbow_availability.rate,
        "baseline_alignment_available_frames": (
            baseline.alignment_availability.available_frames
        ),
        "baseline_alignment_denominator_frames": (
            baseline.alignment_availability.denominator_frames
        ),
        "baseline_alignment_availability_rate": (baseline.alignment_availability.rate),
        "baseline_selected_side_available_frames": (
            baseline.selected_side_availability.available_frames
        ),
        "baseline_selected_side_denominator_frames": (
            baseline.selected_side_availability.denominator_frames
        ),
        "baseline_selected_side_availability_rate": (
            baseline.selected_side_availability.rate
        ),
        "baseline_instantaneous_side_change_count": baseline.side_change_count,
        "enhanced_analyzed_frame_count": enhanced.analyzed_frame_count,
        "enhanced_mean_measured_processing_time_ms": (
            enhanced.mean_measured_processing_time_ms
        ),
        "enhanced_median_measured_processing_time_ms": (
            enhanced.median_measured_processing_time_ms
        ),
        "enhanced_measured_analysis_throughput_fps": (
            enhanced.measured_analysis_throughput_fps
        ),
        "enhanced_pose_available_frames": (enhanced.pose_availability.available_frames),
        "enhanced_pose_denominator_frames": (
            enhanced.pose_availability.denominator_frames
        ),
        "enhanced_pose_availability_rate": enhanced.pose_availability.rate,
        "enhanced_elbow_valid_frames": (enhanced.elbow_availability.available_frames),
        "enhanced_elbow_denominator_frames": (
            enhanced.elbow_availability.denominator_frames
        ),
        "enhanced_elbow_availability_rate": enhanced.elbow_availability.rate,
        "enhanced_alignment_valid_frames": (
            enhanced.alignment_availability.available_frames
        ),
        "enhanced_alignment_denominator_frames": (
            enhanced.alignment_availability.denominator_frames
        ),
        "enhanced_alignment_availability_rate": (enhanced.alignment_availability.rate),
        "enhanced_selected_side_available_frames": (
            enhanced.selected_side_availability.available_frames
        ),
        "enhanced_selected_side_denominator_frames": (
            enhanced.selected_side_availability.denominator_frames
        ),
        "enhanced_selected_side_availability_rate": (
            enhanced.selected_side_availability.rate
        ),
        "enhanced_stable_side_change_count": enhanced.side_change_count,
        "enhanced_predicted_unscorable_count": (repetitions.predicted_unscorable_count),
        "enhanced_predicted_unscorable_rate": (repetitions.predicted_unscorable_rate),
        "enhanced_alignment_coverage_observation_count": (
            repetitions.alignment_coverage_observation_count
        ),
        "enhanced_mean_alignment_valid_ratio": (repetitions.mean_alignment_valid_ratio),
        "human_evaluable_attempt_count": human.evaluable_attempt_count,
        "human_alignment_evidence_adequate_count": human.adequate_attempt_count,
        "human_alignment_evidence_inadequate_count": human.inadequate_attempt_count,
        "human_alignment_evidence_adequate_rate": human.adequate_attempt_rate,
    }


def aggregate_formal_evaluation(
    *,
    baseline_results: Sequence[DetectionSummary],
    enhanced_results: Sequence[EnhancedClipEvaluation],
    clip_contexts: Sequence[EvaluationClipContext],
    split: str,
    tolerance_seconds: float = (DEFAULT_EVENT_TOLERANCE_SECONDS),
) -> FormalEvaluationReport:
    """Pool complete baseline and enhanced results for one manifest split.

    Both methods and clip contexts must cover exactly the same clips and use one
    supplied positive finite tolerance. Per-clip tolerance frames are checked
    against each source FPS. Detection metrics pool counts across clips;
    enhanced-only classification pools confusion-matrix cells from matched
    events. Per-clip rows and class rows use deterministic ordering. No event
    matching, parameter selection or provenance capture occurs here.
    """
    try:
        tolerance = float(tolerance_seconds)
    except (TypeError, ValueError) as error:
        raise ValueError("Event tolerance must be a positive finite number") from error

    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("Event tolerance must be a positive finite number")

    contexts_by_clip = _validated_contexts(
        clip_contexts,
        split=split,
    )
    baseline_by_clip = _index_detection_results(
        baseline_results,
        method="baseline",
    )
    enhanced_by_clip = _index_enhanced_results(
        enhanced_results,
    )
    baseline_clip_ids = set(baseline_by_clip)
    enhanced_clip_ids = set(enhanced_by_clip)
    context_clip_ids = set(contexts_by_clip)

    if baseline_clip_ids != enhanced_clip_ids:
        raise ValueError(
            "Baseline and enhanced clip sets must match exactly; "
            f"baseline-only={sorted(baseline_clip_ids - enhanced_clip_ids)}, "
            f"enhanced-only={sorted(enhanced_clip_ids - baseline_clip_ids)}"
        )

    if baseline_clip_ids != context_clip_ids:
        raise ValueError(
            "Clip contexts must match the evaluated clip set exactly; "
            f"results-only={sorted(baseline_clip_ids - context_clip_ids)}, "
            f"contexts-only={sorted(context_clip_ids - baseline_clip_ids)}"
        )

    ordered_clip_ids = tuple(sorted(baseline_clip_ids))

    for clip_id in ordered_clip_ids:
        context = contexts_by_clip[clip_id]
        baseline = baseline_by_clip[clip_id]
        enhanced = enhanced_by_clip[clip_id]
        _validate_detection_summary(
            baseline,
            context=context,
            method="baseline",
            tolerance_seconds=tolerance,
        )
        _validate_detection_summary(
            enhanced.detection,
            context=context,
            method="enhanced",
            tolerance_seconds=tolerance,
        )
        _validate_enhanced_result(enhanced)

        if (
            baseline.ground_truth_event_count
            != enhanced.detection.ground_truth_event_count
        ):
            raise ValueError(
                f"Clip {clip_id!r} has inconsistent ground-truth counts between methods"
            )

    ordered_baseline = tuple(baseline_by_clip[clip_id] for clip_id in ordered_clip_ids)
    ordered_enhanced = tuple(enhanced_by_clip[clip_id] for clip_id in ordered_clip_ids)
    baseline_aggregate = _aggregate_detection(ordered_baseline)
    enhanced_aggregate = _aggregate_detection(
        tuple(result.detection for result in ordered_enhanced)
    )
    pooled_classification = _aggregate_classification(ordered_enhanced)

    if (
        pooled_classification.evaluated_matched_repetitions
        != enhanced_aggregate.total_matched_events
    ):
        raise ValueError(
            "Pooled classification count must equal enhanced matched events"
        )

    pooled_detection_recall = _aggregate_detection_recall_by_class(
        ordered_enhanced,
        enhanced_aggregate,
    )
    per_clip_rows = tuple(
        PerClipFormalMetrics(
            clip_id=clip_id,
            **_per_clip_evidence_fields(contexts_by_clip[clip_id]),
            ground_truth_repetition_count=(
                baseline_by_clip[clip_id].ground_truth_event_count
            ),
            baseline_predicted_count=(baseline_by_clip[clip_id].predicted_event_count),
            baseline_signed_count_error=(baseline_by_clip[clip_id].signed_count_error),
            baseline_absolute_count_error=(
                baseline_by_clip[clip_id].absolute_count_error
            ),
            baseline_matches=(baseline_by_clip[clip_id].matched_events),
            baseline_misses=(baseline_by_clip[clip_id].missed_annotations),
            baseline_extras=(baseline_by_clip[clip_id].extra_predictions),
            baseline_event_precision=(baseline_by_clip[clip_id].event_precision),
            baseline_event_recall=(baseline_by_clip[clip_id].event_recall),
            baseline_event_f1=(baseline_by_clip[clip_id].event_f1),
            enhanced_predicted_count=(
                enhanced_by_clip[clip_id].detection.predicted_event_count
            ),
            enhanced_signed_count_error=(
                enhanced_by_clip[clip_id].detection.signed_count_error
            ),
            enhanced_absolute_count_error=(
                enhanced_by_clip[clip_id].detection.absolute_count_error
            ),
            enhanced_matches=(enhanced_by_clip[clip_id].detection.matched_events),
            enhanced_misses=(enhanced_by_clip[clip_id].detection.missed_annotations),
            enhanced_extras=(enhanced_by_clip[clip_id].detection.extra_predictions),
            enhanced_event_precision=(
                enhanced_by_clip[clip_id].detection.event_precision
            ),
            enhanced_event_recall=(enhanced_by_clip[clip_id].detection.event_recall),
            enhanced_event_f1=(enhanced_by_clip[clip_id].detection.event_f1),
            enhanced_matched_classification_count=(
                enhanced_by_clip[clip_id].classification.evaluated_matched_repetitions
            ),
            enhanced_classification_accuracy=(
                enhanced_by_clip[clip_id].classification.accuracy
            ),
            enhanced_classification_macro_f1=(
                enhanced_by_clip[clip_id].classification.macro_f1
            ),
        )
        for clip_id in ordered_clip_ids
    )

    return FormalEvaluationReport(
        report_schema_version=(FORMAL_EVALUATION_REPORT_SCHEMA_VERSION),
        split=split,
        event_tolerance_seconds=tolerance,
        ordered_clip_ids=ordered_clip_ids,
        baseline_detection=baseline_aggregate,
        enhanced_detection=enhanced_aggregate,
        enhanced_classification=pooled_classification,
        enhanced_detection_recall_by_ground_truth_class=(pooled_detection_recall),
        per_clip_metrics=per_clip_rows,
        descriptive_evidence=aggregate_formal_evidence(
            tuple(
                contexts_by_clip[clip_id].evidence_metrics
                or unavailable_clip_evidence()
                for clip_id in ordered_clip_ids
            )
        ),
    )


def formal_evaluation_output_paths(
    output_directory: str | Path,
    run_id: str,
) -> FormalEvaluationOutputPaths:
    """Derive every output filename from one safe evaluation run ID."""
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("Evaluation run ID must be a non-blank string")

    if (
        Path(run_id).name != run_id
        or run_id in {".", ".."}
        or any(character in run_id for character in '<>:"/\\|?*')
    ):
        raise ValueError("Evaluation run ID must be a safe file-name component")

    output_root = Path(output_directory)

    return FormalEvaluationOutputPaths(
        report_json=(output_root / f"{run_id}_formal_evaluation.json"),
        per_clip_csv=(output_root / f"{run_id}_per_clip_metrics.csv"),
        confusion_matrix_csv=(
            output_root / (f"{run_id}_classification_confusion_matrix.csv")
        ),
        classification_per_class_csv=(
            output_root / f"{run_id}_classification_per_class.csv"
        ),
        detection_recall_by_class_csv=(
            output_root / f"{run_id}_detection_recall_by_class.csv"
        ),
        metadata_json=(output_root / f"{run_id}_evaluation_metadata.json"),
    )


def _write_json_file(
    output_path: Path,
    document: Mapping[str, Any],
) -> None:
    with output_path.open(
        "x",
        encoding="utf-8",
        newline="\n",
    ) as output_file:
        json.dump(
            document,
            output_file,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        output_file.write("\n")
        output_file.flush()
        os.fsync(output_file.fileno())


def _write_csv_file(
    output_path: Path,
    *,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, object]],
) -> None:
    with output_path.open(
        "x",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
        output_file.flush()
        os.fsync(output_file.fileno())


def _temporary_path(final_path: Path) -> Path:
    return final_path.with_name(f".{final_path.name}.{uuid.uuid4().hex}.tmp")


def _write_report_temporary_files(
    report: FormalEvaluationReport,
    output_paths: FormalEvaluationOutputPaths,
    staged_paths: Mapping[Path, Path],
) -> None:
    """Stage deterministic metric JSON/CSVs before final-path replacement."""
    _write_json_file(
        staged_paths[output_paths.report_json],
        report.to_dict(),
    )
    _write_csv_file(
        staged_paths[output_paths.per_clip_csv],
        fieldnames=PER_CLIP_COLUMNS,
        rows=tuple(row.to_dict() for row in report.per_clip_metrics),
    )
    confusion_columns = (
        "ground_truth_class",
        *SUPPORTED_FORM_CLASSES,
    )
    confusion_rows = []

    for row_index, label in enumerate(SUPPORTED_FORM_CLASSES):
        row: dict[str, object] = {
            "ground_truth_class": label,
        }
        row.update(
            {
                predicted_label: (
                    report.enhanced_classification.confusion_matrix[row_index][
                        column_index
                    ]
                )
                for column_index, predicted_label in enumerate(SUPPORTED_FORM_CLASSES)
            }
        )
        confusion_rows.append(row)

    _write_csv_file(
        staged_paths[output_paths.confusion_matrix_csv],
        fieldnames=confusion_columns,
        rows=confusion_rows,
    )
    _write_csv_file(
        staged_paths[output_paths.classification_per_class_csv],
        fieldnames=CLASSIFICATION_PER_CLASS_COLUMNS,
        rows=tuple(
            {
                "class_label": row.label,
                "true_positives": row.true_positives,
                "false_positives": row.false_positives,
                "false_negatives": row.false_negatives,
                "support": row.support,
                "precision": row.precision,
                "recall": row.recall,
                "f1": row.f1,
            }
            for row in report.enhanced_classification.per_class
        ),
    )
    _write_csv_file(
        staged_paths[output_paths.detection_recall_by_class_csv],
        fieldnames=DETECTION_RECALL_COLUMNS,
        rows=tuple(
            {
                "class_label": row.label,
                "support": row.ground_truth_support,
                "matched": (row.matched_ground_truth_repetitions),
                "missed": (row.missed_ground_truth_repetitions),
                "recall": row.recall,
            }
            for row in (report.enhanced_detection_recall_by_ground_truth_class)
        ),
    )


def _metadata_path(
    output_path: Path,
    repository_root: Path,
) -> str:
    resolved = output_path.resolve()

    try:
        return resolved.relative_to(repository_root.resolve()).as_posix()
    except ValueError:
        return resolved.name


def _ordered_source_run_metadata(
    report: FormalEvaluationReport,
    source_run_provenance: Sequence[SourceRunProvenance],
) -> dict[str, list[dict[str, object]]]:
    """Validate complete privacy-safe source binding and order it by method/clip."""
    records = tuple(source_run_provenance)

    if any(not isinstance(record, SourceRunProvenance) for record in records):
        raise ValueError(
            "Source-run provenance must contain only SourceRunProvenance records"
        )

    record_keys = [(record.method, record.clip_id) for record in records]

    if len(record_keys) != len(set(record_keys)):
        raise ValueError("Source-run provenance contains duplicate method/clip records")

    report_clip_ids = set(report.ordered_clip_ids)
    grouped: dict[str, list[SourceRunProvenance]] = {
        "baseline": [],
        "enhanced": [],
    }

    for record in records:
        if record.method not in grouped:
            raise ValueError(
                "Source-run provenance method must be baseline or enhanced"
            )

        for source_path in (
            record.source_metadata_path,
            record.consumed_output_path,
            record.descriptive_frame_csv_path,
        ):
            if (
                not isinstance(source_path, str)
                or not source_path.strip()
                or Path(source_path).is_absolute()
            ):
                raise ValueError(
                    "Source-run provenance paths must be "
                    "privacy-safe relative identifiers"
                )

        if record.split != report.split:
            raise ValueError(
                f"Source-run provenance for clip {record.clip_id!r} "
                "does not match the report split"
            )

        grouped[record.method].append(record)

    for method, method_records in grouped.items():
        method_clip_ids = {record.clip_id for record in method_records}

        if method_clip_ids != report_clip_ids:
            raise ValueError(
                f"{method.title()} source-run provenance must match "
                "the report clip set exactly"
            )

    return {
        method: [
            record.to_dict()
            for record in sorted(
                method_records,
                key=lambda record: record.clip_id,
            )
        ]
        for method, method_records in grouped.items()
    }


def _evidence_metadata(
    evidence: EvaluationEvidenceProvenance,
) -> dict[str, object]:
    """Validate privacy-safe immutable evidence provenance for publication."""
    if not isinstance(evidence, EvaluationEvidenceProvenance):
        raise ValueError(
            "Evidence provenance must be an EvaluationEvidenceProvenance record"
        )
    for source_path in (
        evidence.manifest_path,
        evidence.annotations_path,
        evidence.review_metadata_path,
    ):
        if (
            not isinstance(source_path, str)
            or not source_path.strip()
            or Path(source_path).is_absolute()
        ):
            raise ValueError(
                "Formal evidence paths must be privacy-safe relative identifiers"
            )
    for hash_value in (
        evidence.manifest_sha256,
        evidence.annotations_sha256,
        evidence.frozen_annotation_sha256,
        evidence.review_metadata_sha256,
    ):
        if (
            not isinstance(hash_value, str)
            or len(hash_value) != 64
            or any(character not in "0123456789abcdef" for character in hash_value)
        ):
            raise ValueError("Formal evidence hashes must be lowercase SHA-256 values")
    if evidence.review_status != "complete":
        raise ValueError("Formal annotation review status must be 'complete'")
    if evidence.annotations_sha256 != evidence.frozen_annotation_sha256:
        raise ValueError("Annotation SHA-256 does not match the frozen review hash")
    return evidence.to_dict()


def _base_evaluation_metadata(
    report: FormalEvaluationReport,
    output_paths: FormalEvaluationOutputPaths,
    *,
    run_id: str,
    repository_root: Path,
    started_utc: str,
    source_runs: Mapping[str, Sequence[Mapping[str, object]]],
    evidence: Mapping[str, object],
    software_versions: Mapping[str, Any] | None,
    git_state: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build non-metric evaluation metadata in its initial running state."""
    return {
        "metadata_schema_version": 2,
        "status": "running",
        "evaluation_run_id": run_id,
        "report_schema_version": report.report_schema_version,
        "split": report.split,
        "ordered_clip_ids": list(report.ordered_clip_ids),
        "evaluated_clip_count": len(report.ordered_clip_ids),
        "event_tolerance_seconds": (report.event_tolerance_seconds),
        "source_runs": {
            method: [dict(record) for record in records]
            for method, records in source_runs.items()
        },
        "formal_evidence": dict(evidence),
        "outputs": {
            name: _metadata_path(path, repository_root)
            for name, path in output_paths.named_paths().items()
        },
        "software": dict(
            software_versions
            if software_versions is not None
            else collect_software_versions()
        ),
        "git": dict(
            git_state if git_state is not None else collect_git_state(repository_root)
        ),
        "timestamps": {
            "started_utc": started_utc,
        },
    }


def write_formal_evaluation_report(
    report: FormalEvaluationReport,
    *,
    output_directory: str | Path,
    run_id: str,
    repository_root: str | Path,
    source_run_provenance: Sequence[SourceRunProvenance],
    evidence_provenance: EvaluationEvidenceProvenance,
    overwrite: bool = False,
    software_versions: Mapping[str, Any] | None = None,
    git_state: Mapping[str, Any] | None = None,
    timestamp_factory: Callable[[], str] = utc_timestamp,
    pre_publication_validation: Callable[[], None] | None = None,
) -> FormalEvaluationOutputPaths:
    """Write deterministic metrics plus separate evaluation provenance.

    The complete output set is collision-checked before directories or files
    are created; replacement requires explicit ``overwrite``. Metric files are
    written to same-directory temporary paths, flushed and then moved to their
    final names. Metadata is atomically replaced as running, completed or
    failed and records timestamps, software, Git state and exact source-run
    bindings; those environment-dependent fields are not part of deterministic
    scientific metric content. Temporary metric files are removed on failure.
    """
    if not isinstance(report, FormalEvaluationReport):
        raise ValueError("Report must be a FormalEvaluationReport instance")

    source_runs = _ordered_source_run_metadata(
        report,
        source_run_provenance,
    )
    evidence = _evidence_metadata(evidence_provenance)

    output_paths = formal_evaluation_output_paths(
        output_directory,
        run_id,
    )
    prepare_output_paths(
        output_paths.all_paths(),
        overwrite=overwrite,
    )
    output_root = Path(output_directory)
    output_root.mkdir(parents=True, exist_ok=True)
    repository = Path(repository_root)
    started_utc = timestamp_factory()
    base_metadata = _base_evaluation_metadata(
        report,
        output_paths,
        run_id=run_id,
        repository_root=repository,
        started_utc=started_utc,
        source_runs=source_runs,
        evidence=evidence,
        software_versions=software_versions,
        git_state=git_state,
    )
    metric_paths = tuple(
        path for path in output_paths.all_paths() if path != output_paths.metadata_json
    )
    staged_paths = {
        final_path: _temporary_path(final_path) for final_path in metric_paths
    }

    try:
        write_json_atomically(
            output_paths.metadata_json,
            base_metadata,
        )
        _write_report_temporary_files(
            report,
            output_paths,
            staged_paths,
        )

        if pre_publication_validation is not None:
            pre_publication_validation()

        for final_path in metric_paths:
            os.replace(staged_paths[final_path], final_path)

        completed_metadata = deepcopy(base_metadata)
        completed_metadata["status"] = "completed"
        completed_metadata["timestamps"] = {
            "started_utc": started_utc,
            "completed_utc": timestamp_factory(),
        }
        write_json_atomically(
            output_paths.metadata_json,
            completed_metadata,
        )
    except Exception as error:
        failed_metadata = deepcopy(base_metadata)
        failed_metadata["status"] = "failed"
        failed_metadata["timestamps"] = {
            "started_utc": started_utc,
            "failed_utc": timestamp_factory(),
        }
        failed_metadata["failure"] = {
            "error_type": type(error).__name__,
            "message": str(error),
        }

        try:
            write_json_atomically(
                output_paths.metadata_json,
                failed_metadata,
            )
        except Exception:
            pass

        raise
    finally:
        for temporary_path in staged_paths.values():
            if temporary_path.exists():
                temporary_path.unlink()

    return output_paths
