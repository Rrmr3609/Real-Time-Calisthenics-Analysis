"""Load and aggregate descriptive evidence for formal evaluation reports.

The metrics in this module are derived only from recorded CSV columns and
validated manual annotations. They do not alter event matching, classification
or either processing method.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import pandas as pd

from evaluation.classification_evaluation import SUPPORTED_FORM_CLASSES


@dataclass(frozen=True)
class AvailabilityMetric:
    """Available-frame count and its explicit evidence denominator."""

    available_frames: int | None
    denominator_frames: int
    rate: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "available_frames": self.available_frames,
            "denominator_frames": self.denominator_frames,
            "rate": self.rate,
        }


@dataclass(frozen=True)
class FrameEvidenceMetrics:
    """Recorded per-frame timing, availability and side stability evidence."""

    method: str
    analyzed_frame_count: int
    timing_frame_count: int
    total_measured_processing_time_ms: float | None
    mean_measured_processing_time_ms: float | None
    median_measured_processing_time_ms: float | None
    measured_analysis_throughput_fps: float | None
    pose_availability: AvailabilityMetric
    elbow_availability: AvailabilityMetric
    alignment_availability: AvailabilityMetric
    selected_side_availability: AvailabilityMetric
    side_change_count: int | None
    side_change_semantics: str
    processing_time_ms_values: tuple[float, ...] | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def to_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "analyzed_frame_count": self.analyzed_frame_count,
            "timing_frame_count": self.timing_frame_count,
            "total_measured_processing_time_ms": (
                self.total_measured_processing_time_ms
            ),
            "mean_measured_processing_time_ms": (self.mean_measured_processing_time_ms),
            "median_measured_processing_time_ms": (
                self.median_measured_processing_time_ms
            ),
            "measured_analysis_throughput_fps": (self.measured_analysis_throughput_fps),
            "pose_availability": self.pose_availability.to_dict(),
            "elbow_availability": self.elbow_availability.to_dict(),
            "alignment_availability": self.alignment_availability.to_dict(),
            "selected_side_availability": (self.selected_side_availability.to_dict()),
            "side_change_count": self.side_change_count,
            "side_change_semantics": self.side_change_semantics,
        }


@dataclass(frozen=True)
class EnhancedRepetitionEvidenceMetrics:
    """Enhanced completed-repetition evidence independent of GT matching."""

    predicted_repetition_count: int
    predicted_unscorable_count: int
    predicted_unscorable_rate: float | None
    alignment_coverage_observation_count: int
    mean_alignment_valid_ratio: float | None
    alignment_valid_ratios: tuple[float, ...] | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def to_dict(self) -> dict[str, object]:
        return {
            "predicted_repetition_count": self.predicted_repetition_count,
            "predicted_unscorable_count": self.predicted_unscorable_count,
            "predicted_unscorable_rate": self.predicted_unscorable_rate,
            "alignment_coverage_observation_count": (
                self.alignment_coverage_observation_count
            ),
            "mean_alignment_valid_ratio": self.mean_alignment_valid_ratio,
        }


@dataclass(frozen=True)
class HumanAlignmentEvidenceMetrics:
    """Human evidence adequacy derived from source-video visibility status."""

    evaluable_attempt_count: int
    adequate_attempt_count: int
    inadequate_attempt_count: int
    adequate_attempt_rate: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "evaluable_attempt_count": self.evaluable_attempt_count,
            "adequate_attempt_count": self.adequate_attempt_count,
            "inadequate_attempt_count": self.inadequate_attempt_count,
            "adequate_attempt_rate": self.adequate_attempt_rate,
            "evidence_basis": "source_video_visibility_status",
        }


@dataclass(frozen=True)
class ClipEvidenceMetrics:
    """All descriptive evidence associated with one manifest clip."""

    baseline_frames: FrameEvidenceMetrics
    enhanced_frames: FrameEvidenceMetrics
    enhanced_repetitions: EnhancedRepetitionEvidenceMetrics
    human_alignment_evidence: HumanAlignmentEvidenceMetrics

    def to_dict(self) -> dict[str, object]:
        return {
            "baseline_frames": self.baseline_frames.to_dict(),
            "enhanced_frames": self.enhanced_frames.to_dict(),
            "enhanced_repetitions": self.enhanced_repetitions.to_dict(),
            "human_alignment_evidence": self.human_alignment_evidence.to_dict(),
        }


@dataclass(frozen=True)
class FormalEvidenceMetrics:
    """Frame-weighted and repetition-weighted cross-clip evidence summary."""

    baseline_frames: FrameEvidenceMetrics
    enhanced_frames: FrameEvidenceMetrics
    enhanced_repetitions: EnhancedRepetitionEvidenceMetrics
    human_alignment_evidence: HumanAlignmentEvidenceMetrics

    def to_dict(self) -> dict[str, object]:
        return {
            "baseline_frames": self.baseline_frames.to_dict(),
            "enhanced_frames": self.enhanced_frames.to_dict(),
            "enhanced_repetitions": self.enhanced_repetitions.to_dict(),
            "human_alignment_evidence": self.human_alignment_evidence.to_dict(),
        }


def _optional_boolean_series(
    data: pd.DataFrame,
    column: str,
    *,
    source_name: str,
) -> pd.Series | None:
    if column not in data.columns:
        return None
    values = data[column].fillna("").astype(str).str.strip().str.lower()
    invalid = ~values.isin({"true", "false"})
    if invalid.any():
        rows = list(data.index[invalid])
        raise ValueError(
            f"{source_name} column {column!r} must contain true or false; "
            f"invalid rows {rows}"
        )
    return values.eq("true")


def _optional_numeric_series(
    data: pd.DataFrame,
    column: str,
    *,
    source_name: str,
    allow_missing: bool,
) -> pd.Series | None:
    if column not in data.columns:
        return None
    text = data[column].fillna("").astype(str).str.strip()
    missing = text.eq("")
    numeric = pd.to_numeric(data[column].where(~missing), errors="coerce")
    invalid = ~missing & (
        numeric.isna() | ~numeric.map(lambda value: math.isfinite(float(value)))
    )
    if not allow_missing:
        invalid |= missing
    if invalid.any():
        rows = list(data.index[invalid])
        raise ValueError(
            f"{source_name} column {column!r} contains invalid numeric "
            f"evidence at rows {rows}"
        )
    return numeric


def _availability(
    available_count: int | None,
    denominator: int,
) -> AvailabilityMetric:
    return AvailabilityMetric(
        available_frames=available_count,
        denominator_frames=(denominator if available_count is not None else 0),
        rate=(
            available_count / denominator
            if available_count is not None and denominator > 0
            else None
        ),
    )


def _validate_frame_identity(
    data: pd.DataFrame,
    *,
    source_name: str,
    expected_run_id: str,
    expected_clip_id: str,
    expected_frame_count: int,
) -> None:
    required = ("run_id", "clip_id", "frame_index")
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"{source_name} is missing required columns: {missing}")
    if len(data) != expected_frame_count:
        raise ValueError(
            f"{source_name} contains {len(data)} frames; expected "
            f"{expected_frame_count}"
        )
    run_ids = set(data["run_id"].fillna("").astype(str).str.strip())
    clip_ids = set(data["clip_id"].fillna("").astype(str).str.strip())
    if run_ids != {expected_run_id} or clip_ids != {expected_clip_id}:
        raise ValueError(
            f"{source_name} run or clip identity does not match source metadata"
        )
    frame_indices = pd.to_numeric(data["frame_index"], errors="coerce")
    invalid = frame_indices.isna() | frame_indices.lt(0) | frame_indices.mod(1).ne(0)
    if invalid.any() or not frame_indices.is_monotonic_increasing:
        raise ValueError(
            f"{source_name} frame indices must be unique increasing integers"
        )
    if frame_indices.duplicated().any():
        raise ValueError(f"{source_name} contains duplicate frame indices")


def load_frame_evidence_metrics(
    csv_path: str | Path,
    *,
    method: str,
    expected_run_id: str,
    expected_clip_id: str,
    expected_frame_count: int,
) -> FrameEvidenceMetrics:
    """Load current or historical frame evidence without fabricating fields."""
    if method not in {"baseline", "enhanced"}:
        raise ValueError("Frame evidence method must be baseline or enhanced")
    path = Path(csv_path)
    data = pd.read_csv(path)
    source_name = str(path)
    _validate_frame_identity(
        data,
        source_name=source_name,
        expected_run_id=expected_run_id,
        expected_clip_id=expected_clip_id,
        expected_frame_count=expected_frame_count,
    )
    frame_count = len(data)
    timing = _optional_numeric_series(
        data,
        "processing_time_ms",
        source_name=source_name,
        allow_missing=False,
    )
    timing_values = (
        tuple(float(value) for value in timing) if timing is not None else None
    )
    if timing_values is not None and any(value < 0.0 for value in timing_values):
        raise ValueError(f"{source_name} processing times must be non-negative")
    total_time = sum(timing_values) if timing_values is not None else None
    timing_count = len(timing_values) if timing_values is not None else 0
    pose = _optional_boolean_series(
        data,
        "pose_detected",
        source_name=source_name,
    )
    if method == "enhanced":
        elbow_valid = _optional_boolean_series(
            data,
            "elbow_feature_valid",
            source_name=source_name,
        )
        alignment_valid = _optional_boolean_series(
            data,
            "alignment_feature_valid",
            source_name=source_name,
        )
    else:
        elbow_angles = _optional_numeric_series(
            data,
            "elbow_angle",
            source_name=source_name,
            allow_missing=True,
        )
        alignment_angles = _optional_numeric_series(
            data,
            "body_alignment_angle",
            source_name=source_name,
            allow_missing=True,
        )
        elbow_valid = elbow_angles.notna() if elbow_angles is not None else None
        alignment_valid = (
            alignment_angles.notna() if alignment_angles is not None else None
        )
    selected_side = None
    side_change_count = None
    if "selected_side" in data.columns:
        selected_values = (
            data["selected_side"].fillna("none").astype(str).str.strip().str.lower()
        )
        selected_side = selected_values.isin({"left", "right"})
        if method == "baseline":
            side_change_count = (
                int(selected_values.ne(selected_values.shift()).sum() - 1)
                if frame_count
                else 0
            )
    if method == "enhanced":
        side_changed = _optional_boolean_series(
            data,
            "side_changed",
            source_name=source_name,
        )
        side_change_count = (
            int(side_changed.sum()) if side_changed is not None else None
        )
    return FrameEvidenceMetrics(
        method=method,
        analyzed_frame_count=frame_count,
        timing_frame_count=timing_count,
        total_measured_processing_time_ms=total_time,
        mean_measured_processing_time_ms=(
            total_time / timing_count
            if total_time is not None and timing_count
            else None
        ),
        median_measured_processing_time_ms=(
            float(statistics.median(timing_values)) if timing_values else None
        ),
        measured_analysis_throughput_fps=(
            timing_count * 1000.0 / total_time
            if total_time is not None and total_time > 0.0
            else None
        ),
        pose_availability=_availability(
            int(pose.sum()) if pose is not None else None,
            frame_count,
        ),
        elbow_availability=_availability(
            int(elbow_valid.sum()) if elbow_valid is not None else None,
            frame_count,
        ),
        alignment_availability=_availability(
            int(alignment_valid.sum()) if alignment_valid is not None else None,
            frame_count,
        ),
        selected_side_availability=_availability(
            int(selected_side.sum()) if selected_side is not None else None,
            frame_count,
        ),
        side_change_count=side_change_count,
        side_change_semantics=(
            "instantaneous_selected_side_state_changes"
            if method == "baseline"
            else "stable_selector_side_changed_events"
        ),
        processing_time_ms_values=timing_values,
    )


def load_enhanced_repetition_evidence_metrics(
    csv_path: str | Path,
) -> EnhancedRepetitionEvidenceMetrics:
    """Load predicted-unscorable and model alignment-coverage evidence."""
    path = Path(csv_path)
    data = pd.read_csv(path)
    source_name = str(path)
    if "predicted_class" not in data.columns:
        raise ValueError(f"{source_name} is missing predicted_class")
    classes = data["predicted_class"].fillna("").astype(str).str.strip()
    invalid_classes = sorted(set(classes) - set(SUPPORTED_FORM_CLASSES))
    if invalid_classes:
        raise ValueError(
            f"{source_name} contains unsupported predicted classes: {invalid_classes}"
        )
    repetition_count = len(data)
    unscorable_count = int(classes.eq("unscorable").sum())
    coverage = _optional_numeric_series(
        data,
        "alignment_valid_ratio",
        source_name=source_name,
        allow_missing=True,
    )
    coverage_values = (
        tuple(float(value) for value in coverage.dropna())
        if coverage is not None
        else None
    )
    if coverage_values is not None and any(
        value < 0.0 or value > 1.0 for value in coverage_values
    ):
        raise ValueError(
            f"{source_name} alignment_valid_ratio values must be between 0 and 1"
        )
    return EnhancedRepetitionEvidenceMetrics(
        predicted_repetition_count=repetition_count,
        predicted_unscorable_count=unscorable_count,
        predicted_unscorable_rate=(
            unscorable_count / repetition_count if repetition_count else None
        ),
        alignment_coverage_observation_count=(
            len(coverage_values) if coverage_values is not None else 0
        ),
        mean_alignment_valid_ratio=(
            sum(coverage_values) / len(coverage_values) if coverage_values else None
        ),
        alignment_valid_ratios=coverage_values,
    )


def human_alignment_evidence_metrics(
    annotations: pd.DataFrame,
    *,
    clip_id: str,
) -> HumanAlignmentEvidenceMetrics:
    """Summarise human assessability separately from model coverage."""
    clip_rows = annotations.loc[
        annotations["clip_id"].fillna("").astype(str).str.strip().eq(clip_id)
    ]
    evaluable = (
        clip_rows["is_evaluable_attempt"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("true")
    )
    rows = clip_rows.loc[evaluable]
    statuses = rows["source_video_visibility_status"].fillna("").astype(str).str.strip()
    total = len(rows)
    inadequate = int(statuses.eq("insufficient").sum())
    adequate = total - inadequate
    return HumanAlignmentEvidenceMetrics(
        evaluable_attempt_count=total,
        adequate_attempt_count=adequate,
        inadequate_attempt_count=inadequate,
        adequate_attempt_rate=(adequate / total if total else None),
    )


def _aggregate_availability(
    metrics: Sequence[AvailabilityMetric],
) -> AvailabilityMetric:
    available = [metric for metric in metrics if metric.available_frames is not None]
    if not available:
        return AvailabilityMetric(None, 0, None)
    numerator = sum(int(metric.available_frames) for metric in available)
    denominator = sum(metric.denominator_frames for metric in available)
    return AvailabilityMetric(
        numerator,
        denominator,
        numerator / denominator if denominator else None,
    )


def aggregate_frame_evidence(
    metrics: Sequence[FrameEvidenceMetrics],
    *,
    method: str,
) -> FrameEvidenceMetrics:
    """Pool per-frame evidence, weighting timings and rates by frames."""
    values = tuple(
        value
        for metric in metrics
        for value in (metric.processing_time_ms_values or ())
    )
    total_time = sum(values) if values else None
    side_changes = [
        metric.side_change_count
        for metric in metrics
        if metric.side_change_count is not None
    ]
    return FrameEvidenceMetrics(
        method=method,
        analyzed_frame_count=sum(metric.analyzed_frame_count for metric in metrics),
        timing_frame_count=len(values),
        total_measured_processing_time_ms=total_time,
        mean_measured_processing_time_ms=(
            total_time / len(values) if total_time is not None else None
        ),
        median_measured_processing_time_ms=(
            float(statistics.median(values)) if values else None
        ),
        measured_analysis_throughput_fps=(
            len(values) * 1000.0 / total_time
            if total_time is not None and total_time > 0.0
            else None
        ),
        pose_availability=_aggregate_availability(
            [metric.pose_availability for metric in metrics]
        ),
        elbow_availability=_aggregate_availability(
            [metric.elbow_availability for metric in metrics]
        ),
        alignment_availability=_aggregate_availability(
            [metric.alignment_availability for metric in metrics]
        ),
        selected_side_availability=_aggregate_availability(
            [metric.selected_side_availability for metric in metrics]
        ),
        side_change_count=(sum(side_changes) if side_changes else None),
        side_change_semantics=(
            "instantaneous_selected_side_state_changes"
            if method == "baseline"
            else "stable_selector_side_changed_events"
        ),
        processing_time_ms_values=values or None,
    )


def aggregate_formal_evidence(
    clip_metrics: Sequence[ClipEvidenceMetrics],
) -> FormalEvidenceMetrics:
    """Pool evidence with explicit frame and repetition denominators."""
    repetitions = [metric.enhanced_repetitions for metric in clip_metrics]
    predicted = sum(metric.predicted_repetition_count for metric in repetitions)
    unscorable = sum(metric.predicted_unscorable_count for metric in repetitions)
    coverage = tuple(
        value
        for metric in repetitions
        for value in (metric.alignment_valid_ratios or ())
    )
    human = [metric.human_alignment_evidence for metric in clip_metrics]
    human_total = sum(metric.evaluable_attempt_count for metric in human)
    human_adequate = sum(metric.adequate_attempt_count for metric in human)
    human_inadequate = sum(metric.inadequate_attempt_count for metric in human)
    return FormalEvidenceMetrics(
        baseline_frames=aggregate_frame_evidence(
            [metric.baseline_frames for metric in clip_metrics],
            method="baseline",
        ),
        enhanced_frames=aggregate_frame_evidence(
            [metric.enhanced_frames for metric in clip_metrics],
            method="enhanced",
        ),
        enhanced_repetitions=EnhancedRepetitionEvidenceMetrics(
            predicted_repetition_count=predicted,
            predicted_unscorable_count=unscorable,
            predicted_unscorable_rate=(unscorable / predicted if predicted else None),
            alignment_coverage_observation_count=len(coverage),
            mean_alignment_valid_ratio=(
                sum(coverage) / len(coverage) if coverage else None
            ),
            alignment_valid_ratios=coverage or None,
        ),
        human_alignment_evidence=HumanAlignmentEvidenceMetrics(
            evaluable_attempt_count=human_total,
            adequate_attempt_count=human_adequate,
            inadequate_attempt_count=human_inadequate,
            adequate_attempt_rate=(
                human_adequate / human_total if human_total else None
            ),
        ),
    )


def unavailable_clip_evidence() -> ClipEvidenceMetrics:
    """Represent historical absence explicitly rather than as observed zero."""

    def unavailable_frames(method: str) -> FrameEvidenceMetrics:
        return FrameEvidenceMetrics(
            method=method,
            analyzed_frame_count=0,
            timing_frame_count=0,
            total_measured_processing_time_ms=None,
            mean_measured_processing_time_ms=None,
            median_measured_processing_time_ms=None,
            measured_analysis_throughput_fps=None,
            pose_availability=AvailabilityMetric(None, 0, None),
            elbow_availability=AvailabilityMetric(None, 0, None),
            alignment_availability=AvailabilityMetric(None, 0, None),
            selected_side_availability=AvailabilityMetric(None, 0, None),
            side_change_count=None,
            side_change_semantics=(
                "instantaneous_selected_side_state_changes"
                if method == "baseline"
                else "stable_selector_side_changed_events"
            ),
        )

    return ClipEvidenceMetrics(
        baseline_frames=unavailable_frames("baseline"),
        enhanced_frames=unavailable_frames("enhanced"),
        enhanced_repetitions=EnhancedRepetitionEvidenceMetrics(0, 0, None, 0, None),
        human_alignment_evidence=HumanAlignmentEvidenceMetrics(0, 0, 0, None),
    )
