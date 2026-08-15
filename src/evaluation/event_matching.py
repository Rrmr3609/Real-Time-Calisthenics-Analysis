"""Match predicted and GT completion events with deterministic chronology."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

from evaluation.repetition_events import (
    GroundTruthRepetitionEvent,
    PredictedRepetitionEvent,
)

DEFAULT_EVENT_TOLERANCE_SECONDS = 0.5


@dataclass(frozen=True)
class MatchedEventPair:
    """One prediction/annotation match and its completion-time error.

    Signed errors are prediction minus GT, so positive values mean the
    predicted completion occurred later. Timing errors are expressed in
    seconds; ``matching_basis`` records whether timestamps or frames were used.
    """

    prediction: PredictedRepetitionEvent
    annotation: GroundTruthRepetitionEvent
    signed_frame_error: int
    signed_timing_error_seconds: float
    absolute_timing_error_seconds: float
    matching_basis: str


@dataclass(frozen=True)
class EventMatchResult:
    """Complete one-to-one result for one clip, method and tolerance."""

    clip_id: str
    method: str
    source_fps: float
    tolerance_seconds: float
    tolerance_frames: int
    matched_pairs: tuple[MatchedEventPair, ...]
    unmatched_predictions: tuple[PredictedRepetitionEvent, ...]
    unmatched_annotations: tuple[GroundTruthRepetitionEvent, ...]


@dataclass(frozen=True)
class _Solution:
    """One chronological dynamic-programming solution candidate."""

    pair_indices: tuple[tuple[int, int], ...]
    total_absolute_error_seconds: float


def _validate_events(
    predictions: Sequence[PredictedRepetitionEvent],
    annotations: Sequence[GroundTruthRepetitionEvent],
    *,
    clip_id: str,
    method: str,
) -> None:
    invalid_predictions = [
        event
        for event in predictions
        if event.clip_id != clip_id or event.method != method
    ]

    if invalid_predictions:
        raise ValueError("All predictions must use the requested clip ID and method")

    invalid_annotations = [event for event in annotations if event.clip_id != clip_id]

    if invalid_annotations:
        raise ValueError("All annotations must use the requested clip ID")

    prediction_ids = [event.predicted_rep_id for event in predictions]

    if len(prediction_ids) != len(set(prediction_ids)):
        raise ValueError(
            "Predicted repetition identifiers must be unique within a clip"
        )

    annotation_ids = [event.ground_truth_attempt_id for event in annotations]

    if len(annotation_ids) != len(set(annotation_ids)):
        raise ValueError(
            "Ground-truth attempt identifiers must be unique within a clip"
        )


def _timing_difference(
    prediction: PredictedRepetitionEvent,
    annotation: GroundTruthRepetitionEvent,
    *,
    source_fps: float,
    tolerance_seconds: float,
    tolerance_frames: int,
) -> tuple[bool, float, str]:
    """Return eligibility, signed seconds and the available timing basis."""
    if (
        prediction.completion_timestamp_ms is not None
        and annotation.completion_timestamp_ms is not None
    ):
        signed_seconds = (
            prediction.completion_timestamp_ms - annotation.completion_timestamp_ms
        ) / 1000.0
        allowed = abs(signed_seconds) <= tolerance_seconds + 1e-12
        return allowed, signed_seconds, "timestamp"

    signed_frame_error = prediction.completion_frame - annotation.completion_frame
    signed_seconds = signed_frame_error / source_fps
    allowed = abs(signed_frame_error) <= tolerance_frames
    return allowed, signed_seconds, "frame"


def _is_better(
    candidate: _Solution,
    incumbent: _Solution,
) -> bool:
    """Apply the matching objective and deterministic final tie-break.

    More pairs always win; equal-cardinality solutions minimise total absolute
    timing error. Remaining ties use lexicographic ordered-event indices.
    """
    candidate_count = len(candidate.pair_indices)
    incumbent_count = len(incumbent.pair_indices)

    if candidate_count != incumbent_count:
        return candidate_count > incumbent_count

    if not math.isclose(
        candidate.total_absolute_error_seconds,
        incumbent.total_absolute_error_seconds,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        return (
            candidate.total_absolute_error_seconds
            < incumbent.total_absolute_error_seconds
        )

    return candidate.pair_indices < incumbent.pair_indices


def match_repetition_events(
    predictions: Sequence[PredictedRepetitionEvent],
    annotations: Sequence[GroundTruthRepetitionEvent],
    *,
    clip_id: str,
    method: str,
    source_fps: float,
    tolerance_seconds: float = (DEFAULT_EVENT_TOLERANCE_SECONDS),
) -> EventMatchResult:
    """Match events one-to-one without allowing chronological crossings.

    Predictions and annotations are ordered by completion frame and stable
    identifier before dynamic programming. The objective first maximises the
    number of pairs within the supplied non-negative tolerance, then minimises
    total absolute timing error, and finally selects the lexicographically
    earliest sequence of ordered index pairs. This last rule makes repeated
    evaluation deterministic when scientifically equivalent solutions remain.

    If both events have recorded timestamps, tolerance and error are evaluated
    in seconds. Otherwise frame differences are used, with the supplied seconds
    converted to ``ceil(tolerance * source_fps)`` frames; returned timing errors
    remain in seconds. The default is the frozen primary formal-evaluation
    tolerance; callers may supply another non-negative value for explicitly
    identified development sensitivity analysis.
    """
    try:
        fps = float(source_fps)
        tolerance = float(tolerance_seconds)
    except (TypeError, ValueError) as error:
        raise ValueError("Source FPS and tolerance must be finite numbers") from error

    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("Source FPS must be a positive number")

    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("Event tolerance cannot be negative")

    _validate_events(
        predictions,
        annotations,
        clip_id=clip_id,
        method=method,
    )
    ordered_predictions = tuple(
        sorted(
            predictions,
            key=lambda event: (
                event.completion_frame,
                event.predicted_rep_id,
            ),
        )
    )
    ordered_annotations = tuple(
        sorted(
            annotations,
            key=lambda event: (
                event.completion_frame,
                event.ground_truth_attempt_id,
            ),
        )
    )
    tolerance_frames = math.ceil(tolerance * fps)

    @lru_cache(maxsize=None)
    def solve(
        prediction_index: int,
        annotation_index: int,
    ) -> _Solution:
        if prediction_index >= len(ordered_predictions) or annotation_index >= len(
            ordered_annotations
        ):
            return _Solution((), 0.0)

        incumbent = solve(
            prediction_index + 1,
            annotation_index,
        )
        skip_annotation = solve(
            prediction_index,
            annotation_index + 1,
        )

        if _is_better(skip_annotation, incumbent):
            incumbent = skip_annotation

        prediction = ordered_predictions[prediction_index]
        annotation = ordered_annotations[annotation_index]
        allowed, signed_seconds, _ = _timing_difference(
            prediction,
            annotation,
            source_fps=fps,
            tolerance_seconds=tolerance,
            tolerance_frames=tolerance_frames,
        )

        if allowed:
            remaining = solve(
                prediction_index + 1,
                annotation_index + 1,
            )
            matched = _Solution(
                pair_indices=(
                    (
                        prediction_index,
                        annotation_index,
                    ),
                    *remaining.pair_indices,
                ),
                total_absolute_error_seconds=(
                    abs(signed_seconds) + remaining.total_absolute_error_seconds
                ),
            )

            if _is_better(matched, incumbent):
                incumbent = matched

        return incumbent

    solution = solve(0, 0)
    matched_pairs = []

    for prediction_index, annotation_index in solution.pair_indices:
        prediction = ordered_predictions[prediction_index]
        annotation = ordered_annotations[annotation_index]
        _, signed_seconds, basis = _timing_difference(
            prediction,
            annotation,
            source_fps=fps,
            tolerance_seconds=tolerance,
            tolerance_frames=tolerance_frames,
        )
        matched_pairs.append(
            MatchedEventPair(
                prediction=prediction,
                annotation=annotation,
                signed_frame_error=(
                    prediction.completion_frame - annotation.completion_frame
                ),
                signed_timing_error_seconds=(signed_seconds),
                absolute_timing_error_seconds=abs(signed_seconds),
                matching_basis=basis,
            )
        )

    matched_prediction_indices = {
        prediction_index for prediction_index, _ in solution.pair_indices
    }
    matched_annotation_indices = {
        annotation_index for _, annotation_index in solution.pair_indices
    }

    return EventMatchResult(
        clip_id=clip_id,
        method=method,
        source_fps=fps,
        tolerance_seconds=tolerance,
        tolerance_frames=tolerance_frames,
        matched_pairs=tuple(matched_pairs),
        unmatched_predictions=tuple(
            event
            for index, event in enumerate(ordered_predictions)
            if index not in matched_prediction_indices
        ),
        unmatched_annotations=tuple(
            event
            for index, event in enumerate(ordered_annotations)
            if index not in matched_annotation_indices
        ),
    )
