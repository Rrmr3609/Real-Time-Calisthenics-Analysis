import pytest

from evaluation.detection_evaluation import (
    evaluate_detection_for_clip,
)
from evaluation.event_matching import (
    match_repetition_events,
)
from evaluation.repetition_events import (
    BaselineRepetitionEvent,
    GroundTruthRepetitionEvent,
)

CLIP_ID = "fictional-clip"
METHOD = "baseline"


def prediction(
    repetition_id,
    frame,
    timestamp_ms=None,
):
    return BaselineRepetitionEvent(
        run_id="fictional-run",
        clip_id=CLIP_ID,
        predicted_rep_id=repetition_id,
        completion_frame=frame,
        completion_timestamp_ms=timestamp_ms,
        resulting_cumulative_count=repetition_id,
    )


def annotation(
    attempt_id,
    frame,
    timestamp_ms=None,
):
    return GroundTruthRepetitionEvent(
        clip_id=CLIP_ID,
        ground_truth_attempt_id=attempt_id,
        completion_frame=frame,
        completion_timestamp_ms=timestamp_ms,
        ground_truth_class="correct",
    )


def match(
    predictions,
    annotations,
    *,
    fps=10.0,
    tolerance=0.5,
):
    return match_repetition_events(
        predictions,
        annotations,
        clip_id=CLIP_ID,
        method=METHOD,
        source_fps=fps,
        tolerance_seconds=tolerance,
    )


def test_exact_event_match():
    result = match(
        [prediction(1, 10, 1000.0)],
        [annotation("A001", 10, 1000.0)],
    )

    assert len(result.matched_pairs) == 1
    assert (
        result.matched_pairs[0]
        .signed_timing_error_seconds
        == 0.0
    )
    assert result.unmatched_predictions == ()
    assert result.unmatched_annotations == ()


def test_tolerance_boundary_is_inclusive():
    result = match(
        [prediction(1, 15)],
        [annotation("A001", 10, 1000.0)],
    )

    assert result.tolerance_frames == 5
    assert len(result.matched_pairs) == 1
    assert (
        result.matched_pairs[0]
        .absolute_timing_error_seconds
        == 0.5
    )


def test_prediction_just_outside_tolerance_is_unmatched():
    result = match(
        [prediction(1, 16)],
        [annotation("A001", 10, 1000.0)],
    )

    assert result.matched_pairs == ()
    assert len(result.unmatched_predictions) == 1
    assert len(result.unmatched_annotations) == 1


def test_one_annotation_near_multiple_predictions_uses_minimum_error():
    result = match(
        [prediction(1, 7), prediction(2, 11)],
        [annotation("A001", 10)],
    )

    assert len(result.matched_pairs) == 1
    assert (
        result.matched_pairs[0]
        .prediction.predicted_rep_id
        == 2
    )
    assert [
        event.predicted_rep_id
        for event in result.unmatched_predictions
    ] == [1]


def test_multiple_annotations_near_one_prediction_use_minimum_error():
    result = match(
        [prediction(1, 10)],
        [
            annotation("A001", 9),
            annotation("A002", 13),
        ],
    )

    assert len(result.matched_pairs) == 1
    assert (
        result.matched_pairs[0]
        .annotation.ground_truth_attempt_id
        == "A001"
    )
    assert [
        event.ground_truth_attempt_id
        for event in result.unmatched_annotations
    ] == ["A002"]


def test_matching_maximises_count_before_minimising_error():
    result = match(
        [prediction(1, 6), prediction(2, 11)],
        [
            annotation("A001", 10),
            annotation("A002", 14),
        ],
        tolerance=0.4,
    )

    assert [
        (
            pair.prediction.predicted_rep_id,
            pair.annotation.ground_truth_attempt_id,
        )
        for pair in result.matched_pairs
    ] == [(1, "A001"), (2, "A002")]


def test_deterministic_tie_uses_earliest_prediction():
    predictions = [
        prediction(2, 11),
        prediction(1, 9),
    ]
    result = match(
        predictions,
        [annotation("A001", 10)],
    )
    repeated = match(
        list(reversed(predictions)),
        [annotation("A001", 10)],
    )

    assert (
        result.matched_pairs[0]
        .prediction.predicted_rep_id
        == 1
    )
    assert result == repeated


def test_missed_and_extra_events_are_reported():
    result, summary = evaluate_detection_for_clip(
        [
            prediction(1, 10),
            prediction(2, 50),
        ],
        [
            annotation("A001", 10),
            annotation("A002", 30),
        ],
        clip_id=CLIP_ID,
        method=METHOD,
        source_fps=10.0,
    )

    assert len(result.matched_pairs) == 1
    assert summary.ground_truth_event_count == 2
    assert summary.predicted_event_count == 2
    assert summary.signed_count_error == 0
    assert summary.absolute_count_error == 0
    assert summary.matched_events == 1
    assert summary.missed_annotations == 1
    assert summary.extra_predictions == 1
    assert summary.event_precision == 0.5
    assert summary.event_recall == 0.5
    assert summary.event_f1 == 0.5


def test_delayed_baseline_completion_remains_extra():
    result = match(
        [prediction(1, 30)],
        [annotation("A001", 10)],
    )

    assert result.matched_pairs == ()
    assert len(result.unmatched_predictions) == 1
    assert len(result.unmatched_annotations) == 1


def test_variable_source_fps_changes_frame_tolerance():
    predictions = [prediction(1, 16)]
    annotations = [annotation("A001", 10)]

    low_fps = match(
        predictions,
        annotations,
        fps=10.0,
    )
    high_fps = match(
        predictions,
        annotations,
        fps=20.0,
    )

    assert low_fps.tolerance_frames == 5
    assert low_fps.matched_pairs == ()
    assert high_fps.tolerance_frames == 10
    assert len(high_fps.matched_pairs) == 1


def test_empty_predictions_report_all_annotations_missed():
    _, summary = evaluate_detection_for_clip(
        [],
        [annotation("A001", 10)],
        clip_id=CLIP_ID,
        method=METHOD,
        source_fps=10.0,
    )

    assert summary.predicted_event_count == 0
    assert summary.missed_annotations == 1
    assert summary.event_precision == 0.0
    assert summary.event_recall == 0.0
    assert summary.event_f1 == 0.0
    assert (
        summary.mean_signed_completion_timing_error_seconds
        is None
    )


def test_empty_annotations_report_all_predictions_extra():
    _, summary = evaluate_detection_for_clip(
        [prediction(1, 10)],
        [],
        clip_id=CLIP_ID,
        method=METHOD,
        source_fps=10.0,
    )

    assert summary.ground_truth_event_count == 0
    assert summary.extra_predictions == 1
    assert summary.event_precision == 0.0
    assert summary.event_recall == 0.0
    assert summary.event_f1 == 0.0


def test_timestamp_errors_are_signed_prediction_minus_annotation():
    result = match(
        [prediction(1, 11, 1200.0)],
        [annotation("A001", 10, 1000.0)],
    )

    pair = result.matched_pairs[0]
    assert pair.matching_basis == "timestamp"
    assert pair.signed_frame_error == 1
    assert pair.signed_timing_error_seconds == pytest.approx(
        0.2
    )
    assert pair.absolute_timing_error_seconds == pytest.approx(
        0.2
    )
