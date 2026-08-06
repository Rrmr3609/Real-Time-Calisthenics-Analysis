import json

import pytest

from evaluation.classification_evaluation import (
    SUPPORTED_FORM_CLASSES,
)
from evaluation.formal_evaluation import (
    evaluate_enhanced_clip,
)
from evaluation.repetition_events import (
    BaselineRepetitionEvent,
    EnhancedRepetitionEvent,
    GroundTruthRepetitionEvent,
)


CLIP_ID = "fictional-clip"
SOURCE_FPS = 10.0


def prediction(
    repetition_id,
    frame,
    predicted_class="correct",
    timestamp_ms=None,
):
    return EnhancedRepetitionEvent(
        run_id="fictional-run",
        clip_id=CLIP_ID,
        predicted_rep_id=repetition_id,
        start_frame=frame - 4,
        bottom_frame=frame - 2,
        completion_frame=frame,
        completion_timestamp_ms=timestamp_ms,
        predicted_class=predicted_class,
    )


def annotation(
    attempt_id,
    frame,
    ground_truth_class="correct",
    timestamp_ms=None,
):
    return GroundTruthRepetitionEvent(
        clip_id=CLIP_ID,
        ground_truth_attempt_id=attempt_id,
        completion_frame=frame,
        completion_timestamp_ms=timestamp_ms,
        ground_truth_class=ground_truth_class,
    )


def evaluate(
    predictions,
    annotations,
    *,
    tolerance=0.5,
):
    return evaluate_enhanced_clip(
        predictions,
        annotations,
        clip_id=CLIP_ID,
        source_fps=SOURCE_FPS,
        tolerance_seconds=tolerance,
    )


def recall_by_label(result):
    return {
        row.label: row
        for row in (
            result.detection_recall_by_ground_truth_class
        )
    }


def test_perfect_matching_and_classification():
    result = evaluate(
        [
            prediction(1, 10, "correct"),
            prediction(2, 30, "insufficient_depth"),
        ],
        [
            annotation("A001", 10, "correct"),
            annotation("A002", 30, "insufficient_depth"),
        ],
    )

    assert result.detection.ground_truth_event_count == 2
    assert result.detection.predicted_event_count == 2
    assert result.detection.matched_events == 2
    assert result.detection.event_f1 == 1.0
    assert result.classification.evaluated_matched_repetitions == 2
    assert result.classification.accuracy == 1.0
    assert result.classification.macro_f1 == 1.0
    assert [
        (
            pair.ground_truth_attempt_id,
            pair.predicted_rep_id,
        )
        for pair in result.matched_pairs
    ] == [("A001", 1), ("A002", 2)]


def test_matching_not_list_position_controls_classification():
    result = evaluate(
        [
            prediction(2, 10, "correct"),
            prediction(1, 30, "insufficient_depth"),
        ],
        [
            annotation("A002", 30, "insufficient_depth"),
            annotation("A001", 10, "correct"),
        ],
    )

    assert [
        (
            pair.ground_truth_attempt_id,
            pair.predicted_rep_id,
            pair.ground_truth_class,
            pair.predicted_class,
        )
        for pair in result.matched_pairs
    ] == [
        ("A001", 2, "correct", "correct"),
        (
            "A002",
            1,
            "insufficient_depth",
            "insufficient_depth",
        ),
    ]
    assert result.classification.accuracy == 1.0


def test_incorrect_matched_class_preserves_matrix_orientation():
    result = evaluate(
        [prediction(1, 10, "insufficient_depth")],
        [annotation("A001", 10, "correct")],
    )

    assert result.classification.confusion_matrix[0][1] == 1
    assert result.classification.confusion_matrix[1][0] == 0
    assert result.classification.accuracy == 0.0
    assert result.classification.macro_f1 == 0.0


def test_ground_truth_miss_is_excluded_from_classification():
    result = evaluate(
        [prediction(1, 10, "correct")],
        [
            annotation("A001", 10, "correct"),
            annotation("A002", 50, "insufficient_depth"),
        ],
    )

    assert result.detection.missed_annotations == 1
    assert result.unmatched_ground_truth_attempt_ids == (
        "A002",
    )
    assert result.classification.evaluated_matched_repetitions == 1
    assert sum(
        sum(row)
        for row in result.classification.confusion_matrix
    ) == 1


def test_extra_prediction_is_excluded_from_classification():
    result = evaluate(
        [
            prediction(1, 10, "correct"),
            prediction(2, 50, "alignment_deviation"),
        ],
        [annotation("A001", 10, "correct")],
    )

    assert result.detection.extra_predictions == 1
    assert result.unmatched_prediction_ids == (2,)
    assert result.classification.evaluated_matched_repetitions == 1
    assert result.classification.confusion_matrix[0][0] == 1
    assert sum(
        sum(row)
        for row in result.classification.confusion_matrix
    ) == 1


def test_detection_recall_is_stratified_and_totals_agree():
    result = evaluate(
        [
            prediction(1, 10, "correct"),
            prediction(2, 200, "insufficient_depth"),
            prediction(3, 500, "alignment_deviation"),
        ],
        [
            annotation("A001", 10, "correct"),
            annotation("A002", 100, "correct"),
            annotation("A003", 200, "insufficient_depth"),
            annotation("A004", 300, "incomplete_extension"),
        ],
    )
    rows = recall_by_label(result)

    assert rows["correct"].ground_truth_support == 2
    assert rows["correct"].matched_ground_truth_repetitions == 1
    assert rows["correct"].missed_ground_truth_repetitions == 1
    assert rows["correct"].recall == 0.5
    assert rows["insufficient_depth"].recall == 1.0
    assert rows["incomplete_extension"].recall == 0.0
    assert rows["alignment_deviation"].recall is None
    assert rows["unscorable"].recall is None
    assert sum(
        row.ground_truth_support for row in rows.values()
    ) == result.detection.ground_truth_event_count == 4
    assert sum(
        row.matched_ground_truth_repetitions
        for row in rows.values()
    ) == result.detection.matched_events == 2
    assert sum(
        row.missed_ground_truth_repetitions
        for row in rows.values()
    ) == result.detection.missed_annotations == 2
    assert result.detection.extra_predictions == 1


def test_empty_ground_truth_and_predictions():
    result = evaluate([], [])

    assert result.detection.ground_truth_event_count == 0
    assert result.detection.predicted_event_count == 0
    assert result.detection.matched_events == 0
    assert result.matched_pairs == ()
    assert result.classification.evaluated_matched_repetitions == 0
    assert result.classification.accuracy is None
    assert result.classification.macro_f1 is None
    assert all(
        row.ground_truth_support == 0
        and row.matched_ground_truth_repetitions == 0
        and row.missed_ground_truth_repetitions == 0
        and row.recall is None
        for row in (
            result.detection_recall_by_ground_truth_class
        )
    )


def test_empty_ground_truth_keeps_predictions_as_extras():
    result = evaluate(
        [prediction(1, 10, "correct")],
        [],
    )

    assert result.detection.extra_predictions == 1
    assert result.unmatched_prediction_ids == (1,)
    assert result.classification.evaluated_matched_repetitions == 0
    assert result.classification.accuracy is None
    assert all(
        row.recall is None
        for row in (
            result.detection_recall_by_ground_truth_class
        )
    )


def test_empty_predictions_keep_ground_truth_as_misses():
    result = evaluate(
        [],
        [
            annotation("A001", 10, "correct"),
            annotation("A002", 30, "correct"),
        ],
    )
    correct = recall_by_label(result)["correct"]

    assert result.detection.missed_annotations == 2
    assert result.unmatched_ground_truth_attempt_ids == (
        "A001",
        "A002",
    )
    assert result.classification.evaluated_matched_repetitions == 0
    assert correct.ground_truth_support == 2
    assert correct.matched_ground_truth_repetitions == 0
    assert correct.missed_ground_truth_repetitions == 2
    assert correct.recall == 0.0


@pytest.mark.parametrize("invalid_label", ["", "   ", "unknown"])
def test_invalid_ground_truth_classes_are_rejected(
    invalid_label,
):
    with pytest.raises(ValueError, match="Ground-truth attempt"):
        evaluate(
            [],
            [annotation("A001", 10, invalid_label)],
        )


@pytest.mark.parametrize("invalid_label", ["", "   ", "unknown"])
def test_invalid_enhanced_classes_are_rejected(
    invalid_label,
):
    with pytest.raises(ValueError, match="Enhanced prediction"):
        evaluate(
            [prediction(1, 10, invalid_label)],
            [],
        )


def test_baseline_event_cannot_enter_enhanced_evaluation():
    baseline_prediction = BaselineRepetitionEvent(
        run_id="fictional-run",
        clip_id=CLIP_ID,
        predicted_rep_id=1,
        completion_frame=10,
        completion_timestamp_ms=None,
        resulting_cumulative_count=1,
    )

    with pytest.raises(
        ValueError,
        match="EnhancedRepetitionEvent",
    ):
        evaluate(
            [baseline_prediction],
            [annotation("A001", 10)],
        )


def test_result_and_class_order_are_deterministic_and_json_safe():
    predictions = [
        prediction(2, 30, "insufficient_depth"),
        prediction(1, 10, "correct"),
    ]
    annotations = [
        annotation("A002", 30, "insufficient_depth"),
        annotation("A001", 10, "correct"),
    ]
    result = evaluate(predictions, annotations)
    repeated = evaluate(
        list(reversed(predictions)),
        list(reversed(annotations)),
    )
    payload = result.to_dict()

    assert result == repeated
    assert tuple(
        row.label
        for row in (
            result.detection_recall_by_ground_truth_class
        )
    ) == SUPPORTED_FORM_CLASSES
    assert payload["classification"]["labels"] == list(
        SUPPORTED_FORM_CLASSES
    )
    assert json.loads(
        json.dumps(payload, sort_keys=True)
    ) == payload


def test_existing_matcher_tolerance_is_preserved():
    result = evaluate(
        [
            prediction(1, 15, "correct"),
            prediction(2, 36, "insufficient_depth"),
        ],
        [
            annotation("A001", 10, "correct"),
            annotation("A002", 30, "insufficient_depth"),
        ],
    )

    assert result.detection.tolerance_seconds == 0.5
    assert result.detection.tolerance_frames == 5
    assert result.detection.matched_events == 1
    assert result.detection.missed_annotations == 1
    assert result.detection.extra_predictions == 1
    assert result.matched_pairs[0].ground_truth_attempt_id == "A001"
    assert result.matched_pairs[0].predicted_rep_id == 1
    assert result.matched_pairs[0].signed_frame_error == 5
    assert result.matched_pairs[0].matching_basis == "frame"
