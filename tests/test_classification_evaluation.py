import json

import pytest

from evaluation.classification_evaluation import (
    SUPPORTED_FORM_CLASSES,
    evaluate_classification,
)


def metrics_by_label(result):
    return {
        metrics.label: metrics
        for metrics in result.per_class
    }


def test_perfect_predictions_across_supported_classes():
    result = evaluate_classification(
        SUPPORTED_FORM_CLASSES,
        SUPPORTED_FORM_CLASSES,
    )

    assert result.evaluated_matched_repetitions == 5
    assert result.labels == SUPPORTED_FORM_CLASSES
    assert result.confusion_matrix == (
        (1, 0, 0, 0, 0),
        (0, 1, 0, 0, 0),
        (0, 0, 1, 0, 0),
        (0, 0, 0, 1, 0),
        (0, 0, 0, 0, 1),
    )
    assert result.accuracy == 1.0
    assert result.macro_f1 == 1.0

    for metrics in result.per_class:
        assert metrics.true_positives == 1
        assert metrics.false_positives == 0
        assert metrics.false_negatives == 0
        assert metrics.support == 1
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1 == 1.0


def test_mixed_predictions_have_explicit_counts_and_metrics():
    result = evaluate_classification(
        [
            "correct",
            "correct",
            "insufficient_depth",
            "incomplete_extension",
            "alignment_deviation",
        ],
        [
            "correct",
            "insufficient_depth",
            "insufficient_depth",
            "correct",
            "unscorable",
        ],
    )

    assert result.confusion_matrix == (
        (1, 1, 0, 0, 0),
        (0, 1, 0, 0, 0),
        (1, 0, 0, 0, 0),
        (0, 0, 0, 0, 1),
        (0, 0, 0, 0, 0),
    )
    per_class = metrics_by_label(result)

    assert per_class["correct"].true_positives == 1
    assert per_class["correct"].false_positives == 1
    assert per_class["correct"].false_negatives == 1
    assert per_class["correct"].support == 2
    assert per_class["correct"].precision == 0.5
    assert per_class["correct"].recall == 0.5
    assert per_class["correct"].f1 == 0.5

    depth = per_class["insufficient_depth"]
    assert depth.true_positives == 1
    assert depth.false_positives == 1
    assert depth.false_negatives == 0
    assert depth.support == 1
    assert depth.precision == 0.5
    assert depth.recall == 1.0
    assert depth.f1 == pytest.approx(2.0 / 3.0)

    assert result.accuracy == 0.4
    assert result.macro_f1 == pytest.approx(7.0 / 24.0)


def test_zero_support_prediction_is_excluded_from_macro_f1():
    result = evaluate_classification(
        ["correct", "correct"],
        ["correct", "unscorable"],
    )
    per_class = metrics_by_label(result)

    unscorable = per_class["unscorable"]
    assert unscorable.true_positives == 0
    assert unscorable.false_positives == 1
    assert unscorable.false_negatives == 0
    assert unscorable.support == 0
    assert unscorable.precision == 0.0
    assert unscorable.recall == 0.0
    assert unscorable.f1 == 0.0
    assert result.macro_f1 == pytest.approx(2.0 / 3.0)


def test_classes_absent_from_ground_truth_and_predictions_remain_zero():
    result = evaluate_classification(
        ["correct"],
        ["correct"],
    )

    assert result.confusion_matrix == (
        (1, 0, 0, 0, 0),
        (0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0),
    )
    assert result.per_class[1].to_dict() == {
        "label": "insufficient_depth",
        "true_positives": 0,
        "false_positives": 0,
        "false_negatives": 0,
        "support": 0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }
    assert result.macro_f1 == 1.0


def test_empty_matched_input_has_defined_empty_result():
    result = evaluate_classification([], [])

    assert result.evaluated_matched_repetitions == 0
    assert result.labels == SUPPORTED_FORM_CLASSES
    assert result.confusion_matrix == (
        (0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0),
    )
    assert all(
        metrics.support == 0
        and metrics.precision == 0.0
        and metrics.recall == 0.0
        and metrics.f1 == 0.0
        for metrics in result.per_class
    )
    assert result.accuracy is None
    assert result.macro_f1 is None


def test_unequal_input_lengths_are_rejected():
    with pytest.raises(ValueError, match="equal lengths"):
        evaluate_classification(
            ["correct"],
            ["correct", "correct"],
        )


def test_unknown_ground_truth_label_is_rejected():
    with pytest.raises(
        ValueError,
        match="ground_truth_labels.*unknown",
    ):
        evaluate_classification(["unknown"], ["correct"])


def test_unknown_predicted_label_is_rejected():
    with pytest.raises(
        ValueError,
        match="predicted_labels.*unknown",
    ):
        evaluate_classification(["correct"], ["unknown"])


@pytest.mark.parametrize(
    ("ground_truth", "prediction", "message"),
    [
        ([""], ["correct"], "ground_truth_labels"),
        (["correct"], ["   "], "predicted_labels"),
    ],
)
def test_blank_input_labels_are_rejected(
    ground_truth,
    prediction,
    message,
):
    with pytest.raises(ValueError, match=message):
        evaluate_classification(
            ground_truth,
            prediction,
        )


@pytest.mark.parametrize(
    ("labels", "message"),
    [
        ((), "must not be empty"),
        (("correct", "correct"), "must be unique"),
        (("correct", ""), "non-empty strings"),
        (("correct", "not_supported"), "Unsupported"),
    ],
)
def test_invalid_configured_reporting_labels_are_rejected(
    labels,
    message,
):
    with pytest.raises(ValueError, match=message):
        evaluate_classification([], [], labels=labels)


def test_custom_reporting_order_is_deterministic_and_sets_matrix_axes():
    labels = (
        "unscorable",
        "correct",
        "alignment_deviation",
    )
    result = evaluate_classification(
        ["correct", "alignment_deviation"],
        ["unscorable", "correct"],
        labels=labels,
    )

    assert result.labels == labels
    assert tuple(
        metrics.label for metrics in result.per_class
    ) == labels
    assert result.confusion_matrix == (
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
    )


def test_json_compatible_dictionary_preserves_reporting_order():
    result = evaluate_classification(
        ["correct"],
        ["insufficient_depth"],
    )

    payload = result.to_dict()
    encoded = json.dumps(payload, sort_keys=True)

    assert payload["labels"] == list(
        SUPPORTED_FORM_CLASSES
    )
    assert payload["confusion_matrix"][0][1] == 1
    assert [
        row["label"] for row in payload["per_class"]
    ] == list(SUPPORTED_FORM_CLASSES)
    assert json.loads(encoded) == payload
