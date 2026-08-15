import pytest

from analysis.repetition_classifier import (
    RepetitionClassifier,
)
from analysis.repetition_result import CompletedRepetition


def make_repetition(
    minimum_elbow_angle=55.0,
    start_top_angle=155.0,
    end_top_angle=156.0,
    alignment_angles=None,
    duration_frames=10,
):
    if alignment_angles is None:
        alignment_angles = (170.0,) * duration_frames

    return CompletedRepetition(
        rep_id=1,
        start_frame=10,
        bottom_frame=15,
        end_frame=19,
        start_top_angle=start_top_angle,
        minimum_elbow_angle=minimum_elbow_angle,
        end_top_angle=end_top_angle,
        duration_frames=duration_frames,
        alignment_angles=tuple(alignment_angles),
    )


def test_correct_repetition():
    classifier = RepetitionClassifier()

    result = classifier.classify(make_repetition())

    assert result.predicted_class == "correct"
    assert result.triggered_rules == ()


def test_insufficient_depth_repetition():
    classifier = RepetitionClassifier()

    result = classifier.classify(make_repetition(minimum_elbow_angle=112.0))

    assert result.predicted_class == "insufficient_depth"
    assert result.insufficient_depth_triggered


def test_extension_uses_return_top_not_initial_top_angle():
    classifier = RepetitionClassifier()

    result = classifier.classify(
        make_repetition(
            start_top_angle=145.0,
            end_top_angle=156.0,
        )
    )

    assert result.predicted_class == "correct"
    assert result.top_extension_angle == 156.0
    assert not result.incomplete_extension_triggered


def test_genuinely_incomplete_return_extension_still_triggers():
    classifier = RepetitionClassifier()

    result = classifier.classify(
        make_repetition(
            start_top_angle=166.0,
            end_top_angle=139.0,
        )
    )

    assert result.predicted_class == "incomplete_extension"
    assert result.top_extension_angle == 139.0
    assert result.incomplete_extension_triggered


def test_alignment_deviation_requires_multiple_frames():
    classifier = RepetitionClassifier(
        alignment_deviation_min_frames=3,
        alignment_deviation_min_ratio=0.20,
    )

    result = classifier.classify(
        make_repetition(
            alignment_angles=(
                170.0,
                158.0,
                157.0,
                156.0,
                168.0,
                169.0,
                170.0,
                171.0,
                172.0,
                173.0,
            )
        )
    )

    assert result.predicted_class == "alignment_deviation"
    assert result.alignment_deviation_frames == 3
    assert result.alignment_deviation_triggered


def test_one_low_alignment_frame_does_not_trigger_rule():
    classifier = RepetitionClassifier(
        alignment_deviation_min_frames=3,
    )

    result = classifier.classify(
        make_repetition(
            alignment_angles=(
                170.0,
                169.0,
                155.0,
                170.0,
                171.0,
                172.0,
                170.0,
                169.0,
                168.0,
                170.0,
            )
        )
    )

    assert result.predicted_class == "correct"
    assert not result.alignment_deviation_triggered


def test_low_alignment_coverage_is_unscorable():
    classifier = RepetitionClassifier(
        minimum_alignment_valid_ratio=0.50,
    )

    result = classifier.classify(
        make_repetition(
            alignment_angles=(170.0, 171.0),
            duration_frames=10,
        )
    )

    assert result.predicted_class == "unscorable"
    assert result.alignment_valid_ratio == pytest.approx(0.20)


def test_elbow_failure_can_be_classified_with_low_alignment_coverage():
    classifier = RepetitionClassifier()

    result = classifier.classify(
        make_repetition(
            minimum_elbow_angle=110.0,
            alignment_angles=(),
            duration_frames=10,
        )
    )

    assert result.predicted_class == "insufficient_depth"


def test_multiple_triggers_are_recorded():
    classifier = RepetitionClassifier()

    result = classifier.classify(
        make_repetition(
            minimum_elbow_angle=110.0,
            start_top_angle=140.0,
            end_top_angle=145.0,
            alignment_angles=(150.0,) * 10,
        )
    )

    assert result.predicted_class == "insufficient_depth"
    assert result.multiple_rules_triggered
    assert len(result.triggered_rules) == 3
