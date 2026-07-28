from analysis.repetition_aggregator import (
    RepetitionFeatureAggregator,
)
from analysis.repetition_result import CompletedRepetition


def make_completed_repetition(
    start_frame=10,
    end_frame=30,
):
    return CompletedRepetition(
        rep_id=1,
        start_frame=start_frame,
        bottom_frame=20,
        end_frame=end_frame,
        start_top_angle=155.0,
        minimum_elbow_angle=90.0,
        end_top_angle=156.0,
        duration_frames=end_frame - start_frame + 1,
    )


def test_collects_alignment_values_during_repetition():
    aggregator = RepetitionFeatureAggregator()

    aggregator.update(
        frame_index=10,
        repetition_window_start_frame=10,
        body_alignment_angle=170.0,
        completed_repetition=None,
    )

    aggregator.update(
        frame_index=20,
        repetition_window_start_frame=10,
        body_alignment_angle=158.0,
        completed_repetition=None,
    )

    aggregator.update(
        frame_index=29,
        repetition_window_start_frame=10,
        body_alignment_angle=165.0,
        completed_repetition=None,
    )

    result = aggregator.update(
        frame_index=30,
        repetition_window_start_frame=10,
        body_alignment_angle=171.0,
        completed_repetition=make_completed_repetition(),
    )

    assert result is not None
    assert result.alignment_angles == (
        170.0,
        158.0,
        165.0,
        171.0,
    )


def test_missing_alignment_values_are_not_inserted():
    aggregator = RepetitionFeatureAggregator()

    aggregator.update(
        frame_index=10,
        repetition_window_start_frame=10,
        body_alignment_angle=170.0,
        completed_repetition=None,
    )

    aggregator.update(
        frame_index=20,
        repetition_window_start_frame=10,
        body_alignment_angle=None,
        completed_repetition=None,
    )

    result = aggregator.update(
        frame_index=30,
        repetition_window_start_frame=10,
        body_alignment_angle=None,
        completed_repetition=make_completed_repetition(),
    )

    assert result is not None
    assert result.alignment_angles == (170.0,)


def test_abandoned_attempt_clears_collected_values():
    aggregator = RepetitionFeatureAggregator()

    aggregator.update(
        frame_index=10,
        repetition_window_start_frame=10,
        body_alignment_angle=150.0,
        completed_repetition=None,
    )

    # A new window start means the previous attempt was abandoned.
    aggregator.update(
        frame_index=20,
        repetition_window_start_frame=20,
        body_alignment_angle=170.0,
        completed_repetition=None,
    )

    aggregator.update(
        frame_index=21,
        repetition_window_start_frame=20,
        body_alignment_angle=175.0,
        completed_repetition=None,
    )

    result = aggregator.update(
        frame_index=30,
        repetition_window_start_frame=20,
        body_alignment_angle=176.0,
        completed_repetition=make_completed_repetition(
            start_frame=20,
            end_frame=30,
        ),
    )

    assert result is not None
    assert 150.0 not in result.alignment_angles
    assert result.alignment_angles == (
        170.0,
        175.0,
        176.0,
    )
