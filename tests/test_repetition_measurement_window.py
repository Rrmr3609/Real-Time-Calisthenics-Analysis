import pytest

from analysis.phase_state_machine import (
    PushUpPhaseStateMachine,
)
from analysis.repetition_aggregator import (
    RepetitionFeatureAggregator,
)
from analysis.repetition_classifier import (
    RepetitionClass,
    RepetitionClassifier,
)


def complete_repetition_sequence():
    return [
        # Confirm the initial top position.
        155.0,
        156.0,
        157.0,
        # Confirm descent.
        124.0,
        123.0,
        122.0,
        # Confirm bottom.
        119.0,
        110.0,
        90.0,
        # Confirm ascent.
        123.0,
        126.0,
        128.0,
        129.0,
        # Confirm the return to top.
        151.0,
        153.0,
        155.0,
    ]


def run_pipeline(angles, alignment_angles):
    machine = PushUpPhaseStateMachine()
    aggregator = RepetitionFeatureAggregator()
    classifier = RepetitionClassifier()
    completed = []

    for frame_index, (elbow, alignment) in enumerate(
        zip(angles, alignment_angles, strict=True)
    ):
        phase_result = machine.update(
            elbow_angle=elbow,
            frame_index=frame_index,
        )
        repetition = aggregator.update(
            frame_index=frame_index,
            repetition_window_start_frame=phase_result[
                "repetition_window_start_frame"
            ],
            body_alignment_angle=alignment,
            completed_repetition=phase_result[
                "completed_repetition"
            ],
        )

        if repetition is not None:
            completed.append(
                (repetition, classifier.classify(repetition))
            )

    return completed


def test_complete_alignment_has_full_window_coverage():
    angles = complete_repetition_sequence()

    completed = run_pipeline(
        angles,
        [170.0] * len(angles),
    )

    assert len(completed) == 1
    repetition, classification = completed[0]
    assert repetition.start_frame == 2
    assert repetition.start_top_angle == 157.0
    assert repetition.end_frame == 15
    assert repetition.end_top_angle == 155.0
    assert repetition.duration_frames == 14
    assert len(repetition.alignment_angles) == 14
    assert classification.alignment_valid_frames == 14
    assert classification.alignment_valid_ratio == 1.0
    assert classification.top_extension_angle == 155.0
    assert (
        classification.predicted_class
        == RepetitionClass.CORRECT.value
    )


def test_missing_alignment_uses_the_same_window_denominator():
    angles = complete_repetition_sequence()
    alignment = [170.0] * len(angles)
    alignment[6] = None
    alignment[11] = None

    completed = run_pipeline(angles, alignment)

    repetition, classification = completed[0]
    assert repetition.duration_frames == 14
    assert classification.alignment_valid_frames == 12
    assert classification.alignment_valid_ratio == pytest.approx(
        12 / 14
    )


def test_non_monotonic_candidate_descent_preserves_earlier_minimum():
    angles = [
        155.0,
        156.0,
        157.0,
        # The earliest candidate is deepest, then the angle rises while
        # the three-frame descent confirmation is still pending.
        90.0,
        124.0,
        123.0,
        119.0,
        118.0,
        117.0,
        126.0,
        127.0,
        128.0,
        151.0,
        153.0,
        155.0,
    ]

    completed = run_pipeline(
        angles,
        [170.0] * len(angles),
    )

    repetition, _ = completed[0]
    assert repetition.minimum_elbow_angle == 90.0
    assert repetition.bottom_frame == 3


def test_interrupted_noisy_descent_candidate_is_discarded():
    angles = [
        155.0,
        156.0,
        157.0,
        # An isolated low candidate must not survive recovery.
        40.0,
        158.0,
        159.0,
        # A later valid repetition.
        124.0,
        123.0,
        122.0,
        119.0,
        110.0,
        95.0,
        126.0,
        128.0,
        129.0,
        151.0,
        153.0,
        155.0,
    ]
    alignment = [
        140.0 if index == 3 else 170.0
        for index in range(len(angles))
    ]

    completed = run_pipeline(angles, alignment)

    assert len(completed) == 1
    repetition, classification = completed[0]
    assert repetition.start_frame == 5
    assert repetition.start_top_angle == 159.0
    assert repetition.end_frame == 17
    assert repetition.duration_frames == 13
    assert repetition.minimum_elbow_angle == 95.0
    assert repetition.bottom_frame == 11
    assert len(repetition.alignment_angles) == 13
    assert 140.0 not in repetition.alignment_angles
    assert classification.minimum_elbow_angle == 95.0
    assert (
        classification.predicted_class
        == RepetitionClass.CORRECT.value
    )


def test_end_top_uses_only_return_confirmation_frames():
    angles = [
        155.0,
        156.0,
        157.0,
        124.0,
        123.0,
        122.0,
        119.0,
        110.0,
        90.0,
        # The final ascent-confirmation value is deliberately higher
        # than all three return-to-top confirmation values.
        126.0,
        128.0,
        160.0,
        151.0,
        152.0,
        153.0,
    ]

    completed = run_pipeline(
        angles,
        [170.0] * len(angles),
    )

    repetition, classification = completed[0]
    assert repetition.end_frame == 14
    assert repetition.end_top_angle == 153.0
    assert repetition.end_top_angle >= 130.0
    assert classification.top_extension_angle == 153.0
    assert (
        classification.predicted_class
        == RepetitionClass.CORRECT.value
    )


def test_brief_missing_elbow_observations_remain_in_window():
    angles = [
        155.0,
        156.0,
        157.0,
        124.0,
        123.0,
        122.0,
        None,
        None,
        119.0,
        110.0,
        90.0,
        126.0,
        128.0,
        129.0,
        151.0,
        153.0,
        155.0,
    ]

    completed = run_pipeline(
        angles,
        [170.0] * len(angles),
    )

    repetition, classification = completed[0]
    assert repetition.start_frame == 2
    assert repetition.end_frame == 16
    assert repetition.duration_frames == 15
    assert classification.alignment_valid_frames == 15
    assert classification.alignment_valid_ratio == 1.0


def test_abandoned_attempt_does_not_leak_into_next_repetition():
    angles = [
        155.0,
        156.0,
        157.0,
        124.0,
        123.0,
        122.0,
        # Confirm a return to top without reaching bottom.
        131.0,
        132.0,
        133.0,
        # Establish a fresh top anchor.
        154.0,
        155.0,
        156.0,
        # Complete the next repetition.
        124.0,
        123.0,
        122.0,
        119.0,
        110.0,
        90.0,
        126.0,
        128.0,
        129.0,
        151.0,
        153.0,
        155.0,
    ]
    alignment = [
        140.0 if index < 9 else 170.0
        for index in range(len(angles))
    ]

    completed = run_pipeline(angles, alignment)

    assert len(completed) == 1
    repetition, classification = completed[0]
    assert repetition.start_frame == 11
    assert repetition.end_frame == 23
    assert repetition.duration_frames == 13
    assert len(repetition.alignment_angles) == 13
    assert 140.0 not in repetition.alignment_angles
    assert classification.alignment_valid_ratio == 1.0
