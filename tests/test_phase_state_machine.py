from analysis.phase_state_machine import (
    PushUpPhaseStateMachine,
)


def run_sequence(machine, angles):
    completed = []
    final_result = None

    for frame_index, angle in enumerate(angles):
        final_result = machine.update(
            elbow_angle=angle,
            frame_index=frame_index,
        )

        repetition = final_result["completed_repetition"]

        if repetition is not None:
            completed.append(repetition)

    return final_result, completed


def correct_repetition_sequence():
    return [
        # Acquire top
        135.0,
        136.0,
        137.0,
        # Confirm descent below 125
        124.0,
        123.0,
        122.0,
        # Confirm bottom below 120
        119.0,
        115.0,
        110.0,
        # Confirm ascent above 125
        123.0,
        126.0,
        128.0,
        129.0,
        # Confirm return to top above 130
        131.0,
        133.0,
        135.0,
    ]


def test_correct_temporal_sequence_counts_one_repetition():
    machine = PushUpPhaseStateMachine()

    final_result, completed = run_sequence(
        machine,
        correct_repetition_sequence(),
    )

    assert final_result["rep_count"] == 1
    assert final_result["phase"] == "top"
    assert len(completed) == 1


def test_completed_repetition_records_minimum_angle():
    machine = PushUpPhaseStateMachine()

    _, completed = run_sequence(
        machine,
        correct_repetition_sequence(),
    )

    repetition = completed[0]

    assert repetition.start_frame == 2
    assert repetition.end_frame == 15
    assert repetition.duration_frames == 14
    assert repetition.start_top_angle == 137.0
    assert repetition.minimum_elbow_angle == 110.0
    assert repetition.bottom_frame == 8
    assert repetition.end_top_angle == 135.0


def test_shallow_attempt_does_not_count():
    machine = PushUpPhaseStateMachine()

    sequence = [
        135.0,
        136.0,
        137.0,
        # Descending begins
        124.0,
        123.0,
        122.0,
        # It never reaches the provisional bottom region
        123.0,
        125.0,
        128.0,
        131.0,
        132.0,
        133.0,
    ]

    final_result, completed = run_sequence(
        machine,
        sequence,
    )

    assert final_result["rep_count"] == 0
    assert final_result["phase"] == "top"
    assert completed == []


def test_incomplete_extension_can_still_be_segmented():
    machine = PushUpPhaseStateMachine()

    sequence = [
        135.0,
        136.0,
        137.0,
        124.0,
        123.0,
        122.0,
        119.0,
        112.0,
        105.0,
        123.0,
        126.0,
        128.0,
        129.0,
        # Above the segmentation threshold but below the future
        # 150-degree form-quality threshold.
        131.0,
        132.0,
        133.0,
    ]

    final_result, completed = run_sequence(
        machine,
        sequence,
    )

    assert final_result["rep_count"] == 1
    assert len(completed) == 1
    assert completed[0].end_top_angle < 150.0


def test_threshold_noise_does_not_start_descent():
    machine = PushUpPhaseStateMachine()

    sequence = [
        135.0,
        136.0,
        137.0,
        # No three consecutive frames below or equal to 125
        124.0,
        126.0,
        124.0,
        126.0,
        124.0,
        126.0,
        135.0,
    ]

    final_result, completed = run_sequence(
        machine,
        sequence,
    )

    assert final_result["phase"] == "top"
    assert final_result["rep_count"] == 0
    assert completed == []


def test_brief_missing_angles_do_not_reset_attempt():
    machine = PushUpPhaseStateMachine(
        missing_grace_frames=3,
    )

    sequence = [
        135.0,
        136.0,
        137.0,
        124.0,
        123.0,
        122.0,
        # Temporary tracking failure
        None,
        None,
        119.0,
        115.0,
        110.0,
        126.0,
        128.0,
        129.0,
        131.0,
        133.0,
        135.0,
    ]

    final_result, completed = run_sequence(
        machine,
        sequence,
    )

    assert final_result["rep_count"] == 1
    assert len(completed) == 1


def test_prolonged_missing_angles_reset_to_waiting():
    machine = PushUpPhaseStateMachine(
        missing_grace_frames=2,
    )

    sequence = [
        135.0,
        136.0,
        137.0,
        124.0,
        123.0,
        122.0,
        None,
        None,
        None,
    ]

    final_result, completed = run_sequence(
        machine,
        sequence,
    )

    assert final_result["phase"] == "waiting"
    assert final_result["rep_count"] == 0
    assert completed == []


def test_remaining_at_top_does_not_duplicate_count():
    machine = PushUpPhaseStateMachine()

    final_result, completed = run_sequence(
        machine,
        [135.0] * 20,
    )

    assert final_result["phase"] == "top"
    assert final_result["rep_count"] == 0
    assert completed == []
