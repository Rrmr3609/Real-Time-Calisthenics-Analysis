from analysis.phase_state_machine import PushUpPhaseStateMachine
from analysis.repetition_aggregator import RepetitionFeatureAggregator
from analysis.repetition_classifier import RepetitionClassifier
from analysis.return_top_finalizer import ReturnTopPeakFinalizer


def run_integrated_pipeline(angles, *, flush=True):
    machine = PushUpPhaseStateMachine()
    aggregator = RepetitionFeatureAggregator()
    finalizer = ReturnTopPeakFinalizer()
    classifier = RepetitionClassifier()
    detected = []
    finalised = []

    for frame_index, angle in enumerate(angles):
        phase_result = machine.update(
            elbow_angle=angle,
            frame_index=frame_index,
        )
        detected_repetition = aggregator.update(
            frame_index=frame_index,
            repetition_window_start_frame=phase_result["repetition_window_start_frame"],
            body_alignment_angle=170.0,
            completed_repetition=phase_result["completed_repetition"],
        )
        if detected_repetition is not None:
            detected.append((frame_index, detected_repetition))

        completed_repetition = finalizer.update(
            detected_repetition=detected_repetition,
            elbow_angle=angle,
            returned_top_phase_active=phase_result["phase"] == "top",
        )
        if completed_repetition is not None:
            finalised.append(
                (
                    frame_index,
                    completed_repetition,
                    classifier.classify(completed_repetition),
                )
            )

    if flush:
        completed_repetition = finalizer.flush()
        if completed_repetition is not None:
            finalised.append(
                (
                    None,
                    completed_repetition,
                    classifier.classify(completed_repetition),
                )
            )

    return machine, detected, finalised, finalizer


def repetition_with_return(return_angles, post_return_angles):
    return [
        155.0,
        156.0,
        157.0,
        124.0,
        123.0,
        122.0,
        119.0,
        110.0,
        90.0,
        126.0,
        128.0,
        129.0,
        *return_angles,
        *post_return_angles,
    ]


def test_later_return_peak_corrects_early_confirmation_without_moving_event():
    angles = repetition_with_return(
        [131.0, 133.0, 135.0],
        [142.0, 151.0, 155.0, 124.0, 123.0, 122.0],
    )

    machine, detected, finalised, _ = run_integrated_pipeline(angles)

    assert machine.rep_count == 1
    assert len(detected) == 1
    detection_frame, detected_repetition = detected[0]
    finalisation_frame, repetition, classification = finalised[0]
    assert detection_frame == 14
    assert detected_repetition.end_frame == 14
    assert detected_repetition.end_top_angle == 135.0
    assert finalisation_frame == 20
    assert repetition.end_frame == detected_repetition.end_frame
    assert repetition.duration_frames == detected_repetition.duration_frames
    assert repetition.end_top_angle == 155.0
    assert classification.top_extension_angle == 155.0
    assert classification.predicted_class == "correct"


def test_return_that_never_reaches_extension_threshold_remains_incomplete():
    angles = repetition_with_return(
        [131.0, 135.0, 138.0],
        [139.0, 137.0, 124.0, 123.0, 122.0],
    )

    _, _, finalised, _ = run_integrated_pipeline(angles)

    _, repetition, classification = finalised[0]
    assert repetition.start_top_angle == 157.0
    assert repetition.end_top_angle == 139.0
    assert classification.predicted_class == "incomplete_extension"


def test_low_initial_top_does_not_override_complete_return_extension():
    angles = [
        140.0,
        141.0,
        142.0,
        124.0,
        123.0,
        122.0,
        119.0,
        110.0,
        90.0,
        126.0,
        128.0,
        129.0,
        131.0,
        140.0,
        151.0,
        155.0,
        124.0,
        123.0,
        122.0,
    ]

    _, _, finalised, _ = run_integrated_pipeline(angles)

    _, repetition, classification = finalised[0]
    assert repetition.start_top_angle == 142.0
    assert repetition.end_top_angle == 155.0
    assert classification.predicted_class == "correct"


def test_stream_end_flush_preserves_final_completed_repetition():
    angles = repetition_with_return(
        [131.0, 133.0, 135.0],
        [145.0, 151.0, 154.0],
    )

    machine, detected, finalised, finalizer = run_integrated_pipeline(
        angles,
        flush=False,
    )

    assert machine.rep_count == 1
    assert len(detected) == 1
    assert finalised == []

    repetition = finalizer.flush()
    classification = RepetitionClassifier().classify(repetition)
    assert repetition.rep_id == 1
    assert repetition.end_frame == detected[0][1].end_frame
    assert repetition.end_top_angle == 154.0
    assert classification.predicted_class == "correct"
    assert finalizer.flush() is None
