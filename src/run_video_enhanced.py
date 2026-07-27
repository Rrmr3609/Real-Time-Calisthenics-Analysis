import argparse
import time

import cv2

from analysis.enhanced_features import EnhancedFeatureProcessor
from analysis.phase_state_machine import (

    PushUpPhaseStateMachine,
)
from capture.video import VideoFileCapture
from pose.estimator import PoseEstimator
from pose.landmarks import extract_landmarks
from utils.csv_logger import CSVLogger
from utils.paths import LOG_DIR, OUTPUT_DIR, create_project_directories
from analysis.repetition_aggregator import (
    RepetitionFeatureAggregator,
)
from analysis.repetition_classifier import (
    RepetitionClassifier,
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Run enhanced feature preprocessing on a recorded video."
        )
    )

    parser.add_argument(
        "--video",
        required=True,
        help="Path to the input video.",
    )

    parser.add_argument(
        "--clip-id",
        required=True,
        help="Identifier written into the output CSV.",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.3,
        help="Exponential moving-average alpha.",
    )

    parser.add_argument(
        "--display",
        action="store_true",
        help="Display processed frames.",
    )

    return parser.parse_args()


def format_angle(value):
    if value is None:
        return "N/A"

    return f"{value:.1f}"


def draw_text(frame, text, position, scale=0.7):
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def main():
    args = parse_arguments()
    create_project_directories()

    capture = VideoFileCapture(args.video)

    pose_estimator = PoseEstimator(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    feature_processor = EnhancedFeatureProcessor(
        smoothing_alpha=args.alpha,
        minimum_visibility=0.5,
        acquisition_frames=3,
        switch_frames=5,
        switch_margin=0.10,
        missing_grace_frames=5,
    )

    phase_machine = PushUpPhaseStateMachine(
    top_region_threshold=130.0,
    bottom_region_threshold=120.0,
    hysteresis=5.0,
    confirmation_frames=3,
    missing_grace_frames=5,
    minimum_rep_frames=8,
    )

    repetition_aggregator = RepetitionFeatureAggregator()

    repetition_classifier = RepetitionClassifier(
        depth_threshold=100.0,
        extension_threshold=150.0,
        alignment_minimum=160.0,
        alignment_deviation_min_frames=3,
        alignment_deviation_min_ratio=0.20,
        minimum_alignment_valid_ratio=0.50,
    )

    feedback_test = ""
    feedback_frames_remaining = 0


    output_path = (
        LOG_DIR
        / f"{args.clip_id}_enhanced_temporal.csv"
    )

    logger = CSVLogger(
        output_path=str(output_path),
        fieldnames=[
            "clip_id",
            "frame_index",
            "video_timestamp_ms",
            "source_fps",
            "processing_time_ms",
            "pose_detected",
            "selected_side",
            "side_changed",
            "left_elbow_visibility_score",
            "right_elbow_visibility_score",
            "elbow_feature_valid",
            "alignment_feature_valid",
            "raw_elbow_angle",
            "smoothed_elbow_angle",
            "raw_alignment_angle",
            "smoothed_alignment_angle",
            "phase",
            "phase_changed",
            "enhanced_rep_count",
            "missing_angle_frames",
            "completed_rep",
            "completed_rep_id",
            "completed_start_frame",
            "completed_bottom_frame",
            "completed_end_frame",
            "completed_start_top_angle",
            "completed_minimum_elbow_angle",
            "completed_end_top_angle",
            "completed_duration_frames",
            "repetition_predicted_class",
            "repetition_multiple_rules",
            "repetition_triggered_rules",
        ],
    )


    repetition_output_path = (
        OUTPUT_DIR
        / f"{args.clip_id}_enhanced_repetitions.csv"
    )

    repetition_logger = CSVLogger(
        output_path=str(repetition_output_path),
        fieldnames=[
            "clip_id",
            "rep_id",
            "start_frame",
            "bottom_frame",
            "end_frame",
            "duration_frames",

            "start_top_angle",
            "minimum_elbow_angle",
            "end_top_angle",
            "top_extension_angle",

            "minimum_alignment_angle",
            "alignment_valid_frames",
            "alignment_valid_ratio",
            "alignment_deviation_frames",
            "alignment_deviation_ratio",

            "insufficient_depth_triggered",
            "incomplete_extension_triggered",
            "alignment_deviation_triggered",
            "multiple_rules_triggered",
            "triggered_rules",

            "predicted_class",
            "classification_reason",

        ],
    )


    try:
        capture.open()

        print(f"Video: {args.video}")
        print(f"Frames: {capture.frame_count}")
        print(f"Source FPS: {capture.source_fps:.2f}")
        print(f"Smoothing alpha: {args.alpha}")
        print(f"Output: {output_path}")
        print(f"Repetition output: {repetition_output_path}")

        while True:
            frame = capture.read()

            if frame is None:
                break

            start_time = time.perf_counter()

            image_height, image_width = frame.shape[:2]
            results = pose_estimator.process(frame)

            pose_detected = bool(results.pose_landmarks)
            landmarks = {}

            if pose_detected:
                landmarks = extract_landmarks(
                    results,
                    image_width,
                    image_height,
                )

            feature_result = feature_processor.update(
                landmarks
            )

            phase_result = phase_machine.update(
                elbow_angle=feature_result[
                    "smoothed_elbow_angle"
                ],
                frame_index=capture.frame_index,
            )


            completed_repetition = (
                repetition_aggregator.update(
                    phase=phase_result["phase"],
                    phase_changed=phase_result[
                        "phase_changed"
                    ],
                    body_alignment_angle=feature_result[
                        "smoothed_alignment_angle"
                    ],
                    completed_repetition=phase_result[
                        "completed_repetition"
                    ],
                )
            )

            classification = None

            if completed_repetition is not None:
                classification = repetition_classifier.classify(
                    completed_repetition
                )
                repetition_logger.write_row(
                {
                    "clip_id": args.clip_id,
                    "rep_id": completed_repetition.rep_id,
                    "start_frame": (
                        completed_repetition.start_frame
                    ),
                    "bottom_frame": (
                        completed_repetition.bottom_frame
                    ),
                    "end_frame": (
                        completed_repetition.end_frame
                    ),
                    "duration_frames": (
                        completed_repetition.duration_frames
                    ),

                    "start_top_angle": (
                        completed_repetition.start_top_angle
                    ),
                    "minimum_elbow_angle": (
                        completed_repetition.minimum_elbow_angle
                    ),
                    "end_top_angle": (
                        completed_repetition.end_top_angle
                    ),
                    "top_extension_angle": (
                        classification.top_extension_angle
                    ),

                    "minimum_alignment_angle": (
                        classification.minimum_alignment_angle
                    ),
                    "alignment_valid_frames": (
                        classification.alignment_valid_frames
                    ),
                    "alignment_valid_ratio": (
                        classification.alignment_valid_ratio
                    ),
                    "alignment_deviation_frames": (
                        classification.alignment_deviation_frames
                    ),
                    "alignment_deviation_ratio": (
                        classification.alignment_deviation_ratio
                    ),

                    "insufficient_depth_triggered": (
                        classification
                        .insufficient_depth_triggered
                    ),
                    "incomplete_extension_triggered": (
                        classification
                        .incomplete_extension_triggered
                    ),
                    "alignment_deviation_triggered": (
                        classification
                        .alignment_deviation_triggered
                    ),
                    "multiple_rules_triggered": (
                        classification
                        .multiple_rules_triggered
                    ),
                    "triggered_rules": "|".join(
                        classification.triggered_rules
                    ),

                    "predicted_class": (
                        classification.predicted_class
                    ),
                    "classification_reason": (
                        classification.classification_reason
                    ),
                }
            )
                feedback_messages = {
                    "correct": (
                        "Rep complete: no predefined deviation"
                    ),
                    "insufficient_depth": (
                        "Rep complete: insufficient depth"
                    ),
                    "incomplete_extension": (
                        "Rep complete: incomplete extension"
                    ),
                    "alignment_deviation": (
                        "Rep complete: alignment deviation"
                    ),
                    "unscorable": (
                        "Rep complete: unscorable"
                    ),
                }

                feedback_text = feedback_messages[
                    classification.predicted_class
                ]

                effective_fps = (
                    capture.source_fps
                    if capture.source_fps > 0
                    else 30.0
                )

                feedback_frames_remaining = max(
                    1,
                    int(effective_fps * 1.5),
                )

            elif feedback_frames_remaining > 0:
                feedback_frames_remaining -= 1

            else:
                feedback_text = ""

            completed = completed_repetition

            processing_time_ms = (
                time.perf_counter() - start_time
            ) * 1000.0

            completed_fields = {
                "completed_rep": completed is not None,
                "completed_rep_id": None,
                "completed_start_frame": None,
                "completed_bottom_frame": None,
                "completed_end_frame": None,
                "completed_start_top_angle": None,
                "completed_minimum_elbow_angle": None,
                "completed_end_top_angle": None,
                "completed_duration_frames": None,
            }

            if completed is not None:
                completed_fields.update(
                    {
                        "completed_rep_id": completed.rep_id,
                        "completed_start_frame": (
                            completed.start_frame
                        ),
                        "completed_bottom_frame": (
                            completed.bottom_frame
                        ),
                        "completed_end_frame": (
                            completed.end_frame
                        ),
                        "completed_start_top_angle": (
                            completed.start_top_angle
                        ),
                        "completed_minimum_elbow_angle": (
                            completed.minimum_elbow_angle
                        ),
                        "completed_end_top_angle": (
                            completed.end_top_angle
                        ),
                        "completed_duration_frames": (
                            completed.duration_frames
                        ),
                    }
                )



            classification_fields = {
                "repetition_predicted_class": None,
                "repetition_multiple_rules": None,
                "repetition_triggered_rules": None,
            }


            if classification is not None:
                classification_fields.update(
                    {
                        "repetition_predicted_class": (
                            classification.predicted_class
                        ),
                        "repetition_multiple_rules": (
                            classification.multiple_rules_triggered
                        ),
                        "repetition_triggered_rules": "|".join(
                            classification.triggered_rules
                        ),
                    }
                )

            logger.write_row(
                {
                    "clip_id": args.clip_id,
                    "frame_index": capture.frame_index,
                    "video_timestamp_ms": capture.timestamp_ms(),
                    "source_fps": capture.source_fps,
                    "processing_time_ms": processing_time_ms,
                    "pose_detected": pose_detected,
                    **feature_result,
                    "phase": phase_result["phase"],
                    "phase_changed": phase_result[
                        "phase_changed"
                    ],
                    "enhanced_rep_count": phase_result[
                        "rep_count"
                    ],
                    "missing_angle_frames": phase_result[
                        "missing_angle_frames"
                    ],
                    **completed_fields,
                    **classification_fields,
                }
            )

            if args.display:
                pose_estimator.draw_landmarks(
                    frame,
                    results,
                )

                draw_text(
                    frame,
                    (
                        "Stable side: "
                        f"{feature_result['selected_side']}"
                    ),
                    (20, 40),
                )

                draw_text(
                    frame,
                    (
                        "Raw elbow: "
                        f"{format_angle(feature_result['raw_elbow_angle'])}"
                    ),
                    (20, 80),
                )

                draw_text(
                    frame,
                    (
                        "Smoothed elbow: "
                        f"{format_angle(feature_result['smoothed_elbow_angle'])}"
                    ),
                    (20, 120),
                )

                draw_text(
                    frame,
                    (
                        "Raw alignment: "
                        f"{format_angle(feature_result['raw_alignment_angle'])}"
                    ),
                    (20, 160),
                )

                draw_text(
                    frame,
                    (
                        "Smoothed alignment: "
                        f"{format_angle(feature_result['smoothed_alignment_angle'])}"
                    ),
                    (20, 200),
                )

                draw_text(
                    frame,
                    f"Phase: {phase_result['phase']}",
                    (20, 240),
                )

                draw_text(
                    frame,
                    (
                        "Enhanced repetitions: "
                        f"{phase_result['rep_count']}"
                    ),
                    (20, 280),
                )

                if feedback_text:
                    draw_text(
                        frame,
                        feedback_text,
                        (20, 320),
                        scale=0.8,
                    )

                cv2.imshow(
                    "Enhanced Preprocessing Development",
                    frame,
                )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        print(
            "Final enhanced repetition count:", 
            phase_machine.rep_count,
        )
        print(
            "Repetition-level CSV:",
            repetition_output_path,
        )

    finally:
        logger.close()
        repetition_logger.close
        capture.release()
        pose_estimator.close()
        cv2.destroyAllWindows()





if __name__ == "__main__":
    main()