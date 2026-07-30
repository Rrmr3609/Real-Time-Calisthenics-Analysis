import argparse
import time
from contextlib import ExitStack
from pathlib import Path

import cv2

from analysis.enhanced_features import EnhancedFeatureProcessor
from analysis.phase_state_machine import PushUpPhaseStateMachine
from analysis.repetition_aggregator import (
    RepetitionFeatureAggregator,
)
from analysis.repetition_classifier import (
    RepetitionClassifier,
)
from capture.video import VideoFileCapture
from config.runtime import (
    ALLOWED_SPLITS,
    apply_cli_overrides,
    load_runtime_config,
)
from pose.estimator import PoseEstimator
from pose.landmarks import extract_landmarks
from utils.csv_logger import (
    CSVLogger,
    ensure_output_paths_available,
    prepare_output_paths,
)
from utils.paths import (
    LOG_DIR,
    OUTPUT_DIR,
    PROJECT_ROOT,
    create_project_directories,
)
from utils.run_provenance import (
    RunMetadataRecorder,
    create_run_metadata,
)


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"

PROCESSING_TIME_DEFINITION = (
    "Per-frame processing_time_ms starts immediately before image-size "
    "inspection and pose estimation, and ends after feature processing, "
    "phase segmentation, repetition aggregation and classification. It "
    "excludes CSV serialisation, optional display rendering, video decoding "
    "and run setup."
)


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Run enhanced push-up analysis on a recorded video."
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
        help="Stable identifier for the source clip.",
    )

    parser.add_argument(
        "--split",
        required=True,
        choices=ALLOWED_SPLITS,
        help="Frozen dataset split for this clip.",
    )

    parser.add_argument(
        "--run-id",
        help=(
            "Unique output-run identifier. Defaults to --clip-id."
        ),
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=(
            "Runtime YAML configuration "
            "(default: configs/default.yaml)."
        ),
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help=(
            "Explicitly override features.ema_alpha from the "
            "runtime configuration."
        ),
    )

    parser.add_argument(
        "--display",
        action="store_true",
        help="Display processed frames.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the complete output set for this run ID.",
    )

    return parser.parse_args(argv)


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


def _capture_metadata(base_metadata, capture):
    source_video = dict(base_metadata["input_video"])
    source_video.update(
        {
            "source_fps": capture.source_fps,
            "frame_count": capture.frame_count,
            "resolution": {
                "width_px": getattr(
                    capture,
                    "width_px",
                    0,
                ),
                "height_px": getattr(
                    capture,
                    "height_px",
                    0,
                ),
            },
        }
    )
    return source_video


def _build_analysis_components(config):
    feature_processor = EnhancedFeatureProcessor(
        smoothing_alpha=config.features.ema_alpha,
        minimum_visibility=(
            config.features.minimum_landmark_visibility
        ),
        acquisition_frames=(
            config.features.side_acquisition_frames
        ),
        switch_frames=config.features.side_switch_frames,
        switch_margin=config.features.side_switch_margin,
        missing_grace_frames=(
            config.features.missing_side_grace_frames
        ),
    )

    phase_machine = PushUpPhaseStateMachine(
        top_region_threshold=(
            config.segmentation.top_region_threshold
        ),
        bottom_region_threshold=(
            config.segmentation.bottom_region_threshold
        ),
        hysteresis=config.segmentation.hysteresis,
        confirmation_frames=(
            config.segmentation.phase_confirmation_frames
        ),
        missing_grace_frames=(
            config.segmentation.missing_angle_grace_frames
        ),
        minimum_rep_frames=(
            config.segmentation.minimum_repetition_frames
        ),
    )

    repetition_classifier = RepetitionClassifier(
        depth_threshold=(
            config.classification.depth_threshold
        ),
        extension_threshold=(
            config.classification.extension_threshold
        ),
        alignment_minimum=(
            config.classification.alignment_minimum
        ),
        alignment_deviation_min_frames=(
            config.classification
            .alignment_deviation_min_frames
        ),
        alignment_deviation_min_ratio=(
            config.classification
            .alignment_deviation_min_ratio
        ),
        minimum_alignment_valid_ratio=(
            config.classification
            .minimum_alignment_valid_ratio
        ),
    )

    return (
        feature_processor,
        phase_machine,
        repetition_classifier,
    )


def main():
    args = parse_arguments()
    create_project_directories()

    loaded_config = load_runtime_config(args.config)
    config, explicit_overrides = apply_cli_overrides(
        loaded_config,
        ema_alpha=args.alpha,
    )
    run_id = args.run_id or args.clip_id

    output_path = (
        LOG_DIR / f"{run_id}_enhanced_temporal.csv"
    )
    repetition_output_path = (
        OUTPUT_DIR / f"{run_id}_enhanced_repetitions.csv"
    )
    metadata_path = (
        OUTPUT_DIR / f"{run_id}_enhanced_metadata.json"
    )
    output_paths = {
        "frame_csv": output_path,
        "repetition_csv": repetition_output_path,
        "metadata_json": metadata_path,
    }

    ensure_output_paths_available(
        output_paths.values(),
        overwrite=args.overwrite,
    )
    base_metadata = create_run_metadata(
        run_id=run_id,
        clip_id=args.clip_id,
        method="enhanced",
        split=args.split,
        video_path=args.video,
        config_path=args.config,
        resolved_config=config.to_dict(),
        explicit_config_overrides=explicit_overrides,
        repository_root=PROJECT_ROOT,
        output_paths=output_paths,
        processing_time_definition=(
            PROCESSING_TIME_DEFINITION
        ),
        display_enabled=args.display,
        overwrite_requested=args.overwrite,
    )

    prepare_output_paths(
        output_paths.values(),
        overwrite=args.overwrite,
    )
    metadata_recorder = RunMetadataRecorder(
        metadata_path,
        base_metadata,
    )

    feedback_text = ""
    feedback_frames_remaining = 0
    capture = None
    processed_frames = 0
    measured_processing_seconds = 0.0
    loop_started = None
    termination_reason = "end_of_stream"

    try:
        with ExitStack() as cleanup:
            cleanup.callback(cv2.destroyAllWindows)

            (
                feature_processor,
                phase_machine,
                repetition_classifier,
            ) = _build_analysis_components(config)
            repetition_aggregator = (
                RepetitionFeatureAggregator()
            )

            capture = VideoFileCapture(args.video)
            cleanup.callback(capture.release)
            capture.open()

            pose_estimator = PoseEstimator(
                min_detection_confidence=(
                    config.pose.minimum_detection_confidence
                ),
                min_tracking_confidence=(
                    config.pose.minimum_tracking_confidence
                ),
            )
            cleanup.callback(pose_estimator.close)

            logger = CSVLogger(
                output_path=str(output_path),
                fieldnames=[
                    "run_id",
                    "clip_id",
                    "frame_index",
                    "video_timestamp_ms",
                    "source_fps",
                    "processing_time_ms",
                    "pose_detected",
                    "selected_side",
                    "selected_elbow_side",
                    "side_changed",
                    "left_elbow_visibility_score",
                    "right_elbow_visibility_score",
                    "left_alignment_visibility_score",
                    "right_alignment_visibility_score",
                    "elbow_feature_valid",
                    "alignment_feature_valid",
                    "opposite_alignment_feature_valid",
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
            cleanup.callback(logger.close)

            repetition_logger = CSVLogger(
                output_path=str(repetition_output_path),
                fieldnames=[
                    "run_id",
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
            cleanup.callback(repetition_logger.close)

            print(f"Video: {args.video}")
            print(f"Run ID: {run_id}")
            print(f"Split: {args.split}")
            print(f"Config: {args.config}")
            print(f"Frames: {capture.frame_count}")
            print(
                f"Source FPS: {capture.source_fps:.2f}"
            )
            print(
                "Smoothing alpha: "
                f"{config.features.ema_alpha}"
            )
            print(f"Output: {output_path}")
            print(
                f"Repetition output: "
                f"{repetition_output_path}"
            )
            print(f"Metadata: {metadata_path}")

            loop_started = time.perf_counter()

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
                        frame_index=capture.frame_index,
                        repetition_window_start_frame=(
                            phase_result[
                                "repetition_window_start_frame"
                            ]
                        ),
                        body_alignment_angle=(
                            feature_result[
                                "smoothed_alignment_angle"
                            ]
                        ),
                        completed_repetition=(
                            phase_result[
                                "completed_repetition"
                            ]
                        ),
                    )
                )

                classification = None

                if completed_repetition is not None:
                    classification = (
                        repetition_classifier.classify(
                            completed_repetition
                        )
                    )
                    repetition_logger.write_row(
                        {
                            "run_id": run_id,
                            "clip_id": args.clip_id,
                            "rep_id": (
                                completed_repetition.rep_id
                            ),
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
                                completed_repetition
                                .duration_frames
                            ),
                            "start_top_angle": (
                                completed_repetition
                                .start_top_angle
                            ),
                            "minimum_elbow_angle": (
                                completed_repetition
                                .minimum_elbow_angle
                            ),
                            "end_top_angle": (
                                completed_repetition
                                .end_top_angle
                            ),
                            "top_extension_angle": (
                                classification
                                .top_extension_angle
                            ),
                            "minimum_alignment_angle": (
                                classification
                                .minimum_alignment_angle
                            ),
                            "alignment_valid_frames": (
                                classification
                                .alignment_valid_frames
                            ),
                            "alignment_valid_ratio": (
                                classification
                                .alignment_valid_ratio
                            ),
                            "alignment_deviation_frames": (
                                classification
                                .alignment_deviation_frames
                            ),
                            "alignment_deviation_ratio": (
                                classification
                                .alignment_deviation_ratio
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
                                classification
                                .predicted_class
                            ),
                            "classification_reason": (
                                classification
                                .classification_reason
                            ),
                        }
                    )
                    feedback_messages = {
                        "correct": (
                            "Rep complete: no predefined "
                            "deviation"
                        ),
                        "insufficient_depth": (
                            "Rep complete: insufficient depth"
                        ),
                        "incomplete_extension": (
                            "Rep complete: incomplete "
                            "extension"
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

                processing_time_ms = (
                    time.perf_counter() - start_time
                ) * 1000.0
                processed_frames += 1
                measured_processing_seconds += (
                    processing_time_ms / 1000.0
                )

                completed_fields = {
                    "completed_rep": (
                        completed_repetition is not None
                    ),
                    "completed_rep_id": None,
                    "completed_start_frame": None,
                    "completed_bottom_frame": None,
                    "completed_end_frame": None,
                    "completed_start_top_angle": None,
                    "completed_minimum_elbow_angle": None,
                    "completed_end_top_angle": None,
                    "completed_duration_frames": None,
                }

                if completed_repetition is not None:
                    completed_fields.update(
                        {
                            "completed_rep_id": (
                                completed_repetition.rep_id
                            ),
                            "completed_start_frame": (
                                completed_repetition.start_frame
                            ),
                            "completed_bottom_frame": (
                                completed_repetition.bottom_frame
                            ),
                            "completed_end_frame": (
                                completed_repetition.end_frame
                            ),
                            "completed_start_top_angle": (
                                completed_repetition
                                .start_top_angle
                            ),
                            "completed_minimum_elbow_angle": (
                                completed_repetition
                                .minimum_elbow_angle
                            ),
                            "completed_end_top_angle": (
                                completed_repetition
                                .end_top_angle
                            ),
                            "completed_duration_frames": (
                                completed_repetition
                                .duration_frames
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
                                classification
                                .multiple_rules_triggered
                            ),
                            "repetition_triggered_rules": (
                                "|".join(
                                    classification
                                    .triggered_rules
                                )
                            ),
                        }
                    )

                logger.write_row(
                    {
                        "run_id": run_id,
                        "clip_id": args.clip_id,
                        "frame_index": capture.frame_index,
                        "video_timestamp_ms": (
                            capture.timestamp_ms()
                        ),
                        "source_fps": capture.source_fps,
                        "processing_time_ms": (
                            processing_time_ms
                        ),
                        "pose_detected": pose_detected,
                        **feature_result,
                        "phase": phase_result["phase"],
                        "phase_changed": (
                            phase_result["phase_changed"]
                        ),
                        "enhanced_rep_count": (
                            phase_result["rep_count"]
                        ),
                        "missing_angle_frames": (
                            phase_result[
                                "missing_angle_frames"
                            ]
                        ),
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

                    if (
                        cv2.waitKey(1) & 0xFF
                        == ord("q")
                    ):
                        termination_reason = (
                            "user_requested"
                        )
                        break

            print(
                "Final enhanced repetition count:",
                phase_machine.rep_count,
            )
            print(
                "Repetition-level CSV:",
                repetition_output_path,
            )

    except BaseException as error:
        source_video = (
            _capture_metadata(base_metadata, capture)
            if capture is not None
            else base_metadata["input_video"]
        )
        wall_seconds = (
            time.perf_counter() - loop_started
            if loop_started is not None
            else 0.0
        )
        metadata_recorder.mark_failed(
            error,
            source_video=source_video,
            processing_summary={
                "processed_frames": processed_frames,
                "measured_processing_seconds": (
                    measured_processing_seconds
                ),
                "loop_wall_seconds": wall_seconds,
                "termination_reason": "error",
            },
        )
        raise

    source_video = _capture_metadata(
        base_metadata,
        capture,
    )
    wall_seconds = (
        time.perf_counter() - loop_started
        if loop_started is not None
        else 0.0
    )
    metadata_recorder.mark_completed(
        source_video=source_video,
        processing_summary={
            "processed_frames": processed_frames,
            "measured_processing_seconds": (
                measured_processing_seconds
            ),
            "loop_wall_seconds": wall_seconds,
            "termination_reason": termination_reason,
            "processed_full_clip": (
                termination_reason == "end_of_stream"
                and (
                    capture.frame_count <= 0
                    or processed_frames == capture.frame_count
                )
            ),
        },
    )


if __name__ == "__main__":
    main()
