"""Process one recorded video with the intentionally simple baseline.

This runner loads validated runtime configuration, performs raw baseline
analysis and writes a frame-level CSV plus completed/failed run metadata. It
processes the video itself; event matching and formal evaluation consume its
outputs later and are deliberately separate.
"""

import argparse
import time
from contextlib import ExitStack
from pathlib import Path

import cv2

from analysis.baseline import BaselinePushUpAnalyser
from capture.video import VideoFileCapture
from config.runtime import ALLOWED_SPLITS, load_runtime_config
from features.angles import calculate_angle
from pose.estimator import PoseEstimator
from pose.landmarks import (
    extract_landmarks,
    feature_landmarks_available,
    feature_visibility_score,
    get_point,
    select_best_elbow_side,
)
from utils.csv_logger import (
    CSVLogger,
    ensure_output_paths_available,
    prepare_output_paths,
)
from utils.paths import (
    LOG_DIR,
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
    "inspection and pose estimation, and ends after landmark extraction, "
    "feature calculation and baseline analysis. It excludes CSV "
    "serialisation, optional display rendering, video decoding and run setup."
)


def parse_arguments(argv=None):
    """Parse recorded-baseline input, identity, config and output options.

    The split is restricted to the shared development/test vocabulary. Run ID
    defaults to clip ID, and replacing the complete output set requires the
    explicit overwrite flag.
    """
    parser = argparse.ArgumentParser(
        description="Run the baseline analyser on a recorded video."
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


def _capture_metadata(base_metadata, capture):
    """Combine immutable input identity with OpenCV source metadata."""
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


def main():
    """Run baseline video processing with complete output provenance.

    The configured video is decoded here and analysed frame by frame. Output
    paths are checked as one set before capture or pose estimation begins;
    existing files fail unless overwrite was explicitly requested. The runner
    writes ``<run_id>_baseline.csv`` and ``<run_id>_baseline_metadata.json``
    under ``experiments/logs``. The frame CSV records source identity,
    measurements, instantaneous selected side, sticky baseline state/count,
    processing time and diagnostic warnings. Warnings are not formal
    repetition classes.

    ``processing_time_ms`` covers image inspection, pose estimation, landmark
    and feature extraction, and baseline analysis. It excludes video decoding,
    CSV serialization, optional display rendering and setup. After that timer
    is finalised for a frame, its baseline CSV row is written and flushed in the
    same loop iteration. The summed per-frame values become measured processing
    seconds; loop wall time is the broader interval and may include decoding,
    incremental row logging, display and other loop/cleanup overhead. Source FPS
    is input-video metadata, not measured throughput; UTC lifecycle timestamps
    are recorded separately from performance timing.

    An ``ExitStack`` owns the video capture, pose estimator, logger and OpenCV
    windows. A broad catch records failed-run metadata after cleanup and then
    re-raises; successful completion records source FPS, frame count, resolution,
    termination reason and whether the complete clip was processed. Formal
    evaluation is a separate workflow over these recorded artefacts.
    """
    args = parse_arguments()
    create_project_directories()

    config = load_runtime_config(args.config)
    run_id = args.run_id or args.clip_id

    output_path = LOG_DIR / f"{run_id}_baseline.csv"
    metadata_path = (
        LOG_DIR / f"{run_id}_baseline_metadata.json"
    )
    output_paths = {
        "frame_csv": output_path,
        "metadata_json": metadata_path,
    }

    ensure_output_paths_available(
        output_paths.values(),
        overwrite=args.overwrite,
    )
    base_metadata = create_run_metadata(
        run_id=run_id,
        clip_id=args.clip_id,
        method="baseline",
        split=args.split,
        video_path=args.video,
        config_path=args.config,
        resolved_config=config.to_dict(),
        explicit_config_overrides={},
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

    capture = None
    processed_frames = 0
    measured_processing_seconds = 0.0
    loop_started = None
    termination_reason = "end_of_stream"

    try:
        with ExitStack() as cleanup:
            cleanup.callback(cv2.destroyAllWindows)

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

            analyser = BaselinePushUpAnalyser(
                top_elbow_angle=(
                    config.baseline.top_elbow_angle
                ),
                bottom_elbow_angle=(
                    config.baseline.bottom_elbow_angle
                ),
                top_extension_warning_threshold=(
                    config.baseline
                    .top_extension_warning_threshold
                ),
                depth_warning_threshold=(
                    config.baseline.depth_warning_threshold
                ),
                alignment_warning_minimum=(
                    config.baseline.alignment_warning_minimum
                ),
            )

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
                    "left_elbow_visibility_score",
                    "right_elbow_visibility_score",
                    "elbow_angle",
                    "body_alignment_angle",
                    "baseline_position",
                    "baseline_rep_count",
                    "baseline_frame_warnings",
                ],
            )
            cleanup.callback(logger.close)

            print(f"Video: {args.video}")
            print(f"Run ID: {run_id}")
            print(f"Split: {args.split}")
            print(f"Config: {args.config}")
            print(f"Frames: {capture.frame_count}")
            print(
                f"Source FPS: {capture.source_fps:.2f}"
            )
            print(f"Output: {output_path}")
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

                selected_side = "none"
                elbow_angle = None
                body_alignment_angle = None
                left_score = None
                right_score = None

                if pose_detected:
                    landmarks = extract_landmarks(
                        results,
                        image_width,
                        image_height,
                    )

                    left_score = feature_visibility_score(
                        landmarks,
                        side="left",
                        feature="elbow",
                    )

                    right_score = feature_visibility_score(
                        landmarks,
                        side="right",
                        feature="elbow",
                    )

                    selected_side = select_best_elbow_side(
                        landmarks,
                        minimum_visibility=(
                            config.features
                            .minimum_landmark_visibility
                        ),
                    )

                    if selected_side != "none":
                        shoulder = get_point(
                            landmarks,
                            f"{selected_side}_shoulder",
                        )
                        elbow = get_point(
                            landmarks,
                            f"{selected_side}_elbow",
                        )
                        wrist = get_point(
                            landmarks,
                            f"{selected_side}_wrist",
                        )

                        if feature_landmarks_available(
                            landmarks,
                            selected_side,
                            feature="elbow",
                            minimum_visibility=(
                                config.features
                                .minimum_landmark_visibility
                            ),
                        ):
                            elbow_angle = calculate_angle(
                                shoulder,
                                elbow,
                                wrist,
                            )

                        if feature_landmarks_available(
                            landmarks,
                            selected_side,
                            feature="alignment",
                            minimum_visibility=(
                                config.features
                                .minimum_landmark_visibility
                            ),
                        ):
                            hip = get_point(
                                landmarks,
                                f"{selected_side}_hip",
                            )
                            ankle = get_point(
                                landmarks,
                                f"{selected_side}_ankle",
                            )

                            body_alignment_angle = (
                                calculate_angle(
                                    shoulder,
                                    hip,
                                    ankle,
                                )
                            )

                baseline_result = analyser.update(
                    elbow_angle=elbow_angle,
                    body_alignment_angle=(
                        body_alignment_angle
                    ),
                )

                warnings = baseline_result["warnings"]
                warning_text = (
                    ", ".join(warnings)
                    if warnings
                    else "No frame warning"
                )

                processing_time_ms = (
                    time.perf_counter() - start_time
                ) * 1000.0
                processed_frames += 1
                measured_processing_seconds += (
                    processing_time_ms / 1000.0
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
                        "selected_side": selected_side,
                        "left_elbow_visibility_score": (
                            left_score
                        ),
                        "right_elbow_visibility_score": (
                            right_score
                        ),
                        "elbow_angle": elbow_angle,
                        "body_alignment_angle": (
                            body_alignment_angle
                        ),
                        "baseline_position": (
                            baseline_result["position"]
                        ),
                        "baseline_rep_count": (
                            baseline_result["rep_count"]
                        ),
                        "baseline_frame_warnings": (
                            warning_text
                        ),
                    }
                )

                if args.display:
                    pose_estimator.draw_landmarks(
                        frame,
                        results,
                    )

                    cv2.putText(
                        frame,
                        (
                            f"Frame: {capture.frame_index} "
                            f"Side: {selected_side} "
                            "Reps: "
                            f"{baseline_result['rep_count']}"
                        ),
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

                    cv2.imshow(
                        "Recorded Video Baseline",
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
                "Final baseline repetition count:",
                analyser.counter.rep_count,
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
