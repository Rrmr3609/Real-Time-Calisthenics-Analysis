import argparse
import time

import cv2

from analysis.enhanced_features import EnhancedFeatureProcessor
from capture.video import VideoFileCapture
from pose.estimator import PoseEstimator
from pose.landmarks import extract_landmarks
from utils.csv_logger import CSVLogger
from utils.paths import LOG_DIR, create_project_directories


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

    output_path = (
        LOG_DIR
        / f"{args.clip_id}_enhanced_features.csv"
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
        ],
    )

    try:
        capture.open()

        print(f"Video: {args.video}")
        print(f"Frames: {capture.frame_count}")
        print(f"Source FPS: {capture.source_fps:.2f}")
        print(f"Smoothing alpha: {args.alpha}")
        print(f"Output: {output_path}")

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

            processing_time_ms = (
                time.perf_counter() - start_time
            ) * 1000.0

            logger.write_row(
                {
                    "clip_id": args.clip_id,
                    "frame_index": capture.frame_index,
                    "video_timestamp_ms": capture.timestamp_ms(),
                    "source_fps": capture.source_fps,
                    "processing_time_ms": processing_time_ms,
                    "pose_detected": pose_detected,
                    **feature_result,
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

                cv2.imshow(
                    "Enhanced Preprocessing Development",
                    frame,
                )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        print("Enhanced preprocessing completed.")

    finally:
        logger.close()
        capture.release()
        pose_estimator.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()