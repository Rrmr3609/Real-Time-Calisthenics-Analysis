import argparse
import time

import cv2

from analysis.baseline import BaselinePushUpAnalyser
from capture.video import VideoFileCapture
from features.angles import calculate_angle
from pose.estimator import PoseEstimator
from pose.landmarks import (
    extract_landmarks,
    feature_landmarks_available,
    feature_visibility_score,
    get_point,
    get_visibility,
    select_best_elbow_side,
)
from utils.csv_logger import CSVLogger
from utils.paths import LOG_DIR, create_project_directories


def parse_arguments():
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
        help="Identifier written into the output log.",
    )

    parser.add_argument(
        "--display",
        action="store_true",
        help="Display processed frames.",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    create_project_directories()

    capture = VideoFileCapture(args.video)
    pose_estimator = PoseEstimator(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    analyser = BaselinePushUpAnalyser()

    output_path = LOG_DIR / f"{args.clip_id}_baseline.csv"

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
            "left_elbow_visibility_score",
            "right_elbow_visibility_score",
            "elbow_angle",
            "body_alignment_angle",
            "baseline_position",
            "baseline_rep_count",
            "baseline_frame_warnings",
        ],
    )

    try:
        capture.open()

        print(f"Video: {args.video}")
        print(f"Frames: {capture.frame_count}")
        print(f"Source FPS: {capture.source_fps:.2f}")
        print(f"Output: {output_path}")

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
                    minimum_visibility=0.5,
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
                        minimum_visibility=0.5,
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
                        minimum_visibility=0.5,
                    ):
                        hip = get_point(
                            landmarks,
                            f"{selected_side}_hip",
                        )
                        ankle = get_point(
                            landmarks,
                            f"{selected_side}_ankle",
                        )

                        body_alignment_angle = calculate_angle(
                            shoulder,
                            hip,
                            ankle,
                        )

            baseline_result = analyser.update(
                elbow_angle=elbow_angle,
                body_alignment_angle=body_alignment_angle,
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

            logger.write_row(
                {
                    "clip_id": args.clip_id,
                    "frame_index": capture.frame_index,
                    "video_timestamp_ms": capture.timestamp_ms(),
                    "source_fps": capture.source_fps,
                    "processing_time_ms": processing_time_ms,
                    "pose_detected": pose_detected,
                    "selected_side": selected_side,
                    "left_elbow_visibility_score": left_score,
                    "right_elbow_visibility_score": right_score,
                    "elbow_angle": elbow_angle,
                    "body_alignment_angle": body_alignment_angle,
                    "baseline_position": baseline_result["position"],
                    "baseline_rep_count": baseline_result["rep_count"],
                    "baseline_frame_warnings": warning_text,
                }
            )

            if args.display:
                pose_estimator.draw_landmarks(frame, results)

                cv2.putText(
                    frame,
                    (
                        f"Frame: {capture.frame_index} "
                        f"Side: {selected_side} "
                        f"Reps: {baseline_result['rep_count']}"
                    ),
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow("Recorded Video Baseline", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        print(
            "Final baseline repetition count:",
            analyser.counter.rep_count,
        )

    finally:
        logger.close()
        capture.release()
        pose_estimator.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()