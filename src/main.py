"""Run the interactive webcam demonstration of the simple baseline.

The demo processes raw frame measurements, displays an annotated OpenCV window
until ``q`` or capture failure, and writes ``experiments/logs/live_feature.csv``.
Its warnings are frame-level diagnostics rather than formal repetition
classifications; enhanced recorded-video analysis and formal evaluation are
separate workflows.
"""

import argparse
import time
from contextlib import ExitStack

import cv2

from analysis.baseline import BaselinePushUpAnalyser
from capture.webcam import WebcamCapture
from utils.paths import LOG_DIR, create_project_directories
from features.angles import calculate_angle
from pose.estimator import PoseEstimator
from pose.landmarks import (
    extract_landmarks,
    get_point,
    get_visibility,
    select_best_elbow_side,
)
from utils.csv_logger import CSVLogger, ensure_output_paths_available


def parse_arguments():
    """Parse the live demo's explicit output-overwrite option."""
    parser = argparse.ArgumentParser(
        description="Run the live webcam baseline analyser."
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the existing live output CSV.",
    )

    return parser.parse_args()


def draw_text(frame, text, position, scale=0.8):
    """Draw one green status line onto the displayed OpenCV frame."""
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
    """Run live baseline capture while owning all operational resources.

    The camera, MediaPipe estimator, CSV logger and OpenCV windows are registered
    for cleanup even when setup or capture fails. The deliberately simple
    baseline uses raw angles and instantaneous side selection; it does not run
    enhanced smoothing, temporal segmentation or formal classification.
    """
    args = parse_arguments()
    create_project_directories()

    output_path = LOG_DIR / "live_feature.csv"
    ensure_output_paths_available(
        [output_path],
        overwrite=args.overwrite,
    )

    cleanup = ExitStack()
    cleanup.callback(cv2.destroyAllWindows)

    try:
        camera = WebcamCapture(
            device_index=0,
            width=1280,
            height=720,
        )
        cleanup.callback(camera.release)
        camera.open()

        pose_estimator = PoseEstimator(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        cleanup.callback(pose_estimator.close)

        baseline_analyser = BaselinePushUpAnalyser()

        logger = CSVLogger(
            output_path=str(output_path),
            fieldnames=[
                "timestamp",
                "fps",
                "selected_side",
                "elbow_angle",
                "body_alignment_angle",
                "shoulder_visibility",
                "elbow_visibility",
                "wrist_visibility",
                "hip_visibility",
                "ankle_visibility",
                "pose_detected",
                "baseline_position",
                "baseline_rep_count",
                "baseline_warnings",
            ],
            overwrite=args.overwrite,
        )
        cleanup.callback(logger.close)

        print("Camera opened successfully. Press 'q' to quit.")

        previous_time = time.time()

        while True:
            frame = camera.read()

            if frame is None:
                print("Warning: failed to read frame from camera.")
                break

            image_height, image_width = frame.shape[:2]
            results = pose_estimator.process(frame)

            current_time = time.time()
            fps = 1.0 / (current_time - previous_time)
            previous_time = current_time

            pose_detected = bool(results.pose_landmarks)

            selected_side = "none"
            elbow_angle = None
            body_alignment_angle = None

            visibility_values = {
                "shoulder_visibility": None,
                "elbow_visibility": None,
                "wrist_visibility": None,
                "hip_visibility": None,
                "ankle_visibility": None,
            }

            if pose_detected:
                frame = pose_estimator.draw_landmarks(frame, results)
                landmarks = extract_landmarks(results, image_width, image_height)

                selected_side = select_best_elbow_side(
                    landmarks,
                    minimum_visibility=0.5,
                )

                if selected_side != "none":
                    shoulder = get_point(landmarks, f"{selected_side}_shoulder")
                    elbow = get_point(landmarks, f"{selected_side}_elbow")
                    wrist = get_point(landmarks, f"{selected_side}_wrist")
                    hip = get_point(landmarks, f"{selected_side}_hip")
                    ankle = get_point(landmarks, f"{selected_side}_ankle")

                    elbow_angle = calculate_angle(shoulder, elbow, wrist)
                    body_alignment_angle = calculate_angle(shoulder, hip, ankle)

                    visibility_values = {
                        "shoulder_visibility": get_visibility(
                            landmarks, f"{selected_side}_shoulder"
                        ),
                        "elbow_visibility": get_visibility(
                            landmarks, f"{selected_side}_elbow"
                        ),
                        "wrist_visibility": get_visibility(
                            landmarks, f"{selected_side}_wrist"
                        ),
                        "hip_visibility": get_visibility(
                            landmarks, f"{selected_side}_hip"
                        ),
                        "ankle_visibility": get_visibility(
                            landmarks, f"{selected_side}_ankle"
                        ),
                    }

            # Keep baseline state aligned with the measurements logged here.
            baseline_result = baseline_analyser.update(
                elbow_angle=elbow_angle,
                body_alignment_angle=body_alignment_angle,
            )

            warnings = baseline_result["warnings"]
            warning_text = ", ".join(warnings) if warnings else "No frame warning"

            logger.write_row(
                {
                    "timestamp": current_time,
                    "fps": fps,
                    "selected_side": selected_side,
                    "elbow_angle": elbow_angle,
                    "body_alignment_angle": body_alignment_angle,
                    **visibility_values,
                    "pose_detected": pose_detected,
                    "baseline_position": baseline_result["position"],
                    "baseline_rep_count": baseline_result["rep_count"],
                    "baseline_warnings": warning_text,
                }
            )

            status_text = "Pose detected" if pose_detected else "No pose detected"

            draw_text(frame, f"{status_text} | FPS: {fps:.1f}", (20, 40), scale=1.0)
            draw_text(frame, f"Side: {selected_side}", (20, 80))

            if elbow_angle is not None:
                draw_text(frame, f"Elbow angle: {elbow_angle:.1f}", (20, 120))
            else:
                draw_text(frame, "Elbow angle: N/A", (20, 120))

            if body_alignment_angle is not None:
                draw_text(frame, f"Body angle: {body_alignment_angle:.1f}", (20, 160))
            else:
                draw_text(frame, "Body angle: N/A", (20, 160))

            draw_text(
                frame,
                (
                    f"Position: {baseline_result['position']} | "
                    f"Reps: {baseline_result['rep_count']}"
                ),
                (20, 200),
            )

            draw_text(
                frame,
                f"Baseline frame warning: {warning_text}",
                (20, 240),
            )
            # Frame warnings are diagnostics, not repetition classifications.
            cv2.imshow("Real-Time Calisthenics Analysis - Baseline Prototype", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except RuntimeError as error:
        print(f"Error: {error}")

    finally:
        cleanup.close()


if __name__ == "__main__":
    main()
