import time
import cv2

from analysis.baseline import BaselinePushUpAnalyser
from capture.webcam import WebcamCapture
from features.angles import calculate_angle
from pose.estimator import PoseEstimator
from pose.landmarks import (
    extract_landmarks,
    get_point,
    get_visibility,
    side_landmarks_available,
)
from utils.csv_logger import CSVLogger


def draw_text(frame, text, position, scale=0.8):
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
    camera = WebcamCapture(device_index=0, width=1280, height=720)

    pose_estimator = PoseEstimator(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    baseline_analyser = BaselinePushUpAnalyser()

    logger = CSVLogger(
        output_path="../experiments/logs/live_feature_log.csv",
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
    )

    try:
        camera.open()
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

                if side_landmarks_available(landmarks, "left", minimum_visibility=0.5):
                    selected_side = "left"
                elif side_landmarks_available(landmarks, "right", minimum_visibility=0.5):
                    selected_side = "right"

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

            # IMPORTANT:
            # The baseline analyser must be updated AFTER elbow_angle and
            # body_alignment_angle have been calculated.
            baseline_result = baseline_analyser.update(
                elbow_angle=elbow_angle,
                body_alignment_angle=body_alignment_angle,
            )

            warnings = baseline_result["warnings"]
            warning_text = ", ".join(warnings) if warnings else "No warning"

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
                f"Position: {baseline_result['position']} | Reps: {baseline_result['rep_count']}",
                (20, 200),
            )

            draw_text(
                frame,
                f"Warning: {warning_text}",
                (20, 240),
            )

            cv2.imshow("Real-Time Calisthenics Analysis - Baseline Prototype", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except RuntimeError as error:
        print(f"Error: {error}")

    finally:
        logger.close()
        camera.release()
        pose_estimator.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()