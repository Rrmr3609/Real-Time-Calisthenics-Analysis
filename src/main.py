import time
import cv2

from capture.webcam import WebcamCapture
from pose.estimator import PoseEstimator
from pose.landmarks import (
    extract_landmarks,
    get_point,
    get_visibility,
    side_landmarks_available,
)
from features.angles import calculate_angle
from utils.csv_logger import CSVLogger


def main():
    camera = WebcamCapture(device_index=0, width=1280, height=720)
    pose_estimator = PoseEstimator(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

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

            elbow_angle = None
            body_alignment_angle = None
            selected_side = "none"

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
                        "shoulder_visibility": get_visibility(landmarks, f"{selected_side}_shoulder"),
                        "elbow_visibility": get_visibility(landmarks, f"{selected_side}_elbow"),
                        "wrist_visibility": get_visibility(landmarks, f"{selected_side}_wrist"),
                        "hip_visibility": get_visibility(landmarks, f"{selected_side}_hip"),
                        "ankle_visibility": get_visibility(landmarks, f"{selected_side}_ankle"),
                    }

            logger.write_row(
                {
                    "timestamp": current_time,
                    "fps": fps,
                    "selected_side": selected_side,
                    "elbow_angle": elbow_angle,
                    "body_alignment_angle": body_alignment_angle,
                    **visibility_values,
                    "pose_detected": pose_detected,
                }
            )

            status_text = "Pose detected" if pose_detected else "No pose detected"

            cv2.putText(
                frame,
                f"{status_text} | FPS: {fps:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"Side: {selected_side}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"Elbow angle: {elbow_angle:.1f}" if elbow_angle is not None else "Elbow angle: N/A",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"Body angle: {body_alignment_angle:.1f}" if body_alignment_angle is not None else "Body angle: N/A",
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Real-Time Calisthenics Analysis - Feature Prototype", frame)

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