import time
import cv2

from capture.webcam import WebcamCapture
from pose.estimator import PoseEstimator


def main():
    camera = WebcamCapture(device_index=0, width=1280, height=720)
    pose_estimator = PoseEstimator(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
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

            results = pose_estimator.process(frame)

            if results.pose_landmarks:
                frame = pose_estimator.draw_landmarks(frame, results)
                status_text = "Pose detected"
            else:
                status_text = "No pose detected"

            current_time = time.time()
            fps = 1.0 / (current_time - previous_time)
            previous_time = current_time

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

            cv2.imshow("Real-Time Calisthenics Analysis - Pose Prototype", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except RuntimeError as error:
        print(f"Error: {error}")

    finally:
        camera.release()
        pose_estimator.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()