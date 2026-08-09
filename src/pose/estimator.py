"""Own and operate the MediaPipe Pose estimator used by the runners."""

import cv2
import mediapipe as mp


class PoseEstimator:
    """Process BGR frames with a stateful MediaPipe Pose instance.

    Detection and tracking confidence values are passed directly to MediaPipe.
    The estimator retains tracking state between frames and owns resources that
    callers must release with :meth:`close`.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame_bgr):
        """Run pose estimation after converting one OpenCV BGR frame to RGB."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.pose.process(frame_rgb)
        frame_rgb.flags.writeable = True
        return results

    def draw_landmarks(self, frame_bgr, results):
        """Draw detected pose landmarks onto ``frame_bgr`` in place.

        The same frame is returned unchanged when no pose is available.
        """
        if not results.pose_landmarks:
            return frame_bgr

        self.mp_drawing.draw_landmarks(
            frame_bgr,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=(
                self.mp_drawing_styles
                .get_default_pose_landmarks_style()
            ),
        )

        return frame_bgr

    def close(self):
        """Release resources owned by the underlying MediaPipe estimator."""
        self.pose.close()
