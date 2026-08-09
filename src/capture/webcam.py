"""Provide explicit lifecycle management for live webcam capture."""

import cv2


class WebcamCapture:
    """Own an OpenCV webcam capture with requested frame dimensions.

    Call :meth:`open` before reading and :meth:`release` when capture ends.
    Width and height are requests to the camera driver and are not guarantees
    of the delivered resolution.
    """

    def __init__(
        self,
        device_index: int = 0,
        width: int = 1280,
        height: int = 720,
    ):
        self.device_index = device_index
        self.width = width
        self.height = height
        self.cap = None

    def open(self) -> None:
        """Open the configured camera device or raise ``RuntimeError``."""
        self.cap = cv2.VideoCapture(self.device_index)

        if not self.cap.isOpened():
            raise RuntimeError(
                "Could not open camera with device index "
                f"{self.device_index}"
            )

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def read(self):
        """Return the next frame, or ``None`` when a camera read fails."""
        if self.cap is None:
            raise RuntimeError("Camera has not been opened.")

        success, frame = self.cap.read()

        if not success:
            return None

        return frame

    def release(self) -> None:
        """Release the underlying OpenCV capture if it was created."""
        if self.cap is not None:
            self.cap.release()
