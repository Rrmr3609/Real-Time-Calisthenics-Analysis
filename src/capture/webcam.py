"""Provide explicit lifecycle management for live webcam capture."""

import sys

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
        self._initial_frame = None

    def open(self) -> None:
        """Open, negotiate and verify the configured camera device.

        DirectShow is preferred on Windows because automatic backend selection
        can hang while negotiating camera properties. A failed DirectShow
        attempt is released before one bounded fallback to OpenCV's normal
        backend selection. Other platforms retain normal backend selection.
        """
        self.release()

        backends = [cv2.CAP_DSHOW, None] if sys.platform.startswith("win") else [None]
        for backend in backends:
            if self._open_with_backend(backend):
                return

        raise RuntimeError(
            f"Could not open camera with device index {self.device_index}"
        )

    def _open_with_backend(self, backend: int | None) -> bool:
        """Try one backend and retain it only after a successful frame read."""
        capture = (
            cv2.VideoCapture(self.device_index)
            if backend is None
            else cv2.VideoCapture(self.device_index, backend)
        )

        try:
            opened = capture.isOpened()
            success = False
            frame = None
            if opened:
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                success, frame = capture.read()
        except Exception:
            capture.release()
            raise

        if not opened or not success or frame is None:
            capture.release()
            return False

        self.cap = capture
        self._initial_frame = frame
        return True

    def read(self):
        """Return the next frame, or ``None`` when a camera read fails."""
        if self.cap is None:
            raise RuntimeError("Camera has not been opened.")

        if self._initial_frame is not None:
            frame = self._initial_frame
            self._initial_frame = None
            return frame

        success, frame = self.cap.read()

        if not success:
            return None

        return frame

    def release(self) -> None:
        """Release the underlying OpenCV capture if it was created."""
        capture = self.cap
        self.cap = None
        self._initial_frame = None
        if capture is not None:
            capture.release()
