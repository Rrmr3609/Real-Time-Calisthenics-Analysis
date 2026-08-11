"""Provide explicit lifecycle management for recorded-video capture."""

from pathlib import Path
from typing import Optional

import cv2


class VideoFileCapture:
    """Own an OpenCV video-file capture and its source metadata.

    Call :meth:`open` before reading and :meth:`release` when processing ends.
    FPS, frame count and pixel dimensions are populated from OpenCV metadata
    during opening. Successful reads receive zero-based integer frame indices.
    """

    def __init__(self, video_path: str):
        self.video_path = Path(video_path)
        self.cap = None

        self.frame_index = -1
        self.source_fps = 0.0
        self.frame_count = 0
        self.width_px = 0
        self.height_px = 0

    def open(self) -> None:
        """Open the source file and populate its available metadata.

        A missing path raises ``FileNotFoundError``; an unreadable source raises
        ``RuntimeError``.
        """
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file does not exist: {self.video_path}")

        self.cap = cv2.VideoCapture(str(self.video_path))

        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.video_path}")

        self.source_fps = float(self.cap.get(cv2.CAP_PROP_FPS))

        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.width_px = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        self.height_px = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read(self):
        """Return the next frame, or ``None`` at end-of-stream/read failure."""
        if self.cap is None:
            raise RuntimeError("Video has not been opened.")

        success, frame = self.cap.read()

        if not success:
            return None

        self.frame_index += 1
        return frame

    def timestamp_ms(self) -> Optional[float]:
        """Return OpenCV's current source position in milliseconds, if open."""
        if self.cap is None:
            return None

        return float(self.cap.get(cv2.CAP_PROP_POS_MSEC))

    def release(self) -> None:
        """Release the underlying OpenCV capture if it was created."""
        if self.cap is not None:
            self.cap.release()
