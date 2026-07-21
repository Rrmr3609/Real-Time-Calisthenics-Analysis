from pathlib import Path
from typing import Optional

import cv2


class VideoFileCapture:
    def __init__(self, video_path: str):
        self.video_path = Path(video_path)
        self.cap = None

        self.frame_index = -1
        self.source_fps = 0.0
        self.frame_count = 0

    def open(self) -> None:
        if not self.video_path.exists():
            raise FileNotFoundError(
                f"Video file does not exist: {self.video_path}"
            )

        self.cap = cv2.VideoCapture(str(self.video_path))

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open video: {self.video_path}"
            )

        self.source_fps = float(
            self.cap.get(cv2.CAP_PROP_FPS)
        )

        self.frame_count = int(
            self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

    def read(self):
        if self.cap is None:
            raise RuntimeError("Video has not been opened.")

        success, frame = self.cap.read()

        if not success:
            return None

        self.frame_index += 1
        return frame

    def timestamp_ms(self) -> Optional[float]:
        if self.cap is None:
            return None

        return float(
            self.cap.get(cv2.CAP_PROP_POS_MSEC)
        )

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()