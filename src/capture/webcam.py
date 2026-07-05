import cv2


class WebcamCapture:
    def __init__(self, device_index: int = 0, width: int = 1280, height: int = 720):
        self.device_index = device_index
        self.width = width
        self.height = height
        self.cap = None

    def open(self) -> None:
        self.cap = cv2.VideoCapture(self.device_index)

        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera with device index {self.device_index}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def read(self):
        if self.cap is None:
            raise RuntimeError("Camera has not been opened.")

        success, frame = self.cap.read()

        if not success:
            return None

        return frame

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()