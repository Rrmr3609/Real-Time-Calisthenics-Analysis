from types import SimpleNamespace

import pytest

from capture import webcam


class FakeCapture:
    def __init__(self, *, opened=True, frames=(), events=None, name="capture"):
        self.opened = opened
        self.frames = list(frames)
        self.events = events if events is not None else []
        self.name = name
        self.set_calls = []
        self.read_calls = 0
        self.release_calls = 0

    def isOpened(self):
        return self.opened

    def set(self, property_id, value):
        self.set_calls.append((property_id, value))
        return True

    def read(self):
        self.read_calls += 1
        if not self.frames:
            return False, None
        return self.frames.pop(0)

    def release(self):
        self.release_calls += 1
        self.events.append(f"release:{self.name}")


def test_windows_prefers_directshow_and_buffers_verified_frame(monkeypatch):
    frame = SimpleNamespace(name="verified frame")
    capture = FakeCapture(frames=[(True, frame), (True, "later")])
    constructor_calls = []

    def create_capture(*args):
        constructor_calls.append(args)
        return capture

    monkeypatch.setattr(webcam.sys, "platform", "win32")
    monkeypatch.setattr(webcam.cv2, "VideoCapture", create_capture)

    camera = webcam.WebcamCapture(device_index=3, width=1280, height=720)
    camera.open()

    assert constructor_calls == [(3, webcam.cv2.CAP_DSHOW)]
    assert capture.set_calls == [
        (webcam.cv2.CAP_PROP_FRAME_WIDTH, 1280),
        (webcam.cv2.CAP_PROP_FRAME_HEIGHT, 720),
    ]
    assert capture.read_calls == 1
    assert camera.read() is frame
    assert capture.read_calls == 1
    assert camera.read() == "later"
    camera.release()
    assert capture.release_calls == 1


def test_non_windows_preserves_default_backend(monkeypatch):
    frame = SimpleNamespace(name="frame")
    capture = FakeCapture(frames=[(True, frame)])
    constructor_calls = []

    def create_capture(*args):
        constructor_calls.append(args)
        return capture

    monkeypatch.setattr(webcam.sys, "platform", "linux")
    monkeypatch.setattr(webcam.cv2, "VideoCapture", create_capture)

    camera = webcam.WebcamCapture(device_index=2)
    camera.open()

    assert constructor_calls == [(2,)]
    assert camera.read() is frame
    camera.release()


def test_failed_directshow_is_released_before_default_fallback(monkeypatch):
    events = []
    directshow = FakeCapture(
        frames=[(False, None)],
        events=events,
        name="directshow",
    )
    fallback_frame = SimpleNamespace(name="fallback frame")
    fallback = FakeCapture(
        frames=[(True, fallback_frame)],
        events=events,
        name="default",
    )

    def create_capture(*args):
        if len(args) == 2:
            events.append("create:directshow")
            return directshow
        events.append("create:default")
        return fallback

    monkeypatch.setattr(webcam.sys, "platform", "win32")
    monkeypatch.setattr(webcam.cv2, "VideoCapture", create_capture)

    camera = webcam.WebcamCapture(device_index=0)
    camera.open()

    assert events[:3] == [
        "create:directshow",
        "release:directshow",
        "create:default",
    ]
    assert directshow.release_calls == 1
    assert camera.read() is fallback_frame
    camera.release()
    assert fallback.release_calls == 1


def test_all_failed_capture_attempts_are_released(monkeypatch):
    directshow = FakeCapture(opened=False, name="directshow")
    fallback = FakeCapture(opened=False, name="default")
    captures = iter([directshow, fallback])

    monkeypatch.setattr(webcam.sys, "platform", "win32")
    monkeypatch.setattr(
        webcam.cv2,
        "VideoCapture",
        lambda *_args: next(captures),
    )

    camera = webcam.WebcamCapture(device_index=4)
    with pytest.raises(
        RuntimeError,
        match="Could not open camera with device index 4",
    ):
        camera.open()

    assert directshow.release_calls == 1
    assert fallback.release_calls == 1
    assert camera.cap is None


def test_unexpected_capture_error_is_released_and_propagated(monkeypatch):
    capture = FakeCapture(frames=[(True, object())])

    def fail_to_set(*_args):
        raise ValueError("unexpected property failure")

    capture.set = fail_to_set
    monkeypatch.setattr(webcam.sys, "platform", "linux")
    monkeypatch.setattr(webcam.cv2, "VideoCapture", lambda *_args: capture)

    camera = webcam.WebcamCapture()
    with pytest.raises(ValueError, match="unexpected property failure"):
        camera.open()

    assert capture.release_calls == 1
    assert camera.cap is None
