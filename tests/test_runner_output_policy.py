import sys
from types import SimpleNamespace

import pytest

import main as live_runner
import run_video
import run_video_enhanced


class FailingReadCapture:
    def __init__(self, *_args, **_kwargs):
        self.frame_count = 1
        self.source_fps = 30.0
        self.released = False

    def open(self):
        return None

    def read(self):
        raise RuntimeError("processing failed")

    def release(self):
        self.released = True


class EmptyCapture(FailingReadCapture):
    def read(self):
        return None


class FakePoseEstimator:
    def __init__(self, *_args, **_kwargs):
        self.closed = False

    def close(self):
        self.closed = True


class FakeLogger:
    def __init__(self, *_args, **_kwargs):
        self.closed = False

    def close(self):
        self.closed = True


@pytest.mark.parametrize(
    ("runner", "arguments"),
    [
        (live_runner, ["main.py", "--overwrite"]),
        (
            run_video,
            [
                "run_video.py",
                "--video",
                "input.mp4",
                "--clip-id",
                "clip",
                "--overwrite",
            ],
        ),
        (
            run_video_enhanced,
            [
                "run_video_enhanced.py",
                "--video",
                "input.mp4",
                "--clip-id",
                "clip",
                "--overwrite",
            ],
        ),
    ],
)
def test_runner_parsers_accept_overwrite(
    runner,
    arguments,
    monkeypatch,
):
    monkeypatch.setattr(sys, "argv", arguments)

    assert runner.parse_arguments().overwrite


def test_enhanced_runner_preflights_both_outputs_before_setup(
    tmp_path,
    monkeypatch,
):
    log_dir = tmp_path / "logs"
    output_dir = tmp_path / "outputs"
    log_dir.mkdir()
    output_dir.mkdir()

    repetition_path = output_dir / "clip_enhanced_repetitions.csv"
    repetition_path.write_text("stale", encoding="utf-8")

    capture_was_created = False

    def create_capture(_video_path):
        nonlocal capture_was_created
        capture_was_created = True
        return FailingReadCapture()

    monkeypatch.setattr(
        run_video_enhanced,
        "parse_arguments",
        lambda: SimpleNamespace(
            video="input.mp4",
            clip_id="clip",
            alpha=0.3,
            display=False,
            overwrite=False,
        ),
    )
    monkeypatch.setattr(run_video_enhanced, "LOG_DIR", log_dir)
    monkeypatch.setattr(run_video_enhanced, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(
        run_video_enhanced,
        "create_project_directories",
        lambda: None,
    )
    monkeypatch.setattr(
        run_video_enhanced,
        "VideoFileCapture",
        create_capture,
    )

    with pytest.raises(FileExistsError, match="--overwrite"):
        run_video_enhanced.main()

    frame_path = log_dir / "clip_enhanced_temporal.csv"
    assert not capture_was_created
    assert not frame_path.exists()
    assert repetition_path.read_text(encoding="utf-8") == "stale"


def test_enhanced_overwrite_replaces_both_output_files(
    tmp_path,
    monkeypatch,
):
    log_dir = tmp_path / "logs"
    output_dir = tmp_path / "outputs"
    log_dir.mkdir()
    output_dir.mkdir()

    frame_path = log_dir / "clip_enhanced_temporal.csv"
    repetition_path = output_dir / "clip_enhanced_repetitions.csv"
    frame_path.write_text("stale frame output", encoding="utf-8")
    repetition_path.write_text(
        "stale repetition output",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        run_video_enhanced,
        "parse_arguments",
        lambda: SimpleNamespace(
            video="input.mp4",
            clip_id="clip",
            alpha=0.3,
            display=False,
            overwrite=True,
        ),
    )
    monkeypatch.setattr(run_video_enhanced, "LOG_DIR", log_dir)
    monkeypatch.setattr(run_video_enhanced, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(
        run_video_enhanced,
        "create_project_directories",
        lambda: None,
    )
    monkeypatch.setattr(
        run_video_enhanced,
        "VideoFileCapture",
        EmptyCapture,
    )
    monkeypatch.setattr(
        run_video_enhanced,
        "PoseEstimator",
        FakePoseEstimator,
    )
    monkeypatch.setattr(
        run_video_enhanced.cv2,
        "destroyAllWindows",
        lambda: None,
    )

    run_video_enhanced.main()

    frame_lines = frame_path.read_text(encoding="utf-8").splitlines()
    repetition_lines = (
        repetition_path.read_text(encoding="utf-8").splitlines()
    )

    assert frame_lines[0].startswith("clip_id,frame_index,")
    assert repetition_lines[0].startswith("clip_id,rep_id,")
    assert "stale" not in frame_path.read_text(encoding="utf-8")
    assert "stale" not in repetition_path.read_text(encoding="utf-8")


def test_enhanced_runner_closes_both_loggers_after_processing_failure(
    tmp_path,
    monkeypatch,
):
    captures = []
    pose_estimators = []
    loggers = []
    windows_destroyed = []

    def create_capture(*args, **kwargs):
        capture = FailingReadCapture(*args, **kwargs)
        captures.append(capture)
        return capture

    def create_pose(*args, **kwargs):
        pose_estimator = FakePoseEstimator(*args, **kwargs)
        pose_estimators.append(pose_estimator)
        return pose_estimator

    def create_logger(*args, **kwargs):
        logger = FakeLogger(*args, **kwargs)
        loggers.append(logger)
        return logger

    monkeypatch.setattr(
        run_video_enhanced,
        "parse_arguments",
        lambda: SimpleNamespace(
            video="input.mp4",
            clip_id="clip",
            alpha=0.3,
            display=False,
            overwrite=False,
        ),
    )
    monkeypatch.setattr(run_video_enhanced, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(
        run_video_enhanced,
        "OUTPUT_DIR",
        tmp_path / "outputs",
    )
    monkeypatch.setattr(
        run_video_enhanced,
        "create_project_directories",
        lambda: None,
    )
    monkeypatch.setattr(
        run_video_enhanced,
        "VideoFileCapture",
        create_capture,
    )
    monkeypatch.setattr(
        run_video_enhanced,
        "PoseEstimator",
        create_pose,
    )
    monkeypatch.setattr(
        run_video_enhanced,
        "CSVLogger",
        create_logger,
    )
    monkeypatch.setattr(
        run_video_enhanced.cv2,
        "destroyAllWindows",
        lambda: windows_destroyed.append(True),
    )

    with pytest.raises(RuntimeError, match="processing failed"):
        run_video_enhanced.main()

    assert len(captures) == 1
    assert captures[0].released
    assert len(pose_estimators) == 1
    assert pose_estimators[0].closed
    assert len(loggers) == 2
    assert all(logger.closed for logger in loggers)
    assert windows_destroyed == [True]


def test_baseline_runner_releases_capture_when_pose_setup_fails(
    tmp_path,
    monkeypatch,
):
    captures = []
    windows_destroyed = []

    def create_capture(*args, **kwargs):
        capture = FailingReadCapture(*args, **kwargs)
        captures.append(capture)
        return capture

    def fail_pose_setup(*_args, **_kwargs):
        raise RuntimeError("pose setup failed")

    monkeypatch.setattr(
        run_video,
        "parse_arguments",
        lambda: SimpleNamespace(
            video="input.mp4",
            clip_id="clip",
            display=False,
            overwrite=False,
        ),
    )
    monkeypatch.setattr(run_video, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(
        run_video,
        "create_project_directories",
        lambda: None,
    )
    monkeypatch.setattr(run_video, "VideoFileCapture", create_capture)
    monkeypatch.setattr(run_video, "PoseEstimator", fail_pose_setup)
    monkeypatch.setattr(
        run_video.cv2,
        "destroyAllWindows",
        lambda: windows_destroyed.append(True),
    )

    with pytest.raises(RuntimeError, match="pose setup failed"):
        run_video.main()

    assert len(captures) == 1
    assert captures[0].released
    assert windows_destroyed == [True]


def test_live_runner_closes_resources_after_processing_failure(
    tmp_path,
    monkeypatch,
):
    captures = []
    pose_estimators = []
    loggers = []
    windows_destroyed = []

    def create_capture(*args, **kwargs):
        capture = FailingReadCapture(*args, **kwargs)
        captures.append(capture)
        return capture

    def create_pose(*args, **kwargs):
        pose_estimator = FakePoseEstimator(*args, **kwargs)
        pose_estimators.append(pose_estimator)
        return pose_estimator

    def create_logger(*args, **kwargs):
        logger = FakeLogger(*args, **kwargs)
        loggers.append(logger)
        return logger

    monkeypatch.setattr(
        live_runner,
        "parse_arguments",
        lambda: SimpleNamespace(overwrite=False),
    )
    monkeypatch.setattr(live_runner, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(
        live_runner,
        "create_project_directories",
        lambda: None,
    )
    monkeypatch.setattr(live_runner, "WebcamCapture", create_capture)
    monkeypatch.setattr(live_runner, "PoseEstimator", create_pose)
    monkeypatch.setattr(live_runner, "CSVLogger", create_logger)
    monkeypatch.setattr(
        live_runner.cv2,
        "destroyAllWindows",
        lambda: windows_destroyed.append(True),
    )

    live_runner.main()

    assert len(captures) == 1
    assert captures[0].released
    assert len(pose_estimators) == 1
    assert pose_estimators[0].closed
    assert len(loggers) == 1
    assert loggers[0].closed
    assert windows_destroyed == [True]
