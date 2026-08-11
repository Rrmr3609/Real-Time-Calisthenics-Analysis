import json
import sys
from types import SimpleNamespace

import pytest

import main as live_runner
import run_video
import run_video_enhanced
from utils.paths import PROJECT_ROOT


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


def fake_run_metadata(**kwargs):
    return {
        "metadata_schema_version": 1,
        "status": "initialised",
        "timestamps": {"started_utc": "2026-07-30T10:00:00+00:00"},
        "input_video": {
            "path": str(kwargs["video_path"]),
            "sha256": "mocked",
            "size_bytes": 1,
            "source_fps": None,
            "frame_count": None,
            "resolution": {
                "width_px": None,
                "height_px": None,
            },
        },
    }


@pytest.fixture(autouse=True)
def mock_recorded_runner_provenance(monkeypatch):
    monkeypatch.setattr(
        run_video,
        "create_run_metadata",
        fake_run_metadata,
    )
    monkeypatch.setattr(
        run_video_enhanced,
        "create_run_metadata",
        fake_run_metadata,
    )


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
                "--split",
                "development",
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
                "--split",
                "development",
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
            split="development",
            run_id=None,
            config=PROJECT_ROOT / "configs" / "default.yaml",
            alpha=None,
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
            split="development",
            run_id=None,
            config=PROJECT_ROOT / "configs" / "default.yaml",
            alpha=None,
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
    repetition_lines = repetition_path.read_text(encoding="utf-8").splitlines()

    metadata_path = output_dir / "clip_enhanced_metadata.json"

    assert frame_lines[0].startswith("run_id,clip_id,frame_index,")
    assert repetition_lines[0].startswith("run_id,clip_id,rep_id,")
    assert "stale" not in frame_path.read_text(encoding="utf-8")
    assert "stale" not in repetition_path.read_text(encoding="utf-8")
    assert (
        json.loads(metadata_path.read_text(encoding="utf-8"))["status"] == "completed"
    )


def test_enhanced_timing_excludes_repetition_csv_write(
    tmp_path,
    monkeypatch,
):
    class FakeClock:
        def __init__(self):
            self.current = 0.0

        def perf_counter(self):
            return self.current

    class OneFrameCapture(FailingReadCapture):
        def __init__(self, *_args, **_kwargs):
            super().__init__()
            self.frame_index = -1
            self.width_px = 640
            self.height_px = 480

        def read(self):
            if self.frame_index >= 0:
                return None

            self.frame_index = 0
            return SimpleNamespace(shape=(480, 640, 3))

        def timestamp_ms(self):
            return 0.0

    clock = FakeClock()
    frame_rows = []
    repetition_rows = []
    completed_repetition = SimpleNamespace(
        rep_id=1,
        start_frame=10,
        bottom_frame=12,
        end_frame=14,
        duration_frames=5,
        start_top_angle=155.0,
        minimum_elbow_angle=90.0,
        end_top_angle=154.0,
    )
    classification = SimpleNamespace(
        top_extension_angle=154.0,
        minimum_alignment_angle=170.0,
        alignment_valid_frames=5,
        alignment_valid_ratio=1.0,
        alignment_deviation_frames=0,
        alignment_deviation_ratio=0.0,
        insufficient_depth_triggered=False,
        incomplete_extension_triggered=False,
        alignment_deviation_triggered=False,
        multiple_rules_triggered=False,
        triggered_rules=(),
        predicted_class="correct",
        classification_reason="No predefined deviation detected.",
    )
    feature_result = {
        "selected_side": "left",
        "selected_elbow_side": "left",
        "side_changed": False,
        "left_elbow_visibility_score": 0.9,
        "right_elbow_visibility_score": 0.8,
        "left_alignment_visibility_score": 0.9,
        "right_alignment_visibility_score": 0.8,
        "elbow_feature_valid": True,
        "alignment_feature_valid": True,
        "opposite_alignment_feature_valid": True,
        "raw_elbow_angle": 154.0,
        "smoothed_elbow_angle": 154.0,
        "raw_alignment_angle": 170.0,
        "smoothed_alignment_angle": 170.0,
    }
    phase_result = {
        "repetition_window_start_frame": 10,
        "completed_repetition": completed_repetition,
        "phase": "top",
        "phase_changed": True,
        "rep_count": 1,
        "missing_angle_frames": 0,
    }

    class TimedPoseEstimator(FakePoseEstimator):
        def process(self, _frame):
            clock.current += 1.0
            return SimpleNamespace(pose_landmarks=None)

    class CapturingLogger(FakeLogger):
        def __init__(self, output_path, **_kwargs):
            super().__init__()
            self.is_repetition_logger = "enhanced_repetitions" in output_path

        def write_row(self, row):
            if self.is_repetition_logger:
                clock.current += 100.0
                repetition_rows.append(dict(row))
            else:
                frame_rows.append(dict(row))

    feature_processor = SimpleNamespace(
        update=lambda _landmarks: feature_result,
    )
    phase_machine = SimpleNamespace(
        update=lambda **_kwargs: phase_result,
        rep_count=1,
    )
    repetition_classifier = SimpleNamespace(
        classify=lambda _repetition: classification,
    )
    repetition_aggregator = SimpleNamespace(
        update=lambda **_kwargs: completed_repetition,
    )

    monkeypatch.setattr(
        run_video_enhanced,
        "parse_arguments",
        lambda: SimpleNamespace(
            video="input.mp4",
            clip_id="clip",
            split="development",
            run_id=None,
            config=PROJECT_ROOT / "configs" / "default.yaml",
            alpha=None,
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
        OneFrameCapture,
    )
    monkeypatch.setattr(
        run_video_enhanced,
        "PoseEstimator",
        TimedPoseEstimator,
    )
    monkeypatch.setattr(
        run_video_enhanced,
        "CSVLogger",
        CapturingLogger,
    )
    monkeypatch.setattr(
        run_video_enhanced,
        "_build_analysis_components",
        lambda _config: (
            feature_processor,
            phase_machine,
            repetition_classifier,
        ),
    )
    monkeypatch.setattr(
        run_video_enhanced,
        "RepetitionFeatureAggregator",
        lambda: repetition_aggregator,
    )
    monkeypatch.setattr(
        run_video_enhanced.time,
        "perf_counter",
        clock.perf_counter,
    )
    monkeypatch.setattr(
        run_video_enhanced.cv2,
        "destroyAllWindows",
        lambda: None,
    )

    run_video_enhanced.main()

    assert len(frame_rows) == 1
    assert frame_rows[0]["processing_time_ms"] == pytest.approx(1000.0)
    assert frame_rows[0]["completed_rep_id"] == 1
    assert len(repetition_rows) == 1
    assert repetition_rows[0]["rep_id"] == 1
    assert repetition_rows[0]["start_frame"] == 10
    assert repetition_rows[0]["end_frame"] == 14
    assert repetition_rows[0]["predicted_class"] == "correct"

    metadata = json.loads(
        (tmp_path / "outputs" / "clip_enhanced_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["processing_summary"][
        "measured_processing_seconds"
    ] == pytest.approx(1.0)


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
            split="development",
            run_id=None,
            config=PROJECT_ROOT / "configs" / "default.yaml",
            alpha=None,
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
    failed_metadata = json.loads(
        (tmp_path / "outputs" / "clip_enhanced_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert failed_metadata["status"] == "failed"
    assert "completed_utc" not in failed_metadata["timestamps"]


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
            split="development",
            run_id=None,
            config=PROJECT_ROOT / "configs" / "default.yaml",
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
    failed_metadata = json.loads(
        (tmp_path / "logs" / "clip_baseline_metadata.json").read_text(encoding="utf-8")
    )
    assert failed_metadata["status"] == "failed"
    assert "completed_utc" not in failed_metadata["timestamps"]


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
