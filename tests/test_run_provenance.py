import hashlib
import json
import subprocess

import pytest

from utils.csv_logger import prepare_output_paths
from utils.run_provenance import (
    RunMetadataRecorder,
    collect_git_state,
    collect_software_versions,
    create_run_metadata,
    sha256_canonical_json,
    sha256_file,
)


def make_base_metadata():
    return {
        "metadata_schema_version": 1,
        "status": "initialised",
        "timestamps": {"started_utc": "2026-07-30T10:00:00+00:00"},
        "input_video": {
            "path": "input.mp4",
            "sha256": "abc",
            "size_bytes": 3,
        },
    }


def test_sha256_file_is_deterministic(tmp_path):
    input_path = tmp_path / "input.bin"
    input_path.write_bytes(b"deterministic content")
    expected = hashlib.sha256(b"deterministic content").hexdigest()

    assert sha256_file(input_path) == expected
    assert sha256_file(input_path) == expected


def test_canonical_json_sha256_is_deterministic_and_order_independent():
    first = {
        "features": {"ema_alpha": 0.4, "enabled": True},
        "threshold": 130,
    }
    reordered = {
        "threshold": 130,
        "features": {"enabled": True, "ema_alpha": 0.4},
    }
    expected_bytes = json.dumps(
        first,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")

    assert sha256_canonical_json(first) == hashlib.sha256(expected_bytes).hexdigest()
    assert sha256_canonical_json(first) == (sha256_canonical_json(reordered))
    assert sha256_canonical_json(first) != (
        sha256_canonical_json({**first, "threshold": 131})
    )


def test_software_versions_use_mocked_distribution_data():
    versions = {
        "opencv-python": "5.0.0",
        "mediapipe": "0.10.14",
        "numpy": "2.5.1",
        "pandas": "3.0.3",
    }

    result = collect_software_versions(version_reader=versions.__getitem__)

    assert result["packages"] == {
        "opencv": "5.0.0",
        "mediapipe": "0.10.14",
        "numpy": "2.5.1",
        "pandas": "3.0.3",
    }
    assert result["python"]
    assert "python_executable" not in result


def test_git_state_uses_mocked_git_results(tmp_path):
    calls = []

    def command_runner(
        command,
        cwd,
        capture_output,
        text,
        check,
    ):
        calls.append(command)

        if command[-2:] == ["rev-parse", "HEAD"]:
            output = "abc123\n"
        elif command[-2:] == [
            "branch",
            "--show-current",
        ]:
            output = "codex/test\n"
        else:
            output = " M src/file.py\n"

        return subprocess.CompletedProcess(
            command,
            0,
            stdout=output,
            stderr="",
        )

    state = collect_git_state(
        tmp_path,
        command_runner=command_runner,
    )

    assert state == {
        "commit": "abc123",
        "branch": "codex/test",
        "dirty": True,
    }
    assert all(
        command[1:3] == ["-c", f"safe.directory={tmp_path.as_posix()}"]
        for command in calls
    )


def test_create_run_metadata_contains_required_provenance(
    tmp_path,
):
    video_path = tmp_path / "clip.mp4"
    config_path = tmp_path / "config.yaml"
    output_path = tmp_path / "run.csv"
    video_path.write_bytes(b"video")
    config_path.write_text("value: 1\n", encoding="utf-8")

    metadata = create_run_metadata(
        run_id="run-01",
        clip_id="clip-01",
        method="enhanced",
        split="development",
        video_path=video_path,
        config_path=config_path,
        resolved_config={"value": 1},
        explicit_config_overrides={"features.ema_alpha": 0.4},
        repository_root=tmp_path,
        output_paths={"frame_csv": output_path},
        processing_time_definition="Measured boundary.",
        display_enabled=False,
        overwrite_requested=False,
        software_versions={
            "python": "3.12.4",
            "packages": {
                "opencv": "5.0.0",
                "mediapipe": "0.10.14",
                "numpy": "2.5.1",
                "pandas": "3.0.3",
            },
        },
        git_state={
            "commit": "abc123",
            "branch": "codex/test",
            "dirty": False,
        },
        timestamp_factory=lambda: "2026-07-30T10:00:00+00:00",
    )

    assert metadata["run_id"] == "run-01"
    assert metadata["clip_id"] == "clip-01"
    assert metadata["method"] == "enhanced"
    assert metadata["split"] == "development"
    assert metadata["input_video"]["size_bytes"] == 5
    assert metadata["input_video"]["path"] == "clip.mp4"
    assert metadata["input_video"]["sha256"] == (hashlib.sha256(b"video").hexdigest())
    assert metadata["configuration"]["source_sha256"] == (
        hashlib.sha256(config_path.read_bytes()).hexdigest()
    )
    assert metadata["configuration"]["explicit_cli_overrides"] == {
        "features.ema_alpha": 0.4
    }
    assert metadata["configuration"]["source_path"] == ("config.yaml")
    assert metadata["outputs"]["frame_csv"] == "run.csv"
    assert str(tmp_path) not in json.dumps(metadata)
    assert metadata["git"]["dirty"] is False
    assert metadata["status"] == "initialised"


def test_stale_metadata_blocks_complete_output_set(
    tmp_path,
):
    csv_path = tmp_path / "run.csv"
    metadata_path = tmp_path / "run_metadata.json"
    metadata_path.write_text("stale", encoding="utf-8")

    with pytest.raises(
        FileExistsError,
        match="run_metadata.json",
    ):
        prepare_output_paths(
            [csv_path, metadata_path],
            overwrite=False,
        )

    assert not csv_path.exists()
    assert metadata_path.read_text(encoding="utf-8") == "stale"


def test_explicit_overwrite_removes_complete_output_set(
    tmp_path,
):
    csv_path = tmp_path / "run.csv"
    metadata_path = tmp_path / "run_metadata.json"
    csv_path.write_text("old csv", encoding="utf-8")
    metadata_path.write_text("old json", encoding="utf-8")

    prepare_output_paths(
        [csv_path, metadata_path],
        overwrite=True,
    )

    assert not csv_path.exists()
    assert not metadata_path.exists()


def test_failed_run_metadata_is_not_marked_completed(
    tmp_path,
):
    output_path = tmp_path / "metadata.json"
    timestamps = iter(
        [
            "2026-07-30T10:01:00+00:00",
        ]
    )
    recorder = RunMetadataRecorder(
        output_path,
        make_base_metadata(),
        timestamp_factory=lambda: next(timestamps),
    )

    recorder.mark_failed(RuntimeError("processing failed"))

    document = json.loads(output_path.read_text(encoding="utf-8"))
    assert document["status"] == "failed"
    assert "completed_utc" not in document["timestamps"]
    assert document["timestamps"]["failed_utc"] == ("2026-07-30T10:01:00+00:00")
    assert document["failure"] == {
        "error_type": "RuntimeError",
        "message": "processing failed",
    }


def test_completed_run_metadata_serialises_atomically(
    tmp_path,
):
    output_path = tmp_path / "metadata.json"
    recorder = RunMetadataRecorder(
        output_path,
        make_base_metadata(),
        timestamp_factory=lambda: "2026-07-30T10:02:00+00:00",
    )

    recorder.mark_completed(
        source_video={
            "path": "input.mp4",
            "sha256": "abc",
            "size_bytes": 3,
            "source_fps": 30.0,
            "frame_count": 90,
            "resolution": {
                "width_px": 1280,
                "height_px": 720,
            },
        },
        processing_summary={
            "processed_frames": 90,
        },
    )

    document = json.loads(output_path.read_text(encoding="utf-8"))
    assert document["status"] == "completed"
    assert document["input_video"]["source_fps"] == 30.0
    assert document["processing_summary"]["processed_frames"] == 90
    assert not list(tmp_path.glob("*.tmp"))
