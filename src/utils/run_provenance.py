from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


PACKAGE_DISTRIBUTIONS = {
    "opencv": "opencv-python",
    "mediapipe": "mediapipe",
    "numpy": "numpy",
    "pandas": "pandas",
}


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(
    file_path: str | Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    path = Path(file_path)
    digest = hashlib.sha256()

    with path.open("rb") as input_file:
        while chunk := input_file.read(chunk_size):
            digest.update(chunk)

    return digest.hexdigest()


def sha256_canonical_json(value: Any) -> str:
    """Hash one JSON value using deterministic canonical serialization."""
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def collect_software_versions(
    version_reader: Callable[[str], str] = metadata.version,
) -> dict[str, Any]:
    packages = {}

    for output_name, distribution_name in (
        PACKAGE_DISTRIBUTIONS.items()
    ):
        try:
            packages[output_name] = version_reader(
                distribution_name
            )
        except metadata.PackageNotFoundError:
            packages[output_name] = None

    return {
        "python": platform.python_version(),
        "python_implementation": (
            platform.python_implementation()
        ),
        "packages": packages,
    }


def _metadata_path(
    path: str | Path,
    repository_root: Path,
) -> str:
    resolved_path = Path(path).resolve()

    try:
        return resolved_path.relative_to(
            repository_root.resolve()
        ).as_posix()
    except ValueError:
        return str(resolved_path)


def _run_git_command(
    repository_root: Path,
    arguments: Sequence[str],
    command_runner: Callable[..., subprocess.CompletedProcess[str]],
) -> str | None:
    safe_directory = repository_root.resolve().as_posix()
    command = [
        "git",
        "-c",
        f"safe.directory={safe_directory}",
        *arguments,
    ]

    try:
        result = command_runner(
            command,
            cwd=repository_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except (FileNotFoundError, OSError):
        return None

    if result.returncode != 0:
        return None

    return result.stdout.strip()


def collect_git_state(
    repository_root: str | Path,
    command_runner: Callable[
        ..., subprocess.CompletedProcess[str]
    ] = subprocess.run,
) -> dict[str, Any]:
    root = Path(repository_root)
    commit = _run_git_command(
        root,
        ["rev-parse", "HEAD"],
        command_runner,
    )
    branch = _run_git_command(
        root,
        ["branch", "--show-current"],
        command_runner,
    )
    status = _run_git_command(
        root,
        ["status", "--porcelain"],
        command_runner,
    )

    return {
        "commit": commit,
        "branch": branch or None,
        "dirty": None if status is None else bool(status),
    }


def create_run_metadata(
    *,
    run_id: str,
    clip_id: str,
    method: str,
    split: str,
    video_path: str | Path,
    config_path: str | Path,
    resolved_config: Mapping[str, Any],
    explicit_config_overrides: Mapping[str, Any],
    repository_root: str | Path,
    output_paths: Mapping[str, str | Path],
    processing_time_definition: str,
    display_enabled: bool,
    overwrite_requested: bool,
    software_versions: Mapping[str, Any] | None = None,
    git_state: Mapping[str, Any] | None = None,
    timestamp_factory: Callable[[], str] = utc_timestamp,
) -> dict[str, Any]:
    video = Path(video_path)
    config = Path(config_path)
    root = Path(repository_root)

    return {
        "metadata_schema_version": 1,
        "status": "initialised",
        "run_id": run_id,
        "clip_id": clip_id,
        "method": method,
        "split": split,
        "timestamps": {
            "started_utc": timestamp_factory(),
        },
        "input_video": {
            "path": _metadata_path(video, root),
            "sha256": sha256_file(video),
            "size_bytes": video.stat().st_size,
            "source_fps": None,
            "frame_count": None,
            "resolution": {
                "width_px": None,
                "height_px": None,
            },
        },
        "configuration": {
            "source_path": _metadata_path(config, root),
            "source_sha256": sha256_file(config),
            "resolved": dict(resolved_config),
            "explicit_cli_overrides": dict(
                explicit_config_overrides
            ),
        },
        "software": dict(
            software_versions
            if software_versions is not None
            else collect_software_versions()
        ),
        "git": dict(
            git_state
            if git_state is not None
            else collect_git_state(root)
        ),
        "runtime_options": {
            "display_enabled": display_enabled,
            "overwrite_requested": overwrite_requested,
        },
        "processing_time_definition": (
            processing_time_definition
        ),
        "outputs": {
            name: _metadata_path(path, root)
            for name, path in output_paths.items()
        },
    }


def _atomic_write_json(
    output_path: Path,
    document: Mapping[str, Any],
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary_path = output_path.with_name(
        f".{output_path.name}.{uuid.uuid4().hex}.tmp"
    )

    try:
        with temporary_path.open(
            "x",
            encoding="utf-8",
            newline="\n",
        ) as output_file:
            json.dump(
                document,
                output_file,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            output_file.write("\n")
            output_file.flush()
            os.fsync(output_file.fileno())

        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def write_json_atomically(
    output_path: str | Path,
    document: Mapping[str, Any],
) -> None:
    """Write one JSON document through an atomic same-directory replace."""
    _atomic_write_json(Path(output_path), document)


class RunMetadataRecorder:
    def __init__(
        self,
        output_path: str | Path,
        base_metadata: Mapping[str, Any],
        timestamp_factory: Callable[[], str] = utc_timestamp,
    ):
        self.output_path = Path(output_path)
        self._base_metadata = deepcopy(dict(base_metadata))
        self._timestamp_factory = timestamp_factory
        self._finalised = False

    def _finalise(
        self,
        status: str,
        updates: Mapping[str, Any],
    ) -> None:
        if self._finalised:
            raise RuntimeError(
                "Run metadata has already been finalised"
            )

        document = deepcopy(self._base_metadata)
        document.update(deepcopy(dict(updates)))
        document["status"] = status
        timestamps = dict(document.get("timestamps", {}))

        if status == "completed":
            timestamps["completed_utc"] = (
                self._timestamp_factory()
            )
        else:
            timestamps["failed_utc"] = (
                self._timestamp_factory()
            )

        document["timestamps"] = timestamps
        _atomic_write_json(self.output_path, document)
        self._finalised = True

    def mark_completed(
        self,
        *,
        source_video: Mapping[str, Any],
        processing_summary: Mapping[str, Any],
    ) -> None:
        self._finalise(
            "completed",
            {
                "input_video": dict(source_video),
                "processing_summary": dict(
                    processing_summary
                ),
            },
        )

    def mark_failed(
        self,
        error: BaseException,
        *,
        source_video: Mapping[str, Any] | None = None,
        processing_summary: Mapping[str, Any] | None = None,
    ) -> None:
        updates: dict[str, Any] = {
            "failure": {
                "error_type": type(error).__name__,
                "message": str(error),
            }
        }

        if source_video is not None:
            updates["input_video"] = dict(source_video)

        if processing_summary is not None:
            updates["processing_summary"] = dict(
                processing_summary
            )

        self._finalise("failed", updates)
