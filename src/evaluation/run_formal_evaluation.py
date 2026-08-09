from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from evaluation.dataset_validation import (
    ALLOWED_SPLITS,
    load_and_validate_evaluation_data,
)
from evaluation.detection_evaluation import (
    DetectionSummary,
    evaluate_detection_for_clip,
)
from evaluation.event_matching import (
    DEFAULT_EVENT_TOLERANCE_SECONDS,
)
from evaluation.formal_evaluation import (
    EnhancedClipEvaluation,
    evaluate_enhanced_clip,
)
from evaluation.formal_reporting import (
    EvaluationClipContext,
    FormalEvaluationOutputPaths,
    SourceRunProvenance,
    aggregate_formal_evaluation,
    write_formal_evaluation_report,
)
from evaluation.repetition_events import (
    BaselineRepetitionEvent,
    EnhancedRepetitionEvent,
    extract_ground_truth_events,
    load_baseline_events,
    load_enhanced_events,
)
from utils.run_provenance import (
    sha256_canonical_json,
    sha256_file,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_FPS_RELATIVE_TOLERANCE = 1e-6
SOURCE_FPS_ABSOLUTE_TOLERANCE = 1e-6


@dataclass(frozen=True)
class _RecordedRun:
    metadata_path: Path
    run_id: str
    clip_id: str
    method: str
    split: str
    source_fps: float
    frame_count: int
    width_px: int
    height_px: int
    processed_frames: int
    input_sha256: str
    output_paths: Mapping[str, Path]
    metadata_sha256: str
    consumed_output_name: str
    consumed_output_sha256: str
    source_git_commit: str | None
    source_git_dirty: bool | None
    resolved_configuration_sha256: str | None


def _required_mapping(
    document: Mapping[str, Any],
    key: str,
    source_name: str,
) -> Mapping[str, Any]:
    value = document.get(key)

    if not isinstance(value, Mapping):
        raise ValueError(
            f"{source_name} requires an object at {key!r}"
        )

    return value


def _required_text(
    document: Mapping[str, Any],
    key: str,
    source_name: str,
) -> str:
    value = document.get(key)

    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{source_name} requires non-blank {key!r}"
        )

    return value.strip()


def _positive_finite_number(
    value: object,
    *,
    description: str,
) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{description} must be a positive finite number"
        ) from error

    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(
            f"{description} must be a positive finite number"
        )

    return number


def _positive_integer(
    value: object,
    *,
    description: str,
) -> int:
    if isinstance(value, bool):
        raise ValueError(
            f"{description} must be a positive integer"
        )

    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{description} must be a positive integer"
        ) from error

    if (
        not math.isfinite(number)
        or number <= 0.0
        or not number.is_integer()
    ):
        raise ValueError(
            f"{description} must be a positive integer"
        )

    return int(number)


def _source_file_sha256(
    file_path: Path,
    *,
    description: str,
) -> str:
    try:
        return sha256_file(file_path)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"{description} disappeared before formal report writing: "
            f"{file_path}"
        ) from error
    except OSError as error:
        raise OSError(
            f"Could not hash {description}: {file_path}"
        ) from error


def _optional_source_git(
    document: Mapping[str, Any],
) -> tuple[str | None, bool | None]:
    git = document.get("git")

    if not isinstance(git, Mapping):
        return None, None

    raw_commit = git.get("commit")
    commit = (
        raw_commit.strip()
        if isinstance(raw_commit, str) and raw_commit.strip()
        else None
    )
    raw_dirty = git.get("dirty")
    dirty = raw_dirty if isinstance(raw_dirty, bool) else None
    return commit, dirty


def _optional_resolved_configuration_sha256(
    document: Mapping[str, Any],
) -> str | None:
    configuration = document.get("configuration")

    if not isinstance(configuration, Mapping):
        return None

    resolved = configuration.get("resolved")

    if not isinstance(resolved, Mapping):
        return None

    return sha256_canonical_json(resolved)


def _resolve_metadata_output_path(raw_path: str) -> Path:
    path = Path(raw_path)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def _load_recorded_run(
    metadata_path: str | Path,
    *,
    expected_method: str,
    selected_split: str,
) -> _RecordedRun:
    path = Path(metadata_path).resolve()
    source_name = str(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Run metadata file does not exist: {path}"
        )

    metadata_sha256 = _source_file_sha256(
        path,
        description="Source-run metadata file",
    )

    try:
        with path.open(encoding="utf-8") as metadata_file:
            document = json.load(metadata_file)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Run metadata is not valid JSON: {path}"
        ) from error

    if not isinstance(document, Mapping):
        raise ValueError(
            f"Run metadata must contain one JSON object: {path}"
        )

    if _source_file_sha256(
        path,
        description="Source-run metadata file",
    ) != metadata_sha256:
        raise RuntimeError(
            f"Source-run metadata changed while being read: {path}"
        )

    if document.get("metadata_schema_version") != 1:
        raise ValueError(
            f"{source_name} has an unsupported metadata schema version"
        )

    status = _required_text(document, "status", source_name)

    if status != "completed":
        raise ValueError(
            f"{source_name} is not a completed recorded-video run; "
            f"status is {status!r}"
        )

    timestamps = _required_mapping(
        document,
        "timestamps",
        source_name,
    )
    _required_text(timestamps, "completed_utc", source_name)
    processing_summary = _required_mapping(
        document,
        "processing_summary",
        source_name,
    )

    if processing_summary.get("processed_full_clip") is not True:
        raise ValueError(
            f"{source_name} did not process the complete recorded clip"
        )

    processed_frames = _positive_integer(
        processing_summary.get("processed_frames"),
        description=f"{source_name} processed frame count",
    )

    method = _required_text(document, "method", source_name)

    if method != expected_method:
        raise ValueError(
            f"{source_name} describes method {method!r}; expected "
            f"{expected_method!r}"
        )

    split = _required_text(document, "split", source_name)

    if split != selected_split:
        raise ValueError(
            f"{source_name} belongs to split {split!r}; selected split "
            f"is {selected_split!r}"
        )

    run_id = _required_text(document, "run_id", source_name)
    clip_id = _required_text(document, "clip_id", source_name)
    input_video = _required_mapping(
        document,
        "input_video",
        source_name,
    )
    input_sha256 = _required_text(
        input_video,
        "sha256",
        source_name,
    ).lower()
    source_fps = _positive_finite_number(
        input_video.get("source_fps"),
        description=f"{source_name} source FPS",
    )
    frame_count = _positive_integer(
        input_video.get("frame_count"),
        description=f"{source_name} input frame count",
    )
    resolution = _required_mapping(
        input_video,
        "resolution",
        source_name,
    )
    width_px = _positive_integer(
        resolution.get("width_px"),
        description=f"{source_name} input width",
    )
    height_px = _positive_integer(
        resolution.get("height_px"),
        description=f"{source_name} input height",
    )

    if processed_frames != frame_count:
        raise ValueError(
            f"{source_name} processed frame count "
            f"{processed_frames} does not match input frame count "
            f"{frame_count}"
        )

    raw_outputs = _required_mapping(
        document,
        "outputs",
        source_name,
    )
    required_output_names = (
        ("frame_csv", "metadata_json")
        if expected_method == "baseline"
        else ("frame_csv", "repetition_csv", "metadata_json")
    )
    missing_output_names = [
        name
        for name in required_output_names
        if name not in raw_outputs
    ]

    if missing_output_names:
        raise ValueError(
            f"{source_name} is missing required output paths: "
            f"{missing_output_names}"
        )

    output_paths: dict[str, Path] = {}

    for name, raw_output_path in raw_outputs.items():
        if (
            not isinstance(name, str)
            or not isinstance(raw_output_path, str)
            or not raw_output_path.strip()
        ):
            raise ValueError(
                f"{source_name} contains an invalid output path entry"
            )

        resolved_path = _resolve_metadata_output_path(
            raw_output_path.strip()
        )

        if not resolved_path.is_file():
            raise FileNotFoundError(
                f"{source_name} references a missing output file "
                f"for {name!r}: {resolved_path}"
            )

        output_paths[name] = resolved_path

    if output_paths["metadata_json"] != path:
        raise ValueError(
            f"{source_name} does not identify itself through "
            "outputs['metadata_json']"
        )

    consumed_output_name = (
        "frame_csv"
        if expected_method == "baseline"
        else "repetition_csv"
    )
    consumed_output_sha256 = _source_file_sha256(
        output_paths[consumed_output_name],
        description=(
            f"{expected_method.title()} consumed output CSV"
        ),
    )
    source_git_commit, source_git_dirty = (
        _optional_source_git(document)
    )
    resolved_configuration_sha256 = (
        _optional_resolved_configuration_sha256(document)
    )

    return _RecordedRun(
        metadata_path=path,
        run_id=run_id,
        clip_id=clip_id,
        method=method,
        split=split,
        source_fps=source_fps,
        frame_count=frame_count,
        width_px=width_px,
        height_px=height_px,
        processed_frames=processed_frames,
        input_sha256=input_sha256,
        output_paths=output_paths,
        metadata_sha256=metadata_sha256,
        consumed_output_name=consumed_output_name,
        consumed_output_sha256=consumed_output_sha256,
        source_git_commit=source_git_commit,
        source_git_dirty=source_git_dirty,
        resolved_configuration_sha256=(
            resolved_configuration_sha256
        ),
    )


def _load_recorded_runs(
    metadata_paths: Sequence[str | Path],
    *,
    expected_method: str,
    selected_split: str,
) -> dict[str, _RecordedRun]:
    if not metadata_paths:
        raise ValueError(
            f"At least one {expected_method} metadata path is required"
        )

    runs_by_clip: dict[str, _RecordedRun] = {}

    for metadata_path in metadata_paths:
        run = _load_recorded_run(
            metadata_path,
            expected_method=expected_method,
            selected_split=selected_split,
        )

        if run.clip_id in runs_by_clip:
            raise ValueError(
                f"Duplicate {expected_method} clip ID "
                f"{run.clip_id!r} in supplied metadata"
            )

        runs_by_clip[run.clip_id] = run

    return runs_by_clip


def _verify_source_files_unchanged(run: _RecordedRun) -> None:
    current_metadata_sha256 = _source_file_sha256(
        run.metadata_path,
        description="Source-run metadata file",
    )

    if current_metadata_sha256 != run.metadata_sha256:
        raise RuntimeError(
            "Source-run metadata changed after validation: "
            f"{run.metadata_path}"
        )

    consumed_output_path = run.output_paths[
        run.consumed_output_name
    ]
    current_output_sha256 = _source_file_sha256(
        consumed_output_path,
        description=(
            f"{run.method.title()} consumed output CSV"
        ),
    )

    if current_output_sha256 != run.consumed_output_sha256:
        raise RuntimeError(
            f"{run.method.title()} consumed output CSV changed "
            f"after validation: {consumed_output_path}"
        )


def _source_identity_path(
    file_path: Path,
    repository_root: Path,
) -> str:
    resolved_path = file_path.resolve()

    try:
        return resolved_path.relative_to(
            repository_root.resolve()
        ).as_posix()
    except ValueError:
        return resolved_path.name


def _build_source_run_provenance(
    run: _RecordedRun,
    *,
    repository_root: Path,
) -> SourceRunProvenance:
    _verify_source_files_unchanged(run)
    consumed_output_path = run.output_paths[
        run.consumed_output_name
    ]
    return SourceRunProvenance(
        clip_id=run.clip_id,
        method=run.method,
        source_run_id=run.run_id,
        split=run.split,
        source_metadata_path=_source_identity_path(
            run.metadata_path,
            repository_root,
        ),
        source_metadata_sha256=run.metadata_sha256,
        consumed_output_name=run.consumed_output_name,
        consumed_output_path=_source_identity_path(
            consumed_output_path,
            repository_root,
        ),
        consumed_output_sha256=run.consumed_output_sha256,
        source_input_video_sha256=run.input_sha256,
        source_git_commit=run.source_git_commit,
        source_git_dirty=run.source_git_dirty,
        resolved_configuration_sha256=(
            run.resolved_configuration_sha256
        ),
    )


def _source_fps_matches(first: float, second: float) -> bool:
    return math.isclose(
        first,
        second,
        rel_tol=SOURCE_FPS_RELATIVE_TOLERANCE,
        abs_tol=SOURCE_FPS_ABSOLUTE_TOLERANCE,
    )


def _validate_complete_split_coverage(
    supplied_clip_ids: set[str],
    selected_manifest_clip_ids: set[str],
    *,
    method: str,
    split: str,
) -> None:
    missing_clip_ids = sorted(
        selected_manifest_clip_ids - supplied_clip_ids
    )
    extra_clip_ids = sorted(
        supplied_clip_ids - selected_manifest_clip_ids
    )

    if missing_clip_ids or extra_clip_ids:
        raise ValueError(
            f"{method.title()} metadata must cover the complete "
            f"manifest split {split!r}; missing={missing_clip_ids}, "
            f"extra={extra_clip_ids}"
        )


def _validate_annotation_presence(
    annotations,
    selected_manifest_clip_ids: set[str],
    *,
    split: str,
) -> None:
    annotated_clip_ids = set(
        annotations["clip_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    missing_clip_ids = sorted(
        selected_manifest_clip_ids - annotated_clip_ids
    )

    if missing_clip_ids:
        raise ValueError(
            f"Every clip in formal manifest split {split!r} must "
            "have at least one annotation row; missing="
            f"{missing_clip_ids}"
        )


def _validate_manifest_run_metadata(
    *,
    clip_id: str,
    manifest_fps: float,
    manifest_frame_count: int,
    manifest_width_px: int,
    manifest_height_px: int,
    baseline_run: _RecordedRun,
    enhanced_run: _RecordedRun,
) -> None:
    if not (
        _source_fps_matches(manifest_fps, baseline_run.source_fps)
        and _source_fps_matches(
            manifest_fps,
            enhanced_run.source_fps,
        )
        and _source_fps_matches(
            baseline_run.source_fps,
            enhanced_run.source_fps,
        )
    ):
        raise ValueError(
            f"Clip {clip_id!r} has inconsistent source FPS: "
            f"manifest={manifest_fps}, "
            f"baseline={baseline_run.source_fps}, "
            f"enhanced={enhanced_run.source_fps}"
        )

    for run in (baseline_run, enhanced_run):
        if run.frame_count != manifest_frame_count:
            raise ValueError(
                f"Clip {clip_id!r} {run.method} frame count "
                f"{run.frame_count} does not match manifest "
                f"{manifest_frame_count}"
            )

        if run.width_px != manifest_width_px:
            raise ValueError(
                f"Clip {clip_id!r} {run.method} width "
                f"{run.width_px} does not match manifest "
                f"{manifest_width_px}"
            )

        if run.height_px != manifest_height_px:
            raise ValueError(
                f"Clip {clip_id!r} {run.method} height "
                f"{run.height_px} does not match manifest "
                f"{manifest_height_px}"
            )


def _validate_loaded_events(
    events: Sequence[
        BaselineRepetitionEvent | EnhancedRepetitionEvent
    ],
    run: _RecordedRun,
) -> None:
    inconsistent_events = [
        event
        for event in events
        if event.clip_id != run.clip_id
        or event.run_id != run.run_id
        or event.method != run.method
    ]

    if inconsistent_events:
        raise ValueError(
            f"Output referenced by {run.metadata_path} contains events "
            "whose run, clip or method identity does not match metadata"
        )


def run_formal_evaluation(
    *,
    manifest_path: str | Path,
    annotations_path: str | Path,
    baseline_metadata_paths: Sequence[str | Path],
    enhanced_metadata_paths: Sequence[str | Path],
    split: str,
    output_directory: str | Path,
    evaluation_run_id: str,
    tolerance_seconds: float = (
        DEFAULT_EVENT_TOLERANCE_SECONDS
    ),
    overwrite: bool = False,
    allow_final_test: bool = False,
) -> FormalEvaluationOutputPaths:
    """Evaluate existing recorded-run outputs and write one report set."""
    selected_split = str(split).strip()

    if selected_split not in ALLOWED_SPLITS:
        raise ValueError(
            f"Evaluation split must be one of {sorted(ALLOWED_SPLITS)}"
        )

    tolerance = _positive_finite_number(
        tolerance_seconds,
        description="Event tolerance",
    )

    if selected_split == "test" and not allow_final_test:
        raise ValueError(
            "Test-split evaluation is disabled by default. Freeze all "
            "development decisions, including event tolerance, before "
            "setting allow_final_test=True."
        )

    manifest, annotations = load_and_validate_evaluation_data(
        Path(manifest_path),
        Path(annotations_path),
    )
    baseline_runs = _load_recorded_runs(
        baseline_metadata_paths,
        expected_method="baseline",
        selected_split=selected_split,
    )
    enhanced_runs = _load_recorded_runs(
        enhanced_metadata_paths,
        expected_method="enhanced",
        selected_split=selected_split,
    )
    baseline_clip_ids = set(baseline_runs)
    enhanced_clip_ids = set(enhanced_runs)

    manifest_clip_ids = (
        manifest["clip_id"].astype(str).str.strip()
    )
    manifest_splits = manifest["split"].astype(str).str.strip()
    selected_manifest_clip_ids = set(
        manifest_clip_ids.loc[
            manifest_splits.eq(selected_split)
        ]
    )
    _validate_complete_split_coverage(
        baseline_clip_ids,
        selected_manifest_clip_ids,
        method="baseline",
        split=selected_split,
    )
    _validate_complete_split_coverage(
        enhanced_clip_ids,
        selected_manifest_clip_ids,
        method="enhanced",
        split=selected_split,
    )
    _validate_annotation_presence(
        annotations,
        selected_manifest_clip_ids,
        split=selected_split,
    )

    source_fps_by_clip = {
        clip_id: float(source_fps)
        for clip_id, source_fps in zip(
            manifest_clip_ids,
            manifest["source_fps"],
        )
    }
    frame_count_by_clip = {
        clip_id: int(frame_count)
        for clip_id, frame_count in zip(
            manifest_clip_ids,
            manifest["frame_count"],
        )
    }
    width_by_clip = {
        clip_id: int(width_px)
        for clip_id, width_px in zip(
            manifest_clip_ids,
            manifest["width_px"],
        )
    }
    height_by_clip = {
        clip_id: int(height_px)
        for clip_id, height_px in zip(
            manifest_clip_ids,
            manifest["height_px"],
        )
    }

    for clip_id in sorted(baseline_clip_ids):
        manifest_fps = source_fps_by_clip[clip_id]
        _validate_manifest_run_metadata(
            clip_id=clip_id,
            manifest_fps=manifest_fps,
            manifest_frame_count=frame_count_by_clip[clip_id],
            manifest_width_px=width_by_clip[clip_id],
            manifest_height_px=height_by_clip[clip_id],
            baseline_run=baseline_runs[clip_id],
            enhanced_run=enhanced_runs[clip_id],
        )

        if (
            baseline_runs[clip_id].input_sha256
            != enhanced_runs[clip_id].input_sha256
        ):
            raise ValueError(
                f"Clip {clip_id!r} baseline and enhanced input "
                "SHA-256 hashes do not match"
            )

    ground_truth_events = extract_ground_truth_events(
        annotations,
        manifest,
        source_name=str(Path(annotations_path)),
        manifest_source_name=str(Path(manifest_path)),
    )
    ground_truth_by_clip = {
        clip_id: tuple(
            event
            for event in ground_truth_events
            if event.clip_id == clip_id
        )
        for clip_id in baseline_clip_ids
    }
    baseline_results: list[DetectionSummary] = []
    enhanced_results: list[EnhancedClipEvaluation] = []
    clip_contexts: list[EvaluationClipContext] = []

    for clip_id in sorted(baseline_clip_ids):
        source_fps = source_fps_by_clip[clip_id]
        baseline_run = baseline_runs[clip_id]
        enhanced_run = enhanced_runs[clip_id]
        baseline_events = load_baseline_events(
            baseline_run.output_paths["frame_csv"],
            source_fps_by_clip={clip_id: source_fps},
            expected_run_id=baseline_run.run_id,
            expected_clip_id=baseline_run.clip_id,
            expected_frame_count=baseline_run.frame_count,
        )
        enhanced_events = load_enhanced_events(
            enhanced_run.output_paths["repetition_csv"],
            source_fps_by_clip={clip_id: source_fps},
        )
        _validate_loaded_events(baseline_events, baseline_run)
        _validate_loaded_events(enhanced_events, enhanced_run)
        _verify_source_files_unchanged(baseline_run)
        _verify_source_files_unchanged(enhanced_run)
        clip_ground_truth = ground_truth_by_clip[clip_id]
        _, baseline_summary = evaluate_detection_for_clip(
            baseline_events,
            clip_ground_truth,
            clip_id=clip_id,
            method="baseline",
            source_fps=source_fps,
            tolerance_seconds=tolerance,
        )
        enhanced_result = evaluate_enhanced_clip(
            enhanced_events,
            clip_ground_truth,
            clip_id=clip_id,
            source_fps=source_fps,
            tolerance_seconds=tolerance,
        )
        baseline_results.append(baseline_summary)
        enhanced_results.append(enhanced_result)
        clip_contexts.append(
            EvaluationClipContext(
                clip_id=clip_id,
                split=selected_split,
                source_fps=source_fps,
            )
        )

    report = aggregate_formal_evaluation(
        baseline_results=baseline_results,
        enhanced_results=enhanced_results,
        clip_contexts=clip_contexts,
        split=selected_split,
        tolerance_seconds=tolerance,
    )
    source_run_provenance = tuple(
        _build_source_run_provenance(
            baseline_runs[clip_id],
            repository_root=PROJECT_ROOT,
        )
        for clip_id in sorted(baseline_runs)
    ) + tuple(
        _build_source_run_provenance(
            enhanced_runs[clip_id],
            repository_root=PROJECT_ROOT,
        )
        for clip_id in sorted(enhanced_runs)
    )
    return write_formal_evaluation_report(
        report,
        output_directory=output_directory,
        run_id=evaluation_run_id,
        repository_root=PROJECT_ROOT,
        source_run_provenance=source_run_provenance,
        overwrite=overwrite,
    )


def parse_arguments(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run formal evaluation over explicit completed baseline and "
            "enhanced recorded-run metadata files."
        )
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Validated dataset manifest CSV.",
    )
    parser.add_argument(
        "--annotations",
        required=True,
        help="Validated manual repetition annotations CSV.",
    )
    parser.add_argument(
        "--baseline-metadata",
        nargs="+",
        required=True,
        help="Explicit baseline provenance metadata JSON paths.",
    )
    parser.add_argument(
        "--enhanced-metadata",
        nargs="+",
        required=True,
        help="Explicit enhanced provenance metadata JSON paths.",
    )
    parser.add_argument(
        "--split",
        choices=sorted(ALLOWED_SPLITS),
        required=True,
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--tolerance-seconds",
        type=float,
        default=DEFAULT_EVENT_TOLERANCE_SECONDS,
        help=(
            "Positive event-matching tolerance. The default 0.5 seconds "
            "is provisional."
        ),
    )
    parser.add_argument(
        "--output-directory",
        required=True,
        help="Directory for the complete formal report set.",
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="Safe evaluation run identifier used in output filenames.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the complete report output set for this run ID.",
    )
    parser.add_argument(
        "--allow-final-test",
        action="store_true",
        help=(
            "Allow test-split evaluation only after development "
            "decisions have been frozen."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_arguments(argv)

    try:
        output_paths = run_formal_evaluation(
            manifest_path=args.manifest,
            annotations_path=args.annotations,
            baseline_metadata_paths=args.baseline_metadata,
            enhanced_metadata_paths=args.enhanced_metadata,
            split=args.split,
            tolerance_seconds=args.tolerance_seconds,
            output_directory=args.output_directory,
            evaluation_run_id=args.run_id,
            overwrite=args.overwrite,
            allow_final_test=args.allow_final_test,
        )
    except (OSError, ValueError) as error:
        print(f"Formal evaluation failed: {error}", file=sys.stderr)
        return 2

    print("Formal evaluation completed.")

    for name, path in output_paths.named_paths().items():
        print(f"{name}: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
