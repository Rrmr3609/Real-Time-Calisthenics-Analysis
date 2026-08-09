import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

import evaluation.run_formal_evaluation as runner_module
from evaluation.formal_reporting import (
    formal_evaluation_output_paths,
)
from evaluation.run_formal_evaluation import (
    main,
    run_formal_evaluation,
)


MANIFEST_COLUMNS = (
    "clip_id",
    "split",
    "video_path",
    "participant_id",
    "camera_view",
    "source_fps",
    "frame_count",
    "width_px",
    "height_px",
    "recording_condition",
    "notes",
)

ANNOTATION_COLUMNS = (
    "clip_id",
    "ground_truth_attempt_id",
    "is_evaluable_attempt",
    "ambiguity_flag",
    "start_top_frame",
    "bottom_turnaround_frame",
    "completion_end_top_frame",
    "ground_truth_class",
    "insufficient_depth_flag",
    "incomplete_extension_flag",
    "alignment_deviation_flag",
    "source_video_visibility_status",
    "annotator_notes",
)

BASELINE_FRAME_COLUMNS = (
    "run_id",
    "clip_id",
    "frame_index",
    "video_timestamp_ms",
    "source_fps",
    "processing_time_ms",
    "pose_detected",
    "selected_side",
    "left_elbow_visibility_score",
    "right_elbow_visibility_score",
    "elbow_angle",
    "body_alignment_angle",
    "baseline_position",
    "baseline_rep_count",
    "baseline_frame_warnings",
)


@dataclass
class FictionalInputs:
    manifest_path: Path
    annotations_path: Path
    baseline_metadata_paths: list[Path]
    enhanced_metadata_paths: list[Path]
    output_directory: Path
    split: str


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_bytes(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_json_sha256(value):
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def expected_source_identity_path(path, repository_root):
    resolved_path = Path(path).resolve()

    try:
        return resolved_path.relative_to(
            Path(repository_root).resolve()
        ).as_posix()
    except ValueError:
        return resolved_path.name


def manifest_row(clip_id, split, fps):
    return {
        "clip_id": clip_id,
        "split": split,
        "video_path": f"data/raw/fictional/{clip_id}.mp4",
        "participant_id": f"P_{clip_id.upper()}",
        "camera_view": "side",
        "source_fps": fps,
        "frame_count": 100,
        "width_px": 640,
        "height_px": 480,
        "recording_condition": "fictional_controlled_indoor",
        "notes": "Fictional test data only.",
    }


def annotation_row(
    clip_id,
    ground_truth_class,
    *,
    completion_frame=15,
):
    return {
        "clip_id": clip_id,
        "ground_truth_attempt_id": "A001",
        "is_evaluable_attempt": True,
        "ambiguity_flag": False,
        "start_top_frame": 5,
        "bottom_turnaround_frame": 10,
        "completion_end_top_frame": completion_frame,
        "ground_truth_class": ground_truth_class,
        "insufficient_depth_flag": (
            ground_truth_class == "insufficient_depth"
        ),
        "incomplete_extension_flag": (
            ground_truth_class == "incomplete_extension"
        ),
        "alignment_deviation_flag": (
            ground_truth_class == "alignment_deviation"
        ),
        "source_video_visibility_status": "sufficient",
        "annotator_notes": "",
    }


def ambiguous_annotation_row(clip_id):
    row = annotation_row(
        clip_id,
        "unscorable",
        completion_frame=15,
    )
    row.update(
        {
            "ground_truth_attempt_id": "F001",
            "is_evaluable_attempt": False,
            "ambiguity_flag": True,
            "bottom_turnaround_frame": "",
            "source_video_visibility_status": (
                "partially_obscured"
            ),
            "annotator_notes": (
                "Fictional ambiguous fragment proving review."
            ),
        }
    )
    return row


def baseline_frame_row(
    *,
    run_id,
    clip_id,
    frame_index,
    source_fps,
    repetition_count,
):
    return {
        "run_id": run_id,
        "clip_id": clip_id,
        "frame_index": frame_index,
        "video_timestamp_ms": (
            frame_index / source_fps * 1000.0
        ),
        "source_fps": source_fps,
        "processing_time_ms": 1.0,
        "pose_detected": True,
        "selected_side": "left",
        "left_elbow_visibility_score": 0.9,
        "right_elbow_visibility_score": 0.8,
        "elbow_angle": 155.0,
        "body_alignment_angle": 170.0,
        "baseline_position": "top",
        "baseline_rep_count": repetition_count,
        "baseline_frame_warnings": "No frame warning",
    }


def recorded_run_metadata(
    *,
    metadata_path,
    run_id,
    clip_id,
    method,
    split,
    source_fps,
    outputs,
):
    return {
        "metadata_schema_version": 1,
        "status": "completed",
        "run_id": run_id,
        "clip_id": clip_id,
        "method": method,
        "split": split,
        "timestamps": {
            "started_utc": "2026-08-08T10:00:00+00:00",
            "completed_utc": "2026-08-08T10:01:00+00:00",
        },
        "processing_summary": {
            "processed_frames": 100,
            "termination_reason": "end_of_stream",
            "processed_full_clip": True,
        },
        "input_video": {
            "path": f"data/raw/fictional/{clip_id}.mp4",
            "sha256": "0" * 64,
            "size_bytes": 100,
            "source_fps": source_fps,
            "frame_count": 100,
            "resolution": {
                "width_px": 640,
                "height_px": 480,
            },
        },
        "configuration": {
            "resolved": {
                "method": method,
                "source_fps": source_fps,
                "fictional_setting": (
                    "baseline-value"
                    if method == "baseline"
                    else "enhanced-value"
                ),
            },
        },
        "git": {
            "commit": f"{method}-{clip_id}-commit",
            "branch": "codex/fictional",
            "dirty": method == "enhanced",
        },
        "outputs": {
            **{
                name: str(Path(path).resolve())
                for name, path in outputs.items()
            },
            "metadata_json": str(metadata_path.resolve()),
        },
    }


def create_run_files(root, clip_id, split, fps, predicted_class):
    run_root = root / clip_id
    baseline_run_id = f"{clip_id}-baseline-run"
    enhanced_run_id = f"{clip_id}-enhanced-run"
    baseline_csv = run_root / "baseline.csv"
    baseline_metadata = run_root / "baseline_metadata.json"
    enhanced_frame_csv = run_root / "enhanced_frames.csv"
    enhanced_repetition_csv = run_root / "enhanced_repetitions.csv"
    enhanced_metadata = run_root / "enhanced_metadata.json"

    write_csv(
        baseline_csv,
        BASELINE_FRAME_COLUMNS,
        tuple(
            baseline_frame_row(
                run_id=baseline_run_id,
                clip_id=clip_id,
                frame_index=frame_index,
                source_fps=fps,
                repetition_count=(
                    1 if frame_index >= 15 else 0
                ),
            )
            for frame_index in range(100)
        ),
    )
    write_csv(
        enhanced_frame_csv,
        ("run_id", "clip_id", "frame_index"),
        (
            {
                "run_id": enhanced_run_id,
                "clip_id": clip_id,
                "frame_index": 0,
            },
        ),
    )
    write_csv(
        enhanced_repetition_csv,
        (
            "run_id",
            "clip_id",
            "rep_id",
            "start_frame",
            "bottom_frame",
            "end_frame",
            "completion_timestamp_ms",
            "source_fps",
            "predicted_class",
        ),
        (
            {
                "run_id": enhanced_run_id,
                "clip_id": clip_id,
                "rep_id": 1,
                "start_frame": 5,
                "bottom_frame": 10,
                "end_frame": 15,
                "completion_timestamp_ms": 15 / fps * 1000.0,
                "source_fps": fps,
                "predicted_class": predicted_class,
            },
        ),
    )
    write_json(
        baseline_metadata,
        recorded_run_metadata(
            metadata_path=baseline_metadata,
            run_id=baseline_run_id,
            clip_id=clip_id,
            method="baseline",
            split=split,
            source_fps=fps,
            outputs={"frame_csv": baseline_csv},
        ),
    )
    write_json(
        enhanced_metadata,
        recorded_run_metadata(
            metadata_path=enhanced_metadata,
            run_id=enhanced_run_id,
            clip_id=clip_id,
            method="enhanced",
            split=split,
            source_fps=fps,
            outputs={
                "frame_csv": enhanced_frame_csv,
                "repetition_csv": enhanced_repetition_csv,
            },
        ),
    )
    return baseline_metadata, enhanced_metadata


def create_fictional_inputs(tmp_path, *, split="development"):
    manifest_path = tmp_path / "manifest.csv"
    annotations_path = tmp_path / "annotations.csv"
    clip_definitions = (
        ("fictional-clip-a", 10.0, "correct"),
        ("fictional-clip-b", 20.0, "insufficient_depth"),
    )
    write_csv(
        manifest_path,
        MANIFEST_COLUMNS,
        tuple(
            manifest_row(clip_id, split, fps)
            for clip_id, fps, _ in clip_definitions
        ),
    )
    write_csv(
        annotations_path,
        ANNOTATION_COLUMNS,
        tuple(
            annotation_row(clip_id, ground_truth_class)
            for clip_id, _, ground_truth_class in clip_definitions
        ),
    )
    baseline_metadata_paths = []
    enhanced_metadata_paths = []

    for clip_id, fps, predicted_class in clip_definitions:
        baseline_metadata, enhanced_metadata = create_run_files(
            tmp_path / "runs",
            clip_id,
            split,
            fps,
            predicted_class,
        )
        baseline_metadata_paths.append(baseline_metadata)
        enhanced_metadata_paths.append(enhanced_metadata)

    return FictionalInputs(
        manifest_path=manifest_path,
        annotations_path=annotations_path,
        baseline_metadata_paths=baseline_metadata_paths,
        enhanced_metadata_paths=enhanced_metadata_paths,
        output_directory=tmp_path / "reports",
        split=split,
    )


def run_inputs(inputs, *, run_id="fictional-evaluation", **overrides):
    arguments = {
        "manifest_path": inputs.manifest_path,
        "annotations_path": inputs.annotations_path,
        "baseline_metadata_paths": inputs.baseline_metadata_paths,
        "enhanced_metadata_paths": inputs.enhanced_metadata_paths,
        "split": inputs.split,
        "tolerance_seconds": 0.75,
        "output_directory": inputs.output_directory,
        "evaluation_run_id": run_id,
    }
    arguments.update(overrides)
    return run_formal_evaluation(**arguments)


def rewrite_metadata(path, update):
    document = read_json(path)
    update(document)
    write_json(path, document)


def clone_metadata(source, destination, update=None):
    document = read_json(source)
    document["outputs"]["metadata_json"] = str(
        destination.resolve()
    )

    if update is not None:
        update(document)

    write_json(destination, document)
    return destination


def cli_arguments(inputs, *, run_id="cli-evaluation"):
    return [
        "--manifest",
        str(inputs.manifest_path),
        "--annotations",
        str(inputs.annotations_path),
        "--baseline-metadata",
        *[str(path) for path in inputs.baseline_metadata_paths],
        "--enhanced-metadata",
        *[str(path) for path in inputs.enhanced_metadata_paths],
        "--split",
        inputs.split,
        "--tolerance-seconds",
        "0.75",
        "--output-directory",
        str(inputs.output_directory),
        "--run-id",
        run_id,
    ]


def test_successful_multi_clip_development_orchestration(tmp_path):
    inputs = create_fictional_inputs(tmp_path)
    output_paths = run_inputs(inputs)
    report = read_json(output_paths.report_json)

    assert report["ordered_clip_ids"] == [
        "fictional-clip-a",
        "fictional-clip-b",
    ]
    assert report["split"] == "development"
    assert report["event_tolerance_seconds"] == 0.75
    assert report["baseline_detection"]["total_matched_events"] == 2
    assert report["enhanced_detection"]["total_matched_events"] == 2


def test_existing_event_loaders_and_enhanced_classification_are_used(
    tmp_path,
):
    inputs = create_fictional_inputs(tmp_path)
    output_paths = run_inputs(inputs)
    report = read_json(output_paths.report_json)
    classification = report["enhanced_classification"]

    assert report["baseline_detection"]["total_predicted_repetitions"] == 2
    assert report["enhanced_detection"]["total_predicted_repetitions"] == 2
    assert classification["evaluated_matched_repetitions"] == 2
    assert classification["accuracy"] == 1.0


def test_complete_formal_report_output_set_is_produced(tmp_path):
    inputs = create_fictional_inputs(tmp_path)
    output_paths = run_inputs(inputs)

    assert len(output_paths.all_paths()) == 6
    assert all(path.is_file() for path in output_paths.all_paths())
    metadata = read_json(output_paths.metadata_json)
    assert metadata["status"] == "completed"
    assert metadata["event_tolerance_seconds"] == 0.75


def test_evaluation_metadata_binds_exact_source_runs(tmp_path):
    inputs = create_fictional_inputs(tmp_path)
    baseline_paths = list(reversed(inputs.baseline_metadata_paths))
    enhanced_paths = list(reversed(inputs.enhanced_metadata_paths))
    output_paths = run_inputs(
        inputs,
        baseline_metadata_paths=baseline_paths,
        enhanced_metadata_paths=enhanced_paths,
    )
    metadata = read_json(output_paths.metadata_json)
    source_runs = metadata["source_runs"]

    assert [
        record["clip_id"] for record in source_runs["baseline"]
    ] == ["fictional-clip-a", "fictional-clip-b"]
    assert [
        record["clip_id"] for record in source_runs["enhanced"]
    ] == ["fictional-clip-a", "fictional-clip-b"]

    for method, metadata_paths in (
        ("baseline", inputs.baseline_metadata_paths),
        ("enhanced", inputs.enhanced_metadata_paths),
    ):
        documents_by_clip = {
            read_json(path)["clip_id"]: (path, read_json(path))
            for path in metadata_paths
        }

        for record in source_runs[method]:
            metadata_path, document = documents_by_clip[
                record["clip_id"]
            ]
            output_name = (
                "frame_csv"
                if method == "baseline"
                else "repetition_csv"
            )
            consumed_path = Path(
                document["outputs"][output_name]
            )

            assert record["method"] == method
            assert record["source_run_id"] == document["run_id"]
            assert record["split"] == "development"
            assert record["source_metadata_file"] == {
                "path": expected_source_identity_path(
                    metadata_path,
                    runner_module.PROJECT_ROOT,
                ),
                "sha256": sha256_bytes(metadata_path),
            }
            assert record["consumed_output_csv"] == {
                "output_name": output_name,
                "path": expected_source_identity_path(
                    consumed_path,
                    runner_module.PROJECT_ROOT,
                ),
                "sha256": sha256_bytes(consumed_path),
            }
            assert record["source_input_video_sha256"] == (
                document["input_video"]["sha256"]
            )
            assert record["source_git"] == {
                "commit": document["git"]["commit"],
                "dirty": document["git"]["dirty"],
            }
            assert record[
                "resolved_configuration_sha256"
            ] == canonical_json_sha256(
                document["configuration"]["resolved"]
            )

    assert source_runs["baseline"][0][
        "resolved_configuration_sha256"
    ] != source_runs["enhanced"][0][
        "resolved_configuration_sha256"
    ]
    assert str(tmp_path.resolve()) not in json.dumps(source_runs)


def test_repository_source_paths_are_relative_and_posix(
    tmp_path,
    monkeypatch,
):
    repository_root = tmp_path / "fictional-repository"
    inputs = create_fictional_inputs(repository_root)
    monkeypatch.setattr(
        runner_module,
        "PROJECT_ROOT",
        repository_root,
    )

    output_paths = run_inputs(inputs)
    source_runs = read_json(output_paths.metadata_json)[
        "source_runs"
    ]

    for method_records in source_runs.values():
        for record in method_records:
            for file_record in (
                record["source_metadata_file"],
                record["consumed_output_csv"],
            ):
                assert file_record["path"].startswith("runs/")
                assert "\\" not in file_record["path"]
                assert not Path(file_record["path"]).is_absolute()


def test_external_source_paths_use_privacy_safe_identifiers(
    tmp_path,
    monkeypatch,
):
    repository_root = tmp_path / "fictional-repository"
    repository_root.mkdir()
    external_root = tmp_path / "fictional-external-sources"
    inputs = create_fictional_inputs(external_root)
    monkeypatch.setattr(
        runner_module,
        "PROJECT_ROOT",
        repository_root,
    )
    output_paths = run_inputs(
        inputs,
        output_directory=repository_root / "reports",
    )
    metadata = read_json(output_paths.metadata_json)
    source_runs = metadata["source_runs"]
    baseline_metadata_path = inputs.baseline_metadata_paths[0]
    baseline_metadata = read_json(baseline_metadata_path)
    baseline_csv = Path(baseline_metadata["outputs"]["frame_csv"])
    baseline_record = source_runs["baseline"][0]

    assert baseline_record["source_metadata_file"] == {
        "path": baseline_metadata_path.name,
        "sha256": sha256_bytes(baseline_metadata_path),
    }
    assert baseline_record["consumed_output_csv"] == {
        "output_name": "frame_csv",
        "path": baseline_csv.name,
        "sha256": sha256_bytes(baseline_csv),
    }

    metadata_text = json.dumps(metadata)
    assert str(external_root.resolve()) not in metadata_text
    assert not any(
        Path(file_record["path"]).is_absolute()
        for method_records in source_runs.values()
        for record in method_records
        for file_record in (
            record["source_metadata_file"],
            record["consumed_output_csv"],
        )
    )


def test_repeated_evaluation_preserves_source_provenance_and_metrics(
    tmp_path,
):
    inputs = create_fictional_inputs(tmp_path)
    first_paths = run_inputs(inputs, run_id="first-evaluation")
    second_paths = run_inputs(inputs, run_id="second-evaluation")

    assert read_json(first_paths.metadata_json)["source_runs"] == (
        read_json(second_paths.metadata_json)["source_runs"]
    )
    assert first_paths.report_json.read_bytes() == (
        second_paths.report_json.read_bytes()
    )
    assert "source_runs" not in read_json(first_paths.report_json)


@pytest.mark.parametrize(
    "removed_source",
    ["metadata", "csv"],
)
def test_source_disappearing_before_provenance_capture_fails(
    tmp_path,
    monkeypatch,
    removed_source,
):
    inputs = create_fictional_inputs(tmp_path)
    baseline_metadata_path = inputs.baseline_metadata_paths[0]
    baseline_metadata = read_json(baseline_metadata_path)
    source_path = (
        baseline_metadata_path
        if removed_source == "metadata"
        else Path(baseline_metadata["outputs"]["frame_csv"])
    )
    original_aggregate = runner_module.aggregate_formal_evaluation

    def aggregate_then_remove_source(*args, **kwargs):
        report = original_aggregate(*args, **kwargs)
        source_path.unlink()
        return report

    monkeypatch.setattr(
        runner_module,
        "aggregate_formal_evaluation",
        aggregate_then_remove_source,
    )

    with pytest.raises(FileNotFoundError, match="disappeared"):
        run_inputs(inputs, run_id="missing-source-evaluation")

    output_paths = formal_evaluation_output_paths(
        inputs.output_directory,
        "missing-source-evaluation",
    )
    assert not any(path.exists() for path in output_paths.all_paths())


def test_consumed_csv_mutation_changes_recorded_hash(tmp_path):
    inputs = create_fictional_inputs(tmp_path)
    first_paths = run_inputs(inputs, run_id="first-evaluation")
    baseline_metadata = read_json(inputs.baseline_metadata_paths[0])
    baseline_csv = Path(baseline_metadata["outputs"]["frame_csv"])
    baseline_csv.write_text(
        baseline_csv.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    second_paths = run_inputs(inputs, run_id="second-evaluation")
    first_hash = read_json(first_paths.metadata_json)["source_runs"][
        "baseline"
    ][0]["consumed_output_csv"]["sha256"]
    second_hash = read_json(second_paths.metadata_json)["source_runs"][
        "baseline"
    ][0]["consumed_output_csv"]["sha256"]

    assert first_hash != second_hash
    assert second_hash == sha256_bytes(baseline_csv)


def test_source_mutation_before_provenance_capture_fails(
    tmp_path,
    monkeypatch,
):
    inputs = create_fictional_inputs(tmp_path)
    baseline_metadata = read_json(inputs.baseline_metadata_paths[0])
    baseline_csv = Path(baseline_metadata["outputs"]["frame_csv"])
    original_aggregate = runner_module.aggregate_formal_evaluation

    def aggregate_then_mutate_source(*args, **kwargs):
        report = original_aggregate(*args, **kwargs)
        baseline_csv.write_text(
            baseline_csv.read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )
        return report

    monkeypatch.setattr(
        runner_module,
        "aggregate_formal_evaluation",
        aggregate_then_mutate_source,
    )

    with pytest.raises(RuntimeError, match="changed after validation"):
        run_inputs(inputs, run_id="mutated-source-evaluation")

    output_paths = formal_evaluation_output_paths(
        inputs.output_directory,
        "mutated-source-evaluation",
    )
    assert not any(path.exists() for path in output_paths.all_paths())


def test_absent_optional_source_provenance_is_not_invented(tmp_path):
    inputs = create_fictional_inputs(tmp_path)
    metadata_path = inputs.baseline_metadata_paths[0]

    def remove_optional_provenance(document):
        document.pop("git")
        document.pop("configuration")

    rewrite_metadata(metadata_path, remove_optional_provenance)
    output_paths = run_inputs(inputs)
    baseline_record = read_json(output_paths.metadata_json)[
        "source_runs"
    ]["baseline"][0]

    assert baseline_record["source_git"] == {
        "commit": None,
        "dirty": None,
    }
    assert baseline_record["resolved_configuration_sha256"] is None


def test_valid_baseline_zero_detection_file_is_accepted(tmp_path):
    inputs = create_fictional_inputs(tmp_path)
    metadata_path = inputs.baseline_metadata_paths[0]
    metadata = read_json(metadata_path)
    alternate_csv = metadata_path.parent / "alternate_baseline.csv"
    write_csv(
        alternate_csv,
        BASELINE_FRAME_COLUMNS,
        tuple(
            baseline_frame_row(
                run_id=metadata["run_id"],
                clip_id=metadata["clip_id"],
                frame_index=frame_index,
                source_fps=metadata["input_video"]["source_fps"],
                repetition_count=0,
            )
            for frame_index in range(100)
        ),
    )
    metadata["outputs"]["frame_csv"] = str(alternate_csv.resolve())
    write_json(metadata_path, metadata)
    output_paths = run_inputs(inputs)
    report = read_json(output_paths.report_json)
    per_clip = {
        row["clip_id"]: row
        for row in report["per_clip_metrics"]
    }

    assert per_clip["fictional-clip-a"]["baseline_predicted_count"] == 0


@pytest.mark.parametrize(
    ("identity_field", "wrong_value", "message"),
    [
        (
            "clip_id",
            "wrong-fictional-clip",
            "clip IDs.*do not match completed-run metadata",
        ),
        (
            "run_id",
            "wrong-fictional-run",
            "run IDs.*do not match completed-run metadata",
        ),
    ],
)
def test_zero_detection_baseline_file_rejects_wrong_identity(
    tmp_path,
    identity_field,
    wrong_value,
    message,
):
    inputs = create_fictional_inputs(tmp_path)
    metadata = read_json(inputs.baseline_metadata_paths[0])
    baseline_csv = Path(metadata["outputs"]["frame_csv"])

    rows = []

    for frame_index in range(100):
        row = baseline_frame_row(
            run_id=metadata["run_id"],
            clip_id=metadata["clip_id"],
            frame_index=frame_index,
            source_fps=metadata["input_video"]["source_fps"],
            repetition_count=0,
        )
        if frame_index == 50:
            row[identity_field] = wrong_value
        rows.append(row)

    write_csv(
        baseline_csv,
        BASELINE_FRAME_COLUMNS,
        rows,
    )

    with pytest.raises(ValueError, match=message):
        run_inputs(inputs)


def test_baseline_frame_row_count_must_match_run_metadata(tmp_path):
    inputs = create_fictional_inputs(tmp_path)
    metadata = read_json(inputs.baseline_metadata_paths[0])
    baseline_csv = Path(metadata["outputs"]["frame_csv"])
    write_csv(
        baseline_csv,
        BASELINE_FRAME_COLUMNS,
        tuple(
            baseline_frame_row(
                run_id=metadata["run_id"],
                clip_id=metadata["clip_id"],
                frame_index=frame_index,
                source_fps=metadata["input_video"]["source_fps"],
                repetition_count=0,
            )
            for frame_index in range(99)
        ),
    )

    with pytest.raises(
        ValueError,
        match="contains 99 frame rows.*records 100 frames",
    ):
        run_inputs(inputs)


def test_baseline_frame_csv_requires_current_runner_header(tmp_path):
    inputs = create_fictional_inputs(tmp_path)
    metadata = read_json(inputs.baseline_metadata_paths[0])
    baseline_csv = Path(metadata["outputs"]["frame_csv"])
    incomplete_header = (
        "run_id",
        "clip_id",
        "frame_index",
        "baseline_rep_count",
    )
    write_csv(
        baseline_csv,
        incomplete_header,
        (
            {
                "run_id": metadata["run_id"],
                "clip_id": metadata["clip_id"],
                "frame_index": frame_index,
                "baseline_rep_count": 0,
            }
            for frame_index in range(100)
        ),
    )

    with pytest.raises(
        ValueError,
        match="missing required columns.*processing_time_ms",
    ):
        run_inputs(inputs)


def test_duplicate_baseline_clip_ids_are_rejected(tmp_path):
    inputs = create_fictional_inputs(tmp_path)
    duplicate_path = clone_metadata(
        inputs.baseline_metadata_paths[0],
        tmp_path / "duplicate_baseline_metadata.json",
    )

    with pytest.raises(ValueError, match="Duplicate baseline clip ID"):
        run_inputs(
            inputs,
            baseline_metadata_paths=[
                *inputs.baseline_metadata_paths,
                duplicate_path,
            ],
        )


def test_duplicate_enhanced_clip_ids_are_rejected(tmp_path):
    inputs = create_fictional_inputs(tmp_path)
    duplicate_path = clone_metadata(
        inputs.enhanced_metadata_paths[0],
        tmp_path / "duplicate_enhanced_metadata.json",
    )

    with pytest.raises(ValueError, match="Duplicate enhanced clip ID"):
        run_inputs(
            inputs,
            enhanced_metadata_paths=[
                *inputs.enhanced_metadata_paths,
                duplicate_path,
            ],
        )


def test_missing_baseline_clip_is_rejected(tmp_path):
    inputs = create_fictional_inputs(tmp_path)

    with pytest.raises(
        ValueError,
        match=(
            "Baseline metadata must cover the complete.*"
            "missing=.*fictional-clip-b"
        ),
    ):
        run_inputs(
            inputs,
            baseline_metadata_paths=inputs.baseline_metadata_paths[:1],
        )


def test_missing_enhanced_clip_is_rejected(tmp_path):
    inputs = create_fictional_inputs(tmp_path)

    with pytest.raises(
        ValueError,
        match=(
            "Enhanced metadata must cover the complete.*"
            "missing=.*fictional-clip-b"
        ),
    ):
        run_inputs(
            inputs,
            enhanced_metadata_paths=inputs.enhanced_metadata_paths[:1],
        )


def test_manifest_split_clip_omitted_from_both_methods_is_rejected(
    tmp_path,
):
    inputs = create_fictional_inputs(tmp_path)

    with pytest.raises(
        ValueError,
        match="complete.*missing=.*fictional-clip-b",
    ):
        run_inputs(
            inputs,
            baseline_metadata_paths=inputs.baseline_metadata_paths[:1],
            enhanced_metadata_paths=inputs.enhanced_metadata_paths[:1],
        )


def test_wrong_metadata_method_is_rejected(tmp_path):
    inputs = create_fictional_inputs(tmp_path)
    rewrite_metadata(
        inputs.baseline_metadata_paths[0],
        lambda document: document.update(method="enhanced"),
    )

    with pytest.raises(ValueError, match="expected 'baseline'"):
        run_inputs(inputs)


def test_wrong_metadata_split_is_rejected(tmp_path):
    inputs = create_fictional_inputs(tmp_path)
    rewrite_metadata(
        inputs.baseline_metadata_paths[0],
        lambda document: document.update(split="test"),
    )

    with pytest.raises(ValueError, match="selected split"):
        run_inputs(inputs)


@pytest.mark.parametrize("status", ["initialised", "failed"])
def test_incomplete_or_failed_run_metadata_is_rejected(tmp_path, status):
    inputs = create_fictional_inputs(tmp_path)
    rewrite_metadata(
        inputs.baseline_metadata_paths[0],
        lambda document: document.update(status=status),
    )

    with pytest.raises(ValueError, match="not a completed"):
        run_inputs(inputs)


def test_completed_metadata_for_partial_processing_is_rejected(tmp_path):
    inputs = create_fictional_inputs(tmp_path)
    rewrite_metadata(
        inputs.baseline_metadata_paths[0],
        lambda document: document["processing_summary"].update(
            processed_full_clip=False
        ),
    )

    with pytest.raises(ValueError, match="complete recorded clip"):
        run_inputs(inputs)


def test_missing_baseline_output_is_rejected(tmp_path):
    inputs = create_fictional_inputs(tmp_path)
    rewrite_metadata(
        inputs.baseline_metadata_paths[0],
        lambda document: document["outputs"].update(
            frame_csv=str((tmp_path / "missing-baseline.csv").resolve())
        ),
    )

    with pytest.raises(FileNotFoundError, match="missing output file"):
        run_inputs(inputs)


def test_missing_enhanced_repetition_output_is_rejected(tmp_path):
    inputs = create_fictional_inputs(tmp_path)
    rewrite_metadata(
        inputs.enhanced_metadata_paths[0],
        lambda document: document["outputs"].update(
            repetition_csv=str(
                (tmp_path / "missing-enhanced.csv").resolve()
            )
        ),
    )

    with pytest.raises(FileNotFoundError, match="missing output file"):
        run_inputs(inputs)


def test_missing_required_metadata_output_path_is_rejected(tmp_path):
    inputs = create_fictional_inputs(tmp_path)

    def remove_repetition_path(document):
        del document["outputs"]["repetition_csv"]

    rewrite_metadata(
        inputs.enhanced_metadata_paths[0],
        remove_repetition_path,
    )

    with pytest.raises(ValueError, match="missing required output paths"):
        run_inputs(inputs)


def test_supplied_clip_absent_from_manifest_is_rejected_as_extra(
    tmp_path,
):
    inputs = create_fictional_inputs(tmp_path)
    with inputs.manifest_path.open(encoding="utf-8") as input_file:
        manifest_rows = list(csv.DictReader(input_file))

    with inputs.annotations_path.open(encoding="utf-8") as input_file:
        annotation_rows = list(csv.DictReader(input_file))

    write_csv(
        inputs.manifest_path,
        MANIFEST_COLUMNS,
        manifest_rows[:1],
    )
    write_csv(
        inputs.annotations_path,
        ANNOTATION_COLUMNS,
        annotation_rows[:1],
    )

    with pytest.raises(
        ValueError,
        match="extra=.*fictional-clip-b",
    ):
        run_inputs(inputs)


def test_unknown_annotation_clip_is_rejected_by_existing_validation(
    tmp_path,
):
    inputs = create_fictional_inputs(tmp_path)
    with inputs.annotations_path.open(encoding="utf-8") as input_file:
        annotation_rows = list(csv.DictReader(input_file))

    unknown_row = dict(annotation_rows[0])
    unknown_row["clip_id"] = "unknown-fictional-clip"
    write_csv(
        inputs.annotations_path,
        ANNOTATION_COLUMNS,
        (*annotation_rows, unknown_row),
    )

    with pytest.raises(ValueError, match="unknown clip IDs"):
        run_inputs(inputs)


def test_selected_clip_without_annotation_rows_is_rejected(tmp_path):
    inputs = create_fictional_inputs(tmp_path)
    with inputs.annotations_path.open(encoding="utf-8") as input_file:
        annotation_rows = list(csv.DictReader(input_file))

    write_csv(
        inputs.annotations_path,
        ANNOTATION_COLUMNS,
        annotation_rows[:1],
    )

    with pytest.raises(
        ValueError,
        match="at least one annotation row.*fictional-clip-b",
    ):
        run_inputs(inputs)


def test_ambiguous_row_counts_as_annotation_review_evidence(tmp_path):
    inputs = create_fictional_inputs(tmp_path)
    write_csv(
        inputs.annotations_path,
        ANNOTATION_COLUMNS,
        (
            annotation_row("fictional-clip-a", "correct"),
            ambiguous_annotation_row("fictional-clip-b"),
        ),
    )

    output_paths = run_inputs(inputs)
    report = read_json(output_paths.report_json)
    per_clip = {
        row["clip_id"]: row
        for row in report["per_clip_metrics"]
    }

    assert per_clip["fictional-clip-b"][
        "ground_truth_repetition_count"
    ] == 0
    assert per_clip["fictional-clip-b"]["baseline_extras"] == 1
    assert per_clip["fictional-clip-b"]["enhanced_extras"] == 1


def test_positive_non_default_tolerance_passes_through(tmp_path):
    inputs = create_fictional_inputs(tmp_path)
    output_paths = run_inputs(inputs, tolerance_seconds=0.25)
    report = read_json(output_paths.report_json)
    metadata = read_json(output_paths.metadata_json)

    assert report["event_tolerance_seconds"] == 0.25
    assert metadata["event_tolerance_seconds"] == 0.25


def test_provisional_default_tolerance_is_available_for_convenience(
    tmp_path,
):
    inputs = create_fictional_inputs(tmp_path)
    output_paths = run_formal_evaluation(
        manifest_path=inputs.manifest_path,
        annotations_path=inputs.annotations_path,
        baseline_metadata_paths=inputs.baseline_metadata_paths,
        enhanced_metadata_paths=inputs.enhanced_metadata_paths,
        split=inputs.split,
        output_directory=inputs.output_directory,
        evaluation_run_id="default-tolerance-evaluation",
    )

    assert read_json(output_paths.report_json)[
        "event_tolerance_seconds"
    ] == 0.5


@pytest.mark.parametrize(
    "tolerance",
    [0.0, -0.1, float("nan"), float("inf")],
)
def test_invalid_tolerance_is_rejected(tmp_path, tolerance):
    inputs = create_fictional_inputs(tmp_path)

    with pytest.raises(ValueError, match="positive finite"):
        run_inputs(inputs, tolerance_seconds=tolerance)


def test_test_split_is_rejected_without_explicit_allow_flag(tmp_path):
    inputs = create_fictional_inputs(tmp_path, split="test")

    with pytest.raises(ValueError, match="disabled by default"):
        run_inputs(inputs)


def test_test_split_is_accepted_with_explicit_allow_flag(tmp_path):
    inputs = create_fictional_inputs(tmp_path, split="test")
    output_paths = run_inputs(inputs, allow_final_test=True)
    report = read_json(output_paths.report_json)

    assert report["split"] == "test"


def test_default_overwrite_protection(tmp_path):
    inputs = create_fictional_inputs(tmp_path)
    run_inputs(inputs)

    with pytest.raises(FileExistsError, match="already exist"):
        run_inputs(inputs)


def test_explicit_overwrite_replaces_complete_output_set(tmp_path):
    inputs = create_fictional_inputs(tmp_path)
    first_paths = run_inputs(inputs)
    first_paths.report_json.write_text("stale", encoding="utf-8")
    second_paths = run_inputs(inputs, overwrite=True)

    assert read_json(second_paths.report_json)["split"] == "development"
    assert all(path.is_file() for path in second_paths.all_paths())


def test_output_paths_and_metric_content_are_deterministic(tmp_path):
    inputs = create_fictional_inputs(tmp_path)
    first_paths = run_inputs(inputs, run_id="first-evaluation")
    second_paths = run_inputs(inputs, run_id="second-evaluation")

    assert first_paths == formal_evaluation_output_paths(
        inputs.output_directory,
        "first-evaluation",
    )
    assert read_json(first_paths.report_json) == read_json(
        second_paths.report_json
    )


def test_inconsistent_source_fps_is_rejected(tmp_path):
    inputs = create_fictional_inputs(tmp_path)
    rewrite_metadata(
        inputs.enhanced_metadata_paths[0],
        lambda document: document["input_video"].update(
            source_fps=11.0
        ),
    )

    with pytest.raises(ValueError, match="inconsistent source FPS"):
        run_inputs(inputs)


@pytest.mark.parametrize(
    ("metadata_list_name", "method"),
    [
        ("baseline_metadata_paths", "baseline"),
        ("enhanced_metadata_paths", "enhanced"),
    ],
)
def test_manifest_frame_count_must_match_each_run(
    tmp_path,
    metadata_list_name,
    method,
):
    inputs = create_fictional_inputs(tmp_path)
    metadata_path = getattr(inputs, metadata_list_name)[0]

    def change_frame_count(document):
        document["input_video"]["frame_count"] = 99
        document["processing_summary"]["processed_frames"] = 99

    rewrite_metadata(metadata_path, change_frame_count)

    with pytest.raises(
        ValueError,
        match=f"{method} frame count 99 does not match manifest 100",
    ):
        run_inputs(inputs)


@pytest.mark.parametrize(
    ("manifest_field", "dimension"),
    [
        ("width_px", "width"),
        ("height_px", "height"),
    ],
)
def test_manifest_resolution_must_match_run_metadata(
    tmp_path,
    manifest_field,
    dimension,
):
    inputs = create_fictional_inputs(tmp_path)
    with inputs.manifest_path.open(encoding="utf-8") as input_file:
        manifest_rows = list(csv.DictReader(input_file))

    manifest_rows[0][manifest_field] = "641"
    write_csv(
        inputs.manifest_path,
        MANIFEST_COLUMNS,
        manifest_rows,
    )

    with pytest.raises(
        ValueError,
        match=f"baseline {dimension} .*does not match manifest 641",
    ):
        run_inputs(inputs)


def test_mismatched_input_hashes_are_rejected(tmp_path):
    inputs = create_fictional_inputs(tmp_path)
    rewrite_metadata(
        inputs.enhanced_metadata_paths[0],
        lambda document: document["input_video"].update(
            sha256="1" * 64
        ),
    )

    with pytest.raises(ValueError, match="SHA-256 hashes do not match"):
        run_inputs(inputs)


def test_cli_success_return_code_and_summary(tmp_path, capsys):
    inputs = create_fictional_inputs(tmp_path)

    assert main(cli_arguments(inputs)) == 0
    captured = capsys.readouterr()
    assert "Formal evaluation completed." in captured.out
    assert "formal_evaluation_json:" in captured.out
    assert captured.err == ""


def test_cli_validation_failure_return_code(tmp_path, capsys):
    inputs = create_fictional_inputs(tmp_path)
    arguments = cli_arguments(inputs)
    tolerance_index = arguments.index("--tolerance-seconds") + 1
    arguments[tolerance_index] = "0"

    assert main(arguments) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "positive finite" in captured.err
