import ast
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from evaluation import annotate_repetitions
from evaluation.dataset_validation import (
    ANNOTATION_COLUMNS,
    MANIFEST_COLUMNS,
    validate_dataset_manifest,
    validate_repetition_annotations,
)


def make_manifest(*, second_clip=False):
    rows = [
        {
            "clip_id": "fictional_dev_01",
            "split": "development",
            "video_path": "data/raw/fictional/fictional_dev_01.mp4",
            "participant_id": "P_FICTIONAL_01",
            "camera_view": "side",
            "source_fps": 30.0,
            "frame_count": 500,
            "width_px": 1280,
            "height_px": 720,
            "recording_condition": "fictional_controlled_condition",
            "notes": "Fictional test row.",
        }
    ]
    if second_clip:
        rows.append(
            {
                **rows[0],
                "clip_id": "fictional_dev_02",
                "video_path": "data/raw/fictional/fictional_dev_02.mp4",
                "participant_id": "P_FICTIONAL_02",
            }
        )
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def make_draft(**overrides):
    values = {
        "start_top_frame": 10,
        "bottom_turnaround_frame": 20,
        "completion_end_top_frame": 30,
    }
    values.update(overrides)
    return annotate_repetitions.AnnotationDraft(**values)


def make_row(*, clip_id="fictional_dev_01", attempt_id="A001", **draft_values):
    return annotate_repetitions.build_annotation_row(
        clip_id=clip_id,
        attempt_id=attempt_id,
        draft=make_draft(**draft_values),
    )


def write_manifest(path, manifest):
    manifest.to_csv(path, index=False)


def test_manifest_technical_metadata_mapping_uses_established_schema():
    metadata = annotate_repetitions.VideoTechnicalMetadata(
        video_path="data/raw/fictional/source.mp4",
        sha256="a" * 64,
        source_fps=29.97,
        frame_count=900,
        width_px=1920,
        height_px=1080,
        duration_seconds=900 / 29.97,
        frame_decodable=True,
    )

    row = annotate_repetitions.build_manifest_row(
        metadata,
        clip_id="fictional_dev_01",
        participant_id="P_FICTIONAL_01",
        camera_view="side",
        recording_condition="fictional_controlled_condition",
        notes="Fictional mapping test.",
    )

    assert tuple(row) == MANIFEST_COLUMNS
    assert row["split"] == "development"
    assert row["video_path"] == metadata.video_path
    assert "sha256" not in row
    validate_dataset_manifest(pd.DataFrame([row]))


@pytest.mark.parametrize(
    ("depth", "extension", "alignment", "expected"),
    [
        (False, False, False, "correct"),
        (True, False, False, "insufficient_depth"),
        (False, True, False, "incomplete_extension"),
        (False, False, True, "alignment_deviation"),
        (True, True, True, "insufficient_depth"),
        (False, True, True, "incomplete_extension"),
    ],
)
def test_class_and_deviation_selection_uses_frozen_priority(
    depth,
    extension,
    alignment,
    expected,
):
    assert (
        annotate_repetitions.canonical_class_for_selection(
            insufficient_depth=depth,
            incomplete_extension=extension,
            alignment_deviation=alignment,
            ambiguity=False,
            visibility_status="sufficient",
        )
        == expected
    )


def test_canonical_annotation_row_retains_multiple_deviation_flags():
    row = make_row(
        insufficient_depth_flag=True,
        incomplete_extension_flag=True,
        alignment_deviation_flag=True,
    )

    assert row["ground_truth_class"] == "insufficient_depth"
    assert row["insufficient_depth_flag"] == "true"
    assert row["incomplete_extension_flag"] == "true"
    assert row["alignment_deviation_flag"] == "true"
    validate_repetition_annotations(pd.DataFrame([row]), make_manifest())


def test_ambiguous_fragment_retains_location_and_protocol_state():
    draft = annotate_repetitions.AnnotationDraft(
        start_top_frame=45,
        ambiguity_flag=True,
        source_video_visibility_status="partially_obscured",
        annotator_notes="Fictional clip-boundary fragment.",
    )

    row = annotate_repetitions.build_annotation_row(
        clip_id="fictional_dev_01",
        attempt_id="F001",
        draft=draft,
    )

    assert row["is_evaluable_attempt"] == "false"
    assert row["ambiguity_flag"] == "true"
    assert row["ground_truth_class"] == "unscorable"
    assert row["start_top_frame"] == 45
    assert row["bottom_turnaround_frame"] == ""
    validate_repetition_annotations(pd.DataFrame([row]), make_manifest())


def test_duplicate_prevention_does_not_replace_existing_row(tmp_path):
    annotations_path = tmp_path / "annotations.csv"
    annotate_repetitions.ensure_annotation_file(annotations_path)
    manifest = make_manifest()
    original = make_row()
    annotate_repetitions.append_annotation_row(
        annotations_path,
        original,
        manifest,
    )
    before = annotations_path.read_bytes()

    with pytest.raises(ValueError, match="Duplicate annotation identity"):
        annotate_repetitions.append_annotation_row(
            annotations_path,
            {**original, "annotator_notes": "Would replace existing work."},
            manifest,
        )

    assert annotations_path.read_bytes() == before


def test_safe_resume_restores_unsaved_draft_and_protects_other_clip(tmp_path):
    checkpoint_path = tmp_path / "annotations.resume.json"
    draft = make_draft(
        incomplete_extension_flag=True,
        annotator_notes="Fictional in-progress note.",
    )
    annotate_repetitions.save_resume_checkpoint(
        checkpoint_path,
        clip_id="fictional_dev_01",
        current_frame=27,
        draft=draft,
    )

    restored = annotate_repetitions.load_resume_checkpoint(
        checkpoint_path,
        clip_id="fictional_dev_01",
    )

    assert restored is not None
    assert restored[0] == 27
    assert restored[1] == draft
    with pytest.raises(ValueError, match="unfinished draft"):
        annotate_repetitions.load_resume_checkpoint(
            checkpoint_path,
            clip_id="fictional_dev_02",
        )


def test_annotation_rows_use_manifest_and_chronological_order(tmp_path):
    annotations_path = tmp_path / "annotations.csv"
    annotate_repetitions.ensure_annotation_file(annotations_path)
    manifest = make_manifest(second_clip=True)

    for row in [
        make_row(clip_id="fictional_dev_02", attempt_id="A001"),
        make_row(
            attempt_id="A002",
            start_top_frame=100,
            bottom_turnaround_frame=110,
            completion_end_top_frame=120,
        ),
        make_row(
            attempt_id="A001",
            start_top_frame=10,
            bottom_turnaround_frame=20,
            completion_end_top_frame=30,
        ),
    ]:
        annotate_repetitions.append_annotation_row(
            annotations_path,
            row,
            manifest,
        )

    identities = [
        (row["clip_id"], row["ground_truth_attempt_id"])
        for row in annotate_repetitions.load_annotation_rows(annotations_path)
    ]
    assert identities == [
        ("fictional_dev_01", "A001"),
        ("fictional_dev_01", "A002"),
        ("fictional_dev_02", "A001"),
    ]


def test_review_metadata_starts_pending_without_freeze_hash(tmp_path):
    annotations_path = tmp_path / "annotations.csv"
    metadata_path = tmp_path / "annotations.review.json"
    annotate_repetitions.ensure_annotation_file(annotations_path)

    document = annotate_repetitions.start_review_metadata(
        metadata_path,
        annotations_path=annotations_path,
        annotator="ANN_FICTIONAL_01",
        now=datetime(2026, 8, 12, 10, 0, tzinfo=timezone.utc),
    )

    assert document["annotator"] == "ANN_FICTIONAL_01"
    assert document["annotation_date"] == "2026-08-12"
    assert document["review_status"] == "in_progress"
    assert document["frozen_annotation_sha256"] is None


def test_review_finalisation_requires_coverage_and_records_exact_hash(tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    annotations_path = tmp_path / "annotations.csv"
    metadata_path = tmp_path / "annotations.review.json"
    manifest = make_manifest()
    write_manifest(manifest_path, manifest)
    annotate_repetitions.ensure_annotation_file(annotations_path)
    annotate_repetitions.start_review_metadata(
        metadata_path,
        annotations_path=annotations_path,
        annotator="ANN_FICTIONAL_01",
    )

    with pytest.raises(ValueError, match="missing"):
        annotate_repetitions.finalise_review_metadata(
            metadata_path,
            manifest_path=manifest_path,
            annotations_path=annotations_path,
            reviewer="REVIEWER_FICTIONAL_01",
        )

    annotate_repetitions.append_annotation_row(
        annotations_path,
        make_row(),
        manifest,
    )
    expected_hash = hashlib.sha256(annotations_path.read_bytes()).hexdigest()
    document = annotate_repetitions.finalise_review_metadata(
        metadata_path,
        manifest_path=manifest_path,
        annotations_path=annotations_path,
        reviewer="REVIEWER_FICTIONAL_01",
        repeat_review_status="complete",
        repeat_reviewer="REVIEWER_FICTIONAL_02",
        notes="Fictional completed review.",
        now=datetime(2026, 8, 13, 9, 30, tzinfo=timezone.utc),
    )

    assert document["review_status"] == "complete"
    assert document["reviewed_by"] == "REVIEWER_FICTIONAL_01"
    assert document["repeat_review_status"] == "complete"
    assert document["frozen_annotation_sha256"] == expected_hash
    assert document["finalised_at_utc"] == "2026-08-13T09:30:00Z"
    with pytest.raises(ValueError, match="frozen"):
        annotate_repetitions.start_review_metadata(
            metadata_path,
            annotations_path=annotations_path,
            annotator="ANN_FICTIONAL_01",
        )


def test_annotation_module_has_no_prediction_pipeline_imports_or_symbols():
    source_path = Path(annotate_repetitions.__file__)
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)

    prohibited_modules = {
        "analysis.baseline",
        "analysis.enhanced_features",
        "analysis.phase_state_machine",
        "analysis.repetition_classifier",
        "pose.estimator",
        "evaluation.repetition_events",
    }
    prohibited_symbols = {
        "EnhancedFeatureProcessor",
        "PushUpPhaseStateMachine",
        "RepetitionClassifier",
        "PoseEstimator",
    }
    assert imported_modules.isdisjoint(prohibited_modules)
    assert all(symbol not in source for symbol in prohibited_symbols)


def test_real_development_manifest_and_empty_annotation_structure_validate():
    project_root = Path(__file__).resolve().parents[1]
    manifest = pd.read_csv(
        project_root / "data" / "manifests" / "development_dataset_manifest.csv"
    )
    annotations = pd.read_csv(
        project_root / "data" / "annotations" / "development_repetition_annotations.csv"
    )

    assert len(manifest) == 12
    assert manifest["clip_id"].tolist() == [
        "dev01_correct",
        "dev02_insufficient_depth",
        "dev03_incomplete_extension",
        "dev04_alignment_deviation",
        "dev05_mixed_fast",
        "dev06_mixed_diagonal",
        "ext_kaggle_01",
        "ext_kaggle_02",
        "ext_kaggle_03",
        "ext_kaggle_04",
        "ext_kaggle_05",
        "ext_kaggle_06",
    ]
    assert manifest["video_path"].tolist() == [
        "data/raw/development/dev01_correct.mp4",
        "data/raw/development/dev02_insufficient_depth.mp4",
        "data/raw/development/dev03_incomplete_extension.mp4",
        "data/raw/development/dev04_alignment_deviation.mp4",
        "data/raw/development/dev05_mixed_fast.mp4",
        "data/raw/development/dev06_mixed_diagonal.mp4",
        ("data/raw/external/kaggle_pushup/Correct sequence/Copy of push up 47.mp4"),
        ("data/raw/external/kaggle_pushup/Correct sequence/Copy of push up 80.mp4"),
        ("data/raw/external/kaggle_pushup/Correct sequence/Copy of push up 164.mp4"),
        "data/raw/external/kaggle_pushup/Wrong sequence/8.mp4",
        ("data/raw/external/kaggle_pushup/Wrong sequence/Copy of push up 42.mp4"),
        ("data/raw/external/kaggle_pushup/Wrong sequence/Copy of push up 81.mp4"),
    ]
    assert set(manifest["split"]) == {"development"}
    assert not manifest["video_path"].str.contains("setup_test").any()
    assert annotations.empty
    assert tuple(annotations.columns) == ANNOTATION_COLUMNS
    validate_repetition_annotations(annotations, manifest)
