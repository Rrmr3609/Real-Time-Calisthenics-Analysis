from pathlib import Path

import pandas as pd
import pytest

from evaluation.dataset_validation import (
    load_and_validate_evaluation_data,
    validate_dataset_manifest,
    validate_repetition_annotations,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_manifest():
    return pd.DataFrame(
        [
            {
                "clip_id": "fictional_clip",
                "split": "development",
                "video_path": (
                    "data/raw/fictional/fictional_clip.mp4"
                ),
                "participant_id": "P_FICTIONAL",
                "camera_view": "side",
                "source_fps": 30.0,
                "frame_count": 500,
                "width_px": 1280,
                "height_px": 720,
                "recording_condition": (
                    "controlled_indoor_even_lighting"
                ),
                "notes": "Fictional test row.",
            }
        ]
    )


def make_annotations():
    return pd.DataFrame(
        [
            {
                "clip_id": "fictional_clip",
                "ground_truth_attempt_id": "A001",
                "is_evaluable_attempt": True,
                "ambiguity_flag": False,
                "start_top_frame": 100,
                "bottom_turnaround_frame": 130,
                "completion_end_top_frame": 160,
                "ground_truth_class": "correct",
                "insufficient_depth_flag": False,
                "incomplete_extension_flag": False,
                "alignment_deviation_flag": False,
                "source_video_visibility_status": "sufficient",
                "annotator_notes": "",
            }
        ]
    )


def test_fictional_example_files_pass_validation():
    manifest, annotations = load_and_validate_evaluation_data(
        manifest_path=(
            PROJECT_ROOT
            / "data"
            / "manifests"
            / "example_dataset_manifest.csv"
        ),
        annotations_path=(
            PROJECT_ROOT
            / "data"
            / "annotations"
            / "example_repetition_annotations.csv"
        ),
    )

    assert len(manifest) == 2
    assert len(annotations) == 4


def test_manifest_rejects_invalid_split():
    manifest = make_manifest()
    manifest.loc[0, "split"] = "validation"

    with pytest.raises(ValueError, match="invalid values"):
        validate_dataset_manifest(manifest)


def test_manifest_rejects_duplicate_clip_id():
    manifest = pd.concat(
        [make_manifest(), make_manifest()],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="duplicate identifiers",
    ):
        validate_dataset_manifest(manifest)


def test_annotations_reject_duplicate_attempt_id_within_clip():
    annotations = pd.concat(
        [make_annotations(), make_annotations()],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="duplicate identifiers",
    ):
        validate_repetition_annotations(
            annotations,
            make_manifest(),
        )


def test_annotations_reject_unknown_clip():
    annotations = make_annotations()
    annotations.loc[0, "clip_id"] = "unknown_clip"

    with pytest.raises(ValueError, match="unknown clip IDs"):
        validate_repetition_annotations(
            annotations,
            make_manifest(),
        )


def test_annotations_reject_invalid_label():
    annotations = make_annotations()
    annotations.loc[0, "ground_truth_class"] = "almost_correct"

    with pytest.raises(ValueError, match="invalid values"):
        validate_repetition_annotations(
            annotations,
            make_manifest(),
        )


def test_annotations_reject_invalid_frame_order():
    annotations = make_annotations()
    annotations.loc[0, "bottom_turnaround_frame"] = 90

    with pytest.raises(ValueError, match="start <= bottom"):
        validate_repetition_annotations(
            annotations,
            make_manifest(),
        )


def test_evaluable_attempt_requires_all_event_frames():
    annotations = make_annotations()
    annotations.loc[0, "bottom_turnaround_frame"] = None

    with pytest.raises(
        ValueError,
        match="require all three frame indices",
    ):
        validate_repetition_annotations(
            annotations,
            make_manifest(),
        )


def test_ambiguous_fragment_is_distinct_from_evaluable_attempt():
    annotations = make_annotations()
    annotations.loc[0, "is_evaluable_attempt"] = False
    annotations.loc[0, "ambiguity_flag"] = True
    annotations.loc[0, "bottom_turnaround_frame"] = None
    annotations.loc[0, "ground_truth_class"] = "unscorable"
    annotations.loc[0, "source_video_visibility_status"] = (
        "partially_obscured"
    )
    annotations.loc[0, "annotator_notes"] = (
        "Fictional ambiguous movement fragment."
    )

    validate_repetition_annotations(
        annotations,
        make_manifest(),
    )

    annotations.loc[0, "is_evaluable_attempt"] = True

    with pytest.raises(
        ValueError,
        match="distinguish evaluable attempts",
    ):
        validate_repetition_annotations(
            annotations,
            make_manifest(),
        )


def test_annotation_frame_must_be_inside_manifest_clip():
    annotations = make_annotations()
    annotations.loc[0, "completion_end_top_frame"] = 500

    with pytest.raises(ValueError, match="outside clip"):
        validate_repetition_annotations(
            annotations,
            make_manifest(),
        )


def test_single_label_must_follow_deviation_priority():
    annotations = make_annotations()
    annotations.loc[0, "insufficient_depth_flag"] = True
    annotations.loc[0, "incomplete_extension_flag"] = True
    annotations.loc[0, "ground_truth_class"] = (
        "incomplete_extension"
    )

    with pytest.raises(
        ValueError,
        match="single-label priority",
    ):
        validate_repetition_annotations(
            annotations,
            make_manifest(),
        )


def test_unscorable_attempt_cannot_assert_deviation_flag():
    annotations = make_annotations()
    annotations.loc[0, "ground_truth_class"] = "unscorable"
    annotations.loc[0, "insufficient_depth_flag"] = True
    annotations.loc[0, "source_video_visibility_status"] = (
        "insufficient"
    )
    annotations.loc[0, "annotator_notes"] = (
        "Fictional row with insufficient source evidence."
    )

    with pytest.raises(
        ValueError,
        match="cannot assert deviation flags",
    ):
        validate_repetition_annotations(
            annotations,
            make_manifest(),
        )
