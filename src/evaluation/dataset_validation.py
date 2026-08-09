"""Validate dataset manifests and manual repetition annotations.

The same schemas support fictional examples and real evaluation inputs. This
module owns row/schema semantics, not complete-split review evidence or binding
to recorded source runs; formal execution performs those checks separately.
"""

import argparse
import math
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

from config.runtime import ALLOWED_SPLITS as _RUNTIME_ALLOWED_SPLITS
from evaluation.validation import require_columns


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

ALLOWED_SPLITS = frozenset(_RUNTIME_ALLOWED_SPLITS)
ALLOWED_CAMERA_VIEWS = frozenset({"side", "side_diagonal"})
ALLOWED_GROUND_TRUTH_CLASSES = frozenset(
    {
        "correct",
        "insufficient_depth",
        "incomplete_extension",
        "alignment_deviation",
        "unscorable",
    }
)
ALLOWED_VISIBILITY_STATUSES = frozenset(
    {
        "sufficient",
        "partially_obscured",
        "insufficient",
    }
)

FRAME_COLUMNS = (
    "start_top_frame",
    "bottom_turnaround_frame",
    "completion_end_top_frame",
)

BOOLEAN_COLUMNS = (
    "is_evaluable_attempt",
    "ambiguity_flag",
    "insufficient_depth_flag",
    "incomplete_extension_flag",
    "alignment_deviation_flag",
)

WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:[\\/]")


def _normalised_text(
    data: pd.DataFrame,
    column: str,
) -> pd.Series:
    return (
        data[column]
        .fillna("")
        .astype(str)
        .str.strip()
    )


def _require_nonempty_text(
    data: pd.DataFrame,
    columns: Iterable[str],
    source_name: str,
) -> None:
    for column in columns:
        empty_mask = _normalised_text(data, column).eq("")

        if empty_mask.any():
            rows = list(data.index[empty_mask])
            raise ValueError(
                f"{source_name} column {column!r} contains empty "
                f"values at rows {rows}"
            )


def _boolean_series(
    data: pd.DataFrame,
    column: str,
    source_name: str,
) -> pd.Series:
    normalised = (
        data[column]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    invalid_mask = ~normalised.isin({"true", "false"})

    if invalid_mask.any():
        invalid_values = sorted(
            set(normalised[invalid_mask])
        )
        raise ValueError(
            f"{source_name} column {column!r} must contain only "
            f"true or false; found {invalid_values}"
        )

    return normalised.eq("true")


def _positive_numeric_series(
    data: pd.DataFrame,
    column: str,
    source_name: str,
    integer: bool,
) -> pd.Series:
    numeric = pd.to_numeric(
        data[column],
        errors="coerce",
    )
    finite = numeric.map(
        lambda value: math.isfinite(value)
        if pd.notna(value)
        else False
    )
    invalid_mask = numeric.isna() | ~finite | numeric.le(0)

    if integer:
        invalid_mask |= numeric.mod(1).ne(0)

    if invalid_mask.any():
        rows = list(data.index[invalid_mask])
        value_type = "positive integers" if integer else "positive numbers"
        raise ValueError(
            f"{source_name} column {column!r} must contain "
            f"{value_type}; invalid rows {rows}"
        )

    return numeric


def _optional_frame_series(
    data: pd.DataFrame,
    column: str,
    source_name: str,
) -> pd.Series:
    text = _normalised_text(data, column)
    missing_mask = text.eq("")
    numeric = pd.to_numeric(
        data[column].where(~missing_mask),
        errors="coerce",
    )
    invalid_mask = (
        ~missing_mask
        & (
            numeric.isna()
            | numeric.lt(0)
            | numeric.mod(1).ne(0)
        )
    )

    if invalid_mask.any():
        rows = list(data.index[invalid_mask])
        raise ValueError(
            f"{source_name} column {column!r} must contain "
            f"non-negative integer frame indices or be empty; "
            f"invalid rows {rows}"
        )

    return numeric


def _reject_duplicate_keys(
    data: pd.DataFrame,
    key_columns: tuple[str, ...],
    source_name: str,
) -> None:
    keys = pd.DataFrame(
        {
            column: _normalised_text(data, column)
            for column in key_columns
        }
    )
    duplicate_mask = keys.duplicated(
        subset=list(key_columns),
        keep=False,
    )

    if not duplicate_mask.any():
        return

    duplicate_keys = (
        keys.loc[duplicate_mask, list(key_columns)]
        .drop_duplicates()
        .to_dict(orient="records")
    )
    raise ValueError(
        f"{source_name} contains duplicate identifiers: "
        f"{duplicate_keys}"
    )


def _validate_allowed_values(
    values: pd.Series,
    allowed_values: frozenset[str],
    column: str,
    source_name: str,
) -> None:
    invalid_values = sorted(
        set(values) - allowed_values
    )

    if invalid_values:
        raise ValueError(
            f"{source_name} column {column!r} contains invalid "
            f"values {invalid_values}; allowed values are "
            f"{sorted(allowed_values)}"
        )


def validate_dataset_manifest(
    manifest: pd.DataFrame,
    source_name: str = "Dataset manifest",
) -> None:
    """Validate clip identity, split, source metadata and local path fields.

    Clip IDs must be unique, split values must use the configured development
    or test vocabulary, camera views must be allowed, and video paths must be
    project-relative. FPS must be positive and finite; frame counts and pixel
    dimensions must be positive integers. Validation does not assert that the
    referenced video exists or change a clip's assigned split.
    """
    require_columns(
        manifest,
        MANIFEST_COLUMNS,
        source_name,
    )

    if manifest.empty:
        raise ValueError(
            f"{source_name} must contain at least one clip"
        )

    _require_nonempty_text(
        manifest,
        (
            "clip_id",
            "split",
            "video_path",
            "participant_id",
            "camera_view",
            "recording_condition",
        ),
        source_name,
    )
    _reject_duplicate_keys(
        manifest,
        ("clip_id",),
        source_name,
    )

    splits = _normalised_text(manifest, "split")
    _validate_allowed_values(
        splits,
        ALLOWED_SPLITS,
        "split",
        source_name,
    )

    camera_views = _normalised_text(
        manifest,
        "camera_view",
    )
    _validate_allowed_values(
        camera_views,
        ALLOWED_CAMERA_VIEWS,
        "camera_view",
        source_name,
    )

    video_paths = _normalised_text(
        manifest,
        "video_path",
    )
    absolute_path_mask = (
        video_paths.str.startswith(("/", "\\"))
        | video_paths.str.match(WINDOWS_ABSOLUTE_PATH)
    )

    if absolute_path_mask.any():
        rows = list(manifest.index[absolute_path_mask])
        raise ValueError(
            f"{source_name} video_path values must be "
            f"project-relative; absolute paths found at rows {rows}"
        )

    _positive_numeric_series(
        manifest,
        "source_fps",
        source_name,
        integer=False,
    )
    _positive_numeric_series(
        manifest,
        "frame_count",
        source_name,
        integer=True,
    )
    _positive_numeric_series(
        manifest,
        "width_px",
        source_name,
        integer=True,
    )
    _positive_numeric_series(
        manifest,
        "height_px",
        source_name,
        integer=True,
    )


def _expected_single_label(
    insufficient_depth: bool,
    incomplete_extension: bool,
    alignment_deviation: bool,
) -> str:
    """Apply the annotation protocol's deterministic deviation priority."""
    if insufficient_depth:
        return "insufficient_depth"

    if incomplete_extension:
        return "incomplete_extension"

    if alignment_deviation:
        return "alignment_deviation"

    return "correct"


def validate_repetition_annotations(
    annotations: pd.DataFrame,
    manifest: pd.DataFrame,
    source_name: str = "Repetition annotations",
    manifest_source_name: str = "Dataset manifest",
) -> None:
    """Validate annotation identity, frames, ambiguity and class semantics.

    Each ``(clip_id, ground_truth_attempt_id)`` pair is unique and must refer to
    a manifest clip. Evaluable attempts require ordered start, turnaround and
    completion frame indices. Ambiguous fragments are explicitly non-evaluable,
    retain at least one locating frame, use the ``unscorable`` class, assert no
    deviation flags and include annotator notes. Evaluable rows follow the
    documented single-label priority, while insufficient source visibility
    requires ``unscorable`` and notes.
    """
    validate_dataset_manifest(
        manifest,
        source_name=manifest_source_name,
    )
    require_columns(
        annotations,
        ANNOTATION_COLUMNS,
        source_name,
    )
    _require_nonempty_text(
        annotations,
        (
            "clip_id",
            "ground_truth_attempt_id",
            "ground_truth_class",
            "source_video_visibility_status",
        ),
        source_name,
    )
    _reject_duplicate_keys(
        annotations,
        ("clip_id", "ground_truth_attempt_id"),
        source_name,
    )

    annotation_clip_ids = _normalised_text(
        annotations,
        "clip_id",
    )
    manifest_clip_ids = set(
        _normalised_text(manifest, "clip_id")
    )
    unknown_clip_ids = sorted(
        set(annotation_clip_ids) - manifest_clip_ids
    )

    if unknown_clip_ids:
        raise ValueError(
            f"{source_name} refers to unknown clip IDs: "
            f"{unknown_clip_ids}"
        )

    ground_truth_classes = _normalised_text(
        annotations,
        "ground_truth_class",
    )
    _validate_allowed_values(
        ground_truth_classes,
        ALLOWED_GROUND_TRUTH_CLASSES,
        "ground_truth_class",
        source_name,
    )

    visibility_statuses = _normalised_text(
        annotations,
        "source_video_visibility_status",
    )
    _validate_allowed_values(
        visibility_statuses,
        ALLOWED_VISIBILITY_STATUSES,
        "source_video_visibility_status",
        source_name,
    )

    boolean_values = {
        column: _boolean_series(
            annotations,
            column,
            source_name,
        )
        for column in BOOLEAN_COLUMNS
    }
    evaluable = boolean_values["is_evaluable_attempt"]
    ambiguous = boolean_values["ambiguity_flag"]

    inconsistent_status = evaluable.eq(ambiguous)

    if inconsistent_status.any():
        rows = list(annotations.index[inconsistent_status])
        raise ValueError(
            f"{source_name} must distinguish evaluable attempts "
            f"from ambiguous fragments; invalid rows {rows}"
        )

    frame_values = {
        column: _optional_frame_series(
            annotations,
            column,
            source_name,
        )
        for column in FRAME_COLUMNS
    }
    start_frames = frame_values["start_top_frame"]
    bottom_frames = frame_values["bottom_turnaround_frame"]
    end_frames = frame_values["completion_end_top_frame"]

    missing_evaluable_frames = (
        evaluable
        & (
            start_frames.isna()
            | bottom_frames.isna()
            | end_frames.isna()
        )
    )

    if missing_evaluable_frames.any():
        rows = list(
            annotations.index[missing_evaluable_frames]
        )
        raise ValueError(
            f"{source_name} evaluable attempts require all three "
            f"frame indices; invalid rows {rows}"
        )

    fragment_has_no_frame = (
        ambiguous
        & start_frames.isna()
        & bottom_frames.isna()
        & end_frames.isna()
    )

    if fragment_has_no_frame.any():
        rows = list(annotations.index[fragment_has_no_frame])
        raise ValueError(
            f"{source_name} ambiguous fragments require at least "
            f"one frame index; invalid rows {rows}"
        )

    ordering_invalid = (
        (
            start_frames.notna()
            & bottom_frames.notna()
            & start_frames.gt(bottom_frames)
        )
        | (
            bottom_frames.notna()
            & end_frames.notna()
            & bottom_frames.gt(end_frames)
        )
        | (
            start_frames.notna()
            & end_frames.notna()
            & start_frames.gt(end_frames)
        )
    )

    if ordering_invalid.any():
        rows = list(annotations.index[ordering_invalid])
        raise ValueError(
            f"{source_name} frame indices must satisfy "
            f"start <= bottom <= completion where present; "
            f"invalid rows {rows}"
        )

    frame_count_by_clip = dict(
        zip(
            _normalised_text(manifest, "clip_id"),
            pd.to_numeric(manifest["frame_count"]),
        )
    )

    for row_index, clip_id in annotation_clip_ids.items():
        frame_count = frame_count_by_clip[clip_id]

        for column, values in frame_values.items():
            frame_value = values.loc[row_index]

            if (
                pd.notna(frame_value)
                and frame_value >= frame_count
            ):
                raise ValueError(
                    f"{source_name} row {row_index} column "
                    f"{column!r} is outside clip {clip_id!r}, "
                    f"which has {int(frame_count)} frames"
                )

    depth_flags = boolean_values[
        "insufficient_depth_flag"
    ]
    extension_flags = boolean_values[
        "incomplete_extension_flag"
    ]
    alignment_flags = boolean_values[
        "alignment_deviation_flag"
    ]
    any_deviation = (
        depth_flags
        | extension_flags
        | alignment_flags
    )

    ambiguous_rule_invalid = (
        ambiguous
        & (
            ground_truth_classes.ne("unscorable")
            | any_deviation
        )
    )

    if ambiguous_rule_invalid.any():
        rows = list(
            annotations.index[ambiguous_rule_invalid]
        )
        raise ValueError(
            f"{source_name} ambiguous fragments must use class "
            f"'unscorable' and false deviation flags; "
            f"invalid rows {rows}"
        )

    unscorable_deviation_invalid = (
        ground_truth_classes.eq("unscorable")
        & any_deviation
    )

    if unscorable_deviation_invalid.any():
        rows = list(
            annotations.index[
                unscorable_deviation_invalid
            ]
        )
        raise ValueError(
            f"{source_name} unscorable rows cannot assert "
            f"deviation flags; invalid rows {rows}"
        )

    insufficient_visibility = visibility_statuses.eq(
        "insufficient"
    )
    evaluable_unscorable = (
        evaluable
        & ground_truth_classes.eq("unscorable")
    )
    visibility_rule_invalid = (
        evaluable
        & (
            insufficient_visibility
            != ground_truth_classes.eq("unscorable")
        )
    )

    if visibility_rule_invalid.any():
        rows = list(
            annotations.index[visibility_rule_invalid]
        )
        raise ValueError(
            f"{source_name} evaluable attempts must use class "
            f"'unscorable' exactly when source visibility is "
            f"insufficient; invalid rows {rows}"
        )

    for row_index in annotations.index[
        evaluable & ~evaluable_unscorable
    ]:
        expected_class = _expected_single_label(
            bool(depth_flags.loc[row_index]),
            bool(extension_flags.loc[row_index]),
            bool(alignment_flags.loc[row_index]),
        )

        if ground_truth_classes.loc[row_index] != expected_class:
            raise ValueError(
                f"{source_name} row {row_index} class "
                f"{ground_truth_classes.loc[row_index]!r} does not "
                f"match deviation flags under the documented "
                f"single-label priority; expected "
                f"{expected_class!r}"
            )

    notes = _normalised_text(
        annotations,
        "annotator_notes",
    )
    notes_required = ambiguous | evaluable_unscorable
    missing_required_notes = notes_required & notes.eq("")

    if missing_required_notes.any():
        rows = list(
            annotations.index[missing_required_notes]
        )
        raise ValueError(
            f"{source_name} ambiguous or unscorable rows require "
            f"annotator notes; invalid rows {rows}"
        )


def load_and_validate_evaluation_data(
    manifest_path: Path,
    annotations_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both CSVs and validate annotations against manifest identities."""
    manifest = pd.read_csv(manifest_path)
    annotations = pd.read_csv(annotations_path)

    validate_repetition_annotations(
        annotations=annotations,
        manifest=manifest,
        source_name=str(annotations_path),
        manifest_source_name=str(manifest_path),
    )

    return manifest, annotations


def parse_arguments() -> argparse.Namespace:
    """Parse paths for the standalone dataset-validation command."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate a formal-evaluation dataset manifest and "
            "repetition annotation CSV."
        )
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Dataset manifest CSV.",
    )
    parser.add_argument(
        "--annotations",
        required=True,
        help="Repetition annotation CSV.",
    )
    return parser.parse_args()


def main() -> None:
    """Validate both CSVs and print their accepted row counts."""
    args = parse_arguments()
    manifest, annotations = load_and_validate_evaluation_data(
        manifest_path=Path(args.manifest),
        annotations_path=Path(args.annotations),
    )
    evaluable_count = _boolean_series(
        annotations,
        "is_evaluable_attempt",
        str(args.annotations),
    ).sum()

    print(
        "Validation passed: "
        f"{len(manifest)} clips, "
        f"{len(annotations)} annotation rows, "
        f"{int(evaluable_count)} evaluable attempts."
    )


if __name__ == "__main__":
    main()
