"""Load validated completion events from baseline, enhanced and GT tables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, TypeAlias

import pandas as pd

from evaluation.dataset_validation import (
    validate_repetition_annotations,
)
from evaluation.validation import require_columns

BASELINE_REQUIRED_COLUMNS = (
    "run_id",
    "clip_id",
    "frame_index",
    "baseline_rep_count",
)

FORMAL_BASELINE_FRAME_COLUMNS = (
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

ENHANCED_REQUIRED_COLUMNS = (
    "run_id",
    "clip_id",
    "rep_id",
    "start_frame",
    "bottom_frame",
    "end_frame",
    "predicted_class",
)


@dataclass(frozen=True)
class BaselineRepetitionEvent:
    """A baseline count increment at its observed completion frame.

    ``predicted_rep_id`` is the resulting cumulative count. The timestamp is in
    milliseconds when recorded or derivable from FPS.
    """

    run_id: str
    clip_id: str
    predicted_rep_id: int
    completion_frame: int
    completion_timestamp_ms: float | None
    resulting_cumulative_count: int
    method: str = "baseline"


@dataclass(frozen=True)
class EnhancedRepetitionEvent:
    """An enhanced repetition ending at its confirmed return-to-top frame.

    Start, bottom and completion fields are integer frame identities. The class
    is the enhanced repetition-level prediction, not a baseline warning.
    """

    run_id: str
    clip_id: str
    predicted_rep_id: int
    start_frame: int
    bottom_frame: int
    completion_frame: int
    completion_timestamp_ms: float | None
    predicted_class: str
    method: str = "enhanced"


@dataclass(frozen=True)
class GroundTruthRepetitionEvent:
    """An evaluable annotated attempt at its completion/end-top frame."""

    clip_id: str
    ground_truth_attempt_id: str
    completion_frame: int
    completion_timestamp_ms: float | None
    ground_truth_class: str
    method: str = "ground_truth"


PredictedRepetitionEvent: TypeAlias = (
    BaselineRepetitionEvent | EnhancedRepetitionEvent
)


def _text_series(
    data: pd.DataFrame,
    column: str,
    source_name: str,
) -> pd.Series:
    values = (
        data[column]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    empty_mask = values.eq("")

    if empty_mask.any():
        rows = list(data.index[empty_mask])
        raise ValueError(
            f"{source_name} column {column!r} contains empty "
            f"values at rows {rows}"
        )

    return values


def _integer_series(
    data: pd.DataFrame,
    column: str,
    source_name: str,
    minimum: int,
) -> pd.Series:
    numeric = pd.to_numeric(
        data[column],
        errors="coerce",
    )
    invalid_mask = (
        numeric.isna()
        | numeric.lt(minimum)
        | numeric.mod(1).ne(0)
    )

    if invalid_mask.any():
        rows = list(data.index[invalid_mask])
        raise ValueError(
            f"{source_name} column {column!r} must contain "
            f"integers of at least {minimum}; invalid rows "
            f"{rows}"
        )

    return numeric.astype(int)


def _optional_nonnegative_number_series(
    data: pd.DataFrame,
    column: str,
    source_name: str,
) -> pd.Series:
    if column not in data.columns:
        return pd.Series(
            float("nan"),
            index=data.index,
            dtype=float,
        )

    text = (
        data[column]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    missing_mask = text.eq("")
    numeric = pd.to_numeric(
        data[column].where(~missing_mask),
        errors="coerce",
    )
    invalid_mask = (
        ~missing_mask
        & (numeric.isna() | numeric.lt(0.0))
    )

    if invalid_mask.any():
        rows = list(data.index[invalid_mask])
        raise ValueError(
            f"{source_name} column {column!r} must contain "
            f"non-negative numbers or be empty; invalid rows "
            f"{rows}"
        )

    return numeric.astype(float)


def _optional_positive_number_series(
    data: pd.DataFrame,
    column: str,
    source_name: str,
) -> pd.Series:
    values = _optional_nonnegative_number_series(
        data,
        column,
        source_name,
    )
    invalid_mask = values.notna() & values.le(0.0)

    if invalid_mask.any():
        rows = list(data.index[invalid_mask])
        raise ValueError(
            f"{source_name} column {column!r} must contain "
            f"positive numbers or be empty; invalid rows {rows}"
        )

    return values


def _normalise_source_fps(
    source_fps_by_clip: Mapping[str, float] | None,
) -> dict[str, float]:
    if source_fps_by_clip is None:
        return {}

    normalised = {}

    for clip_id, source_fps in source_fps_by_clip.items():
        try:
            fps = float(source_fps)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Source FPS for clip {clip_id!r} must be "
                "a positive number"
            ) from error

        if fps <= 0.0:
            raise ValueError(
                f"Source FPS for clip {clip_id!r} must be "
                "a positive number"
            )

        normalised[str(clip_id).strip()] = fps

    return normalised


def _completion_timestamp(
    *,
    row_index: int,
    clip_id: str,
    completion_frame: int,
    recorded_timestamps: pd.Series,
    row_source_fps: pd.Series,
    source_fps_by_clip: Mapping[str, float],
) -> float | None:
    """Use a recorded millisecond timestamp or derive one from frame and FPS."""
    recorded_timestamp = recorded_timestamps.loc[row_index]

    if pd.notna(recorded_timestamp):
        return float(recorded_timestamp)

    source_fps = row_source_fps.loc[row_index]

    if pd.isna(source_fps):
        source_fps = source_fps_by_clip.get(clip_id)

    if source_fps is None or pd.isna(source_fps):
        return None

    return completion_frame / float(source_fps) * 1000.0


def _validate_event_identifiers(
    events: list[PredictedRepetitionEvent],
    source_name: str,
) -> None:
    identifiers = [
        (event.clip_id, event.predicted_rep_id)
        for event in events
    ]

    if len(identifiers) != len(set(identifiers)):
        raise ValueError(
            f"{source_name} contains duplicate "
            "(clip_id, predicted_rep_id) identifiers"
        )


def _validate_expected_baseline_frames(
    *,
    run_ids: pd.Series,
    clip_ids: pd.Series,
    frames: pd.Series,
    source_name: str,
    expected_run_id: str | None,
    expected_clip_id: str | None,
    expected_frame_count: int | None,
) -> None:
    """Bind a formal baseline table to completed-run identity and coverage."""
    if expected_run_id is not None:
        actual_run_ids = sorted(set(run_ids))

        if actual_run_ids != [expected_run_id]:
            raise ValueError(
                f"{source_name} run IDs {actual_run_ids} do not "
                f"match completed-run metadata {expected_run_id!r}"
            )

    if expected_clip_id is not None:
        actual_clip_ids = sorted(set(clip_ids))

        if actual_clip_ids != [expected_clip_id]:
            raise ValueError(
                f"{source_name} clip IDs {actual_clip_ids} do not "
                f"match completed-run metadata {expected_clip_id!r}"
            )

    if expected_frame_count is None:
        return

    if len(frames) != expected_frame_count:
        raise ValueError(
            f"{source_name} contains {len(frames)} frame rows; "
            "completed-run metadata records "
            f"{expected_frame_count} frames"
        )

    expected_frames = list(range(expected_frame_count))

    if frames.tolist() != expected_frames:
        raise ValueError(
            f"{source_name} frame indices must cover every frame "
            f"from 0 to {expected_frame_count - 1} in order"
        )


def extract_baseline_events(
    frame_data: pd.DataFrame,
    *,
    source_name: str = "Baseline frame data",
    source_fps_by_clip: Mapping[str, float] | None = None,
    expected_run_id: str | None = None,
    expected_clip_id: str | None = None,
    expected_frame_count: int | None = None,
) -> list[BaselineRepetitionEvent]:
    """Validate a baseline frame table before extracting count increments.

    Each increase of exactly one in ``baseline_rep_count`` defines a completion
    event on that row's frame. Frames must increase and counts must never fall
    or jump. When completed-run expectations are supplied, the full baseline
    frame schema, exact run/clip identity and every frame from zero through the
    recorded frame count are required before extraction. A structurally
    complete file with no count increases is valid and returns no events.

    Baseline positions, warnings and feature columns remain diagnostic; formal
    event identity comes from run, clip, frame and cumulative-count fields.
    """
    require_columns(
        frame_data,
        BASELINE_REQUIRED_COLUMNS,
        source_name,
    )

    if any(
        value is not None
        for value in (
            expected_run_id,
            expected_clip_id,
            expected_frame_count,
        )
    ):
        require_columns(
            frame_data,
            FORMAL_BASELINE_FRAME_COLUMNS,
            source_name,
        )

    source_fps = _normalise_source_fps(
        source_fps_by_clip
    )
    run_ids = _text_series(
        frame_data,
        "run_id",
        source_name,
    )
    clip_ids = _text_series(
        frame_data,
        "clip_id",
        source_name,
    )
    frames = _integer_series(
        frame_data,
        "frame_index",
        source_name,
        minimum=0,
    )
    counts = _integer_series(
        frame_data,
        "baseline_rep_count",
        source_name,
        minimum=0,
    )
    _validate_expected_baseline_frames(
        run_ids=run_ids,
        clip_ids=clip_ids,
        frames=frames,
        source_name=source_name,
        expected_run_id=expected_run_id,
        expected_clip_id=expected_clip_id,
        expected_frame_count=expected_frame_count,
    )
    timestamps = _optional_nonnegative_number_series(
        frame_data,
        "video_timestamp_ms",
        source_name,
    )
    row_fps = _optional_positive_number_series(
        frame_data,
        "source_fps",
        source_name,
    )
    working = pd.DataFrame(
        {
            "run_id": run_ids,
            "clip_id": clip_ids,
            "frame_index": frames,
            "baseline_rep_count": counts,
            "video_timestamp_ms": timestamps,
            "source_fps": row_fps,
        }
    )
    events: list[BaselineRepetitionEvent] = []

    for (run_id, clip_id), rows in working.groupby(
        ["run_id", "clip_id"],
        sort=False,
    ):
        frame_differences = rows["frame_index"].diff()

        if frame_differences.iloc[1:].le(0).any():
            raise ValueError(
                f"{source_name} frame indices must be strictly "
                f"increasing for run {run_id!r}, clip "
                f"{clip_id!r}"
            )

        count_differences = rows[
            "baseline_rep_count"
        ].diff()

        if count_differences.iloc[1:].lt(0).any():
            raise ValueError(
                f"{source_name} baseline counts must be "
                f"non-decreasing for run {run_id!r}, clip "
                f"{clip_id!r}"
            )

        if count_differences.iloc[1:].gt(1).any():
            raise ValueError(
                f"{source_name} baseline count increments must "
                f"be exactly one for run {run_id!r}, clip "
                f"{clip_id!r}"
            )

        completion_rows = rows.loc[
            count_differences.eq(1)
        ]

        for row_index, row in completion_rows.iterrows():
            completion_frame = int(row["frame_index"])
            count = int(row["baseline_rep_count"])
            events.append(
                BaselineRepetitionEvent(
                    run_id=run_id,
                    clip_id=clip_id,
                    predicted_rep_id=count,
                    completion_frame=completion_frame,
                    completion_timestamp_ms=(
                        _completion_timestamp(
                            row_index=row_index,
                            clip_id=clip_id,
                            completion_frame=(
                                completion_frame
                            ),
                            recorded_timestamps=timestamps,
                            row_source_fps=row_fps,
                            source_fps_by_clip=source_fps,
                        )
                    ),
                    resulting_cumulative_count=count,
                )
            )

    _validate_event_identifiers(events, source_name)
    completion_keys = [
        (event.clip_id, event.completion_frame)
        for event in events
    ]

    if len(completion_keys) != len(set(completion_keys)):
        raise ValueError(
            f"{source_name} contains duplicate completion "
            "frames within a clip"
        )

    return sorted(
        events,
        key=lambda event: (
            event.clip_id,
            event.completion_frame,
            event.predicted_rep_id,
        ),
    )


def load_baseline_events(
    csv_path: str | Path,
    *,
    source_fps_by_clip: Mapping[str, float] | None = None,
    expected_run_id: str | None = None,
    expected_clip_id: str | None = None,
    expected_frame_count: int | None = None,
) -> list[BaselineRepetitionEvent]:
    """Read a baseline frame CSV and extract its validated completion events."""
    path = Path(csv_path)
    return extract_baseline_events(
        pd.read_csv(path),
        source_name=str(path),
        source_fps_by_clip=source_fps_by_clip,
        expected_run_id=expected_run_id,
        expected_clip_id=expected_clip_id,
        expected_frame_count=expected_frame_count,
    )


def extract_enhanced_events(
    repetition_data: pd.DataFrame,
    *,
    source_name: str = "Enhanced repetition data",
    source_fps_by_clip: Mapping[str, float] | None = None,
) -> list[EnhancedRepetitionEvent]:
    """Validate enhanced repetition rows and load completion events.

    IDs must be unique within each clip and frames must satisfy start <= bottom
    <= completion. ``end_frame`` is the formal completion event; recorded
    millisecond timestamps are preferred and FPS supplies the fallback.
    """
    require_columns(
        repetition_data,
        ENHANCED_REQUIRED_COLUMNS,
        source_name,
    )
    source_fps = _normalise_source_fps(
        source_fps_by_clip
    )
    run_ids = _text_series(
        repetition_data,
        "run_id",
        source_name,
    )
    clip_ids = _text_series(
        repetition_data,
        "clip_id",
        source_name,
    )
    repetition_ids = _integer_series(
        repetition_data,
        "rep_id",
        source_name,
        minimum=1,
    )
    start_frames = _integer_series(
        repetition_data,
        "start_frame",
        source_name,
        minimum=0,
    )
    bottom_frames = _integer_series(
        repetition_data,
        "bottom_frame",
        source_name,
        minimum=0,
    )
    end_frames = _integer_series(
        repetition_data,
        "end_frame",
        source_name,
        minimum=0,
    )
    predicted_classes = _text_series(
        repetition_data,
        "predicted_class",
        source_name,
    )
    timestamps = _optional_nonnegative_number_series(
        repetition_data,
        "completion_timestamp_ms",
        source_name,
    )
    row_fps = _optional_positive_number_series(
        repetition_data,
        "source_fps",
        source_name,
    )

    invalid_order = (
        start_frames.gt(bottom_frames)
        | bottom_frames.gt(end_frames)
    )

    if invalid_order.any():
        rows = list(repetition_data.index[invalid_order])
        raise ValueError(
            f"{source_name} frames must satisfy "
            f"start_frame <= bottom_frame <= end_frame; "
            f"invalid rows {rows}"
        )

    events = []

    for row_index in repetition_data.index:
        clip_id = clip_ids.loc[row_index]
        completion_frame = int(end_frames.loc[row_index])
        events.append(
            EnhancedRepetitionEvent(
                run_id=run_ids.loc[row_index],
                clip_id=clip_id,
                predicted_rep_id=int(
                    repetition_ids.loc[row_index]
                ),
                start_frame=int(
                    start_frames.loc[row_index]
                ),
                bottom_frame=int(
                    bottom_frames.loc[row_index]
                ),
                completion_frame=completion_frame,
                completion_timestamp_ms=(
                    _completion_timestamp(
                        row_index=row_index,
                        clip_id=clip_id,
                        completion_frame=completion_frame,
                        recorded_timestamps=timestamps,
                        row_source_fps=row_fps,
                        source_fps_by_clip=source_fps,
                    )
                ),
                predicted_class=(
                    predicted_classes.loc[row_index]
                ),
            )
        )

    _validate_event_identifiers(events, source_name)

    return sorted(
        events,
        key=lambda event: (
            event.clip_id,
            event.completion_frame,
            event.predicted_rep_id,
        ),
    )


def load_enhanced_events(
    csv_path: str | Path,
    *,
    source_fps_by_clip: Mapping[str, float] | None = None,
) -> list[EnhancedRepetitionEvent]:
    """Read an enhanced repetition CSV and load its validated events."""
    path = Path(csv_path)
    return extract_enhanced_events(
        pd.read_csv(path),
        source_name=str(path),
        source_fps_by_clip=source_fps_by_clip,
    )


def extract_ground_truth_events(
    annotations: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    source_name: str = "Repetition annotations",
    manifest_source_name: str = "Dataset manifest",
) -> list[GroundTruthRepetitionEvent]:
    """Extract evaluable GT events after full annotation validation.

    The annotated completion/end-top frame is the formal event identity.
    Ambiguous fragments remain documented annotation rows but are deliberately
    excluded because ``is_evaluable_attempt`` is false.
    """
    validate_repetition_annotations(
        annotations,
        manifest,
        source_name=source_name,
        manifest_source_name=manifest_source_name,
    )
    manifest_clip_ids = (
        manifest["clip_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    source_fps_by_clip = dict(
        zip(
            manifest_clip_ids,
            pd.to_numeric(manifest["source_fps"]),
        )
    )
    evaluable = (
        annotations["is_evaluable_attempt"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("true")
    )
    events = []

    for row_index, row in annotations.loc[
        evaluable
    ].iterrows():
        clip_id = str(row["clip_id"]).strip()
        completion_frame = int(
            row["completion_end_top_frame"]
        )
        source_fps = float(
            source_fps_by_clip[clip_id]
        )
        events.append(
            GroundTruthRepetitionEvent(
                clip_id=clip_id,
                ground_truth_attempt_id=str(
                    row["ground_truth_attempt_id"]
                ).strip(),
                completion_frame=completion_frame,
                completion_timestamp_ms=(
                    completion_frame
                    / source_fps
                    * 1000.0
                ),
                ground_truth_class=str(
                    row["ground_truth_class"]
                ).strip(),
            )
        )

    return sorted(
        events,
        key=lambda event: (
            event.clip_id,
            event.completion_frame,
            event.ground_truth_attempt_id,
        ),
    )
