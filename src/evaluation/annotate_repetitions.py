"""Frame-accurate source-video annotation without prediction leakage.

This viewer decodes one manifest video and records human-selected event frames,
form labels and visibility decisions. It deliberately does not import pose,
baseline, enhanced-feature, temporal-segmentation or classification code.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import tempfile
import textwrap
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from evaluation.dataset_validation import (
    ALLOWED_VISIBILITY_STATUSES,
    ANNOTATION_COLUMNS,
    validate_dataset_manifest,
    validate_repetition_annotations,
)
from utils.paths import PROJECT_ROOT

WINDOW_TITLE = "Manual repetition annotation - source video only"
REVIEW_SCHEMA_VERSION = 1
REPEAT_REVIEW_STATUSES = frozenset({"not_performed", "pending", "complete"})
INITIAL_WINDOW_WIDTH = 1600
INITIAL_WINDOW_HEIGHT = 900
PANEL_WIDTH = 500
MIN_PANEL_WIDTH = 400
PANEL_PADDING = 18
PANEL_FONT_SCALE = 0.58
PANEL_LINE_HEIGHT = 29

FRIENDLY_CLASS_LABELS = {
    "correct": "Meets project criteria",
    "insufficient_depth": "Insufficient depth",
    "incomplete_extension": "Incomplete extension",
    "alignment_deviation": "Body alignment deviation",
    "unscorable": "Unscorable",
}


@dataclass(frozen=True)
class VideoTechnicalMetadata:
    """Non-algorithmic identity and container metadata for one source video."""

    video_path: str
    sha256: str
    source_fps: float
    frame_count: int
    width_px: int
    height_px: int
    duration_seconds: float
    frame_decodable: bool


@dataclass
class AnnotationDraft:
    """Unsaved human selections for the current source-video attempt."""

    start_top_frame: int | None = None
    bottom_turnaround_frame: int | None = None
    completion_end_top_frame: int | None = None
    insufficient_depth_flag: bool = False
    incomplete_extension_flag: bool = False
    alignment_deviation_flag: bool = False
    ambiguity_flag: bool = False
    source_video_visibility_status: str = "sufficient"
    annotator_notes: str = ""

    @property
    def is_blank(self) -> bool:
        """Return whether the draft contains no attempt-specific decisions."""
        return (
            self.start_top_frame is None
            and self.bottom_turnaround_frame is None
            and self.completion_end_top_frame is None
            and not self.insufficient_depth_flag
            and not self.incomplete_extension_flag
            and not self.alignment_deviation_flag
            and not self.ambiguity_flag
            and self.source_video_visibility_status == "sufficient"
            and not self.annotator_notes
        )

    def to_dict(self) -> dict[str, object]:
        """Return a serialisable checkpoint representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> AnnotationDraft:
        """Restore a checkpoint while rejecting unknown or missing fields."""
        expected = set(cls.__dataclass_fields__)
        if set(value) != expected:
            raise ValueError("Annotation resume checkpoint has an invalid draft schema")
        draft = cls(**value)
        if draft.source_video_visibility_status not in ALLOWED_VISIBILITY_STATUSES:
            raise ValueError("Annotation resume checkpoint has invalid visibility")
        return draft


def inspect_video_metadata(video_path: Path) -> VideoTechnicalMetadata:
    """Read SHA-256 and OpenCV metadata, decoding one frame only."""
    path = Path(video_path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    capture = cv2.VideoCapture(str(path))
    try:
        opened = capture.isOpened()
        source_fps = float(capture.get(cv2.CAP_PROP_FPS)) if opened else 0.0
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if opened else 0
        width_px = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) if opened else 0
        height_px = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) if opened else 0
        decoded, frame = capture.read() if opened else (False, None)
    finally:
        capture.release()

    frame_decodable = bool(decoded and frame is not None)
    if (
        not opened
        or not frame_decodable
        or not math.isfinite(source_fps)
        or source_fps <= 0.0
        or frame_count <= 0
        or width_px <= 0
        or height_px <= 0
    ):
        raise ValueError(f"Video cannot supply valid technical metadata: {path}")

    return VideoTechnicalMetadata(
        video_path=_portable_path_identity(path),
        sha256=digest.hexdigest(),
        source_fps=source_fps,
        frame_count=frame_count,
        width_px=width_px,
        height_px=height_px,
        duration_seconds=frame_count / source_fps,
        frame_decodable=frame_decodable,
    )


def build_manifest_row(
    metadata: VideoTechnicalMetadata,
    *,
    clip_id: str,
    participant_id: str,
    camera_view: str,
    recording_condition: str,
    notes: str = "",
) -> dict[str, object]:
    """Map inspected technical metadata into the established manifest schema."""
    return {
        "clip_id": clip_id,
        "split": "development",
        "video_path": metadata.video_path,
        "video_sha256": metadata.sha256,
        "participant_id": participant_id,
        "camera_view": camera_view,
        "source_fps": metadata.source_fps,
        "frame_count": metadata.frame_count,
        "width_px": metadata.width_px,
        "height_px": metadata.height_px,
        "recording_condition": recording_condition,
        "notes": notes,
    }


def canonical_class_for_selection(
    *,
    insufficient_depth: bool,
    incomplete_extension: bool,
    alignment_deviation: bool,
    ambiguity: bool,
    visibility_status: str,
) -> str:
    """Apply the existing manual-GT class and deviation priority semantics."""
    if visibility_status not in ALLOWED_VISIBILITY_STATUSES:
        raise ValueError(f"Unsupported source visibility: {visibility_status!r}")
    if ambiguity or visibility_status == "insufficient":
        return "unscorable"
    if insufficient_depth:
        return "insufficient_depth"
    if incomplete_extension:
        return "incomplete_extension"
    if alignment_deviation:
        return "alignment_deviation"
    return "correct"


def build_annotation_row(
    *,
    clip_id: str,
    attempt_id: str,
    draft: AnnotationDraft,
) -> dict[str, object]:
    """Build one canonical schema row from explicit human selections."""
    frames = (
        draft.start_top_frame,
        draft.bottom_turnaround_frame,
        draft.completion_end_top_frame,
    )
    for frame in frames:
        if frame is not None and (not isinstance(frame, int) or frame < 0):
            raise ValueError("Annotation frames must be non-negative integers")

    flags = (
        draft.insufficient_depth_flag,
        draft.incomplete_extension_flag,
        draft.alignment_deviation_flag,
    )
    notes = draft.annotator_notes.strip()

    if draft.ambiguity_flag:
        if all(frame is None for frame in frames):
            raise ValueError("Ambiguous fragments require at least one locating frame")
        if any(flags):
            raise ValueError("Ambiguous fragments cannot assert deviation flags")
        if not notes:
            raise ValueError("Ambiguous fragments require annotator notes")
    else:
        if any(frame is None for frame in frames):
            raise ValueError("Evaluable attempts require all three event frames")
        start, bottom, completion = frames
        if not start <= bottom <= completion:
            raise ValueError("Event frames must satisfy start <= bottom <= completion")

    if draft.source_video_visibility_status == "insufficient":
        if any(flags):
            raise ValueError("Unscorable attempts cannot assert deviation flags")
        if not notes:
            raise ValueError("Insufficient source visibility requires annotator notes")

    ground_truth_class = canonical_class_for_selection(
        insufficient_depth=draft.insufficient_depth_flag,
        incomplete_extension=draft.incomplete_extension_flag,
        alignment_deviation=draft.alignment_deviation_flag,
        ambiguity=draft.ambiguity_flag,
        visibility_status=draft.source_video_visibility_status,
    )
    return {
        "clip_id": clip_id.strip(),
        "ground_truth_attempt_id": attempt_id.strip(),
        "is_evaluable_attempt": str(not draft.ambiguity_flag).lower(),
        "ambiguity_flag": str(draft.ambiguity_flag).lower(),
        "start_top_frame": "" if frames[0] is None else frames[0],
        "bottom_turnaround_frame": "" if frames[1] is None else frames[1],
        "completion_end_top_frame": "" if frames[2] is None else frames[2],
        "ground_truth_class": ground_truth_class,
        "insufficient_depth_flag": str(draft.insufficient_depth_flag).lower(),
        "incomplete_extension_flag": str(draft.incomplete_extension_flag).lower(),
        "alignment_deviation_flag": str(draft.alignment_deviation_flag).lower(),
        "source_video_visibility_status": (draft.source_video_visibility_status),
        "annotator_notes": notes,
    }


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    """Load and validate the established dataset manifest."""
    manifest = pd.read_csv(manifest_path)
    validate_dataset_manifest(manifest, source_name=str(manifest_path))
    return manifest


def load_annotation_rows(annotations_path: Path) -> list[dict[str, str]]:
    """Load existing rows without altering or silently replacing them."""
    path = Path(annotations_path)
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != ANNOTATION_COLUMNS:
            raise ValueError(
                f"Annotation CSV header must exactly match {ANNOTATION_COLUMNS}"
            )
        return [dict(row) for row in reader]


def ensure_annotation_file(annotations_path: Path) -> None:
    """Create only a missing schema header; never replace existing annotations."""
    path = Path(annotations_path)
    if path.exists():
        load_annotation_rows(path)
        return
    _write_annotation_rows_atomic(path, [])


def next_attempt_id(
    rows: list[dict[str, str]],
    *,
    clip_id: str,
    ambiguous: bool,
) -> str:
    """Return the next stable A/F identifier for one clip."""
    prefix = "F" if ambiguous else "A"
    existing_numbers = []
    for row in rows:
        attempt_id = row["ground_truth_attempt_id"]
        if row["clip_id"] != clip_id or not attempt_id.startswith(prefix):
            continue
        suffix = attempt_id[len(prefix) :]
        if suffix.isdigit():
            existing_numbers.append(int(suffix))
    return f"{prefix}{max(existing_numbers, default=0) + 1:03d}"


def sort_annotation_rows(
    rows: list[dict[str, object]],
    manifest: pd.DataFrame,
) -> list[dict[str, object]]:
    """Order by manifest clip, locating frame and stable attempt identifier."""
    clip_order = {
        str(clip_id).strip(): index for index, clip_id in enumerate(manifest["clip_id"])
    }

    def key(row: dict[str, object]) -> tuple[int, int | float, str]:
        locating_frames = []
        for column in (
            "start_top_frame",
            "bottom_turnaround_frame",
            "completion_end_top_frame",
        ):
            value = str(row[column]).strip()
            if value:
                locating_frames.append(int(float(value)))
        return (
            clip_order[str(row["clip_id"]).strip()],
            min(locating_frames) if locating_frames else math.inf,
            str(row["ground_truth_attempt_id"]).strip(),
        )

    return sorted(rows, key=key)


def append_annotation_row(
    annotations_path: Path,
    row: dict[str, object],
    manifest: pd.DataFrame,
) -> list[dict[str, object]]:
    """Validate and atomically append one unique row in deterministic order."""
    existing: list[dict[str, object]] = load_annotation_rows(annotations_path)
    identity = (
        str(row["clip_id"]).strip(),
        str(row["ground_truth_attempt_id"]).strip(),
    )
    existing_identities = {
        (
            str(item["clip_id"]).strip(),
            str(item["ground_truth_attempt_id"]).strip(),
        )
        for item in existing
    }
    if identity in existing_identities:
        raise ValueError(f"Duplicate annotation identity: {identity}")

    combined = sort_annotation_rows([*existing, row], manifest)
    validate_repetition_annotations(
        pd.DataFrame(combined, columns=ANNOTATION_COLUMNS),
        manifest,
        source_name=str(annotations_path),
    )
    _write_annotation_rows_atomic(annotations_path, combined)
    return combined


def _require_editable_review(
    metadata_path: Path,
    *,
    annotations_path: Path,
) -> None:
    """Require an active review record before replacing a saved annotation."""
    path = Path(metadata_path)
    if not path.is_file():
        raise ValueError("Review metadata must exist before correcting annotations")

    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != REVIEW_SCHEMA_VERSION:
        raise ValueError("Unsupported annotation review metadata schema")
    if document.get("annotation_file") != _portable_path_identity(annotations_path):
        raise ValueError("Review metadata identifies a different annotation CSV")

    review_status = document.get("review_status")
    if review_status == "complete":
        raise ValueError("The annotation CSV is frozen and cannot be corrected")
    if review_status not in {"not_started", "in_progress"}:
        raise ValueError("Annotation review metadata has an invalid review status")


def replace_annotation_row(
    annotations_path: Path,
    row: dict[str, object],
    manifest: pd.DataFrame,
    *,
    metadata_path: Path,
) -> list[dict[str, object]]:
    """Atomically replace one existing identity while review remains open."""
    _require_editable_review(
        metadata_path,
        annotations_path=annotations_path,
    )
    existing: list[dict[str, object]] = load_annotation_rows(annotations_path)
    identity = (
        str(row["clip_id"]).strip(),
        str(row["ground_truth_attempt_id"]).strip(),
    )
    matching_indices = [
        index
        for index, item in enumerate(existing)
        if (
            str(item["clip_id"]).strip(),
            str(item["ground_truth_attempt_id"]).strip(),
        )
        == identity
    ]
    if len(matching_indices) != 1:
        raise ValueError(
            f"Correction requires exactly one existing annotation identity: {identity}"
        )

    combined = [*existing]
    combined[matching_indices[0]] = row
    combined = sort_annotation_rows(combined, manifest)
    validate_repetition_annotations(
        pd.DataFrame(combined, columns=ANNOTATION_COLUMNS),
        manifest,
        source_name=str(annotations_path),
    )
    _write_annotation_rows_atomic(annotations_path, combined)
    return combined


def _write_annotation_rows_atomic(
    annotations_path: Path,
    rows: list[dict[str, object]],
) -> None:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=ANNOTATION_COLUMNS,
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    _atomic_write_text(Path(annotations_path), output.getvalue())


def _portable_path_identity(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.name


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        with temporary_path.open("w", encoding="utf-8", newline="") as handle:
            handle.write(content)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def review_metadata_path(annotations_path: Path) -> Path:
    """Return the adjacent tracked review-record path."""
    return Path(annotations_path).with_suffix(".review.json")


def resume_checkpoint_path(annotations_path: Path) -> Path:
    """Return the adjacent ignored in-progress checkpoint path."""
    return Path(annotations_path).with_suffix(".resume.json")


def start_review_metadata(
    metadata_path: Path,
    *,
    annotations_path: Path,
    annotator: str,
    now: datetime | None = None,
) -> dict[str, object]:
    """Create or resume an explicitly non-final annotation review record."""
    annotator = annotator.strip()
    if not annotator:
        raise ValueError("Annotator identifier must not be empty")
    timestamp = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    path = Path(metadata_path)
    expected_annotation_file = _portable_path_identity(annotations_path)

    if path.exists():
        document = json.loads(path.read_text(encoding="utf-8"))
    else:
        document = {
            "schema_version": REVIEW_SCHEMA_VERSION,
            "annotation_file": expected_annotation_file,
            "annotator": None,
            "annotation_date": None,
            "review_status": "not_started",
            "reviewed_by": None,
            "review_date": None,
            "repeat_review_status": "not_performed",
            "repeat_reviewed_by": None,
            "notes": "",
            "adjudication_notes": "",
            "frozen_annotation_sha256": None,
            "finalised_at_utc": None,
        }

    if document.get("schema_version") != REVIEW_SCHEMA_VERSION:
        raise ValueError("Unsupported annotation review metadata schema")
    if document.get("annotation_file") != expected_annotation_file:
        raise ValueError("Review metadata identifies a different annotation CSV")
    if document.get("review_status") == "complete":
        raise ValueError("The annotation CSV has been frozen and cannot be resumed")
    existing_annotator = document.get("annotator")
    if existing_annotator not in {None, annotator}:
        raise ValueError(
            f"Review metadata already identifies annotator {existing_annotator!r}"
        )

    document["annotator"] = annotator
    document["annotation_date"] = (
        document.get("annotation_date") or timestamp.date().isoformat()
    )
    document["review_status"] = "in_progress"
    _atomic_write_text(path, json.dumps(document, indent=2) + "\n")
    return document


def finalise_review_metadata(
    metadata_path: Path,
    *,
    manifest_path: Path,
    annotations_path: Path,
    reviewer: str,
    repeat_review_status: str = "not_performed",
    repeat_reviewer: str | None = None,
    notes: str = "",
    adjudication_notes: str = "",
    now: datetime | None = None,
) -> dict[str, object]:
    """Explicitly validate, cover, hash and freeze completed human review."""
    reviewer = reviewer.strip()
    if not reviewer:
        raise ValueError("Reviewer identifier must not be empty")
    if repeat_review_status not in REPEAT_REVIEW_STATUSES:
        raise ValueError("Unsupported repeat review status")
    if repeat_review_status == "complete" and not (repeat_reviewer or "").strip():
        raise ValueError("A completed repeat review requires a reviewer identifier")

    manifest = load_manifest(manifest_path)
    rows = load_annotation_rows(annotations_path)
    annotations = pd.DataFrame(rows, columns=ANNOTATION_COLUMNS)
    validate_repetition_annotations(
        annotations,
        manifest,
        source_name=str(annotations_path),
        manifest_source_name=str(manifest_path),
    )
    manifest_clips = {str(value).strip() for value in manifest["clip_id"]}
    annotated_clips = {str(row["clip_id"]).strip() for row in rows}
    missing_clips = sorted(manifest_clips - annotated_clips)
    if missing_clips:
        raise ValueError(
            "Cannot finalise review before every manifest clip has a retained "
            f"annotation row; missing {missing_clips}"
        )

    path = Path(metadata_path)
    if not path.exists():
        raise ValueError("Review metadata must be started before finalisation")
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != REVIEW_SCHEMA_VERSION:
        raise ValueError("Unsupported annotation review metadata schema")
    if document.get("annotation_file") != _portable_path_identity(annotations_path):
        raise ValueError("Review metadata identifies a different annotation CSV")
    if document.get("review_status") == "complete":
        raise ValueError("Annotation review is already finalised")
    if not document.get("annotator"):
        raise ValueError("Review metadata does not identify the annotator")

    timestamp = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    document.update(
        {
            "review_status": "complete",
            "reviewed_by": reviewer,
            "review_date": timestamp.date().isoformat(),
            "repeat_review_status": repeat_review_status,
            "repeat_reviewed_by": (repeat_reviewer or "").strip() or None,
            "notes": notes.strip(),
            "adjudication_notes": adjudication_notes.strip(),
            "frozen_annotation_sha256": _sha256_file(annotations_path),
            "finalised_at_utc": timestamp.isoformat().replace("+00:00", "Z"),
        }
    )
    _atomic_write_text(path, json.dumps(document, indent=2) + "\n")
    return document


def save_resume_checkpoint(
    checkpoint_path: Path,
    *,
    clip_id: str,
    current_frame: int,
    draft: AnnotationDraft,
) -> None:
    """Atomically retain the current unsaved selections for safe resumption."""
    document = {
        "schema_version": 1,
        "clip_id": clip_id,
        "current_frame": int(current_frame),
        "draft": draft.to_dict(),
    }
    _atomic_write_text(
        Path(checkpoint_path),
        json.dumps(document, indent=2) + "\n",
    )


def load_resume_checkpoint(
    checkpoint_path: Path,
    *,
    clip_id: str,
) -> tuple[int, AnnotationDraft] | None:
    """Restore a matching draft and protect unfinished work for another clip."""
    path = Path(checkpoint_path)
    if not path.exists():
        return None
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1:
        raise ValueError("Unsupported annotation resume checkpoint schema")
    draft = AnnotationDraft.from_dict(document.get("draft", {}))
    saved_clip_id = str(document.get("clip_id", ""))
    if saved_clip_id != clip_id:
        if not draft.is_blank:
            raise ValueError(
                f"An unfinished draft exists for {saved_clip_id!r}; resume it first"
            )
        return None
    return int(document.get("current_frame", 0)), draft


def _manifest_clip_row(manifest: pd.DataFrame, clip_id: str) -> pd.Series:
    matches = manifest[manifest["clip_id"].astype(str).str.strip().eq(clip_id)]
    if len(matches) != 1:
        raise ValueError(f"Manifest must contain exactly one clip {clip_id!r}")
    return matches.iloc[0]


def _resolve_manifest_video_path(video_path: str) -> Path:
    path = (PROJECT_ROOT / video_path).resolve()
    try:
        path.relative_to(PROJECT_ROOT.resolve())
    except ValueError as error:
        raise ValueError("Manifest video path escapes the repository") from error
    if not path.is_file():
        raise FileNotFoundError(f"Manifest video does not exist: {video_path}")
    return path


def _verify_capture_metadata(capture, manifest_row: pd.Series) -> None:
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    expected_fps = float(manifest_row["source_fps"])
    expected = (
        int(manifest_row["frame_count"]),
        int(manifest_row["width_px"]),
        int(manifest_row["height_px"]),
    )
    if (
        not math.isclose(fps, expected_fps, rel_tol=1e-9, abs_tol=1e-9)
        or (frame_count, width, height) != expected
    ):
        raise ValueError(
            "Source video metadata does not match the selected manifest row: "
            f"actual fps/frames/resolution={fps}/{frame_count}/{width}x{height}, "
            f"manifest={expected_fps}/{expected[0]}/{expected[1]}x{expected[2]}"
        )


def _read_frame(capture, frame_index: int, frame_count: int):
    index = min(max(int(frame_index), 0), frame_count - 1)
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    success, frame = capture.read()
    if not success or frame is None:
        raise RuntimeError(f"Could not decode source frame {index}")
    return index, frame


def _window_content_size(
    default_width: int = INITIAL_WINDOW_WIDTH,
    default_height: int = INITIAL_WINDOW_HEIGHT,
) -> tuple[int, int]:
    """Return the current drawable window size or the large initial default."""
    try:
        _, _, width, height = cv2.getWindowImageRect(WINDOW_TITLE)
    except cv2.error:
        return default_width, default_height
    if width <= 0 or height <= 0:
        return default_width, default_height
    return width, height


def _window_closed() -> bool:
    try:
        return cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1.0
    except cv2.error:
        return False


def _friendly_frame(value: int | None) -> str:
    return "—" if value is None else str(value)


def _draft_class(draft: AnnotationDraft) -> str:
    return canonical_class_for_selection(
        insufficient_depth=draft.insufficient_depth_flag,
        incomplete_extension=draft.incomplete_extension_flag,
        alignment_deviation=draft.alignment_deviation_flag,
        ambiguity=draft.ambiguity_flag,
        visibility_status=draft.source_video_visibility_status,
    )


def _annotation_panel_width(display_width: int) -> int:
    """Return a readable panel width independent of source-video resolution."""
    if display_width <= 0:
        raise ValueError("Display width must be positive")
    return min(PANEL_WIDTH, max(MIN_PANEL_WIDTH, round(display_width * 0.32)))


def _annotation_layout(
    *,
    source_width: int,
    source_height: int,
    display_width: int,
    display_height: int,
) -> dict[str, tuple[int, int, int, int]]:
    """Fit the complete source beside a fixed-pixel annotation panel."""
    if min(source_width, source_height, display_width, display_height) <= 0:
        raise ValueError("Source and display dimensions must be positive")
    panel_width = _annotation_panel_width(display_width)
    video_area_width = max(1, display_width - panel_width)
    scale = min(video_area_width / source_width, display_height / source_height)
    scaled_width = max(1, min(video_area_width, round(source_width * scale)))
    scaled_height = max(1, min(display_height, round(source_height * scale)))
    video_x = (video_area_width - scaled_width) // 2
    video_y = (display_height - scaled_height) // 2
    return {
        "video": (video_x, video_y, scaled_width, scaled_height),
        "panel": (video_area_width, 0, panel_width, display_height),
    }


def _annotation_panel_lines(
    *,
    clip_id: str,
    frame_index: int,
    frame_count: int,
    fps: float,
    playing: bool,
    draft: AnnotationDraft,
    saved_rows: int,
    status_message: str,
    wrap_width: int,
) -> list[str]:
    """Build concise viewer text independently of source-frame rendering."""
    elapsed = frame_index / fps
    duration = frame_count / fps
    progress = 100.0 * (frame_index + 1) / frame_count
    flags = []
    if draft.insufficient_depth_flag:
        flags.append("depth")
    if draft.incomplete_extension_flag:
        flags.append("extension")
    if draft.alignment_deviation_flag:
        flags.append("alignment")

    state = "AMBIGUOUS FRAGMENT" if draft.ambiguity_flag else "EVALUABLE ATTEMPT"
    lines = [
        "ANNOTATION CONTROLS",
        f"Clip: {clip_id}",
        f"Frame {frame_index}/{frame_count - 1} ({progress:.1f}%)",
        (
            f"Time {elapsed:.2f}/{duration:.2f}s | "
            f"{'PLAYING' if playing else 'PAUSED'} | Saved: {saved_rows}"
        ),
        (
            f"A start: {_friendly_frame(draft.start_top_frame)} | "
            f"B bottom: {_friendly_frame(draft.bottom_turnaround_frame)}"
        ),
        f"E end/top: {_friendly_frame(draft.completion_end_top_frame)}",
        f"State: {state}",
        f"Class: {FRIENDLY_CLASS_LABELS[_draft_class(draft)]}",
        f"Flags: {', '.join(flags) if flags else 'none'}",
        f"Visibility: {draft.source_video_visibility_status}",
    ]
    lines.extend(
        textwrap.wrap(
            f"Note: {draft.annotator_notes or 'none'}",
            width=wrap_width,
            subsequent_indent="  ",
        )
    )
    lines.extend(
        [
            "",
            "Space play/pause | Q/Esc quit",
            ",/. or arrows: +/-1 | [/] +/-10",
            "A start | B bottom | E end/top",
            "1 correct | 2 depth | 3 extension",
            "4 alignment | 5 unscorable",
            "V visibility | M ambiguous | N note",
            "R reset | S save",
        ]
    )
    lines.extend(
        textwrap.wrap(
            status_message,
            width=wrap_width,
            initial_indent="Status: ",
            subsequent_indent="  ",
        )
    )
    return lines


def _build_annotation_canvas(
    frame,
    *,
    clip_id: str,
    frame_index: int,
    frame_count: int,
    fps: float,
    playing: bool,
    draft: AnnotationDraft,
    saved_rows: int,
    status_message: str,
    display_width: int,
    display_height: int,
):
    """Scale source evidence first, then render a crisp fixed-pixel panel."""
    source_height, source_width = frame.shape[:2]
    layout = _annotation_layout(
        source_width=source_width,
        source_height=source_height,
        display_width=display_width,
        display_height=display_height,
    )
    video_x, video_y, video_width, video_height = layout["video"]
    panel_x, _, panel_width, _ = layout["panel"]
    canvas = np.full(
        (display_height, display_width, 3),
        (15, 18, 20),
        dtype=np.uint8,
    )
    if (video_width, video_height) == (source_width, source_height):
        scaled_frame = frame
    else:
        interpolation = (
            cv2.INTER_AREA
            if video_width < source_width or video_height < source_height
            else cv2.INTER_NEAREST
        )
        scaled_frame = cv2.resize(
            frame,
            (video_width, video_height),
            interpolation=interpolation,
        )
    canvas[
        video_y : video_y + video_height,
        video_x : video_x + video_width,
    ] = scaled_frame
    wrap_width = max(
        28,
        round((panel_width - (2 * PANEL_PADDING)) / (17 * PANEL_FONT_SCALE)),
    )
    lines = _annotation_panel_lines(
        clip_id=clip_id,
        frame_index=frame_index,
        frame_count=frame_count,
        fps=fps,
        playing=playing,
        draft=draft,
        saved_rows=saved_rows,
        status_message=status_message,
        wrap_width=wrap_width,
    )
    available_line_height = max(
        18,
        (display_height - (2 * PANEL_PADDING)) // max(1, len(lines)),
    )
    line_height = min(PANEL_LINE_HEIGHT, available_line_height)
    text_x = panel_x + PANEL_PADDING
    y_position = PANEL_PADDING + line_height
    for line in lines:
        cv2.putText(
            canvas,
            line,
            (text_x, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            PANEL_FONT_SCALE,
            (240, 240, 240),
            2,
            cv2.LINE_AA,
        )
        y_position += line_height
    return canvas


def annotation_key_action(key_code: int) -> tuple[str, int | None]:
    """Map one OpenCV key code to the established annotation action."""
    if key_code < 0:
        return "none", None
    if key_code in {0x250000, 65361}:
        return "move", -1
    if key_code in {0x270000, 65363}:
        return "move", 1

    key = key_code & 0xFF
    if key == 27:
        return "quit", None
    if key == 32:
        return "toggle_play", None

    character = chr(key).lower() if 0 <= key <= 255 else ""
    if character == "q":
        return "quit", None
    if character == ",":
        return "move", -1
    if character == ".":
        return "move", 1
    if character == "[":
        return "move", -10
    if character == "]":
        return "move", 10

    actions = {
        "a": "mark_start",
        "b": "mark_bottom",
        "e": "mark_end",
        "1": "select_correct",
        "2": "toggle_depth",
        "3": "toggle_extension",
        "4": "toggle_alignment",
        "5": "select_unscorable",
        "m": "toggle_ambiguous",
        "v": "cycle_visibility",
        "n": "edit_note",
        "r": "reset",
        "s": "save",
    }
    return actions.get(character, "none"), None


def _cycle_visibility(current: str) -> str:
    statuses = ["sufficient", "partially_obscured", "insufficient"]
    return statuses[(statuses.index(current) + 1) % len(statuses)]


def _draft_from_saved_row(row: dict[str, object]) -> AnnotationDraft:
    """Restore a schema row as an explicitly editable viewer draft."""

    def frame_value(column: str) -> int | None:
        raw_value = str(row[column]).strip()
        return None if not raw_value else int(float(raw_value))

    def boolean_value(column: str) -> bool:
        raw_value = str(row[column]).strip().lower()
        if raw_value not in {"true", "false"}:
            raise ValueError(f"Saved annotation has invalid {column!r}")
        return raw_value == "true"

    return AnnotationDraft(
        start_top_frame=frame_value("start_top_frame"),
        bottom_turnaround_frame=frame_value("bottom_turnaround_frame"),
        completion_end_top_frame=frame_value("completion_end_top_frame"),
        insufficient_depth_flag=boolean_value("insufficient_depth_flag"),
        incomplete_extension_flag=boolean_value("incomplete_extension_flag"),
        alignment_deviation_flag=boolean_value("alignment_deviation_flag"),
        ambiguity_flag=boolean_value("ambiguity_flag"),
        source_video_visibility_status=str(
            row["source_video_visibility_status"]
        ).strip(),
        annotator_notes=str(row["annotator_notes"]).strip(),
    )


def _load_correction_draft(
    rows: list[dict[str, object]],
    *,
    clip_id: str,
    attempt_id: str,
) -> tuple[int, AnnotationDraft]:
    """Load one exact saved identity and choose its first locating frame."""
    matches = [
        row
        for row in rows
        if str(row["clip_id"]).strip() == clip_id
        and str(row["ground_truth_attempt_id"]).strip() == attempt_id
    ]
    if len(matches) != 1:
        raise ValueError(
            "Correction requires exactly one existing annotation identity: "
            f"{(clip_id, attempt_id)}"
        )
    draft = _draft_from_saved_row(matches[0])
    locating_frames = [
        frame
        for frame in (
            draft.start_top_frame,
            draft.bottom_turnaround_frame,
            draft.completion_end_top_frame,
        )
        if frame is not None
    ]
    if not locating_frames:
        raise ValueError("Saved annotation has no locating frame")
    return min(locating_frames), draft


def annotate_clip(
    *,
    manifest: pd.DataFrame,
    annotations_path: Path,
    clip_id: str,
    metadata_path: Path,
    correction_attempt_id: str | None = None,
) -> int:
    """Run the source-only frame viewer and save rows only on explicit request."""
    manifest_row = _manifest_clip_row(manifest, clip_id)
    video_path = _resolve_manifest_video_path(str(manifest_row["video_path"]))
    frame_count = int(manifest_row["frame_count"])
    fps = float(manifest_row["source_fps"])
    rows: list[dict[str, object]] = load_annotation_rows(annotations_path)
    checkpoint_path = resume_checkpoint_path(annotations_path)
    resumed = load_resume_checkpoint(checkpoint_path, clip_id=clip_id)
    correction_mode = correction_attempt_id is not None
    if correction_mode:
        if resumed is not None and not resumed[1].is_blank:
            raise ValueError(
                "Save or reset the unfinished annotation draft before correcting "
                "a saved row"
            )
        current_index, draft = _load_correction_draft(
            rows,
            clip_id=clip_id,
            attempt_id=correction_attempt_id,
        )
    else:
        current_index, draft = resumed or (0, AnnotationDraft())
    current_index = min(max(current_index, 0), frame_count - 1)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open source video: {manifest_row['video_path']}")

    playing = False
    status_message = (
        f"Correcting {clip_id}/{correction_attempt_id}; no predictions displayed"
        if correction_mode
        else "Source video only — no predictions are displayed"
    )
    try:
        _verify_capture_metadata(capture, manifest_row)
        current_index, frame = _read_frame(capture, current_index, frame_count)
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
        cv2.resizeWindow(
            WINDOW_TITLE,
            INITIAL_WINDOW_WIDTH,
            INITIAL_WINDOW_HEIGHT,
        )

        while True:
            saved_for_clip = sum(str(row["clip_id"]).strip() == clip_id for row in rows)
            display_width, display_height = _window_content_size()
            display = _build_annotation_canvas(
                frame,
                clip_id=clip_id,
                frame_index=current_index,
                frame_count=frame_count,
                fps=fps,
                playing=playing,
                draft=draft,
                saved_rows=saved_for_clip,
                status_message=status_message,
                display_width=display_width,
                display_height=display_height,
            )
            cv2.imshow(WINDOW_TITLE, display)
            delay = max(1, round(1000.0 / fps)) if playing else 30
            key_code = cv2.waitKeyEx(delay)

            if _window_closed():
                break
            if key_code < 0:
                if playing:
                    if current_index >= frame_count - 1:
                        playing = False
                        status_message = "Reached the final frame"
                    else:
                        current_index, frame = _read_frame(
                            capture,
                            current_index + 1,
                            frame_count,
                        )
                continue

            action, movement = annotation_key_action(key_code)
            if action == "quit":
                break
            if action == "toggle_play":
                playing = not playing
                status_message = "Playing" if playing else "Paused"
                continue

            if action == "move":
                playing = False
                current_index, frame = _read_frame(
                    capture,
                    current_index + movement,
                    frame_count,
                )
                if not correction_mode:
                    save_resume_checkpoint(
                        checkpoint_path,
                        clip_id=clip_id,
                        current_frame=current_index,
                        draft=draft,
                    )
                continue

            changed = True
            if action == "mark_start":
                draft.start_top_frame = current_index
                status_message = "Start/top frame marked"
            elif action == "mark_bottom":
                draft.bottom_turnaround_frame = current_index
                status_message = "Bottom/turnaround frame marked"
            elif action == "mark_end":
                draft.completion_end_top_frame = current_index
                status_message = "Completion/end-top frame marked"
            elif action == "select_correct":
                draft.ambiguity_flag = False
                if draft.source_video_visibility_status == "insufficient":
                    draft.source_video_visibility_status = "sufficient"
                draft.insufficient_depth_flag = False
                draft.incomplete_extension_flag = False
                draft.alignment_deviation_flag = False
                status_message = "Class selection: meets project criteria"
            elif action in {
                "toggle_depth",
                "toggle_extension",
                "toggle_alignment",
            }:
                draft.ambiguity_flag = False
                if draft.source_video_visibility_status == "insufficient":
                    draft.source_video_visibility_status = "sufficient"
                attribute = {
                    "toggle_depth": "insufficient_depth_flag",
                    "toggle_extension": "incomplete_extension_flag",
                    "toggle_alignment": "alignment_deviation_flag",
                }[action]
                setattr(draft, attribute, not getattr(draft, attribute))
                status_message = "Deviation flag toggled"
            elif action == "select_unscorable":
                draft.ambiguity_flag = False
                draft.source_video_visibility_status = "insufficient"
                draft.insufficient_depth_flag = False
                draft.incomplete_extension_flag = False
                draft.alignment_deviation_flag = False
                status_message = "Class selection: unscorable"
            elif action == "toggle_ambiguous":
                draft.ambiguity_flag = not draft.ambiguity_flag
                if draft.ambiguity_flag:
                    draft.insufficient_depth_flag = False
                    draft.incomplete_extension_flag = False
                    draft.alignment_deviation_flag = False
                status_message = "Ambiguous fragment state toggled"
            elif action == "cycle_visibility":
                draft.source_video_visibility_status = _cycle_visibility(
                    draft.source_video_visibility_status
                )
                if draft.source_video_visibility_status == "insufficient":
                    draft.insufficient_depth_flag = False
                    draft.incomplete_extension_flag = False
                    draft.alignment_deviation_flag = False
                status_message = "Source visibility changed"
            elif action == "edit_note":
                playing = False
                try:
                    draft.annotator_notes = input(
                        "Annotation note (blank clears): "
                    ).strip()
                    status_message = "Annotation note updated"
                except EOFError:
                    status_message = "No console input available; note unchanged"
            elif action == "reset":
                draft = AnnotationDraft()
                status_message = "Unsaved draft reset"
            elif action == "save":
                attempt_id = correction_attempt_id or next_attempt_id(
                    [dict(row) for row in rows],
                    clip_id=clip_id,
                    ambiguous=draft.ambiguity_flag,
                )
                try:
                    row = build_annotation_row(
                        clip_id=clip_id,
                        attempt_id=attempt_id,
                        draft=draft,
                    )
                    if correction_mode:
                        rows = replace_annotation_row(
                            annotations_path,
                            row,
                            manifest,
                            metadata_path=metadata_path,
                        )
                    else:
                        rows = append_annotation_row(
                            annotations_path,
                            row,
                            manifest,
                        )
                except ValueError as error:
                    status_message = f"NOT SAVED: {error}"
                else:
                    if correction_mode:
                        print(f"Corrected annotation {clip_id}/{attempt_id}.")
                        break
                    status_message = f"Saved {clip_id}/{attempt_id}"
                    draft = AnnotationDraft()
            else:
                changed = False

            if changed and not correction_mode:
                save_resume_checkpoint(
                    checkpoint_path,
                    clip_id=clip_id,
                    current_frame=current_index,
                    draft=draft,
                )
    finally:
        if not correction_mode:
            save_resume_checkpoint(
                checkpoint_path,
                clip_id=clip_id,
                current_frame=current_index,
                draft=draft,
            )
        capture.release()
        cv2.destroyAllWindows()

    return sum(str(row["clip_id"]).strip() == clip_id for row in rows)


def parse_arguments(argv=None) -> argparse.Namespace:
    """Parse annotation-viewer or explicit review-finalisation options."""
    parser = argparse.ArgumentParser(
        description=(
            "Review one manifest source video frame by frame and save independent "
            "manual repetition annotations without algorithm predictions."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--clip-id", help="Manifest clip ID to annotate.")
    parser.add_argument("--annotator", help="Anonymised annotator identifier.")
    parser.add_argument(
        "--correct-attempt-id",
        help=(
            "Load and atomically replace one saved attempt/fragment identity "
            "while review remains open."
        ),
    )
    parser.add_argument(
        "--review-metadata",
        type=Path,
        help="Adjacent review JSON path; defaults from --annotations.",
    )
    parser.add_argument(
        "--finalise-review",
        action="store_true",
        help="Validate full clip coverage, hash the CSV and freeze review metadata.",
    )
    parser.add_argument("--reviewer", help="Reviewer identifier for finalisation.")
    parser.add_argument(
        "--repeat-review-status",
        choices=sorted(REPEAT_REVIEW_STATUSES),
        default="not_performed",
    )
    parser.add_argument("--repeat-reviewer")
    parser.add_argument("--review-notes", default="")
    parser.add_argument("--adjudication-notes", default="")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    """Run source annotation or an explicit, separately requested review freeze."""
    args = parse_arguments(argv)
    metadata_path = args.review_metadata or review_metadata_path(args.annotations)

    if args.finalise_review:
        if not args.reviewer:
            raise ValueError("--reviewer is required with --finalise-review")
        document = finalise_review_metadata(
            metadata_path,
            manifest_path=args.manifest,
            annotations_path=args.annotations,
            reviewer=args.reviewer,
            repeat_review_status=args.repeat_review_status,
            repeat_reviewer=args.repeat_reviewer,
            notes=args.review_notes,
            adjudication_notes=args.adjudication_notes,
        )
        print(
            "Annotation review finalised: "
            f"SHA-256 {document['frozen_annotation_sha256']}"
        )
        return 0

    if not args.clip_id:
        raise ValueError("--clip-id is required for annotation")
    if not args.annotator:
        raise ValueError("--annotator is required for annotation")

    manifest = load_manifest(args.manifest)
    _manifest_clip_row(manifest, args.clip_id)
    ensure_annotation_file(args.annotations)
    start_review_metadata(
        metadata_path,
        annotations_path=args.annotations,
        annotator=args.annotator,
    )
    saved_rows = annotate_clip(
        manifest=manifest,
        annotations_path=args.annotations,
        clip_id=args.clip_id,
        metadata_path=metadata_path,
        correction_attempt_id=args.correct_attempt_id,
    )
    print(f"Annotation viewer closed: {saved_rows} saved rows for {args.clip_id}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
