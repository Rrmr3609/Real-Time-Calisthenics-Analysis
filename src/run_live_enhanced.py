"""Run the enhanced push-up pipeline with live, non-medical feedback."""

from __future__ import annotations

import argparse
import math
import sys
import textwrap
import time
from contextlib import ExitStack
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import cv2

from analysis.enhanced_features import EnhancedFeatureProcessor
from analysis.phase_state_machine import PushUpPhaseStateMachine
from analysis.repetition_aggregator import RepetitionFeatureAggregator
from analysis.repetition_classifier import (
    RepetitionClassification,
    RepetitionClassifier,
)
from analysis.return_top_finalizer import ReturnTopPeakFinalizer
from capture.webcam import WebcamCapture
from config.runtime import apply_cli_overrides, load_runtime_config
from pose.estimator import PoseEstimator
from pose.landmarks import extract_landmarks
from utils.paths import OUTPUT_DIR, PROJECT_ROOT
from utils.run_provenance import sha256_file

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"
LIVE_SESSION_OUTPUT_DIR = OUTPUT_DIR / "live_sessions"
WINDOW_TITLE = "Real-Time Calisthenics Analysis - Enhanced"
FEEDBACK_DURATION_SECONDS = 2.5
MAX_PRESENTATION_WIDTH = 1280
MAX_PRESENTATION_HEIGHT = 800

PRIMARY_FEEDBACK_MESSAGES = {
    "correct": "Rep meets the project form criteria.",
    "insufficient_depth": "Try lowering further before returning to the top.",
    "incomplete_extension": "Extend the elbows more fully at the top.",
    "alignment_deviation": "Keep the shoulders, hips and ankles more aligned.",
    "unscorable": "Improve body visibility and remain roughly side-on.",
}

CATEGORY_LABELS = {
    "correct": "FORM CRITERIA MET",
    "insufficient_depth": "DEPTH",
    "incomplete_extension": "EXTENSION",
    "alignment_deviation": "BODY ALIGNMENT",
    "unscorable": "NOT SCORED",
}

REPORT_CLASS_LABELS = {
    "correct": "Meets project criteria",
    "insufficient_depth": "Insufficient depth",
    "incomplete_extension": "Incomplete extension",
    "alignment_deviation": "Body alignment deviation",
    "unscorable": "Not scored",
}

DEVIATION_DESCRIPTIONS = {
    "insufficient_depth": "Insufficient depth",
    "incomplete_extension": "incomplete elbow extension",
    "alignment_deviation": "body alignment deviation",
}

FEEDBACK_TONES = {
    "correct": "positive",
    "insufficient_depth": "attention",
    "incomplete_extension": "attention",
    "alignment_deviation": "attention",
    "unscorable": "neutral",
}

PHASE_LABELS = {
    "waiting": "Getting ready",
    "top": "Ready at the top",
    "descending": "Lowering",
    "bottom": "At the bottom",
    "ascending": "Returning to top",
}

TONE_COLOURS = {
    "positive": (90, 220, 90),
    "attention": (0, 190, 255),
    "neutral": (220, 210, 175),
}


@dataclass(frozen=True)
class FeedbackContent:
    """Presentation text derived from one completed-repetition result."""

    headline: str
    primary: str
    secondary: tuple[str, ...]
    tone: str


@dataclass(frozen=True)
class LiveRepetitionResult:
    """Small user-facing record of one completed classified repetition."""

    rep_id: int
    predicted_class: str
    category: str
    feedback: str
    additional_observations: tuple[str, ...]
    alignment_unassessed: bool
    tone: str


@dataclass
class LiveSessionResults:
    """Completed repetition results retained for the summary and text report."""

    started_at_utc: datetime
    repetitions: list[LiveRepetitionResult] = field(default_factory=list)

    def record(
        self,
        classification: RepetitionClassification,
        *,
        minimum_alignment_valid_ratio: float,
    ) -> LiveRepetitionResult:
        """Store one classifier result without recreating classification logic."""
        result = build_live_repetition_result(
            classification,
            minimum_alignment_valid_ratio=minimum_alignment_valid_ratio,
        )
        self.repetitions.append(result)
        return result

    @property
    def total_repetitions(self) -> int:
        """Return the number of completed repetitions in this session."""
        return len(self.repetitions)

    def class_counts(self) -> dict[str, int]:
        """Return all supported class counts, including zero-valued classes."""
        counts = dict.fromkeys(CATEGORY_LABELS, 0)
        for repetition in self.repetitions:
            counts[repetition.predicted_class] += 1
        return counts


@dataclass(frozen=True)
class SummaryPage:
    """One viewport-sized page of completed live repetitions."""

    index: int
    page_count: int
    rows_per_page: int
    first_position: int
    last_position: int
    repetitions: tuple[LiveRepetitionResult, ...]


class TimedFeedbackPresenter:
    """Retain the latest completed-repetition message for a fixed interval."""

    def __init__(self, duration_seconds: float = FEEDBACK_DURATION_SECONDS):
        duration = float(duration_seconds)
        if not math.isfinite(duration) or duration <= 0.0:
            raise ValueError("Feedback duration must be a positive finite number")

        self.duration_seconds = duration
        self._content: FeedbackContent | None = None
        self._expires_at = 0.0

    def record(
        self,
        content: FeedbackContent,
        *,
        now: float,
    ) -> FeedbackContent:
        """Replace any prior message with the latest completed repetition."""
        self._content = content
        self._expires_at = float(now) + self.duration_seconds
        return self._content

    def current(self, now: float) -> FeedbackContent | None:
        """Return active feedback, clearing it once the interval expires."""
        if self._content is None:
            return None

        if float(now) >= self._expires_at:
            self._content = None
            return None

        return self._content


def parse_arguments(argv=None):
    """Parse the small enhanced-live camera and configuration interface."""
    parser = argparse.ArgumentParser(
        description="Run live enhanced push-up feedback from a webcam."
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="OpenCV camera device index (default: 0).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Runtime YAML configuration (default: configs/default.yaml).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Explicitly override features.ema_alpha.",
    )
    return parser.parse_args(argv)


def build_completed_feedback(
    classification: RepetitionClassification,
    *,
    minimum_alignment_valid_ratio: float,
) -> FeedbackContent:
    """Translate classification evidence into readable, non-medical text."""
    result = build_live_repetition_result(
        classification,
        minimum_alignment_valid_ratio=minimum_alignment_valid_ratio,
    )
    return feedback_for_repetition(result)


def build_live_repetition_result(
    classification: RepetitionClassification,
    *,
    minimum_alignment_valid_ratio: float,
) -> LiveRepetitionResult:
    """Convert classifier evidence into the minimal live-session record."""
    predicted_class = classification.predicted_class

    try:
        category = CATEGORY_LABELS[predicted_class]
        feedback = PRIMARY_FEEDBACK_MESSAGES[predicted_class]
        tone = FEEDBACK_TONES[predicted_class]
    except KeyError as error:
        raise ValueError(
            f"Unsupported repetition class: {predicted_class!r}"
        ) from error

    additional_rules = [
        rule for rule in classification.triggered_rules if rule != predicted_class
    ]

    try:
        additional_observations = tuple(
            DEVIATION_DESCRIPTIONS[rule] for rule in additional_rules
        )
    except KeyError as error:
        raise ValueError(f"Unsupported triggered rule: {error.args[0]!r}") from error

    alignment_unassessed = classification.alignment_valid_ratio < float(
        minimum_alignment_valid_ratio
    )

    return LiveRepetitionResult(
        rep_id=classification.rep_id,
        predicted_class=predicted_class,
        category=category,
        feedback=feedback,
        additional_observations=additional_observations,
        alignment_unassessed=alignment_unassessed,
        tone=tone,
    )


def feedback_for_repetition(repetition: LiveRepetitionResult) -> FeedbackContent:
    """Build prominent and secondary live text from a stored result."""
    secondary = []
    if repetition.additional_observations:
        secondary.append(
            "Also observed: " + ", ".join(repetition.additional_observations)
        )
    if repetition.alignment_unassessed:
        secondary.append("Alignment not assessed: insufficient body visibility")

    return FeedbackContent(
        headline=f"REP {repetition.rep_id} — {repetition.category}",
        primary=repetition.feedback,
        secondary=tuple(secondary),
        tone=repetition.tone,
    )


def neutral_guidance(
    *,
    phase: str,
    selected_side: str,
    elbow_feature_valid: bool,
) -> str:
    """Return positioning or movement guidance without judging form."""
    if selected_side == "none" or not elbow_feature_valid:
        return "Move fully into view and remain roughly side-on"

    if phase in {"waiting", "top"}:
        return "Ready — begin a push-up"

    if phase == "descending":
        return "Tracking repetition..."

    if phase in {"bottom", "ascending"}:
        return "Return to the top to complete the repetition"

    return "Tracking movement..."


def exit_key_requested(key_code: int, *, allow_enter: bool = False) -> bool:
    """Return whether an OpenCV key code requests a clean screen exit."""
    if key_code < 0:
        return False

    key = key_code & 0xFF
    exit_keys = {ord("q"), ord("Q"), 27}
    if allow_enter:
        exit_keys.update({10, 13})
    return key in exit_keys


def window_close_requested(window_title: str = WINDOW_TITLE) -> bool:
    """Detect a user-closed OpenCV window where the active backend supports it."""
    try:
        return cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1.0
    except cv2.error:
        return False


def presentation_window_size(
    frame_width: int,
    frame_height: int,
    *,
    maximum_width: int = MAX_PRESENTATION_WIDTH,
    maximum_height: int = MAX_PRESENTATION_HEIGHT,
) -> tuple[int, int]:
    """Return an aspect-preserving initial size without upscaling the frame."""
    dimensions = (frame_width, frame_height, maximum_width, maximum_height)
    if any(int(value) <= 0 for value in dimensions):
        raise ValueError("Presentation dimensions must be positive integers")

    scale = min(
        1.0,
        maximum_width / frame_width,
        maximum_height / frame_height,
    )
    return (
        max(1, round(frame_width * scale)),
        max(1, round(frame_height * scale)),
    )


def configure_presentation_window(frame) -> tuple[int, int]:
    """Create a resizable aspect-preserving window sized from the source frame."""
    frame_height, frame_width = frame.shape[:2]
    window_width, window_height = presentation_window_size(
        frame_width,
        frame_height,
    )
    cv2.namedWindow(
        WINDOW_TITLE,
        cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO,
    )
    cv2.resizeWindow(WINDOW_TITLE, window_width, window_height)
    return window_width, window_height


def tracking_status(*, selected_side: str, elbow_feature_valid: bool) -> str:
    """Describe tracking availability without showing internal side identifiers."""
    if selected_side == "none" or not elbow_feature_valid:
        return "Tracking: Finding body position"
    return "Tracking: Body position found"


def build_session_summary_lines(session: LiveSessionResults) -> list[str]:
    """Return human-readable summary lines independent of OpenCV rendering."""
    counts = session.class_counts()
    total = session.total_repetitions
    correct = counts["correct"]

    if total:
        criteria_line = (
            "Repetitions meeting project criteria: "
            f"{correct}/{total} ({(100.0 * correct / total):.0f}%)"
        )
    else:
        criteria_line = "Repetitions meeting project criteria: 0"

    lines = [
        "SESSION SUMMARY",
        f"Completed repetitions: {total}",
        criteria_line,
        f"Insufficient depth: {counts['insufficient_depth']}",
        f"Incomplete extension: {counts['incomplete_extension']}",
        f"Body alignment deviation: {counts['alignment_deviation']}",
        f"Not scored: {counts['unscorable']}",
    ]

    if session.repetitions:
        lines.append("")
        lines.append("PER-REPETITION SUMMARY")
        lines.extend(
            f"Rep {repetition.rep_id}: {repetition.category}"
            for repetition in session.repetitions
        )
    else:
        lines.append("No completed repetitions were detected.")

    return lines


def summary_rows_per_page(viewport_height: int) -> int:
    """Calculate repetition rows that fit between summary totals and footer."""
    if viewport_height <= 0:
        raise ValueError("Summary viewport height must be positive")

    ui_scale = max(0.65, min(1.0, viewport_height / 650.0))
    margin = max(16, round(34 * ui_scale))
    line_height = max(20, round(28 * ui_scale))
    compact_line_height = max(16, round(22 * ui_scale))

    y_position = margin + round(18 * ui_scale)
    y_position += line_height + round(8 * ui_scale)
    y_position += 6 * line_height
    y_position += line_height
    y_position += line_height + round(8 * ui_scale)
    y_position += compact_line_height

    repetitions_bottom = viewport_height - margin - round(48 * ui_scale)
    available_height = max(0, repetitions_bottom - y_position)
    return max(1, available_height // line_height)


def build_summary_page(
    session: LiveSessionResults,
    *,
    viewport_height: int,
    requested_index: int,
) -> SummaryPage:
    """Return a clamped deterministic page for the current summary viewport."""
    rows_per_page = summary_rows_per_page(viewport_height)
    total = session.total_repetitions
    page_count = max(1, math.ceil(total / rows_per_page))
    page_index = min(max(int(requested_index), 0), page_count - 1)
    start = page_index * rows_per_page
    end = min(start + rows_per_page, total)

    return SummaryPage(
        index=page_index,
        page_count=page_count,
        rows_per_page=rows_per_page,
        first_position=start + 1 if total else 0,
        last_position=end,
        repetitions=tuple(session.repetitions[start:end]),
    )


def summary_navigation_delta(key_code: int) -> int:
    """Map common OpenCV extended navigation key codes to a page direction."""
    previous_page_keys = {
        0x210000,  # PageUp on Windows
        0x260000,  # Up on Windows
        65362,  # Up on common Unix backends
        65365,  # PageUp on common Unix backends
    }
    next_page_keys = {
        0x220000,  # PageDown on Windows
        0x280000,  # Down on Windows
        65364,  # Down on common Unix backends
        65366,  # PageDown on common Unix backends
    }
    if key_code in previous_page_keys:
        return -1
    if key_code in next_page_keys:
        return 1
    return 0


def navigate_summary_page(
    current_index: int,
    *,
    page_count: int,
    key_code: int,
) -> int:
    """Apply one navigation key while keeping the summary page in range."""
    if page_count < 1:
        raise ValueError("Summary page count must be at least one")
    return min(
        max(current_index + summary_navigation_delta(key_code), 0),
        page_count - 1,
    )


def configuration_identity(config_path: Path) -> str:
    """Return a repository-relative or basename-only configuration identity."""
    resolved = Path(config_path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.name


def render_live_session_report(
    session: LiveSessionResults,
    *,
    config_identity: str,
    config_sha256: str,
) -> str:
    """Render a local human-readable report, not formal evaluation evidence."""
    started_at = session.started_at_utc.astimezone(timezone.utc)
    timestamp = started_at.isoformat().replace("+00:00", "Z")
    counts = session.class_counts()
    total = session.total_repetitions
    correct = counts["correct"]

    criteria_value = (
        f"{correct}/{total} ({(100.0 * correct / total):.1f}%)" if total else "0"
    )
    lines = [
        "Enhanced Live Push-Up Session Report",
        "====================================",
        "This local session summary is not formal evaluation evidence.",
        "",
        f"Session timestamp (UTC): {timestamp}",
        f"Configuration: {config_identity}",
        f"Configuration SHA-256: {config_sha256}",
        "",
        "Session totals",
        "--------------",
        f"Completed repetitions: {total}",
        f"Repetitions meeting project criteria: {criteria_value}",
        f"Insufficient depth: {counts['insufficient_depth']}",
        f"Incomplete extension: {counts['incomplete_extension']}",
        f"Body alignment deviation: {counts['alignment_deviation']}",
        f"Not scored: {counts['unscorable']}",
    ]

    if not session.repetitions:
        lines.extend(["", "No completed repetitions were detected."])
        return "\n".join(lines) + "\n"

    lines.extend(["", "Completed repetitions", "---------------------"])
    for repetition in session.repetitions:
        lines.extend(
            [
                (
                    f"Rep {repetition.rep_id} — "
                    f"{REPORT_CLASS_LABELS[repetition.predicted_class]}"
                ),
                f"  Feedback: {repetition.feedback}",
            ]
        )
        if repetition.additional_observations:
            lines.append(
                "  Additional observations: "
                + ", ".join(repetition.additional_observations)
            )
        if repetition.alignment_unassessed:
            lines.append(
                "  Alignment evidence: Not assessed because body visibility "
                "was insufficient."
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def save_live_session_report(
    session: LiveSessionResults,
    *,
    config_identity: str,
    config_sha256: str,
    output_dir: Path = LIVE_SESSION_OUTPUT_DIR,
) -> Path:
    """Atomically create a collision-safe local text report for the session."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    timestamp = session.started_at_utc.astimezone(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    report = render_live_session_report(
        session,
        config_identity=config_identity,
        config_sha256=config_sha256,
    )

    sequence = 1
    while True:
        suffix = "" if sequence == 1 else f"_{sequence:02d}"
        report_path = destination / f"enhanced_live_{timestamp}{suffix}.txt"
        try:
            with report_path.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(report)
            return report_path
        except FileExistsError:
            sequence += 1


def _put_text(
    frame,
    text: str,
    position: tuple[int, int],
    *,
    scale: float,
    colour: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        colour,
        thickness,
        cv2.LINE_AA,
    )


def draw_live_overlay(
    frame,
    *,
    repetition_count: int,
    phase: str,
    selected_side: str,
    elbow_feature_valid: bool,
    fps: float,
    feedback_headline: str,
    primary_message: str,
    secondary_messages: tuple[str, ...],
    tone: str,
    completed_feedback_active: bool,
) -> None:
    """Draw a compact responsive card without exposing diagnostic labels."""
    image_height, image_width = frame.shape[:2]
    ui_scale = max(
        0.68,
        min(1.0, image_width / 1050.0, image_height / 650.0),
    )
    margin = max(8, round(14 * ui_scale))
    padding = max(10, round(16 * ui_scale))
    panel_width = min(
        image_width - (2 * margin),
        round(520 * ui_scale),
        max(round(image_width * 0.38), round(350 * ui_scale)),
    )
    panel_height = min(
        image_height - (2 * margin),
        round(300 * ui_scale),
    )
    panel_left = margin
    panel_top = margin
    panel_right = panel_left + panel_width
    panel_bottom = panel_top + panel_height
    text_left = panel_left + padding

    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (panel_left, panel_top),
        (panel_right, panel_bottom),
        (18, 22, 25),
        -1,
    )
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0.0, frame)

    white = (245, 245, 245)
    secondary = (195, 200, 205)
    muted = (160, 168, 174)
    tone_colour = TONE_COLOURS[tone]
    phase_label = PHASE_LABELS.get(phase, "Tracking")
    status = tracking_status(
        selected_side=selected_side,
        elbow_feature_valid=elbow_feature_valid,
    )

    _put_text(
        frame,
        "ENHANCED PUSH-UP ANALYSIS",
        (text_left, panel_top + round(25 * ui_scale)),
        scale=0.48 * ui_scale,
        colour=secondary,
        thickness=1,
    )
    _put_text(
        frame,
        str(repetition_count),
        (text_left, panel_top + round(72 * ui_scale)),
        scale=1.38 * ui_scale,
        colour=white,
        thickness=max(2, round(3 * ui_scale)),
    )
    _put_text(
        frame,
        "COMPLETED REPETITIONS",
        (text_left + round(55 * ui_scale), panel_top + round(67 * ui_scale)),
        scale=0.42 * ui_scale,
        colour=secondary,
        thickness=1,
    )

    headline = feedback_headline if completed_feedback_active else "LIVE GUIDANCE"
    _put_text(
        frame,
        headline,
        (text_left, panel_top + round(106 * ui_scale)),
        scale=0.58 * ui_scale,
        colour=tone_colour,
    )

    wrap_width = max(24, round((panel_width - (2 * padding)) / (7 * ui_scale)))
    y_position = panel_top + round(132 * ui_scale)
    status_top = panel_bottom - round(72 * ui_scale)
    line_height = max(16, round(22 * ui_scale))
    for line in textwrap.wrap(primary_message, width=wrap_width) or [""]:
        if y_position >= status_top:
            break
        _put_text(
            frame,
            line,
            (text_left, y_position),
            scale=0.52 * ui_scale,
            colour=white,
            thickness=1,
        )
        y_position += line_height

    for message in secondary_messages:
        for line in textwrap.wrap(message, width=wrap_width + 8):
            if y_position >= status_top:
                break
            _put_text(
                frame,
                line,
                (text_left, y_position),
                scale=0.42 * ui_scale,
                colour=secondary,
                thickness=1,
            )
            y_position += max(14, round(18 * ui_scale))

    _put_text(
        frame,
        f"Movement: {phase_label}",
        (text_left, panel_bottom - round(52 * ui_scale)),
        scale=0.45 * ui_scale,
        colour=secondary,
        thickness=1,
    )
    _put_text(
        frame,
        status,
        (text_left, panel_bottom - round(31 * ui_scale)),
        scale=0.43 * ui_scale,
        colour=secondary,
        thickness=1,
    )
    _put_text(
        frame,
        f"{fps:.1f} FPS    Q: finish session    Esc: exit",
        (text_left, panel_bottom - round(11 * ui_scale)),
        scale=0.34 * ui_scale,
        colour=muted,
        thickness=1,
    )


def draw_session_summary(
    frame,
    *,
    session: LiveSessionResults,
    report_name: str,
    page_index: int = 0,
) -> SummaryPage:
    """Draw one responsive end-of-session summary page and return its bounds."""
    frame[:] = (24, 28, 31)
    image_height, image_width = frame.shape[:2]
    ui_scale = max(0.65, min(1.0, image_width / 1000.0, image_height / 650.0))
    margin = max(16, round(34 * ui_scale))
    white = (245, 245, 245)
    secondary = (190, 198, 204)
    accent = TONE_COLOURS["positive"]

    lines = build_session_summary_lines(session)[:7]
    y_position = margin + round(18 * ui_scale)
    line_height = max(20, round(28 * ui_scale))

    for index, line in enumerate(lines):
        is_heading = line == "SESSION SUMMARY"
        is_total = index == 1
        _put_text(
            frame,
            line,
            (margin, y_position),
            scale=(0.78 if is_heading else 0.58 if is_total else 0.48) * ui_scale,
            colour=accent if is_heading else white if is_total else secondary,
            thickness=2 if is_heading or is_total else 1,
        )
        y_position += line_height + (round(8 * ui_scale) if is_heading else 0)

    page = build_summary_page(
        session,
        viewport_height=image_height,
        requested_index=page_index,
    )

    if session.repetitions:
        y_position += line_height
        _put_text(
            frame,
            "PER-REPETITION SUMMARY",
            (margin, y_position),
            scale=0.78 * ui_scale,
            colour=accent,
            thickness=2,
        )
        y_position += line_height + round(8 * ui_scale)

        if page.page_count > 1:
            _put_text(
                frame,
                (
                    f"Showing reps {page.first_position}-{page.last_position} of "
                    f"{session.total_repetitions} — Page {page.index + 1}/"
                    f"{page.page_count}"
                ),
                (margin, y_position),
                scale=0.42 * ui_scale,
                colour=secondary,
                thickness=1,
            )
            y_position += max(16, round(22 * ui_scale))

        for repetition in page.repetitions:
            _put_text(
                frame,
                f"Rep {repetition.rep_id}: {repetition.category}",
                (margin, y_position),
                scale=0.48 * ui_scale,
                colour=secondary,
                thickness=1,
            )
            y_position += line_height
    else:
        y_position += line_height
        _put_text(
            frame,
            "No completed repetitions were detected.",
            (margin, y_position),
            scale=0.48 * ui_scale,
            colour=secondary,
            thickness=1,
        )

    _put_text(
        frame,
        f"Saved locally: {report_name}",
        (margin, image_height - margin - round(20 * ui_scale)),
        scale=0.38 * ui_scale,
        colour=secondary,
        thickness=1,
    )
    _put_text(
        frame,
        (
            "Up / Down / PageUp / PageDown: pages    Q / Esc / Enter: close"
            if page.page_count > 1
            else "Q / Esc / Enter: close summary"
        ),
        (margin, image_height - margin),
        scale=0.38 * ui_scale,
        colour=secondary,
        thickness=1,
    )
    return page


def show_session_summary(
    last_frame,
    *,
    session: LiveSessionResults,
    report_path: Path,
) -> None:
    """Display a navigable summary until a supported close event."""
    configure_presentation_window(last_frame)
    page_index = 0

    while True:
        summary_frame = last_frame.copy()
        page = draw_session_summary(
            summary_frame,
            session=session,
            report_name=report_path.name,
            page_index=page_index,
        )
        cv2.imshow(WINDOW_TITLE, summary_frame)

        key_code = cv2.waitKeyEx(30)
        if exit_key_requested(key_code, allow_enter=True):
            return
        if window_close_requested():
            return
        page_index = navigate_summary_page(
            page.index,
            page_count=page.page_count,
            key_code=key_code,
        )


def _is_expected_camera_unavailable(
    error: RuntimeError,
    *,
    camera_index: int,
) -> bool:
    """Recognise the explicit availability error raised by ``WebcamCapture``."""
    return str(error) == f"Could not open camera with device index {camera_index}"


def main() -> int:
    """Run enhanced webcam analysis while safely owning live resources."""
    args = parse_arguments()
    loaded_config = load_runtime_config(args.config)
    config, _ = apply_cli_overrides(
        loaded_config,
        ema_alpha=args.alpha,
    )

    feature_processor = EnhancedFeatureProcessor(
        smoothing_alpha=config.features.ema_alpha,
        minimum_visibility=config.features.minimum_landmark_visibility,
        acquisition_frames=config.features.side_acquisition_frames,
        switch_frames=config.features.side_switch_frames,
        switch_margin=config.features.side_switch_margin,
        missing_grace_frames=config.features.missing_side_grace_frames,
    )
    phase_machine = PushUpPhaseStateMachine(
        top_region_threshold=config.segmentation.top_region_threshold,
        bottom_region_threshold=config.segmentation.bottom_region_threshold,
        hysteresis=config.segmentation.hysteresis,
        confirmation_frames=config.segmentation.phase_confirmation_frames,
        missing_grace_frames=config.segmentation.missing_angle_grace_frames,
        minimum_rep_frames=config.segmentation.minimum_repetition_frames,
    )
    repetition_aggregator = RepetitionFeatureAggregator()
    return_top_finalizer = ReturnTopPeakFinalizer()
    repetition_classifier = RepetitionClassifier(
        depth_threshold=config.classification.depth_threshold,
        extension_threshold=config.classification.extension_threshold,
        alignment_minimum=config.classification.alignment_minimum,
        alignment_deviation_min_frames=(
            config.classification.alignment_deviation_min_frames
        ),
        alignment_deviation_min_ratio=(
            config.classification.alignment_deviation_min_ratio
        ),
        minimum_alignment_valid_ratio=(
            config.classification.minimum_alignment_valid_ratio
        ),
    )
    feedback_presenter = TimedFeedbackPresenter()
    config_identity = configuration_identity(args.config)
    config_sha256 = sha256_file(Path(args.config))

    with ExitStack() as window_cleanup:
        window_cleanup.callback(cv2.destroyAllWindows)

        with ExitStack() as live_cleanup:
            camera = WebcamCapture(
                device_index=args.camera_index,
                width=1280,
                height=720,
            )
            live_cleanup.callback(camera.release)
            try:
                camera.open()
            except RuntimeError as error:
                if not _is_expected_camera_unavailable(
                    error,
                    camera_index=args.camera_index,
                ):
                    raise
                print(
                    f"Unable to open camera {args.camera_index}. Check that a "
                    "webcam is connected and not being used by another application.",
                    file=sys.stderr,
                )
                return 1

            pose_estimator = PoseEstimator(
                min_detection_confidence=(config.pose.minimum_detection_confidence),
                min_tracking_confidence=config.pose.minimum_tracking_confidence,
            )
            live_cleanup.callback(pose_estimator.close)

            initial_frame = camera.read()
            if initial_frame is None:
                raise RuntimeError("Failed to read a frame from the webcam")
            configure_presentation_window(initial_frame)
            print(
                "Enhanced live feedback started. Focus the video window and press "
                "Q or Esc to finish."
            )

            session = LiveSessionResults(started_at_utc=datetime.now(timezone.utc))
            frame_index = -1
            previous_frame_time = time.perf_counter()
            last_frame = None
            exit_reason = ""

            while True:
                frame = initial_frame
                initial_frame = None
                if frame is None:
                    frame = camera.read()
                if frame is None:
                    raise RuntimeError("Failed to read a frame from the webcam")

                frame_index += 1
                image_height, image_width = frame.shape[:2]
                results = pose_estimator.process(frame)
                pose_detected = bool(results.pose_landmarks)
                landmarks = {}

                if pose_detected:
                    landmarks = extract_landmarks(
                        results,
                        image_width,
                        image_height,
                    )

                feature_result = feature_processor.update(landmarks)
                phase_result = phase_machine.update(
                    elbow_angle=feature_result["smoothed_elbow_angle"],
                    frame_index=frame_index,
                )
                detected_repetition = repetition_aggregator.update(
                    frame_index=frame_index,
                    repetition_window_start_frame=(
                        phase_result["repetition_window_start_frame"]
                    ),
                    body_alignment_angle=(feature_result["smoothed_alignment_angle"]),
                    completed_repetition=phase_result["completed_repetition"],
                )
                completed_repetition = return_top_finalizer.update(
                    detected_repetition=detected_repetition,
                    elbow_angle=feature_result["smoothed_elbow_angle"],
                    returned_top_phase_active=(phase_result["phase"] == "top"),
                )

                now = time.perf_counter()
                elapsed = now - previous_frame_time
                fps = 1.0 / elapsed if elapsed > 0.0 else 0.0
                previous_frame_time = now

                if completed_repetition is not None:
                    classification = repetition_classifier.classify(
                        completed_repetition
                    )
                    repetition_result = session.record(
                        classification,
                        minimum_alignment_valid_ratio=(
                            config.classification.minimum_alignment_valid_ratio
                        ),
                    )
                    feedback_presenter.record(
                        feedback_for_repetition(repetition_result),
                        now=now,
                    )

                active_feedback = feedback_presenter.current(now)
                if active_feedback is None:
                    feedback_headline = "LIVE GUIDANCE"
                    primary_message = neutral_guidance(
                        phase=phase_result["phase"],
                        selected_side=feature_result["selected_elbow_side"],
                        elbow_feature_valid=feature_result["elbow_feature_valid"],
                    )
                    secondary_messages = ()
                    tone = "neutral"
                else:
                    feedback_headline = active_feedback.headline
                    primary_message = active_feedback.primary
                    secondary_messages = active_feedback.secondary
                    tone = active_feedback.tone

                pose_estimator.draw_landmarks(frame, results)
                draw_live_overlay(
                    frame,
                    repetition_count=phase_result["rep_count"],
                    phase=phase_result["phase"],
                    selected_side=feature_result["selected_elbow_side"],
                    elbow_feature_valid=feature_result["elbow_feature_valid"],
                    fps=fps,
                    feedback_headline=feedback_headline,
                    primary_message=primary_message,
                    secondary_messages=secondary_messages,
                    tone=tone,
                    completed_feedback_active=active_feedback is not None,
                )

                last_frame = frame
                cv2.imshow(WINDOW_TITLE, frame)
                key_code = cv2.waitKeyEx(15)
                if exit_key_requested(key_code):
                    exit_reason = "key"
                    break
                if window_close_requested():
                    exit_reason = "window_closed"
                    break

            completed_repetition = return_top_finalizer.flush()
            if completed_repetition is not None:
                classification = repetition_classifier.classify(completed_repetition)
                session.record(
                    classification,
                    minimum_alignment_valid_ratio=(
                        config.classification.minimum_alignment_valid_ratio
                    ),
                )

        report_path = save_live_session_report(
            session,
            config_identity=config_identity,
            config_sha256=config_sha256,
        )
        try:
            report_location = report_path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            report_location = report_path.name
        print(f"Session report saved to {report_location}")

        if exit_reason != "window_closed":
            show_session_summary(
                last_frame,
                session=session,
                report_path=report_path,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
