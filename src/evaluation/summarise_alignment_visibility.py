import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from evaluation.validation import (
    reject_duplicate_repetitions,
    require_columns,
)


FRAME_COLUMNS = (
    "clip_id",
    "phase",
    "selected_elbow_side",
    "side_changed",
    "elbow_feature_valid",
    "alignment_feature_valid",
    "opposite_alignment_feature_valid",
)

REPETITION_COLUMNS = (
    "clip_id",
    "rep_id",
    "alignment_valid_ratio",
    "predicted_class",
)

PHASE_ORDER = (
    "waiting",
    "top",
    "descending",
    "bottom",
    "ascending",
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Summarise alignment availability for an enhanced run."
        )
    )

    parser.add_argument(
        "--frame-input",
        required=True,
        help="Enhanced frame-level CSV.",
    )
    parser.add_argument(
        "--repetition-input",
        required=True,
        help="Enhanced repetition-level CSV.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output text summary.",
    )
    parser.add_argument(
        "--summary-date",
        default=date.today().isoformat(),
        help="Date written into the summary (YYYY-MM-DD).",
    )

    return parser.parse_args()


def boolean_series(
    data: pd.DataFrame,
    column: str,
    source_name: str,
) -> pd.Series:
    values = data[column]

    if values.empty:
        return pd.Series(index=values.index, dtype=bool)

    normalised = (
        values
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    invalid_values = sorted(
        set(normalised) - {"true", "false"}
    )

    if invalid_values:
        raise ValueError(
            f"{source_name} column {column!r} contains invalid "
            f"boolean values: {invalid_values}"
        )

    return normalised.eq("true")


def rate_text(count: int, total: int) -> str:
    if total == 0:
        return "nan"

    return f"{count / total:.3f}"


def phase_sort_key(phase: str) -> tuple[int, str]:
    if phase in PHASE_ORDER:
        return PHASE_ORDER.index(phase), phase

    return len(PHASE_ORDER), phase


def build_summary(
    frame_data: pd.DataFrame,
    repetition_data: pd.DataFrame,
    summary_date: str,
    frame_source: str,
    repetition_source: str,
) -> str:
    require_columns(
        frame_data,
        FRAME_COLUMNS,
        frame_source,
    )
    require_columns(
        repetition_data,
        REPETITION_COLUMNS,
        repetition_source,
    )
    reject_duplicate_repetitions(
        repetition_data,
        source_name=repetition_source,
    )

    elbow_valid = boolean_series(
        frame_data,
        "elbow_feature_valid",
        frame_source,
    )
    alignment_valid = boolean_series(
        frame_data,
        "alignment_feature_valid",
        frame_source,
    )
    opposite_alignment_valid = boolean_series(
        frame_data,
        "opposite_alignment_feature_valid",
        frame_source,
    )
    side_changed = boolean_series(
        frame_data,
        "side_changed",
        frame_source,
    )

    selected_elbow_side = (
        frame_data["selected_elbow_side"]
        .fillna("none")
        .astype(str)
    )
    selected_side_available = selected_elbow_side.isin(
        {"left", "right"}
    )

    rescue_opportunity = (
        selected_side_available
        & ~alignment_valid
        & opposite_alignment_valid
    )

    previous_side = selected_elbow_side.shift()
    direct_side_switch = (
        selected_elbow_side.isin({"left", "right"})
        & previous_side.isin({"left", "right"})
        & selected_elbow_side.ne(previous_side)
    )

    total_frames = len(frame_data)
    elbow_valid_count = int(elbow_valid.sum())
    alignment_valid_count = int(alignment_valid.sum())
    alignment_invalid_count = int(
        (selected_side_available & ~alignment_valid).sum()
    )
    opposite_valid_count = int(opposite_alignment_valid.sum())
    rescue_count = int(rescue_opportunity.sum())
    side_change_count = int(side_changed.sum())
    direct_switch_count = int(direct_side_switch.sum())

    phase_values = (
        frame_data["phase"]
        .fillna("missing")
        .astype(str)
    )
    phase_lines = []

    for phase in sorted(
        phase_values.unique(),
        key=phase_sort_key,
    ):
        phase_mask = phase_values.eq(phase)
        phase_total = int(phase_mask.sum())
        phase_elbow_count = int(
            (elbow_valid & phase_mask).sum()
        )
        phase_alignment_count = int(
            (alignment_valid & phase_mask).sum()
        )
        phase_rescue_count = int(
            (rescue_opportunity & phase_mask).sum()
        )

        phase_lines.append(
            (
                f"- {phase}: frames={phase_total}, "
                f"elbow_valid_rate="
                f"{rate_text(phase_elbow_count, phase_total)}, "
                f"alignment_valid_rate="
                f"{rate_text(phase_alignment_count, phase_total)}, "
                f"opposite_side_rescue_frames="
                f"{phase_rescue_count}"
            )
        )

    alignment_coverage = pd.to_numeric(
        repetition_data["alignment_valid_ratio"],
        errors="coerce",
    )
    mean_alignment_coverage = alignment_coverage.mean()
    mean_coverage_text = (
        f"{mean_alignment_coverage:.3f}"
        if pd.notna(mean_alignment_coverage)
        else "nan"
    )
    unscorable_count = int(
        repetition_data["predicted_class"]
        .astype(str)
        .eq("unscorable")
        .sum()
    )

    clip_ids = sorted(
        set(
            frame_data["clip_id"]
            .dropna()
            .astype(str)
        )
        | set(
            repetition_data["clip_id"]
            .dropna()
            .astype(str)
        )
    )

    summary = f"""Alignment visibility diagnostic — {summary_date}

Clip IDs: {clip_ids}
Frame input: {frame_source}
Repetition input: {repetition_source}

Total frames: {total_frames}
Elbow-valid frames: {elbow_valid_count} ({rate_text(elbow_valid_count, total_frames)})
Alignment-valid frames on elbow-selected side: {alignment_valid_count} ({rate_text(alignment_valid_count, total_frames)})
Opposite-side alignment-valid frames: {opposite_valid_count} ({rate_text(opposite_valid_count, total_frames)})
Selected-side alignment-invalid frames: {alignment_invalid_count}
Opposite-side rescue opportunities: {rescue_count} ({rate_text(rescue_count, alignment_invalid_count)} of selected-side alignment-invalid frames)
Selected elbow-side change frames: {side_change_count}
Direct left/right elbow-side switches: {direct_switch_count}

Phase-grouped availability:
{chr(10).join(phase_lines)}

Repetition summary:
Completed repetitions: {len(repetition_data)}
Mean repetition alignment coverage: {mean_coverage_text}
Unscorable repetitions: {unscorable_count}

Definition:
An opposite-side rescue opportunity is a frame where an elbow side is selected,
alignment is invalid on that selected side, and the opposite side has valid
shoulder-hip-ankle visibility under the existing visibility threshold.

Important:
This is a diagnostic on development data, not a formal evaluation result.
No side-selection behaviour or classifier threshold is changed by this report.
"""

    return summary


def main():
    args = parse_arguments()

    frame_path = Path(args.frame_input)
    repetition_path = Path(args.repetition_input)
    output_path = Path(args.output)

    frame_data = pd.read_csv(frame_path)
    repetition_data = pd.read_csv(repetition_path)

    summary = build_summary(
        frame_data=frame_data,
        repetition_data=repetition_data,
        summary_date=args.summary_date,
        frame_source=str(frame_path),
        repetition_source=str(repetition_path),
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        summary,
        encoding="utf-8",
    )

    print(summary)


if __name__ == "__main__":
    main()
