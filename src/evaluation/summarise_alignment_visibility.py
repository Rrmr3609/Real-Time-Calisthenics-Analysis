"""Summarise enhanced alignment availability for development diagnostics.

Inputs are enhanced frame and repetition CSVs containing boolean availability,
phase, side-selection and alignment-coverage fields. The utility writes a
caller-dated UTF-8 diagnostic summary and does not change analysis behaviour.
"""

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


def iso_summary_date(value: str) -> str:
    """Validate a reproducible ISO calendar date for summary rendering."""
    try:
        parsed = date.fromisoformat(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "summary date must use ISO YYYY-MM-DD"
        ) from error

    if parsed.isoformat() != value:
        raise argparse.ArgumentTypeError("summary date must use ISO YYYY-MM-DD")

    return value


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description=("Summarise alignment availability for an enhanced run.")
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
        required=True,
        type=iso_summary_date,
        help="Date written into the summary (YYYY-MM-DD).",
    )

    parser.add_argument(
        "--minimum-alignment-valid-ratio",
        type=float,
        default=0.50,
        help=(
            "Minimum repetition alignment coverage required for "
            "independently scorable alignment evidence."
        ),
    )

    return parser.parse_args(argv)


def boolean_series(
    data: pd.DataFrame,
    column: str,
    source_name: str,
) -> pd.Series:
    values = data[column]

    if values.empty:
        return pd.Series(index=values.index, dtype=bool)

    normalised = values.fillna("").astype(str).str.strip().str.lower()
    invalid_values = sorted(set(normalised) - {"true", "false"})

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
    minimum_alignment_valid_ratio: float = 0.50,
) -> str:
    if not 0.0 <= minimum_alignment_valid_ratio <= 1.0:
        raise ValueError("minimum_alignment_valid_ratio must be between 0 and 1")

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

    selected_elbow_side = frame_data["selected_elbow_side"].fillna("none").astype(str)
    selected_side_available = selected_elbow_side.isin({"left", "right"})

    rescue_opportunity = (
        selected_side_available & ~alignment_valid & opposite_alignment_valid
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
    alignment_invalid_count = int((selected_side_available & ~alignment_valid).sum())
    opposite_valid_count = int(opposite_alignment_valid.sum())
    rescue_count = int(rescue_opportunity.sum())
    side_change_count = int(side_changed.sum())
    direct_switch_count = int(direct_side_switch.sum())

    phase_values = frame_data["phase"].fillna("missing").astype(str)
    phase_lines = []

    for phase in sorted(
        phase_values.unique(),
        key=phase_sort_key,
    ):
        phase_mask = phase_values.eq(phase)
        phase_total = int(phase_mask.sum())
        phase_elbow_count = int((elbow_valid & phase_mask).sum())
        phase_alignment_count = int((alignment_valid & phase_mask).sum())
        phase_rescue_count = int((rescue_opportunity & phase_mask).sum())

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
        f"{mean_alignment_coverage:.3f}" if pd.notna(mean_alignment_coverage) else "nan"
    )
    final_class_unscorable_count = int(
        repetition_data["predicted_class"].astype(str).eq("unscorable").sum()
    )
    alignment_evidence_unscorable_count = int(
        alignment_coverage.lt(minimum_alignment_valid_ratio).sum()
    )

    clip_ids = sorted(
        set(frame_data["clip_id"].dropna().astype(str))
        | set(repetition_data["clip_id"].dropna().astype(str))
    )

    summary = f"""Alignment visibility development diagnostic — {summary_date}

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
Final predicted-class unscorable repetitions: {final_class_unscorable_count}
Alignment-evidence-unscorable repetitions (coverage < {minimum_alignment_valid_ratio:.3f}): {alignment_evidence_unscorable_count}

Definition:
An opposite-side rescue opportunity is a frame where an elbow side is selected,
alignment is invalid on that selected side, and the opposite side has valid
shoulder-hip-ankle visibility under the existing visibility threshold.

The final predicted-class count follows classifier priority. The independent
alignment-evidence count ignores the final label and reports every repetition
whose alignment coverage is below the stated minimum valid ratio.

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

    if not frame_path.is_file():
        raise FileNotFoundError(f"Enhanced frame CSV does not exist: {frame_path}")

    if not repetition_path.is_file():
        raise FileNotFoundError(
            f"Enhanced repetition CSV does not exist: {repetition_path}"
        )

    frame_data = pd.read_csv(frame_path)
    repetition_data = pd.read_csv(repetition_path)

    summary = build_summary(
        frame_data=frame_data,
        repetition_data=repetition_data,
        summary_date=args.summary_date,
        frame_source=str(frame_path),
        repetition_source=str(repetition_path),
        minimum_alignment_valid_ratio=(args.minimum_alignment_valid_ratio),
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
