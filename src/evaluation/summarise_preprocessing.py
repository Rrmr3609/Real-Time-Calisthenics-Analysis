"""Summarise enhanced frame preprocessing for development diagnostics.

The input is an enhanced frame CSV containing angles in degrees and processing
times in milliseconds. The utility writes a caller-dated UTF-8 text summary;
it does not produce formal evaluation evidence.
"""

import argparse
from datetime import date
from pathlib import Path

import pandas as pd


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


def development_id(value: str) -> str:
    """Require a non-blank caller-supplied development identity."""
    normalised = value.strip()

    if not normalised:
        raise argparse.ArgumentTypeError("development ID cannot be blank")

    return normalised


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description=("Summarise enhanced preprocessing behaviour.")
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Enhanced feature CSV.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Text summary path.",
    )
    parser.add_argument(
        "--summary-date",
        required=True,
        type=iso_summary_date,
        help="Date rendered in the summary (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--development-id",
        required=True,
        type=development_id,
        help="Caller-supplied development clip or run identity.",
    )

    return parser.parse_args(argv)


def count_state_changes(series: pd.Series) -> int:
    values = series.fillna("none").astype(str)

    if values.empty:
        return 0

    return int(values.ne(values.shift()).sum() - 1)


def mean_absolute_frame_change(series: pd.Series) -> float:
    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    return float(values.diff().abs().mean())


def main():
    args = parse_arguments()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.is_file():
        raise FileNotFoundError(f"Enhanced feature CSV does not exist: {input_path}")

    data = pd.read_csv(input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_frames = len(data)

    pose_rate = data["pose_detected"].mean()
    elbow_valid_rate = data["elbow_feature_valid"].mean()
    alignment_valid_rate = data["alignment_feature_valid"].mean()

    no_side_rate = (data["selected_side"].fillna("none") == "none").mean()

    side_changes = count_state_changes(data["selected_side"])

    raw_elbow_change = mean_absolute_frame_change(data["raw_elbow_angle"])

    smoothed_elbow_change = mean_absolute_frame_change(data["smoothed_elbow_angle"])

    mean_processing_time = data["processing_time_ms"].mean()

    median_processing_time = data["processing_time_ms"].median()

    summary = f"""Enhanced preprocessing development diagnostic — {args.summary_date}

Development ID: {args.development_id}
Input: {input_path}

Total frames: {total_frames}
Pose-detected frame rate: {pose_rate:.3f}
Elbow-feature valid rate: {elbow_valid_rate:.3f}
Alignment-feature valid rate: {alignment_valid_rate:.3f}
No-selected-side frame rate: {no_side_rate:.3f}
Selected-side state changes: {side_changes}
Mean absolute raw elbow frame change: {raw_elbow_change:.3f} degrees
Mean absolute smoothed elbow frame change: {smoothed_elbow_change:.3f} degrees
Mean processing time per frame: {mean_processing_time:.3f} ms
Median processing time per frame: {median_processing_time:.3f} ms

Important:
This is a development diagnostic, not a formal evaluation result.
Values describe only the supplied enhanced frame-level CSV.
"""

    print(summary)
    output_path.write_text(
        summary,
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
