"""Summarise enhanced temporal outputs for development diagnostics.

The input is an enhanced temporal CSV containing frame identities, phases,
angles in degrees and processing times in milliseconds. The utility writes a
caller-dated UTF-8 text summary, not human ground truth or formal evidence.
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
        raise argparse.ArgumentTypeError(
            "summary date must use ISO YYYY-MM-DD"
        )

    return value


def development_id(value: str) -> str:
    """Require a non-blank caller-supplied development identity."""
    normalised = value.strip()

    if not normalised:
        raise argparse.ArgumentTypeError(
            "development ID cannot be blank"
        )

    return normalised


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Summarise enhanced phase-detection behaviour."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Enhanced temporal CSV.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output text file.",
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


def main():
    args = parse_arguments()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Enhanced temporal CSV does not exist: {input_path}"
        )

    data = pd.read_csv(input_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_frames = len(data)

    final_rep_count = int(
        data["enhanced_rep_count"].iloc[-1]
    )

    completed_rows = data[
        data["completed_rep"].astype(str).str.lower()
        == "true"
    ]

    phase_change_count = int(
        data["phase"].astype(str)
        .ne(data["phase"].astype(str).shift())
        .sum()
        - 1
    )

    phase_counts = (
        data["phase"]
        .value_counts(dropna=False)
        .to_dict()
    )

    elbow_valid_rate = (
        data["elbow_feature_valid"].mean()
    )

    mean_processing_time = (
        data["processing_time_ms"].mean()
    )

    median_processing_time = (
        data["processing_time_ms"].median()
    )

    completed_lines = []

    for _, row in completed_rows.iterrows():
        completed_lines.append(
            (
                f"Rep {int(row['completed_rep_id'])}: "
                f"frames {int(row['completed_start_frame'])}"
                f"-{int(row['completed_end_frame'])}, "
                f"bottom frame "
                f"{int(row['completed_bottom_frame'])}, "
                f"minimum angle "
                f"{float(row['completed_minimum_elbow_angle']):.2f}, "
                f"ending angle "
                f"{float(row['completed_end_top_angle']):.2f}"
            )
        )

    if not completed_lines:
        completed_lines.append(
            "No completed repetitions were detected."
        )

    summary = f"""Enhanced phase-detection development diagnostic — {args.summary_date}

Development ID: {args.development_id}
Input: {input_path}

Total frames: {total_frames}
Final enhanced repetition count: {final_rep_count}
Completed-repetition event rows: {len(completed_rows)}
Detected phase changes: {phase_change_count}
Elbow-feature valid rate: {elbow_valid_rate:.3f}
Mean processing time per frame: {mean_processing_time:.3f} ms
Median processing time per frame: {median_processing_time:.3f} ms

Frames by phase:
{phase_counts}

Completed repetition details:
{chr(10).join(completed_lines)}

Important:
This is a development diagnostic, not a formal accuracy result.
Completed-repetition boundaries are algorithm outputs, not human ground truth.
"""

    print(summary)

    output_path.write_text(
        summary,
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
