import argparse
from pathlib import Path

import pandas as pd


def parse_arguments():
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

    return parser.parse_args()


def main():
    args = parse_arguments()

    input_path = Path(args.input)
    output_path = Path(args.output)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = pd.read_csv(input_path)

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

    summary = f"""Enhanced phase-detection smoke test — 22 July 2026

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
This is an engineering smoke test on one setup recording.
It is not a formal accuracy evaluation and the temporal
parameters have not yet been calibrated on a development set.
"""

    print(summary)

    output_path.write_text(
        summary,
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()