import argparse
from pathlib import Path

import pandas as pd


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Summarise enhanced preprocessing behaviour."
        )
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

    return parser.parse_args()


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
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(input_path)

    total_frames = len(data)

    pose_rate = data["pose_detected"].mean()
    elbow_valid_rate = data["elbow_feature_valid"].mean()
    alignment_valid_rate = (
        data["alignment_feature_valid"].mean()
    )

    no_side_rate = (
        data["selected_side"].fillna("none") == "none"
    ).mean()

    side_changes = count_state_changes(
        data["selected_side"]
    )

    raw_elbow_change = mean_absolute_frame_change(
        data["raw_elbow_angle"]
    )

    smoothed_elbow_change = mean_absolute_frame_change(
        data["smoothed_elbow_angle"]
    )

    mean_processing_time = (
        data["processing_time_ms"].mean()
    )

    median_processing_time = (
        data["processing_time_ms"].median()
    )

    summary = f"""Enhanced preprocessing smoke test — 21 July 2026

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
This is an engineering smoke test, not a formal evaluation result.
The enhanced method does not yet include push-up phase detection,
repetition counting or repetition-level form classification.
"""

    print(summary)
    output_path.write_text(
        summary,
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()