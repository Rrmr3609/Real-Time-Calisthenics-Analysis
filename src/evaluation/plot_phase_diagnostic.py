import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Plot elbow angles and detected phase transitions."
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
        help="Output image path.",
    )

    parser.add_argument(
        "--top-threshold",
        type=float,
        default=130.0,
    )

    parser.add_argument(
        "--bottom-threshold",
        type=float,
        default=120.0,
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

    frame_index = data["frame_index"]

    raw_angle = pd.to_numeric(
        data["raw_elbow_angle"],
        errors="coerce",
    )

    smoothed_angle = pd.to_numeric(
        data["smoothed_elbow_angle"],
        errors="coerce",
    )

    plt.figure(figsize=(12, 6))

    plt.plot(
        frame_index,
        raw_angle,
        label="Raw elbow angle",
        linewidth=1.0,
    )

    plt.plot(
        frame_index,
        smoothed_angle,
        label="Smoothed elbow angle",
        linewidth=2.0,
    )

    plt.axhline(
        args.top_threshold,
        linestyle="--",
        label="Segmentation top region",
    )

    plt.axhline(
        args.bottom_threshold,
        linestyle=":",
        label="Segmentation bottom region",
    )

    phase_changes = data[
        data["phase"].astype(str)
        != data["phase"].astype(str).shift()
    ]

    for _, row in phase_changes.iterrows():
        frame = int(row["frame_index"])
        phase = str(row["phase"])

        plt.axvline(
            frame,
            linewidth=0.7,
            alpha=0.5,
        )

        plt.text(
            frame,
            0.98,
            phase,
            rotation=90,
            verticalalignment="top",
            transform=plt.gca().get_xaxis_transform(),
            fontsize=7,
        )

    completed_rows = data[
        data["completed_rep"].astype(str).str.lower()
        == "true"
    ]

    for _, row in completed_rows.iterrows():
        frame = int(row["frame_index"])
        rep_id = row["completed_rep_id"]

        plt.scatter(
            [frame],
            [row["smoothed_elbow_angle"]],
            label=(
                f"Completed rep {rep_id}"
                if len(completed_rows) == 1
                else None
            ),
        )

    plt.xlabel("Frame index")
    plt.ylabel("Elbow angle (degrees)")
    plt.title(
        "Enhanced temporal push-up phase detection"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved phase plot to: {output_path}")


if __name__ == "__main__":
    main()