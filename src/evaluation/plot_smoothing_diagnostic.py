import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Plot raw and smoothed elbow-angle traces."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Enhanced feature CSV.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output image path.",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

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

    plt.figure(figsize=(10, 5))
    plt.plot(
        frame_index,
        raw_angle,
        label="Raw elbow angle",
        linewidth=1.0,
    )
    plt.plot(
        frame_index,
        smoothed_angle,
        label="EMA-smoothed elbow angle",
        linewidth=2.0,
    )

    plt.xlabel("Frame index")
    plt.ylabel("Elbow angle (degrees)")
    plt.title("Raw and smoothed elbow angles")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved diagnostic plot to: {output_path}")


if __name__ == "__main__":
    main()