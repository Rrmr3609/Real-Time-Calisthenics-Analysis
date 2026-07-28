import argparse
from pathlib import Path

import pandas as pd

from evaluation.validation import reject_duplicate_repetitions


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Summarise repetition-level classification outputs."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Enhanced repetition-level CSV.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output text summary.",
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
    reject_duplicate_repetitions(
        data,
        source_name=str(input_path),
    )

    repetition_count = len(data)

    if data.empty:
        class_counts = {}
        mean_alignment_coverage = float("nan")
        multiple_rule_count = 0
    else:
        class_counts = (
            data["predicted_class"]
            .value_counts(dropna=False)
            .to_dict()
        )

        mean_alignment_coverage = (
            pd.to_numeric(
                data["alignment_valid_ratio"],
                errors="coerce",
            ).mean()
        )

        multiple_rule_count = int(
            data["multiple_rules_triggered"]
            .astype(str)
            .str.lower()
            .eq("true")
            .sum()
        )

    repetition_lines = []

    for _, row in data.iterrows():
        repetition_lines.append(
            (
                f"Rep {int(row['rep_id'])}: "
                f"class={row['predicted_class']}, "
                f"minimum elbow="
                f"{float(row['minimum_elbow_angle']):.2f}, "
                f"top extension="
                f"{float(row['top_extension_angle']):.2f}, "
                f"alignment coverage="
                f"{float(row['alignment_valid_ratio']):.3f}, "
                f"rules={row['triggered_rules']}"
            )
        )

    if not repetition_lines:
        repetition_lines.append(
            "No completed repetitions were classified."
        )

    summary = f"""Enhanced repetition-classification smoke test — 23 July 2026

Completed repetitions classified: {repetition_count}
Predicted class counts: {class_counts}
Mean alignment valid ratio: {mean_alignment_coverage:.3f}
Repetitions with multiple triggered rules: {multiple_rule_count}

Repetition details:
{chr(10).join(repetition_lines)}

Important:
This is an engineering smoke test using one existing setup video.
It is not a formal classification evaluation. The thresholds have not
yet been calibrated using the planned development recordings.
"""

    print(summary)

    output_path.write_text(
        summary,
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
