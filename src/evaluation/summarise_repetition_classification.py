"""Summarise enhanced repetition predictions for development diagnostics.

The input is an enhanced repetition CSV containing angles in degrees, coverage
ratios and predicted classes. The utility writes a caller-dated UTF-8 text
summary; it does not calculate formal classification metrics.
"""

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from evaluation.validation import reject_duplicate_repetitions


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
        description=("Summarise repetition-level classification outputs.")
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
        raise FileNotFoundError(f"Enhanced repetition CSV does not exist: {input_path}")

    data = pd.read_csv(input_path)
    reject_duplicate_repetitions(
        data,
        source_name=str(input_path),
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    repetition_count = len(data)

    if data.empty:
        class_counts = {}
        mean_alignment_coverage = float("nan")
        multiple_rule_count = 0
    else:
        class_counts = data["predicted_class"].value_counts(dropna=False).to_dict()

        mean_alignment_coverage = pd.to_numeric(
            data["alignment_valid_ratio"],
            errors="coerce",
        ).mean()

        multiple_rule_count = int(
            data["multiple_rules_triggered"].astype(str).str.lower().eq("true").sum()
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
        repetition_lines.append("No completed repetitions were classified.")

    summary = f"""Enhanced repetition-classification development diagnostic — {args.summary_date}

Development ID: {args.development_id}
Input: {input_path}

Completed repetitions classified: {repetition_count}
Predicted class counts: {class_counts}
Mean alignment valid ratio: {mean_alignment_coverage:.3f}
Repetitions with multiple triggered rules: {multiple_rule_count}

Repetition details:
{chr(10).join(repetition_lines)}

Important:
This is a development diagnostic, not a formal classification result.
Predicted classes describe only the supplied enhanced repetition CSV.
"""

    print(summary)

    output_path.write_text(
        summary,
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
