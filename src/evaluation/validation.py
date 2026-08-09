"""Provide shared structural validation for evaluation data tables."""

from typing import Iterable

import pandas as pd


REPETITION_KEY_COLUMNS = ("clip_id", "rep_id")


def require_columns(
    data: pd.DataFrame,
    columns: Iterable[str],
    source_name: str,
) -> None:
    """Require named columns before any row-level validation is attempted."""
    missing_columns = [
        column
        for column in columns
        if column not in data.columns
    ]

    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(
            f"{source_name} is missing required columns: "
            f"{missing_text}"
        )


def reject_duplicate_repetitions(
    data: pd.DataFrame,
    source_name: str = "Repetition data",
) -> None:
    """Reject duplicate ``(clip_id, rep_id)`` rows before summarisation."""
    require_columns(
        data,
        REPETITION_KEY_COLUMNS,
        source_name,
    )

    duplicate_mask = data.duplicated(
        subset=list(REPETITION_KEY_COLUMNS),
        keep=False,
    )

    if not duplicate_mask.any():
        return

    duplicate_keys = (
        data.loc[duplicate_mask, list(REPETITION_KEY_COLUMNS)]
        .drop_duplicates()
        .to_dict(orient="records")
    )

    formatted_keys = ", ".join(
        (
            f"(clip_id={key['clip_id']!r}, "
            f"rep_id={key['rep_id']!r})"
        )
        for key in duplicate_keys
    )

    raise ValueError(
        f"{source_name} contains duplicate (clip_id, rep_id) rows: "
        f"{formatted_keys}"
    )
