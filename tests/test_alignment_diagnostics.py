import sys

import pandas as pd
import pytest

from evaluation.summarise_alignment_visibility import build_summary
from evaluation import summarise_repetition_classification


def make_frame_data():
    return pd.DataFrame(
        {
            "clip_id": ["clip"] * 5,
            "phase": [
                "waiting",
                "top",
                "descending",
                "bottom",
                "ascending",
            ],
            "selected_elbow_side": [
                "none",
                "left",
                "left",
                "left",
                "right",
            ],
            "side_changed": [
                False,
                True,
                False,
                False,
                True,
            ],
            "elbow_feature_valid": [
                False,
                True,
                True,
                True,
                True,
            ],
            "alignment_feature_valid": [
                False,
                True,
                False,
                False,
                True,
            ],
            "opposite_alignment_feature_valid": [
                False,
                True,
                True,
                False,
                False,
            ],
        }
    )


def make_repetition_data():
    return pd.DataFrame(
        {
            "clip_id": ["clip", "clip"],
            "rep_id": [1, 2],
            "alignment_valid_ratio": [0.25, 0.75],
            "predicted_class": ["unscorable", "correct"],
        }
    )


def test_alignment_summary_reports_requested_metrics():
    summary = build_summary(
        frame_data=make_frame_data(),
        repetition_data=make_repetition_data(),
        summary_date="2026-07-28",
        frame_source="frames.csv",
        repetition_source="repetitions.csv",
    )

    assert "Elbow-valid frames: 4 (0.800)" in summary
    assert (
        "Alignment-valid frames on elbow-selected side: 2 (0.400)"
        in summary
    )
    assert "Opposite-side rescue opportunities: 1 (0.500" in summary
    assert "Selected elbow-side change frames: 2" in summary
    assert "Direct left/right elbow-side switches: 1" in summary
    assert "- descending: frames=1" in summary
    assert "Mean repetition alignment coverage: 0.500" in summary
    assert "Unscorable repetitions: 1" in summary


def test_alignment_summary_rejects_duplicate_repetitions():
    repetitions = make_repetition_data()
    repetitions = pd.concat(
        [repetitions, repetitions.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match=r"duplicate \(clip_id, rep_id\)",
    ):
        build_summary(
            frame_data=make_frame_data(),
            repetition_data=repetitions,
            summary_date="2026-07-28",
            frame_source="frames.csv",
            repetition_source="repetitions.csv",
        )


def test_classification_summary_rejects_duplicate_repetitions(
    tmp_path,
    monkeypatch,
):
    input_path = tmp_path / "repetitions.csv"
    output_path = tmp_path / "summary.txt"

    repetitions = pd.DataFrame(
        {
            "clip_id": ["clip", "clip"],
            "rep_id": [1, 1],
            "minimum_elbow_angle": [90.0, 90.0],
            "top_extension_angle": [155.0, 155.0],
            "alignment_valid_ratio": [0.5, 0.5],
            "multiple_rules_triggered": [False, False],
            "triggered_rules": ["", ""],
            "predicted_class": ["correct", "correct"],
        }
    )
    repetitions.to_csv(input_path, index=False)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarise_repetition_classification.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
    )

    with pytest.raises(
        ValueError,
        match=r"duplicate \(clip_id, rep_id\)",
    ):
        summarise_repetition_classification.main()

    assert not output_path.exists()
