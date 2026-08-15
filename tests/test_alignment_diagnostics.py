import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd
import pytest

from evaluation import (
    summarise_alignment_visibility,
    summarise_phase_detection,
    summarise_preprocessing,
    summarise_repetition_classification,
)
from evaluation.summarise_alignment_visibility import build_summary

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_MODULES = (
    summarise_preprocessing,
    summarise_phase_detection,
    summarise_repetition_classification,
    summarise_alignment_visibility,
)


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
    assert "Alignment-valid frames on elbow-selected side: 2 (0.400)" in summary
    assert "Opposite-side rescue opportunities: 1 (0.500" in summary
    assert "Selected elbow-side change frames: 2" in summary
    assert "Direct left/right elbow-side switches: 1" in summary
    assert "- descending: frames=1" in summary
    assert "Mean repetition alignment coverage: 0.500" in summary
    assert "Final predicted-class unscorable repetitions: 1" in summary
    assert "Alignment-evidence-unscorable repetitions (coverage < 0.500): 1" in summary
    assert summary.startswith(
        "Alignment visibility development diagnostic — 2026-07-28"
    )


@pytest.mark.parametrize("module", SUMMARY_MODULES)
def test_summary_date_accepts_strict_iso_date(module):
    assert module.iso_summary_date("2026-08-11") == "2026-08-11"


@pytest.mark.parametrize("module", SUMMARY_MODULES)
@pytest.mark.parametrize(
    "invalid_date",
    ["11 August 2026", "20260811", "2026-02-30"],
)
def test_summary_date_rejects_invalid_values(module, invalid_date):
    with pytest.raises(
        argparse.ArgumentTypeError,
        match="ISO YYYY-MM-DD",
    ):
        module.iso_summary_date(invalid_date)


def test_preprocessing_summary_uses_supplied_identity_and_date(
    tmp_path,
    monkeypatch,
):
    input_path = tmp_path / "frames.csv"
    output_path = tmp_path / "preprocessing.txt"
    pd.DataFrame(
        {
            "pose_detected": [True, True],
            "elbow_feature_valid": [True, True],
            "alignment_feature_valid": [True, False],
            "selected_side": ["left", "left"],
            "raw_elbow_angle": [160.0, 150.0],
            "smoothed_elbow_angle": [160.0, 155.0],
            "processing_time_ms": [10.0, 12.0],
        }
    ).to_csv(input_path, index=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarise_preprocessing.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--summary-date",
            "2026-08-11",
            "--development-id",
            "development-run-001",
        ],
    )

    summarise_preprocessing.main()

    summary = output_path.read_text(encoding="utf-8")
    assert "development diagnostic — 2026-08-11" in summary
    assert "Development ID: development-run-001" in summary
    assert " ".join(("21", "July", "2026")) not in summary


def test_phase_summary_uses_supplied_identity_and_date(
    tmp_path,
    monkeypatch,
):
    input_path = tmp_path / "temporal.csv"
    output_path = tmp_path / "phase.txt"
    pd.DataFrame(
        {
            "enhanced_rep_count": [0, 0],
            "completed_rep": [False, False],
            "phase": ["waiting", "top"],
            "elbow_feature_valid": [False, True],
            "processing_time_ms": [10.0, 12.0],
        }
    ).to_csv(input_path, index=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarise_phase_detection.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--summary-date",
            "2026-08-11",
            "--development-id",
            "development-run-002",
        ],
    )

    summarise_phase_detection.main()

    summary = output_path.read_text(encoding="utf-8")
    assert "development diagnostic — 2026-08-11" in summary
    assert "Development ID: development-run-002" in summary
    assert " ".join(("22", "July", "2026")) not in summary
    assert "not human ground truth" in summary


def test_classification_summary_uses_supplied_identity_and_date(
    tmp_path,
    monkeypatch,
):
    input_path = tmp_path / "repetitions.csv"
    output_path = tmp_path / "classification.txt"
    pd.DataFrame(
        columns=[
            "clip_id",
            "rep_id",
            "predicted_class",
            "alignment_valid_ratio",
            "multiple_rules_triggered",
            "minimum_elbow_angle",
            "top_extension_angle",
            "triggered_rules",
        ]
    ).to_csv(input_path, index=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarise_repetition_classification.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--summary-date",
            "2026-08-11",
            "--development-id",
            "development-run-003",
        ],
    )

    summarise_repetition_classification.main()

    summary = output_path.read_text(encoding="utf-8")
    assert "development diagnostic — 2026-08-11" in summary
    assert "Development ID: development-run-003" in summary
    assert " ".join(("23", "July", "2026")) not in summary


def test_historical_summary_content_is_preserved_across_line_endings():
    summaries = PROJECT_ROOT / "results" / "development" / "summaries"
    expected_sha256 = {
        "2026-07-21_enhanced_preprocessing_summary.txt": (
            "670c3eca75906904cca13a7d955fd1d5e56d8dd37d2f85e3eab24570e54e7103"
        ),
        "2026-07-22_phase_detection_summary.txt": (
            "c39e8e47b18a1ccf0bed8dbc50274e9e9b641452e51f5523470f171d58551c94"
        ),
        "2026-07-23_repetition_classification_summary.txt": (
            "d10430fb0475430a5c6629339399391fda3351e6cffca4e92b281c852d0809d5"
        ),
    }

    for filename, expected_hash in expected_sha256.items():
        content = (summaries / filename).read_bytes()
        # Preserve historical text while tolerating Git/platform newline conversion.
        canonical_content = content.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        assert hashlib.sha256(canonical_content).hexdigest() == expected_hash


def test_alignment_summary_separates_evidence_from_final_class():
    repetitions = make_repetition_data()
    repetitions.loc[0, "predicted_class"] = "incomplete_extension"

    summary = build_summary(
        frame_data=make_frame_data(),
        repetition_data=repetitions,
        summary_date="2026-07-28",
        frame_source="frames.csv",
        repetition_source="repetitions.csv",
        minimum_alignment_valid_ratio=0.50,
    )

    assert "Final predicted-class unscorable repetitions: 0" in summary
    assert "Alignment-evidence-unscorable repetitions (coverage < 0.500): 1" in summary


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
            "--summary-date",
            "2026-08-11",
            "--development-id",
            "duplicate-repetition-test",
        ],
    )

    with pytest.raises(
        ValueError,
        match=r"duplicate \(clip_id, rep_id\)",
    ):
        summarise_repetition_classification.main()

    assert not output_path.exists()
