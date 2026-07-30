import json
import sys

import pandas as pd

from evaluation.detection_evaluation import main


def test_detection_cli_reports_only_detection_metrics(
    tmp_path,
    monkeypatch,
    capsys,
):
    manifest_path = tmp_path / "manifest.csv"
    annotations_path = tmp_path / "annotations.csv"
    predictions_path = tmp_path / "predictions.csv"

    pd.DataFrame(
        [
            {
                "clip_id": "fictional-clip",
                "split": "development",
                "video_path": (
                    "data/raw/fictional/clip.mp4"
                ),
                "participant_id": "P_FICTIONAL",
                "camera_view": "side",
                "source_fps": 20.0,
                "frame_count": 100,
                "width_px": 1280,
                "height_px": 720,
                "recording_condition": (
                    "controlled_fictional"
                ),
                "notes": "",
            }
        ]
    ).to_csv(manifest_path, index=False)
    pd.DataFrame(
        [
            {
                "clip_id": "fictional-clip",
                "ground_truth_attempt_id": "A001",
                "is_evaluable_attempt": True,
                "ambiguity_flag": False,
                "start_top_frame": 10,
                "bottom_turnaround_frame": 15,
                "completion_end_top_frame": 20,
                "ground_truth_class": "correct",
                "insufficient_depth_flag": False,
                "incomplete_extension_flag": False,
                "alignment_deviation_flag": False,
                "source_video_visibility_status": (
                    "sufficient"
                ),
                "annotator_notes": "",
            }
        ]
    ).to_csv(annotations_path, index=False)
    pd.DataFrame(
        [
            {
                "run_id": "fictional-run",
                "clip_id": "fictional-clip",
                "rep_id": 1,
                "start_frame": 10,
                "bottom_frame": 15,
                "end_frame": 20,
                "predicted_class": "correct",
            }
        ]
    ).to_csv(predictions_path, index=False)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "detection_evaluation.py",
            "--method",
            "enhanced",
            "--predictions",
            str(predictions_path),
            "--manifest",
            str(manifest_path),
            "--annotations",
            str(annotations_path),
            "--clip-id",
            "fictional-clip",
        ],
    )

    main()

    report = json.loads(capsys.readouterr().out)
    summary = report["summary"]

    assert summary["ground_truth_event_count"] == 1
    assert summary["predicted_event_count"] == 1
    assert summary["matched_events"] == 1
    assert summary["event_f1"] == 1.0
    assert "classification_accuracy" not in summary
    assert "confusion_matrix" not in report
