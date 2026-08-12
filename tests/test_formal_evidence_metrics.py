import csv

import pandas as pd

from evaluation.formal_evidence_metrics import (
    ClipEvidenceMetrics,
    aggregate_formal_evidence,
    human_alignment_evidence_metrics,
    load_enhanced_repetition_evidence_metrics,
    load_frame_evidence_metrics,
)


def write_csv(path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def frame_rows(run_id, clip_id, timings, *, method):
    rows = []
    for index, timing in enumerate(timings):
        row = {
            "run_id": run_id,
            "clip_id": clip_id,
            "frame_index": index,
            "processing_time_ms": timing,
            "pose_detected": index != 1,
            "selected_side": ("left" if index != 1 else "right"),
        }
        if method == "baseline":
            row.update(
                elbow_angle=("" if index == 1 else 150.0),
                body_alignment_angle=170.0,
            )
        else:
            row.update(
                elbow_feature_valid=index != 1,
                alignment_feature_valid=index == 0,
                side_changed=index == 1,
            )
        rows.append(row)
    return rows


def load_frames(tmp_path, clip_id, timings, *, method):
    rows = frame_rows(f"{method}-run", clip_id, timings, method=method)
    path = tmp_path / f"{method}-{clip_id}.csv"
    write_csv(path, tuple(rows[0]), rows)
    return load_frame_evidence_metrics(
        path,
        method=method,
        expected_run_id=f"{method}-run",
        expected_clip_id=clip_id,
        expected_frame_count=len(rows),
    )


def repetition_metrics(tmp_path, clip_id, classes, coverage):
    rows = [
        {
            "predicted_class": class_label,
            "alignment_valid_ratio": ratio,
        }
        for class_label, ratio in zip(classes, coverage)
    ]
    path = tmp_path / f"repetitions-{clip_id}.csv"
    write_csv(path, tuple(rows[0]), rows)
    return load_enhanced_repetition_evidence_metrics(path)


def human_metrics(clip_id, statuses):
    rows = [
        {
            "clip_id": clip_id,
            "is_evaluable_attempt": True,
            "source_video_visibility_status": status,
        }
        for status in statuses
    ]
    rows.append(
        {
            "clip_id": clip_id,
            "is_evaluable_attempt": False,
            "source_video_visibility_status": "insufficient",
        }
    )
    return human_alignment_evidence_metrics(pd.DataFrame(rows), clip_id=clip_id)


def test_frame_metrics_use_recorded_analysis_timings_and_explicit_denominators(
    tmp_path,
):
    baseline = load_frames(tmp_path, "clip-a", (1.0, 3.0), method="baseline")
    enhanced = load_frames(tmp_path, "clip-a", (2.0, 4.0), method="enhanced")

    assert baseline.analyzed_frame_count == 2
    assert baseline.mean_measured_processing_time_ms == 2.0
    assert baseline.median_measured_processing_time_ms == 2.0
    assert baseline.measured_analysis_throughput_fps == 500.0
    assert baseline.pose_availability.to_dict() == {
        "available_frames": 1,
        "denominator_frames": 2,
        "rate": 0.5,
    }
    assert baseline.elbow_availability.rate == 0.5
    assert baseline.alignment_availability.rate == 1.0
    assert baseline.selected_side_availability.rate == 1.0
    assert baseline.side_change_count == 1
    assert baseline.side_change_semantics == (
        "instantaneous_selected_side_state_changes"
    )
    assert enhanced.elbow_availability.rate == 0.5
    assert enhanced.alignment_availability.rate == 0.5
    assert enhanced.side_change_count == 1
    assert enhanced.side_change_semantics == "stable_selector_side_changed_events"


def test_aggregate_processing_and_availability_are_frame_weighted(tmp_path):
    clip_a = ClipEvidenceMetrics(
        baseline_frames=load_frames(
            tmp_path,
            "clip-a",
            (1.0,),
            method="baseline",
        ),
        enhanced_frames=load_frames(
            tmp_path,
            "clip-a",
            (2.0,),
            method="enhanced",
        ),
        enhanced_repetitions=repetition_metrics(
            tmp_path,
            "clip-a",
            ("correct",),
            (0.25,),
        ),
        human_alignment_evidence=human_metrics("clip-a", ("sufficient",)),
    )
    clip_b = ClipEvidenceMetrics(
        baseline_frames=load_frames(
            tmp_path,
            "clip-b",
            (3.0, 5.0, 7.0),
            method="baseline",
        ),
        enhanced_frames=load_frames(
            tmp_path,
            "clip-b",
            (4.0, 6.0, 8.0),
            method="enhanced",
        ),
        enhanced_repetitions=repetition_metrics(
            tmp_path,
            "clip-b",
            ("unscorable", "correct"),
            (0.5, 1.0),
        ),
        human_alignment_evidence=human_metrics(
            "clip-b",
            ("partially_obscured", "insufficient"),
        ),
    )

    aggregate = aggregate_formal_evidence((clip_a, clip_b))

    assert aggregate.baseline_frames.analyzed_frame_count == 4
    assert aggregate.baseline_frames.mean_measured_processing_time_ms == 4.0
    assert aggregate.baseline_frames.median_measured_processing_time_ms == 4.0
    assert aggregate.baseline_frames.measured_analysis_throughput_fps == 250.0
    assert aggregate.baseline_frames.pose_availability.available_frames == 3
    assert aggregate.baseline_frames.pose_availability.denominator_frames == 4
    assert aggregate.baseline_frames.pose_availability.rate == 0.75
    assert aggregate.enhanced_repetitions.predicted_unscorable_count == 1
    assert aggregate.enhanced_repetitions.predicted_unscorable_rate == 1 / 3
    assert aggregate.enhanced_repetitions.mean_alignment_valid_ratio == 1.75 / 3
    assert aggregate.human_alignment_evidence.to_dict() == {
        "evaluable_attempt_count": 3,
        "adequate_attempt_count": 2,
        "inadequate_attempt_count": 1,
        "adequate_attempt_rate": 2 / 3,
        "evidence_basis": "source_video_visibility_status",
    }


def test_historical_missing_columns_remain_unavailable(tmp_path):
    path = tmp_path / "historical.csv"
    write_csv(
        path,
        ("run_id", "clip_id", "frame_index"),
        ({"run_id": "old-run", "clip_id": "old-clip", "frame_index": 0},),
    )

    metrics = load_frame_evidence_metrics(
        path,
        method="enhanced",
        expected_run_id="old-run",
        expected_clip_id="old-clip",
        expected_frame_count=1,
    )

    assert metrics.analyzed_frame_count == 1
    assert metrics.mean_measured_processing_time_ms is None
    assert metrics.measured_analysis_throughput_fps is None
    assert metrics.pose_availability.available_frames is None
    assert metrics.pose_availability.denominator_frames == 0
    assert metrics.pose_availability.rate is None
    assert metrics.side_change_count is None


def test_human_evidence_excludes_ambiguous_fragments_and_is_not_model_coverage():
    metrics = human_metrics(
        "clip-a",
        ("sufficient", "partially_obscured", "insufficient"),
    )

    assert metrics.evaluable_attempt_count == 3
    assert metrics.adequate_attempt_count == 2
    assert metrics.inadequate_attempt_count == 1
    assert metrics.adequate_attempt_rate == 2 / 3
