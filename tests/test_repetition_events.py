import pandas as pd
import pytest

from evaluation.repetition_events import (
    extract_baseline_events,
    extract_enhanced_events,
    extract_ground_truth_events,
    load_baseline_events,
)


def make_baseline_rows():
    return pd.DataFrame(
        {
            "run_id": ["run-1"] * 5,
            "clip_id": ["fictional-clip"] * 5,
            "frame_index": [0, 1, 2, 3, 4],
            "video_timestamp_ms": [
                0.0,
                100.0,
                200.0,
                300.0,
                400.0,
            ],
            "source_fps": [10.0] * 5,
            "baseline_rep_count": [0, 0, 1, 1, 2],
        }
    )


def make_enhanced_rows():
    return pd.DataFrame(
        [
            {
                "run_id": "run-2",
                "clip_id": "fictional-clip",
                "rep_id": 1,
                "start_frame": 10,
                "bottom_frame": 15,
                "end_frame": 20,
                "predicted_class": "correct",
            },
            {
                "run_id": "run-2",
                "clip_id": "fictional-clip",
                "rep_id": 2,
                "start_frame": 30,
                "bottom_frame": 35,
                "end_frame": 40,
                "predicted_class": "insufficient_depth",
            },
        ]
    )


def make_manifest():
    return pd.DataFrame(
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
                "frame_count": 200,
                "width_px": 1280,
                "height_px": 720,
                "recording_condition": (
                    "controlled_fictional"
                ),
                "notes": "",
            }
        ]
    )


def annotation_row(
    attempt_id,
    *,
    evaluable,
    ambiguous,
    start,
    bottom,
    end,
    ground_truth_class="correct",
    notes="",
):
    return {
        "clip_id": "fictional-clip",
        "ground_truth_attempt_id": attempt_id,
        "is_evaluable_attempt": evaluable,
        "ambiguity_flag": ambiguous,
        "start_top_frame": start,
        "bottom_turnaround_frame": bottom,
        "completion_end_top_frame": end,
        "ground_truth_class": ground_truth_class,
        "insufficient_depth_flag": False,
        "incomplete_extension_flag": False,
        "alignment_deviation_flag": False,
        "source_video_visibility_status": (
            "sufficient"
            if evaluable
            else "partially_obscured"
        ),
        "annotator_notes": notes,
    }


def test_baseline_completion_events_follow_count_increases():
    events = extract_baseline_events(
        make_baseline_rows()
    )

    assert [
        (
            event.predicted_rep_id,
            event.completion_frame,
            event.completion_timestamp_ms,
            event.resulting_cumulative_count,
            event.method,
        )
        for event in events
    ] == [
        (1, 2, 200.0, 1, "baseline"),
        (2, 4, 400.0, 2, "baseline"),
    ]
    assert all(
        event.run_id == "run-1"
        and event.clip_id == "fictional-clip"
        for event in events
    )


@pytest.mark.parametrize(
    ("counts", "message"),
    [
        ([0, 1, 0, 0, 0], "non-decreasing"),
        ([0, 2, 2, 2, 2], "exactly one"),
    ],
)
def test_baseline_rejects_malformed_counts(
    counts,
    message,
):
    rows = make_baseline_rows()
    rows["baseline_rep_count"] = counts

    with pytest.raises(ValueError, match=message):
        extract_baseline_events(rows)


def test_baseline_rejects_non_increasing_frames():
    rows = make_baseline_rows()
    rows["frame_index"] = [0, 1, 1, 3, 4]

    with pytest.raises(
        ValueError,
        match="strictly increasing",
    ):
        extract_baseline_events(rows)


def test_baseline_rejects_duplicate_completion_frames():
    first_run = make_baseline_rows().iloc[:3].copy()
    second_run = first_run.copy()
    second_run["run_id"] = "run-2"
    second_run["baseline_rep_count"] = [1, 1, 2]
    combined = pd.concat(
        [first_run, second_run],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="duplicate completion frames",
    ):
        extract_baseline_events(combined)


def test_baseline_rejects_duplicate_predicted_ids():
    first_run = pd.DataFrame(
        {
            "run_id": ["run-1", "run-1"],
            "clip_id": ["fictional-clip"] * 2,
            "frame_index": [0, 2],
            "baseline_rep_count": [0, 1],
        }
    )
    second_run = pd.DataFrame(
        {
            "run_id": ["run-2", "run-2"],
            "clip_id": ["fictional-clip"] * 2,
            "frame_index": [0, 3],
            "baseline_rep_count": [0, 1],
        }
    )

    with pytest.raises(
        ValueError,
        match="duplicate.*identifiers",
    ):
        extract_baseline_events(
            pd.concat(
                [first_run, second_run],
                ignore_index=True,
            )
        )


def test_baseline_csv_loader_is_reusable(tmp_path):
    input_path = tmp_path / "baseline.csv"
    make_baseline_rows().to_csv(input_path, index=False)

    events = load_baseline_events(input_path)

    assert [
        event.completion_frame for event in events
    ] == [2, 4]


def test_enhanced_events_derive_completion_time_from_fps():
    events = extract_enhanced_events(
        make_enhanced_rows(),
        source_fps_by_clip={"fictional-clip": 20.0},
    )

    assert events[0].completion_timestamp_ms == 1000.0
    assert events[1].completion_timestamp_ms == 2000.0
    assert events[0].start_frame == 10
    assert events[0].bottom_frame == 15
    assert events[0].completion_frame == 20
    assert events[0].predicted_class == "correct"
    assert events[0].method == "enhanced"


def test_enhanced_rejects_duplicate_event_ids():
    rows = make_enhanced_rows()
    rows.loc[1, "rep_id"] = 1

    with pytest.raises(
        ValueError,
        match="duplicate.*identifiers",
    ):
        extract_enhanced_events(rows)


def test_enhanced_rejects_invalid_frame_order():
    rows = make_enhanced_rows()
    rows.loc[0, "bottom_frame"] = 21

    with pytest.raises(
        ValueError,
        match="start_frame <= bottom_frame <= end_frame",
    ):
        extract_enhanced_events(rows)


def test_ground_truth_keeps_evaluable_and_excludes_ambiguous():
    annotations = pd.DataFrame(
        [
            annotation_row(
                "A001",
                evaluable=True,
                ambiguous=False,
                start=10,
                bottom=15,
                end=20,
            ),
            annotation_row(
                "F001",
                evaluable=False,
                ambiguous=True,
                start=40,
                bottom=None,
                end=45,
                ground_truth_class="unscorable",
                notes="Fictional ambiguous fragment.",
            ),
        ]
    )

    events = extract_ground_truth_events(
        annotations,
        make_manifest(),
    )

    assert len(events) == 1
    assert events[0].ground_truth_attempt_id == "A001"
    assert events[0].completion_frame == 20
    assert events[0].completion_timestamp_ms == 1000.0


def test_ground_truth_loader_supports_no_annotations():
    annotations = pd.DataFrame(
        columns=[
            "clip_id",
            "ground_truth_attempt_id",
            "is_evaluable_attempt",
            "ambiguity_flag",
            "start_top_frame",
            "bottom_turnaround_frame",
            "completion_end_top_frame",
            "ground_truth_class",
            "insufficient_depth_flag",
            "incomplete_extension_flag",
            "alignment_deviation_flag",
            "source_video_visibility_status",
            "annotator_notes",
        ]
    )

    assert extract_ground_truth_events(
        annotations,
        make_manifest(),
    ) == []
