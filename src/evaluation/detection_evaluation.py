"""Evaluate per-clip repetition completion-event detection."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Sequence

from evaluation.dataset_validation import (
    load_and_validate_evaluation_data,
)
from evaluation.event_matching import (
    DEFAULT_EVENT_TOLERANCE_SECONDS,
    EventMatchResult,
    match_repetition_events,
)
from evaluation.repetition_events import (
    GroundTruthRepetitionEvent,
    PredictedRepetitionEvent,
    extract_ground_truth_events,
    load_baseline_events,
    load_enhanced_events,
)


@dataclass(frozen=True)
class DetectionSummary:
    """Detection and completion-timing metrics for one clip and method.

    Signed count error is predicted count minus GT count. Completion-timing
    means are measured in seconds and are absent when there are no matches;
    event precision, recall and F1 use ``0.0`` for zero denominators.
    """

    run_id: str | None
    clip_id: str
    method: str
    ground_truth_event_count: int
    predicted_event_count: int
    signed_count_error: int
    absolute_count_error: int
    matched_events: int
    missed_annotations: int
    extra_predictions: int
    event_precision: float
    event_recall: float
    event_f1: float
    mean_signed_completion_timing_error_seconds: (
        float | None
    )
    mean_absolute_completion_timing_error_seconds: (
        float | None
    )
    tolerance_seconds: float
    tolerance_frames: int

    def to_dict(self) -> dict[str, object]:
        """Return the summary fields in a serialization-ready mapping."""
        return asdict(self)


def summarise_detection(
    match_result: EventMatchResult,
) -> DetectionSummary:
    """Summarise matches, misses and extras from one matching result.

    A match is a true-positive completion event, an unmatched annotation is a
    miss and an unmatched prediction is an extra. Empty prediction and/or GT
    sets remain valid and produce zero event rates; timing means require at
    least one matched pair.
    """
    ground_truth_count = (
        len(match_result.matched_pairs)
        + len(match_result.unmatched_annotations)
    )
    predicted_count = (
        len(match_result.matched_pairs)
        + len(match_result.unmatched_predictions)
    )
    matched_count = len(match_result.matched_pairs)
    precision = (
        matched_count / predicted_count
        if predicted_count > 0
        else 0.0
    )
    recall = (
        matched_count / ground_truth_count
        if ground_truth_count > 0
        else 0.0
    )
    event_f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0.0
        else 0.0
    )
    signed_errors = [
        pair.signed_timing_error_seconds
        for pair in match_result.matched_pairs
    ]
    absolute_errors = [
        pair.absolute_timing_error_seconds
        for pair in match_result.matched_pairs
    ]
    run_ids = {
        event.run_id
        for event in (
            pair.prediction
            for pair in match_result.matched_pairs
        )
    }
    run_ids.update(
        event.run_id
        for event in match_result.unmatched_predictions
    )

    if len(run_ids) > 1:
        raise ValueError(
            "A per-clip detection summary cannot mix run IDs"
        )

    return DetectionSummary(
        run_id=next(iter(run_ids), None),
        clip_id=match_result.clip_id,
        method=match_result.method,
        ground_truth_event_count=ground_truth_count,
        predicted_event_count=predicted_count,
        signed_count_error=(
            predicted_count - ground_truth_count
        ),
        absolute_count_error=abs(
            predicted_count - ground_truth_count
        ),
        matched_events=matched_count,
        missed_annotations=len(
            match_result.unmatched_annotations
        ),
        extra_predictions=len(
            match_result.unmatched_predictions
        ),
        event_precision=precision,
        event_recall=recall,
        event_f1=event_f1,
        mean_signed_completion_timing_error_seconds=(
            fmean(signed_errors)
            if signed_errors
            else None
        ),
        mean_absolute_completion_timing_error_seconds=(
            fmean(absolute_errors)
            if absolute_errors
            else None
        ),
        tolerance_seconds=(
            match_result.tolerance_seconds
        ),
        tolerance_frames=match_result.tolerance_frames,
    )


def evaluate_detection_for_clip(
    predictions: Sequence[PredictedRepetitionEvent],
    annotations: Sequence[GroundTruthRepetitionEvent],
    *,
    clip_id: str,
    method: str,
    source_fps: float,
    tolerance_seconds: float = (
        DEFAULT_EVENT_TOLERANCE_SECONDS
    ),
) -> tuple[EventMatchResult, DetectionSummary]:
    """Match one clip's events and return its detection-only summary.

    ``source_fps`` converts frame differences to seconds when recorded
    timestamps are unavailable. Classification is deliberately outside this
    function.
    """
    match_result = match_repetition_events(
        predictions,
        annotations,
        clip_id=clip_id,
        method=method,
        source_fps=source_fps,
        tolerance_seconds=tolerance_seconds,
    )
    return match_result, summarise_detection(
        match_result
    )


def _match_payload(
    match_result: EventMatchResult,
    summary: DetectionSummary,
) -> dict[str, object]:
    """Build the detailed, detection-only JSON payload used by the CLI."""
    return {
        "summary": summary.to_dict(),
        "matched_event_pairs": [
            {
                "predicted_rep_id": (
                    pair.prediction.predicted_rep_id
                ),
                "ground_truth_attempt_id": (
                    pair.annotation
                    .ground_truth_attempt_id
                ),
                "predicted_completion_frame": (
                    pair.prediction.completion_frame
                ),
                "annotated_completion_frame": (
                    pair.annotation.completion_frame
                ),
                "signed_frame_error": (
                    pair.signed_frame_error
                ),
                "signed_timing_error_seconds": (
                    pair.signed_timing_error_seconds
                ),
                "absolute_timing_error_seconds": (
                    pair.absolute_timing_error_seconds
                ),
                "matching_basis": pair.matching_basis,
            }
            for pair in match_result.matched_pairs
        ],
        "unmatched_prediction_ids": [
            event.predicted_rep_id
            for event in match_result.unmatched_predictions
        ],
        "unmatched_ground_truth_attempt_ids": [
            event.ground_truth_attempt_id
            for event in match_result.unmatched_annotations
        ],
    }


def parse_arguments(argv=None) -> argparse.Namespace:
    """Parse arguments for one-clip detection evaluation."""
    parser = argparse.ArgumentParser(
        description=(
            "Extract and match repetition completion events, "
            "then report detection-only metrics for one clip."
        )
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=("baseline", "enhanced"),
        help="Prediction method represented by the CSV.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help=(
            "Baseline frame CSV or enhanced repetition CSV."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Validated dataset manifest CSV.",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        required=True,
        help="Validated repetition annotation CSV.",
    )
    parser.add_argument(
        "--clip-id",
        required=True,
        help="Clip to report.",
    )
    parser.add_argument(
        "--tolerance-seconds",
        type=float,
        default=DEFAULT_EVENT_TOLERANCE_SECONDS,
        help="One-to-one completion-event tolerance.",
    )
    return parser.parse_args(argv)


def main() -> None:
    """Validate inputs and print one clip's detection result as JSON."""
    args = parse_arguments()
    manifest, annotations = (
        load_and_validate_evaluation_data(
            manifest_path=args.manifest,
            annotations_path=args.annotations,
        )
    )
    manifest_clip_ids = (
        manifest["clip_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    selected_manifest = manifest.loc[
        manifest_clip_ids.eq(args.clip_id)
    ]

    if selected_manifest.empty:
        raise ValueError(
            f"Clip {args.clip_id!r} is not present in "
            "the dataset manifest"
        )

    source_fps_by_clip = dict(
        zip(
            manifest_clip_ids,
            manifest["source_fps"],
        )
    )

    if args.method == "baseline":
        all_predictions = load_baseline_events(
            args.predictions,
            source_fps_by_clip=source_fps_by_clip,
        )
    else:
        all_predictions = load_enhanced_events(
            args.predictions,
            source_fps_by_clip=source_fps_by_clip,
        )

    all_ground_truth = extract_ground_truth_events(
        annotations,
        manifest,
        source_name=str(args.annotations),
        manifest_source_name=str(args.manifest),
    )
    predictions_for_clip = [
        event
        for event in all_predictions
        if event.clip_id == args.clip_id
    ]
    ground_truth_for_clip = [
        event
        for event in all_ground_truth
        if event.clip_id == args.clip_id
    ]
    match_result, summary = evaluate_detection_for_clip(
        predictions_for_clip,
        ground_truth_for_clip,
        clip_id=args.clip_id,
        method=args.method,
        source_fps=float(
            selected_manifest.iloc[0]["source_fps"]
        ),
        tolerance_seconds=args.tolerance_seconds,
    )

    print(
        json.dumps(
            _match_payload(match_result, summary),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
