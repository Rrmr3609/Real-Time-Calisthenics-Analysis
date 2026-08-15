import csv
import json
import math
from collections import Counter
from dataclasses import replace

import pytest

import evaluation.formal_reporting as formal_reporting
from evaluation.classification_evaluation import (
    SUPPORTED_FORM_CLASSES,
    evaluate_classification,
)
from evaluation.detection_evaluation import DetectionSummary
from evaluation.formal_evaluation import (
    EnhancedClipEvaluation,
    GroundTruthClassDetectionRecall,
    MatchedClassificationPair,
)
from evaluation.formal_reporting import (
    PER_CLIP_COLUMNS,
    EvaluationClipContext,
    EvaluationEvidenceProvenance,
    SourceRunProvenance,
    aggregate_formal_evaluation,
    formal_evaluation_output_paths,
    write_formal_evaluation_report,
)

FPS = 10.0
TOLERANCE = 0.5


def source_run_provenance(clip_id, method):
    return SourceRunProvenance(
        clip_id=clip_id,
        method=method,
        source_run_id=f"{method}-{clip_id}-run",
        split="development",
        source_metadata_path=(f"runs/{method}-{clip_id}_metadata.json"),
        source_metadata_sha256=("a" * 64 if method == "baseline" else "b" * 64),
        consumed_output_name=(
            "frame_csv" if method == "baseline" else "repetition_csv"
        ),
        consumed_output_path=(f"runs/{method}-{clip_id}.csv"),
        consumed_output_sha256=("c" * 64 if method == "baseline" else "d" * 64),
        descriptive_frame_csv_path=(f"runs/{method}-{clip_id}-frames.csv"),
        descriptive_frame_csv_sha256=("1" * 64 if method == "baseline" else "2" * 64),
        source_input_video_sha256="e" * 64,
        source_git_commit="abc123",
        source_git_dirty=False,
        resolved_configuration_sha256="f" * 64,
    )


def evidence_provenance():
    return EvaluationEvidenceProvenance(
        manifest_path="data/manifests/fictional.csv",
        manifest_sha256="3" * 64,
        annotations_path="data/annotations/fictional.csv",
        annotations_sha256="4" * 64,
        frozen_annotation_sha256="4" * 64,
        review_metadata_path="data/annotations/fictional.review.json",
        review_metadata_sha256="5" * 64,
        review_status="complete",
    )


def detection(
    clip_id,
    method,
    *,
    ground_truth,
    predicted,
    matched,
    mean_signed_timing=None,
    mean_absolute_timing=None,
    tolerance=TOLERANCE,
    fps=FPS,
):
    misses = ground_truth - matched
    extras = predicted - matched
    precision = matched / predicted if predicted else 0.0
    recall = matched / ground_truth if ground_truth else 0.0
    event_f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0.0
        else 0.0
    )

    if matched and mean_signed_timing is None:
        mean_signed_timing = 0.0

    if matched and mean_absolute_timing is None:
        mean_absolute_timing = 0.0

    return DetectionSummary(
        run_id=f"{method}-{clip_id}",
        clip_id=clip_id,
        method=method,
        ground_truth_event_count=ground_truth,
        predicted_event_count=predicted,
        signed_count_error=predicted - ground_truth,
        absolute_count_error=abs(predicted - ground_truth),
        matched_events=matched,
        missed_annotations=misses,
        extra_predictions=extras,
        event_precision=precision,
        event_recall=recall,
        event_f1=event_f1,
        mean_signed_completion_timing_error_seconds=(mean_signed_timing),
        mean_absolute_completion_timing_error_seconds=(mean_absolute_timing),
        tolerance_seconds=tolerance,
        tolerance_frames=math.ceil(tolerance * fps),
    )


def enhanced(
    clip_id,
    *,
    ground_truth_labels=(),
    predicted_labels=(),
    missed_labels=(),
    extra_predictions=0,
    mean_signed_timing=None,
    mean_absolute_timing=None,
    tolerance=TOLERANCE,
    fps=FPS,
):
    classification = evaluate_classification(
        ground_truth_labels,
        predicted_labels,
    )
    matched = len(ground_truth_labels)
    ground_truth_count = matched + len(missed_labels)
    predicted_count = matched + extra_predictions
    summary = detection(
        clip_id,
        "enhanced",
        ground_truth=ground_truth_count,
        predicted=predicted_count,
        matched=matched,
        mean_signed_timing=mean_signed_timing,
        mean_absolute_timing=mean_absolute_timing,
        tolerance=tolerance,
        fps=fps,
    )
    support = Counter((*ground_truth_labels, *missed_labels))
    matched_counts = Counter(ground_truth_labels)
    missed_counts = Counter(missed_labels)
    recall_rows = tuple(
        GroundTruthClassDetectionRecall(
            label=label,
            ground_truth_support=support[label],
            matched_ground_truth_repetitions=(matched_counts[label]),
            missed_ground_truth_repetitions=(missed_counts[label]),
            recall=(matched_counts[label] / support[label] if support[label] else None),
        )
        for label in SUPPORTED_FORM_CLASSES
    )
    matched_pairs = tuple(
        MatchedClassificationPair(
            ground_truth_attempt_id=f"A{index + 1:03d}",
            predicted_rep_id=index + 1,
            ground_truth_completion_frame=10 * (index + 1),
            predicted_completion_frame=10 * (index + 1),
            ground_truth_class=ground_truth_label,
            predicted_class=predicted_label,
            signed_frame_error=0,
            signed_timing_error_seconds=0.0,
            absolute_timing_error_seconds=0.0,
            matching_basis="frame",
        )
        for index, (
            ground_truth_label,
            predicted_label,
        ) in enumerate(zip(ground_truth_labels, predicted_labels))
    )

    return EnhancedClipEvaluation(
        detection=summary,
        matched_pairs=matched_pairs,
        unmatched_prediction_ids=tuple(range(matched + 1, predicted_count + 1)),
        unmatched_ground_truth_attempt_ids=tuple(
            f"M{index + 1:03d}" for index in range(len(missed_labels))
        ),
        classification=classification,
        detection_recall_by_ground_truth_class=recall_rows,
    )


def context(
    clip_id,
    *,
    split="development",
    fps=FPS,
):
    return EvaluationClipContext(
        clip_id=clip_id,
        split=split,
        source_fps=fps,
    )


def aggregate(
    baseline_results,
    enhanced_results,
    *,
    contexts=None,
    split="development",
    tolerance=TOLERANCE,
):
    if contexts is None:
        contexts = [context(result.clip_id) for result in baseline_results]

    return aggregate_formal_evaluation(
        baseline_results=baseline_results,
        enhanced_results=enhanced_results,
        clip_contexts=contexts,
        split=split,
        tolerance_seconds=tolerance,
    )


def perfect_report(tolerance=TOLERANCE):
    baseline = detection(
        "clip-a",
        "baseline",
        ground_truth=1,
        predicted=1,
        matched=1,
        tolerance=tolerance,
    )
    enhanced_result = enhanced(
        "clip-a",
        ground_truth_labels=("correct",),
        predicted_labels=("correct",),
        tolerance=tolerance,
    )
    return aggregate(
        [baseline],
        [enhanced_result],
        tolerance=tolerance,
    )


def test_perfect_detection_over_multiple_clips():
    baseline_results = [
        detection(
            "clip-b",
            "baseline",
            ground_truth=2,
            predicted=2,
            matched=2,
        ),
        detection(
            "clip-a",
            "baseline",
            ground_truth=1,
            predicted=1,
            matched=1,
        ),
    ]
    enhanced_results = [
        enhanced(
            "clip-a",
            ground_truth_labels=("correct",),
            predicted_labels=("correct",),
        ),
        enhanced(
            "clip-b",
            ground_truth_labels=(
                "insufficient_depth",
                "incomplete_extension",
            ),
            predicted_labels=(
                "insufficient_depth",
                "incomplete_extension",
            ),
        ),
    ]
    report = aggregate(baseline_results, enhanced_results)

    assert report.ordered_clip_ids == ("clip-a", "clip-b")
    assert report.baseline_detection.evaluated_clips == 2
    assert report.baseline_detection.total_ground_truth_repetitions == 3
    assert report.baseline_detection.total_matched_events == 3
    assert report.baseline_detection.pooled_event_f1 == 1.0
    assert report.enhanced_detection.pooled_event_f1 == 1.0
    assert report.enhanced_classification.accuracy == 1.0


def test_pooled_detection_uses_summed_counts_not_mean_f1():
    baseline_results = [
        detection(
            "clip-a",
            "baseline",
            ground_truth=1,
            predicted=1,
            matched=1,
        ),
        detection(
            "clip-b",
            "baseline",
            ground_truth=3,
            predicted=3,
            matched=1,
        ),
    ]
    enhanced_results = [
        enhanced(
            "clip-a",
            ground_truth_labels=("correct",),
            predicted_labels=("correct",),
        ),
        enhanced(
            "clip-b",
            ground_truth_labels=("correct",),
            predicted_labels=("correct",),
            missed_labels=("correct", "correct"),
            extra_predictions=2,
        ),
    ]
    report = aggregate(baseline_results, enhanced_results)
    pooled = report.baseline_detection

    assert pooled.total_matched_events == 2
    assert pooled.total_misses == 2
    assert pooled.total_extras == 2
    assert pooled.pooled_event_precision == 0.5
    assert pooled.pooled_event_recall == 0.5
    assert pooled.pooled_event_f1 == 0.5
    clip_f1_mean = sum(result.event_f1 for result in baseline_results) / 2
    assert clip_f1_mean == pytest.approx(2.0 / 3.0)
    assert pooled.pooled_event_f1 != clip_f1_mean


def test_count_error_means_and_exact_count_accuracy():
    baseline_results = [
        detection(
            "clip-a",
            "baseline",
            ground_truth=2,
            predicted=3,
            matched=2,
        ),
        detection(
            "clip-b",
            "baseline",
            ground_truth=4,
            predicted=2,
            matched=2,
        ),
        detection(
            "clip-c",
            "baseline",
            ground_truth=1,
            predicted=1,
            matched=1,
        ),
    ]
    enhanced_results = [
        enhanced(
            "clip-a",
            ground_truth_labels=("correct", "correct"),
            predicted_labels=("correct", "correct"),
            extra_predictions=1,
        ),
        enhanced(
            "clip-b",
            ground_truth_labels=("correct", "correct"),
            predicted_labels=("correct", "correct"),
            missed_labels=("correct", "correct"),
        ),
        enhanced(
            "clip-c",
            ground_truth_labels=("correct",),
            predicted_labels=("correct",),
        ),
    ]
    pooled = aggregate(
        baseline_results,
        enhanced_results,
    ).baseline_detection

    assert pooled.total_signed_count_error == -1
    assert pooled.mean_signed_count_error == pytest.approx(-1.0 / 3.0)
    assert pooled.mean_absolute_count_error == 1.0
    assert pooled.exact_count_clip_count == 1
    assert pooled.exact_count_clip_accuracy == pytest.approx(1.0 / 3.0)


def test_timing_means_are_weighted_by_matched_observations():
    baseline_results = [
        detection(
            "clip-a",
            "baseline",
            ground_truth=1,
            predicted=1,
            matched=1,
            mean_signed_timing=1.0,
            mean_absolute_timing=1.0,
        ),
        detection(
            "clip-b",
            "baseline",
            ground_truth=3,
            predicted=3,
            matched=3,
            mean_signed_timing=0.0,
            mean_absolute_timing=0.2,
        ),
    ]
    enhanced_results = [
        enhanced(
            "clip-a",
            ground_truth_labels=("correct",),
            predicted_labels=("correct",),
        ),
        enhanced(
            "clip-b",
            ground_truth_labels=("correct",) * 3,
            predicted_labels=("correct",) * 3,
        ),
    ]
    pooled = aggregate(
        baseline_results,
        enhanced_results,
    ).baseline_detection

    assert pooled.total_matched_timing_observations == 4
    assert pooled.mean_signed_completion_timing_error_seconds == 0.25
    assert pooled.mean_absolute_completion_timing_error_seconds == pytest.approx(0.4)


def test_no_timing_observations_returns_none():
    baseline = detection(
        "clip-a",
        "baseline",
        ground_truth=1,
        predicted=0,
        matched=0,
    )
    enhanced_result = enhanced(
        "clip-a",
        missed_labels=("correct",),
    )
    report = aggregate([baseline], [enhanced_result])

    assert report.baseline_detection.total_matched_timing_observations == 0
    assert report.baseline_detection.mean_signed_completion_timing_error_seconds is None
    assert (
        report.enhanced_detection.mean_absolute_completion_timing_error_seconds is None
    )


def test_enhanced_classification_pools_raw_confusion_counts():
    baseline_results = [
        detection(
            "clip-a",
            "baseline",
            ground_truth=2,
            predicted=2,
            matched=2,
        ),
        detection(
            "clip-b",
            "baseline",
            ground_truth=2,
            predicted=2,
            matched=2,
        ),
    ]
    enhanced_results = [
        enhanced(
            "clip-a",
            ground_truth_labels=("correct", "correct"),
            predicted_labels=(
                "correct",
                "insufficient_depth",
            ),
        ),
        enhanced(
            "clip-b",
            ground_truth_labels=(
                "insufficient_depth",
                "incomplete_extension",
            ),
            predicted_labels=(
                "insufficient_depth",
                "correct",
            ),
        ),
    ]
    classification = aggregate(
        baseline_results,
        enhanced_results,
    ).enhanced_classification

    assert classification.confusion_matrix == (
        (1, 1, 0, 0, 0),
        (0, 1, 0, 0, 0),
        (1, 0, 0, 0, 0),
        (0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0),
    )
    correct = classification.per_class[0]
    depth = classification.per_class[1]
    extension = classification.per_class[2]
    assert (
        correct.true_positives,
        correct.false_positives,
        correct.false_negatives,
    ) == (1, 1, 1)
    assert (
        depth.true_positives,
        depth.false_positives,
        depth.false_negatives,
    ) == (1, 1, 0)
    assert extension.false_negatives == 1
    assert classification.accuracy == 0.5
    assert classification.macro_f1 == pytest.approx(7.0 / 18.0)


def test_predicted_zero_support_class_does_not_reduce_macro_f1():
    baseline = detection(
        "clip-a",
        "baseline",
        ground_truth=2,
        predicted=2,
        matched=2,
    )
    enhanced_result = enhanced(
        "clip-a",
        ground_truth_labels=("correct", "correct"),
        predicted_labels=("correct", "unscorable"),
    )
    classification = aggregate(
        [baseline],
        [enhanced_result],
    ).enhanced_classification
    unscorable = classification.per_class[4]

    assert unscorable.support == 0
    assert unscorable.false_positives == 1
    assert classification.macro_f1 == pytest.approx(2.0 / 3.0)


def test_no_enhanced_matches_has_empty_classification():
    baseline = detection(
        "clip-a",
        "baseline",
        ground_truth=1,
        predicted=0,
        matched=0,
    )
    enhanced_result = enhanced(
        "clip-a",
        missed_labels=("correct",),
    )
    classification = aggregate(
        [baseline],
        [enhanced_result],
    ).enhanced_classification

    assert classification.evaluated_matched_repetitions == 0
    assert classification.accuracy is None
    assert classification.macro_f1 is None
    assert all(count == 0 for row in classification.confusion_matrix for count in row)


def test_detection_recall_by_class_is_pooled_and_validated():
    baseline_results = [
        detection(
            "clip-a",
            "baseline",
            ground_truth=2,
            predicted=1,
            matched=1,
        ),
        detection(
            "clip-b",
            "baseline",
            ground_truth=2,
            predicted=1,
            matched=1,
        ),
    ]
    enhanced_results = [
        enhanced(
            "clip-a",
            ground_truth_labels=("correct",),
            predicted_labels=("correct",),
            missed_labels=("correct",),
        ),
        enhanced(
            "clip-b",
            ground_truth_labels=("insufficient_depth",),
            predicted_labels=("insufficient_depth",),
            missed_labels=("incomplete_extension",),
        ),
    ]
    report = aggregate(baseline_results, enhanced_results)
    rows = {
        row.label: row
        for row in (report.enhanced_detection_recall_by_ground_truth_class)
    }

    assert rows["correct"].ground_truth_support == 2
    assert rows["correct"].matched_ground_truth_repetitions == 1
    assert rows["correct"].missed_ground_truth_repetitions == 1
    assert rows["correct"].recall == 0.5
    assert rows["insufficient_depth"].recall == 1.0
    assert rows["incomplete_extension"].recall == 0.0
    assert rows["alignment_deviation"].recall is None
    assert rows["unscorable"].recall is None
    assert (
        sum(row.ground_truth_support for row in rows.values())
        == report.enhanced_detection.total_ground_truth_repetitions
    )
    assert (
        sum(row.matched_ground_truth_repetitions for row in rows.values())
        == report.enhanced_detection.total_matched_events
    )
    assert (
        sum(row.missed_ground_truth_repetitions for row in rows.values())
        == report.enhanced_detection.total_misses
    )


def test_per_clip_rows_and_serialisation_are_deterministic():
    baseline_results = [
        detection(
            "clip-b",
            "baseline",
            ground_truth=1,
            predicted=0,
            matched=0,
        ),
        detection(
            "clip-a",
            "baseline",
            ground_truth=1,
            predicted=1,
            matched=1,
        ),
    ]
    enhanced_results = [
        enhanced(
            "clip-b",
            missed_labels=("correct",),
        ),
        enhanced(
            "clip-a",
            ground_truth_labels=("correct",),
            predicted_labels=("insufficient_depth",),
        ),
    ]
    report = aggregate(baseline_results, enhanced_results)
    repeated = aggregate(
        list(reversed(baseline_results)),
        list(reversed(enhanced_results)),
    )
    first_row = report.per_clip_metrics[0]
    payload = report.to_dict()

    assert report == repeated
    assert report.ordered_clip_ids == ("clip-a", "clip-b")
    assert tuple(row.clip_id for row in report.per_clip_metrics) == ("clip-a", "clip-b")
    assert first_row.ground_truth_repetition_count == 1
    assert first_row.baseline_predicted_count == 1
    assert first_row.enhanced_matches == 1
    assert first_row.enhanced_classification_accuracy == 0.0
    assert (
        tuple(
            row.label
            for row in (report.enhanced_detection_recall_by_ground_truth_class)
        )
        == SUPPORTED_FORM_CLASSES
    )
    assert json.loads(json.dumps(payload, sort_keys=True)) == payload


@pytest.mark.parametrize(
    "duplicate_target",
    ["baseline", "enhanced", "context"],
)
def test_duplicate_clip_ids_are_rejected(duplicate_target):
    baseline = detection(
        "clip-a",
        "baseline",
        ground_truth=1,
        predicted=1,
        matched=1,
    )
    enhanced_result = enhanced(
        "clip-a",
        ground_truth_labels=("correct",),
        predicted_labels=("correct",),
    )
    baseline_results = [baseline]
    enhanced_results = [enhanced_result]
    contexts = [context("clip-a")]

    if duplicate_target == "baseline":
        baseline_results.append(baseline)
    elif duplicate_target == "enhanced":
        enhanced_results.append(enhanced_result)
    else:
        contexts.append(context("clip-a"))

    with pytest.raises(ValueError, match="Duplicate"):
        aggregate(
            baseline_results,
            enhanced_results,
            contexts=contexts,
        )


def test_mismatched_method_clip_sets_are_rejected():
    baseline = detection(
        "clip-a",
        "baseline",
        ground_truth=1,
        predicted=1,
        matched=1,
    )
    enhanced_result = enhanced(
        "clip-b",
        ground_truth_labels=("correct",),
        predicted_labels=("correct",),
    )

    with pytest.raises(ValueError, match="clip sets"):
        aggregate(
            [baseline],
            [enhanced_result],
            contexts=[context("clip-a"), context("clip-b")],
        )


def test_inconsistent_ground_truth_counts_are_rejected():
    baseline = detection(
        "clip-a",
        "baseline",
        ground_truth=2,
        predicted=1,
        matched=1,
    )
    enhanced_result = enhanced(
        "clip-a",
        ground_truth_labels=("correct",),
        predicted_labels=("correct",),
    )

    with pytest.raises(ValueError, match="ground-truth counts"):
        aggregate([baseline], [enhanced_result])


def test_mixed_splits_are_rejected():
    baseline_results = [
        detection(
            "clip-a",
            "baseline",
            ground_truth=0,
            predicted=0,
            matched=0,
        ),
        detection(
            "clip-b",
            "baseline",
            ground_truth=0,
            predicted=0,
            matched=0,
        ),
    ]
    enhanced_results = [
        enhanced("clip-a"),
        enhanced("clip-b"),
    ]

    with pytest.raises(ValueError, match="Mixed evaluation splits"):
        aggregate(
            baseline_results,
            enhanced_results,
            contexts=[
                context("clip-a", split="development"),
                context("clip-b", split="test"),
            ],
        )


def test_unsupported_split_is_rejected():
    with pytest.raises(ValueError, match="Unsupported"):
        aggregate([], [], contexts=[], split="calibration")


def test_valid_non_default_tolerance_is_preserved():
    report = perfect_report(tolerance=0.75)

    assert report.event_tolerance_seconds == 0.75
    assert report.baseline_detection.evaluated_clips == 1
    assert report.enhanced_detection.evaluated_clips == 1


def test_mismatched_result_and_report_tolerances_are_rejected():
    baseline = detection(
        "clip-a",
        "baseline",
        ground_truth=0,
        predicted=0,
        matched=0,
        tolerance=0.4,
    )
    enhanced_result = enhanced(
        "clip-a",
        tolerance=0.4,
    )

    with pytest.raises(ValueError, match="tolerance"):
        aggregate(
            [baseline],
            [enhanced_result],
            tolerance=0.5,
        )


@pytest.mark.parametrize(
    "invalid_tolerance",
    [0.0, -0.1, float("nan"), float("inf")],
)
def test_non_positive_or_non_finite_tolerance_is_rejected(
    invalid_tolerance,
):
    with pytest.raises(
        ValueError,
        match="positive finite",
    ):
        aggregate(
            [],
            [],
            contexts=[],
            tolerance=invalid_tolerance,
        )


def test_inconsistent_source_fps_is_rejected():
    baseline = detection(
        "clip-a",
        "baseline",
        ground_truth=0,
        predicted=0,
        matched=0,
        fps=FPS,
    )
    enhanced_result = enhanced("clip-a", fps=FPS)

    with pytest.raises(ValueError, match="source FPS"):
        aggregate(
            [baseline],
            [enhanced_result],
            contexts=[context("clip-a", fps=20.0)],
        )


def test_empty_input_is_deterministic_and_not_misleading():
    report = aggregate([], [], contexts=[])

    assert report.ordered_clip_ids == ()
    assert report.baseline_detection.evaluated_clips == 0
    assert report.baseline_detection.pooled_event_precision == 0.0
    assert report.baseline_detection.pooled_event_recall == 0.0
    assert report.baseline_detection.pooled_event_f1 == 0.0
    assert report.baseline_detection.mean_signed_count_error is None
    assert report.baseline_detection.mean_absolute_count_error is None
    assert report.baseline_detection.exact_count_clip_accuracy is None
    assert report.enhanced_classification.accuracy is None
    assert report.enhanced_classification.macro_f1 is None
    assert report.per_clip_metrics == ()


def write_report(report, output_directory, **kwargs):
    timestamps = iter(
        [
            "2026-08-07T10:00:00+00:00",
            "2026-08-07T10:01:00+00:00",
            "2026-08-07T10:02:00+00:00",
        ]
    )
    provenance = kwargs.pop(
        "source_run_provenance",
        tuple(
            source_run_provenance(clip_id, method)
            for method in ("baseline", "enhanced")
            for clip_id in report.ordered_clip_ids
        ),
    )
    return write_formal_evaluation_report(
        report,
        output_directory=output_directory,
        run_id="fictional-evaluation",
        repository_root=output_directory,
        source_run_provenance=provenance,
        evidence_provenance=evidence_provenance(),
        software_versions={
            "python": "3.12.4",
            "packages": {"pandas": "2.2.2"},
        },
        git_state={
            "commit": "abc123",
            "branch": "codex/test",
            "dirty": False,
        },
        timestamp_factory=lambda: next(timestamps),
        **kwargs,
    )


def test_writer_produces_exact_confusion_matrix_layout(tmp_path):
    paths = write_report(perfect_report(), tmp_path)
    content = paths.confusion_matrix_csv.read_text(encoding="utf-8")

    assert content == (
        "ground_truth_class,correct,insufficient_depth,"
        "incomplete_extension,alignment_deviation,unscorable\n"
        "correct,1,0,0,0,0\n"
        "insufficient_depth,0,0,0,0,0\n"
        "incomplete_extension,0,0,0,0,0\n"
        "alignment_deviation,0,0,0,0,0\n"
        "unscorable,0,0,0,0,0\n"
    )


def test_writer_produces_exact_classification_layout(tmp_path):
    paths = write_report(perfect_report(), tmp_path)
    content = paths.classification_per_class_csv.read_text(encoding="utf-8")

    assert content == (
        "class_label,true_positives,false_positives,"
        "false_negatives,support,precision,recall,f1\n"
        "correct,1,0,0,1,1.0,1.0,1.0\n"
        "insufficient_depth,0,0,0,0,0.0,0.0,0.0\n"
        "incomplete_extension,0,0,0,0,0.0,0.0,0.0\n"
        "alignment_deviation,0,0,0,0,0.0,0.0,0.0\n"
        "unscorable,0,0,0,0,0.0,0.0,0.0\n"
    )


def test_writer_produces_exact_per_clip_layout(tmp_path):
    paths = write_report(perfect_report(), tmp_path)
    content = paths.per_clip_csv.read_text(encoding="utf-8")
    expected_header = ",".join(PER_CLIP_COLUMNS)

    rows = list(csv.DictReader(content.splitlines()))

    assert content.splitlines()[0] == expected_header
    assert len(rows) == 1
    assert rows[0]["clip_id"] == "clip-a"
    assert rows[0]["source_fps"] == "10.0"
    assert rows[0]["baseline_mean_measured_processing_time_ms"] == ""


def test_writer_produces_detection_recall_layout(tmp_path):
    paths = write_report(perfect_report(), tmp_path)
    content = paths.detection_recall_by_class_csv.read_text(encoding="utf-8")

    assert content == (
        "class_label,support,matched,missed,recall\n"
        "correct,1,1,0,1.0\n"
        "insufficient_depth,0,0,0,\n"
        "incomplete_extension,0,0,0,\n"
        "alignment_deviation,0,0,0,\n"
        "unscorable,0,0,0,\n"
    )


def test_writer_refuses_existing_partial_output_set(tmp_path):
    paths = formal_evaluation_output_paths(
        tmp_path,
        "fictional-evaluation",
    )
    paths.per_clip_csv.write_text("stale", encoding="utf-8")

    with pytest.raises(FileExistsError, match="per_clip"):
        write_report(perfect_report(), tmp_path)

    assert paths.per_clip_csv.read_text(encoding="utf-8") == "stale"
    assert not paths.report_json.exists()
    assert not paths.metadata_json.exists()


def test_writer_overwrites_the_complete_output_set(tmp_path):
    paths = formal_evaluation_output_paths(
        tmp_path,
        "fictional-evaluation",
    )

    for output_path in paths.all_paths():
        output_path.write_text("stale", encoding="utf-8")

    written = write_report(
        perfect_report(),
        tmp_path,
        overwrite=True,
    )

    assert written == paths
    assert all(path.exists() for path in paths.all_paths())
    assert all(
        path.read_text(encoding="utf-8") != "stale" for path in paths.all_paths()
    )


def test_writer_metadata_is_completed_atomically(tmp_path):
    report = perfect_report(tolerance=0.75)
    paths = write_report(report, tmp_path)
    report_document = json.loads(paths.report_json.read_text(encoding="utf-8"))
    metadata = json.loads(paths.metadata_json.read_text(encoding="utf-8"))

    assert report_document == report.to_dict()
    assert metadata["status"] == "completed"
    assert metadata["evaluation_run_id"] == ("fictional-evaluation")
    assert metadata["metadata_schema_version"] == 2
    assert metadata["report_schema_version"] == 2
    assert metadata["split"] == "development"
    assert metadata["ordered_clip_ids"] == ["clip-a"]
    assert metadata["evaluated_clip_count"] == 1
    assert report_document["event_tolerance_seconds"] == 0.75
    assert metadata["event_tolerance_seconds"] == 0.75
    assert "source_runs" not in report_document
    assert set(metadata["source_runs"]) == {
        "baseline",
        "enhanced",
    }
    assert metadata["formal_evidence"] == evidence_provenance().to_dict()
    assert (
        metadata["source_runs"]["baseline"][0]["source_run_id"] == "baseline-clip-a-run"
    )
    assert (
        metadata["source_runs"]["enhanced"][0]["consumed_output_csv"]["output_name"]
        == "repetition_csv"
    )
    assert metadata["software"]["python"] == "3.12.4"
    assert metadata["git"]["commit"] == "abc123"
    assert metadata["timestamps"] == {
        "started_utc": "2026-08-07T10:00:00+00:00",
        "completed_utc": "2026-08-07T10:01:00+00:00",
    }
    assert "failed_utc" not in metadata["timestamps"]
    assert set(metadata["outputs"]) == set(paths.named_paths())
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob(".*.tmp"))


def test_writer_requires_complete_source_provenance(tmp_path):
    with pytest.raises(
        ValueError,
        match="Enhanced source-run provenance",
    ):
        write_report(
            perfect_report(),
            tmp_path,
            source_run_provenance=(source_run_provenance("clip-a", "baseline"),),
        )

    paths = formal_evaluation_output_paths(
        tmp_path,
        "fictional-evaluation",
    )
    assert not any(path.exists() for path in paths.all_paths())


def test_writer_rejects_absolute_source_provenance_path(tmp_path):
    baseline = source_run_provenance("clip-a", "baseline")
    unsafe_baseline = replace(
        baseline,
        source_metadata_path=str((tmp_path / "source.json").resolve()),
    )

    with pytest.raises(ValueError, match="privacy-safe"):
        write_report(
            perfect_report(),
            tmp_path,
            source_run_provenance=(
                unsafe_baseline,
                source_run_provenance("clip-a", "enhanced"),
            ),
        )


def test_failed_write_records_failed_metadata(
    tmp_path,
    monkeypatch,
):
    def fail_write(*args, **kwargs):
        raise OSError("fictional report write failure")

    monkeypatch.setattr(
        formal_reporting,
        "_write_report_temporary_files",
        fail_write,
    )

    with pytest.raises(OSError, match="fictional"):
        write_report(perfect_report(), tmp_path)

    paths = formal_evaluation_output_paths(
        tmp_path,
        "fictional-evaluation",
    )
    metadata = json.loads(paths.metadata_json.read_text(encoding="utf-8"))
    assert metadata["status"] == "failed"
    assert metadata["timestamps"] == {
        "started_utc": "2026-08-07T10:00:00+00:00",
        "failed_utc": "2026-08-07T10:01:00+00:00",
    }
    assert "completed_utc" not in metadata["timestamps"]
    assert metadata["failure"] == {
        "error_type": "OSError",
        "message": "fictional report write failure",
    }
    assert all(
        not path.exists() for path in paths.all_paths() if path != paths.metadata_json
    )


def test_deterministic_metric_files_repeat_identically(tmp_path):
    report = perfect_report()
    first_paths = write_report(report, tmp_path / "first")
    second_paths = write_report(report, tmp_path / "second")
    first_metric_paths = first_paths.all_paths()[:-1]
    second_metric_paths = second_paths.all_paths()[:-1]

    assert [path.read_bytes() for path in first_metric_paths] == [
        path.read_bytes() for path in second_metric_paths
    ]
