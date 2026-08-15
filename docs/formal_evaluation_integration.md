# Enhanced per-clip formal-evaluation integration

## Purpose and scope

The per-clip integration layer composes the existing event-detection and
enhanced classification foundations for already loaded event objects. It does
not load CSVs, run videos, write evaluation outputs or aggregate across clips.

Formal repetition classification remains enhanced-only. Baseline repetitions
may be evaluated through the separate detection pipeline, but baseline frame
warnings are not repetition classes and cannot enter this integration layer.

## Evaluation sequence

For one clip, `evaluate_enhanced_clip` performs these steps:

1. Validate that the inputs are ground-truth and enhanced event objects with
   supported form labels.
2. Call the existing per-clip detection evaluator with method `enhanced`.
3. Retain the chronological one-to-one mapping returned by the existing event
   matcher.
4. Read the ground-truth and predicted labels directly from each matched pair,
   preserving matcher order.
5. Pass only those paired label sequences to `evaluate_classification`.
6. Summarise detection recall by ground-truth form class.

There is no second match for classification and no pairing by repetition ID or
input-list position. The event matcher remains the single source of truth.

The primary event tolerance is frozen at `0.5` seconds. A caller may supply
another positive finite value; cross-clip reporting verifies that every
baseline and enhanced result uses the same supplied value and matching frame
tolerance. Alternative values are development sensitivity evidence only, and
the completed final-test evaluation used the frozen primary value.

## Misses and extras

An unmatched ground-truth event remains a detection miss. An unmatched
enhanced prediction remains a detection extra. Neither is inserted into the
classification confusion matrix, so the number of evaluated classification
rows always equals the number of matched events.

The result retains matched identifiers and timing errors, plus unmatched
prediction and annotation identifiers, so the matching decision is auditable.

## Detection recall by ground-truth class

Each class in `SUPPORTED_FORM_CLASSES` appears in deterministic reporting
order with:

- ground-truth support;
- matched ground-truth repetitions;
- missed ground-truth repetitions;
- recall, calculated as `matched / support`.

When class support is zero, recall is `None`: no detection opportunity existed
for that class. When support is positive but every event is missed, recall is
`0.0`. Extra predictions do not affect this ground-truth-class recall.

The integration validates that per-class support, match and miss totals equal
their corresponding clip-level detection totals.

## Deterministic result structure

`EnhancedClipEvaluation.to_dict()` returns JSON-compatible nested records:

- `detection`: the existing `DetectionSummary`, including configured tolerance;
- `matched_pairs`: matched identifiers, classes, frames and timing errors;
- `unmatched_prediction_ids`;
- `unmatched_ground_truth_attempt_ids`;
- `classification`: the existing enhanced classification evaluation;
- `detection_recall_by_ground_truth_class`.

Confusion-matrix rows remain ground truth and columns remain enhanced
predictions. The class order is imported from the existing classification
module rather than duplicated.

## Empty inputs

With no annotations and no predictions, detection follows its existing empty
behavior, the matched-pair list is empty, classification accuracy and macro F1
are `None`, and every class-recall row contains zero counts and recall `None`.

Predictions without annotations remain extras and do not enter classification.
Annotations without predictions remain misses; supported classes with positive
ground-truth support then have recall `0.0`.

## Current limitations

This layer deliberately provides no:

- cross-clip aggregation;
- command-line interface;
- CSV or JSON output writing;
- plots or final result tables;
- confidence intervals;
- runtime evaluation;
- final-test execution.

The subsequent [cross-clip reporting foundation](formal_evaluation_reporting.md)
pools these native per-clip results without repeating event matching.
