# Cross-clip formal-evaluation reporting

## Scope

The reporting foundation aggregates already evaluated per-clip results. It
does not load videos, run pose estimation, discover result files or repeat
event matching.

Baseline reporting is detection-only. Enhanced reporting contains detection,
detection recall stratified by ground-truth form class, and classification for
one-to-one matched repetitions. Baseline warnings are never interpreted as
repetition classes, and misses or extras never enter a confusion matrix.

`EvaluationClipContext` supplies the clip ID, frozen dataset split, source FPS
and recorded descriptive evidence that are not stored in the existing
per-clip metric types. The aggregator continues to consume the
repository-native `DetectionSummary` and `EnhancedClipEvaluation` objects
directly.

## Input validation

One report requires:

- unique, non-blank clip IDs;
- identical baseline, enhanced and context clip sets;
- one supported split (`development` or `test`);
- the same ground-truth count for both methods on each clip;
- source FPS consistent with each method's recorded frame tolerance;
- one positive finite supplied event tolerance used by every clip and method;
- internally consistent native detection and enhanced evaluation results.

No unmatched clip or invalid result is silently removed. Development and test
clips cannot be mixed in one report.

The current provisional default is `0.5` seconds. The reporting layer does not
select or tune this value: it accepts a supplied tolerance, verifies the same
value and corresponding frame tolerance for every result, and records it in
the report and metadata. Development-set evidence must be used to choose and
freeze the final tolerance before final-test evaluation begins.

## Pooled detection metrics

Counts are summed across clips separately for baseline and enhanced methods.
The pooled metrics are:

```text
precision = total matches / (total matches + total extras)
recall = total matches / (total matches + total misses)
F1 = 2 * precision * recall / (precision + recall)
```

Zero denominators follow the established detection convention and produce
`0.0`. Pooled F1 is calculated from pooled precision and recall; per-clip F1
values are not averaged.

Signed count error is `predicted - ground truth`. The report contains the sum
of signed errors, the arithmetic mean of per-clip signed errors, and the
arithmetic mean of per-clip absolute errors. Exact-count clip accuracy is the
number of clips with zero signed count error divided by the number of clips.

Count-error means and exact-count accuracy are `None` when no clips exist.

## Timing aggregation

Completion-timing means are weighted by each clip's number of matched events:

```text
weighted mean =
    sum(clip timing mean * clip matched events)
    / total matched events
```

This is equivalent to pooling all matched timing observations and prevents a
clip with one match from receiving the same weight as a clip with many matches.
Timing means are `None` when there are no matched timing observations.

Processing performance is a separate measurement. `processing_time_ms` is the
runner's explicitly timed per-frame computer-vision/analysis region; it
excludes video decoding, display, CSV serialization and setup/initialisation.
Per-clip mean and median use recorded frame observations. Measured analysis
throughput is:

```text
timed frame observations * 1000 / sum(processing_time_ms)
```

Aggregate processing mean, median and throughput pool every recorded frame
observation, so clips are frame-weighted rather than given equal weight.
`source_fps` remains the video time-base and is reported separately; it is not
measured analysis throughput.

## Descriptive availability and stability evidence

Availability uses explicit frame counts and denominators. Baseline pose,
elbow-angle, body-alignment-angle and valid selected-side availability are
derived from its recorded frame values. Enhanced pose, elbow-valid,
alignment-valid and selected-side availability use its established boolean and
side columns. Aggregate rates sum available-frame numerators and evidence-frame
denominators before division. A historical output without a column reports
`None` with a zero evidence denominator rather than an invented zero rate.

Baseline side changes count transitions in its instantaneous/stateless
`selected_side` sequence. Enhanced side changes sum its stable selector's
recorded `side_changed` events. These are descriptive stability measures, not
accuracy measures, and the two semantics remain labelled separately.

Enhanced predicted unscorable rate is the number of completed/predicted
repetitions whose canonical `predicted_class` is `unscorable`, divided by all
enhanced completed/predicted repetitions. It is independent of GT ambiguous
fragments. Model-derived alignment coverage retains the existing
`alignment_valid_ratio`: per-repetition values remain in the provenance-bound
enhanced repetition CSV, while the report gives per-clip and aggregate means.
The aggregate mean pools available repetition observations, so it is weighted
per repetition, not per clip.

Human alignment-evidence adequacy is reported separately from model coverage.
It uses validated annotation `source_video_visibility_status` on evaluable
attempts: `sufficient` and `partially_obscured` are adequate under the existing
protocol, while `insufficient` is inadequate. Ambiguous fragments are not in
this denominator. This describes independent source-video assessability and
must not be inferred from model landmark coverage.

## Enhanced pooled classification

The raw enhanced confusion-matrix counts are summed cell by cell in
`SUPPORTED_FORM_CLASSES` order. Classification precision, recall, F1,
accuracy and macro F1 are then recomputed from that pooled matrix using the
shared classification evaluator.

Rows are ground truth and columns are predictions. Macro F1 is the unweighted
mean of pooled per-class F1 values only for classes with pooled ground-truth
support greater than zero. Zero-support classes remain visible in the matrix
and per-class table. No per-clip classification metric is averaged to create a
pooled metric.

The pooled classification count must equal the total enhanced matched-event
count. With no matched classifications, the matrix is zero and accuracy and
macro F1 are `None`.

## Detection recall by ground-truth class

Enhanced class-stratified detection counts are summed across clips. For each
supported class:

```text
recall = matched ground-truth repetitions / ground-truth support
```

Recall is `None` when pooled support is zero and `0.0` when positive support is
entirely missed. Extra predictions do not affect this metric. Aggregated class
support, match and miss totals are checked against overall enhanced detection.

## Deterministic report result

`aggregate_formal_evaluation` returns `FormalEvaluationReport`, containing:

- report schema version;
- split and event tolerance;
- clip IDs sorted lexicographically;
- baseline and enhanced pooled detection;
- enhanced pooled classification;
- enhanced detection recall by ground-truth class;
- baseline/enhanced processing, availability and side-stability evidence;
- enhanced unscorable and model alignment-coverage evidence;
- independent human annotation evidence adequacy;
- deterministic per-clip audit rows.

The result contains no timestamp. Repeating the same aggregation produces the
same `to_dict()` content, ready for `json.dumps`.

An empty input collection is supported. It produces zero counts, `0.0` pooled
detection rates, `None` clip-level means and exact-count accuracy, empty
per-clip rows, a zero classification matrix, and `None` classification
accuracy and macro F1.

## Output files

`write_formal_evaluation_report` writes this complete set:

1. `<run_id>_formal_evaluation.json`
2. `<run_id>_per_clip_metrics.csv`
3. `<run_id>_classification_confusion_matrix.csv`
4. `<run_id>_classification_per_class.csv`
5. `<run_id>_detection_recall_by_class.csv`
6. `<run_id>_evaluation_metadata.json`

The per-clip CSV contains the count, match, miss, extra and event metrics for
both methods; enhanced matched-classification results; source FPS; processing,
availability and side-stability evidence; enhanced predicted-unscorable and
alignment-coverage evidence; and human evidence adequacy. Rows are ordered by
clip ID.

The confusion-matrix CSV starts with `ground_truth_class`; the remaining
columns and all rows use supported class order. The classification-per-class
CSV records label, TP, FP, FN, support, precision, recall and F1. The detection
recall CSV records label, support, matched, missed and recall. `None` is
written as an empty CSV field.

All metric files use UTF-8, LF line endings, stable headers and deterministic
row ordering.

## Collision policy and atomicity

By default, any existing member of the complete output set causes a clear
failure before writing. `overwrite=True` applies to the entire set; it is not
possible to replace only selected files through this API. Files are never
opened in append mode.

Metric files are fully staged to same-directory temporary paths before being
replaced. Metadata is written atomically and finalized as `completed` only
after every metric file has been installed. A practical write failure records
`failed` metadata without a completion timestamp, so a partial set cannot
appear successfully complete.

## Evaluation-run provenance

The metadata file is separate from deterministic metric content. It records:

- evaluation run ID and metadata/report schema versions;
- split, ordered clip IDs and evaluated clip count;
- event tolerance and generated output paths;
- Python and relevant package versions;
- Git commit, branch and dirty state;
- start and completion or failure timestamps;
- `running`, `completed` or `failed` lifecycle status.

It additionally contains baseline and enhanced source-run records ordered by
clip ID. Each record binds the result to the source clip, method, run ID and
split; the source metadata file path and SHA-256; the consumed baseline frame
CSV or enhanced repetition CSV path and SHA-256; the input-video SHA-256
inherited from source metadata; source Git commit and dirty state where
present; and a canonical resolved-configuration SHA-256 where available.
Each record also binds the frame CSV used for descriptive reporting. A separate
`formal_evidence` object binds the exact manifest, annotation CSV and review
record by privacy-safe identity and SHA-256, and records the frozen annotation
hash and completed review status.

Source metadata and CSV hashing is streamed. Resolved configuration hashing
uses UTF-8 canonical JSON with sorted keys and stable compact separators.
Repository-contained source paths are recorded relative to the repository with
POSIX separators. External source paths are represented by basename only, so
absolute machine-specific paths are not published; hashes retain exact file
identity.

The source/evidence records are lifecycle provenance only. They are not added to
`FormalEvaluationReport.to_dict()` or any metric file, so identical metric
inputs continue to produce identical deterministic metric content. Missing
upstream Git or resolved-configuration values are recorded as `None`, not
fabricated. Input videos are not rehashed by evaluation; their existing source
metadata hashes are preserved.

## Current limitations and evaluation warning

The formal CLI consumes explicitly supplied completed-run metadata and does not
discover results, process videos, create plots or calculate confidence
intervals. Historical source outputs that lack newer descriptive columns
retain unavailable values rather than being rewritten.

The CLI supports guarded final-test execution through `--allow-final-test`, but
only after development decisions, frozen evidence and the final event tolerance
have been fixed as required by the formal execution protocol.

All development decisions—including thresholds, annotation rules and class
priority—must be frozen before the final test results are evaluated. The
current `0.5`-second tolerance is a provisional default, not a permanently
fixed reporting constraint. Its final value must be justified and frozen
using development evidence before final-test evaluation begins. Final-test
outputs must not be used for tuning.
