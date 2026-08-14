# Repetition-event detection evaluation

## Scope

This evaluation layer extracts repetition completion events for the existing
baseline and enhanced methods, loads evaluable annotated attempts, performs
deterministic one-to-one matching and reports detection-only metrics.

It does not calculate form-classification accuracy, confusion matrices, macro
F1, form-category performance, runtime comparisons or plots.

## Event definitions

### Baseline prediction

A baseline completion event occurs on a frame where
`baseline_rep_count` increases by exactly one relative to the preceding row
for the same run and clip. The resulting count is the predicted repetition ID.
No repetition window or form class is inferred for the baseline.

The extractor requires `run_id`, `clip_id`, `frame_index` and
`baseline_rep_count`. It retains `video_timestamp_ms` when present and can
derive a timestamp from `source_fps` when necessary.

### Enhanced prediction

Each enhanced repetition CSV row is one prediction. Its `end_frame` is the
completion event. The event retains the existing start, bottom and end frames,
predicted repetition ID and predicted class. The class is carried as
traceability data but is not evaluated by this detection-only layer.

A completion timestamp is taken from `completion_timestamp_ms` if a future
input provides it; otherwise it is derived as:

```text
completion_timestamp_ms = end_frame / source_fps * 1000
```

The manifest supplies source FPS to the CLI.

### Ground truth

The annotation and manifest are passed through the existing schema validator.
Every row with `is_evaluable_attempt=true` becomes a ground-truth event at
`completion_end_top_frame`. Ordinary evaluable attempts remain included even
if neither method predicts them. Rows that the annotation protocol explicitly
marks as ambiguous fragments are excluded.

## Validation

Baseline extraction rejects:

- missing required columns or empty run/clip identifiers;
- invalid, decreasing or repeated frame indices within a run and clip;
- negative, non-integer or decreasing cumulative counts;
- cumulative-count jumps larger than one;
- duplicate `(clip_id, predicted_rep_id)` events;
- duplicate completion frames within a clip;
- malformed timestamps or source FPS.

Enhanced loading rejects:

- missing required columns or empty identifiers/classes;
- invalid repetition IDs or frame indices;
- frames not satisfying `start <= bottom <= end`;
- duplicate `(clip_id, predicted_rep_id)` events;
- malformed timestamps or source FPS.

Ground-truth loading preserves all decisions enforced by the annotation
protocol, including duplicate-ID, unknown-clip, visibility, ambiguity and
frame-bound validation.

## Deterministic one-to-one matching

The primary tolerance is frozen at `0.5` seconds following development-only
sensitivity analysis. Final-test evaluation must use this frozen value.

When both events have timestamps, candidate eligibility and error use their
timestamp difference. Otherwise:

```text
tolerance_frames = ceil(tolerance_seconds * source_fps)
timing_error_seconds =
    (predicted_frame - annotated_frame) / source_fps
```

The matcher uses chronological dynamic programming with this ordered
objective:

1. maximise the number of matched pairs;
2. among maximum-cardinality solutions, minimise total absolute timing error;
3. for an exact tie, choose the lexicographically earliest chronological pair
   sequence.

This produces deterministic, non-crossing, one-to-one matches. Signed error is
prediction time minus annotation time. Unmatched predictions are extras;
unmatched evaluable annotations are misses.

## Detection summary

For one clip and method, the CLI reports:

- ground-truth and predicted event counts;
- signed and absolute count error;
- matched, missed and extra events;
- event precision, recall and F1;
- mean signed and absolute completion-timing error in seconds;
- tolerance in seconds and frames.

Precision, recall and F1 use `0.0` when their denominator is zero. Mean timing
errors are `null` when no events match.

## Fictional command example

From the repository root in PowerShell:

```powershell
$env:PYTHONPATH = "$PWD\src"

& ".\.venv\Scripts\python.exe" `
  src\evaluation\detection_evaluation.py `
  --method enhanced `
  --predictions `
    "experiments\outputs\fictional_run_enhanced_repetitions.csv" `
  --manifest "data\manifests\fictional_manifest.csv" `
  --annotations "data\annotations\fictional_annotations.csv" `
  --clip-id "fictional_clip_001" `
  --tolerance-seconds 0.5
```

The example names are fictional and do not represent an existing recording or
annotation. The command writes the JSON report to standard output and does not
modify its inputs.
