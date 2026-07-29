# Baseline-versus-enhanced comparison design

## A. Baseline purpose

Scientifically, the baseline represents a deliberately simple raw-threshold
comparator. It asks:

> How well can push-up repetitions be detected using unsmoothed frame-level
> elbow angles and two fixed threshold crossings, without enhanced temporal
> reasoning?

Technically, the baseline uses raw MediaPipe-derived angles, instantaneous
visibility-based elbow-side selection, and a strict `top -> bottom -> top`
counter. It does not use smoothing, enhanced hysteresis, consecutive-frame
confirmation, missing-frame tolerance, or explicit movement phases.

The baseline is therefore a strict repetition detector. It is not a general
attempt segmenter and does not perform repetition-level form classification.
Its 150-degree top and 100-degree bottom thresholds are provisional operational
project rules, not universal definitions of correct push-up form.

## B. Baseline outputs

The existing outputs have the following valid interpretations:

| Output | Valid interpretation | Invalid interpretation |
| --- | --- | --- |
| Frame-level position | The counter's sticky `unknown`, `top`, or `bottom` state | A complete physical phase or current posture on every frame |
| Frame-level warnings | Raw diagnostic messages based on the current angle and sticky state | Repetition-level form classifications |
| Cumulative repetition count | Number of strict `top -> bottom -> top` threshold cycles | Number of all visible push-up attempts |
| Inferred completion event | A frame where `baseline_rep_count` increases | The exact biomechanical end of every attempted repetition |
| Repetition-level measurements | None currently produced | Depth, extension, alignment coverage, or duration for a baseline repetition |

A baseline completion event can be inferred exactly from the recorded frame
CSV as a row where `baseline_rep_count` increases from the previous row. The
new count supplies the sequential baseline repetition ID. A preceding
`top -> bottom` transition can be reconstructed as threshold-event metadata,
but this does not create a full baseline repetition measurement window.

## C. Warning semantics

Baseline warnings should remain diagnostic-only frame messages and should be
excluded from the primary formal repetition-classification evaluation.

They should not be redesigned or removed because they are part of the existing
simple prototype. Redesigning them into temporally aggregated repetition rules
would add capabilities that belong to the enhanced method.

A minimal secondary analysis may report warning-frame counts or rates among
frames where the required feature is valid. Warning rates may also be described
within manually annotated attempt windows or stratified by ground-truth class.
Such results must be labelled exploratory frame-level diagnostics and must not
be converted into baseline class predictions or a baseline confusion matrix.

This preserves the baseline while avoiding the misleading interpretation that
ordinary descent and ascent warning frames classify an entire repetition.

## D. Repetition definition

A baseline repetition qualifies only when all of the following occur in order:

1. A valid elbow observation reaches at least 150 degrees, placing the counter
   in `top`.
2. A later valid observation reaches at most 100 degrees, causing a
   `top -> bottom` transition and setting the bottom-reached flag.
3. A later valid observation reaches at least 150 degrees, causing
   `bottom -> top`, incrementing the count, and producing the completion event.

Intermediate values do not change the sticky position.

- A shallow attempt that never reaches 100 degrees produces no baseline
  repetition and is a missed annotated event.
- An incomplete-extension attempt that never reaches 150 degrees may not
  initialise or complete a baseline cycle. It is not automatically an
  `incomplete_extension` prediction.
- A missing elbow observation leaves position, bottom-reached state, and count
  unchanged. The baseline may complete a repetition after a tracking gap.
- A movement fragment that does not complete the full threshold sequence is
  not counted. An unfinished state at the end of a clip produces no event.
- Every additional complete threshold cycle produces an additional event.
  Threshold noise can therefore create extra events because the baseline has
  no consecutive-frame confirmation.
- An initial observation at or below 100 degrees cannot later count unless a
  valid top state was observed first.
- Because there is no timeout, a bottom-reached flag may persist until a much
  later top crossing. A delayed event should be handled by event matching, not
  manually reassigned.

The analyser is reset at the start of every clip.

## E. Fair shared metrics

The same manually annotated clips should support direct comparison of:

### Repetition detection

- predicted repetition count per clip;
- signed count error (`predicted - ground truth`);
- absolute count error and mean absolute count error;
- exact-count clip accuracy;
- matched completion events;
- missed annotated events;
- extra predicted events;
- event precision, recall, and F1;
- signed and absolute completion-timing error for matched events.

Detection recall should also be stratified by ground-truth form class. This
shows whether the strict baseline systematically misses shallow or
incomplete-extension attempts without pretending that it classified them.

### Runtime

- processing time per frame;
- median and interquartile range of processing time;
- effective processing throughput;
- total clip processing time.

Runs must use the same recordings, resolution, MediaPipe settings, hardware,
display setting, timing boundaries, and frozen software/configuration version.

### Feature availability

- pose-detected frames;
- elbow-valid frames and rate;
- alignment-valid frames and rate;
- frames with no selected elbow side;
- side-selection changes where supported.

These metrics describe each method's effective input availability. They do not
imply that the selectors behave identically. Sticky baseline positions and
enhanced phases must not be compared as equivalent frame labels.

## F. Classification comparison

Direct repetition-classification comparison is not defensible.

The enhanced method segments attempted repetitions using permissive temporal
thresholds and then classifies them. The baseline may fail to detect the
attempts most likely to show insufficient depth or incomplete extension, while
its warnings are frame-level and it has no repetition window, single-label
priority, or evidence-coverage rule.

Constructing baseline repetition classes from warning strings would introduce
selection bias and silently add a classifier after implementation.

The approved alternative is:

1. Compare repetition detection for both methods, including detection recall
   stratified by annotated form class.
2. Evaluate the enhanced repetition classifier against ground-truth classes
   for matched enhanced events.
3. Report baseline warning rates only as secondary diagnostics.
4. State explicitly that baseline classification accuracy, macro F1, and a
   baseline confusion matrix do not exist.

An oracle-window analysis that applies raw rules inside ground-truth windows
could be considered only as a separately named auxiliary analysis. It would
not represent the deployed baseline or the primary end-to-end comparison.

## G. Event matching

The baseline predicted completion frame is the frame where cumulative
`baseline_rep_count` increases. The enhanced predicted completion frame is
`CompletedRepetition.end_frame`, the final frame confirming return to top.

The annotated reference should be the first frame after ascent at which an
annotator judges that the attempt has returned to its end/top position. This
must be based on the visible movement rather than either method's numeric
threshold.

The provisional matching tolerance is plus or minus 0.5 seconds. Timestamps
should be used where possible. If matching by frame index:

```text
tolerance_frames = ceil(0.5 * source_fps)
```

The tolerance must be frozen using development data and annotation
repeatability before final test results are examined.

Matching must be chronological and one-to-one:

- candidate pairs are permitted only within the tolerance;
- the selected matching first maximises the number of pairs and then minimises
  total absolute timing error;
- only one prediction may match an annotation;
- unmatched predictions are extra events;
- unmatched annotations are missed events;
- a delayed baseline completion outside tolerance remains extra while the
  original annotation remains missed.

Signed timing error is `predicted time - annotated time`.

## H. Annotation implications

Annotations describe visible attempts independently of either method's output.
Every evaluable attempt should record:

- clip and ground-truth attempt identifiers;
- start/top, bottom/turnaround, and completion/end-top frames;
- whether it is an evaluable attempt or an ambiguous fragment;
- a single ground-truth class;
- individual deviation flags;
- source-video visibility status;
- annotator notes where necessary.

Shallow and incomplete-extension attempts remain annotated attempts when they
contain an identifiable descent and return. They remain part of the
ground-truth count even when the baseline cannot cross its strict thresholds.

Computer-vision tracking loss does not remove an otherwise visible
human-annotatable attempt. If the source video itself provides insufficient
evidence, the attempt is marked `unscorable`. Clip-boundary fragments or
ambiguous movements are recorded separately and handled by a preregistered
inclusion rule. The same annotations and inclusion decisions apply to both
methods.

## I. Recommended minimal code changes

### Required after approval

- Add an evaluation annotation schema and validator.
- Add a baseline event extractor based on count-increment rows without changing
  the counter.
- Add an enhanced event loader based on `end_frame`.
- Implement the frozen one-to-one event matcher and count/event metrics.
- Implement enhanced-only repetition classification metrics.
- Add detection recall stratified by annotated form class for both methods.
- Add common runtime and feature-availability summaries.
- Validate identifiers, monotonic baseline counts, and duplicate events.
- Add focused tests for matching tolerances, ambiguous matches, missed events,
  extra events, and delayed baseline completions.
- Record evaluation configuration, source FPS, method/version identifier, and
  tolerance with every result.

### Optional

- Export a derived baseline event CSV with completion frame and timestamp.
- Include the preceding bottom-transition frame as baseline-specific metadata.
- Produce descriptive baseline warning-frame rates.
- Report confidence intervals across clips if the test-set size supports them.
- Perform repeated runtime runs and report median throughput.
- Add an explicitly labelled oracle-window raw-rule analysis as supplementary
  evidence only.

### Rejected

- Adding smoothing to the baseline.
- Adding hysteresis, consecutive-frame confirmation, or missing-frame timeout.
- Replacing strict thresholds with enhanced segmentation thresholds.
- Adding enhanced movement phases or repetition aggregation to the baseline.
- Turning baseline warnings into repetition classes.
- Changing thresholds to improve evaluation results.
- Selecting the matching tolerance on final-test results.
- Treating sticky baseline positions as equivalent to enhanced phases.
- Applying enhanced side selection while continuing to call the result the
  original baseline.

## J. Dissertation justification

The comparison remains meaningful because both methods process the same
manually annotated recordings under the same controlled conditions while
representing intentionally different levels of algorithmic complexity.

The baseline measures strict raw-threshold crossing. The enhanced method
measures the added capability of visibility handling, stable side selection,
smoothing, temporal segmentation, and repetition-level aggregation. Missed
shallow or incomplete-extension attempts are therefore legitimate baseline
outcomes rather than protocol defects.

Both methods can be compared directly for repetition detection, event timing,
runtime, and feature availability. Only the enhanced method is evaluated for
repetition-level form classification. Baseline warnings remain diagnostic
frame messages. This provides a transparent comparison without upgrading the
baseline or claiming superiority over a baseline classifier that does not
exist.
