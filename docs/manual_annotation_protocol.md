# Manual annotation protocol

## Purpose

This protocol defines how recorded push-up attempts are entered into the
formal-evaluation manifest and repetition annotation CSVs. Annotation is
independent of baseline or enhanced predictions. The same frozen annotations
must be used for both methods.

The example files are fictional:

- `data/examples/manifests/example_dataset_manifest.csv`
- `data/examples/annotations/example_repetition_annotations.csv`

They demonstrate the schema only and do not describe existing recordings or
participants.

## Dataset manifest schema

Each manifest row represents one recorded clip.

| Column | Type | Rule |
| --- | --- | --- |
| `clip_id` | text | Required and unique |
| `split` | enum | `development` or `test` |
| `video_path` | text | Required project-relative local path |
| `participant_id` | text | Required anonymised identifier |
| `camera_view` | enum | `side` or `side_diagonal` |
| `source_fps` | number | Finite and greater than zero |
| `frame_count` | integer | Greater than zero |
| `width_px` | integer | Greater than zero |
| `height_px` | integer | Greater than zero |
| `recording_condition` | text | Required concise controlled-condition description |
| `notes` | text | Optional contextual notes; do not include identifying information |

Assign the development/test split before formal analysis. Development clips may
support calibration and protocol refinement. Test clips must not be used to
tune thresholds, matching tolerances, or decision rules.

Formal execution requires both methods to cover every manifest clip in the
chosen split. It also checks manifest FPS, frame count, width and height against
both completed-run provenance records rather than silently preferring either
source.

## Repetition annotation schema

Each row represents either one evaluable attempt or one ambiguous fragment.

| Column | Type | Rule |
| --- | --- | --- |
| `clip_id` | text | Must exist in the manifest |
| `ground_truth_attempt_id` | text | Unique within the clip |
| `is_evaluable_attempt` | boolean | `true` only for a complete annotatable attempt |
| `ambiguity_flag` | boolean | Exact inverse of `is_evaluable_attempt` |
| `start_top_frame` | nullable integer | Required for evaluable attempts |
| `bottom_turnaround_frame` | nullable integer | Required for evaluable attempts |
| `completion_end_top_frame` | nullable integer | Required for evaluable attempts |
| `ground_truth_class` | enum | `correct`, `insufficient_depth`, `incomplete_extension`, `alignment_deviation`, or `unscorable` |
| `insufficient_depth_flag` | boolean | Individual observable deviation flag |
| `incomplete_extension_flag` | boolean | Individual observable deviation flag |
| `alignment_deviation_flag` | boolean | Individual observable deviation flag |
| `source_video_visibility_status` | enum | `sufficient`, `partially_obscured`, or `insufficient` |
| `annotator_notes` | text | Required for ambiguous or unscorable rows |

Frame indices are zero-based and must lie inside the clip. Where present, they
must satisfy:

```text
start_top_frame <= bottom_turnaround_frame <= completion_end_top_frame
```

## Identifying an attempt

An evaluable attempt is a visible down-and-up push-up movement with:

1. an identifiable starting/top posture;
2. an identifiable lowest point or direction change;
3. an identifiable return/end-top posture.

An attempt remains evaluable when it is shallow or has incomplete extension.
Those deviations may cause a method not to segment it, but they do not remove
it from ground truth.

Do not use the baseline 150/100-degree thresholds or enhanced segmentation
thresholds to decide whether an attempt exists.

## Selecting event frames

Review the source video frame by frame:

1. `start_top_frame`: the representative starting/top frame immediately before
   the visible descent.
2. `bottom_turnaround_frame`: the lowest visible point or the frame where
   descent changes to ascent.
3. `completion_end_top_frame`: the first frame after ascent where the attempt
   has visibly returned to its end/top posture.

Use the visible movement rather than method outputs or logged angle thresholds.
Do not view baseline or enhanced predictions while assigning ground truth.

## Evaluable attempts and ambiguous fragments

Set:

```text
is_evaluable_attempt=true
ambiguity_flag=false
```

only when all three event frames can be identified. All three frame columns are
then required.

Use:

```text
is_evaluable_attempt=false
ambiguity_flag=true
ground_truth_class=unscorable
```

for a clip-boundary fragment, incidental movement, or sequence that cannot
confidently be identified as a complete attempt. Set all deviation flags to
`false`, populate every identifiable frame field, and explain the ambiguity in
`annotator_notes`. At least one frame field is required to locate the fragment.

Ambiguous fragments are retained for auditability but are not ground-truth
repetitions for count or event evaluation.

For formal execution, every selected manifest clip must have at least one
annotation row to prove that it was manually reviewed. An ambiguous row counts
as this review evidence while remaining excluded from event metrics. Do not
invent an annotation for a genuine zero-attempt clip: the current annotation
workflow has no explicit review-complete representation for such clips, so
zero-attempt formal clips are not currently supported.

## Visibility and unscorable attempts

- `sufficient`: the source video clearly supports the temporal and form labels.
- `partially_obscured`: some evidence is obscured, but the annotation remains
  defensible.
- `insufficient`: the movement timing is identifiable, but the source video
  cannot support a reliable form class.

An evaluable attempt with `insufficient` source visibility must use
`ground_truth_class=unscorable` and include a note. Conversely, an evaluable
`unscorable` attempt must have `insufficient` visibility. An unscorable row
must not assert a deviation flag because the available source evidence does
not support a defensible form-deviation decision.

Computer-vision tracking loss alone is not a reason to mark source-video
visibility insufficient if a human can still annotate the source recording.

## Form labels and deviation flags

Record every visible deviation using the three boolean flags. Assign the
single ground-truth class with this frozen priority:

1. `insufficient_depth`
2. `incomplete_extension`
3. `alignment_deviation`
4. `correct` when no deviation flag is true

This retains multi-deviation evidence while producing the single label required
by the current classifier evaluation. The labels describe project-defined
observable categories under controlled recording conditions; they are not
medical or universal form judgements.

## Annotation workflow

1. Confirm the clip has one manifest row and the correct frozen split.
2. Review the source video without method predictions.
3. Record every evaluable attempt in chronological order.
4. Record ambiguous fragments separately rather than silently dropping them.
5. Recheck event-frame ordering and source visibility.
6. Recheck the single label against all deviation flags and the priority above.
7. Add notes for ambiguity, insufficient visibility, or difficult decisions.
8. Run schema validation.
9. Resolve validation errors before freezing the annotations.
10. Confirm every clip in the formal split has at least one annotation row.
11. Preserve the frozen annotation file unchanged during formal evaluation.

## Validation command

From PowerShell:

```powershell
$env:PYTHONPATH = "$PWD\src"

& ".\.venv\Scripts\python.exe" `
  src\evaluation\dataset_validation.py `
  --manifest "data\examples\manifests\example_dataset_manifest.csv" `
  --annotations "data\examples\annotations\example_repetition_annotations.csv"
```

The validator checks required columns, allowed labels and splits, numeric
metadata, duplicate identifiers, unknown clips, frame ordering, clip bounds,
attempt/fragment status, visibility rules, deviation flags, and single-label
priority. Formal orchestration additionally checks complete split coverage,
per-clip annotation presence and manifest/run provenance agreement. It does not
change ambiguity handling, perform new event matching or alter metrics.

## Quality control

Before formal evaluation, write and freeze a short annotator decision guide
with representative fictional or development examples. If more than one
annotator is available, independently annotate a subset and document agreement
and adjudication. Protocol changes must be completed on development data before
the final test annotations are analysed.
