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

## Development annotation viewer

The active real-development inputs are:

- `data/manifests/development_dataset_manifest.csv`;
- `data/annotations/development_repetition_annotations.csv`;
- `data/annotations/development_repetition_annotations.review.json`.

The annotation CSV initially contains only the established schema header. Do
not add recalled attempt counts or derive labels from filenames or external
source folders. Start the source-only viewer for one manifest clip with:

```powershell
$env:PYTHONPATH = "$PWD\src"

& ".\.venv\Scripts\python.exe" `
  src\evaluation\annotate_repetitions.py `
  --manifest "data\manifests\development_dataset_manifest.csv" `
  --annotations "data\annotations\development_repetition_annotations.csv" `
  --clip-id "dev01_correct" `
  --annotator "ANN01"
```

`ANN01` is an example anonymised annotator identifier. The manifest resolves
the project-relative raw-video path; no personal absolute path is stored. The
viewer displays source frames only. It does not import or show pose landmarks,
angles, predicted phases, predicted repetitions, predicted classes, baseline
outputs or enhanced outputs.

### Viewer controls

| Key | Action |
| --- | --- |
| Space | Play or pause |
| `,` / `.` or Left / Right | Step backward or forward one frame |
| `[` / `]` | Jump backward or forward ten frames |
| `A` | Mark the representative start/top frame immediately before descent |
| `B` | Mark the lowest point or descent-to-ascent turnaround frame |
| `E` | Mark the completion/end-top frame |
| `1` | Select no visible deviation (`correct`) and clear deviation flags |
| `2` | Toggle insufficient-depth evidence |
| `3` | Toggle incomplete-extension evidence |
| `4` | Toggle alignment-deviation evidence |
| `5` | Select insufficient source visibility (`unscorable`) |
| `V` | Cycle source visibility: sufficient, partially obscured, insufficient |
| `M` | Toggle ambiguous-fragment status |
| `N` | Enter or clear an optional note in the terminal |
| `R` | Reset the unsaved draft |
| `S` | Validate and atomically save the current row |
| `Q` or Esc | Close the viewer, retaining saved rows and the resume checkpoint |

The completion mark remains the first frame after ascent where the attempt has
visibly returned to its ending/top posture. It is a visual source-video
decision, never an application of the baseline or enhanced top thresholds.

Keys `2`, `3` and `4` may be combined. The viewer stores every selected
deviation flag and derives the canonical single class using the frozen
priority: insufficient depth, incomplete extension, alignment deviation, then
correct. It never infers a class from the clip ID, filename, rough performed
count or Kaggle source grouping.

An ambiguous fragment may retain any identifiable start, turnaround or end
frame, but requires at least one locating frame and an explanatory note. It is
saved as non-evaluable and unscorable with false deviation flags, exactly as
required by the schema. An evaluable unscorable attempt requires all three
event frames, insufficient source visibility and a note.

### Resume and output safety

Each explicit save appends a new unique `(clip_id, ground_truth_attempt_id)`
row, validates the complete CSV and atomically replaces the file. Existing
identities are never overwritten. Rows are ordered by manifest clip order,
locating frame and stable attempt ID. A single ignored adjacent `.resume.json`
checkpoint retains the current frame and unsaved draft; it prevents unfinished
work for one clip from being silently replaced by another clip. The annotation
CSV remains one shared file rather than a collection of per-video files.

### Review and freeze record

Opening the viewer changes the adjacent review record from `not_started` to
`in_progress` and records the anonymised annotator and annotation date. It does
not mark review complete. After every manifest clip has at least one retained
evaluable-attempt or ambiguous-fragment row and human review is complete,
explicitly freeze the validated CSV with:

```powershell
$env:PYTHONPATH = "$PWD\src"

& ".\.venv\Scripts\python.exe" `
  src\evaluation\annotate_repetitions.py `
  --manifest "data\manifests\development_dataset_manifest.csv" `
  --annotations "data\annotations\development_repetition_annotations.csv" `
  --finalise-review `
  --reviewer "REVIEWER01" `
  --repeat-review-status "not_performed"
```

Finalisation refuses incomplete manifest coverage, revalidates the established
schema, records reviewer/repeat-review/adjudication fields and stores the exact
SHA-256 of the frozen annotation CSV. Once complete, the viewer refuses to
resume that frozen annotation file. A repeat review is optional; when marked
`complete`, its reviewer identifier is required.

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
