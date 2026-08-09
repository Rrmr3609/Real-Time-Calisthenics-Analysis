# Formal evaluation execution

## Scope

`src/evaluation/run_formal_evaluation.py` is the thin execution layer for the
existing formal-evaluation pipeline. It reads previously generated recorded-run
outputs through their explicit provenance metadata. It does not discover files,
process video, tune event tolerance or reimplement any metric.

Baseline remains detection-only. Enhanced evaluation contains detection and
classification for repetitions paired by the existing deterministic event
matcher. Unmatched enhanced predictions and annotations remain detection extras
and misses; they do not enter the classification confusion matrix.

## Required order of work

### 1. Generate recorded-run outputs

Run both recorded-video methods once for every clip, using the same frozen
manifest clip ID and split. Give each invocation an explicit run ID. The
recorded runners write frame/repetition CSV files and completed provenance JSON
metadata.

The orchestration command consumes the metadata JSON paths, not guessed output
filenames. Each metadata document identifies its run, clip, method, split,
source FPS, frame count, resolution and generated output paths.

The evaluation metadata captures a streaming SHA-256 identity for every
supplied source-run metadata file and for the exact CSV consumed from it:
baseline `frame_csv` and enhanced `repetition_csv`. It also inherits the input
video SHA-256, source Git commit/dirty state and a deterministic hash of the
resolved configuration when those values exist in the source metadata. Source
metadata files are not modified.

### 2. Prepare the dataset manifest

Create one manifest row per clip using the schema in
`manual_annotation_protocol.md`. Assign `development` or `test` before formal
analysis. Do not change a test clip to development to bypass the execution
safeguard. Formal execution always covers every manifest clip in the chosen
split; neither method may supply only a subset or add another clip.

### 3. Prepare and validate annotations

Annotate every evaluable attempt independently of method predictions. Retain
ambiguous fragments according to the manual protocol. Validate the complete
manifest and annotations before evaluation:

```powershell
$env:PYTHONPATH = "$PWD\src"

& ".\.venv\Scripts\python.exe" `
  src\evaluation\dataset_validation.py `
  --manifest "data\examples\manifests\example_dataset_manifest.csv" `
  --annotations "data\examples\annotations\example_repetition_annotations.csv"
```

The supplied example CSVs are fictional schema examples only. They do not
describe real recordings, participants or evaluation results.

Every clip in the selected formal split must have at least one annotation row
as evidence that it was manually reviewed. An ambiguous-fragment row satisfies
this review-presence safeguard, but remains excluded from ground-truth event
metrics under the existing ambiguity rule. The current schema has no explicit
review-complete representation for a genuine zero-attempt clip, so such clips
are not supported by this formal workflow.

### 4. Run development evaluation

Supply every baseline and enhanced metadata path explicitly. The following
PowerShell command uses fictional names:

```powershell
$env:PYTHONPATH = "$PWD\src"

& ".\.venv\Scripts\python.exe" `
  src\evaluation\run_formal_evaluation.py `
  --manifest "data\manifests\fictional_development_manifest.csv" `
  --annotations "data\annotations\fictional_development_annotations.csv" `
  --baseline-metadata `
    "experiments\logs\fictional-a-baseline_metadata.json" `
    "experiments\logs\fictional-b-baseline_metadata.json" `
  --enhanced-metadata `
    "experiments\outputs\fictional-a-enhanced_metadata.json" `
    "experiments\outputs\fictional-b-enhanced_metadata.json" `
  --split development `
  --tolerance-seconds 0.5 `
  --output-directory "results\formal\development" `
  --run-id "fictional-development-evaluation-001"
```

`--baseline-metadata` and `--enhanced-metadata` each accept one or more paths.
Each method's clip set must equal the complete chosen manifest split. The
default tolerance is the current provisional `0.5` seconds, but another
positive finite value may be supplied. The command passes that value through
unchanged to every clip and method.

Existing report files cause a clear failure. Add `--overwrite` only when the
complete report set for that evaluation run ID should be replaced.

### 5. Justify and freeze event tolerance

The execution layer does not select or optimise event tolerance. Use only
development-set evidence to justify the final value, document the decision and
freeze it before evaluating the test split. All clips in one report use the
same supplied tolerance, and the reporting layer verifies its frame equivalent
against each clip's source FPS.

### 6. Enable test evaluation only after freezing decisions

Test execution is deliberately refused unless `--allow-final-test` is present.
After annotations, thresholds, class rules and event tolerance have been
frozen, run the same explicit command with:

```powershell
  --split test `
  --tolerance-seconds <frozen-development-value> `
  --allow-final-test
```

The flag is an accidental-use safeguard, not evidence that the test set is
ready. It must not be used while development decisions remain open. Test output
must never be used to retune the tolerance or any analysis rule.

## Validation before metrics

The orchestrator rejects the complete run rather than skipping a bad clip when:

- a manifest or annotation fails the existing schema validation;
- metadata is absent, malformed, not schema version 1, not completed or does
  not record full-clip processing;
- completed metadata lacks a completion timestamp;
- baseline/enhanced metadata has the wrong method or selected split;
- a clip ID is duplicated within one method;
- either method omits a selected manifest clip or supplies an extra clip;
- a selected manifest clip has no annotation row proving manual review;
- source FPS differs between manifest, baseline and enhanced provenance;
- frame count, width or height differs between the manifest and either run;
- baseline and enhanced input SHA-256 hashes differ for one clip;
- a required metadata output path is missing or its file does not exist;
- the baseline frame CSV lacks the current runner header, contains a run or
  clip identity contradicting metadata, omits/duplicates frame indices, or has
  a row count different from completed-run metadata;
- loaded event run, clip or method identity contradicts its metadata;
- a source metadata file or consumed CSV disappears or changes after it has
  been accepted and before formal report writing;
- event tolerance is zero, negative, NaN or infinite;
- test evaluation lacks the explicit final-test flag; or
- any report output already exists without `--overwrite`.

Project-relative output paths in runner metadata are resolved from the
repository root. Absolute paths are used as recorded. Baseline events are
loaded from the metadata `frame_csv`; enhanced events are loaded from the
metadata `repetition_csv`.

The current baseline frame CSV has `run_id` and `clip_id` columns but no method
column. Formal loading validates every identity field that exists in that CSV;
the baseline method identity itself is validated from completed-run metadata.

Source metadata and consumed CSV paths stored in final evaluation metadata are
repository-relative POSIX-style paths when the files are inside the repository.
For external files, only the basename is published; the streaming SHA-256 still
provides deterministic file identity without exposing an absolute machine path.

## Generated report files

For evaluation run ID `<run-id>`, the command writes:

1. `<run-id>_formal_evaluation.json`
2. `<run-id>_per_clip_metrics.csv`
3. `<run-id>_classification_confusion_matrix.csv`
4. `<run-id>_classification_per_class.csv`
5. `<run-id>_detection_recall_by_class.csv`
6. `<run-id>_evaluation_metadata.json`

The output schemas, aggregation formulas, deterministic ordering, overwrite
policy and atomic metadata lifecycle are documented in
`formal_evaluation_reporting.md`.

## Python API

The same workflow is available through:

```python
run_formal_evaluation(
    manifest_path=...,
    annotations_path=...,
    baseline_metadata_paths=[...],
    enhanced_metadata_paths=[...],
    split="development",
    tolerance_seconds=0.5,
    output_directory=...,
    evaluation_run_id=...,
    overwrite=False,
    allow_final_test=False,
)
```

It returns `FormalEvaluationOutputPaths` for the complete generated report set.
