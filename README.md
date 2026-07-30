# Real-Time Calisthenics Analysis Using Computer Vision

MSc Artificial Intelligence project at Queen Mary University of London.

## Current Scope

The project focuses on real-time analysis of one calisthenic exercise: the push-up.

The controlled implementation conditions are:

- one person visible;
- controlled side-view camera orientation;
- standard webcam input in the current prototype;
- recorded-video runners for the formal evaluation workflow;
- repetition counting and movement-state analysis;
- three predefined observable form-deviation categories;
- immediate non-medical feedback.

## Form Categories

The planned repetition-level evaluation categories are:

1. Correct repetition under the project-defined conditions
2. Insufficient depth
3. Incomplete elbow extension
4. Shoulder-hip-ankle alignment deviation

The current live baseline displays provisional warnings. A live "No warning" message is not treated as proof that a repetition is correct. Correct and incorrect repetition labels will be defined through manual annotation during evaluation.

## Implemented Baseline Prototype

The current prototype:

- opens a live webcam feed using OpenCV;
- runs MediaPipe Pose on each frame;
- draws detected pose landmarks;
- extracts selected landmarks for the visible body side;
- calculates elbow angle from shoulder-elbow-wrist landmarks;
- calculates shoulder-hip-ankle body-alignment angle;
- displays FPS, selected side, angles, position, repetition count and warning text;
- logs per-frame features and baseline outputs to CSV;
- applies a provisional raw-threshold baseline for push-up analysis.

Run the live baseline from PowerShell:

```powershell
$env:PYTHONPATH = "$PWD\src"
& ".\.venv\Scripts\python.exe" src\main.py
```

## Current Baseline Rules

The current baseline uses provisional operational thresholds:

- top position: elbow angle >= 150 degrees;
- bottom position: elbow angle <= 100 degrees;
- repetition count: top -> bottom -> top;
- body-alignment warning: shoulder-hip-ankle angle < 160 degrees.

These thresholds are project-specific development thresholds. They are not universal definitions of correct push-up form and are not medical or injury-risk claims.


## Recorded-run configuration

Both recorded-video runners load validated typed settings from
`configs/default.yaml`. The complete field reference, precedence rules,
metadata schema and reproduction commands are documented in
`docs/runtime_configuration.md`.

Every recorded command requires `--split development|test`. `--run-id`
defaults to `--clip-id`; use an explicit unique run ID for repeated
experiments. The enhanced `--alpha` option is an explicit override of
`features.ema_alpha` and is recorded in metadata.

## Enhanced Preprocessing — In Development

An enhanced preprocessing layer has been added separately from the frozen baseline.

The current enhanced preprocessing includes:

- visibility-based body-side scoring;
- confirmation across multiple frames before acquiring or switching sides;
- a margin requirement before switching to the opposite side;
- a short grace period for missing side visibility;
- exponential moving-average smoothing of elbow and body-alignment angles;
- raw and smoothed feature logging for recorded videos;
- left/right elbow and alignment visibility scores;
- the elbow-selected side and whether the opposite side could provide valid
  alignment landmarks.

Example:

```powershell
$env:PYTHONPATH = "$PWD\src"

& ".\.venv\Scripts\python.exe" src\run_video_enhanced.py `
  --video "data\raw\development\example.mp4" `
  --clip-id "example" `
  --split development `
  --run-id "example_enhanced_001" `
  --config "configs\default.yaml" `
  --display
```

Create an alignment-availability diagnostic from one enhanced run:

```powershell
& ".\.venv\Scripts\python.exe" `
  src\evaluation\summarise_alignment_visibility.py `
  --frame-input `
    "experiments\logs\<run-id>_enhanced_temporal.csv" `
  --repetition-input `
    "experiments\outputs\<run-id>_enhanced_repetitions.csv" `
  --output `
    "results\testing\2026-07-28_alignment_visibility_diagnostic_summary.txt" `
  --summary-date "2026-07-28" `
  --minimum-alignment-valid-ratio 0.50
```

The diagnostic reports overall and phase-grouped feature availability,
opposite-side rescue opportunities, elbow-side changes, mean repetition
alignment coverage, the final predicted-class `unscorable` count and the
independent count of repetitions whose alignment evidence is unscorable below
the configured minimum-valid-ratio threshold. Repetition summaries reject
duplicate `(clip_id, rep_id)` rows. These diagnostics do not change the
elbow-driven selector or any classifier threshold.

### Enhanced repetition measurement window

Enhanced repetition measurements use one closed, inclusive frame interval.
The interval starts at the frame containing the maximum genuine top
observation (`elbow_angle >= top_region_threshold`) available before the
contiguous descent-confirmation sequence. That anchor is frozen when the
candidate sequence begins. The interval ends at the final
consecutive-confirmation frame that completes the return to top. Therefore:

- `duration_frames` is `end_frame - start_frame + 1`;
- only the contiguous descent-candidate frames that confirm the transition are
  retained; an interruption discards the earlier candidate sequence and moves
  the tentative start forward;
- tolerated missing-elbow frames after descent confirmation remain inside the
  interval;
- the minimum elbow angle and bottom frame use every valid elbow observation
  in the interval, including the retained confirmation candidates;
- `start_top_angle` comes from the genuine top anchor, while `end_top_angle`
  uses only the valid frames in the confirmed return-to-top sequence;
- body-alignment observations are collected from the same start and end
  frames; missing alignment values are not fabricated;
- alignment coverage is the number of valid alignment observations divided by
  `duration_frames`, so valid alignment on every interval frame gives `1.0`.

Hysteresis, consecutive-frame confirmation and missing-frame tolerance still
control phase transitions. If an attempt returns to top without reaching the
provisional bottom region, its tentative repetition measurements are discarded
and a new top frame begins the next tentative window. An interrupted,
unconfirmed descent candidate is likewise discarded. A new genuine top
observation is then required before a later candidate sequence can contribute
to a completed repetition.

## Evaluation Data Foundation

The approved baseline-versus-enhanced comparison design is documented in
`docs/baseline_evaluation_design.md`. The manual procedure and complete CSV
column definitions are in `docs/manual_annotation_protocol.md`.

Fictional schema examples are provided at:

- `data/manifests/example_dataset_manifest.csv`;
- `data/annotations/example_repetition_annotations.csv`.

They do not describe real recordings or participants. Validate a manifest and
annotation file from PowerShell with:

```powershell
$env:PYTHONPATH = "$PWD\src"

& ".\.venv\Scripts\python.exe" `
  src\evaluation\dataset_validation.py `
  --manifest "data\manifests\example_dataset_manifest.csv" `
  --annotations "data\annotations\example_repetition_annotations.csv"
```

Detection-event extraction, deterministic one-to-one matching and per-clip
detection metrics are implemented in
`src/evaluation/detection_evaluation.py`. The matching rule, validation
behavior and fictional command example are documented in
`docs/event_detection_evaluation.md`.

This stage reports repetition detection only. Classification metrics,
confusion matrices, form-category performance, runtime comparisons, plots and
formal final-test evaluation remain unimplemented.

  
## Recorded-Video Processing

The baseline analyser can process a saved video so that the same recording can later be evaluated using both the baseline and enhanced methods.

Example:

```powershell
$env:PYTHONPATH = "$PWD\src"

& ".\.venv\Scripts\python.exe" src\run_video.py `
  --video "data\raw\development\example.mp4" `
  --clip-id "example" `
  --split development `
  --run-id "example_baseline_001" `
  --config "configs\default.yaml" `
  --display
```

## Output File Safety

CSV outputs never append a second complete run. By default, each runner fails
clearly if its target output already exists:

- live baseline: `experiments/logs/live_feature.csv`;
- recorded baseline: `experiments/logs/<run-id>_baseline.csv`;
- baseline metadata: `experiments/logs/<run-id>_baseline_metadata.json`;
- enhanced frame log: `experiments/logs/<run-id>_enhanced_temporal.csv`;
- enhanced repetitions: `experiments/outputs/<run-id>_enhanced_repetitions.csv`;
- enhanced metadata: `experiments/outputs/<run-id>_enhanced_metadata.json`.

Each recorded runner checks its complete CSV and metadata output set before
video processing begins. Every CSV row carries the same `run_id`; metadata
records the clip/method/split, input and configuration hashes, resolved
settings and overrides, source properties, software/Git provenance, timing
definition and output paths. To intentionally replace the complete set, add
the explicit `--overwrite` option:

```powershell
& ".\.venv\Scripts\python.exe" src\run_video_enhanced.py `
  --video "data\raw\development\example.mp4" `
  --clip-id "example" `
  --split development `
  --run-id "example_enhanced_001" `
  --config "configs\default.yaml" `
  --overwrite
```

## Current Limitations

The current baseline does not yet include:

- temporal smoothing;
- hysteresis;
- consecutive-frame validation;
- explicit descending and ascending phases;
- feedback cooldown;
- full repetition-level form classification;
- formal manually labelled evaluation results.

The enhanced temporal method is implemented. The labelled evaluation dataset,
quantitative baseline-versus-enhanced evaluation and final result analysis
have not yet been completed.

## Repository Structure

- `src/` — application source code
- `tests/` — software unit tests
- `configs/` — runtime and experimental configuration
- `data/` — local recordings, annotations and processed data
- `experiments/` — experiment configurations, logs and generated outputs
- `results/` — final tables, figures and documented failure cases

Raw identifiable recordings are not committed to this repository.

## Installation

```powershell
& ".\.venv\Scripts\python.exe" -m pip install `
  -r requirements.txt
```
