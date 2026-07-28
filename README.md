# Real-Time Calisthenics Analysis Using Computer Vision

MSc Artificial Intelligence project at Queen Mary University of London.

## Current Scope

The project focuses on real-time analysis of one calisthenic exercise: the push-up.

The controlled implementation conditions are:

- one person visible;
- controlled side-view camera orientation;
- standard webcam input in the current prototype;
- recorded-video evaluation input planned for the formal evaluation stage;
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


## Enhanced Preprocessing — In Development

An enhanced preprocessing layer has been added separately from the frozen baseline.

The current enhanced preprocessing includes:

- visibility-based body-side scoring;
- confirmation across multiple frames before acquiring or switching sides;
- a margin requirement before switching to the opposite side;
- a short grace period for missing side visibility;
- exponential moving-average smoothing of elbow and body-alignment angles;
- raw and smoothed feature logging for recorded videos.

Example:

```powershell
$env:PYTHONPATH = "$PWD\src"

& ".\.venv\Scripts\python.exe" src\run_video_enhanced.py `
  --video "data\raw\development\example.mp4" `
  --clip-id "example" `
  --alpha 0.3 `
  --display
```

  
## Recorded-Video Processing

The baseline analyser can process a saved video so that the same recording can later be evaluated using both the baseline and enhanced methods.

Example:

```powershell
$env:PYTHONPATH = "$PWD\src"

& ".\.venv\Scripts\python.exe" src\run_video.py `
  --video "data\raw\development\example.mp4" `
  --clip-id "example" `
  --display
```

## Output File Safety

CSV outputs never append a second complete run. By default, each runner fails
clearly if its target output already exists:

- live baseline: `experiments/logs/live_feature.csv`;
- recorded baseline: `experiments/logs/<clip-id>_baseline.csv`;
- enhanced frame log: `experiments/logs/<clip-id>_enhanced_temporal.csv`;
- enhanced repetitions: `experiments/outputs/<clip-id>_enhanced_repetitions.csv`.

For the enhanced runner, both output paths are checked before video processing
begins. To intentionally replace the output or outputs for a runner, add the
explicit `--overwrite` option:

```powershell
& ".\.venv\Scripts\python.exe" src\run_video_enhanced.py `
  --video "data\raw\development\example.mp4" `
  --clip-id "example" `
  --alpha 0.3 `
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

The enhanced temporal method, labelled evaluation dataset, quantitative evaluation and final result analysis have not yet been completed.

## Repository Structure

- `src/` — application source code
- `tests/` — software unit tests
- `configs/` — runtime and experimental configuration
- `data/` — local recordings, annotations and processed data
- `experiments/` — experiment configurations, logs and generated outputs
- `results/` — final tables, figures and documented failure cases

Raw identifiable recordings are not committed to this repository.

## Installation

```bash
pip install -r requirements.txt
```
