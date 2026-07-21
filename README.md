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

## Current Baseline Rules

The current baseline uses provisional operational thresholds:

- top position: elbow angle >= 150 degrees;
- bottom position: elbow angle <= 100 degrees;
- repetition count: top -> bottom -> top;
- body-alignment warning: shoulder-hip-ankle angle < 160 degrees.

These thresholds are project-specific development thresholds. They are not universal definitions of correct push-up form and are not medical or injury-risk claims.


## Recorded-Video Processing

The baseline analyser can process a saved video so that the same recording can later be evaluated using both the baseline and enhanced methods.

Example:

```bash
PYTHONPATH=src python src/run_video.py \
  --video data/raw/development/example.mp4 \
  --clip-id example \
  --display
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