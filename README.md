# Real-Time Calisthenics Analysis Using Computer Vision

MSc Artificial Intelligence project at Queen Mary University of London.

## Current Scope

The project focuses on real-time analysis of one calisthenic exercise:
the push-up.

The controlled implementation conditions are:

- one person visible;
- side-view camera orientation;
- standard webcam or recorded-video input;
- repetition and movement-phase detection;
- three predefined observable form deviations;
- immediate non-medical feedback.

## Form Categories

1. Correct repetition
2. Insufficient depth
3. Incomplete elbow extension
4. Shoulder-hip-ankle alignment deviation

## Methods

The project will compare:

1. A baseline method using raw pose landmarks and frame-level thresholds.
2. An enhanced method using confidence handling, temporal smoothing,
   movement-phase modelling and stable transition rules.

## Repository Structure

- `src/` — application source code
- `tests/` — software unit tests
- `configs/` — runtime and experimental configuration
- `data/` — local recordings, annotations and processed data
- `experiments/` — experiment configurations, logs and generated outputs
- `results/` — final tables, figures and documented failure cases

Raw identifiable recordings are not committed to this repository.

## Status

Project structure initialised. Pose-estimation implementation has not yet
been completed.# Real-Time-Calisthenics-Analysis
