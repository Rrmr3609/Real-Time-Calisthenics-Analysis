# Real-Time Calisthenics Analysis Using Computer Vision

An MSc Artificial Intelligence project from Queen Mary University of London
that analyses push-ups with OpenCV and MediaPipe Pose. It compares an
intentionally simple raw-threshold baseline with an enhanced temporal method
and provides observable, non-medical exercise-form feedback.

## What the System Does

- processes a live webcam feed or a recorded video;
- supports one visible person in a controlled side or side-diagonal view;
- extracts elbow and shoulder-hip-ankle angles from pose landmarks;
- counts push-up repetitions using baseline and enhanced approaches;
- classifies enhanced repetitions as `correct`, `insufficient_depth`,
  `incomplete_extension`, `alignment_deviation` or `unscorable`;
- records frame, repetition, configuration and provenance data for evaluation.

The thresholds are provisional project settings for controlled recordings.
They are not universal form definitions or medical guidance.

## Baseline vs Enhanced

| Area | Baseline | Enhanced |
| --- | --- | --- |
| Features | Raw visible-side angles | Visibility checks, stable elbow-side selection and EMA smoothing |
| Counting | Sticky top/bottom thresholds | Explicit temporal phases, hysteresis and confirmation frames |
| Missing observations | State retained until a valid angle returns | Bounded missing-frame tolerance |
| Form output | Diagnostic frame warnings only | Deterministic completed-repetition classification |
| Evaluation role | Repetition-detection comparator | Repetition detection and matched classification |

The baseline remains deliberately limited so that it provides a fair reference
rather than a second version of the enhanced method. See the
[comparison design](docs/baseline_evaluation_design.md) for exact semantics.

## Current Status

- The live baseline and both recorded-video runners are implemented.
- An enhanced webcam entry point provides end-user completed-repetition
  feedback, a session summary and a local text report.
- Enhanced temporal segmentation and repetition classification are implemented.
- Annotation validation, deterministic event matching, per-clip evaluation and
  cross-clip formal reporting are implemented and tested with fictional data.
- Development diagnostics are retained under `results/development/`.
- A real labelled evaluation dataset and formal experimental results are not yet
  complete.
- The `0.5`-second event-matching tolerance is provisional. Development
  ground truth must justify and freeze the final value before test evaluation.

## Quick Start

The project targets Python 3.12 and PowerShell on Windows.

Create a virtual environment and install runtime dependencies:

```powershell
py -3.12 -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
```

For testing and repository development, install the pinned development tools:

```powershell
& ".\.venv\Scripts\python.exe" -m pip install -r requirements-dev.txt
```

Run the live baseline:

```powershell
& ".\.venv\Scripts\python.exe" src\main.py
```

Run the enhanced live feedback system:

```powershell
& ".\.venv\Scripts\python.exe" src\run_live_enhanced.py `
  --camera-index 0 `
  --config "configs\default.yaml"
```

The enhanced live mode is the end-user real-time feedback interface. Focus its
video window and use `Q` or `Esc` to finish; the summary also closes with Enter.
Long summaries use Up/Down or PageUp/PageDown to move between repetition pages.
Each normally finished session writes an ignored human-readable report under
`experiments/outputs/live_sessions/`. These local summaries are not formal
evaluation evidence.

Run the baseline on a recorded development clip:

```powershell
& ".\.venv\Scripts\python.exe" src\run_video.py `
  --video "data\raw\development\example.mp4" `
  --clip-id "example" `
  --split development `
  --run-id "example_baseline_001" `
  --config "configs\default.yaml"
```

Run the enhanced method on the same clip:

```powershell
& ".\.venv\Scripts\python.exe" src\run_video_enhanced.py `
  --video "data\raw\development\example.mp4" `
  --clip-id "example" `
  --split development `
  --run-id "example_enhanced_001" `
  --config "configs\default.yaml"
```

The recorded baseline and enhanced modes support reproducible offline video
processing and evaluation; they are not presented as polished end-user GUIs.
Recorded runs write ignored outputs under `experiments/logs/` and
`experiments/outputs/`. Each run records its resolved configuration, source
identity, software versions, Git state and timing definition. Existing output
sets fail clearly unless replacement is explicitly requested with `--overwrite`.
Detailed options are in the
[runtime and provenance guide](docs/runtime_configuration.md).

## Evaluation Workflow

1. Record or select development and final-test clips under controlled capture
   conditions.
2. Create the dataset manifest and annotate attempts independently of method
   predictions.
3. Run both methods on the same clips with traceable run IDs.
4. Validate the manifest and annotations.
5. Choose and freeze development decisions, then execute formal evaluation.

Use only fictional example CSVs when learning the schemas:

- `data/examples/manifests/example_dataset_manifest.csv`;
- `data/examples/annotations/example_repetition_annotations.csv`.

The [manual annotation protocol](docs/manual_annotation_protocol.md) defines
ground truth. The [formal execution guide](docs/formal_evaluation_execution.md)
defines the development-first safeguards and commands.

## Repository Structure

- `src/` — runners, analysis, pose processing, evaluation and shared utilities;
- `tests/` — unit and integration tests using fictional or temporary data;
- `configs/` — validated runtime configuration;
- `data/examples/` — fictional manifest and annotation examples;
- `data/raw/` — ignored local recordings;
- `experiments/` — ignored generated run logs and outputs, created as needed;
- `results/development/` — retained historical development diagnostics;
- `results/formal/` — formal outputs created by the evaluation command;
- `docs/` — scientific design, annotation and evaluation documentation.

## Documentation

The [documentation index](docs/README.md) gives a one-sentence purpose for
every technical document. Start with:

- [runtime configuration and provenance](docs/runtime_configuration.md);
- [baseline-versus-enhanced design](docs/baseline_evaluation_design.md);
- [manual annotation protocol](docs/manual_annotation_protocol.md);
- [formal evaluation execution](docs/formal_evaluation_execution.md).

## Testing and Quality

The repository configures pytest and Ruff in `pyproject.toml`; manual
`PYTHONPATH` setup is not required for tests.

```powershell
& ".\.venv\Scripts\python.exe" -m pytest -q
& ".\.venv\Scripts\python.exe" -m ruff check src tests
& ".\.venv\Scripts\python.exe" -m ruff format --check src tests
```

## Limitations

- one push-up participant only;
- controlled side or side-diagonal camera placement;
- 2D pose evidence depends on visibility and recording quality;
- no model training, multi-person tracking or 3D reconstruction;
- alignment may be unscorable when the required landmarks are unavailable;
- no clinical, rehabilitation or injury-prevention interpretation;
- formal conclusions require the unfinished labelled evaluation dataset.

## Data and Privacy

Raw identifiable recordings and generated experiment runs are ignored by Git.
Only fictional schema examples and selected non-identifying development
evidence are tracked. Dataset splits must keep development decisions separate
from final-test evaluation.

## Academic Context

This repository supports an MSc project investigating whether transparent
temporal processing improves push-up analysis over a simple threshold baseline
under controlled computer-vision conditions.
