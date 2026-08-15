# Real-Time Calisthenics Analysis Using Computer Vision

An MSc Artificial Intelligence project from Queen Mary University of London
for real-time push-up analysis using OpenCV and MediaPipe Pose. The project
compares an intentionally simple raw-threshold baseline with an enhanced
temporal method and provides observable, non-medical exercise-form feedback.

The implemented research scope is deliberately restricted to push-ups performed
by one visible participant in a controlled side or side-diagonal camera view.

## What the System Does

- processes a live webcam feed or a recorded video;
- supports one visible person in a controlled side or side-diagonal view;
- extracts elbow and shoulder-hip-ankle angles from MediaPipe pose landmarks;
- counts push-up repetitions using baseline and enhanced approaches;
- classifies enhanced repetitions as `correct`, `insufficient_depth`,
  `incomplete_extension`, `alignment_deviation` or `unscorable`;
- records frame-level, repetition-level, configuration and provenance data;
- supports manual ground-truth annotation and formal event/classification
  evaluation.

The numeric thresholds are frozen project settings developed for the controlled
recordings used in this study. They are not universal exercise-form definitions,
medical guidance or rehabilitation criteria.

## Baseline vs Enhanced

| Area | Baseline | Enhanced |
| --- | --- | --- |
| Features | Raw visible-side angles | Visibility checks, stable elbow-side selection and EMA smoothing |
| Counting | Sticky top-bottom-top threshold state | Explicit temporal phases, hysteresis and confirmation frames |
| Missing observations | No enhanced temporal recovery logic | Bounded missing-frame tolerance |
| Form output | Diagnostic frame warnings only | Deterministic completed-repetition classification |
| Evaluation role | Repetition-detection comparator | Repetition detection and matched repetition classification |

The baseline remains deliberately simple so that it acts as a meaningful
reference rather than a second version of the enhanced method. Baseline warning
flags are frame-level diagnostics and are not treated as repetition classes.

See the
[baseline evaluation design](docs/baseline_evaluation_design.md)
for the exact comparison semantics.

## Frozen Enhanced Configuration

The final enhanced scientific configuration was frozen before held-out test
evaluation.

Key values are:

- EMA smoothing alpha: `0.3`;
- segmentation top region: `130 degrees`;
- segmentation bottom region: `120 degrees`;
- hysteresis: `5 degrees`;
- phase confirmation: `3` frames;
- minimum repetition duration: `8` frames;
- insufficient-depth classification threshold: `65 degrees`;
- incomplete-extension classification threshold: `150 degrees`;
- alignment minimum: `160 degrees`;
- alignment-deviation minimum: `3` frames and `20%` of valid observations;
- minimum valid alignment coverage: `50%`;
- primary event-matching tolerance: `0.50 seconds`.

The full configuration is stored in
[`configs/default.yaml`](configs/default.yaml).
The development freeze and decision rationale are recorded in
[`docs/development_scientific_freeze.md`](docs/development_scientific_freeze.md).

## Final Evaluation Summary

Formal machine-readable outputs are committed under
[`results/formal/`](results/formal/).
The
[`formal-results index`](results/formal/README.md)
distinguishes primary, sensitivity and historical development runs.

### Development set

The development ground truth contained 43 evaluable repetitions across 12
clips. At the frozen primary `0.50`-second event tolerance:

| Metric | Baseline | Enhanced |
| --- | ---: | ---: |
| Event precision | 0.775 | 0.975 |
| Event recall | 0.721 | 0.907 |
| Event F1 | 0.747 | 0.940 |
| Exact-count clip accuracy | 9/12 | 11/12 |
| Count MAE | 0.917 | 0.250 |
| Completion-timing MAE | 0.337 s | 0.242 s |

For the 39 enhanced repetitions matched to ground truth within the event
tolerance, classification accuracy was `87.18%` and macro F1 was `0.869`.

Development-only tolerance sensitivity was also evaluated at `0.25`, `0.75`
and `1.00` seconds. The originally specified `0.50`-second value remained the
primary tolerance rather than being replaced post-hoc by the value producing
the highest development score.

### Held-out test set

The final held-out set contained 4 newly recorded clips and 28 manually
annotated evaluable repetitions from one anonymised participant. It therefore
provides clip-level held-out evidence, not participant-independent
generalisation evidence. Ground truth was frozen before either system was run
on these clips.

Both systems produced exactly 28 repetitions in total and achieved exact
repetition-count accuracy on all 4 clips. Timing-aware event matching at the
frozen `0.50`-second tolerance showed:

| Metric | Baseline | Enhanced |
| --- | ---: | ---: |
| Event precision | 0.357 | 0.643 |
| Event recall | 0.357 | 0.643 |
| Event F1 | 0.357 | 0.643 |
| Exact-count clip accuracy | 4/4 | 4/4 |
| Count MAE | 0.000 | 0.000 |
| Completion-timing MAE | 0.463 s | 0.335 s |

For the 18 enhanced repetitions matched to ground truth within the event
tolerance, classification accuracy was `77.78%` and macro F1 was `0.705`.

A clear held-out limitation was observed for incomplete extension: all four
temporally matched ground-truth incomplete-extension repetitions were
classified as `correct`. The frozen system was not retuned after observing this
held-out result.

## Current Status

The implementation and scientific evaluation are complete.

Completed components include:

- live baseline push-up analysis;
- recorded-video baseline analysis;
- recorded-video enhanced temporal analysis;
- enhanced live completed-repetition feedback and session summaries;
- stable pose-side selection and visibility handling;
- temporal phase segmentation and repetition aggregation;
- deterministic repetition classification;
- source-only manual ground-truth annotation;
- dataset and annotation validation;
- deterministic event matching;
- per-clip and pooled detection metrics;
- enhanced matched-repetition classification evaluation;
- development calibration and tolerance sensitivity analysis;
- frozen held-out evaluation;
- preserved formal results and evaluation provenance.

No classifier or event-matching parameter was retuned using held-out test
performance.

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

### Live baseline

```powershell
& ".\.venv\Scripts\python.exe" src\main.py
```

The baseline writes the fixed path `experiments/logs/live_feature.csv`.
If that file already exists, rerun with `--overwrite` to replace it.

### Enhanced live feedback

```powershell
& ".\.venv\Scripts\python.exe" src\run_live_enhanced.py `
  --camera-index 0 `
  --config "configs\default.yaml"
```

The enhanced live mode is the primary real-time user interface. Focus its video
window and use `Q` or `Esc` to finish. A normally completed session writes a
local human-readable summary under
`experiments/outputs/live_sessions/`.

These session summaries are not formal evaluation evidence.

### Recorded baseline

```powershell
& ".\.venv\Scripts\python.exe" src\run_video.py `
  --video "data\raw\development\example.mp4" `
  --clip-id "example" `
  --split development `
  --run-id "example_baseline_001" `
  --config "configs\default.yaml"
```

### Recorded enhanced method

```powershell
& ".\.venv\Scripts\python.exe" src\run_video_enhanced.py `
  --video "data\raw\development\example.mp4" `
  --clip-id "example" `
  --split development `
  --run-id "example_enhanced_001" `
  --config "configs\default.yaml"
```

Recorded runs write generated outputs under `experiments/logs/` and
`experiments/outputs/`. These directories are intentionally ignored by Git.
Each run records configuration, source identity, software versions, Git state
and timing provenance. Existing output sets are protected unless replacement
is explicitly requested with `--overwrite`.

See the
[runtime and provenance guide](docs/runtime_configuration.md)
for detailed runner options.

## Evaluation Workflow

The scientific workflow used in the project was:

1. define controlled recording and annotation protocols;
2. construct and manually annotate the development set independently of system
   predictions;
3. freeze development ground truth;
4. run and evaluate the baseline and enhanced methods on development data;
5. fix the causal return-top extension implementation issue identified during
   development;
6. calibrate the enhanced insufficient-depth threshold using development data
   only;
7. perform event-tolerance sensitivity analysis;
8. freeze the complete scientific configuration and the primary
   `0.50`-second event tolerance;
9. pre-register the held-out collection protocol;
10. collect and manually annotate fresh held-out recordings without viewing
    system predictions;
11. freeze held-out ground truth;
12. run the frozen baseline and enhanced systems on the held-out set once;
13. report the result without test-set retuning.

The held-out collection was reduced from the originally pre-registered six
clips to four for feasibility. This deviation was documented before annotation
or system prediction and is recorded in
[`docs/held_out_test_collection_protocol.md`](docs/held_out_test_collection_protocol.md).

Use only the fictional example CSVs when learning the schemas:

- `data/examples/manifests/example_dataset_manifest.csv`;
- `data/examples/annotations/example_repetition_annotations.csv`.

The
[manual annotation protocol](docs/manual_annotation_protocol.md)
defines ground-truth semantics and the source-only frame annotation viewer.
Real dataset metadata and provenance are indexed in
[`data/README.md`](data/README.md).

The
[formal evaluation execution guide](docs/formal_evaluation_execution.md)
documents the development-first safeguards and final-test protection.

## Repository Structure

- `src/` - runners, analysis, pose processing, evaluation and shared utilities;
- `tests/` - unit and integration tests using fictional or temporary data;
- `configs/` - validated runtime configuration;
- `data/examples/` - fictional manifest and annotation examples;
- `data/manifests/` - tracked development/test dataset metadata;
- `data/annotations/` - tracked frozen manual annotations and review metadata;
- `data/raw/` - ignored local recordings;
- `experiments/` - ignored generated frame/repetition run outputs;
- `results/development/` - retained historical development diagnostics;
- `results/formal/` - committed formal development and held-out evidence;
- `docs/` - scientific design, annotation, evaluation and provenance documents.

## Documentation

The
[documentation index](docs/README.md)
gives the purpose of each technical document.

Important documents include:

- [runtime configuration and provenance](docs/runtime_configuration.md);
- [baseline-versus-enhanced design](docs/baseline_evaluation_design.md);
- [manual annotation protocol](docs/manual_annotation_protocol.md);
- [event detection evaluation](docs/event_detection_evaluation.md);
- [classification evaluation](docs/classification_evaluation.md);
- [formal evaluation execution](docs/formal_evaluation_execution.md);
- [formal evaluation reporting](docs/formal_evaluation_reporting.md);
- [development scientific freeze](docs/development_scientific_freeze.md);
- [held-out test collection protocol](docs/held_out_test_collection_protocol.md);
- [formal result index](results/formal/README.md).

## Testing and Quality

The repository configures pytest and Ruff in `pyproject.toml`; manual
`PYTHONPATH` configuration is not required.

```powershell
& ".\.venv\Scripts\python.exe" -m pytest -q
& ".\.venv\Scripts\python.exe" -m ruff check src tests
& ".\.venv\Scripts\python.exe" -m ruff format --check src tests
```

## Limitations

- the implemented exercise scope is push-ups only;
- one visible participant is supported;
- evaluation used controlled side and side-diagonal views;
- 2D pose evidence depends on landmark visibility, camera placement and
  recording quality;
- the evaluation dataset is small, particularly the four-clip held-out set;
- the project does not train a new pose model;
- no multi-person tracking or 3D reconstruction is implemented;
- thresholds were calibrated for this controlled project rather than validated
  as universal biomechanical standards;
- alignment can become `unscorable` when required landmarks do not provide
  sufficient valid evidence;
- incomplete-extension classification showed poor held-out generalisation;
- completion timestamps are algorithmic event estimates and can differ from
  visually annotated completion frames even when total repetition counts agree;
- no clinical, rehabilitation or injury-prevention interpretation is provided.

## Data, Privacy and Reproducibility

Raw identifiable recordings are excluded from Git, as are generated experiment
runs under `experiments/`.

Tracked material includes dataset manifests, frozen annotation tables, review
metadata, documentation and formal aggregate evaluation outputs. Video hashes
in the manifests preserve source identity for the recordings used in the
experiments.

Because the raw participant recordings are not stored in the Git repository, a
repository checkout alone cannot regenerate the exact recorded-video inference
runs from source video. The committed formal results preserve the reported
evaluation evidence and provenance, while the source code and fictional example
schemas allow the analysis and evaluation workflow to be inspected and tested.

Development decisions and held-out evaluation are kept explicitly separate to
reduce test-set leakage.

## Academic Context

This repository supports an MSc Artificial Intelligence project investigating
whether transparent temporal processing improves push-up analysis over a simple
threshold baseline under controlled computer-vision conditions.
