# AGENTS.md

## Project purpose

This repository contains an MSc Artificial Intelligence project titled
"Real-Time Calisthenics Analysis Using Computer Vision".

The implemented and approved scope is deliberately narrow:

- one exercise: the push-up;
- one person visible at a time;
- controlled side or side-diagonal camera view;
- OpenCV video/webcam capture;
- MediaPipe Pose landmark estimation;
- baseline and enhanced repetition counting;
- repetition-level classification into:
  - correct;
  - insufficient depth;
  - incomplete elbow extension;
  - shoulder-hip-ankle alignment deviation;
- non-medical feedback;
- recorded-video evaluation comparing the baseline and enhanced methods.

Do not broaden the project to additional exercises, multiple-person tracking,
3D pose estimation, model training, mobile deployment, or medical/injury claims.

## Repository routing

Prioritise these paths:

- `src/main.py`: live webcam baseline entry point.
- `src/run_video.py`: recorded-video baseline runner.
- `src/run_video_enhanced.py`: recorded-video enhanced runner.
- `src/analysis/baseline.py`: baseline analyser.
- `src/analysis/phase_state_machine.py`: enhanced temporal segmentation.
- `src/analysis/enhanced_features.py`: visibility handling, side selection,
  angle extraction, and smoothing.
- `src/analysis/repetition_aggregator.py`: repetition-level alignment collection.
- `src/analysis/repetition_classifier.py`: deterministic repetition classifier.
- `src/pose/`: MediaPipe processing and landmark extraction.
- `src/evaluation/`: diagnostic and summary scripts.
- `tests/`: unit tests.
- `README.md` and `change_log.md`: current documentation.
- `data/annotations/` and `data/manifests/`: future evaluation ground truth.
- `experiments/logs/` and `experiments/outputs/`: generated outputs.
- `results/`: final tables, figures, summaries, and failure cases.

Do not inspect or modify `.venv/`, `.pytest_cache/`, `__pycache__/`, `.git/`,
large raw videos, or generated experiment outputs unless the task explicitly
requires a named file from those locations.

## Environment

The project is developed on Windows using PowerShell and Python 3.12.

Before running project modules in PowerShell, use:

```powershell
$env:PYTHONPATH = "$PWD\src"
```

Preferred test command:

```powershell
& ".\.venv\Scripts\python.exe" -m pytest -q
```

If the virtual environment is unavailable, report that fact before attempting
to create or replace it.

## Scientific and engineering guardrails

1. Preserve the approved scope.
2. Treat thresholds as provisional operational project thresholds, not
   universal definitions of correct push-up form.
3. Do not make medical, injury-prevention, rehabilitation, or clinical claims.
4. Keep baseline and enhanced methods clearly separated.
5. Do not silently improve, tune, or refactor the baseline in a way that makes
   the comparison unfair.
6. Do not change thresholds without:
   - naming the exact threshold;
   - explaining the reason;
   - identifying whether it was tuned on development data;
   - updating configuration and documentation;
   - adding or updating tests.
7. Never tune parameters on the final test set.
8. Preserve raw measurements and log enough metadata for reproducibility.
9. Avoid data leakage between development/calibration and final evaluation.
10. Prefer transparent deterministic logic over unnecessary machine learning.

## Change discipline

For every implementation task:

1. Inspect the relevant files and tests first.
2. State the intended change and the files expected to change.
3. Make the smallest coherent change.
4. Add or update focused tests.
5. Run the relevant tests, then run the full test suite.
6. Review the Git diff for accidental edits.
7. Summarise:
   - files changed;
   - behaviour changed;
   - tests run and results;
   - unresolved risks;
   - commands needed to reproduce the result.

Do not perform broad rewrites, mass formatting, dependency upgrades, or
architecture changes unless explicitly requested.

Do not commit, push, delete branches, remove data, or overwrite experimental
results without explicit permission.

## Generated files and logging

Generated CSVs must not silently append a second complete run to an existing
experiment file. A run must either:

- fail clearly when the output exists;
- overwrite only when an explicit `--overwrite` option is supplied; or
- use a unique run identifier.

Never mix runs silently.

Frame-level logs and repetition-level outputs must include enough identifiers
to trace them to the input clip, configuration, and method.

## Current known issues to verify

Do not assume these descriptions are correct without checking the code, but
investigate them early:

1. `src/run_video_enhanced.py` appears to reference
   `repetition_logger.close` without calling it.
2. The enhanced runner appears to initialise `feedback_test` but later use
   `feedback_text`.
3. `CSVLogger` opens files in append mode, which can duplicate complete runs.
4. The baseline depth and extension frame-warning conditions may be unreachable
   because the position thresholds and warning thresholds are identical.
5. Enhanced side selection is driven by elbow visibility while body alignment
   requires shoulder, hip, and ankle visibility on the selected side. This may
   explain very low alignment coverage and `unscorable` repetitions.
6. The README contains statements that may now be stale because enhanced
   temporal segmentation and repetition classification have been implemented.

## Preferred solution for alignment visibility

Investigate before implementing. The likely design is feature-specific stable
side selection:

- one stable selector for elbow features;
- one stable selector for alignment features;
- log both selected sides and both left/right visibility scores;
- reset only the smoother associated with the feature whose side changed;
- never insert stale values on invalid frames;
- preserve temporal consistency;
- add focused tests;
- compare alignment coverage before and after on the same development clip.

Do not merely lower `minimum_alignment_valid_ratio` to hide missing data.

## Evaluation expectations

The eventual formal evaluation should use the same manually annotated test
clips for baseline and enhanced methods.

Keep development/calibration data separate from final test data.

Expected outputs should include, where supported by the annotations:

- ground-truth repetition count per clip;
- predicted repetition count per method;
- count error or absolute count error;
- repetition event matching rule and tolerance;
- classification confusion matrix;
- per-class precision, recall, and F1;
- macro F1 and overall accuracy;
- unscorable rate and alignment coverage;
- feature availability and side-switch statistics;
- processing time per frame and effective throughput;
- documented failure cases.

Do not invent results. Report missing annotations or insufficient evidence.

## Documentation

Update `README.md` when user-facing setup, commands, outputs, or current
implementation status changes.

Update `change_log.md` for material implementation/evaluation milestones.

Keep dissertation prose separate from source-code comments. Codex may help
organise evidence and outline technical descriptions, but every factual claim
must be traceable to code, logs, annotations, results, or cited literature.
