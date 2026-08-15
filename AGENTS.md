# AGENTS.md

This file provides guidance for automated coding agents and contributors; it is
not examiner-facing scientific evidence.

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
  - unscorable when evidence is insufficient;
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
- `src/evaluation/`: annotation, validation, diagnostic, summary and formal
  evaluation modules.
- `tests/`: unit tests.
- `README.md` and `change_log.md`: current documentation.
- `data/annotations/` and `data/manifests/`: frozen development and held-out
  evaluation ground truth and dataset metadata.
- `experiments/logs/` and `experiments/outputs/`: generated outputs.
- `results/`: final tables, figures, summaries, and failure cases.

Do not inspect or modify `.venv/`, `.pytest_cache/`, `__pycache__/`, `.git/`,
large raw videos, or generated experiment outputs unless the task explicitly
requires a named file from those locations.

## Environment

The project is developed on Windows using PowerShell and Python 3.12.

Run user-facing commands from the repository root through the entry points in
`src/`. The live and recorded runners, dataset validator, annotation viewer and
formal evaluator do not require a manually configured `PYTHONPATH`.

Preferred test command:

```powershell
& ".\.venv\Scripts\python.exe" -m pytest -q
```

If the virtual environment is unavailable, report that fact before attempting
to create or replace it.

## Scientific and engineering guardrails

1. Preserve the approved scope.
2. Treat thresholds as frozen, project-specific operational thresholds, not
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

## Current evaluation cautions

- The primary `0.5`-second event tolerance is frozen from development evidence
  and must not be retuned from held-out results.
- Alignment availability is a reported evidence limitation. Do not hide it by
  lowering `minimum_alignment_valid_ratio` or introducing feature-specific side
  selection without new approved development evidence.
- Baseline warnings remain diagnostic frame messages and must not be converted
  into repetition classes.

## Evaluation expectations

The completed formal evaluation used the same manually annotated test clips
for baseline and enhanced methods.

Keep development/calibration data separate from final test data.

The committed formal outputs include, where supported by the annotations:

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
