# Project Change Log

## 28 July 2026 — Alignment visibility diagnostics

- Added left/right elbow and body-alignment visibility scores to enhanced
  frame-level logging.
- Added an explicit selected-elbow-side field and a diagnostic flag showing
  whether the opposite side had valid alignment landmarks.
- Added a development-data diagnostic summary covering elbow and alignment
  validity, opposite-side rescue opportunities, side changes, phase-grouped
  availability, repetition alignment coverage and unscorable repetitions.
- Made repetition summaries reject duplicate `(clip_id, rep_id)` rows.
- Added focused feature-processing and diagnostic-summary tests.

No selector behaviour, temporal threshold, classifier threshold or classifier
priority was changed.

## 28 July 2026 — Output integrity and resource cleanup

- Changed CSV creation to fail clearly when an output already exists instead
  of silently appending another complete run.
- Added an explicit `--overwrite` option to the live, recorded baseline and
  recorded enhanced runners.
- Added a paired preflight check for the enhanced frame-level and
  repetition-level output paths.
- Ensured new and explicitly overwritten CSVs always receive a header,
  including replacement of an existing zero-byte file.
- Ensured captures, pose estimators, CSV loggers and OpenCV windows are cleaned
  up when setup or processing raises an exception.
- Fixed the enhanced repetition logger cleanup and made the feedback text
  variable consistent.
- Added focused tests for CSV collision policy, overwrite behaviour,
  zero-byte files, paired output checks and runner resource cleanup.

No analysis thresholds, temporal behaviour, side selection, baseline semantics
or classifier priority were changed.

## 1 July 2026 — Scope reduction

### Original scope

The original project definition proposed a general real-time calisthenics-analysis system covering multiple exercises.

### Revised scope

The implementation and evaluation scope was reduced to one exercise: the push-up.

The revised controlled conditions are:

- one person;
- a controlled side or side-diagonal camera view;
- ordinary webcam or recorded video input;
- MediaPipe Pose landmark estimation;
- repetition counting;
- three predefined observable form-deviation categories:
  - insufficient depth;
  - incomplete elbow extension;
  - shoulder-hip-ankle alignment deviation;
- non-medical feedback.

### Reason for the change

The original multi-exercise scope was too broad for the remaining implementation and evaluation period. Limiting the project to one exercise allows a complete implementation, manually labelled evaluation, baseline-versus-enhanced comparison and critical failure-case analysis.

### Approval

The revised one-exercise push-up scope was discussed with and approved by supervisor Nafi Ahmad before implementation resumed on 1 July 2026.
