# Project Change Log

## 30 July 2026 - Repetition-event detection evaluation

- Added validated baseline completion-event extraction from cumulative-count
  increments without changing baseline processing.
- Added enhanced repetition-event and evaluable ground-truth event loaders.
- Added deterministic chronological one-to-one matching that first maximises
  match count, then minimises total absolute timing error, with a documented
  deterministic tie rule.
- Added per-clip, per-method count, matching, precision, recall, F1 and
  completion-timing summaries with the provisional 0.5-second tolerance.
- Added a JSON stdout CLI and focused fictional-data tests for tolerance
  boundaries, ambiguous candidate sets, matching objectives, missed/extra
  events, delayed completions, variable FPS and malformed inputs.

No classification metric, plot, runtime comparison or final-test evaluation
was added. No analysis threshold, baseline behavior, segmentation, classifier,
side selector, UI or historical generated result was changed.

## 30 July 2026 - Runtime configuration and run provenance

- Made `configs/default.yaml` the validated primary settings source for both
  recorded-video runners while preserving every existing numeric value.
- Added typed pose, baseline, feature, segmentation and classification
  configuration objects with required-field, unknown-field, type and range
  validation.
- Added mandatory development/test split identity, optional run IDs that
  default to clip IDs, and an explicit recorded `--alpha` precedence rule.
- Added a shared run ID to baseline frame, enhanced frame and enhanced
  repetition CSV rows.
- Added atomic JSON run metadata containing input/config hashes, source video
  properties, the complete resolved configuration, explicit overrides,
  installed core-library versions, Git state, UTC lifecycle timestamps,
  timing-boundary definitions and generated-output paths.
- Extended output preflight to each run's complete CSV/metadata set and made
  explicit overwrite remove that complete old set before opening new writers.
- Added focused tests for config validation and serialisation, override
  precedence, split validation, hashing, mocked software/Git provenance,
  metadata lifecycle and complete-set collision handling.
- Pinned `requirements.txt` to the compatible versions already installed in
  the existing Python 3.12 environment; no package was upgraded.

No threshold, temporal behavior, selector behavior, baseline semantics,
classifier priority, annotation schema, UI behavior or historical experiment
output was changed. Event matching and formal metrics remain unimplemented.

## 29 July 2026 - Evaluation dataset foundation

- Saved the approved baseline-versus-enhanced comparison design.
- Defined CSV schemas for clip manifests, evaluable push-up attempts and
  ambiguous movement fragments.
- Added validation for allowed splits, camera views, form labels, visibility
  states, frame ordering and bounds, duplicate identifiers, unknown clips,
  attempt status and single-label priority.
- Added fictional manifest and annotation examples without assigning labels to
  any existing recording.
- Documented the manual annotation procedure and added focused schema tests.

Event matching and formal evaluation metrics are not implemented in this
milestone. No analysis threshold, state machine, classifier, side selector or
UI behaviour was changed.

## 28 July 2026 - Consistent repetition measurement windows

- Defined each enhanced repetition as one closed, inclusive interval from the
  maximum genuine top observation before descent through the frame confirming
  the completed return to top.
- Made repetition start, end, duration, top angles, minimum elbow angle,
  body-alignment observations and the alignment-coverage denominator use that
  same interval.
- Buffered only the contiguous descent-candidate sequence that confirms the
  transition. Candidate measurements are backfilled on confirmation, while an
  interrupted or noisy sequence is discarded before it can affect the
  repetition window or classification.
- Restricted the starting extension measurement to genuine top-region
  observations and the ending extension measurement to the contiguous valid
  frames that confirm the return to top.
- Kept tolerated missing-observation frames in the duration and eligible for
  feature aggregation, without inserting fabricated alignment values.
- Added integrated state-machine, aggregation and classification tests for
  complete and missing alignment coverage, non-monotonic candidate descent,
  brief missing elbow observations, genuine top anchoring, return-to-top
  measurement and abandoned attempts.
- Split the alignment diagnostic into final predicted-class `unscorable`
  repetitions and repetitions with independently unscorable alignment
  evidence.

No threshold, hysteresis rule, confirmation count, missing-frame tolerance,
classifier priority or side-selection behaviour was changed.

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
