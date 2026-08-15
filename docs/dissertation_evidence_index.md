# Dissertation evidence index

This is an internal evidence map for writing the dissertation. It is not
dissertation prose and does not replace the machine-readable results. Values
below are transcribed from tracked files; linked JSON and CSV files retain the
authoritative full floating-point precision.

## 1. Scope and scientific freeze

- The original multi-exercise proposal, approved reduction to push-up only and
  controlled operating conditions are recorded in
  [`change_log.md`](../change_log.md#1-july-2026--scope-reduction).
- The current push-up-only, one-visible-person, side or side-diagonal scope is
  summarised in the repository [`README`](../README.md#what-the-system-does) and
  [`AGENTS.md`](../AGENTS.md#project-purpose).
- [`configs/default.yaml`](../configs/default.yaml) is the authoritative frozen
  runtime configuration. The decision record is
  [`development_scientific_freeze.md`](development_scientific_freeze.md).

Key frozen enhanced settings are:

| Setting | Frozen value | Source |
| --- | ---: | --- |
| EMA alpha | 0.3 | [`configs/default.yaml`](../configs/default.yaml) |
| Top / bottom region | 130 / 120 degrees | [`configs/default.yaml`](../configs/default.yaml) |
| Hysteresis | 5 degrees | [`configs/default.yaml`](../configs/default.yaml) |
| Phase confirmation | 3 frames | [`configs/default.yaml`](../configs/default.yaml) |
| Missing-angle grace | 5 frames | [`configs/default.yaml`](../configs/default.yaml) |
| Minimum repetition duration | 8 frames | [`configs/default.yaml`](../configs/default.yaml) |
| Depth / extension thresholds | 65 / 150 degrees | [`configs/default.yaml`](../configs/default.yaml) |
| Alignment minimum | 160 degrees | [`configs/default.yaml`](../configs/default.yaml) |
| Alignment deviation rule | at least 3 frames and 0.20 of valid observations | [`configs/default.yaml`](../configs/default.yaml) |
| Minimum valid alignment coverage | 0.50 | [`configs/default.yaml`](../configs/default.yaml) |
| Primary event tolerance | 0.50 seconds | [`development_scientific_freeze.md`](development_scientific_freeze.md#frozen-event-matching-tolerance) |

Tracked freeze history:

| Commit / identity | Meaning | Tracked evidence |
| --- | --- | --- |
| `4945443` | Development ground truth frozen | [`development_scientific_freeze.md`](development_scientific_freeze.md#status) |
| `abbad69` | Causal return-top extension finalisation | [`development_scientific_freeze.md`](development_scientific_freeze.md#status) |
| `2ed7d75` | Development-only depth calibration | [`development_scientific_freeze.md`](development_scientific_freeze.md#status) |
| `40a204c` | Complete development scientific configuration frozen | [`held_out_test_collection_protocol.md`](held_out_test_collection_protocol.md#purpose) |
| `a0407d1` | Commit recorded by the final held-out result after test ground-truth freeze | [`final_test_t050_evaluation_metadata.json`](../results/formal/test/final_test_t050_evaluation_metadata.json) |

## 2. Dataset and ground truth

| Evidence | Development | Held-out test | Sources |
| --- | ---: | ---: | --- |
| Clips | 12 | 4 | [development manifest](../data/manifests/development_dataset_manifest.csv); [test manifest](../data/manifests/test_dataset_manifest.csv) |
| Evaluable attempts | 43 | 28 | [development annotations](../data/annotations/development_repetition_annotations.csv); [test annotations](../data/annotations/test_repetition_annotations.csv) |
| Non-evaluable ambiguous fragments | 1 | 0 | Same annotation CSVs |
| `correct` evaluable attempts | 14 | 13 | Same annotation CSVs |
| `insufficient_depth` evaluable attempts | 9 | 5 | Same annotation CSVs |
| `incomplete_extension` evaluable attempts | 10 | 5 | Same annotation CSVs |
| `alignment_deviation` evaluable attempts | 10 | 4 | Same annotation CSVs |
| `unscorable` evaluable attempts | 0 | 1 | Same annotation CSVs |

The single non-evaluable development row is retained as an ambiguous
`unscorable` fragment but is excluded from event and classification metrics
under the [manual annotation protocol](manual_annotation_protocol.md#evaluable-attempts-and-ambiguous-fragments).

Participant scope:

- Development contains six self-recorded clips under `P_DEV_01` and six
  external clips with clip-local `P_EXT_KAGGLE_*` unknown-identity surrogates.
  Those surrogates do not establish six independent people. Sources:
  [development manifest](../data/manifests/development_dataset_manifest.csv)
  and [`data/README.md`](../data/README.md#external-development-source).
- All four held-out clips use `P_TEST_01`. They are fresh held-out recordings
  from one anonymised participant, so they are clip-level rather than
  participant-independent evidence. Sources:
  [test manifest](../data/manifests/test_dataset_manifest.csv) and
  [collection protocol](held_out_test_collection_protocol.md#test-set-structure).

Annotation freeze evidence:

| Split | Frozen SHA-256 | Review status / finalised UTC | Source |
| --- | --- | --- | --- |
| Development | `307c9a15b2604d838e453c540d370d13570c43e92f7c45fb798c6d58d3a2fdb7` | complete / `2026-08-13T21:14:22.922122Z` | [development review metadata](../data/annotations/development_repetition_annotations.review.json) |
| Test | `ca8f93ac28095d70ce207b0bfa27b33b628f3966b90510ad78826eb32fd8e27c` | complete / `2026-08-15T01:46:08.846195Z` | [test review metadata](../data/annotations/test_repetition_annotations.review.json) |

## 3. Development progression

Detection was unchanged across these `0.50`-second runs; the progression is in
the enhanced repetition classification result.

| Run | Status | Enhanced detection F1 | Matched classification n | Accuracy | Macro F1 | Source |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `development_t050` | Historical: before return-top fix and depth calibration | 0.9397590361 | 39 | 0.4358974359 | 0.3335622711 | [formal JSON](../results/formal/development/development_t050_formal_evaluation.json) |
| `development_postfix_t050` | Historical: after return-top fix, before depth calibration | 0.9397590361 | 39 | 0.6666666667 | 0.6016081871 | [formal JSON](../results/formal/development/development_postfix_t050_formal_evaluation.json) |
| `development_calibrated65_t050` | Primary frozen development result | 0.9397590361 | 39 | 0.8717948718 | 0.8694444444 | [formal JSON](../results/formal/development/development_calibrated65_t050_formal_evaluation.json) |

The authoritative status distinction is in the
[formal result index](../results/formal/README.md#historical-development-results).

## 4. Primary development metrics

Primary source:
[`development_calibrated65_t050_formal_evaluation.json`](../results/formal/development/development_calibrated65_t050_formal_evaluation.json).
Per-clip values are in
[`development_calibrated65_t050_per_clip_metrics.csv`](../results/formal/development/development_calibrated65_t050_per_clip_metrics.csv).

| Detection metric | Baseline | Enhanced |
| --- | ---: | ---: |
| Ground truth / predictions | 43 / 40 | 43 / 40 |
| Matches / misses / extras | 31 / 12 / 9 | 39 / 4 / 1 |
| Precision | 0.775 | 0.975 |
| Recall | 0.7209302326 | 0.9069767442 |
| F1 | 0.7469879518 | 0.9397590361 |
| Exact-count clips | 9/12 (0.75) | 11/12 (0.9166666667) |
| Count MAE | 0.9166666667 | 0.25 |
| Completion-timing MAE | 0.3370379159 s | 0.2417134619 s |

Enhanced classification used 39 temporally matched repetitions: accuracy
`0.8717948718`, macro F1 `0.8694444444`. Sources:
[formal JSON](../results/formal/development/development_calibrated65_t050_formal_evaluation.json),
[per-class CSV](../results/formal/development/development_calibrated65_t050_classification_per_class.csv)
and [confusion matrix CSV](../results/formal/development/development_calibrated65_t050_classification_confusion_matrix.csv).

| Matched GT class | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| `correct` | 13 | 0.7647058824 | 1.0 | 0.8666666667 |
| `insufficient_depth` | 9 | 0.8888888889 | 0.8888888889 | 0.8888888889 |
| `incomplete_extension` | 10 | 1.0 | 0.8 | 0.8888888889 |
| `alignment_deviation` | 7 | 1.0 | 0.7142857143 | 0.8333333333 |
| `unscorable` | 0 | 0.0 | 0.0 | 0.0 |

Per-class support is the matched classification subset, not the full annotation
class total.

## 5. Event-tolerance sensitivity

All rows use the calibrated-65 development outputs. `0.50` seconds remains the
primary frozen value; the other rows are sensitivity evidence only.

| Tolerance | Baseline P / R / F1 | Enhanced P / R / F1 | Enhanced matches | Source |
| ---: | --- | --- | ---: | --- |
| 0.25 s | 0.025 / 0.0232558140 / 0.0240963855 | 0.525 / 0.4883720930 / 0.5060240964 | 21 | [formal JSON](../results/formal/development/development_calibrated65_t025_formal_evaluation.json) |
| **0.50 s primary** | 0.775 / 0.7209302326 / 0.7469879518 | 0.975 / 0.9069767442 / 0.9397590361 | 39 | [formal JSON](../results/formal/development/development_calibrated65_t050_formal_evaluation.json) |
| 0.75 s | 0.8 / 0.7441860465 / 0.7710843373 | 1.0 / 0.9302325581 / 0.9638554217 | 40 | [formal JSON](../results/formal/development/development_calibrated65_t075_formal_evaluation.json) |
| 1.00 s | 0.825 / 0.7674418605 / 0.7951807229 | 1.0 / 0.9302325581 / 0.9638554217 | 40 | [formal JSON](../results/formal/development/development_calibrated65_t100_formal_evaluation.json) |

The pre-existing `0.50`-second value was retained rather than selecting a
post-hoc maximum; see the
[freeze rationale](development_scientific_freeze.md#frozen-event-matching-tolerance).

## 6. Held-out final metrics

Primary source:
[`final_test_t050_formal_evaluation.json`](../results/formal/test/final_test_t050_formal_evaluation.json).
Per-clip values are in
[`final_test_t050_per_clip_metrics.csv`](../results/formal/test/final_test_t050_per_clip_metrics.csv).

| Detection metric | Baseline | Enhanced |
| --- | ---: | ---: |
| Ground truth / predictions | 28 / 28 | 28 / 28 |
| Matches / misses / extras | 10 / 18 / 18 | 18 / 10 / 10 |
| Precision | 0.3571428571 | 0.6428571429 |
| Recall | 0.3571428571 | 0.6428571429 |
| F1 | 0.3571428571 | 0.6428571429 |
| Exact-count clips | 4/4 (1.0) | 4/4 (1.0) |
| Count MAE | 0.0 | 0.0 |
| Completion-timing MAE | 0.4631111312 s | 0.3351848587 s |

Enhanced classification used 18 temporally matched repetitions: accuracy
`0.7777777778`, macro F1 `0.7045454545`.

| Matched GT class | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| `correct` | 9 | 0.6923076923 | 1.0 | 0.8181818182 |
| `insufficient_depth` | 3 | 1.0 | 1.0 | 1.0 |
| `incomplete_extension` | 4 | 0.0 | 0.0 | 0.0 |
| `alignment_deviation` | 2 | 1.0 | 1.0 | 1.0 |
| `unscorable` | 0 | 0.0 | 0.0 | 0.0 |

Source:
[`final_test_t050_classification_per_class.csv`](../results/formal/test/final_test_t050_classification_per_class.csv).

Held-out confusion matrix (rows are ground truth; columns are predictions):

| GT \\ predicted | correct | insufficient depth | incomplete extension | alignment deviation | unscorable |
| --- | ---: | ---: | ---: | ---: | ---: |
| correct | 9 | 0 | 0 | 0 | 0 |
| insufficient depth | 0 | 3 | 0 | 0 | 0 |
| incomplete extension | 4 | 0 | 0 | 0 | 0 |
| alignment deviation | 0 | 0 | 0 | 2 | 0 |
| unscorable | 0 | 0 | 0 | 0 | 0 |

Source:
[`final_test_t050_classification_confusion_matrix.csv`](../results/formal/test/final_test_t050_classification_confusion_matrix.csv).

Enhanced detection recall by full ground-truth class:

| GT class | Full support | Matched | Missed | Recall |
| --- | ---: | ---: | ---: | ---: |
| `correct` | 13 | 9 | 4 | 0.6923076923 |
| `insufficient_depth` | 5 | 3 | 2 | 0.6 |
| `incomplete_extension` | 5 | 4 | 1 | 0.8 |
| `alignment_deviation` | 4 | 2 | 2 | 0.5 |
| `unscorable` | 1 | 0 | 1 | 0.0 |

Source:
[`final_test_t050_detection_recall_by_class.csv`](../results/formal/test/final_test_t050_detection_recall_by_class.csv).

## 7. Error and failure-analysis evidence

- The held-out set contains five annotated incomplete-extension attempts. Four
  were temporally matched, and all four were predicted `correct`; the fifth was
  a detection miss. This is supported jointly by the
  [detection-recall CSV](../results/formal/test/final_test_t050_detection_recall_by_class.csv)
  and [confusion matrix](../results/formal/test/final_test_t050_classification_confusion_matrix.csv).
- The [per-clip CSV](../results/formal/test/final_test_t050_per_clip_metrics.csv)
  localises the four classification errors to three errors in `test02` and one
  in `test04`; it does not retain per-repetition matched identities or the
  class of each error. The aggregate confusion matrix supplies the class-level
  evidence. No unsupported causal explanation should be inferred.
- Five primary-development matched repetitions remained misclassified. The
  exact off-diagonal cells are in the
  [development confusion matrix](../results/formal/development/development_calibrated65_t050_classification_confusion_matrix.csv),
  while the decision not to tune them away is documented in
  [`development_scientific_freeze.md`](development_scientific_freeze.md#remaining-classification-errors).

The formal report set does not include a per-repetition matched-pair export or
failure-case images. Those are not present in formal evidence.

## 8. Runtime and feature-availability evidence

Definitions and denominators are fixed by
[`formal_evaluation_reporting.md`](formal_evaluation_reporting.md#timing-aggregation).
Processing times are milliseconds in the explicitly timed per-frame
computer-vision/analysis region. Measured throughput is timed frame
observations divided by summed processing time; it is not source-video FPS.
Aggregate availability rates pool frame numerators and denominators.

### Processing performance

| Split / method | Timed frames | Total processing ms | Mean ms/frame | Median ms/frame | Measured throughput FPS | Source |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Development baseline | 4,913 | 230374.923300 | 46.890886 | 46.017400 | 21.326106 | [formal JSON](../results/formal/development/development_calibrated65_t050_formal_evaluation.json) |
| Development enhanced | 4,913 | 231837.778000 | 47.188638 | 46.260700 | 21.191542 | [formal JSON](../results/formal/development/development_calibrated65_t050_formal_evaluation.json) |
| Held-out baseline | 4,574 | 236070.557200 | 51.611403 | 49.506050 | 19.375563 | [formal JSON](../results/formal/test/final_test_t050_formal_evaluation.json) |
| Held-out enhanced | 4,574 | 236625.595600 | 51.732749 | 49.260100 | 19.330115 | [formal JSON](../results/formal/test/final_test_t050_formal_evaluation.json) |

### Frame availability and side stability

Each availability cell is `available / denominator (rate)`. Denominators equal
the analysed frame count for these primary results.

| Split / method | Pose | Elbow | Alignment | Selected side | Side changes | Source |
| --- | --- | --- | --- | --- | ---: | --- |
| Development baseline | 4913/4913 (1.0) | 4886/4913 (0.994504) | 4886/4913 (0.994504) | 4886/4913 (0.994504) | 8 instantaneous selected-side state changes | [formal JSON](../results/formal/development/development_calibrated65_t050_formal_evaluation.json) |
| Development enhanced | 4913/4913 (1.0) | 4856/4913 (0.988398) | 4872/4913 (0.991655) | 4872/4913 (0.991655) | 18 stable-selector `side_changed` events | [formal JSON](../results/formal/development/development_calibrated65_t050_formal_evaluation.json) |
| Held-out baseline | 4574/4574 (1.0) | 4574/4574 (1.0) | 4574/4574 (1.0) | 4574/4574 (1.0) | 0 instantaneous selected-side state changes | [formal JSON](../results/formal/test/final_test_t050_formal_evaluation.json) |
| Held-out enhanced | 4574/4574 (1.0) | 4566/4574 (0.998251) | 4566/4574 (0.998251) | 4566/4574 (0.998251) | 4 stable-selector `side_changed` events | [formal JSON](../results/formal/test/final_test_t050_formal_evaluation.json) |

The baseline and enhanced side-change semantics differ and must not be treated
as the same state variable.

### Repetition and human alignment evidence

| Evidence | Development | Held-out test | Source |
| --- | ---: | ---: | --- |
| Enhanced predicted repetitions | 40 | 28 | Primary formal JSONs above |
| Predicted `unscorable` count / rate | 0 / 0.0 | 0 / 0.0 | Primary formal JSONs above |
| Alignment-coverage observations | 40 | 28 | Primary formal JSONs above |
| Mean model `alignment_valid_ratio` | 1.0 | 1.0 | Primary formal JSONs above |
| Human-evaluable attempts | 43 | 28 | Primary formal JSONs above |
| Human alignment evidence adequate | 43/43 (1.0) | 27/28 (0.9642857143) | Primary formal JSONs above |
| Human alignment evidence inadequate | 0 | 1 | Primary formal JSONs above |

Model alignment coverage and human source-video assessability are separate
measures. Per-clip versions of every field above are in the primary
[development](../results/formal/development/development_calibrated65_t050_per_clip_metrics.csv)
and [held-out](../results/formal/test/final_test_t050_per_clip_metrics.csv)
per-clip CSVs.

Not present in the formal evidence: loop wall time, processing-time IQR,
repeated-run variability, confidence intervals and hardware identity. These
must not be inferred from the recorded mean, median or throughput values.

## 9. Repository and testing evidence

- Pytest discovery and Python-path configuration are in
  [`pyproject.toml`](../pyproject.toml); pinned test and lint dependencies are in
  [`requirements-dev.txt`](../requirements-dev.txt).
- The [`tests/`](../tests/) directory contains focused unit and integration
  coverage, including direct user-entry-point help checks in
  [`test_cli_entrypoints.py`](../tests/test_cli_entrypoints.py).
- [`change_log.md`](../change_log.md) records historical additions to focused
  test coverage, but no tracked file records the latest complete-suite result.

**Local certification to preserve during final clean-clone verification:** the
current working-session report is `410 passed` with two third-party protobuf
deprecation warnings, with Ruff check and Ruff format check passing. This is a
local certification note supplied for final packaging, not pre-existing
committed evidence. It should be replaced or supplemented by a preserved
clean-clone verification record if one is created before submission.

## 10. Dissertation-ready outputs still to create

Create these presentation artefacts from the linked evidence without rerunning
or retuning the held-out evaluation:

- primary baseline-versus-enhanced development and held-out results table;
- held-out enhanced classification confusion-matrix figure;
- concise error-analysis table or examples, explicitly including the
  incomplete-extension failure;
- threshold and event-tolerance rationale text based on the scientific freeze;
- runtime and feature-availability table using the recorded units and
  denominators above;
- limitations text covering the four-clip, one-participant held-out scope,
  unmatched-event classification subset and missing repeated-run uncertainty.

The [formal result index](../results/formal/README.md) remains the authoritative
guide to primary, sensitivity and historical result families. Its dirty-state
provenance note must be retained when describing reproducibility.
