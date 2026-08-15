# Formal Evaluation Results

This directory contains the preserved formal evaluation evidence for the
project.

## Primary results

### Final development result

`development/development_calibrated65_t050_*`

This is the primary frozen development evaluation after:

1. causal return-top extension finalisation;
2. development-only calibration of the enhanced insufficient-depth threshold
   to 65 degrees;
3. freezing the primary event-matching tolerance at 0.50 seconds.

Primary development results:

- enhanced detection precision: 0.975;
- enhanced detection recall: 0.9069767442;
- enhanced detection F1: 0.9397590361;
- exact-count clip accuracy: 11/12;
- enhanced matched-repetition classification accuracy: 0.8717948718;
- enhanced matched-repetition macro F1: 0.8694444444.

### Final held-out test result

`test/final_test_t050_*`

This is the single held-out evaluation performed after development
configuration and test ground truth were frozen.

Primary held-out results:

- both methods predicted 28 repetitions for 28 ground-truth repetitions;
- baseline event precision/recall/F1: 0.3571428571;
- enhanced event precision/recall/F1: 0.6428571429;
- baseline completion-timing MAE: 0.4631111312 seconds;
- enhanced completion-timing MAE: 0.3351848587 seconds;
- enhanced matched-repetition classification accuracy: 0.7777777778;
- enhanced matched-repetition macro F1: 0.7045454545.

The held-out classification result exposed a remaining limitation:
all four temporally matched ground-truth incomplete-extension repetitions were
classified as correct. The frozen system was not retuned after this result.

## Development sensitivity analysis

The following runs vary only the event-matching tolerance:

- `development_calibrated65_t025_*` - 0.25 seconds;
- `development_calibrated65_t050_*` - 0.50 seconds, primary;
- `development_calibrated65_t075_*` - 0.75 seconds;
- `development_calibrated65_t100_*` - 1.00 seconds.

These runs are retained as sensitivity evidence. The 0.50-second value was
retained as the primary tolerance rather than selecting the post-hoc
development value producing the highest F1.

## Historical development results

`development_t050_*`

Original formal development evaluation before the causal return-top extension
fix and depth-threshold calibration.

`development_postfix_t050_*`

Formal development evaluation after the causal return-top extension fix but
before the enhanced depth threshold was calibrated from 100 to 65 degrees.

These historical runs are intentionally retained to preserve the experimental
and debugging trail. They are not the final reported development result.

## File families

Each formal run contains:

- `*_formal_evaluation.json` - complete machine-readable evaluation result;
- `*_evaluation_metadata.json` - provenance and execution metadata;
- `*_per_clip_metrics.csv` - per-clip metrics;
- `*_classification_confusion_matrix.csv` - enhanced classification matrix;
- `*_classification_per_class.csv` - enhanced per-class metrics;
- `*_detection_recall_by_class.csv` - event detection recall by ground-truth
  class.

Development results must not be presented as held-out generalisation
performance. The final held-out test results must not be used for retuning.
