# Development Scientific Freeze

## Status

Development-stage scientific decisions are frozen before held-out test
evaluation.

The final development configuration is based only on the frozen development
ground truth and development-set experiments. Held-out test data must not be
used to alter thresholds, matching tolerance, class priority, segmentation,
smoothing, side selection, or any other analysis rule.

The development ground-truth annotations remain frozen with SHA-256:

`307c9a15b2604d838e453c540d370d13570c43e92f7c45fb798c6d58d3a2fdb7`

Relevant implementation history:

- `4945443` - freeze development ground-truth annotations;
- `abbad69` - fix causal return-top extension finalisation;
- `2ed7d75` - calibrate enhanced depth threshold on development data.

## Frozen enhanced configuration

The final enhanced classification settings are:

- insufficient-depth threshold: `65.0` degrees;
- incomplete-extension threshold: `150.0` degrees;
- minimum alignment threshold: `160.0` degrees;
- alignment-deviation minimum frames: `3`;
- alignment-deviation minimum ratio: `0.20`;
- minimum valid alignment coverage: `0.50`.

The enhanced segmentation and temporal-processing configuration is otherwise
unchanged from the committed default configuration.

The baseline remains unchanged and intentionally simple. Its repetition and
warning thresholds are not modified by enhanced-system calibration.

## Depth-threshold calibration

The enhanced insufficient-depth threshold was calibrated using frozen
development data only.

Matched development repetitions showed substantial separation between
insufficient-depth attempts and the other classes, with one overlapping
insufficient-depth example. A development-only threshold sweep identified a
broad stable region around 65-70 degrees. `65.0` degrees was selected as the
lowest stable value above the largest observed correctly labelled depth value
(`64.06` degrees).

The development annotations were not changed as a consequence of system
predictions.

## Remaining classification errors

At the frozen primary event tolerance, the enhanced classifier correctly
classified 34 of 39 matched development repetitions:

- accuracy: `0.8717948718`;
- macro F1: `0.8694444444`.

Five matched repetitions remained misclassified. Inspection showed that these
cases could not be resolved cleanly by moving the existing scalar thresholds
without creating new errors in correctly classified development repetitions.
The remaining disagreements are therefore retained as development limitations
rather than tuned away.

No further classification-threshold tuning is permitted using held-out test
results.

## Frozen event-matching tolerance

The primary formal event-matching tolerance is frozen at `0.50` seconds.

Development sensitivity analysis produced:

| Tolerance (s) | Baseline F1 | Enhanced F1 | Enhanced matches |
| ---: | ---: | ---: | ---: |
| 0.25 | 0.024 | 0.506 | 21 |
| 0.50 | 0.747 | 0.940 | 39 |
| 0.75 | 0.771 | 0.964 | 40 |
| 1.00 | 0.795 | 0.964 | 40 |

The pre-existing `0.50`-second value is retained as the primary tolerance
rather than selecting the post-hoc value that maximises development F1.
The wider values are retained only as sensitivity evidence. Across all
reasonable tolerances from 0.50 to 1.00 seconds, the enhanced method
substantially outperforms the baseline.

Held-out test evaluation must use the frozen `0.50`-second tolerance.

## Primary frozen development result

At `0.50` seconds the enhanced detector achieved:

- event precision: `0.975`;
- event recall: `0.9069767442`;
- event F1: `0.9397590361`;
- matched events: `39 / 43`;
- exact-count clip accuracy: `11 / 12` (`0.9166666667`);
- mean absolute count error: `0.25`;
- mean absolute completion-timing error: `0.2417134619` seconds.

At the same tolerance the baseline detector achieved:

- event precision: `0.775`;
- event recall: `0.7209302326`;
- event F1: `0.7469879518`.

The enhanced matched-repetition classification result was:

- evaluated matched repetitions: `39`;
- accuracy: `0.8717948718`;
- macro F1: `0.8694444444`.

These are development results and must not be presented as held-out
generalisation performance.

## Final-test rule

After this freeze:

1. held-out test clips are defined independently of development tuning;
2. test ground truth is annotated without viewing system predictions;
3. test annotations are reviewed and frozen before formal system evaluation;
4. the frozen implementation and `0.50`-second tolerance are used unchanged;
5. final-test results are reported once and are not used for retuning.

Any post-freeze code change capable of affecting scientific outputs must be
explicitly documented and must invalidate the existing scientific freeze
before held-out evaluation proceeds.
