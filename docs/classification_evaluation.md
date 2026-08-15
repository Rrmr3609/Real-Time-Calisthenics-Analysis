# Enhanced repetition-classification evaluation

## Scope

This module provides pure classification metrics for the enhanced method. It
accepts ground-truth and enhanced predicted class labels only after the
corresponding repetitions have been paired by deterministic one-to-one event
matching.

Baseline warnings are frame-level diagnostics and are not repetition classes.
The baseline therefore has no classification metrics or confusion matrix.

Missed ground-truth attempts and extra predictions remain detection outcomes.
They are not inserted into the classification confusion matrix because doing
so would combine event detection and form classification into one ambiguous
measure. This separation also prevents a missed attempt from being silently
treated as a form-label error.

The module is in-memory and writes no files. The implemented cross-clip
reporting layer consumes its results separately; this module itself does not
load matches, orchestrate datasets or produce final evaluation reports.

## Supported classes and reporting order

The deterministic reporting order is:

1. `correct`
2. `insufficient_depth`
3. `incomplete_extension`
4. `alignment_deviation`
5. `unscorable`

This is a reporting order only. It is separate from, and does not change, the
classifier's rule priority.

## Confusion matrix

Rows represent ground-truth classes and columns represent enhanced predicted
classes. For example, a `correct` ground-truth repetition predicted as
`insufficient_depth` increments the cell in the `correct` row and
`insufficient_depth` column.

All supported classes remain present even when they have no observations.

## Metrics

For each class:

```text
precision = true positives / (true positives + false positives)
recall = true positives / (true positives + false negatives)
F1 = 2 * precision * recall / (precision + recall)
support = number of matched repetitions with that ground-truth class
```

Overall accuracy is the number of correct predictions divided by the number
of evaluated matched repetitions.

Macro F1 is the unweighted mean of per-class F1 values only for classes whose
ground-truth support is greater than zero. A supported class that is absent
from the evaluated ground truth still appears in the output but does not
artificially reduce macro F1. A class with ground-truth support remains in the
macro mean even when its F1 is zero.

Any zero denominator in per-class precision, recall or F1 produces `0.0`.

## Empty matched input

When no repetitions are matched:

- the evaluated count is zero;
- the complete confusion matrix contains zeros;
- every per-class count and metric is zero or `0.0`;
- accuracy is `None`;
- macro F1 is `None`.

The `None` values distinguish an undefined aggregate from a measured value of
zero.

## Public API

```python
from evaluation.classification_evaluation import (
    SUPPORTED_FORM_CLASSES,
    evaluate_classification,
)

result = evaluate_classification(
    ground_truth_labels,
    enhanced_predicted_labels,
    labels=SUPPORTED_FORM_CLASSES,
)
payload = result.to_dict()
```

The two label sequences must have equal lengths because each position denotes
one already matched repetition. Inputs with blank or unsupported labels are
rejected. The immutable result preserves label and matrix ordering, while
`to_dict()` converts it into deterministic JSON-compatible lists and records.

The [per-clip formal-evaluation integration](formal_evaluation_integration.md)
uses this API only after the existing event matcher has established the
one-to-one enhanced repetition pairs.

Cross-clip reporting pools raw confusion-matrix counts and uses the same metric
implementation to recompute the final classification results; it does not
average per-clip classification scores.
