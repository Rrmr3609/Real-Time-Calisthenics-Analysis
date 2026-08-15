# Development evidence

Everything under `results/development/` is retained historical development
evidence. It is not formal evaluation on the final test dataset.

## Directory roles

- `test_runs/` contains engineering pytest snapshots and development
  checkpoints. Filenames containing `final_tests` identify the final
  test-suite run for that development task, not evaluation on the final
  dataset or test split.
- `summaries/` contains smoke-test results, diagnostics and development
  analysis summaries.
- `figures/` contains development diagnostic visualisations.

## Provenance and limitations

Some summaries reference ignored transient inputs under `experiments/logs/`
or `experiments/outputs/`. Those inputs are not necessarily available in a
clean clone. These outputs are retained for traceability; an older result does
not necessarily describe the behaviour of the current software.

Tracked generator utilities exist for the phase and smoothing figures and for
the preprocessing, phase-detection, repetition-classification and alignment
summaries under `src/evaluation/`. The original command provenance for the
2026-07-20 video-smoke summary and the pytest snapshots is not recorded.

The 2026-07-23 repetition-classification summary contains duplicated rows from
an earlier CSV append-integrity defect. It must not be interpreted as formal
classification evidence. The alignment-visibility summaries likewise record
development investigations, not final accuracy results.

Formal evaluation evidence is retained separately under `results/formal/`.
