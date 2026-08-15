# Documentation

This index separates operational guidance, method design and evaluation
protocols. The repository [README](../README.md) provides the short project
overview and quick start.

## Getting started and runtime

- [Recorded-run configuration and provenance](runtime_configuration.md) —
  defines validated settings, output identity, repetition measurement windows,
  provenance and reproduction commands.

## Method and scientific design

- [Baseline-versus-enhanced comparison design](baseline_evaluation_design.md) —
  defines the intentionally simple baseline, fair shared metrics and the limits
  of direct classification comparison.
- [Development scientific freeze](development_scientific_freeze.md) — records
  the final development-only threshold rationale, primary event tolerance,
  annotation hash and no-test-retuning rule.

## Annotation and per-clip evaluation

- [Manual annotation protocol](manual_annotation_protocol.md) — defines the
  dataset manifest, repetition annotations, ambiguity handling, source-only
  annotation viewer and review/freeze procedure.
- [Held-out test collection protocol](held_out_test_collection_protocol.md) —
  records the pre-evaluation collection rules, documented four-clip deviation,
  frozen configuration and held-out safeguards.
- [Repetition-event detection evaluation](event_detection_evaluation.md) —
  specifies event extraction, validation, deterministic matching and detection
  metrics.
- [Enhanced repetition-classification evaluation](classification_evaluation.md)
  — defines supported classes, confusion-matrix orientation and classification
  metrics.
- [Enhanced per-clip formal-evaluation integration](formal_evaluation_integration.md)
  — explains how matched enhanced events feed detection and classification
  results without rematching.

## Cross-clip formal evaluation and reproducibility

- [Cross-clip formal-evaluation reporting](formal_evaluation_reporting.md) —
  defines pooled reporting, timing aggregation, output files and evaluation-run
  provenance.
- [Formal evaluation execution](formal_evaluation_execution.md) — gives the
  development-first execution order, safeguards and generated report set.

## Evidence indexes

- [Development evidence index](../results/development/README.md) — identifies
  retained historical diagnostics and their interpretation limits.
- [Formal result index](../results/formal/README.md) — distinguishes primary,
  sensitivity and historical formal result families.
- [Dissertation evidence index](dissertation_evidence_index.md) — maps frozen
  scope, data, metrics, runtime evidence and limitations to exact tracked
  sources for use while writing the dissertation.
