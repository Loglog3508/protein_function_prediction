# GPU F1/AUC Iteration Design

## Goal

Run a new GPU sequence-model iteration that targets Macro F1 >= 0.50 on the
test-like tail validation set while preserving the current continuous Macro
AUC baseline of 0.815045.

## Scope

The work adds configurable loss and validation behavior to the existing CNN
trainer, creates a tail-specific validation split without changing the fixed
seed-42 split, and evaluates GPU scores with the existing F1-first threshold
and AUC-secondary protocol. Existing experiment outputs remain immutable.

## Design

The trainer will support an asymmetric multilabel loss for the highly sparse
500-label target. Positive weights remain configurable and are derived from
training support; negative focusing and probability clipping are optional and
disabled by default so the existing BCE behavior remains reproducible. A
candidate config can also choose a deeper max/mean pooled CNN and sequence
statistics, but all runs require CUDA.

Validation IDs will be the ordered tail (`protein_id >= P112734`) and training
IDs will be the complement. The tail split is stored under a new iteration-4
directory and does not replace `artifacts/metrics/splits/seed42/`.

Candidate GPU score files will be evaluated with five threshold seeds and
support-shrunk per-label F1 thresholds. A candidate is eligible only when its
continuous Macro AUC is at least 0.815045 on the same tail rows. Among eligible
candidates, mean cross-fit Macro F1 is the primary ranking criterion; standard
deviation and predicted-positive rate are tie-break diagnostics. The baseline
candidate is always retained as a rollback reference.

## Verification

Unit tests cover the new loss and trainer configuration validation. A smoke GPU
run verifies CUDA enforcement, score shapes, and summary metrics before the
longer candidate sweep. Full results are written under fresh `EXP-20260928-*`
paths and summarized in a new experiment report.
