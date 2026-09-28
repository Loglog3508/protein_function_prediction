# GPU F1/AUC Iteration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a GPU-trained, tail-validated candidate path that can improve Macro F1 without accepting a continuous Macro AUC regression.

**Architecture:** Extend `src/train_gpu.py` with a small, tested loss/config layer while preserving the existing CNN entry point. Generate an iteration-4 tail split, train fresh GPU candidates, and score them with the repository's existing threshold/F1/AUC utilities; select only candidates meeting the AUC gate.

**Tech Stack:** Python, PyTorch CUDA, NumPy, pandas, scikit-learn metrics, pytest.

**Spec:** `docs/superpowers/specs/2026-09-28-gpu-f1-auc-iteration-design.md`

## Global Constraints

- Preserve `artifacts/metrics/splits/seed42/` exactly.
- Use a new experiment ID and output prefix for every run.
- Require CUDA for GPU candidates; never silently fall back to CPU.
- Keep generated runs, submissions, `.npz`, and `.joblib` local.
- Reject any candidate with continuous Macro AUC below `0.815045` on the same tail validation rows.

---

### Task 1: Add Tested GPU Loss Helpers

**Files:**
- Modify: `src/train_gpu.py`
- Test: `tests/test_train_gpu.py`

**Interfaces:**
- Produces `asymmetric_bce_loss(logits, targets, positive_weight, gamma_positive, gamma_negative, probability_clip)`.
- Produces `validate_gpu_training_config(config)` for loss and CUDA settings.

- [ ] Write tests for zero-focused loss behavior and invalid loss configuration.
- [ ] Run the focused tests and observe the expected missing-symbol failures.
- [ ] Implement the minimal helpers and call config validation from `run_gpu_evaluation`.
- [ ] Run focused tests, then the complete test suite.

### Task 2: Add Iteration-4 Tail Split and GPU Configurations

**Files:**
- Create: `configs/iteration4_gpu_tail_asl_screen.json`
- Create: `configs/iteration4_gpu_tail_candidate_*.json`
- Create: `artifacts/metrics/splits/iteration4_tail/train_ids.csv` (local generated artifact)
- Create: `artifacts/metrics/splits/iteration4_tail/validation_ids.csv` (local generated artifact)

**Interfaces:**
- Consumes the existing `data/train.csv` and `src.shift.tail_distribution_mask` convention.
- Produces fresh run/metric prefixes beginning with `EXP-20260928-`.

- [ ] Generate and verify the complement/tail IDs without modifying seed-42 IDs.
- [ ] Add a small screen of ASL/positive-weight/CNN settings with `require_cuda=true`.
- [ ] Run one short CUDA smoke candidate and verify output shapes and GPU metadata.

### Task 3: Evaluate GPU Scores Under the AUC Gate

**Files:**
- Create: `src/sweep_gpu_f1_auc.py`
- Test: `tests/test_sweep_gpu_f1_auc.py`
- Create: `configs/iteration4_gpu_f1_auc.json`

**Interfaces:**
- Consumes GPU `scores.npz` files with `validation_scores`, `validation_ids`, and `label_columns`.
- Produces a leaderboard, seed results, selected thresholds, and a JSON summary.

- [ ] Write tests for AUC-gate filtering and F1-first ranking.
- [ ] Run focused tests and verify they fail before implementation.
- [ ] Implement candidate loading, five-seed threshold evaluation, AUC filtering, and rollback baseline handling.
- [ ] Run focused and full tests.

### Task 4: Run GPU Screening and Finalize the Best Eligible Model

**Files:**
- Create: `reports/experiments/EXP-20260928-iteration4-gpu.md`
- Create: `artifacts/metrics/EXP-20260928-*` (local metrics)
- Create: `artifacts/runs/EXP-20260928-*` (local runs)

- [ ] Run the short CUDA screen and inspect loss, AUC, F1, and prediction density.
- [ ] Run the full five-seed threshold evaluation for candidates that pass AUC.
- [ ] Retrain the selected configuration on all training rows only after the gate passes.
- [ ] Run the full test suite and verify the final report contains exact metrics and paths.
