# ESM Label-wise Iteration Design

## Goal

Raise the competition-style label-wise Macro F1 above `0.5` while requiring
continuous Macro AUC to remain at or above the iteration-4 reference
`0.8280390455031759`. Results below either gate remain diagnostic and do not
replace the current rollback submission.

## Baselines and Constraints

- The current best strict local F1 is iteration 5: `0.3351783934801672`, with
  AUC `0.8211324963363611`.
- The AUC rollback reference is iteration 4: `0.8280390455031759`.
- Preserve all existing seed-42 split IDs and use new experiment IDs beginning
  at `EXP-20260929-080`.
- Keep embeddings, model weights, `.npz`, `.joblib`, and submissions local.
- The target machine has an 8 GB RTX 4060 Laptop GPU. Frozen feature extraction
  is the first stage; LoRA or partial fine-tuning is conditional on a successful
  frozen-feature result.
- The selected pretrained model is `facebook/esm2_t12_35M_UR50D`, which has a
  480-dimensional hidden state and a 1,026-token positional limit.

## Architecture

### 1. Shared ESM representation

Load ESM-2 once on the GPU and encode every unique protein sequence. Residue
tokens are split into overlapping windows of at most 1,024 residues. For each
window, mean-pool non-special-token representations. Aggregate multiple windows
with a length-weighted mean and element-wise maximum, then concatenate both
vectors into a 960-dimensional protein representation.

The extractor is deterministic, resumable, uses automatic mixed precision, and
writes row-aligned shard files under `artifacts/runs/`. Each shard records model
name, sequence hash, protein IDs, window settings, dtype, and feature shape.
The extractor rejects stale shards whose metadata does not match the request.

### 2. Label-wise classifiers

Train one lightweight binary classifier per label from the shared embeddings.
The first candidates are logistic regression models with support-stratified
regularization and class weighting. A constant score is used only when a
training fold contains one class. The same fold assignments and row order are
used for every label, so all OOF score matrices remain directly comparable.

The label-wise classifiers produce continuous scores for validation and test
rows. Thresholds are not selected during classifier fitting.

### 3. Leakage-resistant evaluation

Use the fixed tail evaluation population of 1,062 rows for comparison with
iterations 4 and 5. Build ESM predictions with outer OOF training. Inside each
outer-training partition, choose label-group regularization from support-
stratified labels; never use an outer evaluation label to select that fold's
classifier or epoch.

For final F1 evaluation, reuse the existing five threshold seeds
`17, 31, 42, 73, 101`. Threshold and label-policy selection occur on one half
and evaluation on the other, then the roles are exchanged. Report mean,
standard deviation, and minimum Macro F1.

### 4. Label-adaptive fusion

Align ESM, k-mer SGD, homology, and optional CNN OOF score matrices by protein ID
and label name. Candidate policies include individual sources and pairwise
blends. Low-support labels use shared support-stratum policies. A label may use
an independent policy only when its cross-seed improvement is stable.

Select policies by Macro F1 subject to the hard continuous Macro AUC gate. The
iteration-4 AUC candidate and iteration-5 homology candidate remain explicit
rollback branches. A production sweep must include a zero-weight new-model
candidate.

### 5. Conditional fine-tuning

Only if frozen ESM features improve strict OOF F1 over iteration 5 without
failing the AUC gate, evaluate LoRA or unfreezing the final transformer blocks.
Use gradient accumulation, mixed precision, gradient checkpointing, and
sequence windows compatible with 8 GB VRAM. Epoch selection must use an inner
split; outer OOF labels remain evaluation-only.

## Experiment Gates

1. Frozen ESM is useful only if strict five-seed F1 exceeds `0.3351783935`.
2. A production candidate must have AUC at least `0.8280390455`.
3. The requested success condition is strict five-seed Macro F1 above `0.5`.
4. A gain restricted to one threshold seed, an in-sample threshold score, or an
   in-sample label-policy score is not accepted.
5. If no candidate reaches `0.5`, report the best verified result and the next
   technically justified stage; do not claim the target was met.

## Implementation Boundaries

- `src/esm_embeddings.py`: model loading, windowing, pooling, shard validation,
  and feature extraction.
- `src/train_esm_labelwise.py`: label-wise OOF and full-fit classifiers.
- `src/sweep_esm_fusion.py`: aligned score loading, support-stratified policies,
  threshold cross-fitting, and AUC-gated ranking.
- New JSON configs under `configs/` use experiment IDs `080` and above.
- Unit tests use fake tokenizers/models and small arrays. They must not download
  weights or require CUDA.
- A separate requirements file records compatible optional ESM dependencies;
  the base CPU workflow remains unchanged.

## Failure Handling

- CUDA absence, out-of-memory errors, invalid sequence characters, model/token
  limit mismatches, incomplete shards, row-order mismatches, duplicate IDs, and
  label-set mismatches fail with actionable messages.
- Existing output paths are checked before work begins. Resume is allowed only
  through validated shard metadata, never by silently overwriting results.
- Network or model-download failure leaves source/config changes intact and is
  reported separately from model quality.

## Verification

- Unit tests cover window coverage, padding exclusion, deterministic pooling,
  shard resume checks, single-class labels, label alignment, no-leak policy
  selection, AUC gating, and output collision prevention.
- Run the complete test suite and `git diff --check` before recording results.
- Record GPU, dependency versions, model revision, hashes, timings, F1, AUC,
  predicted-positive rate, and per-support-stratum diagnostics in the report.
