# ESM Label-wise Iteration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and evaluate a frozen ESM-2 representation with 500 label-wise classifiers and AUC-gated fusion, targeting strict five-seed Macro F1 above `0.5` and Macro AUC at least `0.8280390455031759`.

**Architecture:** Extract one deterministic 960-dimensional representation per protein by windowing long sequences through ESM-2 35M and concatenating length-weighted mean and max pooled embeddings. Train one lightweight binary classifier per label, align its validation scores with existing k-mer/homology/CNN scores, and select thresholds and fusion policies only through cross-fitting.

**Tech Stack:** Python 3.13, PyTorch CUDA, Hugging Face Transformers, safetensors, NumPy, pandas, SciPy, scikit-learn, pytest.

**Spec:** `docs/superpowers/specs/2026-09-29-esm-labelwise-iteration-design.md`

## Global Constraints

- Preserve `artifacts/metrics/splits/seed42/` and all historical outputs.
- Use new experiment IDs beginning at `EXP-20260929-080`.
- Keep embeddings, weights, `.npz`, `.joblib`, and submissions local.
- Use `facebook/esm2_t12_35M_UR50D`, 1,022-residue windows, overlap 128, and AMP on CUDA.
- Reject production candidates with Macro AUC below `0.8280390455031759`.
- Do not claim success unless strict five-seed Macro F1 is greater than `0.5`.

---

### Task 1: Deterministic ESM windowing and pooling

**Files:**
- Create: `requirements-esm.txt`
- Create: `src/esm_embeddings.py`
- Create: `tests/test_esm_embeddings.py`

**Interfaces:**
- Produces: `sequence_windows(sequence: str, window_size: int, overlap: int) -> list[tuple[int, str]]`.
- Produces: `pool_residue_embeddings(window_embeddings, window_lengths) -> np.ndarray` returning mean/max concatenation.
- Produces: `validate_embedding_metadata(expected: dict, actual: dict) -> None`.

- [ ] **Step 1: Write failing window-coverage and pooling tests**

```python
def test_sequence_windows_cover_long_sequence():
    sequence = "A" * 2500
    windows = sequence_windows(sequence, window_size=1022, overlap=128)
    assert windows[0][0] == 0
    assert windows[-1][0] + len(windows[-1][1]) == len(sequence)
    assert all(len(value) <= 1022 for _, value in windows)

def test_pooling_concatenates_weighted_mean_and_max():
    pooled = pool_residue_embeddings(
        [np.array([1.0, 3.0]), np.array([5.0, 1.0])], [1, 3]
    )
    np.testing.assert_allclose(pooled, [4.0, 1.5, 5.0, 3.0])
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run: `python -m pytest tests/test_esm_embeddings.py -q`

Expected: FAIL because `src.esm_embeddings` does not exist.

- [ ] **Step 3: Implement pure window, pooling, and metadata helpers**

Implement strict validation for empty sequences, window range, overlap range,
finite embeddings, hidden-dimension consistency, model name, model revision,
protein IDs, sequence hashes, pooling mode, and dtype.

- [ ] **Step 4: Run focused tests**

Run: `python -m pytest tests/test_esm_embeddings.py -q`

Expected: PASS without CUDA, network access, or model downloads.

- [ ] **Step 5: Commit**

```shell
git add requirements-esm.txt src/esm_embeddings.py tests/test_esm_embeddings.py
git commit -m "feat: add deterministic ESM embedding primitives"
```

### Task 2: Resumable GPU embedding extraction

**Files:**
- Modify: `src/esm_embeddings.py`
- Modify: `tests/test_esm_embeddings.py`
- Create: `configs/iteration6_esm2_35m_embeddings.json`

**Interfaces:**
- Produces: `extract_embedding_shards(config_path: str | Path) -> Path`.
- CLI: `python -m src.esm_embeddings --config <path>`.
- Shards contain `protein_ids`, `embeddings`, and a JSON metadata sidecar.

- [ ] **Step 1: Add failing fake-model extraction and resume tests**

```python
def test_extractor_skips_only_valid_completed_shards(tmp_path, fake_backend):
    first = extract_embedding_shards(make_config(tmp_path, fake_backend))
    mtimes = shard_mtimes(first)
    second = extract_embedding_shards(make_config(tmp_path, fake_backend))
    assert shard_mtimes(second) == mtimes

def test_extractor_rejects_stale_metadata(tmp_path, fake_backend):
    output = extract_embedding_shards(make_config(tmp_path, fake_backend))
    corrupt_model_name(output)
    with pytest.raises(ValueError, match="model_name"):
        extract_embedding_shards(make_config(tmp_path, fake_backend))
```

- [ ] **Step 2: Verify the new tests fail**

Run: `python -m pytest tests/test_esm_embeddings.py -q`

Expected: FAIL because extraction is not implemented.

- [ ] **Step 3: Implement dependency-isolated model loading and extraction**

Import Transformers only inside the real backend. Load `AutoTokenizer` and
`AutoModel`, exclude special and padding tokens from pooling, use `torch.inference_mode`,
CUDA autocast, configurable batch size, atomic shard writes, and validated resume.
Record the resolved model commit hash and library versions.

- [ ] **Step 4: Run unit tests and a ten-sequence GPU smoke extraction**

Run: `python -m pytest tests/test_esm_embeddings.py -q`

Run: `python -m src.esm_embeddings --config configs/iteration6_esm2_35m_embeddings_smoke.json`

Expected: tests pass; smoke output has ten unique IDs and shape `(10, 960)`.

- [ ] **Step 5: Commit**

```shell
git add src/esm_embeddings.py tests/test_esm_embeddings.py configs/iteration6_esm2_35m_embeddings.json
git commit -m "feat: add resumable ESM GPU extraction"
```

### Task 3: Label-wise classifier screening and strict validation scoring

**Files:**
- Create: `src/train_esm_labelwise.py`
- Create: `tests/test_train_esm_labelwise.py`
- Create: `configs/iteration6_esm2_labelwise_screen.json`
- Create: `configs/iteration6_esm2_labelwise_full.json`

**Interfaces:**
- Produces: `fit_labelwise_scores(train_x, train_y, eval_x, config) -> np.ndarray`.
- Produces: `screen_regularization_by_support(...) -> pandas.DataFrame`.
- CLI writes validation/test scores with IDs and label names plus a concise summary.

- [ ] **Step 1: Write failing tests for independent labels and constants**

```python
def test_labelwise_scores_fit_each_column_independently():
    scores = fit_labelwise_scores(train_x, train_y, eval_x, config)
    assert scores.shape == (len(eval_x), train_y.shape[1])
    assert not np.allclose(scores[:, 0], scores[:, 1])

def test_single_class_label_uses_finite_constant_score():
    scores = fit_labelwise_scores(train_x, np.zeros((4, 1)), eval_x, config)
    assert np.isfinite(scores).all()
    assert np.unique(scores).size == 1
```

- [ ] **Step 2: Verify focused tests fail**

Run: `python -m pytest tests/test_train_esm_labelwise.py -q`

Expected: FAIL because the trainer does not exist.

- [ ] **Step 3: Implement support-stratified SGD/logistic screening**

Use support-stratified label samples rather than frequency-ordered first labels.
Screen regularization only on an inner training split. Fit all 500 label models
with the selected support-stratum settings, save continuous scores, and never
select thresholds in this module.

- [ ] **Step 4: Run focused and full tests**

Run: `python -m pytest tests/test_train_esm_labelwise.py tests/test_metrics.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```shell
git add src/train_esm_labelwise.py tests/test_train_esm_labelwise.py configs/iteration6_esm2_labelwise_screen.json configs/iteration6_esm2_labelwise_full.json
git commit -m "feat: train label-wise classifiers on ESM features"
```

### Task 4: AUC-gated label-adaptive fusion

**Files:**
- Create: `src/sweep_esm_fusion.py`
- Create: `tests/test_sweep_esm_fusion.py`
- Create: `configs/iteration6_esm_fusion.json`

**Interfaces:**
- Produces: `align_score_sources(...) -> dict[str, np.ndarray]`.
- Produces: `crossfit_label_policies(target, sources, seeds, config) -> tuple[pd.DataFrame, pd.DataFrame]`.
- CLI writes seed results, leaderboard, thresholds, label policies, and summary.

- [ ] **Step 1: Write failing alignment, leakage, rollback, and AUC tests**

```python
def test_mismatched_ids_or_labels_are_rejected():
    with pytest.raises(ValueError, match="IDs or labels"):
        align_score_sources(reference, mismatched)

def test_production_candidates_include_rollback_and_enforce_auc_gate():
    ranked = rank_policy_results(results, minimum_auc=0.8280390455031759)
    assert "rollback" in set(ranked.candidate)
    assert not ranked.loc[ranked.continuous_macro_auc < 0.8280390455031759, "eligible"].any()
```

- [ ] **Step 2: Verify focused tests fail**

Run: `python -m pytest tests/test_sweep_esm_fusion.py -q`

Expected: FAIL because fusion is not implemented.

- [ ] **Step 3: Implement cross-fitted support-stratum and label policies**

Align all matrices by names, enumerate individual and pairwise blends, fit a
policy on one threshold fold, evaluate it on the other, and swap. Use shared
support-stratum policies by default; enable independent label policies only
above configured support and stability thresholds. Rank eligible candidates by
mean F1, then minimum F1, standard deviation, and AUC.

- [ ] **Step 4: Run focused and full tests**

Run: `python -m pytest tests/test_sweep_esm_fusion.py -q`

Run: `python -m pytest -q -p no:cacheprovider`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```shell
git add src/sweep_esm_fusion.py tests/test_sweep_esm_fusion.py configs/iteration6_esm_fusion.json
git commit -m "feat: add AUC-gated ESM label fusion"
```

### Task 5: Run full experiments and record verified outcomes

**Files:**
- Create: `artifacts/metrics/EXP-20260929-080-*-summary.json`
- Create: `artifacts/metrics/EXP-20260929-081-*-summary.json`
- Create: `artifacts/metrics/EXP-20260929-082-*-summary.json`
- Create: `reports/experiments/EXP-20260929-iteration6-esm.md`
- Modify: `docs/iteration2_handoff.md`

**Interfaces:**
- Consumes validated local embedding shards and historical aligned score files.
- Produces strict five-seed F1, F1 standard deviation/minimum, continuous AUC,
  predicted-positive rate, per-support diagnostics, and a go/no-go decision.

- [ ] **Step 1: Install optional dependencies in the active environment**

Run: `python -m pip install -r requirements-esm.txt`

Expected: Transformers and safetensors import successfully without changing
machine-specific paths in the repository.

- [ ] **Step 2: Extract all train/test embeddings with validated resume**

Run: `python -m src.esm_embeddings --config configs/iteration6_esm2_35m_embeddings.json`

Expected: every train/test ID occurs exactly once; feature width is 960; shards
record the resolved ESM model revision and no stale shard is accepted.

- [ ] **Step 3: Screen and fit all label-wise classifiers**

Run: `python -m src.train_esm_labelwise --config configs/iteration6_esm2_labelwise_screen.json`

Run: `python -m src.train_esm_labelwise --config configs/iteration6_esm2_labelwise_full.json`

Expected: aligned `(1062, 500)` validation scores and `(28450, 500)` test scores,
or the actual test row count read from `data/test.csv`; no hard-coded row count.

- [ ] **Step 4: Run strict fusion and threshold evaluation**

Run: `python -m src.sweep_esm_fusion --config configs/iteration6_esm_fusion.json`

Expected: rollback is present, AUC-ineligible candidates are retained only as
diagnostics, and production selection uses no outer evaluation labels.

- [ ] **Step 5: Record outcome without overstating it**

Write the experiment report with exact commands, model revision, runtime, GPU,
all gates, and comparisons against iterations 4 and 5. If F1 is not above 0.5,
record the best verified result and whether conditional LoRA is justified.

- [ ] **Step 6: Verify and commit concise artifacts**

Run: `python -m pytest -q -p no:cacheprovider`

Run: `git diff --check`

Run: `git lfs status`

Commit configs, source, tests, concise metrics, and report. Do not commit model
weights, embedding shards, `.npz`, `.joblib`, submission CSVs, or the local ZIP.
