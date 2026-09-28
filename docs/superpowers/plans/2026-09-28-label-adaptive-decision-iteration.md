# 按标签自适应模型与决策实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在第五次全局 `90% 对齐 + 10% SGD` 的基础上，验证每个标签分别选择模型来源、融合权重和决策规则是否能稳定提升排行榜代理 Macro F1。

**Architecture:** 先固定第五次实验的尾段样本、标签顺序和五个 threshold split seeds，再生成对齐近邻、k-mer KNN、SGD 及可用 GPU 分数的同一行同一标签 OOF 矩阵。标签级策略只在一折选择、另一折评估，最后以五 seed Macro F1 为第一排序键，AUC、稳定性和预测密度作为约束与决胜信息；全局纯对齐和第五次全局融合始终保留为回退分支。

**Tech Stack:** Python、NumPy、pandas、scikit-learn、SciPy 稀疏 TF-IDF、现有 `src.metrics`/`src.thresholds` 评估工具；GPU 仅使用仓库已有且环境可用的分支。

**Spec:** `reports/experiments/EXP-20260928-iteration5-homology.md` 中的第五次结果、规则约束和下一次迭代目标。

## Global Constraints

- 竞赛主目标是所有有效标签的 Macro F1；AUC 只作辅助，F1 优先。
- 保持尾段验证口径：`protein_id >= P112734` 的 1,062 条样本，以及 threshold split seeds `17、31、42、73、101`。
- 所有标签级模型、权重、阈值、Top-k 和相似度门槛选择都必须由 OOF 结果驱动，不能使用测试标签或线上成绩反向调参。
- 保留 `artifacts/metrics/splits/seed42/`，不得覆盖历史划分；每次实验使用新实验编号和输出前缀。
- 只使用竞赛提供的训练和测试 CSV；未获书面许可前不使用 ESM、ProtBERT 或其他外部预训练权重。
- `artifacts/runs/`、提交 CSV、`.npz` 和 `.joblib` 只保留本地；提交前检查 Git LFS 状态。
- 不修改项目目录和 `D:\cuda` 之外的文件，不把机器解释器路径或缓存路径写入仓库。

### Task 1: 固定输入和 OOF 评分矩阵

**Files:**
- Create: `configs/iteration6_label_adaptive_oof.json`
- Create: `src/label_adaptive.py`
- Test: `tests/test_label_adaptive.py`

**Interfaces:**
- Produces `build_candidate_score_matrix(...) -> dict[str, np.ndarray]`，返回统一 `validation_ids`、`label_columns` 和每个候选来源的二维分数矩阵。
- 候选键固定为 `alignment`、`knn`、`sgd`、`alignment_sgd`、`alignment_knn`，GPU 分支只有在已有 OOF 文件和兼容环境均满足时才加入。

- [ ] **Step 1: 写输入一致性测试**

```python
def test_candidate_matrix_rejects_mismatched_labels():
    with pytest.raises(ValueError, match="label columns"):
        align_candidate_columns(
            np.zeros((2, 2), dtype=np.float32),
            ["GO:1", "GO:2"],
            ["GO:2", "GO:3"],
        )
```

- [ ] **Step 2: 运行测试确认初始失败**

Run: `python -m pytest tests/test_label_adaptive.py::test_candidate_matrix_rejects_mismatched_labels -q`

Expected: FAIL because the new label-adaptive alignment helper is not implemented yet.

- [ ] **Step 3: 固定数据来源并实现矩阵装配**

使用第五次已有 OOF 产物作为可复用输入，按标签名对齐列，不按列位置盲拼；任何 ID、行数或标签集合不一致都立即失败。输出写入新的 `artifacts/runs/EXP-20260928-071-iteration6-label-adaptive-oof/`，不得覆盖 `EXP-20260928-066` 至 `070`。

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_label_adaptive.py -q`

Expected: PASS，且矩阵的每个来源均为 `(1062, 500)`，ID 和标签顺序完全一致。

### Task 2: 生成按标签候选策略表

**Files:**
- Modify: `src/label_adaptive.py`
- Create: `configs/iteration6_label_adaptive_policy.json`
- Test: `tests/test_label_adaptive.py`

**Interfaces:**
- Produces `enumerate_label_candidates(score_matrix, target, config) -> pandas.DataFrame`。
- 每行至少包含 `label`、`support`、`source`、`weight`、`threshold_rule`、`threshold`、`oof_macro_f1`、`oof_auc`、`predicted_positive_rate`。

- [ ] **Step 1: 为支持度分层和候选策略写测试**

```python
def test_candidate_table_contains_support_strata_and_required_columns():
    table = enumerate_label_candidates(score_matrix, target, config)
    assert {"label", "support", "source", "weight", "threshold_rule"} <= set(table)
    assert table["support"].min() >= 0
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_label_adaptive.py::test_candidate_table_contains_support_strata_and_required_columns -q`

Expected: FAIL until candidate enumeration is implemented.

- [ ] **Step 3: 实现候选枚举**

按标签支持度分层生成共享策略，并在支持度足够、跨 seed 稳定时允许独立策略。模型来源比较纯对齐、纯 KNN、纯 SGD、对齐+SGD 和对齐+KNN；决策比较支持度收缩阈值、统一阈值、Top-k 和相似度不足时拒绝正例。

- [ ] **Step 4: 运行测试并保存候选表**

Run: `python -m pytest tests/test_label_adaptive.py -q`

Expected: PASS；候选表保存为 `artifacts/metrics/EXP-20260928-071-iteration6-label-adaptive-candidates.csv`。

### Task 3: 用双折 OOF 选择标签策略并独立评估

**Files:**
- Modify: `src/label_adaptive.py`
- Create: `src/sweep_label_adaptive.py`
- Test: `tests/test_label_adaptive.py`

**Interfaces:**
- Produces `select_label_policies(score_matrix, target, seeds, config) -> tuple[pandas.DataFrame, pandas.DataFrame]`，分别返回标签策略表和候选汇总表。

- [ ] **Step 1: 写防泄漏测试**

```python
def test_policy_selection_does_not_score_on_selection_fold():
    policy_table, summary = select_label_policies(score_matrix, target, seeds=[17], config=config)
    assert "selection_fold" in policy_table
    assert "evaluation_fold_macro_f1" in policy_table
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_label_adaptive.py::test_policy_selection_does_not_score_on_selection_fold -q`

Expected: FAIL until the two-fold policy selector exists.

- [ ] **Step 3: 实现双折选择和五 seed 汇总**

将尾段 OOF 行再分为两折：用折 A 选择每个标签的来源、权重和决策规则，在折 B 评估；交换 A/B 后合并。先按五 seed Macro F1 排序；仅当候选与最高 F1 的差值不超过 `0.002` 时，才比较 Macro AUC、F1 标准差、最小 seed F1 和预测正例率。AUC 低于 `0.815045` 的候选标记为 F1 诊断候选，不得直接替换生产回退方案。

- [ ] **Step 4: 运行完整 OOF 扫描**

Run: `python -m src.sweep_label_adaptive --config configs/iteration6_label_adaptive_policy.json`

Expected: 生成标签级策略表、候选 leaderboard、五 seed 结果和独立折差异；结果不覆盖历史 `EXP-20260928-066` 至 `070`。

### Task 4: 全量重训、提交校验和回退比较

**Files:**
- Create: `configs/iteration6_final_label_adaptive.json`
- Create: `src/finalize_label_adaptive.py`
- Test: `tests/test_label_adaptive_submission.py`

**Interfaces:**
- Produces `submit_template_v1.csv` and metadata under `artifacts/submissions/EXP-20260928-071-iteration6-label-adaptive/`。
- Finalizer must accept a policy table and emit the exact training-label order from `data/train.csv`。

- [ ] **Step 1: 写提交结构测试**

```python
def test_final_submission_is_binary_and_matches_test_ids(path, test_path):
    result = validate_submission_file(path, test_path=test_path)
    assert result["rows"] == 28450
    assert result["binary_labels"] is True
```

- [ ] **Step 2: 实现全量重训和策略应用**

在全部训练数据上重训需要的候选模型，仅使用已通过独立折评估的策略表生成测试预测；生成前先检查策略表标签集合、测试 ID 顺序、阈值范围和预测矩阵形状。

- [ ] **Step 3: 对比回退方案并校验提交**

Run: `python -m src.finalize_label_adaptive --config configs/iteration6_final_label_adaptive.json`

Expected: 提交 CSV 为 `28,450 × 501`，无空值，标签仅为 `0/1`，ID 顺序与 `data/test.csv` 完全一致；同时保留第五次全局 `90% 对齐 + 10% SGD` 作为可重新生成的回退提交，不因 F1 诊断候选自动替换。

### Task 5: 记录第六次实验和趋势数据

**Files:**
- Create: `reports/experiments/EXP-20260928-iteration6-label-adaptive.md`
- Modify: `reports/蛋白质功能预测迭代趋势.xlsx`

- [ ] **Step 1: 记录可比指标**

写入五 seed Macro F1 均值、标准差、最小值、连续 Macro AUC、预测正例率、每样本标签数、标签级策略数量以及与第五次回退方案的差值；明确区分本地代理成绩和平台返回成绩。

- [ ] **Step 2: 更新趋势表和折线图**

新增第六次一行，并同时绘制 Macro F1、Macro AUC、预测正例率；不修改历史迭代数据。

- [ ] **Step 3: 运行文档与工作区校验**

Run: `git diff --check`

Expected: 无空白错误；报告不把未提交的第六次候选写成最终平台成绩。

## 选择与验收标准

1. 默认最终候选必须在相同尾段口径下将五 seed Macro F1 稳定提高到第五次 `0.335178` 以上。
2. AUC 目标下限为第三次基线 `0.815045`；若只提升 F1 而 AUC 下降，必须单独标记为 F1 诊断候选，不替换回退提交。
3. 标签级独立策略若相对支持度分层共享策略没有跨 seed 的稳定增益，则退回共享策略，避免 500 个标签过拟合。
4. 只有完整提交校验通过且候选选择记录可追溯时，才生成用于平台评分的新 `submit_template_v1.csv`。
5. 第六次结束后才更新综合 PDF；本轮计划和实验报告不提前声称平台成绩。
