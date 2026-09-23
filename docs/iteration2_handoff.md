# 第二轮迭代交接

## 当前结论

第一轮已完成并通过完整 500 标签验证。当前主模型：

- 特征：字符级 3-5-mer TF-IDF，`min_df=5`，`max_features=120000`，`sublinear_tf=true`
- 模型：`SGDClassifier(loss="log_loss", class_weight="balanced")`
- 正则：`alpha=5e-6`，不使用参数平均
- 阈值：逐标签最优阈值向全局阈值收缩，`shrinkage=25`
- 固定阈值 0.5 Macro F1：0.362882
- 双向交叉拟合 Macro F1：0.379956
- 五个阈值二分 seed 均值：0.380168，标准差 0.000541

旧主模型的交叉拟合 Macro F1 为 0.317516。第一轮绝对提升 0.062440，相对提升约 19.7%。详细过程见 `reports/experiments/EXP-20260923-iteration1.md`。

## 已验证和淘汰

- `alpha` 从 `5e-5` 降至 `7e-6` 持续改善；30k 词表下 `5e-6` 与 `3e-6` 略回落，`1e-6` 明显回落。
- 参数平均 `average=True` 在四个配对正则点上全部弱于普通 SGD，第二轮无需重试。
- 词表从 30k 到 60k、120k 持续提升，尚未证明 120k 是容量上限。
- 旧模型、CNN、标签共现传播、相近 SGD 融合和 protein ID 邻域传播均未带来收益。
- 阈值步长从 0.05 细化到 0.01 没有稳定实质收益；保持 0.05 即可。

## 二轮优先顺序

1. 在支持度分层 100 标签上测试 `max_features=200000` 和不设上限，先使用 `alpha=5e-6`；记录内存、非零元素数和 Macro F1。
2. 将 3-mer 与 4-5-mer 拆成两个 TF-IDF block，分别限制词表后拼接，避免短 k-mer 挤占长 motif；与当前 120k 单词表严格同口径比较。
3. 仅在新表示胜出后跑完整 500 标签。不要为每个筛选候选重复全量训练。
4. 若开展 GPU 路线，根据协作者机器的操作系统、驱动和 CUDA 版本安装兼容的 PyTorch，先验证 `torch.cuda.is_available()`。不要依赖原实验机器的解释器或缓存路径；使用外部权重前须确认竞赛规则。
5. 新模型产生验证分数后，再考虑按标签选择模型或 rank/概率校准融合；不要直接重复已经失败的全局简单加权。

## 复现命令

首次克隆先运行 `git lfs install` 和 `git lfs pull`，确保竞赛 CSV 不是 LFS 指针文件。随后激活协作者自己的 Python 环境并执行：

```shell
python -m pytest -q -p no:cacheprovider
python -m src.train --config configs/iteration1_kmer35_sgd_full.json --evaluate
python -m src.thresholds --scores artifacts/runs/EXP-20260923-025-iteration1-kmer35-sgd-full/scores.npz --train data/train.csv --output-prefix artifacts/metrics/EXP-20260923-025-iteration1-kmer35-sgd-thresholds --seed 42 --holdout-fraction 0.5 --shrinkage 25
```

筛选器使用方式：

```shell
python -m src.sweep_sgd --config configs/iteration1_vocab120k_screen.json
```

`src.sweep_sgd` 会复用一次构建的 TF-IDF 矩阵、保存每个候选的连续分数，并支持从已完成候选继续运行。

## 原始运行机器上的可复用产物

以下文件按 `.gitignore` 不进入 Git。它们仅在生成第一轮结果的原始工作区中存在；使用该工作区时可直接复用：

- 验证与测试分数：`artifacts/runs/EXP-20260923-025-iteration1-kmer35-sgd-full/scores.npz`
- 正式全量测试分数：`artifacts/metrics/EXP-20260923-026-iteration1-final-test-scores.npz`
- 正式向量器：`artifacts/metrics/EXP-20260923-026-iteration1-final-vectorizer.joblib`
- 首选提交：`artifacts/submissions/submission_EXP-20260923-026.csv`

其他协作者在新机器克隆仓库后不会得到这些派生产物，需要按上面的相对路径命令重新生成。所有已提交配置、数据划分、阈值和轻量指标均不依赖原机器目录结构。

## 实验编号与基线

- 下一实验编号从 `EXP-20260923-027` 开始。
- 二轮所有筛选必须包含当前 120k、`alpha=5e-6` 基线，或复用第一轮相同的支持度分层标签列表进行严格比较。
- 机器可读排行榜：`artifacts/metrics/leaderboard.csv`
- 提交顺序：`docs/submission_plan.md`
