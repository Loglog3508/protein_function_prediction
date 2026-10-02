# 第十二次迭代：理化性质分组 k-mer

日期：2026-09-30。本轮实验 118 已完成；未生成或上传平台提交。

## 目标与口径

- 目标：完整 500 标签 Macro F1 达到 0.5；当前生产回退仍为实验 103 的 0.338248。
- 规则：仅使用竞赛训练 CSV；允许使用预训练权重，但本轮未新增外部数据。
- 验证：固定 `iteration4_tail` 的 1,062 条验证 ID；训练数据 112,734 条；支持度分层抽取 100 个标签；五个 threshold seeds `17、31、42、73、101`。
- 模型：balanced SGD，`alpha=5e-6`，尾段样本权重 2，阈值收缩 50。

## 方法

在原始 120k 维 3–5-mer TF-IDF 后追加一个理化分组序列 block。20 种氨基酸被划分为 6 个互斥组：

`AGPST / C / DENQ / FWY / HKR / ILMV`

分组 block 使用 3–6-mer、最多 60k 特征、缩放 0.25；原始 k-mer block 保持不变。未知残基保持位置但不产生训练词表特征。纯原始 k-mer 使用已完成的配对分数作为回退。

## 结果

| 候选 | 五 seed Macro F1 | 标准差 | 最小 F1 | 连续 Macro AUC | 预测正例率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw-baseline | 0.316422 | 0.001335 | 0.313781 | 0.803188 | 0.110979 |
| **raw + grouped 0.25** | **0.317236** | 0.001299 | **0.315198** | **0.804960** | 0.113531 |

相对配对基线，Macro F1 增加 `0.000815`，连续 AUC 增加 `0.001771`。五个 threshold seeds 并非全部改善，且预测正例率进一步升高，未解决系统性过预测问题。

## 决策

1. 不扩展该表示到完整 500 标签；增益不足以覆盖全量训练成本和筛选选择偏差。
2. 保留分组 k-mer 代码作为后续混合表示组件，不将其作为新的生产回退。
3. 下一步转向 OOF 标签级先验校准，重点验证低频标签的预测正例率、Precision、Average Precision 和 F1 是否同步改善。

## 产物与复现

- 配置：`configs/iteration12_grouped_kmer_screen.json`
- 指标：`artifacts/metrics/EXP-20260930-118-grouped-kmer-screen-summary.json`
- 报表：`artifacts/metrics/EXP-20260930-118-grouped-kmer-screen-leaderboard.csv`
- 本地分数保存在 `artifacts/runs/EXP-20260930-118-grouped-kmer-screen/`，不提交 Git。

```powershell
python -m src.sweep_labelwise_sgd --config configs/iteration12_grouped_kmer_screen.json
```
