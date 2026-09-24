# 第二次提分迭代：GPU CNN 池化、类别权重与统计特征

- 日期：2026-09-24
- 固定划分：90,920 条训练、22,876 条验证，seed 42
- 主评价：500 标签双向交叉拟合收缩阈值 Macro F1
- GPU：NVIDIA GeForce RTX 4060 Laptop GPU
- 环境：PyTorch 2.9.1+cu126，AMP 开启
- 规则约束：只使用竞赛 CSV，不使用外部数据或预训练权重

## 目标与改动

本轮复查 `docs/iteration2_handoff.md` 后，沿用固定 seed-42 划分和
`shrinkage=25` 阈值流程，评估从零训练的 GPU 模型是否能超过第一轮
120k 3-5-mer TF-IDF + SGD 主模型（Macro F1 0.379956）。

GPU 模型新增了三项可配置能力：

- 对有效序列位置执行 max/mean 双池化，避免 padding 参与新池化模式。
- 提高稀有标签正类损失权重上限。
- 将竞赛数据内的氨基酸组成、二肽、性质和长度统计投影后并入 CNN 分类头。

## 实验结果

| 实验 | 关键变化 | 轮数 | 最佳固定阈值 F1 | 交叉拟合收缩阈值 F1 | 训练与验证秒数 |
| --- | --- | ---: | ---: | ---: | ---: |
| EXP-20260924-027 | max/mean 双池化 | 6 | 0.142696 | 0.203078 | 221.7 |
| EXP-20260924-028 | 正类权重上限 20，dropout 0.1 | 8 | 0.179759 | 0.215152 | 279.2 |
| EXP-20260924-029 | 增加 474 维序列统计分支 | 8 | 0.205274 | 0.231648 | 282.1 |
| EXP-20260924-030 | 统计隐藏层 128，训练 16 轮 | 16 | **0.232949** | **0.246370** | 463.0 |

相对旧 CUDA CNN 的 0.184126，最佳 GPU 候选绝对提升 0.062244，约
33.8%。统计分支和延长训练均产生稳定增益，但最佳结果仍比当前主模型
低 0.133587，因此不进入正式提交，也不与主模型重复进行简单全局加权。

## 结论

本轮确认 RTX 4060 Laptop GPU 与当前 PyTorch/CUDA 环境可稳定完成全量
500 标签训练。轻量 CNN 的主要限制不是阈值，而是从零训练的序列表征仍
弱于大词表 TF-IDF 的 motif 记忆能力。后续提分应优先执行交接文档中的
200k/不限词表和分块 3-mer、4-5-mer 表示筛选；GPU CNN 只保留为规则允许
外部预训练权重时的后续接口，不替换当前主模型。

## 复现

```powershell
python -m src.train_gpu --config configs/iteration2_cnn_gpu_stats_long.json
python -m src.thresholds --scores artifacts/runs/EXP-20260924-030-iteration2-cnn-gpu-stats-long/scores.npz --train data/train.csv --output-prefix artifacts/metrics/EXP-20260924-030-iteration2-cnn-gpu-stats-long-thresholds --seed 42 --holdout-fraction 0.5 --shrinkage 25
```

完整模型、连续分数和测试分数保存在本地 `artifacts/runs/`；阈值摘要和
逐标签诊断保存在本地 `artifacts/metrics/`，不覆盖第一轮产物。
