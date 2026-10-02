# 第十四次迭代：Macro F1 逐标签诊断与 20 条过拟合测试

日期：2026-09-30。本轮是错误诊断，不是新模型成绩；没有生成或上传平台提交。

## 1. 有效交叉拟合分类报告

使用实验 103 的 500 标签连续分数、固定 `iteration4_tail` 验证集和 seed-42 两折阈值交叉拟合，避免使用全验证集拟合阈值造成乐观偏差：

- Macro F1：`0.338552`
- 真实正例率：`5.9166%`
- 预测正例率：`9.1454%`
- 分类报告 Macro 平均：Precision `0.28`、Recall `0.45`、F1 `0.34`
- 最差标签包括 `label_430`（验证正例 2，F1 0）、`label_361`（11，F1 0）、`label_298`（7，F1 0）、`label_281`（15，F1 0.0357）。这些类别普遍 FP 明显多于 TP。

完整报告和逐类混淆矩阵已保存，不在迭代表中重复粘贴 500 行：

- `artifacts/metrics/EXP-20260930-120-f1-diagnostics-crossfit42-classification-report.txt`
- `artifacts/metrics/EXP-20260930-120-f1-diagnostics-crossfit42-confusion-matrices.csv`
- `artifacts/metrics/EXP-20260930-120-f1-diagnostics-crossfit42-score-diagnostics.csv`

## 2. 支持度统计

验证集 500 个标签中：

| 验证正例数 | 标签数 |
| ---: | ---: |
| ≤2 | 1 |
| ≤5 | 7 |
| ≤10 | 70 |
| ≤20 | 179 |
| ≤50 | 353 |
| ≤100 | 438 |

因此“个位数标签导致 Macro F1 被拖低”成立，但不是全部原因：很多训练支持度超过 1,000 的标签同样出现低 F1，且低频标签的预测数量经常是真实数量的 2～7 倍。

## 3. 20 条过拟合测试

- 20 条真实训练行 + identity 特征 + RandomForest：500 标签，239 个有正例标签，训练集 Macro F1=`1.0`，逐元素准确率=`1.0`，分数全为有限值。
- 同 20 条真实序列 + 21 维 composition + SGD：Macro F1=`0.018901`，分数全为有限值。

identity 测试证明标签模型训练、预测和多标签聚合链路可以记忆小数据；真实序列特征无法记忆，说明主要瓶颈是表示和低频排序，不是 CUDA 或基础训练代码错误。

## 4. 决策

不能合并稀有标签，也不能更换竞赛指标；这会改变提交任务。已把标准 `focal_bce` 接入 ESM 微调器，并准备下一轮配置：

- `configs/iteration14_esm_last4_focal_full_epoch1.json`
- 解冻 ESM-2 35M 最后 4 层；
- 使用标准 focal BCE，`gamma=2.0`；
- 仍只使用竞赛训练数据和允许的预训练权重。

历史状态已更新（2026-09-30）：实验 117 已完成，默认阈值 F1=0.143612、AUC=0.699404。
第十五次审计按 EXP-103 口径复核后，五 seed F1=0.152414。
实验 121 focal 尚未启动；当前只运行实验 125 的三 epoch 单变量对照。
详见 `reports/experiments/EXP-20260930-126-esm-provenance-audit.md`。

### 补充：真实序列与 frozen ESM 的 20 条测试

EXP-122 的真实 3–5-mer TF-IDF + SGD 和 EXP-123 的 frozen ESM-35M +
近乎无正则 LogisticRegression 均达到训练 Macro F1=1.0、逐元素准确率=1.0。
500 个输出标签中 239 个有正例；按竞赛规则计算 F1 时跳过 261 个无正例标签。
因此 composition 的低训练 F1 只能说明该低维特征无法记忆这 20 条样本，
不能据此断言所有序列表征链路都存在错误或所有其他 bug 都已排除。
