# 蛋白质功能预测（Protein Function Prediction）

**多标签分类 · 序列数据 · 评估指标 Macro F1**

> 给定蛋白质的氨基酸序列，预测其功能标签（`label_0` ~ `label_499`）。
> 这是一个多标签分类问题：每个蛋白质可以有零个、一个或多个功能标签。

## 赛题背景

蛋白质是生命活动的主要执行者，其功能由氨基酸序列决定。目前已建立标准化的功能注释体系（按分子功能、生物过程、细胞组分等维度组织），但实验手段确定功能成本高、周期长，大量蛋白质仍缺乏功能注释。自动化功能预测可以大幅加速注释过程，是生物信息学的重要方向。

本竞赛面向高年级本科生与研究生，旨在真实生物数据场景中锻炼大数据分析、机器学习建模与训练能力。

## 数据说明

数据由竞赛主办方（杭州睿数科技有限公司 / 海豚实验室 与 杭州分子势能科技有限公司）提供，源自公开的蛋白质功能注释数据库，选取出现频率最高的 500 个功能术语作为标签，并重映射为 `label_0` ~ `label_499`。

| 文件 | 规模 | 列 | 说明 |
| --- | --- | --- | --- |
| `data/train.csv` | 44,000 | 502 | `protein_id` + `sequence` + 500 个标签，含标签 |
| `data/test.csv` | 11,000 | 2 | `protein_id` + `sequence`，仅序列 |
| `data/submit_template_v1.csv` | 11,000 | 501 | 提交模板 `protein_id` + `label_0` ~ `label_499` |

数据特征：

- 标签数 500，标签密度约 3.4%（稀疏标签场景），训练集中每个标签均有正例
- 每条序列平均 17 个标签（标准差约 8.5，范围 1 ~ 55）
- 序列长度平均 300（标准差 116，最短 100，最长 500，中位数 301）
- 仅含 20 种标准氨基酸，无缺失值
- 训练集 : 测试集 ≈ 4 : 1

## 提交格式

CSV 文件，列为 `protein_id,label_0,...,label_499`：

- `protein_id`：字符串，必须与测试集一致
- `label_X`：整数 `0` 或 `1`

可直接以 `data/submit_template_v1.csv` 为骨架填充，输出 `submission.csv`。

## 基线方案

`baseline-v2.ipynb`：

1. 特征工程：氨基酸组成（20 维频率）+ 序列长度（log1p）
2. 建模：对 500 个标签分别训练 `RandomForestClassifier`（`class_weight='balanced'`）
3. 输出：`submission.csv`

> 进阶方向：使用 ESM / ProtBERT 等蛋白质预训练语言模型提取序列嵌入，再以多标签分类头（或逐标签分类器）建模，预期显著优于手工组成特征。

## 目录结构

```
data/train.csv               训练集（含 500 个标签）
data/test.csv                测试集（仅序列）
data/submit_template_v1.csv  提交模板
baseline-v2.ipynb            基线模型
RULES.md                     赛题规则与评测细则
```

## 运行方式

```bash
pip install pandas numpy scikit-learn
jupyter notebook baseline-v2.ipynb
```

运行后生成 `submission.csv`，即可提交。

## 竞赛规则要点

- 仅可使用竞赛提供的数据，禁止引入外部数据集
- 每支队伍每个赛道每天最多提交 3 次
- 排名前三的队伍须在竞赛结束后 48 小时内提交完整可运行代码
- 队伍规模 1 ~ 3 人
- 完整赛题规则、评分细则与 Macro F1 计算方式见 `RULES.md`
