# 官方原始文件与外部数据合并训练集

日期：2026-10-12。数据准备编号：`EXT-20261012-379-official-raw-augmented-training`。

已直接读取官方压缩包 `protein_function_prediction_data_20260926.zip`，完整保留官方训练表，再加入外部蛋白，生成新的统一训练数据集。构建不读取上轮 EXP378 补全表，不按低、中、高频标签或旧 fit、monitor、calibration、audit 角色筛选原始记录。此阶段只准备数据，未启动模型训练。

## 实际生成结果

| 项目 | 数量 |
|---|---:|
| 官方原始训练记录 | 113,796 |
| 外部新增唯一蛋白序列 | 13,522 |
| 新训练集总记录 | **127,318** |
| 官方原有匿名标签 | 500 |
| 保留并逐格核验的官方标签值 | 56,898,000 |
| 匹配到支持证据 GO 注释的官方记录 | 65,551 |
| 暂无本次支持证据 GO 注释的官方记录 | 48,245 |
| 含 GO 辅助监督的训练记录 | 79,073 |
| GO 词表维度 | 28,861 |
| 传播后已知 GO 阳性单元格 | 3,138,204 |
| 可追溯的直接 GO 证据记录 | 433,620 |

官方训练记录全部保留，顺序及全部 502 列值与压缩包一致。即使同序列的原始记录标签不同，也保留原记录，不合并、不判断哪条标签正确。外部蛋白按完整序列合并重复 accession；去除与任何官方 train 或 test 序列完全相同的外部样本。没有将完全相同序列去重表述成同源簇隔离。

13,522 是相对于官方原始训练与测试数据的新增序列数。此前外部准备资产中可能已有这些序列，不能理解为在上轮新增数量之外又增加 13,522 条。

## 原始来源

官方 `train.csv`、`test.csv` 和模板的内容哈希均与官方压缩包对应成员一致。构建实际从压缩包读取 train 和 test；原始文件保持原状。

外部来源为本地已下载并核验的 UniProtKB/Swiss-Prot 完整 reviewed 快照 `uniprot_sprot.dat.gz`，本次扫描 575,748 条记录。下载记录的 Last-Modified 为 2026-09-03。GO 本体为 `go-basic.obo`，版本 `releases/2026-07-26`。本次复用固定原始快照，没有将其描述为今日新下载的数据。

序列匹配使用已验证、不使用标签的固定氨基酸置换映射恢复官方序列。新 `train.csv` 中外部序列反向转换到官方相同编码规则；真实氨基酸序列另存，用于蛋白预训练模型输入。没有依据蛋白编号的字面相似性关联记录。

所有输入文件、准备源码与生成产物的 SHA-256 保存于新数据目录的 `receipt.json`。配置未引用旧训练角色、分层标签列表或 EXP378 补全表。

## 文件及用途

数据目录：`artifacts/runs/EXT-20261012-379-official-raw-augmented-training/`。

| 文件 | 用途 |
|---|---|
| `train.csv` | 新的 127,318 行训练主表；保留官方 502 列格式 |
| `canonical-sequences.csv` | 与训练主表按 ID 和行号对齐的真实氨基酸序列 |
| `main-labels.npz` | 全部样本的 500 标签 targets 与 known 掩码 |
| `go-annotations.csv` | 每个样本的 GO 直接注释、传播注释、证据分类与来源 |
| `go-positive-known.npz` | 全部样本的 GO 已知阳性稀疏矩阵 |
| `go-vocabulary.csv` | GO 矩阵列号与 GO ID 的对应表 |
| `sequence-accessions.csv` | 序列与外部 accession 的对应关系 |
| `go-direct-evidence.csv` | 每条 accession–GO–证据代码–证据来源的精确对应 |
| `test.csv`、`submit_template_v1.csv` | 官方测试与模板的原始字节副本 |
| `canonical-test-sequences.csv` | 官方测试序列的真实氨基酸版本 |
| `dataset-schema.json`、`receipt.json` | 数据结构、处理规则、计数及来源校验 |
| `independent-verification.json` | 独立逐项核验结果 |

`train.csv` 约 190 MB；数据文件保留本地，未提交或推送。

## 标签的真实含义和训练用法

官方样本的 `label_0` 至 `label_499` 全部保持原值，`main_known` 全为真。外部样本没有官方 500 匿名标签的权威对应，CSV 中这些格子留空，`main_known` 全为假。数组里的外部 targets=0 仅是存储占位，不代表已确认阴性。

因此，这是一份“原始 500 标签监督 + 外部真实 GO 辅助监督”的新训练集；不能把它当成每条样本都具备 500 个完整真值的普通多标签数据。仅用原 500 标签损失时，外部样本不产生该损失；要让它们参与学习，需要接入 GO 辅助任务或其他适合未知标签的目标。没有制造匿名标签与 GO 的推测映射或伪真值。

GO 保留实验类 `EXP/IDA/IPI/IMP/IGI/IEP/HTP/HDA/HMP/HGI/HEP`，人工整理类 `TAS/IC` 单独记录。`IEA`、`ND` 与显式 `NOT` 不用作阳性，过滤过时或不存在的 GO 项。传播仅沿 `is_a` 与 `part_of`。

GO 稀疏矩阵的非零项是已知阳性，零项是未知。不能将全部零项当负例直接计算普通全矩阵 BCE；后续应选择明确处理阳性/未知数据的学习方法。只拟合阳性也不能自动提供有效的负例区分。

读取新数据应使用专门提供的加载器：

```python
from src.official_raw_training_data import load_raw_training_dataset

data = load_raw_training_dataset(
    "artifacts/runs/EXT-20261012-379-official-raw-augmented-training"
)
batch = data.batch([0, 113796])
```

批次同时返回真实序列、500 标签及已知掩码、GO 阳性矩阵和来源类别。已有只接受完整 500 标签的普通加载流程不能直接替代该加载器。

本次数据资产没有提前划分训练/验证。后续用于历史对照时，保持固定 seed42 验证 ID，并将这些验证样本及其 GO 监督隔离。其他新实验需先明确划分与泄漏控制，再拟合或调参。测试标签未补入训练，测试记录没有并入训练主表。

## 已完成的核验

生成模块及相关数据处理测试：21 项通过。

独立核验直接读取官方压缩包，逐行核对全部原始 502 列、行序和 56,898,000 个官方标签值，修改数量为 0，原始记录过滤数量为 0。外部 500 标签列全部留空，已知掩码全部为 0；外部序列与官方 train/test 完全相同序列重合数量为 0。

独立解析器重新扫描外部原始快照全部 575,748 条记录，确认所有满足本次证据政策且不属于官方数据的唯一序列均已纳入；逐项复核 433,620 条直接证据的 accession、GO、证据代码与来源，使用独立本体遍历重建全部 3,138,204 个 GO 阳性格。全部输入、准备源码和输出哈希一致，官方 test 与模板逐字节一致。

实际加载完整新数据集并读取混合官方/外部批次，ID 顺序、真实序列、500 标签掩码和外部 GO 监督均可用。独立核验结论为 `passed=true`；这是数据正确性检查，不代表新模型已经训练或分数已经提升。

## 复现

```powershell
python -m scripts.build_official_raw_training --config configs/official_raw_augmented_training_20261012.json
python -m scripts.verify_official_raw_training --config configs/official_raw_augmented_training_20261012.json
```

当前编号的产物已存在，脚本会拒绝覆盖。需要再次生成时，复制配置并使用新的实验编号、输出目录及指标前缀。
