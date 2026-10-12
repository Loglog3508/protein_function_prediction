# 从官方原始数据直接构建的统一训练集

本数据包含官方全部 113,796 行和额外 13,522 条外部蛋白，共 127,318 行。直接读取官方压缩包，没有读取 EXP378 补全表，没有筛选低、中、高频标签，没有按旧 fit/monitor 等角色过滤或重新划分。

`train.csv` 的前 113,796 行与官方原训练表的全部 502 列值及顺序一致。追加行的序列编码与官方原序列保持一致；其 500 个匿名标签未知，CSV 中留空。`main-labels.npz` 明确记录 targets 与 known 掩码，外部占位 0 的 known 全为 0。

`canonical-sequences.csv` 是真实氨基酸序列，用于预训练蛋白模型。`go-annotations.csv` 与 `go-positive-known.npz` 对齐全部训练行，提供外部真实 GO 辅助监督。未注释 GO 格是未知，不是负例。这是主任务 500 标签加 GO 辅助任务的训练数据，不能把外部未知标签补 0 后用于普通 500 标签 BCE。

读取方式：

```python
from src.official_raw_training_data import load_raw_training_dataset

data = load_raw_training_dataset("artifacts/runs/EXT-20261012-379-official-raw-augmented-training")
batch = data.batch([0, 113796])
```

训练批次包含 `sequences`、`main_targets`、`main_known`、`go_positive_known` 和 `source_kind`。主任务损失必须应用 main_known。GO 非零项是已知阳性，不能将其所有零项监督为阴性。

官方 test 和模板原样附带，不并入训练数据。数据构建完成后再进行训练/验证划分；用于历史实验比较时，原 seed42 验证 ID 及其 GO 监督须隔离。外部去除完全相同序列，不声称已去除所有同源蛋白。原始数据、历史实验和暂停的模型训练均未修改或重启。
