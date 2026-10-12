"""Read official archive members directly and build a complete unsplit training asset."""

import argparse
import hashlib
import json
import time
import zipfile
from pathlib import Path

import pandas as pd

from scripts.enrich_original_data import iter_metadata_records
from src.external_go_data import load_go_ontology
from src.official_raw_training_data import assemble_raw_training_dataset
from src.residue_cipher import ALPHABET
from src.run_decoded_esm import file_sha256


def build(config_path):
    started = time.monotonic()
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    output = Path(cfg["output_dir"])
    summary_path = Path(cfg["metrics_prefix"] + "-summary.json")
    if output.exists() or summary_path.exists():
        raise FileExistsError("New raw dataset experiment and output required")
    before = {path: file_sha256(path) for path in cfg["expected_source_hashes"]}
    if before != cfg["expected_source_hashes"]:
        raise ValueError("An official or external raw source changed")
    source_paths = [str(Path(config_path)), "scripts/build_official_raw_training.py",
                    "src/official_raw_training_data.py", "scripts/enrich_original_data.py",
                    "src/external_go_data.py", "src/data.py"]
    source_hashes = {path: file_sha256(path) for path in source_paths}
    mapping_receipt = json.loads(Path(cfg["residue_mapping"]).read_text(encoding="utf-8"))
    mapping = mapping_receipt["mapping"]
    if (mapping_receipt["labels_used"] is not False or mapping_receipt["full_sequences_verified"] is not True
            or set(mapping) != ALPHABET or set(mapping.values()) != ALPHABET):
        raise ValueError("A complete label-free source-bound residue mapping is required")
    labels = [f"label_{index}" for index in range(cfg["expected_label_count"])]
    archive_hashes = {}
    with zipfile.ZipFile(cfg["official_archive"]) as archive:
        for member in (cfg["train_member"], cfg["test_member"], cfg["template_member"]):
            checksum = hashlib.sha256()
            with archive.open(member) as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    checksum.update(block)
            archive_hashes[member] = checksum.hexdigest()
            if archive_hashes[member] != before[member]:
                raise ValueError("Original archive member differs from official CSV")
        with archive.open(cfg["train_member"]) as handle:
            training = pd.read_csv(handle, dtype={"protein_id": str, "sequence": str,
                                                 **{label: "uint8" for label in labels}})
        with archive.open(cfg["test_member"]) as handle:
            test = pd.read_csv(handle, dtype=str, keep_default_na=False)
    if (len(training) != cfg["expected_official_rows"] or len(test) != cfg["expected_test_rows"]
            or training.columns.tolist() != ["protein_id", "sequence", *labels]):
        raise ValueError("Official full dataset rows or label schema differ")
    print("Reading full official archive training rows; no earlier dataset roles or label groups", flush=True)

    def external_records():
        for count, record in enumerate(iter_metadata_records(cfg["external_path"]), start=1):
            if count % 50000 == 0:
                print(json.dumps(dict(reviewed_records_read=count, elapsed_seconds=round(time.monotonic() - started))), flush=True)
            yield record

    summary = assemble_raw_training_dataset(training, test, mapping, external_records(),
                                            load_go_ontology(cfg["ontology_path"]), output)
    with zipfile.ZipFile(cfg["official_archive"]) as archive:
        (output / "test.csv").write_bytes(archive.read(cfg["test_member"]))
        (output / "submit_template_v1.csv").write_bytes(archive.read(cfg["template_member"]))
    canonical_test = test.copy()
    canonical_test["sequence"] = canonical_test.sequence.str.translate(str.maketrans(mapping))
    canonical_test.to_csv(output / "canonical-test-sequences.csv", index=False)
    readme = f"""# 从官方原始数据直接构建的统一训练集

本数据包含官方全部 {summary['official_rows']:,} 行和额外 {summary['external_rows']:,} 条外部蛋白，共 {summary['total_rows']:,} 行。直接读取官方压缩包，没有读取 EXP378 补全表，没有筛选低、中、高频标签，没有按旧 fit/monitor 等角色过滤或重新划分。

`train.csv` 的前 {summary['official_rows']:,} 行与官方原训练表的全部 502 列值及顺序一致。追加行的序列编码与官方原序列保持一致；其 500 个匿名标签未知，CSV 中留空。`main-labels.npz` 明确记录 targets 与 known 掩码，外部占位 0 的 known 全为 0。

`canonical-sequences.csv` 是真实氨基酸序列，用于预训练蛋白模型。`go-annotations.csv` 与 `go-positive-known.npz` 对齐全部训练行，提供外部真实 GO 辅助监督。未注释 GO 格是未知，不是负例。这是主任务 500 标签加 GO 辅助任务的训练数据，不能把外部未知标签补 0 后用于普通 500 标签 BCE。

读取方式：

```python
from src.official_raw_training_data import load_raw_training_dataset

data = load_raw_training_dataset("{cfg['output_dir']}")
batch = data.batch([0, {summary['official_rows']}])
```

训练批次包含 `sequences`、`main_targets`、`main_known`、`go_positive_known` 和 `source_kind`。主任务损失必须应用 main_known。GO 非零项是已知阳性，不能将其所有零项监督为阴性。

官方 test 和模板原样附带，不并入训练数据。数据构建完成后再进行训练/验证划分；用于历史实验比较时，原 seed42 验证 ID 及其 GO 监督须隔离。外部去除完全相同序列，不声称已去除所有同源蛋白。原始数据、历史实验和暂停的模型训练均未修改或重启。
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    after = {path: file_sha256(path) for path in before}
    if before != after or source_hashes != {path: file_sha256(path) for path in source_paths}:
        raise ValueError("A raw input or preparation source changed during this run")
    receipt = dict(experiment_id=cfg["experiment_id"], **summary,
                   input_source_hashes=before, executed_source_hashes=source_hashes,
                   official_archive_member_hashes=archive_hashes,
                   official_archive_is_primary_input=True,
                   originals_and_prior_datasets_overwritten=False,
                   sequence_mapping_applied_to_all_rows=True,
                   elapsed_seconds=round(time.monotonic() - started, 3),
                   output_hashes={path.name: file_sha256(path) for path in output.iterdir() if path.is_file()})
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2), flush=True)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    build(parser.parse_args().config)
