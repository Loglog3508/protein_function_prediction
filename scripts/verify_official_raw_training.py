"""Independently compare the augmented dataset against both raw sources."""

import argparse
import gzip
import hashlib
import json
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from scripts.verify_external_enrichment import ontology_graph
from src.official_raw_training_data import load_raw_training_dataset


ALLOWED_CODES = {"EXP", "IDA", "IPI", "IMP", "IGI", "IEP", "HTP", "HDA",
                 "HMP", "HGI", "HEP", "TAS", "IC"}


def require(condition, description):
    if not condition:
        raise ValueError(description)


def file_digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def seq_digest(value):
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def fields(value):
    return set(value.split(";")) if value else set()


def independent_raw_records(path):
    """Separate minimal parser; retain per-accession evidence without builder imports."""
    accession, sequence_parts, annotations, in_sequence = "", [], set(), False
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("ID   "):
                require("Reviewed;" in line, "Non-reviewed external record")
                accession, sequence_parts, annotations, in_sequence = "", [], set(), False
            elif line.startswith("AC   ") and not accession:
                accession = line[5:].split(";")[0].strip()
            elif line.startswith("DR   GO; "):
                parts = [value.strip() for value in line[5:].split(";")]
                if len(parts) >= 4 and ":" in parts[3]:
                    code, source = parts[3].split(":", 1)
                    if code in ALLOWED_CODES and not any(part.startswith("NOT") for part in parts):
                        annotations.add((parts[1], code, source.rstrip("."), accession))
            elif line.startswith("SQ   "):
                in_sequence = True
            elif line.startswith("//"):
                sequence = "".join(sequence_parts)
                require(bool(accession and sequence), "Incomplete raw external record")
                yield sequence, annotations
                in_sequence = False
            elif in_sequence:
                sequence_parts.append("".join(line.split()))


def verify(config_path):
    started = time.monotonic()
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    output = Path(cfg["output_dir"])
    result_path = output / "independent-verification.json"
    metric_path = Path(cfg["metrics_prefix"] + "-independent-verification.json")
    require(not result_path.exists() and not metric_path.exists(), "Use a fresh verification output")
    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    require(receipt["input_source_hashes"] == cfg["expected_source_hashes"], "Input hashes differ from configuration")
    for path, checksum in receipt["input_source_hashes"].items():
        require(file_digest(path) == checksum, "Raw source changed: " + path)
    for path, checksum in receipt["executed_source_hashes"].items():
        require(file_digest(path) == checksum, "Preparation source changed: " + path)
    for name, checksum in receipt["output_hashes"].items():
        require(file_digest(output / name) == checksum, "Generated output changed: " + name)
    schema = json.loads((output / "dataset-schema.json").read_text(encoding="utf-8"))
    n = cfg["expected_official_rows"]
    total = schema["total_rows"]
    labels = [f"label_{index}" for index in range(cfg["expected_label_count"])]
    mapping = json.loads(Path(cfg["residue_mapping"]).read_text(encoding="utf-8"))["mapping"]
    translator = str.maketrans(mapping)
    canonical = pd.read_csv(output / "canonical-sequences.csv", dtype=str, keep_default_na=False)
    annotations = pd.read_csv(output / "go-annotations.csv", dtype=str, keep_default_na=False)
    with np.load(output / "main-labels.npz", allow_pickle=False) as saved:
        ids, label_names, targets, known = (saved[key] for key in ("protein_ids", "label_columns", "targets", "known"))
    require(targets.shape == known.shape == (total, len(labels)), "Main supervision dimensions")
    require(label_names.tolist() == labels, "Main label order")
    require(len(canonical) == len(annotations) == total, "Aligned table row counts")
    require(ids.tolist() == canonical.protein_id.tolist() == annotations.protein_id.tolist(), "Aligned IDs")
    require(len(set(ids)) == total, "Duplicate protein IDs")
    require(bool(known[:n].all()) and not known[n:].any() and not targets[n:].any(), "Known/unknown masks")
    require(annotations.source_kind.tolist() == ["official"] * n + ["uniprot_external"] * (total - n), "Source flags")
    require(all(schema[key] is False for key in ("old_enriched_dataset_used", "prior_protocol_or_label_groups_used",
                                               "dataset_split_performed", "training_started", "original_labels_modified")),
            "Raw dataset must not use earlier groups or roles")
    checked_labels, row = 0, 0
    with zipfile.ZipFile(cfg["official_archive"]) as archive:
        generated = pd.read_csv(output / "train.csv", dtype=str, keep_default_na=False, iterator=True)
        with archive.open(cfg["train_member"]) as handle:
            for original in pd.read_csv(handle, dtype=str, keep_default_na=False, chunksize=5000):
                chunk = generated.get_chunk(len(original))
                require(original.reset_index(drop=True).equals(chunk.reset_index(drop=True)), "Original row values or order changed")
                require(original.columns.tolist() == ["protein_id", "sequence", *labels], "Official complete schema")
                stop = row + len(original)
                require(np.array_equal(original[labels].to_numpy(np.uint8), targets[row:stop]), "Original label cell changed")
                require(original.sequence.str.translate(translator).tolist() == canonical.sequence.iloc[row:stop].tolist(), "Official residue decoding")
                row = stop
                checked_labels += len(original) * len(labels)
        require(row == n, "Some original rows were filtered")
        for chunk in generated:
            stop = row + len(chunk)
            require(chunk[labels].eq("").all().all(), "External anonymous labels must remain blank")
            require(chunk.protein_id.tolist() == ids[row:stop].tolist(), "External ID order")
            require(chunk.sequence.str.translate(translator).tolist() == canonical.sequence.iloc[row:stop].tolist(), "External residue encoding")
            row = stop
        generated.close()
        require(row == total, "Training CSV total rows")
        for key, name in (("test_member", "test.csv"), ("template_member", "submit_template_v1.csv")):
            require((output / name).read_bytes() == archive.read(cfg[key]), "Official test or template bytes changed")
        with archive.open(cfg["test_member"]) as handle:
            test = pd.read_csv(handle, dtype=str, keep_default_na=False)
    canonical_test = pd.read_csv(output / "canonical-test-sequences.csv", dtype=str, keep_default_na=False)
    require(len(test) == cfg["expected_test_rows"], "Official test row count")
    require(test.protein_id.tolist() == canonical_test.protein_id.tolist(), "Canonical test IDs")
    require(test.sequence.str.translate(translator).tolist() == canonical_test.sequence.tolist(), "Canonical test decoding")
    original_set = set(canonical.sequence.iloc[:n])
    test_set = set(canonical_test.sequence)
    external_set = set(canonical.sequence.iloc[n:])
    require(len(external_set) == total - n and not external_set & (original_set | test_set), "External complete-sequence duplicates")
    sequence_hashes = canonical.sequence.map(seq_digest).tolist()
    require(sequence_hashes == annotations.sequence_sha256.tolist(), "Canonical sequence hashes")
    require(ids[n:].tolist() == ["UP_EXT_" + value[:24] for value in sequence_hashes[n:]], "External deterministic IDs")
    print("All official rows and original label cells agree; independently scanning external raw records", flush=True)
    graph, aliases = ontology_graph(cfg["ontology_path"])
    closure_cache = {}

    def ancestors(identifier):
        if identifier not in closure_cache:
            seen, pending = set(), [identifier]
            while pending:
                value = pending.pop()
                current = aliases.get(value, value)
                if current in seen or current not in graph or graph[current]["obsolete"]:
                    continue
                seen.add(current)
                pending.extend(graph[current]["parents"])
            closure_cache[identifier] = seen
        return closure_cache[identifier]

    expected_evidence = defaultdict(set)
    eligible_external, record_count, supported_count, excluded_test = set(), 0, 0, 0
    for sequence, raw_evidence in independent_raw_records(cfg["external_path"]):
        record_count += 1
        if sequence in test_set:
            excluded_test += 1
            continue
        accepted = {value for value in raw_evidence if ancestors(value[0])}
        if not accepted:
            continue
        supported_count += 1
        if sequence not in original_set:
            eligible_external.add(sequence)
        expected_evidence[seq_digest(sequence)].update(accepted)
        if record_count % 100000 == 0:
            print(json.dumps(dict(independently_scanned=record_count)), flush=True)
    require(eligible_external == external_set, "External dataset omits or adds eligible raw proteins")
    require(record_count == schema["source_records_scanned"] and supported_count == schema["supported_non_test_records"]
            and excluded_test == schema["excluded_official_test_records"], "Raw record counts")
    evidence = pd.read_csv(output / "go-direct-evidence.csv", dtype=str, keep_default_na=False)
    actual_evidence = defaultdict(set)
    origins = dict(zip(annotations.sequence_sha256, annotations.source_kind, strict=True))
    for item in evidence.itertuples(index=False):
        require(item.source_kind == origins[item.sequence_sha256], "Evidence source kind")
        actual_evidence[item.sequence_sha256].add((item.go_id, item.evidence_code, item.evidence_source, item.accession))
    require(actual_evidence == expected_evidence, "Accession/GO/code/source differs from raw Swiss-Prot")
    require(len(evidence) == sum(map(len, actual_evidence.values())), "Duplicate direct evidence")
    accessions = pd.read_csv(output / "sequence-accessions.csv", dtype=str, keep_default_na=False)
    actual_accessions = {(item.sequence_sha256, item.source_kind, item.accession) for item in accessions.itertuples(index=False)}
    expected_accessions = {(sequence_hash, origins[sequence_hash], value[3])
                           for sequence_hash, values in expected_evidence.items() for value in values}
    require(actual_accessions == expected_accessions and len(actual_accessions) == len(accessions), "Source accession attribution")
    vocabulary = pd.read_csv(output / "go-vocabulary.csv", dtype={"go_id": str})
    go_ids = vocabulary.go_id.to_numpy()
    require(vocabulary.column.tolist() == list(range(len(vocabulary))) and vocabulary.go_id.is_unique, "GO column mapping")
    matrix = sparse.load_npz(output / "go-positive-known.npz").tocsr()
    require(matrix.shape == (total, schema["go_terms"]) and matrix.has_canonical_format
            and np.all(matrix.data == 1) and matrix.nnz == schema["go_positive_known_cells"], "GO positive matrix")
    go_vocabulary = set()
    official_go_rows = 0
    for index, item in enumerate(annotations.itertuples(index=False)):
        values = expected_evidence.get(item.sequence_sha256, set())
        direct = {value[0] for value in values}
        propagated = set().union(*(ancestors(value) for value in direct))
        experimental = {value[0] for value in values if value[1] not in {"TAS", "IC"}}
        curated = {value[0] for value in values if value[1] in {"TAS", "IC"}}
        actual = set(go_ids[matrix.indices[matrix.indptr[index]:matrix.indptr[index + 1]]])
        require(fields(item.go_direct) == direct and fields(item.go_propagated) == propagated == actual, "GO row reconstruction")
        require(fields(item.go_experimental) == experimental and fields(item.go_curated) == curated, "GO evidence categories")
        require(fields(item.source_accessions) == {value[3] for value in values}, "Row accessions")
        go_vocabulary.update(propagated)
        official_go_rows += int(index < n and bool(direct))
        require(index < n or bool(direct), "External row lacks GO supervision")
    require(go_vocabulary == set(go_ids) and official_go_rows == schema["official_rows_with_go"], "GO vocabulary and original coverage")
    loaded = load_raw_training_dataset(output)
    batch = loaded.batch([n, 0])
    require(batch["protein_ids"].tolist() == [ids[n], ids[0]], "Loader requested row order")
    require(not batch["main_known"][0].any() and batch["main_known"][1].all(), "Loader mask semantics")
    require(batch["go_positive_known"][0].nnz > 0, "External GO training targets unavailable")
    result = dict(experiment_id=cfg["experiment_id"], passed=True, official_rows=n,
                  external_rows=total - n, total_training_rows=total,
                  original_label_cells_compared=checked_labels, original_label_cells_changed=0,
                  original_row_order_preserved=True, original_rows_filtered=0,
                  all_source_and_output_hashes_equal=True, official_test_and_template_bytes_identical=True,
                  external_exact_sequence_overlap=0, all_eligible_external_sequences_included=True,
                  raw_external_records_independently_read=record_count, raw_evidence_rows_verified=len(evidence),
                  official_rows_with_go=official_go_rows, go_terms=len(go_ids),
                  go_positive_cells_independently_reconstructed=int(matrix.nnz),
                  external_main_known_cells=0, external_csv_labels_all_blank=True,
                  dataset_loader_and_mixed_batch_verified=True, old_role_or_label_group_filtering_used=False,
                  verification_source_sha256=file_digest(__file__), receipt_sha256=file_digest(output / "receipt.json"),
                  elapsed_seconds=round(time.monotonic() - started, 3))
    text = json.dumps(result, indent=2)
    result_path.write_text(text, encoding="utf-8")
    metric_path.write_text(text, encoding="utf-8")
    print(text, flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    verify(parser.parse_args().config)
