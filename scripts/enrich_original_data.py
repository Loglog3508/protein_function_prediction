"""Prepare source-bound raw metadata and positive-only GO auxiliary data."""

import argparse
import csv
import gzip
import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from src.external_go_data import REFERENCE_EVIDENCE_CODES, load_go_ontology, propagate_go
from src.residue_cipher import decode_sequence
from src.run_decoded_esm import file_sha256, verified_existing_inputs


EXPERIMENTAL_CODES = frozenset({"EXP", "IDA", "IPI", "IMP", "IGI", "IEP",
                              "HTP", "HDA", "HMP", "HGI", "HEP"})
METADATA_FIELDS = ("uniprot_accessions", "uniprot_entry_names", "protein_names",
                   "gene_names", "organism_names", "organism_ids", "ec_numbers")
GO_FIELDS = ("go_direct_supported", "go_ancestors_supported",
             "go_direct_experimental", "go_direct_curated")


def sequence_hash(sequence):
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def joined(values):
    return ";".join(sorted({str(value) for value in values if value}))


def clean_evidence_text(value):
    return re.sub(r"\s*\{[^}]*\}", "", value).strip()


def iter_metadata_records(path):
    record = None
    in_sequence = False
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            prefix = line[:2]
            if prefix == "ID":
                if record is not None:
                    raise ValueError("Unterminated Swiss-Prot record")
                match = re.fullmatch(r"ID\s+(\S+)\s+Reviewed;\s+(\d+) AA\.\s*", line)
                if match is None:
                    raise ValueError("Expected a complete reviewed Swiss-Prot record")
                record = dict(entry_name=match[1], declared_length=int(match[2]),
                              accessions=[], descriptions=[], genes=[], organisms=[],
                              taxonomy=[], sequence_parts=[], annotations=[], entry_dates=[])
                in_sequence = False
            elif record is None:
                if line.strip():
                    raise ValueError("Content outside a Swiss-Prot record")
            elif prefix == "AC":
                record["accessions"].extend(part.strip() for part in line[5:].split(";") if part.strip())
            elif prefix == "DE":
                record["descriptions"].append(line[5:].strip())
            elif prefix == "GN":
                record["genes"].append(line[5:].strip())
            elif prefix == "OS":
                record["organisms"].append(line[5:].strip())
            elif prefix == "OX":
                taxonomy = clean_evidence_text(line.split("=", 1)[1].split(";", 1)[0])
                record["taxonomy"].extend(re.findall(r"\d+", taxonomy))
            elif prefix == "DT":
                record["entry_dates"].append(line[5:].strip())
            elif line.startswith("DR   GO; "):
                fields = [part.strip() for part in line[5:].split(";")]
                if len(fields) >= 4:
                    code, separator, source = fields[3].partition(":")
                    if (separator and code in REFERENCE_EVIDENCE_CODES
                            and not any(part.startswith("NOT") for part in fields)):
                        record["annotations"].append((fields[1], code, source.rstrip(".")))
            elif prefix == "SQ":
                in_sequence = True
            elif prefix == "//":
                sequence = "".join(record.pop("sequence_parts"))
                if (not record["accessions"] or len(sequence) != record["declared_length"]
                        or not sequence.isalpha() or not sequence.isupper()):
                    raise ValueError("Incomplete accession or invalid Swiss-Prot sequence")
                record["sequence"] = sequence
                record["accession"] = record["accessions"][0]
                description = clean_evidence_text(" ".join(record.pop("descriptions")))
                record["protein_names"] = re.findall(r"(?:RecName|AltName|SubName): Full=([^;]+);", description)
                record["ec_numbers"] = re.findall(r"\bEC=([^;]+);", description)
                record["gene_names"] = re.findall(r"(?:Name|Synonyms|OrderedLocusNames|ORFNames)=([^;]+);",
                                                   clean_evidence_text(" ".join(record.pop("genes"))))
                record["organism_name"] = " ".join(record.pop("organisms")).rstrip(".")
                yield record
                record = None
                in_sequence = False
            elif in_sequence:
                record["sequence_parts"].append("".join(line.split()))
    if record is not None:
        raise ValueError("Truncated Swiss-Prot input")


def role_maps(cfg, training_ids):
    seed_train = pd.read_csv(cfg["seed42_train_path"], dtype=str).protein_id.tolist()
    seed_validation = pd.read_csv(cfg["seed42_validation_path"], dtype=str).protein_id.tolist()
    if (len(set(seed_train)) != len(seed_train) or len(set(seed_validation)) != len(seed_validation)
            or set(seed_train) & set(seed_validation)
            or set(seed_train) | set(seed_validation) != set(training_ids)):
        raise ValueError("Fixed seed42 IDs must partition every original training row once")
    seed_roles = {identifier: "train" for identifier in seed_train}
    seed_roles.update({identifier: "validation" for identifier in seed_validation})
    protocol_roles = {}
    with np.load(cfg["protocol_path"], allow_pickle=False) as protocol:
        base = protocol["training_ids"].astype(str)
        for role in ("fit", "monitor", "calibration", "audit", "quarantine"):
            for identifier in base[protocol[role + "_indices"]]:
                if identifier in protocol_roles:
                    raise ValueError("Protocol roles overlap")
                protocol_roles[identifier] = role
        for identifier in protocol["legacy_validation_ids"].astype(str):
            if identifier in protocol_roles:
                raise ValueError("Legacy validation overlaps protocol training")
            protocol_roles[identifier] = "legacy_validation"
    if set(protocol_roles) != set(training_ids):
        raise ValueError("Protocol roles must partition the original training data")
    return seed_roles, protocol_roles


def aggregate_metadata(records):
    return dict(
        uniprot_accessions=joined(record["accession"] for record in records),
        uniprot_entry_names=joined(record["entry_name"] for record in records),
        protein_names=joined(value for record in records for value in record["protein_names"]),
        gene_names=joined(value for record in records for value in record["gene_names"]),
        organism_names=joined(record["organism_name"] for record in records),
        organism_ids=joined(value for record in records for value in record["taxonomy"]),
        ec_numbers=joined(value for record in records for value in record["ec_numbers"]),
    )


def go_values(records, ontology):
    annotations = {annotation for record in records for annotation in record["annotations"]}
    valid = {annotation for annotation in annotations if propagate_go([annotation[0]], ontology)}
    direct = {annotation[0] for annotation in valid}
    experimental = {annotation[0] for annotation in valid if annotation[1] in EXPERIMENTAL_CODES}
    curated = {annotation[0] for annotation in valid if annotation[1] in {"TAS", "IC"}}
    return dict(go_direct_supported=joined(direct),
                go_ancestors_supported=joined(propagate_go(direct, ontology)),
                go_direct_experimental=joined(experimental), go_direct_curated=joined(curated)), valid


def read_go_descriptions(path):
    descriptions = {}
    current = None
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line == "[Term]":
            current = {}
        elif line.startswith("["):
            current = None
        elif current is not None:
            if line.startswith("id: "):
                current["go_id"] = line[4:]
                descriptions[line[4:]] = current
            elif line.startswith("name: "):
                current["name"] = line[6:]
            elif line.startswith("namespace: "):
                current["namespace"] = line[11:]
    return descriptions


def prepare(cfg_path):
    started = time.monotonic()
    cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    output = Path(cfg["output_dir"])
    if output.exists():
        raise FileExistsError("Use a fresh experiment ID and output directory")
    expected = cfg["expected_source_hashes"]
    before = {path: file_sha256(path) for path in expected}
    if before != expected:
        raise ValueError("An original, split, mapping, or external source changed")
    external_receipt = json.loads(Path(cfg["external_receipt"]).read_text(encoding="utf-8"))
    if (external_receipt["sha256"] != before[cfg["external_path"]]
            or external_receipt["records"] != cfg["expected_external_records"]):
        raise ValueError("External snapshot receipt mismatch")
    decoded_cfg = json.loads(Path(cfg["decoded_config"]).read_text(encoding="utf-8"))
    decoded_paths = verified_existing_inputs(decoded_cfg, Path.cwd())
    additional_sources = [cfg["external_receipt"], cfg["decoded_config"], cfg["live_source_check"],
                          *map(str, decoded_paths.values())]
    before.update({path: file_sha256(path) for path in additional_sources})
    restored = {name: pd.read_csv(path, dtype=str, keep_default_na=False)
                for name, path in decoded_paths.items()}
    seed_roles, protocol_roles = role_maps(cfg, restored["train"].protein_id.tolist())
    original_sequences = set(restored["train"].sequence) | set(restored["test"].sequence)
    if set(restored["train"].sequence) & set(restored["test"].sequence):
        raise ValueError("Train and test restored sequences overlap")
    eligible_fit_sequences = set(restored["train"].loc[
        restored["train"].protein_id.map(protocol_roles).eq("fit")
        & restored["train"].protein_id.map(seed_roles).eq("train"), "sequence"])
    ontology = load_go_ontology(cfg["ontology_path"])
    matching = defaultdict(list)
    novel = defaultdict(list)
    total = supported = live_matches = 0
    live = json.loads(Path(cfg["live_source_check"]).read_text(encoding="utf-8"))
    live_accession = live["primaryAccession"]
    live_sequence = live["sequence"]["value"]
    print("Verified sources and split identities; scanning complete reviewed snapshot", flush=True)
    for record in iter_metadata_records(cfg["external_path"]):
        total += 1
        if record["accession"] == live_accession:
            if record["sequence"] != live_sequence:
                raise ValueError("Live official sequence differs from pinned reviewed snapshot")
            live_matches += 1
        supported += bool(record["annotations"])
        sequence = record["sequence"]
        if sequence in original_sequences:
            matching[sequence].append(record)
        elif record["annotations"] and propagate_go({row[0] for row in record["annotations"]}, ontology):
            novel[sequence].append(record)
        if total % 50000 == 0:
            print(json.dumps(dict(scanned=total, matched_sequences=len(matching),
                                  novel_supported_sequences=len(novel),
                                  seconds=round(time.monotonic() - started))), flush=True)
    if total != cfg["expected_external_records"] or live_matches != 1:
        raise ValueError("Reviewed snapshot record count or live source check failed")
    reference_matches = set()
    reference_original_pairs = set()
    for chunk in pd.read_csv(cfg["reference_tsv"], sep="\t", dtype=str, keep_default_na=False, chunksize=50000):
        rows = chunk.loc[chunk.Sequence.isin(original_sequences)]
        reference_matches.update(rows.Sequence)
        reference_original_pairs.update(zip(rows.Entry, rows.Sequence))
    parsed_original_pairs = {(record["accession"], sequence)
                             for sequence, records in matching.items() for record in records}
    if reference_matches != set(matching) or reference_original_pairs != parsed_original_pairs:
        raise ValueError("Independent reviewed TSV and flat-file exact matches differ")
    output.mkdir(parents=True)
    metadata_cache = {sequence: aggregate_metadata(records) for sequence, records in matching.items()}
    go_cache = {sequence: go_values(records, ontology)[0]
                for sequence, records in matching.items() if sequence in eligible_fit_sequences}
    label_columns = [f"label_{index}" for index in range(500)]
    missing_original = Counter({"train": 0, "test": 0})
    input_counts = Counter()
    stats = {}
    missing_rows = []
    for name in ("train", "test"):
        restored_by_id = restored[name].set_index("protein_id").sequence.to_dict()
        rows = []
        for protein_id, sequence in restored[name].itertuples(index=False, name=None):
            records = matching.get(sequence, [])
            role = protocol_roles[protein_id] if name == "train" else "test"
            seed_role = seed_roles[protein_id] if name == "train" else "test"
            permitted = role == "fit" and seed_role == "train"
            values = go_cache.get(sequence, {}) if permitted else {}
            row = dict(protein_id=protein_id, sequence_restored=sequence,
                       sequence_sha256=sequence_hash(sequence), protocol_role=role,
                       seed42_partition=seed_role, external_match_status="exact" if records else "unmatched",
                       uniprot_accession_count=len(records),
                       external_source="UniProtKB/Swiss-Prot" if records else "",
                       external_metadata_ambiguous=int(len(records) > 1),
                       external_go_policy="auxiliary_fit_only" if permitted else "withheld_evaluation",
                       **metadata_cache.get(sequence, {key: "" for key in METADATA_FIELDS}),
                       **{key: values.get(key, "") for key in GO_FIELDS})
            rows.append(row)
            if not records:
                missing_rows.append(dict(protein_id=protein_id, dataset=name, protocol_role=role,
                                         sequence_sha256=row["sequence_sha256"],
                                         reason="no_exact_sequence_in_reviewed_snapshot"))
        metadata = pd.DataFrame(rows).set_index("protein_id")
        original_columns = pd.read_csv(cfg[name + "_path"], nrows=0).columns.tolist()
        if original_columns != (["protein_id", "sequence"] + label_columns if name == "train" else ["protein_id", "sequence"]):
            raise ValueError("Unexpected original CSV schema")
        destination = output / (name + "-enriched.csv")
        for chunk_number, chunk in enumerate(pd.read_csv(cfg[name + "_path"], dtype=str, keep_default_na=False, chunksize=12000)):
            if chunk.eq("").any().any():
                missing_original[name] += int(chunk.eq("").sum().sum())
            if name == "train" and not chunk[label_columns].isin(["0", "1"]).all().all():
                raise ValueError("Original labels are not binary")
            decoded_mapping = json.loads(Path(decoded_cfg["mapping_path"]).read_text(encoding="utf-8"))["mapping"]
            if any(decode_sequence(sequence, decoded_mapping) != restored_by_id[protein_id]
                   for protein_id, sequence in chunk[["protein_id", "sequence"]].itertuples(index=False, name=None)):
                raise ValueError("Restored sequence differs from frozen mapping")
            additions = metadata.loc[chunk.protein_id].reset_index(drop=True)
            pd.concat([chunk.reset_index(drop=True), additions], axis=1).to_csv(
                destination, mode="w" if chunk_number == 0 else "a", header=chunk_number == 0, index=False)
            input_counts[name] += len(chunk)
        if input_counts[name] != len(restored[name]):
            raise ValueError("Lost original rows during enrichment")
        stats[name] = dict(rows=input_counts[name], exact_matched_rows=int(metadata.external_match_status.eq("exact").sum()),
                           unmatched_rows=int(metadata.external_match_status.eq("unmatched").sum()),
                           metadata_ambiguous_rows=int(metadata.external_metadata_ambiguous.sum()),
                           eligible_fit_go_rows=int(metadata.go_direct_supported.ne("").sum()),
                           newly_populated_cells={field: int(metadata[field].ne("").sum())
                                                  for field in (*METADATA_FIELDS, *GO_FIELDS)},
                           protocol_roles=metadata.protocol_role.value_counts().to_dict(),
                           seed42_partitions=metadata.seed42_partition.value_counts().to_dict())
        print(json.dumps(dict(written=destination.as_posix(), **stats[name])), flush=True)
    pd.DataFrame(missing_rows).to_csv(output / "unmatched-original-records.csv", index=False)
    reference_metadata = []
    direct_evidence = []
    for sequence, records in matching.items():
        for record in records:
            reference_metadata.append(dict(accession=record["accession"], sequence_sha256=sequence_hash(sequence),
                                           entry_name=record["entry_name"], organism_ids=joined(record["taxonomy"]),
                                           organism_name=record["organism_name"], protein_names=joined(record["protein_names"]),
                                           gene_names=joined(record["gene_names"]), ec_numbers=joined(record["ec_numbers"]),
                                           entry_dates=" | ".join(record["entry_dates"])))
    pd.DataFrame(reference_metadata).to_csv(output / "matched-reference-metadata.csv", index=False)
    auxiliary_rows = []
    for sequence in sorted(eligible_fit_sequences & set(matching)):
        values = go_cache[sequence]
        if values["go_direct_supported"]:
            auxiliary_rows.append(dict(auxiliary_id="fitseq_" + sequence_hash(sequence),
                                       source_kind="original_fit", sequence=sequence,
                                       **aggregate_metadata(matching[sequence]), **values))
    novel_rows = []
    for sequence in sorted(novel, key=sequence_hash):
        values, annotations = go_values(novel[sequence], ontology)
        row = dict(auxiliary_id="external_" + sequence_hash(sequence), source_kind="external_novel",
                   sequence=sequence, **aggregate_metadata(novel[sequence]), **values)
        novel_rows.append(row)
        auxiliary_rows.append(row)
    for row in auxiliary_rows:
        records = matching[row["sequence"]] if row["source_kind"] == "original_fit" else novel[row["sequence"]]
        for record in records:
            for go_id, code, source in set(record["annotations"]):
                if propagate_go([go_id], ontology):
                    direct_evidence.append(dict(auxiliary_id=row["auxiliary_id"], accession=record["accession"],
                                                go_id=go_id, evidence_code=code, evidence_source=source,
                                                evidence_kind="experimental" if code in EXPERIMENTAL_CODES else "curated",
                                                source_kind=row["source_kind"]))
    auxiliary = pd.DataFrame(auxiliary_rows)
    auxiliary.to_csv(output / "auxiliary-go-sequences.csv", index=False)
    pd.DataFrame(novel_rows).to_csv(output / "external-go-sequences.csv", index=False)
    evidence = pd.DataFrame(direct_evidence).drop_duplicates().sort_values(["auxiliary_id", "accession", "go_id", "evidence_code"])
    evidence.to_csv(output / "go-direct-evidence.csv", index=False)
    fit_members = restored["train"].loc[restored["train"].protein_id.map(protocol_roles).eq("fit")
                                        & restored["train"].protein_id.map(seed_roles).eq("train")].copy()
    fit_members["auxiliary_id"] = fit_members.sequence.map(lambda sequence: "fitseq_" + sequence_hash(sequence))
    fit_members.loc[fit_members.auxiliary_id.isin(auxiliary.auxiliary_id), ["protein_id", "auxiliary_id"]].to_csv(
        output / "original-fit-go-members.csv", index=False)
    go_ids = sorted({identifier for row in auxiliary_rows for identifier in row["go_ancestors_supported"].split(";") if identifier})
    positions = {identifier: index for index, identifier in enumerate(go_ids)}
    indices, indptr = [], [0]
    for row in auxiliary_rows:
        indices.extend(sorted(positions[identifier] for identifier in row["go_ancestors_supported"].split(";") if identifier))
        indptr.append(len(indices))
    matrix = sparse.csr_matrix((np.ones(len(indices), np.uint8), np.array(indices, np.int32), np.array(indptr, np.int64)),
                               shape=(len(auxiliary), len(go_ids)))
    sparse.save_npz(output / "auxiliary-go-positive-known.npz", matrix)
    np.savez_compressed(output / "external-anonymous-labels-unknown.npz",
                        auxiliary_ids=np.array([row["auxiliary_id"] for row in novel_rows], dtype=str),
                        label_columns=np.array(label_columns, dtype=str),
                        targets=np.zeros((len(novel_rows), 500), np.uint8),
                        known=np.zeros((len(novel_rows), 500), np.uint8))
    descriptions = read_go_descriptions(cfg["ontology_path"])
    pd.DataFrame([dict(column=index, go_id=identifier, name=descriptions[identifier]["name"],
                       namespace=descriptions[identifier]["namespace"])
                  for index, identifier in enumerate(go_ids)]).to_csv(output / "go-vocabulary.csv", index=False)
    if set(novel) & original_sequences:
        raise ValueError("External novel samples overlap original train/test")
    after = {path: file_sha256(path) for path in before}
    if before != after:
        raise ValueError("An input changed while preparing enrichment")
    summary = dict(experiment_id=cfg["experiment_id"], input_source_hashes=before,
                   config_sha256=file_sha256(cfg_path), preparation_source_sha256=file_sha256(__file__),
                   original_csv_modified=False, original_label_cells_modified=0,
                   confirmed_anonymous_label_additions=0, model_training_started=False,
                   original_missing_cells=dict(missing_original), original_datasets=stats,
                   reviewed_records_scanned=total, reviewed_records_with_supported_go=supported,
                   live_official_sample_verified=dict(accession=live_accession, exact_sequence_equal=True,
                                                      entry_version=live["entryAudit"]["entryVersion"]),
                   exact_match_pairs_independently_equal_to_reviewed_tsv=True,
                   external_snapshot=external_receipt, ontology_release="2026-07-26",
                   auxiliary_go=dict(rows=len(auxiliary), original_fit_unique_sequences=int(auxiliary.source_kind.eq("original_fit").sum()),
                                     external_novel_unique_sequences=len(novel_rows), go_terms=len(go_ids),
                                     positive_known_cells=int(matrix.nnz), direct_evidence_rows=len(evidence),
                                     allowed_evidence_codes=sorted(REFERENCE_EVIDENCE_CODES),
                                     absent_annotations_are_unknown=True,
                                     zero_entries_are_not_supervised_negatives=True,
                                     novel_exact_disjoint_from_all_original=True,
                                     homology_disjoint_not_claimed=True,
                                     anonymous_known_cells=0,
                                     original_go_role_policy="intersection_of_protocol_fit_and_seed42_train"),
                   elapsed_seconds=round(time.monotonic() - started, 3))
    summary["output_hashes"] = {path.name: file_sha256(path) for path in output.iterdir() if path.is_file()}
    (output / "receipt.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    Path("artifacts/metrics", cfg["experiment_id"] + "-summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    arguments = parser.parse_args()
    prepare(arguments.config)
