"""Unsplit official rows plus external GO supervision with explicit unknown masks."""

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from .data import label_columns
from .external_go_data import REFERENCE_EVIDENCE_CODES, propagate_go


EXPERIMENTAL_CODES = REFERENCE_EVIDENCE_CODES - {"TAS", "IC"}


def sequence_digest(sequence):
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def assemble_raw_training_dataset(training, test, mapping, records, ontology, output_dir):
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError("Use a new raw training dataset directory")
    labels = label_columns(training.columns)
    if (training.columns.tolist() != ["protein_id", "sequence", *labels]
            or not labels or test.columns.tolist() != ["protein_id", "sequence"]
            or training.isna().any().any() or test.isna().any().any()
            or not training.protein_id.is_unique or not test.protein_id.is_unique
            or not training[labels].isin([0, 1]).all().all()
            or set(training.protein_id) & set(test.protein_id)
            or len(set(mapping.values())) != len(mapping)):
        raise ValueError("Complete original schema, binary labels, unique IDs and residue mapping required")
    translator = str.maketrans(mapping)
    inverse = str.maketrans({value: key for key, value in mapping.items()})
    original_sequences = training.sequence.str.translate(translator).tolist()
    original_set = set(original_sequences)
    test_set = set(test.sequence.str.translate(translator))
    if original_set & test_set:
        raise ValueError("Official train/test complete sequences overlap")
    original_annotations = defaultdict(lambda: {"accessions": set(), "annotations": set()})
    external_annotations = defaultdict(lambda: {"accessions": set(), "annotations": set()})
    total_records = supported_records = excluded_test_records = 0
    for record in records:
        total_records += 1
        sequence = record["sequence"]
        accepted = {(*annotation, record["accession"]) for annotation in record["annotations"]
                    if annotation[1] in REFERENCE_EVIDENCE_CODES
                    and propagate_go([annotation[0]], ontology)}
        if sequence in test_set:
            excluded_test_records += 1
            continue
        if not accepted:
            continue
        supported_records += 1
        entry = original_annotations[sequence] if sequence in original_set else external_annotations[sequence]
        entry["accessions"].add(record["accession"])
        entry["annotations"].update(accepted)
        if total_records % 50000 == 0:
            print(json.dumps(dict(scanned=total_records, original_go_sequences=len(original_annotations),
                                  external_go_sequences=len(external_annotations))), flush=True)
    external_sequences = sorted(external_annotations, key=sequence_digest)
    external_ids = ["UP_EXT_" + sequence_digest(sequence)[:24] for sequence in external_sequences]
    ids = training.protein_id.astype(str).tolist() + external_ids
    sequences = original_sequences + external_sequences
    if len(set(ids)) != len(ids) or set(external_sequences) & (original_set | test_set):
        raise ValueError("External IDs or sequences collide with original data")
    origins = ["official"] * len(training) + ["uniprot_external"] * len(external_sequences)
    annotations = []
    evidence = []
    for origin, entries in (("official", original_annotations), ("uniprot_external", external_annotations)):
        for sequence, entry in entries.items():
            for accession in sorted(entry["accessions"]):
                evidence.append(dict(sequence_sha256=sequence_digest(sequence), source_kind=origin,
                                     accession=accession))
    annotation_values = {}
    for entries in (original_annotations, external_annotations):
        for sequence, entry in entries.items():
            direct = {annotation[0] for annotation in entry["annotations"]}
            experimental = {annotation[0] for annotation in entry["annotations"] if annotation[1] in EXPERIMENTAL_CODES}
            curated = {annotation[0] for annotation in entry["annotations"] if annotation[1] in {"TAS", "IC"}}
            annotation_values[sequence] = dict(
                source_accessions=";".join(sorted(entry["accessions"])),
                go_direct=";".join(sorted(direct)), go_propagated=";".join(sorted(propagate_go(direct, ontology))),
                go_experimental=";".join(sorted(experimental)), go_curated=";".join(sorted(curated)))
    for protein_id, sequence, origin in zip(ids, sequences, origins, strict=True):
        annotations.append(dict(protein_id=protein_id, source_kind=origin,
                                sequence_sha256=sequence_digest(sequence),
                                **annotation_values.get(sequence, dict(source_accessions="", go_direct="",
                                                                      go_propagated="", go_experimental="", go_curated=""))))
    annotations_frame = pd.DataFrame(annotations)
    vocabulary = sorted({identifier for row in annotations for identifier in row["go_propagated"].split(";") if identifier})
    positions = {identifier: index for index, identifier in enumerate(vocabulary)}
    indices, indptr = [], [0]
    for row in annotations:
        indices.extend(sorted(positions[identifier] for identifier in row["go_propagated"].split(";") if identifier))
        indptr.append(len(indices))
    go_matrix = sparse.csr_matrix((np.ones(len(indices), np.uint8), np.array(indices, np.int32),
                                  np.array(indptr, np.int64)), shape=(len(ids), len(vocabulary)))
    targets = np.zeros((len(ids), len(labels)), np.uint8)
    targets[:len(training)] = training[labels].to_numpy(np.uint8)
    known = np.zeros(targets.shape, np.uint8)
    known[:len(training)] = 1
    output.mkdir(parents=True)
    training.to_csv(output / "train.csv", index=False)
    for start in range(0, len(external_sequences), 1000):
        end = min(start + 1000, len(external_sequences))
        extension = pd.DataFrame({"protein_id": external_ids[start:end],
                                  "sequence": [sequence.translate(inverse) for sequence in external_sequences[start:end]],
                                  **{label: [""] * (end - start) for label in labels}})
        extension.to_csv(output / "train.csv", mode="a", header=False, index=False)
    pd.DataFrame(dict(protein_id=ids, sequence=sequences)).to_csv(output / "canonical-sequences.csv", index=False)
    annotations_frame.to_csv(output / "go-annotations.csv", index=False)
    pd.DataFrame(evidence, columns=["sequence_sha256", "source_kind", "accession"]).to_csv(
        output / "sequence-accessions.csv", index=False)
    with (output / "go-direct-evidence.csv").open("w", encoding="utf-8", newline="") as handle:
        import csv
        writer = csv.writer(handle)
        writer.writerow(["sequence_sha256", "source_kind", "accession", "go_id", "evidence_code", "evidence_source"])
        for origin, entries in (("official", original_annotations), ("uniprot_external", external_annotations)):
            for sequence, entry in entries.items():
                for go_id, code, source, accession in sorted(entry["annotations"]):
                    writer.writerow([sequence_digest(sequence), origin, accession, go_id, code, source])
    np.savez_compressed(output / "main-labels.npz", protein_ids=np.array(ids, dtype=str),
                        label_columns=np.array(labels, dtype=str), targets=targets, known=known)
    sparse.save_npz(output / "go-positive-known.npz", go_matrix)
    pd.DataFrame(dict(column=np.arange(len(vocabulary)), go_id=vocabulary)).to_csv(output / "go-vocabulary.csv", index=False)
    summary = dict(official_rows=len(training), external_rows=len(external_sequences), total_rows=len(ids),
                   main_label_count=len(labels), original_main_known_cells=len(training) * len(labels),
                   external_main_known_cells=0,
                   official_rows_with_go=int(annotations_frame.iloc[:len(training)].go_direct.ne("").sum()),
                   official_rows_without_go=int(annotations_frame.iloc[:len(training)].go_direct.eq("").sum()),
                   go_terms=len(vocabulary), go_positive_known_cells=int(go_matrix.nnz),
                   source_records_scanned=total_records, supported_non_test_records=supported_records,
                   excluded_official_test_records=excluded_test_records,
                   original_train_rows_filtered=0, original_row_order_preserved=True,
                   original_labels_modified=False, external_unknown_csv_value="empty",
                   external_main_targets_are_placeholders=True, go_absence_means_unknown=True,
                   old_enriched_dataset_used=False, prior_protocol_or_label_groups_used=False,
                   dataset_split_performed=False, training_started=False,
                   external_sequences_exact_disjoint=True, homology_disjoint_not_claimed=True,
                   label_mapping_inferred=False, evidence_codes=sorted(REFERENCE_EVIDENCE_CODES))
    (output / "dataset-schema.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


@dataclass
class RawTrainingDataset:
    protein_ids: np.ndarray
    sequences: np.ndarray
    main_targets: np.ndarray
    main_known: np.ndarray
    label_columns: np.ndarray
    go_positive_known: sparse.csr_matrix
    go_ids: np.ndarray
    source_kind: np.ndarray

    def batch(self, row_indices):
        indices = np.asarray(row_indices)
        if (indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer)
                or np.any(indices < 0) or np.any(indices >= len(self.protein_ids))):
            raise ValueError("Valid integer training row indices required")
        return dict(protein_ids=self.protein_ids[indices], sequences=self.sequences[indices],
                    main_targets=self.main_targets[indices], main_known=self.main_known[indices],
                    go_positive_known=self.go_positive_known[indices], source_kind=self.source_kind[indices])


def load_raw_training_dataset(directory):
    root = Path(directory)
    schema = json.loads((root / "dataset-schema.json").read_text(encoding="utf-8"))
    canonical = pd.read_csv(root / "canonical-sequences.csv", dtype=str, keep_default_na=False)
    annotations = pd.read_csv(root / "go-annotations.csv", usecols=["protein_id", "source_kind"], dtype=str)
    csv_ids = pd.read_csv(root / "train.csv", usecols=["protein_id"], dtype=str).protein_id.to_numpy()
    header = pd.read_csv(root / "train.csv", nrows=0).columns.tolist()
    with np.load(root / "main-labels.npz", allow_pickle=False) as saved:
        ids, labels = saved["protein_ids"], saved["label_columns"]
        targets, known = saved["targets"], saved["known"]
    vocabulary = pd.read_csv(root / "go-vocabulary.csv", dtype={"go_id": str})
    go_matrix = sparse.load_npz(root / "go-positive-known.npz").tocsr()
    if (len(ids) != schema["total_rows"] or len(set(ids)) != len(ids)
            or not np.array_equal(ids, canonical.protein_id.to_numpy())
            or not np.array_equal(ids, annotations.protein_id.to_numpy())
            or not np.array_equal(ids, csv_ids)
            or header != ["protein_id", "sequence", *labels.tolist()]):
        raise ValueError("Training IDs, row order or schema differ")
    if (targets.shape != (len(ids), schema["main_label_count"]) or known.shape != targets.shape
            or not np.isin(targets, [0, 1]).all() or not np.isin(known, [0, 1]).all()):
        raise ValueError("Aligned binary target and known masks required")
    official_rows = schema["official_rows"]
    if (not known[:official_rows].all() or known[official_rows:].any() or targets[official_rows:].any()
            or not annotations.source_kind.iloc[:official_rows].eq("official").all()
            or not annotations.source_kind.iloc[official_rows:].eq("uniprot_external").all()):
        raise ValueError("Official labels must be known and external main labels must remain unknown")
    if (go_matrix.shape != (len(ids), schema["go_terms"]) or not go_matrix.has_canonical_format
            or not np.all(go_matrix.data == 1) or go_matrix.nnz != schema["go_positive_known_cells"]
            or vocabulary.column.tolist() != list(range(schema["go_terms"]))
            or not vocabulary.go_id.is_unique or canonical.sequence.eq("").any()):
        raise ValueError("Canonical sequences and binary positive-only GO data required")
    return RawTrainingDataset(ids, canonical.sequence.to_numpy(dtype=object), targets, known.astype(bool),
                              labels, go_matrix, vocabulary.go_id.to_numpy(), annotations.source_kind.to_numpy())
