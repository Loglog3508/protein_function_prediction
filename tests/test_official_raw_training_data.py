import json

import numpy as np
import pandas as pd
import pytest
from scipy import sparse


def raw_inputs():
    training = pd.DataFrame({
        "protein_id": ["P0", "P1", "P2"], "sequence": ["AAA", "GGG", "AAA"],
        "label_0": [1, 0, 0], "label_1": [0, 1, 1], "label_2": [1, 1, 0],
    })
    test = pd.DataFrame({"protein_id": ["T0"], "sequence": ["TTT"]})
    mapping = {"A": "G", "G": "A"}
    ontology = {"terms": {"GO:1": {"parents": set(), "obsolete": False},
                           "GO:2": {"parents": {"GO:1"}, "obsolete": False},
                           "GO:3": {"parents": set(), "obsolete": True}},
                "aliases": {}, "closure": {}}
    records = [
        {"accession": "U0", "sequence": "GGG", "annotations": [("GO:2", "IDA", "Test")]},
        {"accession": "U1", "sequence": "CCC", "annotations": [("GO:2", "IDA", "Test")]},
        {"accession": "U2", "sequence": "CCC", "annotations": [("GO:1", "TAS", "Test")]},
        {"accession": "U3", "sequence": "TTT", "annotations": [("GO:2", "IDA", "Test")]},
        {"accession": "U4", "sequence": "VVV", "annotations": [("GO:2", "IEA", "Test")]},
        {"accession": "U5", "sequence": "LLL", "annotations": [("GO:3", "IDA", "Test")]},
    ]
    return training, test, mapping, ontology, records


def test_new_training_set_preserves_all_official_rows_including_sequence_label_conflicts(tmp_path):
    from src.official_raw_training_data import assemble_raw_training_dataset
    training, test, mapping, ontology, records = raw_inputs()
    destination = tmp_path / "dataset"
    assemble_raw_training_dataset(training, test, mapping, records, ontology, destination)
    combined = pd.read_csv(destination / "train.csv", keep_default_na=False, dtype=str)
    assert combined.columns.tolist() == training.columns.tolist()
    pd.testing.assert_frame_equal(combined.iloc[:len(training)].reset_index(drop=True), training.astype(str))
    assert len(combined) == len(training) + 1
    assert combined.loc[len(training), "sequence"] == "CCC"
    assert not any("role" in column or "group" in column for column in combined.columns)


def test_external_anonymous_labels_remain_unknown_and_go_supervision_is_available(tmp_path):
    from src.official_raw_training_data import assemble_raw_training_dataset, load_raw_training_dataset
    training, test, mapping, ontology, records = raw_inputs()
    destination = tmp_path / "dataset"
    assemble_raw_training_dataset(training, test, mapping, records, ontology, destination)
    data = load_raw_training_dataset(destination)
    np.testing.assert_array_equal(data.main_targets[:3], training[["label_0", "label_1", "label_2"]].to_numpy())
    assert data.main_known[:3].all()
    assert not data.main_known[3].any()
    assert not data.main_targets[3].any()
    assert data.go_ids.tolist() == ["GO:1", "GO:2"]
    np.testing.assert_array_equal(data.go_positive_known.toarray(), [[1, 1], [0, 0], [1, 1], [1, 1]])
    assert data.sequences.tolist() == ["GGG", "AAA", "GGG", "CCC"]
    combined = pd.read_csv(destination / "train.csv", keep_default_na=False, dtype=str)
    assert combined.loc[3, ["label_0", "label_1", "label_2"]].eq("").all()


def test_direct_go_evidence_keeps_each_annotation_linked_to_its_actual_accession(tmp_path):
    from src.official_raw_training_data import assemble_raw_training_dataset
    training, test, mapping, ontology, records = raw_inputs()
    destination = tmp_path / "dataset"
    assemble_raw_training_dataset(training, test, mapping, records, ontology, destination)
    evidence = pd.read_csv(destination / "go-direct-evidence.csv", dtype=str)
    external = evidence.loc[evidence.source_kind.eq("uniprot_external")]
    assert "accession" in external.columns
    assert set(zip(external.accession, external.go_id, external.evidence_code)) == {
        ("U1", "GO:2", "IDA"), ("U2", "GO:1", "TAS")}


def test_batch_returns_requested_row_order_and_masks(tmp_path):
    from src.official_raw_training_data import assemble_raw_training_dataset, load_raw_training_dataset
    training, test, mapping, ontology, records = raw_inputs()
    assemble_raw_training_dataset(training, test, mapping, records, ontology, tmp_path / "dataset")
    data = load_raw_training_dataset(tmp_path / "dataset")
    batch = data.batch([3, 0])
    assert batch["sequences"].tolist() == ["CCC", "GGG"]
    assert not batch["main_known"][0].any()
    assert batch["main_known"][1].all()
    assert batch["go_positive_known"].shape == (2, 2)
    np.testing.assert_array_equal(batch["main_targets"][1], [1, 0, 1])


def test_original_test_sequences_are_excluded_and_external_duplicates_are_merged(tmp_path):
    from src.official_raw_training_data import assemble_raw_training_dataset
    training, test, mapping, ontology, records = raw_inputs()
    summary = assemble_raw_training_dataset(training, test, mapping, records, ontology, tmp_path / "dataset")
    annotations = pd.read_csv(tmp_path / "dataset" / "go-annotations.csv", keep_default_na=False)
    assert summary["external_rows"] == 1
    assert summary["official_rows"] == 3
    assert summary["official_rows_with_go"] == 2
    assert summary["official_rows_without_go"] == 1
    assert annotations.iloc[3].source_accessions == "U1;U2"
    assert not annotations.source_accessions.str.contains("U3").any()
    assert not annotations.source_accessions.str.contains("U4").any()
    assert not annotations.source_accessions.str.contains("U5").any()


def test_external_sequences_use_the_same_residue_encoding_as_official_rows(tmp_path):
    from src.official_raw_training_data import assemble_raw_training_dataset, load_raw_training_dataset
    training, test, mapping, ontology, records = raw_inputs()
    records.append({"accession": "U6", "sequence": "AGG", "annotations": [("GO:2", "IDA", "Test")]})
    assemble_raw_training_dataset(training, test, mapping, records, ontology, tmp_path / "dataset")
    combined = pd.read_csv(tmp_path / "dataset" / "train.csv", dtype=str, keep_default_na=False)
    canonical = pd.read_csv(tmp_path / "dataset" / "canonical-sequences.csv", dtype=str)
    assert combined.sequence.tolist() == [value.translate(str.maketrans({"G": "A", "A": "G"}))
                                          for value in canonical.sequence]
    data = load_raw_training_dataset(tmp_path / "dataset")
    assert set(data.sequences) == {"GGG", "AAA", "CCC", "AGG"}


def test_existing_dataset_is_never_overwritten(tmp_path):
    from src.official_raw_training_data import assemble_raw_training_dataset
    training, test, mapping, ontology, records = raw_inputs()
    destination = tmp_path / "dataset"
    destination.mkdir()
    with pytest.raises(FileExistsError):
        assemble_raw_training_dataset(training, test, mapping, records, ontology, destination)


def test_loader_rejects_reordered_target_ids(tmp_path):
    from src.official_raw_training_data import assemble_raw_training_dataset, load_raw_training_dataset
    training, test, mapping, ontology, records = raw_inputs()
    destination = tmp_path / "dataset"
    assemble_raw_training_dataset(training, test, mapping, records, ontology, destination)
    with np.load(destination / "main-labels.npz", allow_pickle=False) as saved:
        contents = {name: saved[name] for name in saved.files}
    contents["protein_ids"] = contents["protein_ids"][::-1]
    np.savez_compressed(destination / "main-labels.npz", **contents)
    with pytest.raises(ValueError, match="IDs|order"):
        load_raw_training_dataset(destination)


def test_loader_rejects_external_placeholder_zero_as_known_negative(tmp_path):
    from src.official_raw_training_data import assemble_raw_training_dataset, load_raw_training_dataset
    training, test, mapping, ontology, records = raw_inputs()
    destination = tmp_path / "dataset"
    assemble_raw_training_dataset(training, test, mapping, records, ontology, destination)
    with np.load(destination / "main-labels.npz", allow_pickle=False) as saved:
        contents = {name: saved[name] for name in saved.files}
    contents["known"][-1] = 1
    np.savez_compressed(destination / "main-labels.npz", **contents)
    with pytest.raises(ValueError, match="unknown|known"):
        load_raw_training_dataset(destination)
