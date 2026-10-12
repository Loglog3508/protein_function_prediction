import gzip

import numpy as np
import pandas as pd
import pytest

from scripts.enrich_original_data import aggregate_metadata, go_values, iter_metadata_records, role_maps


def test_reviewed_parser_preserves_evidence_and_excludes_unknown_and_not(tmp_path):
    path = tmp_path / "reviewed.dat.gz"
    text = "\n".join([
        "ID   TEST_HUMAN Reviewed; 4 AA.", "AC   P00001; Q00001;",
        "DE   RecName: Full=Example enzyme {ECO:0000269|PubMed:1};",
        "DE   EC=1.2.3.4;", "GN   Name=GENE1; Synonyms=OTHER1;",
        "OS   Homo sapiens (Human).", "OX   NCBI_TaxID=9606 {ECO:0000313};",
        "DR   GO; GO:0000001; P:test; IDA:UniProtKB.",
        "DR   GO; GO:0000002; P:test; TAS:UniProtKB.",
        "DR   GO; GO:0000003; P:test; IEA:InterPro.",
        "DR   GO; GO:0000004; P:test; ND:UniProtKB.",
        "DR   GO; GO:0000005; P:test; IDA:UniProtKB; NOT;",
        "SQ   SEQUENCE 4 AA;", "     AC UX", "//", "",
    ])
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(text)
    record, = iter_metadata_records(path)
    assert record["sequence"] == "ACUX"
    assert record["accession"] == "P00001"
    assert record["taxonomy"] == ["9606"]
    assert record["protein_names"] == ["Example enzyme"]
    assert record["ec_numbers"] == ["1.2.3.4"]
    assert set(record["annotations"]) == {("GO:0000001", "IDA", "UniProtKB"),
                                           ("GO:0000002", "TAS", "UniProtKB")}


@pytest.mark.parametrize("ending", ["", "     A\n//\n"])
def test_reviewed_parser_rejects_truncation_or_wrong_length(tmp_path, ending):
    path = tmp_path / "invalid.dat.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("ID   TEST_HUMAN Reviewed; 4 AA.\nAC   P00001;\nSQ   SEQUENCE 4 AA;\n" + ending)
    with pytest.raises(ValueError):
        list(iter_metadata_records(path))


def test_identical_sequence_metadata_keeps_multiple_accessions_and_species():
    first = dict(accession="P00001", entry_name="ONE", protein_names=["Alpha"],
                 gene_names=[], organism_name="Human", taxonomy=["9606"], ec_numbers=[])
    second = dict(accession="P00002", entry_name="TWO", protein_names=["Beta"],
                  gene_names=["GENE"], organism_name="Mouse", taxonomy=["10090"], ec_numbers=[])
    metadata = aggregate_metadata([first, second])
    assert metadata["uniprot_accessions"] == "P00001;P00002"
    assert metadata["organism_ids"] == "10090;9606"
    assert metadata["protein_names"] == "Alpha;Beta"


def test_go_evidence_distinguishes_experimental_curated_and_obsolete():
    ontology = dict(terms={
        "GO:0000001": dict(parents={"GO:0000003"}, obsolete=False),
        "GO:0000002": dict(parents=set(), obsolete=False),
        "GO:0000003": dict(parents=set(), obsolete=False),
        "GO:0000004": dict(parents=set(), obsolete=True)}, aliases={}, closure={})
    records = [dict(annotations=[("GO:0000001", "IDA", "UniProtKB"),
                                  ("GO:0000002", "TAS", "UniProtKB"),
                                  ("GO:0000004", "IDA", "UniProtKB")])]
    values, annotations = go_values(records, ontology)
    assert values["go_direct_supported"] == "GO:0000001;GO:0000002"
    assert values["go_ancestors_supported"] == "GO:0000001;GO:0000002;GO:0000003"
    assert values["go_direct_experimental"] == "GO:0000001"
    assert values["go_direct_curated"] == "GO:0000002"
    assert len(annotations) == 2


def test_roles_keep_seed42_validation_and_legacy_evaluation(tmp_path):
    pd.DataFrame({"protein_id": ["A", "B", "C", "D", "E"]}).to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame({"protein_id": ["F"]}).to_csv(tmp_path / "validation.csv", index=False)
    np.savez(tmp_path / "protocol.npz", training_ids=np.array(["A", "B", "C", "D", "E"]),
             legacy_validation_ids=np.array(["F"]), fit_indices=[0], monitor_indices=[1],
             calibration_indices=[2], audit_indices=[3], quarantine_indices=[4])
    config = dict(seed42_train_path=tmp_path / "train.csv",
                  seed42_validation_path=tmp_path / "validation.csv", protocol_path=tmp_path / "protocol.npz")
    seed, roles = role_maps(config, ["A", "B", "C", "D", "E", "F"])
    assert seed["F"] == "validation"
    assert roles == {"A": "fit", "B": "monitor", "C": "calibration", "D": "audit",
                     "E": "quarantine", "F": "legacy_validation"}
