"""Independently verify all raw rows, role isolation, evidence, and sparse GO bags."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def fields(value):
    return set(value.split(";")) if value else set()


def ontology_graph(path):
    graph, aliases = {}, {}
    current = None
    for text in Path(path).read_text(encoding="utf-8").splitlines():
        line = text.strip()
        if line == "[Term]":
            current = {"parents": set(), "obsolete": False, "aliases": []}
        elif line.startswith("["):
            current = None
        elif current is not None:
            if line.startswith("id: "):
                graph[line[4:]] = current
            elif line.startswith("is_a: "):
                current["parents"].add(line[6:].split()[0])
            elif line.startswith("relationship: part_of "):
                current["parents"].add(line.split()[2])
            elif line.startswith("alt_id: "):
                current["aliases"].append(line[8:])
            elif line == "is_obsolete: true":
                current["obsolete"] = True
    for identifier, term in graph.items():
        if not term["obsolete"]:
            aliases.update({alias: identifier for alias in term["aliases"]})
    return graph, aliases


def verify(config_path):
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    output = Path(cfg["output_dir"])
    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    result_path = output / "independent-verification.json"
    if result_path.exists():
        raise FileExistsError("Do not overwrite a prior verification")
    assert digest(config_path) == receipt["config_sha256"]
    assert digest("scripts/enrich_original_data.py") == receipt["preparation_source_sha256"]
    assert all(digest(path) == checksum for path, checksum in receipt["input_source_hashes"].items())
    assert all(digest(output / name) == checksum for name, checksum in receipt["output_hashes"].items())
    labels = [f"label_{index}" for index in range(500)]
    mapping_cfg = json.loads(Path(cfg["decoded_config"]).read_text(encoding="utf-8"))
    mapping = json.loads(Path(mapping_cfg["mapping_path"]).read_text(encoding="utf-8"))["mapping"]
    translator = str.maketrans(mapping)
    seed_fit = set(pd.read_csv(cfg["seed42_train_path"], dtype=str).protein_id)
    seed_validation = set(pd.read_csv(cfg["seed42_validation_path"], dtype=str).protein_id)
    assert not seed_fit & seed_validation
    expected_roles = {}
    with np.load(cfg["protocol_path"], allow_pickle=False) as protocol:
        base = protocol["training_ids"].astype(str)
        for role in ("fit", "monitor", "calibration", "audit", "quarantine"):
            expected_roles.update({identifier: role for identifier in base[protocol[role + "_indices"]]})
        expected_roles.update({identifier: "legacy_validation" for identifier in protocol["legacy_validation_ids"].astype(str)})
    go_columns = ["go_direct_supported", "go_ancestors_supported", "go_direct_experimental", "go_direct_curated"]
    metadata_columns = ["protein_id", "sequence_restored", "sequence_sha256", "protocol_role", "seed42_partition",
                        "external_match_status", "uniprot_accessions", "uniprot_accession_count", "protein_names",
                        "organism_ids", "gene_names", "external_go_policy", *go_columns]
    enriched, checks = {}, {}
    original_label_cells_checked = 0
    for name in ("train", "test"):
        count = 0
        source_columns = pd.read_csv(cfg[name + "_path"], nrows=0).columns.tolist()
        original_reader = pd.read_csv(cfg[name + "_path"], dtype=str, keep_default_na=False, chunksize=10000)
        enriched_reader = pd.read_csv(output / (name + "-enriched.csv"), dtype=str, keep_default_na=False, chunksize=10000)
        for source, extended in zip(original_reader, enriched_reader, strict=True):
            assert extended.columns[:len(source_columns)].tolist() == source_columns
            assert source.equals(extended[source_columns])
            assert source.sequence.map(lambda sequence: sequence.translate(translator)).tolist() == extended.sequence_restored.tolist()
            assert extended.sequence_restored.map(lambda sequence: hashlib.sha256(sequence.encode("ascii")).hexdigest()).tolist() == extended.sequence_sha256.tolist()
            if name == "train":
                assert source[labels].isin(["0", "1"]).all().all()
                original_label_cells_checked += len(source) * 500
            count += len(source)
        assert count == receipt["original_datasets"][name]["rows"]
        frame = pd.read_csv(output / (name + "-enriched.csv"), usecols=metadata_columns, dtype=str, keep_default_na=False)
        assert frame.protein_id.is_unique
        matched = frame.external_match_status.eq("exact")
        assert matched.equals(frame.uniprot_accessions.ne(""))
        assert frame.uniprot_accession_count.astype(int).tolist() == frame.uniprot_accessions.map(lambda value: len(fields(value))).tolist()
        assert int(matched.sum()) == receipt["original_datasets"][name]["exact_matched_rows"]
        if name == "train":
            assert frame.protein_id.map(expected_roles).equals(frame.protocol_role)
            seed_expected = frame.protein_id.map(lambda identifier: "train" if identifier in seed_fit else "validation")
            assert seed_expected.equals(frame.seed42_partition)
            assert set(frame.protein_id) == seed_fit | seed_validation
            permitted = frame.protocol_role.eq("fit") & frame.seed42_partition.eq("train")
            assert frame.loc[~permitted, go_columns].eq("").all().all()
            assert frame.loc[permitted, "external_go_policy"].eq("auxiliary_fit_only").all()
            assert frame.loc[~permitted, "external_go_policy"].eq("withheld_evaluation").all()
        else:
            assert frame[go_columns].eq("").all().all()
            assert not any(column.startswith("label_") for column in pd.read_csv(output / "test-enriched.csv", nrows=0).columns)
        enriched[name] = frame
        checks[name] = dict(original_rows_and_columns_equal=True, row_order_equal=True, frozen_sequence_translation_equal=True,
                            metadata_match_flags_equal=True, evaluation_go_withheld=True, rows=count)
    original_sequences = set(enriched["train"].sequence_restored) | set(enriched["test"].sequence_restored)
    unmatched = pd.read_csv(output / "unmatched-original-records.csv", dtype=str, keep_default_na=False)
    expected_unmatched = {(name, row.protein_id) for name, frame in enriched.items()
                          for row in frame.loc[frame.external_match_status.eq("unmatched")].itertuples()}
    assert set(zip(unmatched.dataset, unmatched.protein_id)) == expected_unmatched
    assert len(unmatched) == len(expected_unmatched)
    reference = pd.read_csv(output / "matched-reference-metadata.csv", dtype=str, keep_default_na=False)
    assert reference.accession.is_unique
    reference_hash = reference.set_index("accession").sequence_sha256.to_dict()
    for frame in enriched.values():
        for row in frame.itertuples():
            assert all(reference_hash[accession] == row.sequence_sha256 for accession in fields(row.uniprot_accessions))
    novel = pd.read_csv(output / "external-go-sequences.csv", dtype=str, keep_default_na=False)
    auxiliary = pd.read_csv(output / "auxiliary-go-sequences.csv", dtype=str, keep_default_na=False)
    assert novel.sequence.is_unique and novel.auxiliary_id.is_unique
    assert auxiliary.sequence.is_unique and auxiliary.auxiliary_id.is_unique
    assert not set(novel.sequence) & original_sequences
    assert auxiliary.loc[auxiliary.source_kind.eq("external_novel")].reset_index(drop=True).equals(novel)
    fit = enriched["train"].loc[enriched["train"].protocol_role.eq("fit") & enriched["train"].seed42_partition.eq("train")]
    fit_with_go = fit.loc[fit.go_direct_supported.ne("")]
    assert set(auxiliary.loc[auxiliary.source_kind.eq("original_fit"), "sequence"]) == set(fit_with_go.sequence_restored)
    reserved_sequences = set(enriched["train"].loc[~(enriched["train"].protocol_role.eq("fit")
                              & enriched["train"].seed42_partition.eq("train")), "sequence_restored"]) | set(enriched["test"].sequence_restored)
    assert not set(auxiliary.loc[auxiliary.source_kind.eq("original_fit"), "sequence"]) & reserved_sequences
    members = pd.read_csv(output / "original-fit-go-members.csv", dtype=str, keep_default_na=False)
    assert members.protein_id.is_unique
    assert set(members.protein_id) == set(fit_with_go.protein_id)
    assert members.auxiliary_id.tolist() == members.protein_id.map(fit_with_go.set_index("protein_id").sequence_sha256).map(lambda value: "fitseq_" + value).tolist()
    evidence = pd.read_csv(output / "go-direct-evidence.csv", dtype=str, keep_default_na=False)
    allowed = set(receipt["auxiliary_go"]["allowed_evidence_codes"])
    assert set(evidence.evidence_code) <= allowed
    assert not set(evidence.evidence_code) & {"IEA", "ND", "NOT"}
    auxiliary_by_id = auxiliary.set_index("auxiliary_id")
    for auxiliary_id, group in evidence.groupby("auxiliary_id", sort=False):
        assert set(group.go_id) == fields(auxiliary_by_id.loc[auxiliary_id, "go_direct_supported"])
        assert set(group.accession) <= fields(auxiliary_by_id.loc[auxiliary_id, "uniprot_accessions"])
        assert set(group.source_kind) == {auxiliary_by_id.loc[auxiliary_id, "source_kind"]}
    assert set(evidence.auxiliary_id) == set(auxiliary.auxiliary_id)
    vocabulary = pd.read_csv(output / "go-vocabulary.csv", dtype=str, keep_default_na=False)
    assert vocabulary.column.astype(int).tolist() == list(range(len(vocabulary)))
    assert vocabulary.go_id.is_unique
    matrix = sparse.load_npz(output / "auxiliary-go-positive-known.npz").tocsr()
    assert matrix.shape == (len(auxiliary), len(vocabulary))
    assert np.all(matrix.data == 1)
    assert matrix.nnz == receipt["auxiliary_go"]["positive_known_cells"]
    identifiers = vocabulary.go_id.to_numpy()
    graph, aliases = ontology_graph(cfg["ontology_path"])
    closure_cache = {}

    def ancestors(identifier):
        if identifier not in closure_cache:
            pending, seen = [aliases.get(identifier, identifier)], set()
            while pending:
                current = pending.pop()
                current = aliases.get(current, current)
                if current in seen or current not in graph or graph[current]["obsolete"]:
                    continue
                seen.add(current)
                pending.extend(graph[current]["parents"])
            closure_cache[identifier] = seen
        return closure_cache[identifier]

    for index, row in enumerate(auxiliary.itertuples()):
        expected_go = set().union(*(ancestors(identifier) for identifier in fields(row.go_direct_supported)))
        actual_go = set(identifiers[matrix.indices[matrix.indptr[index]:matrix.indptr[index + 1]]])
        assert expected_go == actual_go == fields(row.go_ancestors_supported)
        assert fields(row.go_direct_experimental) | fields(row.go_direct_curated) == fields(row.go_direct_supported)
    with np.load(output / "external-anonymous-labels-unknown.npz", allow_pickle=False) as anonymous:
        assert anonymous["targets"].shape == anonymous["known"].shape == (len(novel), 500)
        assert not anonymous["targets"].any() and not anonymous["known"].any()
        assert anonymous["label_columns"].tolist() == labels
        assert anonymous["auxiliary_ids"].tolist() == novel.auxiliary_id.tolist()
    result = dict(experiment_id=cfg["experiment_id"], passed=True, datasets=checks,
                  all_source_and_output_hashes_equal=True, original_label_cells_compared=original_label_cells_checked,
                  original_label_cells_changed=0, novel_sequence_count=len(novel), auxiliary_sequence_count=len(auxiliary),
                  go_positive_cells_verified=matrix.nnz, direct_evidence_rows_verified=len(evidence),
                  all_go_rows_independently_reconstructed=True, heldout_sequence_overlap=0,
                  anonymous_external_known_cells=0, anonymous_external_targets_not_claimed=True,
                  test_annotations_not_exposed=True, unmatched_records_reconciled=len(unmatched),
                  verification_source_sha256=digest(__file__), receipt_sha256=digest(output / "receipt.json"))
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    Path("artifacts/metrics", cfg["experiment_id"] + "-independent-verification.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    verify(parser.parse_args().config)
