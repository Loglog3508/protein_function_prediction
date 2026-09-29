import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.sweep_esm_fusion import (
    align_score_sources,
    assert_fusion_outputs_absent,
    crossfit_label_policies,
    rank_policy_results,
    run_fusion,
    validate_fusion_config,
)


def _payload(ids, labels, scores):
    return {
        "protein_ids": np.asarray(ids, dtype=str),
        "label_columns": np.asarray(labels, dtype=str),
        "scores": np.asarray(scores, dtype=float),
    }


def test_mismatched_ids_or_labels_are_rejected():
    reference = _payload(["p1", "p2"], ["a", "b"], [[0.1, 0.2], [0.3, 0.4]])
    mismatched = _payload(["p1", "p3"], ["a", "b"], [[0.1, 0.2], [0.3, 0.4]])
    with pytest.raises(ValueError, match="IDs or labels"):
        align_score_sources(reference, {"esm": mismatched})


def test_alignment_reorders_rows_and_labels_by_exact_names():
    reference = _payload(["p1", "p2"], ["a", "b"], [[0.1, 0.2], [0.3, 0.4]])
    source = _payload(["p2", "p1"], ["b", "a"], [[0.8, 0.7], [0.6, 0.5]])
    aligned = align_score_sources(reference, {"esm": source})
    np.testing.assert_allclose(aligned["esm"], [[0.5, 0.6], [0.7, 0.8]])


def test_production_candidates_include_rollback_and_enforce_auc_gate():
    rows = pd.DataFrame(
        [
            {
                "candidate": "rollback",
                "seed": 17,
                "macro_f1": 0.32,
                "continuous_macro_auc": 0.8280390455031759,
            },
            {
                "candidate": "esm+sgd",
                "seed": 17,
                "macro_f1": 0.51,
                "continuous_macro_auc": 0.82,
            },
        ]
    )
    ranked = rank_policy_results(rows, minimum_auc=0.8280390455031759)
    assert "rollback" in set(ranked.candidate)
    assert not ranked.loc[ranked.continuous_macro_auc < 0.8280390455031759, "eligible"].any()
    assert ranked.iloc[0].candidate == "rollback"


def test_crossfit_uses_only_selection_half_for_threshold_and_policy():
    # Source 0 is useful only on the evaluation half; selecting it in-sample
    # would produce a false perfect score.
    target = np.array([[1], [1], [0], [0], [1], [1], [0], [0]], dtype=np.uint8)
    sources = {
        "sgd": np.array([[0.9], [0.8], [0.2], [0.1], [0.1], [0.2], [0.8], [0.9]]),
        "esm": np.array([[0.8], [0.7], [0.3], [0.2], [0.9], [0.8], [0.2], [0.1]]),
    }
    config = {"thresholds": [0.5], "min_support": 1, "independent_min_support": 99}
    seed_rows, _ = crossfit_label_policies(target, sources, [17], config)
    assert len(seed_rows) >= 2 * 5
    assert set(seed_rows["evaluation_fold"]) == {0, 1}
    assert seed_rows["selection_fold"].ne(seed_rows["evaluation_fold"]).all()
    # At least one held-out fold exposes the source conflict; in-sample scoring
    # would report a perfect score for every source on this fixture.
    assert float(seed_rows["macro_f1"].min()) < 1.0


def test_support_strata_and_zero_weight_new_model_are_reported():
    target = np.array([[1, 0], [0, 1], [1, 0], [0, 0], [1, 0], [0, 1]], dtype=np.uint8)
    sources = {
        "sgd": np.array([[.9, .1], [.1, .8], [.8, .2], [.2, .2], [.7, .1], [.1, .7]]),
        "esm": np.full((6, 2), .5),
        "homology": np.full((6, 2), .4),
    }
    config = {
        "thresholds": [0.5],
        "min_support": 1,
        "independent_min_support": 3,
        "stability_tolerance": 0.0,
        "new_model": "esm",
    }
    seed_rows, policies = crossfit_label_policies(target, sources, [17, 31], config)
    assert "esm_zero_weight" in set(seed_rows.candidate)
    assert any(name.startswith("sgd+") for name in seed_rows.candidate)
    assert "support_stratum" in policies.columns
    assert (policies["policy_scope"] == "shared").any()


def test_output_collision_is_rejected_before_writes(tmp_path):
    existing = tmp_path / "leaderboard.csv"
    existing.write_text("old", encoding="utf-8")
    with pytest.raises(FileExistsError, match="new experiment ID"):
        assert_fusion_outputs_absent([tmp_path / "run", existing])


def _cli_fixture(tmp_path, *, duplicate_score_id=False, duplicate_label=False):
    ids = [f"p{i}" for i in range(8)]
    labels = ["a", "b"]
    target = np.array([[1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1]])
    scores = np.where(target, 0.8, 0.2)
    data = tmp_path / "train.csv"
    # Input order and label-column order deliberately differ from score order.
    pd.DataFrame({"protein_id": ids[::-1], "b": target[::-1, 1], "a": target[::-1, 0]}).to_csv(data, index=False)
    split_dir = tmp_path / "artifacts" / "metrics" / "splits" / "seed42"
    split_dir.mkdir(parents=True)
    pd.DataFrame({"protein_id": ids}).to_csv(split_dir / "validation_ids.csv", index=False)
    score_ids = ids.copy()
    if duplicate_score_id:
        score_ids[-1] = score_ids[0]
    score_labels = labels if not duplicate_label else ["a", "a"]
    paths = {}
    for name in ("esm", "sgd", "rollback_auc", "rollback_homology"):
        path = tmp_path / f"{name}.npz"
        np.savez_compressed(path, validation_scores=scores, validation_ids=score_ids, label_columns=score_labels)
        paths[name] = path.name
    config = {
        "experiment_id": "EXP-TEST-083",
        "seeds": [17, 31, 42, 73, 101],
        "minimum_auc": 0.5,
        "diagnostic_only": True,
        "fixture_mode": True,
        "data": {"train_path": "train.csv"},
        "split": {"validation_ids": "artifacts/metrics/splits/seed42/validation_ids.csv"},
        "sources": paths,
        "new_model": "esm",
        "zero_weight_source": "rollback_auc",
        "thresholds": [0.5],
        "output_dir": "run",
        "metrics_prefix": "metrics/fusion",
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def test_cli_loads_targets_from_train_csv_and_writes_thresholds(tmp_path):
    config_path = _cli_fixture(tmp_path)
    summary_path = run_fusion(config_path, project_root=tmp_path)
    assert summary_path.exists()
    leaderboard = pd.read_csv(tmp_path / "metrics" / "fusion-leaderboard.csv")
    assert {"rollback_auc", "rollback_homology", "esm_zero_weight"}.issubset(set(leaderboard.candidate))
    thresholds = pd.read_csv(tmp_path / "metrics" / "fusion-thresholds.csv")
    assert {"candidate", "seed", "evaluation_fold", "label", "threshold"}.issubset(thresholds.columns)
    assert len(thresholds) > 0
    assert set(thresholds.label) == {"a", "b"}
    assert (thresholds.threshold == 0.5).all()


def test_cli_rejects_duplicate_score_ids_and_labels(tmp_path):
    for option, expected in (("duplicate_score_id", "unique"), ("duplicate_label", "unique")):
        case = tmp_path / option
        case.mkdir()
        config_path = _cli_fixture(case, **{option: True})
        with pytest.raises(ValueError, match=expected):
            run_fusion(config_path, project_root=case)
        assert not (case / "run").exists()


def test_cli_rejects_duplicate_or_mismatched_validation_ids(tmp_path):
    config_path = _cli_fixture(tmp_path)
    path = tmp_path / "artifacts" / "metrics" / "splits" / "seed42" / "validation_ids.csv"
    ids = pd.read_csv(path)
    ids.loc[7, "protein_id"] = ids.loc[0, "protein_id"]
    ids.to_csv(path, index=False)
    with pytest.raises(ValueError, match="unique"):
        run_fusion(config_path, project_root=tmp_path)
    ids.loc[7, "protein_id"] = "absent"
    ids.to_csv(path, index=False)
    with pytest.raises(ValueError, match="IDs"):
        run_fusion(config_path, project_root=tmp_path)


def test_cli_rejects_permuted_canonical_ids_and_mismatched_source_labels(tmp_path):
    config_path = _cli_fixture(tmp_path)
    path = tmp_path / "artifacts" / "metrics" / "splits" / "seed42" / "validation_ids.csv"
    ids = pd.read_csv(path)
    ids.iloc[[0, 1]] = ids.iloc[[1, 0]].to_numpy()
    ids.to_csv(path, index=False)
    with pytest.raises(ValueError, match="order"):
        run_fusion(config_path, project_root=tmp_path)
    ids.iloc[[0, 1]] = ids.iloc[[1, 0]].to_numpy()
    ids.to_csv(path, index=False)
    source = tmp_path / "sgd.npz"
    with np.load(source, allow_pickle=False) as saved:
        scores = saved["validation_scores"]
        score_ids = saved["validation_ids"]
    np.savez_compressed(source, validation_scores=scores, validation_ids=score_ids, label_columns=["a", "other"])
    with pytest.raises(ValueError, match="IDs or labels"):
        run_fusion(config_path, project_root=tmp_path)


def test_cli_output_collision_preserves_existing_files(tmp_path):
    config_path = _cli_fixture(tmp_path)
    run_fusion(config_path, project_root=tmp_path)
    artifact = tmp_path / "metrics" / "fusion-thresholds.csv"
    original = artifact.read_bytes()
    with pytest.raises(FileExistsError, match="new experiment ID"):
        run_fusion(config_path, project_root=tmp_path)
    assert artifact.read_bytes() == original


def test_candidate_auc_uses_only_evaluation_rows():
    target = np.array([[1], [0], [1], [0], [1], [0], [1], [0], [1], [0], [1], [0]], dtype=np.uint8)
    scores = np.array([[.9], [.1], [.8], [.2], [.7], [.3], [.6], [.4], [.55], [.45], [.51], [.49]])
    config = {"thresholds": [0.5], "new_model": "esm", "zero_weight_source": "sgd"}
    first, _ = crossfit_label_policies(target, {"sgd": scores, "esm": scores}, [17], config)
    selection = np.array_split(np.random.default_rng(17).permutation(len(target)), 2)[0]
    changed = target.copy()
    changed[selection] = 1 - changed[selection]
    second, _ = crossfit_label_policies(changed, {"sgd": scores, "esm": scores}, [17], config)
    a = first.query("candidate == 'sgd' and evaluation_fold == 0").iloc[0].continuous_macro_auc
    b = second.query("candidate == 'sgd' and evaluation_fold == 0").iloc[0].continuous_macro_auc
    assert a == pytest.approx(b)


def test_unstable_independent_label_falls_back_to_shared_policy():
    target = np.array([[1, 1], [0, 1], [1, 0], [0, 0]] * 5, dtype=np.uint8)
    scores = np.array([[.9, .4], [.1, .9], [.7, .2], [.3, .8]] * 5)
    config = {
        "thresholds": [0.3, 0.5, 0.7],
        "new_model": "esm",
        "independent_label_policies": True,
        "independent_min_support": 2,
        "stability_tolerance": 0.0,
        "stability_seeds": [17, 31, 42],
    }
    _, policies = crossfit_label_policies(target, {"esm": scores}, [17], config)
    assert "stability_passed" in policies.columns
    unstable = policies.loc[~policies.stability_passed]
    assert not unstable.empty
    assert (unstable.policy_scope == "shared").all()


def test_stable_independent_label_can_use_its_own_threshold():
    target = np.tile(np.array([[1, 1], [0, 0]], dtype=np.uint8), (20, 1))
    scores = np.tile(np.array([[.9, .4], [.6, .1]]), (20, 1))
    config = {
        "thresholds": [0.3, 0.7],
        "independent_label_policies": True,
        "independent_min_support": 5,
        "stability_tolerance": 0.5,
        "stability_seeds": [17, 31, 42],
    }
    _, policies = crossfit_label_policies(target, {"sgd": scores}, [17], config)
    label_zero = policies.query("candidate == 'sgd' and label_index == 0")
    assert (label_zero.policy_scope == "independent").all()
    np.testing.assert_allclose(label_zero.threshold.to_numpy(), [0.7, 0.7])


def test_cli_does_not_select_when_every_candidate_fails_auc_gate(tmp_path):
    config_path = _cli_fixture(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["minimum_auc"] = 0.8
    config_path.write_text(json.dumps(config), encoding="utf-8")
    for name in config["sources"]:
        path = tmp_path / config["sources"][name]
        with np.load(path, allow_pickle=False) as saved:
            ids = saved["validation_ids"]
            labels = saved["label_columns"]
        np.savez_compressed(path, validation_scores=np.full((8, 2), .5), validation_ids=ids, label_columns=labels)
    summary_path = run_fusion(config_path, project_root=tmp_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["selected"] is None
    assert not pd.read_csv(tmp_path / "metrics" / "fusion-leaderboard.csv").eligible.any()


def test_production_config_binds_esm_and_rollback_to_iteration4_tail():
    root = Path(__file__).resolve().parents[1]
    fusion = json.loads((root / "configs" / "iteration6_esm_fusion.json").read_text(encoding="utf-8"))
    labelwise = json.loads((root / "configs" / "iteration6_esm2_labelwise_full.json").read_text(encoding="utf-8"))
    expected = "artifacts/metrics/splits/iteration4_tail/validation_ids.csv"
    assert fusion["split"]["validation_ids"] == expected
    assert labelwise["split"]["validation_ids"] == expected
    assert labelwise["split"]["kind"] == "iteration4_tail"
    assert fusion["sources"]["esm"] == labelwise["output_dir"] + "/scores.npz"
    assert "iteration4-gpu-oof-fixed8-f1-auc/best-scores.npz" in fusion["sources"]["rollback_auc"]
    assert len(pd.read_csv(root / expected)) == 1062


def test_production_fusion_rejects_seed42_validation_cohort():
    config = {
        "sources": {"esm": "esm.npz", "rollback_auc": "auc.npz", "rollback_homology": "homology.npz"},
        "seeds": [17, 31, 42, 73, 101],
        "split": {"kind": "seed42", "validation_ids": "artifacts/metrics/splits/seed42/validation_ids.csv"},
    }
    with pytest.raises(ValueError, match="iteration4_tail"):
        validate_fusion_config(config)
