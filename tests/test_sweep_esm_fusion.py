import numpy as np
import pandas as pd
import pytest

from src.sweep_esm_fusion import (
    align_score_sources,
    assert_fusion_outputs_absent,
    crossfit_label_policies,
    rank_policy_results,
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
