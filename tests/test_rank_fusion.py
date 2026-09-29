import numpy as np

from src.rank_fusion import (
    crossfit_rank_fusion,
    fit_rank_fusion_thresholds,
)


def test_rank_fusion_thresholds_use_exact_boundaries_and_shrinkage():
    target = np.array([[1], [0], [0], [0]], dtype=np.uint8)
    scores = np.array([[0.61], [0.60], [0.21], [0.20]], dtype=np.float32)

    thresholds, global_threshold, diagnostics = fit_rank_fusion_thresholds(
        target, scores, shrinkage=0.0
    )

    np.testing.assert_allclose(thresholds, [0.61])
    np.testing.assert_allclose(global_threshold, 0.50)
    assert diagnostics.iloc[0]["f1"] == 1.0


def test_rank_fusion_thresholds_select_shared_threshold_by_global_f1():
    target = np.array([[1], [0], [0], [0]], dtype=np.uint8)
    scores = np.array([[0.61], [0.60], [0.21], [0.20]], dtype=np.float32)

    _, global_threshold, _ = fit_rank_fusion_thresholds(
        target, scores, shrinkage=1.0, global_candidates=[0.2, 0.6]
    )

    np.testing.assert_allclose(global_threshold, 0.6)


def test_rank_fusion_crossfit_scans_weights_and_keeps_selection_fold_isolated():
    target = np.tile(np.array([[1, 0], [0, 1], [0, 0], [1, 1]], dtype=np.uint8), (8, 1))
    base = np.where(target, 0.8, 0.2).astype(np.float32)
    new = np.where(target, 0.7, 0.3).astype(np.float32)

    rows = crossfit_rank_fusion(
        target,
        base,
        new,
        weights=[0.0, 0.5],
        shrinkages=[0.0, 2.0],
        seeds=[17],
    )

    assert set(rows.columns) >= {
        "new_weight",
        "shrinkage",
        "seed",
        "selection_fold",
        "evaluation_fold",
        "macro_f1",
        "seed_macro_f1",
        "continuous_macro_auc",
    }
    assert len(rows) == 8
    assert (rows.query("shrinkage == 0").macro_f1 == 1.0).all()
    assert (rows.seed_macro_f1 == 1.0).all()
    assert (rows.macro_f1 > 0).all()
