"""F1-first selection rules for the homology/SGD blend."""

import pandas as pd

import numpy as np
import pytest

from src.sweep_homology_blend import align_score_columns, select_f1_auc_tradeoff


def test_auc_breaks_only_near_f1_ties():
    candidates = pd.DataFrame(
        [
            {"name": "highest_f1", "mean_crossfit_macro_f1": 0.335, "continuous_macro_auc": 0.799},
            {"name": "balanced", "mean_crossfit_macro_f1": 0.334, "continuous_macro_auc": 0.815},
            {"name": "auc_only", "mean_crossfit_macro_f1": 0.330, "continuous_macro_auc": 0.900},
        ]
    )
    assert select_f1_auc_tradeoff(candidates, f1_tolerance=0.002)["name"] == "balanced"


def test_aligns_scores_by_label_name_and_rejects_missing_labels():
    aligned = align_score_columns(
        np.array([[0.8, 0.2]], dtype=np.float32),
        ["label_1", "label_0"],
        ["label_0", "label_1"],
    )
    assert aligned.tolist() == [[pytest.approx(0.2), pytest.approx(0.8)]]
    with pytest.raises(ValueError, match="label"):
        align_score_columns(np.array([[0.8]], dtype=np.float32), ["label_1"], ["label_0"])
