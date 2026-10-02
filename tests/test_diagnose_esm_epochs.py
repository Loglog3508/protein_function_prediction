import importlib

import numpy as np
import pytest

from src.rank_fusion import crossfit_rank_fusion


def implementation():
    return importlib.import_module("src.diagnose_esm_epochs")


def test_support_bands_have_explicit_boundaries():
    diagnose = implementation()
    actual = diagnose.support_bands(np.array([0, 1, 5, 6, 10, 11, 20, 21, 50, 51]), [1, 6, 11, 21, 51])
    assert actual.tolist() == ["0", "1-5", "1-5", "6-10", "6-10", "11-20", "11-20", "21-50", "21-50", "51+"]
    assert diagnose.support_bands(np.array([19, 20, 99, 100, 999, 1000]), [20, 100, 1000]).tolist() == ["0-19", "20-99", "20-99", "100-999", "100-999", "1000+"]


def test_crossfit_predictions_reproduce_existing_protocol():
    diagnose = implementation()
    target = np.array([[1, 0], [0, 1], [1, 0], [0, 1], [0, 0], [1, 1]], dtype=np.uint8)
    scores = np.array([[.8, .2], [.3, .7], [.6, .4], [.2, .6], [.4, .3], [.7, .8]], dtype=np.float32)
    for seed in [17, 31]:
        predictions, thresholds, fold_rows = diagnose.crossfit_predictions(target, scores, seed, 25.0)
        expected = crossfit_rank_fusion(target, scores, scores, weights=[0.0], shrinkages=[25.0], seeds=[seed]).iloc[0]
        assert diagnose.macro_f1_skip_empty(target, predictions) == pytest.approx(expected.seed_macro_f1, abs=1e-12)
        assert predictions.mean() == pytest.approx(expected.seed_predicted_positive_rate, abs=1e-12)
        assert thresholds.shape == (2, 2)
        assert len(fold_rows) == 2
        assert np.isin(predictions, [0, 1]).all()
        reapplied = diagnose.apply_fold_thresholds(scores, thresholds, seed)
        np.testing.assert_array_equal(predictions, reapplied)


def test_label_diagnostics_and_pair_contributions_exclude_empty_labels():
    diagnose = implementation()
    target = np.array([[1, 1, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.uint8)
    scores = np.array([[.9, .8, .3], [.7, .3, .3], [.2, .2, .3], [.1, .1, .3]], dtype=np.float32)
    before_predictions = np.array([[1, 1, 1], [1, 0, 1], [1, 0, 0], [0, 0, 0]], dtype=np.uint8)
    after_predictions = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.uint8)
    before = diagnose.diagnose_labels(target, scores, before_predictions, np.array([8, 20, 2]), ["a", "b", "empty"])
    after = diagnose.diagnose_labels(target, scores, after_predictions, np.array([8, 20, 2]), ["a", "b", "empty"])
    paired = diagnose.paired_label_changes(before, after)
    assert before.predicted_positive.sum() > after.predicted_positive.sum()
    assert paired.loc[paired.label == "b", "new_zero_prediction"].item()
    assert paired.loc[paired.label == "b", "new_zero_tp"].item()
    assert not paired.loc[paired.label == "empty", "new_zero_prediction"].item()
    assert paired.macro_f1_contribution.sum() == pytest.approx(diagnose.macro_f1_skip_empty(target, after_predictions) - diagnose.macro_f1_skip_empty(target, before_predictions))
    assert paired.macro_auc_contribution.sum() == pytest.approx(0)
    assert paired.macro_ap_contribution.sum() == pytest.approx(0)
    assert before.loc[before.label == "empty", "roc_auc"].isna().all()
    assert before.loc[before.label == "a", "average_precision"].item() == pytest.approx(1)
    grouped = diagnose.summarize_strata(after, len(target), "validation_stratum")
    assert grouped.label_count.sum() == 3
    assert grouped.tp.sum() == 1


def test_input_validation_rejects_nonfinite_nonbinary_and_negative_support():
    diagnose = implementation()
    target = np.array([[1], [0]], dtype=np.uint8)
    scores = np.array([[.8], [.1]], dtype=np.float32)
    for invalid_scores, invalid_predictions, support in [(scores * np.nan, target, [1]), (scores, [[2], [0]], [1]), (scores, target, [-1])]:
        with pytest.raises(ValueError):
            diagnose.diagnose_labels(target, invalid_scores, invalid_predictions, np.asarray(support), ["label_0"])


def test_output_prefix_refuses_existing_artifacts(tmp_path):
    diagnose = implementation()
    prefix = tmp_path / "run"
    prefix.with_name("run-summary.json").write_text("old", encoding="utf-8")
    with pytest.raises(FileExistsError):
        diagnose.output_paths(prefix)
    assert prefix.with_name("run-summary.json").read_text(encoding="utf-8") == "old"

def test_score_overlap_uses_positive_negative_medians_and_negative_std():
    diagnose = implementation()
    target = np.array([[1], [1], [0], [0], [0]], dtype=np.uint8)
    scores = np.array([[.8], [.6], [.2], [.4], [.6]], dtype=np.float32)
    result = diagnose.score_overlap(target, scores)
    expected = (np.median([.8, .6]) - np.median([.2, .4, .6])) / np.std([.2, .4, .6])
    assert result["positive_median"].item() == pytest.approx(.7)
    assert result["negative_median"].item() == pytest.approx(.4)
    assert result["negative_std"].item() == pytest.approx(np.std([.2, .4, .6]))
    assert result["median_gap_over_negative_std"].item() == pytest.approx(expected)


def test_score_overlap_handles_degenerate_label_without_finite_value():
    diagnose = implementation()
    target = np.array([[1], [0], [0]], dtype=np.uint8)
    scores = np.array([[.5], [.2], [.2]], dtype=np.float32)
    result = diagnose.score_overlap(target, scores)
    assert result["negative_std"].item() == pytest.approx(0)
    assert np.isnan(result["median_gap_over_negative_std"].item())


def test_training_tertiles_are_support_ordered_and_cover_each_label():
    diagnose = implementation()
    groups = diagnose.training_tertiles(np.array([40, 5, 80, 20, 5, 200]))
    assert groups.tolist() == ["middle", "low", "high", "middle", "low", "high"]


def test_prevalence_decomposition_counts_empty_label_and_concentration():
    diagnose = implementation()
    frame = diagnose.diagnose_labels(
        np.array([[1, 0], [0, 0], [0, 0]], dtype=np.uint8),
        np.array([[.9, .4], [.2, .4], [.1, .4]], dtype=np.float32),
        np.array([[1, 1], [0, 1], [0, 1]], dtype=np.uint8),
        np.array([20, 1]), ["positive", "empty"],
    )
    result = diagnose.prevalence_decomposition(frame, 3)
    assert result["overpredicted_labels"] == 1
    assert result["predicted_positive"] == 4
    assert result["positive_excess"] == 3
    assert result["zero_support_predicted_positive"] == 3
    assert result["top10_positive_excess_share"] == pytest.approx(1)


def test_overlap_distribution_excludes_missing_positives():
    diagnose = implementation()
    frame = diagnose.diagnose_labels(
        np.array([[1, 0], [0, 0], [0, 0]], dtype=np.uint8),
        np.array([[.9, .4], [.2, .4], [.1, .4]], dtype=np.float32),
        np.array([[1, 0], [0, 0], [0, 0]], dtype=np.uint8),
        np.array([20, 1]), ["positive", "empty"],
    )
    result = diagnose.overlap_distribution(frame)
    assert result["defined_labels"] == 1
    assert result["undefined_labels"] == 1
    assert result["gap_le_zero_labels"] == 0
    assert result["gap_lt_one_labels"] == 0


def test_prevalence_unsigned_counts_preserve_positive_excess_and_deficit():
    import pandas as pd
    diagnose = implementation()
    frame = pd.DataFrame({
        "true_support": np.array([3, 1], dtype=np.uint64),
        "predicted_positive": np.array([1, 3], dtype=np.uint64),
        "tp": [1, 1], "fp": [0, 2], "fn": [2, 0],
    })
    result = diagnose.prevalence_decomposition(frame, 5)
    assert result["positive_excess"] == pytest.approx(2)
    assert result["positive_deficit"] == pytest.approx(2)
    assert result["net_excess"] == pytest.approx(0)
    assert result["top10_positive_excess_share"] == pytest.approx(1)
    assert result["positive_excess"] - result["positive_deficit"] == pytest.approx(result["fp"] - result["fn"])
