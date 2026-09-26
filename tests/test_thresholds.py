import unittest

import numpy as np

from src.thresholds import (
    crossfit_shrunk_auc_threshold_score,
    fit_shrunk_thresholds,
    fit_shrunk_auc_thresholds,
    samplewise_label_count_summary,
    select_global_auc_threshold,
    select_label_auc_thresholds,
    select_global_threshold,
    select_label_thresholds,
    shrink_label_thresholds,
    threshold_predictions,
    reorder_label_thresholds,
)


class ThresholdTests(unittest.TestCase):
    def setUp(self):
        self.target = np.array(
            [[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.uint8
        )
        self.scores = np.array(
            [[0.9, 0.2], [0.8, 0.3], [0.4, 0.7], [0.2, 0.6]], dtype=np.float32
        )

    def test_global_threshold_scan_selects_best_threshold(self):
        threshold, scan = select_global_threshold(
            self.target, self.scores, np.array([0.2, 0.5, 0.8])
        )
        self.assertEqual(threshold, 0.5)
        self.assertEqual(scan.iloc[0]["macro_f1"], 1.0)

    def test_label_thresholds_and_shrinkage(self):
        thresholds, diagnostics = select_label_thresholds(
            self.target, self.scores, global_threshold=0.5, candidates=np.array([0.5, 0.8])
        )
        np.testing.assert_allclose(thresholds, [0.5, 0.5])
        shrunk = shrink_label_thresholds(
            thresholds, diagnostics["support"].to_numpy(), global_threshold=0.5, shrinkage=2
        )
        np.testing.assert_allclose(shrunk, [0.5, 0.5])

    def test_prediction_and_samplewise_summary(self):
        predictions = threshold_predictions(self.scores, np.array([0.5, 0.5]))
        np.testing.assert_array_equal(predictions, self.target)
        summary = samplewise_label_count_summary(self.target, predictions)
        self.assertEqual(summary["true"]["mean"], 1.0)
        self.assertEqual(summary["predicted"]["max"], 1)

    def test_auc_thresholds_maximize_balanced_separation(self):
        target = np.array(
            [[0, 0], [0, 0], [1, 0], [1, 1], [1, 1], [1, 1]], dtype=np.uint8
        )
        scores = np.array(
            [
                [0.10, 0.05],
                [0.20, 0.10],
                [0.45, 0.40],
                [0.55, 0.60],
                [0.70, 0.80],
                [0.90, 0.95],
            ],
            dtype=np.float32,
        )
        candidates = np.array([0.3, 0.5, 0.7], dtype=np.float32)

        global_threshold, scan = select_global_auc_threshold(
            target, scores, candidates
        )
        label_thresholds, diagnostics = select_label_auc_thresholds(
            target,
            scores,
            global_threshold=global_threshold,
            candidates=candidates,
        )

        self.assertEqual(global_threshold, 0.5)
        self.assertEqual(scan.iloc[0]["macro_auc"], 0.9375)
        np.testing.assert_allclose(label_thresholds, [0.3, 0.5])
        self.assertEqual(diagnostics["auc"].tolist(), [1.0, 1.0])

    def test_auc_threshold_crossfit_returns_binary_auc(self):
        result = crossfit_shrunk_auc_threshold_score(
            np.tile(self.target, (4, 1)),
            np.tile(self.scores, (4, 1)),
            seed=42,
            shrinkage=2,
            candidates=np.array([0.3, 0.5, 0.7]),
        )

        self.assertEqual(result["macro_auc"], 1.0)
        self.assertEqual(result["predicted_positive_rate"], 0.5)

    def test_final_auc_thresholds_are_fit_on_all_rows(self):
        thresholds, global_threshold = fit_shrunk_auc_thresholds(
            self.target,
            self.scores,
            shrinkage=2,
            candidates=np.array([0.3, 0.5, 0.7]),
        )

        self.assertEqual(global_threshold, 0.5)
        np.testing.assert_allclose(thresholds, [0.5, 0.5])

    def test_final_f1_thresholds_are_fit_on_all_rows(self):
        thresholds, global_threshold = fit_shrunk_thresholds(
            self.target,
            self.scores,
            shrinkage=2,
            candidates=np.array([0.3, 0.5, 0.7]),
        )

        self.assertEqual(global_threshold, 0.5)
        np.testing.assert_allclose(thresholds, [0.5, 0.5])

    def test_reorder_label_thresholds_uses_label_names(self):
        reordered = reorder_label_thresholds(
            np.array([0.2, 0.8], dtype=np.float32),
            ["label_b", "label_a"],
            ["label_a", "label_b"],
        )
        np.testing.assert_allclose(reordered, [0.8, 0.2])


if __name__ == "__main__":
    unittest.main()
