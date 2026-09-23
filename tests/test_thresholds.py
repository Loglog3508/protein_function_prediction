import unittest

import numpy as np

from src.thresholds import (
    samplewise_label_count_summary,
    select_global_threshold,
    select_label_thresholds,
    shrink_label_thresholds,
    threshold_predictions,
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


if __name__ == "__main__":
    unittest.main()
