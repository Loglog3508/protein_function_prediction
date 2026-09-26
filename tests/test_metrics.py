import unittest

import numpy as np

import src.metrics
from src.metrics import macro_f1_skip_empty, macro_roc_auc_skip_degenerate


class MacroF1Tests(unittest.TestCase):
    def test_perfect_prediction_scores_one(self):
        target = np.array([[1, 0], [0, 1], [1, 0]], dtype=np.int8)

        self.assertEqual(macro_f1_skip_empty(target, target.copy()), 1.0)

    def test_all_zero_prediction_scores_zero_for_active_labels(self):
        target = np.array([[1, 0], [0, 1]], dtype=np.int8)
        prediction = np.zeros_like(target)

        self.assertEqual(macro_f1_skip_empty(target, prediction), 0.0)

    def test_label_without_true_positives_is_skipped(self):
        target = np.array([[1, 0], [0, 0]], dtype=np.int8)
        prediction = np.array([[1, 1], [0, 1]], dtype=np.int8)

        self.assertEqual(macro_f1_skip_empty(target, prediction), 1.0)

    def test_all_empty_labels_are_rejected(self):
        target = np.zeros((2, 2), dtype=np.int8)

        with self.assertRaisesRegex(ValueError, "positive"):
            macro_f1_skip_empty(target, target)

    def test_dimension_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "equal shape"):
            macro_f1_skip_empty(np.zeros((2, 2)), np.zeros((2, 3)))

    def test_per_label_diagnostics_report_errors_and_rates(self):
        diagnostics_function = getattr(
            src.metrics, "per_label_classification_metrics", None
        )
        self.assertIsNotNone(diagnostics_function, "per-label diagnostics are missing")
        target = np.array([[1, 0], [0, 1], [1, 0]], dtype=np.int8)
        prediction = np.array([[1, 1], [1, 0], [0, 0]], dtype=np.int8)

        diagnostics = diagnostics_function(
            target, prediction, ["label_0", "label_1"]
        )

        first = diagnostics.iloc[0]
        second = diagnostics.iloc[1]
        self.assertEqual((first.tp, first.fp, first.fn), (1, 1, 1))
        self.assertEqual(first.f1, 0.5)
        self.assertEqual(second.f1, 0.0)
        self.assertAlmostEqual(first.predicted_positive_rate, 2 / 3)


class MacroRocAucTests(unittest.TestCase):
    def test_degenerate_labels_are_skipped(self):
        target = np.array(
            [[0, 1, 0], [0, 1, 0], [1, 1, 0], [1, 1, 0]], dtype=np.uint8
        )
        scores = np.array(
            [[0.1, 0.9, 0.2], [0.2, 0.8, 0.3], [0.8, 0.7, 0.4], [0.9, 0.6, 0.5]],
            dtype=np.float32,
        )

        self.assertEqual(macro_roc_auc_skip_degenerate(target, scores), 1.0)

    def test_all_degenerate_labels_are_rejected(self):
        target = np.ones((3, 2), dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "both classes"):
            macro_roc_auc_skip_degenerate(target, target.astype(np.float32))


if __name__ == "__main__":
    unittest.main()
