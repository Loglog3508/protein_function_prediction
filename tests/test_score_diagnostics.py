import unittest

import numpy as np

import src.metrics


class ScoreDiagnosticsTests(unittest.TestCase):
    def test_label_diagnostics_distinguish_ranking_false_positives_and_empty_labels(self):
        diagnose = getattr(src.metrics, "per_label_score_diagnostics", None)
        self.assertIsNotNone(diagnose)
        target = np.array([[1, 1, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
        predicted = np.array([[1, 0, 1], [1, 0, 0], [0, 0, 0], [0, 0, 0]])
        scores = np.array([[0.9, 0.1, 0.9], [0.8, 0.8, 0.1], [0.2, 0.6, 0.1], [0.1, 0.2, 0.1]])
        frame = diagnose(target, predicted, scores, ["a", "b", "c"], np.array([9, 3, 1]))
        self.assertAlmostEqual(frame.loc[0, "precision"], 0.5)
        self.assertAlmostEqual(frame.loc[0, "recall"], 1.0)
        self.assertAlmostEqual(frame.loc[0, "predicted_to_true_ratio"], 2.0)
        self.assertAlmostEqual(frame.loc[0, "average_precision"], 1.0)
        self.assertAlmostEqual(frame.loc[1, "average_precision"], 0.25)
        self.assertTrue(np.isnan(frame.loc[2, "average_precision"]))
        self.assertTrue(np.isnan(frame.loc[2, "predicted_to_true_ratio"]))
        self.assertEqual(frame.loc[2, "predicted_positive"], 1)
        self.assertEqual(frame.loc[1, "training_support"], 3)

    def test_diagnostics_reject_misaligned_nonfinite_and_nonbinary_inputs(self):
        diagnose = getattr(src.metrics, "per_label_score_diagnostics", None)
        self.assertIsNotNone(diagnose)
        target = np.array([[1], [0]])
        for support, scores, prediction in (
            ([1, 2], [[0.9], [0.1]], target),
            ([1], [[float("nan")], [0.1]], target),
            ([1], [[0.9], [0.1]], [[2], [0]]),
        ):
            with self.subTest(support=support, scores=scores), self.assertRaises(ValueError):
                diagnose(target, prediction, scores, ["a"], np.asarray(support))


if __name__ == "__main__":
    unittest.main()
