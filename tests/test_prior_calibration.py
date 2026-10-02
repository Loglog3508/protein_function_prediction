import unittest

import numpy as np

from src.prior_calibration import fit_support_calibrators, transform_support_calibrated_scores


class PriorCalibrationTests(unittest.TestCase):
    def test_support_calibration_returns_finite_scores_and_preserves_shape(self):
        scores = np.array([[0.9, 0.2], [0.8, 0.1], [0.1, 0.8], [0.2, 0.9]], dtype=np.float32)
        target = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.uint8)
        calibrators = fit_support_calibrators(scores, target, np.array([10, 100]))
        transformed = transform_support_calibrated_scores(scores, calibrators, np.array([10, 100]))
        self.assertEqual(transformed.shape, scores.shape)
        self.assertTrue(np.isfinite(transformed).all())
        self.assertTrue(((transformed >= 0) & (transformed <= 1)).all())

    def test_calibration_rejects_misaligned_inputs(self):
        with self.assertRaises(ValueError):
            fit_support_calibrators(np.zeros((2, 1)), np.zeros((2, 2)), np.array([1]))
        with self.assertRaises(ValueError):
            transform_support_calibrated_scores(
                np.zeros((2, 1)), {}, np.array([1, 2])
            )


if __name__ == "__main__":
    unittest.main()
