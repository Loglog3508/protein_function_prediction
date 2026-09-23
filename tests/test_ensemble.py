import unittest

import numpy as np

from src.ensemble import apply_label_association, build_label_association


class EnsembleTests(unittest.TestCase):
    def test_label_association_uses_positive_cooccurrence(self):
        target = np.array(
            [[1, 1, 0], [1, 1, 0], [0, 0, 1], [0, 0, 1]], dtype=np.uint8
        )
        association = build_label_association(target, top_k=1)

        self.assertEqual(association.shape, (3, 3))
        self.assertGreater(association[0, 1], 0)
        self.assertEqual(association[0, 0], 0)

    def test_association_blend_preserves_shape_and_range(self):
        scores = np.array([[0.8, 0.1], [0.2, 0.7]], dtype=np.float32)
        association = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        blended = apply_label_association(scores, association, 0.1)

        self.assertEqual(blended.shape, scores.shape)
        self.assertTrue(((blended >= 0) & (blended <= 1)).all())


if __name__ == "__main__":
    unittest.main()
