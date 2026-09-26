import unittest

import numpy as np
from scipy import sparse

from src.knn import weighted_knn_label_scores


class WeightedKnnTests(unittest.TestCase):
    def test_cosine_neighbors_transfer_matching_labels(self):
        training_features = sparse.csr_matrix(
            np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32)
        )
        training_target = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.uint8)
        evaluation_features = sparse.csr_matrix(
            np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        )

        scores = weighted_knn_label_scores(
            training_features,
            training_target,
            evaluation_features,
            n_neighbors=2,
            similarity_power=2.0,
        )

        self.assertEqual(scores.shape, (2, 2))
        self.assertGreater(scores[0, 0], scores[0, 1])
        self.assertGreater(scores[1, 1], scores[1, 0])
        self.assertTrue(((scores >= 0) & (scores <= 1)).all())

    def test_invalid_neighbor_count_is_rejected(self):
        features = sparse.eye(2, dtype=np.float32, format="csr")
        target = np.eye(2, dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "neighbors"):
            weighted_knn_label_scores(
                features, target, features, n_neighbors=3
            )

    def test_all_positive_neighbors_do_not_exceed_one_from_rounding(self):
        count = 100
        training_features = sparse.csr_matrix(
            np.array(
                [[1.0, index / (count * 3)] for index in range(count)],
                dtype=np.float32,
            )
        )
        target = np.ones((count, 1), dtype=np.uint8)
        query = sparse.csr_matrix(np.array([[1.0, 0.0]], dtype=np.float32))

        scores = weighted_knn_label_scores(
            training_features,
            target,
            query,
            n_neighbors=count,
            similarity_power=2.0,
        )

        self.assertLessEqual(float(scores[0, 0]), 1.0)


if __name__ == "__main__":
    unittest.main()
