import unittest

import numpy as np
import torch

from src.train_gpu import _build_model, encode_sequences


class GpuTrainingTests(unittest.TestCase):
    def test_sequence_encoding_preserves_head_and_tail(self):
        encoded = encode_sequences(["ACDEFG", "AX"], max_length=4)

        self.assertEqual(encoded.shape, (2, 4))
        self.assertEqual(encoded.dtype, np.uint8)
        self.assertTrue((encoded[0] > 0).all())
        self.assertTrue((encoded[1, :2] > 0).all())
        self.assertTrue((encoded[1, 2:] == 0).all())

    def test_sequence_encoding_rejects_tiny_limit(self):
        with self.assertRaises(ValueError):
            encode_sequences(["ACD"], max_length=1)

    def test_max_mean_pooling_doubles_classifier_features(self):
        model = _build_model(
            torch,
            label_count=5,
            model_config={
                "embedding_dim": 4,
                "channels": 3,
                "kernels": [3, 5],
                "pooling": "max_mean",
            },
        )

        logits = model(torch.tensor([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]]))

        self.assertEqual(tuple(logits.shape), (2, 5))
        self.assertEqual(model.output.in_features, 3 * 2 * 2)

    def test_pooling_rejects_unknown_mode(self):
        with self.assertRaisesRegex(ValueError, "pooling"):
            _build_model(
                torch,
                label_count=2,
                model_config={
                    "embedding_dim": 4,
                    "channels": 3,
                    "kernels": [3],
                    "pooling": "median",
                },
            )

    def test_sequence_statistics_branch_accepts_side_features(self):
        model = _build_model(
            torch,
            label_count=3,
            model_config={
                "embedding_dim": 4,
                "channels": 3,
                "kernels": [3],
                "pooling": "max_mean",
                "statistics_dimensions": 2,
                "statistics_hidden": 5,
            },
        )

        logits = model(
            torch.tensor([[1, 2, 0, 0]]),
            torch.tensor([[0.25, 0.75]]),
        )

        self.assertEqual(tuple(logits.shape), (1, 3))
        self.assertEqual(model.output.in_features, 3 * 2 + 5)


if __name__ == "__main__":
    unittest.main()
