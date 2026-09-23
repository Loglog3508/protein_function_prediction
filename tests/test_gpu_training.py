import unittest

import numpy as np

from src.train_gpu import encode_sequences


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


if __name__ == "__main__":
    unittest.main()
