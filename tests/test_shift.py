import unittest

import numpy as np

from src.shift import tail_distribution_mask


class TailDistributionTests(unittest.TestCase):
    def test_numeric_protein_id_cutoff_selects_tail_rows(self):
        mask = tail_distribution_mask(
            ["P112733", "P112734", "P142245"], cutoff=112734
        )

        np.testing.assert_array_equal(mask, [False, True, True])

    def test_invalid_protein_id_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "protein ID"):
            tail_distribution_mask(["bad-id"], cutoff=112734)


if __name__ == "__main__":
    unittest.main()
