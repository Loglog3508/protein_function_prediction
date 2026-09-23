import unittest

import pandas as pd

from src.sweep_sgd import select_support_stratified_labels


class SweepSgdTests(unittest.TestCase):
    def test_support_stratified_selection_covers_frequency_range(self):
        target = pd.DataFrame(
            {
                "label_0": [1, 1, 1, 1, 1],
                "label_1": [1, 1, 1, 1, 0],
                "label_2": [1, 1, 1, 0, 0],
                "label_3": [1, 1, 0, 0, 0],
                "label_4": [1, 0, 0, 0, 0],
            }
        )

        selected = select_support_stratified_labels(target, 3)

        self.assertEqual(selected["label"].tolist(), ["label_0", "label_2", "label_4"])
        self.assertEqual(selected["positive_count"].tolist(), [5, 3, 1])
        self.assertEqual(selected["support_rank"].tolist(), [0, 2, 4])

    def test_selection_rejects_invalid_count(self):
        target = pd.DataFrame({"label_0": [0, 1]})
        with self.assertRaises(ValueError):
            select_support_stratified_labels(target, 0)
        with self.assertRaises(ValueError):
            select_support_stratified_labels(target, 2)


if __name__ == "__main__":
    unittest.main()
