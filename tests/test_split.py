import importlib
import unittest

import numpy as np


class SplitTests(unittest.TestCase):
    def test_multilabel_split_is_deterministic_and_keeps_groups_together(self):
        try:
            split_module = importlib.import_module("src.split")
        except ModuleNotFoundError:
            self.fail("split module is missing")
        make_split = getattr(split_module, "make_group_multilabel_split", None)
        self.assertIsNotNone(make_split, "group multilabel split function is missing")
        groups = np.array(
            ["same", "same", "g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8"]
        )
        target = np.array(
            [
                [1, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [1, 0, 1],
                [0, 1, 1],
                [0, 1, 0],
                [0, 0, 1],
                [1, 1, 1],
                [0, 1, 1],
                [1, 0, 1],
            ],
            dtype=np.uint8,
        )

        train_indices, validation_indices = make_split(
            target, groups, validation_size=0.3, seed=42
        )
        repeated_train, repeated_validation = make_split(
            target, groups, validation_size=0.3, seed=42
        )

        self.assertEqual(set(train_indices) & set(validation_indices), set())
        self.assertEqual(
            sorted(np.concatenate([train_indices, validation_indices]).tolist()),
            list(range(len(target))),
        )
        self.assertEqual(0 in train_indices, 1 in train_indices)
        np.testing.assert_array_equal(train_indices, repeated_train)
        np.testing.assert_array_equal(validation_indices, repeated_validation)
        self.assertTrue((target[validation_indices].sum(axis=0) > 0).all())


if __name__ == "__main__":
    unittest.main()
