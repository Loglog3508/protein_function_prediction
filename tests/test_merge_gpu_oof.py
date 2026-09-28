import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.merge_gpu_oof import merge_gpu_oof


class MergeGpuOofTests(unittest.TestCase):
    def test_reorders_disjoint_fold_rows_and_labels_to_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.npz"
            first = root / "first.npz"
            second = root / "second.npz"
            output = root / "merged" / "scores.npz"
            np.savez_compressed(
                reference,
                validation_scores=np.zeros((3, 2), dtype=np.float32),
                validation_ids=np.asarray(["p2", "p1", "p3"]),
                label_columns=np.asarray(["label_b", "label_a"]),
            )
            np.savez_compressed(
                first,
                validation_scores=np.asarray([[1.0, 2.0]], dtype=np.float32),
                validation_ids=np.asarray(["p1"]),
                label_columns=np.asarray(["label_a", "label_b"]),
            )
            np.savez_compressed(
                second,
                validation_scores=np.asarray(
                    [[3.0, 4.0], [5.0, 6.0]], dtype=np.float32
                ),
                validation_ids=np.asarray(["p3", "p2"]),
                label_columns=np.asarray(["label_a", "label_b"]),
            )

            summary = merge_gpu_oof(reference, [first, second], output)

            with np.load(output, allow_pickle=False) as saved:
                np.testing.assert_array_equal(
                    saved["validation_ids"], ["p2", "p1", "p3"]
                )
                np.testing.assert_array_equal(
                    saved["label_columns"], ["label_b", "label_a"]
                )
                np.testing.assert_array_equal(
                    saved["validation_scores"],
                    [[6.0, 5.0], [2.0, 1.0], [4.0, 3.0]],
                )
            self.assertEqual(summary["fold_rows"], [1, 2])
            self.assertEqual(summary["unique_validation_ids"], 3)

    def test_rejects_duplicate_validation_ids_across_folds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.npz"
            first = root / "first.npz"
            second = root / "second.npz"
            np.savez_compressed(
                reference,
                validation_scores=np.zeros((1, 1), dtype=np.float32),
                validation_ids=np.asarray(["p1"]),
                label_columns=np.asarray(["label_a"]),
            )
            for path in (first, second):
                np.savez_compressed(
                    path,
                    validation_scores=np.ones((1, 1), dtype=np.float32),
                    validation_ids=np.asarray(["p1"]),
                    label_columns=np.asarray(["label_a"]),
                )

            with self.assertRaisesRegex(ValueError, "duplicate validation ID"):
                merge_gpu_oof(reference, [first, second], root / "output" / "x.npz")

    def test_rejects_duplicate_fold_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.npz"
            fold = root / "fold.npz"
            np.savez_compressed(
                reference,
                validation_scores=np.zeros((1, 2), dtype=np.float32),
                validation_ids=np.asarray(["p1"]),
                label_columns=np.asarray(["label_a", "label_b"]),
            )
            np.savez_compressed(
                fold,
                validation_scores=np.ones((1, 3), dtype=np.float32),
                validation_ids=np.asarray(["p1"]),
                label_columns=np.asarray(["label_a", "label_a", "label_b"]),
            )

            with self.assertRaisesRegex(ValueError, "labels must be unique"):
                merge_gpu_oof(reference, [fold], root / "output" / "x.npz")

    def test_rejects_score_shape_that_does_not_match_ids_and_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            malformed = root / "malformed.npz"
            np.savez_compressed(
                malformed,
                validation_scores=np.ones((2, 1), dtype=np.float32),
                validation_ids=np.asarray(["p1"]),
                label_columns=np.asarray(["label_a"]),
            )

            with self.assertRaisesRegex(ValueError, "score shape"):
                merge_gpu_oof(malformed, [], root / "output" / "x.npz")


if __name__ == "__main__":
    unittest.main()
