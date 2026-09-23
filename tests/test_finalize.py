import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.finalize import run_final_training


class FinalTrainingTests(unittest.TestCase):
    def test_final_training_uses_all_rows_and_validates_submission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data").mkdir()
            train = pd.DataFrame(
                {
                    "protein_id": [f"P{i}" for i in range(8)],
                    "sequence": ["AAAA", "CCCC", "ACAC", "CACA"] * 2,
                    "label_0": [1, 0, 1, 0] * 2,
                    "label_1": [0, 1, 0, 1] * 2,
                }
            )
            test = pd.DataFrame(
                {"protein_id": ["T0", "T1"], "sequence": ["AAAA", "CCCC"]}
            )
            train.to_csv(root / "data/train.csv", index=False)
            test.to_csv(root / "data/test.csv", index=False)
            config = {
                "experiment_id": "final-test",
                "seed": 42,
                "data": {"train_path": "data/train.csv", "test_path": "data/test.csv"},
                "features": {
                    "type": "kmer_tfidf",
                    "k_min": 2,
                    "k_max": 2,
                    "min_df": 1,
                    "max_features": 10,
                    "sublinear_tf": True,
                },
                "model": {
                    "type": "sgd",
                    "class_weight": "balanced",
                    "alpha": 0.0001,
                    "max_iter": 20,
                    "tol": 0.001,
                    "n_jobs": 1,
                },
            }
            (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
            thresholds = {
                "labels": ["label_0", "label_1"],
                "thresholds": [0.5, 0.5],
                "strategy": "test",
            }
            (root / "thresholds.json").write_text(
                json.dumps(thresholds), encoding="utf-8"
            )

            metadata_path = run_final_training(
                root / "config.json",
                thresholds_path=root / "thresholds.json",
                submission_path="artifacts/submission.csv",
                metadata_path="artifacts/final-metadata.json",
                project_root=root,
            )

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["training_scope"], "all_training_rows")
            self.assertEqual(metadata["train_rows"], 8)
            self.assertEqual(metadata["submission_validation"]["rows"], 2)
            self.assertTrue((root / "artifacts/submission.csv").exists())


if __name__ == "__main__":
    unittest.main()
