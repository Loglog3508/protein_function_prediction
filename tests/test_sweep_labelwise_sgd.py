import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import numpy as np

from src.sweep_labelwise_sgd import _support_multipliers, run_labelwise_weight_sweep


class LabelwiseWeightSweepTests(unittest.TestCase):
    def test_empty_support_stratum_is_rejected_before_training(self):
        target = np.ones((150, 3), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "empty"):
            _support_multipliers(
                target,
                {"multipliers": {"high": 1.0, "low": 2.0}},
                [{"name": "high", "min_support": 100}, {"name": "low", "max_support": 99}],
            )

    def test_support_strata_apply_different_training_weights(self):
        target = np.zeros((10, 3), dtype=np.uint8)
        target[:8, 0] = 1
        target[:4, 1] = 1
        target[:1, 2] = 1
        multipliers = _support_multipliers(
            target,
            {"multipliers": {"high": 0.75, "medium": 1.0, "low": 1.5}},
            [
                {"name": "high", "min_support": 6},
                {"name": "medium", "min_support": 2, "max_support": 5},
                {"name": "low", "max_support": 1},
            ],
        )
        np.testing.assert_array_equal(multipliers, [0.75, 1.0, 1.5])

    def test_support_stratified_sweep_writes_scores_and_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for index in range(16):
                rows.append(
                    {
                        "protein_id": f"P{index:06d}",
                        "sequence": "AAAA" if index % 2 == 0 else "CCCC",
                        "label_0": int(index % 2 == 0),
                        "label_1": int(index % 4 == 0 or (index >= 12 and index % 2 == 1)),
                    }
                )
            pd.DataFrame(rows).to_csv(root / "train.csv", index=False)
            config = {
                "experiment_id": "labelwise-smoke",
                "seed": 42,
                "data": {"train_path": "train.csv"},
                "distribution": {"cutoff": 12, "folds": 2},
                "selection": {"label_count": 2},
                "features": {"type": "composition"},
                "model": {
                    "type": "sgd",
                    "class_weight": "balanced",
                    "alpha": 0.0001,
                    "max_iter": 50,
                    "tol": 0.001,
                    "n_jobs": 1,
                },
                "support_strata": [
                    {"name": "high", "min_support": 6},
                    {"name": "low", "max_support": 5},
                ],
                "candidates": [
                    {
                        "name": "balanced",
                        "multipliers": {"high": 1.0, "medium": 1.0, "low": 1.0},
                    },
                    {
                        "name": "rare-boost",
                        "multipliers": {"high": 0.8, "medium": 1.0, "low": 1.5},
                    },
                ],
                "threshold": 0.5,
                "thresholds": {"shrinkage": 2.0},
                "seeds": [17, 31, 42, 73, 101],
                "output_dir": "runs/labelwise-smoke",
                "metrics_prefix": "metrics/labelwise-smoke",
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")

            summary_path = run_labelwise_weight_sweep(config_path, project_root=root)

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["early_rows"], 12)
            self.assertEqual(summary["tail_rows"], 4)
            leaderboard = pd.read_csv(root / "metrics/labelwise-smoke-leaderboard.csv")
            self.assertEqual(set(leaderboard["candidate"]), {"balanced", "rare-boost"})
            self.assertTrue(leaderboard["crossfit_macro_f1"].between(0.0, 1.0).all())
            self.assertIn("mean_crossfit_macro_f1", leaderboard.columns)
            self.assertEqual(len(summary["threshold_seeds"]), 5)
            self.assertTrue((root / "runs/labelwise-smoke/balanced-scores.npz").exists())


if __name__ == "__main__":
    unittest.main()
