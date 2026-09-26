import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.sweep_shift import run_tail_sweep


class TailSweepTests(unittest.TestCase):
    def test_tail_sweep_writes_oof_auc_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for index in range(12):
                tail_position = index - 8
                rows.append(
                    {
                        "protein_id": f"P{index:06d}",
                        "sequence": "AAAA" if index % 2 == 0 else "CCCC",
                        "label_0": int(index % 2 == 0),
                        "label_1": int(index % 2 == 1),
                    }
                )
            data_path = root / "train.csv"
            pd.DataFrame(rows).to_csv(data_path, index=False)
            config = {
                "experiment_id": "tail-smoke",
                "seed": 42,
                "data": {"train_path": "train.csv"},
                "distribution": {"cutoff": 8, "folds": 2},
                "selection": {"label_count": 2},
                "features": {"type": "composition"},
                "model": {
                    "type": "sgd",
                    "class_weight": None,
                    "alpha": 0.0001,
                    "max_iter": 50,
                    "tol": 0.001,
                    "n_jobs": 1,
                },
                "candidates": [
                    {"name": "weight-1", "tail_weight": 1.0},
                    {"name": "weight-4", "tail_weight": 4.0},
                ],
                "thresholds": {"shrinkage": 2.0},
                "output_dir": "runs/tail-smoke",
                "metrics_prefix": "metrics/tail-smoke",
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")

            summary_path = run_tail_sweep(config_path, project_root=root)

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["early_rows"], 8)
            self.assertEqual(summary["tail_rows"], 4)
            self.assertEqual(summary["selected_label_count"], 2)
            leaderboard = pd.read_csv(root / "metrics/tail-smoke-leaderboard.csv")
            self.assertEqual(set(leaderboard["candidate"]), {"weight-1", "weight-4"})
            self.assertTrue(
                leaderboard["continuous_macro_auc"].between(0.0, 1.0).all()
            )
            self.assertTrue(
                leaderboard["crossfit_binary_macro_auc"].between(0.0, 1.0).all()
            )
            self.assertTrue(
                (root / "runs/tail-smoke/weight-1-scores.npz").exists()
            )


if __name__ == "__main__":
    unittest.main()
