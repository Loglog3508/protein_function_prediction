import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.sweep_knn_shift import run_tail_knn_blend_sweep


class TailKnnBlendSweepTests(unittest.TestCase):
    def test_tail_knn_sweep_compares_primary_and_blended_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = pd.DataFrame(
                {
                    "protein_id": [f"P{i:06d}" for i in range(12)],
                    "sequence": ["AAAA" if i % 2 == 0 else "CCCC" for i in range(12)],
                    "label_0": [int(i % 2 == 0) for i in range(12)],
                    "label_1": [int(i % 2 == 1) for i in range(12)],
                }
            )
            train.to_csv(root / "train.csv", index=False)
            primary_path = root / "primary.npz"
            np.savez_compressed(
                primary_path,
                validation_scores=np.array(
                    [[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]],
                    dtype=np.float32,
                ),
                validation_ids=np.array(
                    ["P000008", "P000009", "P000010", "P000011"]
                ),
                label_columns=np.array(["label_0", "label_1"]),
            )
            config = {
                "experiment_id": "knn-tail-smoke",
                "seed": 42,
                "data": {"train_path": "train.csv"},
                "distribution": {"cutoff": 8, "folds": 2},
                "primary_scores": "primary.npz",
                "features": {"type": "composition"},
                "knn": {
                    "n_neighbors": 2,
                    "similarity_power": 2.0,
                    "query_batch_size": 2,
                },
                "primary_weights": [1.0, 0.5],
                "thresholds": {"shrinkage": 2.0},
                "output_dir": "runs/knn-tail-smoke",
                "metrics_prefix": "metrics/knn-tail-smoke",
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")

            summary_path = run_tail_knn_blend_sweep(config_path, project_root=root)

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["tail_rows"], 4)
            leaderboard = pd.read_csv(root / "metrics/knn-tail-smoke-leaderboard.csv")
            self.assertEqual(set(leaderboard["candidate"]), {"primary-1.00", "primary-0.50"})
            self.assertTrue((root / "runs/knn-tail-smoke/knn-scores.npz").exists())
            self.assertTrue((root / "runs/knn-tail-smoke/best-scores.npz").exists())


if __name__ == "__main__":
    unittest.main()
