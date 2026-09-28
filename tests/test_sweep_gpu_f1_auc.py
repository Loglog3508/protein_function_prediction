import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.sweep_gpu_f1_auc import (
    assert_experiment_outputs_absent,
    rank_candidate_results,
    validate_sweep_config,
)


class GpuF1AucSweepTests(unittest.TestCase):
    def test_auc_gate_rejects_higher_f1_candidate(self):
        rows = pd.DataFrame(
            [
                {
                    "candidate": "baseline",
                    "gpu_weight": 0.0,
                    "mean_crossfit_macro_f1": 0.32,
                    "std_crossfit_macro_f1": 0.01,
                    "min_crossfit_macro_f1": 0.30,
                    "continuous_macro_auc": 0.816,
                    "mean_predicted_positive_rate": 0.10,
                    "seed_count": 5,
                },
                {
                    "candidate": "gpu-heavy",
                    "gpu_weight": 0.8,
                    "mean_crossfit_macro_f1": 0.51,
                    "std_crossfit_macro_f1": 0.02,
                    "min_crossfit_macro_f1": 0.48,
                    "continuous_macro_auc": 0.81,
                    "mean_predicted_positive_rate": 0.09,
                    "seed_count": 5,
                },
            ]
        )

        ranked = rank_candidate_results(rows, minimum_auc=0.815045)

        self.assertEqual(ranked.iloc[0]["candidate"], "baseline")
        self.assertTrue(bool(ranked.iloc[0]["auc_eligible"]))
        self.assertFalse(bool(ranked.iloc[1]["auc_eligible"]))

    def test_f1_is_primary_within_auc_eligible_candidates(self):
        rows = pd.DataFrame(
            [
                {
                    "candidate": "higher-auc",
                    "gpu_weight": 0.1,
                    "mean_crossfit_macro_f1": 0.40,
                    "std_crossfit_macro_f1": 0.01,
                    "min_crossfit_macro_f1": 0.38,
                    "continuous_macro_auc": 0.90,
                    "mean_predicted_positive_rate": 0.10,
                    "seed_count": 5,
                },
                {
                    "candidate": "higher-f1",
                    "gpu_weight": 0.2,
                    "mean_crossfit_macro_f1": 0.45,
                    "std_crossfit_macro_f1": 0.02,
                    "min_crossfit_macro_f1": 0.41,
                    "continuous_macro_auc": 0.82,
                    "mean_predicted_positive_rate": 0.10,
                    "seed_count": 5,
                },
            ]
        )

        ranked = rank_candidate_results(rows, minimum_auc=0.815045)

        self.assertEqual(ranked.iloc[0]["candidate"], "higher-f1")

    def test_final_sweep_requires_hard_auc_gate_and_rollback_baseline(self):
        config = {
            "minimum_auc": 0.8,
            "gpu_weights": [0.2, 0.4],
        }
        with self.assertRaisesRegex(ValueError, "minimum_auc"):
            validate_sweep_config(config)
        config["minimum_auc"] = 0.8150451732223777
        with self.assertRaisesRegex(ValueError, "rollback baseline"):
            validate_sweep_config(config)
        config["gpu_weights"].insert(0, 0.0)
        self.assertIsNone(validate_sweep_config(config))

    def test_diagnostic_sweep_may_use_lower_auc_gate(self):
        self.assertIsNone(
            validate_sweep_config(
                {
                    "diagnostic_only": True,
                    "minimum_auc": 0.0,
                    "gpu_weights": [1.0],
                }
            )
        )

    def test_preflight_rejects_any_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stale_thresholds = root / "thresholds.json"
            stale_thresholds.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "new experiment ID"):
                assert_experiment_outputs_absent([root / "run", stale_thresholds])


if __name__ == "__main__":
    unittest.main()
