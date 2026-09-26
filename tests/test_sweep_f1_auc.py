import unittest

import pandas as pd

from src.sweep_f1_auc import aggregate_seed_results


class F1AucSweepTests(unittest.TestCase):
    def test_aggregate_seed_results_ranks_mean_f1_before_auc(self):
        rows = pd.DataFrame(
            [
                {
                    "candidate": "primary-0.20",
                    "primary_weight": 0.2,
                    "shrinkage": 10.0,
                    "seed": 1,
                    "crossfit_macro_f1": 0.32,
                    "continuous_macro_auc": 0.81,
                    "predicted_positive_rate": 0.10,
                },
                {
                    "candidate": "primary-0.20",
                    "primary_weight": 0.2,
                    "shrinkage": 10.0,
                    "seed": 2,
                    "crossfit_macro_f1": 0.34,
                    "continuous_macro_auc": 0.81,
                    "predicted_positive_rate": 0.12,
                },
                {
                    "candidate": "primary-0.40",
                    "primary_weight": 0.4,
                    "shrinkage": 10.0,
                    "seed": 1,
                    "crossfit_macro_f1": 0.329,
                    "continuous_macro_auc": 0.82,
                    "predicted_positive_rate": 0.11,
                },
                {
                    "candidate": "primary-0.40",
                    "primary_weight": 0.4,
                    "shrinkage": 10.0,
                    "seed": 2,
                    "crossfit_macro_f1": 0.329,
                    "continuous_macro_auc": 0.82,
                    "predicted_positive_rate": 0.11,
                },
            ]
        )

        leaderboard = aggregate_seed_results(rows)

        self.assertEqual(leaderboard.iloc[0]["candidate"], "primary-0.20")
        self.assertAlmostEqual(leaderboard.iloc[0]["mean_crossfit_macro_f1"], 0.33)
        self.assertGreater(leaderboard.iloc[0]["std_crossfit_macro_f1"], 0.0)
        self.assertEqual(leaderboard.iloc[0]["seed_count"], 2)


if __name__ == "__main__":
    unittest.main()
