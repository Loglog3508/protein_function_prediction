import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import src.predict
import src.train
import src.validate_submission


class SmokePipelineTests(unittest.TestCase):
    def test_configured_smoke_run_trains_predicts_and_validates(self):
        run_training = getattr(src.train, "run_training", None)
        run_prediction = getattr(src.predict, "run_prediction", None)
        validate_submission_file = getattr(
            src.validate_submission, "validate_submission_file", None
        )
        self.assertIsNotNone(run_training, "training entry point is missing")
        self.assertIsNotNone(run_prediction, "prediction entry point is missing")
        self.assertIsNotNone(
            validate_submission_file, "submission validator is missing"
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            data_dir.mkdir()
            train = pd.DataFrame(
                {
                    "protein_id": [f"P{i:03d}" for i in range(12)],
                    "sequence": [
                        "ACDEFG",
                        "AAAAAA",
                        "CCCCCC",
                        "DDDDDD",
                        "EEEEEE",
                        "FFFFFF",
                        "GGGGGG",
                        "HHHHHH",
                        "IIIIII",
                        "KKKKKK",
                        "LLLLLL",
                        "MMMMMM",
                    ],
                    "label_0": [0, 1] * 6,
                    "label_1": [0, 0, 1, 1] * 3,
                    "label_2": [1, 0, 0] * 4,
                }
            )
            test = pd.DataFrame(
                {
                    "protein_id": ["T000", "T001", "T002", "T003"],
                    "sequence": ["ACDEFG", "AAAAAA", "CCCCCC", "MMMMMM"],
                }
            )
            train.to_csv(data_dir / "train.csv", index=False)
            test.to_csv(data_dir / "test.csv", index=False)
            config = {
                "experiment_id": "test-smoke",
                "seed": 42,
                "data": {
                    "train_path": "data/train.csv",
                    "test_path": "data/test.csv",
                    "max_train_samples": 10,
                    "max_test_samples": 3,
                    "max_labels": 2,
                },
                "split": {"validation_size": 0.25},
                "features": {"type": "composition"},
                "model": {
                    "type": "random_forest",
                    "n_estimators": 2,
                    "max_depth": 2,
                    "class_weight": "balanced",
                    "n_jobs": 1,
                },
                "threshold": 0.5,
                "output_dir": "artifacts/runs/test-smoke",
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")

            run_dir = run_training(config_path, project_root=root)
            submission_path = root / "artifacts" / "submissions" / "smoke.csv"
            run_prediction(
                config_path,
                run_dir=run_dir,
                output_path=submission_path,
                project_root=root,
            )
            summary = validate_submission_file(
                submission_path,
                test_path=data_dir / "test.csv",
                expected_label_columns=["label_0", "label_1", "label_2"],
                max_test_samples=3,
            )

            submission = pd.read_csv(submission_path)
            self.assertTrue((run_dir / "model.joblib").exists())
            self.assertTrue((run_dir / "metrics.json").exists())
            self.assertEqual(
                submission.columns.tolist(),
                ["protein_id", "label_0", "label_1", "label_2"],
            )
            self.assertEqual(submission["protein_id"].tolist(), test.protein_id[:3].tolist())
            self.assertTrue(submission.iloc[:, 1:].isin([0, 1]).all().all())
            self.assertEqual(summary["rows"], 3)
            self.assertEqual(summary["labels"], 3)


if __name__ == "__main__":
    unittest.main()
