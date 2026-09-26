import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import src.train


class EvaluationRunnerTests(unittest.TestCase):
    def test_fixed_split_evaluation_saves_scores_and_diagnostics(self):
        run_evaluation = getattr(src.train, "run_evaluation", None)
        self.assertIsNotNone(run_evaluation, "fixed-split evaluation runner is missing")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            split_dir = root / "splits"
            data_dir.mkdir()
            split_dir.mkdir()
            train = pd.DataFrame(
                {
                    "protein_id": [f"P{i:02d}" for i in range(20)],
                    "sequence": ["ACDEFG" if i % 2 else "AAAAAA" for i in range(20)],
                    "label_0": [0, 1] * 10,
                    "label_1": [0, 0, 1, 1] * 5,
                }
            )
            train.to_csv(data_dir / "train.csv", index=False)
            pd.DataFrame({"protein_id": train.protein_id[:15]}).to_csv(
                split_dir / "train_ids.csv", index=False
            )
            pd.DataFrame({"protein_id": train.protein_id[15:]}).to_csv(
                split_dir / "validation_ids.csv", index=False
            )
            config = {
                "experiment_id": "test-evaluation",
                "seed": 42,
                "data": {"train_path": "data/train.csv", "max_labels": None},
                "split": {
                    "train_ids": "splits/train_ids.csv",
                    "validation_ids": "splits/validation_ids.csv",
                },
                "features": {"type": "composition"},
                "model": {
                    "type": "random_forest",
                    "n_estimators": 2,
                    "max_depth": 2,
                    "class_weight": "balanced",
                    "n_jobs": 1,
                },
                "threshold": 0.5,
                "output_dir": "artifacts/runs/test-evaluation",
                "metrics_prefix": "artifacts/metrics/test-evaluation",
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")

            summary_path = run_evaluation(config_path, project_root=root)

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["train_rows"], 15)
            self.assertEqual(summary["validation_rows"], 5)
            self.assertEqual(summary["label_count"], 2)
            self.assertGreaterEqual(summary["fit_seconds"], 0.0)
            self.assertGreaterEqual(summary["inference_seconds"], 0.0)
            self.assertTrue((root / "artifacts/runs/test-evaluation/scores.npz").exists())
            diagnostics = pd.read_csv(
                root / "artifacts/metrics/test-evaluation-per-label.csv"
            )
            self.assertEqual(diagnostics["label"].tolist(), ["label_0", "label_1"])

    def test_kmer_evaluation_records_sparse_matrix_resources(self):
        run_evaluation = getattr(src.train, "run_evaluation", None)
        self.assertIsNotNone(run_evaluation)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data").mkdir()
            (root / "splits").mkdir()
            train = pd.DataFrame(
                {
                    "protein_id": [f"P{i:02d}" for i in range(12)],
                    "sequence": [
                        "ACDEFG",
                        "ACDXXX",
                        "CDEFGH",
                        "DEFGHI",
                        "EFGHIK",
                        "FGHIKL",
                    ]
                    * 2,
                    "label_0": [0, 1] * 6,
                    "label_1": [0, 0, 1, 1] * 3,
                }
            )
            train.to_csv(root / "data/train.csv", index=False)
            test = pd.DataFrame(
                {"protein_id": ["T0", "T1"], "sequence": ["ACDEFG", "FGHIKL"]}
            )
            test.to_csv(root / "data/test.csv", index=False)
            pd.DataFrame({"protein_id": train.protein_id[:8]}).to_csv(
                root / "splits/train_ids.csv", index=False
            )
            pd.DataFrame({"protein_id": train.protein_id[8:]}).to_csv(
                root / "splits/validation_ids.csv", index=False
            )
            config = {
                "experiment_id": "test-kmer",
                "seed": 42,
                "data": {
                    "train_path": "data/train.csv",
                    "test_path": "data/test.csv",
                    "max_labels": None,
                },
                "split": {
                    "train_ids": "splits/train_ids.csv",
                    "validation_ids": "splits/validation_ids.csv",
                },
                "features": {
                    "type": "kmer_tfidf",
                    "k_min": 3,
                    "k_max": 3,
                    "min_df": 1,
                    "max_features": 20,
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
                "threshold": 0.5,
                "output_dir": "artifacts/runs/test-kmer",
                "metrics_prefix": "artifacts/metrics/test-kmer",
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")

            summary_path = run_evaluation(config_path, project_root=root)

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            resources = summary["feature_resources"]
            self.assertTrue(resources["sparse"])
            self.assertGreater(resources["vocabulary_size"], 0)
            self.assertLessEqual(resources["vocabulary_size"], 20)
            self.assertEqual(resources["train_shape"][0], 8)
            self.assertEqual(resources["validation_shape"][0], 4)
            self.assertGreater(resources["train_nnz"], 0)
            self.assertGreater(resources["train_bytes"], 0)
            with np.load(root / "artifacts/runs/test-kmer/scores.npz") as saved:
                self.assertEqual(saved["validation_scores"].shape, (4, 2))
                self.assertEqual(saved["test_scores"].shape, (2, 2))
                self.assertEqual(saved["test_ids"].tolist(), ["T0", "T1"])

    def test_kmer_statistics_features_are_appended(self):
        training = pd.DataFrame({"sequence": ["ACDE", "AAAA", "CCCC"]})
        validation = pd.DataFrame({"sequence": ["ACAC"]})
        build_matrices = getattr(src.train, "_build_feature_matrices")

        train_matrix, validation_matrix, resources, _ = build_matrices(
            training,
            validation,
            {
                "type": "kmer_tfidf_statistics",
                "k_min": 2,
                "k_max": 2,
                "min_df": 1,
                "max_features": 10,
                "sublinear_tf": True,
            },
        )

        self.assertTrue(resources["sparse"])
        self.assertEqual(resources["statistics_dimensions"], 433)
        self.assertEqual(train_matrix.shape[1], resources["vocabulary_size"] + 433)
        self.assertEqual(validation_matrix.shape[1], train_matrix.shape[1])

    def test_kmer_blocks_keep_separate_vocabularies(self):
        training = pd.DataFrame(
            {"sequence": ["ACDEFG", "AAAAAA", "CCCCCC", "ACACAC"]}
        )
        validation = pd.DataFrame({"sequence": ["ACDEAC"]})
        build_matrices = getattr(src.train, "_build_feature_matrices")

        train_matrix, validation_matrix, resources, transformer = build_matrices(
            training,
            validation,
            {
                "type": "kmer_tfidf_blocks",
                "blocks": [
                    {
                        "name": "short",
                        "k_min": 3,
                        "k_max": 3,
                        "min_df": 1,
                        "max_features": 5,
                        "sublinear_tf": True,
                    },
                    {
                        "name": "long",
                        "k_min": 4,
                        "k_max": 5,
                        "min_df": 1,
                        "max_features": 7,
                        "sublinear_tf": True,
                    },
                ],
            },
        )

        self.assertTrue(resources["sparse"])
        self.assertEqual(set(resources["block_vocabulary_sizes"]), {"short", "long"})
        self.assertEqual(
            resources["vocabulary_size"],
            sum(resources["block_vocabulary_sizes"].values()),
        )
        self.assertEqual(train_matrix.shape[1], resources["vocabulary_size"])
        self.assertEqual(validation_matrix.shape[1], train_matrix.shape[1])
        self.assertEqual(transformer.transform(["AAAAAA"]).shape[1], train_matrix.shape[1])


if __name__ == "__main__":
    unittest.main()
