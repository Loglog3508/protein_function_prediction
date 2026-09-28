"""Submission generation from an alignment-neighbor model."""

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.finalize_homology import run_final_homology


def test_final_homology_writes_full_binary_submission_without_overwriting():
    with tempfile.TemporaryDirectory(dir="artifacts/metrics") as temporary:
        root = Path(temporary)
        train = pd.DataFrame(
            {
                "protein_id": ["P000001", "P000002", "P000003", "P000004"],
                "sequence": ["AAAAAC", "RRRRRC", "AAAACC", "RRRRCC"],
                "label_0": [1, 0, 1, 0],
                "label_1": [0, 1, 0, 1],
            }
        )
        test = pd.DataFrame(
            {"protein_id": ["P000005", "P000006"], "sequence": ["AAAAAA", "RRRRRR"]}
        )
        train.to_csv(root / "train.csv", index=False)
        test.to_csv(root / "test.csv", index=False)
        (root / "thresholds.json").write_text(
            json.dumps({"labels": ["label_0", "label_1"], "thresholds": [0.5, 0.5]}),
            encoding="utf-8",
        )
        config = {
            "experiment_id": "test-homology-final",
            "seed": 42,
            "data": {"train_path": str(root / "train.csv"), "test_path": str(root / "test.csv")},
            "features": {"type": "kmer_tfidf", "k_min": 1, "k_max": 2, "min_df": 1, "max_features": 100, "sublinear_tf": True},
            "neighbors": {"n_neighbors": 2, "query_batch_size": 1, "max_length": 32, "alignment_power": 2.0},
            "distribution": {"cutoff": 3, "tail_weight": 2.0},
            "model": {"type": "sgd", "class_weight": "balanced", "alpha": 0.000005, "max_iter": 50, "tol": 0.001},
            "thresholds_path": str(root / "thresholds.json"),
            "submission_path": str(root / "submit_template_v1.csv"),
            "metadata_path": str(root / "metadata.json"),
        }
        (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
        run_final_homology(root / "config.json")
        submission = pd.read_csv(root / "submit_template_v1.csv")
        assert submission.columns.tolist() == ["protein_id", "label_0", "label_1"]
        assert submission["protein_id"].tolist() == test["protein_id"].tolist()
        assert set(submission[["label_0", "label_1"]].to_numpy().ravel()) <= {0, 1}
        with pytest.raises(FileExistsError):
            run_final_homology(root / "config.json")
