import pandas as pd
import numpy as np

from src.finalize_rank_fusion import run_rank_fusion, select_rank_fusion_candidate


def test_select_rank_fusion_candidate_prioritizes_f1_under_auc_gate():
    leaderboard = pd.DataFrame(
        [
            {"new_weight": 0.10, "shrinkage": 10.0, "mean_crossfit_macro_f1": 0.34, "continuous_macro_auc": 0.82},
            {"new_weight": 0.20, "shrinkage": 10.0, "mean_crossfit_macro_f1": 0.35, "continuous_macro_auc": 0.81},
            {"new_weight": 0.30, "shrinkage": 10.0, "mean_crossfit_macro_f1": 0.33, "continuous_macro_auc": 0.84},
        ]
    )

    selected = select_rank_fusion_candidate(leaderboard, minimum_auc=0.81)

    assert selected["new_weight"] == 0.20


def test_run_rank_fusion_writes_valid_submission(tmp_path):
    labels = ["label_0", "label_1"]
    train_ids = [f"p{i}" for i in range(8)]
    test_ids = [f"t{i}" for i in range(4)]
    target = np.tile(np.array([[1, 0], [0, 1], [0, 0], [1, 1]], dtype=np.uint8), (2, 1))
    base_oof = np.where(target, 0.8, 0.2).astype(np.float32)
    new_oof = np.where(target, 0.7, 0.3).astype(np.float32)
    base_test = np.array([[0.8, 0.2], [0.2, 0.8], [0.2, 0.2], [0.8, 0.8]], dtype=np.float32)
    new_test = base_test.copy()
    train = pd.DataFrame({"protein_id": train_ids, **{label: target[:, index] for index, label in enumerate(labels)}})
    train.to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame({"protein_id": test_ids}).to_csv(tmp_path / "test.csv", index=False)

    def save(path, key, ids, values):
        np.savez_compressed(path, **{key: values, key.replace("scores", "ids"): np.asarray(ids), "label_columns": np.asarray(labels)})

    save(tmp_path / "base_oof.npz", "validation_scores", train_ids, base_oof)
    save(tmp_path / "new_oof.npz", "validation_scores", train_ids, new_oof)
    save(tmp_path / "base_test.npz", "test_scores", test_ids, base_test)
    save(tmp_path / "new_test.npz", "test_scores", test_ids, new_test)
    config = {
        "experiment_id": "test-rank-fusion",
        "data": {"train_path": "train.csv", "test_path": "test.csv"},
        "base_oof": "base_oof.npz",
        "new_oof": "new_oof.npz",
        "base_test": "base_test.npz",
        "new_test": "new_test.npz",
        "weights": [0.0, 0.5],
        "shrinkages": [0.0],
        "seeds": [17],
        "minimum_auc": 0.5,
        "output_dir": "run",
        "submission_path": "submission/submit_template_v1.csv",
        "metrics_prefix": "metrics/rank-fusion",
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(__import__("json").dumps(config), encoding="utf-8")

    summary_path = run_rank_fusion(config_path, project_root=tmp_path)

    assert summary_path.exists()
    submission = pd.read_csv(tmp_path / "submission/submit_template_v1.csv")
    assert submission.columns.tolist() == ["protein_id", *labels]
    assert submission[labels].isin([0, 1]).all().all()
