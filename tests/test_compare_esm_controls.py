import importlib
import json

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from src.rank_fusion import crossfit_rank_fusion


def test_control_comparison_uses_continuous_scores_and_existing_calibration():
    compare = importlib.import_module("src.compare_esm_controls")
    target = np.array([[1, 0], [0, 1], [1, 0], [0, 1], [0, 0], [1, 1]], dtype=np.uint8)
    scores = np.array([[.8, .2], [.3, .7], [.6, .4], [.2, .6], [.4, .3], [.7, .8]], dtype=np.float32)
    result, labels = compare.summarize_scores(target, scores, ["label_0", "label_1"])
    assert result["macro_auc"] == pytest.approx(np.mean([roc_auc_score(target[:, index], scores[:, index]) for index in range(2)]))
    assert result["macro_ap"] == pytest.approx(np.mean([average_precision_score(target[:, index], scores[:, index]) for index in range(2)]))
    assert result["default05"]["predicted_positive_rate"] == pytest.approx((scores >= .5).mean())
    assert len(result["seeds"]) == 5
    assert len(labels) == 4
    assert result["crossfit"]["net_excess"] == pytest.approx(result["crossfit"]["fp"] - result["crossfit"]["fn"])
    for seed in result["seeds"]:
        assert seed["true_positive_rate"] == pytest.approx(target.mean())
        assert 0 <= seed["positive_excess"] <= seed["predicted_positive"]
        assert 0 <= seed["positive_deficit"] <= seed["true_positive"]
        assert 0 <= seed["top10_positive_excess_share"] <= 1
        assert seed["positive_excess"] - seed["positive_deficit"] == pytest.approx(seed["fp"] - seed["fn"])


def test_control_comparison_rejects_invalid_scores_before_calibration():
    compare = importlib.import_module("src.compare_esm_controls")
    target = np.array([[1], [0]], dtype=np.uint8)
    for scores in [np.array([[np.nan], [.3]]), np.array([[1.1], [.3]]), np.array([[.3]])]:
        with pytest.raises(ValueError):
            compare.summarize_scores(target, scores, ["label_0"])


@pytest.fixture
def saved_control(tmp_path):
    compare = importlib.import_module("src.compare_esm_controls")
    target = np.array([[1, 0], [0, 1], [1, 0], [0, 1], [0, 0], [1, 1]], dtype=np.uint8)
    baseline_scores = np.array([[.8, .2], [.3, .7], [.6, .4], [.2, .6], [.4, .3], [.7, .8]], dtype=np.float32)
    ids, labels = [f"protein_{index}" for index in range(6)], ["label_0", "label_1"]
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    import pandas as pd
    pd.DataFrame({"protein_id": ids, "label_0": target[:, 0], "label_1": target[:, 1]}).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"protein_id": ids}).to_csv(data_dir / "ids.csv", index=False)
    paths = {}
    for role, scores in [("baseline", baseline_scores), ("candidate", baseline_scores + .01)]:
        run_dir = tmp_path / role / "epochs"
        run_dir.mkdir(parents=True)
        config = {"experiment_id": role, "output_dir": role, "training": {"epochs": 6},
                  "data": {"train_path": "data/train.csv"}, "split": {"validation_ids": "data/ids.csv"}}
        paths[role] = tmp_path / f"{role}.json"
        paths[role].write_text(json.dumps(config), encoding="utf-8")
        np.savez_compressed(run_dir / "epoch06-validation_scores.npz", epoch=6,
                            validation_ids=np.asarray(ids), label_columns=np.asarray(labels), validation_scores=scores)
    seeds = crossfit_rank_fusion(target, baseline_scores, baseline_scores, weights=[0.0], shrinkages=[25.0], seeds=list(compare.SEEDS))
    reference = {"experiment_id": "baseline",
                 "score_provenance": [{"epoch": 6, "sha256": compare.digest(tmp_path / "baseline/epochs/epoch06-validation_scores.npz")}],
                 "history": [{"epoch": 6, "validation_macro_auc": .0,
                              "mean_crossfit_macro_f1": float(seeds.seed_macro_f1.mean()),
                              "mean_crossfit_predicted_positive_rate": float(seeds.seed_predicted_positive_rate.mean())}]}
    reference["history"][0]["validation_macro_auc"] = float(np.mean([roc_auc_score(target[:, index], baseline_scores[:, index]) for index in range(2)]))
    paths["reference"] = tmp_path / "reference.json"
    paths["reference"].write_text(json.dumps(reference), encoding="utf-8")
    return compare, paths


def test_saved_epoch_comparison_verifies_identity_metrics_and_refuses_overwrite(tmp_path, saved_control):
    compare, paths = saved_control
    prefix = tmp_path / "metrics/control"
    result = compare.evaluate_epoch(tmp_path, paths["candidate"], paths["baseline"], paths["reference"], 6, prefix)
    assert result["epoch"] == 6
    assert result["delta"]["macro_auc"] == pytest.approx(0)
    assert result["delta"]["macro_ap"] == pytest.approx(0)
    assert not result["gates"]["ap_above_same_epoch_asl"]
    assert len(result["provenance"]["score_sha256"]["candidate"]) == 64
    summary_path = tmp_path / result["outputs"]["summary"]
    old = summary_path.read_bytes()
    with pytest.raises(FileExistsError):
        compare.evaluate_epoch(tmp_path, paths["candidate"], paths["baseline"], paths["reference"], 6, prefix)
    assert summary_path.read_bytes() == old


def test_saved_epoch_comparison_rejects_changed_historical_baseline(tmp_path, saved_control):
    compare, paths = saved_control
    reference = json.loads(paths["reference"].read_text())
    reference["score_provenance"][0]["sha256"] = "0" * 64
    paths["reference"].write_text(json.dumps(reference), encoding="utf-8")
    with pytest.raises(ValueError, match="historical ASL"):
        compare.evaluate_epoch(tmp_path, paths["candidate"], paths["baseline"], paths["reference"], 6, tmp_path / "metrics/control")
    assert not (tmp_path / "metrics").exists()
