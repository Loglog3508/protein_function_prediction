import hashlib
import json
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score, roc_auc_score

from src.diagnose_esm_epochs import run_diagnostics
from src.rank_fusion import crossfit_rank_fusion, fit_rank_fusion_thresholds


SEEDS = (17, 31, 42, 73, 101)
EPOCHS = [6, 8]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def independent_counts(target, predictions):
    true_positive = ((target == 1) & (predictions == 1)).sum(axis=0)
    predicted = predictions.sum(axis=0)
    return {
        "predicted_positive": predicted,
        "tp": true_positive,
        "fp": predicted - true_positive,
        "fn": target.sum(axis=0) - true_positive,
        "f1": f1_score(target, predictions, average=None, zero_division=0),
    }


@pytest.fixture
def synthetic_run(tmp_path):
    labels = [f"label_{index}" for index in range(6)]
    target = np.array([
        [1, 0, 1, 1, 1, 0], [0, 0, 1, 0, 1, 1],
        [1, 0, 1, 0, 1, 1], [0, 0, 1, 0, 1, 0],
        [1, 0, 1, 1, 0, 0], [0, 0, 1, 0, 0, 1],
        [1, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 1],
        [1, 0, 1, 1, 0, 0], [0, 0, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 1], [0, 0, 1, 0, 0, 0],
    ], dtype=np.uint8)
    training_support = np.array([6, 1, 10, 3, 0, 8])
    training_target = (np.arange(12)[:, None] < training_support).astype(np.uint8)
    training_ids = np.array([f"train_{index}" for index in range(12)])
    validation_ids = np.array([f"validation_{index}" for index in range(12)])[::-1]
    frame = pd.DataFrame(np.vstack([training_target, target]), columns=labels)
    frame.insert(0, "sequence", "ACDEFG")
    frame.insert(0, "protein_id", np.concatenate([training_ids, validation_ids]))
    frame.sample(frac=1, random_state=42).to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame({"protein_id": training_ids}).to_csv(tmp_path / "train_ids.csv", index=False)
    pd.DataFrame({"protein_id": validation_ids}).to_csv(tmp_path / "validation_ids.csv", index=False)
    run_dir = tmp_path / "saved_run"
    (run_dir / "epochs").mkdir(parents=True)
    config = {
        "output_dir": "saved_run", "data": {"train_path": "train.csv"},
        "split": {"train_ids": "train_ids.csv", "validation_ids": "validation_ids.csv"},
    }
    config_path = tmp_path / "config.json"
    reference_path = tmp_path / "reference.json"
    write_json(config_path, config)
    summary_path = run_dir / "summary.json"
    write_json(summary_path, {
        "experiment_id": "synthetic-saved-esm", "validation_rows": len(target),
        "train_rows": len(training_ids), "label_count": len(labels),
    })
    reference = {
        "experiment_id": "synthetic-saved-esm", "source_summary_sha256": sha256(summary_path),
        "history": [], "score_provenance": [],
    }
    generator = np.random.default_rng(42)
    saved_scores, predictions, score_paths = {}, {}, {}
    for epoch in EPOCHS:
        scores = generator.uniform(.05, .95, target.shape).astype(np.float32)
        scores[:, 1] = np.linspace(.65, .85, len(target), dtype=np.float32)
        scores[target[:, 4] == 0, 4] = .25 if epoch == 6 else .65
        scores[target[:, 4] == 1, 4] = [.7, .8, .3, .6] if epoch == 6 else [.8, .3, .7, .1]
        saved_scores[epoch] = scores
        path = run_dir / "epochs" / f"epoch{epoch:02d}-validation_scores.npz"
        np.savez_compressed(path, epoch=np.array(epoch), validation_ids=validation_ids,
                            label_columns=np.array(labels), validation_scores=scores)
        score_paths[epoch] = path
        protocol = crossfit_rank_fusion(
            target, scores, scores, weights=[0.0], shrinkages=[25.0], seeds=SEEDS,
        ).groupby("seed", sort=False).first()
        seed_predictions = []
        for seed in SEEDS:
            first, second = np.array_split(np.random.default_rng(seed).permutation(len(target)), 2)
            predicted = np.zeros_like(target)
            for selection, evaluation in ((first, second), (second, first)):
                thresholds, _, _ = fit_rank_fusion_thresholds(target[selection], scores[selection], shrinkage=25.0)
                predicted[evaluation] = scores[evaluation] >= thresholds
            seed_predictions.append(predicted)
        predictions[epoch] = seed_predictions
        auc = np.mean([
            roc_auc_score(target[:, index], scores[:, index])
            for index in range(len(labels)) if 0 < target[:, index].sum() < len(target)
        ])
        reference["history"].append({
            "epoch": epoch, "validation_macro_auc": float(auc),
            "validation_macro_f1": float(independent_counts(target, scores >= .5)["f1"][target.sum(axis=0) > 0].mean()),
            "default_predicted_positive_rate": float((scores >= .5).mean()),
            "mean_crossfit_macro_f1": float(protocol.seed_macro_f1.mean()),
            "mean_crossfit_predicted_positive_rate": float(protocol.seed_predicted_positive_rate.mean()),
        })
        reference["score_provenance"].append({"epoch": epoch, "sha256": sha256(path)})
    write_json(reference_path, reference)
    return {
        "root": tmp_path, "config": config_path, "reference_path": reference_path,
        "reference": reference, "summary_path": summary_path, "score_paths": score_paths,
        "target": target, "training_support": training_support, "labels": labels,
        "scores": saved_scores, "predictions": predictions,
        "prefix": tmp_path / "outputs" / "synthetic-epoch-diagnostics",
    }


def run_synthetic(case):
    return run_diagnostics(case["root"], case["config"], case["reference_path"], case["prefix"], EPOCHS)


def test_run_diagnostics_reproduces_saved_cpu_epochs_and_conserves_counts(synthetic_run):
    case = synthetic_run
    torch_imported_before = "torch" in sys.modules
    sources = [case["config"], case["reference_path"], case["summary_path"], *case["score_paths"].values()]
    before = {path: sha256(path) for path in sources}
    result = run_synthetic(case)
    assert result["runtime"] == {
        "cpu_only": True, "torch_imported": False, "training_started": False, "submission_generated": False,
    }
    assert ("torch" in sys.modules) == torch_imported_before
    assert result["epochs"] == EPOCHS
    assert result["seeds"] == list(SEEDS)
    assert (result["validation_rows"], result["training_rows"], result["label_count"]) == (12, 12, 6)
    assert result["source_summary_sha256"] == before[case["summary_path"]]
    assert result["reference_sha256"] == before[case["reference_path"]]
    for entry in result["score_provenance"]:
        assert entry["sha256"] == before[case["score_paths"][entry["epoch"]]]
    outputs = {key: case["root"] / value for key, value in result["outputs"].items()}
    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs.values())
    assert outputs["curve"].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert json.loads(outputs["summary"].read_text(encoding="utf-8")) == result
    overlap = pd.read_csv(outputs["overlap"])
    assert len(overlap) == len(EPOCHS) * len(case["labels"])
    assert not overlap.duplicated(["epoch", "label"]).any()
    tertiles = np.array(["middle", "low", "high", "middle", "low", "high"])
    target = case["target"]
    support = target.sum(axis=0)
    for epoch in EPOCHS:
        rows = overlap.loc[overlap.epoch == epoch].set_index("label").loc[case["labels"]]
        np.testing.assert_array_equal(rows.training_support, case["training_support"])
        np.testing.assert_array_equal(rows.true_support, support)
        np.testing.assert_array_equal(rows.training_tertile, tertiles)
        for index, label in enumerate(case["labels"]):
            scores = case["scores"][epoch][:, index].astype(np.float64)
            positive, negative = scores[target[:, index] == 1], scores[target[:, index] == 0]
            positive_median = np.median(positive) if len(positive) else np.nan
            negative_median = np.median(negative) if len(negative) else np.nan
            negative_std = negative.std(ddof=0) if len(negative) else np.nan
            gap = (positive_median - negative_median) / negative_std if negative_std > 0 else np.nan
            np.testing.assert_allclose(
                rows.loc[label, ["positive_median", "negative_median", "negative_std", "median_gap_over_negative_std"]].to_numpy(dtype=float),
                [positive_median, negative_median, negative_std, gap], rtol=1e-12, atol=1e-12, equal_nan=True,
            )
        assert rows.loc["label_4", "negative_std"] == 0
        assert rows.loc[["label_1", "label_2", "label_4"], "median_gap_over_negative_std"].isna().all()
    label_rows = pd.read_csv(outputs["labels"])
    seed_rows = pd.read_csv(outputs["seed_labels"])
    strata = pd.read_csv(outputs["strata"])
    assert len(label_rows) == 2 * len(overlap)
    assert len(seed_rows) == len(SEEDS) * len(overlap)
    assert len(pd.read_csv(outputs["thresholds"])) == 2 * len(SEEDS) * len(overlap)
    for epoch in EPOCHS:
        for seed, prediction in zip(SEEDS, case["predictions"][epoch]):
            actual = seed_rows[(seed_rows.epoch == epoch) & (seed_rows.seed == seed)].set_index("label").loc[case["labels"]]
            for column, expected in independent_counts(target, prediction).items():
                np.testing.assert_allclose(actual[column], expected, atol=1e-12)
        for mode in ("default05", "crossfit"):
            predictions = [case["scores"][epoch] >= .5] if mode == "default05" else case["predictions"][epoch]
            counts = [independent_counts(target, prediction) for prediction in predictions]
            actual = label_rows[(label_rows.epoch == epoch) & (label_rows["mode"] == mode)].set_index("label").loc[case["labels"]]
            for column in counts[0]:
                np.testing.assert_allclose(actual[column], np.mean([entry[column] for entry in counts], axis=0), atol=1e-12)
            global_row = next(row for row in result["global"] if row["epoch"] == epoch and row["mode"] == mode)
            expected_f1 = np.mean([entry["f1"][support > 0].mean() for entry in counts])
            assert global_row["macro_f1"] == pytest.approx(expected_f1, abs=1e-12)
            assert global_row["predicted_positive_rate"] == pytest.approx(np.mean([prediction.mean() for prediction in predictions]), abs=1e-12)
            for axis in ("training_stratum", "training_tertile", "validation_stratum"):
                groups = strata[(strata.epoch == epoch) & (strata["mode"] == mode) & (strata.axis == axis)]
                assert groups.label_count.sum() == len(case["labels"])
                assert groups.f1_label_count.sum() == (support > 0).sum()
                assert groups.auc_label_count.sum() == ((support > 0) & (support < len(target))).sum()
                assert (groups.predicted_positive_rate * groups.label_count).sum() / len(case["labels"]) == pytest.approx(global_row["predicted_positive_rate"])
                for column in ("tp", "fp", "fn"):
                    assert groups[column].sum() == pytest.approx(global_row[column])
                if axis == "training_tertile":
                    assert groups.set_index("stratum").label_count.to_dict() == {"low": 2, "middle": 2, "high": 2}
    distributions = pd.read_csv(outputs["overlap_distribution"])
    prevalence = pd.read_csv(outputs["prevalence"])
    assert len(distributions) == len(prevalence) == 4 * len(EPOCHS)
    for epoch in EPOCHS:
        epoch_overlap = overlap[overlap.epoch == epoch].set_index("label").loc[case["labels"]]
        mean_predicted = np.mean([prediction.sum(axis=0) for prediction in case["predictions"][epoch]], axis=0)
        for stratum in ("all", "low", "middle", "high"):
            selected = np.ones(len(support), dtype=bool) if stratum == "all" else tertiles == stratum
            distribution = distributions[(distributions.epoch == epoch) & (distributions.stratum == stratum)].iloc[0]
            gap = epoch_overlap.loc[selected, "median_gap_over_negative_std"].dropna()
            assert distribution.label_count == selected.sum()
            assert distribution.defined_labels == len(gap)
            assert distribution.undefined_labels == selected.sum() - len(gap)
            expected_bins = {
                "gap_le_zero_labels": (gap <= 0).sum(), "gap_zero_to_one_labels": ((gap > 0) & (gap < 1)).sum(),
                "gap_one_to_two_labels": ((gap >= 1) & (gap < 2)).sum(), "gap_ge_two_labels": (gap >= 2).sum(),
            }
            for column, expected in expected_bins.items():
                assert distribution[column] == expected
            assert sum(distribution[column] for column in expected_bins) == distribution.defined_labels
            decomposition = prevalence[(prevalence.epoch == epoch) & (prevalence.stratum == stratum)].iloc[0]
            difference = mean_predicted[selected] - support[selected]
            assert decomposition.label_count == selected.sum()
            assert decomposition.predicted_positive == pytest.approx(mean_predicted[selected].sum())
            assert decomposition.true_positive == support[selected].sum()
            assert decomposition.positive_excess == pytest.approx(np.maximum(difference, 0).sum())
            assert decomposition.positive_deficit == pytest.approx(np.maximum(-difference, 0).sum())
            assert decomposition.net_excess == pytest.approx(decomposition.positive_excess - decomposition.positive_deficit)
            assert decomposition.net_excess == pytest.approx(decomposition.predicted_positive - decomposition.true_positive)
            assert decomposition.predicted_positive == pytest.approx(decomposition.tp + decomposition.fp)
            assert decomposition.true_positive == pytest.approx(decomposition.tp + decomposition.fn)
            assert decomposition.predicted_positive_rate == pytest.approx(decomposition.predicted_positive / (len(target) * selected.sum()))
            assert decomposition.zero_support_predicted_positive == pytest.approx(mean_predicted[selected & (support == 0)].sum())
            assert decomposition.overpredicted_labels == (difference > 0).sum()
            assert decomposition.underpredicted_labels == (difference < 0).sum()
        groups = prevalence[(prevalence.epoch == epoch) & (prevalence.stratum != "all")]
        total = prevalence[(prevalence.epoch == epoch) & (prevalence.stratum == "all")].iloc[0]
        for column in ("label_count", "predicted_positive", "true_positive", "positive_excess", "positive_deficit", "net_excess", "tp", "fp", "fn", "zero_support_predicted_positive"):
            assert groups[column].sum() == pytest.approx(total[column])
    counterfactuals = pd.read_csv(outputs["counterfactuals"])
    np.testing.assert_allclose(counterfactuals.score_change_component + counterfactuals.threshold_refit_component, counterfactuals.total_f1_change, atol=1e-12)
    after_outputs = {path: sha256(path) for path in outputs.values()}
    with pytest.raises(FileExistsError, match="fresh diagnostic output prefix"):
        run_synthetic(case)
    assert after_outputs == {path: sha256(path) for path in outputs.values()}
    assert before == {path: sha256(path) for path in sources}


@pytest.mark.parametrize("corruption, message", [
    ("summary_hash", "Reference does not match"),
    ("score_hash", "Historical epoch scores changed"),
    ("validation_ids", "Saved epoch, ID or label order mismatch"),
    ("label_columns", "Saved epoch, ID or label order mismatch"),
    ("epoch", "Saved epoch, ID or label order mismatch"),
    ("reference_f1", "do not reproduce reference calibration"),
    ("reference_rate", "do not reproduce reference calibration"),
])
def test_run_diagnostics_rejects_changed_provenance_and_protocol(synthetic_run, corruption, message):
    case = synthetic_run
    reference = case["reference"]
    if corruption == "summary_hash":
        reference["source_summary_sha256"] = "0" * 64
    elif corruption == "score_hash":
        reference["score_provenance"][0]["sha256"] = "0" * 64
    elif corruption in ("validation_ids", "label_columns", "epoch"):
        path = case["score_paths"][EPOCHS[0]]
        with np.load(path, allow_pickle=False) as saved:
            values = {key: saved[key].copy() for key in saved.files}
        values[corruption] = np.array(99) if corruption == "epoch" else values[corruption][::-1]
        np.savez_compressed(path, **values)
        reference["score_provenance"][0]["sha256"] = sha256(path)
    else:
        key = "mean_crossfit_macro_f1" if corruption == "reference_f1" else "mean_crossfit_predicted_positive_rate"
        reference["history"][0][key] += .1
    write_json(case["reference_path"], reference)
    with pytest.raises(ValueError, match=message):
        run_synthetic(case)
    assert not case["prefix"].parent.exists()
