"""Diagnose support-stratified OOF calibration on an existing score matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import label_columns
from .metrics import macro_f1_skip_empty, macro_roc_auc_skip_degenerate
from .prior_calibration import fit_support_calibrators, transform_support_calibrated_scores
from .thresholds import crossfit_shrunk_threshold_score


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _split_indices(size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if size < 4:
        raise ValueError("at least four validation rows are required")
    first, second = np.array_split(np.random.default_rng(seed).permutation(size), 2)
    return first, second


def run_calibration(config_path: str | Path, *, project_root: str | Path | None = None) -> Path:
    root = Path(project_root or Path.cwd())
    config = json.loads(_resolve(root, config_path).read_text(encoding="utf-8"))
    output_prefix = _resolve(root, config["output_prefix"])
    summary_path = output_prefix.with_name(output_prefix.name + "-summary.json")
    if summary_path.exists():
        raise FileExistsError("calibration outputs already exist; use a new experiment ID")
    with np.load(_resolve(root, config["scores_path"]), allow_pickle=False) as saved:
        scores = saved["validation_scores"].astype(np.float32)
        validation_ids = saved["validation_ids"].astype(str)
        labels = saved["label_columns"].astype(str).tolist()
    train = pd.read_csv(_resolve(root, config["train_path"]))
    train["protein_id"] = train["protein_id"].astype(str)
    indexed = train.set_index("protein_id")
    target = indexed.loc[validation_ids, labels].to_numpy(dtype=np.uint8)
    training_ids = pd.read_csv(_resolve(root, config["training_ids"]))["protein_id"].astype(str)
    support = indexed.loc[training_ids, labels].sum(axis=0).to_numpy(dtype=np.int64)
    if scores.shape != target.shape:
        raise ValueError("score and target matrices do not align")
    threshold_seeds = [int(seed) for seed in config.get("threshold_seeds", [42])]
    calibration_seeds = [int(seed) for seed in config.get("calibration_seeds", [42])]
    shrinkages = [float(value) for value in config.get("shrinkages", [25.0, 50.0])]
    rows = []
    for calibration_seed in calibration_seeds:
        first, second = _split_indices(len(target), calibration_seed)
        calibrated = np.zeros_like(scores, dtype=np.float32)
        for fitting, evaluation in ((first, second), (second, first)):
            model = fit_support_calibrators(
                scores[fitting], target[fitting], support,
                support_bins=tuple(config.get("support_bins", [20, 100, 1000])),
            )
            calibrated[evaluation] = transform_support_calibrated_scores(
                scores[evaluation], model, support
            )
        for shrinkage in shrinkages:
            for threshold_seed in threshold_seeds:
                result = crossfit_shrunk_threshold_score(
                    target, calibrated, seed=threshold_seed, shrinkage=shrinkage
                )
                rows.append({
                    "calibration_seed": calibration_seed,
                    "threshold_seed": threshold_seed,
                    "shrinkage": shrinkage,
                    "macro_f1": float(result["macro_f1"]),
                    "predicted_positive_rate": float(result["predicted_positive_rate"]),
                    "continuous_macro_auc": macro_roc_auc_skip_degenerate(target, calibrated),
                })
    frame = pd.DataFrame(rows)
    grouped = frame.groupby("shrinkage", as_index=False).agg(
        mean_macro_f1=("macro_f1", "mean"),
        std_macro_f1=("macro_f1", "std"),
        min_macro_f1=("macro_f1", "min"),
        mean_predicted_positive_rate=("predicted_positive_rate", "mean"),
        continuous_macro_auc=("continuous_macro_auc", "mean"),
    ).sort_values(["mean_macro_f1", "min_macro_f1"], ascending=False, ignore_index=True)
    grouped = grouped.rename(columns={
        "mean_macro_f1": "mean_crossfit_macro_f1",
        "std_macro_f1": "std_crossfit_macro_f1",
        "min_macro_f1": "min_crossfit_macro_f1",
    })
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_prefix.with_name(output_prefix.name + "-results.csv"), index=False)
    summary = {
        "experiment_id": config["experiment_id"],
        "validation_rows": len(target),
        "label_count": len(labels),
        "support_bins": config.get("support_bins", [20, 100, 1000]),
        "baseline": {
            "macro_f1": float(np.mean([
                crossfit_shrunk_threshold_score(target, scores, seed=seed, shrinkage=25.0)["macro_f1"]
                for seed in threshold_seeds
            ])),
            "continuous_macro_auc": macro_roc_auc_skip_degenerate(target, scores),
        },
        "leaderboard": grouped.to_dict(orient="records"),
        "outputs": {"results": str(output_prefix.with_name(output_prefix.name + "-results.csv"))},
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(run_calibration(args.config))


if __name__ == "__main__":
    main()
