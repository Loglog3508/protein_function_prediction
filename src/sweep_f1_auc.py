"""Select the third-iteration blend with Macro F1 primary and AUC secondary."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config
from .metrics import macro_f1_skip_empty, macro_roc_auc_skip_degenerate
from .thresholds import (
    crossfit_shrunk_threshold_score,
    fit_shrunk_thresholds,
    reorder_label_thresholds,
)
from .train import _resolve


def aggregate_seed_results(rows: pd.DataFrame) -> pd.DataFrame:
    """Aggregate repeated cross-fit results and rank F1 before AUC."""
    grouped = (
        rows.groupby(["candidate", "primary_weight", "shrinkage"], as_index=False)
        .agg(
            mean_crossfit_macro_f1=("crossfit_macro_f1", "mean"),
            std_crossfit_macro_f1=("crossfit_macro_f1", "std"),
            min_crossfit_macro_f1=("crossfit_macro_f1", "min"),
            continuous_macro_auc=("continuous_macro_auc", "mean"),
            mean_predicted_positive_rate=("predicted_positive_rate", "mean"),
            seed_count=("seed", "nunique"),
        )
        .fillna({"std_crossfit_macro_f1": 0.0})
    )
    return grouped.sort_values(
        [
            "mean_crossfit_macro_f1",
            "continuous_macro_auc",
            "std_crossfit_macro_f1",
            "shrinkage",
            "primary_weight",
        ],
        ascending=[False, False, True, True, False],
        ignore_index=True,
    )


def run_f1_auc_sweep(
    config_path: str | Path, *, project_root: str | Path | None = None
) -> Path:
    """Evaluate saved SGD/KNN scores using the competition's F1-first objective."""
    root = Path(project_root) if project_root is not None else Path.cwd()
    config = load_config(config_path)
    run_dir = _resolve(root, config["output_dir"])
    metrics_prefix = _resolve(root, config["metrics_prefix"])
    leaderboard_path = metrics_prefix.with_name(metrics_prefix.name + "-leaderboard.csv")
    seed_results_path = metrics_prefix.with_name(
        metrics_prefix.name + "-seed-results.csv"
    )
    summary_path = metrics_prefix.with_name(metrics_prefix.name + "-summary.json")
    if run_dir.exists() or leaderboard_path.exists() or summary_path.exists():
        raise FileExistsError("experiment outputs already exist; use a new experiment ID")
    run_dir.mkdir(parents=True)
    metrics_prefix.parent.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    with np.load(_resolve(root, config["primary_scores"]), allow_pickle=False) as saved:
        primary_scores = saved["validation_scores"].astype(np.float32)
        validation_ids = saved["validation_ids"].astype(str)
        labels = saved["label_columns"].astype(str).tolist()
    with np.load(_resolve(root, config["knn_scores"]), allow_pickle=False) as saved:
        knn_scores = saved["validation_scores"].astype(np.float32)
        if not np.array_equal(validation_ids, saved["validation_ids"].astype(str)):
            raise ValueError("primary and KNN validation IDs do not match")
        if labels != saved["label_columns"].astype(str).tolist():
            raise ValueError("primary and KNN label columns do not match")
    if primary_scores.shape != knn_scores.shape:
        raise ValueError("primary and KNN score shapes do not match")

    train_path = _resolve(root, config["data"]["train_path"])
    train = pd.read_csv(train_path, usecols=["protein_id", *labels])
    destination_labels = [
        column for column in pd.read_csv(train_path, nrows=0).columns
        if str(column).startswith("label_")
    ]
    indexed = train.set_index("protein_id", drop=False)
    if not pd.Index(validation_ids).isin(indexed.index).all():
        raise ValueError("saved validation IDs are absent from training data")
    target = indexed.loc[validation_ids, labels].to_numpy(dtype=np.uint8)

    candidates = np.asarray(config["thresholds"]["candidates"], dtype=np.float32)
    rows = []
    score_candidates: dict[str, np.ndarray] = {}
    seeds = [int(value) for value in config.get("seeds", [config["seed"]])]
    for weight_value in config["primary_weights"]:
        primary_weight = float(weight_value)
        if not 0 <= primary_weight <= 1:
            raise ValueError("primary weights must be between zero and one")
        scores = (
            primary_weight * primary_scores
            + (1.0 - primary_weight) * knn_scores
        ).astype(np.float32)
        score_name = f"primary-{primary_weight:.2f}"
        score_candidates[score_name] = scores
        for shrinkage_value in config["thresholds"]["shrinkages"]:
            shrinkage = float(shrinkage_value)
            for seed in seeds:
                result = crossfit_shrunk_threshold_score(
                    target,
                    scores,
                    seed=seed,
                    shrinkage=shrinkage,
                    candidates=candidates,
                )
                rows.append(
                    {
                        "candidate": score_name,
                        "primary_weight": primary_weight,
                        "shrinkage": shrinkage,
                        "seed": seed,
                        "crossfit_macro_f1": float(result["macro_f1"]),
                        "continuous_macro_auc": macro_roc_auc_skip_degenerate(
                            target, scores
                        ),
                        "fixed_0.5_macro_f1": macro_f1_skip_empty(
                            target, (scores >= 0.5).astype(np.uint8)
                        ),
                        "predicted_positive_rate": float(
                            result["predicted_positive_rate"]
                        ),
                    }
                )

    seed_results = pd.DataFrame(rows)
    seed_results.to_csv(seed_results_path, index=False)
    leaderboard = aggregate_seed_results(seed_results)
    leaderboard.to_csv(leaderboard_path, index=False)
    best = leaderboard.iloc[0]
    best_name = str(best["candidate"])
    best_scores = score_candidates[best_name]
    best_thresholds, global_threshold = fit_shrunk_thresholds(
        target,
        best_scores,
        shrinkage=float(best["shrinkage"]),
        candidates=candidates,
    )
    np.savez_compressed(
        run_dir / "best-scores.npz",
        validation_scores=best_scores,
        validation_ids=validation_ids.astype(np.str_),
        label_columns=np.asarray(labels, dtype=np.str_),
    )
    ordered_thresholds = reorder_label_thresholds(
        best_thresholds, labels, destination_labels
    )
    threshold_path = metrics_prefix.with_name(metrics_prefix.name + "-thresholds.json")
    threshold_path.write_text(
        json.dumps(
            {
                "experiment_id": config["experiment_id"],
                "strategy": "f1_primary_auc_secondary",
                "global_threshold": global_threshold,
                "shrinkage": float(best["shrinkage"]),
                "candidate_grid": candidates.tolist(),
                "labels": destination_labels,
                "thresholds": ordered_thresholds.tolist(),
                "selected": best.to_dict(),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    summary = {
        "experiment_id": config["experiment_id"],
        "objective": "Macro F1 primary; continuous Macro AUC secondary",
        "validation_rows": len(validation_ids),
        "label_count": len(labels),
        "seeds": seeds,
        "best": best.to_dict(),
        "outputs": {
            "leaderboard": str(leaderboard_path),
            "seed_results": str(seed_results_path),
            "best_scores": str(run_dir / "best-scores.npz"),
            "thresholds": str(threshold_path),
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(run_f1_auc_sweep(args.config))


if __name__ == "__main__":
    main()
