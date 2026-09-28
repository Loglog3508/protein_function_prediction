"""F1-first, AUC-secondary fusion of homology and SGD tail scores."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import macro_roc_auc_skip_degenerate
from .thresholds import crossfit_shrunk_threshold_score, fit_shrunk_thresholds
from .train import _resolve


def align_score_columns(scores: np.ndarray, source_labels: list[str], target_labels: list[str]) -> np.ndarray:
    if (
        scores.ndim != 2
        or scores.shape[1] != len(source_labels)
        or len(set(source_labels)) != len(source_labels)
        or len(set(target_labels)) != len(target_labels)
        or set(source_labels) != set(target_labels)
    ):
        raise ValueError("score label columns must be unique and match by name")
    positions = {label: index for index, label in enumerate(source_labels)}
    return scores[:, [positions[label] for label in target_labels]]


def select_f1_auc_tradeoff(leaderboard: pd.DataFrame, *, f1_tolerance: float) -> pd.Series:
    """Choose highest AUC among candidates effectively tied on Macro F1."""
    if leaderboard.empty or f1_tolerance < 0:
        raise ValueError("leaderboard must be nonempty and F1 tolerance non-negative")
    best_f1 = leaderboard["mean_crossfit_macro_f1"].max()
    eligible = leaderboard.loc[
        leaderboard["mean_crossfit_macro_f1"] >= best_f1 - f1_tolerance
    ]
    return eligible.sort_values(
        ["continuous_macro_auc", "mean_crossfit_macro_f1"],
        ascending=[False, False],
    ).iloc[0]


def run_homology_sgd_blend(config_path: str | Path, *, project_root: str | Path | None = None) -> Path:
    root = Path(project_root) if project_root is not None else Path.cwd()
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    run_dir = _resolve(root, config["output_dir"])
    prefix = _resolve(root, config["metrics_prefix"])
    summary_path = prefix.with_name(prefix.name + "-summary.json")
    if run_dir.exists() or summary_path.exists():
        raise FileExistsError("experiment outputs already exist; use a new experiment ID")

    with np.load(_resolve(root, config["homology_scores"]), allow_pickle=False) as saved:
        homology_scores = saved["alignment_scores"].astype(np.float32)
        ids = saved["validation_ids"].astype(str)
        labels = saved["label_columns"].astype(str).tolist()
    with np.load(_resolve(root, config["sgd_scores"]), allow_pickle=False) as saved:
        sgd_scores = saved["validation_scores"].astype(np.float32)
        sgd_ids = saved["validation_ids"].astype(str)
        sgd_labels = saved["label_columns"].astype(str).tolist()
    if not np.array_equal(ids, sgd_ids) or homology_scores.shape != sgd_scores.shape:
        raise ValueError("homology and SGD score IDs and shapes must match")
    sgd_scores = align_score_columns(sgd_scores, sgd_labels, labels)
    train = pd.read_csv(_resolve(root, config["data"]["train_path"]), usecols=["protein_id", *labels]).set_index("protein_id")
    if not pd.Index(ids).isin(train.index).all():
        raise ValueError("validation IDs must occur in training data")
    target = train.loc[ids, labels].to_numpy(dtype=np.uint8)
    candidates = np.asarray(config["thresholds"]["candidates"], dtype=np.float32)
    rows = []
    for weight in config["sgd_weights"]:
        weight = float(weight)
        scores = ((1 - weight) * homology_scores + weight * sgd_scores).astype(np.float32)
        auc = macro_roc_auc_skip_degenerate(target, scores)
        for shrinkage in config["thresholds"]["shrinkages"]:
            results = [
                crossfit_shrunk_threshold_score(target, scores, seed=int(seed), shrinkage=float(shrinkage), candidates=candidates)
                for seed in config["seeds"]
            ]
            f1 = np.asarray([result["macro_f1"] for result in results])
            rows.append({
                "sgd_weight": weight,
                "shrinkage": float(shrinkage),
                "mean_crossfit_macro_f1": float(f1.mean()),
                "std_crossfit_macro_f1": float(f1.std()),
                "min_crossfit_macro_f1": float(f1.min()),
                "continuous_macro_auc": auc,
                "mean_predicted_positive_rate": float(np.mean([result["predicted_positive_rate"] for result in results])),
            })
    leaderboard = pd.DataFrame(rows).sort_values(
        ["mean_crossfit_macro_f1", "continuous_macro_auc"],
        ascending=[False, False], ignore_index=True,
    )
    best = select_f1_auc_tradeoff(leaderboard, f1_tolerance=float(config["f1_tolerance"]))
    selected = ((1 - best["sgd_weight"]) * homology_scores + best["sgd_weight"] * sgd_scores).astype(np.float32)
    thresholds, global_threshold = fit_shrunk_thresholds(target, selected, shrinkage=float(best["shrinkage"]), candidates=candidates)
    run_dir.mkdir(parents=True)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    leaderboard.to_csv(prefix.with_name(prefix.name + "-leaderboard.csv"), index=False)
    np.savez_compressed(run_dir / "best-scores.npz", validation_scores=selected, validation_ids=ids.astype(np.str_), label_columns=np.asarray(labels, dtype=np.str_))
    prefix.with_name(prefix.name + "-thresholds.json").write_text(json.dumps({
        "experiment_id": config["experiment_id"],
        "labels": labels,
        "thresholds": thresholds.tolist(),
        "global_threshold": global_threshold,
        "shrinkage": float(best["shrinkage"]),
        "sgd_weight": float(best["sgd_weight"]),
        "f1_tolerance": float(config["f1_tolerance"]),
    }, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps({
        "experiment_id": config["experiment_id"],
        "validation_rows": len(ids),
        "label_count": len(labels),
        "f1_tolerance": float(config["f1_tolerance"]),
        "highest_f1": float(leaderboard.iloc[0]["mean_crossfit_macro_f1"]),
        "selected": best.to_dict(),
    }, indent=2), encoding="utf-8")
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(run_homology_sgd_blend(args.config))


if __name__ == "__main__":
    main()
