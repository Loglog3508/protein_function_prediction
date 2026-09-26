"""Evaluate test-like tail blends of SGD and sequence-neighbor scores."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from .config import load_config
from .knn import weighted_knn_label_scores
from .metrics import macro_f1_skip_empty, macro_roc_auc_skip_degenerate
from .shift import tail_distribution_mask
from .sweep_shift import _stack_features
from .thresholds import (
    crossfit_shrunk_auc_threshold_score,
    crossfit_shrunk_threshold_score,
)
from .train import _build_feature_matrices, _resolve


def run_tail_knn_blend_sweep(
    config_path: str | Path, *, project_root: str | Path | None = None
) -> Path:
    """Build tail OOF neighbor scores and compare blends with primary scores."""
    root = Path(project_root) if project_root is not None else Path.cwd()
    config = load_config(config_path)
    run_dir = _resolve(root, config["output_dir"])
    metrics_prefix = _resolve(root, config["metrics_prefix"])
    leaderboard_path = metrics_prefix.with_name(
        metrics_prefix.name + "-leaderboard.csv"
    )
    summary_path = metrics_prefix.with_name(metrics_prefix.name + "-summary.json")
    if run_dir.exists() or leaderboard_path.exists() or summary_path.exists():
        raise FileExistsError("experiment outputs already exist; use a new experiment ID")
    run_dir.mkdir(parents=True)
    metrics_prefix.parent.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    primary_path = _resolve(root, config["primary_scores"])
    with np.load(primary_path, allow_pickle=False) as saved:
        primary_scores = saved["validation_scores"].astype(np.float32)
        validation_ids = saved["validation_ids"].astype(str)
        labels = saved["label_columns"].astype(str).tolist()

    train_path = _resolve(root, config["data"]["train_path"])
    train = pd.read_csv(
        train_path,
        usecols=["protein_id", "sequence", *labels],
        dtype={label: np.uint8 for label in labels},
    )
    cutoff = int(config["distribution"]["cutoff"])
    folds = int(config["distribution"].get("folds", 2))
    tail_mask = tail_distribution_mask(train["protein_id"].tolist(), cutoff=cutoff)
    early = train.loc[~tail_mask].reset_index(drop=True)
    tail = train.loc[tail_mask].reset_index(drop=True)
    if tail["protein_id"].tolist() != validation_ids.tolist():
        raise ValueError("primary score IDs do not match the labeled tail order")
    if primary_scores.shape != (len(tail), len(labels)):
        raise ValueError("primary score shape does not match tail rows and labels")
    early_target = early[labels].to_numpy(dtype=np.uint8)
    tail_target = tail[labels].to_numpy(dtype=np.uint8)

    feature_started = time.perf_counter()
    early_features, tail_features, resources, _ = _build_feature_matrices(
        early, tail, config["features"]
    )
    feature_seconds = time.perf_counter() - feature_started
    splitter = MultilabelStratifiedKFold(
        n_splits=folds, shuffle=True, random_state=config["seed"]
    )
    fold_indices = list(splitter.split(np.zeros(len(tail)), tail_target))

    knn_config = config["knn"]
    knn_scores = np.zeros_like(primary_scores, dtype=np.float32)
    knn_started = time.perf_counter()
    for tail_train_index, tail_eval_index in fold_indices:
        training_features = _stack_features(
            early_features, tail_features[tail_train_index]
        )
        training_target = np.vstack(
            [early_target, tail_target[tail_train_index]]
        )
        knn_scores[tail_eval_index] = weighted_knn_label_scores(
            training_features,
            training_target,
            tail_features[tail_eval_index],
            n_neighbors=int(knn_config["n_neighbors"]),
            similarity_power=float(knn_config.get("similarity_power", 1.0)),
            query_batch_size=int(knn_config.get("query_batch_size", 64)),
        )
    knn_seconds = time.perf_counter() - knn_started
    np.savez_compressed(
        run_dir / "knn-scores.npz",
        validation_scores=knn_scores,
        validation_ids=validation_ids.astype(np.str_),
        label_columns=np.asarray(labels, dtype=np.str_),
    )

    shrinkage = float(config.get("thresholds", {}).get("shrinkage", 25.0))
    rows = []
    score_candidates = {}
    for primary_weight_value in config["primary_weights"]:
        primary_weight = float(primary_weight_value)
        if not 0 <= primary_weight <= 1:
            raise ValueError("primary weights must be between zero and one")
        name = f"primary-{primary_weight:.2f}"
        scores = (
            primary_weight * primary_scores
            + (1.0 - primary_weight) * knn_scores
        ).astype(np.float32)
        score_candidates[name] = scores
        auc_result = crossfit_shrunk_auc_threshold_score(
            tail_target,
            scores,
            seed=config["seed"],
            shrinkage=shrinkage,
        )
        f1_result = crossfit_shrunk_threshold_score(
            tail_target,
            scores,
            seed=config["seed"],
            shrinkage=shrinkage,
        )
        fixed = (scores >= 0.5).astype(np.uint8)
        rows.append(
            {
                "candidate": name,
                "primary_weight": primary_weight,
                "continuous_macro_auc": macro_roc_auc_skip_degenerate(
                    tail_target, scores
                ),
                "fixed_binary_macro_auc": macro_roc_auc_skip_degenerate(
                    tail_target, fixed
                ),
                "crossfit_binary_macro_auc": float(auc_result["macro_auc"]),
                "fixed_macro_f1": macro_f1_skip_empty(tail_target, fixed),
                "crossfit_macro_f1": float(f1_result["macro_f1"]),
                "predicted_positive_rate": float(
                    auc_result["predicted_positive_rate"]
                ),
            }
        )
    leaderboard = pd.DataFrame(rows).sort_values(
        ["crossfit_binary_macro_auc", "continuous_macro_auc", "candidate"],
        ascending=[False, False, True],
        ignore_index=True,
    )
    leaderboard.to_csv(leaderboard_path, index=False)
    best_name = str(leaderboard.iloc[0]["candidate"])
    np.savez_compressed(
        run_dir / "best-scores.npz",
        validation_scores=score_candidates[best_name],
        validation_ids=validation_ids.astype(np.str_),
        label_columns=np.asarray(labels, dtype=np.str_),
    )
    summary = {
        "experiment_id": config["experiment_id"],
        "primary_scores": str(primary_path),
        "cutoff": cutoff,
        "folds": folds,
        "early_rows": len(early),
        "tail_rows": len(tail),
        "label_count": len(labels),
        "feature_seconds": feature_seconds,
        "knn_seconds": knn_seconds,
        "feature_resources": resources,
        "best_candidate": leaderboard.iloc[0].to_dict(),
        "outputs": {
            "leaderboard": str(leaderboard_path),
            "knn_scores": str(run_dir / "knn-scores.npz"),
            "best_scores": str(run_dir / "best-scores.npz"),
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
    print(run_tail_knn_blend_sweep(args.config))


if __name__ == "__main__":
    main()
