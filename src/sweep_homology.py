"""Evaluate alignment-weighted label transfer on the test-like tail."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from .data import label_columns
from .homology import alignment_neighbor_scores
from .knn import weighted_knn_label_scores
from .metrics import macro_roc_auc_skip_degenerate
from .shift import tail_distribution_mask
from .sweep_shift import _stack_features
from .thresholds import crossfit_shrunk_threshold_score, fit_shrunk_thresholds
from .train import _build_feature_matrices, _resolve


def run_tail_homology_sweep(config_path: str | Path, *, project_root: str | Path | None = None) -> Path:
    root = Path(project_root) if project_root is not None else Path.cwd()
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    run_dir = _resolve(root, config["output_dir"])
    prefix = _resolve(root, config["metrics_prefix"])
    summary_path = prefix.with_name(prefix.name + "-summary.json")
    if run_dir.exists() or summary_path.exists():
        raise FileExistsError("experiment outputs already exist; use a new experiment ID")
    run_dir.mkdir(parents=True)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    train_path = _resolve(root, config["data"]["train_path"])
    labels = label_columns(pd.read_csv(train_path, nrows=0).columns)
    train = pd.read_csv(
        train_path,
        usecols=["protein_id", "sequence", *labels],
        dtype={label: np.uint8 for label in labels},
    )
    mask = tail_distribution_mask(train["protein_id"].tolist(), cutoff=int(config["distribution"]["cutoff"]))
    early = train.loc[~mask].reset_index(drop=True)
    tail = train.loc[mask].reset_index(drop=True)
    early_target = early[labels].to_numpy(dtype=np.uint8)
    target = tail[labels].to_numpy(dtype=np.uint8)
    started = time.perf_counter()
    early_features, tail_features, resources, _ = _build_feature_matrices(early, tail, config["features"])
    feature_seconds = time.perf_counter() - started
    splitter = MultilabelStratifiedKFold(n_splits=int(config["distribution"]["folds"]), shuffle=True, random_state=int(config["seed"]))
    cosine_scores = np.zeros(target.shape, dtype=np.float32)
    alignment_scores = np.zeros(target.shape, dtype=np.float32)
    neighbor_config = config["neighbors"]
    started = time.perf_counter()
    for tail_train_index, tail_eval_index in splitter.split(np.zeros(len(tail)), target):
        training_features = _stack_features(early_features, tail_features[tail_train_index])
        training_target = np.vstack([early_target, target[tail_train_index]])
        training_sequences = [*early["sequence"], *tail.iloc[tail_train_index]["sequence"]]
        evaluation_sequences = tail.iloc[tail_eval_index]["sequence"].tolist()
        evaluation_features = tail_features[tail_eval_index]
        cosine_scores[tail_eval_index] = weighted_knn_label_scores(
            training_features, training_target, evaluation_features,
            n_neighbors=int(neighbor_config["n_neighbors"]),
            similarity_power=float(neighbor_config["similarity_power"]),
            query_batch_size=int(neighbor_config["query_batch_size"]),
        )
        alignment_scores[tail_eval_index] = alignment_neighbor_scores(
            training_sequences, training_target, training_features,
            evaluation_sequences, evaluation_features,
            n_neighbors=int(neighbor_config["n_neighbors"]),
            query_batch_size=int(neighbor_config["query_batch_size"]),
            max_length=int(neighbor_config["max_length"]),
            alignment_power=float(neighbor_config["alignment_power"]),
        )
    neighbor_seconds = time.perf_counter() - started
    ids = tail["protein_id"].to_numpy(dtype=np.str_)
    np.savez_compressed(run_dir / "scores.npz", cosine_scores=cosine_scores, alignment_scores=alignment_scores, validation_ids=ids, label_columns=np.asarray(labels, dtype=np.str_))

    candidates = np.asarray(config["thresholds"]["candidates"], dtype=np.float32)
    rows = []
    for weight in config["alignment_weights"]:
        weight = float(weight)
        scores = ((1 - weight) * cosine_scores + weight * alignment_scores).astype(np.float32)
        auc = macro_roc_auc_skip_degenerate(target, scores)
        for shrinkage in config["thresholds"]["shrinkages"]:
            results = [
                crossfit_shrunk_threshold_score(target, scores, seed=int(seed), shrinkage=float(shrinkage), candidates=candidates)
                for seed in config["seeds"]
            ]
            f1 = np.asarray([result["macro_f1"] for result in results])
            rows.append({
                "alignment_weight": weight,
                "shrinkage": float(shrinkage),
                "mean_crossfit_macro_f1": float(f1.mean()),
                "std_crossfit_macro_f1": float(f1.std()),
                "min_crossfit_macro_f1": float(f1.min()),
                "continuous_macro_auc": auc,
                "mean_predicted_positive_rate": float(np.mean([result["predicted_positive_rate"] for result in results])),
            })
    leaderboard = pd.DataFrame(rows).sort_values(
        ["mean_crossfit_macro_f1", "continuous_macro_auc", "std_crossfit_macro_f1"],
        ascending=[False, False, True], ignore_index=True,
    )
    leaderboard.to_csv(prefix.with_name(prefix.name + "-leaderboard.csv"), index=False)
    best = leaderboard.iloc[0]
    selected_scores = ((1 - best["alignment_weight"]) * cosine_scores + best["alignment_weight"] * alignment_scores).astype(np.float32)
    thresholds, global_threshold = fit_shrunk_thresholds(target, selected_scores, shrinkage=float(best["shrinkage"]), candidates=candidates)
    thresholds_path = prefix.with_name(prefix.name + "-thresholds.json")
    thresholds_path.write_text(json.dumps({
        "experiment_id": config["experiment_id"],
        "labels": labels,
        "thresholds": thresholds.tolist(),
        "global_threshold": global_threshold,
        "shrinkage": float(best["shrinkage"]),
        "alignment_weight": float(best["alignment_weight"]),
    }, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps({
        "experiment_id": config["experiment_id"],
        "validation_rows": len(tail),
        "label_count": len(labels),
        "feature_seconds": feature_seconds,
        "neighbor_seconds": neighbor_seconds,
        "feature_resources": resources,
        "best": best.to_dict(),
        "iteration4_f1_reference": 0.32419172996031437,
        "beats_iteration4_reference": bool(best["mean_crossfit_macro_f1"] > 0.32419172996031437),
    }, indent=2), encoding="utf-8")
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(run_tail_homology_sweep(args.config))


if __name__ == "__main__":
    main()
