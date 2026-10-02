"""Leakage-safe nested OOF label-chain screening for protein labels."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from scipy import sparse

from .config import load_config
from .data import label_columns
from .hybrid_features import combine_sparse_dense_features
from .metrics import macro_f1_skip_empty, macro_roc_auc_skip_degenerate
from .shift import tail_distribution_mask
from .sweep_sgd import _candidate_name, select_support_stratified_labels
from .thresholds import crossfit_shrunk_threshold_score
from .train import _build_feature_matrices, _resolve, fit_label_models


def _validate_score_matrix(scores, *, rows: int, labels: int) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    if scores.ndim != 2 or scores.shape != (rows, labels):
        raise ValueError("score matrix has an unexpected shape")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("score matrix must be finite and in [0, 1]")
    return scores


def _select_context_indices(target: np.ndarray, count: int) -> np.ndarray:
    """Select globally useful context labels from support-normalized co-occurrence."""
    target = np.asarray(target, dtype=np.float64)
    if target.ndim != 2 or not 1 <= count <= target.shape[1]:
        raise ValueError("target must be two-dimensional and count must fit labels")
    support = target.sum(axis=0)
    cooccurrence = target.T @ target
    np.fill_diagonal(cooccurrence, 0.0)
    normalized = cooccurrence / np.sqrt(np.maximum(support[:, None] * support[None, :], 1.0))
    score = normalized.sum(axis=0)
    order = np.lexsort((np.arange(target.shape[1]), -score))
    return np.asarray(order[:count], dtype=np.int64)


def _build_stage2_features(kmer_features, score_features, *, mode: str, scale: float):
    score_features = _validate_score_matrix(
        score_features, rows=kmer_features.shape[0], labels=score_features.shape[1]
    )
    if not np.isfinite(scale) or scale < 0:
        raise ValueError("score scale must be finite and non-negative")
    if mode == "scores-only":
        return score_features * np.float32(scale)
    if mode == "kmer+scores":
        return combine_sparse_dense_features(kmer_features, score_features, scale=scale)
    raise ValueError("stage2 mode must be scores-only or kmer+scores")


def _stack_features(first, second):
    if sparse.issparse(first):
        return sparse.vstack([first, second], format="csr")
    return np.vstack([first, second])


def _fit_stage1(
    training_features,
    training_target,
    evaluation_features,
    *,
    model_config: dict,
    seed: int,
    sample_weight: np.ndarray,
) -> np.ndarray:
    sample_weight = np.asarray(sample_weight, dtype=np.float64)
    if sample_weight.shape != (training_features.shape[0],):
        raise ValueError("stage1 sample weights must match training rows")
    _, scores = fit_label_models(
        training_features,
        training_target,
        evaluation_features,
        model_config=model_config,
        seed=seed,
        progress_every=25,
        retain_models=False,
        training_sample_weight=sample_weight,
    )
    return _validate_score_matrix(
        scores, rows=evaluation_features.shape[0], labels=training_target.shape[1]
    )


def _nested_stage1_scores(
    early_features,
    tail_features,
    early_target: np.ndarray,
    tail_target: np.ndarray,
    tail_train_index: np.ndarray,
    tail_eval_index: np.ndarray,
    *,
    model_config: dict,
    seed: int,
    tail_weight: float,
    inner_folds: int,
) -> tuple[np.ndarray, np.ndarray]:
    outer_train_features = _stack_features(early_features, tail_features[tail_train_index])
    outer_train_target = np.vstack([early_target, tail_target[tail_train_index]])
    inner_splitter = MultilabelStratifiedKFold(
        n_splits=inner_folds, shuffle=True, random_state=seed
    )
    inner_indices = list(
        inner_splitter.split(np.zeros(len(outer_train_target)), outer_train_target)
    )
    train_oof = np.zeros_like(outer_train_target, dtype=np.float32)
    early_rows = len(early_target)
    for inner_number, (inner_train, inner_eval) in enumerate(inner_indices, start=1):
        inner_train_features = outer_train_features[inner_train]
        inner_eval_features = outer_train_features[inner_eval]
        inner_scores = _fit_stage1(
            inner_train_features,
            outer_train_target[inner_train],
            inner_eval_features,
            model_config=model_config,
            seed=seed + inner_number,
            sample_weight=np.where(
                inner_train >= early_rows, tail_weight, 1.0
            ).astype(np.float64),
        )
        train_oof[inner_eval] = inner_scores
    eval_scores = _fit_stage1(
        outer_train_features,
        outer_train_target,
        tail_features[tail_eval_index],
        model_config=model_config,
        seed=seed + inner_folds + 1,
        sample_weight=np.concatenate(
            [
                np.ones(len(early_target), dtype=np.float64),
                np.full(len(tail_train_index), tail_weight, dtype=np.float64),
            ]
        ),
    )
    return train_oof, eval_scores


def _write_progress(path: Path, rows: list[dict]) -> None:
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(
            ["mean_crossfit_macro_f1", "continuous_macro_auc", "candidate"],
            ascending=[False, False, True],
            ignore_index=True,
        )
    frame.to_csv(path, index=False)


def run_oof_chain_sweep(
    config_path: str | Path, *, project_root: str | Path | None = None
) -> Path:
    root = Path(project_root) if project_root is not None else Path.cwd()
    config = load_config(config_path)
    train_path = _resolve(root, config["data"]["train_path"])
    all_labels = label_columns(pd.read_csv(train_path, nrows=0).columns)
    train = pd.read_csv(train_path, dtype={label: np.uint8 for label in all_labels})
    cutoff = int(config["distribution"]["cutoff"])
    outer_folds = int(config["distribution"].get("folds", 2))
    inner_folds = int(config["distribution"].get("inner_folds", 2))
    if outer_folds < 2 or inner_folds < 2:
        raise ValueError("outer and inner folds must be at least two")
    tail_mask = tail_distribution_mask(train["protein_id"].tolist(), cutoff=cutoff)
    early = train.loc[~tail_mask].reset_index(drop=True)
    tail = train.loc[tail_mask].reset_index(drop=True)
    if len(tail) < outer_folds:
        raise ValueError("tail partition is too small for the requested folds")
    selection = select_support_stratified_labels(
        early[all_labels], int(config["selection"]["label_count"])
    )
    labels = selection["label"].tolist()
    early_target = early[labels].to_numpy(dtype=np.uint8)
    tail_target = tail[labels].to_numpy(dtype=np.uint8)
    expected_ids = tail["protein_id"].to_numpy(dtype=np.str_)
    canonical = pd.read_csv(_resolve(root, config["validation_ids"]))["protein_id"].to_numpy(dtype=np.str_)
    if not np.array_equal(expected_ids, canonical):
        raise ValueError("tail IDs must match canonical validation IDs")

    run_dir = _resolve(root, config["output_dir"])
    prefix = _resolve(root, config["metrics_prefix"])
    summary_path = prefix.with_name(prefix.name + "-summary.json")
    if run_dir.exists() or summary_path.exists():
        raise FileExistsError("experiment outputs already exist; use a new experiment ID")
    run_dir.mkdir(parents=True)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    selection.to_csv(prefix.with_name(prefix.name + "-labels.csv"), index=False)

    started = time.perf_counter()
    early_features, tail_features, resources, _ = _build_feature_matrices(
        early, tail, config["features"]
    )
    feature_seconds = time.perf_counter() - started
    context_indices = _select_context_indices(early_target, int(config["context_count"]))
    context_labels = [labels[index] for index in context_indices]
    splitter = MultilabelStratifiedKFold(
        n_splits=outer_folds, shuffle=True, random_state=int(config["seed"])
    )
    fold_indices = list(splitter.split(np.zeros(len(tail)), tail_target))
    stage1_config = dict(config["stage1_model"])
    stage1_oof = np.zeros_like(tail_target, dtype=np.float32)
    stage2_scores: dict[str, np.ndarray] = {
        candidate["name"]: np.zeros_like(tail_target, dtype=np.float32)
        for candidate in config["candidates"]
    }
    stage1_seconds = 0.0
    stage2_seconds = 0.0
    for outer_number, (tail_train_index, tail_eval_index) in enumerate(fold_indices, start=1):
        fold_started = time.perf_counter()
        train_oof, eval_stage1 = _nested_stage1_scores(
            early_features,
            tail_features,
            early_target,
            tail_target,
            tail_train_index,
            tail_eval_index,
            model_config=stage1_config,
            seed=int(config["seed"]) + outer_number * 100,
            tail_weight=float(config.get("tail_weight", 1.0)),
            inner_folds=inner_folds,
        )
        stage1_seconds += time.perf_counter() - fold_started
        stage1_oof[tail_eval_index] = eval_stage1
        outer_train_features = _stack_features(early_features, tail_features[tail_train_index])
        outer_train_target = np.vstack([early_target, tail_target[tail_train_index]])
        context_train = train_oof[:, context_indices]
        context_eval = eval_stage1[:, context_indices]
        for candidate in config["candidates"]:
            name = _candidate_name(candidate)
            if name == "stage1":
                stage2_scores[name][tail_eval_index] = eval_stage1
                continue
            mode = str(candidate["mode"])
            scale = float(candidate.get("score_scale", 1.0))
            train_stage2 = _build_stage2_features(
                outer_train_features, train_oof[:, context_indices], mode=mode, scale=scale
            )
            eval_stage2 = _build_stage2_features(
                tail_features[tail_eval_index], context_eval, mode=mode, scale=scale
            )
            model_started = time.perf_counter()
            scores = _fit_stage1(
                train_stage2,
                outer_train_target,
                eval_stage2,
                model_config={**config["stage2_model"], **candidate.get("model", {})},
                seed=int(config["seed"]) + outer_number,
                sample_weight=np.concatenate(
                    [
                        np.ones(len(early_target), dtype=np.float64),
                        np.full(
                            len(tail_train_index),
                            float(config.get("tail_weight", 1.0)),
                            dtype=np.float64,
                        ),
                    ]
                ),
            )
            stage2_seconds += time.perf_counter() - model_started
            stage2_scores[name][tail_eval_index] = scores
        print(f"completed outer fold {outer_number}/{outer_folds}", flush=True)

    candidates = []
    threshold_seeds = [int(seed) for seed in config["seeds"]]
    shrinkage = float(config["thresholds"]["shrinkage"])
    for candidate in config["candidates"]:
        name = _candidate_name(candidate)
        scores = _validate_score_matrix(stage2_scores[name], rows=len(tail), labels=len(labels))
        seed_results = [
            crossfit_shrunk_threshold_score(tail_target, scores, seed=seed, shrinkage=shrinkage)
            for seed in threshold_seeds
        ]
        f1 = np.asarray([item["macro_f1"] for item in seed_results], dtype=float)
        row = {
            "candidate": name,
            "mode": candidate.get("mode", "stage1"),
            "context_count": len(context_indices),
            "context_labels": context_labels,
            "mean_crossfit_macro_f1": float(f1.mean()),
            "std_crossfit_macro_f1": float(f1.std()),
            "min_crossfit_macro_f1": float(f1.min()),
            "continuous_macro_auc": float(macro_roc_auc_skip_degenerate(tail_target, scores)),
            "mean_predicted_positive_rate": float(np.mean([item["predicted_positive_rate"] for item in seed_results])),
            "seed_results": [{"seed": seed, "macro_f1": float(item["macro_f1"])} for seed, item in zip(threshold_seeds, seed_results, strict=True)],
        }
        np.savez_compressed(
            run_dir / f"{name}-scores.npz",
            validation_scores=scores,
            validation_ids=expected_ids,
            label_columns=np.asarray(labels, dtype=np.str_),
        )
        candidates.append(row)
        print(json.dumps({key: row[key] for key in ("candidate", "mean_crossfit_macro_f1", "continuous_macro_auc")}), flush=True)
    leaderboard = pd.DataFrame(candidates).sort_values(
        ["mean_crossfit_macro_f1", "continuous_macro_auc", "candidate"],
        ascending=[False, False, True], ignore_index=True
    )
    leaderboard.to_csv(prefix.with_name(prefix.name + "-leaderboard.csv"), index=False)
    summary = {
        "experiment_id": config["experiment_id"],
        "validation_rows": len(tail),
        "selected_label_count": len(labels),
        "outer_folds": outer_folds,
        "inner_folds": inner_folds,
        "threshold_seeds": threshold_seeds,
        "feature_seconds": feature_seconds,
        "stage1_seconds": stage1_seconds,
        "stage2_seconds": stage2_seconds,
        "feature_resources": resources,
        "context_labels": context_labels,
        "leaderboard": leaderboard.to_dict(orient="records"),
        "outputs": {"run_dir": str(run_dir), "leaderboard": str(prefix.with_name(prefix.name + "-leaderboard.csv"))},
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(run_oof_chain_sweep(args.config))


if __name__ == "__main__":
    main()
