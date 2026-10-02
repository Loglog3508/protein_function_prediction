"""Screen support-stratified positive weights for tail-distribution SGD."""

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
from .metrics import macro_f1_skip_empty, macro_roc_auc_skip_degenerate
from .shift import tail_distribution_mask
from .sweep_sgd import _candidate_name, select_support_stratified_labels
from .thresholds import crossfit_shrunk_threshold_score
from .train import _build_feature_matrices, _resolve, fit_label_models


def _stack_features(first, second):
    if sparse.issparse(first):
        return sparse.vstack([first, second], format="csr")
    return np.vstack([first, second])


def _support_multipliers(
    target: np.ndarray,
    candidate: dict,
    strata: list[dict],
) -> np.ndarray:
    """Map each label's training support to the candidate multiplier."""
    support = np.asarray(target, dtype=np.uint8).sum(axis=0)
    values = np.empty(support.shape[0], dtype=np.float64)
    assigned = np.zeros(support.shape[0], dtype=bool)
    candidate_values = candidate.get("multipliers", {})
    for stratum in strata:
        name = stratum["name"]
        if name not in candidate_values:
            raise ValueError(f"candidate is missing multiplier for stratum {name}")
        multiplier = float(candidate_values[name])
        if not np.isfinite(multiplier) or multiplier <= 0:
            raise ValueError("support multipliers must be finite and positive")
        mask = np.ones(support.shape[0], dtype=bool)
        if "min_support" in stratum:
            mask &= support >= int(stratum["min_support"])
        if "max_support" in stratum:
            mask &= support <= int(stratum["max_support"])
        if not mask.any():
            raise ValueError(f"support stratum {name} is empty")
        if (assigned & mask).any():
            raise ValueError("support strata overlap")
        values[mask] = multiplier
        assigned |= mask
    if not assigned.all():
        raise ValueError("support strata do not cover every label")
    return values


def _write_progress(path: Path, rows: list[dict]) -> None:
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(
            ["mean_crossfit_macro_f1", "continuous_macro_auc", "candidate"],
            ascending=[False, False, True],
            ignore_index=True,
        )
    frame.to_csv(path, index=False)


def run_labelwise_weight_sweep(
    config_path: str | Path, *, project_root: str | Path | None = None
) -> Path:
    """Run two-fold tail OOF evaluation for support-stratified class weights."""
    root = Path(project_root) if project_root is not None else Path.cwd()
    config = load_config(config_path)
    train_path = _resolve(root, config["data"]["train_path"])
    columns = pd.read_csv(train_path, nrows=0).columns
    all_labels = label_columns(columns)
    train = pd.read_csv(train_path, dtype={label: np.uint8 for label in all_labels})

    distribution = config["distribution"]
    cutoff = int(distribution["cutoff"])
    folds = int(distribution.get("folds", 2))
    if folds < 2:
        raise ValueError("tail sweep requires at least two folds")
    tail_mask = tail_distribution_mask(train["protein_id"].tolist(), cutoff=cutoff)
    early = train.loc[~tail_mask].reset_index(drop=True)
    tail = train.loc[tail_mask].reset_index(drop=True)
    if early.empty or len(tail) < folds:
        raise ValueError("cutoff must produce non-empty early and tail partitions")

    selection = select_support_stratified_labels(
        early[all_labels], int(config["selection"]["label_count"])
    )
    selected_labels = selection["label"].tolist()
    early_target = early[selected_labels].to_numpy(dtype=np.uint8)
    tail_target = tail[selected_labels].to_numpy(dtype=np.uint8)
    expected_ids = tail["protein_id"].to_numpy(dtype=np.str_)
    if "validation_ids" in config:
        canonical_ids = pd.read_csv(_resolve(root, config["validation_ids"]))["protein_id"].to_numpy(dtype=np.str_)
        if not np.array_equal(expected_ids, canonical_ids):
            raise ValueError("tail IDs must match canonical validation IDs")
    for candidate in config["candidates"]:
        _support_multipliers(early_target, candidate, config["support_strata"])
    threshold_seeds = list(config.get("seeds", [config["seed"]]))
    if not threshold_seeds:
        raise ValueError("at least one threshold seed is required")

    run_dir = _resolve(root, config["output_dir"])
    metrics_prefix = _resolve(root, config["metrics_prefix"])
    if run_dir.exists() or metrics_prefix.with_name(
        metrics_prefix.name + "-summary.json"
    ).exists():
        raise FileExistsError("experiment outputs already exist; use a new experiment ID")
    run_dir.mkdir(parents=True)
    metrics_prefix.parent.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    labels_path = metrics_prefix.with_name(metrics_prefix.name + "-labels.csv")
    leaderboard_path = metrics_prefix.with_name(
        metrics_prefix.name + "-leaderboard.csv"
    )
    summary_path = metrics_prefix.with_name(metrics_prefix.name + "-summary.json")
    selection.to_csv(labels_path, index=False)

    feature_started = time.perf_counter()
    early_features, tail_features, resources, _ = _build_feature_matrices(
        early, tail, config["features"]
    )
    feature_seconds = time.perf_counter() - feature_started

    splitter = MultilabelStratifiedKFold(
        n_splits=folds, shuffle=True, random_state=config["seed"]
    )
    fold_indices = list(splitter.split(np.zeros(len(tail)), tail_target))
    rows: list[dict] = []
    base_model = config["model"]
    shrinkage = float(config.get("thresholds", {}).get("shrinkage", 25.0))
    fixed_threshold = float(config.get("threshold", 0.5))
    support_strata = list(config["support_strata"])

    for candidate in config["candidates"]:
        name = _candidate_name(candidate)
        candidate_summary_path = run_dir / f"{name}-summary.json"
        scores_path = run_dir / f"{name}-scores.npz"
        model_config = {**base_model, **candidate.get("model", {})}
        oof_scores = np.zeros(tail_target.shape, dtype=np.float32)
        fit_seconds = 0.0
        inference_seconds = 0.0
        started = time.perf_counter()
        tail_weight = float(candidate.get("tail_weight", 1.0))
        candidate_shrinkage = float(candidate.get("shrinkage", shrinkage))
        if tail_weight <= 0:
            raise ValueError("tail_weight must be positive")
        if candidate_shrinkage < 0:
            raise ValueError("shrinkage must be non-negative")

        active_folds = fold_indices
        if "reuse_scores" in candidate:
            with np.load(_resolve(root, candidate["reuse_scores"]), allow_pickle=False) as saved:
                if not np.array_equal(saved["validation_ids"].astype(str), expected_ids):
                    raise ValueError("reused score IDs must match validation order")
                if saved["label_columns"].astype(str).tolist() != selected_labels:
                    raise ValueError("reused score labels must match selected labels")
                oof_scores = saved["validation_scores"].astype(np.float32)
            if oof_scores.shape != tail_target.shape or not np.isfinite(oof_scores).all():
                raise ValueError("reused score shape or values are invalid")
            active_folds = []

        for fold_index, (tail_train_index, tail_eval_index) in enumerate(
            active_folds, start=1
        ):
            fold_features = _stack_features(
                early_features, tail_features[tail_train_index]
            )
            fold_target = np.vstack(
                [early_target, tail_target[tail_train_index]]
            )
            multipliers = _support_multipliers(
                fold_target, candidate, support_strata
            )
            if np.allclose(multipliers, 1.0):
                multipliers_for_fit = None
            else:
                multipliers_for_fit = multipliers
            sample_weight = np.concatenate(
                [
                    np.ones(len(early), dtype=np.float64),
                    np.full(len(tail_train_index), tail_weight, dtype=np.float64),
                ]
            )
            timing: dict[str, float] = {}
            _, fold_scores = fit_label_models(
                fold_features,
                fold_target,
                tail_features[tail_eval_index],
                model_config=model_config,
                seed=config["seed"] + fold_index,
                progress_every=25,
                retain_models=False,
                timing_stats=timing,
                training_sample_weight=sample_weight,
                label_positive_weight_multipliers=multipliers_for_fit,
            )
            oof_scores[tail_eval_index] = fold_scores
            fit_seconds += timing["fit_seconds"]
            inference_seconds += timing["inference_seconds"]

        fixed_predictions = (oof_scores >= fixed_threshold).astype(np.uint8)
        threshold_results = [
            crossfit_shrunk_threshold_score(
                tail_target, oof_scores, seed=int(seed), shrinkage=candidate_shrinkage
            )
            for seed in threshold_seeds
        ]
        seed_f1 = np.asarray([result["macro_f1"] for result in threshold_results])
        row = {
            "candidate": name,
            "multipliers": candidate["multipliers"],
            "tail_weight": tail_weight,
            "shrinkage": candidate_shrinkage,
            "fixed_macro_f1": macro_f1_skip_empty(tail_target, fixed_predictions),
            "crossfit_macro_f1": float(seed_f1.mean()),
            "mean_crossfit_macro_f1": float(seed_f1.mean()),
            "std_crossfit_macro_f1": float(seed_f1.std()),
            "min_crossfit_macro_f1": float(seed_f1.min()),
            "seed_results": [
                {"seed": int(seed), "macro_f1": float(result["macro_f1"])}
                for seed, result in zip(threshold_seeds, threshold_results, strict=True)
            ],
            "continuous_macro_auc": macro_roc_auc_skip_degenerate(
                tail_target, oof_scores
            ),
            "predicted_positive_rate": float(
                np.mean([result["predicted_positive_rate"] for result in threshold_results])
            ),
            "fit_seconds": fit_seconds,
            "inference_seconds": inference_seconds,
            "elapsed_seconds": time.perf_counter() - started,
        }
        np.savez_compressed(
            scores_path,
            validation_scores=oof_scores,
            validation_ids=tail["protein_id"].to_numpy(dtype=np.str_),
            label_columns=np.asarray(selected_labels, dtype=np.str_),
        )
        candidate_summary_path.write_text(
            json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        rows.append(row)
        _write_progress(leaderboard_path, rows)
        print(f"completed {name}: {row['crossfit_macro_f1']:.6f}", flush=True)

    leaderboard = pd.DataFrame(rows).sort_values(
        ["mean_crossfit_macro_f1", "continuous_macro_auc", "candidate"],
        ascending=[False, False, True],
        ignore_index=True,
    )
    leaderboard.to_csv(leaderboard_path, index=False)
    summary = {
        "experiment_id": config["experiment_id"],
        "cutoff": cutoff,
        "folds": folds,
        "early_rows": len(early),
        "tail_rows": len(tail),
        "selected_label_count": len(selected_labels),
        "selection": "evenly_spaced_support_ranks",
        "threshold_seeds": threshold_seeds,
        "training_support_range": [int(early_target.sum(axis=0).min()), int(early_target.sum(axis=0).max())],
        "feature_seconds": feature_seconds,
        "feature_resources": resources,
        "best_candidate": leaderboard.iloc[0].to_dict(),
        "outputs": {
            "labels": str(labels_path),
            "leaderboard": str(leaderboard_path),
            "run_dir": str(run_dir),
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
    print(run_labelwise_weight_sweep(args.config))


if __name__ == "__main__":
    main()
