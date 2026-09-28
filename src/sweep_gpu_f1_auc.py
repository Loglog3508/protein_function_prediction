"""Select GPU score blends under a hard continuous Macro AUC constraint."""

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

REQUIRED_MINIMUM_AUC = 0.8150451732223777


def validate_sweep_config(config: dict) -> None:
    """Require the production AUC gate and a rollback baseline."""
    diagnostic_only = bool(config.get("diagnostic_only", False))
    minimum_auc = float(config["minimum_auc"])
    weights = [float(value) for value in config["gpu_weights"]]
    if not diagnostic_only and minimum_auc < REQUIRED_MINIMUM_AUC:
        raise ValueError(
            f"minimum_auc must be at least {REQUIRED_MINIMUM_AUC}"
        )
    if not diagnostic_only and 0.0 not in weights:
        raise ValueError("production sweep must include a 0% GPU rollback baseline")


def assert_experiment_outputs_absent(paths: list[Path]) -> None:
    """Reject every stale output before creating a new experiment."""
    if any(path.exists() for path in paths):
        raise FileExistsError("experiment outputs already exist; use a new experiment ID")


def rank_candidate_results(
    leaderboard: pd.DataFrame, *, minimum_auc: float
) -> pd.DataFrame:
    """Rank eligible candidates by F1 while retaining rejected diagnostics."""
    required = {
        "candidate",
        "gpu_weight",
        "mean_crossfit_macro_f1",
        "std_crossfit_macro_f1",
        "min_crossfit_macro_f1",
        "continuous_macro_auc",
        "mean_predicted_positive_rate",
        "seed_count",
    }
    missing = required - set(leaderboard.columns)
    if missing:
        raise ValueError(f"leaderboard is missing columns: {sorted(missing)}")
    ranked = leaderboard.copy()
    ranked["auc_eligible"] = ranked["continuous_macro_auc"] >= float(minimum_auc)
    return ranked.sort_values(
        [
            "auc_eligible",
            "mean_crossfit_macro_f1",
            "continuous_macro_auc",
            "std_crossfit_macro_f1",
            "gpu_weight",
        ],
        ascending=[False, False, False, True, True],
        ignore_index=True,
    )


def _load_scores(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    with np.load(path, allow_pickle=False) as saved:
        return (
            saved["validation_scores"].astype(np.float32),
            saved["validation_ids"].astype(str),
            saved["label_columns"].astype(str).tolist(),
        )


def _aggregate_seed_results(rows: pd.DataFrame) -> pd.DataFrame:
    return (
        rows.groupby(
            ["candidate", "gpu_source", "gpu_weight", "shrinkage"],
            as_index=False,
        )
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


def run_gpu_f1_auc_sweep(
    config_path: str | Path, *, project_root: str | Path | None = None
) -> Path:
    """Evaluate baseline/GPU blends and select only candidates above the AUC gate."""
    root = Path(project_root) if project_root is not None else Path.cwd()
    config = load_config(config_path)
    validate_sweep_config(config)
    run_dir = _resolve(root, config["output_dir"])
    metrics_prefix = _resolve(root, config["metrics_prefix"])
    leaderboard_path = metrics_prefix.with_name(metrics_prefix.name + "-leaderboard.csv")
    seed_results_path = metrics_prefix.with_name(metrics_prefix.name + "-seed-results.csv")
    summary_path = metrics_prefix.with_name(metrics_prefix.name + "-summary.json")
    threshold_path = metrics_prefix.with_name(metrics_prefix.name + "-thresholds.json")
    assert_experiment_outputs_absent(
        [
            run_dir,
            leaderboard_path,
            seed_results_path,
            summary_path,
            threshold_path,
        ]
    )
    run_dir.mkdir(parents=True)
    metrics_prefix.parent.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    baseline_scores, validation_ids, labels = _load_scores(
        _resolve(root, config["baseline_scores"])
    )
    train_path = _resolve(root, config["data"]["train_path"])
    train = pd.read_csv(train_path, usecols=["protein_id", *labels])
    indexed = train.set_index("protein_id", drop=False)
    if not pd.Index(validation_ids).isin(indexed.index).all():
        raise ValueError("saved validation IDs are absent from training data")
    target = indexed.loc[validation_ids, labels].to_numpy(dtype=np.uint8)
    destination_labels = [
        column
        for column in pd.read_csv(train_path, nrows=0).columns
        if str(column).startswith("label_")
    ]
    candidates = np.asarray(config["thresholds"]["candidates"], dtype=np.float32)
    seeds = [int(value) for value in config["seeds"]]
    rows: list[dict] = []
    score_candidates: dict[str, np.ndarray] = {}

    gpu_entries = config["gpu_candidates"]
    for gpu_index, gpu_entry in enumerate(gpu_entries):
        gpu_name = str(gpu_entry["name"])
        gpu_scores, gpu_ids, gpu_labels = _load_scores(
            _resolve(root, gpu_entry["scores"])
        )
        if not np.array_equal(validation_ids, gpu_ids) or labels != gpu_labels:
            raise ValueError(f"GPU candidate {gpu_name} IDs or labels do not match")
        if baseline_scores.shape != gpu_scores.shape:
            raise ValueError(f"GPU candidate {gpu_name} score shape does not match")
        for weight_value in config["gpu_weights"]:
            gpu_weight = float(weight_value)
            if not 0 <= gpu_weight <= 1:
                raise ValueError("GPU weights must be between zero and one")
            if gpu_weight == 0 and gpu_index:
                continue
            name = "baseline" if gpu_weight == 0 else f"{gpu_name}-gpu-{gpu_weight:.3f}"
            scores = (
                (1.0 - gpu_weight) * baseline_scores + gpu_weight * gpu_scores
            ).astype(np.float32)
            score_candidates[name] = scores
            continuous_auc = macro_roc_auc_skip_degenerate(target, scores)
            fixed_f1 = macro_f1_skip_empty(
                target, (scores >= 0.5).astype(np.uint8)
            )
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
                            "candidate": name,
                            "gpu_source": gpu_name if gpu_weight else "none",
                            "gpu_weight": gpu_weight,
                            "shrinkage": shrinkage,
                            "seed": seed,
                            "crossfit_macro_f1": float(result["macro_f1"]),
                            "continuous_macro_auc": continuous_auc,
                            "fixed_0.5_macro_f1": fixed_f1,
                            "predicted_positive_rate": float(
                                result["predicted_positive_rate"]
                            ),
                        }
                    )

    seed_results = pd.DataFrame(rows)
    seed_results.to_csv(seed_results_path, index=False)
    minimum_auc = float(config["minimum_auc"])
    leaderboard = rank_candidate_results(
        _aggregate_seed_results(seed_results), minimum_auc=minimum_auc
    )
    leaderboard.to_csv(leaderboard_path, index=False)
    eligible = leaderboard.loc[leaderboard["auc_eligible"]]
    if eligible.empty:
        raise RuntimeError("no candidate satisfies the continuous Macro AUC gate")
    best = eligible.iloc[0]
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
    threshold_path.write_text(
        json.dumps(
            {
                "experiment_id": config["experiment_id"],
                "strategy": (
                    "diagnostic_f1_primary"
                    if config.get("diagnostic_only", False)
                    else "f1_primary_hard_auc_gate"
                ),
                "minimum_auc": minimum_auc,
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
        "objective": (
            "Diagnostic Macro F1 sweep without production eligibility"
            if config.get("diagnostic_only", False)
            else "Macro F1 primary under a hard continuous Macro AUC gate"
        ),
        "diagnostic_only": bool(config.get("diagnostic_only", False)),
        "minimum_auc": minimum_auc,
        "validation_rows": len(validation_ids),
        "label_count": len(labels),
        "seeds": seeds,
        "best": best.to_dict(),
        "target_f1_reached": float(best["mean_crossfit_macro_f1"])
        >= float(config.get("target_f1", 0.5)),
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
    print(run_gpu_f1_auc_sweep(args.config))


if __name__ == "__main__":
    main()
