"""Screen SGD hyperparameters while reusing one fixed TF-IDF matrix."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config
from .data import label_columns
from .metrics import macro_f1_skip_empty
from .thresholds import crossfit_shrunk_threshold_score
from .train import _build_feature_matrices, _resolve, fit_label_models


def select_support_stratified_labels(
    target: pd.DataFrame, count: int
) -> pd.DataFrame:
    """Select evenly spaced labels after ranking them by positive support."""
    if not 1 <= count <= target.shape[1]:
        raise ValueError("label selection count must be between 1 and label count")
    supports = target.sum(axis=0).to_numpy(dtype=np.int64)
    support_order = np.argsort(-supports, kind="stable")
    rank_positions = np.rint(np.linspace(0, len(support_order) - 1, count)).astype(int)
    selected_indices = support_order[rank_positions]
    return pd.DataFrame(
        {
            "support_rank": rank_positions,
            "label_index": selected_indices,
            "label": target.columns.to_numpy()[selected_indices],
            "positive_count": supports[selected_indices],
        }
    )


def _candidate_name(candidate: dict) -> str:
    name = candidate.get("name")
    if not name or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for character in name):
        raise ValueError("candidate name must use lowercase ASCII letters, digits, '-' or '_'")
    return name


def _write_progress(path: Path, rows: list[dict]) -> None:
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(
            ["crossfit_macro_f1", "candidate"],
            ascending=[False, True],
            ignore_index=True,
        )
    frame.to_csv(path, index=False)


def run_sweep(
    config_path: str | Path, *, project_root: str | Path | None = None
) -> Path:
    """Run or resume a fixed-split SGD screen and persist each candidate."""
    root = Path(project_root) if project_root is not None else Path.cwd()
    config = load_config(config_path)
    train_path = _resolve(root, config["data"]["train_path"])
    columns = pd.read_csv(train_path, nrows=0).columns
    all_labels = label_columns(columns)
    train = pd.read_csv(
        train_path,
        dtype={label: np.uint8 for label in all_labels},
    )

    split_config = config["split"]
    training_ids = pd.read_csv(_resolve(root, split_config["train_ids"]))[
        "protein_id"
    ].tolist()
    validation_ids = pd.read_csv(_resolve(root, split_config["validation_ids"]))[
        "protein_id"
    ].tolist()
    indexed = train.set_index("protein_id", drop=False)
    missing = (set(training_ids) | set(validation_ids)) - set(indexed.index)
    if missing:
        raise ValueError("split contains protein IDs absent from training data")
    training = indexed.loc[training_ids]
    validation = indexed.loc[validation_ids]

    selection = select_support_stratified_labels(
        training[all_labels], int(config["selection"]["label_count"])
    )
    selected_labels = selection["label"].tolist()
    target = validation[selected_labels].to_numpy(dtype=np.uint8)

    run_dir = _resolve(root, config["output_dir"])
    metrics_prefix = _resolve(root, config["metrics_prefix"])
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_prefix.parent.mkdir(parents=True, exist_ok=True)
    saved_config_path = run_dir / "config.json"
    if saved_config_path.exists():
        saved_config = json.loads(saved_config_path.read_text(encoding="utf-8"))
        if saved_config != config:
            raise ValueError("existing sweep directory belongs to a different config")
    else:
        saved_config_path.write_text(
            json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    selection_path = metrics_prefix.with_name(metrics_prefix.name + "-labels.csv")
    leaderboard_path = metrics_prefix.with_name(
        metrics_prefix.name + "-leaderboard.csv"
    )
    summary_path = metrics_prefix.with_name(metrics_prefix.name + "-summary.json")
    selection.to_csv(selection_path, index=False)

    feature_started = time.perf_counter()
    training_features, validation_features, resources, _ = _build_feature_matrices(
        training, validation, config["features"]
    )
    feature_seconds = time.perf_counter() - feature_started

    rows: list[dict] = []
    base_model = config["model"]
    shrinkage = float(config.get("thresholds", {}).get("shrinkage", 25.0))
    for candidate in config["candidates"]:
        name = _candidate_name(candidate)
        candidate_summary_path = run_dir / f"{name}-summary.json"
        scores_path = run_dir / f"{name}-scores.npz"
        if candidate_summary_path.exists() and scores_path.exists():
            row = json.loads(candidate_summary_path.read_text(encoding="utf-8"))
            rows.append(row)
            _write_progress(leaderboard_path, rows)
            print(f"resumed {name}: {row['crossfit_macro_f1']:.6f}", flush=True)
            continue

        model_config = {**base_model, **candidate.get("model", {})}
        started = time.perf_counter()
        timing: dict[str, float] = {}
        _, scores = fit_label_models(
            training_features,
            training[selected_labels].to_numpy(dtype=np.uint8),
            validation_features,
            model_config=model_config,
            seed=config["seed"],
            progress_every=25,
            retain_models=False,
            timing_stats=timing,
        )
        crossfit = crossfit_shrunk_threshold_score(
            target,
            scores,
            seed=config["seed"],
            shrinkage=shrinkage,
        )
        fixed_predictions = (scores >= float(config.get("threshold", 0.5))).astype(
            np.uint8
        )
        row = {
            "candidate": name,
            "alpha": float(model_config.get("alpha", 0.0001)),
            "average": bool(model_config.get("average", False)),
            "fixed_macro_f1": macro_f1_skip_empty(target, fixed_predictions),
            "crossfit_macro_f1": float(crossfit["macro_f1"]),
            "predicted_positive_rate": float(crossfit["predicted_positive_rate"]),
            "fit_seconds": float(timing["fit_seconds"]),
            "inference_seconds": float(timing["inference_seconds"]),
            "elapsed_seconds": time.perf_counter() - started,
        }
        np.savez_compressed(
            scores_path,
            validation_scores=scores.astype(np.float32),
            validation_ids=np.asarray(validation_ids, dtype=np.str_),
            label_columns=np.asarray(selected_labels, dtype=np.str_),
        )
        candidate_summary_path.write_text(
            json.dumps(row, indent=2), encoding="utf-8"
        )
        rows.append(row)
        _write_progress(leaderboard_path, rows)
        print(f"completed {name}: {row['crossfit_macro_f1']:.6f}", flush=True)

    leaderboard = pd.DataFrame(rows).sort_values(
        ["crossfit_macro_f1", "candidate"],
        ascending=[False, True],
        ignore_index=True,
    )
    summary = {
        "experiment_id": config["experiment_id"],
        "train_rows": len(training),
        "validation_rows": len(validation),
        "selected_label_count": len(selected_labels),
        "selection": "evenly_spaced_support_ranks",
        "feature_seconds": feature_seconds,
        "feature_resources": resources,
        "best_candidate": leaderboard.iloc[0].to_dict(),
        "outputs": {
            "labels": str(selection_path),
            "leaderboard": str(leaderboard_path),
            "run_dir": str(run_dir),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(run_sweep(args.config))


if __name__ == "__main__":
    main()
