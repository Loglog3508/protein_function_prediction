"""Select and finalize a rank-thresholded fusion submission."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .rank_fusion import crossfit_rank_fusion, fit_rank_fusion_thresholds
from .thresholds import threshold_predictions
from .train import _resolve
from .validate_submission import validate_submission_file


def _load_scores(path: Path, key: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    with np.load(path, allow_pickle=False) as saved:
        ids_key = "validation_ids" if key == "validation_scores" else "test_ids"
        if key not in saved or ids_key not in saved or "label_columns" not in saved:
            raise ValueError(f"score file {path} lacks {key}, {ids_key}, or label_columns")
        scores = np.asarray(saved[key], dtype=np.float32)
        ids = np.asarray(saved[ids_key], dtype=str)
        labels = [str(value) for value in np.asarray(saved["label_columns"], dtype=str)]
    if scores.ndim != 2 or len(ids) != scores.shape[0] or len(labels) != scores.shape[1]:
        raise ValueError(f"score file {path} has inconsistent shapes")
    return scores, ids, labels


def _align_scores(
    scores: np.ndarray,
    ids: np.ndarray,
    labels: list[str],
    target_ids: np.ndarray,
    target_labels: list[str],
) -> np.ndarray:
    if set(ids.tolist()) != set(target_ids.tolist()) or set(labels) != set(target_labels):
        raise ValueError("score IDs or labels do not match the reference")
    row_positions = {value: index for index, value in enumerate(ids.tolist())}
    column_positions = {value: index for index, value in enumerate(labels)}
    return scores[
        [row_positions[value] for value in target_ids.tolist()],
        :,
    ][:, [column_positions[value] for value in target_labels]].astype(np.float32, copy=False)


def select_rank_fusion_candidate(
    leaderboard: pd.DataFrame, *, minimum_auc: float
) -> pd.Series:
    """Select highest F1 among rows meeting the continuous-AUC floor."""
    required = {"mean_crossfit_macro_f1", "continuous_macro_auc"}
    if leaderboard.empty or not required.issubset(leaderboard.columns):
        raise ValueError("leaderboard must contain candidate metrics")
    eligible = leaderboard.loc[leaderboard["continuous_macro_auc"] >= float(minimum_auc)].copy()
    if eligible.empty:
        raise ValueError("no rank-fusion candidate satisfies the minimum AUC")
    sort_columns = ["mean_crossfit_macro_f1", "continuous_macro_auc"]
    ascending = [False, False]
    if "std_crossfit_macro_f1" in eligible:
        sort_columns.append("std_crossfit_macro_f1")
        ascending.append(True)
    return eligible.sort_values(sort_columns, ascending=ascending, ignore_index=True).iloc[0]


def _group_seed_results(rows: pd.DataFrame) -> pd.DataFrame:
    seed_rows = (
        rows.groupby(["new_weight", "shrinkage", "seed"], as_index=False)
        .agg(
            macro_f1=("seed_macro_f1", "first"),
            continuous_macro_auc=("seed_continuous_macro_auc", "first"),
            predicted_positive_rate=("seed_predicted_positive_rate", "first"),
        )
    )
    grouped = (
        seed_rows.groupby(["new_weight", "shrinkage"], as_index=False)
        .agg(
            mean_crossfit_macro_f1=("macro_f1", "mean"),
            std_crossfit_macro_f1=("macro_f1", "std"),
            min_crossfit_macro_f1=("macro_f1", "min"),
            continuous_macro_auc=("continuous_macro_auc", "mean"),
            mean_predicted_positive_rate=("predicted_positive_rate", "mean"),
        )
    )
    grouped["std_crossfit_macro_f1"] = grouped["std_crossfit_macro_f1"].fillna(0.0)
    return grouped


def _write_thresholds(path: Path, labels: list[str], diagnostics: pd.DataFrame, thresholds: np.ndarray) -> None:
    output = diagnostics.copy()
    output["label"] = labels
    output["final_threshold"] = thresholds
    path.parent.mkdir(parents=True, exist_ok=True)
    output[["label_index", "label", "support", "threshold", "final_threshold", "f1", "used_global_fallback"]].to_csv(path, index=False)


def run_rank_fusion(config_path: str | Path, *, project_root: str | Path | None = None) -> Path:
    root = Path(project_root) if project_root is not None else Path.cwd()
    config: dict[str, Any] = json.loads(Path(config_path).read_text(encoding="utf-8"))
    metrics_prefix = _resolve(root, config["metrics_prefix"])
    output_dir = _resolve(root, config["output_dir"])
    submission_path = _resolve(root, config["submission_path"])
    summary_path = metrics_prefix.with_name(metrics_prefix.name + "-summary.json")
    thresholds_path = metrics_prefix.with_name(metrics_prefix.name + "-thresholds.csv")
    for path in (summary_path, thresholds_path, submission_path, output_dir):
        if path.exists():
            raise FileExistsError(f"experiment output already exists: {path}")

    base_oof, oof_ids, labels = _load_scores(_resolve(root, config["base_oof"]), "validation_scores")
    new_oof_raw, new_oof_ids, new_oof_labels = _load_scores(_resolve(root, config["new_oof"]), "validation_scores")
    new_oof = _align_scores(new_oof_raw, new_oof_ids, new_oof_labels, oof_ids, labels)
    train_path = _resolve(root, config["data"]["train_path"])
    train = pd.read_csv(train_path, usecols=["protein_id", *labels]).set_index("protein_id")
    if not pd.Index(oof_ids).isin(train.index).all():
        raise ValueError("OOF validation IDs are absent from training data")
    target = train.loc[oof_ids.tolist(), labels].to_numpy(dtype=np.uint8)

    seed_rows = crossfit_rank_fusion(
        target,
        base_oof,
        new_oof,
        weights=config["weights"],
        shrinkages=config["shrinkages"],
        seeds=config["seeds"],
    )
    leaderboard = _group_seed_results(seed_rows)
    leaderboard_path = metrics_prefix.with_name(metrics_prefix.name + "-leaderboard.csv")
    seed_results_path = metrics_prefix.with_name(metrics_prefix.name + "-seed-results.csv")
    leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
    seed_rows.to_csv(seed_results_path, index=False)
    leaderboard.to_csv(leaderboard_path, index=False)
    selected = select_rank_fusion_candidate(leaderboard, minimum_auc=float(config["minimum_auc"]))

    selected_weight = float(selected["new_weight"])
    selected_shrinkage = float(selected["shrinkage"])
    validation_scores = ((1.0 - selected_weight) * base_oof + selected_weight * new_oof).astype(np.float32)
    thresholds, global_threshold, diagnostics = fit_rank_fusion_thresholds(
        target, validation_scores, shrinkage=selected_shrinkage
    )
    _write_thresholds(thresholds_path, labels, diagnostics, thresholds)

    base_test, test_ids, test_labels = _load_scores(_resolve(root, config["base_test"]), "test_scores")
    new_test_raw, new_test_ids, new_test_labels = _load_scores(_resolve(root, config["new_test"]), "test_scores")
    base_test = _align_scores(base_test, test_ids, test_labels, test_ids, labels)
    new_test = _align_scores(new_test_raw, new_test_ids, new_test_labels, test_ids, labels)
    test_scores = ((1.0 - selected_weight) * base_test + selected_weight * new_test).astype(np.float32)
    predictions = threshold_predictions(test_scores, thresholds)
    test_path = _resolve(root, config["data"]["test_path"])
    test = pd.read_csv(test_path, usecols=["protein_id"])
    if not np.array_equal(test["protein_id"].astype(str).to_numpy(), test_ids):
        raise ValueError("test score IDs do not match data/test.csv order")
    submission = pd.DataFrame(predictions, columns=labels)
    submission.insert(0, "protein_id", test_ids)
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(submission_path, index=False)
    validation = validate_submission_file(submission_path, test_path=test_path, expected_label_columns=labels)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "scores.npz",
        validation_scores=validation_scores,
        validation_ids=oof_ids,
        test_scores=test_scores,
        test_ids=test_ids,
        label_columns=np.asarray(labels, dtype=np.str_),
    )
    metadata = {
        "experiment_id": config["experiment_id"],
        "selected": selected.to_dict(),
        "global_threshold": global_threshold,
        "threshold_shrinkage": selected_shrinkage,
        "validation_rows": len(oof_ids),
        "test_rows": len(test_ids),
        "label_count": len(labels),
        "predicted_positive_rate": float(predictions.mean()),
        "submission_validation": validation,
        "outputs": {
            "submission": str(submission_path),
            "scores": str(output_dir / "scores.npz"),
            "thresholds": str(thresholds_path),
            "leaderboard": str(leaderboard_path),
            "seed_results": str(seed_results_path),
        },
    }
    summary_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(run_rank_fusion(args.config))


if __name__ == "__main__":
    main()
