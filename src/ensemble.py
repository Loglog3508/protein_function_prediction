"""Evaluate score fusion and label-cooccurrence post-processing."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .thresholds import crossfit_shrunk_threshold_score, fit_shrunk_thresholds


def _load_scores(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        return {name: saved[name] for name in saved.files}


def _aligned_score_bundles(primary_path: str | Path, secondary_path: str | Path):
    primary = _load_scores(primary_path)
    secondary = _load_scores(secondary_path)
    for key in ("validation_ids", "label_columns"):
        if not np.array_equal(primary[key], secondary[key]):
            raise ValueError(f"score bundles do not align on {key}")
    if "test_scores" in primary or "test_scores" in secondary:
        if not (
            "test_scores" in primary
            and "test_scores" in secondary
            and np.array_equal(primary["test_ids"], secondary["test_ids"])
        ):
            raise ValueError("score bundles do not align on test data")
    return primary, secondary


def build_label_association(target: np.ndarray, *, top_k: int = 12) -> np.ndarray:
    """Build positive label associations using training labels only."""
    target = np.asarray(target, dtype=np.float32)
    if target.ndim != 2:
        raise ValueError("target must be two-dimensional")
    support = target.sum(axis=0)
    prevalence = target.mean(axis=0)
    cooccurrence = target.T @ target
    conditional = np.divide(
        cooccurrence,
        support[:, np.newaxis],
        out=np.zeros_like(cooccurrence),
        where=support[:, np.newaxis] != 0,
    )
    association = np.maximum(conditional - prevalence[np.newaxis, :], 0.0)
    np.fill_diagonal(association, 0.0)
    if top_k < association.shape[0]:
        for target_index in range(association.shape[1]):
            column = association[:, target_index]
            remove = np.argpartition(column, -top_k)[:-top_k]
            column[remove] = 0.0
    column_sum = association.sum(axis=0, keepdims=True)
    return np.divide(
        association,
        column_sum,
        out=np.zeros_like(association),
        where=column_sum != 0,
    ).astype(np.float32)


def apply_label_association(
    scores: np.ndarray, association: np.ndarray, weight: float
) -> np.ndarray:
    """Blend model scores with predictions propagated through label associations."""
    if not 0 <= weight <= 1:
        raise ValueError("association weight must be between 0 and 1")
    propagated = np.asarray(scores, dtype=np.float32) @ association
    return np.clip((1.0 - weight) * scores + weight * propagated, 0.0, 1.0)


def run_ensemble_evaluation(
    primary_path: str | Path,
    secondary_path: str | Path,
    *,
    train_path: str | Path,
    train_ids_path: str | Path,
    output_prefix: str | Path,
    seed: int = 42,
    shrinkage: float = 25.0,
) -> Path:
    """Compare score blends and cooccurrence post-processing with cross-fitting."""
    primary, secondary = _aligned_score_bundles(primary_path, secondary_path)
    labels = primary["label_columns"].astype(str).tolist()
    validation_ids = primary["validation_ids"].astype(str)
    train = pd.read_csv(train_path, usecols=["protein_id", *labels])
    indexed = train.set_index("protein_id", drop=False)
    target = indexed.loc[validation_ids, labels].to_numpy(dtype=np.uint8)
    training_ids = pd.read_csv(train_ids_path)["protein_id"].astype(str)
    training_target = indexed.loc[training_ids, labels].to_numpy(dtype=np.uint8)
    association = build_label_association(training_target)

    validation_candidates = {"primary": primary["validation_scores"]}
    test_candidates = {
        "primary": primary.get("test_scores"),
    }
    for primary_weight in (0.95, 0.9, 0.8):
        name = f"fusion_primary_{primary_weight:.2f}"
        validation_candidates[name] = (
            primary_weight * primary["validation_scores"]
            + (1.0 - primary_weight) * secondary["validation_scores"]
        ).astype(np.float32)
        if "test_scores" in primary:
            test_candidates[name] = (
                primary_weight * primary["test_scores"]
                + (1.0 - primary_weight) * secondary["test_scores"]
            ).astype(np.float32)
    for weight in (0.02, 0.05, 0.1):
        name = f"label_association_{weight:.2f}"
        validation_candidates[name] = apply_label_association(
            primary["validation_scores"], association, weight
        )
        if "test_scores" in primary:
            test_candidates[name] = apply_label_association(
                primary["test_scores"], association, weight
            )

    rows = []
    for name, scores in validation_candidates.items():
        result = crossfit_shrunk_threshold_score(
            target, scores, seed=seed, shrinkage=shrinkage
        )
        rows.append(
            {
                "candidate": name,
                "crossfit_macro_f1": result["macro_f1"],
                "predicted_positive_rate": result["predicted_positive_rate"],
            }
        )
    leaderboard = pd.DataFrame(rows).sort_values(
        ["crossfit_macro_f1", "candidate"], ascending=[False, True]
    )
    best_name = str(leaderboard.iloc[0]["candidate"])
    best_validation_scores = validation_candidates[best_name]
    thresholds, global_threshold = fit_shrunk_thresholds(
        target, best_validation_scores, shrinkage=shrinkage
    )

    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    leaderboard_path = output_prefix.with_name(output_prefix.name + "-leaderboard.csv")
    scores_path = output_prefix.with_name(output_prefix.name + "-scores.npz")
    thresholds_path = output_prefix.with_name(output_prefix.name + "-thresholds.json")
    summary_path = output_prefix.with_name(output_prefix.name + "-summary.json")
    leaderboard.to_csv(leaderboard_path, index=False)
    payload = {
        "validation_scores": best_validation_scores.astype(np.float32),
        "validation_ids": primary["validation_ids"],
        "label_columns": primary["label_columns"],
    }
    if test_candidates.get(best_name) is not None:
        payload["test_scores"] = test_candidates[best_name].astype(np.float32)
        payload["test_ids"] = primary["test_ids"]
    np.savez_compressed(scores_path, **payload)
    thresholds_path.write_text(
        json.dumps(
            {
                "strategy": "per_label_shrunk",
                "candidate": best_name,
                "global_threshold": global_threshold,
                "shrinkage": shrinkage,
                "labels": labels,
                "thresholds": thresholds.tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    summary = {
        "primary_scores": str(primary_path),
        "secondary_scores": str(secondary_path),
        "best_candidate": best_name,
        "best_crossfit_macro_f1": float(leaderboard.iloc[0]["crossfit_macro_f1"]),
        "leaderboard": leaderboard.to_dict(orient="records"),
        "outputs": {
            "leaderboard": str(leaderboard_path),
            "scores": str(scores_path),
            "thresholds": str(thresholds_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", required=True)
    parser.add_argument("--secondary", required=True)
    parser.add_argument("--train", default="data/train.csv")
    parser.add_argument(
        "--train-ids", default="artifacts/metrics/splits/seed42/train_ids.csv"
    )
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shrinkage", type=float, default=25.0)
    args = parser.parse_args()
    print(
        run_ensemble_evaluation(
            args.primary,
            args.secondary,
            train_path=args.train,
            train_ids_path=args.train_ids,
            output_prefix=args.output_prefix,
            seed=args.seed,
            shrinkage=args.shrinkage,
        )
    )


if __name__ == "__main__":
    main()
