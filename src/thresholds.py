"""Threshold selection and validation for multilabel predictions."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import macro_f1_skip_empty


def _validate_inputs(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    if y_true.shape != scores.shape or y_true.ndim != 2:
        raise ValueError("y_true and scores must be two-dimensional with equal shape")
    if not np.isfinite(scores).all():
        raise ValueError("scores must contain only finite values")
    if ((scores < 0) | (scores > 1)).any():
        raise ValueError("scores must be between 0 and 1")
    return y_true.astype(np.uint8, copy=False), scores.astype(np.float32, copy=False)


def threshold_predictions(scores: np.ndarray, thresholds: float | np.ndarray) -> np.ndarray:
    """Convert positive scores into binary predictions."""
    scores = np.asarray(scores)
    thresholds = np.asarray(thresholds)
    if scores.ndim != 2:
        raise ValueError("scores must be two-dimensional")
    if thresholds.ndim == 0:
        return (scores >= thresholds).astype(np.uint8)
    if thresholds.shape != (scores.shape[1],):
        raise ValueError("per-label thresholds must match score columns")
    return (scores >= thresholds[np.newaxis, :]).astype(np.uint8)


def _candidate_thresholds(candidates: np.ndarray | None) -> np.ndarray:
    if candidates is None:
        candidates = np.arange(0.05, 1.0, 0.05, dtype=np.float32)
    candidates = np.asarray(candidates, dtype=np.float32)
    if candidates.ndim != 1 or len(candidates) == 0:
        raise ValueError("threshold candidates must be a non-empty one-dimensional array")
    if ((candidates < 0) | (candidates > 1)).any():
        raise ValueError("threshold candidates must be between 0 and 1")
    return np.unique(candidates)


def scan_global_thresholds(
    y_true: np.ndarray,
    scores: np.ndarray,
    candidates: np.ndarray | None = None,
) -> pd.DataFrame:
    """Evaluate one threshold shared by all labels."""
    y_true, scores = _validate_inputs(y_true, scores)
    candidates = _candidate_thresholds(candidates)
    rows = []
    for threshold in candidates:
        predictions = threshold_predictions(scores, float(threshold))
        rows.append(
            {
                "threshold": float(threshold),
                "macro_f1": macro_f1_skip_empty(y_true, predictions),
                "predicted_positive_rate": float(predictions.mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["macro_f1", "threshold"], ascending=[False, True], ignore_index=True
    )


def select_global_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    candidates: np.ndarray | None = None,
) -> tuple[float, pd.DataFrame]:
    """Select the best shared threshold and return the complete scan."""
    scan = scan_global_thresholds(y_true, scores, candidates)
    return float(scan.iloc[0]["threshold"]), scan


def select_label_thresholds(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    global_threshold: float = 0.5,
    candidates: np.ndarray | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Select a threshold per label using label-wise F1 on the supplied split."""
    y_true, scores = _validate_inputs(y_true, scores)
    candidates = _candidate_thresholds(candidates)
    thresholds = np.full(scores.shape[1], global_threshold, dtype=np.float32)
    rows = []
    for label_index in range(scores.shape[1]):
        target = y_true[:, label_index]
        support = int(target.sum())
        best_threshold = float(global_threshold)
        best_f1 = 0.0
        if support:
            for threshold in candidates:
                prediction = scores[:, label_index] >= threshold
                true_positive = int((target & prediction).sum())
                false_positive = int(((target == 0) & prediction).sum())
                false_negative = int((target & ~prediction).sum())
                denominator = 2 * true_positive + false_positive + false_negative
                f1 = (2 * true_positive / denominator) if denominator else 0.0
                if f1 > best_f1 or (
                    f1 == best_f1
                    and abs(float(threshold) - global_threshold)
                    < abs(best_threshold - global_threshold)
                ):
                    best_f1 = f1
                    best_threshold = float(threshold)
        thresholds[label_index] = best_threshold
        rows.append(
            {
                "label_index": label_index,
                "support": support,
                "threshold": best_threshold,
                "f1": best_f1,
                "used_global_fallback": support == 0,
            }
        )
    return thresholds, pd.DataFrame(rows)


def shrink_label_thresholds(
    label_thresholds: np.ndarray,
    supports: np.ndarray,
    *,
    global_threshold: float,
    shrinkage: float,
) -> np.ndarray:
    """Shrink rare-label thresholds toward a shared threshold."""
    label_thresholds = np.asarray(label_thresholds, dtype=np.float32)
    supports = np.asarray(supports, dtype=np.float32)
    if label_thresholds.ndim != 1 or supports.shape != label_thresholds.shape:
        raise ValueError("label thresholds and supports must have equal one-dimensional shape")
    if shrinkage < 0:
        raise ValueError("shrinkage must be non-negative")
    weight = supports / (supports + shrinkage) if shrinkage else np.ones_like(supports)
    return (weight * label_thresholds + (1.0 - weight) * global_threshold).astype(
        np.float32
    )


def samplewise_label_count_summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Summarize the true and predicted number of labels per sample."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape or y_true.ndim != 2:
        raise ValueError("y_true and y_pred must be two-dimensional with equal shape")
    summary = {}
    for name, values in (("true", y_true.sum(axis=1)), ("predicted", y_pred.sum(axis=1))):
        summary[name] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "p90": float(np.quantile(values, 0.9)),
            "max": int(values.max()),
        }
    return summary


def crossfit_shrunk_threshold_score(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    seed: int = 42,
    shrinkage: float = 25.0,
    candidates: np.ndarray | None = None,
) -> dict:
    """Evaluate shrunk per-label thresholds with two-fold cross-fitting."""
    y_true, scores = _validate_inputs(y_true, scores)
    candidates = _candidate_thresholds(candidates)
    rng = np.random.default_rng(seed)
    first, second = np.array_split(rng.permutation(len(y_true)), 2)
    predictions = np.zeros_like(y_true, dtype=np.uint8)
    fold_thresholds = []
    for selection_index, evaluation_index in ((first, second), (second, first)):
        global_threshold, _ = select_global_threshold(
            y_true[selection_index], scores[selection_index], candidates
        )
        label_thresholds, _ = select_label_thresholds(
            y_true[selection_index],
            scores[selection_index],
            global_threshold=global_threshold,
            candidates=candidates,
        )
        thresholds = shrink_label_thresholds(
            label_thresholds,
            y_true[selection_index].sum(axis=0),
            global_threshold=global_threshold,
            shrinkage=shrinkage,
        )
        fold_thresholds.append(thresholds)
        predictions[evaluation_index] = threshold_predictions(
            scores[evaluation_index], thresholds
        )
    return {
        "macro_f1": macro_f1_skip_empty(y_true, predictions),
        "predicted_positive_rate": float(predictions.mean()),
        "fold_thresholds": fold_thresholds,
    }


def fit_shrunk_thresholds(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    shrinkage: float = 25.0,
    candidates: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Fit final shrunk per-label thresholds on all supplied rows."""
    y_true, scores = _validate_inputs(y_true, scores)
    candidates = _candidate_thresholds(candidates)
    global_threshold, _ = select_global_threshold(y_true, scores, candidates)
    label_thresholds, _ = select_label_thresholds(
        y_true,
        scores,
        global_threshold=global_threshold,
        candidates=candidates,
    )
    return (
        shrink_label_thresholds(
            label_thresholds,
            y_true.sum(axis=0),
            global_threshold=global_threshold,
            shrinkage=shrinkage,
        ),
        global_threshold,
    )


def _resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def run_threshold_optimization(
    scores_path: str | Path,
    train_path: str | Path,
    output_prefix: str | Path,
    *,
    project_root: str | Path | None = None,
    seed: int = 42,
    holdout_fraction: float = 0.5,
    shrinkage: float = 25.0,
    max_labels: int | None = None,
) -> Path:
    """Optimize thresholds from saved validation scores and persist diagnostics."""
    root = Path(project_root) if project_root is not None else Path.cwd()
    with np.load(_resolve(root, scores_path), allow_pickle=False) as saved:
        scores = saved["validation_scores"]
        validation_ids = saved["validation_ids"].astype(str)
        labels = saved["label_columns"].astype(str).tolist()
    if max_labels is not None:
        if max_labels <= 0:
            raise ValueError("max_labels must be positive")
        labels = labels[:max_labels]
        scores = scores[:, :max_labels]
    train = pd.read_csv(_resolve(root, train_path), usecols=["protein_id", *labels])
    indexed = train.set_index("protein_id", drop=False)
    if not pd.Index(validation_ids).isin(indexed.index).all():
        raise ValueError("saved validation IDs are absent from training data")
    target = indexed.loc[validation_ids, labels].to_numpy(dtype=np.uint8)
    target, scores = _validate_inputs(target, scores)
    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be between 0 and 1")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(target))
    split_at = max(1, min(len(target) - 1, int(len(target) * (1 - holdout_fraction))))
    calibration_index = order[:split_at]
    holdout_index = order[split_at:]
    candidates = np.arange(0.05, 1.0, 0.05, dtype=np.float32)
    global_threshold, global_scan = select_global_threshold(
        target[calibration_index], scores[calibration_index], candidates
    )
    label_thresholds, label_scan = select_label_thresholds(
        target[calibration_index],
        scores[calibration_index],
        global_threshold=global_threshold,
        candidates=candidates,
    )
    supports = target[calibration_index].sum(axis=0)
    shrunk_thresholds = shrink_label_thresholds(
        label_thresholds,
        supports,
        global_threshold=global_threshold,
        shrinkage=shrinkage,
    )
    strategies = {
        "fixed_0.5": np.full(len(labels), 0.5, dtype=np.float32),
        "global_selected": np.full(len(labels), global_threshold, dtype=np.float32),
        "per_label_selected": label_thresholds,
        "per_label_shrunk": shrunk_thresholds,
    }
    strategy_rows = []
    for name, thresholds in strategies.items():
        calibration_predictions = threshold_predictions(scores[calibration_index], thresholds)
        holdout_predictions = threshold_predictions(scores[holdout_index], thresholds)
        strategy_rows.append(
            {
                "strategy": name,
                "calibration_macro_f1": macro_f1_skip_empty(
                    target[calibration_index], calibration_predictions
                ),
                "holdout_macro_f1": macro_f1_skip_empty(
                    target[holdout_index], holdout_predictions
                ),
                "calibration_predicted_positive_rate": float(calibration_predictions.mean()),
                "holdout_predicted_positive_rate": float(holdout_predictions.mean()),
            }
        )

    crossfit_predictions = np.zeros_like(target, dtype=np.uint8)
    for selection_index, evaluation_index in (
        (calibration_index, holdout_index),
        (holdout_index, calibration_index),
    ):
        fold_global, _ = select_global_threshold(
            target[selection_index], scores[selection_index], candidates
        )
        fold_label_thresholds, _ = select_label_thresholds(
            target[selection_index],
            scores[selection_index],
            global_threshold=fold_global,
            candidates=candidates,
        )
        fold_thresholds = shrink_label_thresholds(
            fold_label_thresholds,
            target[selection_index].sum(axis=0),
            global_threshold=fold_global,
            shrinkage=shrinkage,
        )
        crossfit_predictions[evaluation_index] = threshold_predictions(
            scores[evaluation_index], fold_thresholds
        )

    final_global_threshold, final_global_scan = select_global_threshold(
        target, scores, candidates
    )
    final_label_thresholds, final_label_scan = select_label_thresholds(
        target,
        scores,
        global_threshold=final_global_threshold,
        candidates=candidates,
    )
    final_thresholds = shrink_label_thresholds(
        final_label_thresholds,
        target.sum(axis=0),
        global_threshold=final_global_threshold,
        shrinkage=shrinkage,
    )
    final_predictions = threshold_predictions(scores, final_thresholds)

    output_prefix = _resolve(root, output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    thresholds_path = output_prefix.with_name(output_prefix.name + "-thresholds.json")
    scan_path = output_prefix.with_name(output_prefix.name + "-global-scan.csv")
    labels_path = output_prefix.with_name(output_prefix.name + "-per-label.csv")
    summary_path = output_prefix.with_name(output_prefix.name + "-summary.json")
    final_global_scan.to_csv(scan_path, index=False)
    final_label_scan = final_label_scan.copy()
    final_label_scan["label"] = labels
    final_label_scan["shrunk_threshold"] = final_thresholds
    final_label_scan["calibration_threshold"] = label_thresholds
    final_label_scan["calibration_shrunk_threshold"] = shrunk_thresholds
    final_label_scan.to_csv(labels_path, index=False)
    thresholds_path.write_text(
        json.dumps(
            {
                "strategy": "per_label_shrunk",
                "selection_data": "full_validation_after_holdout_validation",
                "global_threshold": final_global_threshold,
                "shrinkage": shrinkage,
                "labels": labels,
                "thresholds": final_thresholds.tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    summary = {
        "scores_path": str(scores_path),
        "train_path": str(train_path),
        "validation_rows": len(target),
        "calibration_rows": len(calibration_index),
        "holdout_rows": len(holdout_index),
        "seed": seed,
        "holdout_fraction": holdout_fraction,
        "strategies": strategy_rows,
        "crossfit_per_label_shrunk": {
            "macro_f1": macro_f1_skip_empty(target, crossfit_predictions),
            "predicted_positive_rate": float(crossfit_predictions.mean()),
        },
        "final_full_validation_fit": {
            "global_threshold": final_global_threshold,
            "macro_f1": macro_f1_skip_empty(target, final_predictions),
            "predicted_positive_rate": float(final_predictions.mean()),
        },
        "true_positive_rate": float(target.mean()),
        "samplewise_label_counts": {
            name: samplewise_label_count_summary(
                target[holdout_index], threshold_predictions(scores[holdout_index], thresholds)
            )
            for name, thresholds in strategies.items()
        },
        "outputs": {
            "thresholds": str(thresholds_path),
            "global_scan": str(scan_path),
            "per_label": str(labels_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--train", default="data/train.csv")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout-fraction", type=float, default=0.5)
    parser.add_argument("--shrinkage", type=float, default=25.0)
    parser.add_argument("--max-labels", type=int)
    args = parser.parse_args()
    print(
        run_threshold_optimization(
            args.scores,
            args.train,
            args.output_prefix,
            seed=args.seed,
            holdout_fraction=args.holdout_fraction,
            shrinkage=args.shrinkage,
            max_labels=args.max_labels,
        )
    )


if __name__ == "__main__":
    main()
