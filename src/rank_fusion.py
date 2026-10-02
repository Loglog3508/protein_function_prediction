"""Leakage-safe rank-boundary thresholding for two score sources."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .metrics import macro_f1_skip_empty, macro_roc_auc_skip_degenerate
from .thresholds import (
    select_label_thresholds_exact,
    shrink_label_thresholds,
    threshold_predictions,
)


def _validate_inputs(
    target: np.ndarray, base_scores: np.ndarray, new_scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target = np.asarray(target, dtype=np.uint8)
    base_scores = np.asarray(base_scores, dtype=np.float32)
    new_scores = np.asarray(new_scores, dtype=np.float32)
    if target.ndim != 2 or base_scores.shape != target.shape or new_scores.shape != target.shape:
        raise ValueError("target and score matrices must have equal two-dimensional shapes")
    if not np.isin(target, [0, 1]).all():
        raise ValueError("target must be binary")
    if not np.isfinite(base_scores).all() or not np.isfinite(new_scores).all():
        raise ValueError("scores must be finite")
    if ((base_scores < 0) | (base_scores > 1)).any() or ((new_scores < 0) | (new_scores > 1)).any():
        raise ValueError("scores must be between 0 and 1")
    return target, base_scores, new_scores


def fit_rank_fusion_thresholds(
    target: np.ndarray,
    scores: np.ndarray,
    *,
    shrinkage: float = 20.0,
    global_candidates: Iterable[float] | None = None,
) -> tuple[np.ndarray, float, pd.DataFrame]:
    """Fit exact per-label F1 thresholds and shrink them toward a shared median."""
    target = np.asarray(target, dtype=np.uint8)
    scores = np.asarray(scores, dtype=np.float32)
    if target.ndim != 2 or scores.shape != target.shape:
        raise ValueError("target and scores must have equal two-dimensional shapes")
    if not np.isin(target, [0, 1]).all():
        raise ValueError("target must be binary")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("scores must be finite and between 0 and 1")
    if shrinkage < 0:
        raise ValueError("shrinkage must be non-negative")
    label_thresholds, diagnostics = select_label_thresholds_exact(target, scores)
    support = target.sum(axis=0).astype(np.float32)
    candidates = (
        np.arange(0.02, 1.0, 0.02, dtype=np.float32)
        if global_candidates is None
        else np.asarray(tuple(global_candidates), dtype=np.float32)
    )
    if candidates.ndim != 1 or not len(candidates) or ((candidates < 0) | (candidates > 1)).any():
        raise ValueError("global threshold candidates must be non-empty and between 0 and 1")
    global_threshold = 0.5
    best_key = (-1.0, -float("inf"))
    for candidate in candidates:
        score = macro_f1_skip_empty(target, threshold_predictions(scores, float(candidate)))
        key = (score, -abs(float(candidate) - 0.5))
        if key > best_key:
            best_key = key
            global_threshold = float(candidate)
    thresholds = shrink_label_thresholds(
        label_thresholds,
        support,
        global_threshold=global_threshold,
        shrinkage=float(shrinkage),
    )
    return thresholds, global_threshold, diagnostics


def crossfit_rank_fusion(
    target: np.ndarray,
    base_scores: np.ndarray,
    new_scores: np.ndarray,
    *,
    weights: Iterable[float],
    shrinkages: Iterable[float],
    seeds: Iterable[int],
) -> pd.DataFrame:
    """Evaluate fusion weights with two-way, selection-fold-only threshold fitting."""
    target, base_scores, new_scores = _validate_inputs(target, base_scores, new_scores)
    weights = tuple(float(value) for value in weights)
    shrinkages = tuple(float(value) for value in shrinkages)
    seeds = tuple(int(value) for value in seeds)
    if not weights or not shrinkages or not seeds:
        raise ValueError("weights, shrinkages, and seeds must be non-empty")
    if any(value < 0 or value > 1 for value in weights):
        raise ValueError("fusion weights must be between 0 and 1")
    if any(value < 0 for value in shrinkages):
        raise ValueError("threshold shrinkages must be non-negative")
    rows: list[dict[str, float | int]] = []
    for new_weight in weights:
        scores = ((1.0 - new_weight) * base_scores + new_weight * new_scores).astype(np.float32)
        for shrinkage in shrinkages:
            for seed in seeds:
                first, second = np.array_split(np.random.default_rng(seed).permutation(len(target)), 2)
                fold_rows: list[dict[str, float | int]] = []
                combined_predictions = np.zeros_like(target, dtype=np.uint8)
                for fold, (selection_index, evaluation_index) in enumerate(
                    ((first, second), (second, first))
                ):
                    thresholds, _, _ = fit_rank_fusion_thresholds(
                        target[selection_index], scores[selection_index], shrinkage=shrinkage
                    )
                    predictions = threshold_predictions(scores[evaluation_index], thresholds)
                    combined_predictions[evaluation_index] = predictions
                    fold_rows.append(
                        {
                            "new_weight": new_weight,
                            "shrinkage": shrinkage,
                            "seed": seed,
                            "selection_fold": 1 - fold,
                            "evaluation_fold": fold,
                            "macro_f1": macro_f1_skip_empty(target[evaluation_index], predictions),
                            "continuous_macro_auc": macro_roc_auc_skip_degenerate(
                                target[evaluation_index], scores[evaluation_index]
                            ),
                            "predicted_positive_rate": float(predictions.mean()),
                        }
                    )
                seed_macro_f1 = macro_f1_skip_empty(target, combined_predictions)
                seed_auc = macro_roc_auc_skip_degenerate(target, scores)
                for row in fold_rows:
                    row["seed_macro_f1"] = seed_macro_f1
                    row["seed_continuous_macro_auc"] = seed_auc
                    row["seed_predicted_positive_rate"] = float(combined_predictions.mean())
                rows.extend(fold_rows)
    return pd.DataFrame(rows)
