"""Support-stratified, OOF-safe score calibration helpers."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


DEFAULT_SUPPORT_BINS = (20, 100, 1000)


def _validate(scores, target, training_support):
    scores = np.asarray(scores, dtype=np.float64)
    target = np.asarray(target)
    training_support = np.asarray(training_support, dtype=np.int64)
    if scores.ndim != 2 or target.shape != scores.shape:
        raise ValueError("scores and target must have equal two-dimensional shapes")
    if training_support.shape != (scores.shape[1],):
        raise ValueError("training support must match the label count")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("scores must be finite and in [0, 1]")
    if not np.isin(target, [0, 1]).all() or (training_support < 0).any():
        raise ValueError("target must be binary and support must be non-negative")
    return scores, target.astype(np.uint8, copy=False), training_support


def _strata(training_support, bins):
    bins = tuple(int(value) for value in bins)
    if any(value <= 0 for value in bins) or tuple(sorted(set(bins))) != bins:
        raise ValueError("support bins must be positive and strictly increasing")
    edges = (0, *bins, np.iinfo(np.int64).max)
    return np.searchsorted(np.asarray(edges[1:], dtype=np.int64), training_support, side="left")


def _logit(values):
    values = np.clip(values, 1e-6, 1 - 1e-6)
    return np.log(values / (1 - values))


def fit_support_calibrators(
    scores: np.ndarray,
    target: np.ndarray,
    training_support: np.ndarray,
    *,
    support_bins: tuple[int, ...] = DEFAULT_SUPPORT_BINS,
) -> dict:
    """Fit one shared Platt calibrator per training-support stratum."""
    scores, target, training_support = _validate(scores, target, training_support)
    strata = _strata(training_support, support_bins)
    calibrators = {}
    for stratum in np.unique(strata):
        labels = np.flatnonzero(strata == stratum)
        x = _logit(scores[:, labels]).reshape(-1, 1)
        y = target[:, labels].reshape(-1)
        key = str(int(stratum))
        if np.unique(y).size < 2:
            calibrators[key] = {"constant": float(y.mean())}
            continue
        model = LogisticRegression(solver="lbfgs", C=1.0, max_iter=200)
        model.fit(x, y)
        calibrators[key] = {
            "coef": float(model.coef_[0, 0]),
            "intercept": float(model.intercept_[0]),
        }
    return {"support_bins": list(support_bins), "calibrators": calibrators}


def transform_support_calibrated_scores(
    scores: np.ndarray,
    calibrators: dict,
    training_support: np.ndarray,
) -> np.ndarray:
    """Apply previously fitted support-stratified calibrators."""
    scores = np.asarray(scores, dtype=np.float64)
    training_support = np.asarray(training_support, dtype=np.int64)
    if scores.ndim != 2 or training_support.shape != (scores.shape[1],):
        raise ValueError("scores and training support must have compatible shapes")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("scores must be finite and in [0, 1]")
    if not isinstance(calibrators, dict) or "calibrators" not in calibrators:
        raise ValueError("calibrators must be fitted support calibrators")
    strata = _strata(training_support, tuple(calibrators.get("support_bins", DEFAULT_SUPPORT_BINS)))
    transformed = np.empty_like(scores, dtype=np.float64)
    for stratum in np.unique(strata):
        key = str(int(stratum))
        if key not in calibrators["calibrators"]:
            raise ValueError("calibrators do not cover every support stratum")
        params = calibrators["calibrators"][key]
        labels = np.flatnonzero(strata == stratum)
        if "constant" in params:
            transformed[:, labels] = params["constant"]
            continue
        logits = float(params["coef"]) * _logit(scores[:, labels]) + float(params["intercept"])
        transformed[:, labels] = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
    return transformed.astype(np.float32)
