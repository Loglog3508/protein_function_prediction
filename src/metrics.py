"""Competition metrics and diagnostics."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def macro_f1_skip_empty(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute label-wise Macro F1, excluding labels without true positives."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape or y_true.ndim != 2:
        raise ValueError("y_true and y_pred must be two-dimensional with equal shape")
    active = y_true.sum(axis=0) > 0
    if not active.any():
        raise ValueError("at least one label must contain a positive sample")
    scores = f1_score(y_true[:, active], y_pred[:, active], average=None, zero_division=0)
    return float(np.mean(scores))


def macro_roc_auc_skip_degenerate(
    y_true: np.ndarray, y_score: np.ndarray
) -> float:
    """Compute label-wise Macro ROC AUC, excluding single-class labels."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_true.shape != y_score.shape or y_true.ndim != 2:
        raise ValueError("y_true and y_score must be two-dimensional with equal shape")
    positive = y_true.sum(axis=0)
    active = (positive > 0) & (positive < len(y_true))
    if not active.any():
        raise ValueError("at least one label must contain both classes")
    return float(roc_auc_score(y_true[:, active], y_score[:, active], average="macro"))


def per_label_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]
) -> pd.DataFrame:
    """Return per-label support, errors, F1, and prediction prevalence."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape or y_true.ndim != 2:
        raise ValueError("y_true and y_pred must be two-dimensional with equal shape")
    if y_true.shape[1] != len(labels):
        raise ValueError("label names must match prediction columns")
    true_positive = ((y_true == 1) & (y_pred == 1)).sum(axis=0)
    false_positive = ((y_true == 0) & (y_pred == 1)).sum(axis=0)
    false_negative = ((y_true == 1) & (y_pred == 0)).sum(axis=0)
    denominator = 2 * true_positive + false_positive + false_negative
    f1 = np.divide(
        2 * true_positive,
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator != 0,
    )
    return pd.DataFrame(
        {
            "label": labels,
            "support": y_true.sum(axis=0),
            "predicted_positive": y_pred.sum(axis=0),
            "predicted_positive_rate": y_pred.mean(axis=0),
            "tp": true_positive,
            "fp": false_positive,
            "fn": false_negative,
            "f1": f1,
        }
    )


def per_label_score_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    labels: list[str],
    training_support: np.ndarray,
) -> pd.DataFrame:
    """Combine binary errors, score ranking, and support for each label."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_score = np.asarray(y_score, dtype=np.float64)
    training_support = np.asarray(training_support)
    if y_true.ndim != 2 or y_pred.shape != y_true.shape or y_score.shape != y_true.shape:
        raise ValueError("target, predictions, and scores must have equal two-dimensional shapes")
    if len(labels) != y_true.shape[1] or training_support.shape != (y_true.shape[1],):
        raise ValueError("labels and training support must match the label count")
    if not np.isfinite(y_score).all():
        raise ValueError("scores must be finite")
    if not np.isin(y_true, [0, 1]).all() or not np.isin(y_pred, [0, 1]).all():
        raise ValueError("targets and predictions must be binary")
    if (training_support < 0).any():
        raise ValueError("training support must be non-negative")
    true_positive = ((y_true == 1) & (y_pred == 1)).sum(axis=0)
    false_positive = ((y_true == 0) & (y_pred == 1)).sum(axis=0)
    false_negative = ((y_true == 1) & (y_pred == 0)).sum(axis=0)
    true_support = y_true.sum(axis=0)
    predicted_positive = y_pred.sum(axis=0)
    precision = np.divide(
        true_positive,
        predicted_positive,
        out=np.zeros_like(true_positive, dtype=np.float64),
        where=predicted_positive != 0,
    )
    recall = np.divide(
        true_positive,
        true_support,
        out=np.zeros_like(true_positive, dtype=np.float64),
        where=true_support != 0,
    )
    ratio = np.divide(
        predicted_positive,
        true_support,
        out=np.full(true_support.shape, np.nan, dtype=np.float64),
        where=true_support != 0,
    )
    average_precision = np.full(true_support.shape, np.nan, dtype=np.float64)
    for index in np.flatnonzero(true_support > 0):
        average_precision[index] = average_precision_score(
            y_true[:, index], y_score[:, index]
        )
    return pd.DataFrame(
        {
            "label": labels,
            "training_support": training_support.astype(np.int64),
            "true_support": true_support,
            "predicted_positive": predicted_positive,
            "predicted_to_true_ratio": ratio,
            "tp": true_positive,
            "fp": false_positive,
            "fn": false_negative,
            "precision": precision,
            "recall": recall,
            "f1": np.divide(
                2 * true_positive,
                2 * true_positive + false_positive + false_negative,
                out=np.zeros_like(true_positive, dtype=np.float64),
                where=(2 * true_positive + false_positive + false_negative) != 0,
            ),
            "average_precision": average_precision,
        }
    )
