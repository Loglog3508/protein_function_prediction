"""Competition metrics and diagnostics."""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score


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
