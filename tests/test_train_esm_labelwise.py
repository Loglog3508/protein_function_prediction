import json

import numpy as np
import pandas as pd
import pytest

from src.train_esm_labelwise import (
    fit_labelwise_scores,
    screen_regularization_by_support,
)


def _config(**overrides):
    config = {
        "seed": 42,
        "model": {
            "type": "logistic_regression",
            "C": 1.0,
            "class_weight": "balanced",
            "max_iter": 200,
        },
    }
    config.update(overrides)
    return config


def test_labelwise_scores_fit_each_column_independently():
    train_x = np.array(
        [[-2.0, 0.0], [-1.0, 0.2], [0.0, 0.0], [1.0, 0.2], [2.0, 0.0]],
        dtype=np.float32,
    )
    train_y = np.array(
        [[0, 1], [0, 1], [0, 0], [1, 0], [1, 0]], dtype=np.uint8
    )
    eval_x = np.array([[-1.5, 0.1], [1.5, 0.1]], dtype=np.float32)

    scores = fit_labelwise_scores(train_x, train_y, eval_x, _config())

    assert scores.shape == (len(eval_x), train_y.shape[1])
    assert not np.allclose(scores[:, 0], scores[:, 1])
    assert np.isfinite(scores).all()
    assert ((scores >= 0.0) & (scores <= 1.0)).all()


def test_single_class_label_uses_finite_constant_score():
    train_x = np.arange(8, dtype=np.float32).reshape(4, 2)
    eval_x = np.array([[0.5, 1.0], [3.0, 2.0]], dtype=np.float32)

    scores = fit_labelwise_scores(
        train_x, np.zeros((4, 1), dtype=np.uint8), eval_x, _config()
    )

    assert np.isfinite(scores).all()
    assert np.unique(scores).size == 1
    assert scores[0, 0] == 0.0


def test_screening_uses_support_strata_and_inner_scores():
    rng = np.random.default_rng(42)
    train_x = rng.normal(size=(40, 3)).astype(np.float32)
    train_y = np.zeros((40, 4), dtype=np.uint8)
    train_y[:, 0] = np.arange(40) < 20
    train_y[:, 1] = np.arange(40) < 10
    train_y[:, 2] = np.arange(40) < 4
    train_y[:, 3] = np.arange(40) < 2
    config = _config(
        screening={
            "alphas": [0.001, 1.0],
            "label_count": 4,
            "validation_size": 0.25,
        },
        support_strata=[
            {"name": "high", "min_support": 10},
            {"name": "low", "max_support": 9},
        ],
    )

    result = screen_regularization_by_support(train_x, train_y, config)

    assert isinstance(result, pd.DataFrame)
    assert {"stratum", "alpha", "inner_macro_auc", "selected"}.issubset(
        result.columns
    )
    assert set(result["stratum"]) == {"high", "low"}
    assert result.groupby("stratum")["selected"].sum().eq(1).all()


def test_screening_rejects_nonfinite_features():
    with pytest.raises(ValueError, match="finite"):
        screen_regularization_by_support(
            np.array([[0.0], [np.inf]]),
            np.array([[0], [1]], dtype=np.uint8),
            _config(),
        )
