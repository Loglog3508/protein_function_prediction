import json

import numpy as np
import pandas as pd
import pytest

from src.train_esm_labelwise import (
    fit_labelwise_scores,
    run_labelwise,
    screen_regularization_by_support,
    _validate_canonical_split,
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


def test_screening_representatives_ignore_inner_eval_support(monkeypatch):
    rng = np.random.default_rng(7)
    train_x = rng.normal(size=(16, 3)).astype(np.float32)
    train_y = np.zeros((16, 2), dtype=np.uint8)
    train_y[:5, 0] = 1
    train_y[:4, 1] = 1
    changed = train_y.copy()
    changed[12:, 1] = 1
    monkeypatch.setattr(
        "src.train_esm_labelwise.train_test_split",
        lambda indices, test_size, random_state: (
            np.arange(12),
            np.arange(12, 16),
        ),
    )
    config = _config(
        screening={"alphas": [0.01, 1.0], "label_count": 1, "validation_size": 0.25},
        support_strata=[{"name": "all", "min_support": 0}],
    )

    baseline = screen_regularization_by_support(train_x, train_y, config)
    result = screen_regularization_by_support(train_x, changed, config)

    pd.testing.assert_frame_equal(baseline, result)


def test_production_split_requires_ordered_iteration4_tail_and_seed42():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    split = root / "artifacts" / "metrics" / "splits" / "iteration4_tail"
    early = pd.read_csv(split / "train_ids.csv")["protein_id"].astype(str).tolist()
    tail = pd.read_csv(split / "validation_ids.csv")["protein_id"].astype(str).tolist()
    config = {
        "seed": 42,
        "split": {
            "kind": "iteration4_tail",
            "train_ids": "artifacts/metrics/splits/iteration4_tail/train_ids.csv",
            "validation_ids": "artifacts/metrics/splits/iteration4_tail/validation_ids.csv",
        },
    }
    assert _validate_canonical_split(config, root, early, tail) is None
    with pytest.raises(ValueError, match="seed 42"):
        _validate_canonical_split({**config, "seed": 7}, root, early, tail)
    with pytest.raises(ValueError, match="canonical iteration4_tail"):
        _validate_canonical_split(config, root, early, tail[::-1])
    wrong = {**config, "split": {**config["split"], "kind": "seed42"}}
    with pytest.raises(ValueError, match="iteration4_tail"):
        _validate_canonical_split(wrong, root, early, tail)
    wrong = {**config, "split": {**config["split"], "validation_ids": "artifacts/metrics/splits/seed42/validation_ids.csv"}}
    with pytest.raises(ValueError, match="iteration4_tail"):
        _validate_canonical_split(wrong, root, early, tail)


def _write_labelwise_fixture(root, tail_labels):
    root.mkdir(parents=True)
    ids = [f"P{i:06d}" for i in range(6)]
    target = [0, 1, 0, 1, *tail_labels]
    pd.DataFrame({"protein_id": ids, "sequence": ["ACD"] * 6, "label_0": target}).to_csv(root / "train.csv", index=False)
    pd.DataFrame({"protein_id": ids[:4]}).to_csv(root / "early.csv", index=False)
    pd.DataFrame({"protein_id": ids[4:]}).to_csv(root / "tail.csv", index=False)
    np.savez_compressed(root / "train-embeddings.npz", protein_ids=ids, embeddings=np.arange(6, dtype=float).reshape(6, 1))
    np.savez_compressed(root / "test-embeddings.npz", protein_ids=["T1", "T2", "T3"], embeddings=np.arange(3, dtype=float).reshape(3, 1))
    config = {
        "experiment_id": "EXP-TEST-ESM",
        "seed": 42,
        "fixture_mode": True,
        "data": {"train_path": "train.csv"},
        "split": {"train_ids": "early.csv", "validation_ids": "tail.csv"},
        "embeddings": {"train": "train-embeddings.npz", "test": "test-embeddings.npz"},
        "model": {"type": "sgd", "alpha": 0.001},
        "output_dir": "run",
        "metrics_prefix": "metrics/esm",
    }
    path = root / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_validation_fit_excludes_tail_and_test_fit_includes_all_labels(tmp_path, monkeypatch):
    calls = []
    screens = []

    def score_by_training_support(train_x, train_y, eval_x, config):
        calls.append((train_x[:, 0].tolist(), train_y[:, 0].tolist(), len(eval_x)))
        return np.full((len(eval_x), train_y.shape[1]), float(train_y[:, 0].mean()), dtype=np.float32)

    def screen_early(train_x, train_y, config, **kwargs):
        screens.append((len(train_x), train_y[:, 0].tolist()))
        return pd.DataFrame([{"stratum": "all", "alpha": .001, "selected": True}])

    monkeypatch.setattr("src.train_esm_labelwise.screen_regularization_by_support", screen_early)
    monkeypatch.setattr("src.train_esm_labelwise.fit_labelwise_scores", score_by_training_support)
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_summary_path = run_labelwise(_write_labelwise_fixture(first, [0, 0]), project_root=first)
    run_labelwise(_write_labelwise_fixture(second, [1, 1]), project_root=second)
    first_summary = json.loads(first_summary_path.read_text(encoding="utf-8"))
    assert first_summary["train_rows"] == 4
    assert first_summary["final_fit_rows"] == 6
    with np.load(first / "run" / "scores.npz", allow_pickle=False) as saved:
        first_validation = saved["validation_scores"].copy()
        first_test = saved["test_scores"].copy()
        assert saved["validation_ids"].tolist() == ["P000004", "P000005"]
        assert first_validation.shape == (2, 1)
        assert first_test.shape == (3, 1)
    with np.load(second / "run" / "scores.npz", allow_pickle=False) as saved:
        np.testing.assert_allclose(first_validation, saved["validation_scores"])
        assert not np.allclose(first_test, saved["test_scores"])
    assert [call[0] for call in calls] == [[0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]] * 2
    assert calls[0][1] == [0, 1, 0, 1]
    assert calls[1][1] == [0, 1, 0, 1, 0, 0]
    assert screens == [(4, [0, 1, 0, 1]), (4, [0, 1, 0, 1])]


def test_labelwise_run_rejects_duplicate_training_protein_ids(tmp_path):
    root = tmp_path / "duplicate"
    config_path = _write_labelwise_fixture(root, [0, 1])
    frame = pd.read_csv(root / "train.csv")
    frame.loc[5, "protein_id"] = frame.loc[4, "protein_id"]
    frame.to_csv(root / "train.csv", index=False)
    with pytest.raises(ValueError, match="training protein IDs must be unique"):
        run_labelwise(config_path, project_root=root)
    assert not (root / "run").exists()


def test_full_refit_keeps_early_selected_label_strata(monkeypatch):
    def score_by_alpha(train_x, target, eval_x, *, model_config, **kwargs):
        return None, np.full((len(eval_x), target.shape[1]), model_config["alpha"], dtype=np.float32)

    monkeypatch.setattr("src.train_esm_labelwise.fit_label_models", score_by_alpha)
    config = _config(
        model={"type": "sgd", "alpha": 0.01},
        support_strata=[{"name": "high", "min_support": 3}, {"name": "low", "max_support": 2}],
        selected_regularization={"high": {"alpha": 0.1}, "low": {"alpha": 0.2}},
        fixed_label_strata=["low", "high"],
    )
    scores = fit_labelwise_scores(
        np.arange(10, dtype=np.float32).reshape(5, 2),
        np.array([[1, 0], [1, 0], [1, 0], [1, 1], [0, 0]], dtype=np.uint8),
        np.zeros((2, 2), dtype=np.float32),
        config,
    )
    np.testing.assert_allclose(scores, [[0.2, 0.1], [0.2, 0.1]])
