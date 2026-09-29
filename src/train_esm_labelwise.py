"""Fit independent classifiers on row-aligned ESM embeddings."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from .config import load_config
from .data import label_columns
from .metrics import macro_roc_auc_skip_degenerate
from .sweep_sgd import select_support_stratified_labels
from .train import fit_label_models


def _as_matrix(values: Any, name: str, *, rows: int | None = None) -> Any:
    matrix = values
    if getattr(matrix, "ndim", None) != 2:
        raise ValueError(f"{name} must be two-dimensional")
    if rows is not None and matrix.shape[0] != rows:
        raise ValueError(f"{name} row count does not match training labels")
    if hasattr(matrix, "data") and not isinstance(matrix, np.ndarray):
        finite = np.isfinite(matrix.data).all()
    else:
        finite = np.isfinite(np.asarray(matrix)).all()
    if not finite:
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def _validate_target(training_target: Any, rows: int) -> np.ndarray:
    target = np.asarray(training_target)
    if target.ndim != 2 or target.shape[0] != rows:
        raise ValueError("training_target must be two-dimensional with matching rows")
    if not np.isfinite(target).all() or not np.isin(target, [0, 1]).all():
        raise ValueError("training_target must contain only binary finite values")
    return target.astype(np.uint8, copy=False)


def _model_config(config: dict[str, Any]) -> dict[str, Any]:
    model = config.get("model", config)
    if not isinstance(model, dict):
        raise ValueError("model configuration must be an object")
    model = dict(model)
    model.setdefault("type", "sgd")
    if model["type"] not in {"sgd", "logistic_regression"}:
        raise ValueError("ESM label-wise models must use sgd or logistic_regression")
    return model


def _support_strata(config: dict[str, Any]) -> list[dict[str, Any]]:
    configured = config.get("support_strata")
    if configured is None:
        return [{"name": "all", "min_support": 0}]
    if not isinstance(configured, list) or not configured:
        raise ValueError("support_strata must be a non-empty list")
    strata: list[dict[str, Any]] = []
    names: set[str] = set()
    for item in configured:
        if not isinstance(item, dict) or not item.get("name"):
            raise ValueError("each support stratum needs a name")
        name = str(item["name"])
        if name in names:
            raise ValueError(f"duplicate support stratum: {name}")
        names.add(name)
        current = dict(item)
        if "min_support" in current:
            current["min_support"] = int(current["min_support"])
        if "max_support" in current:
            current["max_support"] = int(current["max_support"])
        if current.get("min_support", 0) < 0:
            raise ValueError("support bounds must be non-negative")
        if "max_support" in current and current["max_support"] < current.get("min_support", 0):
            raise ValueError("support stratum max_support must not be below min_support")
        strata.append(current)
    return strata


def _stratum_name(support: int, strata: list[dict[str, Any]]) -> str:
    for stratum in strata:
        minimum = int(stratum.get("min_support", 0))
        maximum = stratum.get("max_support")
        if support >= minimum and (maximum is None or support <= int(maximum)):
            return str(stratum["name"])
    # A zero-support label can appear after an inner split even when the
    # configured lower bound starts above zero. Keep it in the lowest-support
    # bucket so it receives the constant-score path during fitting.
    lowest = min(strata, key=lambda item: int(item.get("min_support", 0)))
    if support < int(lowest.get("min_support", 0)):
        return str(lowest["name"])
    raise ValueError(f"support count {support} does not belong to a support stratum")


def _regularization_model(
    base: dict[str, Any],
    config: dict[str, Any],
    stratum: dict[str, Any],
    selected: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model = dict(base)
    model.update(stratum.get("model", {}))
    by_stratum = config.get("regularization_by_stratum", {})
    if isinstance(by_stratum, dict) and stratum["name"] in by_stratum:
        value = by_stratum[stratum["name"]]
        if isinstance(value, dict):
            model.update(value)
        elif model["type"] == "sgd":
            model["alpha"] = float(value)
        else:
            model["C"] = float(value)
    if selected is not None:
        if model["type"] == "sgd":
            model["alpha"] = float(selected["alpha"])
        else:
            model["C"] = float(selected["alpha"])
    for key in ("alpha", "C"):
        if key in stratum:
            model[key] = float(stratum[key])
    return model


def _label_strata(target: np.ndarray, config: dict[str, Any]) -> tuple[list[str], np.ndarray]:
    supports = target.sum(axis=0).astype(int)
    strata = _support_strata(config)
    names = [_stratum_name(int(value), strata) for value in supports]
    return names, supports


def fit_labelwise_scores(
    train_x: Any, train_y: Any, eval_x: Any, config: dict[str, Any]
) -> np.ndarray:
    """Fit one independent binary model per label and return positive scores."""
    train_x = _as_matrix(train_x, "train_x")
    eval_x = _as_matrix(eval_x, "eval_x")
    if train_x.shape[1] != eval_x.shape[1]:
        raise ValueError("train_x and eval_x feature widths must match")
    target = _validate_target(train_y, train_x.shape[0])
    base_model = _model_config(config)
    strata = _support_strata(config)
    stratum_names, _ = _label_strata(target, config)

    selected_by_name: dict[str, dict[str, Any]] = {}
    selected = config.get("selected_regularization")
    if isinstance(selected, dict):
        selected_by_name = {str(key): value for key, value in selected.items()}

    scores = np.zeros((eval_x.shape[0], target.shape[1]), dtype=np.float32)
    for name in {stratum["name"] for stratum in strata}:
        indices = [index for index, value in enumerate(stratum_names) if value == name]
        if not indices:
            continue
        stratum = next(value for value in strata if value["name"] == name)
        model_config = _regularization_model(
            base_model, config, stratum, selected_by_name.get(name)
        )
        _, group_scores = fit_label_models(
            train_x,
            target[:, indices],
            eval_x,
            model_config=model_config,
            seed=int(config.get("seed", 42)),
            progress_every=None,
            retain_models=False,
        )
        scores[:, indices] = group_scores
    if not np.isfinite(scores).all():
        raise RuntimeError("label-wise classifier produced non-finite scores")
    return np.clip(scores, 0.0, 1.0)


def _screen_candidates(config: dict[str, Any], model: dict[str, Any]) -> list[dict[str, Any]]:
    screening = config.get("screening", {})
    candidates = screening.get("candidates")
    if candidates is not None:
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("screening candidates must be a non-empty list")
        values = []
        for candidate in candidates:
            if isinstance(candidate, dict):
                value = candidate.get("alpha", candidate.get("C"))
            else:
                value = candidate
            if value is None:
                raise ValueError("screening candidate needs alpha or C")
            values.append({"alpha": float(value)})
        return values
    values = screening.get("alphas")
    if values is None:
        values = screening.get("Cs")
    if values is None:
        values = [model.get("alpha", 0.0001) if model["type"] == "sgd" else model.get("C", 1.0)]
    return [{"alpha": float(value)} for value in values]


def _inner_macro_auc(target: np.ndarray, scores: np.ndarray) -> float:
    values = []
    for index in range(target.shape[1]):
        if np.unique(target[:, index]).size < 2:
            continue
        values.append(roc_auc_score(target[:, index], scores[:, index]))
    return float(np.mean(values)) if values else 0.0


def screen_regularization_by_support(
    train_x: Any,
    train_y: Any,
    config: dict[str, Any],
    *,
    label_names: list[str] | None = None,
) -> pd.DataFrame:
    """Choose regularization independently for support strata on an inner split."""
    train_x = _as_matrix(train_x, "train_x")
    target = _validate_target(train_y, train_x.shape[0])
    if train_x.shape[0] < 4:
        raise ValueError("at least four training rows are required for screening")
    if label_names is not None and len(label_names) != target.shape[1]:
        raise ValueError("label names must match training target columns")
    model = _model_config(config)
    screening = config.get("screening", {})
    validation_size = float(screening.get("validation_size", 0.2))
    if not 0 < validation_size < 1:
        raise ValueError("screening validation_size must be between zero and one")
    inner_train, inner_eval = train_test_split(
        np.arange(target.shape[0]),
        test_size=validation_size,
        random_state=int(config.get("seed", 42)),
    )
    strata = _support_strata(config)
    # Support grouping and representative-label selection are fit-only decisions.
    # Derive them from the inner partition so the inner evaluation labels remain
    # scoring-only observations.
    inner_stratum_names, inner_supports = _label_strata(target[inner_train], config)
    label_count = int(screening.get("label_count", target.shape[1]))
    selection = select_support_stratified_labels(
        pd.DataFrame(target, columns=label_names or [f"label_{i}" for i in range(target.shape[1])]),
        min(label_count, target.shape[1]),
    )
    selected_indices = set(selection["label_index"].astype(int).tolist())
    rows: list[dict[str, Any]] = []
    for stratum in strata:
        name = stratum["name"]
        indices = [
            index
            for index, value in enumerate(inner_stratum_names)
            if value == name and index in selected_indices
        ]
        if not indices:
            continue
        for candidate in _screen_candidates(config, model):
            candidate_model = _regularization_model(model, config, stratum, candidate)
            _, candidate_scores = fit_label_models(
                train_x[inner_train],
                target[inner_train][:, indices],
                train_x[inner_eval],
                model_config=candidate_model,
                seed=int(config.get("seed", 42)),
                retain_models=False,
            )
            score = _inner_macro_auc(target[inner_eval][:, indices], candidate_scores)
            rows.append(
                {
                    "stratum": str(name),
                    "alpha": float(candidate["alpha"]),
                    "inner_macro_auc": score,
                    "selected": False,
                    "label_count": len(indices),
                    "min_support": int(min(inner_supports[indices])),
                    "max_support": int(max(inner_supports[indices])),
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("screening selected no labels in any support stratum")
    for name, group in result.groupby("stratum", sort=False):
        winner = group.sort_values(["inner_macro_auc", "alpha"], ascending=[False, True]).index[0]
        result.loc[winner, "selected"] = True
    return result.reset_index(drop=True)


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _load_embedding(path: Path, expected_source: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        keys = set(saved.files)
        if not {"protein_ids", "embeddings"}.issubset(keys):
            raise ValueError(f"{expected_source} embedding file must contain protein_ids and embeddings")
        ids = saved["protein_ids"].astype(str)
        embeddings = saved["embeddings"]
    if embeddings.ndim != 2 or embeddings.shape[0] != len(ids):
        raise ValueError(f"{expected_source} embedding rows do not match IDs")
    if len(set(ids.tolist())) != len(ids):
        raise ValueError(f"{expected_source} embedding IDs must be unique")
    _as_matrix(embeddings, f"{expected_source} embeddings")
    return embeddings, ids


def run_labelwise(config_path: str | Path, *, project_root: str | Path | None = None) -> Path:
    """Run a configured validation/test label-wise score experiment."""
    root = Path(project_root) if project_root is not None else Path.cwd()
    config = load_config(config_path)
    run_dir = _resolve(root, config["output_dir"])
    metrics_prefix = _resolve(root, config["metrics_prefix"])
    summary_path = metrics_prefix.with_name(metrics_prefix.name + "-summary.json")
    screen_path = metrics_prefix.with_name(metrics_prefix.name + "-screen.csv")
    score_path = run_dir / "scores.npz"
    if run_dir.exists() or any(
        path.exists() for path in (summary_path, screen_path, score_path)
    ):
        raise FileExistsError("experiment outputs already exist; use a new experiment ID")
    run_dir.mkdir(parents=True)
    metrics_prefix.parent.mkdir(parents=True, exist_ok=True)

    data = config["data"]
    train_frame = pd.read_csv(_resolve(root, data["train_path"]))
    labels = label_columns(train_frame.columns)
    train_ids = pd.read_csv(_resolve(root, config["split"]["train_ids"]))["protein_id"].astype(str).tolist()
    validation_ids = pd.read_csv(_resolve(root, config["split"]["validation_ids"]))["protein_id"].astype(str).tolist()
    if len(set(train_ids)) != len(train_ids) or len(set(validation_ids)) != len(validation_ids):
        raise ValueError("fixed split IDs must be unique")
    train_embeddings, embedding_ids = _load_embedding(
        _resolve(root, config["embeddings"]["train"]), "train"
    )
    positions = {value: index for index, value in enumerate(embedding_ids.tolist())}
    if set(train_frame["protein_id"].astype(str)) != set(positions):
        raise ValueError("train embedding IDs do not match training data IDs")
    if set(train_ids) & set(validation_ids) or set(train_ids) | set(validation_ids) != set(positions):
        raise ValueError("fixed split IDs must partition the training embedding IDs")
    train_positions = [positions[value] for value in train_ids]
    validation_positions = [positions[value] for value in validation_ids]
    train_x = train_embeddings[train_positions]
    validation_x = train_embeddings[validation_positions]
    train_y = train_frame.set_index("protein_id").loc[train_ids, labels].to_numpy(dtype=np.uint8)

    screen = screen_regularization_by_support(train_x, train_y, config, label_names=labels)
    screen.to_csv(screen_path, index=False)
    selected = {
        str(row.stratum): {"alpha": float(row.alpha)}
        for row in screen.loc[screen["selected"]].itertuples()
    }
    fit_config = dict(config)
    fit_config["selected_regularization"] = selected
    started = time.perf_counter()
    test_scores = None
    test_ids = None
    eval_x = validation_x
    if config["embeddings"].get("test"):
        test_embeddings, test_ids = _load_embedding(
            _resolve(root, config["embeddings"]["test"]), "test"
        )
        eval_x = np.vstack([validation_x, test_embeddings])
    all_scores = fit_labelwise_scores(train_x, train_y, eval_x, fit_config)
    validation_scores = all_scores[: len(validation_x)]
    if test_ids is not None:
        test_scores = all_scores[len(validation_x) :]
    payload = {
        "validation_scores": validation_scores.astype(np.float32),
        "validation_ids": np.asarray(validation_ids, dtype=np.str_),
        "label_columns": np.asarray(labels, dtype=np.str_),
    }
    if test_scores is not None:
        payload["test_scores"] = test_scores.astype(np.float32)
        payload["test_ids"] = test_ids.astype(np.str_)
    np.savez_compressed(score_path, **payload)
    try:
        validation_auc = macro_roc_auc_skip_degenerate(
            train_frame.set_index("protein_id").loc[validation_ids, labels].to_numpy(dtype=np.uint8),
            validation_scores,
        )
    except ValueError:
        validation_auc = None
    summary = {
        "experiment_id": config["experiment_id"],
        "seed": int(config.get("seed", 42)),
        "train_rows": len(train_ids),
        "validation_rows": len(validation_ids),
        "test_rows": 0 if test_ids is None else len(test_ids),
        "label_count": len(labels),
        "continuous_validation_macro_auc": validation_auc,
        "screening_rows": len(screen),
        "selected_support_strata": sorted(selected),
        "elapsed_seconds": time.perf_counter() - started,
        "outputs": {"scores": str(score_path), "screen": str(screen_path)},
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(run_labelwise(args.config))


if __name__ == "__main__":
    main()
