"""Model training entry point."""

import argparse
import hashlib
import json
import platform
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split

from .config import load_config
from .data import label_columns, load_training_data
from .features import (
    build_kmer_block_vectorizer,
    build_kmer_vectorizer,
    extract_composition_features,
    extract_sequence_statistics,
)
from .metrics import macro_f1_skip_empty, per_label_classification_metrics


def _resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _class_weight(model_config: dict):
    if "positive_class_weight" in model_config:
        positive_weight = float(model_config["positive_class_weight"])
        if positive_weight <= 0:
            raise ValueError("positive_class_weight must be positive")
        return {0: 1.0, 1: positive_weight}
    return model_config.get("class_weight")


def _scaled_label_class_weight(
    model_config: dict,
    target: np.ndarray,
    multiplier: float | None,
):
    """Return an optional per-label class-weight override."""
    if multiplier is None:
        return _class_weight(model_config)
    if not np.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("label positive weight multiplier must be positive")
    base_weight = _class_weight(model_config)
    if base_weight == "balanced":
        positive = int(target.sum())
        negative = int(target.size - positive)
        if positive == 0 or negative == 0:
            return base_weight
        total = float(target.size)
        return {
            0: total / (2.0 * negative),
            1: total / (2.0 * positive) * multiplier,
        }
    if base_weight is None:
        return {0: 1.0, 1: multiplier}
    if isinstance(base_weight, dict):
        scaled = dict(base_weight)
        scaled[1] = float(scaled.get(1, 1.0)) * multiplier
        return scaled
    raise ValueError("label weight multipliers require dict, balanced, or no class weight")


def fit_label_models(
    training_features,
    training_target: np.ndarray,
    evaluation_features,
    *,
    model_config: dict,
    seed: int,
    progress_every: int | None = None,
    retain_models: bool = True,
    timing_stats: dict[str, float] | None = None,
    training_sample_weight: np.ndarray | None = None,
    label_positive_weight_multipliers: np.ndarray | None = None,
) -> tuple[list[object], np.ndarray]:
    """Fit one binary model per label and return continuous positive scores."""
    training_target = np.asarray(training_target)
    if training_target.ndim != 2:
        raise ValueError("training_target must be two-dimensional")
    if training_sample_weight is not None:
        training_sample_weight = np.asarray(training_sample_weight, dtype=np.float64)
        if training_sample_weight.shape != (training_target.shape[0],):
            raise ValueError("training sample weights must match training rows")
        if not np.isfinite(training_sample_weight).all() or (
            training_sample_weight < 0
        ).any():
            raise ValueError("training sample weights must be finite and non-negative")
    if label_positive_weight_multipliers is not None:
        label_positive_weight_multipliers = np.asarray(
            label_positive_weight_multipliers, dtype=np.float64
        )
        if label_positive_weight_multipliers.shape != (training_target.shape[1],):
            raise ValueError(
                "label positive weight multipliers must match label count"
            )
        if not np.isfinite(label_positive_weight_multipliers).all() or (
            label_positive_weight_multipliers <= 0
        ).any():
            raise ValueError(
                "label positive weight multipliers must be finite and positive"
            )
    model_type = model_config["type"]
    if model_type not in {"random_forest", "sgd", "logistic_regression"}:
        raise ValueError("unsupported model type")
    label_n_jobs = model_config.get("label_n_jobs", 1)
    if isinstance(label_n_jobs, bool) or not isinstance(label_n_jobs, int) or label_n_jobs < 1:
        raise ValueError("label_n_jobs must be a positive integer")
    model_n_jobs = 1 if label_n_jobs > 1 else model_config.get("n_jobs", 1)
    models: list[object] = []
    fit_seconds = 0.0
    inference_seconds = 0.0
    scores = np.zeros(
        (evaluation_features.shape[0], training_target.shape[1]), dtype=np.float32
    )
    def fit_one_label(label_index: int):
        target = training_target[:, label_index]
        label_multiplier = (
            None
            if label_positive_weight_multipliers is None
            else float(label_positive_weight_multipliers[label_index])
        )
        class_weight = _scaled_label_class_weight(
            model_config, target, label_multiplier
        )
        classes = np.unique(target)
        if len(classes) == 1:
            value = int(classes[0])
            inference_started = time.perf_counter()
            score = np.float32(value)
            return (
                {"constant": value} if retain_models else None,
                score,
                0.0,
                time.perf_counter() - inference_started,
                False,
            )
        if model_type == "random_forest":
            model = RandomForestClassifier(
                n_estimators=model_config["n_estimators"],
                max_depth=model_config.get("max_depth"),
                class_weight=class_weight,
                n_jobs=model_n_jobs,
                random_state=seed,
            )
        elif model_type == "sgd":
            model = SGDClassifier(
                loss="log_loss",
                alpha=model_config.get("alpha", 0.0001),
                average=model_config.get("average", False),
                class_weight=class_weight,
                max_iter=model_config.get("max_iter", 1000),
                tol=model_config.get("tol", 1e-3),
                random_state=seed,
                n_jobs=model_n_jobs,
            )
        else:
            model = LogisticRegression(
                C=model_config.get("C", 1.0),
                class_weight=class_weight,
                max_iter=model_config.get("max_iter", 1000),
                tol=model_config.get("tol", 1e-4),
                random_state=seed,
                n_jobs=model_n_jobs,
                solver="liblinear",
            )
        fit_started = time.perf_counter()
        model.fit(training_features, target, sample_weight=training_sample_weight)
        label_fit_seconds = time.perf_counter() - fit_started
        positive_index = int(np.flatnonzero(model.classes_ == 1)[0])
        inference_started = time.perf_counter()
        label_scores = model.predict_proba(evaluation_features)[:, positive_index]
        label_inference_seconds = time.perf_counter() - inference_started
        return (
            model if retain_models else None,
            label_scores,
            label_fit_seconds,
            label_inference_seconds,
            True,
        )

    def collect(results):
        nonlocal fit_seconds, inference_seconds
        for label_index, (model, label_scores, label_fit, label_inference, trained) in enumerate(results):
            assignment_started = time.perf_counter()
            scores[:, label_index] = label_scores
            fit_seconds += label_fit
            inference_seconds += label_inference + time.perf_counter() - assignment_started
            if retain_models:
                models.append(model)
            if trained and progress_every and (label_index + 1) % progress_every == 0:
                print(
                    f"trained {label_index + 1}/{training_target.shape[1]} labels",
                    flush=True,
                )

    indices = range(training_target.shape[1])
    if label_n_jobs == 1:
        collect(map(fit_one_label, indices))
    else:
        with ThreadPoolExecutor(max_workers=label_n_jobs) as executor:
            collect(executor.map(fit_one_label, indices))
    if timing_stats is not None:
        timing_stats.update(
            fit_seconds=fit_seconds,
            inference_seconds=inference_seconds,
        )
    return models, scores


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state(root: Path) -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return {"commit": commit, "dirty": bool(status)}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _sparse_bytes(matrix) -> int:
    return int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)


def _matrix_resources(matrix, prefix: str) -> dict:
    is_sparse = sparse.issparse(matrix)
    return {
        f"{prefix}_shape": list(matrix.shape),
        f"{prefix}_nnz": int(matrix.nnz if is_sparse else np.count_nonzero(matrix)),
        f"{prefix}_bytes": (
            _sparse_bytes(matrix) if is_sparse else int(matrix.nbytes)
        ),
    }


def _build_feature_matrices(training, validation, feature_config: dict):
    feature_type = feature_config["type"]
    if feature_type in {"composition", "sequence_statistics"}:
        extractor = (
            extract_composition_features
            if feature_type == "composition"
            else extract_sequence_statistics
        )
        training_features = extractor(training["sequence"])
        validation_features = extractor(validation["sequence"])
        resources = {
            "sparse": False,
            "vocabulary_size": None,
            **_matrix_resources(training_features, "train"),
            **_matrix_resources(validation_features, "validation"),
        }
        return training_features, validation_features, resources, None
    if feature_type not in {
        "kmer_tfidf",
        "kmer_tfidf_statistics",
        "kmer_tfidf_blocks",
    }:
        raise ValueError(f"unsupported feature type: {feature_type}")
    if feature_type == "kmer_tfidf_blocks":
        vectorizer = build_kmer_block_vectorizer(feature_config["blocks"])
    else:
        vectorizer = build_kmer_vectorizer(
            k_min=feature_config["k_min"],
            k_max=feature_config["k_max"],
            min_df=feature_config["min_df"],
            max_features=feature_config.get("max_features"),
            sublinear_tf=feature_config.get("sublinear_tf", True),
        )
    training_features = vectorizer.fit_transform(training["sequence"])
    validation_features = vectorizer.transform(validation["sequence"])
    if feature_type == "kmer_tfidf_statistics":
        training_features = sparse.hstack(
            [
                training_features,
                sparse.csr_matrix(extract_sequence_statistics(training["sequence"])),
            ],
            format="csr",
            dtype=np.float32,
        )
        validation_features = sparse.hstack(
            [
                validation_features,
                sparse.csr_matrix(extract_sequence_statistics(validation["sequence"])),
            ],
            format="csr",
            dtype=np.float32,
        )
    if not sparse.issparse(training_features) or not sparse.issparse(
        validation_features
    ):
        raise RuntimeError("k-mer feature matrices must remain sparse")
    vocabulary_size = (
        vectorizer.vocabulary_size
        if feature_type == "kmer_tfidf_blocks"
        else len(vectorizer.vocabulary_)
    )
    resources = {
        "sparse": True,
        "vocabulary_size": vocabulary_size,
        "block_vocabulary_sizes": (
            vectorizer.block_vocabulary_sizes
            if feature_type == "kmer_tfidf_blocks"
            else None
        ),
        "statistics_dimensions": (
            training_features.shape[1] - vocabulary_size
            if feature_type == "kmer_tfidf_statistics"
            else 0
        ),
        **_matrix_resources(training_features, "train"),
        **_matrix_resources(validation_features, "validation"),
    }
    return training_features, validation_features, resources, vectorizer


def run_evaluation(
    config_path: str | Path, *, project_root: str | Path | None = None
) -> Path:
    """Run a fixed-split experiment and persist scores and diagnostics."""
    root = Path(project_root) if project_root is not None else Path.cwd()
    config = load_config(config_path)
    train_path = _resolve(root, config["data"]["train_path"])
    columns = pd.read_csv(train_path, nrows=0).columns
    all_labels = label_columns(columns)
    train = pd.read_csv(
        train_path, dtype={label: np.uint8 for label in all_labels}
    )
    max_labels = config["data"].get("max_labels")
    labels = all_labels[:max_labels] if max_labels else all_labels
    split_config = config["split"]
    train_ids = pd.read_csv(_resolve(root, split_config["train_ids"]))[
        "protein_id"
    ].tolist()
    validation_ids = pd.read_csv(
        _resolve(root, split_config["validation_ids"])
    )["protein_id"].tolist()
    indexed = train.set_index("protein_id", drop=False)
    missing = (set(train_ids) | set(validation_ids)) - set(indexed.index)
    if missing:
        raise ValueError("split contains protein IDs absent from training data")
    training = indexed.loc[train_ids]
    validation = indexed.loc[validation_ids]

    run_dir = _resolve(root, config["output_dir"])
    metrics_prefix = _resolve(root, config["metrics_prefix"])
    summary_path = metrics_prefix.with_name(metrics_prefix.name + "-summary.json")
    diagnostics_path = metrics_prefix.with_name(
        metrics_prefix.name + "-per-label.csv"
    )
    if run_dir.exists() or summary_path.exists() or diagnostics_path.exists():
        raise FileExistsError("experiment outputs already exist; use a new experiment ID")
    run_dir.mkdir(parents=True)
    metrics_prefix.parent.mkdir(parents=True, exist_ok=True)

    feature_started = time.perf_counter()
    (
        training_features,
        validation_features,
        feature_resources,
        vectorizer,
    ) = _build_feature_matrices(training, validation, config["features"])
    feature_seconds = time.perf_counter() - feature_started
    if vectorizer is not None:
        joblib.dump(vectorizer, run_dir / "vectorizer.joblib")
    test = None
    test_features = None
    if config["data"].get("test_path"):
        test = pd.read_csv(_resolve(root, config["data"]["test_path"]))
        if vectorizer is None:
            extractor = (
                extract_sequence_statistics
                if config["features"]["type"] == "sequence_statistics"
                else extract_composition_features
            )
            test_features = extractor(test["sequence"])
        else:
            test_features = vectorizer.transform(test["sequence"])
            if config["features"]["type"] == "kmer_tfidf_statistics":
                test_features = sparse.hstack(
                    [
                        test_features,
                        sparse.csr_matrix(extract_sequence_statistics(test["sequence"])),
                    ],
                    format="csr",
                    dtype=np.float32,
                )
            if not sparse.issparse(test_features):
                raise RuntimeError("k-mer test features must remain sparse")
        feature_resources.update(_matrix_resources(test_features, "test"))
        if sparse.issparse(validation_features):
            evaluation_features = sparse.vstack(
                [validation_features, test_features], format="csr"
            )
        else:
            evaluation_features = np.vstack([validation_features, test_features])
    else:
        evaluation_features = validation_features
    model_started = time.perf_counter()
    timing_stats: dict[str, float] = {}
    _, evaluation_scores = fit_label_models(
        training_features,
        training[labels].to_numpy(dtype=np.uint8),
        evaluation_features,
        model_config=config["model"],
        seed=config["seed"],
        progress_every=25,
        retain_models=False,
        timing_stats=timing_stats,
    )
    validation_scores = evaluation_scores[: len(validation)]
    test_scores = evaluation_scores[len(validation) :] if test is not None else None
    model_seconds = time.perf_counter() - model_started
    predictions = (validation_scores >= config["threshold"]).astype(np.uint8)
    target = validation[labels].to_numpy(dtype=np.uint8)
    diagnostics = per_label_classification_metrics(target, predictions, labels)
    diagnostics.to_csv(diagnostics_path, index=False)
    scores_path = run_dir / "scores.npz"
    score_payload = {
        "validation_scores": validation_scores.astype(np.float32),
        "validation_predictions": predictions,
        "validation_ids": np.asarray(validation_ids, dtype=np.str_),
        "label_columns": np.asarray(labels, dtype=np.str_),
    }
    if test is not None:
        score_payload["test_scores"] = test_scores.astype(np.float32)
        score_payload["test_ids"] = test["protein_id"].to_numpy(dtype=np.str_)
    np.savez_compressed(scores_path, **score_payload)
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    summary = {
        "experiment_id": config["experiment_id"],
        "seed": config["seed"],
        "data_sha256": _sha256(train_path),
        "validation_split": split_config,
        "features": config["features"],
        "feature_resources": feature_resources,
        "model": config["model"],
        "threshold": config["threshold"],
        "train_rows": len(training),
        "validation_rows": len(validation),
        "label_count": len(labels),
        "macro_f1": macro_f1_skip_empty(target, predictions),
        "true_positive_rate": float(target.mean()),
        "predicted_positive_rate": float(predictions.mean()),
        "zero_f1_labels": int((diagnostics["f1"] == 0).sum()),
        "feature_seconds": feature_seconds,
        "fit_seconds": timing_stats["fit_seconds"],
        "inference_seconds": timing_stats["inference_seconds"],
        "train_and_validation_seconds": model_seconds,
        "rss_megabytes": psutil.Process().memory_info().rss / (1024**2),
        "scores_sha256": _sha256(scores_path),
        "python": platform.python_version(),
        "git": _git_state(root),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary_path


def run_training(
    config_path: str | Path, *, project_root: str | Path | None = None
) -> Path:
    """Train a configured composition/random-forest experiment."""
    root = Path(project_root) if project_root is not None else Path.cwd()
    config = load_config(config_path)
    data_config = config["data"]
    train, all_labels = load_training_data(
        _resolve(root, data_config["train_path"]),
        max_samples=data_config.get("max_train_samples"),
        seed=config["seed"],
    )
    max_labels = data_config.get("max_labels")
    trained_labels = all_labels[:max_labels] if max_labels else all_labels
    indices = np.arange(len(train))
    train_indices, validation_indices = train_test_split(
        indices,
        test_size=config["split"]["validation_size"],
        random_state=config["seed"],
    )
    features = extract_composition_features(train["sequence"])
    model_config = config["model"]
    if config["features"]["type"] != "composition":
        raise ValueError("smoke training supports only composition features")
    if model_config["type"] != "random_forest":
        raise ValueError("smoke training supports only random_forest models")

    started = time.perf_counter()
    validation_target = train.iloc[validation_indices][trained_labels].to_numpy(dtype=np.int8)
    fitted_models, validation_scores = fit_label_models(
        features[train_indices],
        train.iloc[train_indices][trained_labels].to_numpy(dtype=np.int8),
        features[validation_indices],
        model_config=model_config,
        seed=config["seed"],
    )
    validation_predictions = (validation_scores >= config["threshold"]).astype(np.int8)
    models = dict(zip(trained_labels, fitted_models, strict=True))
    metrics = {
        "experiment_id": config["experiment_id"],
        "macro_f1": macro_f1_skip_empty(validation_target, validation_predictions),
        "train_rows": int(len(train_indices)),
        "validation_rows": int(len(validation_indices)),
        "labels_trained": len(trained_labels),
        "elapsed_seconds": time.perf_counter() - started,
        "rss_megabytes": psutil.Process().memory_info().rss / (1024**2),
    }
    run_dir = _resolve(root, config["output_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "all_label_columns": all_labels,
        "trained_label_columns": trained_labels,
        "models": models,
        "feature_type": config["features"]["type"],
        "seed": config["seed"],
    }
    joblib.dump(bundle, run_dir / "model.joblib")
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    if args.evaluate:
        print(run_evaluation(args.config))
    else:
        print(run_training(args.config))


if __name__ == "__main__":
    main()
