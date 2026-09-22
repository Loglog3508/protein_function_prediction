"""Train the final model on all training rows and create a validated submission."""

import argparse
import hashlib
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import load_config
from .data import label_columns
from .features import build_kmer_vectorizer
from .train import fit_label_models
from .validate_submission import validate_submission_file


def _resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


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


def _load_thresholds(path: Path, labels: list[str]) -> tuple[np.ndarray, dict]:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata["labels"] != labels:
        raise ValueError("threshold labels do not match training labels")
    thresholds = np.asarray(metadata["thresholds"], dtype=np.float32)
    if thresholds.shape != (len(labels),):
        raise ValueError("threshold count does not match training labels")
    return thresholds, metadata


def run_final_training(
    config_path: str | Path,
    *,
    thresholds_path: str | Path,
    submission_path: str | Path,
    metadata_path: str | Path,
    project_root: str | Path | None = None,
) -> Path:
    """Fit the selected k-mer model on all rows and write a submission."""
    root = Path(project_root) if project_root is not None else Path.cwd()
    config = load_config(config_path)
    if config["features"]["type"] != "kmer_tfidf":
        raise ValueError("final training currently supports kmer_tfidf features")
    train_path = _resolve(root, config["data"]["train_path"])
    test_path = _resolve(root, config["data"]["test_path"])
    columns = pd.read_csv(train_path, nrows=0).columns
    labels = label_columns(columns)
    train = pd.read_csv(
        train_path,
        dtype={label: np.uint8 for label in labels},
    )
    test = pd.read_csv(test_path)
    thresholds, threshold_metadata = _load_thresholds(
        _resolve(root, thresholds_path), labels
    )

    destination = _resolve(root, submission_path)
    metadata_destination = _resolve(root, metadata_path)
    if destination.exists() or metadata_destination.exists():
        raise FileExistsError("final outputs already exist; choose new paths")
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata_destination.parent.mkdir(parents=True, exist_ok=True)

    feature_started = time.perf_counter()
    feature_config = config["features"]
    vectorizer = build_kmer_vectorizer(
        k_min=feature_config["k_min"],
        k_max=feature_config["k_max"],
        min_df=feature_config["min_df"],
        max_features=feature_config.get("max_features"),
        sublinear_tf=feature_config.get("sublinear_tf", True),
    )
    training_features = vectorizer.fit_transform(train["sequence"])
    test_features = vectorizer.transform(test["sequence"])
    feature_seconds = time.perf_counter() - feature_started

    model_started = time.perf_counter()
    timing_stats = {}
    _, test_scores = fit_label_models(
        training_features,
        train[labels].to_numpy(dtype=np.uint8),
        test_features,
        model_config=config["model"],
        seed=config["seed"],
        progress_every=25,
        retain_models=False,
        timing_stats=timing_stats,
    )
    model_seconds = time.perf_counter() - model_started
    predictions = (test_scores >= thresholds[np.newaxis, :]).astype(np.uint8)
    submission = pd.DataFrame(predictions, columns=labels)
    submission.insert(0, "protein_id", test["protein_id"].to_numpy())
    submission.to_csv(destination, index=False)
    validation = validate_submission_file(
        destination,
        test_path=test_path,
        expected_label_columns=labels,
    )
    scores_path = metadata_destination.with_name(
        metadata_destination.stem.replace("-metadata", "-test-scores") + ".npz"
    )
    np.savez_compressed(
        scores_path,
        test_scores=test_scores.astype(np.float32),
        test_ids=test["protein_id"].to_numpy(dtype=np.str_),
        label_columns=np.asarray(labels, dtype=np.str_),
    )
    vectorizer_path = metadata_destination.with_name(
        metadata_destination.stem.replace("-metadata", "-vectorizer") + ".joblib"
    )
    joblib.dump(vectorizer, vectorizer_path)
    metadata = {
        "experiment_id": config["experiment_id"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_scope": "all_training_rows",
        "train_rows": len(train),
        "test_rows": len(test),
        "label_count": len(labels),
        "data_sha256": _sha256(train_path),
        "test_sha256": _sha256(test_path),
        "config": config,
        "thresholds": threshold_metadata,
        "feature_resources": {
            "vocabulary_size": len(vectorizer.vocabulary_),
            "train_shape": list(training_features.shape),
            "train_nnz": int(training_features.nnz),
            "test_shape": list(test_features.shape),
            "test_nnz": int(test_features.nnz),
        },
        "timing": {
            "feature_seconds": feature_seconds,
            "fit_seconds": timing_stats["fit_seconds"],
            "inference_seconds": timing_stats["inference_seconds"],
            "model_seconds": model_seconds,
        },
        "submission_validation": validation,
        "submission_sha256": _sha256(destination),
        "test_scores_sha256": _sha256(scores_path),
        "python": platform.python_version(),
        "git": _git_state(root),
        "outputs": {
            "submission": str(destination),
            "test_scores": str(scores_path),
            "vectorizer": str(vectorizer_path),
        },
    }
    metadata_destination.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metadata_destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--submission", required=True)
    parser.add_argument("--metadata", required=True)
    args = parser.parse_args()
    print(
        run_final_training(
            args.config,
            thresholds_path=args.thresholds,
            submission_path=args.submission,
            metadata_path=args.metadata,
        )
    )


if __name__ == "__main__":
    main()
