"""Train an alignment-neighbor model on all rows and create a submission."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data import label_columns
from .features import build_kmer_vectorizer
from .finalize import _sha256
from .homology import alignment_neighbor_scores
from .shift import tail_distribution_mask
from .thresholds import threshold_predictions
from .train import _resolve, fit_label_models
from .validate_submission import validate_submission_file


def run_final_homology(config_path: str | Path, *, project_root: str | Path | None = None) -> Path:
    root = Path(project_root) if project_root is not None else Path.cwd()
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    train_path = _resolve(root, config["data"]["train_path"])
    test_path = _resolve(root, config["data"]["test_path"])
    thresholds_path = _resolve(root, config["thresholds_path"])
    submission_path = _resolve(root, config["submission_path"])
    metadata_path = _resolve(root, config["metadata_path"])
    scores_path = _resolve(root, config.get("scores_path", str(metadata_path.with_name("test_scores.npz"))))
    if any(path.exists() for path in (submission_path, metadata_path, scores_path)):
        raise FileExistsError("final outputs already exist; choose new paths")

    labels = label_columns(pd.read_csv(train_path, nrows=0).columns)
    threshold_data = json.loads(thresholds_path.read_text(encoding="utf-8"))
    if threshold_data["labels"] != labels:
        raise ValueError("threshold labels do not match training labels")
    thresholds = np.asarray(threshold_data["thresholds"], dtype=np.float32)
    if thresholds.shape != (len(labels),):
        raise ValueError("threshold count does not match training labels")
    sgd_weight = float(threshold_data.get("sgd_weight", 0.0))
    if not 0 <= sgd_weight <= 1:
        raise ValueError("SGD weight must be between zero and one")
    train = pd.read_csv(train_path, usecols=["protein_id", "sequence", *labels], dtype={label: np.uint8 for label in labels})
    test = pd.read_csv(test_path)

    started = time.perf_counter()
    features = config["features"]
    vectorizer = build_kmer_vectorizer(
        k_min=int(features["k_min"]), k_max=int(features["k_max"]),
        min_df=features["min_df"], max_features=features.get("max_features"),
        sublinear_tf=bool(features.get("sublinear_tf", True)),
    )
    training_features = vectorizer.fit_transform(train["sequence"])
    test_features = vectorizer.transform(test["sequence"])
    feature_seconds = time.perf_counter() - started
    target = train[labels].to_numpy(dtype=np.uint8)
    neighbors = config["neighbors"]
    started = time.perf_counter()
    alignment_scores = alignment_neighbor_scores(
        train["sequence"], target, training_features, test["sequence"], test_features,
        n_neighbors=int(neighbors["n_neighbors"]),
        query_batch_size=int(neighbors["query_batch_size"]),
        max_length=int(neighbors["max_length"]),
        alignment_power=float(neighbors["alignment_power"]),
    )
    alignment_seconds = time.perf_counter() - started
    sgd_seconds = 0.0
    if sgd_weight:
        distribution = config["distribution"]
        tail_mask = tail_distribution_mask(train["protein_id"].tolist(), cutoff=int(distribution["cutoff"]))
        row_weight = np.where(tail_mask, float(distribution["tail_weight"]), 1.0)
        started = time.perf_counter()
        _, sgd_scores = fit_label_models(
            training_features, target, test_features,
            model_config=config["model"], seed=int(config["seed"]),
            progress_every=25, retain_models=False,
            training_sample_weight=row_weight,
        )
        sgd_seconds = time.perf_counter() - started
        scores = ((1 - sgd_weight) * alignment_scores + sgd_weight * sgd_scores).astype(np.float32)
    else:
        scores = alignment_scores
    predictions = threshold_predictions(scores, thresholds)
    submission = pd.DataFrame(predictions, columns=labels)
    submission.insert(0, "protein_id", test["protein_id"].to_numpy())
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(submission_path, index=False)
    validation = validate_submission_file(submission_path, test_path=test_path, expected_label_columns=labels)
    np.savez_compressed(scores_path, test_scores=scores, test_ids=test["protein_id"].to_numpy(dtype=np.str_), label_columns=np.asarray(labels, dtype=np.str_))
    metadata_path.write_text(json.dumps({
        "experiment_id": config["experiment_id"],
        "train_rows": len(train),
        "test_rows": len(test),
        "label_count": len(labels),
        "sgd_weight": sgd_weight,
        "predicted_positive_rate": float(predictions.mean()),
        "average_labels_per_protein": float(predictions.sum(axis=1).mean()),
        "vocabulary_size": len(vectorizer.vocabulary_),
        "timing_seconds": {"features": feature_seconds, "alignment": alignment_seconds, "sgd": sgd_seconds},
        "submission_validation": validation,
        "submission_sha256": _sha256(submission_path),
        "test_scores_sha256": _sha256(scores_path),
    }, indent=2), encoding="utf-8")
    return metadata_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(run_final_homology(args.config))


if __name__ == "__main__":
    main()
