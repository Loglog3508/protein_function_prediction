"""Screen feature-level fusion of k-mer TF-IDF and ESM embeddings."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data import label_columns
from .hybrid_features import combine_sparse_dense_features, validate_embedding_ids
from .metrics import macro_roc_auc_skip_degenerate
from .sweep_sgd import select_support_stratified_labels
from .thresholds import crossfit_shrunk_threshold_score
from .train import _build_feature_matrices, _resolve, fit_label_models


def _load_embeddings(path: Path, expected_ids: list[str]) -> np.ndarray:
    with np.load(path, allow_pickle=False) as stored:
        if set(stored.files) != {"protein_ids", "embeddings"}:
            raise ValueError("embedding file must contain protein_ids and embeddings")
        ids = stored["protein_ids"].astype(str).tolist()
        validate_embedding_ids(expected_ids, ids)
        embeddings = stored["embeddings"].astype(np.float32)
    if embeddings.ndim != 2 or not np.isfinite(embeddings).all():
        raise ValueError("embeddings must be finite and two-dimensional")
    return embeddings


def run_hybrid_sweep(
    config_path: str | Path, *, project_root: str | Path | None = None
) -> Path:
    root = Path(project_root) if project_root is not None else Path.cwd()
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    run_dir = _resolve(root, config["output_dir"])
    metrics_prefix = _resolve(root, config["metrics_prefix"])
    summary_path = metrics_prefix.with_name(metrics_prefix.name + "-summary.json")
    leaderboard_path = metrics_prefix.with_name(metrics_prefix.name + "-leaderboard.csv")
    if run_dir.exists() or summary_path.exists() or leaderboard_path.exists():
        raise FileExistsError("experiment outputs already exist; use a new experiment ID")

    train_path = _resolve(root, config["data"]["train_path"])
    train = pd.read_csv(train_path)
    labels = label_columns(train.columns)
    train["protein_id"] = train["protein_id"].astype(str)
    indexed = train.set_index("protein_id", drop=False)
    split = config["split"]
    training_ids = pd.read_csv(_resolve(root, split["train_ids"]))["protein_id"].astype(str).tolist()
    validation_ids = pd.read_csv(_resolve(root, split["validation_ids"]))["protein_id"].astype(str).tolist()
    if set(training_ids) & set(validation_ids):
        raise ValueError("training and validation IDs overlap")
    if set(training_ids) | set(validation_ids) != set(indexed.index):
        raise ValueError("training and validation IDs do not cover the training data")
    training = indexed.loc[training_ids]
    validation = indexed.loc[validation_ids]

    selection = select_support_stratified_labels(
        training[labels], int(config["selection"]["label_count"])
    )
    selected_labels = selection["label"].tolist()
    target = validation[selected_labels].to_numpy(dtype=np.uint8)
    train_target = training[selected_labels].to_numpy(dtype=np.uint8)

    feature_started = time.perf_counter()
    train_kmer, validation_kmer, resources, _ = _build_feature_matrices(
        training, validation, config["features"]
    )
    feature_seconds = time.perf_counter() - feature_started
    embedding_path = _resolve(root, config["embeddings"]["train_path"])
    full_embeddings = _load_embeddings(embedding_path, train["protein_id"].tolist())
    position = {protein_id: index for index, protein_id in enumerate(train["protein_id"])}
    train_embeddings = full_embeddings[[position[protein_id] for protein_id in training_ids]]
    validation_embeddings = full_embeddings[[position[protein_id] for protein_id in validation_ids]]

    run_dir.mkdir(parents=True)
    metrics_prefix.parent.mkdir(parents=True, exist_ok=True)
    selection.to_csv(metrics_prefix.with_name(metrics_prefix.name + "-labels.csv"), index=False)
    candidates = []
    model_config = dict(config["model"])
    seeds = tuple(int(seed) for seed in config["seeds"])
    threshold_candidates = np.asarray(config["thresholds"]["candidates"], dtype=np.float32)
    shrinkage = float(config["thresholds"]["shrinkage"])
    for scale in config["embeddings"]["scales"]:
        scale = float(scale)
        if scale == 0:
            train_features = train_kmer
            validation_features = validation_kmer
        else:
            train_features = combine_sparse_dense_features(
                train_kmer, train_embeddings, scale=scale
            )
            validation_features = combine_sparse_dense_features(
                validation_kmer, validation_embeddings, scale=scale
            )
        started = time.perf_counter()
        _, scores = fit_label_models(
            train_features,
            train_target,
            validation_features,
            model_config=model_config,
            seed=int(config["seed"]),
            progress_every=int(config.get("progress_every", 25)),
            retain_models=False,
        )
        seed_results = [
            crossfit_shrunk_threshold_score(
                target,
                scores,
                seed=seed,
                shrinkage=shrinkage,
                candidates=threshold_candidates,
            )
            for seed in seeds
        ]
        f1 = np.asarray([result["macro_f1"] for result in seed_results], dtype=float)
        row = {
            "embedding_scale": scale,
            "mean_crossfit_macro_f1": float(f1.mean()),
            "std_crossfit_macro_f1": float(f1.std()),
            "min_crossfit_macro_f1": float(f1.min()),
            "continuous_macro_auc": float(macro_roc_auc_skip_degenerate(target, scores)),
            "mean_predicted_positive_rate": float(
                np.mean([result["predicted_positive_rate"] for result in seed_results])
            ),
            "fit_seconds": float(time.perf_counter() - started),
        }
        candidates.append(row)
        np.savez_compressed(
            run_dir / f"scale-{scale:g}-scores.npz",
            validation_scores=scores.astype(np.float32),
            validation_ids=np.asarray(validation_ids, dtype=np.str_),
            label_columns=np.asarray(selected_labels, dtype=np.str_),
        )
        print(json.dumps(row, ensure_ascii=False), flush=True)
    leaderboard = pd.DataFrame(candidates).sort_values(
        ["mean_crossfit_macro_f1", "continuous_macro_auc", "std_crossfit_macro_f1"],
        ascending=[False, False, True],
        ignore_index=True,
    )
    leaderboard.to_csv(leaderboard_path, index=False)
    summary_path.write_text(
        json.dumps(
            {
                "experiment_id": config["experiment_id"],
                "validation_rows": len(validation),
                "label_count": len(selected_labels),
                "selected_labels": selected_labels,
                "feature_seconds": feature_seconds,
                "feature_resources": resources,
                "leaderboard": leaderboard.to_dict(orient="records"),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(run_hybrid_sweep(args.config))


if __name__ == "__main__":
    main()
