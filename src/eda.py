"""Exploratory data analysis outputs."""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import sparse

from .data import label_columns

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def _describe(values: np.ndarray) -> dict:
    values = np.asarray(values)
    return {
        "min": float(values.min()),
        "q1": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "q3": float(np.quantile(values, 0.75)),
        "max": float(values.max()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
    }


def summarize_frames(
    train: pd.DataFrame, test: pd.DataFrame, labels: list[str]
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calculate label, sequence, amino-acid, and leakage statistics."""
    target = train[labels].to_numpy(dtype=np.uint8)
    positive_counts = target.sum(axis=0, dtype=np.int64)
    label_distribution = pd.DataFrame(
        {
            "label": labels,
            "positive_count": positive_counts,
            "positive_rate": positive_counts / len(train),
        }
    )
    target_sparse = sparse.csr_matrix(target, dtype=np.int32)
    matrix = (target_sparse.T @ target_sparse).toarray()
    upper_row, upper_column = np.triu_indices(len(labels), k=1)
    cooccurrence = pd.DataFrame(
        {
            "label_a": np.asarray(labels)[upper_row],
            "label_b": np.asarray(labels)[upper_column],
            "count": matrix[upper_row, upper_column].astype(np.int64),
        }
    ).sort_values(["count", "label_a", "label_b"], ascending=[False, True, True])
    cooccurrence = cooccurrence.head(100).reset_index(drop=True)

    train_sequences = train["sequence"].astype(str)
    test_sequences = test["sequence"].astype(str)
    train_counter = Counter("".join(train_sequences))
    test_counter = Counter("".join(test_sequences))
    characters = sorted(set(train_counter) | set(test_counter))
    amino_acids = pd.DataFrame(
        {
            "amino_acid": characters,
            "train_count": [train_counter[character] for character in characters],
            "test_count": [test_counter[character] for character in characters],
        }
    )
    train_length = train_sequences.str.len().to_numpy()
    test_length = test_sequences.str.len().to_numpy()
    labels_per_sequence = target.sum(axis=1, dtype=np.int64)
    train_sequence_set = set(train_sequences)
    test_sequence_set = set(test_sequences)
    summary = {
        "rows": {"train": len(train), "test": len(test)},
        "label_count": len(labels),
        "label_density": float(target.mean()),
        "labels_per_sequence": _describe(labels_per_sequence),
        "sequence_length": {
            "train": _describe(train_length),
            "test": _describe(test_length),
        },
        "leakage": {
            "train_duplicate_id_rows": int(
                train["protein_id"].duplicated(keep=False).sum()
            ),
            "test_duplicate_id_rows": int(
                test["protein_id"].duplicated(keep=False).sum()
            ),
            "train_duplicate_sequence_rows": int(
                train_sequences.duplicated(keep=False).sum()
            ),
            "test_duplicate_sequence_rows": int(
                test_sequences.duplicated(keep=False).sum()
            ),
            "train_test_shared_ids": len(
                set(train["protein_id"]) & set(test["protein_id"])
            ),
            "train_test_shared_unique_sequences": len(
                train_sequence_set & test_sequence_set
            ),
        },
    }
    return summary, label_distribution, cooccurrence, amino_acids


def _save_figures(
    train: pd.DataFrame,
    test: pd.DataFrame,
    label_distribution: pd.DataFrame,
    figure_dir: Path,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 4.8))
    ordered = label_distribution.sort_values("positive_count", ascending=False)
    plt.plot(np.arange(len(ordered)), ordered["positive_count"], linewidth=1.5)
    plt.yscale("log")
    plt.xlabel("Labels sorted by frequency")
    plt.ylabel("Positive samples (log scale)")
    plt.tight_layout()
    plt.savefig(figure_dir / "label_long_tail.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4.8))
    train[label_columns(train.columns)].sum(axis=1).clip(upper=150).hist(
        bins=50, color="#2563eb"
    )
    plt.xlabel("Positive labels per sequence (values above 150 clipped)")
    plt.ylabel("Sequences")
    plt.tight_layout()
    plt.savefig(figure_dir / "labels_per_sequence.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4.8))
    plt.hist(
        train["sequence"].str.len().clip(upper=3000),
        bins=60,
        alpha=0.65,
        label="train",
    )
    plt.hist(
        test["sequence"].str.len().clip(upper=3000),
        bins=60,
        alpha=0.55,
        label="test",
    )
    plt.xlabel("Sequence length (values above 3000 clipped)")
    plt.ylabel("Sequences")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "sequence_length_train_test.png", dpi=160)
    plt.close()


def run_eda(project_root: str | Path = ".") -> dict:
    """Run full-data EDA and save reproducible tables, JSON, and figures."""
    root = Path(project_root)
    train_path = root / "data" / "train.csv"
    test_path = root / "data" / "test.csv"
    header = pd.read_csv(train_path, nrows=0)
    labels = label_columns(header.columns)
    train = pd.read_csv(train_path, dtype={label: np.uint8 for label in labels})
    test = pd.read_csv(test_path)
    summary, distribution, cooccurrence, amino_acids = summarize_frames(
        train, test, labels
    )
    metric_dir = root / "artifacts" / "metrics"
    metric_dir.mkdir(parents=True, exist_ok=True)
    (metric_dir / "eda_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    distribution.to_csv(metric_dir / "label_distribution.csv", index=False)
    cooccurrence.to_csv(metric_dir / "label_cooccurrence_top100.csv", index=False)
    amino_acids.to_csv(metric_dir / "amino_acid_frequency.csv", index=False)
    _save_figures(train, test, distribution, root / "reports" / "figures")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    print(json.dumps(run_eda(args.project_root), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
