"""Leakage-safe multilabel validation splitting."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from .data import label_columns


def make_group_multilabel_split(
    target: np.ndarray,
    groups: np.ndarray,
    *,
    validation_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Stratify unique groups by their union of labels, then expand to rows."""
    target = np.asarray(target)
    groups = np.asarray(groups)
    if target.ndim != 2 or len(groups) != len(target):
        raise ValueError("target must be 2D and groups must match its rows")
    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1")
    group_codes, unique_groups = pd.factorize(groups, sort=False)
    group_target = np.zeros((len(unique_groups), target.shape[1]), dtype=np.uint8)
    np.maximum.at(group_target, group_codes, target.astype(np.uint8, copy=False))
    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=validation_size, random_state=seed
    )
    group_indices = np.arange(len(unique_groups))
    train_groups, validation_groups = next(
        splitter.split(group_indices.reshape(-1, 1), group_target)
    )
    train_mask = np.isin(group_codes, train_groups)
    validation_mask = np.isin(group_codes, validation_groups)
    return np.flatnonzero(train_mask), np.flatnonzero(validation_mask)


def save_validation_split(
    train_path: str | Path,
    output_dir: str | Path,
    *,
    validation_size: float = 0.2,
    seed: int = 42,
) -> dict:
    """Create and persist train/validation IDs and label-balance diagnostics."""
    train_path = Path(train_path)
    header = pd.read_csv(train_path, nrows=0)
    labels = label_columns(header.columns)
    train = pd.read_csv(train_path, dtype={label: np.uint8 for label in labels})
    target = train[labels].to_numpy(dtype=np.uint8)
    train_indices, validation_indices = make_group_multilabel_split(
        target,
        train["sequence"].to_numpy(),
        validation_size=validation_size,
        seed=seed,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train.loc[train_indices, ["protein_id"]].to_csv(
        output_dir / "train_ids.csv", index=False
    )
    train.loc[validation_indices, ["protein_id"]].to_csv(
        output_dir / "validation_ids.csv", index=False
    )
    full_positive = target.sum(axis=0, dtype=np.int64)
    train_positive = target[train_indices].sum(axis=0, dtype=np.int64)
    validation_positive = target[validation_indices].sum(axis=0, dtype=np.int64)
    balance = pd.DataFrame(
        {
            "label": labels,
            "full_positive": full_positive,
            "train_positive": train_positive,
            "validation_positive": validation_positive,
            "full_rate": full_positive / len(target),
            "train_rate": train_positive / len(train_indices),
            "validation_rate": validation_positive / len(validation_indices),
        }
    )
    balance["absolute_rate_shift"] = (
        balance["validation_rate"] - balance["full_rate"]
    ).abs()
    balance.to_csv(output_dir / "label_balance.csv", index=False)
    train_sequences = set(train.iloc[train_indices]["sequence"])
    validation_sequences = set(train.iloc[validation_indices]["sequence"])
    summary = {
        "seed": seed,
        "validation_size_requested": validation_size,
        "train_rows": len(train_indices),
        "validation_rows": len(validation_indices),
        "validation_fraction_actual": len(validation_indices) / len(train),
        "labels_with_validation_positives": int((validation_positive > 0).sum()),
        "label_count": len(labels),
        "max_absolute_rate_shift": float(balance["absolute_rate_shift"].max()),
        "mean_absolute_rate_shift": float(balance["absolute_rate_shift"].mean()),
        "shared_sequences_between_splits": len(
            train_sequences & validation_sequences
        ),
    }
    (output_dir / "split_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="data/train.csv")
    parser.add_argument("--output-dir", default="artifacts/metrics/splits/seed42")
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    summary = save_validation_split(
        args.train,
        args.output_dir,
        validation_size=args.validation_size,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
