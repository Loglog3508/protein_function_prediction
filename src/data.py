"""Dataset loading and validation."""

from pathlib import Path
from typing import Iterable

import pandas as pd


def label_columns(columns: Iterable[str]) -> list[str]:
    """Return numeric label columns and reject gaps or malformed names."""
    labels = [column for column in columns if column.startswith("label_")]
    try:
        labels.sort(key=lambda column: int(column.removeprefix("label_")))
    except ValueError as exc:
        raise ValueError("label columns must end with integers") from exc
    expected = [f"label_{index}" for index in range(len(labels))]
    if labels != expected:
        raise ValueError("label columns must be contiguous from label_0")
    return labels


def load_training_data(
    path: str | Path,
    *,
    max_samples: int | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, list[str]]:
    """Load training data, optionally taking a reproducible sample."""
    frame = pd.read_csv(path)
    labels = label_columns(frame.columns)
    required = {"protein_id", "sequence"}
    if not required.issubset(frame.columns) or not labels:
        raise ValueError("training data must contain IDs, sequences, and labels")
    if frame[["protein_id", "sequence", *labels]].isna().any().any():
        raise ValueError("training data contains missing values")
    if max_samples is not None and max_samples < len(frame):
        frame = frame.sample(n=max_samples, random_state=seed).reset_index(drop=True)
    return frame, labels


def load_test_data(
    path: str | Path, *, max_samples: int | None = None
) -> pd.DataFrame:
    """Load test data in submission order, optionally truncating for smoke runs."""
    frame = pd.read_csv(path, nrows=max_samples)
    if frame.columns.tolist() != ["protein_id", "sequence"]:
        raise ValueError("test data columns must be protein_id and sequence")
    if frame.isna().any().any() or frame["protein_id"].duplicated().any():
        raise ValueError("test data contains missing values or duplicate IDs")
    return frame
