"""ESM-2 embedding extraction and deterministic feature pooling."""

import hashlib
from typing import Any, Iterable

import numpy as np


DEFAULT_MODEL_NAME = "facebook/esm2_t12_35M_UR50D"
DEFAULT_WINDOW_SIZE = 1022
DEFAULT_OVERLAP = 128


def sequence_windows(
    sequence: str, *, window_size: int = DEFAULT_WINDOW_SIZE, overlap: int = DEFAULT_OVERLAP
) -> list[tuple[int, str]]:
    """Return overlapping residue windows that cover a sequence exactly."""
    sequence = str(sequence)
    if not sequence:
        raise ValueError("sequence cannot be empty")
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if overlap < 0 or overlap >= window_size:
        raise ValueError("overlap must be non-negative and smaller than window_size")
    step = window_size - overlap
    windows: list[tuple[int, str]] = []
    start = 0
    while start < len(sequence):
        end = min(start + window_size, len(sequence))
        windows.append((start, sequence[start:end]))
        if end == len(sequence):
            break
        start += step
        if start + window_size >= len(sequence):
            start = len(sequence) - window_size
    return windows


def pool_residue_embeddings(
    window_embeddings: Iterable[np.ndarray], window_lengths: Iterable[int]
) -> np.ndarray:
    """Concatenate length-weighted mean and element-wise maximum window pools."""
    embeddings = [np.asarray(value, dtype=np.float32) for value in window_embeddings]
    lengths = [int(value) for value in window_lengths]
    if not embeddings or len(embeddings) != len(lengths):
        raise ValueError("window embeddings and lengths must be non-empty and aligned")
    if any(value <= 0 for value in lengths):
        raise ValueError("window lengths must be positive")
    if any(value.ndim != 1 for value in embeddings):
        raise ValueError("window embeddings must be one-dimensional")
    dimensions = {value.shape[0] for value in embeddings}
    if len(dimensions) != 1:
        raise ValueError("window embeddings must have one consistent dimension")
    matrix = np.stack(embeddings, axis=0)
    weights = np.asarray(lengths, dtype=np.float32)
    mean = np.average(matrix, axis=0, weights=weights)
    maximum = np.max(matrix, axis=0)
    pooled = np.concatenate([mean, maximum]).astype(np.float32, copy=False)
    if not np.isfinite(pooled).all():
        raise ValueError("pooled embedding contains non-finite values")
    return pooled


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(str(sequence).encode("utf-8")).hexdigest()


def validate_embedding_metadata(expected: dict[str, Any], actual: dict[str, Any]) -> None:
    """Require every requested metadata field to match the stored shard."""
    for key, expected_value in expected.items():
        if key not in actual:
            raise ValueError(f"embedding metadata is missing {key}")
        if actual[key] != expected_value:
            raise ValueError(
                f"embedding metadata {key} mismatch: "
                f"expected {expected_value!r}, got {actual[key]!r}"
            )
