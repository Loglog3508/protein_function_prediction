"""Safe feature-level fusion of sparse sequence features and dense embeddings."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy import sparse


def validate_embedding_ids(expected_ids: Iterable[str], actual_ids: Iterable[str]) -> None:
    expected = np.asarray(list(expected_ids), dtype=str)
    actual = np.asarray(list(actual_ids), dtype=str)
    if expected.ndim != 1 or actual.ndim != 1:
        raise ValueError("embedding IDs must be one-dimensional")
    if len(expected) != len(actual):
        raise ValueError("embedding IDs have different lengths")
    if len(set(expected.tolist())) != len(expected) or len(set(actual.tolist())) != len(actual):
        raise ValueError("embedding IDs must be unique")
    if not np.array_equal(expected, actual):
        raise ValueError("embedding IDs must match exactly in order")


def combine_sparse_dense_features(
    sparse_features,
    dense_features: np.ndarray,
    *,
    scale: float = 1.0,
):
    if not sparse.issparse(sparse_features) or sparse_features.ndim != 2:
        raise ValueError("sparse_features must be a two-dimensional sparse matrix")
    if not np.isfinite(sparse_features.data).all():
        raise ValueError("sparse_features must contain only finite values")
    dense_features = np.asarray(dense_features, dtype=np.float32)
    if dense_features.ndim != 2 or dense_features.shape[0] != sparse_features.shape[0]:
        raise ValueError("dense_features must have the same row count as sparse_features")
    if not np.isfinite(dense_features).all():
        raise ValueError("dense_features must contain only finite values")
    if not np.isfinite(scale) or scale < 0:
        raise ValueError("scale must be finite and non-negative")
    dense_sparse = sparse.csr_matrix(dense_features * np.float32(scale))
    return sparse.hstack(
        [sparse_features.astype(np.float32), dense_sparse],
        format="csr",
        dtype=np.float32,
    )
