"""Tests for alignment-weighted sequence-neighbor transfer."""

import numpy as np
import pytest
from scipy import sparse

from src.homology import alignment_neighbor_scores


def test_alignment_weights_identical_sequence_above_unrelated_neighbor():
    features = sparse.csr_matrix(np.ones((2, 1), dtype=np.float32))
    query = sparse.csr_matrix(np.ones((1, 1), dtype=np.float32))
    scores = alignment_neighbor_scores(
        ["AAAA", "RRRR"],
        np.array([[1, 0], [0, 1]], dtype=np.uint8),
        features,
        ["AAAA"],
        query,
        n_neighbors=2,
        query_batch_size=1,
    )
    assert scores.shape == (1, 2)
    assert scores[0, 0] > 0.9
    assert scores[0, 1] < 0.1


def test_alignment_accepts_nonstandard_amino_acids():
    features = sparse.csr_matrix(np.ones((1, 1), dtype=np.float32))
    scores = alignment_neighbor_scores(
        ["ACUOZ"],
        np.array([[1]], dtype=np.uint8),
        features,
        ["ACUOZ"],
        features,
        n_neighbors=1,
    )
    assert scores[0, 0] == pytest.approx(1.0)


def test_alignment_rejects_mismatched_training_rows():
    features = sparse.csr_matrix(np.ones((2, 1), dtype=np.float32))
    with pytest.raises(ValueError, match="training rows"):
        alignment_neighbor_scores(
            ["AAAA"],
            np.array([[1]], dtype=np.uint8),
            features,
            ["AAAA"],
            features[:1],
            n_neighbors=1,
        )
