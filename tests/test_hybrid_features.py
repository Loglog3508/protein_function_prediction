import numpy as np
import pytest
from scipy import sparse

from src.hybrid_features import combine_sparse_dense_features, validate_embedding_ids


def test_combine_sparse_dense_features_preserves_rows_and_scale():
    sparse_features = sparse.csr_matrix([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    dense_features = np.array([[3.0, 4.0], [5.0, 6.0]], dtype=np.float32)

    combined = combine_sparse_dense_features(
        sparse_features, dense_features, scale=0.5
    )

    assert sparse.issparse(combined)
    assert combined.shape == (2, 4)
    np.testing.assert_allclose(
        combined.toarray(),
        [[1.0, 0.0, 1.5, 2.0], [0.0, 2.0, 2.5, 3.0]],
    )


def test_combine_sparse_dense_features_rejects_invalid_dense_values():
    with pytest.raises(ValueError, match="finite"):
        combine_sparse_dense_features(
            sparse.csr_matrix([[1.0]]), np.array([[np.inf]], dtype=np.float32)
        )


def test_validate_embedding_ids_requires_exact_order():
    validate_embedding_ids(["P1", "P2"], ["P1", "P2"])
    with pytest.raises(ValueError, match="order"):
        validate_embedding_ids(["P1", "P2"], ["P2", "P1"])


def test_combine_rejects_nonfinite_sparse_values():
    with pytest.raises(ValueError, match="finite"):
        combine_sparse_dense_features(
            sparse.csr_matrix([[np.nan]]), np.array([[1.0]], dtype=np.float32)
        )
