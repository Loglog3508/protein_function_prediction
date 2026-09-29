import numpy as np
import pytest

from src.esm_embeddings import (
    pool_residue_embeddings,
    sequence_windows,
    validate_embedding_metadata,
)


def test_sequence_windows_cover_long_sequence():
    sequence = "A" * 2500

    windows = sequence_windows(sequence, window_size=1022, overlap=128)

    assert windows[0][0] == 0
    assert windows[-1][0] + len(windows[-1][1]) == len(sequence)
    assert all(len(value) <= 1022 for _, value in windows)
    covered = np.zeros(len(sequence), dtype=bool)
    for start, value in windows:
        covered[start : start + len(value)] = True
    assert covered.all()


@pytest.mark.parametrize(
    ("sequence", "window_size", "overlap", "message"),
    [
        ("", 10, 2, "empty"),
        ("AAAA", 0, 0, "window_size"),
        ("AAAA", 4, -1, "overlap"),
        ("AAAA", 4, 4, "overlap"),
    ],
)
def test_sequence_windows_reject_invalid_inputs(
    sequence, window_size, overlap, message
):
    with pytest.raises(ValueError, match=message):
        sequence_windows(sequence, window_size=window_size, overlap=overlap)


def test_pooling_concatenates_weighted_mean_and_max():
    pooled = pool_residue_embeddings(
        [np.array([1.0, 3.0]), np.array([5.0, 1.0])], [1, 3]
    )

    np.testing.assert_allclose(pooled, [4.0, 1.5, 5.0, 3.0])


def test_pooling_rejects_mismatched_dimensions():
    with pytest.raises(ValueError, match="dimension"):
        pool_residue_embeddings(
            [np.array([1.0, 2.0]), np.array([3.0])], [1, 1]
        )


def test_validate_embedding_metadata_names_mismatch():
    expected = {
        "model_name": "facebook/esm2_t12_35M_UR50D",
        "window_size": 1022,
        "dtype": "float16",
    }
    actual = {**expected, "model_name": "other/model"}

    with pytest.raises(ValueError, match="model_name"):
        validate_embedding_metadata(expected, actual)


def test_validate_embedding_metadata_accepts_expected_subset():
    expected = {"model_name": "esm", "window_size": 1022}
    actual = {**expected, "row_count": 10, "feature_width": 960}

    validate_embedding_metadata(expected, actual)

