"""Sequence feature extraction."""

from collections.abc import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def build_kmer_vectorizer(
    *,
    k_min: int,
    k_max: int,
    min_df: int | float,
    max_features: int | None,
    sublinear_tf: bool,
) -> TfidfVectorizer:
    """Create a memory-bounded character k-mer TF-IDF vectorizer."""
    if not 1 <= k_min <= k_max:
        raise ValueError("k-mer range must satisfy 1 <= k_min <= k_max")
    return TfidfVectorizer(
        analyzer="char",
        ngram_range=(k_min, k_max),
        min_df=min_df,
        max_features=max_features,
        sublinear_tf=sublinear_tf,
        lowercase=False,
        dtype=np.float32,
    )


def extract_composition_features(sequences: Iterable[str]) -> np.ndarray:
    """Build 20 amino-acid frequencies plus log-transformed length."""
    sequences = list(sequences)
    features = np.zeros((len(sequences), len(AMINO_ACIDS) + 1), dtype=np.float32)
    indices = {amino_acid: index for index, amino_acid in enumerate(AMINO_ACIDS)}
    for row, sequence in enumerate(sequences):
        length = len(sequence)
        features[row, -1] = np.log1p(length)
        if not length:
            continue
        for amino_acid in sequence:
            index = indices.get(amino_acid)
            if index is not None:
                features[row, index] += 1
        features[row, :-1] /= length
    return features
