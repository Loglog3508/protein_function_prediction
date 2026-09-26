"""Sequence feature extraction."""

from collections.abc import Iterable

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
DIPEPTIDES = tuple(first + second for first in AMINO_ACIDS for second in AMINO_ACIDS)
PROPERTY_GROUPS = {
    "hydrophobic": frozenset("AVILMFWY"),
    "polar": frozenset("STNQCY"),
    "positive": frozenset("KRH"),
    "negative": frozenset("DE"),
    "aromatic": frozenset("FWYH"),
    "small": frozenset("AGSTCVP"),
}
LENGTH_BINS = (100, 250, 500, 1000, 2000)


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


class KmerBlockVectorizer:
    """Fit and concatenate independently capped character k-mer blocks."""

    def __init__(self, blocks: list[dict]):
        if not blocks:
            raise ValueError("at least one k-mer block is required")
        self.names = []
        self.weights = []
        self.vectorizers = []
        for block in blocks:
            name = str(block["name"])
            if name in self.names:
                raise ValueError("k-mer block names must be unique")
            self.names.append(name)
            self.weights.append(float(block.get("weight", 1.0)))
            self.vectorizers.append(
                build_kmer_vectorizer(
                    k_min=int(block["k_min"]),
                    k_max=int(block["k_max"]),
                    min_df=block["min_df"],
                    max_features=block.get("max_features"),
                    sublinear_tf=bool(block.get("sublinear_tf", True)),
                )
            )

    def _combine(self, matrices):
        weighted = [
            matrix if weight == 1.0 else matrix.multiply(weight)
            for matrix, weight in zip(matrices, self.weights, strict=True)
        ]
        return sparse.hstack(weighted, format="csr", dtype=np.float32)

    def fit_transform(self, sequences: Iterable[str]):
        values = list(sequences)
        return self._combine(
            [vectorizer.fit_transform(values) for vectorizer in self.vectorizers]
        )

    def transform(self, sequences: Iterable[str]):
        values = list(sequences)
        return self._combine(
            [vectorizer.transform(values) for vectorizer in self.vectorizers]
        )

    @property
    def block_vocabulary_sizes(self) -> dict[str, int]:
        return {
            name: len(vectorizer.vocabulary_)
            for name, vectorizer in zip(
                self.names, self.vectorizers, strict=True
            )
        }

    @property
    def vocabulary_size(self) -> int:
        return sum(self.block_vocabulary_sizes.values())


def build_kmer_block_vectorizer(blocks: list[dict]) -> KmerBlockVectorizer:
    """Create a transformer that preserves separate k-mer vocabularies."""
    return KmerBlockVectorizer(blocks)


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


def extract_sequence_statistics(sequences: Iterable[str]) -> np.ndarray:
    """Build composition, dipeptide, property, and length-bin features."""
    sequences = list(sequences)
    composition = extract_composition_features(sequences)
    length_dimensions = len(LENGTH_BINS) + 1
    extra_dimensions = len(DIPEPTIDES) + len(PROPERTY_GROUPS) + length_dimensions
    features = np.zeros(
        (len(sequences), composition.shape[1] + extra_dimensions), dtype=np.float32
    )
    features[:, : composition.shape[1]] = composition
    dipeptide_offset = composition.shape[1]
    dipeptide_indices = {value: index for index, value in enumerate(DIPEPTIDES)}
    property_offset = dipeptide_offset + len(DIPEPTIDES)
    length_offset = property_offset + len(PROPERTY_GROUPS)
    for row, sequence in enumerate(sequences):
        length = len(sequence)
        pair_count = max(length - 1, 0)
        if pair_count:
            for position in range(pair_count):
                index = dipeptide_indices.get(sequence[position : position + 2])
                if index is not None:
                    features[row, dipeptide_offset + index] += 1
            features[row, dipeptide_offset:property_offset] /= pair_count
        if length:
            for group_index, amino_acids in enumerate(PROPERTY_GROUPS.values()):
                features[row, property_offset + group_index] = (
                    sum(amino_acid in amino_acids for amino_acid in sequence) / length
                )
        length_bin = int(np.searchsorted(LENGTH_BINS, length, side="right"))
        features[row, length_offset + length_bin] = 1.0
    return features
