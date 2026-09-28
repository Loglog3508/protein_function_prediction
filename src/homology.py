"""Label transfer from locally aligned protein neighbors."""

import numpy as np
from sklearn.neighbors import NearestNeighbors


_STANDARD = frozenset("ACDEFGHIKLMNPQRSTVWY")


def _alignable(sequence: str, max_length: int) -> str:
    return "".join(amino_acid if amino_acid in _STANDARD else "X" for amino_acid in sequence.upper()[:max_length])


def alignment_neighbor_scores(
    training_sequences,
    training_target: np.ndarray,
    training_features,
    evaluation_sequences,
    evaluation_features,
    *,
    n_neighbors: int = 20,
    query_batch_size: int = 32,
    max_length: int = 2048,
    alignment_power: float = 2.0,
) -> np.ndarray:
    """Retrieve cosine neighbors, then weight their labels by local alignment."""
    import parasail

    training_sequences = list(training_sequences)
    evaluation_sequences = list(evaluation_sequences)
    training_target = np.asarray(training_target)
    if (
        training_target.ndim != 2
        or len(training_sequences) != training_features.shape[0]
        or len(training_sequences) != training_target.shape[0]
    ):
        raise ValueError("training rows must match sequences, features, and targets")
    if len(evaluation_sequences) != evaluation_features.shape[0]:
        raise ValueError("evaluation rows must match sequences and features")
    if not 1 <= n_neighbors <= len(training_sequences):
        raise ValueError("n_neighbors must be between one and the training row count")
    if query_batch_size <= 0 or max_length <= 0 or alignment_power <= 0:
        raise ValueError("batch size, maximum length, and alignment power must be positive")

    neighbors = NearestNeighbors(
        n_neighbors=n_neighbors, metric="cosine", algorithm="brute", n_jobs=-1
    )
    neighbors.fit(training_features)
    scores = np.empty((len(evaluation_sequences), training_target.shape[1]), dtype=np.float32)
    cached_sequences = {}
    for start in range(0, len(evaluation_sequences), query_batch_size):
        stop = min(start + query_batch_size, len(evaluation_sequences))
        distances, indices = neighbors.kneighbors(evaluation_features[start:stop])
        for row, query_index in enumerate(range(start, stop)):
            query = _alignable(evaluation_sequences[query_index], max_length)
            weights = np.zeros(n_neighbors, dtype=np.float32)
            for rank, training_index in enumerate(indices[row]):
                training_index = int(training_index)
                reference = cached_sequences.get(training_index)
                if reference is None:
                    reference = _alignable(training_sequences[training_index], max_length)
                    cached_sequences[training_index] = reference
                if query and reference:
                    alignment = parasail.sw_striped_16(
                        query, reference, 11, 1, parasail.blosum62
                    ).score
                    similarity = min(1.0, alignment / (5.0 * min(len(query), len(reference))))
                    weights[rank] = similarity**alignment_power
            if weights.sum() == 0:
                weights = np.clip(1.0 - distances[row], 0.0, 1.0).astype(np.float32)
                if weights.sum() == 0:
                    weights.fill(1.0)
            scores[query_index] = np.average(
                training_target[indices[row]], axis=0, weights=weights
            )
    return np.clip(scores, 0.0, 1.0)
