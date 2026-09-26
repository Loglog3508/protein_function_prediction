"""Sequence-neighbor label transfer using sparse cosine similarity."""

import numpy as np
from sklearn.neighbors import NearestNeighbors


def weighted_knn_label_scores(
    training_features,
    training_target: np.ndarray,
    evaluation_features,
    *,
    n_neighbors: int,
    similarity_power: float = 1.0,
    query_batch_size: int = 64,
) -> np.ndarray:
    """Average neighbor labels with powered cosine-similarity weights."""
    training_target = np.asarray(training_target)
    if training_target.ndim != 2 or training_target.shape[0] != training_features.shape[0]:
        raise ValueError("training targets must match training feature rows")
    if not 1 <= n_neighbors <= training_features.shape[0]:
        raise ValueError("neighbors must be between one and the training row count")
    if similarity_power <= 0:
        raise ValueError("similarity_power must be positive")
    if query_batch_size <= 0:
        raise ValueError("query_batch_size must be positive")

    model = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )
    model.fit(training_features)
    scores = np.zeros(
        (evaluation_features.shape[0], training_target.shape[1]), dtype=np.float32
    )
    for start in range(0, evaluation_features.shape[0], query_batch_size):
        stop = min(start + query_batch_size, evaluation_features.shape[0])
        distances, indices = model.kneighbors(evaluation_features[start:stop])
        similarities = np.clip(1.0 - distances, 0.0, 1.0) ** similarity_power
        denominator = similarities.sum(axis=1, keepdims=True)
        zero_weight = denominator[:, 0] == 0
        if zero_weight.any():
            similarities[zero_weight] = 1.0
            denominator[zero_weight] = n_neighbors
        neighbor_targets = training_target[indices]
        scores[start:stop] = np.einsum(
            "bk,bkl->bl", similarities, neighbor_targets, optimize=True
        ) / denominator
        np.clip(scores[start:stop], 0.0, 1.0, out=scores[start:stop])
    return scores
