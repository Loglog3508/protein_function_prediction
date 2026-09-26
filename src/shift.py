"""Helpers for the competition's ordered train/test distribution shift."""

import numpy as np


def tail_distribution_mask(
    protein_ids: list[str] | np.ndarray, *, cutoff: int
) -> np.ndarray:
    """Select protein IDs whose numeric suffix is at or beyond the cutoff."""
    mask = []
    for protein_id in protein_ids:
        value = str(protein_id)
        if len(value) < 2 or value[0] != "P" or not value[1:].isdigit():
            raise ValueError(f"invalid protein ID: {value}")
        mask.append(int(value[1:]) >= cutoff)
    return np.asarray(mask, dtype=bool)
