"""Experiment configuration loading."""

import json
from pathlib import Path


def load_config(path: str | Path) -> dict:
    """Load a JSON experiment config and validate shared settings."""
    with Path(path).open(encoding="utf-8") as handle:
        config = json.load(handle)
    seed = config.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("config seed must be a non-negative integer")
    return config
