"""Prediction and submission generation entry point."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import load_config
from .data import load_test_data
from .features import extract_composition_features


def _resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def run_prediction(
    config_path: str | Path,
    *,
    run_dir: str | Path,
    output_path: str | Path,
    project_root: str | Path | None = None,
) -> Path:
    """Load a trained run and create a submission-shaped CSV."""
    root = Path(project_root) if project_root is not None else Path.cwd()
    config = load_config(config_path)
    data_config = config["data"]
    test = load_test_data(
        _resolve(root, data_config["test_path"]),
        max_samples=data_config.get("max_test_samples"),
    )
    bundle = joblib.load(_resolve(root, run_dir) / "model.joblib")
    if bundle["feature_type"] != "composition":
        raise ValueError("prediction supports only composition features")
    features = extract_composition_features(test["sequence"])
    predictions = np.zeros(
        (len(test), len(bundle["all_label_columns"])), dtype=np.int8
    )
    positions = {
        label: index for index, label in enumerate(bundle["all_label_columns"])
    }
    for label in bundle["trained_label_columns"]:
        model = bundle["models"][label]
        if isinstance(model, dict):
            values = np.full(len(test), model["constant"], dtype=np.int8)
        else:
            values = model.predict(features).astype(np.int8)
        predictions[:, positions[label]] = values
    submission = pd.DataFrame(predictions, columns=bundle["all_label_columns"])
    submission.insert(0, "protein_id", test["protein_id"].to_numpy())
    destination = _resolve(root, output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(destination, index=False)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(run_prediction(args.config, run_dir=args.run_dir, output_path=args.output))


if __name__ == "__main__":
    main()
