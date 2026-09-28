"""Merge disjoint GPU validation folds into a reference score order."""

import argparse
import json
from pathlib import Path

import numpy as np


def _load_scores(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    with np.load(path, allow_pickle=False) as saved:
        scores = saved["validation_scores"].astype(np.float32)
        ids = saved["validation_ids"].astype(str)
        labels = saved["label_columns"].astype(str).tolist()
    if scores.shape != (len(ids), len(labels)):
        raise ValueError("score shape must match validation IDs and labels")
    if len(set(ids)) != len(ids):
        raise ValueError("validation IDs must be unique within each score file")
    if len(set(labels)) != len(labels):
        raise ValueError("labels must be unique within each score file")
    return scores, ids, labels


def merge_gpu_oof(
    reference_path: str | Path,
    fold_paths: list[str | Path],
    output_path: str | Path,
) -> dict:
    """Merge folds, reordering rows and labels to match a reference NPZ."""
    _, reference_ids, reference_labels = _load_scores(Path(reference_path))
    if len(set(reference_ids)) != len(reference_ids):
        raise ValueError("reference validation IDs are not unique")

    rows_by_id: dict[str, np.ndarray] = {}
    fold_rows: list[int] = []
    for fold_path in fold_paths:
        fold_scores, fold_ids, fold_labels = _load_scores(Path(fold_path))
        if set(fold_labels) != set(reference_labels):
            raise ValueError(f"fold label set does not match: {fold_path}")
        positions = {label: index for index, label in enumerate(fold_labels)}
        reordered = fold_scores[:, [positions[label] for label in reference_labels]]
        fold_rows.append(len(fold_ids))
        for protein_id, row in zip(fold_ids, reordered, strict=True):
            if protein_id in rows_by_id:
                raise ValueError(f"duplicate validation ID across folds: {protein_id}")
            rows_by_id[protein_id] = row

    missing = [protein_id for protein_id in reference_ids if protein_id not in rows_by_id]
    extras = sorted(set(rows_by_id) - set(reference_ids))
    if missing or extras:
        raise ValueError(
            f"fold coverage mismatch: {len(missing)} missing, {len(extras)} extras"
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        output,
        validation_scores=np.stack([rows_by_id[value] for value in reference_ids]),
        validation_ids=reference_ids.astype(np.str_),
        label_columns=np.asarray(reference_labels, dtype=np.str_),
    )
    return {
        "reference_scores": str(reference_path),
        "fold_scores": [str(path) for path in fold_paths],
        "fold_rows": fold_rows,
        "validation_rows": len(reference_ids),
        "label_count": len(reference_labels),
        "unique_validation_ids": len(rows_by_id),
        "output": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--fold", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()
    summary_path = Path(args.summary)
    if summary_path.exists():
        raise FileExistsError("summary already exists; use a new experiment ID")
    summary = merge_gpu_oof(args.reference, args.fold, args.output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary_path)


if __name__ == "__main__":
    main()
