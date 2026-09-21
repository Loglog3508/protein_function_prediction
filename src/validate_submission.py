"""Strict submission validation entry point."""

import argparse
import json
from pathlib import Path

import pandas as pd


def validate_submission_file(
    submission_path: str | Path,
    *,
    test_path: str | Path,
    expected_label_columns: list[str],
    max_test_samples: int | None = None,
) -> dict:
    """Validate schema, IDs, missing values, and binary label values."""
    submission = pd.read_csv(submission_path)
    test = pd.read_csv(test_path, usecols=["protein_id"], nrows=max_test_samples)
    expected_columns = ["protein_id", *expected_label_columns]
    if submission.columns.tolist() != expected_columns:
        raise ValueError("submission columns or column order are invalid")
    if len(submission) != len(test):
        raise ValueError("submission row count does not match test data")
    if submission["protein_id"].duplicated().any():
        raise ValueError("submission contains duplicate protein IDs")
    if submission["protein_id"].tolist() != test["protein_id"].tolist():
        raise ValueError("submission protein IDs or order do not match test data")
    if submission.isna().any().any():
        raise ValueError("submission contains missing values")
    labels = submission[expected_label_columns]
    if not labels.isin([0, 1]).all().all():
        raise ValueError("submission labels must contain only 0 or 1")
    if not all(pd.api.types.is_integer_dtype(labels[column]) for column in labels):
        raise ValueError("submission labels must use integer values")
    return {
        "rows": len(submission),
        "columns": len(submission.columns),
        "labels": len(expected_label_columns),
        "positive_rate": float(labels.to_numpy().mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", required=True)
    parser.add_argument("--test", default="data/test.csv")
    parser.add_argument("--label-count", type=int, default=500)
    parser.add_argument("--max-test-samples", type=int)
    args = parser.parse_args()
    summary = validate_submission_file(
        args.submission,
        test_path=args.test,
        expected_label_columns=[f"label_{index}" for index in range(args.label_count)],
        max_test_samples=args.max_test_samples,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
