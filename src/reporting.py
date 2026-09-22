"""Build the experiment leaderboard and final report figures."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def build_final_artifacts(project_root: str | Path = ".") -> dict:
    """Create a concise leaderboard and report-ready figures."""
    root = Path(project_root)
    metrics = root / "artifacts/metrics"
    figures = root / "reports/figures"
    stage5 = _read_json(metrics / "EXP-20260922-010-stage5-thresholds-summary.json")
    low_alpha = _read_json(
        metrics / "EXP-20260922-017-kmer35-sgd-low-alpha-thresholds-summary.json"
    )
    cnn = _read_json(metrics / "EXP-20260922-016-cnn-gpu-thresholds-summary.json")
    ensemble = _read_json(metrics / "EXP-20260922-018-stage6-ensemble-summary.json")
    rf = _read_json(metrics / "EXP-20260921-002-rf-baseline-summary.json")
    sequence_statistics = _read_json(
        metrics / "EXP-20260922-013-sequence-statistics-thresholds-summary.json"
    )
    fixed_low_alpha = _read_json(
        metrics / "EXP-20260922-017-kmer35-sgd-low-alpha-full-summary.json"
    )

    leaderboard = pd.DataFrame(
        [
            {
                "rank_basis": "crossfit_or_fixed_validation",
                "experiment": "Low-alpha 3-5-mer SGD + shrunk thresholds",
                "experiment_id": "EXP-20260922-017",
                "macro_f1": low_alpha["crossfit_per_label_shrunk"]["macro_f1"],
                "label_count": 500,
                "prediction_positive_rate": low_alpha["crossfit_per_label_shrunk"][
                    "predicted_positive_rate"
                ],
                "role": "primary",
            },
            {
                "rank_basis": "crossfit_or_fixed_validation",
                "experiment": "3-5-mer SGD + shrunk thresholds",
                "experiment_id": "EXP-20260922-010",
                "macro_f1": stage5["crossfit_per_label_shrunk"]["macro_f1"],
                "label_count": 500,
                "prediction_positive_rate": stage5["crossfit_per_label_shrunk"][
                    "predicted_positive_rate"
                ],
                "role": "backup",
            },
            {
                "rank_basis": "crossfit_or_fixed_validation",
                "experiment": "Sequence statistics SGD + shrunk thresholds (50 labels)",
                "experiment_id": "EXP-20260922-013",
                "macro_f1": sequence_statistics["crossfit_per_label_shrunk"][
                    "macro_f1"
                ],
                "label_count": 50,
                "prediction_positive_rate": sequence_statistics[
                    "crossfit_per_label_shrunk"
                ]["predicted_positive_rate"],
                "role": "screen_only",
            },
            {
                "rank_basis": "crossfit_or_fixed_validation",
                "experiment": "CUDA CNN + shrunk thresholds",
                "experiment_id": "EXP-20260922-016",
                "macro_f1": cnn["crossfit_per_label_shrunk"]["macro_f1"],
                "label_count": 500,
                "prediction_positive_rate": cnn["crossfit_per_label_shrunk"][
                    "predicted_positive_rate"
                ],
                "role": "alternate",
            },
            {
                "rank_basis": "crossfit_or_fixed_validation",
                "experiment": "Random forest baseline at 0.5",
                "experiment_id": "EXP-20260921-002",
                "macro_f1": rf["macro_f1"],
                "label_count": 500,
                "prediction_positive_rate": rf["predicted_positive_rate"],
                "role": "baseline",
            },
        ]
    ).sort_values("macro_f1", ascending=False, ignore_index=True)
    leaderboard.insert(0, "rank", "screen")
    full_mask = leaderboard["label_count"] == 500
    leaderboard.loc[full_mask, "rank"] = np.arange(1, int(full_mask.sum()) + 1).astype(str)
    leaderboard_path = metrics / "leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)

    comparable = leaderboard[leaderboard["label_count"] == 500]
    plt.figure(figsize=(8.2, 4.5))
    bars = plt.barh(
        comparable["experiment"],
        comparable["macro_f1"],
        color=["#187A5B", "#3D78A5", "#B85C38", "#747474"],
    )
    plt.gca().invert_yaxis()
    plt.xlabel("Macro F1")
    plt.title("Comparable 500-label validation results")
    plt.xlim(0, 0.35)
    for bar, value in zip(bars, comparable["macro_f1"], strict=True):
        plt.text(value + 0.004, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center")
    _save_figure(figures / "model_comparison.png")

    threshold_rows = pd.DataFrame(low_alpha["strategies"])
    plt.figure(figsize=(7.4, 4.3))
    colors = ["#747474", "#3D78A5", "#B85C38", "#187A5B"]
    bars = plt.bar(
        ["Fixed 0.5", "Global", "Per-label", "Shrunk"],
        threshold_rows["holdout_macro_f1"],
        color=colors,
    )
    plt.ylabel("Holdout Macro F1")
    plt.title("Threshold strategy comparison")
    plt.ylim(0.2, 0.34)
    for bar, value in zip(bars, threshold_rows["holdout_macro_f1"], strict=True):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.003, f"{value:.3f}", ha="center")
    _save_figure(figures / "threshold_comparison.png")

    counts = low_alpha["samplewise_label_counts"]
    names = ["True", "Fixed 0.5", "Global", "Shrunk"]
    means = [
        counts["fixed_0.5"]["true"]["mean"],
        counts["fixed_0.5"]["predicted"]["mean"],
        counts["global_selected"]["predicted"]["mean"],
        counts["per_label_shrunk"]["predicted"]["mean"],
    ]
    plt.figure(figsize=(7.2, 4.3))
    bars = plt.bar(names, means, color=["#222222", "#B85C38", "#3D78A5", "#187A5B"])
    plt.ylabel("Mean labels per sequence")
    plt.title("Prediction cardinality diagnostic")
    for bar, value in zip(bars, means, strict=True):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.1f}", ha="center")
    _save_figure(figures / "label_count_diagnostic.png")

    per_label = pd.read_csv(
        metrics / "EXP-20260922-017-kmer35-sgd-low-alpha-thresholds-per-label.csv"
    )
    plt.figure(figsize=(7.2, 4.5))
    plt.scatter(
        per_label["support"],
        per_label["f1"],
        s=15,
        alpha=0.55,
        color="#3D78A5",
        edgecolors="none",
    )
    plt.xscale("log")
    plt.xlabel("Validation support (log scale)")
    plt.ylabel("Per-label F1")
    plt.title("Long-tail error analysis")
    _save_figure(figures / "support_vs_f1.png")

    summary = {
        "best_model": leaderboard[leaderboard["role"] == "primary"].iloc[0].to_dict(),
        "ensemble_best_candidate": ensemble["best_candidate"],
        "ensemble_best_crossfit_macro_f1": ensemble["best_crossfit_macro_f1"],
        "fixed_low_alpha_macro_f1": fixed_low_alpha["macro_f1"],
        "outputs": {
            "leaderboard": str(leaderboard_path),
            "figures": [
                str(figures / "model_comparison.png"),
                str(figures / "threshold_comparison.png"),
                str(figures / "label_count_diagnostic.png"),
                str(figures / "support_vs_f1.png"),
            ],
        },
    }
    summary_path = metrics / "final_report_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    print(json.dumps(build_final_artifacts(args.project_root), indent=2))


if __name__ == "__main__":
    main()
