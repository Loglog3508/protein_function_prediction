"""Build figures for the third-iteration competition report."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "artifacts" / "metrics"
FIGURES = ROOT / "reports" / "figures"


def save_blend_tradeoff() -> None:
    rows = pd.read_csv(
        METRICS / "EXP-20260926-044-iteration3-f1-stability-leaderboard.csv"
    )
    rows = rows.loc[rows.groupby("primary_weight")["mean_crossfit_macro_f1"].idxmax()]
    rows = rows.sort_values("primary_weight")
    figure, left = plt.subplots(figsize=(8.2, 4.8))
    right = left.twinx()
    left.plot(
        rows["primary_weight"],
        rows["mean_crossfit_macro_f1"],
        marker="o",
        linewidth=2,
        color="#1f4e79",
        label="Macro F1",
    )
    right.plot(
        rows["primary_weight"],
        rows["continuous_macro_auc"],
        marker="s",
        linewidth=2,
        color="#c55a11",
        label="Macro AUC",
    )
    left.axvline(0.1, color="#666666", linestyle="--", linewidth=1)
    left.set_xlabel("SGD weight in SGD/KNN blend")
    left.set_ylabel("Five-seed mean Macro F1", color="#1f4e79")
    right.set_ylabel("Continuous Macro AUC", color="#c55a11")
    left.grid(axis="y", alpha=0.25)
    handles = left.get_lines()[:1] + right.get_lines()[:1]
    left.legend(handles, [line.get_label() for line in handles], loc="lower center")
    figure.tight_layout()
    figure.savefig(FIGURES / "iteration3_f1_auc_tradeoff.png", dpi=180)
    plt.close(figure)


def save_seed_stability() -> None:
    rows = pd.read_csv(
        METRICS / "EXP-20260926-044-iteration3-f1-stability-seed-results.csv"
    )
    rows = rows[(rows["primary_weight"] == 0.1) & (rows["shrinkage"] == 10.0)]
    rows = rows.sort_values("seed")
    figure, axis = plt.subplots(figsize=(8.2, 4.5))
    bars = axis.bar(
        rows["seed"].astype(str),
        rows["crossfit_macro_f1"],
        color="#4472c4",
        width=0.62,
    )
    axis.axhline(
        rows["crossfit_macro_f1"].mean(),
        color="#c55a11",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean {rows['crossfit_macro_f1'].mean():.6f}",
    )
    axis.set_xlabel("Threshold split seed")
    axis.set_ylabel("Cross-fit Macro F1")
    axis.set_ylim(0.315, 0.327)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="lower right")
    for bar, value in zip(bars, rows["crossfit_macro_f1"], strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.00015,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    figure.tight_layout()
    figure.savefig(FIGURES / "iteration3_seed_stability.png", dpi=180)
    plt.close(figure)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    save_blend_tradeoff()
    save_seed_stability()


if __name__ == "__main__":
    main()
