"""CPU-only, paired label diagnostics for already-saved ESM validation scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .data import label_columns
from .metrics import macro_f1_skip_empty, per_label_score_diagnostics
from .rank_fusion import fit_rank_fusion_thresholds
from .thresholds import threshold_predictions


SEEDS = (17, 31, 42, 73, 101)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def support_bands(values: np.ndarray, edges: list[int]) -> np.ndarray:
    values = np.asarray(values)
    if not edges or edges != sorted(set(edges)) or edges[0] <= 0:
        raise ValueError("support edges must be increasing positive integers")
    if not np.isfinite(values).all() or (values < 0).any() or (values != np.floor(values)).any():
        raise ValueError("support must be finite non-negative integers")
    starts = [0, *edges]
    names = [str(start) if end - start == 1 else f"{start}-{end - 1}" for start, end in zip(starts, edges)]
    names.append(f"{edges[-1]}+")
    return np.asarray(names)[np.searchsorted(edges, values, side="right")]


def training_tertiles(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    support_bands(values, [1])
    groups = np.empty(len(values), dtype="<U6")
    for name, indices in zip(("low", "middle", "high"), np.array_split(np.argsort(values, kind="stable"), 3)):
        groups[indices] = name
    return groups


def fold_indices(row_count: int, seed: int):
    if row_count < 2:
        raise ValueError("cross-fitting requires at least two rows")
    first, second = np.array_split(np.random.default_rng(seed).permutation(row_count), 2)
    return ((first, second), (second, first))


def apply_fold_thresholds(scores: np.ndarray, thresholds: np.ndarray, seed: int) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    if scores.ndim != 2 or thresholds.shape != (2, scores.shape[1]) or not np.isfinite(thresholds).all():
        raise ValueError("fold thresholds must match score labels")
    predictions = np.zeros(scores.shape, dtype=np.uint8)
    for fold, (_, evaluation) in enumerate(fold_indices(len(scores), seed)):
        predictions[evaluation] = threshold_predictions(scores[evaluation], thresholds[fold])
    return predictions


def crossfit_predictions(target: np.ndarray, scores: np.ndarray, seed: int, shrinkage: float = 25.0):
    target = np.asarray(target)
    scores = np.asarray(scores, dtype=np.float32)
    thresholds = []
    receipts = []
    for fold, (selection, evaluation) in enumerate(fold_indices(len(target), seed)):
        fitted, global_threshold, _ = fit_rank_fusion_thresholds(target[selection], scores[selection], shrinkage=shrinkage)
        thresholds.append(fitted)
        receipts.append({"seed": seed, "fold": fold, "selection_rows": len(selection), "evaluation_rows": len(evaluation), "global_threshold": global_threshold})
    threshold_matrix = np.asarray(thresholds)
    return apply_fold_thresholds(scores, threshold_matrix, seed), threshold_matrix, receipts


def prediction_metrics(target: np.ndarray, predictions: np.ndarray) -> dict[str, np.ndarray]:
    target = np.asarray(target)
    predictions = np.asarray(predictions)
    if target.ndim != 2 or predictions.shape != target.shape or not np.isin(target, [0, 1]).all() or not np.isin(predictions, [0, 1]).all():
        raise ValueError("targets and predictions must be aligned binary matrices")
    support = target.sum(axis=0)
    predicted = predictions.sum(axis=0)
    true_positive = ((target == 1) & (predictions == 1)).sum(axis=0)
    false_positive = predicted - true_positive
    false_negative = support - true_positive
    divide = lambda numerator, denominator: np.divide(numerator, denominator, out=np.zeros(len(support), dtype=float), where=denominator != 0)
    return {"predicted_positive": predicted, "tp": true_positive, "fp": false_positive, "fn": false_negative, "precision": divide(true_positive, predicted), "recall": divide(true_positive, support), "f1": divide(2 * true_positive, 2 * true_positive + false_positive + false_negative), "predicted_to_true_ratio": np.divide(predicted, support, out=np.full(len(support), np.nan), where=support != 0)}


def score_overlap(target: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    target = np.asarray(target)
    scores = np.asarray(scores, dtype=np.float64)
    if target.ndim != 2 or scores.shape != target.shape or not np.isin(target, [0, 1]).all():
        raise ValueError("target and scores must be aligned binary matrices")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("scores must be finite probabilities")
    rows = []
    for index in range(target.shape[1]):
        positive = scores[target[:, index] == 1, index]
        negative = scores[target[:, index] == 0, index]
        positive_median = float(np.median(positive)) if len(positive) else np.nan
        negative_median = float(np.median(negative)) if len(negative) else np.nan
        negative_std = float(negative.std(ddof=0)) if len(negative) else np.nan
        gap = positive_median - negative_median
        rows.append({
            "positive_median": positive_median,
            "negative_median": negative_median,
            "negative_std": negative_std,
            "median_gap_over_negative_std": gap / negative_std if negative_std > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def diagnose_labels(target, scores, predictions, training_support, labels) -> pd.DataFrame:
    scores = np.asarray(scores, dtype=np.float32)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("scores must be finite probabilities")
    frame = per_label_score_diagnostics(target, predictions, scores, labels, training_support)
    frame["roc_auc"] = [roc_auc_score(target[:, index], scores[:, index]) if 0 < support < len(target) else np.nan for index, support in enumerate(frame.true_support)]
    frame["score_mean"] = scores.mean(axis=0, dtype=np.float64)
    frame["score_std"] = scores.std(axis=0, dtype=np.float64)
    frame["score_min"] = scores.min(axis=0)
    frame["score_max"] = scores.max(axis=0)
    support = frame.true_support.to_numpy()
    frame["positive_score_mean"] = np.divide((scores * target).sum(axis=0, dtype=float), support, out=np.full(len(labels), np.nan), where=support > 0)
    negative = len(target) - support
    frame["negative_score_mean"] = np.divide((scores * (1 - target)).sum(axis=0, dtype=float), negative, out=np.full(len(labels), np.nan), where=negative > 0)
    overlap = score_overlap(target, scores)
    for column in overlap.columns:
        frame[column] = overlap[column].to_numpy()
    frame["training_stratum"] = support_bands(training_support, [20, 100, 1000])
    frame["training_tertile"] = training_tertiles(training_support)
    frame["validation_stratum"] = support_bands(support, [1, 6, 11, 21, 51])
    frame["f1_std"] = 0.0
    return frame


def summarize_strata(frame: pd.DataFrame, row_count: int, axis: str | None = None) -> pd.DataFrame:
    rows = []
    grouped = [("all", frame)] if axis is None else frame.groupby(axis, sort=False)
    for name, subset in grouped:
        eligible = subset.true_support > 0
        true_positive, false_positive, false_negative = (float(subset[column].sum()) for column in ("tp", "fp", "fn"))
        rows.append({"stratum": name, "label_count": len(subset), "f1_label_count": int(eligible.sum()), "auc_label_count": int(subset.roc_auc.notna().sum()), "macro_f1": float(subset.loc[eligible, "f1"].mean()), "macro_auc": float(subset.roc_auc.mean()), "macro_average_precision": float(subset.average_precision.mean()), "tp": true_positive, "fp": false_positive, "fn": false_negative, "predicted_positive_rate": float(subset.predicted_positive.sum() / (row_count * len(subset))), "true_positive_rate": float(subset.true_support.sum() / (row_count * len(subset))), "precision_micro": true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0, "recall_micro": true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0, "median_gap_over_negative_std_mean": float(subset.median_gap_over_negative_std.mean()), "median_gap_over_negative_std_median": float(subset.median_gap_over_negative_std.median()), "median_gap_over_negative_std_p25": float(subset.median_gap_over_negative_std.quantile(0.25)), "median_gap_over_negative_std_p75": float(subset.median_gap_over_negative_std.quantile(0.75))})
    return pd.DataFrame(rows)


def paired_label_changes(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    if before.label.tolist() != after.label.tolist() or not np.array_equal(before.true_support, after.true_support):
        raise ValueError("paired labels and supports must match")
    paired = before[["label", "training_support", "true_support", "training_stratum", "training_tertile", "validation_stratum"]].copy()
    for column in ["f1", "roc_auc", "average_precision", "predicted_positive", "tp", "fp", "fn", "precision", "recall", "score_mean", "score_std"]:
        paired[f"before_{column}"] = before[column].to_numpy()
        paired[f"after_{column}"] = after[column].to_numpy()
        paired[f"delta_{column}"] = after[column].to_numpy() - before[column].to_numpy()
    eligible = paired.true_support > 0
    paired["macro_f1_contribution"] = np.where(eligible, paired.delta_f1 / max(1, int(eligible.sum())), 0.0)
    auc_eligible = paired.delta_roc_auc.notna()
    paired["macro_auc_contribution"] = np.where(auc_eligible, paired.delta_roc_auc / max(1, int(auc_eligible.sum())), 0.0)
    ap_eligible = paired.delta_average_precision.notna()
    paired["macro_ap_contribution"] = np.where(ap_eligible, paired.delta_average_precision / max(1, int(ap_eligible.sum())), 0.0)
    paired["new_zero_prediction"] = eligible & (paired.before_predicted_positive > 0) & (paired.after_predicted_positive == 0)
    paired["new_zero_tp"] = eligible & (paired.before_tp > 0) & (paired.after_tp == 0)
    paired["auc_up_f1_down"] = eligible & (paired.delta_roc_auc > 1e-12) & (paired.delta_f1 < -1e-12)
    paired["score_compression"] = eligible & (paired.before_score_std > 0) & (paired.after_score_std <= 0.1 * paired.before_score_std)
    return paired


def pair_summary(paired: pd.DataFrame) -> dict:
    negative = -paired.macro_f1_contribution.clip(upper=0)
    total_negative = float(negative.sum())
    eligible = paired.true_support > 0
    return {"delta_macro_f1": float(paired.macro_f1_contribution.sum()), "delta_macro_auc": float(paired.macro_auc_contribution.sum()), "gross_f1_decline": total_negative, "gross_f1_gain": float(paired.macro_f1_contribution.clip(lower=0).sum()), "f1_down_labels": int((eligible & (paired.delta_f1 < -1e-12)).sum()), "f1_up_labels": int((eligible & (paired.delta_f1 > 1e-12)).sum()), "auc_up_f1_down_labels": int(paired.auc_up_f1_down.sum()), "top10_decline_share": float(negative.nlargest(10).sum() / total_negative) if total_negative else 0.0, "top20_decline_share": float(negative.nlargest(20).sum() / total_negative) if total_negative else 0.0, "new_zero_prediction_labels": int(paired.new_zero_prediction.sum()), "new_zero_tp_labels": int(paired.new_zero_tp.sum()), "zero_tp_decline_share": float(negative[paired.new_zero_tp].sum() / total_negative) if total_negative else 0.0, "score_compression_labels": int(paired.score_compression.sum()), "delta_tp": float(paired.delta_tp.sum()), "delta_fp": float(paired.delta_fp.sum()), "delta_fn": float(paired.delta_fn.sum()), "delta_predicted_positive": float(paired.delta_predicted_positive.sum())}


def output_paths(prefix: Path) -> dict[str, Path]:
    suffixes = {"summary": "-summary.json", "labels": "-per-label.csv", "seed_labels": "-per-label-seeds.csv", "strata": "-strata.csv", "pairs": "-paired-labels.csv", "paired_strata": "-paired-strata.csv", "counterfactuals": "-counterfactuals.csv", "thresholds": "-thresholds.csv", "curve": "-curves.png"}
    outputs = {name: prefix.with_name(prefix.name + suffix) for name, suffix in suffixes.items()}
    outputs.update({name: prefix.with_name(prefix.name + suffix) for name, suffix in {
        "overlap": "-overlap.csv",
        "overlap_distribution": "-overlap-distribution.csv",
        "prevalence": "-prevalence-decomposition.csv",
        "support_contrasts": "-support-contrasts.csv",
    }.items()})
    if any(path.exists() for path in outputs.values()):
        raise FileExistsError("Use a fresh diagnostic output prefix")
    return outputs


def json_records(frame: pd.DataFrame) -> list[dict]:
    return frame.astype(object).where(frame.notna(), None).to_dict(orient="records")


def overlap_distribution(frame: pd.DataFrame) -> dict:
    gap = frame.median_gap_over_negative_std.dropna()
    result = {
        "label_count": len(frame), "defined_labels": len(gap), "undefined_labels": len(frame) - len(gap),
        "training_support_min": int(frame.training_support.min()),
        "training_support_max": int(frame.training_support.max()),
        "macro_auc": float(frame.roc_auc.mean()),
        "mean": float(gap.mean()), "std": float(gap.std(ddof=0)),
        "gap_le_zero_labels": int((gap <= 0).sum()),
        "gap_lt_one_labels": int((gap < 1).sum()),
        "gap_zero_to_one_labels": int(((gap > 0) & (gap < 1)).sum()),
        "gap_one_to_two_labels": int(((gap >= 1) & (gap < 2)).sum()),
        "gap_ge_two_labels": int((gap >= 2).sum()),
        "auc_below_half_labels": int((frame.roc_auc < .5).sum()),
        "positive_median_below_half_labels": int((frame.positive_median < .5).sum()),
        "negative_median_above_half_labels": int((frame.negative_median > .5).sum()),
    }
    result.update({f"p{int(quantile * 100):02d}": float(gap.quantile(quantile)) for quantile in [0, .05, .1, .25, .5, .75, .9, .95, 1]})
    return result


def prevalence_decomposition(frame: pd.DataFrame, row_count: int) -> dict:
    difference = frame.predicted_positive.astype(np.float64) - frame.true_support.astype(np.float64)
    excess = difference.clip(lower=0)
    excess_sum = float(excess.sum())
    predicted_sum = float(frame.predicted_positive.sum())
    result = {
        "label_count": len(frame), "true_positive": float(frame.true_support.sum()),
        "predicted_positive": predicted_sum,
        "predicted_positive_rate": predicted_sum / (row_count * len(frame)),
        "true_positive_rate": float(frame.true_support.sum() / (row_count * len(frame))),
        "positive_excess": excess_sum,
        "positive_deficit": float((-difference).clip(lower=0).sum()),
        "net_excess": float(difference.sum()),
        "overpredicted_labels": int((frame.predicted_positive > frame.true_support).sum()),
        "underpredicted_labels": int((frame.predicted_positive < frame.true_support).sum()),
        "overpredicted_supported_labels": int(((frame.predicted_positive > frame.true_support) & (frame.true_support > 0)).sum()),
        "supported_labels": int((frame.true_support > 0).sum()),
        "overpredicted_twice_labels": int(((frame.predicted_positive > 2 * frame.true_support) & (frame.true_support > 0)).sum()),
        "zero_support_labels": int((frame.true_support == 0).sum()),
        "zero_support_predicted_positive": float(frame.loc[frame.true_support == 0, "predicted_positive"].sum()),
        "tp": float(frame.tp.sum()), "fp": float(frame.fp.sum()), "fn": float(frame.fn.sum()),
    }
    for count in [10, 20, 50, 100]:
        result[f"top{count}_positive_excess_share"] = float(excess.nlargest(count).sum() / excess_sum) if excess_sum else 0.0
        result[f"top{count}_predicted_positive_share"] = float(frame.predicted_positive.nlargest(count).sum() / predicted_sum) if predicted_sum else 0.0
    return result


def support_contrasts(frame: pd.DataFrame) -> list[dict]:
    generator = np.random.default_rng(42)
    rows = []
    for column in ["roc_auc", "median_gap_over_negative_std", "f1"]:
        high = frame.loc[frame.training_tertile == "high", column].dropna().to_numpy()
        low = frame.loc[frame.training_tertile == "low", column].dropna().to_numpy()
        if not len(high) or not len(low):
            continue
        bootstrap = generator.choice(high, (2000, len(high))).mean(axis=1) - generator.choice(low, (2000, len(low))).mean(axis=1)
        rows.append({"metric": column, "high_minus_low": float(high.mean() - low.mean()),
                     "ci95_low": float(np.quantile(bootstrap, .025)), "ci95_high": float(np.quantile(bootstrap, .975)),
                     "high_labels": len(high), "low_labels": len(low), "bootstrap_replicates": 2000})
    return rows


def run_diagnostics(root: Path, config_path: Path, reference_path: Path, prefix: Path, epochs: list[int]) -> dict:
    torch_imported_before = "torch" in sys.modules
    outputs = output_paths(prefix)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    run_dir = root / config["output_dir"]
    summary_path = run_dir / "summary.json"
    source_digest = digest(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if source_digest != reference["source_summary_sha256"] or summary["experiment_id"] != reference["experiment_id"]:
        raise ValueError("Reference does not match the completed training run")
    labels = label_columns(pd.read_csv(root / config["data"]["train_path"], nrows=0).columns)
    validation_ids = pd.read_csv(root / config["split"]["validation_ids"])["protein_id"].astype(str).to_numpy()
    training_ids = pd.read_csv(root / config["split"]["train_ids"])["protein_id"].astype(str).to_numpy()
    if set(validation_ids) & set(training_ids) or len(set(validation_ids)) != len(validation_ids) or len(set(training_ids)) != len(training_ids):
        raise ValueError("Split IDs must be unique and disjoint")
    frame = pd.read_csv(root / config["data"]["train_path"], usecols=["protein_id", *labels], dtype={label: np.uint8 for label in labels})
    frame["protein_id"] = frame["protein_id"].astype(str)
    frame = frame.set_index("protein_id")
    target = frame.loc[validation_ids, labels].to_numpy(dtype=np.uint8)
    training_support = frame.loc[training_ids, labels].sum().to_numpy()
    if target.shape != (summary["validation_rows"], summary["label_count"]) or len(training_ids) != summary["train_rows"]:
        raise ValueError("Data dimensions differ from the completed run")
    del frame
    verified = {entry["epoch"]: entry for entry in reference["history"]}
    expected_hashes = {entry["epoch"]: entry["sha256"] for entry in reference["score_provenance"]}
    defaults, seed_frames, threshold_rows, counterfactual_rows, provenance = [], [], [], [], []
    baseline_thresholds = {}
    baseline_f1 = {}
    for epoch in epochs:
        path = run_dir / "epochs" / f"epoch{epoch:02d}-validation_scores.npz"
        before_digest = digest(path)
        if before_digest != expected_hashes[epoch]:
            raise ValueError("Historical epoch scores changed")
        with np.load(path, allow_pickle=False) as saved:
            if saved["epoch"].item() != epoch or not np.array_equal(saved["validation_ids"].astype(str), validation_ids) or saved["label_columns"].astype(str).tolist() != labels:
                raise ValueError("Saved epoch, ID or label order mismatch")
            scores = saved["validation_scores"].astype(np.float32)
        default = diagnose_labels(target, scores, scores >= 0.5, training_support, labels)
        default["epoch"], default["mode"] = epoch, "default05"
        defaults.append(default)
        if abs(summarize_strata(default, len(target)).iloc[0].macro_auc - verified[epoch]["validation_macro_auc"]) > 1e-12:
            raise ValueError("Recomputed AUC differs from the reference")
        for seed in SEEDS:
            predictions, thresholds, receipts = crossfit_predictions(target, scores, seed)
            calibrated = default.copy()
            for name, values in prediction_metrics(target, predictions).items():
                calibrated[name] = values
            calibrated["mode"], calibrated["seed"] = "crossfit", seed
            calibrated["threshold_mean"] = thresholds.mean(axis=0)
            seed_frames.append(calibrated)
            for fold, (selection, _) in enumerate(fold_indices(len(target), seed)):
                for index, label in enumerate(labels):
                    threshold_rows.append({"epoch": epoch, "label": label, "threshold": float(thresholds[fold, index]), "selection_support": int(target[selection, index].sum()), **receipts[fold]})
            own_f1 = macro_f1_skip_empty(target, predictions)
            if epoch == epochs[0]:
                baseline_thresholds[seed], baseline_f1[seed] = thresholds, own_f1
            else:
                fixed_predictions = apply_fold_thresholds(scores, baseline_thresholds[seed], seed)
                fixed_f1 = macro_f1_skip_empty(target, fixed_predictions)
                counterfactual_rows.append({"before_epoch": epochs[0], "after_epoch": epoch, "seed": seed, "baseline_f1": baseline_f1[seed], "fixed_threshold_f1": fixed_f1, "refitted_f1": own_f1, "score_change_component": fixed_f1 - baseline_f1[seed], "threshold_refit_component": own_f1 - fixed_f1, "total_f1_change": own_f1 - baseline_f1[seed], "fixed_predicted_positive_rate": float(fixed_predictions.mean()), "refitted_predicted_positive_rate": float(predictions.mean())})
        if digest(path) != before_digest:
            raise ValueError("Epoch scores changed during diagnosis")
        provenance.append({"epoch": epoch, "path": path.relative_to(root).as_posix(), "sha256": before_digest})
        print(json.dumps({"diagnosed_epoch": epoch, "cpu_only": True}), flush=True)
    seeds = pd.concat(seed_frames, ignore_index=True)
    keys = ["epoch", "mode", "label", "training_stratum", "training_tertile", "validation_stratum"]
    grouped = seeds.drop(columns="seed").groupby(keys, sort=False)
    calibrated = grouped.mean(numeric_only=True)
    calibrated["f1_std"] = grouped.f1.std(ddof=0)
    combined = pd.concat([pd.concat(defaults, ignore_index=True), calibrated.reset_index()], ignore_index=True)
    global_rows, strata_rows, paired_rows, paired_groups, pair_results = [], [], [], [], []
    for (epoch, mode), subset in combined.groupby(["epoch", "mode"], sort=True):
        global_rows.append({"epoch": int(epoch), "mode": mode, **summarize_strata(subset, len(target)).iloc[0].to_dict()})
        expected_f1 = verified[epoch]["mean_crossfit_macro_f1" if mode == "crossfit" else "validation_macro_f1"]
        expected_rate = verified[epoch]["mean_crossfit_predicted_positive_rate" if mode == "crossfit" else "default_predicted_positive_rate"]
        if abs(global_rows[-1]["macro_f1"] - expected_f1) > 1e-12 or abs(global_rows[-1]["predicted_positive_rate"] - expected_rate) > 1e-12:
            raise ValueError("Diagnostic predictions do not reproduce reference calibration")
        for axis in ["training_stratum", "training_tertile", "validation_stratum"]:
            for entry in summarize_strata(subset, len(target), axis).to_dict(orient="records"):
                strata_rows.append({"epoch": int(epoch), "mode": mode, "axis": axis, **entry})
        if epoch == epochs[0]:
            continue
        before = combined[(combined.epoch == epochs[0]) & (combined["mode"] == mode)].set_index("label").loc[labels].reset_index()
        after = subset.set_index("label").loc[labels].reset_index()
        paired = paired_label_changes(before, after)
        paired["before_epoch"], paired["after_epoch"], paired["mode"] = epochs[0], epoch, mode
        paired_rows.append(paired)
        pair_results.append({"before_epoch": epochs[0], "after_epoch": int(epoch), "mode": mode, **pair_summary(paired)})
        for axis in ["training_stratum", "training_tertile", "validation_stratum"]:
            for name, group in paired.groupby(axis, sort=False):
                paired_groups.append({"before_epoch": epochs[0], "after_epoch": int(epoch), "mode": mode, "axis": axis, "stratum": name, "label_count": len(group), **pair_summary(group)})
    counterfactuals = pd.DataFrame(counterfactual_rows)
    counterfactual_summary = counterfactuals.groupby("after_epoch")[list(counterfactuals.select_dtypes(include="number").columns.difference(["before_epoch", "after_epoch", "seed"]))].mean().reset_index()
    global_frame = pd.DataFrame(global_rows)
    strata = pd.DataFrame(strata_rows)
    pair_groups = pd.DataFrame(paired_groups)
    overlap = pd.concat(defaults, ignore_index=True)[[
        "epoch", "label", "training_support", "true_support", "training_tertile",
        "positive_median", "negative_median", "negative_std", "median_gap_over_negative_std", "roc_auc",
    ]]
    overlap_rows, prevalence_rows, contrast_rows = [], [], []
    for (epoch, mode), subset in combined.groupby(["epoch", "mode"], sort=True):
        groups = [("all", subset), *list(subset.groupby("training_tertile", sort=False))]
        if mode == "default05":
            for stratum, group in groups:
                overlap_rows.append({"epoch": int(epoch), "stratum": stratum, **overlap_distribution(group)})
        if mode == "crossfit":
            for stratum, group in groups:
                prevalence_rows.append({"epoch": int(epoch), "stratum": stratum, **prevalence_decomposition(group, len(target))})
            contrast_rows.extend({"epoch": int(epoch), **entry} for entry in support_contrasts(subset))
    result = {"analysis_id": prefix.name, "source_experiment_id": summary["experiment_id"], "epochs": epochs, "seeds": list(SEEDS), "validation_rows": len(target), "label_count": len(labels), "training_rows": len(training_ids), "source_summary_sha256": source_digest, "reference_sha256": digest(reference_path), "score_provenance": provenance, "protocol": {"shrinkage": 25, "global_grid_step": 0.02, "calibration": "two-fold selection-only thresholds, merged evaluation predictions", "label_averaging": "per-label F1 averaged across seeds, not F1 of averaged counts", "collapse_definition": "new zero prediction or zero TP with positive validation support; crossfit means zero means every seed is zero", "score_compression_flag": "score std falls to <=10% of baseline; not proof of hidden representation collapse", "counterfactual": "apply baseline selection-fold thresholds to later evaluation-fold scores; ordered descriptive decomposition, not causal attribution"}, "global": json_records(global_frame), "pairs": pair_results, "counterfactual_means": json_records(counterfactual_summary), "outputs": {name: path.relative_to(root).as_posix() for name, path in outputs.items()}, "runtime": {"cpu_only": True, "torch_imported": not torch_imported_before and "torch" in sys.modules, "training_started": False, "submission_generated": False}}
    result["overlap_distributions"] = json_records(pd.DataFrame(overlap_rows))
    result["prevalence_decompositions"] = json_records(pd.DataFrame(prevalence_rows))
    result["support_contrasts"] = json_records(pd.DataFrame(contrast_rows))
    result["protocol"].update({
        "overlap_indicator": "(positive median - negative median) / negative std(ddof=0); undefined with missing class or zero negative std; NOT an overlap coefficient",
        "overlap_bins": "<=0, (0,1), [1,2), >=2; descriptive heuristic, not calibrated biological criteria",
        "training_tertiles": "stable ascending training support order, np.array_split into low/middle/high; ties use label order; validation labels never used to define groups",
        "support_uncertainty": "2000 label bootstrap samples, seed42, high-minus-low mean; labels are correlated and this is not a causal test or independent training replication",
        "prevalence_excess": "sum(max(mean predicted count over 5 seeds - true count,0)); positive deficit separate; counts include empty-validation labels",
    })
    if result["runtime"]["torch_imported"] or digest(summary_path) != source_digest:
        raise ValueError("Unexpected model runtime import or changed training summary")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for key, table in {"labels": combined, "seed_labels": seeds, "strata": strata, "pairs": pd.concat(paired_rows, ignore_index=True), "paired_strata": pair_groups, "counterfactuals": counterfactuals, "thresholds": pd.DataFrame(threshold_rows)}.items():
        table.to_csv(outputs[key], index=False)
    for key, table in {"overlap": overlap, "overlap_distribution": pd.DataFrame(overlap_rows), "prevalence": pd.DataFrame(prevalence_rows), "support_contrasts": pd.DataFrame(contrast_rows)}.items():
        table.to_csv(outputs[key], index=False)
    plot_diagnostics(global_frame, pair_groups, epochs, outputs["curve"])
    outputs["summary"].write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(global_frame[["epoch", "mode", "macro_auc", "macro_average_precision", "macro_f1", "predicted_positive_rate", "precision_micro", "recall_micro"]].to_string(index=False))
    return result


def plot_diagnostics(global_frame: pd.DataFrame, pairs: pd.DataFrame, epochs: list[int], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=160)
    for mode, label in [("default05", "默认 0.5"), ("crossfit", "五 seed 校准")]:
        subset = global_frame[global_frame["mode"] == mode]
        axes[0, 0].plot(subset.epoch, subset.predicted_positive_rate * 100, marker="o", label=label)
    axes[0, 0].axhline(global_frame.true_positive_rate.iloc[0] * 100, color="gray", linestyle="--", label="真实正例率")
    axes[0, 0].set_title("全局预测正例率（%）")
    calibrated = global_frame[global_frame["mode"] == "crossfit"]
    axes[0, 1].plot(calibrated.epoch, calibrated.precision_micro, marker="o", label="Micro precision（平均计数）")
    axes[0, 1].plot(calibrated.epoch, calibrated.recall_micro, marker="o", label="Micro recall（平均计数）")
    axes[0, 1].set_title("校准后的全局错误结构")
    selected = pairs[(pairs.after_epoch == 8) & (pairs["mode"] == "crossfit") & (pairs.axis == "validation_stratum")]
    axes[1, 0].bar(selected.stratum, selected.delta_macro_f1, color="#bd4434")
    axes[1, 0].set_title("ep6→ep8：各验证支持度层的 Macro F1 贡献")
    axes[1, 1].bar(selected.stratum, selected.delta_macro_auc, color="#1f4e78")
    axes[1, 1].set_title("ep6→ep8：各验证支持度层的 Macro AUC 贡献")
    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.2)
    for axis in axes[0]:
        axis.set_xticks(epochs)
        axis.set_xlabel("epoch")
        axis.legend()
    for axis in axes[1]:
        axis.axhline(0, color="gray", linewidth=0.8)
        axis.set_xlabel("验证正例支持度")
    figure.suptitle("EXP-128：复用 EXP-127 后五轮分数的 CPU 诊断", fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(output, facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--epochs", type=int, nargs="+", default=[6, 7, 8, 9, 10])
    arguments = parser.parse_args()
    if len(arguments.epochs) < 2 or arguments.epochs != sorted(set(arguments.epochs)):
        parser.error("epochs must be increasing and unique")
    root = arguments.root.resolve()
    run_diagnostics(root, root / arguments.config, root / arguments.reference, root / arguments.output_prefix, arguments.epochs)


if __name__ == "__main__":
    main()
