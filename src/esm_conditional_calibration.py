"""Leakage-safe support-stratum conditional calibration for saved ESM scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .diagnose_esm_epochs import crossfit_predictions, fold_indices, training_tertiles
from .metrics import macro_f1_skip_empty, macro_roc_auc_skip_degenerate
from .rank_fusion import fit_rank_fusion_thresholds
from .thresholds import shrink_label_thresholds, threshold_predictions


DEFAULT_VARIANTS = ("baseline_global_prior", "middle_only_stratum_prior", "all_strata_prior")
DEFAULT_STRATA = ("low", "middle", "high")


def _validate_matrix(target: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    target = np.asarray(target)
    scores = np.asarray(scores, dtype=np.float32)
    if target.ndim != 2 or scores.shape != target.shape:
        raise ValueError("target and scores must have equal two-dimensional shapes")
    if not np.isin(target, [0, 1]).all():
        raise ValueError("target must be binary")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("scores must be finite probabilities")
    return target.astype(np.uint8), scores


def _grid(values: Iterable[float] | None) -> np.ndarray:
    if values is None:
        values = np.arange(0.02, 1.0, 0.02, dtype=np.float32)
    values = np.asarray(tuple(values), dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("prior grid must be a non-empty finite one-dimensional sequence")
    if ((values <= 0) | (values >= 1)).any():
        raise ValueError("prior grid values must be in (0, 1)")
    return values


def select_support_prior(
    target: np.ndarray,
    scores: np.ndarray,
    candidates: Iterable[float],
    *,
    label_indices: np.ndarray,
    fallback: float | None = None,
) -> tuple[float, pd.DataFrame]:
    """Select a shared prior using only the selection fold and one support stratum."""
    target, scores = _validate_matrix(target, scores)
    labels = np.asarray(label_indices, dtype=np.int64)
    if labels.ndim != 1 or not len(labels) or (labels < 0).any() or (labels >= target.shape[1]).any():
        raise ValueError("label_indices must contain at least one valid label")
    candidates = _grid(candidates)
    rows = []
    subset_target = target[:, labels]
    subset_scores = scores[:, labels]
    for position, candidate in enumerate(candidates):
        predictions = subset_scores >= float(candidate)
        if subset_target.sum() and (subset_target.shape[0] * subset_target.shape[1] - subset_target.sum()):
            score = macro_f1_skip_empty(subset_target, predictions)
        else:
            score = 0.0
        rows.append(
            {
                "threshold": float(candidate),
                "macro_f1": float(score),
                "distance_to_half": abs(float(candidate) - 0.5),
                "candidate_order": position,
                "label_count": int(len(labels)),
                "positive_count": int(subset_target.sum()),
                "fallback": False,
            }
        )
    diagnostics = pd.DataFrame(rows)
    if not int(subset_target.sum()) or not int((subset_target == 0).sum()):
        if fallback is None:
            raise ValueError("cannot select a prior from a degenerate stratum without fallback")
        selected = float(fallback)
        diagnostics["fallback"] = True
    else:
        selected_row = max(
            rows,
            key=lambda row: (
                row["macro_f1"],
                -row["distance_to_half"],
                -row["candidate_order"],
            ),
        )
        selected = float(selected_row["threshold"])
    diagnostics["selected"] = np.isclose(diagnostics["threshold"], selected)
    return selected, diagnostics


def _variant_strata(variant: str) -> tuple[str, ...]:
    if variant == "baseline_global_prior":
        return ()
    if variant == "middle_only_stratum_prior":
        return ("middle",)
    if variant == "all_strata_prior":
        return DEFAULT_STRATA
    raise ValueError(f"unknown calibration variant: {variant}")


def fit_variant_thresholds(
    target: np.ndarray,
    scores: np.ndarray,
    strata: np.ndarray,
    *,
    variant: str,
    prior_grid: Iterable[float] | None = None,
    shrinkage: float = 25.0,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Fit baseline and one conditional threshold vector on a selection fold."""
    target, scores = _validate_matrix(target, scores)
    strata = np.asarray(strata)
    if strata.shape != (target.shape[1],) or not set(strata).issubset(set(DEFAULT_STRATA)):
        raise ValueError("strata must contain one low/middle/high value per label")
    if shrinkage < 0 or not np.isfinite(shrinkage):
        raise ValueError("shrinkage must be a finite non-negative value")
    prior_grid = _grid(prior_grid)
    baseline, global_prior, _ = fit_rank_fusion_thresholds(
        target, scores, shrinkage=float(shrinkage)
    )
    label_thresholds, _, _ = fit_rank_fusion_thresholds(
        target, scores, shrinkage=0.0
    )
    # fit_rank_fusion_thresholds with shrinkage=0 returns the exact label thresholds.
    supports = target.sum(axis=0).astype(np.float32)
    variant_thresholds = baseline.copy()
    rows: list[dict] = []
    applicable = set(_variant_strata(variant))
    for stratum in DEFAULT_STRATA:
        indices = np.flatnonzero(strata == stratum)
        if stratum in applicable:
            prior, diagnostics = select_support_prior(
                target, scores, prior_grid, label_indices=indices, fallback=global_prior
            )
        else:
            prior = float(global_prior)
            diagnostics = pd.DataFrame()
        if stratum in applicable:
            variant_thresholds[indices] = shrink_label_thresholds(
                label_thresholds[indices],
                supports[indices],
                global_threshold=prior,
                shrinkage=float(shrinkage),
            )
        for label_index in indices:
            rows.append(
                {
                    "variant": variant,
                    "stratum": stratum,
                    "label_index": int(label_index),
                    "selection_support": int(supports[label_index]),
                    "global_prior": float(global_prior),
                    "stratum_prior": float(prior),
                    "baseline_threshold": float(baseline[label_index]),
                    "variant_threshold": float(variant_thresholds[label_index]),
                    "changed": bool(not np.isclose(baseline[label_index], variant_thresholds[label_index])),
                    "prior_fallback": bool(len(diagnostics) and diagnostics["fallback"].all()),
                    "zero_positive_fallback": bool(supports[label_index] == 0),
                    "label_threshold": float(label_thresholds[label_index]),
                    "prior_weight": float(25 / (supports[label_index] + 25)) if shrinkage == 25 else float(shrinkage / (supports[label_index] + shrinkage)) if shrinkage else 0.0,
                }
            )
    frame = pd.DataFrame(rows)
    # Preserve every selection-grid candidate, including degenerate fallbacks.
    grids = []
    for stratum in DEFAULT_STRATA:
        if stratum not in applicable:
            continue
        indices = np.flatnonzero(strata == stratum)
        _, diagnostic = select_support_prior(target, scores, prior_grid, label_indices=indices, fallback=global_prior)
        diagnostic = diagnostic.assign(variant=variant, stratum=stratum)
        grids.extend(diagnostic.to_dict(orient="records"))
    frame.attrs["prior_grid"] = grids
    return baseline.astype(np.float32), variant_thresholds.astype(np.float32), frame


def _raw_metrics(target: np.ndarray, scores: np.ndarray, label_indices: np.ndarray) -> tuple[float, float]:
    labels = np.asarray(label_indices, dtype=np.int64)
    subset_target, subset_scores = target[:, labels], scores[:, labels]
    support = subset_target.sum(axis=0)
    auc_defined = (support > 0) & (support < len(target))
    # No exception is swallowed: explicitly mark a wholly degenerate subset.
    auc = macro_roc_auc_skip_degenerate(subset_target, subset_scores) if auc_defined.any() else float("nan")
    active = subset_target.sum(axis=0) > 0
    ap = float(
        np.mean(
            [average_precision_score(subset_target[:, i], subset_scores[:, i]) for i in np.flatnonzero(active)]
        )
    ) if active.any() else float("nan")
    return float(auc), ap


def _metric_row(
    target: np.ndarray,
    scores: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    *,
    variant: str,
    seed: int,
    fold: int | str,
    stratum: str,
    raw_metrics: tuple[float, float] | None = None,
) -> dict:
    true_support = target[:, labels].sum(axis=0).astype(np.float64)
    predicted_support = predictions[:, labels].sum(axis=0).astype(np.float64)
    tp = ((target[:, labels] == 1) & (predictions[:, labels] == 1)).sum(axis=0).astype(np.float64)
    fp = predicted_support - tp
    fn = true_support - tp
    eligible = true_support > 0
    f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp), where=(2 * tp + fp + fn) != 0)
    over = predicted_support > true_support
    raw_auc, raw_ap = raw_metrics if raw_metrics is not None else _raw_metrics(target, scores, labels)
    return {
        "variant": variant,
        "seed": seed,
        "fold": fold,
        "stratum": stratum,
        "label_count": int(len(labels)),
        "auc_label_count": int(((true_support > 0) & (true_support < len(target))).sum()),
        "ap_label_count": int(eligible.sum()),
        "macro_f1": float(f1[eligible].mean()) if eligible.any() else float("nan"),
        "raw_auc": raw_auc,
        "raw_ap": raw_ap,
        "predicted_positive_rate": float(predicted_support.sum() / (len(target) * len(labels))),
        "true_positive_rate": float(true_support.sum() / (len(target) * len(labels))),
        "overpred_labels": int(over.sum()),
        "tp": float(tp.sum()),
        "fp": float(fp.sum()),
        "fn": float(fn.sum()),
        "excess": float(np.maximum(predicted_support - true_support, 0).sum()),
    }


def _resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _read_scores(path: Path, expected_epoch: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    with np.load(path, allow_pickle=False) as saved:
        keys = set(saved.files)
        required = {"validation_scores", "validation_ids", "label_columns", "epoch"}
        if not required.issubset(keys):
            raise ValueError(f"score file is missing keys: {sorted(required - keys)}")
        if saved["epoch"].size != 1 or saved["epoch"].item() != expected_epoch:
            raise ValueError("score epoch does not match the bound configuration")
        scores = saved["validation_scores"]
        ids = saved["validation_ids"].astype(str)
        labels = saved["label_columns"].astype(str).tolist()
    return scores, ids, labels


def aggregate_label_results(per_seed_labels: pd.DataFrame, *, validation_rows: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Average label counts before computing excess; average already-computed F1."""
    keys = ["variant", "label", "training_tertile"]
    values = [key for key in ("label_index", "training_support", "true_support", "predicted_positive", "tp", "fp", "fn", "f1", "roc_auc", "average_precision") if key in per_seed_labels]
    averaged = per_seed_labels.groupby(keys, sort=False, as_index=False)[values].mean()
    rows = []
    for variant, frame in averaged.groupby("variant", sort=False):
        for stratum in ("all", "high", "middle", "low"):
            subset = frame if stratum == "all" else frame.loc[frame.training_tertile == stratum]
            if subset.empty:
                continue
            eligible = subset.true_support > 0
            row = {"variant": variant, "stratum": stratum, "label_count": len(subset),
                   "calF1": float(subset.loc[eligible, "f1"].mean()),
                   "cal_pos_rate": float(subset.predicted_positive.sum() / (validation_rows * len(subset))),
                   "true_pos_rate": float(subset.true_support.sum() / (validation_rows * len(subset))),
                   "overpred_labels": int((subset.predicted_positive > subset.true_support).sum()),
                   "excess": float((subset.predicted_positive - subset.true_support).clip(lower=0).sum()),
                   "TP": float(subset.tp.sum()), "FP": float(subset.fp.sum()), "FN": float(subset.fn.sum())}
            if "roc_auc" in subset:
                row["raw_AUC"] = float(subset.roc_auc.mean())
                row["raw_AP"] = float(subset.loc[eligible, "average_precision"].mean())
            rows.append(row)
    return pd.DataFrame(rows), averaged


def run_conditional_calibration(
    config: dict,
    *,
    project_root: str | Path | None = None,
) -> Path:
    """Run EXP-136 on a saved score matrix and write a complete CPU audit."""
    root = Path(project_root) if project_root is not None else Path.cwd()
    if config.get("execution", {}).get("enabled") is not True:
        raise ValueError("execution.enabled must be true for all runner entry points")
    if config["execution"].get("cpu_only") is not True:
        raise ValueError("conditional calibration must be CPU-only")
    for forbidden in ("training_allowed", "test_prediction_allowed", "submission_allowed", "fusion_allowed", "overwrite_allowed", "auto_enqueue"):
        if config["execution"].get(forbidden, False) is not False:
            raise ValueError(f"execution.{forbidden} must be false")
    if 'torch' in sys.modules:
        raise ValueError("use a fresh CPU evaluation process without torch imports")
    contract_path = config.get("prepared_config")
    additional_sources = []
    if contract_path:
        prepared_path = _resolve(root, contract_path)
        prepared = json.loads(prepared_path.read_text(encoding='utf-8'))
        for section in ("input", "stratification", "calibration", "evaluation"):
            if config.get(section) != prepared.get(section):
                raise ValueError(f"execution config changed the prepared {section} contract")
        review = config['final_review']
        decision_path = _resolve(root, review['decisions_path'])
        if hashlib.sha256(decision_path.read_bytes()).hexdigest() != review['decisions_sha256']:
            raise ValueError("B/C final attribution evidence changed")
        decisions = json.loads(decision_path.read_text(encoding='utf-8'))
        if any(decisions[group]['state'] != 'confirmed_for_completed_suite' for group in ('A','B','C')):
            raise ValueError("A/B/C final attribution is incomplete")
        additional_sources = [prepared_path, decision_path]
    input_cfg = config["input"]
    scores_path = _resolve(root, input_cfg["scores_path"])
    if not scores_path.exists():
        raise FileNotFoundError(f"saved score matrix not found: {scores_path}")
    if not input_cfg.get("scores_sha256"):
        raise ValueError("a bound score SHA-256 is required")
    actual_hash = hashlib.sha256(scores_path.read_bytes()).hexdigest()
    if actual_hash != input_cfg["scores_sha256"]:
        raise ValueError("saved score matrix SHA-256 does not match the prepared configuration")
    scores, validation_ids, labels = _read_scores(scores_path, int(input_cfg["epoch"]))
    expected_rows = input_cfg.get("expected_validation_rows")
    expected_labels = input_cfg.get("expected_label_count")
    if expected_rows is not None and len(validation_ids) != int(expected_rows):
        raise ValueError("validation row count does not match the prepared configuration")
    if expected_labels is not None and len(labels) != int(expected_labels):
        raise ValueError("label count does not match the prepared configuration")
    if labels != [f"label_{i}" for i in range(len(labels))]:
        raise ValueError("label order must exactly match label_0..label_N")
    canonical_validation = pd.read_csv(_resolve(root, input_cfg["validation_ids"]), dtype={"protein_id": str})["protein_id"].to_numpy()
    if not np.array_equal(validation_ids, canonical_validation) or len(set(validation_ids)) != len(validation_ids):
        raise ValueError("score validation IDs must match the fixed validation IDs in exact order")
    scores = np.asarray(scores, dtype=np.float32)
    if scores.shape != (len(validation_ids), len(labels)):
        raise ValueError("score matrix shape does not match IDs and labels")
    train_path = _resolve(root, input_cfg["train_path"])
    train = pd.read_csv(train_path, usecols=["protein_id", *labels])
    training_ids = pd.read_csv(_resolve(root, input_cfg["training_ids"]))["protein_id"].astype(str).to_numpy()
    validation_ids = validation_ids.astype(str)
    indexed = train.assign(protein_id=train["protein_id"].astype(str)).set_index("protein_id")
    if not indexed.index.is_unique or len(set(training_ids)) != len(training_ids):
        raise ValueError("protein IDs must be unique")
    if set(training_ids) & set(validation_ids):
        raise ValueError("training and validation IDs must be disjoint")
    if not pd.Index(validation_ids).isin(indexed.index).all() or not pd.Index(training_ids).isin(indexed.index).all():
        raise ValueError("training or validation IDs are absent from the training data")
    target = indexed.loc[validation_ids, labels].to_numpy()
    target, scores = _validate_matrix(target, scores)
    training_target = indexed.loc[training_ids, labels].to_numpy()
    if not np.isin(training_target, [0, 1]).all():
        raise ValueError("training labels must be binary")
    training_support = training_target.astype(np.uint8).sum(axis=0)
    strata = training_tertiles(training_support)
    expected_groups = config.get('stratification', {}).get('expected_group_sizes')
    if expected_groups and [int((strata == name).sum()) for name in DEFAULT_STRATA] != expected_groups:
        raise ValueError("training-support stratum sizes do not match the prepared configuration")
    if scores.shape == (1062, 500) and (len(training_ids) != 112734 or not contract_path):
        raise ValueError("production evaluation requires fixed training rows and prepared contract binding")
    calibration_cfg = config.get("calibration", {})
    seeds = tuple(int(seed) for seed in calibration_cfg.get("seeds", [17, 31, 42, 73, 101]))
    shrinkage = float(calibration_cfg.get("shrinkage", 25.0))
    if seeds != (17, 31, 42, 73, 101) or shrinkage != 25.0:
        raise ValueError("prepared protocol requires five fixed seeds and shrinkage 25")
    if calibration_cfg.get("score_transform", "identity") != "identity" or calibration_cfg.get("shared_platt", False) or calibration_cfg.get("full_validation_fit", False):
        raise ValueError("prepared protocol preserves raw scores and forbids full-validation fitting")
    prior_grid = _grid(
        np.arange(
            float(calibration_cfg.get("prior_grid", {}).get("start", 0.02)),
            float(calibration_cfg.get("prior_grid", {}).get("stop_exclusive", 1.0)),
            float(calibration_cfg.get("prior_grid", {}).get("step", 0.02)),
            dtype=np.float32,
        )
    )
    if not np.array_equal(prior_grid, _grid(None)):
        raise ValueError("prior grid must exactly reuse the historical float32 0.02 grid")
    raw_variants = calibration_cfg.get("variants", DEFAULT_VARIANTS)
    # The prepared JSON records variant metadata as objects, while small
    # programmatic callers commonly pass just names.  Normalize both forms
    # to the canonical names used in output tables.
    variants = tuple(
        item.get("name") if isinstance(item, dict) else str(item)
        for item in raw_variants
    )
    if set(variants) != set(DEFAULT_VARIANTS):
        raise ValueError("configuration must contain exactly the three prepared calibration variants")
    if len(variants) != len(DEFAULT_VARIANTS):
        raise ValueError("configuration must not repeat calibration variants")
    prefix = _resolve(root, config["output_prefix"])
    suffixes = {"summary": "-summary.json", "seed": "-per-seed.csv", "strata": "-strata.csv", "thresholds": "-thresholds.csv", "labels": "-per-label.csv", "label_seeds": "-per-label-seeds.csv", "prior_grid": "-prior-grid.csv", "verification": "-verification.json"}
    outputs = {name: prefix.with_name(prefix.name + suffix) for name, suffix in suffixes.items()}
    run_dir = _resolve(root, config["output_dir"])
    if any(path.exists() for path in outputs.values()) or run_dir.exists():
        raise FileExistsError("Use a fresh conditional calibration output prefix")

    seed_rows: list[dict] = []
    threshold_rows: list[dict] = []
    grid_rows: list[dict] = []
    per_label_rows: list[dict] = []
    predictions_by_variant: dict[str, list[np.ndarray]] = {variant: [] for variant in variants}
    indices_by_stratum = {name: np.arange(len(labels)) if name == "all" else np.flatnonzero(strata == name) for name in ("all", "high", "middle", "low")}
    full_raw = {name: _raw_metrics(target, scores, indices) for name, indices in indices_by_stratum.items()}
    # Require the production reference values when evaluating the bound 500-label matrix.
    if scores.shape == (1062, 500):
        if not np.isclose(full_raw["all"][0], .7680116652409509, rtol=0, atol=1e-12):
            raise ValueError("bound EXP-127 epoch6 AUC does not match the historical reference")
    per_label_raw = [_raw_metrics(target, scores, np.asarray([index])) for index in range(len(labels))]
    source_files = [scores_path, train_path, _resolve(root, input_cfg['training_ids']), _resolve(root, input_cfg['validation_ids']), Path(__file__), root/'src/rank_fusion.py', root/'src/thresholds.py', root/'src/diagnose_esm_epochs.py', *additional_sources]
    source_files.extend(path for path in (root/'artifacts/metrics/splits/seed42/train_ids.csv', root/'artifacts/metrics/splits/seed42/validation_ids.csv') if path.exists())
    source_hashes = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files if path.exists()}
    baseline_matches = True
    for seed in seeds:
        combined = {variant: np.zeros_like(target, dtype=np.uint8) for variant in variants}
        for fold, (selection, evaluation) in enumerate(fold_indices(len(target), seed)):
            fold_raw = {name: _raw_metrics(target[evaluation], scores[evaluation], indices) for name, indices in indices_by_stratum.items()}
            fitted: dict[str, np.ndarray] = {}
            baseline_thresholds = None
            for variant in variants:
                baseline, thresholds, records = fit_variant_thresholds(
                    target[selection], scores[selection], strata,
                    variant=variant, prior_grid=prior_grid, shrinkage=shrinkage,
                )
                if baseline_thresholds is None:
                    baseline_thresholds = baseline
                elif not np.array_equal(baseline_thresholds, baseline):
                    raise AssertionError("baseline thresholds changed across variants")
                fitted[variant] = thresholds
                records = records.copy()
                for grid_row in records.attrs.get("prior_grid", []):
                    grid_rows.append({**grid_row, "seed": seed, "fold": fold, "selection_rows": len(selection)})
                records["seed"] = seed
                records["fold"] = fold
                records["selection_rows"] = len(selection)
                records["evaluation_rows"] = len(evaluation)
                threshold_rows.extend(records.to_dict(orient="records"))
                if variant == "middle_only_stratum_prior" and not np.array_equal(thresholds[strata != 'middle'], baseline_thresholds[strata != 'middle']):
                    raise AssertionError("primary variant changed high/low reference thresholds")
                combined[variant][evaluation] = threshold_predictions(scores[evaluation], thresholds)
            for variant in variants:
                for stratum in ("all", "high", "middle", "low"):
                    labels_for_stratum = np.arange(len(labels)) if stratum == "all" else np.flatnonzero(strata == stratum)
                    seed_rows.append(_metric_row(target[evaluation], scores[evaluation], combined[variant][evaluation], labels_for_stratum, variant=variant, seed=seed, fold=fold, stratum=stratum, raw_metrics=fold_raw[stratum]))
        historical_predictions, historical_thresholds, _ = crossfit_predictions(target, scores, seed, shrinkage)
        if not np.array_equal(combined['baseline_global_prior'], historical_predictions):
            raise AssertionError("paired baseline does not reproduce historical crossfit predictions")
        baseline_records = [r for r in threshold_rows if r['variant']=='baseline_global_prior' and r['seed']==seed]
        for fold in (0, 1):
            actual = np.array([r['variant_threshold'] for r in sorted((r for r in baseline_records if r['fold']==fold), key=lambda r:r['label_index'])], dtype=np.float32)
            if not np.array_equal(actual, historical_thresholds[fold]):
                raise AssertionError("paired baseline thresholds do not reproduce historical thresholds")
        for variant in variants:
            predictions_by_variant[variant].append(combined[variant])
            for stratum in ("all", "high", "middle", "low"):
                labels_for_stratum = np.arange(len(labels)) if stratum == "all" else np.flatnonzero(strata == stratum)
                row = _metric_row(target, scores, combined[variant], labels_for_stratum, variant=variant, seed=seed, fold="combined", stratum=stratum, raw_metrics=full_raw[stratum])
                seed_rows.append(row)
            for index, label in enumerate(labels):
                truth = int(target[:, index].sum())
                predicted = int(combined[variant][:, index].sum())
                tp = int(((target[:, index] == 1) & (combined[variant][:, index] == 1)).sum())
                per_label_rows.append({"variant": variant, "seed": seed, "label": label, "label_index": index, "training_support": int(training_support[index]), "training_tertile": strata[index], "true_support": truth, "predicted_positive": predicted, "tp": tp, "fp": predicted-tp, "fn": truth-tp, "f1": 2*tp/(predicted+truth) if predicted+truth else 0.0, "roc_auc": per_label_raw[index][0], "average_precision": per_label_raw[index][1]})
        print(json.dumps({'seed_completed': seed, 'completed_seeds': len(predictions_by_variant['baseline_global_prior']), 'total_seeds': len(seeds)}), flush=True)

    seed_frame = pd.DataFrame(seed_rows)
    label_seed_frame = pd.DataFrame(per_label_rows)
    strata_frame, mean_labels = aggregate_label_results(label_seed_frame, validation_rows=len(target))
    strata_frame['raw_AUC'] = [full_raw[name][0] for name in strata_frame.stratum]
    strata_frame['raw_AP'] = [full_raw[name][1] for name in strata_frame.stratum]
    strata_frame['calF1_std'] = [float(seed_frame.loc[(seed_frame.variant == row.variant) & (seed_frame.stratum == row.stratum) & (seed_frame.fold == 'combined'), 'macro_f1'].std(ddof=0)) for row in strata_frame.itertuples()]
    raw_equal = all(rows[column].nunique(dropna=False) == 1 for _, rows in strata_frame.groupby('stratum') for column in ('raw_AUC','raw_AP'))
    if not raw_equal:
        raise AssertionError("raw AP/AUC changed across calibration variants")
    comparisons = []
    for row in strata_frame.itertuples():
        reference = strata_frame.loc[(strata_frame.variant == 'baseline_global_prior') & (strata_frame.stratum == row.stratum)].iloc[0]
        density_improves = abs(row.cal_pos_rate-row.true_pos_rate) < abs(reference.cal_pos_rate-reference.true_pos_rate)
        gates = {'calF1_improves': bool(row.calF1 > reference.calF1), 'density_gap_decreases': bool(density_improves), 'overpred_decreases': bool(row.overpred_labels < reference.overpred_labels), 'excess_decreases': bool(row.excess < reference.excess)}
        comparisons.append({'variant': row.variant, 'stratum': row.stratum, 'role': 'paired_reference' if row.variant == 'baseline_global_prior' else 'primary' if row.variant == 'middle_only_stratum_prior' else 'diagnostic', 'gates': gates, 'joint_improvement': all(gates.values()), 'delta': {key: float(getattr(row,key)-reference[key]) for key in ('calF1','raw_AUC','raw_AP','cal_pos_rate','overpred_labels','excess','TP','FP','FN')}})
    if scores.shape == (1062, 500):
        reference = strata_frame.loc[(strata_frame.variant == 'baseline_global_prior') & (strata_frame.stratum == 'all')].iloc[0]
        for key, expected in {'calF1': .21403278309062362, 'cal_pos_rate': .12043540489642184, 'overpred_labels': 472, 'excess': 32684.8, 'FP': 45434.0, 'FN': 12899.8}.items():
            if not np.isclose(reference[key], expected, rtol=0, atol=1e-8):
                raise AssertionError(f"paired historical reference {key} does not match EXP-128")
    for path, expected_hash in source_hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected_hash:
            raise AssertionError("input or calibration source changed during evaluation")
    run_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(run_dir/'evaluation-predictions.npz', **{name: np.stack(values) for name,values in predictions_by_variant.items()}, validation_ids=validation_ids, label_columns=np.asarray(labels), seeds=np.asarray(seeds))
    for frame, path in ((seed_frame, outputs["seed"]), (strata_frame, outputs["strata"]), (pd.DataFrame(threshold_rows), outputs["thresholds"]), (mean_labels, outputs["labels"]), (label_seed_frame, outputs['label_seeds']), (pd.DataFrame(grid_rows), outputs['prior_grid'])):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    def output_name(path: Path) -> str:
        try:
            return str(path.relative_to(root))
        except ValueError:
            # Tests and ad-hoc audits may intentionally write outside the
            # repository root; retain an unambiguous absolute path there.
            return str(path)

    summary = {
        "experiment_id": config.get("experiment_id", prefix.name),
        "baseline_id": input_cfg.get("baseline_id"),
        "validation_rows": int(len(target)),
        "label_count": int(len(labels)),
        "seeds": list(seeds),
        "variants": list(variants),
        "strata_counts": {name: int((strata == name).sum()) for name in DEFAULT_STRATA},
        "raw_metrics_equal_across_variants": raw_equal,
        "baseline_matches_historical_predictions": baseline_matches,
        "primary_variant": "middle_only_stratum_prior",
        "diagnostic_variant": "all_strata_prior",
        "metrics": strata_frame.to_dict(orient='records'),
        "paired_comparisons": comparisons,
        "scores_sha256": hashlib.sha256(scores_path.read_bytes()).hexdigest(),
        "protocol": {"selection_only": True, "crossfit": "two_fold_bidirectional", "shrinkage": shrinkage, "prior_grid": prior_grid.tolist(), "stratification": "training IDs only, stable ascending support tertiles", "count_aggregation": "mean_label_counts_across_seeds_then_excess", "f1_aggregation": "mean_seed_per_label_F1_skip_empty"},
        "runtime": {"cpu_only": True, "torch_imported": 'torch' in sys.modules, "training_started": False, "test_prediction_started": False, "submission_generated": False},
        "outputs": {name: output_name(path) for name, path in outputs.items() if name != "verification"},
    }
    outputs["summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    verification = {"source_scores_sha256": summary["scores_sha256"], "source_hashes": source_hashes, "baseline_matches_historical_predictions": baseline_matches, "baseline_matches_historical_thresholds": True, "primary_unchanged_strata_exact": True, "raw_metrics_equal_across_variants": raw_equal, "threshold_rows": int(len(threshold_rows)), "seed_rows": int(len(seed_rows)), "output_hashes": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in outputs.items() if path.exists() and name != "verification"}, "predictions_sha256": hashlib.sha256((run_dir/'evaluation-predictions.npz').read_bytes()).hexdigest()}
    outputs["verification"].write_text(json.dumps(verification, indent=2), encoding="utf-8")
    return outputs["summary"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not config.get("execution", {}).get("enabled", False):
        raise SystemExit("conditional calibration config is prepared_not_started; execution.enabled must be true")
    print(run_conditional_calibration(config, project_root=config_path.parent.parent))


if __name__ == "__main__":
    main()
