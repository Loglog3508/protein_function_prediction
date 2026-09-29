"""AUC-gated, label-adaptive fusion for aligned continuous score sources."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .metrics import macro_f1_skip_empty, macro_roc_auc_skip_degenerate
from .thresholds import threshold_predictions


DEFAULT_SEEDS = (17, 31, 42, 73, 101)
DEFAULT_MINIMUM_AUC = 0.8280390455031759


def _field(payload: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in payload:
            return payload[name]
    raise ValueError(f"score payload must contain one of {names}")


def _parse_payload(payload: Any, *, name: str) -> tuple[np.ndarray | None, list[str] | None, np.ndarray]:
    if isinstance(payload, Mapping):
        ids_value = next((payload[k] for k in ("protein_ids", "validation_ids", "ids") if k in payload), None)
        labels_value = next((payload[k] for k in ("label_columns", "labels", "label_names") if k in payload), None)
        scores = np.asarray(_field(payload, "scores", "validation_scores", "score_matrix"), dtype=np.float32)
        ids = None if ids_value is None else np.asarray(ids_value, dtype=str)
        labels = None if labels_value is None else [str(value) for value in labels_value]
    elif isinstance(payload, tuple) and len(payload) == 3:
        ids = np.asarray(payload[0], dtype=str)
        labels = [str(value) for value in payload[1]]
        scores = np.asarray(payload[2], dtype=np.float32)
    else:
        ids = labels = None
        scores = np.asarray(payload, dtype=np.float32)
    if scores.ndim != 2 or not np.isfinite(scores).all():
        raise ValueError(f"{name} scores must be a finite two-dimensional array")
    if ((scores < 0) | (scores > 1)).any():
        raise ValueError(f"{name} scores must be between 0 and 1")
    if ids is not None:
        if len(ids) != scores.shape[0] or len(set(ids.tolist())) != len(ids):
            raise ValueError(f"{name} protein IDs must be unique and match score rows")
    if labels is not None:
        if len(labels) != scores.shape[1] or len(set(labels)) != len(labels):
            raise ValueError(f"{name} labels must be unique and match score columns")
    return ids, labels, scores


def align_score_sources(reference: Any, sources: Mapping[str, Any] | Any) -> dict[str, np.ndarray]:
    """Align score matrices to a reference's exact protein and label order.

    Payloads may use ``protein_ids``/``validation_ids``, ``label_columns`` and
    ``scores``/``validation_scores``. Every source must contain exactly the
    same ID and label sets as the reference; rows and columns are reordered by
    name, never by position.
    """
    ref_ids, ref_labels, _ = _parse_payload(reference, name="reference")
    if ref_ids is None or ref_labels is None:
        raise ValueError("reference must include protein IDs and labels")
    if isinstance(sources, Mapping) and not any(
        key in sources for key in ("scores", "validation_scores", "score_matrix")
    ):
        source_map = dict(sources)
    else:
        source_map = {"source": sources}
    aligned: dict[str, np.ndarray] = {}
    for source_name, payload in source_map.items():
        ids, labels, scores = _parse_payload(payload, name=str(source_name))
        if ids is None or labels is None:
            raise ValueError(f"{source_name} must include protein IDs and labels")
        if set(ids.tolist()) != set(ref_ids.tolist()) or set(labels) != set(ref_labels):
            raise ValueError("score source IDs or labels do not match reference")
        source_rows = {value: index for index, value in enumerate(ids.tolist())}
        source_cols = {value: index for index, value in enumerate(labels)}
        aligned[source_name] = scores[np.ix_(
            [source_rows[value] for value in ref_ids.tolist()],
            [source_cols[value] for value in ref_labels],
        )].astype(np.float32, copy=False)
    return aligned


def _target_array(target: Any) -> tuple[np.ndarray, np.ndarray | None, list[str] | None]:
    if isinstance(target, Mapping):
        ids_value = next((target[k] for k in ("protein_ids", "validation_ids", "ids") if k in target), None)
        labels_value = next((target[k] for k in ("label_columns", "labels", "label_names") if k in target), None)
        values = target.get("target", target.get("y_true"))
        if values is None:
            raise ValueError("target payload must contain target or y_true")
        array = np.asarray(values, dtype=np.uint8)
        ids = None if ids_value is None else np.asarray(ids_value, dtype=str)
        labels = None if labels_value is None else [str(value) for value in labels_value]
    else:
        array = np.asarray(target, dtype=np.uint8)
        ids = labels = None
    if array.ndim != 2 or not np.isin(array, [0, 1]).all():
        raise ValueError("target must be a two-dimensional binary matrix")
    if ids is not None and (len(ids) != len(array) or len(set(ids.tolist())) != len(ids)):
        raise ValueError("target protein IDs must be unique and match target rows")
    if labels is not None and (len(labels) != array.shape[1] or len(set(labels)) != len(labels)):
        raise ValueError("target labels must be unique and match target columns")
    return array, ids, labels


def _threshold_candidates(config: Mapping[str, Any]) -> np.ndarray:
    values = config.get("thresholds", config.get("threshold_candidates"))
    if isinstance(values, Mapping):
        values = values.get("candidates")
    if values is None:
        values = np.arange(0.05, 1.0, 0.05)
    candidates = np.unique(np.asarray(values, dtype=np.float32))
    if candidates.ndim != 1 or not len(candidates) or ((candidates < 0) | (candidates > 1)).any():
        raise ValueError("threshold candidates must be a non-empty array between 0 and 1")
    return candidates


def _candidate_scores(sources: Mapping[str, np.ndarray], config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    if not sources:
        raise ValueError("at least one score source is required")
    names = list(sources)
    result = {name: np.asarray(scores, dtype=np.float32) for name, scores in sources.items()}
    for left, right in itertools.combinations(names, 2):
        result[f"{left}+{right}"] = ((result[left] + result[right]) / 2.0).astype(np.float32)
    new_model = config.get("new_model", "esm")
    baseline = config.get("zero_weight_source")
    if baseline is None:
        baseline = next((name for name in names if name != new_model), names[0])
    if baseline not in result:
        raise ValueError(f"zero_weight_source {baseline!r} is not present")
    if new_model in names:
        result[f"{new_model}_zero_weight"] = result[baseline].copy()
    rollback_source = config.get("rollback_source", baseline)
    if rollback_source not in result:
        raise ValueError(f"rollback_source {rollback_source!r} is not present")
    result["rollback"] = result[rollback_source].copy()
    return result


def _fit_thresholds(y_true: np.ndarray, scores: np.ndarray, candidates: np.ndarray) -> tuple[np.ndarray, float]:
    thresholds = np.full(scores.shape[1], 0.5, dtype=np.float32)
    for label_index in range(scores.shape[1]):
        target = y_true[:, label_index]
        best = (-1.0, 0.5)
        for candidate in candidates:
            prediction = scores[:, label_index] >= float(candidate)
            tp = int((target & prediction).sum())
            fp = int(((target == 0) & prediction).sum())
            fn = int((target & ~prediction).sum())
            denominator = 2 * tp + fp + fn
            f1 = 2 * tp / denominator if denominator else 0.0
            value = (f1, -abs(float(candidate) - 0.5))
            if value > (best[0], -abs(best[1] - 0.5)):
                best = (f1, float(candidate))
        thresholds[label_index] = best[1]
    global_threshold = float(np.median(thresholds))
    return thresholds, global_threshold


def _fit_group_thresholds(y_true: np.ndarray, scores: np.ndarray, candidates: np.ndarray, groups: np.ndarray) -> np.ndarray:
    thresholds = np.full(scores.shape[1], 0.5, dtype=np.float32)
    for group in np.unique(groups):
        labels = np.flatnonzero(groups == group)
        if not y_true[:, labels].any():
            continue
        best = (-1.0, 0.5)
        for candidate in candidates:
            predictions = scores[:, labels] >= float(candidate)
            value = macro_f1_skip_empty(y_true[:, labels], predictions)
            tie = -abs(float(candidate) - 0.5)
            if (value, tie) > (best[0], -abs(best[1] - 0.5)):
                best = (value, float(candidate))
        thresholds[labels] = best[1]
    return thresholds


def _safe_f1(y_true: np.ndarray, predictions: np.ndarray) -> float:
    return macro_f1_skip_empty(y_true, predictions) if y_true.any() else 0.0


def _adaptive_policy(
    fit_target: np.ndarray,
    candidates: Mapping[str, np.ndarray],
    selection_index: np.ndarray,
    threshold_values: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Choose one source per support group using selection rows only."""
    selected = np.empty(fit_target.shape[1], dtype=object)
    thresholds = np.full(fit_target.shape[1], 0.5, dtype=np.float32)
    for group in np.unique(groups):
        labels = np.flatnonzero(groups == group)
        if not fit_target[:, labels].any():
            selected[labels] = next(iter(candidates))
            continue
        best = (-1.0, "", 0.5)
        for name, scores in candidates.items():
            fit_scores = scores[selection_index][:, labels]
            for threshold in threshold_values:
                f1 = _safe_f1(fit_target[:, labels], fit_scores >= float(threshold))
                key = (f1, -abs(float(threshold) - 0.5))
                if key > (best[0], -abs(best[2] - 0.5)):
                    best = (f1, name, float(threshold))
        selected[labels] = best[1]
        thresholds[labels] = best[2]
    return thresholds, selected.astype(str).tolist()


def _support_groups(y_true: np.ndarray, config: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    support = y_true.sum(axis=0).astype(int)
    strata = config.get("support_strata")
    if strata:
        groups = np.empty(y_true.shape[1], dtype=object)
        for index, count in enumerate(support):
            name = "other"
            for stratum in strata:
                minimum = int(stratum.get("min_support", 0))
                maximum = int(stratum.get("max_support", 10**9))
                if minimum <= count <= maximum:
                    name = str(stratum["name"])
                    break
            groups[index] = name
    else:
        minimum = int(config.get("independent_min_support", 100))
        groups = np.where(support >= minimum, "high", "low").astype(object)
    return support, groups


def crossfit_label_policies(
    target: Any,
    sources: Mapping[str, Any],
    seeds: list[int] | tuple[int, ...] | None,
    config: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate source policies with two-way threshold cross-fitting.

    Each seed creates two roles: thresholds and support policies are fitted on
    one half and scored on the other, then the roles are exchanged. The policy
    table records only fit-fold decisions, while seed rows contain held-out
    Macro F1 and continuous Macro AUC.
    """
    config = dict(config or {})
    y_true, target_ids, target_labels = _target_array(target)
    arrays: dict[str, np.ndarray] = {}
    for name, payload in sources.items():
        ids, labels, values = _parse_payload(payload, name=str(name))
        if target_ids is not None and ids is not None:
            if set(ids.tolist()) != set(target_ids.tolist()):
                raise ValueError("score source IDs or labels do not match target")
            positions = {value: index for index, value in enumerate(ids.tolist())}
            values = values[[positions[value] for value in target_ids.tolist()]]
        if target_labels is not None and labels is not None:
            if set(labels) != set(target_labels):
                raise ValueError("score source IDs or labels do not match target")
            positions = {value: index for index, value in enumerate(labels)}
            values = values[:, [positions[value] for value in target_labels]]
        if values.shape != y_true.shape:
            raise ValueError("score source shape does not match target")
        arrays[name] = values
    candidates = _candidate_scores(arrays, config)
    threshold_values = _threshold_candidates(config)
    independent_enabled = bool(config.get("independent_label_policies", False))
    if independent_enabled:
        if "independent_min_support" not in config:
            raise ValueError("independent label policies require independent_min_support")
        if "stability_tolerance" not in config:
            raise ValueError("independent label policies require stability_tolerance")
        if float(config["stability_tolerance"]) < 0:
            raise ValueError("stability_tolerance must be non-negative")
    seed_values = DEFAULT_SEEDS if seeds is None else tuple(int(seed) for seed in seeds)
    if not seed_values:
        raise ValueError("at least one threshold seed is required")
    result_rows: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []
    for seed in seed_values:
        rng = np.random.default_rng(seed)
        first, second = np.array_split(rng.permutation(len(y_true)), 2)
        for fold, (selection_index, evaluation_index) in enumerate(((first, second), (second, first))):
            fit_target = y_true[selection_index]
            _, groups = _support_groups(fit_target, config)
            independent_min = int(config.get("independent_min_support", 100))
            allow_independent = independent_enabled
            if allow_independent:
                independent = fit_target.sum(axis=0) >= independent_min
                groups = groups.astype(object)
                groups[independent] = np.asarray([f"label_{i}" for i in np.flatnonzero(independent)], dtype=object)
            for candidate_name, candidate_scores in candidates.items():
                fit_scores = candidate_scores[selection_index]
                thresholds = _fit_group_thresholds(fit_target, fit_scores, threshold_values, groups)
                predictions = threshold_predictions(candidate_scores[evaluation_index], thresholds)
                result_rows.append(
                    {
                        "candidate": candidate_name,
                        "seed": int(seed),
                        "selection_fold": 1 - fold,
                        "evaluation_fold": fold,
                        "macro_f1": _safe_f1(y_true[evaluation_index], predictions),
                        "continuous_macro_auc": macro_roc_auc_skip_degenerate(y_true, candidate_scores),
                        "predicted_positive_rate": float(predictions.mean()),
                    }
                )
                for label_index, group in enumerate(groups):
                    support = int(fit_target[:, label_index].sum())
                    policy_rows.append(
                        {
                            "candidate": candidate_name,
                            "seed": int(seed),
                            "selection_fold": 1 - fold,
                            "label_index": label_index,
                            "support": support,
                            "support_stratum": str(group),
                            "policy_scope": "independent" if str(group).startswith("label_") else "shared",
                            "source": candidate_name,
                            "threshold": float(thresholds[label_index]),
                        }
                    )
            adaptive_thresholds, chosen = _adaptive_policy(
                fit_target, candidates, selection_index, threshold_values, groups
            )
            adaptive_scores = np.column_stack(
                [candidates[name][:, label_index] for label_index, name in enumerate(chosen)]
            )
            predictions = threshold_predictions(adaptive_scores[evaluation_index], adaptive_thresholds)
            result_rows.append(
                {
                    "candidate": "adaptive",
                    "seed": int(seed),
                    "selection_fold": 1 - fold,
                    "evaluation_fold": fold,
                    "macro_f1": _safe_f1(y_true[evaluation_index], predictions),
                    "continuous_macro_auc": macro_roc_auc_skip_degenerate(y_true, adaptive_scores),
                    "predicted_positive_rate": float(predictions.mean()),
                }
            )
            for label_index, group in enumerate(groups):
                policy_rows.append(
                    {
                        "candidate": "adaptive",
                        "seed": int(seed),
                        "selection_fold": 1 - fold,
                        "label_index": label_index,
                        "support": int(fit_target[:, label_index].sum()),
                        "support_stratum": str(group),
                        "policy_scope": "independent" if str(group).startswith("label_") else "shared",
                        "source": chosen[label_index],
                        "threshold": float(adaptive_thresholds[label_index]),
                    }
                )
    return pd.DataFrame(result_rows), pd.DataFrame(policy_rows)


def rank_policy_results(results: pd.DataFrame, *, minimum_auc: float = DEFAULT_MINIMUM_AUC) -> pd.DataFrame:
    """Aggregate seed rows and rank candidates under a hard AUC gate."""
    if results.empty:
        raise ValueError("policy results must be non-empty")
    if minimum_auc < 0 or minimum_auc > 1:
        raise ValueError("minimum_auc must be between 0 and 1")
    if "mean_crossfit_macro_f1" in results.columns:
        ranked = results.copy()
    else:
        grouped = results.groupby("candidate", sort=False)
        aggregations = dict(
            mean_crossfit_macro_f1=("macro_f1", "mean"),
            std_crossfit_macro_f1=("macro_f1", "std"),
            min_crossfit_macro_f1=("macro_f1", "min"),
            continuous_macro_auc=("continuous_macro_auc", "mean"),
            seed_count=("seed", "nunique"),
        )
        if "predicted_positive_rate" in results.columns:
            aggregations["mean_predicted_positive_rate"] = ("predicted_positive_rate", "mean")
        ranked = grouped.agg(**aggregations).reset_index()
        ranked["std_crossfit_macro_f1"] = ranked["std_crossfit_macro_f1"].fillna(0.0)
    if "candidate" not in ranked.columns:
        ranked = ranked.reset_index()
    ranked["eligible"] = ranked["continuous_macro_auc"] >= float(minimum_auc)
    ranked["auc_eligible"] = ranked["eligible"]
    ranked = ranked.sort_values(
        ["eligible", "mean_crossfit_macro_f1", "min_crossfit_macro_f1", "std_crossfit_macro_f1", "continuous_macro_auc"],
        ascending=[False, False, False, True, False],
        ignore_index=True,
    )
    return ranked


def assert_fusion_outputs_absent(paths: list[str | Path]) -> None:
    if any(Path(path).exists() for path in paths):
        raise FileExistsError("experiment outputs already exist; use a new experiment ID")


def validate_fusion_config(config: Mapping[str, Any]) -> None:
    if not bool(config.get("diagnostic_only", False)) and float(config.get("minimum_auc", DEFAULT_MINIMUM_AUC)) < DEFAULT_MINIMUM_AUC:
        raise ValueError("production minimum_auc must preserve the rollback gate")
    seeds = tuple(int(seed) for seed in config.get("seeds", DEFAULT_SEEDS))
    if seeds != DEFAULT_SEEDS:
        raise ValueError("production fusion must use threshold seeds 17, 31, 42, 73, 101")
    if not config.get("sources"):
        raise ValueError("fusion config must define sources")


def run_fusion(config_path: str | Path, *, project_root: str | Path | None = None) -> Path:
    root = Path(project_root) if project_root is not None else Path.cwd()
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    validate_fusion_config(config)
    prefix = Path(config["metrics_prefix"])
    if not prefix.is_absolute():
        prefix = root / prefix
    run_dir = Path(config["output_dir"])
    if not run_dir.is_absolute():
        run_dir = root / run_dir
    paths = [run_dir]
    paths.extend(
        prefix.with_name(prefix.name + suffix)
        for suffix in (
            "-seed-results.csv",
            "-leaderboard.csv",
            "-thresholds.csv",
            "-label-policies.csv",
            "-summary.json",
        )
    )
    assert_fusion_outputs_absent(paths)
    source_payloads = {}
    reference = None
    for name, source_path in config["sources"].items():
        path = Path(source_path)
        if not path.is_absolute():
            path = root / path
        with np.load(path, allow_pickle=False) as saved:
            payload = {key: saved[key] for key in saved.files}
        if reference is None:
            reference = payload
        source_payloads[name] = payload
    aligned = align_score_sources(reference, source_payloads)
    target_path = Path(config["target_path"])
    if not target_path.is_absolute():
        target_path = root / target_path
    with np.load(target_path, allow_pickle=False) as saved:
        target = {key: saved[key] for key in saved.files}
    seed_results, policies = crossfit_label_policies(target, aligned, config.get("seeds", DEFAULT_SEEDS), config)
    leaderboard = rank_policy_results(seed_results, minimum_auc=float(config.get("minimum_auc", DEFAULT_MINIMUM_AUC)))
    run_dir.mkdir(parents=True, exist_ok=True)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    seed_results.to_csv(prefix.with_name(prefix.name + "-seed-results.csv"), index=False)
    leaderboard.to_csv(prefix.with_name(prefix.name + "-leaderboard.csv"), index=False)
    policies.to_csv(prefix.with_name(prefix.name + "-label-policies.csv"), index=False)
    pd.DataFrame().to_csv(prefix.with_name(prefix.name + "-thresholds.csv"), index=False)
    summary = {"experiment_id": config["experiment_id"], "selected": leaderboard.iloc[0].to_dict(), "seed_count": len(config.get("seeds", DEFAULT_SEEDS))}
    summary_path = prefix.with_name(prefix.name + "-summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(run_fusion(args.config))


if __name__ == "__main__":
    main()
