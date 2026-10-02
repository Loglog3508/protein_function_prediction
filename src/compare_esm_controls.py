"""CPU-only same-epoch AP/AUC/overprediction comparisons for saved ESM scores."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import average_precision_score, roc_auc_score

from .data import label_columns
from .diagnose_esm_epochs import (
    SEEDS,
    crossfit_predictions,
    digest,
    prediction_metrics,
    prevalence_decomposition,
)


def summarize_scores(target: np.ndarray, scores: np.ndarray, labels: list[str]) -> tuple[dict, pd.DataFrame]:
    target = np.asarray(target)
    scores = np.asarray(scores, dtype=np.float32)
    if target.ndim != 2 or scores.shape != target.shape or len(labels) != target.shape[1] or not np.isin(target, [0, 1]).all():
        raise ValueError("targets, scores and labels must be aligned binary matrices")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("scores must be finite probabilities")
    support = target.sum(axis=0)
    auc = np.asarray([roc_auc_score(target[:, index], scores[:, index]) if 0 < count < len(target) else np.nan for index, count in enumerate(support)])
    ap = np.asarray([average_precision_score(target[:, index], scores[:, index]) if count else np.nan for index, count in enumerate(support)])

    def frame_for(predictions):
        frame = pd.DataFrame({"label": labels, "true_support": support, "roc_auc": auc, "average_precision": ap})
        for column, values in prediction_metrics(target, predictions).items():
            frame[column] = values
        return frame

    def aggregate(frame):
        result = prevalence_decomposition(frame, len(target))
        result["macro_f1"] = float(frame.loc[frame.true_support > 0, "f1"].mean())
        result["overpredicted_label_fraction"] = result["overpredicted_labels"] / len(labels)
        result["predicted_to_true_ratio"] = result["predicted_positive"] / result["true_positive"] if result["true_positive"] else None
        return result

    default = frame_for(scores >= .5)
    seed_frames, seeds = [], []
    for seed in SEEDS:
        predictions, _, receipts = crossfit_predictions(target, scores, seed, 25.0)
        frame = frame_for(predictions)
        seed_frames.append(frame)
        seeds.append({"seed": seed, "global_thresholds": [entry["global_threshold"] for entry in receipts], **aggregate(frame)})
    calibrated = pd.concat(seed_frames).groupby("label", sort=False).mean(numeric_only=True).loc[labels].reset_index()
    default["mode"], calibrated["mode"] = "default05", "crossfit"
    crossfit = aggregate(calibrated)
    crossfit["macro_f1_std"] = float(np.std([entry["macro_f1"] for entry in seeds]))
    result = {"macro_auc": float(pd.Series(auc).mean()), "macro_ap": float(pd.Series(ap).mean()),
              "auc_label_count": int(np.isfinite(auc).sum()), "ap_label_count": int(np.isfinite(ap).sum()),
              "default05": aggregate(default), "crossfit": crossfit, "seeds": seeds}
    return result, pd.concat([default, calibrated], ignore_index=True)


def cap_decision_gates(baseline: dict, candidate: dict) -> dict:
    """Same-epoch cap criteria; shrinking density alone is insufficient."""
    truth = baseline["crossfit"]["true_positive_rate"]
    baseline_gap = abs(baseline["crossfit"]["predicted_positive_rate"] - truth)
    candidate_gap = abs(candidate["crossfit"]["predicted_positive_rate"] - truth)
    gates = {
        "calibrated_rate_closer_to_true": candidate_gap < baseline_gap - 1e-12,
        "calibrated_overpredicted_labels_lower": candidate["crossfit"]["overpredicted_labels"] < baseline["crossfit"]["overpredicted_labels"],
        "ap_above_same_epoch_asl": candidate["macro_ap"] > baseline["macro_ap"] + 1e-12,
        "auc_not_below_same_epoch_asl": candidate["macro_auc"] >= baseline["macro_auc"] - 1e-12,
    }
    gates["cap_joint_criterion"] = all(gates.values())
    return gates


def read_scores(path: Path, epoch: int, ids: np.ndarray, labels: list[str]) -> np.ndarray:
    with np.load(path, allow_pickle=False) as saved:
        if saved["epoch"].item() != epoch or saved["label_columns"].astype(str).tolist() != labels or not np.array_equal(saved["validation_ids"].astype(str), ids):
            raise ValueError("saved epoch, label or validation ID order mismatch")
        return saved["validation_scores"].astype(np.float32)


def evaluate_epoch(root: Path, candidate_path: Path, baseline_path: Path, reference_path: Path, epoch: int, prefix: Path) -> dict:
    paths = {name: prefix.with_name(prefix.name + suffix) for name, suffix in {
        "summary": "-summary.json", "labels": "-per-label.csv", "seeds": "-seeds.csv",
    }.items()}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("use a fresh same-epoch comparison prefix")
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if candidate["split"] != baseline["split"] or candidate["data"] != baseline["data"]:
        raise ValueError("same-epoch controls require the same fixed split and source data")
    if epoch < 1 or epoch > int(candidate["training"]["epochs"]) or epoch > int(baseline["training"]["epochs"]):
        raise ValueError("epoch lies outside the common training budget")
    train_path = root / baseline["data"]["train_path"]
    labels = label_columns(pd.read_csv(train_path, nrows=0).columns)
    ids_path = root / baseline["split"]["validation_ids"]
    ids = pd.read_csv(ids_path).protein_id.astype(str).to_numpy()
    frame = pd.read_csv(train_path, usecols=["protein_id", *labels], dtype={label: np.uint8 for label in labels})
    frame.protein_id = frame.protein_id.astype(str)
    if frame.protein_id.duplicated().any() or len(np.unique(ids)) != len(ids):
        raise ValueError("source or validation IDs are duplicated")
    target = frame.set_index("protein_id").loc[ids, labels].to_numpy(dtype=np.uint8)
    del frame
    baseline_scores = root / baseline["output_dir"] / "epochs" / f"epoch{epoch:02d}-validation_scores.npz"
    candidate_scores = root / candidate["output_dir"] / "epochs" / f"epoch{epoch:02d}-validation_scores.npz"
    hashes = {"baseline": digest(baseline_scores), "candidate": digest(candidate_scores)}
    expected = next(entry["sha256"] for entry in reference["score_provenance"] if entry["epoch"] == epoch)
    if hashes["baseline"] != expected or reference["experiment_id"] != baseline["experiment_id"]:
        raise ValueError("historical ASL scores do not match the verified reference")
    results, tables, seed_rows = {}, [], []
    for role, config, score_path in [("baseline", baseline, baseline_scores), ("candidate", candidate, candidate_scores)]:
        metrics, table = summarize_scores(target, read_scores(score_path, epoch, ids, labels), labels)
        metrics["experiment_id"] = config["experiment_id"]
        results[role] = metrics
        table["role"], table["epoch"] = role, epoch
        tables.append(table)
        seed_rows.extend({"role": role, "epoch": epoch, **entry} for entry in metrics["seeds"])
        if digest(score_path) != hashes[role]:
            raise ValueError("scores changed during CPU evaluation")
    historical = next(entry for entry in reference["history"] if entry["epoch"] == epoch)
    for actual, expected_value in [
        (results["baseline"]["macro_auc"], historical["validation_macro_auc"]),
        (results["baseline"]["crossfit"]["macro_f1"], historical["mean_crossfit_macro_f1"]),
        (results["baseline"]["crossfit"]["predicted_positive_rate"], historical["mean_crossfit_predicted_positive_rate"]),
    ]:
        if abs(actual - expected_value) > 1e-12:
            raise ValueError("ASL reference metrics were not reproduced")
    delta = {name: results["candidate"][name] - results["baseline"][name] for name in ["macro_auc", "macro_ap"]}
    delta.update({"calibrated_" + name: results["candidate"]["crossfit"][name] - results["baseline"]["crossfit"][name]
                  for name in ["predicted_positive_rate", "overpredicted_label_fraction", "positive_excess", "macro_f1"]})
    result = {"epoch": epoch, "candidate": results["candidate"], "baseline": results["baseline"], "delta": delta,
              "gates": {"ap_above_same_epoch_asl": delta["macro_ap"] > 1e-12,
                        "calibrated_predicted_rate_lower": delta["calibrated_predicted_positive_rate"] < -1e-12,
                        "calibrated_positive_excess_lower": delta["calibrated_positive_excess"] < -1e-12,
                        "calibrated_overpredicted_label_fraction_lower": delta["calibrated_overpredicted_label_fraction"] < -1e-12,
                        **cap_decision_gates(results["baseline"], results["candidate"])},
              "protocol": {"seeds": list(SEEDS), "shrinkage": 25, "global_grid_step": .02,
                           "same_epoch_only": True, "no_parameter_search": True, "f1": "mean of per-seed F1, not F1 of mean counts"},
              "provenance": {"score_sha256": hashes, "baseline_config_sha256": digest(baseline_path),
                             "candidate_config_sha256": digest(candidate_path), "validation_ids_sha256": digest(ids_path)},
              "runtime": {"cpu_only": True, "torch_imported": "torch" in sys.modules, "training_started": False, "submission_generated": False},
              "outputs": {name: path.relative_to(root).as_posix() for name, path in paths.items()}}
    prefix.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(tables).to_csv(paths["labels"], index=False)
    pd.DataFrame(seed_rows).to_csv(paths["seeds"], index=False)
    temporary = paths["summary"].with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(paths["summary"])
    return result


def watch_controls(root: Path, state_path: Path, baseline: Path, reference: Path, output_dir: Path, interval: float, analysis_tag: str = "") -> None:
    suffix = f"-{analysis_tag}" if analysis_tag else ""
    print(json.dumps({"watcher_status": "ready", "cpu_only": True, "torch_imported": "torch" in sys.modules, "analysis_tag": analysis_tag}), flush=True)
    while True:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        for task in state["tasks"]:
            config_path = root / task["config"]
            if digest(config_path) != task["config_sha256"]:
                raise ValueError("authorized configuration changed while controls were running")
            history_path = root / task["output_dir"] / "history.json"
            if not history_path.exists():
                continue
            history = json.loads(history_path.read_text(encoding="utf-8"))["history"]
            for entry in history:
                epoch = int(entry["epoch"])
                prefix = output_dir / f"{task['id']}-epoch{epoch:02d}-asl-control{suffix}"
                summary_path = prefix.with_name(prefix.name + "-summary.json")
                if summary_path.exists():
                    continue
                result = evaluate_epoch(root, config_path, baseline, reference, epoch, prefix)
                print(json.dumps({"experiment_id": task["id"], "epoch": epoch, "delta": result["delta"], "gates": result["gates"]}), flush=True)
        if all(task["status"] == "completed" for task in state["tasks"]):
            for task in state["tasks"]:
                config = json.loads((root / task["config"]).read_text(encoding="utf-8"))
                expected_epochs = list(range(1, int(config["training"]["epochs"]) + 1))
                run_dir = root / task["output_dir"]
                final = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
                if final["experiment_id"] != task["id"] or [int(entry["epoch"]) for entry in final["history"]] != expected_epochs:
                    raise ValueError("terminal run summary does not prove the full authorized epoch budget")
                for epoch in expected_epochs:
                    if not (output_dir / f"{task['id']}-epoch{epoch:02d}-asl-control{suffix}-summary.json").is_file():
                        raise ValueError("terminal run is missing a same-epoch AP/AUC/overprediction evaluation")
            print(json.dumps({"evaluation_status": "all_completed", "gpu_used": False}), flush=True)
            return
        if state.get("status") == "attention_required":
            raise RuntimeError(state.get("error", "training needs attention"))
        active = [task for task in state["tasks"] if task["status"] == "running"]
        for task in active:
            if not psutil.pid_exists(task["pid"]):
                supervisor_alive = psutil.pid_exists(state["supervisor_pid"])
                if not supervisor_alive:
                    raise RuntimeError("running state is stale; training and supervisor PIDs are gone")
        time.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--baseline-config", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--watch-state", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--output-prefix", type=Path)
    parser.add_argument("--interval", type=float, default=30)
    parser.add_argument("--analysis-tag", default="")
    arguments = parser.parse_args()
    if arguments.interval <= 0:
        parser.error("interval must be positive")
    if arguments.analysis_tag and (not arguments.analysis_tag.isascii() or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in arguments.analysis_tag)):
        parser.error("analysis-tag must contain only ASCII letters, digits, hyphens or underscores")
    root = arguments.root.resolve()
    if arguments.watch_state:
        if not arguments.output_dir:
            parser.error("watch mode requires output-dir")
        watch_controls(root, root / arguments.watch_state, root / arguments.baseline_config, root / arguments.reference, root / arguments.output_dir, arguments.interval, arguments.analysis_tag)
    else:
        if not arguments.config or not arguments.epoch or not arguments.output_prefix:
            parser.error("single epoch requires config, epoch and output-prefix")
        result = evaluate_epoch(root, root / arguments.config, root / arguments.baseline_config, root / arguments.reference, arguments.epoch, root / arguments.output_prefix)
        print(json.dumps({"epoch": arguments.epoch, "delta": result["delta"], "gates": result["gates"]}))


if __name__ == "__main__":
    main()
