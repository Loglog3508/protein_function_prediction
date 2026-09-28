"""Train a compact protein sequence CNN with PyTorch and CUDA."""

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data import label_columns
from .features import extract_sequence_statistics
from .metrics import macro_f1_skip_empty, macro_roc_auc_skip_degenerate

AMINO_ACID_TOKENS = "ACDEFGHIKLMNPQRSTVWYBOUXZ"
TOKEN_LOOKUP = np.zeros(256, dtype=np.uint8)
for _token_index, _amino_acid in enumerate(AMINO_ACID_TOKENS, start=1):
    TOKEN_LOOKUP[ord(_amino_acid)] = _token_index


def encode_sequences(sequences, max_length: int) -> np.ndarray:
    """Encode protein sequences with deterministic head-tail truncation."""
    if max_length < 2:
        raise ValueError("max_length must be at least 2")
    encoded = np.zeros((len(sequences), max_length), dtype=np.uint8)
    head_length = max_length // 2
    tail_length = max_length - head_length
    for row, sequence in enumerate(sequences):
        sequence = str(sequence)
        if len(sequence) > max_length:
            sequence = sequence[:head_length] + sequence[-tail_length:]
        values = np.frombuffer(sequence.encode("ascii", errors="ignore"), dtype=np.uint8)
        encoded[row, : len(values)] = TOKEN_LOOKUP[values]
    return encoded


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _seed_everything(seed: int, torch) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def validate_gpu_training_config(config: dict) -> None:
    """Validate CUDA and asymmetric-loss settings before allocating a model."""
    training = config.get("training", {})
    if training.get("require_cuda", True) is not True:
        raise ValueError("GPU experiments must set training.require_cuda=true")
    checkpoint_metric = str(training.get("checkpoint_metric", "macro_f1_at_0.5"))
    if checkpoint_metric not in {"macro_f1_at_0.5", "macro_auc", "last_epoch"}:
        raise ValueError(
            "training.checkpoint_metric must be macro_f1_at_0.5, macro_auc, or last_epoch"
        )
    loss = training.get("loss", {"name": "bce"})
    name = str(loss.get("name", "bce"))
    if name not in {"bce", "asymmetric_bce"}:
        raise ValueError("training.loss.name must be bce or asymmetric_bce")
    for key in ("gamma_positive", "gamma_negative"):
        value = float(loss.get(key, 0.0))
        if value < 0:
            raise ValueError(f"training.loss.{key} must be non-negative")
    probability_clip = float(loss.get("probability_clip", 0.0))
    if not 0 <= probability_clip < 1:
        raise ValueError("training.loss.probability_clip must be in [0, 1)")


def asymmetric_bce_loss(
    logits,
    targets,
    positive_weight,
    *,
    gamma_positive: float = 0.0,
    gamma_negative: float = 0.0,
    probability_clip: float = 0.0,
    reduction: str = "mean",
):
    """Compute an asymmetric BCE loss for sparse multilabel targets."""
    import torch

    if logits.shape != targets.shape:
        raise ValueError("logits and targets must have the same shape")
    if gamma_positive < 0 or gamma_negative < 0:
        raise ValueError("focusing exponents must be non-negative")
    if not 0 <= probability_clip < 1:
        raise ValueError("probability_clip must be in [0, 1)")
    if reduction not in {"mean", "none"}:
        raise ValueError("reduction must be mean or none")
    positive_weight = torch.as_tensor(
        positive_weight, dtype=logits.dtype, device=logits.device
    )
    if positive_weight.ndim == 1:
        positive_weight = positive_weight.unsqueeze(0)
    if positive_weight.shape[-1] != logits.shape[-1]:
        raise ValueError("positive_weight must match the label dimension")
    probabilities = torch.sigmoid(logits)
    positive_probability = probabilities.clamp_min(torch.finfo(logits.dtype).eps)
    negative_probability = (1.0 - probabilities).clamp_min(
        torch.finfo(logits.dtype).eps
    )
    if probability_clip:
        negative_probability = (negative_probability + probability_clip).clamp(max=1.0)
    positive_focus = (1.0 - positive_probability).pow(gamma_positive)
    negative_focus = (1.0 - negative_probability).pow(gamma_negative)
    log_likelihood = (
        targets * positive_weight * positive_focus * positive_probability.log()
        + (1.0 - targets) * negative_focus * negative_probability.log()
    )
    loss = -log_likelihood
    return loss.mean() if reduction == "mean" else loss


def sample_weights_for_ids(protein_ids, weighting: dict | None) -> np.ndarray:
    """Return per-row weights for the ordered tail distribution."""
    weights = np.ones(len(protein_ids), dtype=np.float32)
    if not weighting:
        return weights
    cutoff = int(weighting["cutoff"])
    tail_weight = float(weighting.get("tail_weight", 1.0))
    if tail_weight <= 0:
        raise ValueError("training.sample_weighting.tail_weight must be positive")
    for index, protein_id in enumerate(protein_ids):
        value = str(protein_id)
        if len(value) < 2 or value[0] != "P" or not value[1:].isdigit():
            raise ValueError(f"invalid protein ID: {value}")
        if int(value[1:]) >= cutoff:
            weights[index] = tail_weight
    return weights


def should_update_checkpoint(
    *, checkpoint_metric: str, score: float, best_score: float
) -> bool:
    """Return whether the current epoch should replace the saved checkpoint."""
    return checkpoint_metric == "last_epoch" or score > best_score


def _build_model(torch, *, label_count: int, model_config: dict):
    pooling_mode = model_config.get("pooling", "max")
    if pooling_mode not in {"max", "mean", "max_mean"}:
        raise ValueError("pooling must be one of: max, mean, max_mean")
    statistics_dimensions = int(model_config.get("statistics_dimensions", 0))
    statistics_hidden = int(model_config.get("statistics_hidden", 64))
    if statistics_dimensions < 0:
        raise ValueError("statistics_dimensions must be non-negative")
    if statistics_dimensions and statistics_hidden <= 0:
        raise ValueError("statistics_hidden must be positive when statistics are enabled")

    class ProteinCNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            embedding_dim = model_config["embedding_dim"]
            channels = model_config["channels"]
            kernels = model_config["kernels"]
            self.embedding = torch.nn.Embedding(
                len(AMINO_ACID_TOKENS) + 1,
                embedding_dim,
                padding_idx=0,
            )
            self.convolutions = torch.nn.ModuleList(
                [
                    torch.nn.Conv1d(
                        embedding_dim,
                        channels,
                        kernel_size=kernel,
                        padding=kernel // 2,
                    )
                    for kernel in kernels
                ]
            )
            self.dropout = torch.nn.Dropout(model_config.get("dropout", 0.2))
            pooling_multiplier = 2 if pooling_mode == "max_mean" else 1
            self.statistics_projection = (
                torch.nn.Sequential(
                    torch.nn.Linear(statistics_dimensions, statistics_hidden),
                    torch.nn.LayerNorm(statistics_hidden),
                    torch.nn.GELU(),
                )
                if statistics_dimensions
                else None
            )
            self.output = torch.nn.Linear(
                channels * len(kernels) * pooling_multiplier
                + (statistics_hidden if statistics_dimensions else 0),
                label_count,
            )

        def forward(self, tokens, statistics=None):
            embedded = self.embedding(tokens.long()).transpose(1, 2)
            pooled = []
            valid_mask = tokens.ne(0).unsqueeze(1)
            for layer in self.convolutions:
                features = torch.nn.functional.gelu(layer(embedded))
                if pooling_mode == "max":
                    pooled.append(torch.amax(features, dim=2))
                    continue
                expanded_mask = valid_mask.expand(-1, features.shape[1], -1)
                masked_features = features.masked_fill(~expanded_mask, 0.0)
                mean_features = masked_features.sum(dim=2) / valid_mask.sum(
                    dim=2
                ).clamp_min(1)
                if pooling_mode == "mean":
                    pooled.append(mean_features)
                else:
                    negative_infinity = torch.finfo(features.dtype).min
                    max_features = features.masked_fill(
                        ~expanded_mask, negative_infinity
                    ).amax(dim=2)
                    pooled.extend((max_features, mean_features))
            if self.statistics_projection is not None:
                if statistics is None:
                    raise ValueError("statistics are required by this model")
                pooled.append(self.statistics_projection(statistics.float()))
            return self.output(self.dropout(torch.cat(pooled, dim=1)))

    return ProteinCNN()


def _predict_scores(torch, model, loader, device, amp_enabled: bool) -> np.ndarray:
    batches = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            tokens = batch[0].to(device, non_blocking=True)
            statistics = (
                batch[1].to(device, non_blocking=True) if len(batch) > 1 else None
            )
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits = model(tokens, statistics)
            batches.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(batches).astype(np.float32, copy=False)


def run_gpu_evaluation(
    config_path: str | Path, *, project_root: str | Path | None = None
) -> Path:
    """Train a sequence CNN and save validation/test scores for comparison."""
    import torch

    root = Path(project_root) if project_root is not None else Path.cwd()
    with Path(config_path).open(encoding="utf-8") as handle:
        config = json.load(handle)
    validate_gpu_training_config(config)
    seed = int(config["seed"])
    _seed_everything(seed, torch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config["training"].get("require_cuda", True) and device.type != "cuda":
        raise RuntimeError("CUDA is required by this experiment")

    train_path = _resolve(root, config["data"]["train_path"])
    columns = pd.read_csv(train_path, nrows=0).columns
    all_labels = label_columns(columns)
    max_labels = config["data"].get("max_labels")
    labels = all_labels[:max_labels] if max_labels else all_labels
    train = pd.read_csv(
        train_path,
        usecols=["protein_id", "sequence", *labels],
        dtype={label: np.uint8 for label in labels},
    )
    split_config = config["split"]
    train_ids = pd.read_csv(_resolve(root, split_config["train_ids"]))[
        "protein_id"
    ].tolist()
    validation_ids = pd.read_csv(
        _resolve(root, split_config["validation_ids"])
    )["protein_id"].tolist()
    indexed = train.set_index("protein_id", drop=False)
    training = indexed.loc[train_ids]
    validation = indexed.loc[validation_ids]
    test = None
    if config["data"].get("test_path"):
        test = pd.read_csv(_resolve(root, config["data"]["test_path"]))

    run_dir = _resolve(root, config["output_dir"])
    metrics_prefix = _resolve(root, config["metrics_prefix"])
    summary_path = metrics_prefix.with_name(metrics_prefix.name + "-summary.json")
    if run_dir.exists() or summary_path.exists():
        raise FileExistsError("experiment outputs already exist; use a new experiment ID")
    run_dir.mkdir(parents=True)
    metrics_prefix.parent.mkdir(parents=True, exist_ok=True)

    max_length = int(config["model"]["max_length"])
    encoding_started = time.perf_counter()
    training_tokens = encode_sequences(training["sequence"].tolist(), max_length)
    validation_tokens = encode_sequences(validation["sequence"].tolist(), max_length)
    test_tokens = (
        encode_sequences(test["sequence"].tolist(), max_length)
        if test is not None
        else None
    )
    encoding_seconds = time.perf_counter() - encoding_started
    training_target = training[labels].to_numpy(dtype=np.uint8)
    validation_target = validation[labels].to_numpy(dtype=np.uint8)
    training_row_weights = sample_weights_for_ids(
        training["protein_id"].tolist(),
        config["training"].get("sample_weighting"),
    )
    use_statistics = bool(config["model"].get("use_sequence_statistics", False))
    training_statistics = (
        extract_sequence_statistics(training["sequence"].tolist())
        if use_statistics
        else None
    )
    validation_statistics = (
        extract_sequence_statistics(validation["sequence"].tolist())
        if use_statistics
        else None
    )
    test_statistics = (
        extract_sequence_statistics(test["sequence"].tolist())
        if use_statistics and test is not None
        else None
    )

    batch_size = int(config["training"]["batch_size"])
    loader_options = {
        "batch_size": batch_size,
        "num_workers": 0,
        "pin_memory": device.type == "cuda",
    }
    if use_statistics:
        training_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(training_tokens),
            torch.from_numpy(training_statistics),
            torch.from_numpy(training_target),
            torch.from_numpy(training_row_weights),
        )
    else:
        training_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(training_tokens),
            torch.from_numpy(training_target),
            torch.from_numpy(training_row_weights),
        )
    generator = torch.Generator().manual_seed(seed)
    training_loader = torch.utils.data.DataLoader(
        training_dataset, shuffle=True, generator=generator, **loader_options
    )
    validation_dataset = (
        torch.utils.data.TensorDataset(
            torch.from_numpy(validation_tokens), torch.from_numpy(validation_statistics)
        )
        if use_statistics
        else torch.utils.data.TensorDataset(torch.from_numpy(validation_tokens))
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset, shuffle=False, **loader_options
    )
    test_loader = (
        torch.utils.data.DataLoader(
            (
                torch.utils.data.TensorDataset(
                    torch.from_numpy(test_tokens), torch.from_numpy(test_statistics)
                )
                if use_statistics
                else torch.utils.data.TensorDataset(torch.from_numpy(test_tokens))
            ),
            shuffle=False,
            **loader_options,
        )
        if test_tokens is not None
        else None
    )

    model_config = dict(config["model"])
    if use_statistics:
        model_config.setdefault("statistics_dimensions", training_statistics.shape[1])
    model = _build_model(torch, label_count=len(labels), model_config=model_config)
    model.to(device)
    positive = training_target.sum(axis=0).astype(np.float32)
    negative = len(training_target) - positive
    positive_weight = np.sqrt((negative + 1.0) / (positive + 1.0))
    positive_weight = np.clip(
        positive_weight,
        1.0,
        float(config["training"].get("max_positive_weight", 8.0)),
    )
    loss_config = config["training"].get("loss", {"name": "bce"})
    loss_name = str(loss_config.get("name", "bce"))
    if loss_name == "bce":
        positive_weight_tensor = torch.from_numpy(positive_weight).to(device)
        loss_function = lambda logits, target: torch.nn.functional.binary_cross_entropy_with_logits(
            logits, target, pos_weight=positive_weight_tensor, reduction="none"
        )
    else:
        loss_function = lambda logits, target: asymmetric_bce_loss(
            logits,
            target,
            positive_weight,
            gamma_positive=float(loss_config.get("gamma_positive", 0.0)),
            gamma_negative=float(loss_config.get("gamma_negative", 0.0)),
            probability_clip=float(loss_config.get("probability_clip", 0.0)),
            reduction="none",
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"].get("weight_decay", 0.01)),
    )
    amp_enabled = bool(config["training"].get("amp", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    history = []
    checkpoint_metric = str(
        config["training"].get("checkpoint_metric", "macro_f1_at_0.5")
    )
    best_score = -1.0
    best_macro_f1 = -1.0
    best_macro_auc = -1.0
    best_state = None
    best_validation_scores = None
    training_started = time.perf_counter()
    for epoch in range(int(config["training"]["epochs"])):
        model.train()
        total_loss = 0.0
        for batch in training_loader:
            if use_statistics:
                tokens, statistics, target, row_weight = batch
                statistics = statistics.to(device, non_blocking=True)
            else:
                tokens, target, row_weight = batch
                statistics = None
            tokens = tokens.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True).float()
            row_weight = row_weight.to(device, non_blocking=True).float()
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                elementwise_loss = loss_function(model(tokens, statistics), target)
                loss = (elementwise_loss.mean(dim=1) * row_weight).sum() / row_weight.sum()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach().cpu()) * len(tokens)
        validation_scores = _predict_scores(
            torch, model, validation_loader, device, amp_enabled
        )
        validation_predictions = (validation_scores >= 0.5).astype(np.uint8)
        macro_f1 = macro_f1_skip_empty(validation_target, validation_predictions)
        macro_auc = macro_roc_auc_skip_degenerate(
            validation_target, validation_scores
        )
        epoch_result = {
            "epoch": epoch + 1,
            "training_loss": total_loss / len(training_dataset),
            "validation_macro_f1_at_0.5": macro_f1,
            "validation_macro_auc": macro_auc,
            "validation_predicted_positive_rate": float(
                validation_predictions.mean()
            ),
        }
        history.append(epoch_result)
        print(json.dumps(epoch_result), flush=True)
        score = macro_auc if checkpoint_metric in {"macro_auc", "last_epoch"} else macro_f1
        if should_update_checkpoint(
            checkpoint_metric=checkpoint_metric,
            score=score,
            best_score=best_score,
        ):
            best_score = score
            best_macro_f1 = macro_f1
            best_macro_auc = macro_auc
            best_validation_scores = validation_scores
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

    model.load_state_dict(best_state)
    test_scores = (
        _predict_scores(torch, model, test_loader, device, amp_enabled)
        if test_loader is not None
        else None
    )
    elapsed_seconds = time.perf_counter() - training_started
    scores_path = run_dir / "scores.npz"
    payload = {
        "validation_scores": best_validation_scores,
        "validation_predictions": (best_validation_scores >= 0.5).astype(np.uint8),
        "validation_ids": np.asarray(validation_ids, dtype=np.str_),
        "label_columns": np.asarray(labels, dtype=np.str_),
    }
    if test is not None:
        payload["test_scores"] = test_scores
        payload["test_ids"] = test["protein_id"].to_numpy(dtype=np.str_)
    np.savez_compressed(scores_path, **payload)
    torch.save(best_state, run_dir / "model.pt")
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    summary = {
        "experiment_id": config["experiment_id"],
        "seed": seed,
        "data_sha256": _sha256(train_path),
        "validation_split": split_config,
        "model": config["model"],
        "training": config["training"],
        "training_row_weight_mean": float(training_row_weights.mean()),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "train_rows": len(training),
        "validation_rows": len(validation),
        "label_count": len(labels),
        "checkpoint_metric": checkpoint_metric,
        "checkpoint_score": best_score,
        "macro_f1": best_macro_f1,
        "validation_macro_auc": best_macro_auc,
        "true_positive_rate": float(validation_target.mean()),
        "predicted_positive_rate": float((best_validation_scores >= 0.5).mean()),
        "encoding_seconds": encoding_seconds,
        "train_and_validation_seconds": elapsed_seconds,
        "history": history,
        "scores_sha256": _sha256(scores_path),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(run_gpu_evaluation(args.config))


if __name__ == "__main__":
    main()
