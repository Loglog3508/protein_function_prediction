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
from .metrics import macro_f1_skip_empty

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
        )
    else:
        training_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(training_tokens), torch.from_numpy(training_target)
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
    loss_function = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.from_numpy(positive_weight).to(device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"].get("weight_decay", 0.01)),
    )
    amp_enabled = bool(config["training"].get("amp", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    history = []
    best_score = -1.0
    best_state = None
    best_validation_scores = None
    training_started = time.perf_counter()
    for epoch in range(int(config["training"]["epochs"])):
        model.train()
        total_loss = 0.0
        for batch in training_loader:
            if use_statistics:
                tokens, statistics, target = batch
                statistics = statistics.to(device, non_blocking=True)
            else:
                tokens, target = batch
                statistics = None
            tokens = tokens.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True).float()
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                loss = loss_function(model(tokens, statistics), target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach().cpu()) * len(tokens)
        validation_scores = _predict_scores(
            torch, model, validation_loader, device, amp_enabled
        )
        validation_predictions = (validation_scores >= 0.5).astype(np.uint8)
        score = macro_f1_skip_empty(validation_target, validation_predictions)
        epoch_result = {
            "epoch": epoch + 1,
            "training_loss": total_loss / len(training_dataset),
            "validation_macro_f1_at_0.5": score,
            "validation_predicted_positive_rate": float(
                validation_predictions.mean()
            ),
        }
        history.append(epoch_result)
        print(json.dumps(epoch_result), flush=True)
        if score > best_score:
            best_score = score
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
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "train_rows": len(training),
        "validation_rows": len(validation),
        "label_count": len(labels),
        "macro_f1": best_score,
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
