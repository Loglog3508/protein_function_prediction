"""Supervised ESM fine-tuning primitives and CUDA training entry point."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .data import label_columns
from .esm_embeddings import sequence_windows
from .metrics import macro_f1_skip_empty, macro_roc_auc_skip_degenerate
from .train_gpu import asymmetric_bce_loss


def validate_finetune_config(config: dict[str, Any]) -> None:
    model_name = str(config.get("model_name", "")).strip()
    if not model_name:
        raise ValueError("model_name must be a non-empty string")
    window_size = int(config.get("window_size", 1022))
    overlap = int(config.get("overlap", 128))
    if not 32 <= window_size <= 1022:
        raise ValueError("window_size must be between 32 and 1022")
    if not 0 <= overlap < window_size:
        raise ValueError("overlap must be non-negative and smaller than window_size")
    training = config.get("training", {})
    if not isinstance(training, dict):
        raise ValueError("training must be an object")
    if training.get("require_cuda", True) is not True:
        raise ValueError("training.require_cuda must be true")
    if not isinstance(training.get("save_epoch_scores", False), bool):
        raise ValueError("training.save_epoch_scores must be boolean")
    _positive_weight_limit(training.get("positive_weight_cap", 20.0))
    for key, minimum in (("epochs", 1), ("batch_size", 1), ("gradient_accumulation", 1)):
        if int(training.get(key, minimum)) < minimum:
            raise ValueError(f"training.{key} must be at least {minimum}")
    for key in ("learning_rate", "encoder_learning_rate", "weight_decay"):
        if float(training.get(key, 0.0)) < 0:
            raise ValueError(f"training.{key} must be non-negative")
    if float(training.get("learning_rate", 1e-3)) <= 0:
        raise ValueError("training.learning_rate must be positive")
    if float(training.get("encoder_learning_rate", 1e-5)) <= 0:
        raise ValueError("training.encoder_learning_rate must be positive")
    unfreeze = int(config.get("unfreeze_last_n_layers", 0))
    if unfreeze < 0:
        raise ValueError("unfreeze_last_n_layers must be non-negative")
    pooling = str(config.get("pooling", "mean_max"))
    if pooling not in {"mean", "max", "mean_max"}:
        raise ValueError("pooling must be mean, max, or mean_max")
    loss = training.get("loss", {})
    if not isinstance(loss, dict):
        raise ValueError("training.loss must be an object")
    if str(loss.get("name", "asymmetric_bce")) not in {"bce", "focal_bce", "asymmetric_bce"}:
        raise ValueError("training.loss.name must be bce, focal_bce, or asymmetric_bce")
    if float(loss.get("gamma", 0.0)) < 0:
        raise ValueError("training.loss.gamma must be non-negative")
    for key in ("gamma_positive", "gamma_negative"):
        if float(loss.get(key, 0.0)) < 0:
            raise ValueError(f"training.loss.{key} must be non-negative")
    probability_clip = float(loss.get("probability_clip", 0.0))
    if not 0 <= probability_clip < 1:
        raise ValueError("training.loss.probability_clip must be in [0, 1)")


def pool_hidden_states(hidden_states, residue_mask):
    if hidden_states.ndim != 3 or residue_mask.ndim != 2:
        raise ValueError("hidden_states must be [batch, length, hidden] and residue_mask [batch, length]")
    if hidden_states.shape[:2] != residue_mask.shape:
        raise ValueError("hidden_states and residue_mask dimensions do not match")
    mask = residue_mask.to(dtype=hidden_states.dtype).unsqueeze(-1)
    counts = mask.sum(dim=1)
    if bool((counts == 0).any()):
        raise ValueError("each sequence must contain at least one residue token")
    mean = (hidden_states * mask).sum(dim=1) / counts
    masked = hidden_states.masked_fill(~residue_mask.unsqueeze(-1), -torch_finfo(hidden_states).max)
    maximum = masked.amax(dim=1)
    return __import__("torch").cat((mean, maximum), dim=-1)


def torch_finfo(tensor):
    return __import__("torch").finfo(tensor.dtype)


def _transformer_layers(encoder) -> Sequence[Any]:
    candidates = (
        getattr(getattr(encoder, "encoder", None), "layer", None),
        getattr(encoder, "layer", None),
        getattr(getattr(encoder, "transformer", None), "layer", None),
    )
    for layers in candidates:
        if layers is not None and len(layers) > 0:
            return layers
    raise ValueError("could not locate transformer layers in encoder")


def configure_encoder_trainability(encoder, *, unfreeze_last_n_layers: int) -> int:
    if unfreeze_last_n_layers < 0:
        raise ValueError("unfreeze_last_n_layers must be non-negative")
    layers = _transformer_layers(encoder)
    if unfreeze_last_n_layers > len(layers):
        raise ValueError("unfreeze_last_n_layers exceeds encoder depth")
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    for layer in layers:
        for parameter in layer.parameters():
            parameter.requires_grad = False
    for layer in layers[-unfreeze_last_n_layers:] if unfreeze_last_n_layers else []:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    return unfreeze_last_n_layers


def build_esm_classifier(torch, encoder, *, label_count: int, config: dict[str, Any]):
    if label_count < 1:
        raise ValueError("label_count must be positive")
    hidden_size = int(getattr(getattr(encoder, "config", None), "hidden_size", 0))
    if hidden_size < 1:
        raise ValueError("encoder.config.hidden_size must be positive")
    pooling = str(config.get("pooling", "mean_max"))
    representation_size = hidden_size * 2 if pooling == "mean_max" else hidden_size
    head_hidden_size = int(config.get("head_hidden_size", 512))
    dropout = float(config.get("dropout", 0.1))
    if head_hidden_size < 1 or not 0 <= dropout < 1:
        raise ValueError("invalid classifier head configuration")

    class ESMProteinClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            self.pooling = pooling
            self.head = torch.nn.Sequential(
                torch.nn.Linear(representation_size, head_hidden_size),
                torch.nn.LayerNorm(head_hidden_size),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(head_hidden_size, label_count),
            )

        def forward(self, input_ids, attention_mask, residue_mask):
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            hidden_states = getattr(outputs, "last_hidden_state", None)
            if hidden_states is None and isinstance(outputs, dict):
                hidden_states = outputs["last_hidden_state"]
            if hidden_states is None:
                raise ValueError("encoder output lacks last_hidden_state")
            if self.pooling == "mean_max":
                representation = pool_hidden_states(hidden_states, residue_mask)
            else:
                mask = residue_mask.to(dtype=hidden_states.dtype).unsqueeze(-1)
                counts = mask.sum(dim=1).clamp_min(1)
                if self.pooling == "mean":
                    representation = (hidden_states * mask).sum(dim=1) / counts
                else:
                    representation = hidden_states.masked_fill(
                        ~residue_mask.unsqueeze(-1), -torch.finfo(hidden_states.dtype).max
                    ).amax(dim=1)
            return self.head(representation)

    return ESMProteinClassifier()


def aggregate_window_scores(
    row_indices: np.ndarray,
    window_scores: np.ndarray,
    *,
    row_count: int,
    mode: str = "mean_max",
) -> np.ndarray:
    row_indices = np.asarray(row_indices)
    window_scores = np.asarray(window_scores, dtype=np.float32)
    if row_indices.ndim != 1 or window_scores.ndim != 2:
        raise ValueError("row_indices must be one-dimensional and scores two-dimensional")
    if len(row_indices) != len(window_scores):
        raise ValueError("row_indices and window_scores must have equal rows")
    if row_count < 1 or np.any(row_indices < 0) or np.any(row_indices >= row_count):
        raise ValueError("window row indices are outside the protein row range")
    if mode not in {"mean", "max", "mean_max"}:
        raise ValueError("mode must be mean, max, or mean_max")
    result = np.zeros((row_count, window_scores.shape[1]), dtype=np.float32)
    for row in range(row_count):
        values = window_scores[row_indices == row]
        if len(values) == 0:
            raise ValueError(f"protein row {row} has no windows")
        mean = values.mean(axis=0)
        maximum = values.max(axis=0)
        if mode == "mean":
            result[row] = mean
        elif mode == "max":
            result[row] = maximum
        else:
            result[row] = 0.5 * (mean + maximum)
    return result


def build_window_records(
    sequences: Iterable[str], *, window_size: int = 1022, overlap: int = 128
) -> tuple[np.ndarray, list[str]]:
    row_indices: list[int] = []
    windows: list[str] = []
    for row, sequence in enumerate(sequences):
        current = sequence_windows(str(sequence), window_size=window_size, overlap=overlap)
        if not current:
            raise ValueError(f"sequence row {row} produced no windows")
        for _, value in current:
            row_indices.append(row)
            windows.append(value)
    return np.asarray(row_indices, dtype=np.int64), windows


def tokenize_windows(tokenizer, sequences: Sequence[str], *, max_length: int):
    encoded = tokenizer(
        list(sequences),
        padding=True,
        truncation=True,
        max_length=max_length + 2,
        return_attention_mask=True,
        return_special_tokens_mask=True,
        return_tensors="pt",
    )
    attention_mask = encoded["attention_mask"].bool()
    special_tokens_mask = encoded["special_tokens_mask"].bool()
    encoded["residue_mask"] = attention_mask & ~special_tokens_mask
    return encoded


def _seed_everything(seed: int, torch) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def _resolve(root: Path, value: str | Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else root / candidate


def _read_ids(path: Path) -> list[str]:
    import pandas as pd

    frame = pd.read_csv(path)
    if "protein_id" not in frame.columns:
        raise ValueError(f"split file lacks protein_id: {path}")
    ids = frame["protein_id"].astype(str).tolist()
    if len(ids) != len(set(ids)):
        raise ValueError(f"split file contains duplicate protein IDs: {path}")
    return ids


def _positive_weight_limit(cap: float | str) -> float:
    """Balanced removes the upper cap while retaining the historical floor of one."""
    if isinstance(cap, str) and cap == "balanced":
        return np.inf
    try:
        value = float(cap)
    except (TypeError, ValueError) as exc:
        raise ValueError("positive_weight_cap must be finite >= 1 or 'balanced'") from exc
    if isinstance(cap, bool) or not np.isfinite(value) or value < 1:
        raise ValueError("positive_weight_cap must be finite >= 1 or 'balanced'")
    return value


def _positive_weights(target: np.ndarray, cap: float | str) -> np.ndarray:
    limit = _positive_weight_limit(cap)
    positive = target.sum(axis=0).astype(np.float32)
    negative = float(len(target)) - positive
    weights = np.divide(
        negative,
        np.maximum(positive, 1.0),
        out=np.ones_like(positive),
        where=positive > 0,
    )
    return np.clip(weights, 1.0, limit).astype(np.float32)


def _load_model_and_tokenizer(config: dict[str, Any]):
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("install requirements-esm.txt before fine-tuning") from exc
    model_name = str(config["model_name"])
    revision = config.get("model_revision")
    load_kwargs = {"revision": revision} if revision else {}
    tokenizer = AutoTokenizer.from_pretrained(model_name, **load_kwargs)
    encoder = AutoModel.from_pretrained(model_name, **load_kwargs)
    return torch, tokenizer, encoder


def _make_collate(tokenizer, targets, *, max_length: int):
    def collate(batch):
        rows, sequences = zip(*batch)
        encoded = tokenize_windows(tokenizer, sequences, max_length=max_length)
        labels = None if targets is None else targets[np.asarray(rows)]
        return (
            np.asarray(rows, dtype=np.int64),
            encoded,
            None if labels is None else __import__("torch").as_tensor(labels, dtype=__import__("torch").float32),
        )

    return collate


def _move_batch(encoded, device):
    return {
        key: value.to(device)
        for key, value in encoded.items()
        if key in {"input_ids", "attention_mask", "residue_mask"}
    }


def _predict_windows(torch, model, loader, device, *, amp_enabled: bool):
    model.eval()
    row_batches: list[np.ndarray] = []
    score_batches: list[np.ndarray] = []
    with torch.inference_mode():
        for rows, encoded, _ in loader:
            batch = _move_batch(encoded, device)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits = model(**batch)
            row_batches.append(rows)
            score_batches.append(torch.sigmoid(logits).float().cpu().numpy())
    if not row_batches:
        raise ValueError("prediction loader produced no batches")
    return np.concatenate(row_batches), np.concatenate(score_batches)


def _loss_for_batch(torch, logits, target, positive_weight, loss_config):
    import torch.nn.functional as functional

    if loss_config.get("name", "asymmetric_bce") == "bce":
        return functional.binary_cross_entropy_with_logits(
            logits, target, pos_weight=positive_weight
        )
    if loss_config.get("name") == "focal_bce":
        elementwise_bce = functional.binary_cross_entropy_with_logits(
            logits, target, pos_weight=positive_weight, reduction="none"
        )
        probability = torch.sigmoid(logits)
        probability_for_target = target * probability + (1.0 - target) * (1.0 - probability)
        focusing = (1.0 - probability_for_target).pow(float(loss_config.get("gamma", 2.0)))
        return (elementwise_bce * focusing).mean()
    return asymmetric_bce_loss(
        logits,
        target,
        positive_weight,
        gamma_positive=float(loss_config.get("gamma_positive", 0.0)),
        gamma_negative=float(loss_config.get("gamma_negative", 1.0)),
        probability_clip=float(loss_config.get("probability_clip", 0.05)),
    )


def _evaluate_loader(
    torch,
    model,
    loader,
    device,
    *,
    row_count: int,
    pooling: str,
    amp_enabled: bool,
):
    row_indices, window_scores = _predict_windows(
        torch, model, loader, device, amp_enabled=amp_enabled
    )
    scores = aggregate_window_scores(
        row_indices, window_scores, row_count=row_count, mode=pooling
    )
    return scores, macro_f1_skip_empty(
        loader.dataset.targets,
        scores >= 0.5,
    ), macro_roc_auc_skip_degenerate(loader.dataset.targets, scores)


def _save_epoch_artifacts(
    output_dir: Path,
    *,
    history: list[dict],
    validation_ids: Sequence[str],
    labels: Sequence[str],
    validation_scores: np.ndarray,
) -> Path:
    epoch = int(history[-1]["epoch"])
    epoch_dir = output_dir / "epochs"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    scores_path = epoch_dir / f"epoch{epoch:02d}-validation_scores.npz"
    if scores_path.exists():
        raise FileExistsError(f"epoch scores already exist: {scores_path}")
    np.savez_compressed(
        scores_path,
        epoch=np.asarray(epoch, dtype=np.int64),
        validation_ids=np.asarray(validation_ids, dtype=np.str_),
        label_columns=np.asarray(labels, dtype=np.str_),
        validation_scores=np.asarray(validation_scores, dtype=np.float32),
    )
    history_temporary = output_dir / "history.json.tmp"
    history_temporary.write_text(
        json.dumps({"history": history}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    history_temporary.replace(output_dir / "history.json")
    return scores_path


def run_finetune(config_path: str | Path, *, project_root: str | Path | None = None) -> Path:
    import pandas as pd
    import torch

    root = Path(project_root or Path.cwd()).resolve()
    config_file = _resolve(root, config_path)
    config = json.loads(config_file.read_text(encoding="utf-8"))
    validate_finetune_config(config)
    _seed_everything(int(config.get("seed", 42)), torch)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for ESM fine-tuning")
    device = torch.device("cuda")

    data_config = config.get("data", {})
    train_path = _resolve(root, data_config.get("train_path", "data/train.csv"))
    train_frame = pd.read_csv(train_path)
    labels = label_columns(train_frame.columns)
    if train_frame["protein_id"].astype(str).duplicated().any():
        raise ValueError("training protein IDs must be unique")
    train_frame["protein_id"] = train_frame["protein_id"].astype(str)
    validation_ids = _read_ids(
        _resolve(root, config["split"]["validation_ids"])
    )
    validation_set = set(validation_ids)
    all_ids = set(train_frame["protein_id"])
    if not validation_set.issubset(all_ids):
        raise ValueError("validation IDs must all occur in training data")
    configured_train = config["split"].get("train_ids")
    training_ids = _read_ids(_resolve(root, configured_train)) if configured_train else [
        value for value in train_frame["protein_id"] if value not in validation_set
    ]
    if set(training_ids) & validation_set:
        raise ValueError("training and validation IDs overlap")
    if set(training_ids) | validation_set != all_ids:
        raise ValueError("training and validation IDs do not cover training data")
    train_part = train_frame.set_index("protein_id").loc[training_ids].reset_index()
    validation_part = train_frame.set_index("protein_id").loc[validation_ids].reset_index()
    if config.get("max_train_rows") is not None:
        limit = int(config["max_train_rows"])
        if limit < 1:
            raise ValueError("max_train_rows must be positive")
        train_part = train_part.iloc[:limit].copy()
    if config.get("max_validation_rows") is not None:
        limit = int(config["max_validation_rows"])
        if limit < 1:
            raise ValueError("max_validation_rows must be positive")
        validation_part = validation_part.iloc[:limit].copy()
    training_ids = train_part["protein_id"].astype(str).tolist()
    validation_ids = validation_part["protein_id"].astype(str).tolist()
    train_target = train_part[labels].to_numpy(dtype=np.float32)
    validation_target = validation_part[labels].to_numpy(dtype=np.float32)

    tokenizer, encoder = None, None
    loaded_torch, tokenizer, encoder = _load_model_and_tokenizer(config)
    if loaded_torch is not torch:
        raise RuntimeError("loaded torch module does not match active runtime")
    model = build_esm_classifier(
        torch,
        encoder,
        label_count=len(labels),
        config=config,
    )
    configure_encoder_trainability(
        encoder,
        unfreeze_last_n_layers=int(config.get("unfreeze_last_n_layers", 0)),
    )
    model.to(device)

    train_rows, train_windows = build_window_records(
        train_part["sequence"].astype(str),
        window_size=int(config.get("window_size", 1022)),
        overlap=int(config.get("overlap", 128)),
    )
    validation_rows, validation_windows = build_window_records(
        validation_part["sequence"].astype(str),
        window_size=int(config.get("window_size", 1022)),
        overlap=int(config.get("overlap", 128)),
    )
    class WindowDataset(torch.utils.data.Dataset):
        def __init__(self, rows, windows, targets):
            self.rows = rows
            self.windows = windows
            self.targets = targets

        def __len__(self):
            return len(self.windows)

        def __getitem__(self, index):
            row = int(self.rows[index])
            return row, self.windows[index]

    batch_size = int(config["training"].get("batch_size", 2))
    train_dataset = WindowDataset(train_rows, train_windows, train_target)
    validation_dataset = WindowDataset(
        validation_rows, validation_windows, validation_target
    )
    collate_train = _make_collate(tokenizer, train_target, max_length=int(config.get("window_size", 1022)))
    collate_validation = _make_collate(tokenizer, validation_target, max_length=int(config.get("window_size", 1022)))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_train,
        num_workers=0,
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_validation,
        num_workers=0,
    )
    loss_config = config["training"].get("loss", {})
    positive_weight = torch.as_tensor(
        _positive_weights(train_target, config["training"].get("positive_weight_cap", 20.0)),
        dtype=torch.float32,
        device=device,
    )
    head_parameters = list(model.head.parameters())
    encoder_parameters = [
        parameter for parameter in model.encoder.parameters() if parameter.requires_grad
    ]
    parameter_groups = [{
        "params": head_parameters,
        "lr": float(config["training"].get("learning_rate", 1e-3)),
    }]
    if encoder_parameters:
        parameter_groups.append({
            "params": encoder_parameters,
            "lr": float(config["training"].get("encoder_learning_rate", 1e-5)),
        })
    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=float(config["training"].get("weight_decay", 0.01)),
    )
    amp_enabled = bool(config["training"].get("amp", True))
    accumulation = int(config["training"].get("gradient_accumulation", 1))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    output_dir = _resolve(root, config["output_dir"])
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    best_score = -np.inf
    best_epoch = 0
    history = []
    checkpoint_metric = str(config["training"].get("checkpoint_metric", "macro_f1"))
    for epoch in range(1, int(config["training"].get("epochs", 1)) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for step, (_, encoded, labels_batch) in enumerate(train_loader, start=1):
            batch = _move_batch(encoded, device)
            labels_batch = labels_batch.to(device)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits = model(**batch)
                loss = _loss_for_batch(
                    torch, logits, labels_batch, positive_weight, loss_config
                ) / accumulation
            scaler.scale(loss).backward()
            if step % accumulation == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(config["training"].get("max_grad_norm", 1.0)),
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()) * accumulation)
        validation_scores, validation_f1, validation_auc = _evaluate_loader(
            torch,
            model,
            validation_loader,
            device,
            row_count=len(validation_part),
            pooling=str(config.get("pooling", "mean_max")),
            amp_enabled=amp_enabled,
        )
        score = validation_auc if checkpoint_metric == "macro_auc" else validation_f1
        history.append(
            {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "validation_macro_f1": validation_f1,
                "validation_macro_auc": validation_auc,
            }
        )
        if config["training"].get("save_epoch_scores", False):
            _save_epoch_artifacts(
                output_dir,
                history=history,
                validation_ids=validation_ids,
                labels=labels,
                validation_scores=validation_scores,
            )
        print(json.dumps(history[-1], ensure_ascii=False), flush=True)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save(model.state_dict(), output_dir / "model.pt")
            np.savez_compressed(
                output_dir / "validation_scores.npz",
                validation_ids=np.asarray(validation_ids, dtype=np.str_),
                label_columns=np.asarray(labels, dtype=np.str_),
                validation_scores=validation_scores.astype(np.float32),
            )
    summary = {
        "experiment_id": config.get("experiment_id"),
        "model_name": config["model_name"],
        "train_rows": int(len(train_part)),
        "validation_rows": int(len(validation_part)),
        "train_windows": int(len(train_windows)),
        "validation_windows": int(len(validation_windows)),
        "label_count": len(labels),
        "best_epoch": best_epoch,
        "history": history,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output_dir / "summary.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(run_finetune(args.config))


if __name__ == "__main__":
    main()
