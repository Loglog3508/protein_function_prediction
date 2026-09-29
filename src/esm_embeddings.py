"""ESM-2 embedding extraction and deterministic feature pooling."""

import argparse
import hashlib
import json
import os
import platform
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_MODEL_NAME = "facebook/esm2_t12_35M_UR50D"
DEFAULT_WINDOW_SIZE = 1022
DEFAULT_OVERLAP = 128


def sequence_windows(
    sequence: str, *, window_size: int = DEFAULT_WINDOW_SIZE, overlap: int = DEFAULT_OVERLAP
) -> list[tuple[int, str]]:
    """Return overlapping residue windows that cover a sequence exactly."""
    sequence = str(sequence)
    if not sequence:
        raise ValueError("sequence cannot be empty")
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if overlap < 0 or overlap >= window_size:
        raise ValueError("overlap must be non-negative and smaller than window_size")
    step = window_size - overlap
    windows: list[tuple[int, str]] = []
    start = 0
    while start < len(sequence):
        end = min(start + window_size, len(sequence))
        windows.append((start, sequence[start:end]))
        if end == len(sequence):
            break
        start += step
        if start + window_size >= len(sequence):
            start = len(sequence) - window_size
    return windows


def pool_residue_embeddings(
    window_embeddings: Iterable[np.ndarray], window_lengths: Iterable[int]
) -> np.ndarray:
    """Concatenate length-weighted mean and element-wise maximum window pools."""
    embeddings = [np.asarray(value, dtype=np.float32) for value in window_embeddings]
    lengths = [int(value) for value in window_lengths]
    if not embeddings or len(embeddings) != len(lengths):
        raise ValueError("window embeddings and lengths must be non-empty and aligned")
    if any(value <= 0 for value in lengths):
        raise ValueError("window lengths must be positive")
    if any(value.ndim != 1 for value in embeddings):
        raise ValueError("window embeddings must be one-dimensional")
    dimensions = {value.shape[0] for value in embeddings}
    if len(dimensions) != 1:
        raise ValueError("window embeddings must have one consistent dimension")
    matrix = np.stack(embeddings, axis=0)
    weights = np.asarray(lengths, dtype=np.float32)
    mean = np.average(matrix, axis=0, weights=weights)
    maximum = np.max(matrix, axis=0)
    pooled = np.concatenate([mean, maximum]).astype(np.float32, copy=False)
    if not np.isfinite(pooled).all():
        raise ValueError("pooled embedding contains non-finite values")
    return pooled


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(str(sequence).encode("utf-8")).hexdigest()


def validate_embedding_metadata(expected: dict[str, Any], actual: dict[str, Any]) -> None:
    """Require every requested metadata field to match the stored shard."""
    for key, expected_value in expected.items():
        if key not in actual:
            raise ValueError(f"embedding metadata is missing {key}")
        if actual[key] != expected_value:
            raise ValueError(
                f"embedding metadata {key} mismatch: "
                f"expected {expected_value!r}, got {actual[key]!r}"
            )


class TransformersEsmBackend:
    """Lazy optional Transformers backend for ESM inference."""

    def __init__(self, config: dict[str, Any]):
        try:
            import torch
            import transformers
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "install requirements-esm.txt before extracting ESM embeddings"
            ) from exc
        if bool(config.get("require_cuda", True)) and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this ESM extraction")
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = str(config.get("model_name", DEFAULT_MODEL_NAME))
        if self.model_name != DEFAULT_MODEL_NAME:
            raise ValueError(
                f"model_name must be canonical {DEFAULT_MODEL_NAME!r}; "
                "use local_model_path for an offline checkpoint"
            )
        self.local_model_path = config.get("local_model_path")
        load_path = str(self.local_model_path or self.model_name)
        resolved_revision = None
        resolved_sha256 = None
        if self.local_model_path:
            resolved_revision, resolved_sha256 = _resolve_local_model_identity(
                config, self.local_model_path
            )
        self.token_max_length = int(
            config.get("token_max_length", DEFAULT_WINDOW_SIZE + 2)
        )
        self.use_amp = bool(config.get("amp", True)) and self.device.type == "cuda"
        self.tokenizer = AutoTokenizer.from_pretrained(
            load_path, local_files_only=bool(self.local_model_path)
        )
        self.model = AutoModel.from_pretrained(
            load_path,
            add_pooling_layer=False,
            local_files_only=bool(self.local_model_path),
        ).eval().to(self.device)
        self.hidden_size = int(self.model.config.hidden_size)
        self.model_revision = str(
            resolved_revision
            or getattr(self.model.config, "_commit_hash", None)
            or "unresolved"
        )
        self.model_sha256 = resolved_sha256
        self.library_versions = {
            "torch": str(torch.__version__),
            "transformers": str(transformers.__version__),
        }

    def embed_windows(self, sequences: list[str]) -> list[np.ndarray]:
        encoded = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.token_max_length,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with self.torch.inference_mode():
            with self.torch.autocast(
                device_type="cuda", dtype=self.torch.float16, enabled=self.use_amp
            ):
                hidden = self.model(**encoded).last_hidden_state
        results = []
        for row in range(len(sequences)):
            tokens = encoded["input_ids"][row].detach().cpu().tolist()
            special = self.tokenizer.get_special_tokens_mask(
                tokens, already_has_special_tokens=True
            )
            mask = encoded["attention_mask"][row].bool() & ~self.torch.as_tensor(
                special, device=self.device, dtype=self.torch.bool
            )
            residue_values = hidden[row][mask]
            if residue_values.numel() == 0:
                raise ValueError("tokenizer produced no residue tokens")
            results.append(
                residue_values.float().mean(dim=0).detach().cpu().numpy()
            )
        return results


def _atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def _read_config(config: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(config, dict):
        return dict(config)
    return json.loads(Path(config).read_text(encoding="utf-8"))


def _source_digest(protein_ids: Iterable[str], sequences: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for protein_id, sequence in zip(protein_ids, sequences, strict=True):
        digest.update(str(protein_id).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(sequence).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValueError(f"cannot read local model weights: {path}") from exc
    return digest.hexdigest()


def _resolve_local_model_identity(
    config: dict[str, Any], model_dir: str | Path
) -> tuple[str, str]:
    """Resolve and verify a local checkpoint's revision and weight hash."""
    model_dir = Path(model_dir)
    manifest_value = config.get("local_model_manifest")
    manifest_path = Path(manifest_value) if manifest_value else model_dir / "model_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"local model manifest is required and must be valid: {manifest_path}"
        ) from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"local model manifest must be an object: {manifest_path}")
    weights_name = str(manifest.get("weights_file", "model.safetensors"))
    weights_path = model_dir / weights_name
    actual_sha256 = _sha256_file(weights_path)
    manifest_sha256 = str(manifest.get("model_sha256", ""))
    if manifest_sha256 != actual_sha256:
        raise ValueError(
            "local model weight hash mismatch: "
            f"manifest {manifest_sha256!r}, actual {actual_sha256!r}"
        )
    expected_sha256 = config.get("model_sha256")
    if expected_sha256 is not None and str(expected_sha256) != actual_sha256:
        raise ValueError(
            "configured model_sha256 mismatch: "
            f"expected {expected_sha256!r}, actual {actual_sha256!r}"
        )
    resolved_revision = str(manifest.get("model_revision", ""))
    if not resolved_revision:
        raise ValueError(f"local model manifest is missing model_revision: {manifest_path}")
    expected_revision = config.get("model_revision")
    if expected_revision is not None and str(expected_revision) != resolved_revision:
        raise ValueError(
            "configured model_revision mismatch: "
            f"expected {expected_revision!r}, actual {resolved_revision!r}"
        )
    return resolved_revision, actual_sha256


_PART_NAME = re.compile(r"^part-(\d{5})$")


def _load_json(path: Path, *, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {description} metadata: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"invalid {description} metadata: {path}")
    return value


def _part_paths(parts_dir: Path) -> dict[int, tuple[Path, Path]]:
    """Return complete part path pairs and reject incomplete/invalid names."""
    candidates: dict[str, set[str]] = {}
    if not parts_dir.exists():
        return {}
    for path in parts_dir.iterdir():
        if not path.is_file():
            raise ValueError(f"invalid embedding part artifact: {path}")
        if path.suffix not in {".npz", ".json"}:
            raise ValueError(f"invalid embedding part artifact: {path}")
        stem = path.stem
        candidates.setdefault(stem, set()).add(path.suffix)
    parts: dict[int, tuple[Path, Path]] = {}
    for stem, suffixes in candidates.items():
        match = _PART_NAME.fullmatch(stem)
        if match is None:
            raise ValueError(f"invalid embedding part name: {stem}")
        if suffixes != {".npz", ".json"}:
            raise ValueError(f"incomplete embedding part: {parts_dir / stem}")
        index = int(match.group(1))
        parts[index] = (parts_dir / f"{stem}.npz", parts_dir / f"{stem}.json")
    return parts


def _load_part(
    paths: tuple[Path, Path],
    *,
    expected: dict[str, Any],
    protein_ids: list[str],
    start: int,
    end: int,
) -> np.ndarray:
    metadata = _load_json(paths[1], description="embedding part")
    validate_embedding_metadata(expected, metadata)
    if metadata.get("row_start") != start or metadata.get("row_end") != end:
        raise ValueError("embedding part row range mismatch")
    try:
        with np.load(paths[0], allow_pickle=False) as stored:
            keys = set(stored.files)
            if keys != {"protein_ids", "embeddings"}:
                raise ValueError("embedding part keys mismatch")
            stored_ids = stored["protein_ids"].astype(str)
            matrix = np.asarray(stored["embeddings"])
    except (OSError, ValueError, KeyError) as exc:
        if isinstance(exc, ValueError) and str(exc) == "embedding part keys mismatch":
            raise
        raise ValueError(f"invalid embedding part data: {paths[0]}") from exc
    expected_ids = np.asarray(protein_ids[start:end], dtype=np.str_)
    if not np.array_equal(stored_ids, expected_ids):
        raise ValueError("embedding part IDs mismatch")
    if matrix.ndim != 2 or matrix.shape[0] != end - start:
        raise ValueError("embedding part row count mismatch")
    if str(matrix.dtype) != str(expected["dtype"]):
        raise ValueError("embedding part dtype mismatch")
    if not np.isfinite(matrix).all():
        raise ValueError("embedding part contains non-finite values")
    feature_width = metadata.get("feature_width")
    if feature_width != int(matrix.shape[1]):
        raise ValueError("embedding part feature width mismatch")
    return matrix


def _backend_embeddings(backend, windows: list[str], batch_size: int) -> list[np.ndarray]:
    results: list[np.ndarray] = []
    if hasattr(backend, "embed_windows"):
        for start in range(0, len(windows), batch_size):
            values = backend.embed_windows(windows[start : start + batch_size])
            results.extend(np.asarray(value, dtype=np.float32) for value in values)
    else:
        results.extend(
            np.asarray(backend.embed_window(window), dtype=np.float32)
            for window in windows
        )
    return results


def _embed_sequences(
    backend,
    sequences: list[str],
    *,
    window_size: int,
    overlap: int,
    batch_size: int,
) -> np.ndarray:
    all_windows: list[str] = []
    spans: list[tuple[int, int, list[int]]] = []
    for sequence in sequences:
        values = sequence_windows(sequence, window_size=window_size, overlap=overlap)
        begin = len(all_windows)
        all_windows.extend(window for _, window in values)
        spans.append((begin, len(all_windows), [len(window) for _, window in values]))
    embedded = _backend_embeddings(backend, all_windows, batch_size)
    pooled = [
        pool_residue_embeddings(embedded[begin:end], lengths)
        for begin, end, lengths in spans
    ]
    return np.stack(pooled).astype(np.float32, copy=False)


def extract_embedding_shards(
    config: str | Path | dict[str, Any], *, backend=None
) -> Path:
    """Extract row-aligned train/test ESM matrices with validated resume."""
    import pandas as pd

    settings = _read_config(config)
    output_dir = Path(settings["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    window_size = int(settings.get("window_size", DEFAULT_WINDOW_SIZE))
    overlap = int(settings.get("overlap", DEFAULT_OVERLAP))
    batch_size = int(settings.get("batch_size", 8))
    shard_rows = int(settings.get("shard_rows", 256))
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if shard_rows <= 0:
        raise ValueError("shard_rows must be positive")
    if backend is None:
        backend = TransformersEsmBackend(settings)
    model_name = str(getattr(backend, "model_name", settings.get("model_name")))
    model_revision = str(
        getattr(backend, "model_revision", None)
        or settings.get("model_revision")
        or "test-backend"
    )
    storage_dtype = str(settings.get("storage_dtype", "float16"))
    runtime_metadata = {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }
    library_versions = getattr(backend, "library_versions", None)
    if isinstance(library_versions, dict):
        runtime_metadata["library_versions"] = dict(library_versions)
    resolved_sha256 = getattr(backend, "model_sha256", None)
    if resolved_sha256 is None and settings.get("model_sha256") is not None:
        resolved_sha256 = str(settings["model_sha256"])
    if resolved_sha256 is not None:
        runtime_metadata["model_sha256"] = str(resolved_sha256)
    outputs: list[str] = []
    for source, path_value in (
        ("train", settings["train_path"]),
        ("test", settings["test_path"]),
    ):
        frame = pd.read_csv(path_value, usecols=["protein_id", "sequence"])
        max_rows = settings.get("max_rows")
        if max_rows is not None:
            max_rows = int(max_rows)
            if max_rows <= 0:
                raise ValueError("max_rows must be positive")
            frame = frame.iloc[:max_rows].copy()
        protein_ids = frame["protein_id"].astype(str).tolist()
        sequences = frame["sequence"].astype(str).tolist()
        if len(set(protein_ids)) != len(protein_ids):
            raise ValueError(f"{source} contains duplicate protein IDs")
        expected = {
            "source": source,
            "model_name": model_name,
            "model_revision": model_revision,
            "window_size": window_size,
            "overlap": overlap,
            "shard_rows": shard_rows,
            "pooling": "length_weighted_mean_plus_max",
            "dtype": storage_dtype,
            "row_count": len(frame),
            "source_sha256": _source_digest(protein_ids, sequences),
        }
        expected.update(runtime_metadata)
        shard_path = output_dir / f"{source}-embeddings.npz"
        metadata_path = output_dir / f"{source}-embeddings.json"
        parts_dir = output_dir / f"{source}-parts"
        parts = _part_paths(parts_dir)
        part_matrices: dict[int, np.ndarray] = {}
        part_count = (len(frame) + shard_rows - 1) // shard_rows
        for index, paths in parts.items():
            if index >= part_count:
                raise ValueError(f"embedding part index out of range: {index}")
            start = index * shard_rows
            end = min(start + shard_rows, len(frame))
            part_expected = {
                **expected,
                "part_index": index,
                "row_start": start,
                "row_end": end,
                "row_count": end - start,
            }
            part_matrices[index] = _load_part(
                paths,
                expected=part_expected,
                protein_ids=protein_ids,
                start=start,
                end=end,
            )
        if shard_path.exists() or metadata_path.exists():
            if not shard_path.exists() or not metadata_path.exists():
                raise ValueError(f"incomplete {source} embedding shard")
            actual = _load_json(metadata_path, description=source)
            validate_embedding_metadata(expected, actual)
            with np.load(shard_path, allow_pickle=False) as stored:
                if set(stored.files) != {"protein_ids", "embeddings"}:
                    raise ValueError(f"stored {source} shard keys mismatch")
                stored_ids = stored["protein_ids"].astype(str)
                if not np.array_equal(stored_ids, np.asarray(protein_ids)):
                    raise ValueError(f"stored {source} shard IDs mismatch")
                matrix = stored["embeddings"]
                if matrix.ndim != 2 or matrix.shape[0] != len(frame):
                    raise ValueError(f"stored {source} shard row count mismatch")
                if str(matrix.dtype) != storage_dtype:
                    raise ValueError(f"stored {source} shard dtype mismatch")
                if not np.isfinite(matrix).all():
                    raise ValueError(f"stored {source} shard contains non-finite values")
                if actual.get("feature_width") != int(matrix.shape[1]):
                    raise ValueError(f"stored {source} shard feature width mismatch")
            outputs.append(str(shard_path))
            continue
        for index in range(part_count):
            if index in part_matrices:
                continue
            start = index * shard_rows
            end = min(start + shard_rows, len(frame))
            matrix = _embed_sequences(
                backend,
                sequences[start:end],
                window_size=window_size,
                overlap=overlap,
                batch_size=batch_size,
            ).astype(storage_dtype)
            if matrix.ndim != 2 or matrix.shape[0] != end - start:
                raise ValueError("extracted embedding part row count mismatch")
            if not np.isfinite(matrix).all():
                raise ValueError("extracted embedding part contains non-finite values")
            part_expected = {
                **expected,
                "part_index": index,
                "row_start": start,
                "row_end": end,
                "row_count": end - start,
                "feature_width": int(matrix.shape[1]),
            }
            part_path = parts_dir / f"part-{index:05d}.npz"
            part_metadata_path = parts_dir / f"part-{index:05d}.json"
            if part_path.exists() or part_metadata_path.exists():
                raise ValueError(f"embedding part appeared during extraction: {part_path}")
            _atomic_save_npz(
                part_path,
                protein_ids=np.asarray(protein_ids[start:end], dtype=np.str_),
                embeddings=matrix,
            )
            _atomic_write_json(part_metadata_path, part_expected)
            part_matrices[index] = matrix

        if len(part_matrices) != part_count:
            raise ValueError(f"incomplete {source} embedding parts")
        widths = {matrix.shape[1] for matrix in part_matrices.values()}
        if len(widths) != 1:
            raise ValueError(f"{source} embedding parts feature width mismatch")
        matrix = np.concatenate(
            [part_matrices[index] for index in range(part_count)], axis=0
        )
        metadata = {
            **expected,
            "feature_width": int(matrix.shape[1]),
        }
        _atomic_save_npz(
            shard_path,
            protein_ids=np.asarray(protein_ids, dtype=np.str_),
            embeddings=matrix,
        )
        _atomic_write_json(metadata_path, metadata)
        outputs.append(str(shard_path))
    manifest = output_dir / "manifest.json"
    _atomic_write_json(
        manifest,
        {
            "experiment_id": settings.get("experiment_id"),
            "model_name": model_name,
            "model_revision": model_revision,
            "outputs": outputs,
        },
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(extract_embedding_shards(args.config))


if __name__ == "__main__":
    main()
