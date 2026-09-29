import json
import numpy as np
import pandas as pd
import pytest

from src.esm_embeddings import (
    extract_embedding_shards,
    pool_residue_embeddings,
    sequence_windows,
    validate_embedding_metadata,
)


class FakeBackend:
    hidden_size = 2
    model_name = "fake/esm"

    def __init__(self):
        self.calls = 0

    def embed_window(self, sequence: str) -> np.ndarray:
        self.calls += 1
        return np.array([len(sequence), sequence.count("A")], dtype=np.float32)


class FailingBackend(FakeBackend):
    def __init__(self, fail_after):
        super().__init__()
        self.fail_after = fail_after

    def embed_window(self, sequence: str) -> np.ndarray:
        if self.calls >= self.fail_after:
            raise RuntimeError("intentional interruption")
        return super().embed_window(sequence)


def test_sequence_windows_cover_long_sequence():
    sequence = "A" * 2500

    windows = sequence_windows(sequence, window_size=1022, overlap=128)

    assert windows[0][0] == 0
    assert windows[-1][0] + len(windows[-1][1]) == len(sequence)
    assert all(len(value) <= 1022 for _, value in windows)
    covered = np.zeros(len(sequence), dtype=bool)
    for start, value in windows:
        covered[start : start + len(value)] = True
    assert covered.all()


@pytest.mark.parametrize(
    ("sequence", "window_size", "overlap", "message"),
    [
        ("", 10, 2, "empty"),
        ("AAAA", 0, 0, "window_size"),
        ("AAAA", 4, -1, "overlap"),
        ("AAAA", 4, 4, "overlap"),
    ],
)
def test_sequence_windows_reject_invalid_inputs(
    sequence, window_size, overlap, message
):
    with pytest.raises(ValueError, match=message):
        sequence_windows(sequence, window_size=window_size, overlap=overlap)


def test_pooling_concatenates_weighted_mean_and_max():
    pooled = pool_residue_embeddings(
        [np.array([1.0, 3.0]), np.array([5.0, 1.0])], [1, 3]
    )

    np.testing.assert_allclose(pooled, [4.0, 1.5, 5.0, 3.0])


def test_pooling_rejects_mismatched_dimensions():
    with pytest.raises(ValueError, match="dimension"):
        pool_residue_embeddings(
            [np.array([1.0, 2.0]), np.array([3.0])], [1, 1]
        )


def test_validate_embedding_metadata_names_mismatch():
    expected = {
        "model_name": "facebook/esm2_t12_35M_UR50D",
        "window_size": 1022,
        "dtype": "float16",
    }
    actual = {**expected, "model_name": "other/model"}

    with pytest.raises(ValueError, match="model_name"):
        validate_embedding_metadata(expected, actual)


def test_validate_embedding_metadata_accepts_expected_subset():
    expected = {"model_name": "esm", "window_size": 1022}
    actual = {**expected, "row_count": 10, "feature_width": 960}

    validate_embedding_metadata(expected, actual)


def _write_input_csvs(root):
    train_path = root / "train.csv"
    test_path = root / "test.csv"
    pd.DataFrame(
        {"protein_id": ["P1", "P2"], "sequence": ["AAAA", "RRRR"]}
    ).to_csv(train_path, index=False)
    pd.DataFrame({"protein_id": ["P3"], "sequence": ["AARR"]}).to_csv(
        test_path, index=False
    )
    return train_path, test_path


def test_extractor_reuses_valid_completed_shards(tmp_path):
    train_path, test_path = _write_input_csvs(tmp_path)
    config = {
        "train_path": str(train_path),
        "test_path": str(test_path),
        "output_dir": str(tmp_path / "embeddings"),
        "model_name": "fake/esm",
        "window_size": 4,
        "overlap": 1,
        "storage_dtype": "float32",
    }
    backend = FakeBackend()

    first = extract_embedding_shards(config, backend=backend)
    calls_after_first = backend.calls
    second = extract_embedding_shards(config, backend=backend)

    assert first == second
    assert backend.calls == calls_after_first
    assert np.load(tmp_path / "embeddings" / "train-embeddings.npz")["embeddings"].shape == (2, 4)


def test_extractor_rejects_stale_metadata(tmp_path):
    train_path, test_path = _write_input_csvs(tmp_path)
    output_dir = tmp_path / "embeddings"
    config = {
        "train_path": str(train_path),
        "test_path": str(test_path),
        "output_dir": str(output_dir),
        "model_name": "fake/esm",
        "storage_dtype": "float32",
    }
    extract_embedding_shards(config, backend=FakeBackend())
    metadata_path = output_dir / "train-embeddings.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["model_name"] = "other/model"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="model_name"):
        extract_embedding_shards(config, backend=FakeBackend())


def test_extractor_honors_smoke_max_rows(tmp_path):
    train_path, test_path = _write_input_csvs(tmp_path)
    output_dir = tmp_path / "embeddings"
    config = {
        "train_path": str(train_path),
        "test_path": str(test_path),
        "output_dir": str(output_dir),
        "model_name": "fake/esm",
        "storage_dtype": "float32",
        "max_rows": 1,
    }

    extract_embedding_shards(config, backend=FakeBackend())

    with np.load(output_dir / "train-embeddings.npz") as stored:
        assert stored["embeddings"].shape == (1, 4)


def test_extractor_records_configured_model_revision(tmp_path):
    train_path, test_path = _write_input_csvs(tmp_path)
    output_dir = tmp_path / "embeddings"
    config = {
        "train_path": str(train_path),
        "test_path": str(test_path),
        "output_dir": str(output_dir),
        "model_name": "fake/esm",
        "model_revision": "abc123",
        "storage_dtype": "float32",
    }

    extract_embedding_shards(config, backend=FakeBackend())

    metadata = json.loads(
        (output_dir / "train-embeddings.json").read_text(encoding="utf-8")
    )
    assert metadata["model_revision"] == "abc123"


def test_extractor_resumes_from_completed_row_shards(tmp_path):
    train_path, test_path = _write_input_csvs(tmp_path)
    output_dir = tmp_path / "embeddings"
    config = {
        "train_path": str(train_path),
        "test_path": str(test_path),
        "output_dir": str(output_dir),
        "model_name": "fake/esm",
        "storage_dtype": "float32",
        "shard_rows": 1,
    }
    with pytest.raises(RuntimeError, match="intentional"):
        extract_embedding_shards(config, backend=FailingBackend(fail_after=1))
    completed_part = output_dir / "train-parts" / "part-00000.npz"
    assert completed_part.exists()
    completed_mtime = completed_part.stat().st_mtime_ns

    backend = FakeBackend()
    extract_embedding_shards(config, backend=backend)

    assert completed_part.stat().st_mtime_ns == completed_mtime
    assert backend.calls == 2


def test_extractor_rejects_incomplete_row_part(tmp_path):
    train_path, test_path = _write_input_csvs(tmp_path)
    output_dir = tmp_path / "embeddings"
    config = {
        "train_path": str(train_path),
        "test_path": str(test_path),
        "output_dir": str(output_dir),
        "model_name": "fake/esm",
        "storage_dtype": "float32",
        "shard_rows": 1,
    }
    parts_dir = output_dir / "train-parts"
    parts_dir.mkdir(parents=True)
    np.savez_compressed(
        parts_dir / "part-00000.npz",
        protein_ids=np.asarray(["P1"]),
        embeddings=np.ones((1, 4), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="incomplete embedding part"):
        extract_embedding_shards(config, backend=FakeBackend())


def test_extractor_rejects_stale_row_part_metadata(tmp_path):
    train_path, test_path = _write_input_csvs(tmp_path)
    output_dir = tmp_path / "embeddings"
    config = {
        "train_path": str(train_path),
        "test_path": str(test_path),
        "output_dir": str(output_dir),
        "model_name": "fake/esm",
        "storage_dtype": "float32",
        "shard_rows": 1,
    }
    extract_embedding_shards(config, backend=FakeBackend())
    metadata_path = output_dir / "train-parts" / "part-00000.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["model_name"] = "other/model"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="model_name"):
        extract_embedding_shards(config, backend=FakeBackend())
