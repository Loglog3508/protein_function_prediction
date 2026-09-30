import json
from types import SimpleNamespace

import numpy as np
import pytest

from src import train_esm_finetune
from src.train_esm_finetune import (
    _loss_for_batch,
    aggregate_window_scores,
    build_esm_classifier,
    build_window_records,
    configure_encoder_trainability,
    pool_hidden_states,
    validate_finetune_config,
)


def test_finetune_config_requires_cuda_and_valid_windowing():
    config = {
        "model_name": "fake/esm",
        "window_size": 1022,
        "overlap": 128,
        "training": {
            "require_cuda": True,
            "epochs": 2,
            "batch_size": 4,
            "learning_rate": 1e-3,
        },
    }

    validate_finetune_config(config)

    with pytest.raises(ValueError, match="window_size"):
        validate_finetune_config({**config, "window_size": 0})
    with pytest.raises(ValueError, match="require_cuda"):
        validate_finetune_config(
            {**config, "training": {**config["training"], "require_cuda": False}}
        )
    focal_config = {
        **config,
        "training": {**config["training"], "loss": {"name": "focal_bce", "gamma": 2.0}},
    }
    validate_finetune_config(focal_config)


def test_finetune_config_rejects_non_boolean_epoch_score_flag():
    config = {
        "model_name": "fake/esm",
        "training": {"require_cuda": True, "save_epoch_scores": "true"},
    }

    with pytest.raises(ValueError, match="save_epoch_scores"):
        validate_finetune_config(config)


def test_epoch_artifacts_preserve_all_scores_and_publish_live_history(tmp_path):
    history = [
        {
            "epoch": 1,
            "loss": 0.5,
            "validation_macro_f1": 0.2,
            "validation_macro_auc": 0.7,
        }
    ]
    scores = np.array([[0.1, 0.9], [0.8, 0.2]], dtype=np.float32)
    first_path = train_esm_finetune._save_epoch_artifacts(
        tmp_path,
        history=history,
        validation_ids=["protein1", "protein2"],
        labels=["label1", "label2"],
        validation_scores=scores,
    )
    first_bytes = first_path.read_bytes()
    history.append(
        {
            "epoch": 2,
            "loss": 0.45,
            "validation_macro_f1": 0.18,
            "validation_macro_auc": 0.69,
        }
    )
    second_path = train_esm_finetune._save_epoch_artifacts(
        tmp_path,
        history=history,
        validation_ids=["protein1", "protein2"],
        labels=["label1", "label2"],
        validation_scores=scores / 2,
    )

    assert first_path.name == "epoch01-validation_scores.npz"
    assert second_path.name == "epoch02-validation_scores.npz"
    assert first_path.read_bytes() == first_bytes
    for epoch, path, expected in ((1, first_path, scores), (2, second_path, scores / 2)):
        with np.load(path, allow_pickle=False) as saved:
            assert saved["epoch"].item() == epoch
            assert saved["validation_ids"].tolist() == ["protein1", "protein2"]
            assert saved["label_columns"].tolist() == ["label1", "label2"]
            np.testing.assert_array_equal(saved["validation_scores"], expected)
    assert json.loads((tmp_path / "history.json").read_text(encoding="utf-8")) == {
        "history": history
    }
    assert not (tmp_path / "history.json.tmp").exists()


def test_epoch_artifacts_refuse_to_overwrite_existing_scores(tmp_path):
    arguments = {
        "history": [{"epoch": 1, "loss": 0.5, "validation_macro_auc": 0.7}],
        "validation_ids": ["protein1"],
        "labels": ["label1"],
        "validation_scores": np.array([[0.5]], dtype=np.float32),
    }
    path = train_esm_finetune._save_epoch_artifacts(tmp_path, **arguments)
    original = path.read_bytes()

    with pytest.raises(FileExistsError):
        train_esm_finetune._save_epoch_artifacts(tmp_path, **arguments)

    assert path.read_bytes() == original


def test_focal_bce_loss_is_finite_and_focuses_hard_examples():
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[5.0, -5.0], [0.0, 0.0]])
    target = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    positive_weight = torch.ones(2)
    loss = _loss_for_batch(
        torch,
        logits,
        target,
        positive_weight,
        {"name": "focal_bce", "gamma": 2.0},
    )
    assert torch.isfinite(loss)
    assert float(loss) < 0.25


def test_pool_hidden_states_returns_mean_max_representation():
    torch = pytest.importorskip("torch")
    hidden = torch.tensor(
        [
            [[1.0, 4.0], [3.0, 2.0], [99.0, 99.0]],
            [[2.0, 8.0], [4.0, 6.0], [6.0, 4.0]],
        ]
    )
    residue_mask = torch.tensor([[True, True, False], [True, True, True]])

    pooled = pool_hidden_states(hidden, residue_mask)

    torch.testing.assert_close(
        pooled,
        torch.tensor(
            [
                [2.0, 3.0, 3.0, 4.0],
                [4.0, 6.0, 6.0, 8.0],
            ]
        ),
    )


def test_aggregate_window_scores_combines_windows_by_protein():
    scores = np.array(
        [
            [0.2, 0.8],
            [0.6, 0.4],
            [0.9, 0.1],
        ],
        dtype=np.float32,
    )

    aggregated = aggregate_window_scores(
        np.array([0, 0, 1]), scores, row_count=2, mode="mean_max"
    )

    np.testing.assert_allclose(
        aggregated,
        np.array([[0.5, 0.7], [0.9, 0.1]], dtype=np.float32),
    )


def test_configure_encoder_trainability_unfreezes_only_last_layers():
    torch = pytest.importorskip("torch")

    class FakeEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = SimpleNamespace(
                layer=torch.nn.ModuleList(
                    [torch.nn.Linear(2, 2) for _ in range(3)]
                )
            )

    encoder = FakeEncoder()
    selected = configure_encoder_trainability(encoder, unfreeze_last_n_layers=1)

    assert selected == 1
    assert all(
        not parameter.requires_grad
        for parameter in encoder.encoder.layer[0].parameters()
    )
    assert all(
        parameter.requires_grad
        for parameter in encoder.encoder.layer[-1].parameters()
    )


def test_esm_classifier_outputs_one_logit_vector_per_window():
    torch = pytest.importorskip("torch")

    class FakeEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=2)
            self.projection = torch.nn.Linear(3, 2)

        def forward(self, input_ids, attention_mask):
            values = input_ids.float().unsqueeze(-1).repeat(1, 1, 3)
            return SimpleNamespace(last_hidden_state=self.projection(values))

    encoder = FakeEncoder()
    model = build_esm_classifier(
        torch,
        encoder,
        label_count=4,
        config={"head_hidden_size": 8, "dropout": 0.0},
    )
    input_ids = torch.ones((2, 3), dtype=torch.long)
    attention_mask = torch.ones((2, 3), dtype=torch.long)
    residue_mask = torch.ones((2, 3), dtype=torch.bool)

    logits = model(input_ids, attention_mask, residue_mask)

    assert logits.shape == (2, 4)
    assert torch.isfinite(logits).all()

def test_build_window_records_preserves_protein_row_mapping():
    rows, windows = build_window_records(
        ["AAAAA", "CCCC"], window_size=4, overlap=1
    )

    assert rows.tolist() == [0, 0, 1]
    assert windows == ["AAAA", "AAAA", "CCCC"]
