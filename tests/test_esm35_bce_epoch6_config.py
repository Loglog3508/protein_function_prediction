import copy
import json
from pathlib import Path

import numpy as np
import pytest

from src.train_esm_finetune import _loss_for_batch, validate_finetune_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/iteration19_esm35_bce_full_epoch6.json"
BASELINE = ROOT / "configs/iteration17_esm35_last4_full_epoch10.json"


def test_bce_config_changes_only_loss_budget_and_experiment_identity():
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    expected = copy.deepcopy(baseline)
    expected["experiment_id"] = "EXP-20261001-130-esm35-bce-last4-full-epoch6"
    expected["output_dir"] = "artifacts/runs/EXP-20261001-130-esm35-bce-last4-full-epoch6"
    expected["training"]["epochs"] = 6
    expected["training"]["loss"] = {"name": "bce"}
    actual = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert actual == expected
    assert actual["training"]["save_epoch_scores"] is True
    validate_finetune_config(actual)


def test_plain_bce_removes_asymmetric_focusing_but_preserves_positive_weights():
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[.1, -.3], [.8, -.7]], requires_grad=True)
    target = torch.tensor([[1., 0.], [0., 1.]])
    positive_weights = torch.tensor([2., 4.])
    actual = _loss_for_batch(torch, logits, target, positive_weights, {"name": "bce"})
    expected = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, pos_weight=positive_weights)
    assert float(actual.detach()) == pytest.approx(float(expected.detach()), abs=1e-12)
    actual.backward()
    assert np.isfinite(logits.grad.numpy()).all()
