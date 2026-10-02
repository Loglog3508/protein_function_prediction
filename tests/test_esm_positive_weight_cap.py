import numpy as np
import pytest

from src.train_esm_finetune import _positive_weights, validate_finetune_config


def test_balanced_removes_only_upper_cap_and_preserves_common_and_empty_labels():
    target = np.zeros((100, 4), dtype=np.float32)
    target[:1, 0] = 1
    target[:10, 1] = 1
    target[:75, 2] = 1
    np.testing.assert_array_equal(_positive_weights(target, "balanced"), [99, 9, 1, 1])
    np.testing.assert_array_equal(_positive_weights(target, 20), [20, 9, 1, 1])
    np.testing.assert_array_equal(_positive_weights(target, 5), [5, 5, 1, 1])
    np.testing.assert_array_equal(_positive_weights(target, 2), [2, 2, 1, 1])


@pytest.mark.parametrize("cap", [0, -1, .5, float("nan"), float("inf"), True, None, "none"])
def test_invalid_caps_fail_before_model_loading(cap):
    with pytest.raises(ValueError, match="positive_weight_cap"):
        validate_finetune_config({"model_name": "fake/esm", "training": {"positive_weight_cap": cap}})


def test_balanced_config_is_accepted_and_zero_support_weights_stay_finite():
    validate_finetune_config({"model_name": "fake/esm", "training": {"positive_weight_cap": "balanced"}})
    weights = _positive_weights(np.zeros((3, 2), dtype=np.float32), "balanced")
    np.testing.assert_array_equal(weights, [1, 1])
    assert weights.dtype == np.float32
