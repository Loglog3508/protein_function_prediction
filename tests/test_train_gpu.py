import unittest

import numpy as np

from src.train_gpu import (
    asymmetric_bce_loss,
    sample_weights_for_ids,
    should_update_checkpoint,
    validate_gpu_training_config,
)


class TrainGpuHelperTests(unittest.TestCase):
    def test_asymmetric_loss_downweights_easy_negative_examples(self):
        import torch

        logits = torch.tensor([[6.0, -6.0], [0.0, 0.0]])
        targets = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        weights = torch.ones(2)
        plain = asymmetric_bce_loss(
            logits,
            targets,
            weights,
            gamma_positive=0.0,
            gamma_negative=0.0,
            probability_clip=0.0,
        )
        focused = asymmetric_bce_loss(
            logits,
            targets,
            weights,
            gamma_positive=0.0,
            gamma_negative=2.0,
            probability_clip=0.0,
        )
        self.assertLess(float(focused), float(plain))

    def test_asymmetric_loss_focuses_hard_negative_more_than_easy_negative(self):
        import torch

        targets = torch.zeros((1, 1))
        weights = torch.ones(1)
        easy = asymmetric_bce_loss(
            torch.tensor([[-6.0]]),
            targets,
            weights,
            gamma_negative=2.0,
        )
        hard = asymmetric_bce_loss(
            torch.tensor([[6.0]]),
            targets,
            weights,
            gamma_negative=2.0,
        )
        self.assertGreater(float(hard), 100 * float(easy))

    def test_config_validation_rejects_cpu_fallback_and_bad_loss(self):
        config = {
            "training": {
                "require_cuda": True,
                "loss": {
                    "name": "asymmetric_bce",
                    "gamma_positive": 0.0,
                    "gamma_negative": 2.0,
                    "probability_clip": 0.05,
                },
            }
        }
        self.assertIsNone(validate_gpu_training_config(config))
        config["training"]["require_cuda"] = False
        with self.assertRaises(ValueError):
            validate_gpu_training_config(config)
        config["training"]["require_cuda"] = True
        config["training"]["loss"]["probability_clip"] = 1.0
        with self.assertRaises(ValueError):
            validate_gpu_training_config(config)

    def test_config_validation_rejects_unknown_checkpoint_metric(self):
        config = {
            "training": {
                "require_cuda": True,
                "checkpoint_metric": "accuracy",
            }
        }
        with self.assertRaises(ValueError):
            validate_gpu_training_config(config)

    def test_last_epoch_checkpoint_replaces_previous_epoch(self):
        config = {
            "training": {
                "require_cuda": True,
                "checkpoint_metric": "last_epoch",
            }
        }
        self.assertIsNone(validate_gpu_training_config(config))
        self.assertTrue(
            should_update_checkpoint(
                checkpoint_metric="last_epoch", score=0.1, best_score=0.9
            )
        )

    def test_asymmetric_loss_accepts_numpy_positive_weights(self):
        import torch

        value = asymmetric_bce_loss(
            torch.zeros((2, 2)),
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            np.ones(2, dtype=np.float32),
            gamma_positive=1.0,
            gamma_negative=1.0,
            probability_clip=0.0,
        )
        self.assertTrue(np.isfinite(float(value)))

    def test_asymmetric_loss_can_return_per_label_losses(self):
        import torch

        value = asymmetric_bce_loss(
            torch.zeros((2, 3)),
            torch.zeros((2, 3)),
            np.ones(3, dtype=np.float32),
            gamma_negative=2.0,
            reduction="none",
        )
        self.assertEqual(tuple(value.shape), (2, 3))

    def test_tail_sample_weights_follow_protein_id_cutoff(self):
        weights = sample_weights_for_ids(
            ["P112733", "P112734", "P120000"],
            {"cutoff": 112734, "tail_weight": 2.5},
        )
        np.testing.assert_allclose(weights, [1.0, 2.5, 2.5])


if __name__ == "__main__":
    unittest.main()
