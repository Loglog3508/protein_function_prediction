import unittest
from contextlib import redirect_stdout
from io import StringIO
from threading import Barrier
from unittest.mock import patch

import numpy as np
from sklearn.linear_model import SGDClassifier

import src.train


class LabelModelTests(unittest.TestCase):
    def test_parallel_sgd_matches_serial_scores_models_and_progress(self):
        features = np.array(
            [
                [0.0, 0.2],
                [0.1, 0.5],
                [0.2, 0.8],
                [0.4, 0.1],
                [0.5, 0.4],
                [0.6, 0.7],
                [0.8, 0.3],
                [1.0, 0.9],
            ],
            dtype=np.float32,
        )
        target = np.array(
            [
                [0, 0, 1, 1, 0],
                [0, 0, 1, 1, 1],
                [0, 0, 0, 1, 1],
                [0, 0, 1, 1, 0],
                [1, 0, 0, 1, 1],
                [1, 0, 1, 1, 0],
                [1, 0, 0, 1, 1],
                [1, 0, 1, 1, 0],
            ],
            dtype=np.uint8,
        )
        weights = np.array([1, 2, 1, 3, 1, 2, 1, 4], dtype=np.float64)
        config = {
            "type": "sgd",
            "alpha": 0.001,
            "max_iter": 100,
            "tol": 1e-4,
            "average": True,
            "n_jobs": -1,
        }
        serial_stats = {}
        parallel_stats = {}
        serial_output = StringIO()
        parallel_output = StringIO()
        with redirect_stdout(serial_output):
            serial_models, serial_scores = src.train.fit_label_models(
                features, target, features[:3], model_config=config, seed=42,
                progress_every=3, timing_stats=serial_stats,
                training_sample_weight=weights,
            )
        with redirect_stdout(parallel_output):
            parallel_models, parallel_scores = src.train.fit_label_models(
                features, target, features[:3],
                model_config={**config, "label_n_jobs": 2}, seed=42,
                progress_every=3, timing_stats=parallel_stats,
                training_sample_weight=weights,
            )

        np.testing.assert_array_equal(parallel_scores, serial_scores)
        self.assertEqual(parallel_scores.shape, (3, 5))
        self.assertEqual(parallel_scores.dtype, np.float32)
        self.assertEqual(parallel_models[1], {"constant": 0})
        self.assertEqual(parallel_models[3], {"constant": 1})
        self.assertEqual(parallel_output.getvalue(), serial_output.getvalue())
        self.assertEqual(parallel_models[0].n_jobs, 1)
        for serial_model, parallel_model in zip(serial_models, parallel_models):
            if isinstance(serial_model, dict):
                self.assertEqual(parallel_model, serial_model)
            else:
                np.testing.assert_array_equal(parallel_model.coef_, serial_model.coef_)
                np.testing.assert_array_equal(parallel_model.intercept_, serial_model.intercept_)
        for stats in (serial_stats, parallel_stats):
            self.assertGreaterEqual(stats["fit_seconds"], 0)
            self.assertGreaterEqual(stats["inference_seconds"], 0)

        discarded, discarded_scores = src.train.fit_label_models(
            features, target, features[:3],
            model_config={**config, "label_n_jobs": 2}, seed=42,
            retain_models=False, training_sample_weight=weights,
        )
        self.assertEqual(discarded, [])
        np.testing.assert_array_equal(discarded_scores, serial_scores)

    def test_parallel_sgd_fits_labels_concurrently(self):
        barrier = Barrier(2)

        class ConcurrentSGD(SGDClassifier):
            def fit(self, X, y, **kwargs):
                barrier.wait(timeout=5)
                return super().fit(X, y, **kwargs)

        features = np.array([[0.0], [0.2], [0.8], [1.0]], dtype=np.float32)
        target = np.array([[0, 1], [0, 0], [1, 0], [1, 1]], dtype=np.uint8)
        with patch.object(src.train, "SGDClassifier", ConcurrentSGD):
            _, scores = src.train.fit_label_models(
                features, target, features,
                model_config={"type": "sgd", "label_n_jobs": 2}, seed=42,
            )
        self.assertEqual(scores.shape, (4, 2))

    def test_training_sample_weights_change_fitted_scores(self):
        features = np.zeros((4, 1), dtype=np.float32)
        target = np.array([[0], [0], [0], [1]], dtype=np.uint8)
        config = {
            "type": "sgd",
            "class_weight": None,
            "alpha": 0.0001,
            "max_iter": 2000,
            "tol": 1e-6,
            "n_jobs": 1,
        }

        _, unweighted = src.train.fit_label_models(
            features, target, features[:1], model_config=config, seed=42
        )
        _, weighted = src.train.fit_label_models(
            features,
            target,
            features[:1],
            model_config=config,
            seed=42,
            training_sample_weight=np.array([1.0, 1.0, 1.0, 20.0]),
        )

        self.assertGreater(weighted[0, 0], unweighted[0, 0])

    def test_custom_positive_class_weight_is_supported(self):
        features = np.array([[0.0], [0.2], [0.8], [1.0]], dtype=np.float32)
        target = np.array([[0], [0], [1], [1]], dtype=np.uint8)
        _, scores = src.train.fit_label_models(
            features,
            target,
            features,
            model_config={
                "type": "sgd",
                "class_weight": None,
                "positive_class_weight": 2.0,
                "max_iter": 20,
                "tol": 0.001,
                "n_jobs": 1,
            },
            seed=42,
        )
        self.assertEqual(scores.shape, (4, 1))
        self.assertTrue(((scores >= 0) & (scores <= 1)).all())

    def test_sgd_parameter_averaging_is_supported(self):
        features = np.array([[0.0], [0.2], [0.8], [1.0]], dtype=np.float32)
        target = np.array([[0], [0], [1], [1]], dtype=np.uint8)

        models, _ = src.train.fit_label_models(
            features,
            target,
            features,
            model_config={
                "type": "sgd",
                "class_weight": "balanced",
                "average": True,
                "max_iter": 20,
                "tol": 0.001,
                "n_jobs": 1,
            },
            seed=42,
        )

        self.assertTrue(models[0].average)

    def test_random_forest_trains_rare_label_and_is_reproducible(self):
        fit_label_models = getattr(src.train, "fit_label_models", None)
        self.assertIsNotNone(fit_label_models, "label model trainer is missing")
        features = np.array(
            [[0.0], [0.2], [0.4], [0.6], [0.8], [1.0]], dtype=np.float32
        )
        target = np.array(
            [[1, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]], dtype=np.uint8
        )
        config = {
            "type": "random_forest",
            "n_estimators": 3,
            "max_depth": 2,
            "class_weight": "balanced",
            "n_jobs": 1,
        }

        models, scores = fit_label_models(
            features, target, features, model_config=config, seed=42
        )
        repeated_models, repeated_scores = fit_label_models(
            features, target, features, model_config=config, seed=42
        )

        self.assertFalse(isinstance(models[0], dict), "rare label was skipped")
        self.assertEqual(models[1], {"constant": 1})
        self.assertEqual(repeated_models[1], {"constant": 1})
        self.assertEqual(scores.shape, (6, 2))
        self.assertTrue(((scores >= 0) & (scores <= 1)).all())
        np.testing.assert_array_equal(scores, repeated_scores)

        discarded_models, discarded_scores = fit_label_models(
            features,
            target,
            features,
            model_config=config,
            seed=42,
            retain_models=False,
        )
        self.assertEqual(discarded_models, [])
        np.testing.assert_array_equal(scores, discarded_scores)

    def test_linear_models_accept_sparse_features_and_return_probabilities(self):
        from scipy import sparse

        features = sparse.csr_matrix(
            np.array(
                [[0.0, 1.0], [0.2, 0.8], [0.8, 0.2], [1.0, 0.0]],
                dtype=np.float32,
            )
        )
        target = np.array([[0], [0], [0], [1]], dtype=np.uint8)
        for model_type in ["sgd", "logistic_regression"]:
            with self.subTest(model_type=model_type):
                config = {
                    "type": model_type,
                    "class_weight": "balanced",
                    "max_iter": 200,
                    "tol": 1e-3,
                    "alpha": 0.0001,
                    "C": 1.0,
                    "n_jobs": 1,
                }
                models, scores = src.train.fit_label_models(
                    features,
                    target,
                    features,
                    model_config=config,
                    seed=42,
                )

                self.assertEqual(len(models), 1)
                self.assertFalse(isinstance(models[0], dict))
                self.assertEqual(scores.shape, (4, 1))
                self.assertTrue(((scores >= 0) & (scores <= 1)).all())


if __name__ == "__main__":
    unittest.main()
