import unittest

import numpy as np

import src.train


class LabelModelTests(unittest.TestCase):
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
