import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse

from src.sweep_oof_chain import (
    _build_stage2_features,
    _select_context_indices,
    _validate_score_matrix,
    _nested_stage1_scores,
)


class OofChainHelperTests(unittest.TestCase):
    def test_nested_scores_never_train_on_the_predicted_row(self):
        early_features = sparse.csr_matrix(np.arange(12, dtype=np.float32).reshape(-1, 1))
        tail_features = sparse.csr_matrix(np.arange(12, 18, dtype=np.float32).reshape(-1, 1))
        early_target = np.array([[index % 2, 1 - index % 2] for index in range(12)], dtype=np.uint8)
        tail_target = np.array([[index % 2, 1 - index % 2] for index in range(6)], dtype=np.uint8)
        tail_train = np.array([0, 2, 4])
        tail_eval = np.array([1, 3, 5])
        calls = []

        def record_fit(training_features, training_target, evaluation_features, **kwargs):
            training_ids = training_features.toarray().ravel().astype(int)
            evaluation_ids = evaluation_features.toarray().ravel().astype(int)
            self.assertFalse(set(training_ids) & set(evaluation_ids))
            self.assertFalse(set(training_ids) & {13, 15, 17})
            np.testing.assert_array_equal(kwargs['sample_weight'], np.where(training_ids >= 12, 2.0, 1.0))
            calls.append((training_ids, evaluation_ids))
            return np.repeat((evaluation_ids / 20.0)[:, None], 2, axis=1).astype(np.float32)

        with patch('src.sweep_oof_chain._fit_stage1', side_effect=record_fit):
            train_oof, eval_scores = _nested_stage1_scores(
                early_features, tail_features, early_target, tail_target,
                tail_train, tail_eval, model_config={'type': 'sgd'}, seed=42,
                tail_weight=2.0, inner_folds=2,
            )
        self.assertEqual(len(calls), 3)
        self.assertEqual(sorted(np.concatenate([item[1] for item in calls[:2]]).tolist()), list(range(12)) + [12, 14, 16])
        np.testing.assert_allclose(train_oof[:, 0], np.r_[np.arange(12), [12, 14, 16]] / 20.0)
        np.testing.assert_allclose(eval_scores[:, 0], [13 / 20.0, 15 / 20.0, 17 / 20.0])

    def test_context_selection_excludes_diagonal_and_is_deterministic(self):
        target = np.array(
            [[1, 1, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 1, 1]],
            dtype=np.uint8,
        )
        first = _select_context_indices(target, 2)
        second = _select_context_indices(target, 2)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), 2)
        self.assertTrue(np.all((first >= 0) & (first < target.shape[1])))

    def test_stage2_features_preserve_rows_and_append_oof_scores(self):
        kmer = sparse.csr_matrix(np.eye(3, dtype=np.float32))
        scores = np.linspace(0.0, 1.0, 12, dtype=np.float32).reshape(3, 4)
        result = _build_stage2_features(kmer, scores, mode="kmer+scores", scale=0.5)
        self.assertEqual(result.shape, (3, 7))
        np.testing.assert_allclose(result[:, -4:].toarray(), scores * 0.5)
        scores_only = _build_stage2_features(kmer, scores, mode="scores-only", scale=1.0)
        self.assertFalse(sparse.issparse(scores_only))
        np.testing.assert_array_equal(scores_only, scores)

    def test_score_matrix_validation_rejects_nonfinite_or_wrong_shape(self):
        with self.assertRaises(ValueError):
            _validate_score_matrix(np.ones((2, 3)), rows=3, labels=3)
        with self.assertRaises(ValueError):
            _validate_score_matrix(np.array([[0.0, np.nan]]), rows=1, labels=2)


if __name__ == "__main__":
    unittest.main()
