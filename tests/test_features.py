import unittest

import numpy as np
from scipy import sparse

import src.features
from src.features import AMINO_ACIDS, extract_composition_features


class CompositionFeatureTests(unittest.TestCase):
    def test_composition_uses_total_length_and_ignores_unknown_dimensions(self):
        features = extract_composition_features(["ACAX"])
        a_index = AMINO_ACIDS.index("A")
        c_index = AMINO_ACIDS.index("C")

        self.assertEqual(features.shape, (1, 21))
        self.assertAlmostEqual(float(features[0, a_index]), 0.5)
        self.assertAlmostEqual(float(features[0, c_index]), 0.25)
        self.assertAlmostEqual(float(features[0, :-1].sum()), 0.75)
        self.assertAlmostEqual(float(features[0, -1]), float(np.log1p(4)))

    def test_empty_sequence_produces_zero_features(self):
        features = extract_composition_features([""])

        np.testing.assert_array_equal(features, np.zeros((1, 21), dtype=np.float32))

    def test_kmer_vectorizer_builds_float32_sparse_character_features(self):
        build_vectorizer = getattr(src.features, "build_kmer_vectorizer", None)
        self.assertIsNotNone(build_vectorizer, "k-mer vectorizer builder is missing")
        vectorizer = build_vectorizer(
            k_min=3,
            k_max=5,
            min_df=1,
            max_features=4,
            sublinear_tf=True,
        )

        matrix = vectorizer.fit_transform(["ABCDE", "ABCXX"])

        self.assertTrue(sparse.issparse(matrix))
        self.assertEqual(matrix.dtype, np.float32)
        self.assertLessEqual(len(vectorizer.vocabulary_), 4)
        self.assertEqual(vectorizer.analyzer, "char")
        self.assertEqual(vectorizer.ngram_range, (3, 5))


if __name__ == "__main__":
    unittest.main()
