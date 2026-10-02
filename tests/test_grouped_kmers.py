import unittest

import numpy as np
from scipy import sparse

from src.features import build_kmer_block_vectorizer


class GroupedKmerTests(unittest.TestCase):
    def test_grouped_block_shares_equivalent_motifs_but_raw_block_stays_distinct(self):
        vectorizer = build_kmer_block_vectorizer([
            {"name": "raw", "k_min": 2, "k_max": 2, "min_df": 1},
            {"name": "groups", "k_min": 2, "k_max": 2, "min_df": 1,
             "alphabet_groups": ["AGPST", "C", "DENQ", "FWY", "HKR", "ILMV"]},
        ])
        matrix = vectorizer.fit_transform(["ACDI", "GCEL"])
        raw_size = vectorizer.block_vocabulary_sizes["raw"]
        self.assertGreater((matrix[0, :raw_size] != matrix[1, :raw_size]).nnz, 0)
        np.testing.assert_allclose(matrix[0, raw_size:].toarray(), matrix[1, raw_size:].toarray())
        self.assertTrue(sparse.isspmatrix_csr(matrix))
        self.assertEqual(matrix.dtype, np.float32)

    def test_unknown_residue_preserves_position_and_transform_keeps_training_vocabulary(self):
        vectorizer = build_kmer_block_vectorizer([
            {"name": "groups", "k_min": 2, "k_max": 2, "min_df": 1,
             "alphabet_groups": ["AGPST", "C", "DENQ", "FWY", "HKR", "ILMV"]}
        ])
        vectorizer.fit_transform(["AC", "ACD"])
        before = dict(vectorizer.vectorizers[0].vocabulary_)
        matrix = vectorizer.transform(["AXC", "GC"])
        self.assertEqual(matrix[0].nnz, 0)
        self.assertGreater(matrix[1].nnz, 0)
        self.assertEqual(before, vectorizer.vectorizers[0].vocabulary_)

    def test_alphabet_rejects_overlap_missing_residues_and_empty_groups(self):
        for groups in (["AC", "ACDEFGHIKLMNPQRSTVWY"], ["AC"], ["", "ACDEFGHIKLMNPQRSTVWY"]):
            with self.subTest(groups=groups), self.assertRaises(ValueError):
                build_kmer_block_vectorizer([
                    {"name": "groups", "k_min": 2, "k_max": 3, "min_df": 1,
                     "alphabet_groups": groups}
                ])


if __name__ == "__main__":
    unittest.main()
