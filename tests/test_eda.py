import unittest

import pandas as pd

import src.eda


class EdaTests(unittest.TestCase):
    def test_summary_reports_label_sequence_and_leakage_statistics(self):
        summarize_frames = getattr(src.eda, "summarize_frames", None)
        self.assertIsNotNone(summarize_frames, "EDA summary function is missing")
        train = pd.DataFrame(
            {
                "protein_id": ["P0", "P1", "P2", "P3"],
                "sequence": ["AAAA", "CCCC", "AAAA", "ACGT"],
                "label_0": [1, 0, 1, 0],
                "label_1": [0, 1, 0, 0],
            }
        )
        test = pd.DataFrame(
            {
                "protein_id": ["T0", "T1"],
                "sequence": ["GGGG", "AAAA"],
            }
        )

        summary, label_distribution, cooccurrence, amino_acids = summarize_frames(
            train, test, ["label_0", "label_1"]
        )

        self.assertEqual(label_distribution["positive_count"].tolist(), [2, 1])
        self.assertEqual(label_distribution["positive_rate"].tolist(), [0.5, 0.25])
        self.assertEqual(summary["labels_per_sequence"]["mean"], 0.75)
        self.assertEqual(summary["leakage"]["train_duplicate_sequence_rows"], 2)
        self.assertEqual(summary["leakage"]["train_test_shared_unique_sequences"], 1)
        self.assertEqual(cooccurrence.iloc[0]["count"], 0)
        train_a = amino_acids.loc[amino_acids["amino_acid"] == "A", "train_count"].item()
        self.assertEqual(train_a, 9)


if __name__ == "__main__":
    unittest.main()
