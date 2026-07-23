import unittest

import numpy as np

from scripts.analyze_results import (
    PAPER_SINGLE_SEED_COMPARISONS,
    paired_statistics,
)


class AnalysisTest(unittest.TestCase):
    def test_paper_comparison_directions_favor_proposed_model(self):
        self.assertIn(
            ("12_cqt_temporal_post_bridge", "11_ds_temporal_post_bridge"),
            PAPER_SINGLE_SEED_COMPARISONS,
        )
        self.assertIn(
            ("13_ds_temporal_shuffled_post_bridge", "11_ds_temporal_post_bridge"),
            PAPER_SINGLE_SEED_COMPARISONS,
        )

    def test_clustered_bootstrap_uses_audio_as_unit(self):
        result = paired_statistics(
            np.asarray([1.0, 3.0, -1.0, 1.0]),
            bootstrap_samples=100,
            seed=42,
            clusters=["a", "a", "b", "b"],
        )
        self.assertEqual(result["paired_samples"], 4)
        self.assertEqual(result["bootstrap_clusters"], 2)
        self.assertEqual(result["bootstrap_unit"], "audio")
        self.assertAlmostEqual(result["paired_mean_difference"], 1.0)


if __name__ == "__main__":
    unittest.main()
