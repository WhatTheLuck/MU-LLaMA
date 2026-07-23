import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.analyze_results import (
    PAPER_SINGLE_SEED_COMPARISONS,
    add_holm_adjustment,
    paired_statistics,
    write_paper_reproducibility_outputs,
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
        self.assertIn("paired_wilcoxon_p_one_sided", result)

    def test_holm_adjustment_is_monotone_and_bounded(self):
        rows = [{"p": 0.01}, {"p": 0.04}, {"p": 0.03}]
        add_holm_adjustment(rows, "p", "adjusted")
        self.assertAlmostEqual(rows[0]["adjusted"], 0.03)
        self.assertAlmostEqual(rows[2]["adjusted"], 0.06)
        self.assertAlmostEqual(rows[1]["adjusted"], 0.06)

    def test_paper_reproducibility_outputs_bind_run_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = {
                "model": {
                    "path": Path("/run/model"),
                    "config": {"training": {"seed": 42}},
                    "completed": {"status": "completed", "config_fingerprint": "abc"},
                    "environment": {"git_commit": "deadbeef", "gpu": "A100"},
                    "parameters": {"trainable_parameters": 123},
                    "split": {"group_overlap": {"train_test": 0}},
                }
            }
            write_paper_reproducibility_outputs(root, ["model"], [], runs)
            configs = json.loads(
                (root / "paper_single_seed_final_configs.json").read_text()
            )
            manifest = json.loads(
                (root / "paper_single_seed_reproducibility_manifest.json").read_text()
            )
            self.assertEqual(configs["model"]["training"]["seed"], 42)
            self.assertEqual(
                manifest["runs"]["model"]["completed"]["config_fingerprint"], "abc"
            )
            self.assertEqual(manifest["reported_runs_per_condition"], 1)


if __name__ == "__main__":
    unittest.main()
