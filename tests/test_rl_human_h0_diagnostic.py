from __future__ import annotations

import unittest

import numpy as np

from threes_rl.human_h0_diagnostic import top_edge_metrics


class HumanH0DiagnosticTests(unittest.TestCase):
    def test_top_edge_metrics_detect_preserved_descending_anchor(self) -> None:
        initial = np.asarray([[1536, 768, 384, 96], [3, 2, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        later = np.asarray([[1536, 768, 384, 192], [3, 2, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

        metrics = top_edge_metrics(later, initial)

        self.assertEqual(metrics["anchor_preserved"], 1)
        self.assertEqual(metrics["top_edge_descending"], 1)
        self.assertGreater(metrics["top_edge_rank_mass_delta"], 0)


if __name__ == "__main__":
    unittest.main()
