from __future__ import annotations

import unittest

import numpy as np

from threes_rl.r1b_coverage_audit import coverage_comparison, summarize_values


class R1bCoverageAuditTests(unittest.TestCase):
    def test_summarize_values_reports_residual_scale(self) -> None:
        summary = summarize_values(np.asarray([-4.0, 0.0, 2.0], dtype=np.float32))

        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["nonzero_count"], 2)
        self.assertAlmostEqual(summary["mean_abs"], 2.0)
        self.assertEqual(summary["max_abs"], 4.0)

    def test_coverage_requires_every_stage_to_retain_eighty_five_percent(self) -> None:
        previous = [
            {"index": 0, "name": "early", "table_entries_touched": 90},
            {"index": 1, "name": "late", "table_entries_touched": 80},
        ]
        current = [
            {"index": 0, "name": "early", "table_entries_touched": 100},
            {"index": 1, "name": "late", "table_entries_touched": 100},
        ]

        rows, saturated = coverage_comparison(previous, current)

        self.assertTrue(rows[0]["stage_saturated"])
        self.assertFalse(rows[1]["stage_saturated"])
        self.assertFalse(saturated)

    def test_coverage_rejects_regression(self) -> None:
        previous = [{"index": 0, "name": "early", "table_entries_touched": 101}]
        current = [{"index": 0, "name": "early", "table_entries_touched": 100}]

        with self.assertRaisesRegex(ValueError, "regressed"):
            coverage_comparison(previous, current)


if __name__ == "__main__":
    unittest.main()
