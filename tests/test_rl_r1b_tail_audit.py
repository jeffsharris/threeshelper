from __future__ import annotations

import unittest

from threes_rl.r1b_tail_audit import replay_corner_metrics, select_cases


def row(score: int) -> dict[str, str]:
    return {"score_minus_starter": str(score)}


class R1bTailAuditTests(unittest.TestCase):
    def test_select_cases_starts_with_largest_paired_losses(self) -> None:
        baseline = {("D2", idx): row(100 + idx) for idx in range(20)}
        candidate = {("D2", idx): row(100 + idx - idx * 10) for idx in range(20)}

        selected, threshold = select_cases(baseline, candidate)

        self.assertEqual(selected[0], ("D2", 19))
        self.assertEqual(len(selected), 12)
        self.assertGreater(threshold, 100.0)

    def test_replay_corner_metrics_detects_anchor_loss(self) -> None:
        replay = {
            "starter_tile": 1536,
            "frames": [
                {"index": 0, "state": {"move_count": 0, "board": [[1536, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}},
                {"index": 1, "state": {"move_count": 1, "board": [[3, 1536, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}},
            ],
        }

        metrics = replay_corner_metrics(replay)

        self.assertEqual(metrics["first_top_left_below_starter_move"], 1)
        self.assertTrue(metrics["final_top_left_below_starter"])
        self.assertFalse(metrics["final_max_at_top_left"])


if __name__ == "__main__":
    unittest.main()
