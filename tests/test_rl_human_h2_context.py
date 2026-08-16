from __future__ import annotations

import unittest

from threes_rl.human_h2_context import (
    cycle_distance,
    permutation_null95,
    select_donor_pairs,
    small_support,
)


def root(root_id: str, ancestry: str, *, plus: float, pending: bool, span: int, small_pos: int):
    counts = {"red": 2, "blue": 2, "gray": 2}
    return {
        "root_id": root_id,
        "ancestry_cluster": ancestry,
        "match_features": {"plus_probability": plus},
        "state": {
            "tile_cycle": {
                "small_counts": counts,
                "small_pos": small_pos,
                "small_seen_total": 100,
                "span_small_pos": span,
                "large_pending": pending,
                "max_tile": 1536,
            }
        },
    }


class HumanH2ContextTests(unittest.TestCase):
    def test_select_donor_pairs_is_ancestry_capped_and_mechanics_only(self) -> None:
        roots = []
        for index in range(5):
            ancestry = f"a{index}"
            roots.extend(
                [
                    root(f"{ancestry}-low", ancestry, plus=0.0, pending=False, span=8, small_pos=2),
                    root(
                        f"{ancestry}-high",
                        ancestry,
                        plus=0.5 - index * 0.05,
                        pending=True,
                        span=19 - index,
                        small_pos=3,
                    ),
                ]
            )

        pairs = select_donor_pairs(roots, count=4)

        self.assertEqual(len(pairs), 4)
        self.assertEqual(len({pair["donor_ancestry"] for pair in pairs}), 4)
        self.assertEqual(pairs[0]["donor_ancestry"], "a0")
        self.assertEqual(pairs[0]["common_preview"], "blue")
        self.assertGreater(pairs[0]["high_plus_probability"], pairs[-1]["high_plus_probability"])

    def test_cycle_distance_is_zero_only_for_equal_cycle_features(self) -> None:
        left = root("x", "a", plus=0.0, pending=False, span=8, small_pos=2)["state"]["tile_cycle"]
        same = {**left, "small_counts": dict(left["small_counts"])}
        changed = {**same, "large_pending": True}

        self.assertEqual(cycle_distance(left, same), 0.0)
        self.assertGreater(cycle_distance(left, changed), 0.0)

    def test_small_support_uses_exact_cycle_probabilities(self) -> None:
        cycle = root("x", "a", plus=0.0, pending=False, span=8, small_pos=6)["state"]["tile_cycle"]

        support = small_support(cycle)

        self.assertEqual(set(support), {"red", "blue", "gray"})
        self.assertAlmostEqual(sum(support.values()), 1.0)

    def test_permutation_null_is_small_for_consistent_large_effect(self) -> None:
        cases = [[1000.0] * 16 for _ in range(24)]

        null95 = permutation_null95(cases, seed=7, repeats=500)

        self.assertLess(null95, 1000.0)


if __name__ == "__main__":
    unittest.main()
