from __future__ import annotations

import unittest

from threes_rl.r15a_context_inventory import (
    ancestry_partition,
    cap_family_share,
    context_bins,
    empties_bin,
    plus_bin,
    round_robin_states,
)


class R15aContextInventoryTests(unittest.TestCase):
    def test_frozen_context_bins_have_expected_boundaries(self) -> None:
        self.assertEqual(plus_bin(0.0), "zero")
        self.assertEqual(plus_bin(0.09), "lt_0.10")
        self.assertEqual(plus_bin(0.10), "0.10_0.25")
        self.assertEqual(plus_bin(0.25), "ge_0.25")
        self.assertEqual(empties_bin(1), "0_1")
        self.assertEqual(empties_bin(3), "2_3")
        self.assertEqual(empties_bin(4), "4_plus")
        bins = context_bins(
            {
                "phase4_stage": "late_1536",
                "p_plus_next": 0.2,
                "visible_preview_kind": "bonus",
                "post_visible_large_pending": True,
                "empty_count": 2,
                "post_visible_small_pos": 8,
            }
        )
        self.assertEqual(
            bins,
            {
                "stage": "late_1536",
                "plus_bin": "0.10_0.25",
                "preview_bin": "bonus",
                "pending": "pending",
                "empties_bin": "2_3",
                "bag_bin": "8_11",
            },
        )

    def test_partition_keeps_whole_family_and_human_separate(self) -> None:
        self.assertEqual(ancestry_partition("corner2_lineage", "root-a"), "family_holdout")
        self.assertEqual(ancestry_partition("human_observed", "root-b"), "human_diagnostic")
        first = ancestry_partition("expectimax_baseline", "root-c")
        second = ancestry_partition("expectimax_baseline", "root-c")
        self.assertEqual(first, second)
        self.assertIn(first, {"train", "ancestry_holdout"})

    def test_family_cap_is_deterministic_and_below_frozen_share(self) -> None:
        families = {
            **{f"a-{index}": "a" for index in range(8)},
            **{f"b-{index}": "b" for index in range(3)},
            **{f"c-{index}": "c" for index in range(3)},
        }
        selected, removed = cap_family_share(set(families), families)
        counts = {family: sum(families[root] == family for root in selected) for family in {"a", "b", "c"}}

        self.assertEqual(selected, cap_family_share(set(families), families)[0])
        self.assertTrue(removed)
        self.assertLessEqual(max(counts.values()) / len(selected), 0.40)

    def test_round_robin_caps_states_and_spreads_ancestries(self) -> None:
        records = {}
        for ancestry in ("a", "b", "c"):
            records[ancestry] = [
                {
                    "record_id": f"{ancestry}-{index}",
                    "ancestry_key": ancestry,
                    "behavior_family": "family",
                    "context_cell": f"cell-{index}",
                }
                for index in range(4)
            ]

        selected = round_robin_states(records, set(records), cap=6)

        self.assertEqual(len(selected), 6)
        self.assertEqual({row["ancestry_key"] for row in selected}, {"a", "b", "c"})
        self.assertLessEqual(max(sum(row["ancestry_key"] == ancestry for row in selected) for ancestry in records), 2)


if __name__ == "__main__":
    unittest.main()
