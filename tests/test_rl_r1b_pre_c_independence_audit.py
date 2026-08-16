from __future__ import annotations

import unittest

from threes_rl.r1b_pre_c_independence_audit import (
    eligible_current_state,
    select_outcome_independent_records,
)


def record(ancestry: str, move: int, *, support: bool = True, outcome: str = "failure") -> dict:
    board = [[1536, 768, 384 if support else 192, 0], [3, 2, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    return {
        "record_id": f"{ancestry}-{move}",
        "ancestry_key": ancestry,
        "root_origin": "fresh",
        "starter_tile": 1536,
        "trajectory_outcome": outcome,
        "state": {"board": board, "move_count": move, "game_over": False},
    }


class R1bPreCIndependenceAuditTests(unittest.TestCase):
    def test_current_state_rule_requires_768_and_384_support(self) -> None:
        self.assertTrue(eligible_current_state(record("a", 10)))
        self.assertFalse(eligible_current_state(record("a", 10, support=False)))

    def test_selection_excludes_sampled_ancestry_and_uses_earliest_frame(self) -> None:
        records = [
            record("clean", 30, outcome="success"),
            record("clean", 20, outcome="failure"),
            record("sampled", 10),
        ]

        selected = select_outcome_independent_records(records, {"sampled"})

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["ancestry_key"], "clean")
        self.assertEqual(selected[0]["state"]["move_count"], 20)

    def test_selection_is_invariant_to_outcome_labels(self) -> None:
        records = [record("clean", 20, outcome="success"), record("clean", 30, outcome="failure")]
        flipped = [dict(row, trajectory_outcome="failure" if row["trajectory_outcome"] == "success" else "success") for row in records]

        original = select_outcome_independent_records(records, set())
        relabeled = select_outcome_independent_records(flipped, set())

        self.assertEqual(original[0]["record_id"], relabeled[0]["record_id"])


if __name__ == "__main__":
    unittest.main()
