import unittest

from threes_rl.frontier_matched_controls import select_matched_records


def _record(record_id, root_seed, rank):
    return {
        "id": record_id,
        "root_seed": root_seed,
        "score_delta": int(rank),
        "frontier": {"rank_tuple": [float(rank)]},
        "features": {"empty_count": 1},
    }


class FrontierMatchedControlsTest(unittest.TestCase):
    def test_bottom_matches_reference_counts_and_excludes_reference(self):
        pool = [
            _record("a_top", 1, 10),
            _record("a_mid", 1, 5),
            _record("a_low", 1, 1),
            _record("b_top", 2, 20),
            _record("b_mid", 2, 12),
            _record("b_low2", 2, 4),
            _record("b_low", 2, 2),
        ]
        reference = [_record("a_top", 1, 10), _record("b_top", 2, 20), _record("b_mid", 2, 12)]

        selected, summary = select_matched_records(pool, reference, mode="bottom", group_by="root_seed")

        self.assertEqual([row["id"] for row in selected], ["a_low", "b_low", "b_low2"])
        self.assertEqual(summary["reference_counts"], {"1": 1, "2": 2})
        self.assertEqual(summary["selected_counts"], {"1": 1, "2": 2})
        self.assertEqual(summary["skipped"]["reference_record"], 3)
        self.assertEqual(summary["shortages"], {})
        self.assertEqual(selected[0]["matched_control"]["mode"], "bottom")

    def test_random_is_deterministic(self):
        pool = [_record(f"a_{idx}", 1, idx) for idx in range(8)]
        reference = [_record("a_7", 1, 7), _record("a_6", 1, 6), _record("a_5", 1, 5)]

        first, first_summary = select_matched_records(pool, reference, mode="random", group_by="root_seed", seed=42)
        second, second_summary = select_matched_records(pool, reference, mode="random", group_by="root_seed", seed=42)

        self.assertEqual([row["id"] for row in first], [row["id"] for row in second])
        self.assertEqual(first_summary["selected_records"], 3)
        self.assertFalse({"a_7", "a_6", "a_5"} & {row["id"] for row in first})
        self.assertEqual(second_summary["shortages"], {})


if __name__ == "__main__":
    unittest.main()
