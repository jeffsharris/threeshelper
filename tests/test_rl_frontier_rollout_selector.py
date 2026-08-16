import unittest

from threes_rl.frontier_rollout_selector import aggregate_action_summary, select_records


def _record(record_id, root_seed, score_delta=0):
    return {
        "id": record_id,
        "root_seed": root_seed,
        "score_delta": score_delta,
        "features": {"empty_count": 1},
    }


class FrontierRolloutSelectorTest(unittest.TestCase):
    def test_aggregate_action_summary_tracks_best_action(self):
        rows = [
            {"case_id": "a", "first_action": "left", "target_hits": 1, "valid_rollouts": 4, "target_rate": 0.25},
            {"case_id": "a", "first_action": "right", "target_hits": 3, "valid_rollouts": 4, "target_rate": 0.75},
        ]

        scores = aggregate_action_summary(rows)

        self.assertEqual(scores["a"]["target_hits"], 4)
        self.assertEqual(scores["a"]["valid_rollouts"], 8)
        self.assertEqual(scores["a"]["target_rate"], 0.5)
        self.assertEqual(scores["a"]["best_action"], "right")
        self.assertEqual(scores["a"]["best_action_rate"], 0.75)

    def test_selects_top_by_best_action_rate_with_per_root_cap(self):
        records = [_record("a", 1), _record("b", 1), _record("c", 2)]
        scores = {
            "a": {"best_action_rate": 0.2, "target_rate": 0.1, "target_hits": 1, "valid_rollouts": 8},
            "b": {"best_action_rate": 0.8, "target_rate": 0.4, "target_hits": 4, "valid_rollouts": 8},
            "c": {"best_action_rate": 0.7, "target_rate": 0.3, "target_hits": 3, "valid_rollouts": 8},
        }

        selected, summary = select_records(records, scores=scores, mode="top", max_records=2, max_per_group=1)

        self.assertEqual([row["id"] for row in selected], ["b", "c"])
        self.assertEqual(summary["selected_counts"], {"1": 1, "2": 1})
        self.assertAlmostEqual(selected[0]["rollout_selector"]["screen"]["best_action_rate"], 0.8)

    def test_reference_matched_bottom_excludes_reference(self):
        records = [_record("a", 1), _record("b", 1), _record("c", 1), _record("d", 2), _record("e", 2)]
        reference = [_record("a", 1), _record("d", 2)]
        scores = {
            "a": {"best_action_rate": 0.9, "target_rate": 0.9, "target_hits": 9, "valid_rollouts": 10},
            "b": {"best_action_rate": 0.1, "target_rate": 0.1, "target_hits": 1, "valid_rollouts": 10},
            "c": {"best_action_rate": 0.3, "target_rate": 0.3, "target_hits": 3, "valid_rollouts": 10},
            "d": {"best_action_rate": 0.8, "target_rate": 0.8, "target_hits": 8, "valid_rollouts": 10},
            "e": {"best_action_rate": 0.2, "target_rate": 0.2, "target_hits": 2, "valid_rollouts": 10},
        }

        selected, summary = select_records(
            records,
            scores=scores,
            mode="bottom",
            reference_records=reference,
            exclude_reference=True,
        )

        self.assertEqual([row["id"] for row in selected], ["b", "e"])
        self.assertEqual(summary["reference_counts"], {"1": 1, "2": 1})
        self.assertEqual(summary["skipped"]["reference_record"], 2)
        self.assertEqual(summary["shortages"], {})


if __name__ == "__main__":
    unittest.main()
