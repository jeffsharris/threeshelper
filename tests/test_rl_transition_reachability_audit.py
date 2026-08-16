import unittest

import numpy as np

from threes_rl.record_replay import state_payload
from threes_rl.sim import SimState, ThreesSim, preview_from_label
from threes_rl.transition_reachability_audit import build_rows, feature_summary, grouped_logistic_probe


def audit_state(top_left: int, empty_count: int) -> SimState:
    board = np.asarray(
        [
            [top_left, 3072, 1536, 768],
            [3, 3, 6, 12],
            [24, 48, 96, 192],
            [384, 0, 0, 0],
        ],
        dtype=np.int32,
    )
    flats = board.reshape(-1)
    for idx in range(max(0, min(3, empty_count))):
        flats[-1 - idx] = 0
    return SimState(
        board=board,
        preview=preview_from_label("gray"),
        small_counts={"red": 4, "blue": 4, "gray": 4},
        small_pos=0,
        small_seen_total=0,
        span_small_pos=0,
        large_pending=False,
        max_tile=int(board.max(initial=0)),
        move_count=100,
        game_over=False,
    )


def raw_feature_state() -> SimState:
    board = np.asarray(
        [
            [3072, 1536, 768, 768],
            [1536, 384, 192, 96],
            [48, 24, 12, 6],
            [3, 2, 1, 0],
        ],
        dtype=np.int32,
    )
    return SimState(
        board=board,
        preview=preview_from_label("gray"),
        small_counts={"red": 4, "blue": 4, "gray": 4},
        small_pos=0,
        small_seen_total=0,
        span_small_pos=0,
        large_pending=False,
        max_tile=int(board.max(initial=0)),
        move_count=100,
        game_over=False,
    )


class TransitionReachabilityAuditTests(unittest.TestCase):
    def test_build_rows_and_feature_summary(self):
        sim = ThreesSim(np.random.default_rng(1), starter_tile=1536)
        records = []
        for idx in range(6):
            success = idx % 2 == 0
            state = audit_state(3072 if success else 3, 2 if success else 0)
            records.append(
                {
                    "id": f"r{idx}",
                    "target_tile": 6144,
                    "outcome": "success" if success else "failure",
                    "source_replay": f"replay-{idx // 2}",
                    "source_next_action": "up",
                    "starter_tile": 1536,
                    "features": {
                        "score_minus_starter": 1000 + idx,
                        "empty_count": 2 if success else 0,
                        "top_left": 3072 if success else 3,
                        "preview": "gray",
                        "corner_risk": "high_corner_risk",
                        "stratum": "endgame_3072p/high_corner_risk",
                        "raw_count_768": 2 if success else 1,
                        "raw_count_1536": 2 if success else 1,
                        "raw_highest_duplicate_tile": 1536 if success else 0,
                        "raw_highest_adjacent_pair_tile": 768 if success else 0,
                        "raw_has_adjacent_768": success,
                        "raw_has_adjacent_1536": False,
                    },
                    "state": state_payload(state, sim),
                }
            )

        rows = build_rows(records, target_tile=6144)
        summary = feature_summary(rows)
        probe = grouped_logistic_probe(rows)

        self.assertEqual(len(rows), 6)
        self.assertEqual(rows[0]["y"], 1)
        self.assertEqual(rows[0]["raw_highest_duplicate_tile"], 1536)
        self.assertEqual(rows[1]["raw_has_adjacent_768"], 0.0)
        self.assertIn("numeric", summary)
        self.assertGreaterEqual(probe["predictions"], 0)

    def test_build_rows_computes_raw_features_when_missing(self):
        sim = ThreesSim(np.random.default_rng(1), starter_tile=1536)
        state = raw_feature_state()
        records = [
            {
                "id": "r",
                "target_tile": 1536,
                "outcome": "success",
                "source_replay": "replay",
                "source_next_action": "up",
                "starter_tile": 1536,
                "features": {
                    "score_minus_starter": 1000,
                    "empty_count": 1,
                    "top_left": 3072,
                    "preview": "gray",
                    "corner_risk": "high_corner_risk",
                    "stratum": "endgame_3072p/high_corner_risk",
                },
                "state": state_payload(state, sim),
            }
        ]

        rows = build_rows(records, target_tile=1536)

        self.assertEqual(rows[0]["raw_count_768"], 2)
        self.assertEqual(rows[0]["raw_count_1536"], 2)
        self.assertEqual(rows[0]["raw_highest_duplicate_tile"], 1536)
        self.assertEqual(rows[0]["raw_highest_adjacent_pair_tile"], 768)
        self.assertEqual(rows[0]["raw_has_adjacent_768"], 1.0)
        self.assertEqual(rows[0]["raw_has_adjacent_1536"], 0.0)
        self.assertEqual(rows[0]["raw_768_min_pair_distance"], 1.0)
        self.assertEqual(rows[0]["raw_768_min_distance_to_1536"], 1.0)
        self.assertEqual(rows[0]["raw_768_adjacent_to_1536"], 1.0)
        self.assertEqual(rows[0]["raw_1536_min_pair_distance"], 2.0)
        self.assertEqual(rows[0]["raw_1536_min_distance_to_3072"], 1.0)
        self.assertEqual(rows[0]["raw_1536_adjacent_to_3072"], 1.0)
        self.assertEqual(rows[0]["max_tile_corner_distance"], 0.0)
        self.assertEqual(rows[0]["corner_count_ge_768"], 2.0)
        self.assertEqual(rows[0]["raw_768_same_line_pair_count"], 1.0)
        self.assertEqual(rows[0]["raw_768_line_blocker_min"], 0.0)
        self.assertEqual(rows[0]["raw_768_clear_merge_pair_count"], 1.0)
        self.assertEqual(rows[0]["raw_768_min_clear_pair_distance"], 1.0)
        self.assertEqual(rows[0]["raw_768_clear_target_min_distance_to_1536"], 1.0)
        self.assertEqual(rows[0]["raw_768_clear_target_adjacent_to_1536"], 1.0)
        self.assertEqual(rows[0]["raw_1536_same_line_pair_count"], 0.0)
        self.assertEqual(rows[0]["raw_1536_clear_merge_pair_count"], 0.0)
        self.assertEqual(rows[0]["raw_1536_min_clear_pair_distance"], 9.0)
        self.assertEqual(rows[0]["cell_00_rank"], 13.0)
        self.assertEqual(rows[0]["cell_01_rank"], 12.0)
        self.assertEqual(rows[0]["cell_02_is_768"], 1.0)
        self.assertEqual(rows[0]["cell_00_is_3072"], 1.0)

    def test_build_rows_can_skip_target_filter_for_milestone_records(self):
        sim = ThreesSim(np.random.default_rng(1), starter_tile=1536)
        state = raw_feature_state()
        records = [
            {
                "id": "milestone",
                "outcome": "success",
                "source_replay": "replay",
                "starter_tile": 1536,
                "state": state_payload(state, sim),
            }
        ]

        self.assertEqual(build_rows(records, target_tile=6144), [])
        self.assertEqual(len(build_rows(records, target_tile=None)), 1)

    def test_build_rows_can_group_by_original_replay(self):
        sim = ThreesSim(np.random.default_rng(1), starter_tile=1536)
        state = raw_feature_state()
        records = []
        for idx in range(2):
            records.append(
                {
                    "id": f"r{idx}",
                    "target_tile": 1536,
                    "outcome": "success" if idx == 0 else "failure",
                    "source_replay": f"continuation-{idx}.json",
                    "source_group": f"frame-{idx}",
                    "original_source_replay": "normal-game/replay.json",
                    "original_source_seed": 42,
                    "starter_tile": 1536,
                    "state": state_payload(state, sim),
                }
            )

        auto_rows = build_rows(records, target_tile=1536)
        replay_rows = build_rows(records, target_tile=1536, group_by="original-replay")

        self.assertNotEqual(auto_rows[0]["group_key"], auto_rows[1]["group_key"])
        self.assertEqual(replay_rows[0]["group_key"], replay_rows[1]["group_key"])
        self.assertIn("normal-game/replay.json", replay_rows[0]["group_key"])


if __name__ == "__main__":
    unittest.main()
