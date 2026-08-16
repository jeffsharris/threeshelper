import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from threes_rl.rare_event_frontier import load_cases, run_frontier, select_diverse_cases, select_first_actions, target_reached
from threes_rl.record_replay import state_payload
from threes_rl.sim import DIRECTION_NAMES, SimState, ThreesSim, direction_index, preview_from_label


def rung_state() -> SimState:
    board = np.asarray(
        [
            [1536, 768, 768, 0],
            [384, 192, 96, 48],
            [24, 12, 6, 3],
            [3, 2, 1, 0],
        ],
        dtype=np.int32,
    )
    return SimState(
        board=board,
        preview=preview_from_label("gray"),
        small_counts={"red": 4, "blue": 4, "gray": 4},
        small_pos=8,
        small_seen_total=0,
        span_small_pos=0,
        large_pending=False,
        max_tile=1536,
        move_count=80,
        game_over=False,
    )


def record_payload(*, root_origin: str) -> dict:
    sim = ThreesSim(np.random.default_rng(7), starter_tile=1536)
    state = rung_state()
    return {
        "id": f"case-{root_origin}",
        "starter_tile": 1536,
        "source_replay": f"{root_origin}/source.json",
        "source_seed": 7,
        "source_frame_index": 80,
        "source_policy": "fixture_policy",
        "source_next_action": "left",
        "root_origin": root_origin,
        "root_replay": f"{root_origin}/root.json" if root_origin == "fresh" else None,
        "root_seed": 7 if root_origin == "fresh" else None,
        "root_frame_index": 0 if root_origin == "fresh" else None,
        "root_policy": "fixture_policy" if root_origin == "fresh" else None,
        "state": state_payload(state, sim),
    }


class RankedPolicy:
    def action_values(self, state, sim):
        legal = [int(action) for action in sim.legal_actions(state)]
        scores = {
            int(direction_index("left")): 4.0,
            int(direction_index("down")): 3.0,
            int(direction_index("up")): 2.0,
            int(direction_index("right")): 1.0,
        }
        return [(action, scores[action]) for action in legal]


class RareEventFrontierTests(unittest.TestCase):
    def test_select_diverse_cases_backfills_without_numpy_equality(self):
        records = []
        for idx in range(3):
            row = record_payload(root_origin="fresh")
            row["id"] = f"case-{idx}"
            row["source_frame_index"] = 80 + idx
            records.append(row)

        cases, rejected = load_cases(records, default_starter_tile=1536, root_origins={"fresh"})
        selected = select_diverse_cases(cases, max_starts=2, seed=123)

        self.assertEqual(rejected, {})
        self.assertEqual(len(selected), 2)
        self.assertEqual(len({case.id for case in selected}), 2)

    def test_frontier_branches_actions_and_filters_root_origin(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            tmp_path = Path(tmp)
            records_path = tmp_path / "records.json"
            out_dir = tmp_path / "out"
            records_path.write_text(
                json.dumps(
                    {
                        "records": [
                            record_payload(root_origin="fresh"),
                            record_payload(root_origin="unknown"),
                        ]
                    }
                )
            )

            payload = run_frontier(
                records_json=[records_path],
                policy_name="greedy",
                target="raw_duplicate_1536",
                horizon=1,
                repeats=1,
                max_starts=8,
                seed=123,
                root_origins={"fresh"},
                case_ids=None,
                default_starter_tile=1536,
                out_dir=out_dir,
            )
            self.assertTrue((out_dir / "frontier_records.json").exists())

        summary = payload["summary"]
        self.assertEqual(summary["cases_total"], 1)
        self.assertEqual(summary["cases_selected"], 1)
        self.assertEqual(summary["rejected"], {"root_origin:unknown": 1})
        self.assertGreaterEqual(summary["target_hits"], 1)
        hit_actions = {
            row["first_action"]
            for row in payload["action_summary"]
            if row["target_hits"] > 0
        }
        self.assertIn("left", hit_actions)

    def test_frontier_can_filter_exact_case_ids(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            tmp_path = Path(tmp)
            records_path = tmp_path / "records.json"
            out_dir = tmp_path / "out"
            records_path.write_text(
                json.dumps(
                    {
                        "records": [
                            record_payload(root_origin="fresh"),
                            {
                                **record_payload(root_origin="fresh"),
                                "id": "other-case",
                                "source_frame_index": 81,
                            },
                        ]
                    }
                )
            )

            payload = run_frontier(
                records_json=[records_path],
                policy_name="greedy",
                target="raw_duplicate_1536",
                horizon=1,
                repeats=1,
                max_starts=0,
                seed=123,
                root_origins={"fresh"},
                case_ids={"case-fresh", "missing-case"},
                default_starter_tile=1536,
                out_dir=out_dir,
            )

        self.assertEqual(payload["case_id_filter"], ["case-fresh", "missing-case"])
        self.assertEqual(payload["summary"]["cases_total"], 1)
        self.assertEqual(payload["summary"]["rejected"]["case_id_filter"], 1)
        self.assertEqual(payload["summary"]["rejected"]["case_id_missing"], 1)
        self.assertEqual({row["case_id"] for row in payload["action_summary"]}, {"case-fresh"})

    def test_select_first_actions_supports_top_two_and_recorded_modes(self):
        cases, rejected = load_cases([record_payload(root_origin="fresh")], default_starter_tile=1536, root_origins={"fresh"})
        self.assertEqual(rejected, {})
        case = cases[0]
        sim = ThreesSim(np.random.default_rng(7), starter_tile=1536)
        rng = np.random.default_rng(9)

        top_two = select_first_actions(policy=RankedPolicy(), case=case, sim=sim, mode="top-two", rng=rng)
        recorded = select_first_actions(policy=RankedPolicy(), case=case, sim=sim, mode="recorded", rng=rng)
        recorded_plus = select_first_actions(
            policy=RankedPolicy(),
            case=case,
            sim=sim,
            mode="recorded-plus-top-two",
            rng=rng,
        )

        self.assertEqual([DIRECTION_NAMES[action] for action in top_two], ["left", "down"])
        self.assertEqual([DIRECTION_NAMES[action] for action in recorded], ["left"])
        self.assertEqual([DIRECTION_NAMES[action] for action in recorded_plus], ["left", "down"])

    def test_frontier_recorded_action_mode_limits_rollouts(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            tmp_path = Path(tmp)
            records_path = tmp_path / "records.json"
            out_dir = tmp_path / "out"
            records_path.write_text(json.dumps({"records": [record_payload(root_origin="fresh")]}))

            payload = run_frontier(
                records_json=[records_path],
                policy_name="greedy",
                target="raw_duplicate_1536",
                horizon=1,
                repeats=3,
                max_starts=0,
                seed=123,
                root_origins={"fresh"},
                case_ids=None,
                default_starter_tile=1536,
                out_dir=out_dir,
                first_action_mode="recorded",
            )

        self.assertEqual(payload["summary"]["first_action_mode"], "recorded")
        self.assertEqual(payload["summary"]["rollouts_planned"], 3)
        self.assertEqual({row["first_action"] for row in payload["action_summary"]}, {"left"})

    def test_frontier_resumes_checkpointed_rollouts(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            tmp_path = Path(tmp)
            records_path = tmp_path / "records.json"
            out_dir = tmp_path / "out"
            records_path.write_text(json.dumps({"records": [record_payload(root_origin="fresh")]}))

            payload1 = run_frontier(
                records_json=[records_path],
                policy_name="greedy",
                target="raw_duplicate_1536",
                horizon=2,
                repeats=2,
                max_starts=0,
                seed=456,
                root_origins={"fresh"},
                case_ids=None,
                default_starter_tile=1536,
                out_dir=out_dir,
                checkpoint_rollouts=True,
            )
            payload2 = run_frontier(
                records_json=[records_path],
                policy_name="greedy",
                target="raw_duplicate_1536",
                horizon=2,
                repeats=2,
                max_starts=0,
                seed=456,
                root_origins={"fresh"},
                case_ids=None,
                default_starter_tile=1536,
                out_dir=out_dir,
                checkpoint_rollouts=True,
            )
            progress_exists = (out_dir / "frontier_progress.json").exists()

        self.assertTrue(progress_exists)
        self.assertGreater(payload1["summary"]["rollouts_ran"], 0)
        self.assertEqual(payload1["summary"]["rollouts_resumed"], 0)
        self.assertEqual(payload2["summary"]["rollouts_ran"], 0)
        self.assertEqual(payload2["summary"]["rollouts_resumed"], payload1["summary"]["rollouts"])
        self.assertEqual(payload2["summary"]["target_hits"], payload1["summary"]["target_hits"])
        self.assertEqual(payload2["action_summary"], payload1["action_summary"])

    def test_reached_targets_use_max_tile_excluding_starter(self):
        state = rung_state()
        self.assertFalse(target_reached(state, 1536, "reached_1536"))
        self.assertFalse(target_reached(state, 1536, "reached_3072"))
        self.assertFalse(target_reached(state, 1536, "reached_6144"))
        state.board[0, 2] = 1536
        self.assertTrue(target_reached(state, 1536, "reached_1536"))
        self.assertFalse(target_reached(state, 1536, "reached_3072"))
        state.board[0, 2] = 3072
        self.assertTrue(target_reached(state, 1536, "reached_1536"))
        self.assertTrue(target_reached(state, 1536, "reached_3072"))
        self.assertFalse(target_reached(state, 1536, "reached_6144"))
        state.board[0, 2] = 6144
        self.assertTrue(target_reached(state, 1536, "reached_3072"))
        self.assertTrue(target_reached(state, 1536, "reached_6144"))

    def test_raw_three_768_no_1536_target_rejects_existing_1536(self):
        state = rung_state()
        state.board[0, 0] = 3
        state.board[2, 2] = 768

        self.assertTrue(target_reached(state, 1536, "raw_three_768_no_1536"))
        state.board[1, 0] = 1536
        self.assertFalse(target_reached(state, 1536, "raw_three_768_no_1536"))

    def test_raw_four_adjacent_768_no_1536_requires_adjacent_support(self):
        state = rung_state()
        state.board[:, :] = np.asarray(
            [
                [3, 3072, 768, 0],
                [384, 192, 96, 48],
                [768, 12, 768, 3],
                [6, 2, 1, 768],
            ],
            dtype=np.int32,
        )

        self.assertFalse(target_reached(state, 1536, "raw_four_adjacent_768_no_1536"))
        state.board[0, 3] = 768
        self.assertTrue(target_reached(state, 1536, "raw_four_adjacent_768_no_1536"))
        state.board[1, 0] = 1536
        self.assertFalse(target_reached(state, 1536, "raw_four_adjacent_768_no_1536"))

    def test_raw_near_adjacent_1536_accepts_diagonal_touch_not_blocked_gap(self):
        state = rung_state()
        state.board[:, :] = np.asarray(
            [
                [1536, 0, 0, 0],
                [0, 1536, 3072, 0],
                [24, 12, 6, 3],
                [3, 2, 1, 0],
            ],
            dtype=np.int32,
        )

        self.assertTrue(target_reached(state, 1536, "raw_near_adjacent_1536"))
        self.assertFalse(target_reached(state, 1536, "raw_adjacent_1536"))
        state.board[:, :] = np.asarray(
            [
                [1536, 3072, 1536, 0],
                [0, 24, 12, 0],
                [24, 12, 6, 3],
                [3, 2, 1, 0],
            ],
            dtype=np.int32,
        )
        self.assertFalse(target_reached(state, 1536, "raw_near_adjacent_1536"))


if __name__ == "__main__":
    unittest.main()
