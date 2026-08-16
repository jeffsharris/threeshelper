import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from threes_rl.post3072_frontier import Post3072Node, run_frontier, select_archive_nodes
from threes_rl.rare_event_frontier import FrontierCase
from threes_rl.record_replay import state_payload
from threes_rl.sim import SimState, ThreesSim, preview_from_label


def state_from_board(board) -> SimState:
    arr = np.asarray(board, dtype=np.int32)
    return SimState(
        board=arr,
        preview=preview_from_label("gray"),
        small_counts={"red": 4, "blue": 4, "gray": 4},
        small_pos=8,
        small_seen_total=40,
        span_small_pos=0,
        large_pending=False,
        max_tile=int(np.max(arr)),
        move_count=220,
        game_over=False,
    )


def post3072_state(*, duplicate_1536: bool = False, raw_768s: int = 2) -> SimState:
    board = np.asarray(
        [
            [1536, 3072, 0, 0],
            [768, 768 if raw_768s >= 2 else 0, 0, 0],
            [768 if raw_768s >= 3 else 0, 0, 0, 0],
            [3, 6, 12, 24],
        ],
        dtype=np.int32,
    )
    if duplicate_1536:
        board[2, 2] = 1536
    return state_from_board(board)


def record_payload(state: SimState, *, record_id: str = "case", root_origin: str = "fresh", seed: int = 7) -> dict:
    sim = ThreesSim(np.random.default_rng(seed), starter_tile=1536)
    return {
        "id": record_id,
        "starter_tile": 1536,
        "source_replay": f"{root_origin}/source_{seed}.json",
        "source_seed": seed,
        "source_frame_index": 220,
        "source_policy": "fixture",
        "root_origin": root_origin,
        "root_replay": f"{root_origin}/root_{seed}.json" if root_origin == "fresh" else None,
        "root_seed": seed if root_origin == "fresh" else None,
        "root_frame_index": 0 if root_origin == "fresh" else None,
        "root_policy": "fixture" if root_origin == "fresh" else None,
        "state": state_payload(state, sim),
    }


def frontier_case(root_seed: int, state: SimState) -> FrontierCase:
    return FrontierCase(
        id=f"case-{root_seed}",
        state=state,
        starter_tile=1536,
        source_replay=f"root-{root_seed}.json",
        source_seed=root_seed,
        source_frame_index=220,
        source_policy="fixture",
        root_origin="fresh",
        root_replay=f"root-{root_seed}.json",
        root_seed=root_seed,
        root_frame_index=0,
        root_policy="fixture",
        root_policy_family="fixture",
        ancestry_key=f"root:fresh:{root_seed}",
        features={},
        raw={},
    )


def node(root_seed: int, *, node_id: str, state: SimState, score_delta: int = 0) -> Post3072Node:
    return Post3072Node(
        case=frontier_case(root_seed, state),
        state=state,
        depth=1,
        path=[node_id],
        seed_path=[root_seed],
        start_score=0,
        score_delta=score_delta,
        parent_id=None,
        node_id=node_id,
    )


class Post3072FrontierTests(unittest.TestCase):
    def test_start_that_already_hits_duplicate_1536_is_exported(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            tmp_path = Path(tmp)
            records_path = tmp_path / "records.json"
            out_dir = tmp_path / "frontier"
            records_path.write_text(json.dumps({"records": [record_payload(post3072_state(duplicate_1536=True))]}))

            payload = run_frontier(
                records_json=[records_path],
                target="raw_duplicate_1536",
                max_depth=0,
                repeats_per_action=1,
                max_starts=0,
                min_start_raw_768=0,
                min_raw_768=0,
                max_nodes_per_depth=8,
                max_per_cell=2,
                seed=20260708,
                root_origins={"fresh"},
                default_starter_tile=1536,
                out_dir=out_dir,
                progress_every=0,
                compact_output=True,
            )

            self.assertEqual(payload["summary"]["cases_selected"], 1)
            self.assertEqual(payload["summary"]["target_records"], 1)
            self.assertEqual(payload["summary"]["target_by_depth"]["0"], 1)
            self.assertTrue(payload["summary"]["compact_output"] if "compact_output" in payload["summary"] else payload["compact_output"])
            self.assertEqual(payload["records"], [])
            self.assertIsNone(payload["records_json"])
            self.assertFalse((out_dir / "records.json").exists())
            self.assertTrue((out_dir / "target_records.json").exists())
            self.assertTrue((out_dir / "post3072_frontier.html").exists())

    def test_filters_unknown_roots_and_pre_3072_states(self):
        pre_3072 = state_from_board(
            [
                [1536, 768, 0, 0],
                [768, 0, 0, 0],
                [0, 0, 0, 0],
                [3, 6, 12, 24],
            ]
        )
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            tmp_path = Path(tmp)
            records_path = tmp_path / "records.json"
            out_dir = tmp_path / "frontier"
            records_path.write_text(
                json.dumps(
                    {
                        "records": [
                            record_payload(post3072_state(), record_id="fresh-ok"),
                            record_payload(pre_3072, record_id="pre-3072"),
                            record_payload(post3072_state(), record_id="unknown", root_origin="unknown"),
                        ]
                    }
                )
            )

            payload = run_frontier(
                records_json=[records_path],
                target="raw_duplicate_1536",
                max_depth=0,
                repeats_per_action=1,
                max_starts=0,
                min_start_raw_768=0,
                min_raw_768=0,
                max_nodes_per_depth=8,
                max_per_cell=2,
                seed=20260708,
                root_origins={"fresh"},
                default_starter_tile=1536,
                out_dir=out_dir,
                progress_every=0,
            )

            self.assertEqual(payload["summary"]["cases_selected"], 1)
            self.assertEqual(payload["summary"]["rejected"]["below_first_3072"], 1)
            self.assertEqual(payload["summary"]["rejected"]["root_origin:unknown"], 1)

    def test_omit_transitions_keeps_archive_records_without_transition_file(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            tmp_path = Path(tmp)
            records_path = tmp_path / "records.json"
            out_dir = tmp_path / "frontier"
            records_path.write_text(json.dumps({"records": [record_payload(post3072_state(duplicate_1536=True))]}))

            payload = run_frontier(
                records_json=[records_path],
                target="raw_duplicate_1536",
                max_depth=0,
                repeats_per_action=1,
                max_starts=0,
                min_start_raw_768=0,
                min_raw_768=0,
                max_nodes_per_depth=8,
                max_per_cell=2,
                seed=20260708,
                root_origins={"fresh"},
                default_starter_tile=1536,
                out_dir=out_dir,
                progress_every=0,
                omit_transitions=True,
            )

            self.assertFalse(payload["compact_output"])
            self.assertTrue(payload["omit_transitions"])
            self.assertGreater(len(payload["records"]), 0)
            self.assertEqual(payload["transitions"], [])
            self.assertTrue((out_dir / "records.json").exists())
            self.assertFalse((out_dir / "transitions.json").exists())

    def test_near1536_rank_profile_prefers_near_duplicate_geometry(self):
        far = state_from_board(
            [
                [1536, 3072, 0, 0],
                [768, 768, 0, 0],
                [0, 0, 1536, 0],
                [3, 6, 12, 24],
            ]
        )
        near = state_from_board(
            [
                [1536, 3072, 0, 0],
                [0, 1536, 0, 0],
                [768, 768, 0, 0],
                [3, 6, 12, 24],
            ]
        )
        candidates = [
            node(1, node_id="far", state=far, score_delta=1000),
            node(2, node_id="near", state=near, score_delta=1),
        ]

        selected = select_archive_nodes(candidates, max_nodes=1, max_per_cell=4, rank_profile="near1536")

        self.assertEqual(selected[0].node_id, "near")

    def test_regen768_rank_profile_prefers_rebuild_material_over_lone_1536(self):
        lone_1536 = state_from_board(
            [
                [1536, 3072, 0, 0],
                [1536, 0, 0, 0],
                [0, 0, 0, 0],
                [3, 6, 12, 24],
            ]
        )
        broad_stock = state_from_board(
            [
                [1536, 3072, 0, 0],
                [768, 384, 384, 0],
                [192, 192, 96, 0],
                [3, 6, 12, 24],
            ]
        )
        candidates = [
            node(1, node_id="lone-1536", state=lone_1536, score_delta=1000),
            node(2, node_id="broad-stock", state=broad_stock, score_delta=1),
        ]

        selected = select_archive_nodes(candidates, max_nodes=1, max_per_cell=4, rank_profile="regen768")

        self.assertEqual(selected[0].node_id, "broad-stock")

    def test_balance_by_root_preserves_weaker_root_candidate(self):
        candidates = [
            node(1, node_id="root1-high", state=post3072_state(raw_768s=3), score_delta=1000),
            node(1, node_id="root1-mid", state=post3072_state(raw_768s=2), score_delta=900),
            node(2, node_id="root2-low", state=post3072_state(raw_768s=2), score_delta=1),
        ]

        unbalanced = select_archive_nodes(candidates, max_nodes=2, max_per_cell=4)
        balanced = select_archive_nodes(candidates, max_nodes=2, max_per_cell=4, balance_by_root=True)

        self.assertEqual([item.node_id for item in unbalanced], ["root1-high", "root1-mid"])
        self.assertEqual({item.case.root_seed for item in balanced}, {1, 2})


if __name__ == "__main__":
    unittest.main()
