import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from threes_rl.record_replay import state_payload
from threes_rl.rare_event_frontier import FrontierCase
from threes_rl.sim import SimState, ThreesSim, preview_from_label
from threes_rl.support_accumulation_frontier import AccumulationNode, _target_hit, run_frontier, select_archive_nodes


def accumulation_state(raw_768s: int, *, raw_1536s: int = 0) -> SimState:
    board = np.asarray(
        [
            [768, 768, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [3, 6, 12, 24],
        ],
        dtype=np.int32,
    )
    if raw_768s >= 3:
        board[1, 0] = 768
    if raw_768s >= 4:
        board[1, 1] = 768
    if raw_1536s:
        board[2, 0] = 1536
    return SimState(
        board=board,
        preview=preview_from_label("gray"),
        small_counts={"red": 4, "blue": 4, "gray": 4},
        small_pos=8,
        small_seen_total=40,
        span_small_pos=0,
        large_pending=False,
        max_tile=int(np.max(board)),
        move_count=80,
        game_over=False,
    )


def custom_state(board) -> SimState:
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
        move_count=80,
        game_over=False,
    )


def fixed_starter_support_state(raw_768s: int, *, built_1536: bool = False) -> SimState:
    board = np.asarray(
        [
            [1536, 768, 768, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [3, 6, 12, 24],
        ],
        dtype=np.int32,
    )
    if raw_768s >= 3:
        board[1, 1] = 768
    if raw_768s >= 4:
        board[2, 2] = 768
    if built_1536:
        board[2, 0] = 1536
    return custom_state(board)


def record_payload(state: SimState, *, record_id: str = "case", root_origin: str = "fresh", seed: int = 7) -> dict:
    sim = ThreesSim(np.random.default_rng(seed), starter_tile=1536)
    return {
        "id": record_id,
        "starter_tile": 1536,
        "source_replay": f"{root_origin}/source.json",
        "source_seed": seed,
        "source_frame_index": 80,
        "source_policy": "fixture",
        "root_origin": root_origin,
        "root_replay": f"{root_origin}/root.json" if root_origin == "fresh" else None,
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
        source_frame_index=0,
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


def accumulation_node(root_seed: int, *, node_id: str, score_delta: int, state: SimState | None = None) -> AccumulationNode:
    state = state or accumulation_state(3)
    return AccumulationNode(
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


class SupportAccumulationFrontierTests(unittest.TestCase):
    def test_balance_by_root_preserves_weaker_root_candidate(self):
        candidates = [
            accumulation_node(1, node_id="root1-high", score_delta=1000),
            accumulation_node(1, node_id="root1-mid", score_delta=900),
            accumulation_node(2, node_id="root2-low", score_delta=1),
        ]

        unbalanced = select_archive_nodes(candidates, max_nodes=2, max_per_cell=4)
        balanced = select_archive_nodes(candidates, max_nodes=2, max_per_cell=4, balance_by_root=True)

        self.assertEqual([node.node_id for node in unbalanced], ["root1-high", "root1-mid"])
        self.assertEqual({node.case.root_seed for node in balanced}, {1, 2})

    def test_buildable_rank_profile_prefers_separated_768s(self):
        adjacent = custom_state(
            [
                [768, 768, 0, 0],
                [768, 0, 0, 0],
                [0, 0, 0, 0],
                [3, 6, 12, 24],
            ]
        )
        separated = custom_state(
            [
                [768, 0, 768, 0],
                [0, 0, 0, 0],
                [768, 0, 0, 0],
                [3, 6, 12, 24],
            ]
        )
        candidates = [
            accumulation_node(1, node_id="adjacent-high", score_delta=1000, state=adjacent),
            accumulation_node(2, node_id="separated-low", score_delta=1, state=separated),
        ]

        default = select_archive_nodes(candidates, max_nodes=1, max_per_cell=4)
        buildable = select_archive_nodes(candidates, max_nodes=1, max_per_cell=4, rank_profile="buildable")

        self.assertEqual(default[0].node_id, "adjacent-high")
        self.assertEqual(buildable[0].node_id, "separated-low")

    def test_bridge_rank_profile_prefers_adjacent_support_over_separated_four(self):
        adjacent_three = custom_state(
            [
                [768, 768, 0, 0],
                [0, 0, 0, 0],
                [768, 0, 0, 0],
                [3, 6, 12, 24],
            ]
        )
        separated_four = custom_state(
            [
                [768, 0, 768, 0],
                [0, 0, 0, 0],
                [768, 0, 768, 0],
                [3, 6, 12, 24],
            ]
        )
        candidates = [
            accumulation_node(1, node_id="separated-four", score_delta=1000, state=separated_four),
            accumulation_node(2, node_id="adjacent-three", score_delta=1, state=adjacent_three),
        ]

        buildable = select_archive_nodes(candidates, max_nodes=1, max_per_cell=4, rank_profile="buildable")
        bridge = select_archive_nodes(candidates, max_nodes=1, max_per_cell=4, rank_profile="bridge")

        self.assertEqual(buildable[0].node_id, "separated-four")
        self.assertEqual(bridge[0].node_id, "adjacent-three")

    def test_second768_rank_profile_prefers_adjacent_384_support(self):
        weak_support = custom_state(
            [
                [1536, 768, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [3, 6, 12, 24],
            ]
        )
        adjacent_384 = custom_state(
            [
                [1536, 768, 0, 0],
                [384, 384, 0, 0],
                [0, 0, 0, 0],
                [3, 6, 12, 24],
            ]
        )
        candidates = [
            accumulation_node(1, node_id="weak-high-score", score_delta=1000, state=weak_support),
            accumulation_node(2, node_id="adjacent-384-low-score", score_delta=1, state=adjacent_384),
        ]

        default = select_archive_nodes(candidates, max_nodes=1, max_per_cell=4)
        second768 = select_archive_nodes(candidates, max_nodes=1, max_per_cell=4, rank_profile="second768")

        self.assertEqual(default[0].node_id, "weak-high-score")
        self.assertEqual(second768[0].node_id, "adjacent-384-low-score")

    def test_supportstock_rank_profile_prefers_broad_lower_support(self):
        adjacent_192 = custom_state(
            [
                [1536, 768, 0, 0],
                [192, 192, 0, 0],
                [0, 0, 0, 0],
                [3, 6, 12, 24],
            ]
        )
        broad_stock = custom_state(
            [
                [1536, 768, 384, 0],
                [192, 96, 48, 0],
                [0, 0, 0, 0],
                [3, 6, 12, 24],
            ]
        )
        candidates = [
            accumulation_node(1, node_id="adjacent-192-high-score", score_delta=1000, state=adjacent_192),
            accumulation_node(2, node_id="broad-stock-low-score", score_delta=1, state=broad_stock),
        ]

        second768 = select_archive_nodes(candidates, max_nodes=1, max_per_cell=4, rank_profile="second768")
        supportstock = select_archive_nodes(candidates, max_nodes=1, max_per_cell=4, rank_profile="supportstock")

        self.assertEqual(second768[0].node_id, "adjacent-192-high-score")
        self.assertEqual(supportstock[0].node_id, "broad-stock-low-score")

    def test_384_support_targets_require_one_768_and_no_built_1536(self):
        no_lower_support = custom_state(
            [
                [1536, 768, 0, 0],
                [96, 0, 0, 0],
                [0, 0, 0, 0],
                [3, 6, 12, 24],
            ]
        )
        one_192 = custom_state(
            [
                [1536, 768, 0, 0],
                [192, 0, 0, 0],
                [0, 0, 0, 0],
                [3, 6, 12, 24],
            ]
        )
        one_384 = custom_state(
            [
                [1536, 768, 0, 0],
                [384, 0, 0, 0],
                [0, 0, 0, 0],
                [3, 6, 12, 24],
            ]
        )
        duplicate_384 = custom_state(
            [
                [1536, 768, 0, 0],
                [384, 0, 0, 384],
                [0, 0, 0, 0],
                [3, 6, 12, 24],
            ]
        )
        adjacent_384 = custom_state(
            [
                [1536, 768, 0, 0],
                [384, 384, 0, 0],
                [0, 0, 0, 0],
                [3, 6, 12, 24],
            ]
        )
        already_two_768 = custom_state(
            [
                [1536, 768, 768, 0],
                [384, 384, 0, 0],
                [0, 0, 0, 0],
                [3, 6, 12, 24],
            ]
        )
        built_1536 = custom_state(
            [
                [1536, 768, 0, 0],
                [384, 384, 1536, 0],
                [0, 0, 0, 0],
                [3, 6, 12, 24],
            ]
        )

        self.assertFalse(_target_hit(no_lower_support, 1536, "raw_one_192_with_one_768_no_1536"))
        self.assertFalse(_target_hit(no_lower_support, 1536, "raw_one_384_with_one_768_no_1536"))
        self.assertTrue(_target_hit(one_192, 1536, "raw_one_192_with_one_768_no_1536"))
        self.assertFalse(_target_hit(one_192, 1536, "raw_one_384_with_one_768_no_1536"))
        self.assertTrue(_target_hit(one_384, 1536, "raw_one_384_with_one_768_no_1536"))
        self.assertFalse(_target_hit(already_two_768, 1536, "raw_one_192_with_one_768_no_1536"))
        self.assertFalse(_target_hit(already_two_768, 1536, "raw_one_384_with_one_768_no_1536"))
        self.assertFalse(_target_hit(built_1536, 1536, "raw_one_192_with_one_768_no_1536"))
        self.assertFalse(_target_hit(built_1536, 1536, "raw_one_384_with_one_768_no_1536"))
        self.assertTrue(_target_hit(duplicate_384, 1536, "raw_two_384_with_one_768_no_1536"))
        self.assertFalse(_target_hit(duplicate_384, 1536, "raw_adjacent_384_with_one_768_no_1536"))
        self.assertTrue(_target_hit(adjacent_384, 1536, "raw_two_384_with_one_768_no_1536"))
        self.assertTrue(_target_hit(adjacent_384, 1536, "raw_adjacent_384_with_one_768_no_1536"))
        self.assertFalse(_target_hit(already_two_768, 1536, "raw_two_384_with_one_768_no_1536"))
        self.assertFalse(_target_hit(built_1536, 1536, "raw_two_384_with_one_768_no_1536"))

    def test_archives_start_that_already_hits_three_768_target(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            tmp_path = Path(tmp)
            records_path = tmp_path / "records.json"
            out_dir = tmp_path / "frontier"
            records_path.write_text(json.dumps({"records": [record_payload(accumulation_state(3))]}))

            payload = run_frontier(
                records_json=[records_path],
                target="raw_three_768_no_1536",
                max_depth=1,
                repeats_per_action=1,
                max_starts=0,
                min_start_raw_768=2,
                min_raw_768=2,
                max_nodes_per_depth=8,
                max_per_cell=2,
                seed=20260708,
                root_origins={"fresh"},
                default_starter_tile=1536,
                require_no_start_1536=True,
                out_dir=out_dir,
                progress_every=0,
            )

            self.assertEqual(payload["summary"]["cases_selected"], 1)
            self.assertGreater(payload["summary"]["target_records"], 0)
            self.assertEqual(payload["summary"]["target_by_depth"]["0"], 1)
            self.assertTrue((out_dir / "target_records.json").exists())
            self.assertTrue((out_dir / "support_accumulation_frontier.html").exists())

    def test_fixed_starter_does_not_count_as_built_1536_for_support_targets(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            tmp_path = Path(tmp)
            records_path = tmp_path / "records.json"
            out_dir = tmp_path / "frontier"
            records_path.write_text(json.dumps({"records": [record_payload(fixed_starter_support_state(3))]}))

            payload = run_frontier(
                records_json=[records_path],
                target="raw_three_768_no_1536",
                max_depth=0,
                repeats_per_action=1,
                max_starts=0,
                min_start_raw_768=2,
                min_raw_768=2,
                max_nodes_per_depth=8,
                max_per_cell=2,
                seed=20260708,
                root_origins={"fresh"},
                default_starter_tile=1536,
                require_no_start_1536=True,
                out_dir=out_dir,
                progress_every=0,
            )

            self.assertEqual(payload["summary"]["cases_selected"], 1)
            self.assertEqual(payload["summary"]["target_records"], 1)
            self.assertEqual(payload["target_records"][0]["features"]["raw_count_1536"], 1)
            self.assertEqual(payload["target_records"][0]["features"]["masked_count_1536"], 0)

    def test_compact_output_keeps_target_records_without_full_archives(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            tmp_path = Path(tmp)
            records_path = tmp_path / "records.json"
            out_dir = tmp_path / "frontier"
            records_path.write_text(json.dumps({"records": [record_payload(fixed_starter_support_state(2))]}))

            payload = run_frontier(
                records_json=[records_path],
                target="raw_two_768_no_1536",
                max_depth=0,
                repeats_per_action=1,
                max_starts=0,
                min_start_raw_768=1,
                min_raw_768=1,
                max_nodes_per_depth=8,
                max_per_cell=2,
                seed=20260708,
                root_origins={"fresh"},
                default_starter_tile=1536,
                require_no_start_1536=True,
                out_dir=out_dir,
                progress_every=0,
                compact_output=True,
            )

            self.assertTrue(payload["compact_output"])
            self.assertEqual(payload["summary"]["target_records"], 1)
            self.assertEqual(payload["records"], [])
            self.assertEqual(payload["transitions"], [])
            self.assertFalse((out_dir / "records.json").exists())
            self.assertFalse((out_dir / "transitions.json").exists())
            self.assertTrue((out_dir / "target_records.json").exists())
            self.assertTrue((out_dir / "summary.json").exists())

    def test_raw_two_768_target_ignores_fixed_starter(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            tmp_path = Path(tmp)
            records_path = tmp_path / "records.json"
            out_dir = tmp_path / "frontier"
            records_path.write_text(json.dumps({"records": [record_payload(fixed_starter_support_state(2))]}))

            payload = run_frontier(
                records_json=[records_path],
                target="raw_two_768_no_1536",
                max_depth=0,
                repeats_per_action=1,
                max_starts=0,
                min_start_raw_768=1,
                min_raw_768=1,
                max_nodes_per_depth=8,
                max_per_cell=2,
                seed=20260708,
                root_origins={"fresh"},
                default_starter_tile=1536,
                require_no_start_1536=True,
                out_dir=out_dir,
                progress_every=0,
            )

            self.assertEqual(payload["summary"]["cases_selected"], 1)
            self.assertEqual(payload["summary"]["target_records"], 1)
            self.assertEqual(payload["target_records"][0]["features"]["raw_count_768"], 2)
            self.assertEqual(payload["target_records"][0]["features"]["raw_count_1536"], 1)
            self.assertEqual(payload["target_records"][0]["features"]["masked_count_1536"], 0)

    def test_raw_adjacent_768_target_requires_adjacency(self):
        adjacent = fixed_starter_support_state(2)
        separated = custom_state(
            [
                [1536, 768, 0, 768],
                [0, 0, 0, 0],
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
                            record_payload(adjacent, record_id="adjacent"),
                            record_payload(separated, record_id="separated"),
                        ]
                    }
                )
            )

            payload = run_frontier(
                records_json=[records_path],
                target="raw_adjacent_768_no_1536",
                max_depth=0,
                repeats_per_action=1,
                max_starts=0,
                min_start_raw_768=1,
                min_raw_768=1,
                max_nodes_per_depth=8,
                max_per_cell=2,
                seed=20260708,
                root_origins={"fresh"},
                default_starter_tile=1536,
                require_no_start_1536=True,
                out_dir=out_dir,
                progress_every=0,
            )

            self.assertEqual(payload["summary"]["cases_selected"], 2)
            self.assertEqual(payload["summary"]["target_records"], 1)
            self.assertTrue(payload["target_records"][0]["features"]["raw_has_adjacent_768"])

    def test_saturate_target_roots_stops_expanding_target_start_roots(self):
        target_start = fixed_starter_support_state(2)
        below_target = custom_state(
            [
                [1536, 768, 0, 0],
                [0, 0, 0, 0],
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
                            record_payload(target_start, record_id="target-start", seed=1),
                            record_payload(below_target, record_id="below-target", seed=2),
                        ]
                    }
                )
            )

            payload = run_frontier(
                records_json=[records_path],
                target="raw_two_768_no_1536",
                max_depth=1,
                repeats_per_action=1,
                max_starts=0,
                min_start_raw_768=1,
                min_raw_768=1,
                max_nodes_per_depth=8,
                max_per_cell=2,
                seed=20260708,
                root_origins={"fresh"},
                default_starter_tile=1536,
                require_no_start_1536=True,
                out_dir=out_dir,
                progress_every=0,
                saturate_target_roots=True,
            )

            self.assertEqual(payload["summary"]["cases_selected"], 2)
            self.assertEqual(payload["summary"]["target_by_depth"]["0"], 1)
            self.assertEqual(payload["summary"]["target_saturated_root_seeds"], ["1"])
            self.assertEqual(payload["depth_rows"][0]["saturated_root"], 1)
            self.assertNotIn("1", {str(row.get("root_key")) for row in payload["transitions"]})

    def test_raw_duplicate_1536_target_counts_first_built_tile_with_fixed_starter(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            tmp_path = Path(tmp)
            records_path = tmp_path / "records.json"
            out_dir = tmp_path / "frontier"
            records_path.write_text(json.dumps({"records": [record_payload(fixed_starter_support_state(2, built_1536=True))]}))

            payload = run_frontier(
                records_json=[records_path],
                target="raw_duplicate_1536",
                max_depth=0,
                repeats_per_action=1,
                max_starts=0,
                min_start_raw_768=2,
                min_raw_768=2,
                max_nodes_per_depth=8,
                max_per_cell=2,
                seed=20260708,
                root_origins={"fresh"},
                default_starter_tile=1536,
                require_no_start_1536=False,
                out_dir=out_dir,
                progress_every=0,
            )

            self.assertEqual(payload["summary"]["cases_selected"], 1)
            self.assertEqual(payload["summary"]["target_records"], 1)
            self.assertEqual(payload["target_records"][0]["features"]["raw_count_1536"], 2)
            self.assertEqual(payload["target_records"][0]["features"]["masked_count_1536"], 1)

    def test_four_adjacent_target_requires_adjacent_support(self):
        adjacent = custom_state(
            [
                [768, 768, 0, 0],
                [768, 0, 768, 0],
                [3072, 0, 0, 0],
                [3, 6, 12, 24],
            ]
        )
        separated = custom_state(
            [
                [768, 0, 768, 0],
                [0, 0, 0, 0],
                [768, 0, 768, 0],
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
                            record_payload(adjacent, record_id="adjacent"),
                            record_payload(separated, record_id="separated"),
                        ]
                    }
                )
            )

            payload = run_frontier(
                records_json=[records_path],
                target="raw_four_adjacent_768_no_1536",
                max_depth=0,
                repeats_per_action=1,
                max_starts=0,
                min_start_raw_768=2,
                min_raw_768=2,
                max_nodes_per_depth=8,
                max_per_cell=2,
                seed=20260708,
                root_origins={"fresh"},
                default_starter_tile=1536,
                require_no_start_1536=True,
                out_dir=out_dir,
                progress_every=0,
            )

            self.assertEqual(payload["summary"]["cases_selected"], 2)
            self.assertEqual(payload["summary"]["target_records"], 1)
            self.assertTrue(payload["target_records"][0]["features"]["raw_has_adjacent_768"])

    def test_filters_unknown_roots_and_starting_1536(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            tmp_path = Path(tmp)
            records_path = tmp_path / "records.json"
            out_dir = tmp_path / "frontier"
            records_path.write_text(
                json.dumps(
                    {
                        "records": [
                            record_payload(accumulation_state(2), record_id="fresh-ok"),
                            record_payload(accumulation_state(3), record_id="unknown", root_origin="unknown"),
                            record_payload(fixed_starter_support_state(3, built_1536=True), record_id="has-1536"),
                        ]
                    }
                )
            )

            payload = run_frontier(
                records_json=[records_path],
                target="raw_three_768_no_1536",
                max_depth=1,
                repeats_per_action=1,
                max_starts=0,
                min_start_raw_768=2,
                min_raw_768=2,
                max_nodes_per_depth=8,
                max_per_cell=2,
                seed=20260708,
                root_origins={"fresh"},
                default_starter_tile=1536,
                require_no_start_1536=True,
                out_dir=out_dir,
                progress_every=0,
            )

            self.assertEqual(payload["summary"]["cases_selected"], 1)
            self.assertEqual(payload["summary"]["rejected"]["root_origin:unknown"], 1)
            self.assertEqual(payload["summary"]["rejected"]["start_has_1536"], 1)


if __name__ == "__main__":
    unittest.main()
