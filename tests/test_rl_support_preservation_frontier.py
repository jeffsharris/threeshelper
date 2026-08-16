import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from threes_rl.record_replay import state_payload
from threes_rl.sim import SimState, ThreesSim, preview_from_label
from threes_rl.support_preservation_frontier import run_frontier


def adjacent_768_state() -> SimState:
    return SimState(
        board=np.asarray(
            [
                [768, 768, 0, 0],
                [3, 6, 12, 24],
                [48, 96, 192, 384],
                [1, 2, 3, 6],
            ],
            dtype=np.int32,
        ),
        preview=preview_from_label("gray"),
        small_counts={"red": 4, "blue": 4, "gray": 4},
        small_pos=8,
        small_seen_total=40,
        span_small_pos=0,
        large_pending=False,
        max_tile=768,
        move_count=80,
        game_over=False,
    )


def support_with_1536_state() -> SimState:
    return SimState(
        board=np.asarray(
            [
                [768, 0, 0, 0],
                [768, 0, 0, 0],
                [0, 0, 768, 768],
                [3, 6, 12, 24],
            ],
            dtype=np.int32,
        ),
        preview=preview_from_label("gray"),
        small_counts={"red": 4, "blue": 4, "gray": 4},
        small_pos=8,
        small_seen_total=40,
        span_small_pos=0,
        large_pending=False,
        max_tile=768,
        move_count=80,
        game_over=False,
    )


def four_768_state() -> SimState:
    return SimState(
        board=np.asarray(
            [
                [768, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 768, 768, 768],
                [3, 6, 12, 24],
            ],
            dtype=np.int32,
        ),
        preview=preview_from_label("gray"),
        small_counts={"red": 4, "blue": 4, "gray": 4},
        small_pos=8,
        small_seen_total=40,
        span_small_pos=0,
        large_pending=False,
        max_tile=768,
        move_count=80,
        game_over=False,
    )


def record_payload() -> dict:
    sim = ThreesSim(np.random.default_rng(7), starter_tile=1536)
    state = adjacent_768_state()
    return {
        "id": "adjacent-768-case",
        "starter_tile": 1536,
        "source_replay": "fixture/source.json",
        "source_seed": 7,
        "source_frame_index": 80,
        "source_policy": "fixture",
        "root_origin": "fresh",
        "root_replay": "fixture/root.json",
        "root_seed": 7,
        "root_frame_index": 0,
        "root_policy": "fixture",
        "state": state_payload(state, sim),
    }


def target_record_payload() -> dict:
    sim = ThreesSim(np.random.default_rng(7), starter_tile=1536)
    state = support_with_1536_state()
    return {
        **record_payload(),
        "id": "support-target-case",
        "state": state_payload(state, sim),
    }


def four_target_record_payload() -> dict:
    sim = ThreesSim(np.random.default_rng(7), starter_tile=1536)
    state = four_768_state()
    return {
        **record_payload(),
        "id": "four-768-target-case",
        "state": state_payload(state, sim),
    }


class SupportPreservationFrontierTests(unittest.TestCase):
    def test_run_frontier_archives_preserved_prefixes(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            tmp_path = Path(tmp)
            records_path = tmp_path / "records.json"
            out_dir = tmp_path / "frontier"
            records_path.write_text(json.dumps({"records": [record_payload()]}))

            payload = run_frontier(
                records_json=[records_path],
                max_depth=1,
                repeats_per_action=1,
                max_starts=0,
                max_nodes_per_depth=8,
                max_per_cell=2,
                seed=20260708,
                root_origins={"fresh"},
                default_starter_tile=1536,
                require_start_adjacent_768=True,
                require_no_start_1536=True,
                out_dir=out_dir,
                progress_every=0,
            )

            self.assertEqual(payload["summary"]["cases_selected"], 1)
            self.assertGreater(payload["summary"]["archive_records"], 1)
            self.assertGreaterEqual(payload["summary"]["deepest_preserved_depth"], 1)
            self.assertEqual(
                len({record["id"] for record in payload["records"]}),
                len(payload["records"]),
            )
            self.assertTrue(any((row.get("features") or {}).get("depth") == 1 for row in payload["records"]))
            self.assertTrue((out_dir / "support_preservation_frontier.json").exists())
            self.assertTrue((out_dir / "records.json").exists())
            self.assertTrue((out_dir / "support_preservation_frontier.html").exists())

    def test_run_frontier_archives_support_with_1536_targets(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            tmp_path = Path(tmp)
            records_path = tmp_path / "records.json"
            out_dir = tmp_path / "frontier"
            records_path.write_text(json.dumps({"records": [target_record_payload()]}))

            payload = run_frontier(
                records_json=[records_path],
                max_depth=1,
                repeats_per_action=1,
                max_starts=0,
                max_nodes_per_depth=8,
                max_per_cell=2,
                seed=20260708,
                root_origins={"fresh"},
                default_starter_tile=1536,
                require_start_adjacent_768=True,
                require_no_start_1536=True,
                out_dir=out_dir,
                progress_every=0,
            )

            self.assertGreater(payload["summary"]["target_records"], 0)
            self.assertGreater(payload["summary"]["support_with_1536_transitions"], 0)
            self.assertTrue((out_dir / "target_records.json").exists())

    def test_run_frontier_archives_four_768_no_1536_targets(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            tmp_path = Path(tmp)
            records_path = tmp_path / "records.json"
            out_dir = tmp_path / "frontier"
            records_path.write_text(json.dumps({"records": [four_target_record_payload()]}))

            payload = run_frontier(
                records_json=[records_path],
                max_depth=1,
                repeats_per_action=1,
                max_starts=0,
                max_nodes_per_depth=8,
                max_per_cell=2,
                seed=20260708,
                root_origins={"fresh"},
                default_starter_tile=1536,
                require_start_adjacent_768=True,
                require_no_start_1536=True,
                out_dir=out_dir,
                progress_every=0,
            )

            self.assertGreater(payload["summary"]["target_records"], 0)
            self.assertTrue(
                any((record.get("features") or {}).get("raw_count_768", 0) >= 4 for record in payload["target_records"])
            )


if __name__ == "__main__":
    unittest.main()
