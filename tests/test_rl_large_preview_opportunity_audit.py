import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

from threes_rl.large_preview_opportunity_audit import audit_replay, run_from_args
from threes_rl.record_replay import state_payload
from threes_rl.sim import SimState, ThreesSim, preview_from_label, score_board


def make_state(board, preview_label="blue", candidates=(), move_count=100):
    arr = np.asarray(board, dtype=np.int32)
    return SimState(
        board=arr,
        preview=preview_from_label(preview_label, candidates),
        small_counts={"red": 4, "blue": 4, "gray": 4},
        small_pos=0,
        small_seen_total=0,
        span_small_pos=0,
        large_pending=bool(candidates),
        max_tile=int(arr.max(initial=0)),
        move_count=move_count,
        game_over=False,
    )


def make_replay():
    sim = ThreesSim(np.random.default_rng(7), starter_tile=1536)
    start = make_state(
        [
            [1536, 3072, 768, 768],
            [384, 192, 96, 48],
            [24, 12, 6, 3],
            [3, 2, 1, 0],
        ],
        move_count=400,
    )
    large_next = make_state(
        [
            [1536, 3072, 768, 768],
            [384, 192, 96, 48],
            [24, 12, 6, 3],
            [3, 2, 1, 0],
        ],
        preview_label="large_candidates",
        candidates=(192, 384, 768),
        move_count=401,
    )
    built_1536 = make_state(
        [
            [1536, 3072, 768, 0],
            [384, 192, 1536, 48],
            [24, 12, 6, 3],
            [3, 2, 1, 0],
        ],
        preview_label="red",
        move_count=402,
    )
    return {
        "seed": 1234,
        "starter_tile": 1536,
        "start_case_id": "fixture-start",
        "start_score": score_board(start.board),
        "final_score": score_board(built_1536.board),
        "final_score_delta": score_board(built_1536.board) - score_board(start.board),
        "final_max_tile_excl_starter": 3072,
        "frames": [
            {"index": 0, "state": state_payload(start, sim), "move": None},
            {
                "index": 1,
                "state": state_payload(large_next, sim),
                "move": {"action": "right", "inserted_value": 1},
            },
            {
                "index": 2,
                "state": state_payload(built_1536, sim),
                "move": {"action": "up", "inserted_value": 2},
            },
        ],
    }


class LargePreviewOpportunityAuditTests(unittest.TestCase):
    def test_audit_replay_links_large_preview_to_raw_1536_milestone(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            path = Path(tmp) / "replay.json"
            path.write_text(json.dumps(make_replay()))

            record = audit_replay(path)

            self.assertEqual(record["preview_after_first_label"], "large_candidates")
            self.assertEqual(record["preview_after_first_max_candidate"], 768)
            self.assertTrue(record["preview_after_first_ge_768"])
            self.assertTrue(record["raw_duplicate_1536"])
            self.assertFalse(record["reached_6144"])

    def test_run_from_args_writes_artifacts(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            tmp_path = Path(tmp)
            replay = tmp_path / "replay.json"
            out_dir = tmp_path / "out"
            replay.write_text(json.dumps(make_replay()))

            payload = run_from_args(
                Namespace(
                    replay_json=[[replay]],
                    replay_glob=None,
                    out_dir=out_dir,
                )
            )

            self.assertEqual(payload["summary"]["records"], 1)
            self.assertEqual(payload["summary"]["preview_after_first_ge_768"], 1)
            self.assertEqual(payload["summary"]["raw_duplicate_1536"], 1)
            self.assertTrue((out_dir / "large_preview_opportunity_audit.json").exists())
            self.assertTrue((out_dir / "records.json").exists())
            self.assertTrue((out_dir / "large_preview_opportunity_audit.html").exists())


if __name__ == "__main__":
    unittest.main()
