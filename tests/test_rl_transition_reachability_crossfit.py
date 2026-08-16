import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

from threes_rl.record_replay import state_payload
from threes_rl.sim import SimState, ThreesSim, preview_from_label
from threes_rl.transition_reachability_crossfit import run_from_args


def make_state(top_left: int, empty_count: int, move_count: int) -> SimState:
    board = np.asarray(
        [
            [top_left, 3072, 1536, 768],
            [1536, 384, 192, 96],
            [48, 24, 12, 6],
            [3, 2, 1, 3],
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
        move_count=move_count,
        game_over=False,
    )


def make_record(sim: ThreesSim, idx: int, *, group: str, success: bool) -> dict:
    state = make_state(3072 if success else 3, 2 if success else 0, 100 + idx)
    group_seed = {"a": 1, "b": 2, "c": 3}[group]
    return {
        "id": f"r{idx}",
        "target_tile": 1536,
        "outcome": "success" if success else "failure",
        "source_replay": f"continuation-{idx}.json",
        "original_source_replay": f"{group}/replay.json",
        "original_source_seed": group_seed,
        "source_seed": idx,
        "source_frame_index": idx * 10,
        "starter_tile": 1536,
        "features": {
            "score_minus_starter": 1000 + idx,
            "empty_count": 2 if success else 0,
            "top_left": 3072 if success else 3,
            "preview": "gray",
            "corner_risk": "high_corner_risk",
            "stratum": "endgame_3072p/high_corner_risk",
            "raw_count_1536": 2,
            "raw_has_adjacent_1536": False,
        },
        "state": state_payload(state, sim),
    }


class TransitionReachabilityCrossfitTests(unittest.TestCase):
    def test_crossfit_scores_and_writes_top_bottom_records(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            tmp_path = Path(tmp)
            sim = ThreesSim(np.random.default_rng(1), starter_tile=1536)
            records = []
            idx = 0
            for group in ("a", "b", "c"):
                for _ in range(4):
                    records.append(make_record(sim, idx, group=group, success=True))
                    idx += 1
                    records.append(make_record(sim, idx, group=group, success=False))
                    idx += 1
            records_path = tmp_path / "records.json"
            out_dir = tmp_path / "crossfit"
            records_path.write_text(json.dumps({"records": records}))

            payload = run_from_args(
                Namespace(
                    records_json=[[records_path]],
                    target_tile=1536,
                    no_target_filter=False,
                    group_by="original-replay",
                    train_cap_per_source_group_outcome=0,
                    candidate_min_feature=["raw_count_1536=2"],
                    candidate_max_feature=[],
                    candidate_feature_equals=["raw_has_adjacent_1536=0"],
                    top_n=3,
                    bottom_n=2,
                    top_source_replay_limit=0,
                    top_source_seed_limit=0,
                    top_frame_min_gap=0,
                    out_dir=out_dir,
                )
            )

            self.assertEqual(payload["summary"]["groups"], 3)
            self.assertEqual(payload["summary"]["scored_records"], len(records))
            self.assertEqual(len(payload["top_candidates"]), 3)
            self.assertEqual(len(payload["bottom_candidates"]), 2)
            self.assertTrue((out_dir / "crossfit_scores.html").exists())
            top_payload = json.loads((out_dir / "top_records.json").read_text())
            bottom_payload = json.loads((out_dir / "bottom_records.json").read_text())
            self.assertEqual(len(top_payload["records"]), 3)
            self.assertEqual(len(bottom_payload["records"]), 2)
            self.assertIn("crossfit_group", top_payload["records"][0])


if __name__ == "__main__":
    unittest.main()
