import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

from threes_rl.record_replay import state_payload
from threes_rl.sim import SimState, ThreesSim, preview_from_label
from threes_rl.transition_reachability_score import run_from_args


def score_state(top_left: int, empty_count: int) -> SimState:
    board = np.asarray(
        [
            [top_left, 3072, 1536, 768],
            [3, 3, 6, 12],
            [24, 48, 96, 192],
            [384, 1, 2, 3],
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


def make_record(sim: ThreesSim, idx: int, *, success: bool | None, top_left: int, empty_count: int) -> dict:
    state = score_state(top_left, empty_count)
    record = {
        "id": f"r{idx}",
        "target_tile": 6144,
        "source_replay": f"replay-{idx // 2}",
        "source_seed": idx,
        "source_frame_index": idx * 10,
        "starter_tile": 1536,
        "features": {
            "score_minus_starter": 1000 + idx,
            "empty_count": empty_count,
            "top_left": top_left,
            "preview": "gray",
            "corner_risk": "medium_corner_risk" if top_left == 3072 else "high_corner_risk",
            "stratum": "endgame_3072p/medium_corner_risk"
            if top_left == 3072
            else "endgame_3072p/high_corner_risk",
        },
        "state": state_payload(state, sim),
    }
    if success is not None:
        record["outcome"] = "success" if success else "failure"
    return record


class TransitionReachabilityScoreTests(unittest.TestCase):
    def test_scores_candidates_and_writes_top_records(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            tmp_path = Path(tmp)
            sim = ThreesSim(np.random.default_rng(1), starter_tile=1536)
            train_records = []
            for idx in range(20):
                success = idx % 2 == 0
                train_records.append(
                    make_record(
                        sim,
                        idx,
                        success=success,
                        top_left=3072 if success else 3,
                        empty_count=2 if success else 0,
                    )
                )
            candidate_records = [
                make_record(sim, 100, success=None, top_left=3072, empty_count=2),
                make_record(sim, 101, success=None, top_left=3, empty_count=0),
            ]
            train_path = tmp_path / "train.json"
            candidate_path = tmp_path / "candidates.json"
            out_dir = tmp_path / "scores"
            train_path.write_text(json.dumps({"records": train_records}))
            candidate_path.write_text(json.dumps({"records": candidate_records}))

            payload = run_from_args(
                Namespace(
                    train_json=[[train_path]],
                    candidate_json=[[candidate_path]],
                    target_tile=6144,
                    candidate_min_feature=[],
                    top_n=2,
                    bottom_n=1,
                    out_dir=out_dir,
                )
            )

            self.assertEqual(payload["summary"]["candidates"], 2)
            self.assertEqual(len(payload["top_candidates"]), 2)
            self.assertGreaterEqual(
                payload["top_candidates"][0]["reachability_prob"],
                payload["top_candidates"][1]["reachability_prob"],
            )
            top_payload = json.loads((out_dir / "top_records.json").read_text())
            self.assertEqual(len(top_payload["records"]), 2)
            self.assertIn("state", top_payload["records"][0])
            bottom_payload = json.loads((out_dir / "bottom_records.json").read_text())
            self.assertEqual(len(bottom_payload["records"]), 1)
            self.assertIn("state", bottom_payload["records"][0])
            self.assertTrue((out_dir / "reachability_scores.html").exists())

    def test_candidate_min_feature_filters_scored_rows(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            tmp_path = Path(tmp)
            sim = ThreesSim(np.random.default_rng(1), starter_tile=1536)
            train_records = []
            for idx in range(20):
                success = idx % 2 == 0
                train_records.append(
                    make_record(
                        sim,
                        idx,
                        success=success,
                        top_left=3072 if success else 3,
                        empty_count=2 if success else 0,
                    )
                )
            candidate_records = [
                make_record(sim, 100, success=None, top_left=3072, empty_count=2),
                make_record(sim, 101, success=None, top_left=3, empty_count=0),
            ]
            train_path = tmp_path / "train.json"
            candidate_path = tmp_path / "candidates.json"
            out_dir = tmp_path / "scores"
            train_path.write_text(json.dumps({"records": train_records}))
            candidate_path.write_text(json.dumps({"records": candidate_records}))

            payload = run_from_args(
                Namespace(
                    train_json=[[train_path]],
                    candidate_json=[[candidate_path]],
                    target_tile=6144,
                    candidate_min_feature=["top_left=3072"],
                    top_n=2,
                    out_dir=out_dir,
                )
            )

            self.assertEqual(payload["raw_candidates"], 2)
            self.assertEqual(payload["filtered_candidates"], 1)
            self.assertEqual(payload["summary"]["candidates"], 1)
            self.assertEqual(payload["top_candidates"][0]["top_left"], 3072.0)

    def test_candidate_max_and_equal_feature_filters_scored_rows(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            tmp_path = Path(tmp)
            sim = ThreesSim(np.random.default_rng(1), starter_tile=1536)
            train_records = []
            for idx in range(20):
                success = idx % 2 == 0
                train_records.append(
                    make_record(
                        sim,
                        idx,
                        success=success,
                        top_left=3072 if success else 3,
                        empty_count=2 if success else 0,
                    )
                )
            candidate_records = [
                make_record(sim, 100, success=None, top_left=3072, empty_count=1),
                make_record(sim, 101, success=None, top_left=3, empty_count=2),
                make_record(sim, 102, success=None, top_left=3072, empty_count=2),
            ]
            candidate_records[0]["features"]["preview"] = "red"
            candidate_records[1]["features"]["preview"] = "red"
            candidate_records[2]["features"]["preview"] = "blue"
            train_path = tmp_path / "train.json"
            candidate_path = tmp_path / "candidates.json"
            out_dir = tmp_path / "scores"
            train_path.write_text(json.dumps({"records": train_records}))
            candidate_path.write_text(json.dumps({"records": candidate_records}))

            payload = run_from_args(
                Namespace(
                    train_json=[[train_path]],
                    candidate_json=[[candidate_path]],
                    target_tile=6144,
                    candidate_min_feature=[],
                    candidate_max_feature=["empty_count=1"],
                    candidate_feature_equals=["preview=red"],
                    top_n=3,
                    out_dir=out_dir,
                )
            )

            self.assertEqual(payload["raw_candidates"], 3)
            self.assertEqual(payload["filtered_candidates"], 1)
            self.assertEqual(payload["summary"]["candidates"], 1)
            self.assertEqual(payload["top_candidates"][0]["preview"], "red")
            self.assertEqual(payload["top_candidates"][0]["empty_count"], 1.0)

    def test_top_source_replay_limit_diversifies_exported_records(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            tmp_path = Path(tmp)
            sim = ThreesSim(np.random.default_rng(1), starter_tile=1536)
            train_records = []
            for idx in range(20):
                success = idx % 2 == 0
                train_records.append(
                    make_record(
                        sim,
                        idx,
                        success=success,
                        top_left=3072 if success else 3,
                        empty_count=2 if success else 0,
                    )
                )
            candidate_records = [
                make_record(sim, 100, success=None, top_left=3072, empty_count=2),
                make_record(sim, 101, success=None, top_left=3072, empty_count=1),
                make_record(sim, 102, success=None, top_left=3072, empty_count=0),
                make_record(sim, 103, success=None, top_left=3, empty_count=0),
            ]
            candidate_records[0]["source_replay"] = "replay-a"
            candidate_records[1]["source_replay"] = "replay-a"
            candidate_records[2]["source_replay"] = "replay-b"
            candidate_records[3]["source_replay"] = "replay-c"
            train_path = tmp_path / "train.json"
            candidate_path = tmp_path / "candidates.json"
            out_dir = tmp_path / "scores"
            train_path.write_text(json.dumps({"records": train_records}))
            candidate_path.write_text(json.dumps({"records": candidate_records}))

            payload = run_from_args(
                Namespace(
                    train_json=[[train_path]],
                    candidate_json=[[candidate_path]],
                    target_tile=6144,
                    candidate_min_feature=[],
                    top_n=3,
                    top_source_replay_limit=1,
                    top_source_seed_limit=0,
                    top_frame_min_gap=0,
                    out_dir=out_dir,
                )
            )

            replays = [row["source_replay"] for row in payload["top_candidates"]]
            self.assertEqual(len(replays), 3)
            self.assertEqual(len(set(replays)), 3)
            top_payload = json.loads((out_dir / "top_records.json").read_text())
            top_replays = [record["source_replay"] for record in top_payload["records"]]
            self.assertEqual(len(top_replays), 3)
            self.assertEqual(len(set(top_replays)), 3)


if __name__ == "__main__":
    unittest.main()
