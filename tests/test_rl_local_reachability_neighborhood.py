import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

from threes_rl.local_reachability_neighborhood import collect_neighborhood_records, run_from_args
from threes_rl.record_replay import state_payload
from threes_rl.sim import SimState, ThreesSim, preview_from_label


def make_state(top_left: int, *, empty_count: int = 1) -> SimState:
    board = np.asarray(
        [
            [top_left, 3072, 1536, 768],
            [384, 192, 96, 48],
            [24, 12, 6, 3],
            [2, 1, 3, 0],
        ],
        dtype=np.int32,
    )
    flat = board.reshape(-1)
    for idx in range(max(0, min(3, empty_count))):
        flat[-1 - idx] = 0
    return SimState(
        board=board,
        preview=preview_from_label("red"),
        small_counts={"red": 4, "blue": 4, "gray": 4},
        small_pos=0,
        small_seen_total=0,
        span_small_pos=0,
        large_pending=False,
        max_tile=int(board.max(initial=0)),
        move_count=100,
        game_over=False,
    )


def write_replay(path: Path) -> None:
    sim = ThreesSim(np.random.default_rng(7), starter_tile=1536)
    frames = []
    for idx, top_left in enumerate([1, 2, 3, 6, 12]):
        frames.append(
            {
                "index": idx,
                "state": state_payload(make_state(top_left, empty_count=1), sim),
                "move": None if idx == 0 else {"action": "left"},
            }
        )
    path.write_text(
        json.dumps(
            {
                "policy": "test-policy",
                "seed": 7,
                "starter_tile": 1536,
                "frames": frames,
            }
        )
    )


def make_train_record(sim: ThreesSim, idx: int, *, success: bool) -> dict:
    state = make_state(3072 if success else 3, empty_count=2 if success else 0)
    return {
        "id": f"train-{idx}",
        "target_tile": 1536,
        "outcome": "success" if success else "failure",
        "source_replay": f"train-{idx // 2}.json",
        "source_seed": idx,
        "source_frame_index": idx,
        "starter_tile": 1536,
        "features": {
            "score_minus_starter": 1000 + idx,
            "empty_count": 2 if success else 0,
            "preview": "red",
            "corner_risk": "medium_corner_risk" if success else "high_corner_risk",
            "stratum": "endgame_3072p/medium_corner_risk"
            if success
            else "endgame_3072p/high_corner_risk",
            "raw_count_768": 1,
            "raw_count_1536": 1,
        },
        "state": state_payload(state, sim),
    }


class LocalReachabilityNeighborhoodTests(unittest.TestCase):
    def test_collects_radius_around_anchor_frame(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            tmp_path = Path(tmp)
            replay_path = tmp_path / "replay.json"
            write_replay(replay_path)
            anchors = [
                {
                    "id": "anchor-a",
                    "source_replay": str(replay_path),
                    "source_frame_index": 2,
                    "starter_tile": 1536,
                }
            ]

            records, summary = collect_neighborhood_records(
                anchors,
                radius=1,
                min_tile=3072,
                phase_filter={"endgame_3072p"},
                default_starter_tile=1536,
            )

            self.assertEqual(summary["records"], 3)
            self.assertEqual([record["source_frame_index"] for record in records], [1, 2, 3])
            self.assertEqual([record["frame_offset"] for record in records], [-1, 0, 1])
            self.assertEqual(records[1]["anchor_id"], "anchor-a")
            self.assertEqual(records[1]["raw_count_1536"], 1)

    def test_run_from_args_scores_and_writes_artifacts(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            tmp_path = Path(tmp)
            replay_path = tmp_path / "replay.json"
            anchor_path = tmp_path / "anchors.json"
            train_path = tmp_path / "train.json"
            out_dir = tmp_path / "neighborhood"
            write_replay(replay_path)
            anchor_path.write_text(
                json.dumps(
                    {
                        "records": [
                            {
                                "id": "anchor-a",
                                "source_replay": str(replay_path),
                                "source_frame_index": 2,
                                "starter_tile": 1536,
                            },
                            {
                                "id": "anchor-b",
                                "source_replay": str(replay_path),
                                "source_frame_index": 3,
                                "starter_tile": 1536,
                            }
                        ]
                    }
                )
            )
            sim = ThreesSim(np.random.default_rng(9), starter_tile=1536)
            train_records = [
                make_train_record(sim, idx, success=(idx % 2 == 0))
                for idx in range(20)
            ]
            train_path.write_text(json.dumps({"records": train_records}))

            payload = run_from_args(
                Namespace(
                    anchor_json=[[anchor_path]],
                    train_json=[[train_path]],
                    target_tile=1536,
                    radius=1,
                    min_tile=3072,
                    phase_filter=["endgame"],
                    default_starter="1536",
                    top_n=5,
                    export_top_n=0,
                    export_source_replay_limit=0,
                    export_source_seed_limit=0,
                    export_frame_min_gap=0,
                    out_dir=out_dir,
                )
            )

            self.assertEqual(payload["summary"]["records"], 6)
            self.assertEqual(payload["summary"]["unique_records"], 4)
            self.assertEqual(payload["summary"]["selected_records"], 4)
            self.assertEqual(len(payload["by_anchor"]), 2)
            self.assertEqual(len(payload["top_records"]), 4)
            self.assertIn("reachability_prob", payload["top_records"][0])
            self.assertTrue((out_dir / "local_reachability_neighborhood.json").exists())
            self.assertTrue((out_dir / "records.json").exists())
            self.assertTrue((out_dir / "selected_records.json").exists())
            self.assertTrue((out_dir / "local_reachability_neighborhood.html").exists())
            records_payload = json.loads((out_dir / "records.json").read_text())
            self.assertIsInstance(records_payload, list)
            self.assertEqual(len(records_payload), 4)
            selected_payload = json.loads((out_dir / "selected_records.json").read_text())
            self.assertEqual(len(selected_payload), 4)


if __name__ == "__main__":
    unittest.main()
