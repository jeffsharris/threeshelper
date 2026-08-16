import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

from threes_rl.high_board_reservoir import collect_reservoir_records, run_from_args
from threes_rl.record_replay import state_payload
from threes_rl.sim import SimState, ThreesSim, preview_from_label


def low_state() -> SimState:
    return SimState(
        board=np.asarray(
            [
                [1536, 96, 0, 0],
                [48, 24, 12, 6],
                [3, 2, 1, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.int32,
        ),
        preview=preview_from_label("gray"),
        small_counts={"red": 4, "blue": 4, "gray": 4},
        small_pos=0,
        small_seen_total=0,
        span_small_pos=0,
        large_pending=False,
        max_tile=1536,
        move_count=10,
        game_over=False,
    )


def high_state() -> SimState:
    return SimState(
        board=np.asarray(
            [
                [1536, 3072, 0, 0],
                [768, 384, 192, 96],
                [48, 24, 12, 6],
                [3, 2, 1, 0],
            ],
            dtype=np.int32,
        ),
        preview=preview_from_label("blue"),
        small_counts={"red": 4, "blue": 4, "gray": 4},
        small_pos=0,
        small_seen_total=0,
        span_small_pos=0,
        large_pending=False,
        max_tile=3072,
        move_count=120,
        game_over=False,
    )


def write_replay(path: Path) -> None:
    sim = ThreesSim(np.random.default_rng(7), starter_tile=1536)
    payload = {
        "policy": "test-policy",
        "seed": 7,
        "starter_tile": 1536,
        "frames": [
            {"index": 0, "state": state_payload(low_state(), sim), "move": None},
            {"index": 1, "state": state_payload(high_state(), sim), "move": {"action": "up"}},
            {"index": 2, "state": state_payload(high_state(), sim), "move": {"action": "left"}},
        ],
    }
    path.write_text(json.dumps(payload))


class HighBoardReservoirTests(unittest.TestCase):
    def test_collect_reservoir_records_filters_and_keeps_source_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            replay_path = Path(tmp) / "replay.json"
            write_replay(replay_path)

            records, summary = collect_reservoir_records(
                [replay_path],
                min_tile=3072,
                phase_filter={"endgame_3072p"},
                corner_risk_filter=None,
                first_per="replay-stratum",
            )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["source_replay"], str(replay_path))
        self.assertEqual(records[0]["source_frame_index"], 1)
        self.assertEqual(records[0]["source_next_action"], "left")
        self.assertEqual(records[0]["phase"], "endgame_3072p")
        self.assertEqual(records[0]["max_tile_excl_starter"], 3072)
        self.assertEqual(summary["records"], 1)
        self.assertEqual(summary["max_tile_thresholds"][">=3072"], 1)
        self.assertGreaterEqual(summary["candidate_frames"], 1)

    def test_run_from_args_writes_reservoir_artifacts(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            tmp_path = Path(tmp)
            replay_path = tmp_path / "replay.json"
            out_dir = tmp_path / "reservoir"
            write_replay(replay_path)

            payload = run_from_args(
                Namespace(
                    replay_json=[[replay_path]],
                    replay_glob=[],
                    min_tile=3072,
                    phase_filter=["endgame"],
                    corner_risk_filter=None,
                    default_starter="1536",
                    first_per="replay-stratum",
                    max_records=10,
                    max_per_stratum=0,
                    sort_by="source",
                    out_dir=out_dir,
                )
            )

            self.assertEqual(payload["summary"]["records"], 1)
            self.assertTrue((out_dir / "reservoir.json").exists())
            self.assertTrue((out_dir / "records.json").exists())
            self.assertTrue((out_dir / "reservoir.html").exists())

    def test_run_from_args_replay_glob_is_recursive(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            tmp_path = Path(tmp)
            nested = tmp_path / "nested" / "deep"
            nested.mkdir(parents=True)
            replay_path = nested / "replay.json"
            out_dir = tmp_path / "reservoir"
            write_replay(replay_path)

            payload = run_from_args(
                Namespace(
                    replay_json=[],
                    replay_glob=[str(tmp_path / "**" / "replay.json")],
                    min_tile=3072,
                    phase_filter=["endgame"],
                    corner_risk_filter=None,
                    default_starter="1536",
                    first_per="replay-stratum",
                    max_records=10,
                    max_per_stratum=0,
                    sort_by="source",
                    out_dir=out_dir,
                )
            )

            self.assertEqual(payload["summary"]["source_replays"], 1)
            self.assertEqual(payload["summary"]["records"], 1)


if __name__ == "__main__":
    unittest.main()
