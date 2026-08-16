import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

from threes_rl.record_replay import state_payload
from threes_rl.sim import SimState, ThreesSim, preview_from_label
from threes_rl.transition_window_reservoir import collect_transition_window_records, run_from_args


def state_with_max(tile: int, move_count: int) -> SimState:
    return SimState(
        board=np.asarray(
            [
                [1536, tile, 0, 0],
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
        max_tile=max(1536, tile),
        move_count=move_count,
        game_over=False,
    )


def write_replay(path: Path) -> None:
    sim = ThreesSim(np.random.default_rng(7), starter_tile=1536)
    states = [
        state_with_max(1536, 10),
        state_with_max(1536, 11),
        state_with_max(3072, 12),
        state_with_max(3072, 13),
        state_with_max(3072, 14),
    ]
    replay = {
        "policy": "human_observed",
        "seed": 7,
        "starter_tile": 1536,
        "frames": [
            {"index": 0, "state": state_payload(states[0], sim), "move": None},
            {"index": 1, "state": state_payload(states[1], sim), "move": {"action": "up"}},
            {"index": 2, "state": state_payload(states[2], sim), "move": {"action": "left"}},
            {"index": 3, "state": state_payload(states[3], sim), "move": {"action": "down"}},
            {"index": 4, "state": state_payload(states[4], sim), "move": {"action": "right"}},
        ],
    }
    path.write_text(json.dumps(replay))


class TransitionWindowReservoirTests(unittest.TestCase):
    def test_collects_success_and_failure_control_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            replay_path = Path(tmp) / "replay.json"
            write_replay(replay_path)

            records, summary = collect_transition_window_records(
                [replay_path],
                targets=[3072, 6144],
                window_size=2,
                include_failures=True,
            )

        self.assertEqual(summary["by_target_outcome"]["3072"]["success"], 2)
        self.assertEqual(summary["by_target_outcome"]["6144"]["failure"], 2)
        success = [record for record in records if record["target_tile"] == 3072 and record["outcome"] == "success"]
        failure = [record for record in records if record["target_tile"] == 6144 and record["outcome"] == "failure"]
        self.assertEqual([record["moves_to_promotion"] for record in success], [2, 1])
        self.assertEqual([record["source_next_action"] for record in success], ["up", "left"])
        self.assertEqual([record["control_tile"] for record in failure], [3072, 3072])
        self.assertTrue(all(record["moves_to_terminal"] is not None for record in failure))
        self.assertIn("legal_count", success[0]["features"])
        self.assertIn("raw_count_768", success[0]["features"])
        self.assertIn("raw_has_adjacent_768", success[0])

    def test_run_from_args_writes_records_json(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            tmp_path = Path(tmp)
            replay_path = tmp_path / "replay.json"
            out_dir = tmp_path / "windows"
            write_replay(replay_path)

            payload = run_from_args(
                Namespace(
                    replay_json=[[replay_path]],
                    replay_glob=[],
                    targets="3072",
                    window_size=2,
                    no_failures=False,
                    starter="1536",
                    max_records=0,
                    out_dir=out_dir,
                )
            )

            self.assertEqual(payload["summary"]["records"], 2)
            self.assertTrue((out_dir / "transition_windows.json").exists())
            self.assertTrue((out_dir / "records.json").exists())
            self.assertTrue((out_dir / "transition_windows.html").exists())


if __name__ == "__main__":
    unittest.main()
