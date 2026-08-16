import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

from threes_rl.first_action_milestone_audit import run_from_args
from threes_rl.record_replay import state_payload
from threes_rl.sim import SimState, ThreesSim, preview_from_label


def adjacent_1536_state(preview="blue") -> SimState:
    return SimState(
        board=np.asarray(
            [
                [1, 12, 3, 0],
                [6, 1536, 12, 1],
                [96, 1536, 96, 2],
                [6, 3072, 3, 2],
            ],
            dtype=np.int32,
        ),
        preview=preview_from_label(preview),
        small_counts={"blue": 4, "red": 4, "gray": 4},
        small_pos=0,
        small_seen_total=400,
        span_small_pos=11,
        large_pending=False,
        max_tile=3072,
        move_count=495,
        game_over=False,
    )


def write_replay(path: Path) -> None:
    sim = ThreesSim(np.random.default_rng(7), starter_tile=1536)
    state = adjacent_1536_state()
    path.write_text(
        json.dumps(
            {
                "policy": "fixture",
                "seed": 7,
                "starter_tile": 1536,
                "frames": [
                    {"index": 0, "state": state_payload(state, sim), "move": None},
                ],
            }
        )
    )


class FirstActionMilestoneAuditTests(unittest.TestCase):
    def test_run_from_args_writes_raw_1536_milestone_labels(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            tmp_path = Path(tmp)
            replay = tmp_path / "replay.json"
            out_dir = tmp_path / "audit"
            write_replay(replay)

            payload = run_from_args(
                Namespace(
                    policy="greedy",
                    replay_json=[[replay]],
                    starter="1536",
                    start_state_min_tile=3072,
                    phase_filter="endgame",
                    max_starts=1,
                    repeats=1,
                    max_moves=2,
                    seed=20260707,
                    target="raw_adjacent_1536",
                    progress_every=0,
                    checkpoint_continuations=False,
                    continuation_progress_json=None,
                    out_dir=out_dir,
                )
            )

            self.assertEqual(payload["summary"]["labels"], 1)
            self.assertGreater(payload["summary"]["continuations"], 0)
            self.assertEqual(payload["summary"]["starts_with_any_raw_adjacent_1536"], 1)
            self.assertTrue((out_dir / "first_action_milestone_audit.json").exists())
            self.assertTrue((out_dir / "records.json").exists())
            self.assertTrue((out_dir / "first_action_milestone_audit.html").exists())


if __name__ == "__main__":
    unittest.main()
