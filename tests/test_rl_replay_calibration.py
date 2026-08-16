import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from threes_rl.record_replay import state_payload
from threes_rl.sim import DIRECTION_NAMES, SimState, ThreesSim, preview_from_label, score_board, simulate_base_move
from threes_rl.train_replay_calibration import ReplayCalibrationConfig, calibrate, replay_examples


def simple_state() -> SimState:
    return SimState(
        board=np.asarray(
            [
                [1, 2, 0, 0],
                [3, 3, 0, 0],
                [0, 0, 0, 0],
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
        max_tile=3,
        move_count=0,
        game_over=False,
    )


class ReplayCalibrationTests(unittest.TestCase):
    def write_simple_replay(self, path: Path) -> None:
        state = simple_state()
        sim = ThreesSim(np.random.default_rng(7), starter_tile=1536)
        action = int(sim.legal_actions(state)[0])
        afterstate, eligible = simulate_base_move(state.board, action)
        self.assertTrue(eligible)
        replay = {
            "policy": "fixture",
            "seed": 7,
            "starter_tile": 1536,
            "final_score": int(score_board(afterstate) + 100),
            "frames": [
                {"index": 0, "state": state_payload(state, sim), "move": None},
                {
                    "index": 1,
                    "state": state_payload(state, sim),
                    "move": {"action": DIRECTION_NAMES[action]},
                },
            ],
        }
        path.write_text(json.dumps(replay))

    def test_replay_examples_compute_remaining_return_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "replay.json"
            self.write_simple_replay(path)

            examples = replay_examples(path, starter_tile=1536)

        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0].target, 100.0)
        self.assertEqual(examples[0].phase, "early_lt384")

    def test_replay_examples_can_filter_phase(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "replay.json"
            self.write_simple_replay(path)

            examples = replay_examples(path, starter_tile=1536, phase_filter={"late_1536"})

        self.assertEqual(examples, [])

    def test_calibrate_writes_checkpoint_and_summary(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            replay_path = Path(tmp) / "replay.json"
            self.write_simple_replay(replay_path)
            run_name = f"replay_calibration_smoke_{Path(tmp).name}"
            config = ReplayCalibrationConfig(
                run_name=run_name,
                replay_json=[str(replay_path)],
                epochs=2,
                pattern_set="tiny",
                alpha=0.01,
                progress_every=1,
            )

            checkpoint = calibrate(config)
            run_dir = checkpoint.parent
            summary = json.loads((run_dir / "summary.json").read_text())

            self.assertEqual(summary["examples"], 1)
            self.assertEqual(summary["updates"], 2)
            self.assertTrue((checkpoint / "meta.json").exists())
            shutil.rmtree(run_dir)


if __name__ == "__main__":
    unittest.main()
