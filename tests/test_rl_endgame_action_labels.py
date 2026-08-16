import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

from threes_rl.continue_from_replays import StartCase
from threes_rl.endgame_action_labels import action_continuation_progress_key, run_from_args
from threes_rl.record_replay import state_payload
from threes_rl.sim import SimState, ThreesSim, preview_from_label


def endgame_state() -> SimState:
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
        small_counts={"blue": 4, "red": 4, "gray": 4},
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
    state = endgame_state()
    path.write_text(
        json.dumps(
            {
                "policy": "test",
                "seed": 7,
                "starter_tile": 1536,
                "final_score": 123456,
                "frames": [
                    {"index": 0, "state": state_payload(state, sim), "move": None},
                ],
            }
        )
    )


class EndgameActionLabelTests(unittest.TestCase):
    def test_run_from_args_writes_labels_report_and_top_replays(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            tmp_path = Path(tmp)
            replay_path = tmp_path / "replay.json"
            out_dir = tmp_path / "labels"
            write_replay(replay_path)

            payload = run_from_args(
                Namespace(
                    policy="greedy",
                    replay_json=[[replay_path]],
                    starter="1536",
                    start_state_min_tile=3072,
                    phase_filter="endgame",
                    corner_risk_filter="high",
                    max_starts=1,
                    repeats=2,
                    max_moves=4,
                    seed=20260706,
                    stability_threshold=0.7,
                    keep_top_games=2,
                    progress_every=0,
                    checkpoint_continuations=False,
                    continuation_progress_json=None,
                    out_dir=out_dir,
                )
            )

            self.assertEqual(payload["summary"]["labels"], 1)
            self.assertEqual(payload["corner_risk_filter"], ["high_corner_risk"])
            self.assertGreater(payload["summary"]["continuations"], 0)
            self.assertIn("features", payload["labels"][0])
            self.assertIn("corner_risk", payload["labels"][0]["features"])
            self.assertIn("stratum", payload["labels"][0]["features"])
            self.assertTrue((out_dir / "endgame_action_labels.json").exists())
            self.assertTrue((out_dir / "endgame_action_labels.html").exists())
            self.assertTrue((out_dir / "top_delta_games" / "manifest.json").exists())
            self.assertTrue((out_dir / "top_games" / "manifest.json").exists())

    def test_run_from_args_resumes_checkpointed_action_continuations(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            tmp_path = Path(tmp)
            replay_path = tmp_path / "replay.json"
            out_dir = tmp_path / "labels"
            write_replay(replay_path)
            args = Namespace(
                policy="greedy",
                replay_json=[[replay_path]],
                starter="1536",
                start_state_min_tile=3072,
                phase_filter="endgame",
                corner_risk_filter="high",
                max_starts=1,
                repeats=1,
                max_moves=4,
                seed=20260706,
                stability_threshold=0.7,
                keep_top_games=2,
                progress_every=0,
                checkpoint_continuations=True,
                continuation_progress_json=None,
                out_dir=out_dir,
            )

            first = run_from_args(args)
            second = run_from_args(args)

            continuations = first["summary"]["continuations"]
            self.assertGreater(continuations, 0)
            self.assertEqual(first["summary"]["continuations_ran"], continuations)
            self.assertEqual(first["summary"]["continuations_resumed"], 0)
            self.assertEqual(second["summary"]["continuations"], continuations)
            self.assertEqual(second["summary"]["continuations_ran"], 0)
            self.assertEqual(second["summary"]["continuations_resumed"], continuations)
            self.assertEqual(first["labels"][0]["action_results"], second["labels"][0]["action_results"])
            self.assertTrue((out_dir / "action_continuation_progress.json").exists())

    def test_action_progress_key_includes_preview_and_tile_cycle(self):
        state_a = endgame_state()
        state_b = endgame_state()
        state_b.preview = preview_from_label("red")
        state_b.small_pos = 1
        case_a = StartCase(
            id="case",
            state=state_a,
            starter_tile=1536,
            source_replay="fixture.json",
            source_seed=1,
            frame_index=10,
            start_score=1,
            start_max_tile_excl_starter=3072,
            phase="endgame_3072p",
        )
        case_b = StartCase(
            id="case",
            state=state_b,
            starter_tile=1536,
            source_replay="fixture.json",
            source_seed=1,
            frame_index=10,
            start_score=1,
            start_max_tile_excl_starter=3072,
            phase="endgame_3072p",
        )

        key_a = action_continuation_progress_key(
            policy_name="greedy",
            start_case=case_a,
            action=0,
            repeat_index=0,
            seed=7,
            max_moves=2,
        )
        key_b = action_continuation_progress_key(
            policy_name="greedy",
            start_case=case_b,
            action=0,
            repeat_index=0,
            seed=7,
            max_moves=2,
        )

        self.assertNotEqual(key_a, key_b)


if __name__ == "__main__":
    unittest.main()
