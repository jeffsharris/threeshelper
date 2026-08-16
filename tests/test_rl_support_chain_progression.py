import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from threes_rl.record_replay import state_payload
from threes_rl.sim import SimState, ThreesSim, preview_from_label
from threes_rl.support_chain_progression import analyze_replay, run


def make_state(board, move_count):
    return SimState(
        board=np.asarray(board, dtype=np.int32),
        preview=preview_from_label("gray"),
        small_counts={"red": 4, "blue": 4, "gray": 4},
        small_pos=0,
        small_seen_total=0,
        span_small_pos=0,
        large_pending=False,
        max_tile=int(np.max(board)),
        move_count=move_count,
        game_over=False,
    )


def write_replay(path: Path, states):
    sim = ThreesSim(np.random.default_rng(7), starter_tile=1536)
    frames = []
    for idx, state in enumerate(states):
        frames.append(
            {
                "index": idx,
                "state": state_payload(state, sim),
                "move": None if idx == 0 else {"action": "left"},
            }
        )
    path.write_text(
        json.dumps(
            {
                "policy": "test",
                "seed": 7,
                "starter_tile": 1536,
                "final_score": int(frames[-1]["state"]["score"]),
                "final_moves": int(states[-1].move_count),
                "frames": frames,
            }
        )
    )


class SupportChainProgressionTests(unittest.TestCase):
    def test_analyze_replay_keeps_raw_support_counts_alongside_masked(self):
        states = [
            make_state(
                [
                    [1536, 3072, 1536, 768],
                    [768, 768, 192, 96],
                    [48, 24, 12, 6],
                    [3, 2, 1, 0],
                ],
                100,
            ),
            make_state(
                [
                    [1536, 3072, 3072, 768],
                    [768, 768, 192, 96],
                    [48, 24, 12, 6],
                    [3, 2, 1, 0],
                ],
                101,
            ),
        ]
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            replay_path = Path(tmp) / "success.json"
            write_replay(replay_path, states)

            case = analyze_replay(replay_path)

        self.assertEqual(case["outcome"], "success")
        self.assertEqual(case["second_3072_event"]["frames_after_first_3072"], 1)
        self.assertEqual(case["milestones"]["first_raw_three_768"]["frames_after_first_3072"], 0)
        self.assertIsNone(case["milestones"]["first_raw_four_768"])
        self.assertEqual(case["pre_event_max_raw_count_768"], 3)
        self.assertEqual(case["milestones"]["first_raw_duplicate_1536"]["frames_after_first_3072"], 0)
        self.assertIsNone(case["milestones"]["first_masked_duplicate_1536"])
        self.assertEqual(case["pre_event_max_raw_duplicate_tile"], 1536)
        self.assertEqual(case["pre_event_max_masked_duplicate_tile"], 768)
        self.assertEqual(case["post_3072_max_raw_duplicate_tile"], 3072)

    def test_analyze_replay_tracks_four_768_material_milestone(self):
        states = [
            make_state(
                [
                    [1536, 3072, 768, 0],
                    [768, 384, 192, 96],
                    [48, 24, 12, 6],
                    [3, 2, 1, 0],
                ],
                100,
            ),
            make_state(
                [
                    [1536, 3072, 768, 768],
                    [768, 768, 192, 96],
                    [48, 24, 12, 6],
                    [3, 2, 1, 0],
                ],
                101,
            ),
        ]
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            replay_path = Path(tmp) / "material.json"
            write_replay(replay_path, states)

            case = analyze_replay(replay_path)

        self.assertEqual(case["outcome"], "failure")
        self.assertEqual(case["milestones"]["first_raw_three_768"]["frames_after_first_3072"], 1)
        self.assertEqual(case["milestones"]["first_raw_four_768"]["frames_after_first_3072"], 1)
        self.assertEqual(case["post_3072_max_raw_count_768"], 4)

    def test_run_summarizes_success_and_failure_replays(self):
        success_states = [
            make_state(
                [
                    [1536, 3072, 1536, 0],
                    [768, 384, 192, 96],
                    [48, 24, 12, 6],
                    [3, 2, 1, 0],
                ],
                100,
            ),
            make_state(
                [
                    [1536, 3072, 3072, 0],
                    [768, 384, 192, 96],
                    [48, 24, 12, 6],
                    [3, 2, 1, 0],
                ],
                101,
            ),
        ]
        failure_states = [
            make_state(
                [
                    [1536, 3072, 768, 0],
                    [384, 192, 96, 48],
                    [24, 12, 6, 3],
                    [3, 2, 1, 0],
                ],
                100,
            )
        ]
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            tmp_path = Path(tmp)
            success_path = tmp_path / "success.json"
            failure_path = tmp_path / "failure.json"
            out_dir = tmp_path / "out"
            write_replay(success_path, success_states)
            write_replay(failure_path, failure_states)

            payload = run([success_path, failure_path], out_dir)

            self.assertEqual(payload["summary"]["by_outcome_count"], {"success": 1, "failure": 1})
            self.assertEqual(payload["summary"]["reached_second_3072"], 1)
            self.assertTrue((out_dir / "support_chain_progression.html").exists())


if __name__ == "__main__":
    unittest.main()
