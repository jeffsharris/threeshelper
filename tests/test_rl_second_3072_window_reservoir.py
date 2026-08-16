import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from threes_rl.record_replay import state_payload
from threes_rl.second_3072_window_reservoir import collect_second_3072_records, run_from_args
from threes_rl.sim import SimState, ThreesSim, preview_from_label


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


def replay_payload(states):
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
    return {
        "policy": "test",
        "seed": 7,
        "starter_tile": 1536,
        "final_score": int(frames[-1]["state"]["score"]),
        "final_moves": int(states[-1].move_count),
        "frames": frames,
    }


class Second3072WindowReservoirTests(unittest.TestCase):
    def test_collects_success_and_failure_windows(self):
        success_states = [
            make_state(
                [
                    [1536, 3072, 768, 0],
                    [384, 192, 96, 48],
                    [24, 12, 6, 3],
                    [3, 2, 1, 0],
                ],
                100,
            ),
            make_state(
                [
                    [1536, 3072, 1536, 0],
                    [384, 192, 96, 48],
                    [24, 12, 6, 3],
                    [3, 2, 1, 0],
                ],
                101,
            ),
            make_state(
                [
                    [1536, 3072, 3072, 0],
                    [384, 192, 96, 48],
                    [24, 12, 6, 3],
                    [3, 2, 1, 0],
                ],
                102,
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
            ),
            make_state(
                [
                    [1536, 3072, 1536, 0],
                    [384, 192, 96, 48],
                    [24, 12, 6, 3],
                    [3, 2, 1, 0],
                ],
                101,
            ),
            make_state(
                [
                    [1536, 3072, 768, 3],
                    [384, 192, 96, 48],
                    [24, 12, 6, 3],
                    [3, 2, 1, 0],
                ],
                102,
            ),
        ]
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            tmp_path = Path(tmp)
            success_path = tmp_path / "success.json"
            failure_path = tmp_path / "failure.json"
            success_path.write_text(json.dumps(replay_payload(success_states)))
            failure_path.write_text(json.dumps(replay_payload(failure_states)))

            records, summary = collect_second_3072_records(
                [success_path, failure_path],
                window_size=2,
                include_failures=True,
                default_starter_tile=1536,
            )

        self.assertEqual(summary["by_outcome"], {"success": 2, "failure": 2})
        success = [record for record in records if record["outcome"] == "success"]
        failure = [record for record in records if record["outcome"] == "failure"]
        self.assertEqual([record["moves_to_event"] for record in success], [2, 1])
        self.assertTrue(all(record["event_kind"] == "visible_two_3072" for record in success))
        self.assertTrue(all(record["moves_to_event"] is None for record in failure))
        self.assertIn("geometry_count_3072", success[0]["features"])

    def test_run_from_args_writes_artifacts(self):
        state = make_state(
            [
                [1536, 3072, 768, 0],
                [384, 192, 96, 48],
                [24, 12, 6, 3],
                [3, 2, 1, 0],
            ],
            100,
        )
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            tmp_path = Path(tmp)
            replay_path = tmp_path / "replay.json"
            out_dir = tmp_path / "out"
            replay_path.write_text(json.dumps(replay_payload([state])))

            payload = run_from_args(
                type(
                    "Args",
                    (),
                    {
                        "replay_json": [[replay_path]],
                        "replay_glob": [],
                        "window_size": 2,
                        "no_failures": False,
                        "starter": "1536",
                        "max_records": 0,
                        "out_dir": out_dir,
                    },
                )()
            )

            self.assertEqual(payload["kind"], "second_3072_window_reservoir")
            self.assertTrue((out_dir / "records.json").exists())
            self.assertTrue((out_dir / "second_3072_windows.html").exists())


if __name__ == "__main__":
    unittest.main()
