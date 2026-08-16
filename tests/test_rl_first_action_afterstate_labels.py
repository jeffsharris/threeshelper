import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

from threes_rl.first_action_afterstate_labels import run_from_args
from threes_rl.record_replay import state_payload
from threes_rl.sim import SimState, ThreesSim, preview_from_label, score_board
from threes_rl.transition_reachability_audit import build_rows


def make_state(board, *, move_count=100, preview="blue") -> SimState:
    arr = np.asarray(board, dtype=np.int32)
    return SimState(
        board=arr,
        preview=preview_from_label(preview),
        small_counts={"red": 4, "blue": 4, "gray": 4},
        small_pos=0,
        small_seen_total=120,
        span_small_pos=0,
        large_pending=False,
        max_tile=int(arr.max(initial=0)),
        move_count=move_count,
        game_over=False,
    )


def replay_for_afterstate(after_state: SimState, *, action: str, seed: int) -> dict:
    sim = ThreesSim(np.random.default_rng(seed), starter_tile=1536)
    start = make_state(
        [
            [0, 3, 6, 3],
            [768, 6, 12, 48],
            [768, 3072, 24, 6],
            [0, 0, 768, 768],
        ],
        move_count=99,
    )
    return {
        "policy": "fixture",
        "seed": seed,
        "starter_tile": 1536,
        "source_replay": f"source-{seed}.json",
        "source_seed": seed,
        "source_frame_index": 10,
        "first_action": action,
        "start_score": int(score_board(start.board)),
        "start_max_tile_excl_starter": 3072,
        "frames": [
            {"index": 0, "state": state_payload(start, sim), "move": None},
            {
                "index": 1,
                "state": state_payload(after_state, sim),
                "move": {
                    "action": action,
                    "requested_action": action,
                    "preview_used": {"kind": "small", "value": 1},
                    "inserted_value": 1,
                    "inserted_pos": [0, 0],
                    "eligible_positions": [[0, 0]],
                    "score_delta": 0,
                    "merge_score_delta": 0,
                },
            },
        ],
        "final_score": int(score_board(after_state.board)),
        "final_score_delta": int(score_board(after_state.board) - score_board(start.board)),
        "final_max_tile_excl_starter": 3072,
    }


class FirstActionAfterstateLabelsTests(unittest.TestCase):
    def test_run_from_args_writes_afterstate_records(self):
        success_after = make_state(
            [
                [0, 3, 6, 3],
                [1536, 1536, 12, 48],
                [768, 3072, 24, 6],
                [0, 0, 768, 768],
            ],
            move_count=100,
            preview="gray",
        )
        failure_after = make_state(
            [
                [0, 3, 6, 3],
                [1536, 6, 12, 48],
                [768, 3072, 1536, 6],
                [0, 0, 768, 768],
            ],
            move_count=100,
            preview="red",
        )
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            tmp_path = Path(tmp)
            progress = tmp_path / "progress.json"
            out_dir = tmp_path / "labels"
            payload = {
                "version": 1,
                "entries": {
                    "success": {
                        "record": {
                            "start_id": "start-success",
                            "action_name": "left",
                            "repeat_index": 0,
                            "seed": 11,
                            "source_replay": "source-success.json",
                            "source_seed": 11,
                            "source_frame_index": 10,
                            "start_score": 1000,
                            "start_max_tile_excl_starter": 3072,
                            "score_delta": 100,
                            "moves_delta": 1,
                            "max_tile_excl_starter": 3072,
                        },
                        "replay": replay_for_afterstate(success_after, action="left", seed=11),
                    },
                    "failure": {
                        "record": {
                            "start_id": "start-failure",
                            "action_name": "down",
                            "repeat_index": 0,
                            "seed": 12,
                            "source_replay": "source-failure.json",
                            "source_seed": 12,
                            "source_frame_index": 20,
                            "start_score": 1000,
                            "start_max_tile_excl_starter": 3072,
                            "score_delta": 50,
                            "moves_delta": 1,
                            "max_tile_excl_starter": 3072,
                        },
                        "replay": replay_for_afterstate(failure_after, action="down", seed=12),
                    },
                },
            }
            progress.write_text(json.dumps(payload))

            result = run_from_args(
                Namespace(
                    progress_json=[[progress]],
                    target="raw_adjacent_1536",
                    after_frame_index=1,
                    out_dir=out_dir,
                )
            )

            self.assertEqual(result["summary"]["records"], 2)
            self.assertEqual(result["summary"]["successes"], 1)
            self.assertEqual(result["summary"]["failures"], 1)
            self.assertTrue((out_dir / "records.json").exists())
            rows = build_rows(result["records"], target_tile=6144)
            self.assertEqual(len(rows), 2)
            self.assertEqual(sorted(row["y"] for row in rows), [0, 1])
            self.assertEqual({row["source_action"] for row in rows}, {"left", "down"})


if __name__ == "__main__":
    unittest.main()
