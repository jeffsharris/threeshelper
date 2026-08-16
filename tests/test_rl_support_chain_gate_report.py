import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from threes_rl.record_replay import state_payload
from threes_rl.sim import SimState, ThreesSim, preview_from_label, score_board
from threes_rl.support_chain_gate_report import run_from_args


def make_state(board, move_count):
    arr = np.asarray(board, dtype=np.int32)
    return SimState(
        board=arr,
        preview=preview_from_label("gray"),
        small_counts={"red": 4, "blue": 4, "gray": 4},
        small_pos=0,
        small_seen_total=0,
        span_small_pos=0,
        large_pending=False,
        max_tile=int(arr.max(initial=0)),
        move_count=move_count,
        game_over=False,
    )


def write_replay(path: Path, *, start_case_id: str, states: list[SimState]) -> None:
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
    start_score = score_board(states[0].board)
    final_score = score_board(states[-1].board)
    path.write_text(
        json.dumps(
            {
                "policy": "test",
                "seed": 7 + len(states),
                "starter_tile": 1536,
                "start_case_id": start_case_id,
                "source_seed": 100,
                "source_frame_index": 20,
                "start_score": int(start_score),
                "final_score": int(final_score),
                "final_score_delta": int(final_score - start_score),
                "final_moves_delta": int(states[-1].move_count - states[0].move_count),
                "final_max_tile_excl_starter": int(max(value for value in states[-1].board.reshape(-1) if value != 1536)),
                "frames": frames,
            }
        )
    )


class SupportChainGateReportTests(unittest.TestCase):
    def test_gate_report_summarizes_stage_rates_by_start(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            success_path = root / "success.json"
            failure_path = root / "failure.json"
            start_json = root / "starts.json"
            out_dir = root / "gate"
            write_replay(
                success_path,
                start_case_id="success-start",
                states=[
                    make_state(
                        [
                            [1536, 3072, 768, 768],
                            [1536, 384, 192, 96],
                            [48, 24, 12, 6],
                            [3, 2, 1, 0],
                        ],
                        100,
                    ),
                    make_state(
                        [
                            [1536, 3072, 3072, 0],
                            [1536, 768, 192, 96],
                            [48, 24, 12, 6],
                            [3, 2, 1, 0],
                        ],
                        101,
                    ),
                    make_state(
                        [
                            [1536, 6144, 0, 0],
                            [768, 384, 192, 96],
                            [48, 24, 12, 6],
                            [3, 2, 1, 0],
                        ],
                        102,
                    ),
                ],
            )
            write_replay(
                failure_path,
                start_case_id="failure-start",
                states=[
                    make_state(
                        [
                            [1536, 3072, 768, 0],
                            [384, 192, 96, 48],
                            [24, 12, 6, 3],
                            [3, 2, 1, 0],
                        ],
                        100,
                    )
                ],
            )
            start_json.write_text(
                json.dumps(
                    {
                        "records": [
                            {
                                "id": "success-start",
                                "reachability_prob": 0.8,
                                "reachability_rank": 2,
                            }
                        ]
                    }
                )
            )

            payload = run_from_args(
                argparse.Namespace(
                    replay_json=[[success_path, failure_path]],
                    replay_glob=[],
                    start_json=start_json,
                    out_dir=out_dir,
                )
            )

            summary = payload["summary"]
            self.assertEqual(summary["continuations"], 2)
            self.assertEqual(summary["starts"], 2)
            self.assertEqual(summary["stage_summary"]["raw_adjacent_768"]["count"], 1)
            self.assertEqual(summary["stage_summary"]["raw_adjacent_1536"]["count"], 1)
            self.assertEqual(summary["stage_summary"]["second_3072"]["count"], 1)
            self.assertEqual(summary["stage_summary"]["reached_6144"]["count"], 1)
            self.assertEqual(payload["start_rows"][0]["start_case_id"], "success-start")
            self.assertEqual(payload["start_rows"][0]["p_reached_6144"], 1.0)
            self.assertEqual(payload["start_rows"][0]["reachability_prob"], 0.8)
            self.assertTrue((out_dir / "support_chain_gate_report.html").exists())

    def test_run_from_args_replay_glob_is_recursive(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            nested = root / "nested" / "deep"
            nested.mkdir(parents=True)
            replay_path = nested / "replay.json"
            out_dir = root / "gate"
            write_replay(
                replay_path,
                start_case_id="success-start",
                states=[
                    make_state(
                        [
                            [1536, 3072, 768, 768],
                            [1536, 384, 192, 96],
                            [48, 24, 12, 6],
                            [3, 2, 1, 0],
                        ],
                        100,
                    ),
                    make_state(
                        [
                            [1536, 6144, 0, 0],
                            [768, 384, 192, 96],
                            [48, 24, 12, 6],
                            [3, 2, 1, 0],
                        ],
                        101,
                    ),
                ],
            )

            payload = run_from_args(
                argparse.Namespace(
                    replay_json=[],
                    replay_glob=[str(root / "**" / "replay.json")],
                    start_json=None,
                    out_dir=out_dir,
                )
            )

            self.assertEqual(payload["summary"]["continuations"], 1)
            self.assertEqual(payload["summary"]["stage_summary"]["reached_6144"]["count"], 1)


if __name__ == "__main__":
    unittest.main()
