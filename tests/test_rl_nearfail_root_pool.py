import json
import tempfile
import unittest
from pathlib import Path

from threes_rl.nearfail_root_pool import run_from_args


def state_payload(board, *, move_count: int, score: int, game_over: bool = False):
    return {
        "board": board,
        "game_over": game_over,
        "max_tile": max(max(row) for row in board),
        "move_count": move_count,
        "preview": {"kind": "blue", "label": "blue", "value": 1, "candidates": []},
        "score": score,
        "tile_cycle": {
            "large_pending": False,
            "max_tile": max(max(row) for row in board),
            "small_counts": {"blue": 1, "red": 2, "gray": 1},
            "small_pos": 8,
            "small_seen_total": 0,
            "span_small_pos": 0,
        },
    }


class NearfailRootPoolTests(unittest.TestCase):
    def test_selects_one_best_state_per_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            replay_path = root / "replay.json"
            replay = {
                "seed": 123,
                "starter_tile": 1536,
                "policy": "ntuple_phaseblend_expectimax2:checkpoint",
                "final_score": 90000,
                "final_moves": 3,
                "frames": [
                    {
                        "index": 0,
                        "move": None,
                        "state": state_payload(
                            [
                                [1536, 768, 0, 0],
                                [1, 2, 3, 0],
                                [0, 1, 2, 3],
                                [0, 0, 0, 0],
                            ],
                            move_count=10,
                            score=80000,
                        ),
                    },
                    {
                        "index": 1,
                        "move": {"action": "left"},
                        "state": state_payload(
                            [
                                [1536, 768, 0, 0],
                                [768, 2, 3, 0],
                                [0, 1, 2, 3],
                                [0, 0, 0, 0],
                            ],
                            move_count=11,
                            score=81000,
                        ),
                    },
                    {
                        "index": 2,
                        "move": {"action": "up"},
                        "state": state_payload(
                            [
                                [1536, 768, 0, 0],
                                [768, 2, 3, 0],
                                [0, 1, 2, 3],
                                [0, 0, 0, 0],
                            ],
                            move_count=12,
                            score=82000,
                            game_over=True,
                        ),
                    },
                ],
            }
            replay_path.write_text(json.dumps(replay))
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "min_tile": 768,
                        "qualified_games": 1,
                        "replays": [
                            {
                                "json": str(replay_path),
                                "seed": 123,
                                "score": 90000,
                                "score_minus_starter": 30951,
                                "moves": 3,
                                "starter_tile": 1536,
                            }
                        ],
                    }
                )
            )

            args = type(
                "Args",
                (),
                {
                    "manifest": manifest_path,
                    "label": "test_nearfail",
                    "threshold": 1536,
                    "min_tile": 768,
                    "starter": "1536",
                    "root_origin": "fresh",
                    "out_dir": root / "out",
                },
            )()
            payload = run_from_args(args)

            records = json.loads((root / "out" / "records.json").read_text())["records"]
            self.assertEqual(payload["summary"]["records"], 1)
            self.assertEqual(payload["summary"]["min_tile"], 768)
            self.assertEqual(payload["summary"]["min_start_raw_768"], 2)
            self.assertEqual(payload["summary"]["raw_count_768_dist"], {"2": 1})
            self.assertEqual(records[0]["root_seed"], 123)
            self.assertEqual(records[0]["source_frame_index"], 1)
            self.assertEqual(records[0]["source_next_action"], "up")
            self.assertEqual(records[0]["raw_count_768"], 2)
            self.assertEqual(records[0]["root_origin"], "fresh")


if __name__ == "__main__":
    unittest.main()
