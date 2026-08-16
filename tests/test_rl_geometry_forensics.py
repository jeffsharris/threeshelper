import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from threes_rl.geometry_forensics import analyze_replay, geometry_features, run, summarize_cases
from threes_rl.record_replay import state_payload
from threes_rl.sim import SimState, ThreesSim, preview_from_label


def make_state(board, move_count=0):
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


class GeometryForensicsTests(unittest.TestCase):
    def test_geometry_features_masks_one_free_starter_tile(self):
        board = np.asarray(
            [
                [1536, 384, 0, 0],
                [0, 768, 0, 0],
                [0, 0, 3072, 1536],
                [0, 0, 0, 0],
            ],
            dtype=np.int32,
        )

        features = geometry_features(board, starter_tile=1536)

        self.assertEqual(features["max_tile_excl_starter"], 3072)
        self.assertEqual(features["primary_max_position"], [2, 2])
        self.assertEqual(features["primary_max_region"], "interior")
        self.assertTrue(features["max_displaced_from_top_left"])
        self.assertEqual(features["count_1536"], 1)
        self.assertTrue(features["adjacent_half_max"])
        self.assertEqual(features["highest_duplicate_tile"], 0)
        self.assertEqual(features["highest_adjacent_pair_tile"], 0)

    def test_geometry_features_reports_highest_duplicate_and_adjacent_pair(self):
        board = np.asarray(
            [
                [1536, 3072, 3072, 0],
                [1536, 1536, 384, 0],
                [768, 192, 96, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.int32,
        )

        features = geometry_features(board, starter_tile=1536)

        self.assertEqual(features["count_3072"], 2)
        self.assertEqual(features["highest_duplicate_tile"], 3072)
        self.assertEqual(features["highest_adjacent_pair_tile"], 3072)
        self.assertTrue(features["adjacent_same_max"])

    def test_analyze_replay_reports_first_3072_and_summary_regions(self):
        sim = ThreesSim(np.random.default_rng(1), starter_tile=1536)
        early = make_state(
            [
                [1536, 384, 0, 0],
                [0, 768, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            move_count=10,
        )
        late = make_state(
            [
                [1536, 384, 0, 0],
                [0, 768, 0, 0],
                [0, 0, 3072, 1536],
                [0, 0, 0, 0],
            ],
            move_count=20,
        )
        replay = {
            "seed": 7,
            "starter_tile": 1536,
            "final_score": 250000,
            "final_moves": 20,
            "frames": [
                {"index": 0, "state": state_payload(early, sim), "move": None},
                {"index": 1, "state": state_payload(late, sim), "move": {"action": "left"}},
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "replay.json"
            path.write_text(json.dumps(replay))
            case = analyze_replay(path)

        self.assertEqual(case["first_3072"]["move_count"], 20)
        self.assertEqual(case["first_3072"]["features"]["primary_max_region"], "interior")
        self.assertIsNone(case["first_two_3072"])
        self.assertEqual(case["post_3072_displaced_rate"], 1.0)
        self.assertEqual(case["post_3072_max_count_3072"], 1)
        self.assertEqual(case["post_3072_max_highest_adjacent_pair_tile"], 0)

        summary = summarize_cases([case])
        self.assertEqual(summary["reached_3072"], 1)
        self.assertEqual(summary["reached_two_3072"], 0)
        self.assertEqual(summary["first_3072_regions"], {"interior": 1})

    def test_run_dedupes_overlapping_replay_paths(self):
        sim = ThreesSim(np.random.default_rng(1), starter_tile=1536)
        state = make_state(
            [
                [1536, 384, 0, 0],
                [0, 768, 0, 0],
                [0, 0, 3072, 1536],
                [0, 0, 0, 0],
            ],
            move_count=20,
        )
        replay = {
            "seed": 7,
            "starter_tile": 1536,
            "final_score": 250000,
            "final_moves": 20,
            "frames": [{"index": 0, "state": state_payload(state, sim), "move": None}],
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "replay.json"
            copied_path = Path(tmp) / "copied_replay.json"
            out_dir = Path(tmp) / "out"
            path.write_text(json.dumps(replay))
            copied_path.write_text(json.dumps(replay))

            payload = run([path, copied_path, path], out_dir)

        self.assertEqual(payload["summary"]["replays"], 1)
        self.assertEqual(payload["source_replays"], [str(path)])


if __name__ == "__main__":
    unittest.main()
