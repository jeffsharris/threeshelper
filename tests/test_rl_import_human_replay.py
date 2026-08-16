import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from threes_rl.import_human_replay import import_events_file, import_game
from threes_rl.record_replay import state_payload
from threes_rl.sim import ThreesSim, board_to_tokens, preview_from_label, simulate_base_move
from threes_rl.train_td import state_from_replay_payload


def initial_board() -> np.ndarray:
    return np.asarray(
        [
            [1536, 0, 1, 0],
            [2, 0, 3, 0],
            [1, 0, 2, 0],
            [3, 1, 2, 0],
        ],
        dtype=np.int32,
    )


def make_events() -> list[dict[str, object]]:
    before = initial_board()
    shifted, eligible = simulate_base_move(before, "left")
    after = shifted.copy()
    after[0, 3] = 1
    return [
        {
            "type": "game_start",
            "game_index": 1,
            "board": board_to_tokens(before),
            "preview_label": "blue",
        },
        {
            "type": "observed_move",
            "game_index": 1,
            "move_index_start": 1,
            "move_index": 1,
            "before_board": board_to_tokens(before),
            "before_preview_label": "blue",
            "after_board": board_to_tokens(after),
            "after_preview_label": "red",
            "unknown_board": False,
            "unknown_preview": False,
            "preview_check": {"valid": True},
            "transition_check": {
                "valid": True,
                "inserted_value": 1,
                "inserted_pos": [0, 3],
                "eligible_positions": [list(pos) for pos in eligible],
                "expected_values": [1],
            },
            "direction": "left",
            "direction_sequence": ["left"],
            "step_count": 1,
            "transition_path": [
                {
                    "direction": "left",
                    "preview_label": "blue",
                    "inserted_value": 1,
                    "inserted_pos": [0, 3],
                    "eligible_positions": [list(pos) for pos in eligible],
                    "expected_values": [1],
                    "after_board": board_to_tokens(after),
                }
            ],
        },
        {
            "type": "game_end",
            "game_index": 1,
            "move_index": 1,
            "scene": "game_over",
        },
    ]


class HumanReplayImportTests(unittest.TestCase):
    def test_import_game_writes_sim_replay_frames(self):
        replay = import_game(make_events(), source_events=Path("events.jsonl"), starter_tile=1536)

        self.assertEqual(replay["policy"], "human_observed")
        self.assertEqual(replay["final_moves"], 1)
        self.assertEqual(len(replay["frames"]), 2)

        first_state = replay["frames"][0]["state"]
        self.assertEqual(first_state["move_count"], 0)
        self.assertEqual(first_state["preview"]["label"], "blue")
        self.assertEqual(first_state["tile_cycle"]["small_pos"], 8)
        self.assertEqual(first_state["tile_cycle"]["small_counts"]["blue"], 1)

        second_state = replay["frames"][1]["state"]
        self.assertEqual(second_state["move_count"], 1)
        self.assertEqual(second_state["preview"]["label"], "red")
        self.assertEqual(second_state["tile_cycle"]["small_pos"], 9)
        self.assertEqual(second_state["tile_cycle"]["small_counts"]["blue"], 0)
        self.assertEqual(replay["frames"][1]["move"]["action"], "left")

        restored = state_from_replay_payload(second_state)
        self.assertEqual(restored.move_count, 1)
        self.assertEqual(restored.preview.label, "red")

    def test_import_events_file_writes_manifest_and_replay(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            tmp_path = Path(tmp)
            events_path = tmp_path / "events.jsonl"
            out_dir = tmp_path / "out"
            events_path.write_text("\n".join(json.dumps(event) for event in make_events()))

            manifest = import_events_file(events_path, out_dir, starter_tile=1536)

            self.assertEqual(manifest["games_imported"], 1)
            self.assertEqual(manifest["games_skipped"], 0)
            replay_path = Path(manifest["replays"][0]["json"])
            self.assertTrue(replay_path.exists())
            self.assertTrue(Path(manifest["replays"][0]["html"]).exists())
            self.assertTrue((out_dir / "manifest.json").exists())

    def test_state_payload_still_round_trips_large_preview_candidates(self):
        sim = ThreesSim(np.random.default_rng(1), starter_tile=1536)
        state = sim.state_from_snapshot(
            initial_board(),
            preview_from_label("large_candidates", (6, 12, 24)),
            ({"red": 1, "blue": 1, "gray": 2}, 8, 21, 0, True, 1536),
        )

        payload = state_payload(state, sim)
        restored = state_from_replay_payload(payload)

        self.assertEqual(restored.preview.candidates, (6, 12, 24))


if __name__ == "__main__":
    unittest.main()
