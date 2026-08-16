from __future__ import annotations

import unittest

from threes_rl.human_h0_roots import feature_distance, state_match_features


class HumanH0RootTests(unittest.TestCase):
    def test_current_state_features_are_deterministic(self) -> None:
        payload = {
            "board": [[1536, 768, 384, 96], [3, 2, 1, 0], [6, 3, 2, 1], [0, 0, 0, 0]],
            "preview": {"kind": "red", "value": 2, "candidates": []},
            "tile_cycle": {
                "small_counts": {"red": 2, "blue": 3, "gray": 1},
                "small_pos": 6,
                "small_seen_total": 30,
                "span_small_pos": 4,
                "large_pending": True,
                "max_tile": 1536,
            },
            "move_count": 200,
            "game_over": False,
        }

        first = state_match_features(payload, 1536)
        second = state_match_features(payload, 1536)

        self.assertEqual(first, second)
        self.assertEqual(feature_distance(first, second), 0.0)
        self.assertIn("plus_probability", first)


if __name__ == "__main__":
    unittest.main()
