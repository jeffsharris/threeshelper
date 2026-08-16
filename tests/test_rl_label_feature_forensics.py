import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from threes_rl.label_feature_forensics import load_cases, run
from threes_rl.record_replay import state_payload
from threes_rl.sim import SimState, ThreesSim, preview_from_label


def make_state() -> SimState:
    return SimState(
        board=np.asarray(
            [
                [1536, 768, 384, 192],
                [96, 48, 24, 12],
                [6, 3, 2, 1],
                [0, 0, 0, 0],
            ],
            dtype=np.int32,
        ),
        preview=preview_from_label("blue"),
        small_counts={"red": 4, "blue": 4, "gray": 4},
        small_pos=0,
        small_seen_total=0,
        span_small_pos=0,
        large_pending=False,
        max_tile=1536,
        move_count=10,
        game_over=False,
    )


class LabelFeatureForensicsTests(unittest.TestCase):
    def test_load_cases_categorizes_stable_flip(self):
        sim = ThreesSim(np.random.default_rng(1), starter_tile=1536)
        state = make_state()
        payload = {
            "labels": [
                {
                    "id": "sample",
                    "seed": 1,
                    "move_count": 10,
                    "base_action": "left",
                    "comparison_action": "up",
                    "top_two_actions": ["left", "up"],
                    "features": {"stratum": "late_1536/high_corner_risk"},
                    "state": state_payload(state, sim),
                    "label": {
                        "actions": ["left", "up"],
                        "stable": True,
                        "stable_winner": "up",
                        "oracle_winner": "up",
                        "same_winner_across_horizons": True,
                        "min_bootstrap_winner_fraction": 0.8,
                        "oracle_regret_at_max_horizon": 123.0,
                    },
                }
            ]
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.json"
            out_dir = Path(tmp) / "out"
            path.write_text(json.dumps(payload))

            cases = load_cases([path])
            report = run([path], out_dir)
            json_exists = (out_dir / "label_feature_forensics.json").exists()
            html_exists = (out_dir / "label_feature_forensics.html").exists()

        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].category, "stable_flip")
        self.assertEqual(cases[0].winner, "up")
        self.assertIn("empty_count", cases[0].deltas)
        self.assertEqual(report["summary"]["categories"]["stable_flip"], 1)
        self.assertTrue(json_exists)
        self.assertTrue(html_exists)


if __name__ == "__main__":
    unittest.main()
