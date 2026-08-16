import json
import tempfile
import unittest
from pathlib import Path

from threes_rl.label_relabel_compare import load_comparisons, run


class LabelRelabelCompareTests(unittest.TestCase):
    def test_compare_counts_retained_and_reversed_flips(self):
        payload = {
            "labels": [
                {
                    "id": "retained",
                    "seed": 1,
                    "move_count": 10,
                    "base_action": "left",
                    "top_two_actions": ["left", "up"],
                    "features": {"stratum": "endgame_3072p/high_corner_risk"},
                    "previous_label": {
                        "stable": True,
                        "stable_winner": "up",
                        "oracle_regret_at_max_horizon": 100.0,
                    },
                    "label": {
                        "stable": True,
                        "stable_winner": "up",
                        "oracle_regret_at_max_horizon": 80.0,
                    },
                },
                {
                    "id": "reversed",
                    "seed": 2,
                    "move_count": 11,
                    "base_action": "down",
                    "top_two_actions": ["down", "right"],
                    "features": {"stratum": "endgame_3072p/high_corner_risk"},
                    "previous_label": {
                        "stable": True,
                        "stable_winner": "right",
                        "oracle_regret_at_max_horizon": 200.0,
                    },
                    "label": {
                        "stable": True,
                        "stable_winner": "down",
                        "oracle_regret_at_max_horizon": 0.0,
                    },
                },
            ]
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "relabel.json"
            out_dir = Path(tmp) / "out"
            path.write_text(json.dumps(payload))
            comparisons = load_comparisons(path)
            report = run(path, out_dir)

        self.assertEqual(len(comparisons), 2)
        self.assertTrue(comparisons[0].retained_stable_flip)
        self.assertTrue(comparisons[1].stable_reversed_to_base)
        self.assertEqual(report["summary"]["previous_stable_flips"], 2)
        self.assertEqual(report["summary"]["retained_stable_flips"], 1)
        self.assertEqual(report["summary"]["stable_reversed_to_base"], 1)


if __name__ == "__main__":
    unittest.main()
