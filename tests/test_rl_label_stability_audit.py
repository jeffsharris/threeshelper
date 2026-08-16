import json
import tempfile
import unittest
from pathlib import Path

from threes_rl.label_stability_audit import load_cases, run


class LabelStabilityAuditTests(unittest.TestCase):
    def test_audit_marks_consistent_bootstrap_flip_as_robust(self):
        payload = {
            "labels": [
                {
                    "id": "robust",
                    "seed": 1,
                    "move_count": 10,
                    "base_action": "left",
                    "top_two_actions": ["left", "up"],
                    "features": {"stratum": "endgame_3072p/high_corner_risk"},
                    "label": {
                        "actions": ["left", "up"],
                        "horizons": [32, 64],
                        "stable": True,
                        "stable_winner": "up",
                        "oracle_winner": "up",
                        "oracle_regret_at_max_horizon": 100.0,
                        "by_action": {
                            "left": {"32": [0, 0, 0, 0], "64": [0, 0, 0, 0]},
                            "up": {"32": [10, 10, 10, 10], "64": [20, 20, 20, 20]},
                        },
                    },
                },
                {
                    "id": "horizon-flip",
                    "seed": 2,
                    "move_count": 11,
                    "base_action": "down",
                    "top_two_actions": ["down", "right"],
                    "features": {"stratum": "endgame_3072p/high_corner_risk"},
                    "label": {
                        "actions": ["down", "right"],
                        "horizons": [32, 64],
                        "stable": False,
                        "oracle_winner": "right",
                        "by_action": {
                            "down": {"32": [10, 10, 10, 10], "64": [0, 0, 0, 0]},
                            "right": {"32": [0, 0, 0, 0], "64": [10, 10, 10, 10]},
                        },
                    },
                },
            ]
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.json"
            out_dir = Path(tmp) / "out"
            path.write_text(json.dumps(payload))

            cases = load_cases([path], threshold=0.70, resamples=100, seed=7)
            report = run([path], out_dir, threshold=0.70, resamples=100, seed=7)
            json_exists = (out_dir / "label_stability_audit.json").exists()
            html_exists = (out_dir / "label_stability_audit.html").exists()
            export_path = out_dir / "robust_flip_samples.json"
            export_exists = export_path.exists()
            export_payload = json.loads(export_path.read_text())

        by_id = {case.id: case for case in cases}
        self.assertTrue(by_id["robust"].robust)
        self.assertTrue(by_id["robust"].robust_flip)
        self.assertFalse(by_id["horizon-flip"].robust)
        self.assertEqual(report["summary"]["robust_flips"], 1)
        self.assertTrue(json_exists)
        self.assertTrue(html_exists)
        self.assertTrue(export_exists)
        self.assertEqual(len(export_payload["samples"]), 1)
        self.assertEqual(export_payload["samples"][0]["id"], "robust")
        self.assertIn("previous_label", export_payload["samples"][0])


if __name__ == "__main__":
    unittest.main()
