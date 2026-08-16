import argparse
import json
import tempfile
import unittest
from pathlib import Path

from threes_rl.reachability_screen_report import run_from_args


class ReachabilityScreenReportTests(unittest.TestCase):
    def test_reports_rank_calibration_from_gate_rows(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            gate_json = root / "gate.json"
            score_json = root / "score.json"
            gate_json.write_text(
                json.dumps(
                    {
                        "start_rows": [
                            {
                                "start_case_id": "a",
                                "source_seed": 1,
                                "source_frame_index": 10,
                                "reachability_rank": 1,
                                "reachability_prob": 0.8,
                                "continuations": 4,
                                "p_reached_6144": 0.0,
                                "p_raw_adjacent_1536": 0.0,
                                "high_score_delta": 100,
                            },
                            {
                                "start_case_id": "b",
                                "source_seed": 2,
                                "source_frame_index": 20,
                                "reachability_rank": 2,
                                "reachability_prob": 0.6,
                                "continuations": 4,
                                "p_reached_6144": 0.5,
                                "p_raw_adjacent_1536": 0.5,
                                "high_score_delta": 200,
                            },
                            {
                                "start_case_id": "c",
                                "source_seed": 3,
                                "source_frame_index": 30,
                                "reachability_rank": 3,
                                "reachability_prob": 0.2,
                                "continuations": 4,
                                "p_reached_6144": 0.25,
                                "p_raw_adjacent_1536": 0.25,
                                "high_score_delta": 300,
                            },
                        ]
                    }
                )
            )
            score_json.write_text(
                json.dumps(
                    {
                        "raw_candidates": 10,
                        "filtered_candidates": 3,
                        "unique_candidates": 3,
                        "summary": {"max_reachability_prob": 0.8},
                        "top_selection": {"selected": 3},
                    }
                )
            )

            payload = run_from_args(
                argparse.Namespace(
                    gate_json=gate_json,
                    score_json=score_json,
                    target_tile=6144,
                    out_dir=root / "out",
                )
            )

            summary = payload["summary"]
            self.assertEqual(summary["starts"], 3)
            self.assertEqual(summary["continuations"], 12)
            self.assertEqual(summary["hits"], 3)
            self.assertAlmostEqual(summary["hit_rate"], 0.25)
            self.assertEqual(summary["hit_starts"], 2)
            self.assertEqual(summary["dead_top_starts"][0]["start_case_id"], "a")
            self.assertEqual(summary["productive_starts"][0]["start_case_id"], "b")
            self.assertEqual(summary["top_buckets"][0]["k"], 3)
            self.assertTrue((root / "out" / "reachability_screen_report.html").exists())


if __name__ == "__main__":
    unittest.main()
