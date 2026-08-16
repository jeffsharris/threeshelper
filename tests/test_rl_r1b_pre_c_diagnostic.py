from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from threes_rl.r1b_pre_c_diagnostic import analyze


class R1bPreCDiagnosticTests(unittest.TestCase):
    def test_analysis_pairs_streams_and_reports_h40_support(self) -> None:
        fields = [
            "root_id", "block", "repeat", "horizon", "logical_seed",
            "deck_stream_id", "slot_stream_id", "policy_stream_id",
            "reached_1536", "score_gain", "survived",
        ]
        with tempfile.TemporaryDirectory() as directory:
            baseline_path = Path(directory) / "baseline.csv"
            candidate_path = Path(directory) / "candidate.csv"
            baseline_rows = []
            candidate_rows = []
            for root in ("r1", "r2"):
                for block in ("A", "B"):
                    for repeat in range(8):
                        for horizon in (10, 20, 40):
                            common = {
                                "root_id": root, "block": block, "repeat": repeat,
                                "horizon": horizon, "logical_seed": repeat,
                                "deck_stream_id": repeat + 10, "slot_stream_id": repeat + 20,
                                "policy_stream_id": repeat + 30,
                            }
                            baseline_rows.append({**common, "reached_1536": 0, "score_gain": 10, "survived": 1})
                            candidate_rows.append({
                                **common,
                                "reached_1536": int(horizon == 40),
                                "score_gain": 20,
                                "survived": 1,
                            })
            for path, rows in ((baseline_path, baseline_rows), (candidate_path, candidate_rows)):
                with path.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)

            result = analyze(baseline_path, candidate_path)

            self.assertEqual(result["decision"], "SUPPORTS_PRE_C")
            self.assertEqual(result["metrics"]["h40"]["reached_1536"]["difference"], 1.0)


if __name__ == "__main__":
    unittest.main()
