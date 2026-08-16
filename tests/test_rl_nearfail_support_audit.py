import json
import tempfile
import unittest
from pathlib import Path

from threes_rl.nearfail_support_audit import run_audit, support_bucket


def record(seed, board):
    return {
        "id": f"case_{seed}",
        "root_origin": "fresh",
        "root_seed": seed,
        "source_seed": seed,
        "starter_tile": 1536,
        "state": {
            "board": board,
            "game_over": False,
            "max_tile": max(max(row) for row in board),
            "move_count": 20,
            "preview": {"kind": "blue", "label": "blue", "value": 1, "candidates": []},
            "score": 80000,
            "tile_cycle": {
                "large_pending": False,
                "max_tile": max(max(row) for row in board),
                "small_counts": {"blue": 1, "red": 2, "gray": 1},
                "small_pos": 0,
                "small_seen_total": 0,
                "span_small_pos": 0,
            },
        },
    }


class NearfailSupportAuditTests(unittest.TestCase):
    def test_support_bucket_order(self):
        self.assertEqual(
            support_bucket({"raw_count_768": 2}),
            "not_one_768",
        )
        self.assertEqual(
            support_bucket(
                {
                    "raw_count_768": 1,
                    "raw_has_adjacent_384": True,
                    "raw_count_384": 2,
                    "raw_has_adjacent_192": False,
                    "raw_count_192": 0,
                }
            ),
            "adjacent_384",
        )
        self.assertEqual(
            support_bucket(
                {
                    "raw_count_768": 1,
                    "raw_has_adjacent_384": False,
                    "raw_count_384": 0,
                    "raw_has_adjacent_192": False,
                    "raw_count_192": 2,
                }
            ),
            "duplicate_192",
        )

    def test_run_audit_counts_support_material(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            records_path = root / "records.json"
            records_path.write_text(
                json.dumps(
                    {
                        "records": [
                            record(1, [[1536, 768, 384, 0], [192, 0, 0, 0], [0, 0, 0, 0], [3, 6, 12, 24]]),
                            record(2, [[1536, 768, 192, 0], [192, 0, 0, 0], [0, 0, 0, 0], [3, 6, 12, 24]]),
                            record(3, [[1536, 768, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [3, 6, 12, 24]]),
                        ]
                    }
                )
            )

            payload = run_audit(records_json=[records_path], out_dir=root / "out")
            summary = payload["summary"]

            self.assertEqual(summary["records"], 3)
            self.assertEqual(summary["one_768_records"], 3)
            self.assertEqual(summary["one_768_with_384"], 1)
            self.assertEqual(summary["one_768_with_192"], 2)
            self.assertEqual(summary["one_768_no_384_192"], 1)
            self.assertEqual(summary["by_support_bucket"]["one_384"], 1)
            self.assertEqual(summary["by_support_bucket"]["duplicate_192"], 1)
            self.assertEqual(summary["support_bucket_stats"]["one_384"]["records"], 1)
            self.assertEqual(summary["support_bucket_stats"]["no_384_192"]["records"], 1)
            self.assertTrue((root / "out" / "report.md").exists())


if __name__ == "__main__":
    unittest.main()
