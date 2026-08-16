import json
import tempfile
import unittest
from pathlib import Path

from threes_rl.duplicate1536_geometry_audit import (
    aggregate_action_summary,
    build_rows,
    pair_geometry,
    run_audit,
)


def record(record_id: str, board: list[list[int]], *, root_seed: int = 1) -> dict:
    return {
        "id": record_id,
        "root_seed": root_seed,
        "root_origin": "fresh",
        "root_policy_family": "test",
        "source_replay": f"seed_{root_seed}.json",
        "root_replay": f"seed_{root_seed}.json",
        "features": {
            "empty_count": sum(1 for row in board for value in row if value == 0),
            "preview": "red",
            "large_pending": False,
            "safe_smalls_until_large_possible": 12,
            "top_left": board[0][0],
            "corner_risk": "high_corner_risk",
        },
        "state": {
            "board": board,
            "game_over": False,
            "legal_mask": [True, False, True, False],
            "move_count": 100,
            "score": 12345,
            "preview": {"label": "red"},
        },
    }


class Duplicate1536GeometryAuditTests(unittest.TestCase):
    def test_pair_geometry_describes_relation_and_blockers(self):
        board = [
            [1536, 2, 3072, 1536],
            [6, 12, 24, 48],
            [3, 6, 12, 24],
            [1, 2, 3, 6],
        ]

        geom = pair_geometry(board)

        self.assertEqual(geom["relation"], "same_row_gap2")
        self.assertEqual(geom["min_distance"], 3)
        self.assertEqual(geom["line_blockers"], [2, 3072])
        self.assertTrue(geom["has_3072_between"])

    def test_build_rows_combines_geometry_and_rollout_labels(self):
        records = [
            record("a", [[1536, 1536, 0, 0], [6, 12, 24, 48], [3, 6, 12, 24], [1, 2, 3, 6]]),
            record("b", [[1536, 2, 3072, 1536], [6, 12, 24, 48], [3, 6, 12, 24], [1, 2, 3, 6]]),
        ]
        labels = aggregate_action_summary(
            [
                {"case_id": "a", "first_action": "left", "target_hits": 4, "valid_rollouts": 4, "target_rate": 1.0},
                {"case_id": "b", "first_action": "right", "target_hits": 0, "valid_rollouts": 4, "target_rate": 0.0},
            ]
        )

        rows = build_rows(records, labels)

        self.assertEqual(rows[0]["geom_relation"], "adjacent")
        self.assertEqual(rows[0]["target_rate"], 1.0)
        self.assertTrue(rows[0]["positive"])
        self.assertEqual(rows[1]["geom_relation"], "same_row_gap2")
        self.assertTrue(rows[1]["geom_has_3072_between"])
        self.assertFalse(rows[1]["positive"])

    def test_run_audit_writes_json_and_report(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            tmp_path = Path(tmp)
            records_path = tmp_path / "records.json"
            frontier_path = tmp_path / "frontier.json"
            out_dir = tmp_path / "out"
            records_path.write_text(
                json.dumps(
                    [
                        record("a", [[1536, 1536, 0, 0], [6, 12, 24, 48], [3, 6, 12, 24], [1, 2, 3, 6]], root_seed=1),
                        record("b", [[1536, 2, 3072, 1536], [6, 12, 24, 48], [3, 6, 12, 24], [1, 2, 3, 6]], root_seed=2),
                    ]
                )
            )
            frontier_path.write_text(
                json.dumps(
                    {
                        "action_summary": [
                            {"case_id": "a", "first_action": "left", "target_hits": 4, "valid_rollouts": 4, "target_rate": 1.0},
                            {"case_id": "b", "first_action": "right", "target_hits": 0, "valid_rollouts": 4, "target_rate": 0.0},
                        ]
                    }
                )
            )

            payload = run_audit(records_json=[records_path], frontier_json=[frontier_path], out_dir=out_dir)

            self.assertEqual(payload["summary"]["records"], 2)
            self.assertEqual(payload["summary"]["positive_records"], 1)
            self.assertTrue((out_dir / "duplicate1536_geometry_audit.json").exists())
            self.assertTrue((out_dir / "report.md").exists())


if __name__ == "__main__":
    unittest.main()
