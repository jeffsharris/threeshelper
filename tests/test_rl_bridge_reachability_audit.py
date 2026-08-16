import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from threes_rl.bridge_reachability_audit import label_bridge_records, run_audit
from threes_rl.record_replay import state_payload
from threes_rl.sim import SimState, ThreesSim, preview_from_label


def make_state() -> SimState:
    board = np.asarray(
        [
            [768, 768, 0, 0],
            [768, 0, 768, 1],
            [48, 96, 3072, 24],
            [6, 12, 3, 2],
        ],
        dtype=np.int32,
    )
    return SimState(
        board=board,
        preview=preview_from_label("red"),
        small_counts={"red": 2, "blue": 1, "gray": 0},
        small_pos=9,
        small_seen_total=80,
        span_small_pos=0,
        large_pending=False,
        max_tile=3072,
        move_count=120,
        game_over=False,
    )


def bridge_record(record_id: str, *, root_seed: int) -> dict:
    sim = ThreesSim(np.random.default_rng(root_seed), starter_tile=1536)
    return {
        "id": record_id,
        "source_replay": f"root-{root_seed}.json",
        "source_seed": root_seed,
        "source_frame_index": 10,
        "root_origin": "fresh",
        "root_replay": f"root-{root_seed}.json",
        "root_seed": root_seed,
        "starter_tile": 1536,
        "state": state_payload(make_state(), sim),
        "features": {
            "empty_count": 2,
            "legal_count": 4,
            "raw_count_768": 4,
            "raw_count_1536": 0,
            "raw_has_adjacent_768": True,
            "raw_768_adjacent_pairs": 1,
            "raw_768_components": 3,
            "raw_768_max_component": 2,
            "raw_768_air_neighbors": 3,
        },
    }


class BridgeReachabilityAuditTests(unittest.TestCase):
    def test_label_bridge_records_from_action_summary(self):
        records = [bridge_record("a", root_seed=1), bridge_record("b", root_seed=2)]
        action_summary = [
            {"case_id": "a", "first_action": "left", "target_hits": 2, "valid_rollouts": 4, "target_rate": 0.5},
            {"case_id": "b", "first_action": "right", "target_hits": 0, "valid_rollouts": 4, "target_rate": 0.0},
        ]

        labeled, summary = label_bridge_records(records, action_summary=action_summary)

        self.assertEqual(summary["labeled_records"], 2)
        self.assertEqual(summary["positive_records"], 1)
        self.assertEqual(summary["negative_records"], 1)
        self.assertEqual([record["outcome"] for record in labeled], ["success", "failure"])

    def test_run_audit_writes_artifacts(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            tmp_path = Path(tmp)
            records_path = tmp_path / "records.json"
            frontier_path = tmp_path / "frontier.json"
            out_dir = tmp_path / "out"
            records_path.write_text(json.dumps([bridge_record("a", root_seed=1), bridge_record("b", root_seed=2)]))
            frontier_path.write_text(
                json.dumps(
                    {
                        "action_summary": [
                            {
                                "case_id": "a",
                                "first_action": "left",
                                "target_hits": 2,
                                "valid_rollouts": 4,
                                "target_rate": 0.5,
                            },
                            {
                                "case_id": "b",
                                "first_action": "right",
                                "target_hits": 0,
                                "valid_rollouts": 4,
                                "target_rate": 0.0,
                            },
                        ]
                    }
                )
            )

            payload = run_audit(records_json=[records_path], frontier_json=frontier_path, out_dir=out_dir)

            self.assertEqual(payload["summary"]["records"], 2)
            self.assertEqual(payload["summary"]["successes"], 1)
            self.assertTrue((out_dir / "bridge_reachability_audit.json").exists())
            self.assertTrue((out_dir / "labeled_bridge_records.json").exists())
            self.assertTrue((out_dir / "bridge_reachability_audit.html").exists())

    def test_run_audit_merges_multiple_frontier_files(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            tmp_path = Path(tmp)
            records_path = tmp_path / "records.json"
            frontier_a = tmp_path / "frontier_a.json"
            frontier_b = tmp_path / "frontier_b.json"
            out_dir = tmp_path / "out"
            records_path.write_text(json.dumps([bridge_record("a", root_seed=1), bridge_record("b", root_seed=2)]))
            frontier_a.write_text(
                json.dumps(
                    {
                        "action_summary": [
                            {
                                "case_id": "a",
                                "first_action": "left",
                                "target_hits": 2,
                                "valid_rollouts": 4,
                                "target_rate": 0.5,
                            }
                        ]
                    }
                )
            )
            frontier_b.write_text(
                json.dumps(
                    {
                        "action_summary": [
                            {
                                "case_id": "b",
                                "first_action": "right",
                                "target_hits": 0,
                                "valid_rollouts": 4,
                                "target_rate": 0.0,
                            }
                        ]
                    }
                )
            )

            payload = run_audit(records_json=[records_path], frontier_json=[frontier_a, frontier_b], out_dir=out_dir)

            self.assertEqual(payload["label_summary"]["labeled_records"], 2)
            self.assertEqual(payload["label_summary"]["positive_records"], 1)
            self.assertEqual(payload["frontier_json"], [str(frontier_a), str(frontier_b)])


if __name__ == "__main__":
    unittest.main()
