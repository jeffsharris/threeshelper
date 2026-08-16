import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from threes_rl.first_action_support_preservation_audit import run_audit
from threes_rl.record_replay import state_payload
from threes_rl.sim import SimState, ThreesSim, preview_from_label


def support_preservation_state() -> SimState:
    return SimState(
        board=np.asarray(
            [
                [768, 0, 0, 0],
                [768, 0, 0, 0],
                [0, 0, 768, 768],
                [3, 6, 12, 24],
            ],
            dtype=np.int32,
        ),
        preview=preview_from_label("gray"),
        small_counts={"red": 4, "blue": 4, "gray": 4},
        small_pos=8,
        small_seen_total=40,
        span_small_pos=0,
        large_pending=False,
        max_tile=768,
        move_count=80,
        game_over=False,
    )


def record_payload() -> dict:
    sim = ThreesSim(np.random.default_rng(7), starter_tile=1536)
    state = support_preservation_state()
    return {
        "id": "support-case",
        "starter_tile": 1536,
        "source_replay": "fixture/source.json",
        "source_seed": 7,
        "source_frame_index": 80,
        "source_policy": "fixture",
        "root_origin": "fresh",
        "root_replay": "fixture/root.json",
        "root_seed": 7,
        "root_frame_index": 0,
        "root_policy": "fixture",
        "state": state_payload(state, sim),
    }


class FirstActionSupportPreservationAuditTests(unittest.TestCase):
    def test_run_audit_detects_support_preserved_at_first_1536(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            tmp_path = Path(tmp)
            records_path = tmp_path / "records.json"
            out_dir = tmp_path / "audit"
            records_path.write_text(json.dumps({"records": [record_payload()]}))

            payload = run_audit(
                records_json=[records_path],
                policy_name="greedy",
                horizon=1,
                repeats=1,
                max_starts=0,
                seed=20260708,
                root_origins={"fresh"},
                default_starter_tile=1536,
                require_start_adjacent_768=False,
                out_dir=out_dir,
                progress_every=0,
            )

            self.assertEqual(payload["summary"]["labels"], 1)
            self.assertEqual(payload["summary"]["start_cases_with_adjacent_768"], 1)
            self.assertGreater(payload["summary"]["rollouts"], 0)
            self.assertGreater(payload["summary"]["p_one1536"], 0.0)
            self.assertGreater(payload["summary"]["p_support_preserved_at_one1536"], 0.0)
            up_rows = [row for row in payload["action_summary"] if row["first_action"] == "up"]
            self.assertEqual(up_rows[0]["p_support_preserved_at_one1536"], 1.0)
            self.assertTrue((out_dir / "first_action_support_preservation_audit.json").exists())
            self.assertTrue((out_dir / "records.json").exists())
            self.assertTrue((out_dir / "first_action_support_preservation_audit.html").exists())


if __name__ == "__main__":
    unittest.main()
