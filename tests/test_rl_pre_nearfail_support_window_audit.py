import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from threes_rl.pre_nearfail_support_window_audit import parse_offsets, run_audit
from threes_rl.record_replay import state_payload
from threes_rl.sim import SimState, ThreesSim, preview_from_label


def state(board, *, move_count):
    arr = np.asarray(board, dtype=np.int32)
    return SimState(
        board=arr,
        preview=preview_from_label("gray"),
        small_counts={"red": 4, "blue": 4, "gray": 4},
        small_pos=0,
        small_seen_total=0,
        span_small_pos=0,
        large_pending=False,
        max_tile=int(arr.max(initial=0)),
        move_count=move_count,
        game_over=False,
    )


def frame(idx, sim, board, *, move_count, action="left"):
    return {
        "index": idx,
        "state": state_payload(state(board, move_count=move_count), sim),
        "move": {"action": action},
    }


class PreNearfailSupportWindowAuditTests(unittest.TestCase):
    def test_parse_offsets_deduplicates_and_validates(self):
        self.assertEqual(parse_offsets("2,1,2,0"), [2, 1, 0])
        with self.assertRaises(ValueError):
            parse_offsets("-1")

    def test_run_audit_reports_support_loss_before_selected_frame(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            replay_path = root / "replay.json"
            sim = ThreesSim(np.random.default_rng(7), starter_tile=1536)
            replay = {
                "policy": "fixture",
                "seed": 7,
                "starter_tile": 1536,
                "root_origin": "fresh",
                "frames": [
                    frame(
                        0,
                        sim,
                        [[1536, 768, 384, 0], [192, 0, 0, 0], [0, 0, 0, 0], [3, 6, 12, 24]],
                        move_count=10,
                    ),
                    frame(
                        1,
                        sim,
                        [[1536, 768, 192, 0], [0, 0, 0, 0], [0, 0, 0, 0], [3, 6, 12, 24]],
                        move_count=11,
                    ),
                    frame(
                        2,
                        sim,
                        [[1536, 768, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [3, 6, 12, 24]],
                        move_count=12,
                    ),
                ],
            }
            replay_path.write_text(json.dumps(replay))
            records_path = root / "records.json"
            records_path.write_text(
                json.dumps(
                    {
                        "records": [
                            {
                                "id": "case",
                                "root_origin": "fresh",
                                "root_seed": 7,
                                "source_seed": 7,
                                "source_replay": str(replay_path),
                                "source_frame_index": 2,
                                "starter_tile": 1536,
                                "state": replay["frames"][2]["state"],
                            }
                        ]
                    }
                )
            )

            payload = run_audit(
                records_json=[records_path],
                out_dir=root / "out",
                offsets=[2, 1, 0],
                final_bucket="no_384_192",
                root_origins={"fresh"},
            )

            summary = payload["summary"]
            self.assertEqual(summary["selected_cases"], 1)
            self.assertEqual(summary["rows"], 3)
            self.assertEqual(summary["support_present_by_offset"], {"2": 1, "1": 1, "0": 0})
            self.assertEqual(summary["lost_support_by_offset"], {"2": 1, "1": 1})
            self.assertEqual(summary["nearest_support_before_zero"], {"1": 1})
            self.assertEqual(summary["support_material_by_offset"]["2"]["has_384"], 1)
            self.assertEqual(summary["support_material_by_offset"]["2"]["has_192"], 1)
            self.assertEqual(summary["support_material_by_offset"]["0"]["has_neither"], 1)
            self.assertEqual(summary["raw_768_by_offset"]["2"]["1"], 1)
            by_offset = summary["by_offset_support_bucket"]
            self.assertEqual(by_offset["2"]["one_384"], 1)
            self.assertEqual(by_offset["1"]["one_192"], 1)
            self.assertEqual(by_offset["0"]["no_384_192"], 1)
            self.assertTrue((root / "out" / "pre_nearfail_support_windows.json").exists())
            self.assertTrue((root / "out" / "report.md").exists())


if __name__ == "__main__":
    unittest.main()
