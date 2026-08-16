import argparse
import json
import tempfile
import unittest
from pathlib import Path

from threes_rl.support_chain_start_labels import run_from_args


def start_record(record_id: str, *, source_replay: str = "source.json", frame_index: int = 10) -> dict:
    return {
        "id": record_id,
        "source_replay": source_replay,
        "source_seed": 1,
        "source_frame_index": frame_index,
        "starter_tile": 1536,
        "state": {
            "board": [
                [1536, 3072, 768, 0],
                [384, 192, 96, 48],
                [24, 12, 6, 3],
                [3, 2, 1, 0],
            ]
        },
    }


class SupportChainStartLabelsTests(unittest.TestCase):
    def test_builds_continuation_and_start_threshold_labels(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            starts = root / "starts.json"
            gate = root / "gate.json"
            starts.write_text(json.dumps({"records": [start_record("a"), start_record("b")]}))
            gate.write_text(
                json.dumps(
                    {
                        "rows": [
                            {"start_case_id": "a", "raw_adjacent_1536": True, "replay_json": "a1.json"},
                            {"start_case_id": "a", "raw_adjacent_1536": False, "replay_json": "a2.json"},
                            {"start_case_id": "b", "raw_adjacent_1536": False, "replay_json": "b1.json"},
                        ],
                        "start_rows": [
                            {"start_case_id": "a", "p_raw_adjacent_1536": 0.5},
                            {"start_case_id": "b", "p_raw_adjacent_1536": 0.0},
                        ],
                    }
                )
            )

            continuation_payload = run_from_args(
                argparse.Namespace(
                    gate_json=[[gate]],
                    start_json=[[starts]],
                    target_stage="raw_adjacent_1536",
                    mode="continuation",
                    success_threshold=0.25,
                    out_dir=root / "continuation",
                )
            )
            threshold_payload = run_from_args(
                argparse.Namespace(
                    gate_json=[[gate]],
                    start_json=[[starts]],
                    target_stage="raw_adjacent_1536",
                    mode="start-threshold",
                    success_threshold=0.25,
                    out_dir=root / "threshold",
                )
            )

            self.assertEqual(continuation_payload["summary"]["records"], 3)
            self.assertEqual(continuation_payload["summary"]["successes"], 1)
            self.assertEqual(continuation_payload["summary"]["failures"], 2)
            self.assertEqual(continuation_payload["records"][0]["target_tile"], 6144)
            self.assertEqual(continuation_payload["records"][0]["base_start_id"], "a")
            self.assertTrue((root / "continuation" / "records.json").exists())

            self.assertEqual(threshold_payload["summary"]["records"], 2)
            self.assertEqual(threshold_payload["summary"]["successes"], 1)
            self.assertEqual(threshold_payload["records"][0]["label_mode"], "start_threshold")
            self.assertEqual(threshold_payload["records"][0]["label_target_rate"], 0.5)
            self.assertTrue((root / "threshold" / "support_chain_start_labels.html").exists())

    def test_falls_back_to_source_replay_and_frame_when_start_id_is_generated(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            starts = root / "starts.json"
            gate = root / "gate.json"
            starts.write_text(
                json.dumps(
                    {
                        "records": [
                            start_record("stable-id", source_replay="source/replay.json", frame_index=42)
                        ]
                    }
                )
            )
            gate.write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "start_case_id": "continuation_0001_score123",
                                "source_replay": "source/replay.json",
                                "source_frame_index": 42,
                                "raw_adjacent_1536": True,
                                "replay_json": "a1.json",
                            }
                        ],
                        "start_rows": [
                            {
                                "start_case_id": "continuation_0001_score123",
                                "source_replay": "source/replay.json",
                                "source_frame_index": 42,
                                "p_raw_adjacent_1536": 1.0,
                            }
                        ],
                    }
                )
            )

            continuation_payload = run_from_args(
                argparse.Namespace(
                    gate_json=[[gate]],
                    start_json=[[starts]],
                    target_stage="raw_adjacent_1536",
                    mode="continuation",
                    success_threshold=0.25,
                    out_dir=root / "continuation",
                )
            )
            threshold_payload = run_from_args(
                argparse.Namespace(
                    gate_json=[[gate]],
                    start_json=[[starts]],
                    target_stage="raw_adjacent_1536",
                    mode="start-threshold",
                    success_threshold=0.25,
                    out_dir=root / "threshold",
                )
            )

            self.assertEqual(continuation_payload["summary"]["records"], 1)
            self.assertEqual(continuation_payload["summary"]["rejected"], {})
            self.assertEqual(continuation_payload["records"][0]["base_start_id"], "stable-id")
            self.assertEqual(threshold_payload["summary"]["records"], 1)
            self.assertEqual(threshold_payload["records"][0]["base_start_id"], "stable-id")


if __name__ == "__main__":
    unittest.main()
