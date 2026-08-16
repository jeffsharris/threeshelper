import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from threes_rl.first_action_path_forensics import run_from_args


def milestone(frame):
    if frame is None:
        return None
    return {"frames_after_first_3072": frame, "frame_position": frame, "frame_index": frame, "move_count": 100 + frame}


class FirstActionPathForensicsTests(unittest.TestCase):
    def test_run_from_args_joins_preview_and_support_records(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            tmp_path = Path(tmp)
            large_path = tmp_path / "large_records.json"
            support_path = tmp_path / "support.json"
            out_dir = tmp_path / "out"
            large_path.write_text(
                json.dumps(
                    [
                        {
                            "source_replay": "replay-success.json",
                            "source_seed": 101,
                            "source_frame_index": 20,
                            "seed": 1,
                            "first_action": "left",
                            "score_delta": 900,
                            "reached_6144": True,
                            "first_large_ge_768": {"max_candidate": 768},
                            "first_large_ge_768_frame_after_start": 7,
                            "preview_after_first_label": "blue",
                        },
                        {
                            "source_replay": "replay-failure.json",
                            "source_seed": 101,
                            "source_frame_index": 20,
                            "seed": 2,
                            "first_action": "right",
                            "score_delta": 100,
                            "reached_6144": False,
                            "first_large_ge_768": None,
                            "first_large_ge_768_frame_after_start": None,
                            "preview_after_first_label": "red",
                        },
                    ]
                )
            )
            support_path.write_text(
                json.dumps(
                    {
                        "cases": [
                            {
                                "source_replay": "replay-success.json",
                                "seed": 1,
                                "outcome": "success",
                                "first_6144": milestone(14),
                                "second_3072_event": milestone(13),
                                "milestones": {
                                    "first_raw_duplicate_1536": milestone(3),
                                    "first_raw_adjacent_pair_1536": milestone(11),
                                },
                            },
                            {
                                "source_replay": "replay-failure.json",
                                "seed": 2,
                                "outcome": "failure",
                                "first_6144": None,
                                "second_3072_event": None,
                                "milestones": {
                                    "first_raw_duplicate_1536": milestone(2),
                                    "first_raw_adjacent_pair_1536": None,
                                },
                            },
                        ]
                    }
                )
            )

            payload = run_from_args(
                Namespace(
                    large_preview_records=large_path,
                    support_chain_json=support_path,
                    out_dir=out_dir,
                )
            )

            overall = payload["summary"]["overall"]
            self.assertEqual(overall["records"], 2)
            self.assertEqual(overall["success"], 1)
            self.assertEqual(overall["raw_duplicate_1536"], 2)
            self.assertEqual(overall["raw_adjacent_1536"], 1)
            self.assertEqual(overall["first_large_ge_768"], 1)
            self.assertEqual(len(payload["summary"]["by_source_action"]), 2)
            self.assertTrue((out_dir / "first_action_path_forensics.json").exists())
            self.assertTrue((out_dir / "records.json").exists())
            self.assertTrue((out_dir / "first_action_path_forensics.html").exists())


if __name__ == "__main__":
    unittest.main()
