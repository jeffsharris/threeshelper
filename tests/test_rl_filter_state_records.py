import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from threes_rl.filter_state_records import filter_records, load_records, run_from_args


def sample_records():
    return [
        {
            "id": "a",
            "synthetic_kind": "adjacent768",
            "target_tile": 3072,
            "target_milestone": "first_3072",
            "outcome": "success",
            "phase": "late_1536",
            "corner_risk": "high_corner_risk",
            "stratum": "late_1536/high_corner_risk",
            "state": {"board": []},
        },
        {
            "id": "b",
            "synthetic_kind": "one1536_adjacent768",
            "target_tile": 6144,
            "target_milestone": "second_3072",
            "outcome": "failure",
            "phase": "endgame_3072p",
            "corner_risk": "medium_corner_risk",
            "stratum": "endgame_3072p/medium_corner_risk",
            "state": {"board": []},
        },
        {
            "id": "c",
            "synthetic_kind": "adjacent768",
            "target_tile": 6144,
            "target_milestone": "second_3072",
            "outcome": "failure",
            "phase": "endgame_3072p",
            "corner_risk": "high_corner_risk",
            "stratum": "endgame_3072p/high_corner_risk",
            "state": {"board": []},
        },
    ]


class FilterStateRecordsTests(unittest.TestCase):
    def test_filter_records_by_target_outcome_and_risk(self):
        records, rejected = filter_records(
            sample_records(),
            target_filter={6144},
            milestone_filter={"second_3072"},
            outcome_filter={"failure"},
            corner_risk_filter={"high_corner_risk"},
        )

        self.assertEqual([record["id"] for record in records], ["c"])
        self.assertEqual(rejected["target_filter"], 1)
        self.assertEqual(rejected["corner_risk_filter"], 1)

    def test_filter_records_by_id(self):
        records, rejected = filter_records(sample_records(), id_filter={"b", "c"})

        self.assertEqual([record["id"] for record in records], ["b", "c"])
        self.assertEqual(rejected["id_filter"], 1)

    def test_filter_records_by_synthetic_kind(self):
        records, rejected = filter_records(sample_records(), synthetic_kind_filter={"adjacent768"})

        self.assertEqual([record["id"] for record in records], ["a", "c"])
        self.assertEqual(rejected["synthetic_kind_filter"], 1)

    def test_run_from_args_writes_filtered_records(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "records.json"
            out_dir = tmp_path / "out"
            source.write_text(json.dumps(sample_records()))

            payload = run_from_args(
                Namespace(
                    state_json=[[source]],
                    id=None,
                    synthetic_kind=None,
                    target="6144",
                    milestone="second_3072",
                    outcome="failure",
                    phase_filter=["endgame"],
                    corner_risk_filter=["high"],
                    max_records=10,
                    out_dir=out_dir,
                )
            )

            self.assertEqual(payload["summary"]["records"], 1)
            self.assertEqual(payload["summary"]["by_target_outcome"], {"6144:failure": 1})
            self.assertEqual(payload["summary"]["by_milestone_outcome"], {"second_3072:failure": 1})
            self.assertTrue((out_dir / "filtered_state_records.json").exists())
            self.assertTrue((out_dir / "records.json").exists())
            self.assertTrue((out_dir / "filtered_state_records.html").exists())


if __name__ == "__main__":
    unittest.main()
