import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from threes_rl.record_replay import state_payload
from threes_rl.sim import SimState, ThreesSim, preview_from_label
from threes_rl.support_ladder_state_generator import generate_records, run_from_args


def make_high_board_state() -> SimState:
    board = np.asarray(
        [
            [1536, 3072, 384, 192],
            [48, 24, 12, 6],
            [3, 2, 1, 0],
            [96, 192, 384, 0],
        ],
        dtype=np.int32,
    )
    return SimState(
        board=board,
        preview=preview_from_label("gray"),
        small_counts={"red": 4, "blue": 4, "gray": 4},
        small_pos=0,
        small_seen_total=0,
        span_small_pos=0,
        large_pending=False,
        max_tile=3072,
        move_count=180,
        game_over=False,
    )


def make_source_record(record_id: str = "source-1") -> dict:
    sim = ThreesSim(np.random.default_rng(7), starter_tile=1536)
    state = make_high_board_state()
    return {
        "id": record_id,
        "source_replay": "fixture/replay.json",
        "source_seed": 123,
        "source_frame_index": 45,
        "starter_tile": 1536,
        "state": state_payload(state, sim),
    }


class SupportLadderStateGeneratorTests(unittest.TestCase):
    def test_generate_records_creates_targeted_support_ladder_modes(self):
        records, summary = generate_records(
            [make_source_record()],
            modes=["adjacent768", "one1536_adjacent768", "adjacent1536"],
            variants_per_mode=1,
            default_starter_tile=1536,
            min_tile=3072,
            max_per_source=0,
        )

        by_mode = {record["synthetic_kind"]: record for record in records}
        self.assertEqual(set(by_mode), {"adjacent768", "one1536_adjacent768", "adjacent1536"})
        self.assertTrue(by_mode["adjacent768"]["raw_has_adjacent_768"])
        self.assertGreaterEqual(by_mode["adjacent768"]["raw_count_768"], 2)
        self.assertTrue(by_mode["one1536_adjacent768"]["raw_has_adjacent_768"])
        self.assertGreaterEqual(by_mode["one1536_adjacent768"]["raw_count_1536"], 2)
        self.assertTrue(by_mode["adjacent1536"]["raw_has_adjacent_1536"])
        self.assertGreaterEqual(by_mode["adjacent1536"]["raw_count_1536"], 2)
        self.assertEqual(summary["records"], 3)
        self.assertEqual(summary["raw_adjacent_1536"], 1)
        self.assertEqual(len({record["id"] for record in records}), 3)

    def test_generate_records_can_cap_outputs_per_source(self):
        records, summary = generate_records(
            [make_source_record()],
            modes=["adjacent768", "one1536_adjacent768", "adjacent1536"],
            variants_per_mode=1,
            default_starter_tile=1536,
            min_tile=3072,
            max_per_source=2,
        )

        self.assertEqual(len(records), 2)
        self.assertEqual(summary["max_per_source"], 2)
        self.assertEqual(summary["source_count_max"], 2)
        self.assertEqual(summary["rejected"]["max_per_source"], 1)

    def test_generate_records_can_stage_768_material_counts(self):
        records, summary = generate_records(
            [make_source_record()],
            modes=["identity", "three768", "four768"],
            variants_per_mode=1,
            default_starter_tile=1536,
            min_tile=3072,
            max_per_source=0,
        )

        by_mode = {record["synthetic_kind"]: record for record in records}
        self.assertEqual(set(by_mode), {"identity", "three768", "four768"})
        self.assertEqual(by_mode["identity"]["raw_count_768"], 0)
        self.assertEqual(by_mode["three768"]["raw_count_768"], 3)
        self.assertEqual(by_mode["four768"]["raw_count_768"], 4)
        self.assertEqual(by_mode["three768"]["raw_count_1536"], 1)
        self.assertEqual(by_mode["four768"]["raw_count_1536"], 1)
        self.assertEqual(summary["mode_counts"]["three768"], 1)
        self.assertEqual(summary["mode_counts"]["four768"], 1)

    def test_run_from_args_writes_payloads_for_continuation_runner(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            tmp_path = Path(tmp)
            source_path = tmp_path / "records.json"
            out_dir = tmp_path / "out"
            source_path.write_text(json.dumps({"records": [make_source_record()]}))

            args = type(
                "Args",
                (),
                {
                    "state_json": [[source_path]],
                    "modes": "adjacent768,adjacent1536",
                    "variants_per_mode": 1,
                    "starter": "1536",
                    "min_tile": 3072,
                    "max_records": 0,
                    "max_per_source": 0,
                    "out_dir": out_dir,
                },
            )()
            payload = run_from_args(args)

            self.assertEqual(payload["summary"]["records"], 2)
            self.assertTrue((out_dir / "synthetic_support_ladder_states.json").exists())
            self.assertTrue((out_dir / "records.json").exists())
            self.assertTrue((out_dir / "synthetic_support_ladder_states.html").exists())
            written = json.loads((out_dir / "synthetic_support_ladder_states.json").read_text())
            self.assertEqual(len(written["records"]), 2)
            self.assertIn("state", written["records"][0])
            self.assertTrue(written["records"][0]["synthetic"])


if __name__ == "__main__":
    unittest.main()
