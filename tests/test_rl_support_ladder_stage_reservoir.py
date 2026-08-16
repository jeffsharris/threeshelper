import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from threes_rl.record_replay import state_payload
from threes_rl.sim import SimState, ThreesSim, preview_from_label
from threes_rl.support_ladder_stage_reservoir import collect_stage_records, run_from_args


def make_state(board, move_count=100) -> SimState:
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


def make_record(record_id: str, board) -> dict:
    sim = ThreesSim(np.random.default_rng(7), starter_tile=1536)
    state = make_state(board)
    return {
        "id": record_id,
        "source_replay": f"fixture/{record_id}.json",
        "source_seed": 123,
        "source_frame_index": 45,
        "starter_tile": 1536,
        "state": state_payload(state, sim),
    }


class SupportLadderStageReservoirTests(unittest.TestCase):
    def test_collect_stage_records_filters_duplicate_768_without_adjacency(self):
        duplicate_no_adjacent = make_record(
            "dup",
            [
                [1536, 3072, 768, 0],
                [384, 192, 96, 48],
                [24, 12, 6, 3],
                [768, 2, 1, 0],
            ],
        )
        adjacent = make_record(
            "adj",
            [
                [1536, 3072, 768, 768],
                [384, 192, 96, 48],
                [24, 12, 6, 3],
                [3, 2, 1, 0],
            ],
        )

        records, summary = collect_stage_records(
            [duplicate_no_adjacent, adjacent],
            stages=["duplicate768_no_adjacent"],
            min_tile=3072,
            default_starter_tile=1536,
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["source_record_id"], "dup")
        self.assertEqual(records[0]["stage"], "duplicate768_no_adjacent")
        self.assertGreaterEqual(records[0]["raw_count_768"], 2)
        self.assertFalse(records[0]["raw_has_adjacent_768"])
        self.assertEqual(summary["raw_duplicate_768"], 1)
        self.assertEqual(summary["raw_adjacent_768"], 0)

    def test_collect_stage_records_finds_one_built_1536_with_adjacent_768(self):
        source = make_record(
            "one-built",
            [
                [1536, 3072, 768, 768],
                [384, 192, 1536, 48],
                [24, 12, 6, 3],
                [3, 2, 1, 0],
            ],
        )

        records, summary = collect_stage_records(
            [source],
            stages=["one1536_adjacent768"],
            min_tile=3072,
            default_starter_tile=1536,
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["stage"], "one1536_adjacent768")
        self.assertEqual(records[0]["masked_count_1536"], 1)
        self.assertTrue(records[0]["raw_has_adjacent_768"])
        self.assertEqual(summary["one_built_1536"], 1)

    def test_collect_stage_records_finds_four_768_without_built_1536(self):
        source = make_record(
            "four-768",
            [
                [1536, 3072, 768, 768],
                [384, 192, 96, 48],
                [24, 12, 768, 3],
                [768, 2, 1, 0],
            ],
        )
        with_built_1536 = make_record(
            "with-built",
            [
                [1536, 3072, 768, 768],
                [384, 192, 1536, 48],
                [24, 12, 768, 3],
                [768, 2, 1, 0],
            ],
        )

        records, summary = collect_stage_records(
            [source, with_built_1536],
            stages=["four768_no_built1536"],
            min_tile=3072,
            default_starter_tile=1536,
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["source_record_id"], "four-768")
        self.assertEqual(records[0]["stage"], "four768_no_built1536")
        self.assertEqual(records[0]["raw_count_768"], 4)
        self.assertEqual(records[0]["masked_count_1536"], 0)
        self.assertEqual(summary["raw_duplicate_768"], 1)

    def test_source_cap_is_applied_after_sorting(self):
        low = make_record(
            "low",
            [
                [1536, 3072, 768, 0],
                [384, 192, 96, 48],
                [24, 12, 6, 3],
                [768, 2, 1, 0],
            ],
        )
        high = make_record(
            "high",
            [
                [1536, 3072, 768, 384],
                [384, 192, 96, 48],
                [24, 12, 6, 3],
                [768, 192, 96, 48],
            ],
        )
        for idx, record in enumerate((low, high)):
            record["source_replay"] = "fixture/shared.json"
            record["source_seed"] = 777
            record["source_frame_index"] = idx

        records, summary = collect_stage_records(
            [low, high],
            stages=["duplicate768_no_adjacent"],
            min_tile=3072,
            default_starter_tile=1536,
            max_per_source=1,
            sort_by="score",
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["source_record_id"], "high")
        self.assertEqual(summary["rejected"]["max_per_source"], 1)

    def test_dedupe_keeps_same_board_with_different_tile_cycle(self):
        board = [
            [1536, 3072, 768, 0],
            [384, 192, 96, 48],
            [24, 12, 6, 3],
            [768, 2, 1, 0],
        ]
        first = make_record("first", board)
        second = make_record("second", board)
        second["state"]["tile_cycle"]["small_pos"] = 1
        second["state"]["tile_cycle"]["small_seen_total"] = 13

        records, summary = collect_stage_records(
            [first, second],
            stages=["duplicate768_no_adjacent"],
            min_tile=3072,
            default_starter_tile=1536,
        )

        self.assertEqual(len(records), 2)
        self.assertEqual(summary["rejected"].get("duplicate_state", 0), 0)

    def test_record_ids_distinguish_same_source_frame_states(self):
        first = make_record(
            "first",
            [
                [1536, 3072, 768, 0],
                [384, 192, 96, 48],
                [24, 12, 6, 3],
                [768, 2, 1, 0],
            ],
        )
        second = make_record(
            "second",
            [
                [1536, 3072, 768, 0],
                [384, 192, 96, 48],
                [24, 12, 6, 3],
                [2, 768, 1, 0],
            ],
        )
        for record in (first, second):
            record["source_replay"] = "fixture/shared.json"
            record["source_seed"] = 777
            record["source_frame_index"] = 45

        records, _summary = collect_stage_records(
            [first, second],
            stages=["duplicate768_no_adjacent"],
            min_tile=3072,
            default_starter_tile=1536,
        )

        self.assertEqual(len(records), 2)
        self.assertNotEqual(records[0]["id"], records[1]["id"])

    def test_run_from_args_writes_artifacts(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            tmp_path = Path(tmp)
            source_path = tmp_path / "records.json"
            out_dir = tmp_path / "out"
            source_path.write_text(
                json.dumps(
                    {
                        "records": [
                            make_record(
                                "dup",
                                [
                                    [1536, 3072, 768, 0],
                                    [384, 192, 96, 48],
                                    [24, 12, 6, 3],
                                    [768, 2, 1, 0],
                                ],
                            )
                        ]
                    }
                )
            )
            args = type(
                "Args",
                (),
                {
                    "state_json": [[source_path]],
                    "stages": "duplicate768_no_adjacent",
                    "min_tile": 3072,
                    "phase_filter": None,
                    "corner_risk_filter": None,
                    "starter": "1536",
                    "max_records": 0,
                    "max_per_source": 0,
                    "max_per_stage": 0,
                    "sort_by": "support",
                    "out_dir": out_dir,
                },
            )()

            payload = run_from_args(args)

            self.assertEqual(payload["summary"]["records"], 1)
            self.assertTrue((out_dir / "support_ladder_stage_reservoir.json").exists())
            self.assertTrue((out_dir / "records.json").exists())
            self.assertTrue((out_dir / "support_ladder_stage_reservoir.html").exists())


if __name__ == "__main__":
    unittest.main()
