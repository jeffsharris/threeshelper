import json
import shutil
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

from threes_rl.continue_from_replays import (
    collect_start_cases,
    load_selection_records,
    run_from_args,
    select_start_cases_from_records,
)
from threes_rl.record_replay import state_payload
from threes_rl.sim import SimState, ThreesSim, preview_from_label


def high_state() -> SimState:
    return SimState(
        board=np.asarray(
            [
                [1536, 3072, 0, 0],
                [768, 384, 192, 96],
                [48, 24, 12, 6],
                [3, 2, 1, 0],
            ],
            dtype=np.int32,
        ),
        preview=preview_from_label("blue"),
        small_counts={"blue": 4, "red": 4, "gray": 4},
        small_pos=0,
        small_seen_total=0,
        span_small_pos=0,
        large_pending=False,
        max_tile=3072,
        move_count=120,
        game_over=False,
    )


def write_replay(path: Path) -> None:
    sim = ThreesSim(np.random.default_rng(7), starter_tile=1536)
    state = high_state()
    payload = {
        "policy": "test",
        "seed": 7,
        "starter_tile": 1536,
        "final_score": 123456,
        "frames": [
            {"index": 0, "state": state_payload(state, sim), "move": None},
        ],
    }
    path.write_text(json.dumps(payload))


class ContinueFromReplayTests(unittest.TestCase):
    def test_collect_start_cases_filters_by_min_tile_and_phase(self):
        with tempfile.TemporaryDirectory() as tmp:
            replay_path = Path(tmp) / "replay.json"
            write_replay(replay_path)

            cases = collect_start_cases(
                [replay_path],
                min_tile=3072,
                phase_filter={"endgame_3072p"},
                default_starter_tile=1536,
            )

        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].start_max_tile_excl_starter, 3072)
        self.assertEqual(cases[0].phase, "endgame_3072p")

    def test_run_from_args_writes_summary_and_top_replay(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            tmp_path = Path(tmp)
            replay_path = tmp_path / "replay.json"
            out_dir = tmp_path / "out"
            write_replay(replay_path)

            summary = run_from_args(
                Namespace(
                    policy="greedy",
                    replay_json=[[replay_path]],
                    starter="1536",
                    start_state_min_tile=3072,
                    phase_filter="endgame",
                    max_starts=1,
                    repeats_per_start=1,
                    max_moves=4,
                    seed=99,
                    keep_top_games=1,
                    progress_every=0,
                    out_dir=out_dir,
                )
            )

            self.assertEqual(summary["continuations"], 1)
            self.assertIn("high_score_delta", summary)
            self.assertTrue((out_dir / "summary.json").exists())
            self.assertTrue((out_dir / "records.json").exists())
            self.assertTrue((out_dir / "top_games" / "manifest.json").exists())
            self.assertTrue((out_dir / "top_delta_games" / "manifest.json").exists())
            records = json.loads((out_dir / "records.json").read_text())
            manifest = json.loads((out_dir / "top_games" / "manifest.json").read_text())
            delta_manifest = json.loads((out_dir / "top_delta_games" / "manifest.json").read_text())
            self.assertEqual(len(records), 1)
            self.assertIn("start_case_id", records[0])
            self.assertEqual(records[0]["source_frame_index"], 0)
            self.assertIn("score_delta", records[0])
            self.assertEqual(len(manifest), 1)
            self.assertIn("start_case_id", manifest[0])
            self.assertEqual(len(delta_manifest), 1)
            self.assertTrue(Path(manifest[0]["json"]).exists())
            self.assertTrue(Path(delta_manifest[0]["json"]).exists())
            shutil.rmtree(out_dir)

    def test_run_from_args_resumes_checkpointed_continuations(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            tmp_path = Path(tmp)
            replay_path = tmp_path / "replay.json"
            out_dir = tmp_path / "out"
            write_replay(replay_path)
            args = Namespace(
                policy="greedy",
                replay_json=[[replay_path]],
                starter="1536",
                start_state_min_tile=3072,
                phase_filter="endgame",
                max_starts=1,
                repeats_per_start=1,
                max_moves=4,
                seed=99,
                keep_top_games=1,
                progress_every=0,
                checkpoint_continuations=True,
                continuation_progress_json=None,
                out_dir=out_dir,
            )

            first = run_from_args(args)
            second = run_from_args(args)

            self.assertEqual(first["continuations"], 1)
            self.assertEqual(first["continuations_ran"], 1)
            self.assertEqual(first["continuations_resumed"], 0)
            self.assertEqual(second["continuations"], 1)
            self.assertEqual(second["continuations_ran"], 0)
            self.assertEqual(second["continuations_resumed"], 1)
            self.assertEqual(first["high_score_delta"], second["high_score_delta"])
            self.assertTrue((out_dir / "continuation_progress.json").exists())

    def test_select_start_cases_from_changed_records_preserves_record_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            first = tmp_path / "first.json"
            second = tmp_path / "second.json"
            write_replay(first)
            write_replay(second)
            cases = collect_start_cases(
                [first, second],
                min_tile=3072,
                phase_filter={"endgame_3072p"},
                default_starter_tile=1536,
            )
            records = [
                {"source_replay": str(second), "frame_index": 0, "changed": True, "top_two_changed": True},
                {"source_replay": str(first), "frame_index": 0, "changed": False, "top_two_changed": True},
                {"source_replay": str(first), "frame_index": 0, "changed": True, "top_two_changed": True},
                {"source_replay": str(second), "frame_index": 999, "changed": True, "top_two_changed": True},
            ]

            selected, stats = select_start_cases_from_records(
                cases,
                records,
                records_filter="changed",
                max_starts=0,
            )

        self.assertEqual([case.source_replay for case in selected], [str(second), str(first)])
        self.assertEqual(stats["records_matched"], 2)
        self.assertEqual(stats["records_skipped_filter"], 1)
        self.assertEqual(stats["records_skipped_missing"], 1)

    def test_select_start_cases_from_records_accepts_source_frame_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "replay.json"
            write_replay(path)
            cases = collect_start_cases(
                [path],
                min_tile=3072,
                phase_filter={"endgame_3072p"},
                default_starter_tile=1536,
            )

            selected, stats = select_start_cases_from_records(
                cases,
                [{"source_replay": str(path), "source_frame_index": 0, "changed": True}],
                records_filter="changed",
                max_starts=0,
            )

        self.assertEqual(len(selected), 1)
        self.assertEqual(stats["records_matched"], 1)

    def test_collect_start_cases_accepts_reservoir_records(self):
        sim = ThreesSim(np.random.default_rng(7), starter_tile=1536)
        state = high_state()
        with tempfile.TemporaryDirectory() as tmp:
            reservoir_path = Path(tmp) / "reservoir.json"
            source_replay = Path(tmp) / "source_replay.json"
            reservoir_path.write_text(
                json.dumps(
                    {
                        "kind": "threes_high_board_reservoir",
                        "records": [
                            {
                                "id": "record-1",
                                "source_replay": str(source_replay),
                                "source_seed": 42,
                                "source_frame_index": 9,
                                "source_origin": "fresh",
                                "root_origin": "fresh",
                                "root_replay": "normal/root/replay.json",
                                "root_seed": 42,
                                "root_frame_index": 0,
                                "root_move_count": 0,
                                "root_policy": "fixture_policy",
                                "starter_tile": 1536,
                                "state": state_payload(state, sim),
                            }
                        ],
                    }
                )
            )

            cases = collect_start_cases(
                [reservoir_path],
                min_tile=3072,
                phase_filter={"endgame_3072p"},
                default_starter_tile=1536,
            )

        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].id, "record-1")
        self.assertEqual(cases[0].source_replay, str(source_replay))
        self.assertEqual(cases[0].source_seed, 42)
        self.assertEqual(cases[0].frame_index, 9)
        self.assertEqual(cases[0].root_origin, "fresh")
        self.assertEqual(cases[0].root_replay, "normal/root/replay.json")
        self.assertEqual(cases[0].start_max_tile_excl_starter, 3072)

    def test_select_start_cases_from_records_matches_reservoir_source_metadata(self):
        sim = ThreesSim(np.random.default_rng(7), starter_tile=1536)
        state = high_state()
        with tempfile.TemporaryDirectory() as tmp:
            reservoir_path = Path(tmp) / "reservoir.json"
            source_replay = Path(tmp) / "source_replay.json"
            reservoir_path.write_text(
                json.dumps(
                    {
                        "records": [
                            {
                                "id": "record-1",
                                "source_replay": str(source_replay),
                                "source_seed": 42,
                                "source_frame_index": 9,
                                "starter_tile": 1536,
                                "state": state_payload(state, sim),
                            }
                        ],
                    }
                )
            )
            cases = collect_start_cases(
                [reservoir_path],
                min_tile=3072,
                phase_filter={"endgame_3072p"},
                default_starter_tile=1536,
            )
            selected, stats = select_start_cases_from_records(
                cases,
                [{"source_replay": str(source_replay), "frame_index": 9, "changed": True}],
                records_filter="changed",
            )

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].id, "record-1")
        self.assertEqual(stats["records_matched"], 1)

    def test_load_selection_records_accepts_artifact_object(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "top_records.json"
            path.write_text(json.dumps({"kind": "top", "records": [{"id": "a"}, {"id": "b"}]}))

            records = load_selection_records(path)

        self.assertEqual([record["id"] for record in records], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
