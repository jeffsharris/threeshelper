import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from threes_rl.human_diagnostics_batch import build_parser, run_batch
from threes_rl.sim import board_to_tokens, simulate_base_move


def make_events() -> list[dict[str, object]]:
    before = np.asarray(
        [
            [1536, 0, 1, 0],
            [2, 0, 3, 0],
            [1, 0, 2, 0],
            [3, 1, 2, 0],
        ],
        dtype=np.int32,
    )
    shifted, eligible = simulate_base_move(before, "left")
    after = shifted.copy()
    after[0, 3] = 1
    return [
        {
            "type": "game_start",
            "game_index": 1,
            "board": board_to_tokens(before),
            "preview_label": "blue",
        },
        {
            "type": "observed_move",
            "game_index": 1,
            "move_index_start": 1,
            "move_index": 1,
            "before_board": board_to_tokens(before),
            "before_preview_label": "blue",
            "after_board": board_to_tokens(after),
            "after_preview_label": "red",
            "unknown_board": False,
            "unknown_preview": False,
            "preview_check": {"valid": True},
            "transition_check": {
                "valid": True,
                "inserted_value": 1,
                "inserted_pos": [0, 3],
                "eligible_positions": [list(pos) for pos in eligible],
                "expected_values": [1],
            },
            "direction": "left",
            "direction_sequence": ["left"],
            "step_count": 1,
            "transition_path": [
                {
                    "direction": "left",
                    "preview_label": "blue",
                    "inserted_value": 1,
                    "inserted_pos": [0, 3],
                    "eligible_positions": [list(pos) for pos in eligible],
                    "expected_values": [1],
                    "after_board": board_to_tokens(after),
                }
            ],
        },
    ]


def batch_args(*parts: str):
    return build_parser().parse_args(list(parts))


def write_processed_session(tmp_path: Path, *, session: str, max_excl_tile: int) -> Path:
    events_path = tmp_path / "human_watch" / session / "events.jsonl"
    events_path.parent.mkdir(parents=True)
    events_path.write_text("{}\n")
    out_dir = tmp_path / "diagnostics" / session
    replay_path = out_dir / "imported_replays" / f"{session}.json"
    replay_path.parent.mkdir(parents=True)
    replay = {
        "starter_tile": 1536,
        "frames": [
            {
                "index": 0,
                "state": {
                    "move_count": 0,
                    "board": [[1536, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                },
            },
            {
                "index": 1,
                "state": {
                    "move_count": 40,
                    "board": [[1536, max_excl_tile, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                },
            },
        ],
    }
    replay_path.write_text(json.dumps(replay))
    manifest_path = out_dir / "human_diagnostics_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "games_seen": 1,
                "games_imported": 1,
                "games_skipped": 0,
                "imports": [
                    {
                        "replays": [
                            {
                                "json": str(replay_path),
                                "final_score": 12345,
                                "final_max_tile": max(1536, max_excl_tile),
                            }
                        ]
                    }
                ],
            }
        )
    )
    return events_path


class HumanDiagnosticsBatchTests(unittest.TestCase):
    def test_batch_reports_waiting_when_no_events_exist(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            tmp_path = Path(tmp)
            args = batch_args(
                "--events-glob",
                str(tmp_path / "human_watch" / "*" / "events.jsonl"),
                "--out-root",
                str(tmp_path / "diagnostics"),
                "--no-policy",
            )

            payload = run_batch(args)

            self.assertEqual(payload["status"], "waiting_for_human_data")
            self.assertEqual(payload["totals"]["sessions"], 0)
            self.assertTrue((tmp_path / "diagnostics" / "human_diagnostics_batch.json").exists())
            self.assertTrue((tmp_path / "diagnostics" / "human_diagnostics_batch.html").exists())

    def test_batch_run_processes_pending_session(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            tmp_path = Path(tmp)
            events_path = tmp_path / "human_watch" / "session_a" / "events.jsonl"
            events_path.parent.mkdir(parents=True)
            events_path.write_text("\n".join(json.dumps(event) for event in make_events()))
            args = batch_args(
                "--events-glob",
                str(tmp_path / "human_watch" / "*" / "events.jsonl"),
                "--out-root",
                str(tmp_path / "diagnostics"),
                "--no-policy",
                "--run",
                "--min-tile",
                "3",
                "--phase-filter",
                "early",
                "--corner-risk-filter",
                "low,medium,high",
                "--transition-targets",
                "3,1536",
                "--support-ladder-targets",
                "raw_one_1536",
            )

            payload = run_batch(args)

            self.assertEqual(payload["status"], "processed")
            self.assertEqual(payload["totals"]["sessions"], 1)
            self.assertEqual(payload["totals"]["processed_sessions"], 1)
            self.assertEqual(payload["totals"]["games_imported"], 1)
            self.assertEqual(payload["totals"]["games_reaching_nonstarter_1536"], 0)
            self.assertEqual(payload["target_intake"]["ready_for_human_root_labeling"], False)
            session = payload["sessions"][0]
            self.assertEqual(session["session_id"], "session_a")
            self.assertEqual(session["status"], "processed")
            self.assertTrue(Path(str(session["manifest_json"])).exists())

            dry_run = run_batch(batch_args(
                "--events-glob",
                str(tmp_path / "human_watch" / "*" / "events.jsonl"),
                "--out-root",
                str(tmp_path / "diagnostics"),
                "--no-policy",
            ))
            self.assertEqual(dry_run["status"], "current")
            self.assertEqual(dry_run["totals"]["current_sessions"], 1)

    def test_batch_counts_nonstarter_milestone_targets_from_replay_frames(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            tmp_path = Path(tmp)
            write_processed_session(tmp_path, session="session_a", max_excl_tile=3072)
            args = batch_args(
                "--events-glob",
                str(tmp_path / "human_watch" / "*" / "events.jsonl"),
                "--out-root",
                str(tmp_path / "diagnostics"),
                "--no-policy",
            )

            payload = run_batch(args)

            self.assertEqual(payload["status"], "current")
            self.assertEqual(payload["totals"]["games_reaching_nonstarter_1536"], 1)
            self.assertEqual(payload["totals"]["games_reaching_3072"], 1)
            self.assertEqual(payload["totals"]["highest_max_tile_excl_starter"], 3072)
            self.assertEqual(payload["target_intake"]["current_games_reaching_nonstarter_1536"], 1)
            self.assertEqual(payload["target_intake"]["current_games_reaching_3072"], 1)
            self.assertEqual(payload["target_intake"]["remaining_games_reaching_nonstarter_1536"], 4)
            self.assertEqual(payload["target_intake"]["remaining_games_reaching_3072"], 0)


if __name__ == "__main__":
    unittest.main()
