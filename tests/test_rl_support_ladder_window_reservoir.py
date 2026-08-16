import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from threes_rl.record_replay import state_payload
from threes_rl.sim import SimState, ThreesSim, preview_from_label
from threes_rl.support_ladder_window_reservoir import collect_support_ladder_records, raw_ladder_features, run_from_args


def make_state(board, move_count):
    return SimState(
        board=np.asarray(board, dtype=np.int32),
        preview=preview_from_label("gray"),
        small_counts={"red": 4, "blue": 4, "gray": 4},
        small_pos=0,
        small_seen_total=0,
        span_small_pos=0,
        large_pending=False,
        max_tile=int(np.max(board)),
        move_count=move_count,
        game_over=False,
    )


def write_replay(path: Path, states, seed=7, source_seed=None, source_frame_index=None):
    sim = ThreesSim(np.random.default_rng(seed), starter_tile=1536)
    frames = []
    for idx, state in enumerate(states):
        frames.append(
            {
                "index": idx,
                "state": state_payload(state, sim),
                "move": None if idx == 0 else {"action": "left"},
            }
        )
    path.write_text(
        json.dumps(
            {
                "policy": "test",
                "seed": seed,
                "source_replay": "original/replay.json" if source_seed is not None else None,
                "source_seed": source_seed,
                "source_frame_index": source_frame_index,
                "starter_tile": 1536,
                "final_score": int(frames[-1]["state"]["score"]),
                "final_moves": int(states[-1].move_count),
                "frames": frames,
            }
        )
    )


def ladder_success_states():
    return [
        make_state(
            [
                [1536, 3072, 768, 0],
                [384, 192, 96, 48],
                [24, 12, 6, 3],
                [3, 2, 1, 0],
            ],
            100,
        ),
        make_state(
            [
                [1536, 3072, 768, 0],
                [384, 192, 96, 48],
                [24, 12, 768, 3],
                [3, 2, 1, 0],
            ],
            101,
        ),
        make_state(
            [
                [1536, 3072, 768, 0],
                [384, 192, 96, 48],
                [24, 12, 768, 3],
                [3, 2, 1, 0],
            ],
            102,
        ),
        make_state(
            [
                [1536, 3072, 768, 0],
                [384, 192, 96, 48],
                [24, 12, 1536, 3],
                [3, 2, 1, 0],
            ],
            103,
        ),
        make_state(
            [
                [1536, 1536, 3072, 0],
                [384, 192, 96, 48],
                [24, 12, 768, 3],
                [3, 2, 1, 0],
            ],
            104,
        ),
        make_state(
            [
                [1536, 3072, 3072, 0],
                [384, 192, 96, 48],
                [24, 12, 768, 3],
                [3, 2, 1, 0],
            ],
            105,
        ),
    ]


class SupportLadderWindowReservoirTests(unittest.TestCase):
    def test_raw_near_adjacent_1536_features_distinguish_diagonal_touch_from_gap(self):
        diagonal_touch = np.asarray(
            [
                [1536, 0, 0, 0],
                [0, 1536, 3072, 0],
                [24, 12, 6, 3],
                [3, 2, 1, 0],
            ],
            dtype=np.int32,
        )
        gap_with_blocker = np.asarray(
            [
                [1536, 3072, 1536, 0],
                [0, 24, 12, 0],
                [24, 12, 6, 3],
                [3, 2, 1, 0],
            ],
            dtype=np.int32,
        )

        diagonal_features = raw_ladder_features(diagonal_touch, starter_tile=1536)
        gap_features = raw_ladder_features(gap_with_blocker, starter_tile=1536)

        self.assertTrue(diagonal_features["raw_has_near_adjacent_1536"])
        self.assertTrue(diagonal_features["raw_has_diagonal_touch_1536"])
        self.assertFalse(diagonal_features["raw_has_adjacent_1536"])
        self.assertEqual(diagonal_features["raw_min_pair_distance_1536"], 2)
        self.assertEqual(diagonal_features["raw_min_pair_chebyshev_1536"], 1)
        self.assertFalse(gap_features["raw_has_near_adjacent_1536"])
        self.assertFalse(gap_features["raw_has_diagonal_touch_1536"])
        self.assertEqual(gap_features["raw_min_pair_distance_1536"], 2)
        self.assertEqual(gap_features["raw_min_pair_chebyshev_1536"], 2)

    def test_collects_success_windows_for_ladder_milestones(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            path = Path(tmp) / "success.json"
            write_replay(path, ladder_success_states())

            records, summary = collect_support_ladder_records(
                [path],
                targets=["raw_duplicate_1536", "raw_adjacent_1536", "second_3072"],
                window_size=2,
            )

        self.assertEqual(summary["milestone_presence_replays"]["raw_duplicate_1536"], 1)
        self.assertEqual(summary["milestone_presence_replays"]["second_3072"], 1)
        by_target_outcome = summary["by_target_outcome"]
        self.assertGreater(by_target_outcome["raw_duplicate_1536:success"], 0)
        self.assertGreater(by_target_outcome["raw_adjacent_1536:success"], 0)
        self.assertGreater(by_target_outcome["second_3072:success"], 0)
        second_records = [record for record in records if record["target_milestone"] == "second_3072"]
        self.assertEqual(second_records[0]["prerequisite_milestone"], "raw_adjacent_1536")
        self.assertEqual(second_records[0]["raw_count_1536"], 2)
        self.assertTrue(second_records[0]["raw_has_adjacent_1536"])

    def test_collects_adjacent_768_with_1536_window_after_raw_one_1536(self):
        states = [
            make_state(
                [
                    [1, 3072, 768, 0],
                    [384, 192, 96, 48],
                    [24, 12, 6, 3],
                    [3, 2, 1, 0],
                ],
                100,
            ),
            make_state(
                [
                    [1, 3072, 768, 768],
                    [384, 192, 96, 48],
                    [24, 12, 6, 3],
                    [3, 2, 1, 0],
                ],
                101,
            ),
            make_state(
                [
                    [1, 3072, 768, 0],
                    [1536, 192, 96, 48],
                    [24, 12, 768, 3],
                    [3, 2, 1, 0],
                ],
                102,
            ),
            make_state(
                [
                    [1, 3072, 768, 768],
                    [1536, 192, 96, 48],
                    [24, 12, 6, 3],
                    [3, 2, 1, 0],
                ],
                103,
            ),
        ]
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            path = Path(tmp) / "success.json"
            write_replay(path, states)

            records, summary = collect_support_ladder_records(
                [path],
                targets=["raw_adjacent_768_with_1536"],
                window_size=4,
                include_failures=False,
            )

        self.assertEqual(summary["milestone_presence_replays"]["raw_one_1536"], 1)
        self.assertEqual(summary["milestone_presence_replays"]["raw_adjacent_768_with_1536"], 1)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["target_milestone"], "raw_adjacent_768_with_1536")
        self.assertEqual(records[0]["prerequisite_milestone"], "raw_one_1536")
        self.assertEqual(records[0]["frame_position"], 2)
        self.assertEqual(records[0]["moves_to_milestone"], 1)
        self.assertEqual(records[0]["raw_count_1536"], 1)

    def test_collects_raw_three_768_no_1536_window_after_duplicate_768(self):
        states = [
            make_state(
                [
                    [1, 3072, 768, 768],
                    [384, 192, 96, 48],
                    [24, 12, 6, 3],
                    [3, 2, 1, 0],
                ],
                100,
            ),
            make_state(
                [
                    [1, 3072, 768, 768],
                    [384, 192, 96, 48],
                    [24, 12, 768, 3],
                    [3, 2, 1, 0],
                ],
                101,
            ),
        ]
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            path = Path(tmp) / "success.json"
            write_replay(path, states)

            records, summary = collect_support_ladder_records(
                [path],
                targets=["raw_three_768_no_1536"],
                window_size=2,
                include_failures=False,
            )

        self.assertEqual(summary["milestone_presence_replays"]["raw_duplicate_768"], 1)
        self.assertEqual(summary["milestone_presence_replays"]["raw_three_768_no_1536"], 1)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["target_milestone"], "raw_three_768_no_1536")
        self.assertEqual(records[0]["prerequisite_milestone"], "raw_duplicate_768")
        self.assertEqual(records[0]["moves_to_milestone"], 1)

    def test_collects_raw_four_adjacent_768_no_1536_window_after_raw_four(self):
        states = [
            make_state(
                [
                    [3, 3072, 768, 0],
                    [384, 192, 96, 48],
                    [768, 12, 768, 3],
                    [6, 2, 1, 768],
                ],
                100,
            ),
            make_state(
                [
                    [3, 3072, 768, 768],
                    [384, 192, 96, 48],
                    [768, 12, 768, 3],
                    [6, 2, 1, 768],
                ],
                101,
            ),
        ]
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            path = Path(tmp) / "success.json"
            write_replay(path, states)

            records, summary = collect_support_ladder_records(
                [path],
                targets=["raw_four_adjacent_768_no_1536"],
                window_size=2,
                include_failures=False,
            )

        self.assertEqual(summary["milestone_presence_replays"]["raw_four_768_no_1536"], 1)
        self.assertEqual(summary["milestone_presence_replays"]["raw_four_adjacent_768_no_1536"], 1)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["target_milestone"], "raw_four_adjacent_768_no_1536")
        self.assertEqual(records[0]["prerequisite_milestone"], "raw_four_768_no_1536")
        self.assertEqual(records[0]["moves_to_milestone"], 1)

    def test_collects_failure_controls_after_prerequisite(self):
        failure_states = ladder_success_states()[:3]
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            path = Path(tmp) / "failure.json"
            write_replay(path, failure_states)

            records, summary = collect_support_ladder_records(
                [path],
                targets=["raw_duplicate_1536"],
                window_size=2,
            )

        self.assertEqual(summary["by_target_outcome"], {"raw_duplicate_1536:failure": len(records)})
        self.assertGreater(len(records), 0)
        self.assertEqual(records[0]["outcome"], "failure")
        self.assertEqual(records[0]["prerequisite_milestone"], "raw_duplicate_768")
        self.assertIsNone(records[0]["moves_to_milestone"])

    def test_records_preserve_original_source_group_when_present(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            path = Path(tmp) / "success.json"
            write_replay(path, ladder_success_states(), seed=99, source_seed=1234, source_frame_index=88)

            records, _summary = collect_support_ladder_records(
                [path],
                targets=["raw_adjacent_1536"],
                window_size=2,
            )

        self.assertGreater(len(records), 0)
        self.assertEqual(records[0]["continuation_seed"], 99)
        self.assertEqual(records[0]["original_source_seed"], 1234)
        self.assertEqual(records[0]["original_source_frame_index"], 88)
        self.assertIn("original/replay.json", records[0]["source_group"])

    def test_run_from_args_writes_artifacts(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            tmp_path = Path(tmp)
            replay_path = tmp_path / "success.json"
            out_dir = tmp_path / "out"
            write_replay(replay_path, ladder_success_states())

            args = type(
                "Args",
                (),
                {
                    "replay_json": [[replay_path]],
                    "replay_glob": [],
                    "exclude_path_substring": [],
                    "targets": "second_3072",
                    "window_size": 2,
                    "no_failures": False,
                    "starter": "1536",
                    "max_records": 0,
                    "out_dir": out_dir,
                },
            )()
            payload = run_from_args(args)

            self.assertEqual(payload["summary"]["by_target_outcome"]["second_3072:success"], 1)
            self.assertTrue((out_dir / "support_ladder_windows.json").exists())
            self.assertTrue((out_dir / "records.json").exists())
            self.assertTrue((out_dir / "support_ladder_windows.html").exists())

    def test_run_from_args_can_exclude_path_substrings(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            tmp_path = Path(tmp)
            replay_path = tmp_path / "success.json"
            excluded_path = tmp_path / "synthetic_success.json"
            out_dir = tmp_path / "out"
            write_replay(replay_path, ladder_success_states())
            write_replay(excluded_path, ladder_success_states())

            args = type(
                "Args",
                (),
                {
                    "replay_json": [[replay_path, excluded_path]],
                    "replay_glob": [],
                    "exclude_path_substring": ["synthetic"],
                    "targets": "second_3072",
                    "window_size": 2,
                    "no_failures": False,
                    "starter": "1536",
                    "max_records": 0,
                    "out_dir": out_dir,
                },
            )()
            payload = run_from_args(args)

            self.assertEqual(payload["summary"]["source_replays"], 1)
            self.assertEqual(payload["summary"]["replays_scanned"], 1)

    def test_run_from_args_replay_glob_is_recursive(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            tmp_path = Path(tmp)
            nested = tmp_path / "nested" / "deep"
            nested.mkdir(parents=True)
            replay_path = nested / "success.json"
            out_dir = tmp_path / "out"
            write_replay(replay_path, ladder_success_states())

            args = type(
                "Args",
                (),
                {
                    "replay_json": [],
                    "replay_glob": [str(tmp_path / "**" / "success.json")],
                    "exclude_path_substring": [],
                    "targets": "second_3072",
                    "window_size": 2,
                    "no_failures": False,
                    "starter": "1536",
                    "max_records": 0,
                    "out_dir": out_dir,
                },
            )()
            payload = run_from_args(args)

            self.assertEqual(payload["summary"]["source_replays"], 1)
            self.assertEqual(payload["summary"]["by_target_outcome"]["second_3072:success"], 1)

    def test_fresh_root_only_filters_by_root_provenance(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            tmp_path = Path(tmp)
            fresh_rooted = tmp_path / "fresh_rooted.json"
            derived = tmp_path / "derived.json"
            out_dir = tmp_path / "out"
            write_replay(fresh_rooted, ladder_success_states(), seed=11)
            write_replay(derived, ladder_success_states(), seed=12)
            fresh_payload = json.loads(fresh_rooted.read_text())
            fresh_payload.update(
                {
                    "replay_origin": "continuation",
                    "root_origin": "fresh",
                    "root_replay": "normal/root/replay.json",
                    "root_seed": 99,
                    "root_frame_index": 0,
                    "root_move_count": 0,
                    "root_score": 59049,
                    "root_policy": "fixture_policy",
                }
            )
            fresh_rooted.write_text(json.dumps(fresh_payload))

            args = type(
                "Args",
                (),
                {
                    "replay_json": [[fresh_rooted, derived]],
                    "replay_glob": [],
                    "exclude_path_substring": [],
                    "targets": "second_3072",
                    "window_size": 2,
                    "no_failures": False,
                    "starter": "1536",
                    "max_records": 0,
                    "fresh_root_only": True,
                    "root_origin": None,
                    "out_dir": out_dir,
                },
            )()
            payload = run_from_args(args)

            self.assertEqual(payload["summary"]["replays_considered"], 2)
            self.assertEqual(payload["summary"]["replays_scanned"], 1)
            self.assertEqual(payload["summary"]["by_root_origin"], {"fresh": 1})
            self.assertEqual(payload["records"][0]["root_replay"], "normal/root/replay.json")


if __name__ == "__main__":
    unittest.main()
