import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from threes_rl.record_replay import state_payload
from threes_rl.sim import SimState, ThreesSim, preview_from_label
from threes_rl.support_chain_milestone_reservoir import extract_replay_records, run


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


def write_replay(path: Path, states):
    sim = ThreesSim(np.random.default_rng(7), starter_tile=1536)
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
                "seed": 7,
                "starter_tile": 1536,
                "final_score": int(frames[-1]["state"]["score"]),
                "final_moves": int(states[-1].move_count),
                "frames": frames,
            }
        )
    )


class SupportChainMilestoneReservoirTests(unittest.TestCase):
    def test_extracts_first_matching_milestone_state(self):
        states = [
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
                    [1536, 3072, 768, 768],
                    [384, 192, 96, 48],
                    [24, 12, 6, 3],
                    [3, 2, 1, 0],
                ],
                101,
            ),
        ]
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            replay_path = Path(tmp) / "failure.json"
            write_replay(replay_path, states)

            records, rejected = extract_replay_records(
                replay_path,
                milestones=["raw_adjacent_768"],
                outcome_filter={"failure"},
            )

        self.assertEqual(rejected, {})
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["target_milestone"], "raw_adjacent_768")
        self.assertEqual(records[0]["outcome"], "failure")
        self.assertEqual(records[0]["frames_after_first_3072"], 1)
        self.assertEqual(records[0]["raw_count_768"], 2)

    def test_extracts_adjacent_768_with_and_without_raw_1536(self):
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
                    [1536, 192, 96, 48],
                    [24, 12, 6, 3],
                    [3, 2, 1, 0],
                ],
                101,
            ),
        ]
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            replay_path = Path(tmp) / "failure.json"
            write_replay(replay_path, states)

            records, rejected = extract_replay_records(
                replay_path,
                milestones=["raw_adjacent_768_no_1536", "raw_adjacent_768_with_1536", "raw_one_1536"],
                outcome_filter={"failure"},
            )

        self.assertEqual(rejected, {})
        by_milestone = {record["target_milestone"]: record for record in records}
        self.assertEqual(by_milestone["raw_adjacent_768_no_1536"]["frame_position"], 0)
        self.assertEqual(by_milestone["raw_adjacent_768_with_1536"]["frame_position"], 1)
        self.assertEqual(by_milestone["raw_one_1536"]["frame_position"], 1)

    def test_extracts_four_raw_768_before_raw_1536(self):
        states = [
            make_state(
                [
                    [1, 3072, 768, 768],
                    [384, 768, 768, 48],
                    [24, 12, 6, 3],
                    [3, 2, 1, 0],
                ],
                100,
            ),
            make_state(
                [
                    [1, 3072, 768, 768],
                    [1536, 768, 768, 48],
                    [24, 12, 6, 3],
                    [3, 2, 1, 0],
                ],
                101,
            ),
        ]
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            replay_path = Path(tmp) / "failure.json"
            write_replay(replay_path, states)

            records, rejected = extract_replay_records(
                replay_path,
                milestones=["raw_four_768_no_1536", "raw_one_1536"],
                outcome_filter={"failure"},
            )

        self.assertEqual(rejected, {})
        by_milestone = {record["target_milestone"]: record for record in records}
        self.assertEqual(by_milestone["raw_four_768_no_1536"]["frame_position"], 0)
        self.assertEqual(by_milestone["raw_four_768_no_1536"]["raw_count_768"], 4)
        self.assertEqual(by_milestone["raw_four_768_no_1536"]["raw_count_1536"], 0)
        self.assertEqual(by_milestone["raw_one_1536"]["frame_position"], 1)

    def test_pre_event_only_rejects_post_event_milestone(self):
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
                    [1, 3072, 3072, 0],
                    [384, 192, 96, 48],
                    [24, 12, 6, 3],
                    [3, 2, 1, 0],
                ],
                101,
            ),
            make_state(
                [
                    [1, 3072, 768, 768],
                    [384, 192, 96, 48],
                    [24, 12, 6, 3],
                    [3, 2, 1, 0],
                ],
                102,
            ),
        ]
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            replay_path = Path(tmp) / "success.json"
            write_replay(replay_path, states)

            records, rejected = extract_replay_records(
                replay_path,
                milestones=["raw_adjacent_768_no_1536"],
                outcome_filter={"success"},
                pre_event_only=True,
            )
            unrestricted, _ = extract_replay_records(
                replay_path,
                milestones=["raw_adjacent_768_no_1536"],
                outcome_filter={"success"},
                pre_event_only=False,
            )

        self.assertEqual(records, [])
        self.assertEqual(rejected, {"missing_raw_adjacent_768_no_1536": 1})
        self.assertEqual(len(unrestricted), 1)
        self.assertEqual(unrestricted[0]["frame_position"], 2)

    def test_run_writes_records_and_summary(self):
        success_states = [
            make_state(
                [
                    [1536, 3072, 768, 768],
                    [384, 192, 96, 48],
                    [24, 12, 6, 3],
                    [3, 2, 1, 0],
                ],
                100,
            ),
            make_state(
                [
                    [1536, 3072, 3072, 0],
                    [384, 192, 96, 48],
                    [24, 12, 6, 3],
                    [3, 2, 1, 0],
                ],
                101,
            ),
        ]
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            tmp_path = Path(tmp)
            replay_path = tmp_path / "success.json"
            out_dir = tmp_path / "out"
            write_replay(replay_path, success_states)

            payload = run(
                [replay_path],
                milestones=["raw_adjacent_768"],
                outcome_filter={"success"},
                out_dir=out_dir,
            )

            self.assertEqual(payload["summary"]["records"], 1)
            self.assertEqual(payload["summary"]["by_outcome"], {"success": 1})
            self.assertTrue((out_dir / "records.json").exists())
            self.assertTrue((out_dir / "support_chain_milestone_reservoir.html").exists())


if __name__ == "__main__":
    unittest.main()
