import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from threes_rl.record_replay import state_payload
from threes_rl.sim import SimState, ThreesSim, preview_from_label
from threes_rl.support_loss_action_labels import run_labels, target_reached


def support_loss_state() -> SimState:
    board = np.asarray(
        [
            [1536, 0, 0, 0],
            [384, 384, 384, 0],
            [0, 0, 0, 0],
            [3, 6, 12, 24],
        ],
        dtype=np.int32,
    )
    return SimState(
        board=board,
        preview=preview_from_label("gray"),
        small_counts={"red": 4, "blue": 4, "gray": 4},
        small_pos=8,
        small_seen_total=40,
        span_small_pos=0,
        large_pending=False,
        max_tile=int(board.max(initial=0)),
        move_count=80,
        game_over=False,
    )


def record_payload() -> dict:
    sim = ThreesSim(np.random.default_rng(7), starter_tile=1536)
    state = support_loss_state()
    return {
        "id": "support-loss-case",
        "starter_tile": 1536,
        "source_replay": "fixture/source.json",
        "source_seed": 7,
        "source_frame_index": 80,
        "source_next_action": "left",
        "source_policy": "fixture",
        "root_origin": "fresh",
        "root_replay": "fixture/root.json",
        "root_seed": 7,
        "root_frame_index": 0,
        "root_policy": "fixture",
        "state": state_payload(state, sim),
    }


class SupportLossActionLabelsTests(unittest.TestCase):
    def test_target_reached_requires_384_and_768_without_built_1536(self):
        state = support_loss_state()
        self.assertFalse(target_reached(state, 1536, "raw_one_384_with_one_768_no_1536"))
        reached = SimState(
            board=np.asarray(
                [
                    [1536, 768, 384, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [3, 6, 12, 24],
                ],
                dtype=np.int32,
            ),
            preview=state.preview,
            small_counts=state.small_counts,
            small_pos=state.small_pos,
            small_seen_total=state.small_seen_total,
            span_small_pos=state.span_small_pos,
            large_pending=False,
            max_tile=1536,
            move_count=81,
            game_over=False,
        )
        self.assertTrue(target_reached(reached, 1536, "raw_one_384_with_one_768_no_1536"))

    def test_run_labels_scores_first_actions_with_crn_rollouts(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            records_path = root / "records.json"
            out_dir = root / "labels"
            records_path.write_text(json.dumps({"records": [record_payload()]}))

            payload = run_labels(
                records_json=[records_path],
                policy_name="greedy",
                target="raw_one_384_with_one_768_no_1536",
                horizon=1,
                repeats=1,
                max_starts=0,
                seed=20260708,
                root_origins={"fresh"},
                default_starter_tile=1536,
                first_action_mode="all",
                out_dir=out_dir,
                progress_every=0,
            )

            self.assertEqual(payload["summary"]["labels"], 1)
            self.assertGreater(payload["summary"]["p_target"], 0.0)
            left = [row for row in payload["action_summary"] if row["first_action"] == "left"]
            self.assertEqual(left[0]["p_target"], 1.0)
            self.assertEqual(payload["labels"][0]["base_action"], "left")
            self.assertTrue((out_dir / "support_loss_action_labels.json").exists())
            self.assertTrue((out_dir / "records.json").exists())
            self.assertTrue((out_dir / "support_loss_action_labels.html").exists())


if __name__ == "__main__":
    unittest.main()
