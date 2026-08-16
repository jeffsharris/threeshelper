import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from threes_rl.ntuple import NtupleValue
from threes_rl.record_replay import state_payload
from threes_rl.sim import SimState, ThreesSim, preview_from_label, simulate_base_move
from threes_rl.train_transition_replay_calibration import (
    TransitionReplayCalibrationConfig,
    calibrate,
)


def make_state(*, success_shape: bool, move_count: int = 0) -> SimState:
    board = np.asarray(
        [
            [3072 if success_shape else 3, 1536, 768, 384],
            [192, 96, 48, 24],
            [12, 6, 3, 2],
            [1, 0, 0, 3 if success_shape else 6],
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
        max_tile=int(board.max(initial=0)),
        move_count=move_count,
        game_over=False,
    )


def write_replay(path: Path, sim: ThreesSim, *, success_shape: bool, final_score: int) -> None:
    before = make_state(success_shape=success_shape, move_count=0)
    after_board, eligible = simulate_base_move(before.board, 3)
    assert eligible
    after = make_state(success_shape=success_shape, move_count=1)
    after.board = after_board
    after.max_tile = int(after_board.max(initial=0))
    payload = {
        "seed": 1 if success_shape else 2,
        "starter_tile": 1536,
        "final_score": int(final_score),
        "frames": [
            {"index": 0, "state": state_payload(before, sim), "move": None},
            {"index": 1, "state": state_payload(after, sim), "move": {"action": "right"}},
        ],
    }
    path.write_text(json.dumps(payload))


def make_record(replay_path: Path, sim: ThreesSim, *, success: bool) -> dict:
    return {
        "id": f"{'success' if success else 'failure'}_record",
        "target_tile": 1536,
        "outcome": "success" if success else "failure",
        "source_replay": str(replay_path),
        "source_seed": 1 if success else 2,
        "source_frame_index": 0,
        "starter_tile": 1536,
        "state": state_payload(make_state(success_shape=success), sim),
        "features": {
            "raw_count_1536": 2,
            "raw_has_adjacent_1536": False,
            "phase": "endgame_3072p",
            "corner_risk": "medium_corner_risk",
        },
    }


class TrainTransitionReplayCalibrationTests(unittest.TestCase):
    def test_calibrates_from_record_replay_suffixes(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            sim = ThreesSim(np.random.default_rng(1), starter_tile=1536)
            success_replay = root / "success.json"
            failure_replay = root / "failure.json"
            write_replay(success_replay, sim, success_shape=True, final_score=1_000_000)
            write_replay(failure_replay, sim, success_shape=False, final_score=1_000)
            records_path = root / "records.json"
            records_path.write_text(
                json.dumps(
                    {
                        "records": [
                            make_record(success_replay, sim, success=True),
                            make_record(failure_replay, sim, success=False),
                        ]
                    }
                )
            )

            checkpoint = calibrate(
                TransitionReplayCalibrationConfig(
                    run_name=str(root / "suffix_cal"),
                    records_json=[str(records_path)],
                    epochs=20,
                    pattern_set="tiny",
                    alpha=0.2,
                    use_tc=True,
                    target_tile=1536,
                    candidate_min_feature={"raw_count_1536": 2.0},
                    candidate_feature_equals={"raw_has_adjacent_1536": "0"},
                    shuffle=True,
                    progress_every=1000,
                )
            )

            summary = json.loads((checkpoint.parent / "summary.json").read_text())
            self.assertEqual(summary["examples"], 2)
            self.assertEqual(summary["source_starts"], 2)
            model = NtupleValue.load(checkpoint)
            success_after, _ = simulate_base_move(make_state(success_shape=True).board, 3)
            failure_after, _ = simulate_base_move(make_state(success_shape=False).board, 3)
            self.assertGreater(model.value(success_after), model.value(failure_after))


if __name__ == "__main__":
    unittest.main()
