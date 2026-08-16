import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from threes_rl.ntuple import NtupleValue
from threes_rl.record_replay import state_payload
from threes_rl.sim import SimState, ThreesSim, preview_from_label
from threes_rl.train_transition_reachability_value import ReachabilityValueConfig, train


def make_state(*, success_shape: bool, move_count: int) -> SimState:
    board = np.asarray(
        [
            [3072 if success_shape else 3, 1536, 1536, 768],
            [384, 192, 96, 48],
            [24, 12, 6, 3],
            [2, 1, 0 if success_shape else 3, 0 if success_shape else 6],
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


def make_record(sim: ThreesSim, idx: int, *, success: bool) -> dict:
    state = make_state(success_shape=success, move_count=100 + idx)
    return {
        "id": f"r{idx}",
        "target_tile": 1536,
        "outcome": "success" if success else "failure",
        "phase": "endgame_3072p",
        "corner_risk": "medium_corner_risk" if success else "high_corner_risk",
        "source_replay": f"replay-{idx}.json",
        "source_seed": idx,
        "source_frame_index": idx * 3,
        "starter_tile": 1536,
        "features": {
            "empty_count": 2 if success else 0,
            "raw_count_1536": 2,
            "raw_has_adjacent_1536": False,
            "preview": "gray",
            "phase": "endgame_3072p",
            "corner_risk": "medium_corner_risk" if success else "high_corner_risk",
        },
        "state": state_payload(state, sim),
    }


class TrainTransitionReachabilityValueTests(unittest.TestCase):
    def test_trains_checkpoint_and_filters_examples(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            tmp_path = Path(tmp)
            sim = ThreesSim(np.random.default_rng(1), starter_tile=1536)
            records = []
            for idx in range(8):
                records.append(make_record(sim, idx, success=idx % 2 == 0))
            filtered_out = make_record(sim, 99, success=True)
            filtered_out["features"]["raw_count_1536"] = 1
            records.append(filtered_out)
            records_path = tmp_path / "records.json"
            records_path.write_text(json.dumps({"records": records}))

            checkpoint = train(
                ReachabilityValueConfig(
                    run_name=str(tmp_path / "reachability_value"),
                    records_json=[str(records_path)],
                    epochs=12,
                    pattern_set="tiny",
                    alpha=0.2,
                    use_tc=True,
                    target_tile=1536,
                    success_target=1000.0,
                    failure_target=-1000.0,
                    shuffle=False,
                    candidate_min_feature={"raw_count_1536": 2.0},
                    candidate_feature_equals={"raw_has_adjacent_1536": "0"},
                    progress_every=1000,
                )
            )

            summary = json.loads((checkpoint.parent / "summary.json").read_text())
            self.assertEqual(summary["examples"], 8)
            self.assertEqual(summary["outcome_counts"], {"success": 4, "failure": 4})
            model = NtupleValue.load(checkpoint)
            success_value = model.value(make_state(success_shape=True, move_count=1).board)
            failure_value = model.value(make_state(success_shape=False, move_count=1).board)
            self.assertGreater(success_value, failure_value)


if __name__ == "__main__":
    unittest.main()
