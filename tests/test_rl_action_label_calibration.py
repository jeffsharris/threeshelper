import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from threes_rl.record_replay import state_payload
from threes_rl.sim import DIRECTION_NAMES, ThreesSim
from threes_rl.train_action_label_calibration import (
    ActionLabelCalibrationConfig,
    calibrate,
    examples_from_endgame_label_file,
    examples_from_swing_label_file,
)
from tests.test_rl_swing_label import simple_state


class ActionLabelCalibrationTests(unittest.TestCase):
    def test_swing_label_examples_are_centered_by_group(self):
        state = simple_state()
        sim = ThreesSim(np.random.default_rng(7), starter_tile=1536)
        actions = [DIRECTION_NAMES[int(action)] for action in sim.legal_actions(state)[:2]]
        payload = {
            "labels": [
                {
                    "id": "sample-1",
                    "state": state_payload(state, sim),
                    "base_action": actions[0],
                    "features": {"corner_risk": "high_corner_risk"},
                    "label": {
                        "horizons": [32, 64],
                        "by_action": {
                            actions[0]: {"64": [10, 20]},
                            actions[1]: {"64": [30, 40]},
                        },
                    },
                }
            ]
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "swing_labels.json"
            path.write_text(json.dumps(payload))
            examples = examples_from_swing_label_file(
                path,
                horizon=64,
                target_mode="centered_afterstate",
                starter_tile=1536,
            )

        self.assertEqual(len(examples), 2)
        self.assertAlmostEqual(sum(example.target for example in examples), 0.0)
        self.assertEqual({example.corner_risk for example in examples}, {"high_corner_risk"})

    def test_endgame_label_examples_load_source_replay_frame(self):
        state = simple_state()
        sim = ThreesSim(np.random.default_rng(7), starter_tile=1536)
        actions = [DIRECTION_NAMES[int(action)] for action in sim.legal_actions(state)[:2]]
        with tempfile.TemporaryDirectory() as tmp:
            replay_path = Path(tmp) / "replay.json"
            replay_path.write_text(
                json.dumps(
                    {
                        "seed": 7,
                        "starter_tile": 1536,
                        "frames": [{"index": 0, "state": state_payload(state, sim), "move": None}],
                    }
                )
            )
            label_path = Path(tmp) / "endgame_action_labels.json"
            label_path.write_text(
                json.dumps(
                    {
                        "labels": [
                            {
                                "id": "label-1",
                                "source_replay": str(replay_path),
                                "source_frame_index": 0,
                                "base_action": actions[0],
                                "features": {"corner_risk": "medium_corner_risk"},
                                "action_results": [
                                    {"action": actions[0], "mean_delta": 10.0},
                                    {"action": actions[1], "mean_delta": 20.0},
                                ],
                            }
                        ]
                    }
                )
            )

            examples = examples_from_endgame_label_file(
                label_path,
                target_mode="centered_afterstate",
                starter_tile=1536,
            )

        self.assertEqual(len(examples), 2)
        self.assertAlmostEqual(sum(example.target for example in examples), 0.0)
        self.assertEqual({example.group_id for example in examples}, {"label-1"})
        self.assertEqual({example.corner_risk for example in examples}, {"medium_corner_risk"})

    def test_confidence_regret_weights_corrective_labels_higher(self):
        state = simple_state()
        sim = ThreesSim(np.random.default_rng(7), starter_tile=1536)
        actions = [DIRECTION_NAMES[int(action)] for action in sim.legal_actions(state)[:2]]
        with tempfile.TemporaryDirectory() as tmp:
            replay_path = Path(tmp) / "replay.json"
            replay_path.write_text(
                json.dumps(
                    {
                        "seed": 7,
                        "starter_tile": 1536,
                        "frames": [{"index": 0, "state": state_payload(state, sim), "move": None}],
                    }
                )
            )
            label_path = Path(tmp) / "endgame_action_labels.json"
            label_path.write_text(
                json.dumps(
                    {
                        "labels": [
                            {
                                "id": "corrective",
                                "source_replay": str(replay_path),
                                "source_frame_index": 0,
                                "base_action": actions[0],
                                "winner": actions[1],
                                "stable": True,
                                "bootstrap_winner_fraction": 0.9,
                                "oracle_regret": 30000.0,
                                "winner_p6144": 0.25,
                                "action_results": [
                                    {"action": actions[0], "mean_delta": 10.0},
                                    {"action": actions[1], "mean_delta": 40.0},
                                ],
                            },
                            {
                                "id": "no-regret",
                                "source_replay": str(replay_path),
                                "source_frame_index": 0,
                                "base_action": actions[0],
                                "winner": actions[0],
                                "stable": True,
                                "bootstrap_winner_fraction": 0.9,
                                "oracle_regret": 0.0,
                                "winner_p6144": 0.0,
                                "action_results": [
                                    {"action": actions[0], "mean_delta": 20.0},
                                    {"action": actions[1], "mean_delta": 10.0},
                                ],
                            },
                        ]
                    }
                )
            )

            examples = examples_from_endgame_label_file(
                label_path,
                target_mode="centered_afterstate",
                starter_tile=1536,
                label_weight_mode="confidence_regret",
            )

        weights_by_group = {example.group_id: example.weight for example in examples}
        self.assertGreater(weights_by_group["corrective"], weights_by_group["no-regret"])
        self.assertGreater(weights_by_group["corrective"], 1.0)
        self.assertLess(weights_by_group["no-regret"], 1.0)

    def test_calibrate_writes_checkpoint(self):
        state = simple_state()
        sim = ThreesSim(np.random.default_rng(7), starter_tile=1536)
        actions = [DIRECTION_NAMES[int(action)] for action in sim.legal_actions(state)[:2]]
        with tempfile.TemporaryDirectory() as tmp:
            label_path = Path(tmp) / "swing_labels.json"
            label_path.write_text(
                json.dumps(
                    {
                        "labels": [
                            {
                                "id": "sample-1",
                                "state": state_payload(state, sim),
                                "base_action": actions[0],
                                "features": {"corner_risk": "medium_corner_risk"},
                                "label": {
                                    "horizons": [64],
                                    "by_action": {
                                        actions[0]: {"64": [10, 20]},
                                        actions[1]: {"64": [30, 40]},
                                    },
                                },
                            }
                        ]
                    }
                )
            )
            run_name = f"test_action_label_calibration_{Path(tmp).name}"
            config = ActionLabelCalibrationConfig(
                run_name=run_name,
                swing_label_json=[str(label_path)],
                epochs=1,
                pattern_set="tiny",
                stage_mode="none",
                alpha=0.01,
                use_tc=False,
                progress_every=0,
            )
            checkpoint = calibrate(config)

        self.assertTrue((checkpoint / "meta.json").exists())


if __name__ == "__main__":
    unittest.main()
