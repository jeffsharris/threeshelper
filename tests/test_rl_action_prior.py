import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from threes_rl.action_prior import (
    FEATURE_NAMES,
    ActionPriorModel,
    ActionPriorPolicy,
    ActionPriorPolicyConfig,
    ActionPriorTrainConfig,
    _built_max_support_features,
    action_features,
    calibrate_action_prior,
    load_pair_examples,
)
from threes_rl.eval import make_policy
from threes_rl.record_replay import state_payload
from threes_rl.sim import DIRECTION_NAMES, ThreesSim
from tests.test_rl_swing_label import simple_state


class FixedTopTwoPolicy:
    def __init__(self, base_action: int, alt_action: int) -> None:
        self.base_action = int(base_action)
        self.alt_action = int(alt_action)

    def action_values(self, state, sim):
        return [(self.base_action, 10.0), (self.alt_action, 9.99)]

    def _select_action(self, action_values, rng):
        return int(max(action_values, key=lambda item: item[1])[0])

    def __call__(self, state, sim, rng):
        return self.base_action


class ActionPriorTests(unittest.TestCase):
    def _write_endgame_label(self, tmp: str, *, base_name: str, winner_name: str) -> Path:
        state = simple_state()
        sim = ThreesSim(np.random.default_rng(7), starter_tile=1536)
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
                            "base_action": base_name,
                            "winner": winner_name,
                            "stable": True,
                            "bootstrap_winner_fraction": 0.95,
                            "oracle_regret": 10000.0,
                            "winner_p6144": 0.0,
                            "features": {
                                "phase": "early_lt384",
                                "corner_risk": "medium_corner_risk",
                            },
                            "action_results": [
                                {"action": base_name, "mean_delta": 10.0},
                                {"action": winner_name, "mean_delta": 30.0},
                            ],
                        }
                    ]
                }
            )
        )
        return label_path

    def test_load_pair_examples_adds_forward_and_reverse_pairs(self):
        state = simple_state()
        sim = ThreesSim(np.random.default_rng(7), starter_tile=1536)
        actions = [DIRECTION_NAMES[int(action)] for action in sim.legal_actions(state)[:2]]
        with tempfile.TemporaryDirectory() as tmp:
            label_path = self._write_endgame_label(tmp, base_name=actions[0], winner_name=actions[1])
            config = ActionPriorTrainConfig(
                run_name="test_action_prior",
                endgame_label_json=[str(label_path)],
                phase_filter=["early_lt384"],
                corner_risk_filter=["medium_corner_risk"],
            )
            examples = load_pair_examples(config)

        self.assertEqual(len(examples), 2)
        self.assertEqual({example.target for example in examples}, {0.0, 1.0})
        self.assertEqual({example.winner for example in examples}, {actions[1]})

    def test_make_policy_parses_action_prior_wrapper(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "prior"
            checkpoint.mkdir()
            (checkpoint / "model.json").write_text(
                json.dumps(
                    {
                        "feature_names": ["action_up"],
                        "weights": [0.0],
                        "bias": 0.0,
                        "feature_mean": [0.0],
                        "feature_std": [1.0],
                        "starter_tile": 1536,
                    }
                )
            )

            policy = make_policy(f"action_prior|greedy|checkpoint={checkpoint}|prob=0.6|max_rate=1|stratum=all")
            stats = policy.summary_stats()

        self.assertEqual(stats["base_policy"], "greedy")
        self.assertEqual(stats["config"]["checkpoint"], str(checkpoint))
        self.assertEqual(stats["config"]["min_probability"], 0.6)

    def test_action_features_include_geometry_score(self):
        state = simple_state()
        features = action_features(state, DIRECTION_NAMES.index("left"), starter_tile=1536)

        self.assertIn("high_tile_geometry_score", FEATURE_NAMES)
        self.assertIn("built_max_stranded", FEATURE_NAMES)
        self.assertEqual(len(features), len(FEATURE_NAMES))

    def test_built_max_support_features_detect_stranded_high_tile(self):
        stranded = np.asarray(
            [
                [1536, 0, 0, 0],
                [0, 0, 3072, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.int32,
        )
        supported = np.asarray(
            [
                [1536, 0, 0, 0],
                [0, 1536, 3072, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.int32,
        )

        stranded_features = _built_max_support_features(stranded, starter_tile=1536)
        supported_features = _built_max_support_features(supported, starter_tile=1536)

        self.assertEqual(stranded_features["built_max_top_left"], 0.0)
        self.assertEqual(stranded_features["built_max_top_or_left_edge"], 0.0)
        self.assertEqual(stranded_features["built_max_has_half_neighbor"], 0.0)
        self.assertEqual(stranded_features["built_max_stranded"], 1.0)
        self.assertEqual(supported_features["built_max_has_half_neighbor"], 1.0)
        self.assertEqual(supported_features["built_max_stranded"], 0.0)

    def test_action_prior_model_aligns_legacy_feature_subset(self):
        model = ActionPriorModel(
            weights=np.asarray([1.0], dtype=np.float64),
            bias=0.0,
            feature_mean=np.asarray([0.0], dtype=np.float64),
            feature_std=np.asarray([1.0], dtype=np.float64),
            feature_names=["action_left"],
            starter_tile=1536,
        )
        state = simple_state()

        probability = model.probability(state, "left", "right")

        self.assertGreaterEqual(probability, 0.0)
        self.assertLessEqual(probability, 1.0)

    def test_action_prior_can_override_base_top_two_choice(self):
        state = simple_state()
        sim = ThreesSim(np.random.default_rng(7), starter_tile=1536)
        legal = sim.legal_actions(state)
        base_action = int(legal[0])
        winner_action = int(legal[1])
        with tempfile.TemporaryDirectory() as tmp:
            label_path = self._write_endgame_label(
                tmp,
                base_name=DIRECTION_NAMES[base_action],
                winner_name=DIRECTION_NAMES[winner_action],
            )
            checkpoint = calibrate_action_prior(
                ActionPriorTrainConfig(
                    run_name=f"test_action_prior_{Path(tmp).name}",
                    endgame_label_json=[str(label_path)],
                    phase_filter=["early_lt384"],
                    corner_risk_filter=["medium_corner_risk"],
                    epochs=80,
                    lr=0.1,
                    l2=0.0,
                    progress_every=0,
                )
            )
            policy = ActionPriorPolicy(
                FixedTopTwoPolicy(base_action, winner_action),
                "fixed",
                ActionPriorPolicyConfig(
                    checkpoint=str(checkpoint),
                    min_probability=0.55,
                    max_base_margin=1.0,
                    max_override_rate=1.0,
                    stratum=None,
                ),
            )

            action = policy(state, sim, np.random.default_rng(11))
            stats = policy.summary_stats()

        self.assertEqual(action, winner_action)
        self.assertEqual(stats["eligible"], 1)
        self.assertEqual(stats["overrides"], 1)


if __name__ == "__main__":
    unittest.main()
