import unittest

import numpy as np

from threes_rl.eval import make_policy
from threes_rl.selective_rollout import RolloutGateConfig, SelectiveRolloutPolicy
from threes_rl.sim import SimState, ThreesSim, preview_from_label


class NearTiePolicy:
    def action_values(self, state, sim):
        legal = sim.legal_actions(state)
        return [(int(action), 100.0 - 0.01 * idx) for idx, action in enumerate(legal)]

    def _select_action(self, action_values, rng):
        return int(max(action_values, key=lambda item: item[1])[0])

    def __call__(self, state, sim, rng):
        return self._select_action(self.action_values(state, sim), rng)


def medium_risk_endgame_state() -> SimState:
    return SimState(
        board=np.asarray(
            [
                [3072, 1536, 768, 384],
                [192, 96, 48, 24],
                [12, 6, 3, 2],
                [1, 0, 0, 0],
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
        move_count=320,
        game_over=False,
    )


class SelectiveRolloutTests(unittest.TestCase):
    def test_make_policy_parses_selective_rollout_wrapper(self):
        policy = make_policy("selective_rollout|greedy|repeats=1|horizon=1|margin=1|max_rate=1")

        stats = policy.summary_stats()

        self.assertEqual(stats["base_policy"], "greedy")
        self.assertEqual(stats["config"]["repeats"], 1)
        self.assertEqual(stats["config"]["horizon"], 1)

    def test_selective_rollout_fires_for_matching_low_margin_stratum(self):
        policy = SelectiveRolloutPolicy(
            NearTiePolicy(),
            "near_tie",
            RolloutGateConfig(margin_threshold=1.0, repeats=1, horizon=1, max_gate_rate=1.0),
        )
        sim = ThreesSim(np.random.default_rng(7), starter_tile=1536)
        state = medium_risk_endgame_state()

        action = policy(state, sim, np.random.default_rng(11))
        stats = policy.summary_stats()

        self.assertIn(action, sim.legal_actions(state))
        self.assertEqual(stats["eligible"], 1)
        self.assertEqual(stats["gated"], 1)
        self.assertEqual(stats["rollouts"], 2)


if __name__ == "__main__":
    unittest.main()
