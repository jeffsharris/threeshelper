import unittest

import numpy as np

from threes_rl.env import ILLEGAL_PENALTY, ThreesEnv
from threes_rl.sim import LEFT, UP, ThreesSim, preview_from_label, score_board


class EnvApiTests(unittest.TestCase):
    def test_gymnasium_check_env_if_available(self):
        try:
            from gymnasium.utils.env_checker import check_env
        except Exception as exc:
            self.skipTest(f"gymnasium check_env unavailable: {exc}")
        check_env(ThreesEnv(seed=1), skip_render_check=True)

    def test_reset_observation_bounds(self):
        env = ThreesEnv(seed=5)
        obs, info = env.reset(seed=5)
        self.assertEqual(obs.shape, env.observation_space.shape)
        self.assertTrue(np.all(obs >= 0.0))
        self.assertTrue(np.all(obs <= 1.0))
        self.assertEqual(info["legal_mask"].shape, (4,))
        self.assertEqual(info["legal_mask"].dtype, np.bool_)

    def test_deterministic_trajectories_with_same_seed(self):
        env1 = ThreesEnv(seed=9)
        env2 = ThreesEnv(seed=9)
        obs1, info1 = env1.reset(seed=9)
        obs2, info2 = env2.reset(seed=9)
        np.testing.assert_array_equal(obs1, obs2)
        rng = np.random.default_rng(99)
        for _ in range(1000):
            legal = np.flatnonzero(info1["legal_mask"])
            if len(legal) == 0:
                obs1, info1 = env1.reset(seed=9)
                obs2, info2 = env2.reset(seed=9)
                continue
            action = int(legal[int(rng.integers(len(legal)))])
            out1 = env1.step(action)
            out2 = env2.step(action)
            np.testing.assert_array_equal(out1[0], out2[0])
            self.assertEqual(out1[1], out2[1])
            self.assertEqual(out1[2], out2[2])
            self.assertEqual(out1[3], out2[3])
            np.testing.assert_array_equal(out1[4]["legal_mask"], out2[4]["legal_mask"])
            if out1[2]:
                obs1, info1 = env1.reset(seed=10)
                obs2, info2 = env2.reset(seed=10)
                np.testing.assert_array_equal(obs1, obs2)
            else:
                info1 = out1[4]
                info2 = out2[4]

    def test_illegal_step_is_noop_with_penalty(self):
        env = ThreesEnv(seed=1)
        env.reset(seed=1)
        sim = env.sim
        snapshot = ({"red": 4, "blue": 4, "gray": 4}, 0, 0, 0, False, 1)
        state = sim.state_from_snapshot(
            [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            preview_from_label("red"),
            snapshot,
        )
        env.state = state
        before_obs = env._obs().copy()
        obs, reward, terminated, truncated, info = env.step(UP)
        np.testing.assert_array_equal(obs, before_obs)
        self.assertEqual(reward, ILLEGAL_PENALTY)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["illegal_action"])

    def test_final_score_reward_for_terminal_merge(self):
        env = ThreesEnv(seed=1, reward_mode="final_score")
        env.reset(seed=1)
        state = env.sim.state_from_snapshot(
            [[6144, 6144, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            preview_from_label("red"),
            ({"red": 4, "blue": 4, "gray": 4}, 0, 0, 0, False, 6144),
        )
        state.game_over = False
        env.state = state
        _obs, reward, terminated, _truncated, info = env.step(LEFT)
        self.assertTrue(terminated)
        self.assertEqual(reward, score_board(env.state.board))
        self.assertEqual(info["final_score"], 1594323)

    def test_reward_modes(self):
        for mode in ("final_score", "score_delta", "merge_delta", "log_score_delta", "survival"):
            env = ThreesEnv(seed=3, reward_mode=mode)
            _obs, info = env.reset(seed=3)
            action = int(np.flatnonzero(info["legal_mask"])[0])
            _obs, reward, _terminated, _truncated, _info = env.step(action)
            self.assertIsInstance(reward, float)


if __name__ == "__main__":
    unittest.main()
