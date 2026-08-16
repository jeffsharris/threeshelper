import unittest

import numpy as np

import mirroring_control as mc
import state_hunt as sh
from threes_rl.sim import (
    DIRECTION_NAMES,
    LEFT,
    TERMINAL_TILE,
    ThreesSim,
    board_to_tokens,
    score_board,
    score_tile,
    simulate_base_move,
)


TILE_VALUES = [0, 1, 2, 3, 6, 12, 24, 48, 96, 192, 384, 768, 1536, 3072, 6144]


class SimRuleTests(unittest.TestCase):
    def test_direction_order_matches_tracker(self):
        self.assertEqual(DIRECTION_NAMES, mc.DIRECTIONS)

    def test_score_table_includes_terminal_tile(self):
        expected = {
            0: 0,
            1: 0,
            2: 0,
            3: 3,
            6: 9,
            12: 27,
            24: 81,
            48: 243,
            96: 729,
            192: 2187,
            384: 6561,
            768: 19683,
            1536: 59049,
            3072: 177147,
            6144: 531441,
            12288: 1594323,
        }
        for value, score in expected.items():
            self.assertEqual(score_tile(value), score)

    def test_starter_tile_starts_top_left(self):
        for seed in range(50):
            sim = ThreesSim(np.random.default_rng(seed), starter_tile=1536)
            state = sim.reset()
            self.assertEqual(int(state.board[0, 0]), 1536)
            self.assertEqual(int(np.count_nonzero(state.board == 1536)), 1)
            self.assertEqual(int(np.count_nonzero(state.board)), 9)

    def test_base_move_matches_tracker_oracle(self):
        rng = np.random.default_rng(20260705)
        adversarial = [
            np.zeros((4, 4), dtype=np.int32),
            np.asarray([[1, 2, 0, 0], [3, 3, 3, 3], [6144, 6144, 0, 0], [1, 1, 2, 2]], dtype=np.int32),
            np.asarray([[0, 0, 0, 3], [0, 0, 6, 6], [0, 12, 12, 12], [24, 24, 24, 24]], dtype=np.int32),
        ]
        random_boards = [rng.choice(TILE_VALUES, size=(4, 4)).astype(np.int32) for _ in range(20000)]
        for board in adversarial + random_boards:
            for action, direction in enumerate(DIRECTION_NAMES):
                ours_board, ours_eligible = simulate_base_move(board, action)
                oracle_board, oracle_eligible = sh.simulate_base_move(board.tolist(), direction)
                self.assertEqual(ours_board.tolist(), oracle_board)
                self.assertEqual(ours_eligible, oracle_eligible)

    def test_terminal_6144_merge_ends_without_spawn(self):
        sim = ThreesSim(np.random.default_rng(1))
        state = sim.reset()
        state.board = np.asarray(
            [
                [6144, 6144, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.int32,
        )
        state.max_tile = 6144
        state.game_over = False
        next_state, info = sim.step(state, LEFT)
        self.assertTrue(info.moved)
        self.assertTrue(info.terminal_merge)
        self.assertIsNone(info.inserted_value)
        self.assertTrue(next_state.game_over)
        self.assertEqual(int(next_state.board[0, 0]), TERMINAL_TILE)
        self.assertEqual(score_board(next_state.board), 1594323)

    def test_simulated_steps_validate_against_tracker(self):
        rng = np.random.default_rng(44)
        checked = 0
        episode = 0
        while checked < 50000:
            sim = ThreesSim(np.random.default_rng(int(rng.integers(1_000_000))), starter_tile=1536 if episode % 2 == 0 else None)
            state = sim.reset()
            while not state.game_over and checked < 50000:
                actions = sim.legal_actions(state)
                action = int(actions[int(rng.integers(len(actions)))])
                before = state
                before_tokens = board_to_tokens(before.board)
                state, info = sim.step(state, action)
                if info.terminal_merge:
                    continue
                after_tokens = board_to_tokens(state.board)
                validation = sh.validate_transition(before_tokens, DIRECTION_NAMES[action], before.preview.label, after_tokens)
                self.assertTrue(validation.valid, validation.reason)
                self.assertEqual(validation.inserted_value, info.inserted_value)
                self.assertEqual(tuple(validation.inserted_pos), info.inserted_pos)
                checked += 1
                if checked >= 50000:
                    break
            episode += 1
        self.assertEqual(checked, 50000)


if __name__ == "__main__":
    unittest.main()
