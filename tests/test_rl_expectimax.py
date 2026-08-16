import tempfile
import unittest
from pathlib import Path

import numpy as np

from threes_rl.eval import AnchorGuardPolicy, make_policy
from threes_rl.expectimax import NtupleExpectimaxPolicy, high_tile_geometry_score
from threes_rl.ntuple import NtupleValue, patterns_for_set
from threes_rl.sim import LEFT, RIGHT, UP, SimState, ThreesSim, preview_from_label, score_tile


class LearnedExpectimaxTests(unittest.TestCase):
    def _save_tiny_checkpoint(self, root: str, name: str = "tiny_ntuple", init: float = 0.0) -> Path:
        path = Path(root) / name
        NtupleValue(patterns_for_set("tiny"), init=init).save(path)
        return path

    def _state(self, board: np.ndarray, max_tile: int) -> SimState:
        return SimState(
            board=board,
            preview=preview_from_label("gray"),
            small_counts={"red": 4, "blue": 4, "gray": 4},
            small_pos=0,
            small_seen_total=0,
            span_small_pos=0,
            large_pending=False,
            max_tile=max_tile,
            move_count=0,
            game_over=False,
        )

    def test_ntuple_expectimax_terminal_action_value_includes_score_delta(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = self._save_tiny_checkpoint(tmp)
            policy = NtupleExpectimaxPolicy(checkpoint, depth=2)
            sim = ThreesSim(np.random.default_rng(1), starter_tile=None)
            state = self._state(
                np.asarray(
                    [
                        [6144, 6144, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                    ],
                    dtype=np.int32,
                ),
                6144,
            )

            value = policy._action_value(state, sim, LEFT, depth=2)

            self.assertEqual(value, score_tile(12288) - 2 * score_tile(6144))

    def test_ntuple_expectimax_adaptive_depth_triggers_on_bonus_preview(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = self._save_tiny_checkpoint(tmp)
            policy = NtupleExpectimaxPolicy(checkpoint, depth=2, adaptive=True)
            board = np.asarray(
                [
                    [1536, 768, 384, 192],
                    [96, 48, 24, 12],
                    [6, 3, 2, 1],
                    [0, 0, 0, 6],
                ],
                dtype=np.int32,
            )
            state = self._state(board, 1536)
            state.preview = preview_from_label("large_candidates", (6, 12, 24))

            self.assertEqual(policy._root_depth(state), 3)

    def test_ntuple_expectimax_adaptive_depth_waits_for_very_tight_boards(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = self._save_tiny_checkpoint(tmp)
            policy = NtupleExpectimaxPolicy(checkpoint, depth=2, adaptive=True)
            board = np.asarray(
                [
                    [1536, 768, 384, 192],
                    [96, 48, 24, 12],
                    [6, 3, 2, 1],
                    [0, 0, 0, 6],
                ],
                dtype=np.int32,
            )
            state = self._state(board, 1536)

            self.assertEqual(policy._root_depth(state), 2)

    def test_post_spawn_leaf_cache_is_preview_independent(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = self._save_tiny_checkpoint(tmp)
            policy = NtupleExpectimaxPolicy(checkpoint, depth=2)
            sim = ThreesSim(np.random.default_rng(1), starter_tile=1536)
            board = np.asarray(
                [
                    [1536, 768, 384, 192],
                    [96, 48, 24, 12],
                    [6, 3, 2, 1],
                    [0, 0, 0, 6],
                ],
                dtype=np.int32,
            )
            gray_state = self._state(board.copy(), 1536)
            blue_state = self._state(board.copy(), 1536)
            blue_state.preview = preview_from_label("blue")
            blue_state.small_pos = 5
            blue_state.small_seen_total = 17

            first = policy._post_spawn_state_value(gray_state, sim)
            second = policy._post_spawn_state_value(blue_state, sim)

            self.assertEqual(first, second)
            self.assertEqual(len(policy._post_spawn_cache), 1)

    def test_cached_transition_outcomes_match_simulator(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = self._save_tiny_checkpoint(tmp)
            policy = NtupleExpectimaxPolicy(checkpoint, depth=2)
            sim = ThreesSim(np.random.default_rng(1), starter_tile=1536)
            board = np.asarray(
                [
                    [1536, 768, 384, 192],
                    [96, 48, 24, 12],
                    [6, 3, 2, 1],
                    [0, 0, 0, 6],
                ],
                dtype=np.int32,
            )
            state = self._state(board, 1536)

            expected = sim.transition_outcomes(state, LEFT, include_info=True, include_next_preview=True)
            actual = policy._transition_outcomes(state, sim, LEFT, include_next_preview=True)

            self.assertEqual(len(actual), len(expected))
            self.assertAlmostEqual(sum(probability for probability, _state, _info in actual), 1.0)
            for (actual_prob, actual_state, actual_info), (expected_prob, expected_state, expected_info) in zip(actual, expected):
                self.assertAlmostEqual(actual_prob, expected_prob)
                np.testing.assert_array_equal(actual_state.board, expected_state.board)
                self.assertEqual(actual_state.preview, expected_state.preview)
                self.assertEqual(actual_state.small_counts, expected_state.small_counts)
                self.assertEqual(actual_state.small_pos, expected_state.small_pos)
                self.assertEqual(actual_state.small_seen_total, expected_state.small_seen_total)
                self.assertEqual(actual_state.span_small_pos, expected_state.span_small_pos)
                self.assertEqual(actual_state.large_pending, expected_state.large_pending)
                self.assertEqual(actual_state.max_tile, expected_state.max_tile)
                self.assertEqual(actual_state.move_count, expected_state.move_count)
                self.assertEqual(actual_info, expected_info)
            self.assertEqual(len(policy._base_move_cache), 1)

    def test_make_policy_parses_ntuple_expectimax_specs(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = self._save_tiny_checkpoint(tmp)

            policy = make_policy(f"ntuple_expectimax2a:{checkpoint}")

            self.assertIsInstance(policy, NtupleExpectimaxPolicy)
            self.assertEqual(policy.depth, 2)
            self.assertTrue(policy.adaptive)

            budgeted = make_policy(f"ntuple_expectimax2b:{checkpoint}")

            self.assertIsInstance(budgeted, NtupleExpectimaxPolicy)
            self.assertEqual(budgeted.depth, 2)
            self.assertTrue(budgeted.adaptive)
            self.assertEqual(budgeted.chance_limit, 12)

    def test_make_policy_wraps_ntuple_expectimax_with_geometry_bonus(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = self._save_tiny_checkpoint(tmp)

            policy = make_policy(f"geometry_bonus|75|3072|ntuple_expectimax2:{checkpoint}")

            self.assertIsInstance(policy, NtupleExpectimaxPolicy)
            self.assertAlmostEqual(policy.geometry_weight, 75.0)
            self.assertEqual(policy.geometry_min_tile, 3072)
            self.assertIn("geometry_bonus|75|3072|", policy.name)

    def test_make_policy_wraps_ntuple_expectimax_with_value_bonus(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = self._save_tiny_checkpoint(tmp, "base", init=1.0)
            bonus = self._save_tiny_checkpoint(tmp, "bonus", init=5.0)

            policy = make_policy(f"value_bonus|{bonus}|0.50|endgame|ntuple_expectimax2:{base}")

            self.assertIsInstance(policy, NtupleExpectimaxPolicy)
            self.assertEqual(policy.bonus_specs, ((bonus, 0.50, 3),))
            self.assertIn("value_bonus|", policy.name)

    def test_make_policy_parses_blended_ntuple_expectimax_specs(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = self._save_tiny_checkpoint(tmp, "base", init=1.0)
            sidecar = self._save_tiny_checkpoint(tmp, "sidecar", init=3.0)

            policy = make_policy(f"ntuple_blend_expectimax2:{base}:{sidecar}:0.25")

            self.assertIsInstance(policy, NtupleExpectimaxPolicy)
            self.assertEqual(policy.depth, 2)
            self.assertEqual(policy.blend_checkpoint, sidecar)
            self.assertAlmostEqual(policy.blend_weight, 0.25)

    def test_make_policy_parses_tiebreak_blended_ntuple_expectimax_specs(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = self._save_tiny_checkpoint(tmp, "base", init=1.0)
            sidecar = self._save_tiny_checkpoint(tmp, "sidecar", init=3.0)

            policy = make_policy(f"ntuple_blend_tiebreak_expectimax2:{base}:{sidecar}:0.25:5")

            self.assertIsInstance(policy, NtupleExpectimaxPolicy)
            self.assertEqual(policy.depth, 2)
            self.assertEqual(policy.blend_checkpoint, sidecar)
            self.assertAlmostEqual(policy.blend_weight, 0.25)
            self.assertAlmostEqual(policy.tie_margin, 5.0)
            self.assertEqual(policy.tie_breaker, "up_left")

    def test_make_policy_parses_multiblended_ntuple_expectimax_specs(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = self._save_tiny_checkpoint(tmp, "base", init=1.0)
            sidecar_a = self._save_tiny_checkpoint(tmp, "sidecar_a", init=3.0)
            sidecar_b = self._save_tiny_checkpoint(tmp, "sidecar_b", init=5.0)

            policy = make_policy(f"ntuple_multiblend_expectimax2:{base}:{sidecar_a}:0.20:{sidecar_b}:0.10")

            self.assertIsInstance(policy, NtupleExpectimaxPolicy)
            self.assertEqual(policy.depth, 2)
            self.assertEqual(policy.blend_checkpoint, sidecar_a)
            self.assertAlmostEqual(policy.blend_weight, 0.20)
            self.assertEqual([path for path, _weight in policy.blend_specs], [sidecar_a, sidecar_b])
            self.assertEqual([weight for _path, weight in policy.blend_specs], [0.20, 0.10])
            self.assertAlmostEqual(policy.base_weight, 0.70)

    def test_make_policy_parses_phaseblended_ntuple_expectimax_specs(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = self._save_tiny_checkpoint(tmp, "base", init=1.0)
            always = self._save_tiny_checkpoint(tmp, "always", init=3.0)
            late = self._save_tiny_checkpoint(tmp, "late", init=5.0)

            policy = make_policy(f"ntuple_phaseblend_expectimax2:{base}:{always}:0.25:all:{late}:0.10:late")

            self.assertIsInstance(policy, NtupleExpectimaxPolicy)
            self.assertEqual(policy.depth, 2)
            self.assertEqual(policy.blend_specs, ((always, 0.25),))
            self.assertEqual(policy.phase_blend_specs, ((late, 0.10, 2),))
            self.assertAlmostEqual(policy.base_weight, 0.75)

    def test_make_policy_parses_additive_phaseblend_ntuple_expectimax_specs(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = self._save_tiny_checkpoint(tmp, "base", init=1.0)
            correction = self._save_tiny_checkpoint(tmp, "correction", init=3.0)

            policy = make_policy(f"ntuple_additive_phaseblend_expectimax2:{base}:{correction}:0.25:all")

            self.assertIsInstance(policy, NtupleExpectimaxPolicy)
            self.assertEqual(policy.ensemble_mode, "additive")
            self.assertEqual(policy.blend_specs, ((correction, 0.25),))
            self.assertAlmostEqual(policy.base_weight, 1.0)

    def test_make_policy_parses_maxblend_ntuple_expectimax_specs(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = self._save_tiny_checkpoint(tmp, "base", init=1.0)
            sidecar_a = self._save_tiny_checkpoint(tmp, "sidecar_a", init=3.0)
            sidecar_b = self._save_tiny_checkpoint(tmp, "sidecar_b", init=5.0)

            policy = make_policy(f"ntuple_maxblend_expectimax2:{base}:{sidecar_a}:{sidecar_b}")

            self.assertIsInstance(policy, NtupleExpectimaxPolicy)
            self.assertEqual(policy.depth, 2)
            self.assertEqual(policy.ensemble_mode, "max")
            self.assertEqual([path for path, _weight in policy.blend_specs], [sidecar_a, sidecar_b])

    def test_blended_ntuple_expectimax_afterstate_value_interpolates(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = self._save_tiny_checkpoint(tmp, "base", init=1.0)
            sidecar = self._save_tiny_checkpoint(tmp, "sidecar", init=3.0)
            base_policy = NtupleExpectimaxPolicy(base, depth=2)
            sidecar_policy = NtupleExpectimaxPolicy(sidecar, depth=2)
            blended = NtupleExpectimaxPolicy(base, depth=2, blend_checkpoint=sidecar, blend_weight=0.25)
            board = np.zeros((4, 4), dtype=np.int32)

            expected = 0.75 * base_policy._afterstate_value(board) + 0.25 * sidecar_policy._afterstate_value(board)

            self.assertAlmostEqual(blended._afterstate_value(board), expected)

    def test_multiblended_ntuple_expectimax_afterstate_value_interpolates(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = self._save_tiny_checkpoint(tmp, "base", init=1.0)
            sidecar_a = self._save_tiny_checkpoint(tmp, "sidecar_a", init=3.0)
            sidecar_b = self._save_tiny_checkpoint(tmp, "sidecar_b", init=5.0)
            base_policy = NtupleExpectimaxPolicy(base, depth=2)
            sidecar_a_policy = NtupleExpectimaxPolicy(sidecar_a, depth=2)
            sidecar_b_policy = NtupleExpectimaxPolicy(sidecar_b, depth=2)
            blended = NtupleExpectimaxPolicy(
                base,
                depth=2,
                blend_specs=[(sidecar_a, 0.20), (sidecar_b, 0.10)],
            )
            board = np.zeros((4, 4), dtype=np.int32)

            expected = (
                0.70 * base_policy._afterstate_value(board)
                + 0.20 * sidecar_a_policy._afterstate_value(board)
                + 0.10 * sidecar_b_policy._afterstate_value(board)
            )

            self.assertAlmostEqual(blended._afterstate_value(board), expected)

    def test_additive_ntuple_expectimax_afterstate_value_preserves_base(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = self._save_tiny_checkpoint(tmp, "base", init=1.0)
            correction = self._save_tiny_checkpoint(tmp, "correction", init=3.0)
            base_policy = NtupleExpectimaxPolicy(base, depth=2)
            correction_policy = NtupleExpectimaxPolicy(correction, depth=2)
            additive = NtupleExpectimaxPolicy(
                base,
                depth=2,
                blend_specs=[(correction, 0.25)],
                ensemble_mode="additive",
            )
            board = np.zeros((4, 4), dtype=np.int32)

            expected = base_policy._afterstate_value(board) + 0.25 * correction_policy._afterstate_value(board)

            self.assertAlmostEqual(additive._afterstate_value(board), expected)

    def test_phaseblended_ntuple_expectimax_only_activates_for_late_boards(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = self._save_tiny_checkpoint(tmp, "base", init=1.0)
            always = self._save_tiny_checkpoint(tmp, "always", init=3.0)
            late = self._save_tiny_checkpoint(tmp, "late", init=5.0)
            base_policy = NtupleExpectimaxPolicy(base, depth=2)
            always_policy = NtupleExpectimaxPolicy(always, depth=2)
            late_policy = NtupleExpectimaxPolicy(late, depth=2)
            phase_blended = NtupleExpectimaxPolicy(
                base,
                depth=2,
                blend_specs=[(always, 0.25)],
                phase_blend_specs=[(late, 0.10, 2)],
            )
            early_board = np.zeros((4, 4), dtype=np.int32)
            late_board = np.asarray(
                [
                    [1536, 1536, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
                dtype=np.int32,
            )

            early_expected = 0.75 * base_policy._afterstate_value(early_board) + 0.25 * always_policy._afterstate_value(early_board)
            late_expected = (
                0.65 * base_policy._afterstate_value(late_board)
                + 0.25 * always_policy._afterstate_value(late_board)
                + 0.10 * late_policy._afterstate_value(late_board)
            )

            self.assertAlmostEqual(phase_blended._afterstate_value(early_board), early_expected)
            self.assertAlmostEqual(phase_blended._afterstate_value(late_board), late_expected)

    def test_phaseblended_ntuple_expectimax_can_gate_by_corner_risk(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = self._save_tiny_checkpoint(tmp, "base", init=1.0)
            medium_sidecar = self._save_tiny_checkpoint(tmp, "medium", init=5.0)
            base_policy = NtupleExpectimaxPolicy(base, depth=2)
            sidecar_policy = NtupleExpectimaxPolicy(medium_sidecar, depth=2)
            policy = make_policy(
                f"ntuple_phaseblend_expectimax2:{base}:{medium_sidecar}:0.10:medium_corner_risk"
            )
            low_board = np.zeros((4, 4), dtype=np.int32)
            medium_board = np.asarray(
                [
                    [3072, 384, 2, 6],
                    [1536, 192, 48, 6],
                    [3, 3, 6, 0],
                    [0, 0, 1, 3],
                ],
                dtype=np.int32,
            )

            self.assertEqual(policy.phase_blend_specs, ((medium_sidecar, 0.10, "risk=medium_corner_risk"),))
            self.assertAlmostEqual(policy._afterstate_value(low_board), base_policy._afterstate_value(low_board))
            expected = 0.90 * base_policy._afterstate_value(medium_board) + 0.10 * sidecar_policy._afterstate_value(medium_board)
            self.assertAlmostEqual(policy._afterstate_value(medium_board), expected)

    def test_value_bonus_adds_after_existing_phaseblend_when_gate_active(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = self._save_tiny_checkpoint(tmp, "base", init=1.0)
            sidecar = self._save_tiny_checkpoint(tmp, "sidecar", init=3.0)
            bonus = self._save_tiny_checkpoint(tmp, "bonus", init=5.0)
            base_policy = NtupleExpectimaxPolicy(base, depth=2)
            sidecar_policy = NtupleExpectimaxPolicy(sidecar, depth=2)
            bonus_policy = NtupleExpectimaxPolicy(bonus, depth=2)
            policy = NtupleExpectimaxPolicy(
                base,
                depth=2,
                blend_specs=[(sidecar, 0.25)],
                bonus_specs=[(bonus, 0.5, 3)],
            )
            early_board = np.zeros((4, 4), dtype=np.int32)
            endgame_board = np.asarray(
                [
                    [3072, 1536, 1536, 768],
                    [384, 192, 96, 48],
                    [24, 12, 6, 3],
                    [2, 1, 0, 0],
                ],
                dtype=np.int32,
            )

            early_expected = 0.75 * base_policy._afterstate_value(early_board) + 0.25 * sidecar_policy._afterstate_value(early_board)
            endgame_expected = (
                0.75 * base_policy._afterstate_value(endgame_board)
                + 0.25 * sidecar_policy._afterstate_value(endgame_board)
                + 0.5 * bonus_policy._afterstate_value(endgame_board)
            )

            self.assertAlmostEqual(policy._afterstate_value(early_board), early_expected)
            self.assertAlmostEqual(policy._afterstate_value(endgame_board), endgame_expected)

    def test_maxblended_ntuple_expectimax_afterstate_value_uses_max(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = self._save_tiny_checkpoint(tmp, "base", init=1.0)
            sidecar_a = self._save_tiny_checkpoint(tmp, "sidecar_a", init=3.0)
            sidecar_b = self._save_tiny_checkpoint(tmp, "sidecar_b", init=5.0)
            sidecar_b_policy = NtupleExpectimaxPolicy(sidecar_b, depth=2)
            blended = NtupleExpectimaxPolicy(
                base,
                depth=2,
                blend_specs=[(sidecar_a, 0.0), (sidecar_b, 0.0)],
                ensemble_mode="max",
            )
            board = np.zeros((4, 4), dtype=np.int32)

            self.assertAlmostEqual(blended._afterstate_value(board), sidecar_b_policy._afterstate_value(board))

    def test_ntuple_expectimax_tiebreak_prefers_up_left_within_margin(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = self._save_tiny_checkpoint(tmp)
            policy = NtupleExpectimaxPolicy(checkpoint, depth=2, tie_margin=1.0, tie_breaker="up_left")
            rng = np.random.default_rng(1)

            action = policy._select_action([(RIGHT, 10.0), (UP, 9.5), (LEFT, 9.4)], rng)

            self.assertEqual(action, UP)

    def test_ntuple_expectimax_tiebreak_keeps_best_outside_margin(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = self._save_tiny_checkpoint(tmp)
            policy = NtupleExpectimaxPolicy(checkpoint, depth=2, tie_margin=0.25, tie_breaker="up_left")
            rng = np.random.default_rng(1)

            action = policy._select_action([(RIGHT, 10.0), (UP, 9.5), (LEFT, 9.4)], rng)

            self.assertEqual(action, RIGHT)

    def test_high_tile_geometry_score_prefers_supported_top_left_shape(self):
        good = np.asarray(
            [
                [3072, 1536, 384, 96],
                [768, 192, 48, 12],
                [24, 6, 3, 2],
                [0, 0, 0, 0],
            ],
            dtype=np.int32,
        )
        stranded = np.asarray(
            [
                [1536, 384, 96, 48],
                [768, 3072, 24, 12],
                [3, 6, 3, 2],
                [1, 2, 3, 1],
            ],
            dtype=np.int32,
        )

        self.assertGreater(
            high_tile_geometry_score(good, min_tile=3072),
            high_tile_geometry_score(stranded, min_tile=3072),
        )

    def test_ntuple_expectimax_geometry_bonus_adds_to_afterstate_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = self._save_tiny_checkpoint(tmp)
            base = NtupleExpectimaxPolicy(checkpoint, depth=2)
            shaped = NtupleExpectimaxPolicy(checkpoint, depth=2, geometry_weight=10.0, geometry_min_tile=3072)
            board = np.asarray(
                [
                    [3072, 1536, 384, 96],
                    [768, 192, 48, 12],
                    [24, 6, 3, 2],
                    [0, 0, 0, 0],
                ],
                dtype=np.int32,
            )

            expected = base._afterstate_value(board) + 10.0 * high_tile_geometry_score(board, min_tile=3072)

            self.assertAlmostEqual(shaped._afterstate_value(board), expected)

    def test_anchor_guard_filters_moves_that_dislodge_large_top_left_tile(self):
        class FakePolicy:
            name = "fake"

            def action_values(self, state, sim):
                return [(RIGHT, 10.0), (UP, 9.0), (LEFT, 8.0)]

            def __call__(self, state, sim, rng):
                return RIGHT

        policy = AnchorGuardPolicy(FakePolicy())
        sim = ThreesSim(np.random.default_rng(1), starter_tile=None)
        state = self._state(
            np.asarray(
                [
                    [1536, 0, 0, 0],
                    [0, 3, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
                dtype=np.int32,
            ),
            1536,
        )

        action = policy(state, sim, np.random.default_rng(2))

        self.assertEqual(action, UP)

    def test_anchor_guard_is_inactive_below_large_anchor_threshold(self):
        class FakePolicy:
            name = "fake"

            def action_values(self, state, sim):
                return [(RIGHT, 10.0), (UP, 9.0), (LEFT, 8.0)]

            def __call__(self, state, sim, rng):
                return RIGHT

        policy = AnchorGuardPolicy(FakePolicy())
        sim = ThreesSim(np.random.default_rng(1), starter_tile=None)
        state = self._state(
            np.asarray(
                [
                    [768, 0, 0, 0],
                    [0, 3, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
                dtype=np.int32,
            ),
            768,
        )

        action = policy(state, sim, np.random.default_rng(2))

        self.assertEqual(action, RIGHT)

    def test_anchor_penalty_can_override_small_unsafe_preference(self):
        class FakePolicy:
            name = "fake"

            def action_values(self, state, sim):
                return [(RIGHT, 10.0), (UP, 9.0), (LEFT, 8.0)]

            def __call__(self, state, sim, rng):
                return RIGHT

        policy = AnchorGuardPolicy(FakePolicy(), penalty=2.0)
        sim = ThreesSim(np.random.default_rng(1), starter_tile=None)
        state = self._state(
            np.asarray(
                [
                    [1536, 0, 0, 0],
                    [0, 3, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
                dtype=np.int32,
            ),
            1536,
        )

        action = policy(state, sim, np.random.default_rng(2))

        self.assertEqual(action, UP)

    def test_anchor_penalty_keeps_large_unsafe_preference(self):
        class FakePolicy:
            name = "fake"

            def action_values(self, state, sim):
                return [(RIGHT, 10.0), (UP, 9.0), (LEFT, 8.0)]

            def __call__(self, state, sim, rng):
                return RIGHT

        policy = AnchorGuardPolicy(FakePolicy(), penalty=0.25)
        sim = ThreesSim(np.random.default_rng(1), starter_tile=None)
        state = self._state(
            np.asarray(
                [
                    [1536, 0, 0, 0],
                    [0, 3, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
                dtype=np.int32,
            ),
            1536,
        )

        action = policy(state, sim, np.random.default_rng(2))

        self.assertEqual(action, RIGHT)

    def test_ntuple_expectimax_budgeted_outcomes_normalize_probability(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = self._save_tiny_checkpoint(tmp)
            policy = NtupleExpectimaxPolicy(checkpoint, depth=2, chance_limit=2)
            sim = ThreesSim(np.random.default_rng(1), starter_tile=None)
            state = self._state(
                np.asarray(
                    [
                        [0, 1, 2, 0],
                        [0, 3, 3, 0],
                        [0, 6, 6, 0],
                        [0, 12, 12, 0],
                    ],
                    dtype=np.int32,
                ),
                1536,
            )
            outcomes = sim.transition_outcomes(state, LEFT, include_info=True)

            budgeted = policy._budgeted_outcomes(outcomes, sim, depth=2)
            exact_leaf = policy._budgeted_outcomes(outcomes, sim, depth=1)

            self.assertEqual(len(budgeted), 2)
            self.assertAlmostEqual(sum(probability for probability, _state, _info in budgeted), 1.0)
            self.assertEqual(len(exact_leaf), len(outcomes))


if __name__ == "__main__":
    unittest.main()
