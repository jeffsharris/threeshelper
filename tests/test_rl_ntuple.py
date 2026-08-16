import csv
import json
import tempfile
import unittest
import shutil
from pathlib import Path

import numpy as np

from threes_rl.ntuple import (
    CORNER_RISK_NAMES,
    NtupleValue,
    ResidualStagedNtupleValue,
    StagedNtupleValue,
    SYMMETRIES,
    afterstate_action_value,
    choose_action,
    corner_risk_bucket_for_board,
    expected_afterstate_target,
    index_for_pattern,
    phase4_index_for_board,
    patterns_for_set,
    rank_board,
)
from threes_rl.eval import make_policy, run_game_with_optional_replay
from threes_rl.run_artifacts import write_json
from threes_rl.sim import LEFT, SimState, ThreesSim, preview_from_label
from threes_rl.train_td import (
    TDConfig,
    StartStateReservoir,
    create_value_model,
    apply_nstep_updates,
    iter_fixed_actor_episodes,
    load_start_states,
    parse_phase_filter,
    play_episode,
    prepare_fixed_actor_jobs,
    start_state_phase_index,
    train,
)


class NtupleTests(unittest.TestCase):
    def _residual_fixture(self, root: Path):
        checkpoints = []
        boards = [
            np.asarray([[1536, 24, 12, 6], [3, 2, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int32),
            np.asarray([[1536, 384, 192, 96], [48, 24, 12, 6], [3, 2, 1, 0], [0, 0, 0, 0]], dtype=np.int32),
            np.asarray([[1536, 1536, 768, 384], [192, 96, 48, 24], [12, 6, 3, 2], [1, 0, 0, 0]], dtype=np.int32),
            np.asarray([[1536, 3072, 1536, 768], [384, 192, 96, 48], [24, 12, 6, 3], [2, 1, 0, 0]], dtype=np.int32),
        ]
        for idx in range(4):
            model = NtupleValue.from_pattern_set("tiny")
            for board_idx, board in enumerate(boards):
                model.update(board, target=float((idx + 1) * (board_idx + 2) * 100), alpha=0.5)
            checkpoint = root / f"component_{idx}"
            model.save(checkpoint)
            checkpoints.append(checkpoint)
        spec = (
            f"ntuple_phaseblend_expectimax2:{checkpoints[0]}"
            f":{checkpoints[1]}:0.25:all"
            f":{checkpoints[2]}:0.05:mid"
            f":{checkpoints[3]}:0.10:endgame"
        )
        policy = make_policy(spec)
        composite = ResidualStagedNtupleValue.from_frozen_blend(
            frozen_policy_spec=spec,
            base_checkpoint=policy.checkpoint,
            blend_specs=list(policy.blend_specs),
            phase_blend_specs=list(policy.phase_blend_specs),
            pattern_set="tiny",
            starter_tile=1536,
        )
        composite.enable_temporal_coherence()
        return composite, policy, boards, spec

    def test_zero_residual_exactly_matches_frozen_incumbent_leaf(self):
        with tempfile.TemporaryDirectory() as tmp:
            composite, policy, boards, _spec = self._residual_fixture(Path(tmp))
            for board in boards:
                self.assertEqual(composite.residual_value(board), 0.0)
                self.assertEqual(composite.value(board), policy._afterstate_value(board))

    def test_residual_update_does_not_mutate_frozen_components(self):
        with tempfile.TemporaryDirectory() as tmp:
            composite, _policy, boards, _spec = self._residual_fixture(Path(tmp))
            frozen_before = [
                [np.asarray(table).copy() for table in model.tables]
                for model in composite.frozen_models
            ]
            before_total = composite.value(boards[0])
            delta = composite.update_tc(boards[0], before_total + 50.0, alpha=0.1)
            self.assertAlmostEqual(delta, 50.0)
            self.assertNotEqual(composite.residual_value(boards[0]), 0.0)
            for model, before_tables in zip(composite.frozen_models, frozen_before):
                for table, before in zip(model.tables, before_tables):
                    np.testing.assert_array_equal(table, before)

    def test_residual_stage_promotion_copies_only_residual_weight_and_tc(self):
        with tempfile.TemporaryDirectory() as tmp:
            composite, _policy, boards, _spec = self._residual_fixture(Path(tmp))
            early, mid = boards[:2]
            composite.update_tc(early, composite.value(early) + 30.0, alpha=0.1)
            residual = composite.residual
            indices = residual.stages[1].indices(mid)
            table_idx, index = indices[0]
            expected_weight = residual._effective_entry(0, table_idx, index, "tables")
            expected_sum = residual._effective_entry(0, table_idx, index, "tc_sum_tables")
            frozen_before = composite.frozen_value(mid)

            composite.update_tc(mid, composite.value(mid) + 10.0, alpha=0.1)

            self.assertTrue(residual.promotion_masks[1][table_idx][index])
            self.assertNotEqual(residual.stages[1].tables[table_idx][index], 0.0)
            self.assertNotEqual(residual.stages[1].tc_sum_tables[table_idx][index], 0.0)
            self.assertLessEqual(abs(float(residual.stages[1].tables[table_idx][index]) - expected_weight), 10.0)
            self.assertLessEqual(abs(float(residual.stages[1].tc_sum_tables[table_idx][index]) - expected_sum), 10.0)
            self.assertEqual(composite.frozen_value(mid), frozen_before)

    def test_residual_composite_save_load_preserves_predictions_and_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            composite, _policy, boards, _spec = self._residual_fixture(root)
            composite.update_tc(boards[1], composite.value(boards[1]) + 25.0, alpha=0.1)
            expected = [composite.value(board) for board in boards]
            promoted = list(composite.residual.promotion_counts)
            checkpoint = root / "composite"
            composite.save(checkpoint)

            loaded = NtupleValue.load(checkpoint)

            self.assertIsInstance(loaded, ResidualStagedNtupleValue)
            self.assertEqual(loaded.residual.promotion_counts, promoted)
            self.assertEqual([loaded.value(board) for board in boards], expected)

    def test_residual_config_rejects_non_incumbent_actor(self):
        with tempfile.TemporaryDirectory() as tmp:
            _composite, _policy, _boards, spec = self._residual_fixture(Path(tmp))
            config = TDConfig(
                run_name="bad_residual_actor",
                games=1,
                pattern_set="tiny",
                stage_mode="phase4",
                stage_weight_promotion=True,
                use_tc=True,
                frozen_incumbent_policy=spec,
                actor_policy="greedy",
            )
            with self.assertRaisesRegex(ValueError, "exact frozen incumbent"):
                create_value_model(config)

    def test_residual_training_actions_match_frozen_actor(self):
        with tempfile.TemporaryDirectory() as tmp:
            composite, policy, _boards, spec = self._residual_fixture(Path(tmp))
            config = TDConfig(
                run_name="fixed_actor_test",
                games=1,
                pattern_set="tiny",
                stage_mode="phase4",
                stage_weight_promotion=True,
                alpha=0.001,
                seed=71,
                starter_tile=1536,
                max_moves=4,
                actor_policy=spec,
                target_mode="nstep",
                n_step=3,
                use_tc=True,
                frozen_incumbent_policy=spec,
            )
            episode = play_episode(composite, config, 1, actor_policy=policy)
            game_seed = config.seed + 1_000_003
            _result, replay = run_game_with_optional_replay(
                policy,
                policy_name=spec,
                seed=game_seed,
                starter_tile=1536,
                max_moves=4,
                capture_replay=True,
            )
            self.assertIsNotNone(replay)
            trained_actions = [frame["move"]["action"] for frame in episode.replay["frames"][1:]]
            frozen_actions = [frame["move"]["action"] for frame in replay["frames"][1:]]
            self.assertEqual(trained_actions, frozen_actions)

    def test_deferred_fixed_actor_updates_match_serial_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            composite, policy, _boards, spec = self._residual_fixture(root)
            checkpoint = root / "initial"
            composite.save(checkpoint)
            serial_model = NtupleValue.load(checkpoint)
            deferred_model = NtupleValue.load(checkpoint)
            config = TDConfig(
                run_name="fixed_actor_deferred_test",
                games=1,
                pattern_set="tiny",
                stage_mode="phase4",
                stage_weight_promotion=True,
                alpha=0.001,
                seed=79,
                starter_tile=1536,
                max_moves=5,
                actor_policy=spec,
                target_mode="nstep",
                n_step=3,
                use_tc=True,
                frozen_incumbent_policy=spec,
                actor_generation_jobs=1,
            )
            serial_episode = play_episode(serial_model, config, 1, actor_policy=policy)
            jobs = prepare_fixed_actor_jobs(config, range(1, 2), None)
            _game_index, deferred_episode = list(iter_fixed_actor_episodes(config, jobs))[0]
            errors, applied, skipped = apply_nstep_updates(
                deferred_model,
                deferred_episode.deferred_nstep_afterstates,
                deferred_episode.result.score,
                config,
            )

            self.assertEqual(deferred_episode.result, serial_episode.result)
            self.assertEqual(applied, serial_episode.updates_applied)
            self.assertEqual(skipped, serial_episode.updates_skipped)
            self.assertEqual(float(np.mean(errors)), serial_episode.mean_abs_td_error)
            for serial_stage, deferred_stage in zip(serial_model.residual.stages, deferred_model.residual.stages):
                for serial_table, deferred_table in zip(serial_stage.tables, deferred_stage.tables):
                    np.testing.assert_array_equal(serial_table, deferred_table)
                for serial_table, deferred_table in zip(serial_stage.tc_sum_tables, deferred_stage.tc_sum_tables):
                    np.testing.assert_array_equal(serial_table, deferred_table)
                for serial_table, deferred_table in zip(serial_stage.tc_abs_tables, deferred_stage.tc_abs_tables):
                    np.testing.assert_array_equal(serial_table, deferred_table)

    def test_symmetry_permutations_round_trip_cells(self):
        identity = tuple(range(16))
        perms = {sym.name: sym.cell_perm for sym in SYMMETRIES}
        self.assertEqual(perms["identity"], identity)
        for sym in SYMMETRIES:
            self.assertEqual(sorted(sym.cell_perm), list(range(16)))
            self.assertEqual(sorted(sym.action_perm), list(range(4)))

    def test_pattern_index_uses_base_16_ranks(self):
        board = np.asarray(
            [
                [0, 1, 2, 3],
                [6, 12, 24, 48],
                [96, 192, 384, 768],
                [1536, 3072, 6144, 12288],
            ],
            dtype=np.int32,
        )
        ranks = rank_board(board)
        self.assertEqual(index_for_pattern(ranks, (0, 1, 2, 3)), 0 * 16**3 + 1 * 16**2 + 2 * 16 + 3)

    def test_big6_pattern_set_uses_eight_six_cell_patterns(self):
        patterns = patterns_for_set("big6")
        self.assertEqual(len(patterns), 8)
        self.assertTrue(all(len(pattern) == 6 for pattern in patterns))

    def test_phase4_index_excludes_free_starter_tile(self):
        early = np.asarray([[1536, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int32)
        mid = np.asarray([[1536, 384, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int32)
        late = np.asarray([[1536, 1536, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int32)
        endgame = np.asarray([[1536, 3072, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int32)

        self.assertEqual(phase4_index_for_board(early, starter_tile=1536), 0)
        self.assertEqual(phase4_index_for_board(mid, starter_tile=1536), 1)
        self.assertEqual(phase4_index_for_board(late, starter_tile=1536), 2)
        self.assertEqual(phase4_index_for_board(endgame, starter_tile=1536), 3)

    def test_corner_risk_bucket_for_board_uses_board_shape(self):
        low = np.asarray([[1536, 384, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int32)
        medium = np.asarray(
            [
                [3072, 384, 2, 6],
                [1536, 192, 48, 6],
                [3, 3, 6, 0],
                [0, 0, 1, 3],
            ],
            dtype=np.int32,
        )
        high = np.asarray(
            [
                [1536, 384, 2, 6],
                [3072, 192, 48, 6],
                [3, 3, 6, 12],
                [6, 12, 1, 3],
            ],
            dtype=np.int32,
        )

        self.assertEqual(corner_risk_bucket_for_board(low, starter_tile=1536), "low_corner_risk")
        self.assertEqual(corner_risk_bucket_for_board(medium, starter_tile=1536), "medium_corner_risk")
        self.assertEqual(corner_risk_bucket_for_board(high, starter_tile=1536), "high_corner_risk")

    def test_parse_phase_filter_accepts_aliases(self):
        self.assertEqual(parse_phase_filter("late,endgame"), ["late_1536", "endgame_3072p"])
        self.assertEqual(parse_phase_filter("mid,middle"), ["mid_384_768"])
        self.assertIsNone(parse_phase_filter(None))

    def test_start_state_reservoir_can_balance_phase_buckets(self):
        def state_with_board(board):
            return SimState(
                board=np.asarray(board, dtype=np.int32),
                preview=preview_from_label("blue"),
                small_counts={"blue": 1, "red": 1, "gray": 1},
                small_pos=0,
                small_seen_total=0,
                span_small_pos=0,
                large_pending=False,
                max_tile=int(np.max(board)),
                move_count=0,
                game_over=False,
            )

        early = state_with_board([[1536, 192, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        mid = state_with_board([[1536, 768, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        late = state_with_board([[1536, 1536, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        endgame = state_with_board([[1536, 3072, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

        self.assertEqual(start_state_phase_index(early, 1536), 0)
        self.assertEqual(start_state_phase_index(mid, 1536), 1)
        self.assertEqual(start_state_phase_index(late, 1536), 2)
        self.assertEqual(start_state_phase_index(endgame, 1536), 3)

        reservoir = StartStateReservoir([early, early, early, mid, late, endgame], 1536, "phase_balanced")
        summary = reservoir.summary()

        self.assertEqual(
            summary["phase_buckets"],
            {
                "early_lt384": 3,
                "mid_384_768": 1,
                "late_1536": 1,
                "endgame_3072p": 1,
            },
        )
        sampled_phases = {
            start_state_phase_index(reservoir.sample(np.random.default_rng(seed)), 1536)
            for seed in range(100)
        }
        self.assertEqual(sampled_phases, {0, 1, 2, 3})

    def test_value_update_round_trips_checkpoint(self):
        model = NtupleValue.from_pattern_set("tiny")
        board = np.zeros((4, 4), dtype=np.int32)
        before = model.value(board)
        delta = model.update(board, target=64.0, alpha=0.5)
        self.assertGreater(delta, 0.0)
        self.assertGreater(model.value(board), before)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt"
            model.save(path)
            loaded = NtupleValue.load(path)
            self.assertAlmostEqual(loaded.value(board), model.value(board))

    def test_value_clone_has_independent_tables(self):
        model = NtupleValue.from_pattern_set("tiny")
        clone = model.clone()
        board = np.zeros((4, 4), dtype=np.int32)

        clone.update(board, target=64.0, alpha=0.5)

        self.assertEqual(model.value(board), 0.0)
        self.assertGreater(clone.value(board), 0.0)

    def test_temporal_coherence_update_round_trips_checkpoint(self):
        model = NtupleValue.from_pattern_set("tiny")
        board = np.zeros((4, 4), dtype=np.int32)
        delta = model.update_tc(board, target=64.0, alpha=0.5)
        self.assertGreater(delta, 0.0)
        self.assertIsNotNone(model.tc_sum_tables)
        self.assertIsNotNone(model.tc_abs_tables)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt"
            model.save(path)
            loaded = NtupleValue.load(path)
            self.assertIsNotNone(loaded.tc_sum_tables)
            self.assertIsNotNone(loaded.tc_abs_tables)
            self.assertAlmostEqual(loaded.value(board), model.value(board))

    def test_staged_value_updates_only_selected_phase_and_round_trips(self):
        model = StagedNtupleValue.from_pattern_set("tiny", starter_tile=1536)
        early = np.asarray([[1536, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int32)
        mid = np.asarray([[1536, 384, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int32)
        mid_before = model.value(mid)

        model.update(early, target=64.0, alpha=0.5)

        self.assertGreater(model.value(early), 0.0)
        self.assertEqual(model.value(mid), mid_before)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "staged"
            model.save(path)
            loaded = NtupleValue.load(path)

        self.assertIsInstance(loaded, StagedNtupleValue)
        self.assertAlmostEqual(loaded.value(early), model.value(early))
        self.assertAlmostEqual(loaded.value(mid), model.value(mid))

    def test_staged_value_can_split_by_phase_and_corner_risk(self):
        model = StagedNtupleValue.from_pattern_set("tiny", stage_mode="phase4_corner3", starter_tile=1536)
        self.assertEqual(len(model.stages), 4 * len(CORNER_RISK_NAMES))
        medium = np.asarray(
            [
                [3072, 384, 2, 6],
                [1536, 192, 48, 6],
                [3, 3, 6, 0],
                [0, 0, 1, 3],
            ],
            dtype=np.int32,
        )
        high = np.asarray(
            [
                [1536, 384, 2, 6],
                [3072, 192, 48, 6],
                [3, 3, 6, 12],
                [6, 12, 1, 3],
            ],
            dtype=np.int32,
        )
        high_before = model.value(high)

        model.update(medium, target=64.0, alpha=0.5)

        self.assertGreater(model.value(medium), 0.0)
        self.assertEqual(model.value(high), high_before)
        self.assertIn("/medium_corner_risk", model.stage_name(medium))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "staged"
            model.save(path)
            loaded = NtupleValue.load(path)

        self.assertIsInstance(loaded, StagedNtupleValue)
        self.assertEqual(loaded.stage_mode, "phase4_corner3")
        self.assertAlmostEqual(loaded.value(medium), model.value(medium))
        self.assertAlmostEqual(loaded.value(high), model.value(high))

    def test_lazy_staged_value_allocates_only_touched_stages(self):
        model = StagedNtupleValue.from_pattern_set("tiny", stage_mode="phase4_corner3", starter_tile=1536, lazy=True)
        self.assertTrue(all(stage is None for stage in model.stages))
        medium = np.asarray(
            [
                [3072, 384, 2, 6],
                [1536, 192, 48, 6],
                [3, 3, 6, 0],
                [0, 0, 1, 3],
            ],
            dtype=np.int32,
        )
        high = np.asarray(
            [
                [1536, 384, 2, 6],
                [3072, 192, 48, 6],
                [3, 3, 6, 12],
                [6, 12, 1, 3],
            ],
            dtype=np.int32,
        )

        model.update(medium, target=64.0, alpha=0.5)

        self.assertEqual(sum(stage is not None for stage in model.stages), 1)
        self.assertGreater(model.value(medium), 0.0)
        self.assertEqual(model.value(high), 0.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lazy_staged"
            model.save(path)
            loaded = NtupleValue.load(path)

        self.assertIsInstance(loaded, StagedNtupleValue)
        self.assertEqual(sum(stage is not None for stage in loaded.stages), 1)
        self.assertAlmostEqual(loaded.value(medium), model.value(medium))
        self.assertEqual(loaded.value(high), 0.0)

    def test_staged_value_can_clone_from_base_model(self):
        base = NtupleValue.from_pattern_set("tiny")
        board = np.zeros((4, 4), dtype=np.int32)
        base.update(board, target=64.0, alpha=0.5)

        staged = StagedNtupleValue.from_base_model(base, starter_tile=1536)
        staged.update(board, target=128.0, alpha=0.5)

        self.assertNotEqual(staged.value(board), base.value(board))
        self.assertGreater(base.value(board), 0.0)

    def _promotion_boards(self):
        early = np.asarray(
            [[1536, 192, 48, 12], [6, 3, 2, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
            dtype=np.int32,
        )
        mid = early.copy()
        mid[0, 1] = 384
        return early, mid

    def test_stage_promotion_first_training_access_copies_previous_stage(self):
        base = NtupleValue.from_pattern_set("tiny")
        early, mid = self._promotion_boards()
        base.update(early, target=96.0, alpha=0.5)
        staged = StagedNtupleValue.from_base_model(base, promotion_enabled=True)
        stage1 = staged.stages[1]
        self.assertIsNotNone(stage1)
        before_masks = sum(int(np.count_nonzero(mask)) for mask in staged.promotion_masks[1])

        inherited_value = staged.value(mid)
        staged.update(mid, target=inherited_value, alpha=0.0)

        self.assertEqual(before_masks, 0)
        self.assertAlmostEqual(stage1.value(mid), inherited_value)  # type: ignore[union-attr]
        self.assertGreater(staged.promotion_counts[1], 0)

    def test_stage_promotion_observes_previous_stage_update_before_first_access(self):
        base = NtupleValue.from_pattern_set("tiny")
        _early, mid = self._promotion_boards()
        staged = StagedNtupleValue.from_base_model(base, promotion_enabled=True)
        stage0 = staged.stages[0]
        self.assertIsNotNone(stage0)

        stage0.update(mid, target=128.0, alpha=0.5)  # type: ignore[union-attr]
        expected = stage0.value(mid)  # type: ignore[union-attr]
        staged.update(mid, target=expected, alpha=0.0)

        self.assertAlmostEqual(staged.stages[1].value(mid), expected)  # type: ignore[union-attr]

    def test_stage_promotion_does_not_overwrite_promoted_entry(self):
        base = NtupleValue.from_pattern_set("tiny")
        _early, mid = self._promotion_boards()
        staged = StagedNtupleValue.from_base_model(base, promotion_enabled=True)
        inherited = staged.value(mid)
        staged.update(mid, target=inherited, alpha=0.0)
        promoted = staged.value(mid)
        count = staged.promotion_counts[1]

        staged.stages[0].update(mid, target=256.0, alpha=0.5)  # type: ignore[union-attr]

        self.assertAlmostEqual(staged.value(mid), promoted)
        self.assertEqual(staged.promotion_counts[1], count)

    def test_stage_promotion_copies_temporal_coherence_state(self):
        base = NtupleValue.from_pattern_set("tiny")
        _early, mid = self._promotion_boards()
        base.enable_temporal_coherence()
        for table_idx, index in set(base.indices(mid)):
            base.tables[table_idx][index] = 3.5
            base.tc_sum_tables[table_idx][index] = 7.0  # type: ignore[index]
            base.tc_abs_tables[table_idx][index] = 9.0  # type: ignore[index]
        staged = StagedNtupleValue.from_base_model(base, promotion_enabled=True)
        inherited = staged.value(mid)

        staged.update_tc(mid, target=inherited, alpha=0.0)

        stage1 = staged.stages[1]
        self.assertIsNotNone(stage1)
        for table_idx, index in set(stage1.indices(mid)):  # type: ignore[union-attr]
            self.assertEqual(float(stage1.tc_sum_tables[table_idx][index]), 7.0)  # type: ignore[index,union-attr]
            self.assertEqual(float(stage1.tc_abs_tables[table_idx][index]), 9.0)  # type: ignore[index,union-attr]

    def test_stage_promotion_round_trip_preserves_masks_and_predictions(self):
        base = NtupleValue.from_pattern_set("tiny")
        _early, mid = self._promotion_boards()
        base.update(mid, target=64.0, alpha=0.5)
        staged = StagedNtupleValue.from_base_model(base, promotion_enabled=True)
        staged.update_tc(mid, target=staged.value(mid) + 4.0, alpha=0.01)
        expected = staged.value(mid)
        expected_count = staged.promotion_counts[1]

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "promoted"
            staged.save(path)
            loaded = NtupleValue.load(path)

        self.assertIsInstance(loaded, StagedNtupleValue)
        self.assertTrue(loaded.promotion_enabled)
        self.assertEqual(loaded.promotion_counts[1], expected_count)
        self.assertAlmostEqual(loaded.value(mid), expected)

    def test_stage_promotion_read_only_value_does_not_mutate_masks(self):
        base = NtupleValue.from_pattern_set("tiny")
        _early, mid = self._promotion_boards()
        base.update(mid, target=64.0, alpha=0.5)
        staged = StagedNtupleValue.from_base_model(base, promotion_enabled=True)
        before = [mask.copy() for mask in staged.promotion_masks[1]]

        self.assertGreater(staged.value(mid), 0.0)

        for old, current in zip(before, staged.promotion_masks[1]):
            np.testing.assert_array_equal(current, old)
        self.assertEqual(staged.promotion_counts[1], 0)

    def test_stage_promotion_rejects_incompatible_patterns(self):
        tiny = NtupleValue.from_pattern_set("tiny")
        small = NtupleValue.from_pattern_set("small")
        with self.assertRaisesRegex(ValueError, "Incompatible patterns"):
            StagedNtupleValue(
                [tiny, small, tiny.clone(), tiny.clone()],
                stage_mode="phase4",
                promotion_enabled=True,
            )

    def test_legacy_staged_checkpoint_loads_without_promotion(self):
        model = StagedNtupleValue.from_pattern_set("tiny")
        _early, mid = self._promotion_boards()
        model.update(mid, target=32.0, alpha=0.5)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy"
            model.save(path)
            loaded = NtupleValue.load(path)

        self.assertIsInstance(loaded, StagedNtupleValue)
        self.assertFalse(loaded.promotion_enabled)
        self.assertAlmostEqual(loaded.value(mid), model.value(mid))

    def test_choose_action_returns_legal_afterstate(self):
        sim = ThreesSim(np.random.default_rng(4), starter_tile=1536)
        state = sim.reset()
        model = NtupleValue(patterns_for_set("tiny"))
        action, afterstate = choose_action(model, state, sim, np.random.default_rng(5))
        self.assertIn(action, sim.legal_actions(state))
        self.assertEqual(afterstate.shape, (4, 4))

    def _dead_after_left_state(self, preview):
        board = np.asarray(
            [
                [0, 6, 12, 24],
                [48, 96, 192, 384],
                [768, 1536, 3072, 6144],
                [6, 12, 24, 48],
            ],
            dtype=np.int32,
        )
        return SimState(
            board=board,
            preview=preview,
            small_counts={"red": 4, "blue": 4, "gray": 4},
            small_pos=0,
            small_seen_total=21,
            span_small_pos=0,
            large_pending=True,
            max_tile=6144,
            move_count=0,
            game_over=False,
        )

    def test_expected_afterstate_target_includes_known_spawn_score(self):
        sim = ThreesSim(np.random.default_rng(4), starter_tile=1536)
        state = self._dead_after_left_state(preview_from_label("gray"))
        model = NtupleValue(patterns_for_set("tiny"))

        target, afterstate = expected_afterstate_target(model, state, sim, LEFT)
        action_value, _ = afterstate_action_value(model, state, sim, LEFT)

        self.assertIsNotNone(afterstate)
        self.assertAlmostEqual(target, 3.0)
        self.assertAlmostEqual(action_value, 3.0)

    def test_expected_afterstate_target_averages_bonus_candidates(self):
        sim = ThreesSim(np.random.default_rng(4), starter_tile=1536)
        state = self._dead_after_left_state(preview_from_label("large_candidates", (6, 12, 48)))
        model = NtupleValue(patterns_for_set("tiny"))

        target, afterstate = expected_afterstate_target(model, state, sim, LEFT)

        self.assertIsNotNone(afterstate)
        self.assertAlmostEqual(target, (9 + 27 + 243) / 3)

    def test_collapsed_next_preview_keeps_preview_independent_target(self):
        sim = ThreesSim(np.random.default_rng(4), starter_tile=1536)
        state = self._dead_after_left_state(preview_from_label("large_candidates", (6, 12, 48)))
        model = NtupleValue(patterns_for_set("tiny"))
        shifted, _eligible = sim.transition_outcomes(state, LEFT, include_next_preview=False)[0][1].board, None
        merge_delta = 0.0

        full = 0.0
        for probability, next_state, info in sim.transition_outcomes(state, LEFT, include_next_preview=True):
            full += probability * (info.score_delta - merge_delta + model.value(shifted) * 0.0)
        collapsed = 0.0
        collapsed_outcomes = sim.transition_outcomes(state, LEFT, include_next_preview=False)
        for probability, _next_state, info in collapsed_outcomes:
            collapsed += probability * (info.score_delta - merge_delta)

        self.assertLess(len(collapsed_outcomes), len(sim.transition_outcomes(state, LEFT, include_next_preview=True)))
        self.assertAlmostEqual(collapsed, full)

    def test_train_td_smoke_writes_artifacts(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            run_name = f"td_smoke_{Path(tmp).name}"
            config = TDConfig(
                run_name=run_name,
                games=3,
                pattern_set="tiny",
                alpha=0.01,
                seed=7,
                progress_every=1,
                checkpoint_every=2,
                keep_top_games=2,
                max_moves=50,
            )
            checkpoint = train(config)
            run_dir = checkpoint.parent
            self.assertTrue((run_dir / "summary.json").exists())
            self.assertTrue((run_dir / "progress.html").exists())
            self.assertTrue((run_dir / "top_games" / "manifest.json").exists())
            self.assertTrue((checkpoint / "meta.json").exists())
            shutil.rmtree(run_dir)

    def test_train_td_mc_actor_smoke_writes_checkpoint(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            run_name = f"td_mc_smoke_{Path(tmp).name}"
            config = TDConfig(
                run_name=run_name,
                games=2,
                pattern_set="tiny",
                alpha=0.01,
                seed=11,
                progress_every=1,
                checkpoint_every=2,
                keep_top_games=1,
                max_moves=12,
                actor_policy="greedy",
                target_mode="mc",
            )
            checkpoint = train(config)
            run_dir = checkpoint.parent
            self.assertTrue((run_dir / "summary.json").exists())
            self.assertTrue((checkpoint / "meta.json").exists())
            shutil.rmtree(run_dir)

    def test_train_td_nstep_tc_smoke_writes_checkpoint(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            run_name = f"td_nstep_smoke_{Path(tmp).name}"
            config = TDConfig(
                run_name=run_name,
                games=2,
                pattern_set="tiny",
                alpha=0.01,
                seed=13,
                progress_every=1,
                checkpoint_every=2,
                keep_top_games=1,
                max_moves=12,
                actor_policy="greedy",
                target_mode="nstep",
                n_step=3,
                use_tc=True,
            )
            checkpoint = train(config)
            run_dir = checkpoint.parent
            meta = (checkpoint / "meta.json").read_text()
            self.assertIn("tc_sum_tables", meta)
            self.assertTrue((run_dir / "summary.json").exists())
            shutil.rmtree(run_dir)

    def test_train_td_phase4_smoke_writes_staged_checkpoint(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            run_name = f"td_phase4_smoke_{Path(tmp).name}"
            config = TDConfig(
                run_name=run_name,
                games=2,
                pattern_set="tiny",
                stage_mode="phase4",
                alpha=0.01,
                seed=19,
                progress_every=1,
                checkpoint_every=2,
                keep_top_games=1,
                max_moves=8,
            )
            checkpoint = train(config)
            run_dir = checkpoint.parent
            meta = json.loads((checkpoint / "meta.json").read_text())

            self.assertEqual(meta["value_type"], "staged")
            self.assertEqual(meta["stage_mode"], "phase4")
            self.assertTrue((checkpoint / "stage_00_early_lt384" / "meta.json").exists())
            shutil.rmtree(run_dir)

    def test_train_td_lazy_phase4_corner3_allocates_touched_stages(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            run_name = f"td_lazy_phase4_corner3_smoke_{Path(tmp).name}"
            config = TDConfig(
                run_name=run_name,
                games=1,
                pattern_set="tiny",
                stage_mode="phase4_corner3",
                lazy_stages=True,
                alpha=0.01,
                seed=31,
                progress_every=1,
                checkpoint_every=1,
                keep_top_games=1,
                max_moves=4,
            )
            checkpoint = train(config)
            run_dir = checkpoint.parent
            meta = json.loads((checkpoint / "meta.json").read_text())
            summary = json.loads((run_dir / "summary.json").read_text())

            self.assertEqual(meta["value_type"], "staged")
            self.assertEqual(meta["stage_mode"], "phase4_corner3")
            self.assertEqual(meta["pattern_set"], "tiny")
            self.assertLess(sum(stage_dir is not None for stage_dir in meta["stage_dirs"]), 12)
            self.assertGreater(sum(stage_dir is not None for stage_dir in meta["stage_dirs"]), 0)
            self.assertEqual(
                summary["allocated_stages"]["allocated_count"],
                sum(stage_dir is not None for stage_dir in meta["stage_dirs"]),
            )
            self.assertEqual(summary["allocated_stages"]["stage_mode"], "phase4_corner3")
            shutil.rmtree(run_dir)

    def test_train_td_phase4_resume_from_single_checkpoint_clones_stages(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            base = NtupleValue.from_pattern_set("tiny")
            base_path = Path(tmp) / "base"
            base.save(base_path)
            run_name = f"td_phase4_resume_smoke_{Path(tmp).name}"
            config = TDConfig(
                run_name=run_name,
                games=1,
                pattern_set="tiny",
                stage_mode="phase4",
                alpha=0.01,
                seed=23,
                progress_every=1,
                checkpoint_every=1,
                keep_top_games=1,
                max_moves=4,
            )
            checkpoint = train(config, resume=base_path)
            run_dir = checkpoint.parent
            meta = json.loads((checkpoint / "meta.json").read_text())

            self.assertEqual(meta["value_type"], "staged")
            shutil.rmtree(run_dir)

    def test_train_td_promoted_stage_resume_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            parent = NtupleValue.from_pattern_set("tiny")
            parent_path = Path(tmp) / "parent"
            parent.save(parent_path)
            split_name = f"td_promoted_resume_{Path(tmp).name}"
            full_name = f"td_promoted_full_{Path(tmp).name}"
            common = dict(
                pattern_set="tiny",
                stage_mode="phase4",
                stage_weight_promotion=True,
                alpha=0.01,
                seed=47,
                progress_every=0,
                checkpoint_every=0,
                keep_top_games=0,
                max_moves=6,
                actor_policy="greedy",
                target_mode="nstep",
                n_step=3,
                use_tc=True,
            )
            first_checkpoint = train(TDConfig(run_name=split_name, games=2, **common), resume=parent_path)
            split_checkpoint = train(TDConfig(run_name=split_name, games=4, **common), resume=first_checkpoint)
            full_checkpoint = train(TDConfig(run_name=full_name, games=4, **common), resume=parent_path)

            split = NtupleValue.load(split_checkpoint)
            full = NtupleValue.load(full_checkpoint)

            self.assertIsInstance(split, StagedNtupleValue)
            self.assertIsInstance(full, StagedNtupleValue)
            self.assertEqual(split.promotion_counts, full.promotion_counts)
            self.assertEqual(split.feature_access_counts, full.feature_access_counts)
            for split_stage, full_stage in zip(split.stages, full.stages):
                self.assertIsNotNone(split_stage)
                self.assertIsNotNone(full_stage)
                for split_table, full_table in zip(split_stage.tables, full_stage.tables):
                    np.testing.assert_array_equal(split_table, full_table)
                for split_table, full_table in zip(split_stage.tc_sum_tables, full_stage.tc_sum_tables):
                    np.testing.assert_array_equal(split_table, full_table)
                for split_table, full_table in zip(split_stage.tc_abs_tables, full_stage.tc_abs_tables):
                    np.testing.assert_array_equal(split_table, full_table)
            split_meta = json.loads((split_checkpoint / "meta.json").read_text())
            self.assertEqual(split_meta["games_completed"], 4)
            self.assertEqual(split_meta["promotion_semantics"], "copy_weight_and_tc_on_first_training_access")
            shutil.rmtree(split_checkpoint.parent)
            shutil.rmtree(full_checkpoint.parent)

    def test_train_td_phase_filter_can_skip_early_updates(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            run_name = f"td_phase_filter_smoke_{Path(tmp).name}"
            config = TDConfig(
                run_name=run_name,
                games=1,
                pattern_set="tiny",
                stage_mode="phase4",
                alpha=0.01,
                seed=29,
                progress_every=1,
                checkpoint_every=1,
                keep_top_games=1,
                max_moves=4,
                train_phase_filter=["late_1536"],
            )
            checkpoint = train(config)
            run_dir = checkpoint.parent
            summary = json.loads((run_dir / "summary.json").read_text())
            with (run_dir / "metrics.csv").open() as fh:
                rows = list(csv.DictReader(fh))

            self.assertEqual(summary["train_phase_filter"], ["late_1536"])
            self.assertEqual(summary["updates_applied"], 0)
            self.assertGreater(summary["updates_skipped"], 0)
            self.assertEqual(int(rows[0]["updates_applied"]), 0)
            shutil.rmtree(run_dir)

    def test_train_td_starter_curriculum_records_per_episode_starter(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            run_name = f"td_curriculum_smoke_{Path(tmp).name}"
            config = TDConfig(
                run_name=run_name,
                games=4,
                pattern_set="tiny",
                alpha=0.01,
                seed=17,
                starter_tile=1536,
                starter_tiles=[None, 96],
                progress_every=2,
                checkpoint_every=4,
                keep_top_games=1,
                max_moves=8,
            )
            checkpoint = train(config)
            run_dir = checkpoint.parent
            with (run_dir / "metrics.csv").open() as fh:
                rows = list(csv.DictReader(fh))
            summary = json.loads((run_dir / "summary.json").read_text())

            self.assertEqual({row["starter_tile"] for row in rows}, {"", "96"})
            self.assertIn("by_starter", summary)
            self.assertEqual(summary["by_starter"]["none"]["games"], 2)
            self.assertEqual(summary["by_starter"]["96"]["games"], 2)
            shutil.rmtree(run_dir)

    def test_load_start_states_filters_replay_frames(self):
        replay = {
            "frames": [
                {
                    "state": {
                        "move_count": 0,
                        "board": [[1536, 768, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                        "max_tile": 1536,
                        "game_over": False,
                        "preview": {"kind": "gray", "label": "gray", "value": 3, "candidates": []},
                        "tile_cycle": {
                            "small_counts": {"red": 4, "blue": 4, "gray": 4},
                            "small_pos": 0,
                            "small_seen_total": 0,
                            "span_small_pos": 0,
                            "large_pending": False,
                            "max_tile": 1536,
                        },
                    }
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "replay.json"
            write_json(path, replay)
            states = load_start_states([str(path)], min_tile=768, starter_tile=1536)
            self.assertEqual(len(states), 1)
            self.assertEqual(int(states[0].board[0, 1]), 768)

    def test_load_start_states_accepts_reservoir_records(self):
        state_payload = {
            "move_count": 12,
            "board": [[1536, 3072, 0, 0], [768, 384, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            "max_tile": 3072,
            "game_over": False,
            "preview": {"kind": "gray", "label": "gray", "value": 3, "candidates": []},
            "tile_cycle": {
                "small_counts": {"red": 4, "blue": 4, "gray": 4},
                "small_pos": 0,
                "small_seen_total": 0,
                "span_small_pos": 0,
                "large_pending": False,
                "max_tile": 3072,
            },
        }
        reservoir = {
            "kind": "threes_high_board_reservoir",
            "records": [
                {"id": "keep", "starter_tile": 1536, "state": state_payload},
                {
                    "id": "skip",
                    "starter_tile": 1536,
                    "state": {
                        **state_payload,
                        "board": [[1536, 768, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                        "max_tile": 1536,
                    },
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reservoir.json"
            write_json(path, reservoir)
            states = load_start_states([str(path)], min_tile=3072, starter_tile=1536)

        self.assertEqual(len(states), 1)
        self.assertEqual(int(states[0].board[0, 1]), 3072)

    def test_train_td_replay_start_uses_relative_continuation_cap(self):
        replay = {
            "starter_tile": 1536,
            "frames": [
                {
                    "state": {
                        "move_count": 120,
                        "board": [
                            [1536, 3072, 0, 0],
                            [768, 384, 192, 96],
                            [48, 24, 12, 6],
                            [3, 2, 1, 0],
                        ],
                        "max_tile": 3072,
                        "game_over": False,
                        "preview": {"kind": "gray", "label": "gray", "value": 3, "candidates": []},
                        "tile_cycle": {
                            "small_counts": {"red": 4, "blue": 4, "gray": 4},
                            "small_pos": 0,
                            "small_seen_total": 0,
                            "span_small_pos": 0,
                            "large_pending": False,
                            "max_tile": 3072,
                        },
                    }
                }
            ],
        }
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            tmp_path = Path(tmp)
            replay_path = tmp_path / "replay.json"
            write_json(replay_path, replay)
            run_name = f"td_replay_continuation_cap_{tmp_path.name}"
            config = TDConfig(
                run_name=run_name,
                games=1,
                pattern_set="tiny",
                alpha=0.01,
                seed=37,
                progress_every=1,
                checkpoint_every=0,
                keep_top_games=0,
                max_moves=4,
                continuation_max_moves=2,
                actor_policy="greedy",
                start_state_replays=[str(replay_path)],
                start_state_prob=1.0,
                start_state_min_tile=3072,
            )
            checkpoint = train(config)
            run_dir = checkpoint.parent
            summary = json.loads((run_dir / "summary.json").read_text())

            self.assertEqual(summary["updates_applied"], 2)
            shutil.rmtree(run_dir)

    def test_train_td_replay_start_writes_episode_provenance(self):
        replay = {
            "starter_tile": 1536,
            "root_origin": "fresh",
            "root_replay": "normal/root/replay.json",
            "root_seed": 123,
            "root_frame_index": 0,
            "root_move_count": 0,
            "root_score": 59049,
            "root_policy": "fixture_policy",
            "frames": [
                {
                    "index": 9,
                    "state": {
                        "move_count": 120,
                        "board": [
                            [1536, 3072, 0, 0],
                            [768, 384, 192, 96],
                            [48, 24, 12, 6],
                            [3, 2, 1, 0],
                        ],
                        "max_tile": 3072,
                        "game_over": False,
                        "preview": {"kind": "gray", "label": "gray", "value": 3, "candidates": []},
                        "tile_cycle": {
                            "small_counts": {"red": 4, "blue": 4, "gray": 4},
                            "small_pos": 0,
                            "small_seen_total": 0,
                            "span_small_pos": 0,
                            "large_pending": False,
                            "max_tile": 3072,
                        },
                    }
                }
            ],
        }
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            tmp_path = Path(tmp)
            replay_path = tmp_path / "replay.json"
            write_json(replay_path, replay)
            run_name = f"td_replay_provenance_{tmp_path.name}"
            config = TDConfig(
                run_name=run_name,
                games=1,
                pattern_set="tiny",
                alpha=0.01,
                seed=38,
                progress_every=1,
                checkpoint_every=0,
                keep_top_games=1,
                max_moves=121,
                actor_policy="greedy",
                start_state_replays=[str(replay_path)],
                start_state_prob=1.0,
                start_state_min_tile=3072,
            )
            checkpoint = train(config)
            run_dir = checkpoint.parent
            manifest = json.loads((run_dir / "top_games" / "manifest.json").read_text())
            top_replay = json.loads(Path(manifest[0]["json"]).read_text())
            with (run_dir / "metrics.csv").open() as fh:
                rows = list(csv.DictReader(fh))

            self.assertEqual(top_replay["replay_origin"], "replay_start")
            self.assertEqual(top_replay["source_replay"], str(replay_path))
            self.assertEqual(top_replay["source_frame_index"], 9)
            self.assertEqual(top_replay["root_origin"], "fresh")
            self.assertEqual(top_replay["root_replay"], "normal/root/replay.json")
            self.assertEqual(rows[0]["replay_origin"], "replay_start")
            shutil.rmtree(run_dir)


if __name__ == "__main__":
    unittest.main()
