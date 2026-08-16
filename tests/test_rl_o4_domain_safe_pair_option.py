from __future__ import annotations

import copy

import numpy as np
import pytest

from threes_rl.baselines import GreedyPolicy
from threes_rl.o4_domain_safe_pair_option import (
    GEOMETRY_WIDTH,
    O4DesignatedPairNet,
    OUTPUT_WIDTH,
    VERSION,
    advance_lineage_base,
    apply_spawn_to_lineage,
    blocker_geometry,
    build_decision_targets,
    canonical_json_hash,
    eligible_blocker_cells,
    exhaustive_blocker_domain_proof,
    initial_lineage,
    option_features,
    parameter_count,
    root_option_eligible,
    schema_manifest,
    schema_sha256,
    select_designated_pair,
    successor_geometry,
    transition_status,
)
from threes_rl.o4_power_contract import (
    POINT_GATE_ODDS_RATIO,
    power_table,
    simulate_mechanism_power,
    stratum_counts,
)
from threes_rl.sim import Preview, SimState, ThreesSim


EXPECTED_SCHEMA_SHA256 = (
    "60a83881d8e8275a4aa2d03df06815d65e5b247b16f36118009f42f2ce3098ba"
)


def _sim() -> ThreesSim:
    return ThreesSim.from_stream_ids(
        deck_stream_id=2_026_072_811,
        slot_stream_id=2_026_072_812,
        starter_tile=1536,
    )


def _state(board: np.ndarray) -> SimState:
    return SimState(
        board=np.asarray(board, dtype=np.int32).copy(),
        preview=Preview("bonus", None, (24, 48, 96)),
        small_counts={"red": 2, "blue": 3, "gray": 4},
        small_pos=3,
        small_seen_total=55,
        span_small_pos=7,
        large_pending=True,
        max_tile=int(np.max(board)),
        move_count=120,
        game_over=False,
    )


def _hard_start_board() -> np.ndarray:
    return np.asarray(
        [
            [1536, 3, 6, 12],
            [48, 24, 48, 6144],
            [0, 3072, 2, 768],
            [0, 96, 192, 384],
        ],
        dtype=np.int32,
    )


def _merge_ready_board() -> np.ndarray:
    return np.asarray(
        [
            [1536, 3, 6, 12],
            [48, 48, 24, 6144],
            [0, 3072, 2, 768],
            [0, 96, 192, 384],
        ],
        dtype=np.int32,
    )


def test_schema_and_parameter_count_are_exact() -> None:
    assert VERSION == "o4_domain_safe_designated_pair_v1"
    assert OUTPUT_WIDTH == 29
    assert GEOMETRY_WIDTH == 8
    assert parameter_count() == 102_557
    assert sum(
        parameter.numel() for parameter in O4DesignatedPairNet().parameters()
    ) == 102_557
    assert schema_sha256() == EXPECTED_SCHEMA_SHA256
    assert schema_manifest()["pair_selection"].startswith(
        "min(0_if_safe_else_1"
    )


def test_schema_hash_changes_for_every_domain_semantic() -> None:
    baseline = schema_manifest()
    mutations = (
        ("version", "changed"),
        ("train_targets", [48, 96]),
        ("blocker_eligible_cells", {"changed": True}),
        ("blocker_density", {"changed": True}),
        ("all_model_inputs_domain", [0.0, 2.0]),
        ("pair_selection", "changed"),
        ("successor_transforms", ("changed",)),
        ("training_schedule", {"changed": True}),
    )
    for field, replacement in mutations:
        changed = copy.deepcopy(baseline)
        changed[field] = replacement
        assert canonical_json_hash(changed) != EXPECTED_SCHEMA_SHA256


def test_blocker_cells_ties_and_adjacent_zero_capacity() -> None:
    assert eligible_blocker_cells(((1, 1), (1, 2))) == ()
    assert eligible_blocker_cells(((1, 2), (1, 1))) == ()
    assert eligible_blocker_cells(((0, 0), (0, 3))) == ((0, 1), (0, 2))
    assert eligible_blocker_cells(((0, 0), (3, 0))) == ((1, 0), (2, 0))
    assert len(eligible_blocker_cells(((0, 0), (3, 3)))) == 14
    board = np.ones((4, 4), dtype=np.int32)
    adjacent = blocker_geometry(board, ((1, 1), (1, 2)))
    assert adjacent.capacity == 0
    assert adjacent.occupied == 0
    assert adjacent.density == 0.0


def test_exhaustive_blocker_density_domain_and_capacity_counts() -> None:
    proof = exhaustive_blocker_domain_proof()
    assert proof["passes"]
    assert proof["coordinate_pairs"] == 120
    assert proof["occupancy_cases"] == 43_296
    assert proof["capacity_pair_counts"] == {
        "0": 24,
        "1": 16,
        "2": 26,
        "4": 24,
        "6": 12,
        "7": 8,
        "10": 8,
        "14": 2,
    }
    assert proof["minimum_density"] == 0.0
    assert proof["maximum_density"] == 1.0


def test_pair_selection_uses_literal_safe_first_key() -> None:
    pair = select_designated_pair(
        _merge_ready_board(),
        1536,
        requested_target=48,
    )
    assert pair is not None
    assert pair.coordinates == ((1, 0), (1, 1))
    assert pair.safe_merge_actions
    assert pair.blocker_capacity == 0
    assert pair.blocker_density == 0.0

    hard = select_designated_pair(
        _hard_start_board(),
        1536,
        requested_target=48,
    )
    assert hard is not None
    assert not hard.safe_merge_actions
    assert hard.blocker_capacity == 1
    assert hard.blocker_density == 1.0


def test_features_and_successor_targets_are_bounded_and_nonmutating() -> None:
    sim = _sim()
    state = _state(_hard_start_board())
    pair = select_designated_pair(
        state.board,
        1536,
        requested_target=48,
    )
    assert pair is not None
    lineage = initial_lineage(pair)
    board_before = state.board.copy()
    lineage_before = lineage.copy()
    for action in sim.legal_actions(state):
        tokens, global_values = option_features(
            state,
            sim,
            starter_tile=1536,
            pair=pair,
            lineage=lineage,
            action=action,
        )
        assert tokens.shape == (16, 37)
        assert global_values.shape == (35,)
        assert np.isfinite(tokens).all() and np.isfinite(global_values).all()
        assert np.all((0.0 <= tokens) & (tokens <= 1.0))
        assert np.all((0.0 <= global_values) & (global_values <= 1.0))
    geometry = successor_geometry(
        state,
        sim,
        lineage=lineage,
        target=pair.target,
    )
    assert geometry.shape == (8,)
    assert np.all((0.0 <= geometry) & (geometry <= 1.0))
    assert geometry[2] == pytest.approx(pair.blocker_density)
    np.testing.assert_array_equal(state.board, board_before)
    np.testing.assert_array_equal(lineage, lineage_before)


def test_random_reachable_legal_transition_property_is_nonvacuous() -> None:
    eligible_states = 0
    legal_transitions = 0
    live_successors = 0
    for game_index in range(64):
        sim = ThreesSim.from_stream_ids(
            deck_stream_id=900_000 + game_index,
            slot_stream_id=910_000 + game_index,
            starter_tile=1536,
        )
        state = sim.reset()
        policy_rng = np.random.default_rng(920_000 + game_index)
        policy = GreedyPolicy()
        for _move in range(500):
            legal = tuple(int(action) for action in sim.legal_actions(state))
            if not legal:
                break
            pair = select_designated_pair(
                state.board,
                1536,
                allowed_targets=(48, 96, 192),
            )
            if (
                pair is not None
                and not pair.safe_merge_actions
                and root_option_eligible(state, sim, 1536)
            ):
                eligible_states += 1
                lineage = initial_lineage(pair)
                board_before = state.board.copy()
                deck_before = copy.deepcopy(sim.deck_rng.bit_generator.state)
                slot_before = copy.deepcopy(sim.slot_rng.bit_generator.state)
                for action in legal:
                    tokens, global_values = option_features(
                        state,
                        sim,
                        starter_tile=1536,
                        pair=pair,
                        lineage=lineage,
                        action=action,
                    )
                    assert np.all((0.0 <= tokens) & (tokens <= 1.0))
                    assert np.all(
                        (0.0 <= global_values) & (global_values <= 1.0)
                    )
                    np.testing.assert_array_equal(state.board, board_before)
                    assert sim.deck_rng.bit_generator.state == deck_before
                    assert sim.slot_rng.bit_generator.state == slot_before

                    base = advance_lineage_base(state.board, lineage, action)
                    branch_sim = copy.deepcopy(sim)
                    next_state, info = branch_sim.step(state, action)
                    assert info.moved
                    shifted = next_state.board.copy()
                    if info.inserted_pos is not None:
                        shifted[info.inserted_pos] = 0
                    np.testing.assert_array_equal(base.board, shifted)
                    assert tuple(base.eligible_slots) == tuple(
                        info.eligible_positions
                    )
                    next_lineage = (
                        base.lineage
                        if info.inserted_pos is None
                        else apply_spawn_to_lineage(
                            base.lineage,
                            info.inserted_pos,
                        )
                    )
                    status = transition_status(
                        next_state,
                        branch_sim,
                        starter_tile=1536,
                        lineage=next_lineage,
                        base_event=base.event,
                    )
                    if status == "live":
                        geometry = successor_geometry(
                            next_state,
                            branch_sim,
                            lineage=next_lineage,
                            target=pair.target,
                        )
                        assert np.isfinite(geometry).all()
                        assert np.all((0.0 <= geometry) & (geometry <= 1.0))
                        targets = build_decision_targets(
                            decision_move=0,
                            terminal_move=40,
                            terminal_status="censor",
                            live_geometry_by_move={
                                10: geometry,
                                20: geometry,
                                40: geometry,
                            },
                        )
                        live_successors += 1
                    else:
                        targets = build_decision_targets(
                            decision_move=0,
                            terminal_move=1,
                            terminal_status=status,
                            live_geometry_by_move={},
                        )
                    assert np.isfinite(targets.geometry).all()
                    assert np.all(
                        (0.0 <= targets.geometry)
                        & (targets.geometry <= 1.0)
                    )
                    legal_transitions += 1
                np.testing.assert_array_equal(state.board, board_before)
                assert sim.deck_rng.bit_generator.state == deck_before
                assert sim.slot_rng.bit_generator.state == slot_before

            action = policy(state, sim, policy_rng)
            state, info = sim.step(state, action)
            assert info.moved

    assert eligible_states == 15
    assert legal_transitions == 55
    assert live_successors > 0


def test_root_eligibility_remains_hard_start_only() -> None:
    sim = _sim()
    assert root_option_eligible(
        _state(_hard_start_board()),
        sim,
        1536,
    )
    assert not root_option_eligible(
        _state(_merge_ready_board()),
        sim,
        1536,
    )


def test_decision_target_wrapper_rejects_unbounded_auxiliary_values() -> None:
    good = build_decision_targets(
        decision_move=0,
        terminal_move=40,
        terminal_status="censor",
        live_geometry_by_move={
            10: np.linspace(0.0, 1.0, 8, dtype=np.float32),
            20: np.linspace(1.0, 0.0, 8, dtype=np.float32),
            40: np.full(8, 0.5, dtype=np.float32),
        },
    )
    assert np.all((0.0 <= good.geometry) & (good.geometry <= 1.0))
    with pytest.raises(ValueError, match=r"escaped \[0,1\]"):
        build_decision_targets(
            decision_move=0,
            terminal_move=40,
            terminal_status="censor",
            live_geometry_by_move={
                10: np.asarray([2.0] + [0.0] * 7, dtype=np.float32),
                20: np.zeros(8, dtype=np.float32),
                40: np.zeros(8, dtype=np.float32),
            },
        )


def test_balanced_n192_power_contract_is_deterministic() -> None:
    assert POINT_GATE_ODDS_RATIO == 1.25
    assert stratum_counts() == {
        "T48:aligned": 32,
        "T48:unaligned": 32,
        "T96:aligned": 32,
        "T96:unaligned": 32,
        "T192:aligned": 32,
        "T192:unaligned": 32,
    }
    first = simulate_mechanism_power(
        1.50,
        draws=16,
        bootstrap_replicates=19,
        seed=123,
    )
    second = simulate_mechanism_power(
        1.50,
        draws=16,
        bootstrap_replicates=19,
        seed=123,
    )
    assert first == second
    table = power_table()
    assert table["passes"]
    assert table["selected_roots"] == 192
    assert table["grid_mde"] == 1.50
    or150 = next(
        row
        for row in table["rows"]
        if row["true_common_odds_ratio"] == 1.50
    )
    assert or150["power_full_gate"] == pytest.approx(0.912109375)
