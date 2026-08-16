from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from threes_rl.o3_designated_pair_option import (
    CHECKPOINTS,
    EVENT_CLASS_NAMES,
    INTEGRATED_TARGETS,
    O3DesignatedPairNet,
    OPTION_HORIZON,
    OUTPUT_WIDTH,
    TRAIN_TARGETS,
    VERSION,
    action_order_components,
    advance_lineage_base,
    balanced_valid_row_weight,
    build_decision_targets,
    canonical_json_hash,
    choose_option_action,
    initial_lineage,
    option_features,
    root_option_eligible,
    schema_manifest,
    schema_sha256,
    select_designated_pair,
)
from threes_rl.o3_power_contract import (
    POINT_GATE_ODDS_RATIO,
    mantel_haenszel_odds_ratio,
    simulate_mechanism_power,
    stratum_counts,
    target_counts,
)
from threes_rl.sim import LEFT, Preview, SimState, ThreesSim


EXPECTED_SCHEMA_SHA256 = (
    "a1c2efa6bd980d32138fb6026c1a5109685db8f1630e1b5fa732b2c2eb983602"
)


def _sim() -> ThreesSim:
    return ThreesSim.from_stream_ids(
        deck_stream_id=2_026_072_801,
        slot_stream_id=2_026_072_802,
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


def _wilson_lower(successes: int, total: int) -> float:
    z = 1.6448536269514722
    proportion = float(successes) / float(total)
    denominator = 1.0 + z * z / float(total)
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return center - margin


def test_schema_model_and_integrated_target_contract() -> None:
    assert VERSION == "o3_event_conditioned_designated_pair_v1"
    assert TRAIN_TARGETS == (48, 96, 192)
    assert INTEGRATED_TARGETS == (48, 96, 192, 384, 768)
    assert OPTION_HORIZON == 40
    assert CHECKPOINTS == (10, 20, 40)
    assert len(EVENT_CLASS_NAMES) == 5
    assert OUTPUT_WIDTH == 29
    assert sum(
        parameter.numel() for parameter in O3DesignatedPairNet().parameters()
    ) == 102_557
    manifest = schema_manifest()
    assert manifest["hard_start"] == (
        "canonical_pair_has_zero_safe_merge_actions"
    )
    assert manifest["incumbent_delegated_targets"] == "target>=1536"
    assert schema_sha256() == EXPECTED_SCHEMA_SHA256


def test_schema_hash_changes_with_contract_fields() -> None:
    baseline = schema_manifest()
    mutations = (
        ("version", "changed"),
        ("integrated_targets", [48, 96, 192, 384, 768, 1536]),
        ("hard_start", "allow_merge_ready"),
        ("decision_label_semantics", {"changed": True}),
        ("action_ranking", {"changed": True}),
        ("lifecycle", {"changed": True}),
    )
    for field, replacement in mutations:
        changed = copy.deepcopy(baseline)
        changed[field] = replacement
        assert canonical_json_hash(changed) != EXPECTED_SCHEMA_SHA256


def test_hard_start_rejects_merge_ready_and_accepts_nontrivial_pair() -> None:
    sim = _sim()
    merge_ready = _state(_merge_ready_board())
    hard_start = _state(_hard_start_board())
    ready_pair = select_designated_pair(
        merge_ready.board,
        1536,
        allowed_targets=TRAIN_TARGETS,
    )
    hard_pair = select_designated_pair(
        hard_start.board,
        1536,
        allowed_targets=TRAIN_TARGETS,
    )
    assert ready_pair is not None and ready_pair.safe_merge_actions
    assert hard_pair is not None and not hard_pair.safe_merge_actions
    assert not root_option_eligible(merge_ready, sim, 1536)
    assert root_option_eligible(hard_start, sim, 1536)


def test_1536_pair_is_never_an_integrated_option_target() -> None:
    board = np.asarray(
        [
            [0, 3, 6, 12],
            [1536, 24, 1536, 48],
            [0, 3072, 2, 768],
            [0, 96, 192, 384],
        ],
        dtype=np.int32,
    )
    assert select_designated_pair(
        board,
        None,
        allowed_targets=INTEGRATED_TARGETS,
    ) is None


def test_designated_lineage_merge_matches_exact_base_move() -> None:
    board = _merge_ready_board()
    pair = select_designated_pair(
        board,
        1536,
        requested_target=48,
        allowed_targets=TRAIN_TARGETS,
    )
    assert pair is not None and LEFT in pair.safe_merge_actions
    before = board.copy()
    moved = advance_lineage_base(board, initial_lineage(pair), LEFT)
    assert moved.event == "designated_pair_merged"
    assert int(np.count_nonzero(moved.lineage == 3)) == 1
    np.testing.assert_array_equal(board, before)


def test_option_features_are_finite_exact_shape_and_nonmutating() -> None:
    sim = _sim()
    state = _state(_hard_start_board())
    pair = select_designated_pair(
        state.board,
        1536,
        allowed_targets=TRAIN_TARGETS,
    )
    assert pair is not None
    lineage = initial_lineage(pair)
    state_before = state.board.copy()
    lineage_before = lineage.copy()
    action = min(sim.legal_actions(state))
    tokens, globals_array = option_features(
        state,
        sim,
        starter_tile=1536,
        pair=pair,
        lineage=lineage,
        action=action,
    )
    assert tokens.shape == (16, 37)
    assert globals_array.shape == (35,)
    assert np.isfinite(tokens).all()
    assert np.isfinite(globals_array).all()
    np.testing.assert_array_equal(state.board, state_before)
    np.testing.assert_array_equal(lineage, lineage_before)


def test_decision_targets_use_relative_offsets_and_true_censor_masks() -> None:
    geometry = {
        move: np.full(8, float(move), dtype=np.float32)
        for move in (10, 15, 20, 25, 40)
    }
    root_censor = build_decision_targets(
        decision_move=0,
        terminal_move=40,
        terminal_status="censor",
        live_geometry_by_move=geometry,
    )
    assert root_censor.event_mask and root_censor.event_class == 4
    assert root_censor.geometry_mask.tolist() == [True, True, True]

    late_censor = build_decision_targets(
        decision_move=5,
        terminal_move=40,
        terminal_status="censor",
        live_geometry_by_move=geometry,
    )
    assert not late_censor.event_mask and late_censor.event_class is None
    assert late_censor.geometry_mask.tolist() == [True, True, False]

    relative_success = build_decision_targets(
        decision_move=5,
        terminal_move=17,
        terminal_status="success",
        live_geometry_by_move=geometry,
    )
    assert relative_success.event_mask and relative_success.event_class == 1
    assert relative_success.geometry_mask.tolist() == [True, False, False]

    relative_failure = build_decision_targets(
        decision_move=11,
        terminal_move=12,
        terminal_status="failure",
        live_geometry_by_move={},
    )
    assert relative_failure.event_mask and relative_failure.event_class == 3
    assert not relative_failure.geometry_mask.any()


def test_balanced_row_weight_exactly_splits_family_root_trajectory_rows() -> None:
    assert balanced_valid_row_weight(
        represented_family_count=4,
        roots_in_family=24,
        trajectories_per_root=12,
        valid_rows_in_trajectory=8,
    ) == pytest.approx(1.0 / (4 * 24 * 12 * 8))
    with pytest.raises(ValueError, match="positive"):
        balanced_valid_row_weight(
            represented_family_count=4,
            roots_in_family=24,
            trajectories_per_root=12,
            valid_rows_in_trajectory=0,
        )


def test_action_ranking_and_safe_merge_override_are_deterministic() -> None:
    tied = np.zeros(OUTPUT_WIDTH, dtype=np.float64)
    assert choose_option_action(
        {3: tied.copy(), 1: tied.copy()},
        remaining_horizon=40,
    ) == 1
    favored = tied.copy()
    favored[0] = 3.0
    assert choose_option_action(
        {0: tied.copy(), 2: favored},
        remaining_horizon=40,
    ) == 2
    assert choose_option_action(
        {0: favored, 2: tied.copy()},
        remaining_horizon=40,
        safe_merge_actions=(2,),
    ) == 2
    components = action_order_components(favored, remaining_horizon=10)
    assert len(components) == 3 and all(math.isfinite(value) for value in components)


def test_weakest_target_sizing_and_resource_arithmetic() -> None:
    lower = _wilson_lower(5, 128)
    assert lower == pytest.approx(0.01914143013104029)
    assert 5_020 * lower == pytest.approx(96.08997925782226)
    assert 1_675 * lower == pytest.approx(32.061895469492484)
    assert 13_805 * lower == pytest.approx(264.2474429590112)
    assert 20_500 * lower == pytest.approx(392.39931768632596)

    nominal_hours = 20_500 * 9.195378486979166 / 3_600.0
    assert nominal_hours == pytest.approx(52.36257193974247)
    assert 2.5 * nominal_hours == pytest.approx(130.9064298493562)
    projected_bytes = math.ceil(
        1.25 * 20_500 * (1_000_401 + 65_536) + 512 * 1024**2
    )
    assert projected_bytes == 27_851_506_537
    assert projected_bytes < 28 * 1024**3


def test_power_contract_matches_frozen_allocations_and_pass_gate() -> None:
    assert POINT_GATE_ODDS_RATIO == 1.25
    assert target_counts(192) == (96, 58, 38)
    assert target_counts(264) == (132, 79, 53)
    assert stratum_counts(192) == {
        "T48:aligned": 48,
        "T48:unaligned": 48,
        "T96:aligned": 29,
        "T96:unaligned": 29,
        "T192:aligned": 19,
        "T192:unaligned": 19,
    }
    assert sum(stratum_counts(264).values()) == 264

    odds_ratio = mantel_haenszel_odds_ratio(
        [
            (
                np.asarray([12.0]),
                np.asarray([8.0]),
                np.asarray([8.0]),
                np.asarray([12.0]),
            )
        ]
    )
    assert odds_ratio.item() == pytest.approx(2.25)

    first = simulate_mechanism_power(
        192,
        1.50,
        draws=8,
        bootstrap_replicates=19,
        seed=123,
    )
    second = simulate_mechanism_power(
        192,
        1.50,
        draws=8,
        bootstrap_replicates=19,
        seed=123,
    )
    assert first == second
    assert first["point_gate_odds_ratio"] == 1.25
    assert 0.0 <= first["power_full_gate"] <= 1.0

    frozen_192 = simulate_mechanism_power(192, 1.50)
    frozen_264 = simulate_mechanism_power(264, 1.50)
    assert frozen_192["power_lower_ci_gt_1"] == 0.9267578125
    assert frozen_192["power_full_gate"] == 0.9169921875
    assert frozen_264["power_lower_ci_gt_1"] == 0.9765625
    assert frozen_264["power_full_gate"] == 0.953125
