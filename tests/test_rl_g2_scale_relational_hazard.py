from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from threes_rl.g2_scale_relational_hazard import (
    FEATURE_NAMES,
    FEATURE_WIDTH,
    _best_pair,
    canonical_orientation,
    feature_vector,
    schema_manifest,
    schema_sha256,
)
from threes_rl.g2_scale_relational_hazard_preflight import (
    PROPOSAL_PATH,
    PROPOSAL_SHA256,
    _compact_state,
    _partition_records,
    _root_summary,
    _write_immutable_json,
    representation_self_audit,
    sha256_path,
    simulate_power,
)
from threes_rl.sim import LEFT, UP, Preview, SimState, ThreesSim


EXPECTED_SCHEMA_SHA256 = (
    "6af0cd515e5886b5fd8bc4d9f52cc9202bd3ed1f149d0ae146829681aea8340e"
)


def _state() -> SimState:
    return SimState(
        board=np.asarray(
            [
                [1536, 384, 192, 24],
                [0, 384, 96, 12],
                [3, 48, 0, 6],
                [1, 0, 2, 0],
            ],
            dtype=np.int32,
        ),
        preview=Preview("bonus", None, (24, 48, 96)),
        small_counts={"red": 2, "blue": 3, "gray": 4},
        small_pos=3,
        small_seen_total=55,
        span_small_pos=7,
        large_pending=True,
        max_tile=1536,
        move_count=120,
        game_over=False,
    )


def _sim() -> ThreesSim:
    return ThreesSim.from_stream_ids(
        deck_stream_id=9_100_001,
        slot_stream_id=9_100_002,
        starter_tile=1536,
    )


def _scaled(state: SimState) -> SimState:
    board = state.board.copy()
    for row in range(4):
        for column in range(4):
            if (row, column) != (0, 0) and int(board[row, column]) >= 3:
                board[row, column] *= 2
    return SimState(
        board=board,
        preview=Preview("bonus", None, (48, 96, 192)),
        small_counts=state.small_counts.copy(),
        small_pos=state.small_pos,
        small_seen_total=state.small_seen_total,
        span_small_pos=state.span_small_pos,
        large_pending=state.large_pending,
        max_tile=state.max_tile,
        move_count=state.move_count,
        game_over=state.game_over,
    )


def test_proposal_and_schema_hash_are_frozen() -> None:
    assert sha256_path(PROPOSAL_PATH) == PROPOSAL_SHA256
    manifest = schema_manifest()
    assert manifest["width"] == FEATURE_WIDTH == 64
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES)) == 64
    assert schema_sha256() == EXPECTED_SCHEMA_SHA256


def test_schema_hash_changes_for_contract_mutations() -> None:
    baseline = schema_manifest()
    baseline_hash = hashlib.sha256(
        json.dumps(baseline, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    mutations = []
    for field, value in (
        ("version", "changed"),
        ("orientation", "changed"),
        ("pair_tie", "changed"),
        ("graph_connectivity", "changed"),
    ):
        payload = json.loads(json.dumps(baseline))
        payload[field] = value
        mutations.append(payload)
    for key, value in (
        ("name", "changed"),
        ("formula", "changed"),
        ("domain", "[0,2]"),
        ("train_standardize", not baseline["columns"][0]["train_standardize"]),
    ):
        payload = json.loads(json.dumps(baseline))
        payload["columns"][0][key] = value
        mutations.append(payload)
    payload = json.loads(json.dumps(baseline))
    payload["columns"][0], payload["columns"][1] = (
        payload["columns"][1],
        payload["columns"][0],
    )
    mutations.append(payload)
    assert all(
        hashlib.sha256(
            json.dumps(item, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        != baseline_hash
        for item in mutations
    )


def test_features_are_scale_and_orientation_equivariant_without_mutation() -> None:
    state = _state()
    sim = _sim()
    board_before = state.board.copy()
    counts_before = state.small_counts.copy()
    deck_before = json.dumps(sim.deck_rng.bit_generator.state, sort_keys=True)
    slot_before = json.dumps(sim.slot_rng.bit_generator.state, sort_keys=True)

    base = feature_vector(
        state,
        sim,
        LEFT,
        target=768,
        horizon=40,
        starter_tile=1536,
    )
    transposed_state = SimState(
        board=state.board.T.copy(),
        preview=state.preview,
        small_counts=state.small_counts.copy(),
        small_pos=state.small_pos,
        small_seen_total=state.small_seen_total,
        span_small_pos=state.span_small_pos,
        large_pending=state.large_pending,
        max_tile=state.max_tile,
        move_count=state.move_count,
        game_over=state.game_over,
    )
    transposed = feature_vector(
        transposed_state,
        sim,
        UP,
        target=768,
        horizon=40,
        starter_tile=1536,
    )
    scaled = feature_vector(
        _scaled(state),
        sim,
        LEFT,
        target=1536,
        horizon=40,
        starter_tile=1536,
    )

    assert base.shape == (64,)
    assert np.all(np.isfinite(base))
    assert np.all((base >= 0.0) & (base <= 1.0))
    np.testing.assert_array_equal(base, transposed)
    np.testing.assert_array_equal(base, scaled)
    np.testing.assert_array_equal(state.board, board_before)
    assert state.small_counts == counts_before
    assert json.dumps(sim.deck_rng.bit_generator.state, sort_keys=True) == deck_before
    assert json.dumps(sim.slot_rng.bit_generator.state, sort_keys=True) == slot_before


def test_canonical_orientation_tie_uses_identity() -> None:
    board = np.asarray(
        [
            [1536, 384, 0, 0],
            [384, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int32,
    )
    canonical, action, transposed = canonical_orientation(
        board,
        UP,
        768,
        1536,
    )
    assert not transposed
    assert action == UP
    np.testing.assert_array_equal(canonical, board)


def test_best_pair_uses_frozen_lexicographic_tie_break() -> None:
    assert _best_pair([]) is None
    assert _best_pair([(1, 1)]) is None
    assert _best_pair([(1, 0), (1, 1), (0, 1)]) == ((0, 1), (1, 1))


def test_missing_pair_and_graph_features_are_finite() -> None:
    state = _state()
    state.board[state.board == 384] = 192
    vector = feature_vector(
        state,
        _sim(),
        LEFT,
        target=768,
        horizon=20,
        starter_tile=1536,
    )
    by_name = dict(zip(FEATURE_NAMES, vector.tolist()))
    assert by_name["parent_pair_exists"] == 0.0
    assert by_name["parent_pair_chebyshev"] == 1.0
    assert by_name["parent_pair_manhattan"] == 1.0
    assert by_name["support_graph_node_fraction"] > 0.0
    assert np.all(np.isfinite(vector))


def test_horizon_and_action_indicators_are_exact_one_hot() -> None:
    vector = feature_vector(
        _state(),
        _sim(),
        LEFT,
        target=768,
        horizon=10,
        starter_tile=1536,
    )
    assert sum(vector[:3]) == 1.0
    assert sum(vector[3:7]) == 1.0


def test_compact_state_excludes_score_and_recorded_action() -> None:
    payload = {
        "board": [[0] * 4 for _ in range(4)],
        "preview": {"kind": "blue", "value": 1, "candidates": []},
        "tile_cycle": {},
        "move_count": 1,
        "game_over": False,
        "legal_actions": ["left"],
        "legal_mask": [False, False, True, False],
        "score": 123,
        "action": "left",
        "move": "left",
    }
    compact = _compact_state(payload)
    assert "score" not in compact
    assert "action" not in compact
    assert "move" not in compact


def _record(root: str, family: str, scale: str) -> dict[str, object]:
    target = {
        "pre768": 768,
        "pre1536": 1536,
        "pre3072_transfer": 3072,
    }[scale]
    return {
        "record_id": f"{root}-{scale}",
        "root_cluster": root,
        "behavior_family": family,
        "scale": scale,
        "target": target,
    }


def test_partition_keeps_roots_whole_and_transfer_untouched() -> None:
    records = [
        _record("fresh:1:1536", "family_a", "pre768"),
        _record("fresh:1:1536", "family_a", "pre3072_transfer"),
        _record("fresh:2:1536", "family_a", "pre1536"),
        _record("fresh:2:1536", "family_a", "pre3072_transfer"),
        _record("fresh:3:1536", "family_b", "pre768"),
    ]
    historical = {
        "sets": {
            "S3_historical_exclusion_union": {"fresh:2:1536"},
            "S3_sealed_surviving_roots": set(),
            "A2_selected_or_labeled_roots": set(),
            "QD5_sealed_pilot_roots": set(),
        }
    }
    partitioned, audit = _partition_records(records, historical)
    rows = {(row["root_cluster"], row["scale"]): row for row in partitioned}
    assert rows[("fresh:1:1536", "pre3072_transfer")]["partition"] == (
        "untouched_transfer"
    )
    assert rows[("fresh:1:1536", "pre768")]["partition"] == (
        "withheld_earlier_from_transfer"
    )
    assert rows[("fresh:2:1536", "pre3072_transfer")]["partition"] == (
        "diagnostic_prior_overlap_transfer"
    )
    assert rows[("fresh:2:1536", "pre1536")]["partition"] in {
        "train",
        "development",
    }
    assert not audit["cross_partition_roots"]


def test_family_balanced_summary_caps_effective_share() -> None:
    rows = []
    for index in range(8):
        rows.append(
            {
                **_record(f"fresh:{index}:1536", "large", "pre768"),
                "partition": "train",
            }
        )
    for index in range(2):
        rows.append(
            {
                **_record(f"fresh:{100+index}:1536", "small", "pre1536"),
                "partition": "train",
            }
        )
    summary = _root_summary(rows, "train")
    assert summary["raw_max_family_share"] == 0.8
    assert summary["effective_family_shares"] == {"large": 0.5, "small": 0.5}
    assert summary["effective_ancestry_count"] < summary["unique_roots"]


def test_representation_self_audit_passes() -> None:
    audit = representation_self_audit()
    assert audit["passes"]
    assert all(audit["checks"].values())
    assert audit["schema_sha256"] == EXPECTED_SCHEMA_SHA256


def test_power_simulation_is_deterministic_and_outcome_free() -> None:
    first = simulate_power(96, 1.75, draws=200)
    second = simulate_power(96, 1.75, draws=200)
    assert first == second
    assert first["roots"] == 96
    assert first["active_roots"] == 28
    assert 0.0 <= first["power_pass_point_or_1_25_and_ci"] <= 1.0


def test_immutable_writer_rejects_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "sealed.json"
    _write_immutable_json(path, {"decision": "HOLD"})
    with pytest.raises(FileExistsError):
        _write_immutable_json(path, {"decision": "READY"})
