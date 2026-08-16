from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from threes_rl.g1_relational_hazard import (
    FEATURE_WIDTH,
    MODEL_PARAMETER_COUNT,
    G1LogisticModel,
    _best_pair,
    _blockers,
    _chebyshev,
    _manhattan,
    _minimum_distance,
    combined_schema_manifest,
    encode_action_afterstate,
    family_balanced_action_weights,
    feature_names,
    implementation_sha256,
    payload_sha256,
    root_equal_action_weights,
    schema_payload,
    schema_sha256,
)
from threes_rl.sim import ThreesSim, simulate_base_move


EXPECTED_RELATIONAL_SCHEMA_SHA256 = (
    "930f68e2fc0c45e635a1a223ff6bd69e3f5a4bc1836dd017e154e70ac2a87e20"
)
EXPECTED_POSITIONAL_SCHEMA_SHA256 = (
    "305ccc536a117f94f6b0af6b7b66e2b64b6bea1b3a355d934d8a001e2ae1c355"
)


def _state_with_board(sim: ThreesSim, board: list[list[int]]):
    state = sim.reset()
    state.board = np.asarray(board, dtype=np.int32)
    state.max_tile = int(np.max(state.board))
    state.game_over = False
    return state


def _assert_state_equal(left, right) -> None:
    np.testing.assert_array_equal(left.board, right.board)
    assert left.preview == right.preview
    assert left.small_counts == right.small_counts
    assert left.small_pos == right.small_pos
    assert left.small_seen_total == right.small_seen_total
    assert left.span_small_pos == right.span_small_pos
    assert left.large_pending == right.large_pending
    assert left.max_tile == right.max_tile
    assert left.move_count == right.move_count
    assert left.game_over == right.game_over


def _feature(features: np.ndarray, mode: str, name: str) -> float:
    return float(features[feature_names(mode).index(name)])


def test_schemas_are_unique_and_capacity_matched() -> None:
    relational = feature_names("relational")
    positional = feature_names("positional")
    assert len(relational) == FEATURE_WIDTH == len(set(relational))
    assert len(positional) == FEATURE_WIDTH == len(set(positional))
    assert MODEL_PARAMETER_COUNT == 65
    assert schema_sha256("relational") == EXPECTED_RELATIONAL_SCHEMA_SHA256
    assert schema_sha256("positional") == EXPECTED_POSITIONAL_SCHEMA_SHA256
    for mode in ("relational", "positional"):
        payload = schema_payload(mode)
        assert [row["index"] for row in payload["features"]] == list(range(64))
        assert [row["name"] for row in payload["features"]] == list(
            feature_names(mode)
        )
        assert all(row["formula"] for row in payload["features"])
    combined = combined_schema_manifest()
    assert combined["shared_feature_count"] == 24
    assert combined["shared_formulas_identical"] is True
    assert (
        combined["relational"]["schema"]["column_partition"]["shared_names"]
        == combined["positional"]["schema"]["column_partition"]["shared_names"]
    )
    assert combined["implementation_sha256"] == implementation_sha256()


def test_schema_hash_covers_order_formula_and_normalization() -> None:
    original = schema_payload("relational")
    original_hash = payload_sha256(original)
    for mutation in ("name", "formula", "normalize_train", "order"):
        payload = copy.deepcopy(original)
        if mutation == "name":
            payload["features"][0]["name"] = "changed"
        elif mutation == "formula":
            payload["features"][0]["formula"] = "changed"
        elif mutation == "normalize_train":
            payload["features"][0]["normalize_train"] = not payload["features"][0][
                "normalize_train"
            ]
        else:
            payload["features"][0], payload["features"][1] = (
                payload["features"][1],
                payload["features"][0],
            )
        assert payload_sha256(payload) != original_hash


def test_schema_contains_no_provenance_or_outcome_inputs() -> None:
    forbidden = (
        "source",
        "frame",
        "seed",
        "wall_clock",
        "future_action",
        "future_outcome",
        "family",
        "source_role",
        "final_score",
    )
    for mode in ("relational", "positional"):
        text = json.dumps(schema_payload(mode), sort_keys=True).lower()
        assert not any(token in text for token in forbidden)


def test_shared_features_match_for_every_legal_action_and_context_is_immutable() -> None:
    sim = ThreesSim.from_stream_ids(
        deck_stream_id=101,
        slot_stream_id=202,
        starter_tile=1536,
    )
    cases = (
        (
            "pre1536",
            [
                [1536, 768, 384, 192],
                [96, 48, 24, 12],
                [6, 3, 2, 1],
                [0, 0, 0, 0],
            ],
        ),
        (
            "pre3072",
            [
                [1536, 1536, 768, 384],
                [192, 96, 48, 24],
                [12, 6, 3, 2],
                [1, 0, 0, 0],
            ],
        ),
    )
    for stratum, board in cases:
        state = _state_with_board(sim, board)
        before = copy.deepcopy(state)
        for action in sim.legal_actions(state):
            relational = encode_action_afterstate(
                state,
                sim,
                action=action,
                stratum=stratum,
                mode="relational",
            )
            positional = encode_action_afterstate(
                state,
                sim,
                action=action,
                stratum=stratum,
                mode="positional",
            )
            np.testing.assert_array_equal(relational[:24], positional[:24])
            assert relational.shape == positional.shape == (FEATURE_WIDTH,)
            assert np.all(np.isfinite(relational))
            assert np.all(np.isfinite(positional))
            assert np.all((relational >= 0.0) & (relational <= 1.0))
            assert np.all((positional >= 0.0) & (positional <= 1.0))
        _assert_state_equal(state, before)


def test_pair_and_missing_conventions_on_formula_edge_states() -> None:
    positions = [(0, 1), (1, 0), (1, 2)]
    assert _best_pair(positions) == ((0, 1), (1, 0))
    assert _minimum_distance([], positions, _manhattan, missing=6) == 6
    assert _minimum_distance([], positions, _chebyshev, missing=3) == 3

    blocked = np.zeros((4, 4), dtype=np.int32)
    blocked[0, 0] = 768
    blocked[0, 2] = 192
    blocked[0, 3] = 768
    assert _blockers(blocked, ((0, 0), (0, 3))) == (1, False)
    blocked[0, 2] = 0
    assert _blockers(blocked, ((0, 0), (0, 3))) == (0, True)

    sim = ThreesSim.from_stream_ids(
        deck_stream_id=505,
        slot_stream_id=606,
        starter_tile=1536,
    )
    no_target = _state_with_board(
        sim,
        [
            [1536, 384, 192, 96],
            [48, 24, 12, 6],
            [3, 2, 1, 0],
            [0, 0, 0, 0],
        ],
    )
    action = sim.legal_actions(no_target)[0]
    no_target_features = encode_action_afterstate(
        no_target,
        sim,
        action=action,
        stratum="pre1536",
        mode="relational",
    )
    assert _feature(no_target_features, "relational", "target_count") == 0.0
    assert (
        _feature(no_target_features, "relational", "target_pair_manhattan")
        == 1.0
    )
    assert (
        _feature(no_target_features, "relational", "target_pair_chebyshev")
        == 1.0
    )

    one_target_no_support = _state_with_board(
        sim,
        [
            [1536, 768, 96, 48],
            [24, 12, 6, 3],
            [2, 1, 0, 0],
            [0, 0, 0, 0],
        ],
    )
    action = sim.legal_actions(one_target_no_support)[0]
    features = encode_action_afterstate(
        one_target_no_support,
        sim,
        action=action,
        stratum="pre1536",
        mode="relational",
    )
    assert _feature(features, "relational", "target_duplicate_exists") == 0.0
    assert _feature(features, "relational", "support_min_manhattan") == 1.0
    assert _feature(features, "relational", "support_min_chebyshev") == 1.0


def test_insertion_slot_geometry_matches_forced_base_move() -> None:
    sim = ThreesSim.from_stream_ids(
        deck_stream_id=707,
        slot_stream_id=808,
        starter_tile=1536,
    )
    state = _state_with_board(
        sim,
        [
            [1536, 768, 384, 0],
            [96, 48, 24, 0],
            [12, 6, 3, 0],
            [2, 1, 0, 0],
        ],
    )
    action = 3
    assert action in sim.legal_actions(state)
    after, slots = simulate_base_move(state.board, action)
    targets = [
        (int(row), int(column))
        for row, column in np.argwhere(after == 768)
    ]
    supports = [
        (int(row), int(column))
        for value in (384, 192)
        for row, column in np.argwhere(after == value)
    ]
    features = encode_action_afterstate(
        state,
        sim,
        action=action,
        stratum="pre1536",
        mode="relational",
    )
    expected_target = min(
        (_manhattan(slot, target) for slot in slots for target in targets),
        default=6,
    ) / 6.0
    expected_support = min(
        (_manhattan(slot, support) for slot in slots for support in supports),
        default=6,
    ) / 6.0
    expected_starter = min(_manhattan(slot, (0, 0)) for slot in slots) / 6.0
    expected_axis = sum(
        any(
            slot[0] == target[0] or slot[1] == target[1]
            for target in targets
        )
        for slot in slots
    ) / len(slots)
    assert _feature(
        features, "relational", "insertion_min_target_manhattan"
    ) == expected_target
    assert _feature(
        features, "relational", "insertion_min_support_manhattan"
    ) == expected_support
    assert _feature(
        features, "relational", "insertion_min_starter_manhattan"
    ) == expected_starter
    assert _feature(
        features, "relational", "insertion_target_axis_alignment"
    ) == expected_axis


def test_illegal_action_is_rejected() -> None:
    sim = ThreesSim.from_stream_ids(
        deck_stream_id=303,
        slot_stream_id=404,
        starter_tile=1536,
    )
    state = sim.reset()
    state.board.fill(0)
    state.board[0, 0] = 1536
    state.max_tile = 1536
    illegal = next(
        action for action in range(4) if action not in sim.legal_actions(state)
    )
    with pytest.raises(ValueError, match="not legal"):
        encode_action_afterstate(
            state,
            sim,
            action=illegal,
            stratum="pre1536",
            mode="relational",
        )


def test_zero_model_is_capacity_matched_and_round_trips(tmp_path) -> None:
    features = np.linspace(0.0, 1.0, FEATURE_WIDTH)
    for mode in ("relational", "positional"):
        model = G1LogisticModel.zero(mode)
        assert model.parameter_count == MODEL_PARAMETER_COUNT
        assert model.predict_probability(features) == 0.5
        directory = tmp_path / mode
        model.save(directory)
        loaded = G1LogisticModel.load(directory, expected_mode=mode)
        assert loaded.predict_probability(features) == 0.5
        np.testing.assert_array_equal(loaded.weights, model.weights)


def test_model_load_rejects_incompatible_schema(tmp_path) -> None:
    directory = tmp_path / "model"
    G1LogisticModel.zero("relational").save(directory)
    metadata = json.loads((directory / "meta.json").read_text())
    metadata["schema_sha256"] = "not-the-frozen-schema"
    (directory / "meta.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="feature schema"):
        G1LogisticModel.load(directory, expected_mode="relational")


@pytest.mark.parametrize(
    ("metadata_key", "metadata_value", "error"),
    (
        ("feature_width", 63, "feature width"),
        ("parameter_count", 64, "parameter count"),
        ("intercept", float("nan"), "Nonfinite"),
    ),
)
def test_model_load_rejects_invalid_metadata(
    tmp_path,
    metadata_key,
    metadata_value,
    error,
) -> None:
    directory = tmp_path / metadata_key
    G1LogisticModel.zero("relational").save(directory)
    metadata = json.loads((directory / "meta.json").read_text())
    metadata[metadata_key] = metadata_value
    (directory / "meta.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=error):
        G1LogisticModel.load(directory, expected_mode="relational")


@pytest.mark.parametrize(
    ("array_name", "array_value", "error"),
    (
        ("weights", float("nan"), "Nonfinite"),
        ("feature_mean", float("inf"), "Nonfinite"),
        ("feature_scale", 0.0, "feature scale"),
        ("feature_scale", -1.0, "feature scale"),
    ),
)
def test_model_load_rejects_invalid_arrays(
    tmp_path,
    array_name,
    array_value,
    error,
) -> None:
    directory = tmp_path / f"{array_name}_{array_value}"
    G1LogisticModel.zero("relational").save(directory)
    with np.load(directory / "arrays.npz") as payload:
        arrays = {name: payload[name].copy() for name in payload.files}
    arrays[array_name][0] = array_value
    np.savez_compressed(directory / "arrays.npz", **arrays)
    with pytest.raises(ValueError, match=error):
        G1LogisticModel.load(directory, expected_mode="relational")


def test_root_equal_weights_ignore_legal_action_count() -> None:
    rows = root_equal_action_weights([2, 3, 4])
    np.testing.assert_allclose([np.sum(row) for row in rows], [1 / 3] * 3)
    assert np.isclose(sum(float(np.sum(row)) for row in rows), 1.0)


def test_family_balanced_weights_match_charter_formula() -> None:
    families = ["a", "a", "b"]
    rows = family_balanced_action_weights(families, [2, 4, 3])
    np.testing.assert_allclose(
        [np.sum(row) for row in rows],
        [0.25, 0.25, 0.5],
    )
    assert np.isclose(sum(float(np.sum(row)) for row in rows), 1.0)
