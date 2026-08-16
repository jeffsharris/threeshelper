from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from threes_rl import o1_p0_preflight as p0
from threes_rl.o1_geometry_option import (
    O1OptionNet,
    VERSION,
    geometry,
    option_features,
    option_status,
    pair_safe_merge_actions,
    schema_manifest,
    schema_sha256,
    tagged_base_move,
)
from threes_rl.sim import (
    DIRECTION_NAMES,
    LEFT,
    Preview,
    SimState,
    ThreesSim,
    simulate_base_move,
)


EXPECTED_SCHEMA_SHA256 = (
    "55dd298ea2bf40a24d8af641d852d5f9c09aff14b1b736a29e6b5a071563772c"
)


def _sim() -> ThreesSim:
    return ThreesSim.from_stream_ids(
        deck_stream_id=2_026_072_611,
        slot_stream_id=2_026_072_612,
        starter_tile=1536,
    )


def _board_with_prespawn_empties(empty_count: int) -> np.ndarray:
    if empty_count == 2:
        return np.asarray(
            [
                [1536, 3, 6, 12],
                [48, 48, 24, 6144],
                [1, 3072, 2, 768],
                [0, 96, 192, 384],
            ],
            dtype=np.int32,
        )
    if empty_count == 3:
        return np.asarray(
            [
                [1536, 3, 6, 12],
                [48, 48, 24, 6144],
                [0, 3072, 2, 768],
                [0, 96, 192, 384],
            ],
            dtype=np.int32,
        )
    raise ValueError("Fixture supports only two or three pre-spawn empties")


def _state(
    board: np.ndarray | None = None,
    *,
    game_over: bool = False,
) -> SimState:
    value = (
        _board_with_prespawn_empties(3)
        if board is None
        else np.asarray(board, dtype=np.int32)
    )
    return SimState(
        board=value.copy(),
        preview=Preview("bonus", None, (24, 48, 96)),
        small_counts={"red": 2, "blue": 3, "gray": 4},
        small_pos=3,
        small_seen_total=55,
        span_small_pos=7,
        large_pending=True,
        max_tile=int(np.max(value)),
        move_count=120,
        game_over=game_over,
    )


def _payload(state: SimState, sim: ThreesSim) -> dict:
    legal = sim.legal_actions(state)
    return {
        "board": state.board.tolist(),
        "preview": {
            "kind": state.preview.kind,
            "value": state.preview.value,
            "candidates": list(state.preview.candidates),
        },
        "tile_cycle": {
            "small_counts": state.small_counts,
            "small_pos": state.small_pos,
            "small_seen_total": state.small_seen_total,
            "span_small_pos": state.span_small_pos,
            "large_pending": state.large_pending,
            "max_tile": state.max_tile,
        },
        "move_count": state.move_count,
        "game_over": state.game_over,
        "legal_actions": [DIRECTION_NAMES[action] for action in legal],
        "legal_mask": [action in legal for action in range(4)],
        "max_tile": state.max_tile,
    }


def test_a3_schema_and_model_are_exactly_frozen() -> None:
    assert VERSION == "o1_goal_conditioned_geometry_option_v1_a3"
    manifest = schema_manifest()
    assert manifest["minimum_safe_empties"] == 2
    assert manifest["minimum_safe_prespawn_empties"] == 3
    assert manifest["pair_specific_tagged_merge"]
    assert manifest["action_conditioned_forward"]
    assert schema_sha256() == EXPECTED_SCHEMA_SHA256
    assert sum(parameter.numel() for parameter in O1OptionNet().parameters()) == 113_780
    assert p0.sha256_path(p0.A4_PATH) == p0.A4_SHA256
    assert p0.sha256_path(p0.A5_PATH) == p0.A5_SHA256
    assert p0.sha256_path(p0.A6_PATH) == p0.A6_SHA256
    assert p0.sha256_path(p0.OLD_TEST_EVIDENCE_PATH) == (
        p0.OLD_TEST_EVIDENCE_FILE_SHA256
    )


def test_schema_hash_changes_with_a3_contract_fields() -> None:
    baseline = schema_manifest()
    for field, replacement in (
        ("version", "changed"),
        ("minimum_safe_empties", 3),
        ("minimum_safe_prespawn_empties", 2),
        ("pair_specific_tagged_merge", False),
        ("action_conditioned_forward", False),
    ):
        changed = copy.deepcopy(baseline)
        changed[field] = replacement
        assert p0.canonical_json_hash(changed) != EXPECTED_SCHEMA_SHA256


def test_a6_evidence_transition_binds_old_and_new_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_path = tmp_path / "old.json"
    new_path = tmp_path / "new.json"
    transition_path = tmp_path / "transition.json"
    runner_path = tmp_path / "runner.py"
    tests_path = tmp_path / "tests.py"
    a6_path = tmp_path / "a6.md"
    runner_path.write_text("runner")
    tests_path.write_text("tests")
    a6_path.write_text("a6")
    p0.write_immutable_json(old_path, {"version": "old"})
    p0.write_immutable_json(new_path, {"version": "new"})
    old_payload = json.loads(old_path.read_text())
    monkeypatch.setattr(p0, "OLD_TEST_EVIDENCE_PATH", old_path)
    monkeypatch.setattr(p0, "TEST_EVIDENCE_PATH", new_path)
    monkeypatch.setattr(p0, "EVIDENCE_TRANSITION_PATH", transition_path)
    monkeypatch.setattr(p0, "RUNNER_PATH", runner_path)
    monkeypatch.setattr(p0, "TEST_PATH", tests_path)
    monkeypatch.setattr(p0, "A6_PATH", a6_path)
    monkeypatch.setattr(p0, "A6_SHA256", p0.sha256_path(a6_path))
    monkeypatch.setattr(
        p0,
        "OLD_TEST_EVIDENCE_FILE_SHA256",
        p0.sha256_path(old_path),
    )
    monkeypatch.setattr(
        p0,
        "OLD_TEST_EVIDENCE_PAYLOAD_SHA256",
        old_payload["canonical_payload_sha256"],
    )
    p0.write_immutable_json(
        transition_path,
        {
            "version": "o1_p0_a6_evidence_transition_v1",
            "a6_amendment_sha256": p0.A6_SHA256,
            "runner_sha256": p0.sha256_path(runner_path),
            "tests_sha256": p0.sha256_path(tests_path),
            "new_test_evidence": p0.artifact_identity(new_path),
            "old_test_evidence": p0.artifact_identity(old_path),
        },
    )
    report = p0.verify_evidence_transition()
    assert report["passes"]
    assert all(report["checks"].values())


def test_tagged_move_matches_simulator_and_never_mutates() -> None:
    rng = np.random.default_rng(2_026_072_601)
    tiles = np.asarray((0, 1, 2, 3, 6, 12, 24, 48, 96), dtype=np.int32)
    for _ in range(40):
        board = rng.choice(tiles, size=(4, 4)).astype(np.int32)
        before = board.copy()
        for action in range(4):
            tagged, eligible, _tags = tagged_base_move(board, action)
            expected, expected_eligible = simulate_base_move(board, action)
            np.testing.assert_array_equal(tagged, expected)
            assert eligible == expected_eligible
        np.testing.assert_array_equal(board, before)


def test_pair_specific_merge_does_not_credit_another_pair() -> None:
    board = np.asarray(
        [
            [1536, 3, 6, 12],
            [48, 48, 24, 96],
            [48, 192, 384, 768],
            [0, 0, 3072, 6144],
        ],
        dtype=np.int32,
    )
    selected_pair = ((1, 0), (2, 0))
    actions = pair_safe_merge_actions(board, selected_pair, 48, 1536)
    assert LEFT not in actions
    shifted, _eligible, tags = tagged_base_move(board, LEFT)
    assert any(int(shifted[coord]) == 96 for coord in tags)
    assert not any(
        frozenset(selected_pair).issubset(provenance)
        for provenance in tags.values()
    )


def test_merge_ready_requires_three_prespawn_empties() -> None:
    two_empty = _board_with_prespawn_empties(2)
    three_empty = _board_with_prespawn_empties(3)
    shifted_two, _, _ = tagged_base_move(two_empty, LEFT)
    shifted_three, _, _ = tagged_base_move(three_empty, LEFT)
    assert int(np.count_nonzero(shifted_two == 0)) == 2
    assert int(np.count_nonzero(shifted_three == 0)) == 3

    pair_two = geometry(two_empty, 1536)
    pair_three = geometry(three_empty, 1536)
    assert pair_two is not None and pair_two.stage == 2
    assert pair_two.safe_merge_actions == ()
    assert pair_three is not None and pair_three.stage == 3
    assert LEFT in pair_three.safe_merge_actions


def test_safe_real_merge_succeeds_for_every_requested_goal() -> None:
    sim = _sim()
    root = _board_with_prespawn_empties(3)
    pair = geometry(root, 1536)
    assert pair is not None
    root_double_count = int(np.count_nonzero(root == 96))
    merged, _eligible, _tags = tagged_base_move(root, LEFT)
    merged[1, 3] = 3
    state = _state(merged)
    assert int(np.count_nonzero(state.board == 0)) == 2
    for goal in (1, 2, 3, 4):
        assert option_status(
            state,
            sim,
            starter_tile=1536,
            target=48,
            requested_goal=goal,
            root_double_count=root_double_count,
        ) == "success"

    unsafe = _state(merged)
    unsafe.board[2, 3] = 6
    assert int(np.count_nonzero(unsafe.board == 0)) == 1
    assert option_status(
        unsafe,
        sim,
        starter_tile=1536,
        target=48,
        requested_goal=1,
        root_double_count=root_double_count,
    ) == "failure"


def test_action_conditioned_features_are_finite_and_nonmutating() -> None:
    sim = _sim()
    state = _state()
    pair = geometry(state.board, 1536)
    assert pair is not None
    board_before = state.board.copy()
    cycle_before = sim.tile_cycle_snapshot(state)
    features = {}
    for action in sim.legal_actions(state):
        spatial, global_values = option_features(
            state,
            sim,
            starter_tile=1536,
            pair_geometry=pair,
            requested_goal=4,
            action=action,
        )
        assert spatial.shape == (16, 4, 4)
        assert global_values.shape == (28,)
        assert np.isfinite(spatial).all()
        assert np.isfinite(global_values).all()
        features[action] = (spatial, global_values)
    assert len(features) >= 2
    assert len({tuple(value[1][-4:]) for value in features.values()}) == len(features)
    np.testing.assert_array_equal(state.board, board_before)
    assert sim.tile_cycle_snapshot(state) == cycle_before


def test_model_save_load_is_deterministic(tmp_path: Path) -> None:
    torch.manual_seed(2_026_072_602)
    model = O1OptionNet().eval()
    spatial = torch.zeros((2, 16, 4, 4), dtype=torch.float32)
    global_values = torch.zeros((2, 28), dtype=torch.float32)
    expected = model(spatial, global_values).detach()
    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)
    restored = O1OptionNet().eval()
    restored.load_state_dict(torch.load(path, weights_only=True))
    actual = restored(spatial, global_values).detach()
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_family_classifier_collapses_aliases() -> None:
    assert p0.family_classifier(
        {"policy": "corner2"}, Path("fresh/replay.json")
    ) == "g1r_corner2"
    assert p0.family_classifier(
        {"policy": "expectimax2"}, Path("fresh/replay.json")
    ) == "g1r_expectimax2"
    assert p0.family_classifier(
        {"policy": "phaseblend"}, Path("fresh/replay.json")
    ) == "g1r_parent_mc1000"
    assert p0.family_classifier(
        {"policy": "replaycal"}, Path("fresh/replay.json")
    ) == "g1r_replaycal"


def test_trajectory_streams_share_crn_and_are_unique() -> None:
    treatment = p0.trajectory_stream_ids(
        "development", 7, replicate=3, incumbent_arm=False
    )
    incumbent = p0.trajectory_stream_ids(
        "development", 7, replicate=3, incumbent_arm=True
    )
    for key in ("logical_seed", "deck_stream_id", "slot_stream_id"):
        assert treatment[key] == incumbent[key]
    assert treatment["policy_stream_id"] != incumbent["policy_stream_id"]
    rows = [
        p0.trajectory_stream_ids(
            "train",
            root,
            round_index=round_index,
            replicate=replicate,
        )
        for root in range(3)
        for round_index in range(4)
        for replicate in range(2)
    ]
    assert len({row["trajectory_code"] for row in rows}) == len(rows)


def test_internal_stream_contract_accepts_only_frozen_pair_coupling() -> None:
    rows = p0._requested_stream_rows(max_test_n=12)
    report = p0._internal_stream_contract(rows)
    assert report["passes"]
    assert report["policy_stream_ids_unique"]
    assert report["cross_namespace_disjoint"]
    expected_paired_codes = (80 + 12) * 8
    assert report["expected_shared_crn_duplicate_counts"] == {
        "logical_seed": expected_paired_codes,
        "deck_stream_id": expected_paired_codes,
        "slot_stream_id": expected_paired_codes,
    }

    missing_arm = copy.deepcopy(rows)
    first_paired_code = next(
        row["trajectory_code"]
        for row in missing_arm
        if row["trajectory_code"] >= 1_000_000
    )
    removed = False
    kept = []
    for row in missing_arm:
        if row["trajectory_code"] == first_paired_code and not removed:
            removed = True
            continue
        kept.append(row)
    assert not p0._internal_stream_contract(kept)["passes"]

    policy_duplicate = copy.deepcopy(rows)
    policy_duplicate[1]["policy_stream_id"] = policy_duplicate[0][
        "policy_stream_id"
    ]
    assert not p0._internal_stream_contract(policy_duplicate)["passes"]

    cross_code_duplicate = copy.deepcopy(rows)
    cross_code_duplicate[1]["logical_seed"] = cross_code_duplicate[0][
        "logical_seed"
    ]
    assert not p0._internal_stream_contract(cross_code_duplicate)["passes"]


def test_source_inventory_hashes_but_does_not_parse_candidate_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_path = tmp_path / "family" / "game" / "replay.json"
    replay_path.parent.mkdir(parents=True)
    replay_path.write_text("{not-json")
    monkeypatch.setattr(p0, "ROOT", tmp_path)
    inventory = p0.source_path_inventory()
    assert inventory["candidate_json_parsed"] is False
    assert inventory["rows"][0]["sha256"] == hashlib.sha256(
        b"{not-json"
    ).hexdigest()


def test_invalid_unselected_source_is_support_loss_not_selected_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sim = _sim()
    valid_state = _state()
    terminal_state = _state(game_over=True)
    valid_path = tmp_path / "valid" / "replay.json"
    invalid_path = tmp_path / "invalid" / "replay.json"
    valid_path.parent.mkdir(parents=True)
    invalid_path.parent.mkdir(parents=True)
    valid_path.write_text(
        json.dumps(
            {
                "starter_tile": 1536,
                "frames": [
                    {"index": 0, "state": _payload(valid_state, sim)},
                    {"index": 1, "state": _payload(terminal_state, sim)},
                ],
            }
        )
    )
    invalid_path.write_text(
        json.dumps(
            {
                "starter_tile": 1536,
                "frames": [
                    {"index": 0, "state": _payload(valid_state, sim)},
                    {"index": 1, "state": "malformed"},
                    {"index": 2, "state": _payload(terminal_state, sim)},
                ],
            }
        )
    )
    monkeypatch.setattr(
        p0,
        "replay_provenance",
        lambda replay, path: {
            "replay_origin": "fresh",
            "root_origin": "fresh",
            "replay_reset_invariant": True,
            "root_seed": 7,
        },
    )
    monkeypatch.setattr(
        p0,
        "canonical_ancestry_id",
        lambda replay, path: f"fresh:{path.parent.name}:1536",
    )
    monkeypatch.setattr(
        p0,
        "family_classifier",
        lambda replay, path: "g1r_corner2",
    )
    inventory = {
        "rows": [
            {
                "path": str(path),
                "sha256": p0.sha256_path(path),
                "eligible_for_content_scan": True,
            }
            for path in (valid_path, invalid_path)
        ]
    }
    records, report = p0.scan_candidate_content(
        inventory,
        {"excluded_roots": []},
    )
    assert [row["root_cluster"] for row in records] == ["fresh:valid:1536"]
    assert report["data_hygiene_failure_count"] == 1
    assert report["counts"]["data_hygiene_disqualified_roots"] == 1
    assert report["selected_integrity"]["passes"]
    assert report["selected_integrity"]["failure_count"] == 0
    access = report["provenance_field_access"]
    assert access["reset_or_root_score_fields_may_be_read"]
    assert access["terminal_completion_flag_read"]
    assert not access["final_or_future_score_fields_read"]
    assert not access["future_milestone_or_terminal_max_tile_fields_read"]
    assert not access["score_or_outcome_fields_used_for_selection"]
    assert not report["recorded_actions_accessed"]


def test_scan_uses_selected_integrity_not_hygiene_failure_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "o1"
    output.mkdir()
    marker_path = output / "marker.json"
    inventory_path = output / "inventory.json"
    exclusion_path = output / "exclusion.json"
    root_manifest_path = output / "roots.json"
    result_path = output / "result.json"
    marker = p0.payload_with_hash(
        {
            "implementation": {"schema_sha256": p0.schema_sha256()},
            "family_evidence": {"passes": True},
            "future_streams": {"consumed": False},
        }
    )
    p0.write_immutable_json(marker_path, marker)
    p0.write_immutable_json(inventory_path, {"rows": []})
    p0.write_immutable_json(exclusion_path, {"excluded_roots": []})
    monkeypatch.setattr(p0, "MARKER_PATH", marker_path)
    monkeypatch.setattr(p0, "SOURCE_INVENTORY_PATH", inventory_path)
    monkeypatch.setattr(p0, "EXCLUSION_PATH", exclusion_path)
    monkeypatch.setattr(p0, "ROOT_MANIFEST_PATH", root_manifest_path)
    monkeypatch.setattr(p0, "RESULT_PATH", result_path)
    monkeypatch.setattr(p0, "ROOT", tmp_path)
    monkeypatch.setattr(
        p0,
        "_revalidate_marker",
        lambda: (marker, {"rows": []}, {"excluded_roots": []}),
    )
    monkeypatch.setattr(
        p0,
        "scan_candidate_content",
        lambda inventory, exclusion: (
            [],
            {
                "support": {},
                "counts": {"data_hygiene_failure_sources": 1},
                "data_hygiene_failure_count": 1,
                "selected_integrity": {
                    "passes": True,
                    "failure_count": 0,
                },
            },
        ),
    )
    monkeypatch.setattr(
        p0,
        "power_table",
        lambda: {"selected_smallest_passing_n": None},
    )
    monkeypatch.setattr(
        p0,
        "allocate_partitions",
        lambda records, selected: ([], {"passes": False}),
    )
    monkeypatch.setattr(
        p0.history,
        "service_health",
        lambda: {
            "passes": True,
            "dashboard_top_scores": list(p0.EXPECTED_TOP_THREE),
        },
    )
    result_identity = p0.scan()
    result = json.loads(result_path.read_text())
    assert result_identity["payload_valid"]
    assert result["decision"] == "HOLD_O1_DATA_OR_POWER"
    assert result["integrity"]["selected_state_failures_zero"]
    assert result["integrity"]["data_hygiene_failures_are_support_loss_only"]


def test_partition_cell_allocator_is_deterministic_and_family_capped() -> None:
    rows = [
        {
            "root_cluster": f"root-{stage}-{index}",
            "behavior_family": p0.GENUINE_FAMILIES[index % 5],
            "stage": stage,
            "scale_band": "early",
        }
        for stage in range(4)
        for index in range(10)
    ]
    first, report = p0._allocate_cells(
        rows,
        cells=tuple((stage,) for stage in range(4)),
        quota_per_cell=5,
        total_target=20,
        key_prefix="test",
    )
    second, second_report = p0._allocate_cells(
        rows,
        cells=tuple((stage,) for stage in range(4)),
        quota_per_cell=5,
        total_target=20,
        key_prefix="test",
    )
    assert [row["root_cluster"] for row in first] == [
        row["root_cluster"] for row in second
    ]
    assert report == second_report
    assert report["passes"]
    assert report["maximum_family_share"] <= 0.40


def test_power_table_mde_is_computed_at_selected_smallest_n(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(p0, "POWER_N_GRID", (144, 192))
    monkeypatch.setattr(p0, "POWER_OR_GRID", (1.25, 1.50, 1.75))

    def fake_power(n_roots: int, odds_ratio: float) -> dict:
        power = (
            0.90
            if n_roots == 144
            else (0.81 if odds_ratio >= 1.50 else 0.60)
        )
        return {
            "n_roots": n_roots,
            "odds_ratio": odds_ratio,
            "power_lower_bound_above_zero": power,
        }

    monkeypatch.setattr(p0, "simulate_common_or_power", fake_power)
    report = p0.power_table()
    assert report["selected_smallest_passing_n"] == 192
    assert report["mde_at_selected_n"] == 1.50
    assert all(row["n_roots"] == 192 for row in report["mde_rows"])
    first = report["or150_rows"][0]
    assert first["power_pass"]
    assert not first["structural_minimum_pass"]
    assert not first["eligible_for_selection"]


def test_scan_requires_a_sealed_marker_before_candidate_content(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(p0, "MARKER_PATH", tmp_path / "missing-marker.json")
    opened = False

    def forbidden_scan(*args: object, **kwargs: object) -> object:
        nonlocal opened
        opened = True
        raise AssertionError("candidate content opened before marker")

    monkeypatch.setattr(p0, "scan_candidate_content", forbidden_scan)
    with pytest.raises(FileNotFoundError):
        p0.scan()
    assert not opened
