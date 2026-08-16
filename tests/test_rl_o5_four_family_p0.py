from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

from threes_rl import o5_four_family_p0 as p0
from threes_rl.baselines import GreedyPolicy
from threes_rl.o4_domain_safe_pair_option import (
    advance_lineage_base,
    apply_spawn_to_lineage,
    build_decision_targets,
    initial_lineage,
    option_features,
    root_option_eligible,
    select_designated_pair,
    successor_geometry,
    transition_status,
)
from threes_rl.sim import ThreesSim


def _reseal() -> dict:
    return {
        "decision": "READY_O3_OPTION_TRAINING_INTEGRITY_RESEALED_V3",
        "selected_pre_serialization_reproduction_sha256": (
            p0.SELECTED_PRE_SERIALIZATION_SHA256
        ),
        "selected_post_json_scientific_payload_sha256": (
            p0.SELECTED_POST_JSON_SHA256
        ),
    }


def _small_source_payloads() -> tuple[dict, dict, dict]:
    membership = []
    candidate_rows = []
    for index, family in enumerate(p0.O3_FAMILY_ORDER):
        root = f"root-{index}"
        replay = f"/sources/{root}.json"
        replay_hash = f"sha-{index}"
        membership.append(
            {
                "root_cluster": root,
                "family": family,
                "source_replay": replay,
                "source_replay_sha256": replay_hash,
                "deck_stream_id": 100 + index,
                "slot_stream_id": 200 + index,
            }
        )
        candidate_rows.append(
            {
                "root_cluster": root,
                "family": family,
                "target": 48,
                "frame_index": index,
                "state_sha1": f"state-{index}",
                "source_replay": replay,
                "source_replay_sha256": replay_hash,
            }
        )
    union = {"membership": membership, "passes": True}
    support = {
        "candidate_rows": candidate_rows,
        "candidate_manifest_sha256": p0.canonical_json_hash(candidate_rows),
        "audit": {"passes": True},
    }
    selected = {
        "selected": [{"root_cluster": "root-0"}],
        "passes": True,
        "deficits": [],
    }
    return union, support, selected


def _allocation_candidates() -> list[dict]:
    candidates = []
    counter = 0
    for family in p0.FAMILY_ORDER:
        for target in p0.TARGET_ORDER:
            required = p0._required_family_target_counts()[(family, target)]
            for _ in range(required):
                candidates.append(
                    {
                        "root_cluster": f"root-{counter:04d}",
                        "family": family,
                        "target": target,
                        "frame_index": counter,
                        "state_sha1": f"state-{counter:04d}",
                    }
                )
                counter += 1
    return candidates


class ForbiddenSentinel(Mapping):
    def __init__(self, allowed: dict, forbidden: set[str]) -> None:
        self.allowed = allowed
        self.forbidden = forbidden

    def __getitem__(self, key):
        if key in self.forbidden:
            raise AssertionError(f"forbidden key accessed: {key}")
        return self.allowed[key]

    def __iter__(self):
        return iter(self.allowed)

    def __len__(self):
        return len(self.allowed)

    def get(self, key, default=None):
        if key in self.forbidden:
            raise AssertionError(f"forbidden key accessed: {key}")
        return self.allowed.get(key, default)


def test_frozen_matrices_have_exact_four_family_marginals() -> None:
    report = p0.validate_frozen_matrices()
    assert report["passes"]
    assert report["combined_family_counts"] == {
        "o5_corner2": 112,
        "o5_expectimax2": 112,
        "o5_parent_mc1000": 112,
        "o5_replaycal": 112,
    }
    assert report["combined_target_counts"] == {48: 150, 96: 149, 192: 149}
    assert p0.ROLE_FAMILY_TARGET_COUNTS["train"] == (
        (16, 16, 16),
        (16, 16, 16),
        (16, 16, 16),
        (16, 16, 16),
    )
    assert p0.ROLE_FAMILY_TARGET_COUNTS["development"] == (
        (6, 5, 5),
        (5, 6, 5),
        (5, 5, 6),
        (6, 5, 5),
    )


def test_domain_contract_reuses_exact_immutable_operator() -> None:
    report = p0.domain_proof()
    assert report["passes"]
    assert report["schema_sha256"] == p0.O4_OPERATOR_SCHEMA_SHA256
    assert report["parameter_count"] == 102_557
    assert report["proof"]["coordinate_pairs"] == 120
    assert report["proof"]["occupancy_cases"] == 43_296
    assert report["proof"]["minimum_density"] == 0.0
    assert report["proof"]["maximum_density"] == 1.0


def test_source_pool_excludes_selected_and_qd_without_substitution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    union, support, selected = _small_source_payloads()
    monkeypatch.setattr(p0, "O3_ACQUISITION_ROOTS", 5)
    monkeypatch.setattr(p0, "O3_SELECTED_ROOTS", 1)
    monkeypatch.setattr(p0, "O3_UNSELECTED_ROOTS", 4)
    report, candidates = p0.source_pool_from_payloads(
        union,
        support,
        selected,
        _reseal(),
    )
    assert report["passes"]
    assert report["selected_roots_excluded"] == 1
    assert report["unselected_root_universe"] == 4
    assert report["qd_candidate_rows_excluded"] == 1
    assert report["qd_candidate_roots_excluded"] == 1
    assert {row["family"] for row in candidates} == set(p0.FAMILY_ORDER[1:])
    assert all(row["root_cluster"] != "root-0" for row in candidates)
    assert all(row["source_family"] != p0.EXCLUDED_O3_FAMILY for row in candidates)


def test_source_pool_rejects_identity_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    union, support, selected = _small_source_payloads()
    support["candidate_rows"][1]["source_replay_sha256"] = "changed"
    support["candidate_manifest_sha256"] = p0.canonical_json_hash(
        support["candidate_rows"]
    )
    monkeypatch.setattr(p0, "O3_ACQUISITION_ROOTS", 5)
    monkeypatch.setattr(p0, "O3_SELECTED_ROOTS", 1)
    monkeypatch.setattr(p0, "O3_UNSELECTED_ROOTS", 4)
    report, _ = p0.source_pool_from_payloads(
        union,
        support,
        selected,
        _reseal(),
    )
    assert not report["passes"]
    assert report["source_identity_failures"]


def test_whitelist_never_accesses_forbidden_score_action_or_max_tile() -> None:
    cycle = ForbiddenSentinel(
        {
            "small_counts": {"red": 2, "blue": 3, "gray": 4},
            "small_pos": 3,
            "small_seen_total": 55,
            "span_small_pos": 7,
            "large_pending": True,
        },
        {"max_tile", "score", "final_score"},
    )
    payload = ForbiddenSentinel(
        {
            "board": [
                [1536, 3, 6, 12],
                [48, 24, 48, 6144],
                [0, 3072, 2, 768],
                [0, 96, 192, 384],
            ],
            "preview": {"kind": "bonus", "candidates": [24, 48, 96]},
            "tile_cycle": cycle,
            "move_count": 120,
            "game_over": False,
        },
        {
            "score",
            "final_score",
            "max_tile",
            "legal_actions",
            "move",
            "action",
            "outcome",
        },
    )
    state, identity = p0.whitelisted_state_payload(payload)
    assert state.max_tile == 6144
    assert "max_tile" not in identity
    assert set(identity) == {
        "board",
        "preview",
        "tile_cycle",
        "move_count",
        "game_over",
    }


def test_source_replay_manifest_rejects_hash_drift_and_training_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    acquisition = tmp_path / "acquisition"
    recovery = tmp_path / "recovery"
    training = tmp_path / "training"
    for directory in (acquisition, recovery, training):
        (directory / "source_replays").mkdir(parents=True)
    good = acquisition / "source_replays" / "one.json"
    good.write_text("{}")
    monkeypatch.setattr(p0, "O3_ACQUISITION_DIR", acquisition)
    monkeypatch.setattr(p0, "O3_RECOVERY_DIR", recovery)
    monkeypatch.setattr(p0, "O3_OPTION_TRAINING_DIR", training)
    row = {
        "root_cluster": "root",
        "source_replay": str(good),
        "source_replay_sha256": hashlib.sha256(good.read_bytes()).hexdigest(),
    }
    assert p0.verify_candidate_source_replays([row])["passes"]
    assert not p0.verify_candidate_source_replays(
        [{**row, "source_replay_sha256": "0" * 64}]
    )["passes"]
    forbidden = training / "source_replays" / "episode.json"
    forbidden.write_text("{}")
    assert not p0.verify_candidate_source_replays(
        [
            {
                **row,
                "source_replay": str(forbidden),
                "source_replay_sha256": hashlib.sha256(
                    forbidden.read_bytes()
                ).hexdigest(),
            }
        ]
    )["passes"]


def test_allocator_exactly_reproduces_every_frozen_cell() -> None:
    candidates = _allocation_candidates()
    first = p0.allocate_candidates(candidates)
    second = p0.allocate_candidates(list(reversed(candidates)))
    assert first["passes"]
    assert len(first["selected"]) == 448
    assert first["selected_manifest_sha256"] == second[
        "selected_manifest_sha256"
    ]
    assert first["role_counts"] == p0.ROLE_COUNTS
    assert first["family_counts"] == {
        family: 112 for family in p0.FAMILY_ORDER
    }
    assert first["target_counts"] == {"T48": 150, "T96": 149, "T192": 149}
    for role in p0.ROLE_ORDER:
        for family_index, family in enumerate(p0.FAMILY_ORDER):
            for target_index, target in enumerate(p0.TARGET_ORDER):
                assert first["role_family_target_counts"][role][family][
                    f"T{target}"
                ] == p0.ROLE_FAMILY_TARGET_COUNTS[role][family_index][
                    target_index
                ]


def test_allocator_is_one_root_and_never_backtracks() -> None:
    candidates = _allocation_candidates()
    replacement = dict(candidates[-1])
    replacement["root_cluster"] = candidates[0]["root_cluster"]
    candidates[-1] = replacement
    report = p0.allocate_candidates(candidates)
    assert not report["passes"]
    assert report["deficits"]
    roots = [row["root_cluster"] for row in report["selected"]]
    assert len(roots) == len(set(roots))
    assert report["checks"]["deterministic_no_backtracking"]


def test_upper_bound_failure_opens_no_replay_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = {
        "upper_bound_feasible": False,
        "upper_bound_checks": {
            "family_target_cells_can_meet_combined_need": False,
            "families_can_meet_combined_need": True,
            "targets_can_meet_combined_need": True,
        },
    }

    def forbidden(_rows):
        raise AssertionError("replay geometry must remain unopened")

    monkeypatch.setattr(p0, "restore_o5_candidates", forbidden)
    restore, allocation = p0.support_and_allocation(source, [])
    assert restore["scan_skipped"]
    assert restore["source_roots_opened"] == 0
    assert not allocation["allocation_attempted"]


def test_fresh_stream_contract_is_exact_and_zero_consumption() -> None:
    rows = p0.future_stream_rows()
    contract = p0.stream_contract(rows)
    assert contract["passes"]
    assert contract["row_count"] == 6_272
    assert contract["purpose_counts"] == {
        "learning": 1_152,
        "option_development": 512,
        "option_untouched_mechanism": 1_536,
        "normal_development": 512,
        "confirmation": 2_560,
    }
    assert contract["stream_bases"] == p0.STREAM_BASES
    assert contract["streams_consumed"] == 0
    assert min(row["logical_seed"] for row in rows) == 181_000_000_000


def test_o3_and_o4_reservations_are_exact_and_disjoint() -> None:
    requested = p0._stream_sets(p0.future_stream_rows())
    o3_audit, o3_rows = p0.o3_learning_stream_audit()
    o4_audit, o4_rows = p0.o4_reservation_audit()
    assert o3_audit["passes"]
    assert o3_audit["learning_rows"] == 1_152
    assert o4_audit["passes"]
    assert o4_audit["reservation_rows"] == 6_272
    for prior in (p0._stream_sets(o3_rows), p0._stream_sets(o4_rows)):
        assert all(
            not requested[field].intersection(prior[field])
            for field in p0.STREAM_FIELDS
        )


def test_policy_audit_uses_semantic_order_not_signature_map_order() -> None:
    report = p0.policy_audit()
    assert report["passes"]
    assert tuple(report["family_order"]) == p0.FAMILY_ORDER
    assert tuple(report["signatures"]) == p0.FAMILY_ORDER
    assert len(report["pairwise"]) == 6
    assert len(set(report["signatures"].values())) == 4
    assert report["excluded_family"] == "o3_qd_v2"
    assert report["checks"]["semantic_family_order_exact"]


def test_collision_audit_detects_external_but_ignores_skipped_bodies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs = tmp_path / "runs"
    training = runs / "training"
    acquisition = runs / "acquisition"
    recovery = runs / "recovery"
    current = runs / "current"
    for directory in (
        training,
        acquisition / "source_replays",
        recovery / "source_replays",
        current,
    ):
        directory.mkdir(parents=True)
    requested = p0.future_stream_rows()[0]
    logical = requested["logical_seed"]
    (training / "episode.json").write_text(
        json.dumps({"logical_seed": logical})
    )
    (acquisition / "source_replays" / "replay.json").write_text(
        json.dumps({"logical_seed": logical})
    )
    external = runs / "external.json"
    external.write_text(json.dumps({"logical_seed": logical}))
    monkeypatch.setattr(p0, "O3_OPTION_TRAINING_DIR", training)
    monkeypatch.setattr(p0, "O3_ACQUISITION_DIR", acquisition)
    monkeypatch.setattr(p0, "O3_RECOVERY_DIR", recovery)
    monkeypatch.setattr(
        p0,
        "o3_learning_stream_audit",
        lambda: (
            {"passes": True, "learning_rows": 1_152},
            [
                {
                    "logical_seed": 1,
                    "deck_stream_id": 2,
                    "slot_stream_id": 3,
                    "policy_stream_id": 4,
                }
            ],
        ),
    )
    monkeypatch.setattr(
        p0,
        "o4_reservation_audit",
        lambda: (
            {"passes": True, "reservation_rows": 6_272},
            [
                {
                    "logical_seed": 5,
                    "deck_stream_id": 6,
                    "slot_stream_id": 7,
                    "policy_stream_id": 8,
                }
            ],
        ),
    )
    report = p0.collision_audit(
        [requested],
        scan_root=runs,
        out_dir=current,
    )
    assert not report["passes"]
    assert report["collisions"]["logical_seed"] == [logical]
    assert report["o3_direct_collisions"]["logical_seed"] == []
    assert report["o4_direct_collisions"]["logical_seed"] == []
    external.unlink()
    clean = p0.collision_audit(
        [requested],
        scan_root=runs,
        out_dir=current,
    )
    assert clean["passes"]


def test_random_reachable_exact_transition_domain_is_nonvacuous() -> None:
    eligible_states = 0
    legal_transitions = 0
    live_successors = 0
    for game_index in range(64):
        simulator = ThreesSim.from_stream_ids(
            deck_stream_id=900_000 + game_index,
            slot_stream_id=910_000 + game_index,
            starter_tile=1536,
        )
        state = simulator.reset()
        policy_rng = np.random.default_rng(920_000 + game_index)
        policy = GreedyPolicy()
        for _move in range(500):
            legal = tuple(
                int(action) for action in simulator.legal_actions(state)
            )
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
                and root_option_eligible(state, simulator, 1536)
            ):
                eligible_states += 1
                lineage = initial_lineage(pair)
                board_before = state.board.copy()
                deck_before = copy.deepcopy(
                    simulator.deck_rng.bit_generator.state
                )
                slot_before = copy.deepcopy(
                    simulator.slot_rng.bit_generator.state
                )
                for action in legal:
                    tokens, globals_ = option_features(
                        state,
                        simulator,
                        starter_tile=1536,
                        pair=pair,
                        lineage=lineage,
                        action=action,
                    )
                    assert np.isfinite(tokens).all()
                    assert np.isfinite(globals_).all()
                    assert np.all((0.0 <= tokens) & (tokens <= 1.0))
                    assert np.all((0.0 <= globals_) & (globals_ <= 1.0))
                    np.testing.assert_array_equal(state.board, board_before)
                    assert (
                        simulator.deck_rng.bit_generator.state == deck_before
                    )
                    assert (
                        simulator.slot_rng.bit_generator.state == slot_before
                    )

                    base = advance_lineage_base(
                        state.board,
                        lineage,
                        action,
                    )
                    branch = copy.deepcopy(simulator)
                    next_state, info = branch.step(state, action)
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
                        branch,
                        starter_tile=1536,
                        lineage=next_lineage,
                        base_event=base.event,
                    )
                    if status == "live":
                        geometry = successor_geometry(
                            next_state,
                            branch,
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
                assert simulator.deck_rng.bit_generator.state == deck_before
                assert simulator.slot_rng.bit_generator.state == slot_before
            action = policy(state, simulator, policy_rng)
            state, info = simulator.step(state, action)
            assert info.moved
    assert eligible_states == 15
    assert legal_transitions == 55
    assert live_successors > 0


def test_open_path_cannot_parse_source_payloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = tmp_path / "o5"
    monkeypatch.setattr(p0, "OUTPUT_DIR", out_dir)
    monkeypatch.setattr(p0, "_load_test_evidence", lambda: {"passes": True})
    calls = []

    def immutable(*, parse_payloads: bool):
        calls.append(parse_payloads)
        assert not parse_payloads
        return {"passes": True, "source_payloads_parsed": False}

    monkeypatch.setattr(p0, "immutable_input_audit", immutable)
    monkeypatch.setattr(
        p0,
        "_heavy_process_audit",
        lambda: {"passes": True},
    )
    monkeypatch.setattr(
        p0.history,
        "service_health",
        lambda: {"passes": True},
    )
    monkeypatch.setattr(p0.history, "current_nice", lambda: 10)
    monkeypatch.setattr(
        p0.shutil,
        "disk_usage",
        lambda _path: type(
            "Disk",
            (),
            {"free": int(150 * 1024**3)},
        )(),
    )
    monkeypatch.setattr(
        p0,
        "_bindings",
        lambda path: {
            "version": p0.VERSION,
            "bound_out_dir": str(path.resolve()),
        },
    )
    marker = p0.open_preflight(out_dir)
    assert calls == [False]
    payload = json.loads(Path(marker["path"]).read_text())
    assert payload["decision"] == "O5_P0_OPENED_ZERO_WORK"
    assert payload["zero_work"]["source_payloads_parsed"] == 0


def test_marker_validation_rejects_binding_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = tmp_path / "o5"
    out_dir.mkdir()
    monkeypatch.setattr(
        p0,
        "_bindings",
        lambda _path: {"version": p0.VERSION, "bound": "expected"},
    )
    p0._write_immutable_json(
        out_dir / p0.MARKER_NAME,
        {
            "version": p0.VERSION,
            "bound": "changed",
            "decision": "O5_P0_OPENED_ZERO_WORK",
            "zero_work": {"games": 0},
        },
        self_hash_field="opened_payload_sha256",
    )
    with pytest.raises(p0.SourceIntegrityError, match="binding mismatch"):
        p0._load_marker(out_dir)


def test_terminal_decision_precedence_is_fail_closed() -> None:
    assert p0._decision(
        integrity_checks={"identity": False},
        support_checks={"support": True},
    ) == "KILL_O5_FOUR_FAMILY_INTEGRITY_OR_REPRESENTATION"
    assert p0._decision(
        integrity_checks={"identity": True},
        support_checks={"support": False},
    ) == "HOLD_O5_FOUR_FAMILY_DATA_SUPPORT"
    assert p0._decision(
        integrity_checks={"identity": True},
        support_checks={"support": True},
    ) == "READY_O5_FOUR_FAMILY_DOMAIN_SAFE_PREFLIGHT"


def test_run_rejects_existing_terminal_without_opening_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = tmp_path / "o5"
    out_dir.mkdir()
    (out_dir / p0.RESULT_NAME).write_text("{}")
    monkeypatch.setattr(p0, "OUTPUT_DIR", out_dir)

    def forbidden(_path):
        raise AssertionError("marker/source must not open after terminal")

    monkeypatch.setattr(p0, "_load_marker", forbidden)
    with pytest.raises(FileExistsError, match="terminal result exists"):
        p0.run_preflight(out_dir)
