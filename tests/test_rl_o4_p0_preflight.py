from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from threes_rl import o4_p0_preflight as p0


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
    for index in range(5):
        family = p0.O3_FAMILY_ORDER[index]
        membership.append(
            {
                "root_cluster": f"root-{index}",
                "family": family,
                "source_replay": f"/tmp/source-{index}.json",
                "source_replay_sha256": f"sha-{index}",
                "deck_stream_id": 100 + index,
                "slot_stream_id": 200 + index,
            }
        )
    candidates = [
        {
            "root_cluster": f"root-{index}",
            "family": p0.O3_FAMILY_ORDER[index],
            "target": 48,
            "frame_index": 1,
            "state_sha1": f"state-{index}",
            "source_replay": f"/tmp/source-{index}.json",
            "source_replay_sha256": f"sha-{index}",
        }
        for index in range(5)
    ]
    union = {"membership": membership, "passes": True}
    support = {
        "candidate_rows": candidates,
        "candidate_manifest_sha256": p0.canonical_json_hash(candidates),
        "audit": {"passes": True},
    }
    selected = {
        "selected": [{"root_cluster": "root-0"}],
        "passes": True,
        "deficits": [],
    }
    return union, support, selected


def test_frozen_family_target_matrices_and_marginals_are_exact() -> None:
    report = p0.validate_frozen_matrices()
    assert report["passes"]
    assert report["combined_family_counts"] == {
        "o4_corner2": 90,
        "o4_expectimax2": 90,
        "o4_parent_mc1000": 90,
        "o4_replaycal": 89,
        "o4_qd_v2": 89,
    }
    assert report["combined_target_counts"] == {48: 150, 96: 149, 192: 149}
    assert p0.ROLE_FAMILY_TARGET_COUNTS["train"][0] == (13, 13, 13)
    assert p0.ROLE_FAMILY_TARGET_COUNTS["development"][-1] == (4, 4, 4)
    assert p0.ROLE_FAMILY_TARGET_COUNTS["untouched_mechanism"][2] == (
        13,
        13,
        13,
    )


def test_source_pool_excludes_exact_selected_roots_and_maps_families(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    union, support, selected = _small_source_payloads()
    monkeypatch.setattr(p0, "O3_ACQUISITION_ROOTS", 5)
    monkeypatch.setattr(p0, "O3_SELECTED_ROOTS", 1)
    monkeypatch.setattr(p0, "O3_UNSELECTED_ROOTS", 4)
    report, candidates, index = p0.source_pool_from_payloads(
        union,
        support,
        selected,
        _reseal(),
    )
    assert report["passes"]
    assert len(index) == 5
    assert report["selected_roots_excluded"] == 1
    assert report["unselected_root_universe"] == 4
    assert all(row["root_cluster"] != "root-0" for row in candidates)
    assert {row["family"] for row in candidates} == set(p0.FAMILY_ORDER[1:])
    assert not report["upper_bound_feasible"]


def test_source_pool_rejects_union_support_identity_drift(
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
    report, _candidates, _index = p0.source_pool_from_payloads(
        union,
        support,
        selected,
        _reseal(),
    )
    assert not report["passes"]
    assert report["source_identity_failures"]


def test_source_replay_verification_rejects_option_training_and_hash_drift(
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
    changed = dict(row, source_replay_sha256="0" * 64)
    assert not p0.verify_candidate_source_replays([changed])["passes"]
    forbidden = training / "source_replays" / "episode.json"
    forbidden.write_text("{}")
    forbidden_row = {
        **row,
        "source_replay": str(forbidden),
        "source_replay_sha256": hashlib.sha256(
            forbidden.read_bytes()
        ).hexdigest(),
    }
    assert not p0.verify_candidate_source_replays([forbidden_row])["passes"]


def _exact_allocation_candidates() -> list[dict]:
    candidates = []
    required = p0._required_family_target_counts()
    counter = 0
    for family in p0.FAMILY_ORDER:
        for target in p0.TARGET_ORDER:
            for _index in range(required[(family, target)]):
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


def test_allocator_exactly_reproduces_all_frozen_cells_deterministically() -> None:
    candidates = _exact_allocation_candidates()
    first = p0.allocate_candidates(candidates)
    second = p0.allocate_candidates(list(reversed(candidates)))
    assert first["passes"]
    assert len(first["selected"]) == 448
    assert first["selected_manifest_sha256"] == second[
        "selected_manifest_sha256"
    ]
    assert first["role_counts"] == p0.ROLE_COUNTS
    assert first["family_counts"] == {
        "o4_corner2": 90,
        "o4_expectimax2": 90,
        "o4_parent_mc1000": 90,
        "o4_replaycal": 89,
        "o4_qd_v2": 89,
    }
    assert first["target_counts"] == {"T48": 150, "T96": 149, "T192": 149}


def test_allocator_never_backtracks_or_reuses_a_multitarget_root() -> None:
    candidates = _exact_allocation_candidates()
    replacement = dict(candidates[-1])
    replacement["root_cluster"] = candidates[0]["root_cluster"]
    candidates[-1] = replacement
    report = p0.allocate_candidates(candidates)
    assert not report["passes"]
    assert report["deficits"]
    roots = [row["root_cluster"] for row in report["selected"]]
    assert len(roots) == len(set(roots))
    assert report["checks"]["deterministic_no_backtracking"]


def test_upper_bound_failure_never_opens_replay_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = {
        "upper_bound_feasible": False,
        "upper_bound_checks": {
            "family_target_cells_can_meet_combined_need": False,
            "families_can_meet_combined_need": False,
            "targets_can_meet_combined_need": True,
        },
    }

    def forbidden(_rows):
        raise AssertionError("replay geometry must remain unopened")

    monkeypatch.setattr(p0, "restore_o4_candidates", forbidden)
    restore, allocation = p0.support_and_allocation(source, [])
    assert restore["scan_skipped"]
    assert restore["source_roots_opened"] == 0
    assert not allocation["allocation_attempted"]
    assert not allocation["passes"]


def test_future_stream_contract_has_no_acquisition_and_exact_crn_counts() -> None:
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
    assert "acquisition" not in contract["purpose_counts"]
    paired = [row for row in rows if "control_policy_stream_id" in row]
    assert all(
        row["control_policy_stream_id"]
        != row["treatment_policy_stream_id"]
        for row in paired
    )
    assert contract["streams_consumed"] == 0


def test_exact_o3_learning_stream_reservation_is_bound_and_disjoint() -> None:
    audit, o3_rows = p0.o3_learning_stream_audit()
    assert audit["passes"]
    assert audit["learning_rows"] == 1_152
    current = p0._stream_sets(p0.future_stream_rows())
    prior = p0._stream_sets(o3_rows)
    assert all(not current[field].intersection(prior[field]) for field in current)


def test_collision_audit_skips_o3_training_bodies_and_detects_collision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs = tmp_path / "runs"
    training = runs / "o3_training"
    acquisition_replays = runs / "o3_acquisition" / "source_replays"
    recovery_replays = runs / "o3_recovery" / "source_replays"
    current = runs / "current"
    for directory in (
        training,
        acquisition_replays,
        recovery_replays,
        current,
    ):
        directory.mkdir(parents=True)
    (training / "episode.json").write_text(
        '{"logical_seed":129000000000}'
    )
    (acquisition_replays / "replay.json").write_text(
        '{"logical_seed":129000000000}'
    )
    (recovery_replays / "replay.json").write_text(
        '{"logical_seed":129000000000}'
    )
    external = runs / "external.json"
    external.write_text('{"logical_seed":129000000000}')
    monkeypatch.setattr(p0, "O3_OPTION_TRAINING_DIR", training)
    monkeypatch.setattr(p0, "O3_ACQUISITION_DIR", runs / "o3_acquisition")
    monkeypatch.setattr(p0, "O3_RECOVERY_DIR", runs / "o3_recovery")
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
    row = p0.future_stream_rows()[0]
    report = p0.collision_audit(
        [row],
        scan_root=runs,
        out_dir=current,
    )
    assert not report["passes"]
    assert report["collisions"]["logical_seed"] == [129_000_000_000]
    excluded = {item["classification"] for item in report["excluded_sources"]}
    assert "o3_option_training_body_unread" in excluded
    assert "o3_acquisition_replay_body_unread" in excluded


def test_domain_and_power_preflight_contracts_pass() -> None:
    domain = p0.domain_proof()
    assert domain["passes"]
    assert domain["parameter_count"] == 102_557
    assert domain["exhaustive"]["occupancy_cases"] == 43_296
    power = p0.power_table()
    assert power["passes"]
    assert power["selected_roots"] == 192
    assert power["grid_mde"] == 1.50


def test_decision_separates_representation_kill_from_support_hold() -> None:
    assert p0._decision(
        integrity_checks={"domain": False},
        support_checks={"support": True},
    ) == "KILL_O4_REPRESENTATION_PREFLIGHT"
    assert p0._decision(
        integrity_checks={"domain": True},
        support_checks={"support": False},
    ) == "HOLD_O4_DATA_SUPPORT"
    assert p0._decision(
        integrity_checks={"domain": True},
        support_checks={"support": True},
    ) == "READY_O4_DOMAIN_SAFE_OPTION_PREFLIGHT"


def test_open_writes_only_zero_work_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = tmp_path / "o4"
    monkeypatch.setattr(p0, "OUTPUT_DIR", out_dir)
    monkeypatch.setattr(
        p0,
        "_load_test_evidence",
        lambda: {
            "passes": True,
            "test_evidence_payload_sha256": "evidence",
        },
    )
    monkeypatch.setattr(
        p0,
        "IMMUTABLE_INPUT_HASHES",
        {},
    )
    monkeypatch.setattr(
        p0,
        "_heavy_process_audit",
        lambda: {"passes": True, "candidates": []},
    )
    monkeypatch.setattr(
        p0.history,
        "service_health",
        lambda: {"passes": True},
    )
    monkeypatch.setattr(p0.history, "current_nice", lambda: 19)
    monkeypatch.setattr(
        p0.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=140 * 1024**3),
    )
    monkeypatch.setattr(p0, "_current_bindings", lambda: {"version": p0.VERSION})
    monkeypatch.setattr(p0, "TEST_EVIDENCE_PATH", tmp_path / "evidence.json")
    p0.TEST_EVIDENCE_PATH.write_text("{}")
    marker = p0.open_preflight(out_dir)
    assert marker["decision"] == "O4_P0_OPENED_ZERO_WORK"
    assert marker["zero_work"]["games"] == 0
    assert marker["zero_work"]["source_replay_bodies_opened"] == 0
    assert [path.name for path in out_dir.iterdir()] == [p0.MARKER_NAME]


def test_marker_rejects_binding_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = tmp_path / "o4"
    out_dir.mkdir()
    monkeypatch.setattr(p0, "OUTPUT_DIR", out_dir)
    monkeypatch.setattr(
        p0,
        "_load_test_evidence",
        lambda: {
            "passes": True,
            "test_evidence_payload_sha256": "evidence",
        },
    )
    monkeypatch.setattr(p0, "_current_bindings", lambda: {"binding": "one"})
    monkeypatch.setattr(p0, "TEST_EVIDENCE_PATH", tmp_path / "evidence.json")
    p0.TEST_EVIDENCE_PATH.write_text("{}")
    payload = {
        **p0._marker_identity(out_dir),
        "decision": "O4_P0_OPENED_ZERO_WORK",
    }
    p0._write_immutable_json(
        out_dir / p0.MARKER_NAME,
        payload,
        self_hash_field="opened_payload_sha256",
    )
    monkeypatch.setattr(p0, "_current_bindings", lambda: {"binding": "two"})
    with pytest.raises(p0.SourceIntegrityError, match="binding"):
        p0._load_marker(out_dir)


def test_source_hash_verification_does_not_parse_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    acquisition = tmp_path / "acquisition"
    replay_dir = acquisition / "source_replays"
    replay_dir.mkdir(parents=True)
    replay = replay_dir / "replay.json"
    replay.write_text("this-is-not-json")
    monkeypatch.setattr(p0, "O3_ACQUISITION_DIR", acquisition)
    monkeypatch.setattr(p0, "O3_RECOVERY_DIR", tmp_path / "recovery")
    monkeypatch.setattr(p0, "O3_OPTION_TRAINING_DIR", tmp_path / "training")
    row = {
        "root_cluster": "root",
        "source_replay": str(replay),
        "source_replay_sha256": hashlib.sha256(replay.read_bytes()).hexdigest(),
    }
    report = p0.verify_candidate_source_replays([row])
    assert report["passes"]
    assert report["replay_bodies_parsed"] is False


def test_whitelist_restore_never_accesses_forbidden_replay_fields() -> None:
    class GuardedMapping(dict):
        def __init__(self, *args, forbidden: set[str], **kwargs):
            super().__init__(*args, **kwargs)
            self.forbidden = forbidden
            self.accessed: list[str] = []

        def __getitem__(self, key):
            if key in self.forbidden:
                raise AssertionError(f"forbidden key accessed: {key}")
            self.accessed.append(str(key))
            return super().__getitem__(key)

        def get(self, key, default=None):
            if key in self.forbidden:
                raise AssertionError(f"forbidden key accessed: {key}")
            self.accessed.append(str(key))
            return super().get(key, default)

    cycle = GuardedMapping(
        {
            "small_counts": {"red": 1, "blue": 2, "gray": 3},
            "small_pos": 4,
            "small_seen_total": 30,
            "span_small_pos": 7,
            "large_pending": False,
            "max_tile": object(),
            "score": object(),
        },
        forbidden={"max_tile", "score"},
    )
    payload = GuardedMapping(
        {
            "board": [
                [1536, 3, 6, 12],
                [48, 24, 48, 96],
                [0, 192, 2, 384],
                [0, 768, 1, 3],
            ],
            "preview": {"kind": "bonus", "candidates": [24, 48, 96]},
            "tile_cycle": cycle,
            "move_count": 120,
            "game_over": False,
            "score": object(),
            "max_tile": object(),
            "legal_actions": object(),
            "move": object(),
            "action": object(),
            "outcome": object(),
            "final_score": object(),
        },
        forbidden={
            "score",
            "max_tile",
            "legal_actions",
            "move",
            "action",
            "outcome",
            "final_score",
        },
    )
    state, identity = p0.whitelisted_state_payload(payload)
    assert state.max_tile == int(np.max(state.board))
    assert set(payload.accessed) == {
        "board",
        "preview",
        "tile_cycle",
        "move_count",
        "game_over",
    }
    assert set(cycle.accessed) == {
        "small_counts",
        "small_pos",
        "small_seen_total",
        "span_small_pos",
        "large_pending",
    }
    assert "max_tile" not in identity["tile_cycle"]
    assert "score" not in identity
