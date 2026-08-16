from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from threes_rl import g1r_acquire_v2_qd5 as pilot
from threes_rl.record_replay import state_payload
from threes_rl.sim import ThreesSim


AUTHORITATIVE_CHARTER_SHA256 = (
    "1f58d73b5f21aed20806f605009b6572bc9587aafa0c7cf2323f54f90ddce003"
)
SUPERSEDED_CHARTER_SHA256 = (
    "06ae8fa29edee9d5e86a6af7e0f63a330fa4cde7d4a0fb8d1f6c2d67ef0daebf"
)


def test_authoritative_charter_hash_and_superseded_hash_are_explicit():
    assert pilot.sha256_path(pilot.CHARTER_PATH) == AUTHORITATIVE_CHARTER_SHA256
    text = pilot.CHARTER_PATH.read_text()
    assert SUPERSEDED_CHARTER_SHA256 in text
    assert "k_f,1536 + k_f,3072 == k_f,any" in text
    for path, expected in pilot.SUPERSEDED_TEST_EVIDENCE:
        assert pilot.sha256_path(path) == expected


def test_preserved_base_and_s3_inputs_match_frozen_hashes():
    audit = pilot._immutable_input_audit()
    assert audit["passes"]
    assert audit["artifacts"]["s3_power"]["actual"] == pilot.S3_POWER_SHA256
    assert (
        audit["artifacts"]["s3_provenance"]["actual"]
        == pilot.S3_PROVENANCE_SHA256
    )
    assert (
        audit["artifacts"]["base_implementation"]["actual"]
        == pilot.BASE_IMPLEMENTATION_SHA256
    )


def test_exact_five_family_order_and_specs():
    assert [family for family, _spec in pilot.policy_slate()] == [
        "g1r_corner2",
        "g1r_expectimax2",
        "g1r_parent_mc1000",
        "g1r_replaycal",
        "g1r_qd_static_archive_oneply_v2_terminal_schema",
    ]
    assert len(dict(pilot.policy_slate())) == 5
    assert dict(pilot.policy_slate())[pilot.QD_FAMILY] == pilot.QD_SPEC


def test_requested_stream_manifest_is_exact_unique_100_rows():
    rows = pilot.requested_stream_manifest()
    assert len(rows) == 100
    assert [row["nominal_family"] for row in rows[::20]] == [
        family for family, _spec in pilot.FAMILY_SLATE
    ]
    assert [row["game_index"] for row in rows[:20]] == list(range(20))
    values = [row[key] for row in rows for key in pilot.STREAM_BASES]
    assert len(values) == len(set(values))
    assert rows[0]["logical_seed"] == 49_000_000_000
    assert rows[20]["logical_seed"] == 49_001_000_000
    assert rows[-1]["policy_stream_id"] == 52_004_000_019


def test_stream_collision_audit_detects_aliases(monkeypatch, tmp_path):
    rows = pilot.requested_stream_manifest()
    monkeypatch.setattr(
        pilot.base,
        "historical_collision_union",
        lambda **_kwargs: (
            {"root_seed": {rows[0]["logical_seed"]}},
            {"matched_source_count": 1, "matched_sources_sha256": "x"},
        ),
    )
    audit = pilot.stream_collision_audit(rows, exclude_dir=tmp_path)
    assert not audit["zero_collisions"]
    assert audit["collisions"]["logical_seed"] == [rows[0]["logical_seed"]]


def _candidate(root: str, stratum: str, frame: int, state_hash: str, family: str):
    return {
        "root_cluster": root,
        "stratum": stratum,
        "source_frame_index": frame,
        "state_sha1": state_hash,
        "behavior_family": family,
        "role": "source_control",
    }


def test_root_cap_is_global_across_strata_and_uses_frozen_hash_argmin():
    family = pilot.FAMILY_SLATE[0][0]
    candidates = [
        _candidate("root-a", "pre1536", 10, "a", family),
        _candidate("root-a", "pre3072", 20, "b", family),
        _candidate("root-b", "pre1536", 30, "c", family),
    ]
    selected = pilot.root_cap_candidates(candidates)
    assert len(selected) == 2
    root_a = [row for row in selected if row["root_cluster"] == "root-a"]
    assert len(root_a) == 1
    expected = min(
        candidates[:2],
        key=lambda row: pilot.canonical_json_hash(
            [
                "G1R-pilot-v2-root-cap",
                row["root_cluster"],
                row["stratum"],
                row["source_frame_index"],
                row["state_sha1"],
            ]
        ),
    )
    assert root_a[0] == expected


def test_yield_projection_enforces_per_family_cross_stratum_conservation():
    completed = [
        {"nominal_family": family}
        for family, _spec in pilot.FAMILY_SLATE
        for _index in range(20)
    ]
    selected = [
        _candidate(f"{family}-a", "pre1536", 1, "a", family)
        for family, _spec in pilot.FAMILY_SLATE
    ] + [
        _candidate(f"{family}-b", "pre3072", 2, "b", family)
        for family, _spec in pilot.FAMILY_SLATE
    ]
    projection = pilot.yield_projection(completed, selected)
    for row in projection["family_rows"]:
        counts = row["counts"]
        assert counts["pre1536"] + counts["pre3072"] == counts["any"]
        assert row["ancestry_unique_conservation_passes"]
    assert projection["checks"]["per_family_cross_stratum_conservation"]


def test_yield_projection_rejects_duplicate_ancestry():
    family = pilot.FAMILY_SLATE[0][0]
    duplicate = [
        _candidate("same-root", "pre1536", 1, "a", family),
        _candidate("same-root", "pre3072", 2, "b", family),
    ]
    with pytest.raises(ValueError, match="ancestry-unique"):
        pilot.yield_projection([], duplicate)


def test_wilson_lower_bound_edges():
    assert pilot.wilson_lower(0, 0) == 0.0
    assert pilot.wilson_lower(0, 20) == 0.0
    assert 0.8 < pilot.wilson_lower(20, 20) < 1.0
    assert pilot.wilson_lower(10, 20) < 0.5


def test_storage_admission_binding_is_ready_and_below_cap():
    audit = pilot.storage_admission_audit()
    assert audit["passes"]
    assert audit["projected_bytes"] == 313_094_177
    assert audit["projected_bytes"] < 4 * 1024**3


def test_load_panel_uses_only_complete_pilot_v1_lock():
    panel, source = pilot._load_panel()
    assert source["file_sha256"] == pilot.PILOT_V1_LOCK_SHA256
    assert panel["panel_sha256"] == pilot.PANEL_SHA256
    assert len(panel["records"]) == 64


class _FakeQD:
    def decision(self, state, sim):
        legal = sim.legal_actions(state)
        return {
            "action": legal[0],
            "tie_count_before_action_priority": 1,
        }


def test_qd_deterministic_action_does_not_mutate_state():
    sim = ThreesSim.from_stream_ids(
        deck_stream_id=901,
        slot_stream_id=902,
        starter_tile=pilot.STARTER_TILE,
    )
    state = sim.reset()
    payload = state_payload(state, sim)
    before = json.loads(json.dumps(payload))
    row = pilot._deterministic_action(pilot.QD_FAMILY, _FakeQD(), payload)
    assert row["state_unmutated"]
    assert payload == before
    assert row["exact_tie_count"] == 1


def test_action_signature_audit_checks_all_ten_pairs(monkeypatch):
    families = [family for family, _spec in pilot.FAMILY_SLATE]
    panel = {
        "records": [
            {"stratum": "pre1536", "state": {"index": index}}
            for index in range(32)
        ]
        + [
            {"stratum": "pre3072", "state": {"index": index}}
            for index in range(32, 64)
        ],
    }

    def synthetic_action(family, index):
        offset = families.index(family)
        if offset == 4:
            return (3 - index) % 4
        return (index + offset) % 4

    def fake_action(family, _policy, payload):
        return {
            "action": synthetic_action(family, payload["index"]),
            "exact_tie_count": 1,
            "state_unmutated": True,
        }

    signatures = {
        family: [synthetic_action(family, index) for index in range(64)]
        for family in families
    }
    expected_hashes = {
        family: pilot.canonical_json_hash(actions)
        for family, actions in signatures.items()
    }
    accepted = {}
    for left_index, left in enumerate(families):
        for right in families[left_index + 1 :]:
            overall = sum(
                a != b for a, b in zip(signatures[left], signatures[right])
            ) / 64
            accepted[(left, right)] = {
                "overall_disagreement": overall,
                "stratum_disagreement": {
                    "pre1536": sum(
                        signatures[left][index] != signatures[right][index]
                        for index in range(32)
                    )
                    / 32,
                    "pre3072": sum(
                        signatures[left][index] != signatures[right][index]
                        for index in range(32, 64)
                    )
                    / 32,
                },
            }
    monkeypatch.setattr(pilot, "_deterministic_action", fake_action)
    monkeypatch.setattr(pilot, "EXPECTED_SIGNATURES", expected_hashes)
    monkeypatch.setattr(pilot, "_accepted_pairwise", lambda: accepted)
    audit = pilot.action_signature_audit(
        {family: object() for family in families}, panel
    )
    assert audit["passes"]
    assert len(audit["pairwise"]) == 10
    assert all(row["accepted_exact_match"] for row in audit["pairwise"])


def test_split_reset_fixture_uses_no_reserved_pilot_stream():
    fixture = pilot.split_reset_roundtrip_fixture()
    assert fixture["passes"]
    assert fixture["checks"]["reserved_pilot_namespace_unused"]


def test_sealed_payload_audit_rejects_hash_change(tmp_path):
    path = tmp_path / "sealed.json"
    payload = {"version": "x", "value": 1}
    payload_hash = pilot.canonical_json_hash(payload)
    path.write_text(
        json.dumps({**payload, "payload_sha256": payload_hash}, sort_keys=True)
    )
    audit = pilot._sealed_payload_audit(
        path,
        expected_file_sha256=pilot.sha256_path(path),
        self_hash_field="payload_sha256",
        expected_payload_sha256=payload_hash,
    )
    assert audit["passes"]
    path.write_text(path.read_text() + "\n")
    changed = pilot._sealed_payload_audit(
        path,
        expected_file_sha256=audit["file_sha256"],
        self_hash_field="payload_sha256",
        expected_payload_sha256=payload_hash,
    )
    assert not changed["passes"]


def test_prepare_preflight_promotes_staging_atomically(monkeypatch, tmp_path):
    final = tmp_path / "pilot_v2_qd5"
    monkeypatch.setattr(pilot, "OUTPUT_DIR", final)

    def fake_prepare(staging, final_dir):
        pilot._write_new_json_atomic(
            staging / "preflight_lock.json",
            {"bound_out_dir": str(final_dir)},
        )
        return {"decision": "READY_G1R_PILOT_V2_QD5_PREFLIGHT"}

    monkeypatch.setattr(pilot, "_prepare_preflight_in_staging", fake_prepare)
    result = pilot.prepare_preflight(final)
    assert result["decision"] == "READY_G1R_PILOT_V2_QD5_PREFLIGHT"
    assert (final / "preflight_lock.json").is_file()
    assert not list(tmp_path.glob("pilot_v2_qd5.staging.*"))


def test_prepare_preflight_rejects_existing_output(monkeypatch, tmp_path):
    final = tmp_path / "pilot_v2_qd5"
    final.mkdir()
    monkeypatch.setattr(pilot, "OUTPUT_DIR", final)
    with pytest.raises(FileExistsError):
        pilot.prepare_preflight(final)


def test_preflight_path_never_invokes_game_evaluator(monkeypatch, tmp_path):
    final = tmp_path / "pilot_v2_qd5"
    monkeypatch.setattr(pilot, "OUTPUT_DIR", final)
    monkeypatch.setattr(
        pilot,
        "iter_eval_job_outputs",
        lambda **_kwargs: pytest.fail("preflight invoked game evaluator"),
    )

    def fake_prepare(staging, final_dir):
        pilot._write_new_json_atomic(
            staging / "preflight_lock.json",
            {
                "decision": "READY_G1R_PILOT_V2_QD5_PREFLIGHT",
                "zero_work": {"games_generated": 0},
            },
        )
        return {"zero_work": {"games_generated": 0}}

    monkeypatch.setattr(pilot, "_prepare_preflight_in_staging", fake_prepare)
    result = pilot.prepare_preflight(final)
    assert result["zero_work"]["games_generated"] == 0


def test_source_has_no_human_session_path_or_score_filter():
    source = pilot.IMPLEMENTATION_PATH.read_text()
    assert "datasets/human_play" not in source
    assert "events.jsonl" not in source
    assert "final_score" not in source
    assert "output.result.score" not in source
