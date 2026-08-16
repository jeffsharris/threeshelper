from __future__ import annotations

import json
from pathlib import Path

import pytest

from threes_rl import g3_scale_transfer_bootstrap_preflight as g3


def test_authoritative_charter_and_input_hashes_are_exact():
    assert g3.sha256_path(g3.CHARTER_PATH) == g3.CHARTER_SHA256
    assert g3.sha256_path(g3.AMENDMENT_PATH) == g3.AMENDMENT_SHA256
    assert g3.schema_sha256() == g3.SCHEMA_SHA256
    assert g3._input_hash_audit()["passes"]


def test_frozen_partition_hashes_reproduce_exactly():
    manifest = g3._json(g3.ROOT_MANIFEST_PATH)
    roots_by_partition = {}
    for partition, expected in g3.PARTITION_LOCKS.items():
        rows, summary = g3._partition_rows(manifest, partition)
        assert all(summary[key] == value for key, value in expected.items())
        roots_by_partition[partition] = {
            row["root_cluster"] for row in rows
        }
    assert not roots_by_partition["train"].intersection(
        roots_by_partition["development"]
    )


def test_stream_rows_share_tapes_across_actions_and_separate_units():
    records = [
        {
            "partition": "train",
            "record_id": "r0",
            "root_cluster": "root0",
            "behavior_family": "family",
            "scale": "pre768",
            "target": 768,
            "state_sha1": "state0",
            "legal_action_ids": [0, 2],
            "legal_actions": ["up", "left"],
        },
        {
            "partition": "development",
            "record_id": "r1",
            "root_cluster": "root1",
            "behavior_family": "family",
            "scale": "pre1536",
            "target": 1536,
            "state_sha1": "state1",
            "legal_action_ids": [1, 3],
            "legal_actions": ["down", "right"],
        },
    ]
    rows = g3.label_stream_rows(records)
    assert len(rows) == 2 * 2 * g3.REPLICATES
    for record_ordinal in range(2):
        for replicate in range(g3.REPLICATES):
            unit = [
                row
                for row in rows
                if row["record_ordinal"] == record_ordinal
                and row["replicate"] == replicate
            ]
            assert len(unit) == 2
            for key, base in g3.STREAM_BASES.items():
                assert {row[key] for row in unit} == {
                    base + 8 * record_ordinal + replicate
                }
    first = {
        (key, row[key])
        for row in rows
        if row["record_ordinal"] == 0
        for key in g3.STREAM_BASES
    }
    second = {
        (key, row[key])
        for row in rows
        if row["record_ordinal"] == 1
        for key in g3.STREAM_BASES
    }
    assert first.isdisjoint(second)


def test_stream_audit_accepts_intended_reuse(monkeypatch, tmp_path):
    records = [
        {
            "partition": "train",
            "record_id": "r0",
            "root_cluster": "root0",
            "behavior_family": "family",
            "scale": "pre768",
            "target": 768,
            "state_sha1": "state0",
            "legal_action_ids": [0, 1, 2],
            "legal_actions": ["up", "down", "left"],
        }
    ]
    rows = g3.label_stream_rows(records)
    monkeypatch.setattr(
        g3.g1r,
        "historical_collision_union",
        lambda exclude_dir: ({}, [{"path": "fixture"}]),
    )
    audit = g3.stream_coupling_audit(rows, exclude_dir=tmp_path)
    assert audit["passes"]
    assert audit["record_replicate_units"] == 8
    assert audit["intended_action_arm_reuses"] == 16


def test_stream_audit_rejects_historical_collision(monkeypatch, tmp_path):
    records = [
        {
            "partition": "train",
            "record_id": "r0",
            "root_cluster": "root0",
            "behavior_family": "family",
            "scale": "pre768",
            "target": 768,
            "state_sha1": "state0",
            "legal_action_ids": [0],
            "legal_actions": ["up"],
        }
    ]
    rows = g3.label_stream_rows(records)
    monkeypatch.setattr(
        g3.g1r,
        "historical_collision_union",
        lambda exclude_dir: (
            {"deck_stream_id": {g3.STREAM_BASES["deck_stream_id"]}},
            [],
        ),
    )
    audit = g3.stream_coupling_audit(rows, exclude_dir=tmp_path)
    assert not audit["passes"]
    assert audit["collisions"]["deck_stream_id"]


def test_compatible_legacy_manifest_requires_exact_contract(tmp_path):
    required_row = {
        "record_id": "r0",
        "action": "up",
        "replicate": 0,
        "root_cluster": "root0",
        "state_sha1": "state0",
        "logical_seed": 57_000_000_000,
        "deck_stream_id": 58_000_000_000,
        "slot_stream_id": 59_000_000_000,
        "policy_stream_id": 60_000_000_000,
    }
    contract = {
        "root_manifest_file_sha256": g3.ROOT_MANIFEST_FILE_SHA256,
        "root_manifest_payload_sha256": g3.ROOT_MANIFEST_PAYLOAD_SHA256,
        "incumbent_policy_file_sha256": g3.INCUMBENT_FILE_SHA256,
        "continuation_policy": "frozen_incumbent_depth2",
        "horizons_from_one_h40_path": True,
        "terminal_right_censoring": True,
        "all_legal_actions": True,
        "replicates": g3.REPLICATES,
        "shared_action_arm_tapes": True,
        "path_provenance": [required_row],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"g3_compatible_label_contract": contract}))
    key = ("r0", "up", 0)
    covered, reports = g3._compatible_legacy_paths(
        [str(path)], {key: required_row}
    )
    assert covered == {key}
    assert reports[0]["compatible"]

    contract["shared_action_arm_tapes"] = False
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"g3_compatible_label_contract": contract}))
    covered, reports = g3._compatible_legacy_paths(
        [str(bad)], {key: required_row}
    )
    assert not covered
    assert not reports[0]["compatible"]


def test_metadata_search_does_not_treat_g2_inputs_as_legacy_labels():
    inventory = g3._legacy_candidate_paths()
    assert str(g3.G2_PREFLIGHT_PATH) in inventory[
        "root_manifest_file_hash_matches"
    ]
    assert not inventory["candidate_metadata_paths"]


def test_find_frame_requires_exactly_one_match():
    replay = {
        "frames": [
            {"index": 4, "state": {"board": [[0] * 4 for _ in range(4)]}}
        ]
    }
    assert g3._find_frame(replay, 4)["board"][0][0] == 0
    with pytest.raises(ValueError):
        g3._find_frame(replay, 5)


def test_one_frozen_ordinary_record_restores_exactly():
    manifest = g3._json(g3.ROOT_MANIFEST_PATH)
    row = next(
        record
        for record in manifest["records"]
        if record["partition"] == "train"
    )
    records, audit = g3.validate_ordinary_records([row])
    assert audit["passes"]
    assert len(records) == 1
    assert records[0]["legal_action_count"] >= 1
    assert len(records[0]["feature_rows_sha256"]) == 64


def test_all_32_transfer_records_restore_and_remain_unique():
    result = g3._json(g3.TRANSFER_RESULT_PATH)
    sources = g3._transfer_sources(result)
    records, audit = g3.validate_transfer_records(sources)
    assert audit["passes"]
    assert len(records) == 32
    assert audit["unique_roots"] == 32
    assert audit["roots_by_family"] == {
        "g2_transfer_corner2": 12,
        "g2_transfer_expectimax2": 1,
        "g2_transfer_phaseblend_incumbent": 19,
    }


def test_transfer_untouched_audit_fails_on_any_external_match(monkeypatch):
    calls = iter([["outside/root.json"], []])
    monkeypatch.setattr(g3, "_rg_matching_paths", lambda patterns, root: next(calls))
    audit = g3.transfer_untouched_audit(
        [{"root_cluster": "root", "state_sha1": "state"}]
    )
    assert not audit["passes"]


def test_n32_power_audit_is_frozen_and_reports_mde():
    audit = g3.power_audit_n32()
    assert audit["roots"] == 32
    assert audit["assumptions"]["draws"] == 10_000
    assert len(audit["rows"]) == len(g3.POWER_OR_GRID)
    assert audit["mde_grid_or"] in g3.POWER_OR_GRID
    passing = [
        row
        for row in audit["rows"]
        if row["target_policy_odds_ratio"] == audit["mde_grid_or"]
    ][0]
    assert passing["attainable"]
    assert passing["power_pass_point_or_1_25_and_ci"] >= 0.80


def test_cost_projection_is_compact_and_deterministic():
    projection = g3.cost_projection(20_000)
    assert projection["projected_incremental_bytes"] < 4 * 1024**3
    assert projection["conservative_seconds_per_path"] > 0
    assert projection["benchmark"]["new_tasks"] > 0


def test_immutable_writer_and_one_shot_output(tmp_path):
    path = tmp_path / "artifact.json"
    g3._write_immutable_json(path, {"a": 1})
    with pytest.raises(FileExistsError):
        g3._write_immutable_json(path, {"a": 2})
    out_dir = tmp_path / "already"
    out_dir.mkdir()
    with pytest.raises(FileExistsError):
        g3.run_preflight(out_dir)


def test_zero_forbidden_work_contract_is_present_in_source():
    source = g3.IMPLEMENTATION_PATH.read_text()
    for token in (
        '"new_labels": 0',
        '"label_values_opened": False',
        '"models_fit": 0',
        '"transfer_outcomes_opened": 0',
        '"policy_outcomes": 0',
        '"score_inspection": False',
    ):
        assert token in source
