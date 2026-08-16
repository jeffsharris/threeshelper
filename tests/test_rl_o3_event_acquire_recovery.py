from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from threes_rl import o3_event_acquire as source
from threes_rl import o3_event_acquire_recovery as recovery


def test_recovery_is_separate_and_original_hashes_are_frozen():
    assert recovery.OUTPUT_DIR != source.OUTPUT_DIR
    assert recovery.RUNNER_PATH != source.RUNNER_PATH
    assert recovery.TEST_PATH != source.TEST_PATH
    assert recovery.ORIGINAL_ARTIFACTS["runner"][1] == (
        "842fee2b41526d6c37770b7deee09500354e9140731753da905c1900e974bd5b"
    )
    assert recovery.ORIGINAL_ARTIFACTS["result"][1] == (
        "f7a967b936894a3d626055e366dc899d4efc52e9ad4b791b8c9a95d6e7fc791a"
    )


def test_real_complement_is_exact_and_content_blind():
    rows = recovery.derive_complement()
    assert len(rows) == 1_510
    assert all(row["role"] == "untouched_mechanism" for row in rows)
    assert {
        family: sum(row["family"] == family for row in rows)
        for family in recovery.FAMILY_ORDER
    } == {family: 302 for family in recovery.FAMILY_ORDER}
    assert {
        family: [
            row["game_index"] for row in rows if row["family"] == family
        ]
        for family in recovery.FAMILY_ORDER
    } == {
        family: list(range(3_798, 4_100))
        for family in recovery.FAMILY_ORDER
    }


def test_complement_rejects_missing_completed_metadata():
    planned = source.acquisition_rows()
    completed = recovery._read_jsonl_metadata(source.COMPLETION_PATH)[:-1]
    with pytest.raises(ValueError, match="complement mismatch"):
        recovery.derive_complement(
            acquisition_rows=planned,
            completion_rows=completed,
        )


def test_round_robin_chunk_order_and_membership():
    rows = recovery.derive_complement()
    chunks = recovery.round_robin_chunks(rows)
    assert len(chunks) == 302
    assert all(len(chunk) == 5 for chunk in chunks)
    assert [row["family"] for row in chunks[0]] == list(
        recovery.FAMILY_ORDER
    )
    assert {row["game_index"] for row in chunks[0]} == {3_798}
    assert {row["game_index"] for row in chunks[-1]} == {4_099}


def test_completion_schema_contains_no_outcome_fields():
    keys = recovery._completion_allowed_keys()
    assert "score" not in keys
    assert "action" not in keys
    assert "max_tile" not in keys
    assert "frames" not in keys
    assert "root_cluster" in keys
    assert "source_replay_sha256" in keys


def test_original_source_audit_hashes_without_parsing_replays(monkeypatch):
    original_read_text = Path.read_text

    def guarded_read_text(path: Path, *args, **kwargs):
        if recovery._is_within(path, source.REPLAY_DIR):
            raise AssertionError("original replay body was parsed")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    audit = recovery.audit_original_source()
    assert audit["passes"]
    assert audit["completion_count"] == 18_990
    assert audit["unique_ancestries"] == 18_990
    assert audit["unique_replay_hashes"] == 18_990
    assert audit["replay_bodies_parsed"] is False


def test_process_parser_ignores_ancestors_and_allows_services():
    table = "\n".join(
        [
            "10 1 python -m threes_rl.o3_event_acquire_recovery execute",
            "11 10 zsh wrapper",
            "20 1 python -m threes_rl.dashboard",
            "21 1 python -m threes_rl.human_play_server",
        ]
    )
    audit = recovery._parse_process_table(table, 11)
    assert audit["passes"]
    assert {row["classification"] for row in audit["candidate_processes"]} == {
        "current_process_or_ancestor",
        "allowed_dashboard_or_recorder",
    }


def test_process_parser_rejects_unrelated_heavy_python():
    table = "\n".join(
        [
            "10 1 python -m threes_rl.o3_event_acquire_recovery execute",
            "11 10 zsh wrapper",
            "30 1 python -m threes_rl.train_td --episodes 100",
        ]
    )
    audit = recovery._parse_process_table(table, 11)
    assert not audit["passes"]
    assert audit["disallowed_processes"] == [
        {
            "pid": 30,
            "ppid": 1,
            "command": "python -m threes_rl.train_td --episodes 100",
        }
    ]


def test_ownership_file_is_atomic_and_exclusive(tmp_path, monkeypatch):
    out = tmp_path / "recovery"
    out.mkdir()
    monkeypatch.setattr(recovery, "OUTPUT_DIR", out)
    monkeypatch.setattr(recovery, "OWNERSHIP_PATH", out / "ownership.json")
    recovery._create_ownership_file(out)
    with pytest.raises(FileExistsError):
        recovery._create_ownership_file(out)
    with recovery.execution_ownership():
        fd = os.open(recovery.OWNERSHIP_PATH, os.O_RDONLY)
        try:
            with pytest.raises(BlockingIOError):
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(fd)


def test_process_guard_evidence_is_append_only(tmp_path, monkeypatch):
    path = tmp_path / "guards.jsonl"
    monkeypatch.setattr(recovery, "PROCESS_GUARD_PATH", path)
    recovery._append_process_guard("first", {"passes": True})
    recovery._append_process_guard("second", {"passes": False})
    rows = recovery._read_jsonl_metadata(path)
    assert [row["stage"] for row in rows] == ["first", "second"]
    assert [row["passes"] for row in rows] == [True, False]


def test_collision_audit_does_not_parse_replay_content(tmp_path, monkeypatch):
    scan_root = tmp_path / "runs"
    replay_dir = scan_root / "sealed" / "source_replays"
    p0_dir = scan_root / "p0"
    out_dir = scan_root / "recovery"
    o2_dir = scan_root / "o2"
    replay_dir.mkdir(parents=True)
    p0_dir.mkdir()
    out_dir.mkdir()
    o2_dir.mkdir()
    (replay_dir / "bad.json").write_text("not-json")
    (p0_dir / "streams.json").write_text("not-json")
    (o2_dir / "support.json").write_text("not-json")
    (scan_root / "manifest.json").write_text(
        json.dumps({"logical_seed": 1})
    )
    monkeypatch.setattr(source, "REPLAY_DIR", replay_dir)
    monkeypatch.setattr(source, "P0_DIR", p0_dir)
    monkeypatch.setattr(recovery, "O2_FORBIDDEN_DIRS", (o2_dir,))
    monkeypatch.setattr(recovery, "ORIGINAL_COMPLETIONS", 1)
    row = {
        "logical_seed": 100,
        "deck_stream_id": 200,
        "slot_stream_id": 300,
        "policy_stream_id": 400,
    }
    audit = recovery.collision_audit(
        [row],
        out_dir=out_dir,
        scan_root=scan_root,
    )
    assert audit["passes"]
    assert audit["replay_bodies_parsed"] is False
    assert audit["exclusion_counts"][
        "original_replay_bytes_hash_bound_unread"
    ] == 1


def _synthetic_union_rows():
    planned = source.acquisition_rows()
    rows = []
    for index, row in enumerate(planned):
        item = {
            key: row[key]
            for key in (
                "family",
                "family_index",
                "game_index",
                "role",
                "planned_root_id",
                "logical_seed",
                "deck_stream_id",
                "slot_stream_id",
                "policy_stream_id",
            )
        }
        item.update(
            {
                "root_cluster": f"root-{index}",
                "source_replay": f"/tmp/replay-{index}.json",
                "source_replay_sha256": f"{index:064x}",
                "complete": True,
                "dashboard_eligible": False,
            }
        )
        rows.append(item)
    return rows


def test_union_contract_detects_exact_membership(monkeypatch, tmp_path):
    all_rows = _synthetic_union_rows()
    original = all_rows[:18_990]
    recovered = all_rows[18_990:]
    original_completion = tmp_path / "original.jsonl"
    original_attempts = tmp_path / "original_attempts.jsonl"
    recovery_completion = tmp_path / "recovery.jsonl"
    recovery_attempts = tmp_path / "recovery_attempts.jsonl"
    original_completion.write_text("completion-fixture\n")
    original_attempts.write_text("attempt-fixture\n")
    recovery_completion.write_text("completion-fixture\n")
    recovery_attempts.write_text("attempt-fixture\n")
    monkeypatch.setattr(source, "COMPLETION_PATH", original_completion)
    monkeypatch.setattr(source, "ATTEMPT_PATH", original_attempts)
    monkeypatch.setattr(recovery, "COMPLETION_PATH", recovery_completion)
    monkeypatch.setattr(recovery, "ATTEMPT_PATH", recovery_attempts)

    def read_fixture(path):
        if path == original_completion:
            return original
        if path == original_attempts:
            return [{}] * 37_980
        if path == recovery_attempts:
            return [{}] * 3_020
        raise AssertionError(f"Unexpected fixture read: {path}")

    monkeypatch.setattr(recovery, "_read_jsonl_metadata", read_fixture)
    source_audit = {
        "passes": True,
        "replay_byte_manifest_sha256": "frozen",
        "original_file_audit": {"passes": True},
    }
    combined, payload = recovery.build_union(recovered, source_audit)
    assert len(combined) == 20_500
    assert payload["passes"]
    assert payload["role_counts"] == {
        "development": 1_675,
        "train": 5_020,
        "untouched_mechanism": 13_805,
    }


def test_union_contract_fails_duplicate_root(monkeypatch, tmp_path):
    all_rows = _synthetic_union_rows()
    all_rows[-1]["root_cluster"] = all_rows[0]["root_cluster"]
    original = all_rows[:18_990]
    recovered = all_rows[18_990:]
    original_completion = tmp_path / "original.jsonl"
    original_attempts = tmp_path / "original_attempts.jsonl"
    recovery_completion = tmp_path / "recovery.jsonl"
    recovery_attempts = tmp_path / "recovery_attempts.jsonl"
    original_completion.write_text("completion-fixture\n")
    original_attempts.write_text("attempt-fixture\n")
    recovery_completion.write_text("completion-fixture\n")
    recovery_attempts.write_text("attempt-fixture\n")
    monkeypatch.setattr(source, "COMPLETION_PATH", original_completion)
    monkeypatch.setattr(source, "ATTEMPT_PATH", original_attempts)
    monkeypatch.setattr(recovery, "COMPLETION_PATH", recovery_completion)
    monkeypatch.setattr(recovery, "ATTEMPT_PATH", recovery_attempts)

    def read_fixture(path):
        if path == original_completion:
            return original
        if path == original_attempts:
            return [{}] * 37_980
        if path == recovery_attempts:
            return [{}] * 3_020
        raise AssertionError(f"Unexpected fixture read: {path}")

    monkeypatch.setattr(recovery, "_read_jsonl_metadata", read_fixture)
    _, payload = recovery.build_union(
        recovered,
        {
            "passes": True,
            "replay_byte_manifest_sha256": "frozen",
            "original_file_audit": {"passes": True},
        },
    )
    assert not payload["passes"]
    assert not payload["checks"]["unique_ancestries"]


def test_marker_loader_requires_ready_preflight(tmp_path, monkeypatch):
    marker = tmp_path / "marker.json"
    preflight = tmp_path / "preflight.json"
    result = tmp_path / "result.json"
    monkeypatch.setattr(recovery, "MARKER_PATH", marker)
    monkeypatch.setattr(recovery, "PREFLIGHT_RESULT_PATH", preflight)
    monkeypatch.setattr(recovery, "RESULT_PATH", result)
    monkeypatch.setattr(recovery, "_marker_identity", lambda: {"version": "x"})
    recovery._write_immutable_json(
        marker,
        {"version": "x"},
        self_hash_field="opened_payload_sha256",
    )
    recovery._write_immutable_json(
        preflight,
        {"decision": "HOLD_O3_ACQUISITION_RECOVERY_PREFLIGHT"},
        self_hash_field="result_payload_sha256",
    )
    with pytest.raises(ValueError, match="not READY"):
        recovery._load_marker()


def test_recovery_attempt_audit_rejects_retry(tmp_path, monkeypatch):
    rows = recovery.derive_complement()[:1]
    attempts = tmp_path / "attempts.jsonl"
    monkeypatch.setattr(recovery, "ATTEMPT_PATH", attempts)
    row = rows[0]
    for attempt_number in (0, 1):
        recovery._append_attempt(
            row,
            attempt_number=attempt_number,
            status="opened",
            chunk_index=0,
        )
        recovery._append_attempt(
            row,
            attempt_number=attempt_number,
            status="completed",
            chunk_index=0,
        )
    completion = {
        "family": row["family"],
        "game_index": row["game_index"],
    }
    audit = recovery.recovery_attempt_audit(rows, [completion])
    assert not audit["passes"]
    assert not audit["checks"]["one_attempt_per_root"]
