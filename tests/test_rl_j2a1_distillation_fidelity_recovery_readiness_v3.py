from __future__ import annotations

import ast
import copy
import json
import math
from pathlib import Path

import pytest

from threes_rl import (
    j2a1_distillation_fidelity_recovery_readiness_v3 as v3,
)


HEX_A = "a" * 64
HEX_B = "b" * 64
HEX_C = "c" * 64
HEX_D = "d" * 64
CONTRACT = "1" * 64


def _attempt_records(
    *,
    roots: list[str],
    started: float = 100.0,
    ended: float = 110.0,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    predecessor = None
    starts = []
    for root in roots:
        unit_id = f"teacher_root|teacher_behavior_cloning|{root}"
        record = v3.payload_with_hash(
            {
                "version": "fixture_attempt_v1",
                "sequence": len(records),
                "predecessor_record_sha256": predecessor,
                "contract_sha256": CONTRACT,
                "event": "started",
                "unit_id": unit_id,
                "unit_type": "teacher_root",
                "attempt_id": f"{unit_id}|attempt=0",
                "wall_started_at": started,
            },
            "attempt_record_sha256",
        )
        records.append(record)
        starts.append(record)
        predecessor = record["attempt_record_sha256"]
    for row_index, (root, start) in enumerate(zip(roots, starts)):
        path = (
            "teacher_roots/teacher_behavior_cloning/"
            f"{row_index:05d}_{root}.bin"
        )
        record = v3.payload_with_hash(
            {
                "version": "fixture_attempt_v1",
                "sequence": len(records),
                "predecessor_record_sha256": predecessor,
                "contract_sha256": CONTRACT,
                "event": "finished",
                "unit_id": start["unit_id"],
                "unit_type": "teacher_root",
                "attempt_id": start["attempt_id"],
                "start_sha256": start["attempt_record_sha256"],
                "wall_ended_at": ended,
                "charged_seconds": ended - started,
                "output_identity": {
                    "root_id": root,
                    "ancestry_id": f"{row_index + 10:064x}",
                    "row_index": row_index,
                    "stage": "teacher_behavior_cloning",
                    "path": path,
                    "bytes": 3,
                    "file_sha256": f"{row_index + 20:064x}",
                    "root_content_sha256": f"{row_index + 30:064x}",
                },
            },
            "attempt_record_sha256",
        )
        records.append(record)
        predecessor = record["attempt_record_sha256"]
    return records


def _completion_record(
    *,
    sequence: int,
    predecessor: str | None,
    row: dict[str, object],
    file_sha: str,
    content_sha: str,
) -> dict[str, object]:
    return v3.payload_with_hash(
        {
            "version": "fixture_completion_v1",
            "sequence": sequence,
            "predecessor_record_sha256": predecessor,
            "contract_sha256": CONTRACT,
            "kind": "teacher_root",
            "root_id": row["root_id"],
            "ancestry_id": row["ancestry_id"],
            "row_index": row["row_index"],
            "stage": row["stage"],
            "relative_path": (
                "teacher_roots/teacher_behavior_cloning/"
                f"{row['row_index']:05d}_{row['root_id']}.bin"
            ),
            "file_sha256": file_sha,
            "content_sha256": content_sha,
            "recovered_orphan": False,
        },
        "completion_record_sha256",
    )


def _rows(count: int = 4) -> list[dict[str, object]]:
    return [
        {
            "row_index": index,
            "root_id": f"{index + 1:064x}",
            "ancestry_id": f"{index + 101:064x}",
            "stage": "teacher_behavior_cloning",
            "streams": {
                "logical_stream_id": 227_000_000_000 + index,
                "deck_stream_id": 228_000_000_000 + index,
                "slot_stream_id": 229_000_000_000 + index,
                "teacher_policy_stream_id": 230_000_000_000 + index,
            },
            "reserved": True,
            "consumed": True,
            "content_opened": False,
        }
        for index in range(count)
    ]


def _recovery_inputs(
    rows: list[dict[str, object]],
    completed_indices: list[int],
) -> tuple[
    dict[str, dict[str, object]],
    dict[str, dict[str, object]],
    list[dict[str, object]],
]:
    completions: dict[str, dict[str, object]] = {}
    outputs: dict[str, dict[str, object]] = {}
    retained: list[dict[str, object]] = []
    predecessor = None
    for sequence, index in enumerate(completed_indices):
        row = rows[index]
        file_sha = f"{index + 201:064x}"
        content_sha = f"{index + 301:064x}"
        completion = _completion_record(
            sequence=sequence,
            predecessor=predecessor,
            row=row,
            file_sha=file_sha,
            content_sha=content_sha,
        )
        predecessor = completion["completion_record_sha256"]
        path = completion["relative_path"]
        completions[str(row["root_id"])] = completion
        outputs[str(row["root_id"])] = {
            "root_id": row["root_id"],
            "ancestry_id": row["ancestry_id"],
            "row_index": row["row_index"],
            "stage": row["stage"],
            "path": path,
            "bytes": 3,
            "file_sha256": file_sha,
            "root_content_sha256": content_sha,
        }
        retained.append(
            {
                "path": path,
                "bytes": 3,
                "file_sha256": file_sha,
            }
        )
    return completions, outputs, retained


def test_eight_collectors_charge_wall_not_aggregate() -> None:
    projection = v3.wall_clock_projection(
        completed_roots=8,
        total_roots=16,
        earliest_start=100.0,
        latest_finish=110.0,
        aggregate_worker_seconds=80.0,
    )
    assert projection["observed_wall_seconds"] == 10.0
    assert projection["aggregate_worker_seconds_descriptive"] == 80.0
    assert projection["projected_total_stage_a_wall_hours"] == pytest.approx(
        20.0 / 3600.0
    )
    assert projection["checks"]["worker_seconds_descriptive_only"]


def test_wall_projection_reproduces_authoritative_v2_pace() -> None:
    projection = v3.wall_clock_projection(
        completed_roots=v3.COMPLETED_ROOTS,
        total_roots=v3.ACTIVE_ROOTS,
        earliest_start=v3.EXPECTED_EARLIEST_START,
        latest_finish=v3.EXPECTED_LATEST_FINISH,
        aggregate_worker_seconds=v3.EXPECTED_WORKER_SECONDS,
    )
    assert projection["remaining_roots"] == 11_288
    assert projection["observed_wall_hours"] == pytest.approx(
        9.05785490612189
    )
    assert projection["projected_total_stage_a_wall_hours"] == pytest.approx(
        42.602824125381694
    )
    assert projection[
        "conservative_total_stage_a_wall_hours"
    ] == pytest.approx(50.989066430196644)
    assert projection["passes"]


def test_wall_projection_rejects_worker_time_as_wall() -> None:
    with pytest.raises(v3.J2A1V3IntegrityError):
        v3.wall_clock_projection(
            completed_roots=8,
            total_roots=16,
            earliest_start=100.0,
            latest_finish=200.0,
            aggregate_worker_seconds=99.0,
        )


def test_attempt_audit_reproduces_eight_collector_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roots = [f"{index + 1:064x}" for index in range(8)]
    records = _attempt_records(roots=roots)
    monkeypatch.setattr(v3, "ATTEMPT_RECORDS", 16)
    monkeypatch.setattr(v3, "COMPLETED_ROOTS", 8)
    monkeypatch.setattr(v3, "EXPECTED_WORKER_SECONDS", 80.0)
    monkeypatch.setattr(v3, "EXPECTED_EARLIEST_START", 100.0)
    monkeypatch.setattr(v3, "EXPECTED_LATEST_FINISH", 110.0)
    monkeypatch.setattr(v3, "EXPECTED_WALL_SECONDS", 10.0)
    audit = v3.audit_attempt_records(records)
    assert audit["aggregate_worker_seconds_descriptive"] == 80.0
    assert audit["top_level_wall_span_seconds"] == 10.0
    assert audit["attempts_abandoned"] == 0


def test_attempt_audit_rejects_hidden_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roots = [HEX_A, HEX_A]
    records = _attempt_records(roots=roots)
    monkeypatch.setattr(v3, "ATTEMPT_RECORDS", 4)
    monkeypatch.setattr(v3, "COMPLETED_ROOTS", 2)
    with pytest.raises(v3.J2A1V3IntegrityError, match="retry|repeats"):
        v3.audit_attempt_records(records)


def test_attempt_audit_rejects_tampered_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = _attempt_records(roots=[HEX_A])
    records[1]["charged_seconds"] = 9.0
    records[1] = v3.payload_with_hash(
        records[1],
        "attempt_record_sha256",
    )
    monkeypatch.setattr(v3, "ATTEMPT_RECORDS", 2)
    monkeypatch.setattr(v3, "COMPLETED_ROOTS", 1)
    with pytest.raises(v3.J2A1V3IntegrityError):
        v3.audit_attempt_records(records)


def test_attempt_audit_rejects_scientific_key_even_if_rehashed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = _attempt_records(roots=[HEX_A])
    records[1]["score"] = 123
    records[1] = v3.payload_with_hash(
        records[1],
        "attempt_record_sha256",
    )
    monkeypatch.setattr(v3, "ATTEMPT_RECORDS", 2)
    monkeypatch.setattr(v3, "COMPLETED_ROOTS", 1)
    with pytest.raises(v3.J2A1V3IntegrityError, match="keys changed"):
        v3.audit_attempt_records(records)


def test_completion_audit_rejects_extra_family_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _rows(1)[0]
    record = _completion_record(
        sequence=0,
        predecessor=None,
        row=row,
        file_sha=HEX_B,
        content_sha=HEX_C,
    )
    record["family"] = "forbidden"
    record = v3.payload_with_hash(record, "completion_record_sha256")
    monkeypatch.setattr(v3, "COMPLETED_ROOTS", 1)
    with pytest.raises(v3.J2A1V3IntegrityError, match="keys changed"):
        v3.audit_completion_records([record])


def test_completion_audit_rejects_duplicate_ancestry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _rows(2)
    rows[1]["ancestry_id"] = rows[0]["ancestry_id"]
    first = _completion_record(
        sequence=0,
        predecessor=None,
        row=rows[0],
        file_sha=HEX_B,
        content_sha=HEX_C,
    )
    second = _completion_record(
        sequence=1,
        predecessor=first["completion_record_sha256"],
        row=rows[1],
        file_sha=HEX_D,
        content_sha="e" * 64,
    )
    monkeypatch.setattr(v3, "COMPLETED_ROOTS", 2)
    with pytest.raises(v3.J2A1V3IntegrityError, match="repeats"):
        v3.audit_completion_records([first, second])


def test_recovery_set_difference_is_canonical_and_out_of_order_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _rows(4)
    completions, outputs, retained = _recovery_inputs(rows, [3, 1])
    monkeypatch.setattr(v3, "COMPLETED_ROOTS", 2)
    monkeypatch.setattr(v3, "REMAINING_ROOTS", 2)
    first = v3.derive_recovery_authority(
        rows,
        completions,
        outputs,
        retained,
    )
    second = v3.derive_recovery_authority(
        rows,
        dict(reversed(list(completions.items()))),
        dict(reversed(list(outputs.items()))),
        list(reversed(retained)),
    )
    assert first["unfinished_rows_sha256"] == second[
        "unfinished_rows_sha256"
    ]
    assert [row["row_index"] for row in first["unfinished_rows"]] == [0, 2]
    assert [row["row_index"] for row in first["completed_refs"]] == [1, 3]


def test_crash_restart_reconstructs_exact_same_recovery_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _rows(4)
    uninterrupted = _recovery_inputs(rows, [1, 3])
    before_crash = _recovery_inputs(rows, [1])
    # The restart sees only authenticated metadata reloaded from durable JSON.
    durable_completion = json.loads(
        json.dumps(before_crash[0], sort_keys=True)
    )
    durable_outputs = json.loads(json.dumps(before_crash[1], sort_keys=True))
    durable_retention = json.loads(
        json.dumps(before_crash[2], sort_keys=True)
    )
    tail_root = str(rows[3]["root_id"])
    durable_completion[tail_root] = copy.deepcopy(
        uninterrupted[0][tail_root]
    )
    durable_outputs[tail_root] = copy.deepcopy(uninterrupted[1][tail_root])
    durable_retention.append(copy.deepcopy(uninterrupted[2][1]))
    monkeypatch.setattr(v3, "COMPLETED_ROOTS", 2)
    monkeypatch.setattr(v3, "REMAINING_ROOTS", 2)
    expected = v3.derive_recovery_authority(rows, *uninterrupted)
    resumed = v3.derive_recovery_authority(
        rows,
        durable_completion,
        durable_outputs,
        durable_retention,
    )
    assert resumed["completed_refs_sha256"] == expected[
        "completed_refs_sha256"
    ]
    assert resumed["unfinished_rows_sha256"] == expected[
        "unfinished_rows_sha256"
    ]
    assert resumed["unfinished_rows"] == expected["unfinished_rows"]


def test_recovery_set_difference_rejects_wrong_stream_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _rows(2)
    completions, outputs, retained = _recovery_inputs(rows, [0])
    outputs[str(rows[0]["root_id"])]["row_index"] = 1
    monkeypatch.setattr(v3, "COMPLETED_ROOTS", 1)
    monkeypatch.setattr(v3, "REMAINING_ROOTS", 1)
    with pytest.raises(v3.J2A1V3IntegrityError, match="cross-binding"):
        v3.derive_recovery_authority(
            rows,
            completions,
            outputs,
            retained,
        )


def test_recovery_set_difference_rejects_replacement_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _rows(2)
    completions, outputs, retained = _recovery_inputs(rows, [0])
    completion = completions.pop(str(rows[0]["root_id"]))
    completions[HEX_A] = {**completion, "root_id": HEX_A}
    monkeypatch.setattr(v3, "COMPLETED_ROOTS", 1)
    monkeypatch.setattr(v3, "REMAINING_ROOTS", 1)
    with pytest.raises(v3.J2A1V3IntegrityError):
        v3.derive_recovery_authority(
            rows,
            completions,
            outputs,
            retained,
        )


def test_hash_only_retention_never_decodes_blob(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_dir = tmp_path / "teacher_roots" / "teacher_behavior_cloning"
    root_dir.mkdir(parents=True)
    blob = root_dir / f"00000_{HEX_A}.bin"
    blob.write_bytes(b"\x80not-json-or-pickle")
    file_row = {
        "path": str(blob.relative_to(tmp_path)),
        "bytes": blob.stat().st_size,
        "file_sha256": v3.sha256_path(blob),
    }
    retention = {
        "files": [file_row],
        "file_count": 1,
        "retained_bytes": file_row["bytes"],
        "canonical_inventory_sha256": v3.canonical_json_hash([file_row]),
        "passes": True,
    }
    monkeypatch.setattr(
        v3,
        "EXPECTED_RETENTION",
        {
            "file_count": 1,
            "retained_bytes": file_row["bytes"],
            "canonical_inventory_sha256":
                retention["canonical_inventory_sha256"],
        },
    )
    monkeypatch.setattr(v3, "COMPLETED_ROOTS", 1)
    calls: list[Path] = []

    def hash_only(path: str | Path) -> str:
        calls.append(Path(path))
        return v3.sha256_path(path)

    audit = v3.hash_only_retention_audit(
        retention,
        execution_dir=tmp_path,
        hash_file=hash_only,
    )
    assert calls == [blob]
    assert audit["root_body_deserializations"] == 0
    assert audit["body_access"] == "streaming SHA-256 and byte count only"


def test_hash_only_retention_rejects_changed_blob(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_dir = tmp_path / "teacher_roots"
    root_dir.mkdir()
    blob = root_dir / "root.bin"
    blob.write_bytes(b"original")
    row = {
        "path": "teacher_roots/root.bin",
        "bytes": len(b"original"),
        "file_sha256": v3.sha256_path(blob),
    }
    retention = {
        "files": [row],
        "file_count": 1,
        "retained_bytes": row["bytes"],
        "canonical_inventory_sha256": v3.canonical_json_hash([row]),
        "passes": True,
    }
    monkeypatch.setattr(
        v3,
        "EXPECTED_RETENTION",
        {
            "file_count": 1,
            "retained_bytes": row["bytes"],
            "canonical_inventory_sha256":
                retention["canonical_inventory_sha256"],
        },
    )
    monkeypatch.setattr(v3, "COMPLETED_ROOTS", 1)
    blob.write_bytes(b"changed!")
    with pytest.raises(v3.J2A1V3IntegrityError):
        v3.hash_only_retention_audit(
            retention,
            execution_dir=tmp_path,
        )


def test_owner_create_once_live_reject_and_dead_reclaim(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "ownership.jsonl"
    first = v3.acquire_or_reclaim_recovery_owner(
        ledger_path=ledger,
        marker_sha256=HEX_A,
        authority_sha256=HEX_B,
        command="recover",
        pid=100,
        process_start_identity="start-a",
        is_live=lambda _: False,
        commit_head_sha256=None,
    )
    assert not first["reclaimed"]
    original_bytes = ledger.read_bytes()
    with pytest.raises(v3.J2A1V3OperationalHold):
        v3.acquire_or_reclaim_recovery_owner(
            ledger_path=ledger,
            marker_sha256=HEX_A,
            authority_sha256=HEX_B,
            command="recover",
            pid=101,
            process_start_identity="start-b",
            is_live=lambda _: True,
            commit_head_sha256=HEX_C,
        )
    assert ledger.read_bytes() == original_bytes
    reclaimed = v3.acquire_or_reclaim_recovery_owner(
        ledger_path=ledger,
        marker_sha256=HEX_A,
        authority_sha256=HEX_B,
        command="recover",
        pid=102,
        process_start_identity="start-c",
        is_live=lambda _: False,
        commit_head_sha256=HEX_C,
    )
    assert reclaimed["reclaimed"]
    records = v3._read_owner_ledger(ledger)
    assert len(records) == 2
    assert records[1]["recovered_owner_record_sha256"] == records[0][
        "owner_record_sha256"
    ]


def test_owner_reclaim_wrong_marker_or_authority_fails(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "ownership.jsonl"
    v3.acquire_or_reclaim_recovery_owner(
        ledger_path=ledger,
        marker_sha256=HEX_A,
        authority_sha256=HEX_B,
        command="recover",
        pid=100,
        process_start_identity="start-a",
        is_live=lambda _: False,
        commit_head_sha256=None,
    )
    before = ledger.read_bytes()
    with pytest.raises(v3.J2A1V3IntegrityError, match="contract"):
        v3.acquire_or_reclaim_recovery_owner(
            ledger_path=ledger,
            marker_sha256=HEX_C,
            authority_sha256=HEX_B,
            command="recover",
            pid=101,
            process_start_identity="start-b",
            is_live=lambda _: False,
            commit_head_sha256=HEX_D,
        )
    assert ledger.read_bytes() == before


def test_owner_reclaim_requires_authenticated_commit_head(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "ownership.jsonl"
    v3.acquire_or_reclaim_recovery_owner(
        ledger_path=ledger,
        marker_sha256=HEX_A,
        authority_sha256=HEX_B,
        command="recover",
        pid=100,
        process_start_identity="start-a",
        is_live=lambda _: False,
        commit_head_sha256=None,
    )
    with pytest.raises(v3.J2A1V3IntegrityError, match="commit head"):
        v3.acquire_or_reclaim_recovery_owner(
            ledger_path=ledger,
            marker_sha256=HEX_A,
            authority_sha256=HEX_B,
            command="recover",
            pid=101,
            process_start_identity="start-b",
            is_live=lambda _: False,
            commit_head_sha256=None,
        )


def test_immutable_json_create_once_and_tamper_reject(
    tmp_path: Path,
) -> None:
    path = tmp_path / "artifact.json"
    payload = {"version": "fixture", "count": 1}
    first = v3.write_immutable_json(path, payload, field="payload_sha256")
    original = path.read_bytes()
    second = v3.write_immutable_json(path, payload, field="payload_sha256")
    assert first == second
    assert path.read_bytes() == original
    with pytest.raises(v3.J2A1V3IntegrityError):
        v3.write_immutable_json(
            path,
            {"version": "fixture", "count": 2},
            field="payload_sha256",
        )


def test_cli_is_readiness_only_and_import_light() -> None:
    parser = v3.build_parser()
    choices = parser._subparsers._group_actions[0].choices
    assert set(choices) == {
        "audit-zero-work",
        "write-test-evidence",
        "prepare",
    }
    tree = ast.parse(v3.RUNNER_PATH.read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    )
    assert not any(
        forbidden in module
        for module in imported
        for forbidden in (
            "torch",
            "numpy",
            "sim",
            "eval",
            "expectimax",
            "teacher",
        )
    )


def test_schema_has_no_execution_route_and_all_counters_zero() -> None:
    schema = v3.schema_payload()
    assert schema["public_commands"] == [
        "audit-zero-work",
        "write-test-evidence",
        "prepare",
    ]
    assert not schema["future_execution"]["authorized"]
    assert not schema["read_boundary"]["root_blob_deserialization"]
    assert all(value == 0 for value in schema["zero_work"].values())


def test_actual_v2_metadata_authority_audit_without_body_decode() -> None:
    retention = v3.load_json(
        v3.V2_EXECUTION_DIR
        / "J2A1_V2_DISTILLATION_FIDELITY_RETENTION.json"
    )
    hashes = {
        str(v3.V2_EXECUTION_DIR / row["path"]): row["file_sha256"]
        for row in retention["files"]
        if row["path"].startswith("teacher_roots/")
    }
    integrity, authority, projection = (
        v3.v2_integrity_and_authority_audit(
            hash_file=lambda path: hashes[str(path)]
        )
    )
    assert integrity["attempt_ledger"]["record_count"] == 6_096
    assert integrity["completion_ledger"]["completed"] == 3_048
    assert integrity["retention"]["root_blob_count"] == 3_048
    assert integrity["scientific_body_deserializations"] == 0
    assert authority["total_rows"] == 14_336
    assert authority["completed_rows"] == 3_048
    assert authority["unfinished_rows_count"] == 11_288
    assert projection["passes"]


def test_prepare_seals_nine_create_once_artifacts_without_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        v3,
        "_source_identities",
        lambda: {"fixture": HEX_A},
    )
    monkeypatch.setattr(
        v3,
        "source_and_input_bindings",
        lambda: {"version": "bindings", "passes": True},
    )
    authority = {
        "version": "authority",
        "total_rows": 14_336,
        "total_streams": 63_488,
        "completed_rows": 3_048,
        "unfinished_rows_count": 11_288,
        "completed_refs_sha256": HEX_A,
        "unfinished_rows_sha256": HEX_B,
        "passes": True,
    }
    projection = {
        "version": "projection",
        "projected_total_stage_a_wall_hours": 42.6,
        "conservative_total_stage_a_wall_hours": 51.0,
        "passes": True,
    }
    monkeypatch.setattr(
        v3,
        "v2_integrity_and_authority_audit",
        lambda hash_file=v3.sha256_path: (
            {"version": "integrity", "passes": True},
            authority,
            projection,
        ),
    )
    monkeypatch.setattr(
        v3,
        "operational_audit",
        lambda **_: {
            "version": "operations",
            "passes": True,
            "checks": {"fixture": True},
        },
    )
    monkeypatch.setattr(
        v3,
        "FUTURE_EXECUTION_DIR",
        tmp_path / "future",
    )
    record = {
        "command": "pytest fixture",
        "passed": 1,
        "failed": 0,
        "deselected": 0,
        "note": "synthetic",
    }
    v3.write_test_evidence(tmp_path, [record])
    result = v3.prepare(tmp_path, include_services=False)
    assert result["decision"] == v3.READY
    assert not result["execution_authorized"]
    assert len(list(tmp_path.iterdir())) == 9
    assert all(value == 0 for value in result["result"]["zero_work"].values())
    with pytest.raises(v3.J2A1V3IntegrityError, match="boundary"):
        v3.prepare(tmp_path, include_services=False)


def test_prepare_holds_on_operational_gate_without_authorizing_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        v3,
        "_source_identities",
        lambda: {"fixture": HEX_A},
    )
    monkeypatch.setattr(
        v3,
        "source_and_input_bindings",
        lambda: {"version": "bindings", "passes": True},
    )
    monkeypatch.setattr(
        v3,
        "v2_integrity_and_authority_audit",
        lambda hash_file=v3.sha256_path: (
            {"version": "integrity", "passes": True},
            {
                "version": "authority",
                "total_rows": 14_336,
                "total_streams": 63_488,
                "completed_rows": 3_048,
                "unfinished_rows_count": 11_288,
                "completed_refs_sha256": HEX_A,
                "unfinished_rows_sha256": HEX_B,
                "passes": True,
            },
            {
                "version": "projection",
                "projected_total_stage_a_wall_hours": 42.6,
                "conservative_total_stage_a_wall_hours": 51.0,
                "passes": True,
            },
        ),
    )
    monkeypatch.setattr(
        v3,
        "operational_audit",
        lambda **_: {
            "version": "operations",
            "passes": False,
            "checks": {"disk": False},
        },
    )
    monkeypatch.setattr(
        v3,
        "FUTURE_EXECUTION_DIR",
        tmp_path / "future",
    )
    v3.write_test_evidence(
        tmp_path,
        [
            {
                "command": "pytest fixture",
                "passed": 1,
                "failed": 0,
                "deselected": 0,
                "note": "synthetic",
            }
        ],
    )
    result = v3.prepare(tmp_path, include_services=False)
    assert result["decision"] == v3.HOLD
    assert not result["execution_authorized"]
    assert result["result"]["hold"]
    assert not result["result"]["collectors_authorized"]


def test_zero_work_audit_rejects_future_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    future = tmp_path / "future"
    future.mkdir()
    monkeypatch.setattr(v3, "FUTURE_EXECUTION_DIR", future)
    monkeypatch.setattr(
        v3,
        "V2_EXECUTION_DIR",
        tmp_path,
    )
    audit = v3.audit_zero_work(tmp_path / "readiness", include_services=False)
    assert not audit["passes"]
    assert not audit["checks"]["future_execution_absent"]


def test_test_evidence_requires_zero_failures_and_current_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        v3,
        "_source_identities",
        lambda: {"fixture": HEX_A},
    )
    with pytest.raises(v3.J2A1V3IntegrityError):
        v3.test_evidence_payload(
            [
                {
                    "command": "pytest",
                    "passed": 1,
                    "failed": 1,
                    "deselected": 0,
                    "note": "bad",
                }
            ]
        )
    payload = v3.write_test_evidence(
        tmp_path,
        [
            {
                "command": "pytest",
                "passed": 2,
                "failed": 0,
                "deselected": 1,
                "note": "good",
            }
        ],
    )
    assert payload["passes"]
    monkeypatch.setattr(
        v3,
        "_source_identities",
        lambda: {"fixture": HEX_B},
    )
    with pytest.raises(v3.J2A1V3IntegrityError, match="changed"):
        v3._load_test_evidence(tmp_path / v3.TEST_EVIDENCE_NAME)


def test_v2_constants_and_future_namespace_are_exact() -> None:
    assert v3.ACTIVE_ROOTS == 14_336
    assert v3.ACTIVE_STREAMS == 63_488
    assert v3.COMPLETED_ROOTS == 3_048
    assert v3.REMAINING_ROOTS == 11_288
    assert v3.ATTEMPT_RECORDS == 6_096
    assert v3.COLLECTORS == 8
    assert v3.RUNTIME_CAP_HOURS == 72.0
    assert not v3.FUTURE_EXECUTION_DIR.exists()


def test_no_scientific_terms_appear_in_recovery_authority_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _rows(2)
    completions, outputs, retained = _recovery_inputs(rows, [0])
    monkeypatch.setattr(v3, "COMPLETED_ROOTS", 1)
    monkeypatch.setattr(v3, "REMAINING_ROOTS", 1)
    authority = v3.derive_recovery_authority(
        rows,
        completions,
        outputs,
        retained,
    )
    encoded = json.dumps(authority)
    for forbidden in (
        '"score"',
        '"action"',
        '"label"',
        '"family"',
        '"metric"',
        '"board"',
        '"transitions"',
    ):
        assert forbidden not in encoded
    assert authority["scientific_content_opened"] == 0
    assert authority["duplicate_stream_consumptions"] == 0
