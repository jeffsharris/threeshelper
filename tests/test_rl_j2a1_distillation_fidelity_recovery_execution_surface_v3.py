from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import pytest

from threes_rl import (
    j2a1_distillation_fidelity_recovery_execution_surface_v3 as surface,
)


HEX_A = "a" * 64
HEX_B = "b" * 64
HEX_C = "c" * 64
HEX_D = "d" * 64
CONTRACT = "1" * 64


class _Clock:
    def __init__(self, values: list[float]) -> None:
        self.values = iter(values)

    def __call__(self) -> float:
        return float(next(self.values))


def _identity(value: str) -> dict[str, object]:
    return {
        "path": f"/fixture/{value}.json",
        "bytes": 1,
        "file_sha256": value * 64,
        "payload_field": "payload_sha256",
        "payload_sha256": value * 64,
    }


def _chain() -> dict[str, object]:
    return {
        "lock": {
            "identity": _identity("1"),
            "payload": {"authorization": _identity("4")},
        },
        "marker_identity": _identity("2"),
        "manifest_identity": _identity("3"),
    }


def _rows(count: int = 4) -> list[dict[str, object]]:
    return [
        {
            "row_index": index,
            "root_id": f"{index + 1:064x}",
            "ancestry_id": f"{index + 101:064x}",
            "stage": (
                "teacher_behavior_cloning"
                if index < max(1, count // 2)
                else "distillation_validation"
            ),
            "streams": {
                "logical_stream_id": 227_000_000_000 + index,
                "deck_stream_id": 228_000_000_000 + index,
                "slot_stream_id": 229_000_000_000 + index,
                "teacher_policy_stream_id": 230_000_000_000 + index,
            },
            "reserved": False,
            "consumed": False,
            "content_opened": False,
        }
        for index in range(count)
    ]


def _completion_ref(
    row: dict[str, object],
    *,
    suffix: int,
) -> dict[str, object]:
    return {
        "root_id": row["root_id"],
        "ancestry_id": row["ancestry_id"],
        "row_index": row["row_index"],
        "stage": row["stage"],
        "relative_path": f"teacher_roots/{row['root_id']}.bin",
        "bytes": 10,
        "file_sha256": f"{suffix + 100:064x}",
        "content_sha256": f"{suffix + 200:064x}",
    }


def _authorization_body(out_dir: Path) -> dict[str, object]:
    return {
        "version": "fixture_authorization_v1",
        "decision": surface.AUTHORIZATION_DECISION,
        "execution_mode": "scientific",
        "scientific_authority": True,
        "execution_root": str(out_dir.resolve()),
        "jobs": 1,
        "authorized_commands": [
            "seal-phase-lock",
            "open",
            "materialize",
            "execute",
        ],
        "execution_authorized": True,
        "ppo_authorized": False,
        "development_authorized": False,
        "confirmation_authorized": False,
        "promotion_authorized": False,
    }


def _patch_phase_chain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    rows: list[dict[str, object]] | None = None,
) -> tuple[Path, Path, Path, list[dict[str, object]]]:
    if rows is None:
        rows = _rows(2)
    out_dir = tmp_path / "execution"
    readiness_dir = tmp_path / "readiness"
    authorization_path = tmp_path / "authorization.json"
    completed_row = copy.deepcopy(_rows(1)[0])
    completed_row["root_id"] = "f" * 64
    completed_row["ancestry_id"] = "e" * 64
    completed = [_completion_ref(completed_row, suffix=1)]
    unfinished_sha = surface.canonical_json_hash(rows)
    completed_sha = surface.canonical_json_hash(completed)
    monkeypatch.setattr(surface, "RECOVERY_ROOTS", len(rows))
    monkeypatch.setattr(
        surface,
        "V2_COMPLETED_ROOTS",
        len(completed),
    )
    monkeypatch.setattr(
        surface,
        "ACTIVE_ROOTS",
        len(rows) + len(completed),
    )
    monkeypatch.setattr(
        surface,
        "EXPECTED_UNFINISHED_SHA256",
        unfinished_sha,
    )
    monkeypatch.setattr(
        surface,
        "EXPECTED_COMPLETED_REFS_SHA256",
        completed_sha,
    )
    monkeypatch.setattr(
        surface,
        "authorization_payload",
        lambda **_kwargs: _authorization_body(out_dir),
    )
    monkeypatch.setattr(
        surface,
        "verify_readiness_package",
        lambda _path: {
            "identities": {
                key: _identity(str(index + 5))
                for index, key in enumerate(surface.READINESS_FIELDS)
            },
            "passes": True,
        },
    )
    monkeypatch.setattr(
        surface,
        "operational_audit",
        lambda **_kwargs: {"passes": True, "checks": {"fixture": True}},
    )
    monkeypatch.setattr(
        surface,
        "load_recovery_authority",
        lambda: {
            "unfinished_rows": copy.deepcopy(rows),
            "completed_refs": copy.deepcopy(completed),
            "completed_refs_sha256": completed_sha,
        },
    )
    surface.write_immutable_json(
        authorization_path,
        _authorization_body(out_dir),
        field="authorization_payload_sha256",
    )
    return readiness_dir, authorization_path, out_dir, rows


def test_authoritative_preflight_sources_and_nine_artifacts_are_exact() -> None:
    audit = surface.source_and_parent_audit(
        require_future_absent=True,
    )
    assert audit["passes"]
    assert len(audit["preflight_artifacts"]) == 9
    assert all(
        row["passes"] for row in audit["preflight_sources"].values()
    )


def test_recovery_authority_is_exact_and_content_blind() -> None:
    audit = surface.authority_audit()
    assert audit["v2_completed_roots"] == 3_048
    assert audit["recovery_roots"] == 11_288
    assert audit["total_authority_roots"] == 14_336
    assert audit["recovery_streams"] == 51_296
    assert audit["completed_bytes_rehashed"] == 3_048
    assert audit["root_body_deserializations"] == 0
    assert audit["family_reads"] == 0
    assert all(audit["checks"].values())


def test_recovery_authority_uses_exact_prefix_and_unfinished_order() -> None:
    authority = surface.load_recovery_authority()
    completed = authority["completed_refs"]
    unfinished = authority["unfinished_rows"]
    assert [row["row_index"] for row in completed] == list(range(3_048))
    bc = [
        row
        for row in unfinished
        if row["stage"] == "teacher_behavior_cloning"
    ]
    validation = [
        row
        for row in unfinished
        if row["stage"] == "distillation_validation"
    ]
    assert [row["row_index"] for row in bc] == list(range(3_048, 8_192))
    assert [row["row_index"] for row in validation] == list(range(6_144))
    assert all(
        int(row["row_index"]) % 8 in range(8) for row in unfinished
    )


def test_readiness_audit_never_calls_root_body_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from threes_rl import (
        j2a1_distillation_fidelity_execution_surface_v2 as v2,
    )

    monkeypatch.setattr(
        v2,
        "load_teacher_root_blob",
        lambda *_args, **_kwargs: pytest.fail("root body opened"),
    )
    assert surface.authority_audit()["passes"]
    assert surface.source_and_parent_audit(
        require_future_absent=True
    )["passes"]


def test_schema_exposes_only_frozen_recovery_commands() -> None:
    parser = surface.build_parser()
    choices = next(
        action.choices
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    assert list(choices) == [
        "audit-zero-work",
        "write-test-evidence",
        "prepare-readiness",
        "seal-phase-lock",
        "open",
        "materialize",
        "execute",
    ]
    for forbidden in (
        "ppo",
        "development",
        "confirmation",
        "promote",
        "reserve",
        "consume",
    ):
        assert forbidden not in choices


def test_authorization_schema_binds_all_preflight_and_v2_evidence() -> None:
    required = set(surface.authorization_schema()["required_fields"])
    assert {
        "recovery_preflight_artifacts",
        "recovery_authority_audit_sha256",
        "v2_bound_artifacts",
        "v2_terminal",
        "v2_retention",
    } <= required
    assert len(surface.PREFLIGHT_ARTIFACTS) == 9


def test_phase_paths_have_reuse_but_no_new_reservation_or_consumption(
    tmp_path: Path,
) -> None:
    paths = surface.phase_paths(tmp_path)
    assert paths["stream_reuse"].name == surface.STREAM_REUSE_NAME
    assert not any("RESERVATION" in path.name for path in paths.values())
    assert not any("CONSUMPTION" in path.name for path in paths.values())


def test_stream_reuse_binds_exact_v2_authority() -> None:
    payload = surface.stream_authority_reuse_payload(_chain())
    assert payload["stream_count"] == 63_488
    assert payload["recovery_row_count"] == 11_288
    assert payload["new_reservations"] == 0
    assert payload["new_consumptions"] == 0
    assert payload["reuse_only"]


def test_projection_uses_wall_and_remains_under_storage_cap() -> None:
    projection = surface.projection_payload()
    assert projection["v2_observed_wall_seconds"] == pytest.approx(
        32_608.277662038803
    )
    assert projection["point_total_stage_a_wall_hours"] == pytest.approx(
        42.602824125381694
    )
    assert projection[
        "conservative_total_stage_a_wall_hours"
    ] == pytest.approx(50.989066430196644)
    assert projection["combined_peak_after_margin_gib"] == pytest.approx(
        20.53877067565918
    )
    assert projection["passes"]


def test_worker_topology_is_exactly_one_by_eight() -> None:
    topology = surface.execution_schema()["worker_topology"]
    assert topology == {
        "top_level_jobs": 1,
        "collectors": 8,
        "single_thread_each": True,
        "shard": "stage-local row_index modulo 8",
        "canonical_merge": True,
        "work_stealing": False,
    }


def test_operational_guard_reuses_process_local_output_accountant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from threes_rl import (
        j2a1_distillation_fidelity_execution_surface_v2 as v2,
    )

    class Accountant:
        def __init__(self) -> None:
            self.snapshots = 0

        def snapshot(self) -> dict[str, object]:
            self.snapshots += 1
            return {
                "output_bytes": 0,
                "output_file_count": 0,
                "full_scan_count": 1,
                "targeted_stat_count": 4,
                "passes": True,
            }

    accountant = Accountant()
    monkeypatch.setattr(
        v2,
        "OutputAccountant",
        lambda _path: pytest.fail("unexpected full namespace scan"),
    )
    monkeypatch.setattr(
        v2,
        "execution_operational_guard",
        lambda **_kwargs: {"passes": True},
    )
    first = surface._phase_operational_guard(
        out_dir=tmp_path,
        cumulative_wall_seconds=surface.V2_WALL_SECONDS,
        include_services=False,
        accountant=accountant,
    )
    second = surface._phase_operational_guard(
        out_dir=tmp_path,
        cumulative_wall_seconds=surface.V2_WALL_SECONDS + 1.0,
        include_services=False,
        accountant=accountant,
    )
    assert first["output_accounting"]["full_scan_count"] == 1
    assert second["output_accounting"]["full_scan_count"] == 1
    assert accountant.snapshots == 4


def test_wall_ledger_charges_top_level_not_eight_worker_sum(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(surface, "V2_WALL_SECONDS", 100.0)
    ledger = surface.TopLevelWallLedger(
        path=tmp_path / "wall.jsonl",
        contract_sha256=CONTRACT,
        wall_clock=_Clock([1_000.0, 1_010.0, 1_010.0]),
    )
    ledger.start(unit_kind="teacher_root_block")
    ledger.heartbeat(unit_kind="teacher_root_block")
    ledger.finish(unit_kind="teacher_root_block")
    assert ledger.summary()["v3_charged_wall_seconds"] == 10.0
    assert ledger.summary()["cumulative_wall_seconds"] == 110.0
    assert ledger.summary()["cumulative_wall_seconds"] != 180.0


def test_wall_ledger_rejects_cross_phase_heartbeat(tmp_path: Path) -> None:
    ledger = surface.TopLevelWallLedger(
        path=tmp_path / "wall.jsonl",
        contract_sha256=CONTRACT,
        wall_clock=_Clock([0.0]),
    )
    ledger.start(unit_kind="teacher_root_block")
    with pytest.raises(surface.J2A1V3SurfaceIntegrityError):
        ledger.heartbeat(unit_kind="distillation_minibatch")


def test_wall_switch_closes_old_segment_before_new(tmp_path: Path) -> None:
    ledger = surface.TopLevelWallLedger(
        path=tmp_path / "wall.jsonl",
        contract_sha256=CONTRACT,
        wall_clock=_Clock([0.0, 5.0, 5.0, 9.0, 9.0]),
    )
    ledger.start(unit_kind="teacher_root_block")
    changed = ledger.switch(unit_kind="distillation_minibatch")
    assert changed["changed"]
    ledger.finish(unit_kind="distillation_minibatch")
    assert ledger.summary()["v3_charged_wall_seconds"] == 9.0


@pytest.mark.parametrize("downtime", [3_600.0, 86_400.0])
def test_dead_process_downtime_uses_same_bounded_charge(
    tmp_path: Path,
    downtime: float,
) -> None:
    path = tmp_path / f"wall-{int(downtime)}.jsonl"
    first = surface.TopLevelWallLedger(
        path=path,
        contract_sha256=CONTRACT,
        wall_clock=_Clock([10.0]),
    )
    first.start(unit_kind="distillation_minibatch")
    recovered = surface.TopLevelWallLedger(
        path=path,
        contract_sha256=CONTRACT,
        wall_clock=_Clock([10.0 + downtime]),
    )
    recovered.abandon_open()
    assert recovered.summary()["v3_charged_wall_seconds"] == 60.0


def test_wall_ledger_tamper_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "wall.jsonl"
    ledger = surface.TopLevelWallLedger(
        path=path,
        contract_sha256=CONTRACT,
        wall_clock=_Clock([0.0, 1.0]),
    )
    ledger.start(unit_kind="teacher_root_block")
    ledger.heartbeat(unit_kind="teacher_root_block")
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    records[-1]["charged_seconds"] = 2.0
    records[-1] = surface.payload_with_hash(
        {
            key: value
            for key, value in records[-1].items()
            if key != "wall_record_sha256"
        },
        "wall_record_sha256",
    )
    path.write_text(
        "\n".join(surface.canonical_json_bytes(row).decode() for row in records)
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(surface.J2A1V3SurfaceIntegrityError):
        surface.TopLevelWallLedger(
            path=path,
            contract_sha256=CONTRACT,
        )


@pytest.mark.parametrize(
    ("integrity", "operational", "scientific", "expected"),
    [
        (True, True, "READY", surface.KILL_EXECUTION),
        (False, True, "READY", surface.KILL_EXECUTION),
        (False, False, "HOLD_FAMILY", "HOLD_FAMILY"),
        (False, False, surface.READY_EXECUTION, surface.READY_EXECUTION),
    ],
)
def test_terminal_precedence(
    integrity: bool,
    operational: bool,
    scientific: str,
    expected: str,
) -> None:
    assert surface.terminal_precedence(
        integrity_failure=integrity,
        operational_failure=operational,
        scientific_decision=scientific,
    ) == expected


def test_stage_b_barrier_requires_exact_union() -> None:
    assert surface.stage_b_barrier(
        v2_completed=3_048,
        v3_completed=11_288,
        union_passes=True,
    )["passes"]
    assert not surface.stage_b_barrier(
        v2_completed=3_048,
        v3_completed=11_287,
        union_passes=True,
    )["passes"]
    assert not surface.stage_b_barrier(
        v2_completed=3_048,
        v3_completed=11_288,
        union_passes=False,
    )["passes"]


def test_canonical_merge_ignores_completion_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _rows(4)
    v2 = {
        str(rows[index]["root_id"]): _completion_ref(
            rows[index],
            suffix=index,
        )
        for index in (1, 0)
    }
    v3 = {
        str(rows[index]["root_id"]): _completion_ref(
            rows[index],
            suffix=index,
        )
        for index in (3, 2)
    }
    monkeypatch.setattr(surface, "ACTIVE_ROOTS", 4)
    merged = surface.canonical_merge_refs(
        full_rows=rows,
        v2_refs=v2,
        v3_refs=v3,
    )
    assert [row["root_id"] for row in merged] == [
        row["root_id"] for row in rows
    ]
    assert [row["source"] for row in merged] == ["v2", "v2", "v3", "v3"]


def test_canonical_merge_rejects_wrong_row_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _rows(2)
    ref = _completion_ref(rows[0], suffix=1)
    ref["ancestry_id"] = HEX_A
    monkeypatch.setattr(surface, "ACTIVE_ROOTS", 2)
    with pytest.raises(surface.J2A1V3SurfaceIntegrityError):
        surface.canonical_merge_refs(
            full_rows=rows,
            v2_refs={str(rows[0]["root_id"]): ref},
            v3_refs={
                str(rows[1]["root_id"]): _completion_ref(
                    rows[1],
                    suffix=2,
                )
            },
        )


def test_commit_chain_detects_tampered_older_file(tmp_path: Path) -> None:
    ledger = tmp_path / "commits.jsonl"
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    surface._append_commit(
        path=ledger,
        contract_sha256=CONTRACT,
        kind="first",
        bound_paths=(first,),
    )
    surface._append_commit(
        path=ledger,
        contract_sha256=CONTRACT,
        kind="second",
        bound_paths=(second,),
    )
    assert len(
        surface.verify_commit_ledger(
            path=ledger,
            contract_sha256=CONTRACT,
        )
    ) == 2
    first.write_bytes(b"changed")
    with pytest.raises(surface.J2A1V3SurfaceIntegrityError):
        surface.verify_commit_ledger(
            path=ledger,
            contract_sha256=CONTRACT,
        )


def test_commit_chain_accepts_authenticated_append_only_prefix(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "commits.jsonl"
    journal = tmp_path / "attempts.jsonl"
    journal.write_bytes(b"one\n")
    surface._append_commit(
        path=ledger,
        contract_sha256=CONTRACT,
        kind="one",
        bound_paths=(journal,),
    )
    with journal.open("ab") as handle:
        handle.write(b"two\n")
    assert surface.verify_commit_ledger(
        path=ledger,
        contract_sha256=CONTRACT,
    )


def test_owner_requires_durable_genesis(tmp_path: Path) -> None:
    with pytest.raises(surface.J2A1V3SurfaceIntegrityError):
        surface.acquire_or_reclaim_owner(
            ledger_path=tmp_path / "owners.jsonl",
            chain=_chain(),
            command="execute",
            commit_ledger_path=tmp_path / "commits.jsonl",
            pid=10,
            process_start_identity="start-a",
            is_live=lambda _record: False,
        )


def _owner_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, dict[str, object]]:
    commits = tmp_path / "commits.jsonl"
    genesis = tmp_path / "genesis.json"
    genesis.write_text("genesis", encoding="utf-8")
    chain = _chain()
    surface._append_commit(
        path=commits,
        contract_sha256=surface.phase_contract_sha256(
            chain,
            command="execute",
        ),
        kind="genesis",
        bound_paths=(genesis,),
    )
    return tmp_path / "owners.jsonl", commits, chain


def test_live_owner_is_rejected(tmp_path: Path) -> None:
    owners, commits, chain = _owner_fixture(tmp_path)
    surface.acquire_or_reclaim_owner(
        ledger_path=owners,
        chain=chain,
        command="execute",
        commit_ledger_path=commits,
        pid=10,
        process_start_identity="start-a",
        is_live=lambda _record: False,
    )
    with pytest.raises(surface.J2A1V3SurfaceOperationalHold):
        surface.acquire_or_reclaim_owner(
            ledger_path=owners,
            chain=chain,
            command="execute",
            commit_ledger_path=commits,
            pid=11,
            process_start_identity="start-b",
            is_live=lambda _record: True,
        )


def test_dead_owner_reclaim_is_append_only_and_verifiable(
    tmp_path: Path,
) -> None:
    owners, commits, chain = _owner_fixture(tmp_path)
    first = surface.acquire_or_reclaim_owner(
        ledger_path=owners,
        chain=chain,
        command="execute",
        commit_ledger_path=commits,
        pid=10,
        process_start_identity="start-a",
        is_live=lambda _record: False,
    )
    second = surface.acquire_or_reclaim_owner(
        ledger_path=owners,
        chain=chain,
        command="execute",
        commit_ledger_path=commits,
        pid=11,
        process_start_identity="start-b",
        is_live=lambda _record: False,
    )
    assert not first["reclaimed"]
    assert second["reclaimed"]
    assert second["owner"]["recovered_owner_record_sha256"] == first[
        "owner"
    ]["owner_record_sha256"]
    audit = surface.verify_current_owner(
        ledger_path=owners,
        expected_owner_sha256=second["owner"]["owner_record_sha256"],
        chain=chain,
        command="execute",
        pid=11,
        process_start_identity="start-b",
    )
    assert audit["passes"]
    assert audit["record_count"] == 2


def test_dead_owner_reclaim_rejects_wrong_contract(tmp_path: Path) -> None:
    owners, commits, chain = _owner_fixture(tmp_path)
    surface.acquire_or_reclaim_owner(
        ledger_path=owners,
        chain=chain,
        command="execute",
        commit_ledger_path=commits,
        pid=10,
        process_start_identity="start-a",
        is_live=lambda _record: False,
    )
    changed = copy.deepcopy(chain)
    changed["marker_identity"] = _identity("9")
    with pytest.raises(surface.J2A1V3SurfaceIntegrityError):
        surface.acquire_or_reclaim_owner(
            ledger_path=owners,
            chain=changed,
            command="execute",
            commit_ledger_path=commits,
            pid=11,
            process_start_identity="start-b",
            is_live=lambda _record: False,
        )


def test_phase_order_is_create_once_and_materializes_only_unfinished(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readiness, authorization, out_dir, rows = _patch_phase_chain(
        monkeypatch,
        tmp_path,
    )
    with pytest.raises(Exception):
        surface.open_phase(
            readiness_dir=readiness,
            authorization_path=authorization,
            out_dir=out_dir,
            jobs=1,
        )
    surface.seal_phase_lock(
        readiness_dir=readiness,
        authorization_path=authorization,
        out_dir=out_dir,
        jobs=1,
        include_operational=False,
    )
    marker = surface.open_phase(
        readiness_dir=readiness,
        authorization_path=authorization,
        out_dir=out_dir,
        jobs=1,
    )
    assert marker["teacher_queries"] == 0
    with pytest.raises(surface.J2A1V3SurfaceIntegrityError):
        surface.open_phase(
            readiness_dir=readiness,
            authorization_path=authorization,
            out_dir=out_dir,
            jobs=1,
        )
    materialized = surface.materialize_phase(
        readiness_dir=readiness,
        authorization_path=authorization,
        out_dir=out_dir,
        jobs=1,
    )
    assert materialized["rows"] == rows
    assert materialized["new_reservations"] == 0
    assert materialized["new_consumptions"] == 0
    assert materialized["teacher_queries"] == 0


def test_marker_tamper_blocks_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readiness, authorization, out_dir, _rows_value = _patch_phase_chain(
        monkeypatch,
        tmp_path,
    )
    surface.seal_phase_lock(
        readiness_dir=readiness,
        authorization_path=authorization,
        out_dir=out_dir,
        jobs=1,
        include_operational=False,
    )
    surface.open_phase(
        readiness_dir=readiness,
        authorization_path=authorization,
        out_dir=out_dir,
        jobs=1,
    )
    marker_path = out_dir / surface.MARKER_NAME
    marker = surface.load_json(marker_path)
    marker["teacher_queries"] = 1
    marker = surface.payload_with_hash(
        {
            key: value
            for key, value in marker.items()
            if key != "execution_marker_payload_sha256"
        },
        "execution_marker_payload_sha256",
    )
    marker_path.write_bytes(surface._serialized_json_bytes(marker))
    with pytest.raises(surface.J2A1V3SurfaceIntegrityError):
        surface.materialize_phase(
            readiness_dir=readiness,
            authorization_path=authorization,
            out_dir=out_dir,
            jobs=1,
        )


def test_checkpoint_authority_requires_exact_checkpoint_and_final_guard(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"checkpoint")
    engine = {
        "decision": surface.READY_EXECUTION,
        "passes": True,
        "checkpoint_authoritative": False,
        "checkpoint_authority_pending_final_operational_guard": True,
        "checkpoint": {
            "path": str(checkpoint),
            "file_sha256": surface.sha256_path(checkpoint),
        },
        "mechanism": {"mechanism_payload_sha256": HEX_A},
        "fidelity": {"fidelity_payload_sha256": HEX_B},
    }
    authority = surface._authorize_v3_checkpoint(
        out_dir=tmp_path,
        engine=engine,
        final_guard_identity=_identity("8"),
        execution_mode="scientific",
    )
    assert authority["authoritative"]
    assert not authority["ppo_execution_authorized"]


def test_checkpoint_mutation_fails_authority(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"checkpoint")
    engine = {
        "decision": surface.READY_EXECUTION,
        "passes": True,
        "checkpoint_authoritative": False,
        "checkpoint_authority_pending_final_operational_guard": True,
        "checkpoint": {
            "path": str(checkpoint),
            "file_sha256": surface.sha256_path(checkpoint),
        },
        "mechanism": {"mechanism_payload_sha256": HEX_A},
        "fidelity": {"fidelity_payload_sha256": HEX_B},
    }
    checkpoint.write_bytes(b"changed")
    with pytest.raises(surface.J2A1V3SurfaceIntegrityError):
        surface._authorize_v3_checkpoint(
            out_dir=tmp_path,
            engine=engine,
            final_guard_identity=_identity("8"),
            execution_mode="scientific",
        )


def test_clean_hold_quarantines_checkpoint(tmp_path: Path) -> None:
    quarantine = surface._quarantine_v3_checkpoint(
        out_dir=tmp_path,
        checkpoint={
            "path": str(tmp_path / "checkpoint.bin"),
            "file_sha256": HEX_A,
        },
        decision="HOLD_FIXTURE",
        predecessor={"fixture_payload_sha256": HEX_B},
    )
    assert quarantine["authoritative"] is False
    assert quarantine["usable_for_ppo"] is False
    assert quarantine["usable_for_development"] is False


def test_retention_cap_fails_before_retention_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "large.bin").write_bytes(b"1234")
    monkeypatch.setattr(surface, "STORAGE_CAP_BYTES", 4)
    with pytest.raises(surface.J2A1V3SurfaceOperationalHold):
        surface._seal_execution_retention(tmp_path)
    assert not (tmp_path / surface.EXECUTION_RETENTION_NAME).exists()


def test_terminal_bundle_roundtrip_and_tamper_detection(
    tmp_path: Path,
) -> None:
    result = surface.seal_terminal(
        out_dir=tmp_path,
        terminal_payload={
            "version": "fixture_terminal_v1",
            "decision": "HOLD_FIXTURE_DATA_SUPPORT",
            "checkpoint_authoritative": False,
            "passes": False,
        },
    )
    assert result["terminal"]["decision"] == "HOLD_FIXTURE_DATA_SUPPORT"
    assert surface._load_terminal_bundle(tmp_path)["passes"]
    evidence = tmp_path / surface.TERMINAL_EVIDENCE_NAME
    evidence.write_bytes(evidence.read_bytes() + b" ")
    with pytest.raises(Exception):
        surface._load_terminal_bundle(tmp_path)


def test_partial_terminal_evidence_finalizes_without_rerun(
    tmp_path: Path,
) -> None:
    payload = {
        "version": "fixture_terminal_v1",
        "decision": "HOLD_FIXTURE_DATA_SUPPORT",
        "checkpoint_authoritative": False,
        "passes": False,
    }
    surface.write_immutable_json(
        tmp_path / surface.TERMINAL_EVIDENCE_NAME,
        payload,
        field="terminal_evidence_payload_sha256",
    )
    resumed = surface._resume_terminal_finalization(tmp_path)
    assert resumed is not None
    assert resumed["terminal"]["decision"] == "HOLD_FIXTURE_DATA_SUPPORT"
    assert (tmp_path / surface.EXECUTION_RETENTION_NAME).is_file()
    assert (tmp_path / surface.TERMINAL_NAME).is_file()


def test_bounded_dispatcher_fixture_recovers_without_requery_or_consumption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from threes_rl import (
        j2a1_distillation_fidelity_execution_surface_v2 as v2,
    )

    readiness, authorization, out_dir, rows = _patch_phase_chain(
        monkeypatch,
        tmp_path,
    )
    surface.seal_phase_lock(
        readiness_dir=readiness,
        authorization_path=authorization,
        out_dir=out_dir,
        jobs=1,
        include_operational=False,
    )
    surface.open_phase(
        readiness_dir=readiness,
        authorization_path=authorization,
        out_dir=out_dir,
        jobs=1,
    )
    surface.materialize_phase(
        readiness_dir=readiness,
        authorization_path=authorization,
        out_dir=out_dir,
        jobs=1,
    )
    monkeypatch.setattr(surface, "V2_WALL_SECONDS", 0.0)
    monkeypatch.setattr(
        surface,
        "_phase_operational_guard",
        lambda **kwargs: {
            "cumulative_top_level_wall_seconds":
                kwargs["cumulative_wall_seconds"],
            "passes": True,
        },
    )
    monkeypatch.setattr(
        v2,
        "execute_distillation_fidelity_engine",
        lambda **_kwargs: pytest.fail("legacy engine reached"),
    )
    calls: list[list[str]] = []

    def bounded_fixture(**kwargs: object) -> dict[str, object]:
        authority_rows = list(kwargs["rows"])
        calls.append([str(row["root_id"]) for row in authority_rows])
        if len(calls) == 1:
            raise surface.J2A1V3PlannedInterruption(
                "planned pre-commit interruption"
            )
        ledger = v2.CompletionLedger(
            path=out_dir / "teacher_root_completions.jsonl",
            contract_sha256=str(kwargs["contract_sha256"]),
            kind="teacher_root",
        )
        for index, row in enumerate(authority_rows):
            ledger.append(
                root_id=str(row["root_id"]),
                ancestry_id=str(row["ancestry_id"]),
                row_index=int(row["row_index"]),
                stage=str(row["stage"]),
                relative_path=f"fixture/{row['root_id']}.bin",
                file_sha256=f"{index + 301:064x}",
                content_sha256=f"{index + 401:064x}",
                recovered_orphan=False,
            )
        kwargs["boundary_callback"](
            "teacher_root_block",
            (ledger.path,),
        )
        return {
            "refs": [
                ledger.by_root[str(row["root_id"])]
                for row in authority_rows
            ],
            "ledger": ledger.summary(),
            "passes": True,
        }

    monkeypatch.setattr(
        v2,
        "bounded_collect_teacher_roots",
        bounded_fixture,
    )

    def fixture_union(
        *,
        out_dir: Path,
        v3_completion_records: list[dict[str, object]],
    ) -> dict[str, object]:
        return surface.write_immutable_json(
            out_dir / surface.UNION_NAME,
            {
                "version": "fixture_union_v1",
                "v2_completed": 1,
                "v3_completed": len(v3_completion_records),
                "total_completed": 1 + len(v3_completion_records),
                "merged_refs": list(v3_completion_records),
                "passes": True,
            },
            field="union_payload_sha256",
        )

    monkeypatch.setattr(surface, "seal_union", fixture_union)
    now = [0.0]

    def wall_clock() -> float:
        now[0] += 1.0
        return now[0]

    with pytest.raises(surface.J2A1V3PlannedInterruption):
        surface.execute_phase_from_artifacts(
            readiness_dir=readiness,
            authorization_path=authorization,
            out_dir=out_dir,
            jobs=1,
            include_operational=False,
            execution_mode="miniature_fixture",
            fixture_collector=lambda _rows: [],
            fixture_post_union=lambda _union: {},
            owner_pid=90_000_001,
            owner_start_identity="fixture-start-1",
            wall_clock=wall_clock,
        )
    result = surface.execute_phase_from_artifacts(
        readiness_dir=readiness,
        authorization_path=authorization,
        out_dir=out_dir,
        jobs=1,
        include_operational=False,
        execution_mode="miniature_fixture",
        fixture_collector=lambda _rows: [],
        fixture_post_union=lambda _union: {
            "decision": "HOLD_FIXTURE_DATA_SUPPORT",
            "checkpoint_authoritative": False,
            "passes": False,
        },
        owner_pid=90_000_002,
        owner_start_identity="fixture-start-2",
        wall_clock=wall_clock,
    )
    expected_roots = [str(row["root_id"]) for row in rows]
    assert calls == [expected_roots, expected_roots]
    assert "f" * 64 not in calls[0]
    terminal = result["terminal"]
    assert terminal["decision"] == "HOLD_FIXTURE_DATA_SUPPORT"
    assert terminal["owner_record_count"] == 2
    assert terminal["v3_completion_summary"]["completed"] == len(rows)
    assert terminal["total_completed_roots"] == len(rows) + 1
    assert terminal["wall"]["v3_charged_wall_seconds"] >= (
        surface.ABANDONED_WALL_CHARGE_SECONDS["teacher_root_block"]
    )
    assert terminal["new_stream_reservations"] == 0
    assert terminal["new_stream_consumptions"] == 0
    assert terminal["v2_completed_roots_requeried"] == 0
    assert not any(
        "RESERVATION" in path.name or "CONSUMPTION" in path.name
        for path in out_dir.iterdir()
    )


def test_test_evidence_and_readiness_are_create_once(
    tmp_path: Path,
) -> None:
    evidence = surface.write_test_evidence(
        output_dir=tmp_path,
        commands=[
            {
                "command": "fixture",
                "passed": 1,
                "failed": 0,
                "note": "synthetic only",
            }
        ],
        deselections=[],
    )
    assert evidence["total_passed"] == 1
    with pytest.raises(surface.J2A1V3SurfaceIntegrityError):
        surface.write_test_evidence(
            output_dir=tmp_path,
            commands=[
                {
                    "command": "changed",
                    "passed": 1,
                    "failed": 0,
                    "note": "changed",
                }
            ],
            deselections=[],
        )
    ready = surface.prepare_readiness(
        output_dir=tmp_path,
        include_operational=False,
    )
    assert ready["decision"] == surface.READY
    assert ready["execution_authorized"] is False
    package = surface.verify_readiness_package(tmp_path)
    assert package["passes"]
    assert len(package["identities"]) == 9
    extra = tmp_path / "unexpected.json"
    extra.write_text("{}\n", encoding="utf-8")
    with pytest.raises(surface.J2A1V3SurfaceIntegrityError):
        surface.verify_readiness_package(tmp_path)
    extra.unlink()
    with pytest.raises(surface.J2A1V3SurfaceIntegrityError):
        surface.prepare_readiness(
            output_dir=tmp_path,
            include_operational=False,
        )


def test_zero_work_and_future_namespaces_are_absent() -> None:
    assert all(value == 0 for value in surface.ZERO_WORK.values())
    assert not surface.FUTURE_AUTHORIZATION_DIR.exists()
    assert not surface.FUTURE_EXECUTION_DIR.exists()
    assert surface.audit_zero_work(
        output_dir=surface.READINESS_DIR,
        include_operational=False,
    )["passes"]


def test_dispatcher_scientific_execute_has_no_fixture_escape(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    called: dict[str, object] = {}

    def fake_execute(**kwargs: object) -> dict[str, object]:
        called.update(kwargs)
        return {"passes": True}

    monkeypatch.setattr(
        surface,
        "execute_phase_from_artifacts",
        fake_execute,
    )
    result = surface.dispatch_cli(
        argparse.Namespace(
            command="execute",
            readiness_dir=tmp_path / "readiness",
            authorization=tmp_path / "authorization.json",
            out_dir=tmp_path / "execution",
            jobs=1,
        )
    )
    assert result["passes"]
    assert called["execution_mode"] == "scientific"
    assert called["jobs"] == 1
    assert "fixture_collector" not in called
    assert "fixture_post_union" not in called
