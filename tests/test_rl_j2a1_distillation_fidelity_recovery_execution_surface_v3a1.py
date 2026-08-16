from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pytest

from threes_rl import (
    j2a1_distillation_fidelity_recovery_execution_surface_v3a1 as surface,
)


COMMANDS = [
    {
        "command": "fixture focused",
        "passed": 1,
        "failed": 0,
        "note": "fixture",
    }
]


def _write_fixture_evidence(output_dir: Path) -> dict[str, object]:
    return surface.write_test_evidence(
        output_dir=output_dir,
        commands=COMMANDS,
        deselections=surface.EXPECTED_DESELECTIONS,
    )


def _operational_pass(**_kwargs: object) -> dict[str, object]:
    return {
        "version": "fixture_operational_v1",
        "passes": True,
    }


def _prepare_fixture(
    output_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    cushion_gib: int,
) -> dict[str, object]:
    _write_fixture_evidence(output_dir)
    monkeypatch.setattr(
        surface.v3,
        "operational_audit",
        _operational_pass,
    )
    current_free = (
        surface.PROJECTED_INCREMENTAL_BYTES
        + surface.HARD_FLOOR_BYTES
        + cushion_gib * 1024**3
    )
    return surface.prepare_readiness(
        output_dir=output_dir,
        include_operational=False,
        free_bytes=current_free,
    )


def test_frozen_incremental_headroom_formula() -> None:
    assert surface.PROJECTED_COMBINED_BYTES == 22_053_337_088
    assert surface.V2_RETAINED_BYTES == 1_782_523_714
    assert surface.PROJECTED_INCREMENTAL_BYTES == 20_270_813_374
    assert surface.PROJECTED_INCREMENTAL_BYTES == (
        surface.PROJECTED_COMBINED_BYTES
        - surface.V2_RETAINED_BYTES
    )


def test_parent_v3_package_verifies_exactly() -> None:
    audit = surface.verify_parent_v3_package()
    assert audit["passes"]
    assert all(audit["checks"].values())
    assert len(audit["artifact_identities"]) == 9


def test_parent_source_drift_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed = dict(surface.PARENT_SOURCE_IDENTITIES)
    key = next(iter(changed))
    changed[key] = "0" * 64
    monkeypatch.setattr(surface, "PARENT_SOURCE_IDENTITIES", changed)
    with pytest.raises(surface.J2A1V3A1IntegrityError):
        surface.verify_parent_v3_package()


def test_parent_extra_file_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    copied = tmp_path / "parent"
    shutil.copytree(surface.PARENT_READINESS_DIR, copied)
    (copied / "EXTRA.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(surface, "PARENT_READINESS_DIR", copied)
    with pytest.raises(Exception):
        surface.verify_parent_v3_package()


def test_parent_missing_file_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    copied = tmp_path / "parent"
    shutil.copytree(surface.PARENT_READINESS_DIR, copied)
    (copied / surface.v3.SCHEMA_NAME).unlink()
    monkeypatch.setattr(surface, "PARENT_READINESS_DIR", copied)
    with pytest.raises(Exception):
        surface.verify_parent_v3_package()


def test_pre_prepare_requires_absent_namespace(tmp_path: Path) -> None:
    output_dir = tmp_path / "absent"
    assert surface.audit_pre_prepare(output_dir=output_dir)["passes"]
    output_dir.mkdir()
    audit = surface.audit_pre_prepare(output_dir=output_dir)
    assert not audit["passes"]
    assert not audit["checks"]["readiness_namespace_absent"]


def test_post_seal_requires_exact_parent_package() -> None:
    audit = surface.audit_post_seal()
    assert audit["passes"]
    assert all(audit["checks"].values())
    assert audit["parent_v3"]["checks"]["exact_nine_file_set"]


def test_parent_preprepare_audit_reproduces_spent_failure() -> None:
    assert not surface.v3.audit_zero_work(
        output_dir=surface.PARENT_READINESS_DIR,
        include_operational=False,
    )["passes"]
    assert surface.audit_post_seal()["passes"]


def test_parent_hold_binds_exact_diagnostic() -> None:
    hold = surface.parent_hold_payload()
    assert hold["decision"] == surface.PARENT_HOLD
    assert hold["parent_surface_spent_for_execution"]
    assert hold["diagnostic"]["passed"] == 41
    assert hold["diagnostic"]["failed"] == 1
    assert hold["diagnostic"]["failure_node"] == surface.PARENT_FAILURE_NODE
    assert hold["passes"]


def test_future_namespace_presence_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    future = tmp_path / "future"
    future.mkdir()
    monkeypatch.setattr(surface, "FUTURE_AUTHORIZATION_DIR", future)
    with pytest.raises(surface.J2A1V3A1IntegrityError):
        surface.verify_parent_v3_package()


def test_zero_work_is_exact() -> None:
    assert all(value == 0 for value in surface.ZERO_WORK.values())
    assert surface._zero_work_exact()


@pytest.mark.parametrize(
    ("integrity", "headroom", "expected"),
    [
        (False, False, surface.KILL),
        (False, True, surface.KILL),
        (True, False, surface.HOLD),
        (True, True, surface.READY),
    ],
)
def test_readiness_decision_precedence(
    integrity: bool,
    headroom: bool,
    expected: str,
) -> None:
    assert surface.readiness_decision(
        integrity_passes=integrity,
        headroom_passes=headroom,
    ) == expected


def test_headroom_ready_at_exact_five_gib_cushion() -> None:
    current = (
        surface.PROJECTED_INCREMENTAL_BYTES
        + surface.HARD_FLOOR_BYTES
        + surface.MINIMUM_CUSHION_BYTES
    )
    audit = surface.calculate_launch_headroom(current)
    assert audit["admission_passes"]
    assert audit["projected_floor_cushion_bytes"] == (
        surface.MINIMUM_CUSHION_BYTES
    )
    assert audit["additional_free_bytes_required"] == 0


def test_headroom_holds_below_five_gib_cushion() -> None:
    current = (
        surface.PROJECTED_INCREMENTAL_BYTES
        + surface.HARD_FLOOR_BYTES
        + 4 * 1024**3
    )
    audit = surface.calculate_launch_headroom(current)
    assert audit["checks"]["projected_free_at_least_100_gib"]
    assert not audit["checks"]["projected_cushion_at_least_5_gib"]
    assert not audit["admission_passes"]
    assert audit["decision"] == surface.HOLD
    assert audit["additional_free_bytes_required"] == 1024**3


def test_headroom_holds_below_hard_floor() -> None:
    current = (
        surface.PROJECTED_INCREMENTAL_BYTES
        + surface.HARD_FLOOR_BYTES
        - 1
    )
    audit = surface.calculate_launch_headroom(current)
    assert not audit["checks"]["projected_free_at_least_100_gib"]
    assert not audit["admission_passes"]


@pytest.mark.parametrize("value", [None, -1, 1.5, "100"])
def test_headroom_unavailable_fails_admission(value: object) -> None:
    audit = surface.calculate_launch_headroom(value)
    assert audit["formula_integrity_passes"]
    assert not audit["calculation_available"]
    assert not audit["admission_passes"]
    assert audit["decision"] == surface.HOLD


def test_headroom_formula_drift_is_integrity_kill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        surface,
        "PROJECTED_COMBINED_BYTES",
        surface.PROJECTED_COMBINED_BYTES + 1,
    )
    audit = surface.calculate_launch_headroom(130 * 1024**3)
    assert not audit["formula_integrity_passes"]
    assert audit["decision"] == surface.KILL


def test_v2_footprint_is_not_double_charged() -> None:
    current = 130 * 1024**3
    audit = surface.calculate_launch_headroom(current)
    assert audit["projected_peak_free_bytes"] == (
        current
        - surface.PROJECTED_COMBINED_BYTES
        + surface.V2_RETAINED_BYTES
    )


def test_cleanup_proposal_authorizes_no_deletion() -> None:
    current = (
        surface.PROJECTED_INCREMENTAL_BYTES
        + surface.HARD_FLOOR_BYTES
        + 2 * 1024**3
    )
    calculation = surface.calculate_launch_headroom(current)
    proposal = surface.cleanup_review_proposal(
        {"calculation": calculation}
    )
    assert proposal["decision"] == "REVIEW_ADDITIONAL_HEADROOM"
    assert proposal["candidate_deletions"] == []
    assert proposal["cleanup_authorized"] is False
    assert proposal["cleanup_performed"] is False
    assert proposal["review_required_before_any_move_or_delete"]
    assert proposal["additional_free_bytes_required"] == 3 * 1024**3


def test_cleanup_proposal_remains_no_delete_when_ready() -> None:
    current = (
        surface.PROJECTED_INCREMENTAL_BYTES
        + surface.HARD_FLOOR_BYTES
        + 6 * 1024**3
    )
    proposal = surface.cleanup_review_proposal(
        {"calculation": surface.calculate_launch_headroom(current)}
    )
    assert proposal["decision"] == "NO_HEADROOM_ACTION_REQUIRED"
    assert proposal["candidate_deletions"] == []
    assert proposal["cleanup_authorized"] is False


def test_schema_exposes_no_execution_or_cleanup_command() -> None:
    schema = surface.readiness_schema()
    assert schema["public_commands"] == [
        "audit-pre-prepare",
        "audit-post-seal",
        "write-test-evidence",
        "prepare-readiness",
    ]
    assert schema["execution_commands"] == []
    assert schema["cleanup_commands"] == []
    assert schema["execution_authorized"] is False
    assert not hasattr(surface, "execute")
    assert not hasattr(surface, "seal_phase_lock")


def test_parser_exposes_only_frozen_commands() -> None:
    parser = surface.build_parser()
    subparsers = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    assert list(subparsers.choices) == [
        "audit-pre-prepare",
        "audit-post-seal",
        "write-test-evidence",
        "prepare-readiness",
    ]
    with pytest.raises(SystemExit):
        parser.parse_args(["execute"])


def test_test_evidence_is_create_once(tmp_path: Path) -> None:
    output_dir = tmp_path / "readiness"
    evidence = _write_fixture_evidence(output_dir)
    assert evidence["passes"]
    assert evidence["total_deselected"] == 8
    with pytest.raises(surface.J2A1V3A1IntegrityError):
        _write_fixture_evidence(output_dir)


def test_test_evidence_rejects_new_deselection(tmp_path: Path) -> None:
    with pytest.raises(surface.J2A1V3A1IntegrityError):
        surface.write_test_evidence(
            output_dir=tmp_path / "readiness",
            commands=COMMANDS,
            deselections=surface.EXPECTED_DESELECTIONS + ["new"],
        )


def test_test_evidence_binds_parent_failure_without_counting_it(
    tmp_path: Path,
) -> None:
    evidence = _write_fixture_evidence(tmp_path / "readiness")
    assert evidence["total_failed"] == 0
    assert evidence["bound_parent_diagnostic"]["failed"] == 1
    assert evidence["bound_parent_diagnostic"]["classification"] == (
        "chronology_orchestration_only"
    )


def test_headroom_audit_keeps_operations_separate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        surface.v3,
        "operational_audit",
        _operational_pass,
    )
    current = (
        surface.PROJECTED_INCREMENTAL_BYTES
        + surface.HARD_FLOOR_BYTES
        + 4 * 1024**3
    )
    audit = surface.launch_headroom_audit(
        output_dir=tmp_path,
        include_operational=False,
        free_bytes=current,
    )
    assert audit["integrity_passes"]
    assert not audit["admission_passes"]
    assert audit["checks"]["parent_operations_pass"]


def test_disk_usage_failure_seals_headroom_hold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        surface.v3,
        "operational_audit",
        _operational_pass,
    )
    monkeypatch.setattr(
        surface.shutil,
        "disk_usage",
        lambda _path: (_ for _ in ()).throw(OSError("fixture")),
    )
    audit = surface.launch_headroom_audit(
        output_dir=tmp_path,
        include_operational=False,
    )
    assert audit["integrity_passes"]
    assert not audit["admission_passes"]
    assert audit["calculation"]["decision"] == surface.HOLD
    assert audit["calculation"]["current_free_bytes"] is None


def test_prepare_seals_headroom_hold_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "readiness"
    sealed = _prepare_fixture(
        output_dir,
        monkeypatch,
        cushion_gib=4,
    )
    assert sealed["decision"] == surface.HOLD
    assert sealed["execution_authorized"] is False
    assert len(list(output_dir.iterdir())) == 10
    result = sealed["result"]
    assert result["hold"] is True
    assert result["continue"] is False
    assert result["checks"]["headroom_integrity"]
    assert not result["checks"]["headroom_admission"]
    assert result["checks"]["cleanup_review_only"]


def test_prepare_can_only_ready_with_full_cushion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sealed = _prepare_fixture(
        tmp_path / "readiness",
        monkeypatch,
        cushion_gib=5,
    )
    assert sealed["decision"] == surface.READY
    assert sealed["result"]["continue"] is True
    assert sealed["result"]["execution_authorized"] is False


def test_readiness_package_reload_is_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "readiness"
    _prepare_fixture(output_dir, monkeypatch, cushion_gib=4)
    verified = surface.verify_readiness_package(output_dir)
    assert verified["passes"]
    assert verified["decision"] == surface.HOLD
    assert len(verified["identities"]) == 10
    assert all(verified["checks"].values())


def test_readiness_file_tamper_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "readiness"
    _prepare_fixture(output_dir, monkeypatch, cushion_gib=4)
    path = output_dir / surface.HEADROOM_AUDIT_NAME
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(Exception):
        surface.verify_readiness_package(output_dir)


def test_readiness_extra_file_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "readiness"
    _prepare_fixture(output_dir, monkeypatch, cushion_gib=4)
    (output_dir / "EXTRA").write_text("x", encoding="utf-8")
    with pytest.raises(surface.J2A1V3A1IntegrityError):
        surface.verify_readiness_package(output_dir)


def test_readiness_missing_file_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "readiness"
    _prepare_fixture(output_dir, monkeypatch, cushion_gib=4)
    (output_dir / surface.LOCK_NAME).unlink()
    with pytest.raises(surface.J2A1V3A1IntegrityError):
        surface.verify_readiness_package(output_dir)


def test_retention_inventory_tamper_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "readiness"
    _prepare_fixture(output_dir, monkeypatch, cushion_gib=4)
    retention_path = output_dir / surface.RETENTION_NAME
    retention = json.loads(retention_path.read_text(encoding="utf-8"))
    retention["canonical_inventory_sha256"] = "0" * 64
    body = {
        key: value
        for key, value in retention.items()
        if key != surface.READINESS_FIELDS["retention"]
    }
    rewritten = surface.v3.payload_with_hash(
        body,
        surface.READINESS_FIELDS["retention"],
    )
    retention_path.write_bytes(surface.v3.canonical_json_bytes(rewritten))
    with pytest.raises(surface.J2A1V3A1IntegrityError):
        surface.verify_readiness_package(output_dir)


def test_authoritative_v3a1_namespace_is_absent_or_exact() -> None:
    if surface.READINESS_DIR.exists():
        assert surface.verify_readiness_package(
            surface.READINESS_DIR
        )["passes"]
    else:
        assert surface.audit_pre_prepare(
            output_dir=surface.READINESS_DIR
        )["passes"]


def test_cli_post_seal_audit_uses_parent_package() -> None:
    args = surface.build_parser().parse_args(["audit-post-seal"])
    result = surface.dispatch_cli(args)
    assert result["passes"]
    assert result["checks"]["parent_exact_nine_file_package"]


def test_no_scientific_namespace_or_work_exists() -> None:
    assert not surface.FUTURE_AUTHORIZATION_DIR.exists()
    assert not surface.FUTURE_EXECUTION_DIR.exists()
    assert not surface.v3.FUTURE_AUTHORIZATION_DIR.exists()
    assert not surface.v3.FUTURE_EXECUTION_DIR.exists()
    assert surface._zero_work_exact()
