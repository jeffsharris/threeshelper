from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from threes_rl import o3_p0_preflight as p0


def _fake_tests() -> dict:
    return {
        "passes": True,
        "focused_tests_passed": 1,
        "regression_tests_passed": 1,
        "test_evidence_payload_sha256": "tests-payload",
    }


def _fake_disk(_path: Path) -> SimpleNamespace:
    return SimpleNamespace(free=140 * 1024**3)


def test_frozen_o3_contract_hashes_and_model_reproduce() -> None:
    audit = p0.frozen_hash_audit()
    assert audit["passes"]
    assert audit["schema_sha256"] == p0.FROZEN_SCHEMA_SHA256
    assert audit["parameter_count"] == 102_557


def test_o2_evidence_reads_aggregate_text_and_byte_hash_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = tmp_path / "ledger.md"
    log = tmp_path / "log.md"
    support = tmp_path / "support.json"
    ledger.write_text(
        "\n".join(
            (
                "**Decision: `HOLD_O2_DATA_SUPPORT`.**",
                "`128` unique ancestries/replay",
                "`T192 5/5/0/9`",
                "`T96 9/9/0/9`",
                "`T48 9/9/2/9`",
                "`7/20`",
            )
        )
    )
    log.write_text(
        "\n".join(
            (
                "`7,192` frames",
                "`267` permitted",
                "`128` unique",
                "`32` roots per family",
                "`T192 5/5/0/9`",
                "`T96 9/9/0/9`",
                "`T48 9/9/2/9`",
                "`7/20`",
            )
        )
    )
    support.write_bytes(b"not-json-and-must-never-be-parsed")
    monkeypatch.setattr(
        p0,
        "O2_SUPPORT_FILE_SHA256",
        hashlib.sha256(support.read_bytes()).hexdigest(),
    )
    report = p0.o2_aggregate_evidence(
        ledger_path=ledger,
        log_path=log,
        support_path=support,
    )
    assert report["passes"]
    assert report["support_json"]["handling"] == "byte_hash_only"
    assert report["checks"]["o2_replay_content_not_read"]


def test_partition_plan_is_whole_root_equal_family_and_exact_size() -> None:
    rows = p0.future_stream_rows()
    plan = p0.partition_plan(rows)
    assert plan["passes"]
    assert plan["root_universe_count"] == 20_500
    assert plan["role_counts"] == {
        "train": 5_020,
        "development": 1_675,
        "untouched_mechanism": 13_805,
    }
    assert all(
        set(counts.values()) == {count // 5}
        for count, counts in (
            (5_020, plan["per_role_family_counts"]["train"]),
            (1_675, plan["per_role_family_counts"]["development"]),
            (
                13_805,
                plan["per_role_family_counts"]["untouched_mechanism"],
            ),
        )
    )


def test_stream_contract_counts_and_pair_coupling_are_exact() -> None:
    rows = p0.future_stream_rows()
    contract = p0.stream_contract(rows)
    assert contract["passes"]
    assert contract["row_count"] == 26_516
    assert contract["purpose_counts"] == {
        "acquisition": 20_500,
        "learning": 1_152,
        "option_development": 256,
        "option_untouched_mechanism": 1_536,
        "normal_development": 512,
        "confirmation": 2_560,
    }
    paired = [
        row for row in rows if "control_policy_stream_id" in row
    ]
    assert paired
    assert all(
        row["control_policy_stream_id"]
        != row["treatment_policy_stream_id"]
        for row in paired
    )


def test_collision_audit_skips_o2_content_and_detects_external_collision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scan_root = tmp_path / "runs"
    support = scan_root / "recovery" / "O2_RECOVERED_SUPPORT.json"
    replay_dir = scan_root / "pilot" / "source_replays"
    support.parent.mkdir(parents=True)
    replay_dir.mkdir(parents=True)
    support.write_text('{"logical_seed":105000000000}')
    replay = replay_dir / "replay.json"
    replay.write_text('{"logical_seed":105000000000}')
    metadata = scan_root / "other.json"
    metadata.write_text('{"logical_seed":1}')
    monkeypatch.setattr(p0, "O2_SUPPORT_PATH", support)
    monkeypatch.setattr(p0, "O2_FORBIDDEN_CONTENT_DIR", replay_dir)

    original_scan = p0.history._scan_history_file

    def guarded(path: Path):
        assert path != support
        assert not p0._is_within(path, replay_dir)
        return original_scan(path)

    monkeypatch.setattr(p0.history, "_scan_history_file", guarded)
    rows = [p0.acquisition_rows()[0]]
    clean = p0.collision_audit(
        rows,
        scan_root=scan_root,
        out_dir=scan_root / "current",
    )
    assert clean["passes"]

    metadata.write_text('{"logical_seed":105000000000}')
    collided = p0.collision_audit(
        rows,
        scan_root=scan_root,
        out_dir=scan_root / "current",
    )
    assert not collided["passes"]
    assert collided["collisions"]["logical_seed"] == [105_000_000_000]


def test_power_contract_reproduces_n192_or150_gate() -> None:
    report = p0.power_contract()
    assert report["passes"]
    assert report["selected_roots"] == 192
    assert report["grid_mde"] == 1.50


def test_scientific_decision_separates_representation_and_readiness() -> None:
    assert p0._decision(
        representation_checks={"representation": True},
        readiness_checks={"readiness": True},
    ) == "READY_O3_EVENT_ACQUISITION"
    assert p0._decision(
        representation_checks={"representation": True},
        readiness_checks={"readiness": False},
    ) == "HOLD_O3_DATA_OR_POWER"
    assert p0._decision(
        representation_checks={"representation": False},
        readiness_checks={"readiness": True},
    ) == "KILL_O3_REPRESENTATION_PREFLIGHT"


def test_open_writes_only_one_zero_work_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = tmp_path / "o3"
    monkeypatch.setattr(p0, "OUTPUT_DIR", out_dir)
    monkeypatch.setattr(p0, "frozen_hash_audit", lambda: {"passes": True})
    monkeypatch.setattr(p0, "_load_test_evidence", _fake_tests)
    monkeypatch.setattr(
        p0,
        "_heavy_process_audit",
        lambda: {"passes": True, "heavy_processes": []},
    )
    monkeypatch.setattr(
        p0.history,
        "service_health",
        lambda: {"passes": True},
    )
    monkeypatch.setattr(p0.history, "current_nice", lambda: 10)
    monkeypatch.setattr(p0.shutil, "disk_usage", _fake_disk)
    monkeypatch.setattr(
        p0,
        "_marker_identity",
        lambda path: {
            "version": p0.VERSION,
            "bound_out_dir": str(path.resolve()),
        },
    )
    marker = p0.open_preflight(out_dir)
    assert marker["decision"] == "O3_P0_OPENED_ZERO_WORK"
    assert list(out_dir.iterdir()) == [out_dir / p0.MARKER_NAME]
    assert p0._verify_self_hash(marker, "opened_payload_sha256")
    with pytest.raises(FileExistsError):
        p0.open_preflight(out_dir)


def test_marker_validation_rejects_binding_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = tmp_path / "o3"
    out_dir.mkdir()
    monkeypatch.setattr(
        p0,
        "_marker_identity",
        lambda path: {"version": p0.VERSION, "binding": "expected"},
    )
    p0._write_immutable_json(
        out_dir / p0.MARKER_NAME,
        {"version": p0.VERSION, "binding": "wrong"},
        self_hash_field="opened_payload_sha256",
    )
    with pytest.raises(ValueError, match="binding mismatch"):
        p0._load_marker(out_dir)


def test_run_seals_ready_result_without_fresh_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = tmp_path / "o3"
    out_dir.mkdir()
    identity = {"version": p0.VERSION, "binding": "exact"}
    monkeypatch.setattr(p0, "_marker_identity", lambda _path: identity)
    p0._write_immutable_json(
        out_dir / p0.MARKER_NAME,
        identity,
        self_hash_field="opened_payload_sha256",
    )
    monkeypatch.setattr(p0, "frozen_hash_audit", lambda: {"passes": True})
    monkeypatch.setattr(p0, "_load_test_evidence", _fake_tests)
    monkeypatch.setattr(
        p0,
        "o2_aggregate_evidence",
        lambda: {"passes": True},
    )
    monkeypatch.setattr(p0, "future_stream_rows", lambda: [])
    monkeypatch.setattr(
        p0,
        "partition_plan",
        lambda _rows: {
            "passes": True,
            "root_universe_count": 20_500,
            "role_counts": p0.ROLE_COUNTS,
        },
    )
    monkeypatch.setattr(
        p0,
        "stream_contract",
        lambda _rows: {
            "passes": True,
            "row_count": 26_516,
            "purpose_counts": {},
        },
    )
    monkeypatch.setattr(
        p0,
        "collision_audit",
        lambda _rows, out_dir: {
            "passes": True,
            "matched_source_count": 1,
            "matched_sources_sha256": "sources",
        },
    )
    monkeypatch.setattr(
        p0,
        "power_contract",
        lambda: {
            "passes": True,
            "rows": [
                {
                    "roots": 192,
                    "true_common_odds_ratio": 1.50,
                    "power_full_gate": 0.90,
                }
            ],
            "selected_roots": 192,
            "grid_mde": 1.50,
        },
    )
    monkeypatch.setattr(
        p0,
        "policy_audit",
        lambda: {
            "passes": True,
            "signature_sha256": {"family": "signature"},
        },
    )
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
    monkeypatch.setattr(p0.shutil, "disk_usage", _fake_disk)
    test_evidence = tmp_path / "test_evidence.json"
    test_evidence.write_text("{}")
    monkeypatch.setattr(p0, "TEST_EVIDENCE_PATH", test_evidence)
    result = p0.run_preflight(out_dir)
    assert result["decision"] == "READY_O3_EVENT_ACQUISITION", json.dumps(
        result,
        sort_keys=True,
    )
    assert result["zero_work"]["fresh_games"] == 0
    assert p0._verify_self_hash(result, "result_payload_sha256")
    with pytest.raises(FileExistsError):
        p0.run_preflight(out_dir)
