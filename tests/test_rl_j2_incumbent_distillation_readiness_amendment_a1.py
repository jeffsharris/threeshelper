from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import pytest

from threes_rl import j2_incumbent_distillation_readiness as j2
from threes_rl import (
    j2_incumbent_distillation_readiness_amendment_a1 as a1,
)


def test_stage_table_derives_every_amended_total() -> None:
    assert a1.derive_stage_totals() == {
        "prospective_rows_or_pairs": 36_096,
        "game_arms": 47_616,
        "unique_streams": 155_904,
        "pre_ppo_teacher_roots": 14_336,
        "online_teacher_roots": 4_096,
        "total_teacher_root_equivalents": 18_432,
    }
    assert 155_904 == (
        4 * 8_192
        + 5 * 6_144
        + 4 * 16_384
        + 5 * 896
        + 5 * 4_480
    )


def test_only_validation_stage_counts_change_from_parent() -> None:
    parent = {row["stage"]: row for row in j2.STAGE_TABLE}
    amended = {row["stage"]: row for row in a1.STAGE_TABLE}
    assert set(parent) == set(amended)
    for stage in parent:
        if stage == "distillation_validation":
            assert parent[stage]["authority_rows"] == 2_048
            assert amended[stage]["authority_rows"] == 6_144
            assert amended[stage]["game_arms"] == 12_288
            assert amended[stage]["pre_ppo_teacher_roots"] == 6_144
            assert amended[stage]["streams"] == parent[stage]["streams"]
        else:
            assert amended[stage] == parent[stage]


def test_stage_table_drift_is_detected() -> None:
    changed = [dict(row) for row in a1.STAGE_TABLE]
    changed[0]["authority_rows"] = 8_193
    assert a1.derive_stage_totals(changed) != a1.EXPECTED_STAGE_TOTALS


def test_miniature_rows_use_versioned_commitments_and_pair_semantics() -> None:
    table = (
        {
            "stage": "distillation_validation",
            "authority_rows": 2,
            "game_arms": 4,
            "pre_ppo_teacher_roots": 2,
            "online_teacher_roots": 0,
            "streams": {
                "logical_stream_id": 231_000_000_000,
                "deck_stream_id": 232_000_000_000,
                "slot_stream_id": 233_000_000_000,
                "student_policy_stream_id": 234_000_000_000,
                "teacher_policy_stream_id": 235_000_000_000,
            },
        },
    )
    rows = a1.build_prospective_rows(table)
    assert len(rows) == 2
    assert rows[0]["root_id"] != rows[1]["root_id"]
    assert rows[0]["ancestry_id"] != rows[1]["ancestry_id"]
    assert rows[0]["streams"]["student_policy_stream_id"] != rows[0][
        "streams"
    ]["teacher_policy_stream_id"]
    core = {
        "stage": "distillation_validation",
        "row_index": 0,
        "streams": rows[0]["streams"],
    }
    assert rows[0]["root_id"] == a1._commitment("j2-a1-root-v1", core)
    assert rows[0]["root_id"] != j2._commitment("j2-root-v1", core)
    assert not rows[0]["content_opened"]
    assert not rows[0]["reserved"]
    assert not rows[0]["consumed"]


def test_full_prospective_authority_is_exact_and_collision_free() -> None:
    report = a1.prospective_authority()
    assert report["passes"]
    assert report["row_count"] == 36_096
    assert report["stream_count"] == 155_904
    assert report["checks"]["paired_policy_streams_distinct"]
    assert report["checks"]["new_root_commitments_distinct_from_parent"]
    assert report["checks"]["new_ancestry_commitments_distinct_from_parent"]
    assert report["checks"]["parent_prospective_was_unopened_unspent"]
    assert report["checks"]["no_spent_213b_226b_collision"]
    assert report["checks"]["no_engineering_250b_255b_collision"]


def test_local_and_parent_identities_are_exact() -> None:
    report = a1.source_and_parent_audit()
    assert report["passes"]
    assert len(report["parent_readiness_artifacts"]) == 10
    assert all(report["parent_readiness_checks"].values())


def test_parent_source_drift_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed = dict(a1.EXPECTED_PARENT_SOURCE_HASHES)
    first = next(iter(changed))
    changed[first] = "0" * 64
    monkeypatch.setattr(a1, "EXPECTED_PARENT_SOURCE_HASHES", changed)
    report = a1.source_and_parent_audit()
    assert not report["passes"]
    assert not report["checks"]["parent_sources_exact"]


def test_parent_readiness_payload_drift_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed = dict(a1.EXPECTED_PARENT_READINESS)
    name = "J2_READINESS_RESULT.json"
    file_sha, field, _payload = changed[name]
    changed[name] = (file_sha, field, "0" * 64)
    monkeypatch.setattr(a1, "EXPECTED_PARENT_READINESS", changed)
    report = a1.source_and_parent_audit()
    assert not report["passes"]
    assert not report["parent_readiness_checks"][name]


def test_pilot_history_binds_all_retained_predecessors() -> None:
    report = a1.pilot_history_audit()
    assert report["passes"]
    assert len(report["v2_retention_predecessors"]) == 26
    assert set(report["v2_measured_artifacts"]) == {
        "central",
        "sensitivity",
        "synchronous",
        "power",
    }
    assert report["checks"]["v2_query_count_exact"]
    assert report["checks"]["v2_scientific_counters_zero"]


def test_pilot_terminal_identity_drift_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        a1,
        "EXPECTED_V2_TERMINAL",
        ("0" * 64, "terminal_payload_sha256", "1" * 64),
    )
    report = a1.pilot_history_audit()
    assert not report["passes"]
    assert not report["checks"]["v2_terminal_exact"]


def test_score_power_is_exact_at_n6144() -> None:
    report = a1.score_fidelity_power()
    assert report["n_pairs"] == 6_144
    assert report["standard_error"] == pytest.approx(
        0.01594719884624465,
        abs=1e-16,
    )
    assert report["score_80pct_mde_percent"] == pytest.approx(
        4.569050397401253,
        abs=1e-12,
    )
    assert report["equal_policy_combined_gate_power"] == pytest.approx(
        0.9719336262231589,
        abs=1e-15,
    )


def test_exact_n6144_power_reproduces_v2() -> None:
    report = a1.power_report()
    assert report["passes"]
    assert (
        report["progression_full_report_sha256"]
        == a1.EXPECTED_V2_POWER_REPORT_SHA256
    )
    assert report["progression_common_or"][
        "worst_case_primary_power"
    ] == pytest.approx(a1.EXPECTED_V2_POWER, abs=0.0)
    assert report["checks"]["worst_power_at_least_080"]


def test_power_workload_cannot_be_reduced() -> None:
    with pytest.raises(a1.J2A1IntegrityError):
        a1.power_report(datasets=767)
    with pytest.raises(a1.J2A1IntegrityError):
        a1.power_report(bootstraps=198)


def test_runtime_storage_projection_uses_measured_p99() -> None:
    report = a1.runtime_storage_projection()
    assert report["passes"]
    distillation = report["distillation"]
    assert distillation["teacher_roots"] == 14_336
    assert distillation["teacher_calls"] == 7_340_032
    assert distillation["observed_p99_seconds"] == pytest.approx(
        a1.EXPECTED_V2_CENTRAL_P99,
        abs=0.0,
    )
    assert distillation["runtime_hours_with_25pct_margin"] == pytest.approx(
        41.98247721555496,
        abs=1e-12,
    )
    assert distillation["storage_with_25pct_margin_bytes"] == 15_099_494_400
    assert report["on_policy_training"][
        "runtime_hours_with_25pct_margin"
    ] == pytest.approx(5.093303720071912, abs=1e-12)


def test_runtime_admission_does_not_fall_back_to_mean_throughput(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = j2.load_hashed_json

    def changed(path: str | Path, *, field: str, root: Path = j2.REPO_ROOT):
        payload = original(path, field=field, root=root)
        if str(path).endswith("J2_TEACHER_PILOT_V2_CENTRAL_COST.json"):
            payload = json.loads(json.dumps(payload))
            payload["parallel_eight_process"]["calls_per_second"] = 1e9
            payload["parallel_eight_process"]["timing_summary"][
                "p99_seconds"
            ] = 1_000.0
        return payload

    monkeypatch.setattr(a1.j2, "load_hashed_json", changed)
    report = a1.runtime_storage_projection()
    assert not report["checks"]["measured_central_p99_exact"]
    assert not report["checks"]["central_p99_below_required_ceiling"]
    assert report["distillation"][
        "observed_calls_per_second_descriptive"
    ] == 1e9


def test_sensitivity_is_mandatory_but_nonconjunctive() -> None:
    report = a1.runtime_storage_projection()
    sensitivity = report["sensitivity_5000_moves"]
    assert sensitivity["diagnostic_not_conjunctive"]
    assert not sensitivity["distillation_runtime_fits_72h"]
    assert not sensitivity["distillation_storage_fits_24gib"]
    assert report["passes"]


def test_measured_memory_and_required_margins_are_explicit() -> None:
    report = a1.runtime_storage_projection()
    assert report["memory"][
        "maximum_contemporaneous_parent_children_rss_bytes"
    ] == a1.EXPECTED_V2_MAX_CONTEMPORANEOUS_RSS
    assert report["memory"][
        "conservative_independent_peak_sum_bytes"
    ] == a1.EXPECTED_V2_CONSERVATIVE_PEAK_RSS
    assert report["memory"]["headroom_bytes"] > 0
    assert report["distillation"]["p99_margin_seconds"] > 0
    assert report["on_policy_training"]["throughput_margin_ratio"] > 1


def test_family_limitation_and_future_support_gate_are_explicit() -> None:
    report = a1.family_support_safeguard()
    assert report["passes"]
    assert report["pilot_natural_feature_family_counts"] == {
        "low_air": 139,
        "low_constrained": 4_861,
        "mid_progression": 0,
        "upper_progression": 0,
    }
    gates = report["future_pre_checkpoint_gates"]
    assert gates["minimum_natural_states_per_family"] == 1_024
    assert gates["minimum_distinct_validation_roots_per_family"] == 256
    assert gates["maximum_natural_family_fraction"] == 0.70
    assert gates["maximum_capped_inventory_family_fraction"] == 0.40
    assert gates["checkpoint_authority_before_pass"] is False


def test_zero_work_audit_rejects_unexpected_file(tmp_path: Path) -> None:
    clean = a1.audit_zero_work(
        output_dir=tmp_path / "clean",
        include_operational=False,
    )
    assert clean["passes"]
    dirty = tmp_path / "dirty"
    dirty.mkdir()
    (dirty / "teacher_labels.json").write_text("[]", encoding="utf-8")
    report = a1.audit_zero_work(
        output_dir=dirty,
        include_operational=False,
    )
    assert not report["passes"]
    assert not report["checks"]["namespace_has_only_allowed_files"]


def test_cli_exposes_only_zero_work_reseal_verbs() -> None:
    parser = a1.build_parser()
    subparser = next(
        action
        for action in parser._actions
        if isinstance(action, __import__("argparse")._SubParsersAction)
    )
    assert set(subparser.choices) == {
        "audit-zero-work",
        "write-test-evidence",
        "prepare",
    }


def test_ast_has_no_execution_or_teacher_surface() -> None:
    tree = ast.parse(a1.RUNNER_PATH.read_text(encoding="utf-8"))
    function_names = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    forbidden = {
        "execute",
        "open",
        "reserve_streams",
        "consume_streams",
        "query_teacher",
        "run_game",
        "train",
        "fit",
        "promote",
    }
    assert not (function_names & forbidden)


def _passing_command() -> dict[str, object]:
    return {
        "name": "focused",
        "command": "pytest focused",
        "passed": 1,
        "failed": 0,
        "deselected": 0,
    }


def test_test_evidence_is_create_once(tmp_path: Path) -> None:
    output = tmp_path / "evidence"
    payload = a1.write_test_evidence(
        commands=[_passing_command()],
        deselections=[],
        output_dir=output,
    )
    assert payload["total_passed"] == 1
    before = (output / a1.TEST_EVIDENCE_NAME).read_bytes()
    with pytest.raises((FileExistsError, a1.J2A1IntegrityError)):
        a1.write_test_evidence(
            commands=[_passing_command()],
            deselections=[],
            output_dir=output,
        )
    assert (output / a1.TEST_EVIDENCE_NAME).read_bytes() == before


def test_failing_test_evidence_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(a1.J2A1IntegrityError):
        a1.write_test_evidence(
            commands=[
                {
                    "name": "bad",
                    "command": "pytest bad",
                    "passed": 1,
                    "failed": 1,
                }
            ],
            deselections=[],
            output_dir=tmp_path / "bad",
        )


def test_readiness_decision_precedence() -> None:
    ready = a1.readiness_decision(
        integrity_checks={"i": True},
        feasibility_checks={"f": True},
        operational_checks={"o": True},
    )
    hold = a1.readiness_decision(
        integrity_checks={"i": True},
        feasibility_checks={"f": False},
        operational_checks={"o": True},
    )
    kill = a1.readiness_decision(
        integrity_checks={"i": False},
        feasibility_checks={"f": True},
        operational_checks={"o": True},
    )
    assert ready["decision"] == a1.READY
    assert hold["decision"] == a1.HOLD
    assert kill["decision"] == a1.KILL


def _small_pass_report() -> dict[str, object]:
    return {"passes": True, "checks": {"pass": True}}


def _patch_prepare_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    *,
    power_pass: bool = True,
    source_pass: bool = True,
) -> None:
    operational = {"nice": True}
    monkeypatch.setattr(
        a1,
        "audit_zero_work",
        lambda **_kwargs: {
            "passes": True,
            "operational": {"checks": operational},
        },
    )
    monkeypatch.setattr(
        a1,
        "validate_test_evidence",
        lambda **_kwargs: {"passes": True},
    )
    monkeypatch.setattr(
        a1,
        "source_and_parent_audit",
        lambda: {"passes": source_pass},
    )
    monkeypatch.setattr(
        a1,
        "pilot_history_audit",
        lambda: {"passes": True},
    )
    monkeypatch.setattr(
        a1,
        "prospective_authority",
        lambda: {"passes": True},
    )
    monkeypatch.setattr(
        a1,
        "power_report",
        lambda **_kwargs: {
            "passes": power_pass,
            "checks": {
                "method": True,
                "worst_power_at_least_080": power_pass,
            },
            "score_fidelity": {"score_80pct_mde_percent": 4.5},
            "progression_common_or": {
                "worst_case_primary_power": (
                    0.81 if power_pass else 0.79
                )
            },
        },
    )
    monkeypatch.setattr(
        a1,
        "runtime_storage_projection",
        lambda: {
            "passes": True,
            "checks": {
                "method": True,
                "central_p99_below_required_ceiling": True,
                "distillation_runtime_within_72h": True,
                "distillation_storage_within_24gib": True,
                "online_runtime_within_72h": True,
                "online_storage_within_24gib": True,
                "online_sync_throughput_above_required_floor": True,
                "measured_memory_within_effective_cap": True,
            },
            "distillation": {},
            "on_policy_training": {},
            "memory": {},
            "sensitivity_5000_moves": {},
        },
    )
    monkeypatch.setattr(
        a1,
        "family_support_safeguard",
        lambda: {
            "passes": True,
            "future_pre_checkpoint_gates": {},
            "pilot_natural_feature_family_counts": {},
        },
    )


def _seed_evidence(path: Path) -> None:
    j2.write_immutable_json(
        path / a1.TEST_EVIDENCE_NAME,
        {"fixture": True},
        field="test_evidence_payload_sha256",
    )


@pytest.mark.parametrize(
    ("power_pass", "source_pass", "expected"),
    [
        (True, True, a1.READY),
        (False, True, a1.HOLD),
        (True, False, a1.KILL),
    ],
)
def test_prepare_seals_decision_precedence_and_retention(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    power_pass: bool,
    source_pass: bool,
    expected: str,
) -> None:
    output = tmp_path / expected
    _seed_evidence(output)
    _patch_prepare_dependencies(
        monkeypatch,
        power_pass=power_pass,
        source_pass=source_pass,
    )
    result = a1.prepare(output_dir=output)
    assert result["decision"] == expected
    assert result["execution_authorized"] is False
    assert result["zero_work"] == a1.ZERO_WORK
    retention = j2.load_hashed_json(
        output / a1.RETENTION_NAME,
        field="retention_payload_sha256",
    )
    assert retention["decision"] == expected
    assert retention["passes"]
    assert retention["file_count"] == 8
    assert not any(
        key.endswith("markers") and value
        for key, value in result["zero_work"].items()
    )


def test_prepare_rejects_reduced_power_workload(tmp_path: Path) -> None:
    with pytest.raises(a1.J2A1IntegrityError):
        a1.prepare(
            output_dir=tmp_path,
            power_datasets=a1.POWER_DATASETS - 1,
        )


def test_retention_inventory_is_canonical(tmp_path: Path) -> None:
    for name in ("a.json", "b.json"):
        (tmp_path / name).write_text(name, encoding="utf-8")
    payload = a1._retention_payload(
        output_dir=tmp_path,
        decision=a1.READY,
        names=("b.json", "a.json"),
    )
    assert [row["path"] for row in payload["files"]] == [
        "a.json",
        "b.json",
    ]
    assert payload["inventory_sha256"] == j2.canonical_json_hash(
        payload["files"]
    )


def test_all_prospective_science_counters_are_zero() -> None:
    assert set(a1.ZERO_WORK) == set(j2.ZERO_WORK)
    assert all(value == 0 for value in a1.ZERO_WORK.values())
    assert not any(
        token in a1.build_parser().format_help().lower()
        for token in ("execute", "reserve", "consume", "teacher-query")
    )
