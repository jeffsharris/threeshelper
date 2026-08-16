from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from threes_rl import j1a_cost_power_preflight as j1a


def _parent_projection() -> dict:
    path = (
        j1a.REPO_ROOT
        / "threes_rl"
        / "runs"
        / "forensics"
        / "j1_implementation_preflight_v1"
        / "J1_RUNTIME_STORAGE_PROJECTION.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _fake_power(
    *,
    n_roots,
    odds_ratio,
    base_rates,
    coupling,
    calibration_name,
    designs,
    bootstraps,
):
    assert calibration_name == "J1"
    return {
        "n_roots": n_roots,
        "roots_per_stream_stratum": n_roots // 8,
        "base_rates": list(base_rates),
        "coupling": coupling,
        "true_odds_ratio": odds_ratio,
        "designs": designs,
        "bootstrap_replicates": bootstraps,
        "gate_point_floor": 1.25,
        "gate_lower_ci_floor": 1.0,
        "seed": 7,
        "full_gate_power": 0.1,
        "monte_carlo_standard_error": 0.01,
        "mean_log_common_or": math.log(odds_ratio),
    }


def _decision_inputs(*, method=True, gate=True):
    return {
        "parent": {"passes": True},
        "arithmetic": {
            "progression": {
                "method_reproduction": {"passes": method},
                "accepted": gate,
            },
            "score": {
                "passes": gate,
                "checks": {
                    "parent_score_method_reproduced_exactly": True,
                },
            },
            "runtime_storage": {
                "passes": gate,
                "checks": {
                    "sealed_parent_projection_reproduced_exactly": True,
                },
            },
        },
        "streams": {"passes": True},
        "evidence": {"passes": True},
        "zero_work": {"passes": True},
        "operational": {"passes": True},
    }


def test_frozen_counts_and_total_arms():
    assert j1a.TRAIN_ROOTS == 16_384
    assert j1a.DEVELOPMENT_PAIRS == 896
    assert j1a.CONFIRMATION_PAIRS == 4_480
    assert j1a.TOTAL_GAME_ARMS == 27_136
    assert j1a.DEVELOPMENT_PAIRS % 64 == 0
    assert j1a.CONFIRMATION_PAIRS % 64 == 0


def test_score_power_reproduces_parent_and_amended_contract():
    parent_development = j1a.score_power_row(1_024)
    parent_confirmation = j1a.score_power_row(5_120)
    assert parent_development["mde_80pct_relative"] == (
        0.11564969641400724
    )
    assert parent_development["power_at_7pct"] == 0.40986098707230906
    assert parent_confirmation["mde_80pct_relative"] == (
        0.050159103285112305
    )
    assert parent_confirmation["power_at_7pct"] == 0.9721287292167085

    development = j1a.score_power_row(896)
    confirmation = j1a.score_power_row(4_480)
    assert development["mde_80pct_relative"] == 0.1241115511719173
    assert development["power_at_7pct"] == 0.3670152848352046
    assert confirmation["mde_80pct_relative"] == 0.05371377900036789
    assert confirmation["power_at_7pct"] == 0.9518340090000377
    assert confirmation["power_at_7pct"] >= 0.95
    assert confirmation["mde_80pct_relative"] < 0.055


def test_stream_contract_exact_and_collision_free():
    report = j1a.stream_contract()
    assert report["passes"]
    assert report["checks"]["total_game_arms_exact"]
    assert report["checks"]["all_namespace_ranges_disjoint"]
    assert report["checks"]["all_ranges_above_historical_ceiling"]
    assert report["checks"]["parent_denylist_contract_passed"]
    assert report["checks"]["amended_ranges_are_exact_parent_prefixes"]
    assert report["checks"][
        "parent_streams_were_not_reserved_or_consumed"
    ]
    assert report["prospective_unique_stream_id_count"] == 92_416
    assert report["checks"]["streams_not_reserved"]
    assert report["checks"]["streams_not_consumed"]
    assert not report["streams_reserved"]
    assert not report["streams_consumed"]
    assert len(report["contract_sha256"]) == 64


def test_parent_identity_audit_reproduces_all_seals():
    report = j1a.parent_identity_audit()
    assert report["passes"]
    assert report["parent_terminal_decision"] == (
        "HOLD_J1_IMPLEMENTATION_PREFLIGHT"
    )
    assert not report["parent_marker_paths"]
    assert all(row["matches"] for row in report["files"].values())
    assert all(row["matches"] for row in report["payloads"].values())


def test_exact_preserved_progression_source_reproduces_published_cells():
    development = j1a.progression_power_summary(
        1_024,
        odds_ratios=(1.50, 2.50),
    )
    confirmation = j1a.progression_power_summary(
        5_120,
        odds_ratios=(1.25, 1.50),
    )
    assert development["worst_by_or"]["1.50"]["power"] == 0.30078125
    assert development["worst_by_or"]["2.50"]["power"] == 0.9453125
    assert development["mde_80pct_grid"] == 2.5
    assert confirmation["worst_by_or"]["1.50"]["power"] == (
        0.8854166666666666
    )
    assert confirmation["mde_80pct_grid"] == 1.5


def test_method_mismatch_holds_before_amended_power():
    report = j1a.progression_power_report(
        simulator=_fake_power,
        datasets=8,
        bootstraps=3,
    )
    assert not report["method_reproduction"]["passes"]
    assert report["amended"] is None
    assert not report["accepted"]
    assert report["decision"] == "HOLD_METHOD_REPRODUCTION"


def test_progression_contract_rejects_invalid_width_and_drift():
    with pytest.raises(ValueError, match="divisible by 8"):
        j1a.progression_power_summary(
            897,
            simulator=_fake_power,
            odds_ratios=(1.5,),
            control_rates=(0.02,),
            couplings=(0.0,),
            datasets=8,
            bootstraps=3,
        )

    def drifted(**kwargs):
        row = dict(_fake_power(**kwargs))
        row["bootstrap_replicates"] -= 1
        return row

    with pytest.raises(j1a.J1AIntegrityError, match="contract drift"):
        j1a.progression_power_summary(
            896,
            simulator=drifted,
            odds_ratios=(1.5,),
            control_rates=(0.02,),
            couplings=(0.0,),
            datasets=8,
            bootstraps=3,
        )


def test_runtime_projection_reproduces_parent_and_clears_headroom():
    report = j1a.runtime_storage_projection(_parent_projection())
    assert report["passes"]
    assert report["parent_reproduction"]["passes"]
    assert all(report["parent_reproduction"]["checks"].values())
    phases = report["amended_phase_projections"]
    assert phases["training"]["complete_game_arms"] == 16_384
    assert phases["development"]["complete_game_arms"] == 1_792
    assert phases["confirmation"]["complete_game_arms"] == 8_960
    assert phases["development"]["runtime_cap_fraction_after_margin"] == (
        0.9016159389719919
    )
    assert phases["confirmation"]["runtime_cap_fraction_after_margin"] == (
        0.9016159389719919
    )
    assert phases["development"]["runtime_at_most_91pct_cap"]
    assert phases["confirmation"]["runtime_at_most_91pct_cap"]
    assert phases["training"][
        "contract_max_5000_move_sensitivity_hours_with_25pct_margin"
    ] == 1.7407226738416486
    assert phases["development"][
        "contract_max_5000_move_sensitivity_hours_with_25pct_margin"
    ] == 211.3162356965606
    assert phases["confirmation"][
        "contract_max_5000_move_sensitivity_hours_with_25pct_margin"
    ] == 1056.581178482803
    assert phases["development"][
        "contract_max_5000_move_sensitivity_runtime_passes"
    ] is False
    assert phases["confirmation"][
        "contract_max_5000_move_sensitivity_runtime_passes"
    ] is False


def test_runtime_projection_fails_parent_fixture_drift():
    parent = _parent_projection()
    parent["fixture_timing"]["actor_batch"]["p90_seconds"] *= 2
    report = j1a.runtime_storage_projection(parent)
    assert not report["parent_reproduction"]["passes"]
    assert not report["checks"][
        "sealed_parent_projection_reproduced_exactly"
    ]
    assert not report["passes"]


def test_decision_precedence_method_reproduction():
    decision, _checks = j1a._decision(
        **_decision_inputs(method=False, gate=False)
    )
    assert decision == "HOLD_METHOD_REPRODUCTION"


def test_decision_ready_and_cost_hold():
    ready, ready_checks = j1a._decision(**_decision_inputs())
    assert ready == "READY_J1A_COST_POWER_AMENDMENT"
    assert all(ready_checks.values())

    held_inputs = _decision_inputs()
    held_inputs["arithmetic"]["runtime_storage"]["passes"] = False
    held, _held_checks = j1a._decision(**held_inputs)
    assert held == "HOLD_J1A_COST_POWER_AMENDMENT"


def test_payload_hash_and_immutable_write(tmp_path):
    path = tmp_path / "artifact.json"
    payload = j1a.write_immutable_json(
        path,
        {"version": "fixture", "value": 1},
        field="payload_sha256",
    )
    assert j1a.verify_payload_hash(payload, "payload_sha256")
    assert (
        json.loads(path.read_text(encoding="utf-8"))["payload_sha256"]
        == payload["payload_sha256"]
    )
    with pytest.raises(FileExistsError):
        j1a.write_immutable_json(
            path,
            {"version": "fixture", "value": 1},
            field="payload_sha256",
        )


def test_test_evidence_requires_fresh_namespace(tmp_path):
    output = tmp_path / "j1a"
    evidence = j1a.write_test_evidence(
        output_dir=output,
        focused_command=j1a.FOCUSED_TEST_COMMAND,
        focused_passed=j1a.FOCUSED_TEST_COUNT,
        parent_command=j1a.PARENT_TEST_COMMAND,
        parent_passed=j1a.PARENT_TEST_COUNT,
    )
    assert evidence["passes"]
    assert evidence["zero_work"] == j1a.ZERO_WORK
    with pytest.raises(FileExistsError, match="must be fresh"):
        j1a.write_test_evidence(
            output_dir=output,
            focused_command=j1a.FOCUSED_TEST_COMMAND,
            focused_passed=j1a.FOCUSED_TEST_COUNT,
            parent_command=j1a.PARENT_TEST_COMMAND,
            parent_passed=j1a.PARENT_TEST_COUNT,
        )


def test_test_evidence_rejects_wrong_count_before_write(tmp_path):
    output = tmp_path / "j1a"
    with pytest.raises(j1a.J1AIntegrityError, match="frozen commands"):
        j1a.write_test_evidence(
            output_dir=output,
            focused_command=j1a.FOCUSED_TEST_COMMAND,
            focused_passed=j1a.FOCUSED_TEST_COUNT - 1,
            parent_command=j1a.PARENT_TEST_COMMAND,
            parent_passed=j1a.PARENT_TEST_COUNT,
        )
    assert not output.exists()


def test_zero_work_fails_closed_on_marker(tmp_path):
    output = tmp_path / "j1a"
    output.mkdir()
    (output / j1a.TEST_EVIDENCE_NAME).write_text("{}", encoding="utf-8")
    (output / "EXECUTION_MARKER.json").write_text("{}", encoding="utf-8")
    report = j1a.zero_work_audit(output)
    assert not report["passes"]
    assert report["forbidden_marker_paths"] == ["EXECUTION_MARKER.json"]


def test_cli_surface_has_no_execution_or_marker_command():
    parser = j1a.build_parser()
    choices = parser._subparsers._group_actions[0].choices
    assert set(choices) == {"write-test-evidence", "prepare"}
    assert "execute" not in choices
    assert "open" not in choices
    assert "reserve" not in choices
    source = j1a.RUNNER_PATH.read_text(encoding="utf-8")
    assert "normal_start_sim(" not in source
    assert "optimizer.step(" not in source
    assert "make_policy(" not in source


def test_expected_o2_power_source_hash_is_literal_and_current():
    path = j1a.REPO_ROOT / "threes_rl" / "o2_online_option_preflight.py"
    assert j1a.sha256_path(path) == j1a.EXPECTED_O2_POWER_SOURCE_SHA256
    assert (
        j1a.PARENT_FILES["threes_rl/o2_online_option_preflight.py"]
        == j1a.EXPECTED_O2_POWER_SOURCE_SHA256
    )


def test_zero_work_contract_is_all_zero():
    assert j1a.ZERO_WORK
    assert all(value == 0 for value in j1a.ZERO_WORK.values())
    assert j1a.OUTPUT_DIR.name == "j1a_cost_power_amendment_v1"
    assert not any(
        name in {
            "EXECUTION_MARKER.json",
            "stream_manifest.json",
            "checkpoint.pt",
        }
        for name in (
            j1a.TEST_EVIDENCE_NAME,
            j1a.ARITHMETIC_NAME,
            j1a.PREFLIGHT_LOCK_NAME,
            j1a.PREFLIGHT_RESULT_NAME,
        )
    )
