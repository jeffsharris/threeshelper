"""Serialization-safe orchestration for the unchanged O4 P0 science."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Mapping

from threes_rl import g1r_acquire as history
from threes_rl import o4_p0_preflight as science
from threes_rl.g1r_qd_admission_v2 import _heavy_process_audit
from threes_rl.s3_power_preflight import sha256_path


VERSION = "o4_domain_safe_p0_v2_serialization"
AMENDMENT_PATH = Path(
    "threes_rl/O4_DOMAIN_SAFE_P0_AMENDMENT_V2_SERIALIZATION.md"
)
RUNNER_PATH = Path("threes_rl/o4_p0_preflight_v2.py")
TEST_PATH = Path("tests/test_rl_o4_p0_preflight_v2.py")
OUTPUT_DIR = Path("threes_rl/runs/forensics/o4_domain_safe_p0_v2")
TEST_EVIDENCE_PATH = Path(
    "threes_rl/runs/forensics/o4_domain_safe_p0_test_evidence_v2.json"
)
MARKER_NAME = "O4_P0_V2_OPENED.json"
RESULT_NAME = "O4_P0_V2_RESULT.json"
V1_HOLD_NAME = "O4_P0_V1_ENGINEERING_HOLD.json"
V1_MARKER_PATH = science.OUTPUT_DIR / science.MARKER_NAME
V1_HOLD_PATH = science.OUTPUT_DIR / V1_HOLD_NAME
V1_MARKER_FILE_SHA256 = (
    "7f84bbd9679b9d6294a0530b47b5ba01749426191a1a3f509bf38a48114723b6"
)
V1_MARKER_PAYLOAD_SHA256 = (
    "854822ffb6bd6b23cae646c684475293e9f65c8b35a86267c5616f39f2d55679"
)
V1_RUNNER_SHA256 = (
    "81fa03580f35de8734ec17973bec1538f1c680285a9beb1d133454a7f5b6c87e"
)
V1_TEST_EVIDENCE_FILE_SHA256 = (
    "9325646eeb94b9b28f5acb6a2d482a0240df89cbfb14d54a6857783f2f8f0938"
)
V1_TEST_EVIDENCE_PAYLOAD_SHA256 = (
    "78fc9c3a4fdbc2717f14b97891fc398538a5650c7d5bc3ea2fe84b1a2b9b883b"
)
V1_ERROR = (
    "SourceIntegrityError: O4 P0 marker binding mismatch: "
    "role_family_target_counts"
)


def _normalize(value: Any) -> Any:
    return json.loads(
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )


def _same_json(left: Any, right: Any) -> bool:
    return science.canonical_json_hash(_normalize(left)) == (
        science.canonical_json_hash(_normalize(right))
    )


def seal_v1_engineering_hold() -> dict[str, Any]:
    if V1_HOLD_PATH.exists():
        raise FileExistsError(f"V1 engineering HOLD already exists: {V1_HOLD_PATH}")
    files = {path.name for path in science.OUTPUT_DIR.iterdir() if path.is_file()}
    if files != {science.MARKER_NAME}:
        raise science.SourceIntegrityError(
            f"V1 namespace was not zero-content: {sorted(files)}"
        )
    marker = json.loads(V1_MARKER_PATH.read_text())
    checks = {
        "v1_marker_file_exact": sha256_path(V1_MARKER_PATH)
        == V1_MARKER_FILE_SHA256,
        "v1_marker_payload_exact": marker.get("opened_payload_sha256")
        == V1_MARKER_PAYLOAD_SHA256,
        "v1_marker_self_hash_valid": science._verify_self_hash(
            marker,
            "opened_payload_sha256",
        ),
        "v1_runner_exact": sha256_path(science.RUNNER_PATH) == V1_RUNNER_SHA256,
        "v1_test_evidence_exact": sha256_path(science.TEST_EVIDENCE_PATH)
        == V1_TEST_EVIDENCE_FILE_SHA256,
        "no_v1_terminal_result": not (
            science.OUTPUT_DIR / science.RESULT_NAME
        ).exists(),
        "only_v1_marker_pre_hold": files == {science.MARKER_NAME},
    }
    if not all(checks.values()):
        raise science.SourceIntegrityError(
            f"V1 engineering HOLD checks failed: {checks}"
        )
    return science._write_immutable_json(
        V1_HOLD_PATH,
        {
            "version": f"{VERSION}_v1_hold",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "decision": "HOLD_O4_P0_V1_ORCHESTRATION",
            "error": V1_ERROR,
            "checks": checks,
            "zero_work": {
                "source_replay_bodies_opened": 0,
                "support_content_opened": 0,
                "games": 0,
                "streams_consumed": 0,
                "labels": 0,
                "models_fit": 0,
                "policy_outcomes": 0,
                "scores_or_actions_inspected": 0,
                "dashboard_changes": 0,
            },
            "continue": "O4 P0 V2 serialization-only amendment",
            "kill": False,
            "promote": False,
        },
        self_hash_field="hold_payload_sha256",
    )


def _load_v1_hold() -> dict[str, Any]:
    payload = json.loads(V1_HOLD_PATH.read_text())
    if not science._verify_self_hash(payload, "hold_payload_sha256"):
        raise science.SourceIntegrityError("V1 engineering HOLD self hash mismatch")
    if payload.get("decision") != "HOLD_O4_P0_V1_ORCHESTRATION":
        raise science.SourceIntegrityError("V1 engineering HOLD decision changed")
    if payload.get("error") != V1_ERROR:
        raise science.SourceIntegrityError("V1 engineering HOLD error changed")
    return payload


def _load_test_evidence() -> dict[str, Any]:
    payload = json.loads(TEST_EVIDENCE_PATH.read_text())
    if not science._verify_self_hash(payload, "test_evidence_payload_sha256"):
        raise science.SourceIntegrityError("O4 V2 test evidence self hash mismatch")
    expected = {
        "amendment_sha256": sha256_path(AMENDMENT_PATH),
        "orchestration_runner_sha256": sha256_path(RUNNER_PATH),
        "orchestration_tests_sha256": sha256_path(TEST_PATH),
        "scientific_runner_sha256": sha256_path(science.RUNNER_PATH),
        "scientific_charter_sha256": sha256_path(science.CHARTER_PATH),
        "scientific_option_sha256": sha256_path(science.OPTION_PATH),
        "scientific_power_sha256": sha256_path(science.POWER_PATH),
        "scientific_option_tests_sha256": sha256_path(science.OPTION_TEST_PATH),
        "scientific_p0_tests_sha256": sha256_path(science.TEST_PATH),
        "v1_marker_file_sha256": sha256_path(V1_MARKER_PATH),
        "v1_hold_file_sha256": sha256_path(V1_HOLD_PATH),
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise science.SourceIntegrityError("O4 V2 test evidence binding changed")
    if not payload.get("passes"):
        raise science.SourceIntegrityError("O4 V2 tests did not pass")
    return payload


def seal_test_evidence(
    *,
    focused_passed: int,
    regression_passed: int,
    commands: list[str],
) -> dict[str, Any]:
    hold = _load_v1_hold()
    return science._write_immutable_json(
        TEST_EVIDENCE_PATH,
        {
            "version": f"{VERSION}_test_evidence",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "amendment_sha256": sha256_path(AMENDMENT_PATH),
            "orchestration_runner_sha256": sha256_path(RUNNER_PATH),
            "orchestration_tests_sha256": sha256_path(TEST_PATH),
            "scientific_runner_sha256": sha256_path(science.RUNNER_PATH),
            "scientific_charter_sha256": sha256_path(science.CHARTER_PATH),
            "scientific_option_sha256": sha256_path(science.OPTION_PATH),
            "scientific_power_sha256": sha256_path(science.POWER_PATH),
            "scientific_option_tests_sha256": sha256_path(
                science.OPTION_TEST_PATH
            ),
            "scientific_p0_tests_sha256": sha256_path(science.TEST_PATH),
            "v1_marker_file_sha256": sha256_path(V1_MARKER_PATH),
            "v1_marker_payload_sha256": V1_MARKER_PAYLOAD_SHA256,
            "v1_hold_file_sha256": sha256_path(V1_HOLD_PATH),
            "v1_hold_payload_sha256": hold["hold_payload_sha256"],
            "focused_tests_passed": int(focused_passed),
            "regression_tests_passed": int(regression_passed),
            "commands": commands,
            "passes": True,
            "scientific_contract_changed": False,
            "games": 0,
            "streams_consumed": 0,
            "source_replay_bodies_opened": 0,
            "labels": 0,
            "models_fit": 0,
            "policy_outcomes": 0,
        },
        self_hash_field="test_evidence_payload_sha256",
    )


def _commands(out_dir: Path) -> dict[str, str]:
    base = (
        "zsh -ic 'no-secrets nice -n 10 env PYTHONPATH=. "
        ".venv/bin/python -m threes_rl.o4_p0_preflight_v2"
    )
    suffix = f" --out-dir {out_dir}'"
    return {
        "open": f"{base} open{suffix}",
        "run": f"{base} run{suffix}",
    }


def _bindings(out_dir: Path) -> dict[str, Any]:
    evidence = _load_test_evidence()
    hold = _load_v1_hold()
    return _normalize(
        {
            "version": VERSION,
            "bound_out_dir": str(out_dir.resolve()),
            "amendment_sha256": sha256_path(AMENDMENT_PATH),
            "orchestration_runner_sha256": sha256_path(RUNNER_PATH),
            "orchestration_tests_sha256": sha256_path(TEST_PATH),
            "test_evidence_file_sha256": sha256_path(TEST_EVIDENCE_PATH),
            "test_evidence_payload_sha256": evidence[
                "test_evidence_payload_sha256"
            ],
            "v1_marker_file_sha256": sha256_path(V1_MARKER_PATH),
            "v1_marker_payload_sha256": V1_MARKER_PAYLOAD_SHA256,
            "v1_hold_file_sha256": sha256_path(V1_HOLD_PATH),
            "v1_hold_payload_sha256": hold["hold_payload_sha256"],
            "scientific_bindings": science._current_bindings(),
            "source_universe_roots": science.O3_ACQUISITION_ROOTS,
            "selected_o3_roots_excluded": science.O3_SELECTED_ROOTS,
            "o4_root_count": science.TOTAL_SELECTED_ROOTS,
            "future_stream_manifest_sha256": science.canonical_json_hash(
                science.future_stream_rows()
            ),
            "future_stream_row_count": len(science.future_stream_rows()),
            "commands": _commands(out_dir),
        }
    )


def open_preflight(out_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    if out_dir.resolve() != OUTPUT_DIR.resolve():
        raise ValueError("O4 P0 V2 output directory is immutable")
    if out_dir.exists():
        raise FileExistsError(f"O4 P0 V2 namespace exists: {out_dir}")
    _load_test_evidence()
    _load_v1_hold()
    immutable = {
        path: {
            "expected": expected,
            "actual": sha256_path(Path(path)),
        }
        for path, expected in science.IMMUTABLE_INPUT_HASHES.items()
    }
    heavy = _heavy_process_audit()
    services = history.service_health()
    free_gib = shutil.disk_usage(out_dir.parent).free / 1024**3
    checks = {
        "v1_hold_exact": True,
        "test_evidence_exact": True,
        "immutable_input_file_hashes_exact": all(
            row["actual"] == row["expected"] for row in immutable.values()
        ),
        "nice_at_least_10": history.current_nice() >= science.MINIMUM_NICE,
        "no_competing_heavy_process": heavy["passes"],
        "free_disk_above_120_gib": free_gib > science.TARGET_FREE_GIB,
        "free_disk_above_100_gib": free_gib > science.MIN_FREE_GIB,
        "services_dashboard_top_three": services["passes"],
        "zero_prior_v2_namespace": True,
        "zero_games_streams_labels_models_outcomes": True,
    }
    if not all(checks.values()):
        raise science.OperationalHold(f"O4 P0 V2 open failed: {checks}")
    marker = {
        **_bindings(out_dir),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "decision": "O4_P0_V2_OPENED_ZERO_WORK",
        "preopen": {
            "heavy_process_audit": heavy,
            "service_health": services,
            "free_gib": free_gib,
            "nice": history.current_nice(),
            "immutable_input_file_hashes": immutable,
        },
        "checks": checks,
        "zero_work": {
            "source_replay_bodies_opened": 0,
            "games": 0,
            "streams_consumed": 0,
            "labels": 0,
            "models_fit": 0,
            "policy_outcomes": 0,
            "scores_or_actions_inspected": 0,
            "dashboard_changes": 0,
        },
    }
    return science._write_immutable_json(
        out_dir / MARKER_NAME,
        marker,
        self_hash_field="opened_payload_sha256",
    )


def _load_marker(out_dir: Path) -> dict[str, Any]:
    path = out_dir / MARKER_NAME
    if not path.is_file():
        raise FileNotFoundError("O4 P0 V2 marker is missing")
    marker = json.loads(path.read_text())
    if not science._verify_self_hash(marker, "opened_payload_sha256"):
        raise science.SourceIntegrityError("O4 P0 V2 marker self hash mismatch")
    expected = _bindings(out_dir)
    for key, value in expected.items():
        if not _same_json(marker.get(key), value):
            raise science.SourceIntegrityError(
                f"O4 P0 V2 marker binding mismatch: {key}"
            )
    return marker


def _write_manifest(
    out_dir: Path,
    name: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    body = science._write_immutable_json(
        out_dir / name,
        payload,
        self_hash_field="payload_sha256",
    )
    return {
        "path": str(out_dir / name),
        "file_sha256": sha256_path(out_dir / name),
        "payload_sha256": body["payload_sha256"],
    }


def run_preflight(out_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    marker = _load_marker(out_dir)
    result_path = out_dir / RESULT_NAME
    if result_path.exists():
        raise FileExistsError("O4 P0 V2 terminal result exists")
    unexpected = {
        path.name for path in out_dir.iterdir() if path.name != MARKER_NAME
    }
    if unexpected:
        raise science.SourceIntegrityError(
            f"O4 P0 V2 namespace contains unexpected work: {sorted(unexpected)}"
        )
    try:
        tests = _load_test_evidence()
        immutable = science.immutable_input_audit()
        domain = science.domain_proof()
        source_report, candidates, _union_index = science.load_source_pool()
        replay_sources = science.verify_candidate_source_replays(candidates)
        restore_report, allocation = science.support_and_allocation(
            source_report,
            candidates,
        )
        rows = science.future_stream_rows()
        streams = science.stream_contract(rows)
        collision = science.collision_audit(rows, out_dir=out_dir)
        power = science.power_table()
        policies = science.policy_audit()
        heavy = _heavy_process_audit()
        services = history.service_health()
        free_gib = shutil.disk_usage(out_dir).free / 1024**3

        artifacts = {
            "domain": _write_manifest(
                out_dir,
                science.DOMAIN_NAME,
                {
                    "version": f"{VERSION}_domain",
                    **domain,
                    "outcomes_opened": False,
                },
            ),
            "source_pool": _write_manifest(
                out_dir,
                science.SOURCE_NAME,
                {
                    "version": f"{VERSION}_source",
                    **source_report,
                    "geometry_restore": restore_report,
                },
            ),
            "source_replays": _write_manifest(
                out_dir,
                science.SOURCE_REPLAY_NAME,
                {
                    "version": f"{VERSION}_source_replays",
                    **replay_sources,
                },
            ),
            "selection": _write_manifest(
                out_dir,
                science.SELECTION_NAME,
                {
                    "version": f"{VERSION}_selection",
                    **allocation,
                    "labels_generated": 0,
                    "policy_outcomes_opened": False,
                },
            ),
            "streams": _write_manifest(
                out_dir,
                science.STREAM_NAME,
                {
                    "version": f"{VERSION}_streams",
                    "rows": rows,
                    "contract": streams,
                    "streams_consumed": 0,
                },
            ),
            "collision": _write_manifest(
                out_dir,
                science.COLLISION_NAME,
                {
                    "version": f"{VERSION}_collision",
                    **collision,
                },
            ),
            "power": _write_manifest(
                out_dir,
                science.POWER_NAME,
                {
                    "version": f"{VERSION}_power",
                    **power,
                    "outcomes_used": False,
                },
            ),
            "policies": _write_manifest(
                out_dir,
                science.POLICY_NAME,
                {
                    "version": f"{VERSION}_policies",
                    **policies,
                },
            ),
        }
        integrity_checks = {
            "immutable_source_identities": immutable["passes"],
            "domain_proof": domain["passes"],
            "source_pool_integrity": source_report["passes"],
            "source_replay_hashes": replay_sources["passes"],
            "permitted_geometry_restore": restore_report["passes"],
            "tests_exact": tests["passes"],
            "stream_contract": streams["passes"],
            "stream_collisions_zero": collision["passes"],
            "policy_identities_and_signatures": policies["passes"],
        }
        support_checks = {
            "raw_support_upper_bound_feasible": source_report[
                "upper_bound_feasible"
            ],
            "exact_448_allocation": allocation["passes"],
            "n192_or150_power": power["passes"],
            "one_heavy_job": heavy["passes"],
            "nice_at_least_10": history.current_nice()
            >= science.MINIMUM_NICE,
            "free_disk_above_120_gib": free_gib > science.TARGET_FREE_GIB,
            "free_disk_above_100_gib": free_gib > science.MIN_FREE_GIB,
            "services_dashboard_top_three": services["passes"],
            "projected_runtime_frozen": science.PROJECTED_ACTIVE_SECONDS
            == 18 * 3_600,
            "projected_storage_below_4_gib": science.PROJECTED_STORAGE_BYTES
            < science.STORAGE_CAP_BYTES,
            "zero_fresh_work": True,
        }
        decision = science._decision(
            integrity_checks=integrity_checks,
            support_checks=support_checks,
        )
        result = {
            "version": VERSION,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "decision": decision,
            "continue": (
                "O4 acquisition/training requires separate authorization"
                if decision == "READY_O4_DOMAIN_SAFE_OPTION_PREFLIGHT"
                else "NONE"
            ),
            "hold": [
                "o4_acquisition",
                "o4_training",
                "o4_mechanism_outcomes",
                "normal_start_development",
                "confirmation",
                "promotion",
            ],
            "kill": decision == "KILL_O4_REPRESENTATION_PREFLIGHT",
            "promote": False,
            "v1_engineering_hold": {
                "path": str(V1_HOLD_PATH),
                "file_sha256": sha256_path(V1_HOLD_PATH),
                "payload_sha256": _load_v1_hold()["hold_payload_sha256"],
            },
            "marker": {
                "path": str(out_dir / MARKER_NAME),
                "file_sha256": sha256_path(out_dir / MARKER_NAME),
                "payload_sha256": marker["opened_payload_sha256"],
            },
            "artifacts": artifacts,
            "summaries": {
                "source_universe": source_report["union_roots"],
                "selected_o3_roots_excluded": source_report[
                    "selected_roots_excluded"
                ],
                "unselected_source_universe": source_report[
                    "unselected_root_universe"
                ],
                "candidate_root_upper_bounds_by_family": source_report[
                    "family_root_upper_bounds"
                ],
                "candidate_root_upper_bounds_by_family_target": source_report[
                    "family_target_root_upper_bounds"
                ],
                "allocation_count": len(allocation["selected"]),
                "allocation_deficits": allocation["deficits"],
                "stream_rows_reserved": streams["row_count"],
                "o3_learning_rows_explicitly_checked": collision[
                    "o3_learning_stream_audit"
                ]["learning_rows"],
                "collision_source_count": collision["matched_source_count"],
                "power_n": power["selected_roots"],
                "power_or150": next(
                    row["power_full_gate"]
                    for row in power["rows"]
                    if row["true_common_odds_ratio"] == 1.50
                ),
                "grid_mde": power["grid_mde"],
                "schema_sha256": domain["schema_sha256"],
                "parameter_count": domain["parameter_count"],
                "family_signatures": policies["signatures"],
            },
            "integrity_checks": integrity_checks,
            "support_checks": support_checks,
            "process": {
                "nice": history.current_nice(),
                "heavy_process_audit": heavy,
                "free_gib": free_gib,
                "service_health": services,
                "projected_active_seconds": science.PROJECTED_ACTIVE_SECONDS,
                "projected_storage_bytes": science.PROJECTED_STORAGE_BYTES,
                "storage_cap_bytes": science.STORAGE_CAP_BYTES,
            },
            "zero_work": {
                "fresh_games": 0,
                "fresh_streams_consumed": 0,
                "o3_option_training_bodies_read": False,
                "o3_selected_replay_bodies_read": False,
                "final_score_action_outcome_fields_read": False,
                "labels": 0,
                "models_fit": 0,
                "policy_outcomes": 0,
                "incumbent_changed": False,
                "dashboard_changed": False,
            },
        }
    except science.SourceIntegrityError as error:
        result = {
            "version": VERSION,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "decision": "KILL_O4_REPRESENTATION_PREFLIGHT",
            "continue": "NONE",
            "hold": ["all_o4_execution"],
            "kill": True,
            "promote": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "zero_work": {
                "fresh_games": 0,
                "fresh_streams_consumed": 0,
                "labels": 0,
                "models_fit": 0,
                "policy_outcomes": 0,
                "incumbent_changed": False,
                "dashboard_changed": False,
            },
        }
    except Exception as error:
        result = {
            "version": VERSION,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "decision": "HOLD_O4_DATA_SUPPORT",
            "continue": "NONE",
            "hold": ["all_o4_execution"],
            "kill": False,
            "promote": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "zero_work": {
                "fresh_games": 0,
                "fresh_streams_consumed": 0,
                "labels": 0,
                "models_fit": 0,
                "policy_outcomes": 0,
                "incumbent_changed": False,
                "dashboard_changed": False,
            },
        }
    return science._write_immutable_json(
        result_path,
        result,
        self_hash_field="result_payload_sha256",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("seal-v1-hold")
    for command in ("open", "run"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    evidence = subparsers.add_parser("seal-test-evidence")
    evidence.add_argument("--focused-passed", type=int, required=True)
    evidence.add_argument("--regression-passed", type=int, required=True)
    evidence.add_argument(
        "--test-command",
        dest="test_commands",
        action="append",
        default=[],
    )
    args = parser.parse_args()
    if args.command == "seal-v1-hold":
        result = seal_v1_engineering_hold()
    elif args.command == "open":
        result = open_preflight(args.out_dir)
    elif args.command == "run":
        result = run_preflight(args.out_dir)
    else:
        result = seal_test_evidence(
            focused_passed=args.focused_passed,
            regression_passed=args.regression_passed,
            commands=list(args.test_commands),
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
