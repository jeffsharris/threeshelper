"""Scan-only recovery for the immutable O2 128-root yield pilot."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from threes_rl import g1r_acquire as history
from threes_rl import o2_yield_pilot as pilot


VERSION = "o2_yield_pilot_scan_recovery_v1"
ROOT = Path("threes_rl/runs")
CHARTER_PATH = Path("threes_rl/O2_YIELD_PILOT_SCAN_RECOVERY_CHARTER.md")
RUNNER_PATH = Path("threes_rl/o2_yield_pilot_scan_recovery.py")
TEST_PATH = Path("tests/test_rl_o2_yield_pilot_scan_recovery.py")
TEST_EVIDENCE_PATH = (
    ROOT / "forensics/o2_yield_pilot_scan_recovery_test_evidence.json"
)
OUTPUT_DIR = ROOT / "forensics/o2_yield_pilot_scan_recovery_v1"
MARKER_PATH = OUTPUT_DIR / "O2_SCAN_RECOVERY_OPENED.json"
SUPPORT_PATH = OUTPUT_DIR / "O2_RECOVERED_SUPPORT.json"
RESULT_PATH = OUTPUT_DIR / "O2_SCAN_RECOVERY_RESULT.json"

ORIGINAL_DIR = ROOT / "forensics/o2_yield_pilot_v1"
ORIGINAL_CHARTER = Path("threes_rl/O2_YIELD_PILOT_EXECUTION_CHARTER.md")
ORIGINAL_RUNNER = Path("threes_rl/o2_yield_pilot.py")
ORIGINAL_TESTS = Path("tests/test_rl_o2_yield_pilot.py")
ORIGINAL_TEST_EVIDENCE = (
    ROOT / "forensics/o2_yield_pilot_test_evidence.json"
)
ORIGINAL_MARKER = ORIGINAL_DIR / "O2_YIELD_PILOT_EXECUTION_OPENED.json"
ORIGINAL_RESULT = ORIGINAL_DIR / "O2_YIELD_PILOT_RESULT.json"
ORIGINAL_ATTEMPTS = ORIGINAL_DIR / "attempts.jsonl"
ORIGINAL_COMPLETIONS = ORIGINAL_DIR / "completed_games.jsonl"
ORIGINAL_RUNTIME = ORIGINAL_DIR / "runtime_state.json"
ORIGINAL_SUPPORT = ORIGINAL_DIR / "O2_PILOT_SUPPORT.json"
ORIGINAL_REPLAY_DIR = ORIGINAL_DIR / "source_replays"

EXPECTED_ORIGINAL_HASHES = {
    ORIGINAL_CHARTER: (
        "4c015f0238ae7e6fe545cf497b58576890a733656995fe0f6507767e2e3391b3"
    ),
    ORIGINAL_RUNNER: (
        "5af6e741612ca6862d821e688cfd92974beae5f33d0b926a815056efb4d95247"
    ),
    ORIGINAL_TESTS: (
        "058bc390a27c51c241b9f7acb6009d193c26c0b9e66dab940746bf31be4751e1"
    ),
    ORIGINAL_TEST_EVIDENCE: (
        "f34f1a45896651b770da25416f1dd351a77c5547610e3a18093291933e15183b"
    ),
    ORIGINAL_MARKER: (
        "bbce7dd41c84ea5c6e1985a70529f284515ae6dec79c405581ce383c4a3c6457"
    ),
    ORIGINAL_RESULT: (
        "f443a76392f09052179d8a9b458dd2d3ff615072c6d48024fafc5ef11b9ce576"
    ),
    ORIGINAL_ATTEMPTS: (
        "33e9062e748495cb2ac7f02be2be223b19abc875de6ea0624e1327fe63750948"
    ),
    ORIGINAL_COMPLETIONS: (
        "de99abc9b096aaa6f12606a4a345417cfdadd7c16b8ed7a66ede4d09ab829eae"
    ),
    ORIGINAL_RUNTIME: (
        "4cc9ab89222a13ad39405ea75cb43abc7f0141fe875faf49b8ccfedb3a259138"
    ),
}
EXPECTED_PAYLOAD_HASHES = {
    ORIGINAL_TEST_EVIDENCE: (
        "bd2626069c186ff42b44adeb9739bbe3b8c311a3e56e5d53ea70111d0e9df2a2"
    ),
    ORIGINAL_MARKER: (
        "23e704420d369c3b3410a8feb4b4620896a1ad96d90a5cc3df0128073234dc4b"
    ),
    ORIGINAL_RESULT: (
        "0f3de62667ac13734ec45118de4179e40f8aaf819b0f570847de285e415333d2"
    ),
}
COMPLETION_FIELDS = {
    "complete",
    "dashboard_eligible",
    "deck_stream_id",
    "family",
    "family_index",
    "game_index",
    "logical_seed",
    "nominal_family",
    "policy_stream_id",
    "root_cluster",
    "slot_stream_id",
    "source_replay",
    "source_replay_sha256",
}
EXPECTED_TOP_THREE = (263670, 261369, 258561)
TOTAL_ROOTS = 128
ATTEMPT_EVENTS = 256
ROOTS_PER_FAMILY = 32
MINIMUM_NICE = 10
MIN_FREE_GIB = 100.0
TARGET_FREE_GIB = 120.0
ORIGINAL_BYTE_LIMIT = 3 * 1024**3
CHARTER_SHA256 = (
    "a7630d4f37c7bde6164d3c3b5f7d9280c4371e268dda482e1b361da4f6197af0"
)
EXECUTE_COMMAND = (
    "zsh -ic 'no-secrets nice -n 10 env PYTHONPATH=. .venv/bin/python "
    "-m threes_rl.o2_yield_pilot_scan_recovery execute --out-dir "
    "threes_rl/runs/forensics/o2_yield_pilot_scan_recovery_v1'"
)


def canonical_json_hash(value: Any) -> str:
    return pilot.canonical_json_hash(value)


def sha256_path(path: Path) -> str:
    return pilot.sha256_path(path)


def write_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    pilot.preflight.write_immutable_json(path, value)


def payload_identity(
    path: Path,
    expected_payload_sha256: str,
) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    return {
        "path": str(path),
        "file_sha256": sha256_path(path),
        "canonical_payload_sha256": payload.get("canonical_payload_sha256"),
        "payload_valid": pilot.preflight.verify_payload_hash(payload),
        "payload_exact": payload.get("canonical_payload_sha256")
        == expected_payload_sha256,
    }


def immutable_original_audit() -> dict[str, Any]:
    files = []
    for path, expected in EXPECTED_ORIGINAL_HASHES.items():
        actual = sha256_path(path) if path.is_file() else None
        files.append(
            {
                "path": str(path),
                "expected_sha256": expected,
                "actual_sha256": actual,
                "passes": actual == expected,
            }
        )
    payloads = {
        str(path): payload_identity(path, expected)
        for path, expected in EXPECTED_PAYLOAD_HASHES.items()
    }
    marker = json.loads(ORIGINAL_MARKER.read_text())
    result = json.loads(ORIGINAL_RESULT.read_text())
    checks = {
        "all_original_files_exact": all(row["passes"] for row in files),
        "all_payloads_exact": all(
            row["payload_valid"] and row["payload_exact"]
            for row in payloads.values()
        ),
        "original_marker_opened": marker.get("decision")
        == "OPENED_O2_YIELD_PILOT",
        "original_terminal_hold": result.get("decision")
        == "HOLD_O2_PILOT_OPERATIONAL_INTEGRITY",
        "exact_original_error": result.get("error_type") == "KeyError"
        and result.get("error") == "'family_game_index'",
        "original_support_absent": not ORIGINAL_SUPPORT.exists(),
    }
    return {
        "files": files,
        "payloads": payloads,
        "checks": checks,
        "passes": all(checks.values()),
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def completion_adapter(row: Mapping[str, Any]) -> dict[str, Any]:
    if "game_index" not in row or "family_game_index" in row:
        raise ValueError("O2 recovery completion schema is not raw v1")
    adapted = dict(row)
    adapted["family_game_index"] = int(row["game_index"])
    if {
        key: value
        for key, value in adapted.items()
        if key != "family_game_index"
    } != dict(row):
        raise ValueError("O2 recovery adapter changed a non-index field")
    return adapted


def source_manifest_audit() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if sha256_path(ORIGINAL_COMPLETIONS) != EXPECTED_ORIGINAL_HASHES[
        ORIGINAL_COMPLETIONS
    ]:
        raise ValueError("O2 original completion file changed")
    rows = _load_jsonl(ORIGINAL_COMPLETIONS)
    marker = json.loads(ORIGINAL_MARKER.read_text())
    stream_rows = [dict(row) for row in marker["stream_rows"]]
    expected_by_key = {
        (str(row["family"]), int(row["family_game_index"])): row
        for row in stream_rows
    }
    manifest = []
    row_checks = []
    for row in rows:
        key = (str(row.get("family")), int(row.get("game_index", -1)))
        expected = expected_by_key.get(key)
        path = Path(str(row.get("source_replay")))
        actual_replay_sha = sha256_path(path) if path.is_file() else None
        checks = {
            "schema_exact": set(row) == COMPLETION_FIELDS,
            "stream_row_exists": expected is not None,
            "family_identity": expected is not None
            and row["nominal_family"] == row["family"]
            and int(row["family_index"]) == int(expected["family_index"]),
            "stream_identity": expected is not None
            and all(
                int(row[field]) == int(expected[field])
                for field in (
                    "logical_seed",
                    "deck_stream_id",
                    "slot_stream_id",
                    "policy_stream_id",
                )
            ),
            "complete": row.get("complete") is True,
            "dashboard_ineligible": row.get("dashboard_eligible") is False,
            "replay_in_original_namespace": path.parent.resolve()
            == ORIGINAL_REPLAY_DIR.resolve(),
            "replay_hash_exact": actual_replay_sha
            == row.get("source_replay_sha256"),
        }
        row_checks.append(all(checks.values()))
        manifest.append(
            {
                "family": row["family"],
                "family_index": int(row["family_index"]),
                "game_index": int(row["game_index"]),
                "logical_seed": int(row["logical_seed"]),
                "deck_stream_id": int(row["deck_stream_id"]),
                "slot_stream_id": int(row["slot_stream_id"]),
                "policy_stream_id": int(row["policy_stream_id"]),
                "root_cluster": row["root_cluster"],
                "source_replay": str(path),
                "source_replay_sha256": row["source_replay_sha256"],
                "source_replay_bytes": (
                    int(path.stat().st_size) if path.is_file() else None
                ),
                "checks": checks,
            }
        )
    manifest.sort(key=lambda row: (row["family_index"], row["game_index"]))
    families = Counter(row["family"] for row in rows)
    roots = [str(row["root_cluster"]) for row in rows]
    paths = [str(row["source_replay"]) for row in rows]
    replay_hashes = [str(row["source_replay_sha256"]) for row in rows]
    checks = {
        "exact_128_rows": len(rows) == TOTAL_ROOTS,
        "all_rows_exact": all(row_checks),
        "equal_32_per_family": families
        == {family: ROOTS_PER_FAMILY for family in pilot.FAMILY_ORDER},
        "unique_ancestries": len(set(roots)) == TOTAL_ROOTS,
        "unique_replay_paths": len(set(paths)) == TOTAL_ROOTS,
        "unique_replay_hashes": len(set(replay_hashes)) == TOTAL_ROOTS,
        "exact_stream_manifest": set(expected_by_key)
        == {(row["family"], row["game_index"]) for row in manifest},
    }
    audit = {
        "completion_rows": len(rows),
        "games_by_family": dict(sorted(families.items())),
        "unique_ancestries": len(set(roots)),
        "unique_replay_paths": len(set(paths)),
        "unique_replay_hashes": len(set(replay_hashes)),
        "source_manifest_sha256": canonical_json_hash(manifest),
        "checks": checks,
        "passes": all(checks.values()),
    }
    return rows, {"manifest": manifest, **audit}


def attempt_and_adapter_audit(
    completions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    attempts = _load_jsonl(ORIGINAL_ATTEMPTS)
    raw_keyerror = False
    raw_error = None
    try:
        pilot._stream_key(completions[0])
    except KeyError as error:
        raw_keyerror = error.args == ("family_game_index",)
        raw_error = repr(error)
    adapted = [completion_adapter(row) for row in completions]
    fixed = pilot.attempt_ledger_audit(
        json.loads(ORIGINAL_MARKER.read_text())["stream_rows"],
        adapted,
    )
    statuses = Counter(str(row.get("status")) for row in attempts)
    attempt_ids = [str(row.get("attempt_id")) for row in attempts]
    checks = {
        "attempt_file_exact": sha256_path(ORIGINAL_ATTEMPTS)
        == EXPECTED_ORIGINAL_HASHES[ORIGINAL_ATTEMPTS],
        "exact_256_events": len(attempts) == ATTEMPT_EVENTS,
        "exact_status_counts": statuses
        == {"opened": TOTAL_ROOTS, "completed": TOTAL_ROOTS},
        "exact_128_attempt_ids": len(set(attempt_ids)) == TOTAL_ROOTS,
        "raw_keyerror_reproduced": raw_keyerror,
        "adapter_only_adds_family_game_index": all(
            adapted_row["family_game_index"] == int(raw["game_index"])
            and {
                key: value
                for key, value in adapted_row.items()
                if key != "family_game_index"
            }
            == dict(raw)
            for raw, adapted_row in zip(completions, adapted, strict=True)
        ),
        "fixed_original_audit_passes": fixed["passes"],
        "zero_retries": fixed["retries"] == 0,
    }
    return {
        "attempt_events": len(attempts),
        "status_counts": dict(sorted(statuses.items())),
        "unique_attempt_ids": len(set(attempt_ids)),
        "raw_error": raw_error,
        "fixed_original_attempt_audit": fixed,
        "adapted_completion_manifest_sha256": canonical_json_hash(adapted),
        "checks": checks,
        "passes": all(checks.values()),
    }


def runtime_audit() -> dict[str, Any]:
    runtime = json.loads(ORIGINAL_RUNTIME.read_text())
    result = json.loads(ORIGINAL_RESULT.read_text())
    original_bytes = history._directory_bytes(ORIGINAL_DIR)
    checks = {
        "runtime_file_exact": sha256_path(ORIGINAL_RUNTIME)
        == EXPECTED_ORIGINAL_HASHES[ORIGINAL_RUNTIME],
        "exact_32_chunks": int(runtime["chunks_completed"]) == 32,
        "exact_128_games": int(runtime["games_completed"]) == TOTAL_ROOTS
        and int(runtime["games_evaluated_charged"]) == TOTAL_ROOTS,
        "runtime_below_six_hours": float(runtime["active_runtime_seconds"])
        < pilot.ACTIVE_RUNTIME_LIMIT,
        "original_storage_below_three_gib": original_bytes
        < ORIGINAL_BYTE_LIMIT,
        "zero_forbidden_work": all(
            int(value) == 0
            for value in result["zero_forbidden_work"].values()
        ),
        "support_absent": not ORIGINAL_SUPPORT.exists(),
    }
    return {
        "runtime": runtime,
        "original_output_bytes": original_bytes,
        "checks": checks,
        "passes": all(checks.values()),
    }


def operational_audit(*, out_dir: Path) -> dict[str, Any]:
    disk = shutil.disk_usage(ROOT)
    free_gib = disk.free / 1024**3
    service = history.service_health()
    heavy = pilot._heavy_process_audit()
    nice = os.getpriority(os.PRIO_PROCESS, 0)
    checks = {
        "nice_at_least_10": nice >= MINIMUM_NICE,
        "free_disk_hard": free_gib >= MIN_FREE_GIB,
        "free_disk_target": free_gib >= TARGET_FREE_GIB,
        "services_healthy": bool(service.get("passes")),
        "top_three_exact": tuple(service.get("dashboard_top_scores", [])[:3])
        == EXPECTED_TOP_THREE,
        "no_competing_heavy_process": bool(heavy.get("passes")),
        "recovery_output_small": (
            history._directory_bytes(out_dir) if out_dir.exists() else 0
        )
        < ORIGINAL_BYTE_LIMIT,
    }
    return {
        "nice": nice,
        "free_bytes": disk.free,
        "free_gib": free_gib,
        "service_health": service,
        "heavy_process_audit": heavy,
        "checks": checks,
        "passes": all(checks.values()),
    }


def seal_test_evidence(
    *,
    focused_passed: int,
    regression_passed: int,
    commands: Sequence[str],
) -> dict[str, Any]:
    if TEST_EVIDENCE_PATH.exists():
        raise FileExistsError("O2 recovery test evidence is immutable")
    payload = {
        "version": f"{VERSION}_test_evidence",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "charter_sha256": sha256_path(CHARTER_PATH),
        "runner_sha256": sha256_path(RUNNER_PATH),
        "tests_sha256": sha256_path(TEST_PATH),
        "focused_passed": int(focused_passed),
        "regression_passed": int(regression_passed),
        "commands": list(commands),
        "passes": focused_passed > 0 and regression_passed > 0,
        "zero_work": {
            "games": 0,
            "streams": 0,
            "replay_content_reads": 0,
            "support_scans": 0,
            "labels": 0,
            "model_fits": 0,
            "policy_outcomes": 0,
        },
    }
    write_immutable_json(TEST_EVIDENCE_PATH, payload)
    return json.loads(TEST_EVIDENCE_PATH.read_text())


def load_test_evidence() -> dict[str, Any]:
    payload = json.loads(TEST_EVIDENCE_PATH.read_text())
    checks = {
        "payload_valid": pilot.preflight.verify_payload_hash(payload),
        "charter": payload.get("charter_sha256") == sha256_path(CHARTER_PATH),
        "runner": payload.get("runner_sha256") == sha256_path(RUNNER_PATH),
        "tests": payload.get("tests_sha256") == sha256_path(TEST_PATH),
        "passes": payload.get("passes") is True,
    }
    if not all(checks.values()):
        raise ValueError(f"O2 recovery test evidence mismatch: {checks}")
    return payload


def current_bindings() -> dict[str, Any]:
    return {
        "charter_sha256": sha256_path(CHARTER_PATH),
        "runner_sha256": sha256_path(RUNNER_PATH),
        "tests_sha256": sha256_path(TEST_PATH),
        "test_evidence_file_sha256": sha256_path(TEST_EVIDENCE_PATH),
        "original_hashes": {
            str(path): sha256_path(path) for path in EXPECTED_ORIGINAL_HASHES
        },
    }


def open_recovery(*, out_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    if out_dir.resolve() != OUTPUT_DIR.resolve():
        raise ValueError("O2 recovery output directory mismatch")
    if out_dir.exists():
        raise FileExistsError("O2 recovery output already exists")
    immutable = immutable_original_audit()
    completions, sources = source_manifest_audit()
    attempts = attempt_and_adapter_audit(completions)
    runtime = runtime_audit()
    operations = operational_audit(out_dir=out_dir)
    tests = load_test_evidence()
    checks = {
        "charter_exact": sha256_path(CHARTER_PATH) == CHARTER_SHA256,
        "immutable_original": immutable["passes"],
        "sources": sources["passes"],
        "attempt_and_adapter": attempts["passes"],
        "runtime": runtime["passes"],
        "operations": operations["passes"],
        "tests": tests["passes"],
        "output_absent": not out_dir.exists(),
        "support_absent": not ORIGINAL_SUPPORT.exists(),
    }
    if not all(checks.values()):
        raise ValueError(f"O2 recovery open checks failed: {checks}")
    marker = {
        "version": VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "decision": "OPENED_O2_SCAN_RECOVERY",
        "bound_out_dir": str(out_dir.resolve()),
        "bound_execute_command": EXECUTE_COMMAND,
        "bindings": current_bindings(),
        "immutable_original_audit": immutable,
        "source_audit": sources,
        "attempt_and_adapter_audit": attempts,
        "runtime_audit": runtime,
        "operational_audit": operations,
        "checks": checks,
        "zero_content": {
            "games_generated": 0,
            "streams_consumed": 0,
            "replay_json_parsed": 0,
            "support_scans": 0,
            "score_action_max_tile_inspection": 0,
            "labels": 0,
            "model_fits": 0,
            "policy_outcomes": 0,
        },
    }
    out_dir.mkdir(parents=True)
    write_immutable_json(MARKER_PATH, marker)
    return json.loads(MARKER_PATH.read_text())


def load_marker(*, out_dir: Path) -> dict[str, Any]:
    if out_dir.resolve() != OUTPUT_DIR.resolve():
        raise ValueError("O2 recovery output directory mismatch")
    if RESULT_PATH.exists():
        raise FileExistsError("O2 recovery terminal result already exists")
    marker = json.loads(MARKER_PATH.read_text())
    if not pilot.preflight.verify_payload_hash(marker):
        raise ValueError("O2 recovery marker payload mismatch")
    completions, sources = source_manifest_audit()
    attempts = attempt_and_adapter_audit(completions)
    runtime = runtime_audit()
    checks = {
        "version": marker.get("version") == VERSION,
        "decision": marker.get("decision") == "OPENED_O2_SCAN_RECOVERY",
        "out_dir": marker.get("bound_out_dir") == str(out_dir.resolve()),
        "command": marker.get("bound_execute_command") == EXECUTE_COMMAND,
        "bindings": marker.get("bindings") == current_bindings(),
        "immutable": immutable_original_audit()["passes"],
        "sources": marker.get("source_audit") == sources,
        "attempts": marker.get("attempt_and_adapter_audit") == attempts,
        "runtime": marker.get("runtime_audit") == runtime,
        "original_support_absent": not ORIGINAL_SUPPORT.exists(),
    }
    if not all(checks.values()):
        raise ValueError(f"O2 recovery marker binding mismatch: {checks}")
    return marker


def _zero_forbidden_work() -> dict[str, int]:
    return {
        "new_games": 0,
        "streams_consumed": 0,
        "policy_evaluations": 0,
        "score_action_max_tile_inspection": 0,
        "corpus_collection": 0,
        "labels": 0,
        "model_fits": 0,
        "incumbent_changes": 0,
        "dashboard_changes": 0,
    }


def execute_recovery(*, out_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    if RESULT_PATH.exists():
        raise FileExistsError("O2 recovery terminal result already exists")
    marker = load_marker(out_dir=out_dir)
    try:
        completions, sources = source_manifest_audit()
        attempts = attempt_and_adapter_audit(completions)
        before = operational_audit(out_dir=out_dir)
        if not (sources["passes"] and attempts["passes"] and before["passes"]):
            raise ValueError("O2 recovery pre-scan integrity changed")
        support = pilot.support_analysis(completions)
        if support.get("decision") not in (
            "READY_O2_CORPUS_COLLECTION",
            "HOLD_O2_DATA_SUPPORT",
        ):
            raise ValueError("O2 recovery support decision is outside A4")
        write_immutable_json(SUPPORT_PATH, support)
        after = operational_audit(out_dir=out_dir)
        if not after["passes"]:
            raise ValueError("O2 recovery post-scan operations failed")
        result = {
            "version": VERSION,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "decision": support["decision"],
            "terminal_status": "HOLD_O2_AFTER_SCAN_RECOVERY_SEAL",
            "continue": (
                "CORPUS_COLLECTION_REQUIRES_RESEARCH_LEAD_REVIEW"
                if support["decision"] == "READY_O2_CORPUS_COLLECTION"
                else "NONE"
            ),
            "hold": [
                "corpus_collection",
                "option_rollouts",
                "labels",
                "model_fit",
                "policy_evaluation",
                "confirmation",
                "promotion",
            ],
            "kill": False,
            "promote": False,
            "original_decision_preserved": "HOLD_O2_PILOT_OPERATIONAL_INTEGRITY",
            "marker": pilot.preflight.artifact_identity(MARKER_PATH),
            "support": pilot.preflight.artifact_identity(SUPPORT_PATH),
            "source_audit": sources,
            "attempt_and_adapter_audit": attempts,
            "support_summary": {
                "candidate_rows": len(support["candidate_rows"]),
                "frames_scanned": support["scan_audit"]["frames_scanned"],
                "structural_passes": support["structural_layer"]["passes"],
                "availability_passes": support["availability_layer"]["passes"],
                "descriptive_1536_count": support["descriptive_1536"][
                    "selected_count"
                ],
            },
            "pre_scan_operations": before,
            "post_scan_operations": after,
            "zero_forbidden_work": _zero_forbidden_work(),
            "dashboard_eligible": False,
        }
        write_immutable_json(RESULT_PATH, result)
        return json.loads(RESULT_PATH.read_text())
    except Exception as error:
        if RESULT_PATH.exists():
            raise
        result = {
            "version": VERSION,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "decision": "HOLD_O2_SCAN_RECOVERY_INTEGRITY",
            "terminal_status": "HOLD_O2_AFTER_SCAN_RECOVERY_SEAL",
            "continue": "NONE",
            "hold": [
                "recovery_retry",
                "corpus_collection",
                "option_rollouts",
                "labels",
                "model_fit",
                "policy_evaluation",
                "confirmation",
                "promotion",
            ],
            "kill": False,
            "promote": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "original_decision_preserved": "HOLD_O2_PILOT_OPERATIONAL_INTEGRITY",
            "marker": pilot.preflight.artifact_identity(MARKER_PATH),
            "support_written": SUPPORT_PATH.exists(),
            "zero_forbidden_work": _zero_forbidden_work(),
            "dashboard_eligible": False,
        }
        write_immutable_json(RESULT_PATH, result)
        return json.loads(RESULT_PATH.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    evidence = subparsers.add_parser("seal-test-evidence")
    evidence.add_argument("--focused-passed", type=int, required=True)
    evidence.add_argument("--regression-passed", type=int, required=True)
    evidence.add_argument(
        "--test-command",
        action="append",
        dest="test_commands",
        required=True,
    )
    open_parser = subparsers.add_parser("open")
    open_parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    execute_parser = subparsers.add_parser("execute")
    execute_parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    if args.command == "seal-test-evidence":
        result = seal_test_evidence(
            focused_passed=args.focused_passed,
            regression_passed=args.regression_passed,
            commands=args.test_commands,
        )
    elif args.command == "open":
        result = open_recovery(out_dir=args.out_dir)
    else:
        result = execute_recovery(out_dir=args.out_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
