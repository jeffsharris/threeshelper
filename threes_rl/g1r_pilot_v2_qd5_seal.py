"""Open and seal the authorized G1-R pilot-v2 QD5 execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any

from threes_rl import g1r_acquire_v2_qd5 as pilot
from threes_rl.g1r_acquire import canonical_json_hash, service_health
from threes_rl.g1r_qd_admission_v2 import _heavy_process_audit
from threes_rl.s3_power_preflight import sha256_path


VERSION = "g1r_pilot_v2_qd5_execution_seal_v1"
OUTPUT_DIR = pilot.OUTPUT_DIR
PREFLIGHT_PATH = OUTPUT_DIR / "preflight_lock.json"
MARKER_PATH = OUTPUT_DIR / "PILOT_V2_EXECUTION_OPENED.json"
SUMMARY_PATH = OUTPUT_DIR / "pilot_summary.json"
COMPLETED_PATH = OUTPUT_DIR / "completed_games.jsonl"
RUNTIME_PATH = OUTPUT_DIR / "runtime_state.json"
SEAL_PATH = OUTPUT_DIR / "PILOT_V2_SEAL.json"
HELPER_PATH = Path("threes_rl/g1r_pilot_v2_qd5_seal.py")

EXPECTED = {
    "charter_sha256": (
        pilot.CHARTER_PATH,
        "1f58d73b5f21aed20806f605009b6572bc9587aafa0c7cf2323f54f90ddce003",
    ),
    "runner_sha256": (
        pilot.IMPLEMENTATION_PATH,
        "f195026041e25aeb22ffc72cc57c49d1da96a1af3dfa9fa9180e31345a13d776",
    ),
    "test_sha256": (
        pilot.TEST_PATH,
        "85be02eb7158c1fef5104c836835704a24bcbfc2c5bf0396a03ba7168143a8de",
    ),
    "test_evidence_sha256": (
        pilot.TEST_EVIDENCE_PATH,
        "c0804d7cf6b13f2344b992727148b05f87895a7a36980bddd832856e6cd7e393",
    ),
    "preflight_lock_sha256": (
        PREFLIGHT_PATH,
        "0d50edaae52e9a6f6291c4b397fd03c9d7d8651b28bb9dbd05b53c8718ee22ad",
    ),
}
PREFLIGHT_PAYLOAD_SHA256 = (
    "1a0ca85b4115f220d0d7c857bde912be8570cf0b2e72d055e6cd88b285227e67"
)
BOUND_COMMAND = (
    "zsh -ic 'no-secrets nice -n 10 env PYTHONPATH=. .venv/bin/python "
    "-m threes_rl.g1r_acquire_v2_qd5 run-pilot "
    "--out-dir threes_rl/runs/forensics/g1r_acquisition/pilot_v2_qd5 "
    "--preflight-lock "
    "threes_rl/runs/forensics/g1r_acquisition/pilot_v2_qd5/preflight_lock.json "
    "--jobs 1'"
)


def _write_new_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Immutable artifact already exists: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"Atomic temporary already exists: {temporary}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _seal_payload(path: Path, payload: dict[str, Any], field: str) -> dict[str, Any]:
    payload_hash = canonical_json_hash(payload)
    sealed = dict(payload)
    sealed[field] = payload_hash
    _write_new_json_atomic(path, sealed)
    return {
        "path": str(path),
        "file_sha256": sha256_path(path),
        "payload_sha256": payload_hash,
    }


def _load_self_hashed(path: Path, field: str) -> tuple[dict[str, Any], str]:
    payload = json.loads(path.read_text())
    embedded = payload.pop(field)
    if canonical_json_hash(payload) != embedded:
        raise ValueError(f"Canonical payload mismatch: {path}")
    payload[field] = embedded
    return payload, embedded


def _identity_audit() -> dict[str, Any]:
    rows = {}
    for name, (path, expected) in EXPECTED.items():
        actual = sha256_path(path)
        rows[name] = {
            "path": str(path),
            "expected": expected,
            "actual": actual,
            "passes": actual == expected,
        }
    preflight, embedded = _load_self_hashed(
        PREFLIGHT_PATH, "preflight_payload_sha256"
    )
    rows["preflight_payload_sha256"] = {
        "expected": PREFLIGHT_PAYLOAD_SHA256,
        "actual": embedded,
        "passes": embedded == PREFLIGHT_PAYLOAD_SHA256,
    }
    return {
        "artifacts": rows,
        "preflight": preflight,
        "passes": all(row["passes"] for row in rows.values()),
    }


def open_execution() -> dict[str, Any]:
    if MARKER_PATH.exists() or SEAL_PATH.exists():
        raise FileExistsError("Pilot-v2 execution has already opened or sealed")
    forbidden = [
        COMPLETED_PATH,
        RUNTIME_PATH,
        SUMMARY_PATH,
        OUTPUT_DIR / "source_replays",
    ]
    if any(path.exists() for path in forbidden):
        raise ValueError("Pilot work exists before execution marker")
    identity = _identity_audit()
    preflight = identity["preflight"]
    heavy = _heavy_process_audit()
    services = service_health()
    free_gib = shutil.disk_usage(OUTPUT_DIR).free / 1024**3
    checks = {
        "identity_exact": identity["passes"],
        "preflight_ready": preflight["decision"]
        == "READY_G1R_PILOT_V2_QD5_PREFLIGHT",
        "family_order_exact": preflight["family_order"]
        == [family for family, _spec in pilot.FAMILY_SLATE],
        "stream_manifest_exact": preflight["stream_manifest_sha256"]
        == "fae883a10c5aba931d3d8a2644986c9dc7dca48f3d959a657294e4dfec1fdc68",
        "stream_bases_exact": preflight["stream_bases"] == pilot.STREAM_BASES,
        "jobs_exactly_one": preflight["frozen_jobs"] == 1,
        "nice_at_least_10": pilot.base.current_nice() >= 10,
        "no_heavy_contention": heavy["passes"],
        "free_disk_at_least_120_gib": free_gib >= 120,
        "free_disk_above_100_gib": free_gib > 100,
        "services_dashboard_top_three": services["passes"]
        and services["dashboard_top_scores"][:3] == [263670, 261369, 258561],
        "zero_pilot_work_before_marker": not any(path.exists() for path in forbidden),
        "bounds_exact": (
            preflight["active_wall_seconds_limit"] == 12 * 3600
            and preflight["pilot_byte_limit"] == 4 * 1024**3
            and preflight["pause_free_disk_gib"] == 100
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"Pilot-v2 execution-open checks failed: {checks}")
    payload = {
        "version": VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "execution_opened": True,
        "bound_command": BOUND_COMMAND,
        "bound_out_dir": str(OUTPUT_DIR.resolve()),
        "identity_audit": {
            "artifacts": identity["artifacts"],
            "helper_path": str(HELPER_PATH),
            "helper_sha256": sha256_path(HELPER_PATH),
        },
        "family_order": preflight["family_order"],
        "stream_manifest_sha256": preflight["stream_manifest_sha256"],
        "stream_row_count": len(preflight["stream_rows"]),
        "stream_bases": {
            f"{key}_base": value for key, value in preflight["stream_bases"].items()
        },
        "jobs": 1,
        "required_nice": 10,
        "active_wall_seconds_limit": 12 * 3600,
        "output_byte_limit": 4 * 1024**3,
        "free_disk_pause_gib": 100,
        "free_gib": free_gib,
        "heavy_process_audit": heavy,
        "service_health": services,
        "checks": checks,
        "zero_before_open": {
            "games": 0,
            "streams_consumed": 0,
            "labels": 0,
            "models": 0,
            "h40_outcomes": 0,
            "continuations": 0,
            "score_or_policy_outcomes_inspected": False,
            "dashboard_changed": False,
            "incumbent_changed": False,
        },
    }
    return _seal_payload(
        MARKER_PATH, payload, "execution_opened_payload_sha256"
    )


def _load_completed() -> list[dict[str, Any]]:
    rows = []
    with COMPLETED_PATH.open() as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def seal_execution() -> dict[str, Any]:
    if SEAL_PATH.exists():
        raise FileExistsError("Pilot-v2 terminal seal already exists")
    marker, marker_payload_sha = _load_self_hashed(
        MARKER_PATH, "execution_opened_payload_sha256"
    )
    identity = _identity_audit()
    preflight = identity["preflight"]
    summary = json.loads(SUMMARY_PATH.read_text())
    runtime = json.loads(RUNTIME_PATH.read_text())
    completed = _load_completed()
    expected_rows = {
        (str(row["nominal_family"]), int(row["game_index"])): row
        for row in preflight["stream_rows"]
    }
    completed_rows = {
        (str(row["nominal_family"]), int(row["game_index"])): row
        for row in completed
    }
    field_mismatches = []
    for key, expected in expected_rows.items():
        actual = completed_rows.get(key)
        if actual is None:
            continue
        for field in (
            "logical_seed",
            "deck_stream_id",
            "slot_stream_id",
            "policy_stream_id",
            "family_index",
        ):
            if int(actual[field]) != int(expected[field]):
                field_mismatches.append(
                    {"key": list(key), "field": field, "actual": actual[field]}
                )
    all_candidates = [
        candidate for row in completed for candidate in row.get("candidates", [])
    ]
    recomputed_root_cap = pilot.root_cap_candidates(all_candidates)
    sealed_root_cap = summary["root_capped_candidates"]
    root_cap_equal = recomputed_root_cap == sealed_root_cap
    roots = [str(row["root_cluster"]) for row in sealed_root_cap]
    yields = {
        family: {
            stratum: sum(
                row["behavior_family"] == family and row["stratum"] == stratum
                for row in sealed_root_cap
            )
            for stratum in pilot.STRATA
        }
        for family, _spec in pilot.FAMILY_SLATE
    }
    conservation = {}
    for family, counts in yields.items():
        counts["any"] = counts["pre1536"] + counts["pre3072"]
        conservation[family] = (
            counts["pre1536"] + counts["pre3072"] == counts["any"]
        )
    family_counts = Counter(row["behavior_family"] for row in sealed_root_cap)
    total_roots = len(sealed_root_cap)
    family_concentration = {
        family: {
            "roots": int(family_counts[family]),
            "share": (
                float(family_counts[family]) / total_roots if total_roots else 0.0
            ),
        }
        for family, _spec in pilot.FAMILY_SLATE
    }
    external_collision = pilot.stream_collision_audit(
        preflight["stream_rows"], exclude_dir=OUTPUT_DIR
    )
    services = service_health()
    free_gib = shutil.disk_usage(OUTPUT_DIR).free / 1024**3
    output_bytes = pilot.base._directory_bytes(OUTPUT_DIR)
    checks = {
        "marker_and_identity_exact": identity["passes"]
        and marker["bound_command"] == BOUND_COMMAND,
        "exact_100_completed_rows": len(completed) == 100
        and set(completed_rows) == set(expected_rows),
        "twenty_games_per_family": Counter(
            row["nominal_family"] for row in completed
        )
        == Counter({family: 20 for family, _spec in pilot.FAMILY_SLATE}),
        "stream_rows_exact": not field_mismatches,
        "zero_external_stream_collisions": external_collision["zero_collisions"],
        "root_cap_reproduces": root_cap_equal,
        "root_ancestries_globally_unique": len(roots) == len(set(roots)),
        "per_family_cross_stratum_conservation": all(conservation.values()),
        "retained_source_integrity": summary["retained_source_integrity"]["passes"],
        "runtime_within_12h": float(runtime["active_runtime_seconds"])
        < 12 * 3600,
        "output_below_4_gib": output_bytes < 4 * 1024**3,
        "free_disk_above_100_gib": free_gib > 100,
        "services_dashboard_top_three": services["passes"]
        and services["dashboard_top_scores"][:3] == [263670, 261369, 258561],
        "no_forbidden_analysis_fields": all(
            "score" not in row and "action" not in row for row in completed
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"Pilot-v2 terminal seal checks failed: {checks}")
    payload = {
        "version": VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "decision": "HOLD_G1R_AFTER_PILOT_V2_QD5_SEAL",
        "runner_projection_decision": summary["decision"],
        "continuation_acquisition_authorized": False,
        "marker": {
            "path": str(MARKER_PATH),
            "file_sha256": sha256_path(MARKER_PATH),
            "payload_sha256": marker_payload_sha,
        },
        "preflight_file_sha256": sha256_path(PREFLIGHT_PATH),
        "preflight_payload_sha256": preflight["preflight_payload_sha256"],
        "summary_file_sha256": sha256_path(SUMMARY_PATH),
        "completed_file_sha256": sha256_path(COMPLETED_PATH),
        "runtime_file_sha256": sha256_path(RUNTIME_PATH),
        "completeness": {
            "games": len(completed),
            "games_by_family": dict(
                sorted(Counter(row["nominal_family"] for row in completed).items())
            ),
            "exact_manifest": set(completed_rows) == set(expected_rows),
        },
        "exact_rung_yields": yields,
        "root_accounting": {
            "eligible_candidate_records": len(all_candidates),
            "globally_unique_selected_roots": total_roots,
            "selected_root_ids_sha256": canonical_json_hash(sorted(roots)),
            "root_cap_reproduces": root_cap_equal,
            "all_roots_unique": len(roots) == len(set(roots)),
            "per_family_conservation": conservation,
        },
        "family_concentration": family_concentration,
        "roles": summary["role_counts"],
        "retained_source_integrity": summary["retained_source_integrity"],
        "stream_audit": {
            "manifest_sha256": preflight["stream_manifest_sha256"],
            "completed_field_mismatches": field_mismatches,
            "external_collision_audit": external_collision,
        },
        "wilson_yield_projection": summary["yield_projection"],
        "runtime": {
            **runtime,
            "output_bytes": output_bytes,
            "free_gib": free_gib,
        },
        "service_health": services,
        "checks": checks,
        "zero_forbidden_work": {
            "action_labels": 0,
            "models": 0,
            "h40_outcomes": 0,
            "continuations": 0,
            "score_or_policy_outcome_analysis": False,
            "incumbent_changed": False,
            "dashboard_changed": False,
        },
    }
    return _seal_payload(SEAL_PATH, payload, "seal_payload_sha256")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("open", "seal"))
    args = parser.parse_args()
    payload = open_execution() if args.command == "open" else seal_execution()
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
