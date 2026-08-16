"""One-shot execution marker and terminal seal for G2 transfer acquisition."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any

from threes_rl import g2_fresh_transfer_acquire as acquisition
from threes_rl.s3_power_preflight import sha256_path


VERSION = "g2_transfer_acquisition_guard_v1"
GUARD_PATH = Path("threes_rl/g2_transfer_acquisition_guard.py")
OUTPUT_DIR = acquisition.OUTPUT_DIR
PREFLIGHT_PATH = OUTPUT_DIR / "preflight_lock.json"
MARKER_PATH = OUTPUT_DIR / "G2_TRANSFER_ACQUISITION_OPENED.json"
RESULT_PATH = OUTPUT_DIR / "G2_TRANSFER_ACQUISITION_RESULT.json"
RUNNER_SHA256 = "66ce0dea164a2c34fe8cbf5e92d35e8797116ed83c7ec15500c03b02c7f87c23"
PREFLIGHT_FILE_SHA256 = (
    "5250e54d081a48525f4ef3a776ac2836259873a45a33f395fbbd0281b545cf44"
)
PREFLIGHT_PAYLOAD_SHA256 = (
    "18d9e851dfe923cfa6efcd68d28bd3faeadd8c8ddb9fcce5615157c411f05993"
)
STREAM_MANIFEST_SHA256 = (
    "8c5aefd39379f4c2ad27c6e11926267673501ca8905346332cdeea98cc89c047"
)
HISTORICAL_UNION_SHA256 = (
    "eead7a4e03528539483d286351bda5fa1496d5a685bb81b5c85f5d70b28cc756"
)
TERMINAL_STATUS = "HOLD_G2_AFTER_FRESH_TRANSFER_ACQUISITION_SEAL"
EXACT_COMMAND = (
    "zsh -ic 'no-secrets nice -n 10 env PYTHONPATH=. .venv/bin/python -m "
    "threes_rl.g2_transfer_acquisition_guard run "
    "--out-dir threes_rl/runs/forensics/g2_fresh_transfer_acquisition_v1 "
    "--preflight-lock "
    "threes_rl/runs/forensics/g2_fresh_transfer_acquisition_v1/"
    "preflight_lock.json --jobs 1'"
)


def canonical_json_hash(value: Any) -> str:
    return acquisition.canonical_json_hash(value)


def _write_new_json(path: Path, payload: dict[str, Any]) -> None:
    acquisition._write_new_json_atomic(path, payload)


def _load_preflight() -> dict[str, Any]:
    if sha256_path(PREFLIGHT_PATH) != PREFLIGHT_FILE_SHA256:
        raise ValueError("Preflight file hash changed")
    lock = json.loads(PREFLIGHT_PATH.read_text())
    embedded = lock.pop("preflight_payload_sha256")
    computed = canonical_json_hash(lock)
    lock["preflight_payload_sha256"] = embedded
    if embedded != PREFLIGHT_PAYLOAD_SHA256 or computed != embedded:
        raise ValueError("Preflight canonical payload changed")
    if lock["decision"] != "READY_G2_FRESH_TRANSFER_ACQUISITION":
        raise ValueError("Preflight is not READY")
    return lock


def _historical_source_hash(lock: dict[str, Any]) -> str:
    sources = lock["historical_stream_collision_audit"]["historical_union"][
        "matched_sources"
    ]
    return canonical_json_hash(sources)


def _work_artifact_paths(out_dir: Path) -> list[Path]:
    names = (
        "completion_rows.jsonl",
        "runtime_state.json",
        "acquisition_summary.json",
        RESULT_PATH.name,
    )
    paths = [out_dir / name for name in names]
    paths.append(out_dir / "qualifying_sources")
    return paths


def create_open_marker(
    *, out_dir: Path = OUTPUT_DIR, preflight_lock: Path = PREFLIGHT_PATH, jobs: int = 1
) -> dict[str, Any]:
    if out_dir.resolve() != OUTPUT_DIR.resolve():
        raise ValueError("Output path differs from frozen path")
    if preflight_lock.resolve() != PREFLIGHT_PATH.resolve():
        raise ValueError("Preflight path differs from frozen path")
    if jobs != 1:
        raise ValueError("jobs must equal one")
    if MARKER_PATH.exists() or RESULT_PATH.exists():
        raise FileExistsError("Execution was already opened or sealed")
    lock = _load_preflight()
    policy_lock, _policies = acquisition.load_and_lock_policies()
    collision = acquisition.stream_collision_audit(
        lock["stream_rows"], exclude_dir=out_dir
    )
    heavy = acquisition._heavy_process_audit()
    services = acquisition.base.service_health()
    free_gib = shutil.disk_usage(out_dir).free / 1024**3
    work_absent = {
        str(path): not path.exists() for path in _work_artifact_paths(out_dir)
    }
    checks = {
        "charter_exact": sha256_path(acquisition.CHARTER_PATH)
        == acquisition.CHARTER_SHA256,
        "runner_exact": sha256_path(acquisition.IMPLEMENTATION_PATH)
        == RUNNER_SHA256,
        "preflight_exact": lock["preflight_payload_sha256"]
        == PREFLIGHT_PAYLOAD_SHA256,
        "stream_manifest_exact": lock["stream_manifest_sha256"]
        == STREAM_MANIFEST_SHA256,
        "historical_union_exact": _historical_source_hash(lock)
        == HISTORICAL_UNION_SHA256,
        "policy_lock_exact": policy_lock["policy_lock_sha256"]
        == lock["policy_lock"]["policy_lock_sha256"],
        "signatures_exact": lock["action_signature_audit"]["passes"],
        "quota_cap_exact": lock["quota_per_family"] == 32
        and lock["game_cap_per_family"] == 640,
        "round_robin_exact": lock["maximum_chunk_size"] == 6
        and lock["frozen_jobs"] == 1,
        "zero_current_collisions": collision["zero_collisions"],
        "nice_at_least_10": acquisition.base.current_nice() >= 10,
        "no_heavy_contention": heavy["passes"],
        "free_disk_above_120_gib": free_gib >= 120,
        "services_dashboard_top_three": services["passes"]
        and services["dashboard_top_scores"][:3] == [263670, 261369, 258561],
        "zero_work_before_marker": all(work_absent.values()),
        "no_active_human_session_read": True,
    }
    if not all(checks.values()):
        raise ValueError(f"Execution-open checks failed: {checks}")
    marker = {
        "version": VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "acquisition_opened": True,
        "terminal_status_after_seal": TERMINAL_STATUS,
        "bound_command": EXACT_COMMAND,
        "bound_out_dir": str(out_dir.resolve()),
        "bound_preflight": str(preflight_lock.resolve()),
        "frozen_jobs": jobs,
        "required_nice": 10,
        "identities": {
            "charter_sha256": acquisition.CHARTER_SHA256,
            "runner_sha256": RUNNER_SHA256,
            "guard_sha256": sha256_path(GUARD_PATH),
            "preflight_file_sha256": PREFLIGHT_FILE_SHA256,
            "preflight_payload_sha256": PREFLIGHT_PAYLOAD_SHA256,
            "stream_manifest_sha256": STREAM_MANIFEST_SHA256,
            "historical_union_sha256": HISTORICAL_UNION_SHA256,
            "policy_lock_sha256": policy_lock["policy_lock_sha256"],
            "signature_audit_sha256": lock["action_signature_audit"][
                "audit_sha256"
            ],
        },
        "family_order": lock["family_order"],
        "policy_lock": lock["policy_lock"],
        "action_signatures": lock["action_signature_audit"],
        "quota_per_family": 32,
        "game_cap_per_family": 640,
        "maximum_chunk_size": 6,
        "stream_bases": lock["stream_bases"],
        "bounds": {
            "active_wall_seconds": 12 * 3600,
            "bytes": 4 * 1024**3,
            "hard_minimum_free_gib": 100,
            "target_free_gib": 120,
        },
        "preopen_collision_audit": {
            "matched_source_count": collision["historical_union"][
                "matched_source_count"
            ],
            "historical_union_sha256": _historical_source_hash(lock),
            "zero_collisions": collision["zero_collisions"],
        },
        "process": heavy,
        "nice": acquisition.base.current_nice(),
        "free_gib": free_gib,
        "services": services,
        "work_artifact_absence": work_absent,
        "checks": checks,
        "zero_before_open": {
            "games": 0,
            "streams_consumed": 0,
            "labels": 0,
            "models": 0,
            "outcomes": 0,
            "continuations": 0,
            "dashboard_changes": 0,
        },
    }
    marker["opened_payload_sha256"] = canonical_json_hash(marker)
    _write_new_json(MARKER_PATH, marker)
    return marker


def _load_marker() -> dict[str, Any]:
    marker = json.loads(MARKER_PATH.read_text())
    embedded = marker.pop("opened_payload_sha256")
    computed = canonical_json_hash(marker)
    marker["opened_payload_sha256"] = embedded
    if embedded != computed:
        raise ValueError("Open marker payload mismatch")
    if marker["identities"]["guard_sha256"] != sha256_path(GUARD_PATH):
        raise ValueError("Guard changed after execution opened")
    if marker["identities"]["runner_sha256"] != sha256_path(
        acquisition.IMPLEMENTATION_PATH
    ):
        raise ValueError("Runner changed after execution opened")
    if marker["bound_command"] != EXACT_COMMAND:
        raise ValueError("Bound command changed")
    return marker


def _load_completion_rows(out_dir: Path) -> list[dict[str, Any]]:
    path = out_dir / "completion_rows.jsonl"
    if not path.is_file():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    forbidden = ("score", "action")
    for row in rows:
        flattened = json.dumps(row, sort_keys=True)
        if any(token in flattened for token in forbidden):
            raise ValueError("Compact completion row contains forbidden outcome data")
    return rows


def _prior_root_seeds() -> set[int]:
    root_manifest = json.loads(acquisition.G2_ROOT_MANIFEST.read_text())
    pattern = re.compile(r":fresh:(\d+):1536$")
    seeds = set()
    for row in root_manifest["records"]:
        match = pattern.search(str(row["root_cluster"]))
        if match:
            seeds.add(int(match.group(1)))
    return seeds


def _retained_source_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    retained = [row for row in rows if row["retained"]]
    failures = []
    roots = []
    source_rows = []
    prior_seeds = _prior_root_seeds()
    overlap = []
    for row in retained:
        replay_path = Path(row["source_replay"])
        state_path = Path(row["source_state"])
        candidate = row["candidate"]
        try:
            replay_sha = sha256_path(replay_path)
            state_sha = sha256_path(state_path)
            state_record = json.loads(state_path.read_text())
            replay = json.loads(replay_path.read_text())
            extracted = acquisition.extract_first_transfer_state(
                replay,
                family=str(row["family"]),
                expected_seed=int(row["logical_seed"]),
            )
            if extracted is None:
                raise ValueError("retained replay no longer qualifies")
            if extracted["state_sha1"] != candidate["state_sha1"]:
                raise ValueError("state hash mismatch")
            if replay_sha != candidate["source_replay_sha256"]:
                raise ValueError("replay hash mismatch")
            if state_sha != candidate["source_state_sha256"]:
                raise ValueError("state artifact hash mismatch")
            if state_record["state_sha1"] != candidate["state_sha1"]:
                raise ValueError("state record mismatch")
            root = str(candidate["root_cluster"])
            roots.append(root)
            if int(row["logical_seed"]) in prior_seeds:
                overlap.append(root)
            source_rows.append(
                {
                    "family": row["family"],
                    "game_index": row["game_index"],
                    "logical_seed": row["logical_seed"],
                    "root_cluster": root,
                    "source_replay": str(replay_path),
                    "source_replay_sha256": replay_sha,
                    "source_state": str(state_path),
                    "source_state_sha256": state_sha,
                    "state_sha1": candidate["state_sha1"],
                    "source_frame_index": candidate["source_frame_index"],
                }
            )
        except Exception as error:
            failures.append(
                {
                    "family": row.get("family"),
                    "game_index": row.get("game_index"),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    checks = {
        "retained_count_96": len(retained) == 96,
        "roots_unique": len(roots) == len(set(roots)),
        "sources_exact": not failures,
        "prior_root_seed_overlap_zero": not overlap,
    }
    return {
        "retained_count": len(retained),
        "unique_roots": len(set(roots)),
        "protected_overlap": overlap,
        "failures": failures,
        "sources": source_rows,
        "source_manifest_sha256": canonical_json_hash(source_rows),
        "checks": checks,
        "passes": all(checks.values()),
    }


def seal_result(
    *,
    out_dir: Path,
    runner_summary: dict[str, Any] | None,
    error: Exception | None,
) -> dict[str, Any]:
    if RESULT_PATH.exists():
        raise FileExistsError("Terminal result already exists")
    marker = _load_marker()
    lock = _load_preflight()
    rows = _load_completion_rows(out_dir)
    by_family = Counter(str(row["family"]) for row in rows)
    retained_by_family = Counter(
        str(row["family"]) for row in rows if row["retained"]
    )
    completed_by_family = Counter(
        str(row["family"]) for row in rows if row["completed"]
    )
    source_audit = _retained_source_audit(rows)
    collision = acquisition.stream_collision_audit(
        lock["stream_rows"], exclude_dir=out_dir
    )
    services = acquisition.base.service_health()
    heavy = acquisition._heavy_process_audit()
    free_gib = shutil.disk_usage(out_dir).free / 1024**3
    output_bytes = acquisition._directory_bytes(out_dir)
    families = lock["family_order"]
    checks = {
        "runner_returned_ready": error is None
        and runner_summary is not None
        and runner_summary["decision"] == "READY_G2_TRANSFER_ROOTS",
        "all_rows_completed": all(row["completed"] for row in rows),
        "quota_32_each": all(retained_by_family[family] == 32 for family in families),
        "family_caps_respected": all(by_family[family] <= 640 for family in families),
        "retained_source_integrity": source_audit["passes"],
        "stream_collision_free": collision["zero_collisions"],
        "runtime_below_12h": runner_summary is not None
        and runner_summary["runtime"]["active_seconds"] < 12 * 3600,
        "storage_below_4gib": output_bytes < 4 * 1024**3,
        "disk_above_100gib": free_gib >= 100,
        "services_dashboard_top_three": services["passes"]
        and services["dashboard_top_scores"][:3] == [263670, 261369, 258561],
        "no_heavy_contention": heavy["passes"],
    }
    decision = (
        "READY_G2_FRESH_TRANSFER_ROOTS"
        if all(checks.values())
        else "HOLD_G2_FRESH_TRANSFER_ACQUISITION"
    )
    result = {
        "version": VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "decision": decision,
        "terminal_status": TERMINAL_STATUS,
        "marker_file_sha256": sha256_path(MARKER_PATH),
        "marker_payload_sha256": marker["opened_payload_sha256"],
        "preflight_file_sha256": PREFLIGHT_FILE_SHA256,
        "preflight_payload_sha256": PREFLIGHT_PAYLOAD_SHA256,
        "bound_command": EXACT_COMMAND,
        "attempted_games": len(rows),
        "attempted_by_family": dict(by_family),
        "completed_by_family": dict(completed_by_family),
        "retained_by_family": dict(retained_by_family),
        "family_concentration": {
            family: retained_by_family[family] / max(1, len(source_audit["sources"]))
            for family in families
        },
        "source_audit": source_audit,
        "stream_integrity": {
            "manifest_sha256": STREAM_MANIFEST_SHA256,
            "historical_union_sha256": HISTORICAL_UNION_SHA256,
            "zero_collisions": collision["zero_collisions"],
            "matched_source_count": collision["historical_union"][
                "matched_source_count"
            ],
        },
        "runtime": None if runner_summary is None else runner_summary["runtime"],
        "output_bytes": output_bytes,
        "free_gib": free_gib,
        "services": services,
        "process": heavy,
        "checks": checks,
        "error": (
            None
            if error is None
            else {
                "type": type(error).__name__,
                "message": str(error),
            }
        ),
        "zero_downstream_work": {
            "labels": 0,
            "models": 0,
            "h10_h20_h40_outcomes": 0,
            "policy_outcomes": 0,
            "continuations": 0,
            "score_or_action_inspection": False,
            "incumbent_changed": False,
            "dashboard_changed": False,
        },
    }
    result["result_payload_sha256"] = canonical_json_hash(result)
    _write_new_json(RESULT_PATH, result)
    return result


def run_once(
    *, out_dir: Path = OUTPUT_DIR, preflight_lock: Path = PREFLIGHT_PATH, jobs: int = 1
) -> dict[str, Any]:
    _load_marker()
    if out_dir.resolve() != OUTPUT_DIR.resolve():
        raise ValueError("Output path differs from marker")
    if preflight_lock.resolve() != PREFLIGHT_PATH.resolve() or jobs != 1:
        raise ValueError("Invocation differs from marker")
    try:
        summary = acquisition.run_acquisition(
            out_dir=out_dir,
            preflight_lock=preflight_lock,
            jobs=jobs,
        )
    except Exception as error:
        return seal_result(out_dir=out_dir, runner_summary=None, error=error)
    return seal_result(out_dir=out_dir, runner_summary=summary, error=None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("open", "run"):
        command = commands.add_parser(name)
        command.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
        command.add_argument("--preflight-lock", type=Path, default=PREFLIGHT_PATH)
        command.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    if args.command == "open":
        payload = create_open_marker(
            out_dir=args.out_dir,
            preflight_lock=args.preflight_lock,
            jobs=args.jobs,
        )
    else:
        payload = run_once(
            out_dir=args.out_dir,
            preflight_lock=args.preflight_lock,
            jobs=args.jobs,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
