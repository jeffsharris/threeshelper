"""Outcome-free supplemental storage audit for the sealed QD-v2 admission."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from threes_rl.g1r_acquire import canonical_json_hash, service_health


WORKSPACE = Path(__file__).resolve().parents[1]
RUNS_DIR = WORKSPACE / "threes_rl/runs"
V1_DIR = RUNS_DIR / "forensics/g1r_qd_admission_v1"
V2_DIR = RUNS_DIR / "forensics/g1r_qd_admission_v2_terminal_schema"
CHARTER_PATH = WORKSPACE / "threes_rl/G1R_QD_V2_STORAGE_AUDIT_CHARTER.md"
INVENTORY_PATH = V2_DIR / "QD_V2_STORAGE_REPLAY_INVENTORY.json"
AUDIT_PATH = V2_DIR / "QD_V2_STORAGE_ADMISSION_AUDIT.json"

CHARTER_SHA256 = (
    "dd51e2745320734f874729529d2a230ab6e381f4a1c8c6b5d586adc88ed03070"
)
CUTOFF_TEXT = "2026-07-25T10:26:56-0700"
CUTOFF = datetime.strptime(CUTOFF_TEXT, "%Y-%m-%dT%H:%M:%S%z")
CUTOFF_NS = int(CUTOFF.timestamp()) * 1_000_000_000

EXPECTED_SEALED_HASHES = {
    "v2_admission_opened": {
        "path": V2_DIR / "ADMISSION_OPENED.json",
        "sha256": (
            "11b21137303fa4cfd258dfe3ff536b227c24fa4cb7db727ca376b970418c5135"
        ),
    },
    "v2_admission_result": {
        "path": V2_DIR / "admission_result.json",
        "sha256": (
            "27bcb3328a02d6dc5094dcc5a8e52b8f27d2f3e4ea7b92f5c1a8153bc1326a8e"
        ),
    },
    "v1_admission_opened": {
        "path": V1_DIR / "ADMISSION_OPENED.json",
        "sha256": (
            "f1faadcf2152b28b0254f36402de4568be4eb056c7dd56c52bbcd51c17d51f6e"
        ),
    },
    "v1_admission_hold": {
        "path": V1_DIR / "HOLD_QD_ADMISSION_ERROR.json",
        "sha256": (
            "205229ce77a34b68ff3fdc31ee0bb83bc917671ae3f9efdb5f5eeb91c0b7068b"
        ),
    },
}

MIB = 1_048_576
GIB = 1_073_741_824
FOUR_GIB = 4 * GIB


def _relative(path: Path) -> str:
    return path.relative_to(WORKSPACE).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file_stat(path: Path) -> os.stat_result:
    result = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(result.st_mode):
        raise ValueError(f"Expected regular file: {path}")
    return result


def _metadata_stable_hash(path: Path) -> tuple[os.stat_result, str]:
    before = _regular_file_stat(path)
    digest = _sha256(path)
    after = _regular_file_stat(path)
    before_key = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_key = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_key != after_key:
        raise RuntimeError(f"File metadata changed during hashing: {path}")
    return after, digest


def _write_new_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Immutable artifact already exists: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"Atomic temporary already exists: {temporary}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _seal_payload(path: Path, payload: dict[str, Any], hash_field: str) -> str:
    payload_hash = canonical_json_hash(payload)
    sealed = dict(payload)
    sealed[hash_field] = payload_hash
    _write_new_json_atomic(path, sealed)
    return payload_hash


def _inside_exact_excluded_root(path: Path) -> bool:
    return path.is_relative_to(V1_DIR) or path.is_relative_to(V2_DIR)


def _inventory_replays() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(RUNS_DIR.rglob("replay.json")):
        if _inside_exact_excluded_root(path):
            continue
        try:
            metadata = _regular_file_stat(path)
        except (FileNotFoundError, ValueError):
            continue
        if metadata.st_mtime_ns >= CUTOFF_NS:
            continue
        metadata, digest = _metadata_stable_hash(path)
        rows.append(
            {
                "path": _relative(path),
                "bytes": int(metadata.st_size),
                "mtime_ns": int(metadata.st_mtime_ns),
                "mtime_utc": datetime.fromtimestamp(
                    metadata.st_mtime, tz=timezone.utc
                ).isoformat(),
                "predates_cutoff": True,
                "sha256": digest,
            }
        )
    if not rows:
        raise RuntimeError("Eligible replay inventory is empty")
    return rows


def _directory_logical_bytes(path: Path) -> tuple[int, int]:
    total = 0
    count = 0
    for candidate in path.rglob("*"):
        try:
            metadata = _regular_file_stat(candidate)
        except (FileNotFoundError, ValueError):
            continue
        total += int(metadata.st_size)
        count += 1
    return total, count


def _sealed_hash_audit() -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for name, expected in EXPECTED_SEALED_HASHES.items():
        path = expected["path"]
        actual = _sha256(path)
        rows[name] = {
            "path": _relative(path),
            "expected_sha256": expected["sha256"],
            "actual_sha256": actual,
            "passes": actual == expected["sha256"],
        }
    charter_actual = _sha256(CHARTER_PATH)
    rows["storage_audit_charter"] = {
        "path": _relative(CHARTER_PATH),
        "expected_sha256": CHARTER_SHA256,
        "actual_sha256": charter_actual,
        "passes": charter_actual == CHARTER_SHA256,
    }
    return {
        "artifacts": rows,
        "passes": all(row["passes"] for row in rows.values()),
    }


def run_audit() -> dict[str, Any]:
    if INVENTORY_PATH.exists() or AUDIT_PATH.exists():
        raise FileExistsError("Supplemental storage audit is one-shot and already opened")

    sealed_before = _sealed_hash_audit()
    if not sealed_before["passes"]:
        raise RuntimeError("Sealed input hash verification failed before inventory")

    base_bytes, base_file_count = _directory_logical_bytes(V2_DIR)
    inventory_rows = _inventory_replays()
    top_ten = sorted(
        inventory_rows, key=lambda row: (-int(row["bytes"]), str(row["path"]))
    )[:10]
    maximum = top_ten[0]

    inventory_payload = {
        "version": "g1r_qd_v2_storage_replay_inventory_v1",
        "charter_sha256": CHARTER_SHA256,
        "cutoff": {
            "admission_opened_at": CUTOFF_TEXT,
            "cutoff_mtime_ns": CUTOFF_NS,
            "comparison": "stat.st_mtime_ns < cutoff_mtime_ns",
        },
        "scan_root": _relative(RUNS_DIR),
        "excluded_exact_roots": [_relative(V1_DIR), _relative(V2_DIR)],
        "content_contract": "raw byte SHA-256 only; replay JSON not parsed",
        "order": "path ascending",
        "replay_count": len(inventory_rows),
        "rows": inventory_rows,
    }
    inventory_payload_hash = _seal_payload(
        INVENTORY_PATH, inventory_payload, "inventory_payload_sha256"
    )
    inventory_file_sha256 = _sha256(INVENTORY_PATH)

    maximum_bytes = int(maximum["bytes"])
    per_game_bytes = maximum_bytes + MIB
    increment_bytes = 120 * per_game_bytes
    pre_overhead_bytes = base_bytes + increment_bytes
    projection_bytes = math.ceil(1.25 * pre_overhead_bytes)
    projection_headroom_bytes = FOUR_GIB - projection_bytes

    disk = shutil.disk_usage(RUNS_DIR)
    free_bytes = int(disk.free)
    try:
        services = service_health()
    except Exception as error:  # pragma: no cover - operational fail-closed path
        services = {
            "passes": False,
            "error_type": type(error).__name__,
            "error": str(error),
        }
    sealed_after = _sealed_hash_audit()

    checks = {
        "sealed_inputs_unchanged": sealed_before["passes"]
        and sealed_after["passes"],
        "inventory_nonempty": bool(inventory_rows),
        "all_inventory_rows_predate_cutoff": all(
            row["predates_cutoff"] and int(row["mtime_ns"]) < CUTOFF_NS
            for row in inventory_rows
        ),
        "projection_strictly_below_4_gib": projection_bytes < FOUR_GIB,
        "free_disk_strictly_above_120_gib": free_bytes > 120 * GIB,
        "services_dashboard_top_three_pass": bool(services["passes"]),
    }
    decision = (
        "READY_QD_STORAGE_ADMISSION"
        if all(checks.values())
        else "KILL_QD_STORAGE_ADMISSION"
    )
    audit_payload = {
        "version": "g1r_qd_v2_storage_admission_audit_v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "decision": decision,
        "supplements_only": _relative(V2_DIR / "admission_result.json"),
        "original_action_family_decision": "READY_QD_FAMILY_ADMISSION",
        "acquisition_authorized": False,
        "charter": {
            "path": _relative(CHARTER_PATH),
            "sha256": CHARTER_SHA256,
        },
        "sealed_input_hash_audit_before": sealed_before,
        "sealed_input_hash_audit_after": sealed_after,
        "inventory": {
            "path": _relative(INVENTORY_PATH),
            "file_sha256": inventory_file_sha256,
            "canonical_payload_sha256": inventory_payload_hash,
            "replay_count": len(inventory_rows),
            "maximum": maximum,
            "top_ten": top_ten,
        },
        "admission_directory_baseline": {
            "path": _relative(V2_DIR),
            "logical_bytes_B": base_bytes,
            "regular_file_count": base_file_count,
            "measurement_timing": (
                "immediately before supplemental inventory/audit artifacts"
            ),
        },
        "projection": {
            "mib_bytes": MIB,
            "gib_bytes": GIB,
            "four_gib_bytes": FOUR_GIB,
            "maximum_replay_bytes_M": maximum_bytes,
            "summary_allowance_per_game_bytes": MIB,
            "replay_plus_summary_bytes_per_game": per_game_bytes,
            "game_count": 120,
            "first_120_game_increment_bytes": increment_bytes,
            "pre_overhead_total_bytes": pre_overhead_bytes,
            "overhead_multiplier": 1.25,
            "projected_bytes_P": projection_bytes,
            "projected_gib_P": projection_bytes / GIB,
            "headroom_to_4_gib_bytes": projection_headroom_bytes,
            "headroom_to_4_gib_gib": projection_headroom_bytes / GIB,
        },
        "disk": {
            "free_bytes": free_bytes,
            "free_gib": free_bytes / GIB,
            "required_strictly_above_gib": 120,
        },
        "services": services,
        "checks": checks,
        "zero_work": {
            "candidate_or_reference_actions": 0,
            "timing_assays": 0,
            "games_generated": 0,
            "streams_consumed": 0,
            "labels_generated": 0,
            "models_fit": 0,
            "continuations_run": 0,
            "score_or_policy_outcomes_inspected": False,
            "incumbent_changed": False,
            "dashboard_changed": False,
        },
        "holds_after_audit": [
            "acquisition",
            "G1-R",
            "incumbent promotion",
            "dashboard promotion",
        ],
    }
    audit_payload_hash = _seal_payload(
        AUDIT_PATH, audit_payload, "audit_payload_sha256"
    )
    audit_file_sha256 = _sha256(AUDIT_PATH)
    return {
        "decision": decision,
        "inventory_path": str(INVENTORY_PATH),
        "inventory_file_sha256": inventory_file_sha256,
        "inventory_payload_sha256": inventory_payload_hash,
        "audit_path": str(AUDIT_PATH),
        "audit_file_sha256": audit_file_sha256,
        "audit_payload_sha256": audit_payload_hash,
        "replay_count": len(inventory_rows),
        "maximum_replay": maximum,
        "base_bytes": base_bytes,
        "projection_bytes": projection_bytes,
        "projection_gib": projection_bytes / GIB,
        "free_gib": free_bytes / GIB,
        "checks": checks,
    }


def main() -> None:
    print(json.dumps(run_audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
