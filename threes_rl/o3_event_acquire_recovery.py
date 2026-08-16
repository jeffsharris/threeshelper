"""Marker-bound recovery for the never-run O3 acquisition complement."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from threes_rl import g1r_acquire as history
from threes_rl import g1r_acquire_v2_qd5 as policy_source
from threes_rl import o3_event_acquire as source
from threes_rl import o3_p0_preflight as p0
from threes_rl.eval import EvalJob, EvalStreamIds, iter_eval_job_outputs
from threes_rl.replay_provenance import ORIGIN_FRESH, direct_root_fields
from threes_rl.restart_manifest import canonical_ancestry_id
from threes_rl.run_artifacts import write_json
from threes_rl.s3_power_preflight import sha256_path


VERSION = "o3_event_acquisition_recovery_v1"
ROOT = Path("threes_rl/runs")
CHARTER_PATH = Path("threes_rl/O3_EVENT_ACQUISITION_RECOVERY_CHARTER.md")
RUNNER_PATH = Path("threes_rl/o3_event_acquire_recovery.py")
TEST_PATH = Path("tests/test_rl_o3_event_acquire_recovery.py")
TEST_EVIDENCE_PATH = (
    ROOT / "forensics/o3_event_acquisition_recovery_test_evidence.json"
)
OUTPUT_DIR = ROOT / "forensics/o3_event_acquisition_recovery_v1"
OWNERSHIP_PATH = OUTPUT_DIR / "O3_RECOVERY_OWNERSHIP.json"
PROCESS_GUARD_PATH = OUTPUT_DIR / "process_guards.jsonl"
SOURCE_AUDIT_PATH = OUTPUT_DIR / "O3_RECOVERY_SOURCE_AUDIT.json"
COMPLEMENT_PATH = OUTPUT_DIR / "O3_RECOVERY_COMPLEMENT.json"
COLLISION_PATH = OUTPUT_DIR / "O3_RECOVERY_COLLISION_AUDIT.json"
MARKER_PATH = OUTPUT_DIR / "O3_RECOVERY_OPENED.json"
PREFLIGHT_RESULT_PATH = OUTPUT_DIR / "O3_RECOVERY_PREFLIGHT_RESULT.json"
ATTEMPT_PATH = OUTPUT_DIR / "attempts.jsonl"
COMPLETION_PATH = OUTPUT_DIR / "completed_games.jsonl"
RUNTIME_PATH = OUTPUT_DIR / "runtime_state.json"
REPLAY_DIR = OUTPUT_DIR / "source_replays"
UNION_PATH = OUTPUT_DIR / "O3_RECOVERY_UNION_MANIFEST.json"
SUPPORT_PATH = OUTPUT_DIR / "O3_RECOVERY_SUPPORT_SCAN.json"
SELECTED_PATH = OUTPUT_DIR / "O3_RECOVERY_SELECTED_ROOTS.json"
RESULT_PATH = OUTPUT_DIR / "O3_RECOVERY_RESULT.json"

ORIGINAL_DIR = source.OUTPUT_DIR
ORIGINAL_ARTIFACTS = {
    "execution_charter": (
        source.CHARTER_PATH,
        "b557c1ce5838d8841a5ebad3dec9c6559de46b5c2c7aefa051f38d483120de38",
        None,
        None,
    ),
    "runner": (
        source.RUNNER_PATH,
        "842fee2b41526d6c37770b7deee09500354e9140731753da905c1900e974bd5b",
        None,
        None,
    ),
    "tests": (
        source.TEST_PATH,
        "cbafa61a29aae6cfae107eac85b8f56519edb356deb11996e5da5379ef1d922e",
        None,
        None,
    ),
    "test_evidence": (
        source.TEST_EVIDENCE_PATH,
        "a353845a06f4c4fac6cba9fd869e1ccfd779d8bb03afa7f913888786600adedd",
        "test_evidence_payload_sha256",
        "372c2685d0bc589154bc644201fdc6f1463a51db729c5517c3b25e9894fdcb84",
    ),
    "marker": (
        source.MARKER_PATH,
        "fcf9e275444ab1cce3b11855b54f84236ba2cb2ba96aa861c444830a99371145",
        "opened_payload_sha256",
        "c8fc61d249478b879c694fb166ae2e4f3d206765d74f0abd889844fd609db404",
    ),
    "result": (
        source.RESULT_PATH,
        "f7a967b936894a3d626055e366dc899d4efc52e9ad4b791b8c9a95d6e7fc791a",
        "result_payload_sha256",
        "677b99bddc124cba128b46031dc67559db5845a86e3013464084712f88e2f9ff",
    ),
    "attempts": (
        source.ATTEMPT_PATH,
        "fbf59e5f1e614667f7e804c5ded4bcdde356f0f9c56a8a743b26aadd0488a4f3",
        None,
        None,
    ),
    "completions": (
        source.COMPLETION_PATH,
        "7b121f99945a15692b0719c0adebd6f30eb2fbb17c0c362d2e792ea79546aeb5",
        None,
        None,
    ),
    "runtime": (
        source.RUNTIME_PATH,
        "e70de8b6531f9c5650007048fd8c9f25e688323ad0c709db29b57ed017f64804",
        None,
        None,
    ),
}
ORIGINAL_COMPLETIONS = 18_990
RECOVERY_ROOTS = 1_510
RECOVERY_ROOTS_PER_FAMILY = 302
FAMILY_ORDER = source.FAMILY_ORDER
FROZEN_JOBS = 1
MINIMUM_NICE = 10
ACTIVE_RUNTIME_LIMIT = 18 * 60 * 60
RECOVERY_BYTE_LIMIT = 6 * 1024**3
COMBINED_BYTE_LIMIT = source.BYTE_LIMIT
MIN_FREE_GIB = source.MIN_FREE_GIB
TARGET_FREE_GIB = source.TARGET_FREE_GIB
CHUNK_SIZE = len(FAMILY_ORDER)
STARTER_TILE = source.STARTER_TILE
MAX_MOVES = source.MAX_MOVES
ATTEMPT_TERMINAL_STATUSES = {
    "completed",
    "completed_recovered",
    "interrupted_no_replay",
}
ALLOWED_SERVICE_TOKENS = (
    "threes_rl.dashboard",
    "threes_rl.human_play_server",
)
HEAVY_TOKENS = (
    "train",
    "eval",
    "acquire",
    "admission",
    "continuation",
    "label",
    "fit",
    "mcts",
)
LIVE_UNBOUND_NAMES = {
    "dashboard.json",
    "score_trends.json",
}
REPLAY_CONTENT_PARTS = {
    "source_replays",
    "replays",
    "human_play",
    "continuations",
    "datasets",
}
O2_FORBIDDEN_DIRS = source.O2_FORBIDDEN_DIRS
DEPENDENCY_PATHS = (
    source.COURSE_CHARTER_PATH,
    source.RUNNER_PATH,
    source.TEST_PATH,
    source.TEST_EVIDENCE_PATH,
    Path("threes_rl/o3_p0_preflight.py"),
    Path("threes_rl/o3_designated_pair_option.py"),
    Path("threes_rl/o3_power_contract.py"),
    Path("threes_rl/eval.py"),
    Path("threes_rl/sim.py"),
    Path("threes_rl/record_replay.py"),
    Path("threes_rl/replay_provenance.py"),
    Path("threes_rl/restart_manifest.py"),
    Path("threes_rl/train_td.py"),
    Path("threes_rl/g1r_acquire.py"),
    Path("threes_rl/g1r_acquire_v2_qd5.py"),
    Path("threes_rl/g1r_qd_admission_v2.py"),
)


def canonical_json_hash(value: Any) -> str:
    return source.canonical_json_hash(value)


def _write_immutable_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    self_hash_field: str,
) -> dict[str, Any]:
    return source._write_immutable_json(
        path,
        payload,
        self_hash_field=self_hash_field,
    )


def _verify_self_hash(payload: Mapping[str, Any], field: str) -> bool:
    return source._verify_self_hash(payload, field)


def _artifact_identity(path: Path, field: str) -> dict[str, Any]:
    return source._artifact_identity(path, field)


def _read_jsonl_metadata(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _original_file_audit() -> dict[str, Any]:
    rows = {}
    for name, (path, expected_file, field, expected_payload) in (
        ORIGINAL_ARTIFACTS.items()
    ):
        file_sha = sha256_path(path)
        checks = {"file_sha256_exact": file_sha == expected_file}
        payload_sha = None
        if field is not None:
            payload = json.loads(path.read_text())
            payload_sha = payload.get(field)
            checks.update(
                {
                    "payload_self_hash_valid": _verify_self_hash(payload, field),
                    "payload_sha256_exact": payload_sha == expected_payload,
                }
            )
        rows[name] = {
            "path": str(path),
            "file_sha256": file_sha,
            "payload_sha256": payload_sha,
            "checks": checks,
            "passes": all(checks.values()),
        }
    checks = {
        "all_original_artifacts_exact": all(
            row["passes"] for row in rows.values()
        ),
        "p0_artifacts_exact": source._sealed_artifact_audit()["passes"],
        "support_never_opened": not source.SUPPORT_PATH.exists(),
        "selected_never_opened": not source.SELECTED_PATH.exists(),
    }
    return {
        "artifacts": rows,
        "checks": checks,
        "passes": all(checks.values()),
    }


def _completion_key(row: Mapping[str, Any]) -> tuple[str, int]:
    return str(row["family"]), int(row["game_index"])


def _completion_allowed_keys() -> set[str]:
    return {
        "complete",
        "dashboard_eligible",
        "deck_stream_id",
        "family",
        "family_index",
        "game_index",
        "logical_seed",
        "planned_root_id",
        "policy_stream_id",
        "role",
        "root_cluster",
        "slot_stream_id",
        "source_replay",
        "source_replay_sha256",
    }


def audit_original_source() -> dict[str, Any]:
    file_audit = _original_file_audit()
    completions = _read_jsonl_metadata(source.COMPLETION_PATH)
    attempts = _read_jsonl_metadata(source.ATTEMPT_PATH)
    expected_rows = {_completion_key(row): row for row in source.acquisition_rows()}
    completion_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    replay_rows = []
    for completion in completions:
        key = _completion_key(completion)
        if key in completion_by_key:
            raise ValueError(f"Duplicate original completion metadata: {key}")
        if set(completion) != _completion_allowed_keys():
            raise ValueError(f"Unexpected original completion schema: {key}")
        planned = expected_rows.get(key)
        if planned is None:
            raise ValueError(f"Original completion outside P0 universe: {key}")
        for field in (
            "family_index",
            "role",
            "planned_root_id",
            "logical_seed",
            "deck_stream_id",
            "slot_stream_id",
            "policy_stream_id",
        ):
            if completion[field] != planned[field]:
                raise ValueError(f"Original completion drift at {key}: {field}")
        replay_path = Path(str(completion["source_replay"]))
        if not replay_path.is_file():
            raise FileNotFoundError(f"Missing original replay: {replay_path}")
        actual_sha = sha256_path(replay_path)
        if actual_sha != completion["source_replay_sha256"]:
            raise ValueError(f"Original replay bytes changed: {replay_path}")
        replay_rows.append(
            {
                "family": key[0],
                "game_index": key[1],
                "path": str(replay_path),
                "bytes": int(replay_path.stat().st_size),
                "sha256": actual_sha,
            }
        )
        completion_by_key[key] = dict(completion)

    statuses = Counter(str(row.get("status")) for row in attempts)
    attempt_ids = Counter(str(row.get("attempt_id")) for row in attempts)
    attempt_keys = Counter(
        (str(row.get("family")), int(row.get("game_index", -1)))
        for row in attempts
        if row.get("status") == "opened"
    )
    families = Counter(str(row["family"]) for row in completions)
    roles = Counter(str(row["role"]) for row in completions)
    roots = [str(row["root_cluster"]) for row in completions]
    replay_hashes = [str(row["source_replay_sha256"]) for row in completions]
    checks = {
        "original_files_exact": file_audit["passes"],
        "exact_18990_completions": len(completions) == ORIGINAL_COMPLETIONS,
        "exact_3798_per_family": families
        == {family: 3_798 for family in FAMILY_ORDER},
        "train_complete": roles["train"] == p0.ROLE_COUNTS["train"],
        "development_complete": roles["development"]
        == p0.ROLE_COUNTS["development"],
        "untouched_partial_exact": roles["untouched_mechanism"] == 12_295,
        "unique_completion_keys": len(completion_by_key) == len(completions),
        "unique_ancestries": len(roots) == len(set(roots)),
        "unique_replay_hashes": len(replay_hashes) == len(set(replay_hashes)),
        "exact_attempt_rows": len(attempts) == 37_980,
        "attempt_statuses_balanced": statuses
        == {"opened": ORIGINAL_COMPLETIONS, "completed": ORIGINAL_COMPLETIONS},
        "attempt_ids_paired": len(attempt_ids) == ORIGINAL_COMPLETIONS
        and all(count == 2 for count in attempt_ids.values()),
        "one_open_per_completion_key": set(attempt_keys) == set(completion_by_key)
        and all(count == 1 for count in attempt_keys.values()),
        "zero_retries": all(
            int(row.get("attempt_number", -1)) == 0 for row in attempts
        ),
        "all_replay_bytes_hash_exact_without_parsing": len(replay_rows)
        == ORIGINAL_COMPLETIONS,
    }
    compact_completions = sorted(
        completion_by_key.values(),
        key=lambda row: (int(row["family_index"]), int(row["game_index"])),
    )
    replay_rows.sort(key=lambda row: (row["family"], row["game_index"]))
    return {
        "original_file_audit": file_audit,
        "completion_count": len(completions),
        "family_counts": dict(sorted(families.items())),
        "role_counts": dict(sorted(roles.items())),
        "unique_ancestries": len(set(roots)),
        "unique_replay_hashes": len(set(replay_hashes)),
        "attempt_rows": len(attempts),
        "attempt_status_counts": dict(sorted(statuses.items())),
        "completion_manifest_sha256": canonical_json_hash(compact_completions),
        "replay_byte_manifest_sha256": canonical_json_hash(replay_rows),
        "replay_byte_manifest": replay_rows,
        "max_replay_bytes": max(
            (int(row["bytes"]) for row in replay_rows),
            default=0,
        ),
        "checks": checks,
        "passes": all(checks.values()),
        "replay_bodies_parsed": False,
        "support_content_opened": False,
        "score_action_max_tile_outcome_fields_read": False,
    }


def derive_complement(
    *,
    acquisition_rows: Sequence[Mapping[str, Any]] | None = None,
    completion_rows: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    planned = [
        dict(row)
        for row in (
            source.acquisition_rows()
            if acquisition_rows is None
            else acquisition_rows
        )
    ]
    completed = (
        _read_jsonl_metadata(source.COMPLETION_PATH)
        if completion_rows is None
        else [dict(row) for row in completion_rows]
    )
    completed_keys = {_completion_key(row) for row in completed}
    if len(completed_keys) != len(completed):
        raise ValueError("Original completion metadata contains duplicate keys")
    complement = [row for row in planned if _completion_key(row) not in completed_keys]
    complement.sort(key=lambda row: (int(row["game_index"]), int(row["family_index"])))
    counts = Counter(str(row["family"]) for row in complement)
    indices = {
        family: [int(row["game_index"]) for row in complement if row["family"] == family]
        for family in FAMILY_ORDER
    }
    checks = {
        "exact_1510": len(complement) == RECOVERY_ROOTS,
        "exact_302_per_family": counts
        == {family: RECOVERY_ROOTS_PER_FAMILY for family in FAMILY_ORDER},
        "all_untouched_mechanism": all(
            row["role"] == "untouched_mechanism" for row in complement
        ),
        "frozen_family_order": tuple(
            dict.fromkeys(str(row["family"]) for row in complement[:CHUNK_SIZE])
        )
        == FAMILY_ORDER,
        "indices_3798_through_4099": all(
            values == list(range(3_798, 4_100)) for values in indices.values()
        ),
        "set_difference_exact": len(complement) + len(completed_keys)
        == source.TOTAL_ROOTS,
    }
    if not all(checks.values()):
        raise ValueError(f"O3 recovery complement mismatch: {checks}")
    return complement


def complement_payload(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    compact = [dict(row) for row in rows]
    return {
        "version": f"{VERSION}_complement",
        "source_p0_stream_manifest_file_sha256": sha256_path(
            source.P0_DIR / p0.STREAM_MANIFEST_NAME
        ),
        "source_completion_file_sha256": sha256_path(source.COMPLETION_PATH),
        "rows": compact,
        "rows_sha256": canonical_json_hash(compact),
        "count": len(compact),
        "family_counts": dict(
            sorted(Counter(str(row["family"]) for row in compact).items())
        ),
        "role_counts": dict(
            sorted(Counter(str(row["role"]) for row in compact).items())
        ),
        "derivation": (
            "P0 acquisition rows minus immutable original completion "
            "(family,game_index) keys"
        ),
        "original_replay_bodies_parsed": False,
        "passes": len(compact) == RECOVERY_ROOTS,
    }


def round_robin_chunks(
    rows: Sequence[Mapping[str, Any]],
) -> list[list[dict[str, Any]]]:
    by_key = {_completion_key(row): dict(row) for row in rows}
    if len(by_key) != RECOVERY_ROOTS:
        raise ValueError("Recovery complement rows are not unique")
    chunks = []
    for game_index in range(3_798, 4_100):
        chunk = [by_key[(family, game_index)] for family in FAMILY_ORDER]
        chunks.append(chunk)
    if len(chunks) != RECOVERY_ROOTS_PER_FAMILY or any(
        len(chunk) != CHUNK_SIZE for chunk in chunks
    ):
        raise ValueError("Recovery round-robin chunk construction failed")
    return chunks


def _is_within(path: Path, directory: Path) -> bool:
    return source._is_within(path, directory)


def _is_replay_content(path: Path) -> bool:
    parts = set(path.parts)
    return path.name == "replay.json" or bool(parts.intersection(REPLAY_CONTENT_PARTS))


def collision_audit(
    rows: Sequence[Mapping[str, Any]],
    *,
    out_dir: Path = OUTPUT_DIR,
    scan_root: Path = ROOT,
) -> dict[str, Any]:
    requested = {
        field: {int(row[field]) for row in rows}
        for field in p0.STREAM_FIELDS
    }
    found: dict[str, set[int]] = defaultdict(set)
    scanned = []
    exclusions = Counter()
    for path in sorted(scan_root.rglob("*")):
        if not path.is_file() or path.suffix not in {".json", ".jsonl", ".csv"}:
            continue
        if _is_within(path, out_dir):
            exclusions["current_recovery_namespace"] += 1
            continue
        if _is_within(path, source.P0_DIR):
            exclusions["immutable_p0_reservation"] += 1
            continue
        if _is_within(path, source.REPLAY_DIR):
            exclusions["original_replay_bytes_hash_bound_unread"] += 1
            continue
        if any(_is_within(path, directory) for directory in O2_FORBIDDEN_DIRS):
            exclusions["o2_content_forbidden_unread"] += 1
            continue
        if path.name in LIVE_UNBOUND_NAMES or _is_within(path, ROOT / "dashboard"):
            exclusions["live_dashboard_unbound"] += 1
            continue
        if _is_replay_content(path):
            exclusions["other_replay_content_unread"] += 1
            continue
        values = history._scan_history_file(path)
        if not values:
            continue
        for field, items in values.items():
            found[field].update(items)
        scanned.append(
            {
                "path": str(path),
                "bytes": int(path.stat().st_size),
                "sha256": sha256_path(path),
                "counts": {
                    field: len(items) for field, items in sorted(values.items())
                },
            }
        )
    collisions = {}
    for field, values in requested.items():
        prior = set(found.get(field, set()))
        if field == "logical_seed":
            for alias in ("seed", "root_seed", "source_seed", "fresh_root_seed"):
                prior.update(found.get(alias, set()))
        collisions[field] = sorted(values.intersection(prior))
    flat = [
        int(row[field])
        for row in rows
        for field in p0.STREAM_FIELDS
    ]
    checks = {
        "requested_ids_internally_unique": len(flat) == len(set(flat)),
        "zero_external_collisions": not any(collisions.values()),
        "p0_reservation_excluded": exclusions["immutable_p0_reservation"] > 0,
        "original_replays_unread": exclusions[
            "original_replay_bytes_hash_bound_unread"
        ]
        == ORIGINAL_COMPLETIONS,
        "o2_content_unread": exclusions["o2_content_forbidden_unread"] > 0,
    }
    return {
        "requested_rows": len(rows),
        "requested_rows_sha256": canonical_json_hash(
            [dict(row) for row in rows]
        ),
        "scanned_source_count": len(scanned),
        "scanned_sources_sha256": canonical_json_hash(scanned),
        "scanned_sources": scanned,
        "exclusion_counts": dict(sorted(exclusions.items())),
        "collisions": collisions,
        "checks": checks,
        "passes": all(checks.values()),
        "replay_bodies_parsed": False,
    }


def _parse_process_table(text: str, current_pid: int) -> dict[str, Any]:
    processes: dict[int, dict[str, Any]] = {}
    for line in text.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
        except ValueError:
            continue
        processes[pid] = {
            "pid": pid,
            "ppid": ppid,
            "command": parts[2] if len(parts) == 3 else "",
        }
    ancestors = {current_pid}
    cursor = current_pid
    while cursor in processes:
        parent = int(processes[cursor]["ppid"])
        if parent <= 0 or parent in ancestors:
            break
        ancestors.add(parent)
        cursor = parent
    candidates = []
    disallowed = []
    for pid, process in sorted(processes.items()):
        command = str(process["command"])
        if "python" not in command or "threes_rl" not in command:
            continue
        if pid in ancestors:
            classification = "current_process_or_ancestor"
        elif any(token in command for token in ALLOWED_SERVICE_TOKENS):
            classification = "allowed_dashboard_or_recorder"
        elif any(token in command for token in HEAVY_TOKENS):
            classification = "disallowed_heavy_threes"
            disallowed.append(dict(process))
        else:
            classification = "other_python_threes_nonheavy"
        candidates.append({**process, "classification": classification})
    return {
        "current_pid": current_pid,
        "ancestor_pids": sorted(ancestors),
        "candidate_processes": candidates,
        "disallowed_processes": disallowed,
        "passes": not disallowed,
    }


def process_audit() -> dict[str, Any]:
    result = subprocess.run(
        ["ps", "ax", "-o", "pid=,ppid=,command="],
        check=True,
        capture_output=True,
        text=True,
    )
    return _parse_process_table(result.stdout, os.getpid())


def _append_process_guard(stage: str, operations: Mapping[str, Any]) -> dict[str, Any]:
    record = {
        "stage": stage,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "pid": os.getpid(),
        **dict(operations),
    }
    history._append_jsonl_row(PROCESS_GUARD_PATH, record)
    return record


def _operational_audit(stage: str) -> dict[str, Any]:
    process = process_audit()
    free_gib = shutil.disk_usage(OUTPUT_DIR.parent).free / 1024**3
    recovery_bytes = (
        history._directory_bytes(OUTPUT_DIR) if OUTPUT_DIR.exists() else 0
    )
    original_bytes = history._directory_bytes(ORIGINAL_DIR)
    services = history.service_health()
    checks = {
        "nice_at_least_10": history.current_nice() >= MINIMUM_NICE,
        "no_competing_heavy_process": process["passes"],
        "free_disk_above_100_gib": free_gib > MIN_FREE_GIB,
        "recovery_output_below_6_gib": recovery_bytes < RECOVERY_BYTE_LIMIT,
        "combined_output_below_28_gib": (
            recovery_bytes + original_bytes < COMBINED_BYTE_LIMIT
        ),
        "services_dashboard_top_three": services["passes"],
    }
    audit = {
        "nice": history.current_nice(),
        "free_gib": free_gib,
        "recovery_output_bytes": recovery_bytes,
        "original_output_bytes": original_bytes,
        "combined_output_bytes": recovery_bytes + original_bytes,
        "process_audit": process,
        "services": services,
        "checks": checks,
        "passes": all(checks.values()),
    }
    _append_process_guard(stage, audit)
    return audit


def _bound_commands(out_dir: Path) -> dict[str, str]:
    prefix = (
        "zsh -ic 'no-secrets nice -n 10 env PYTHONPATH=. "
        ".venv/bin/python -m threes_rl.o3_event_acquire_recovery"
    )
    suffix = f" --out-dir {out_dir} --jobs 1'"
    return {
        "open": f"{prefix} open{suffix}",
        "execute": f"{prefix} execute{suffix}",
    }


def _dependency_manifest() -> dict[str, Any]:
    rows = [
        {
            "path": str(path),
            "bytes": int(path.stat().st_size),
            "sha256": sha256_path(path),
        }
        for path in DEPENDENCY_PATHS
    ]
    return {"rows": rows, "manifest_sha256": canonical_json_hash(rows)}


def _create_ownership_file(out_dir: Path) -> dict[str, Any]:
    payload = {
        "version": f"{VERSION}_ownership",
        "bound_out_dir": str(out_dir.resolve()),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "commands": _bound_commands(out_dir),
        "jobs": FROZEN_JOBS,
        "minimum_nice": MINIMUM_NICE,
        "ownership_semantics": (
            "atomically-created identity file plus execution-lifetime "
            "nonblocking exclusive flock"
        ),
    }
    body = dict(payload)
    body["ownership_payload_sha256"] = canonical_json_hash(body)
    raw = (json.dumps(body, indent=2, sort_keys=True) + "\n").encode("ascii")
    fd = os.open(
        OWNERSHIP_PATH,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o644,
    )
    try:
        os.write(fd, raw)
        os.fsync(fd)
    finally:
        os.close(fd)
    return body


def _load_ownership() -> dict[str, Any]:
    payload = json.loads(OWNERSHIP_PATH.read_text())
    if not _verify_self_hash(payload, "ownership_payload_sha256"):
        raise ValueError("Recovery ownership payload mismatch")
    expected = {
        "version": f"{VERSION}_ownership",
        "bound_out_dir": str(OUTPUT_DIR.resolve()),
        "commands": _bound_commands(OUTPUT_DIR),
        "jobs": FROZEN_JOBS,
        "minimum_nice": MINIMUM_NICE,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"Recovery ownership identity changed: {key}")
    return payload


@contextmanager
def execution_ownership() -> Iterable[None]:
    _load_ownership()
    fd = os.open(OWNERSHIP_PATH, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        os.close(fd)
        raise RuntimeError("O3 recovery ownership lock is already held") from error
    try:
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _load_test_evidence() -> dict[str, Any]:
    payload = json.loads(TEST_EVIDENCE_PATH.read_text())
    if not _verify_self_hash(payload, "test_evidence_payload_sha256"):
        raise ValueError("Recovery test evidence payload mismatch")
    expected = {
        "charter_sha256": sha256_path(CHARTER_PATH),
        "runner_sha256": sha256_path(RUNNER_PATH),
        "tests_sha256": sha256_path(TEST_PATH),
        "original_runner_sha256": sha256_path(source.RUNNER_PATH),
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("Recovery test evidence source mismatch")
    if not payload.get("passes"):
        raise ValueError("Recovery test evidence is not passing")
    return payload


def seal_test_evidence(
    *,
    focused_passed: int,
    regression_passed: int,
    commands: Sequence[str],
) -> dict[str, Any]:
    return _write_immutable_json(
        TEST_EVIDENCE_PATH,
        {
            "version": f"{VERSION}_test_evidence",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "charter_sha256": sha256_path(CHARTER_PATH),
            "runner_sha256": sha256_path(RUNNER_PATH),
            "tests_sha256": sha256_path(TEST_PATH),
            "original_runner_sha256": sha256_path(source.RUNNER_PATH),
            "focused_tests_passed": int(focused_passed),
            "regression_tests_passed": int(regression_passed),
            "commands": list(commands),
            "passes": focused_passed > 0 and regression_passed > 0,
            "games_generated": 0,
            "streams_consumed": 0,
            "original_replay_bodies_parsed": False,
            "support_content_opened": False,
            "labels_generated": 0,
            "models_fit": 0,
            "policy_outcomes_opened": False,
        },
        self_hash_field="test_evidence_payload_sha256",
    )


def _storage_projection(source_audit: Mapping[str, Any]) -> dict[str, Any]:
    original_bytes = history._directory_bytes(ORIGINAL_DIR)
    static_recovery_bytes = history._directory_bytes(OUTPUT_DIR)
    maximum_replay = int(source_audit["max_replay_bytes"])
    projected_recovery = math.ceil(
        1.25
        * (
            static_recovery_bytes
            + RECOVERY_ROOTS * (maximum_replay + 1024**2)
        )
    )
    projected_combined = original_bytes + projected_recovery
    checks = {
        "projected_recovery_below_6_gib": projected_recovery
        < RECOVERY_BYTE_LIMIT,
        "projected_combined_below_28_gib": projected_combined
        < COMBINED_BYTE_LIMIT,
    }
    return {
        "original_bytes": original_bytes,
        "static_recovery_bytes": static_recovery_bytes,
        "maximum_original_replay_bytes": maximum_replay,
        "per_root_summary_allowance_bytes": 1024**2,
        "overhead_multiplier": 1.25,
        "projected_recovery_bytes": projected_recovery,
        "projected_combined_bytes": projected_combined,
        "checks": checks,
        "passes": all(checks.values()),
    }


def _marker_identity() -> dict[str, Any]:
    tests = _load_test_evidence()
    dependency = _dependency_manifest()
    return {
        "version": VERSION,
        "bound_out_dir": str(OUTPUT_DIR.resolve()),
        "charter_sha256": sha256_path(CHARTER_PATH),
        "runner_sha256": sha256_path(RUNNER_PATH),
        "tests_sha256": sha256_path(TEST_PATH),
        "test_evidence_file_sha256": sha256_path(TEST_EVIDENCE_PATH),
        "test_evidence_payload_sha256": tests[
            "test_evidence_payload_sha256"
        ],
        "original_runner_sha256": sha256_path(source.RUNNER_PATH),
        "source_audit": _artifact_identity(
            SOURCE_AUDIT_PATH, "source_audit_payload_sha256"
        ),
        "complement": _artifact_identity(
            COMPLEMENT_PATH, "complement_payload_sha256"
        ),
        "collision": _artifact_identity(
            COLLISION_PATH, "collision_payload_sha256"
        ),
        "ownership": _artifact_identity(
            OWNERSHIP_PATH, "ownership_payload_sha256"
        ),
        "dependency_manifest_sha256": dependency["manifest_sha256"],
        "dependency_manifest": dependency["rows"],
        "family_order": list(FAMILY_ORDER),
        "recovery_roots": RECOVERY_ROOTS,
        "roots_per_family": RECOVERY_ROOTS_PER_FAMILY,
        "jobs": FROZEN_JOBS,
        "minimum_nice": MINIMUM_NICE,
        "active_runtime_limit_seconds": ACTIVE_RUNTIME_LIMIT,
        "recovery_byte_limit": RECOVERY_BYTE_LIMIT,
        "combined_byte_limit": COMBINED_BYTE_LIMIT,
        "minimum_free_gib": MIN_FREE_GIB,
        "target_free_gib": TARGET_FREE_GIB,
        "commands": _bound_commands(OUTPUT_DIR),
    }


def open_preflight(
    *,
    out_dir: Path = OUTPUT_DIR,
    jobs: int = FROZEN_JOBS,
) -> dict[str, Any]:
    if out_dir.resolve() != OUTPUT_DIR.resolve() or jobs != FROZEN_JOBS:
        raise ValueError("O3 recovery open identity mismatch")
    if out_dir.exists():
        raise FileExistsError(f"O3 recovery namespace exists: {out_dir}")
    out_dir.mkdir(parents=True)
    _create_ownership_file(out_dir)
    try:
        source_audit = audit_original_source()
        complement = derive_complement()
        complement_artifact = _write_immutable_json(
            COMPLEMENT_PATH,
            complement_payload(complement),
            self_hash_field="complement_payload_sha256",
        )
        source_artifact = _write_immutable_json(
            SOURCE_AUDIT_PATH,
            {
                "version": f"{VERSION}_source_audit",
                **source_audit,
            },
            self_hash_field="source_audit_payload_sha256",
        )
        collision = collision_audit(complement)
        collision_artifact = _write_immutable_json(
            COLLISION_PATH,
            {
                "version": f"{VERSION}_collision",
                **collision,
            },
            self_hash_field="collision_payload_sha256",
        )
        policies, policy_audit = source._load_policies()
        del policies
        operations = _operational_audit("preflight_open")
        projection = _storage_projection(source_audit)
        checks = {
            "source_audit_pass": source_audit["passes"],
            "complement_exact": complement_artifact["passes"],
            "collision_free": collision_artifact["passes"],
            "policies_and_signatures_exact": policy_audit["passes"],
            "operations_pass": operations["passes"],
            "free_disk_target_met": operations["free_gib"] > TARGET_FREE_GIB,
            "storage_projection_pass": projection["passes"],
            "zero_games_streams_support_labels_models_outcomes": True,
        }
        decision = (
            "READY_O3_ACQUISITION_RECOVERY"
            if all(checks.values())
            else "HOLD_O3_ACQUISITION_RECOVERY_PREFLIGHT"
        )
        marker = _write_immutable_json(
            MARKER_PATH,
            {
                **_marker_identity(),
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "decision": "O3_RECOVERY_OPENED_ZERO_WORK",
                "preflight_decision": decision,
                "policy_audit": policy_audit,
                "operations": operations,
                "storage_projection": projection,
                "checks": checks,
                "zero_work": _zero_forbidden_work(),
            },
            self_hash_field="opened_payload_sha256",
        )
        result = {
            "version": f"{VERSION}_preflight",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "decision": decision,
            "continue": (
                "EXACT_MARKER_BOUND_1510_ROOT_RECOVERY"
                if decision == "READY_O3_ACQUISITION_RECOVERY"
                else "NONE"
            ),
            "hold": (
                []
                if decision == "READY_O3_ACQUISITION_RECOVERY"
                else ["recovery_execution", "all_downstream_o3"]
            ),
            "kill": False,
            "promote": False,
            "marker": _artifact_identity(
                MARKER_PATH, "opened_payload_sha256"
            ),
            "source_audit": _artifact_identity(
                SOURCE_AUDIT_PATH, "source_audit_payload_sha256"
            ),
            "complement": _artifact_identity(
                COMPLEMENT_PATH, "complement_payload_sha256"
            ),
            "collision": _artifact_identity(
                COLLISION_PATH, "collision_payload_sha256"
            ),
            "checks": checks,
            "zero_work": _zero_forbidden_work(),
        }
        return _write_immutable_json(
            PREFLIGHT_RESULT_PATH,
            result,
            self_hash_field="result_payload_sha256",
        )
    except Exception as error:
        if MARKER_PATH.exists() or PREFLIGHT_RESULT_PATH.exists():
            raise
        operations = _operational_audit("preflight_error")
        marker = _write_immutable_json(
            MARKER_PATH,
            {
                "version": VERSION,
                "bound_out_dir": str(OUTPUT_DIR.resolve()),
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "decision": "O3_RECOVERY_OPENED_ZERO_WORK",
                "preflight_decision": "HOLD_O3_ACQUISITION_RECOVERY_PREFLIGHT",
                "error_type": type(error).__name__,
                "error": str(error),
                "operations": operations,
                "zero_work": _zero_forbidden_work(),
            },
            self_hash_field="opened_payload_sha256",
        )
        return _write_immutable_json(
            PREFLIGHT_RESULT_PATH,
            {
                "version": f"{VERSION}_preflight",
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "decision": "HOLD_O3_ACQUISITION_RECOVERY_PREFLIGHT",
                "continue": "NONE",
                "hold": ["recovery_execution", "all_downstream_o3"],
                "kill": False,
                "promote": False,
                "marker": _artifact_identity(
                    MARKER_PATH, "opened_payload_sha256"
                ),
                "error_type": type(error).__name__,
                "error": str(error),
                "zero_work": _zero_forbidden_work(),
            },
            self_hash_field="result_payload_sha256",
        )


def _zero_forbidden_work() -> dict[str, Any]:
    return {
        "games_generated": 0,
        "streams_consumed": 0,
        "original_replay_bodies_parsed": False,
        "support_content_opened": False,
        "labels_generated": 0,
        "models_fit": 0,
        "option_rollouts": 0,
        "policy_outcomes_opened": False,
        "scores_actions_max_tiles_opened": False,
        "incumbent_changes": 0,
        "dashboard_changes": 0,
    }


def _load_marker() -> dict[str, Any]:
    marker = json.loads(MARKER_PATH.read_text())
    if not _verify_self_hash(marker, "opened_payload_sha256"):
        raise ValueError("Recovery marker payload mismatch")
    current = _marker_identity()
    for key, value in current.items():
        if marker.get(key) != value:
            raise ValueError(f"Recovery marker binding changed: {key}")
    preflight = json.loads(PREFLIGHT_RESULT_PATH.read_text())
    if not _verify_self_hash(preflight, "result_payload_sha256"):
        raise ValueError("Recovery preflight payload mismatch")
    if preflight.get("decision") != "READY_O3_ACQUISITION_RECOVERY":
        raise ValueError("Recovery preflight is not READY")
    if RESULT_PATH.exists():
        raise FileExistsError("Recovery already has a terminal result")
    return marker


def _runtime_state() -> dict[str, Any]:
    if RUNTIME_PATH.is_file():
        return json.loads(RUNTIME_PATH.read_text())
    return {
        "active_runtime_seconds": 0.0,
        "evaluation_batches_charged": 0,
        "games_evaluated_charged": 0,
        "games_completed": 0,
        "chunks_completed": 0,
    }


def _attempt_id(row: Mapping[str, Any], attempt_number: int) -> str:
    return hashlib.sha256(
        (
            "O3-acquisition-recovery-v1|"
            f"{row['family']}|{row['game_index']}|{attempt_number}|"
            f"{row['logical_seed']}|{row['deck_stream_id']}|"
            f"{row['slot_stream_id']}|{row['policy_stream_id']}"
        ).encode("ascii")
    ).hexdigest()


def _append_attempt(
    row: Mapping[str, Any],
    *,
    attempt_number: int,
    status: str,
    chunk_index: int,
) -> None:
    if status not in {"opened", *ATTEMPT_TERMINAL_STATUSES}:
        raise ValueError(f"Unknown recovery attempt status: {status}")
    history._append_jsonl_row(
        ATTEMPT_PATH,
        {
            "attempt_id": _attempt_id(row, attempt_number),
            "attempt_number": attempt_number,
            "status": status,
            "family": str(row["family"]),
            "family_index": int(row["family_index"]),
            "game_index": int(row["game_index"]),
            "chunk_index": int(chunk_index),
            "logical_seed": int(row["logical_seed"]),
            "deck_stream_id": int(row["deck_stream_id"]),
            "slot_stream_id": int(row["slot_stream_id"]),
            "policy_stream_id": int(row["policy_stream_id"]),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )


def _load_attempts(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int], list[dict[str, Any]]]:
    expected = {_completion_key(row): dict(row) for row in rows}
    grouped: dict[tuple[str, int], dict[int, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    if ATTEMPT_PATH.is_file():
        for event in _read_jsonl_metadata(ATTEMPT_PATH):
            key = (str(event["family"]), int(event["game_index"]))
            if key not in expected:
                raise ValueError(f"Recovery attempt outside complement: {key}")
            row = expected[key]
            for field in (
                "family_index",
                "logical_seed",
                "deck_stream_id",
                "slot_stream_id",
                "policy_stream_id",
            ):
                if int(event[field]) != int(row[field]):
                    raise ValueError(f"Recovery attempt drift at {key}: {field}")
            grouped[key][int(event["attempt_number"])].append(
                str(event["status"])
            )
    result = {}
    for key, attempts in grouped.items():
        lifecycles = []
        for number in sorted(attempts):
            if number != len(lifecycles):
                raise ValueError(f"Recovery attempt number gap: {key}")
            statuses = attempts[number]
            if not statuses or statuses[0] != "opened" or len(statuses) > 2:
                raise ValueError(f"Malformed recovery attempt: {key}")
            lifecycles.append(
                {"attempt_number": number, "statuses": statuses}
            )
        result[key] = lifecycles
    return result


def _replay_path(row: Mapping[str, Any]) -> Path:
    return REPLAY_DIR / (
        f"{row['family']}_game_{int(row['game_index']):05d}_"
        f"seed_{int(row['logical_seed'])}.json"
    )


def _completion_from_replay(
    replay: Mapping[str, Any],
    *,
    replay_path: Path,
    row: Mapping[str, Any],
) -> dict[str, Any]:
    completion = source._completion_from_replay(
        replay,
        replay_path=replay_path,
        stream_row=row,
    )
    return completion


def _store_output(output: Any, row: Mapping[str, Any]) -> dict[str, Any]:
    replay = output.replay
    if not isinstance(replay, dict):
        raise ValueError("Recovery evaluator omitted replay capture")
    path = _replay_path(row)
    if path.exists():
        raise FileExistsError(f"Recovery replay already exists: {path}")
    replay.update(
        direct_root_fields(
            origin=ORIGIN_FRESH,
            seed=int(row["logical_seed"]),
            policy=str(row["family"]),
            replay_path=path,
            first_score=None,
        )
    )
    replay["behavior_family"] = str(row["family"])
    replay["nominal_family"] = str(row["family"])
    replay["acquisition_policy_spec"] = source.POLICY_SPECS[str(row["family"])]
    replay["o3_event_acquisition_recovery"] = True
    replay["planned_root_id"] = str(row["planned_root_id"])
    replay["acquisition_role"] = str(row["role"])
    replay["dashboard_eligible"] = False
    _write_immutable_json(
        path,
        replay,
        self_hash_field="o3_recovery_replay_payload_sha256",
    )
    stored = json.loads(path.read_text())
    return _completion_from_replay(stored, replay_path=path, row=row)


def _load_completions() -> dict[tuple[str, int], dict[str, Any]]:
    result = {}
    if COMPLETION_PATH.is_file():
        for row in _read_jsonl_metadata(COMPLETION_PATH):
            key = _completion_key(row)
            if key in result:
                raise ValueError(f"Duplicate recovery completion: {key}")
            result[key] = row
    return result


def _verify_recovery_completions(
    completions: Mapping[tuple[str, int], Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    original_roots: set[str],
) -> None:
    expected = {_completion_key(row): dict(row) for row in rows}
    roots = set()
    for key, completion in completions.items():
        row = expected.get(key)
        if row is None:
            raise ValueError(f"Recovery completion outside complement: {key}")
        path = Path(str(completion["source_replay"]))
        if path != _replay_path(row) or not path.is_file():
            raise ValueError(f"Missing recovery replay: {key}")
        if sha256_path(path) != completion["source_replay_sha256"]:
            raise ValueError(f"Recovery replay hash changed: {key}")
        replay = json.loads(path.read_text())
        restored = _completion_from_replay(replay, replay_path=path, row=row)
        if restored != dict(completion):
            raise ValueError(f"Recovery completion/replay mismatch: {key}")
        root = str(completion["root_cluster"])
        if root in roots or root in original_roots:
            raise ValueError(f"Recovery ancestry collision: {root}")
        roots.add(root)


def _guard_execution(stage: str, runtime: Mapping[str, Any]) -> dict[str, Any]:
    operations = _operational_audit(stage)
    checks = dict(operations["checks"])
    checks["active_runtime_below_18h"] = (
        float(runtime["active_runtime_seconds"]) <= ACTIVE_RUNTIME_LIMIT
    )
    if not all(checks.values()):
        raise history.AcquisitionPause(
            "HOLD_O3_ACQUISITION_RECOVERY_INTEGRITY",
            f"O3 recovery operational guard failed: {checks}",
        )
    return {**operations, "checks": checks, "passes": True}


def collect_recovery(
    rows: Sequence[Mapping[str, Any]],
    *,
    jobs: int,
    policies: Mapping[str, Any],
    original_roots: set[str],
) -> list[dict[str, Any]]:
    chunks = round_robin_chunks(rows)
    completions = _load_completions()
    _verify_recovery_completions(completions, rows, original_roots)
    attempts = _load_attempts(rows)
    runtime = _runtime_state()
    REPLAY_DIR.mkdir(exist_ok=True)
    active_policies = dict(policies)

    for chunk_index, chunk in enumerate(chunks):
        pending = [row for row in chunk if _completion_key(row) not in completions]
        if not pending:
            continue
        _guard_execution(f"before_chunk_{chunk_index}", runtime)
        for row in pending:
            key = _completion_key(row)
            lifecycles = attempts.setdefault(key, [])
            if lifecycles and len(lifecycles[-1]["statuses"]) == 1:
                if _replay_path(row).exists():
                    replay = json.loads(_replay_path(row).read_text())
                    completion = _completion_from_replay(
                        replay,
                        replay_path=_replay_path(row),
                        row=row,
                    )
                    history._append_jsonl_row(COMPLETION_PATH, completion)
                    completions[key] = completion
                    _append_attempt(
                        row,
                        attempt_number=int(lifecycles[-1]["attempt_number"]),
                        status="completed_recovered",
                        chunk_index=chunk_index,
                    )
                    lifecycles[-1]["statuses"].append("completed_recovered")
                    continue
                _append_attempt(
                    row,
                    attempt_number=int(lifecycles[-1]["attempt_number"]),
                    status="interrupted_no_replay",
                    chunk_index=chunk_index,
                )
                lifecycles[-1]["statuses"].append("interrupted_no_replay")
            attempt_number = len(lifecycles)
            _append_attempt(
                row,
                attempt_number=attempt_number,
                status="opened",
                chunk_index=chunk_index,
            )
            lifecycles.append(
                {
                    "attempt_number": attempt_number,
                    "statuses": ["opened"],
                }
            )
            family = str(row["family"])
            job = EvalJob(
                index=0,
                seed=int(row["logical_seed"]),
                starter_tile=STARTER_TILE,
                stream_ids=EvalStreamIds(
                    deck_stream_id=int(row["deck_stream_id"]),
                    slot_stream_id=int(row["slot_stream_id"]),
                    policy_stream_id=int(row["policy_stream_id"]),
                ),
            )
            started = time.perf_counter()
            try:
                outputs = list(
                    iter_eval_job_outputs(
                        policy=active_policies[family],
                        policy_name=source.POLICY_SPECS[family],
                        eval_jobs=[job],
                        max_moves=MAX_MOVES,
                        capture_replay=True,
                        jobs=jobs,
                    )
                )
            finally:
                elapsed = time.perf_counter() - started
                runtime["active_runtime_seconds"] = (
                    float(runtime["active_runtime_seconds"]) + elapsed
                )
                runtime["evaluation_batches_charged"] = (
                    int(runtime["evaluation_batches_charged"]) + 1
                )
                runtime["games_evaluated_charged"] = (
                    int(runtime["games_evaluated_charged"]) + 1
                )
                write_json(RUNTIME_PATH, runtime)
            if len(outputs) != 1 or int(outputs[0].index) != 0:
                raise ValueError("Recovery evaluator returned invalid output")
            completion = _store_output(outputs[0], row)
            if completion["root_cluster"] in original_roots:
                raise ValueError("Recovery root collides with original ancestry")
            history._append_jsonl_row(COMPLETION_PATH, completion)
            completions[key] = completion
            _append_attempt(
                row,
                attempt_number=attempt_number,
                status="completed",
                chunk_index=chunk_index,
            )
            lifecycles[-1]["statuses"].append("completed")
        runtime["chunks_completed"] = int(runtime["chunks_completed"]) + 1
        runtime["games_completed"] = len(completions)
        write_json(RUNTIME_PATH, runtime)
        _guard_execution(f"after_chunk_{chunk_index}", runtime)
        if len(completions) % 100 == 0:
            print(
                json.dumps(
                    {
                        "phase": "o3_acquisition_recovery",
                        "completed": len(completions),
                        "total": RECOVERY_ROOTS,
                        "active_runtime_seconds": runtime[
                            "active_runtime_seconds"
                        ],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    expected = {_completion_key(row) for row in rows}
    if set(completions) != expected:
        raise ValueError("Recovery did not complete exact complement")
    result = [completions[key] for key in sorted(expected)]
    _verify_recovery_completions(completions, rows, original_roots)
    return result


def recovery_attempt_audit(
    rows: Sequence[Mapping[str, Any]],
    completions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    lifecycles = _load_attempts(rows)
    expected = {_completion_key(row) for row in rows}
    completed = {_completion_key(row) for row in completions}
    status_counts = Counter()
    opened = 0
    for root_attempts in lifecycles.values():
        opened += len(root_attempts)
        for lifecycle in root_attempts:
            status_counts.update(lifecycle["statuses"])
    checks = {
        "all_roots_attempted": set(lifecycles) == expected,
        "all_roots_completed": completed == expected,
        "one_attempt_per_root": opened == len(expected),
        "all_attempts_paired": all(
            lifecycle["statuses"] in (
                ["opened", "completed"],
                ["opened", "completed_recovered"],
            )
            for root_attempts in lifecycles.values()
            for lifecycle in root_attempts
        ),
        "zero_retries": opened == RECOVERY_ROOTS,
    }
    return {
        "attempt_rows": sum(status_counts.values()),
        "attempts_opened": opened,
        "status_counts": dict(sorted(status_counts.items())),
        "file_sha256": sha256_path(ATTEMPT_PATH),
        "checks": checks,
        "passes": all(checks.values()),
    }


def build_union(
    recovery_rows: Sequence[Mapping[str, Any]],
    source_audit: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    original = _read_jsonl_metadata(source.COMPLETION_PATH)
    combined = [dict(row) for row in original] + [
        dict(row) for row in recovery_rows
    ]
    combined.sort(key=lambda row: (int(row["family_index"]), int(row["game_index"])))
    planned = {
        _completion_key(row): dict(row) for row in source.acquisition_rows()
    }
    keys = [_completion_key(row) for row in combined]
    families = Counter(str(row["family"]) for row in combined)
    roles = Counter(str(row["role"]) for row in combined)
    roots = [str(row["root_cluster"]) for row in combined]
    replay_hashes = [str(row["source_replay_sha256"]) for row in combined]
    membership = []
    role_drift = []
    stream_drift = []
    for row in combined:
        key = _completion_key(row)
        expected = planned.get(key)
        if expected is None:
            role_drift.append({"key": key, "reason": "not_in_p0"})
            continue
        if row["role"] != expected["role"]:
            role_drift.append(
                {"key": key, "actual": row["role"], "expected": expected["role"]}
            )
        for field in (
            "logical_seed",
            "deck_stream_id",
            "slot_stream_id",
            "policy_stream_id",
        ):
            if int(row[field]) != int(expected[field]):
                stream_drift.append({"key": key, "field": field})
        membership.append(
            {
                "family": row["family"],
                "family_index": int(row["family_index"]),
                "game_index": int(row["game_index"]),
                "role": row["role"],
                "planned_root_id": row["planned_root_id"],
                "logical_seed": int(row["logical_seed"]),
                "deck_stream_id": int(row["deck_stream_id"]),
                "slot_stream_id": int(row["slot_stream_id"]),
                "policy_stream_id": int(row["policy_stream_id"]),
                "root_cluster": row["root_cluster"],
                "source_replay": row["source_replay"],
                "source_replay_sha256": row["source_replay_sha256"],
            }
        )
    original_attempts = _read_jsonl_metadata(source.ATTEMPT_PATH)
    recovery_attempts = _read_jsonl_metadata(ATTEMPT_PATH)
    checks = {
        "source_revalidated": source_audit["passes"],
        "exact_20500": len(combined) == source.TOTAL_ROOTS,
        "exact_p0_membership": set(keys) == set(planned),
        "no_duplicate_keys": len(keys) == len(set(keys)),
        "exact_4100_per_family": families
        == {family: source.ROOTS_PER_FAMILY for family in FAMILY_ORDER},
        "role_counts_exact": roles == p0.ROLE_COUNTS,
        "unique_ancestries": len(roots) == len(set(roots)),
        "unique_replay_hashes": len(replay_hashes) == len(set(replay_hashes)),
        "zero_role_drift": not role_drift,
        "zero_stream_drift": not stream_drift,
        "original_attempt_rows_exact": len(original_attempts) == 37_980,
        "recovery_attempt_rows_exact": len(recovery_attempts)
        == 2 * RECOVERY_ROOTS,
        "original_top_level_hashes_unchanged": source_audit[
            "original_file_audit"
        ]["passes"],
    }
    payload = {
        "version": f"{VERSION}_union",
        "original_completion_file_sha256": sha256_path(source.COMPLETION_PATH),
        "recovery_completion_file_sha256": sha256_path(COMPLETION_PATH),
        "original_replay_byte_manifest_sha256": source_audit[
            "replay_byte_manifest_sha256"
        ],
        "membership": membership,
        "membership_sha256": canonical_json_hash(membership),
        "family_counts": dict(sorted(families.items())),
        "role_counts": dict(sorted(roles.items())),
        "unique_ancestries": len(set(roots)),
        "unique_replay_hashes": len(set(replay_hashes)),
        "role_drift": role_drift,
        "stream_drift": stream_drift,
        "checks": checks,
        "passes": all(checks.values()),
        "support_content_opened": False,
        "score_action_max_tile_outcome_fields_read": False,
    }
    return combined, payload


def _seal_terminal_error(error: Exception) -> dict[str, Any]:
    decision = (
        error.decision
        if isinstance(error, history.AcquisitionPause)
        else "HOLD_O3_ACQUISITION_RECOVERY_INTEGRITY"
    )
    return _write_immutable_json(
        RESULT_PATH,
        {
            "version": VERSION,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "decision": decision,
            "continue": "NONE",
            "hold": [
                "recovery_retry",
                "support_scan",
                "option_training",
                "mechanism_test",
                "normal_start_development",
                "confirmation",
                "promotion",
            ],
            "kill": False,
            "promote": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "marker": _artifact_identity(
                MARKER_PATH, "opened_payload_sha256"
            ),
            "completion_rows": len(_load_completions()),
            "attempt_rows": (
                len(_read_jsonl_metadata(ATTEMPT_PATH))
                if ATTEMPT_PATH.is_file()
                else 0
            ),
            "runtime": _runtime_state(),
            "output_bytes": history._directory_bytes(OUTPUT_DIR),
            "zero_forbidden_work": _zero_forbidden_work(),
            "dashboard_eligible": False,
        },
        self_hash_field="result_payload_sha256",
    )


def execute(
    *,
    out_dir: Path = OUTPUT_DIR,
    jobs: int = FROZEN_JOBS,
) -> dict[str, Any]:
    if out_dir.resolve() != OUTPUT_DIR.resolve() or jobs != FROZEN_JOBS:
        raise ValueError("Recovery execute identity mismatch")
    _load_marker()
    with execution_ownership():
        try:
            source_audit = audit_original_source()
            if not source_audit["passes"]:
                raise ValueError("Original source audit changed before execute")
            rows = derive_complement()
            complement = json.loads(COMPLEMENT_PATH.read_text())
            if (
                complement["rows_sha256"]
                != canonical_json_hash([dict(row) for row in rows])
            ):
                raise ValueError("Recovery complement changed before execute")
            collision = collision_audit(rows)
            frozen_collision = json.loads(COLLISION_PATH.read_text())
            if not collision["passes"] or (
                collision["scanned_sources_sha256"]
                != frozen_collision["scanned_sources_sha256"]
            ):
                raise ValueError("Recovery collision inventory changed")
            policies, policy_audit = source._load_policies()
            if not policy_audit["passes"]:
                raise ValueError("Recovery policy lock changed")
            _guard_execution("execute_start", _runtime_state())
            original_rows = _read_jsonl_metadata(source.COMPLETION_PATH)
            original_roots = {
                str(row["root_cluster"]) for row in original_rows
            }
            recovered = collect_recovery(
                rows,
                jobs=jobs,
                policies=policies,
                original_roots=original_roots,
            )
            _guard_execution("post_collection", _runtime_state())
            attempts = recovery_attempt_audit(rows, recovered)
            if not attempts["passes"]:
                raise ValueError("Recovery attempt ledger failed")
            source_audit = audit_original_source()
            union_rows, union_payload = build_union(recovered, source_audit)
            union = _write_immutable_json(
                UNION_PATH,
                union_payload,
                self_hash_field="union_payload_sha256",
            )
            if not union["passes"]:
                raise history.AcquisitionPause(
                    "HOLD_O3_ACQUISITION_UNION_INTEGRITY",
                    "O3 recovery virtual union failed",
                )

            candidates, support_audit = source.scan_support(union_rows)
            allocation = source.allocate_candidates(candidates)
            support = _write_immutable_json(
                SUPPORT_PATH,
                {
                    "version": f"{VERSION}_support",
                    "union": _artifact_identity(
                        UNION_PATH, "union_payload_sha256"
                    ),
                    "audit": support_audit,
                    "candidate_rows": candidates,
                    "candidate_manifest_sha256": canonical_json_hash(candidates),
                    "stage_descriptive_only": True,
                    "outcomes_compared": False,
                },
                self_hash_field="support_payload_sha256",
            )
            selected = _write_immutable_json(
                SELECTED_PATH,
                {
                    "version": f"{VERSION}_selected",
                    **allocation,
                    "labels_generated": 0,
                    "policy_outcomes_opened": False,
                },
                self_hash_field="selected_payload_sha256",
            )
            terminal_collision = collision_audit(rows)
            if not terminal_collision["passes"]:
                raise ValueError("Recovery terminal collision audit failed")
            terminal_operations = _guard_execution(
                "terminal", _runtime_state()
            )
            decision = (
                "READY_O3_OPTION_TRAINING"
                if support_audit["passes"] and allocation["passes"]
                else "HOLD_O3_DATA_OR_POWER"
            )
            return _write_immutable_json(
                RESULT_PATH,
                {
                    "version": VERSION,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "decision": decision,
                    "continue": (
                        "O3_FROZEN_OPTION_TRAINING"
                        if decision == "READY_O3_OPTION_TRAINING"
                        else "NONE"
                    ),
                    "hold": (
                        []
                        if decision == "READY_O3_OPTION_TRAINING"
                        else [
                            "option_training",
                            "mechanism_test",
                            "normal_start_development",
                            "confirmation",
                            "promotion",
                        ]
                    ),
                    "kill": False,
                    "promote": False,
                    "marker": _artifact_identity(
                        MARKER_PATH, "opened_payload_sha256"
                    ),
                    "preflight": _artifact_identity(
                        PREFLIGHT_RESULT_PATH, "result_payload_sha256"
                    ),
                    "attempt_ledger": attempts,
                    "recovery_completion_rows": len(recovered),
                    "recovery_games_by_family": dict(
                        sorted(Counter(row["family"] for row in recovered).items())
                    ),
                    "union": _artifact_identity(
                        UNION_PATH, "union_payload_sha256"
                    ),
                    "support": _artifact_identity(
                        SUPPORT_PATH, "support_payload_sha256"
                    ),
                    "selected": _artifact_identity(
                        SELECTED_PATH, "selected_payload_sha256"
                    ),
                    "support_summary": {
                        "candidate_rows": len(candidates),
                        "candidate_roots": support_audit["candidate_roots"],
                        "candidate_counts_by_role_target": support_audit[
                            "candidate_counts_by_role_target"
                        ],
                        "allocation_passes": allocation["passes"],
                        "per_role": allocation["per_role"],
                        "deficits": allocation["deficits"],
                    },
                    "runtime": _runtime_state(),
                    "output_bytes": history._directory_bytes(OUTPUT_DIR),
                    "process_guard_file_sha256": sha256_path(
                        PROCESS_GUARD_PATH
                    ),
                    "terminal_collision": {
                        "scanned_source_count": terminal_collision[
                            "scanned_source_count"
                        ],
                        "scanned_sources_sha256": terminal_collision[
                            "scanned_sources_sha256"
                        ],
                        "collisions": terminal_collision["collisions"],
                        "passes": terminal_collision["passes"],
                    },
                    "terminal_operations": terminal_operations,
                    "zero_forbidden_work": {
                        **_zero_forbidden_work(),
                        "games_generated": RECOVERY_ROOTS,
                        "streams_consumed": 4 * RECOVERY_ROOTS,
                        "support_content_opened": True,
                    },
                    "dashboard_eligible": False,
                },
                self_hash_field="result_payload_sha256",
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as error:
            if RESULT_PATH.exists():
                raise
            return _seal_terminal_error(error)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    tests = subparsers.add_parser("seal-test-evidence")
    tests.add_argument("--focused-passed", type=int, required=True)
    tests.add_argument("--regression-passed", type=int, required=True)
    tests.add_argument(
        "--test-command",
        action="append",
        dest="test_commands",
        required=True,
    )

    open_parser = subparsers.add_parser("open")
    open_parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    open_parser.add_argument("--jobs", type=int, default=FROZEN_JOBS)

    execute_parser = subparsers.add_parser("execute")
    execute_parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    execute_parser.add_argument("--jobs", type=int, default=FROZEN_JOBS)

    args = parser.parse_args()
    if args.command == "seal-test-evidence":
        result = seal_test_evidence(
            focused_passed=args.focused_passed,
            regression_passed=args.regression_passed,
            commands=args.test_commands,
        )
    elif args.command == "open":
        result = open_preflight(out_dir=args.out_dir, jobs=args.jobs)
    else:
        result = execute(out_dir=args.out_dir, jobs=args.jobs)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
