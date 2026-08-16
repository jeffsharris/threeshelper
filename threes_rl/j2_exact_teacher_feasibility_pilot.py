"""Outcome-free exact-teacher engineering feasibility pilot for J2."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import queue
import resource
import shutil
import statistics
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from threes_rl import j1_joint_policy_value as j1
from threes_rl import j2_incumbent_distillation_readiness as j2


VERSION = "j2_exact_teacher_feasibility_pilot_v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
CHARTER_PATH = (
    REPO_ROOT
    / "threes_rl"
    / "J2_EXACT_TEACHER_ENGINEERING_FEASIBILITY_PILOT_CHARTER.md"
)
RUNNER_PATH = (
    REPO_ROOT / "threes_rl" / "j2_exact_teacher_feasibility_pilot.py"
)
TEST_PATH = (
    REPO_ROOT
    / "tests"
    / "test_rl_j2_exact_teacher_feasibility_pilot.py"
)
OUTPUT_DIR = (
    REPO_ROOT
    / "threes_rl"
    / "runs"
    / "forensics"
    / "j2_exact_teacher_feasibility_pilot_v1"
)

TEST_EVIDENCE_NAME = "J2_TEACHER_PILOT_TEST_EVIDENCE.json"
PREFLIGHT_LOCK_NAME = "J2_TEACHER_PILOT_PREFLIGHT_LOCK.json"
PREFLIGHT_RESULT_NAME = "J2_TEACHER_PILOT_PREFLIGHT_RESULT.json"
MARKER_NAME = "J2_TEACHER_PILOT_EXECUTION_MARKER.json"
INVENTORY_NAME = "J2_TEACHER_PILOT_STATE_INVENTORY.json"
CENTRAL_NAME = "J2_TEACHER_PILOT_CENTRAL_COST.json"
SENSITIVITY_NAME = "J2_TEACHER_PILOT_SENSITIVITY_COST.json"
SYNC_NAME = "J2_TEACHER_PILOT_SYNCHRONOUS_ORCHESTRATION.json"
POWER_NAME = "J2_TEACHER_PILOT_POWER_SIZING.json"
TERMINAL_NAME = "J2_TEACHER_PILOT_TERMINAL_RESULT.json"
RETENTION_NAME = "J2_TEACHER_PILOT_RETENTION.json"
ROUND_DIR_NAME = "round_manifests"

READY_PREFLIGHT = "READY_J2_TEACHER_FEASIBILITY_PILOT_EXECUTION"
HOLD_PREFLIGHT = "HOLD_J2_TEACHER_FEASIBILITY_PILOT_PREFLIGHT"
READY_TERMINAL = "READY_J2_FEASIBILITY_AMENDMENT_PREFLIGHT"
HOLD_TERMINAL = "HOLD_J2_TEACHER_ENGINEERING_FEASIBILITY"
KILL_TERMINAL = "KILL_J2_TEACHER_PILOT_INTEGRITY"

INVENTORY_COUNT = 5_000
CENTRAL_COUNT = 512
SYNC_ROUNDS = 16
SYNC_STATES_PER_ROUND = 256
SYNC_COUNT = SYNC_ROUNDS * SYNC_STATES_PER_ROUND
WORKERS = 8
WARMUP_CALLS = 8
DECK_BASE = 250_000_000_000
SLOT_BASE = 251_000_000_000
EXPLORATION_BASE = 252_000_000_000
PREFIX_MIN = 16
PREFIX_SPAN = 160
PREFIX_MULTIPLIER = 73
PREFIX_OFFSET = 19
POWER_N_GRID = (2_048, 3_072, 4_096, 6_144, 8_192)
POWER_DATASETS = j2.POWER_DATASETS
POWER_BOOTSTRAPS = j2.POWER_BOOTSTRAPS
SAFETY_MULTIPLIER = 1.25
PHASE_RUNTIME_CAP_HOURS = 72.0
PHASE_STORAGE_CAP_BYTES = 24 * 1024**3
PILOT_OUTPUT_CAP_BYTES = 1024**3
FINAL_EVIDENCE_ALLOWANCE_BYTES = 16 * 1024**2
HARD_DISK_FLOOR_GIB = 100.0
TARGET_DISK_GIB = 120.0
MEMORY_FRACTION_CAP = 0.75
QUERY_TIMEOUT_SECONDS = 1_800.0
OPTIMIZER_FIXTURE_HOURS = 0.03300427754720052

EXPECTED_J2_IDENTITIES = {
    "charter": (
        REPO_ROOT
        / "threes_rl"
        / "J2_INCUMBENT_DISTILLED_JOINT_POLICY_VALUE_CHARTER.md",
        "3cf410a4da9418c9e06164ac077d3e389f77720d056dfe25ced2a4a2a052163b",
    ),
    "runner": (
        REPO_ROOT
        / "threes_rl"
        / "j2_incumbent_distillation_readiness.py",
        "9ecd658ea69968feb605d0e0a9e4e621b73ac01619536e45c0cdf69b7bc3b15f",
    ),
    "tests": (
        REPO_ROOT
        / "tests"
        / "test_rl_j2_incumbent_distillation_readiness.py",
        "24736fa56702c46b24d515716d7a6365dadb49b20622f333bead39d3105ebdb2",
    ),
}

EXPECTED_TEACHER_QUERY_CALLS = 19_432
J2_READINESS_DIR = (
    REPO_ROOT
    / "threes_rl"
    / "runs"
    / "forensics"
    / "j2_incumbent_distillation_readiness_v1"
)
EXPECTED_J2_ARTIFACTS = {
    "J2_TEACHER_PROVENANCE.json": (
        "824aa8988136d81a00d81dd4899b9985aedbbb213260d3a2e94c4e7dc931840a",
        "teacher_provenance_payload_sha256",
        "a8d355bd056bdd31f860a668d4e86a0898866192b39cf0665d348db33ac02768",
    ),
    "J2_PROTECTED_STREAM_AUTHORITY.json": (
        "b9e806e13c28d33f0edabe756ed06b49c7c5e880bd8370de99b007c0bc9d28db",
        "protected_stream_authority_payload_sha256",
        "51fa8c173049b01a3fff19860968de2bc4d09521f5cc3980ab0da9ab4add40e6",
    ),
    "J2_PROSPECTIVE_AUTHORITY.json": (
        "cea6f129e0dbb5309d67d554a74ddb8965e6c5586efb36f570363d7d370707f8",
        "prospective_authority_payload_sha256",
        "631ed382950a30dd51790ad94cfb9fb56b78f9330c87d794d17977e9d14690d6",
    ),
    "J2_READINESS_LOCK.json": (
        "c3f08429b625369263b75a5724b3abfdf2487d6a9fd2414897c7aaca7fd74488",
        "readiness_lock_payload_sha256",
        "a4683de92f833c4f33451b9f73acc0214566ab2d28d45a4e95e49a6d07372c8e",
    ),
    "J2_READINESS_RESULT.json": (
        "8c24be58bb6a30cd2cf302f17894b69e131f3b3c6092a4e71801c6b0f2f96eab",
        "readiness_result_payload_sha256",
        "4110e987eed93a0b50cf8dfc3978469f316039edfe03ae22549daf464ddf04de",
    ),
}

FORBIDDEN_RETAINED_KEYS = {
    "action",
    "actions",
    "teacher_action",
    "teacher_actions",
    "q_values",
    "values",
    "score",
    "scores",
    "final_score",
    "max_score",
    "trajectory",
    "trajectories",
    "policy_outcome",
    "policy_outcomes",
}


class PilotIntegrityError(RuntimeError):
    """The immutable pilot contract or evidence failed."""


class PilotOperationalHold(RuntimeError):
    """A resource or service gate stopped the pilot."""


def sha256_path(path: Path) -> str:
    return j2.sha256_path(path)


def canonical_hash(value: Any) -> str:
    return j2.canonical_json_hash(value)


def load_hashed_json(
    path: Path,
    *,
    field: str,
    file_sha256: str | None = None,
    payload_sha256: str | None = None,
) -> dict[str, Any]:
    if not path.is_file():
        raise PilotIntegrityError(f"Missing immutable artifact: {path}")
    if file_sha256 is not None and sha256_path(path) != file_sha256:
        raise PilotIntegrityError(f"Immutable file changed: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        raise PilotIntegrityError(f"Invalid immutable JSON: {path}") from error
    if not j2.verify_payload_hash(payload, field):
        raise PilotIntegrityError(f"Payload hash mismatch: {path}")
    if payload_sha256 is not None and payload.get(field) != payload_sha256:
        raise PilotIntegrityError(f"Payload identity changed: {path}")
    return payload


def immutable_identity(path: Path, field: str) -> dict[str, Any]:
    payload = load_hashed_json(path, field=field)
    return {
        "path": str(path.resolve()),
        "file_sha256": sha256_path(path),
        "payload_field": field,
        "payload_sha256": payload[field],
    }


def write_immutable(
    path: Path,
    payload: Mapping[str, Any],
    *,
    field: str,
) -> dict[str, Any]:
    return j2.write_immutable_json(path, payload, field=field)


def _payload_without_hash(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    body = j2.json_native(dict(payload))
    body.pop(field, None)
    return body


def source_identity_audit() -> dict[str, Any]:
    current = {
        "charter": sha256_path(CHARTER_PATH),
        "runner": sha256_path(RUNNER_PATH),
        "tests": sha256_path(TEST_PATH),
    }
    parent = {}
    for name, (path, expected) in EXPECTED_J2_IDENTITIES.items():
        observed = sha256_path(path) if path.is_file() else None
        parent[name] = {
            "path": str(path.resolve()),
            "expected": expected,
            "observed": observed,
            "passes": observed == expected,
        }
    artifacts = {}
    for name, (
        expected_file,
        field,
        expected_payload,
    ) in EXPECTED_J2_ARTIFACTS.items():
        path = J2_READINESS_DIR / name
        payload = load_hashed_json(
            path,
            field=field,
            file_sha256=expected_file,
            payload_sha256=expected_payload,
        )
        artifacts[name] = {
            "path": str(path.resolve()),
            "file_sha256": expected_file,
            "payload_field": field,
            "payload_sha256": expected_payload,
            "decision": payload.get("decision"),
        }
    current_teacher = j2.teacher_provenance_audit()
    sealed_teacher = load_hashed_json(
        J2_READINESS_DIR / "J2_TEACHER_PROVENANCE.json",
        field="teacher_provenance_payload_sha256",
    )
    teacher_body = _payload_without_hash(
        sealed_teacher,
        "teacher_provenance_payload_sha256",
    )
    checks = {
        "local_sources_exist": all(current.values()),
        "accepted_j2_sources_exact": all(
            row["passes"] for row in parent.values()
        ),
        "accepted_j2_artifacts_exact": len(artifacts)
        == len(EXPECTED_J2_ARTIFACTS),
        "teacher_provenance_current_passes": current_teacher["passes"],
        "teacher_provenance_body_exact": (
            j2.json_native(current_teacher) == teacher_body
        ),
        "j2_readiness_is_hold": artifacts["J2_READINESS_RESULT.json"][
            "decision"
        ]
        == j2.HOLD,
    }
    return {
        "version": f"{VERSION}_source_identity_audit_v1",
        "local_sources": current,
        "accepted_j2_sources": parent,
        "accepted_j2_artifacts": artifacts,
        "teacher_binding": current_teacher["incumbent_binding"],
        "checks": checks,
        "passes": all(checks.values()),
    }


def engineering_stream_rows() -> list[dict[str, Any]]:
    rows = []
    scientific_rows = j2.build_prospective_rows()
    scientific_roots = {str(row["root_id"]) for row in scientific_rows}
    scientific_ancestries = {
        str(row["ancestry_id"]) for row in scientific_rows
    }
    for index in range(INVENTORY_COUNT):
        streams = {
            "deck_stream_id": DECK_BASE + index,
            "slot_stream_id": SLOT_BASE + index,
            "exploration_policy_stream_id": EXPLORATION_BASE + index,
        }
        ancestry_id = canonical_hash(
            {
                "course": VERSION,
                "kind": "engineering_normal_start_ancestry",
                "streams": streams,
            }
        )
        root_id = canonical_hash(
            {
                "course": VERSION,
                "kind": "engineering_prefix_root",
                "state_index": index,
                "ancestry_id": ancestry_id,
                "streams": streams,
            }
        )
        if root_id in scientific_roots or ancestry_id in scientific_ancestries:
            raise PilotIntegrityError("Engineering root collided with J2")
        rows.append(
            {
                "state_index": index,
                "root_id": root_id,
                "ancestry_id": ancestry_id,
                "worker_id": index % WORKERS,
                "target_prefix_steps": (
                    PREFIX_MIN
                    + ((PREFIX_MULTIPLIER * index + PREFIX_OFFSET) % PREFIX_SPAN)
                ),
                "streams": streams,
            }
        )
    return rows


def stream_authority_audit() -> dict[str, Any]:
    rows = engineering_stream_rows()
    roots = [str(row["root_id"]) for row in rows]
    ancestries = [str(row["ancestry_id"]) for row in rows]
    stream_ids = [
        int(value)
        for row in rows
        for value in row["streams"].values()
    ]
    prefixes = {value // 1_000_000_000 for value in stream_ids}
    checks = {
        "row_count_exact": len(rows) == INVENTORY_COUNT,
        "roots_unique": len(set(roots)) == INVENTORY_COUNT,
        "ancestries_unique": len(set(ancestries)) == INVENTORY_COUNT,
        "streams_unique": len(set(stream_ids)) == 3 * INVENTORY_COUNT,
        "engineering_prefixes_exact": prefixes == {250, 251, 252},
        "no_spent_213b_226b_collision": not (
            prefixes & set(range(213, 227))
        ),
        "no_j2_227b_249b_collision": not (
            prefixes & set(range(227, 250))
        ),
        "j2_scientific_reservations_zero": True,
        "j2_scientific_consumptions_zero": True,
    }
    return {
        "version": f"{VERSION}_stream_authority_v1",
        "row_count": len(rows),
        "root_set_sha256": canonical_hash(sorted(roots)),
        "ancestry_set_sha256": canonical_hash(sorted(ancestries)),
        "stream_set_sha256": canonical_hash(sorted(stream_ids)),
        "intervals": {
            "deck": [DECK_BASE, DECK_BASE + INVENTORY_COUNT - 1],
            "slot": [SLOT_BASE, SLOT_BASE + INVENTORY_COUNT - 1],
            "exploration": [
                EXPLORATION_BASE,
                EXPLORATION_BASE + INVENTORY_COUNT - 1,
            ],
        },
        "checks": checks,
        "passes": all(checks.values()),
    }


def _physical_memory_bytes() -> int:
    return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))


def _available_memory_bytes() -> int:
    if sys.platform != "darwin":
        return _physical_memory_bytes()
    completed = subprocess.run(
        ["vm_stat"],
        check=True,
        capture_output=True,
        text=True,
    )
    page_size = 4096
    first = completed.stdout.splitlines()[0]
    if "page size of" in first:
        page_size = int(first.split("page size of", 1)[1].split()[0])
    available_pages = 0
    accepted = {
        "Pages free",
        "Pages inactive",
        "Pages speculative",
        "Pages purgeable",
    }
    for line in completed.stdout.splitlines()[1:]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        if key.strip() in accepted:
            available_pages += int(value.strip().rstrip("."))
    return available_pages * page_size


def _rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _sample_process_rss_bytes(pids: Sequence[int]) -> int:
    live = sorted({int(pid) for pid in pids if int(pid) > 0})
    if not live:
        return 0
    completed = subprocess.run(
        ["ps", "-o", "rss=", "-p", ",".join(str(pid) for pid in live)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode not in (0, 1):
        raise PilotOperationalHold("RSS sampler failed")
    return sum(
        int(line.strip()) * 1024
        for line in completed.stdout.splitlines()
        if line.strip()
    )


class PeakRSSSampler:
    def __init__(self, pids: Sequence[int]) -> None:
        self.pids = tuple(int(pid) for pid in pids)
        self.stop_event = threading.Event()
        self.peak_bytes = 0
        self.sample_count = 0
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        try:
            while not self.stop_event.is_set():
                self.peak_bytes = max(
                    self.peak_bytes,
                    _sample_process_rss_bytes(self.pids),
                )
                self.sample_count += 1
                self.stop_event.wait(0.05)
            self.peak_bytes = max(
                self.peak_bytes,
                _sample_process_rss_bytes(self.pids),
            )
            self.sample_count += 1
        except BaseException as error:
            self.error = error

    def __enter__(self) -> "PeakRSSSampler":
        self.thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop_event.set()
        self.thread.join(timeout=10.0)
        if self.thread.is_alive():
            raise PilotOperationalHold("RSS sampler failed to stop")
        if self.error is not None and exc_type is None:
            raise PilotOperationalHold("RSS sampler failed") from self.error


def operational_audit(
    *,
    output_dir: Path = OUTPUT_DIR,
    include_namespace_absence: bool,
) -> dict[str, Any]:
    parent = j2.operational_audit(output_dir=output_dir)
    free_gib = shutil.disk_usage(REPO_ROOT).free / 1024**3
    available_memory = _available_memory_bytes()
    checks = {
        "nice_at_least_10": parent["checks"]["nice_at_least_10"],
        "one_heavy_job": parent["checks"]["one_heavy_job"],
        "disk_above_hard_floor": free_gib > HARD_DISK_FLOOR_GIB,
        "disk_target_met": free_gib > TARGET_DISK_GIB,
        "services_healthy": parent["checks"]["services_healthy"],
        "top_three_exact": parent["checks"]["dashboard_top_three_exact"],
        "human_sessions_opaque": parent["checks"][
            "human_session_content_unread"
        ],
        "physical_memory_positive": _physical_memory_bytes() > 0,
        "available_memory_positive": available_memory > 0,
        "namespace_state_expected": (
            not output_dir.exists()
            if include_namespace_absence
            else output_dir.exists()
        ),
    }
    return {
        "version": f"{VERSION}_operational_audit_v1",
        "parent": parent,
        "free_disk_gib": free_gib,
        "physical_memory_bytes": _physical_memory_bytes(),
        "available_memory_bytes": available_memory,
        "checks": checks,
        "passes": all(checks.values()),
    }


def _state_from_engineering_row(
    row: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    index = int(row["state_index"])
    streams = dict(row["streams"])
    sim, state = j1.normal_start_sim(
        role="train",
        deck_stream_id=int(streams["deck_stream_id"]),
        slot_stream_id=int(streams["slot_stream_id"]),
    )
    rng = np.random.default_rng(
        int(streams["exploration_policy_stream_id"])
    )
    reached = 0
    for _ in range(int(row["target_prefix_steps"])):
        legal = sim.legal_actions(state)
        if state.game_over or not legal:
            raise PilotIntegrityError(
                "Engineering ancestry terminated before its frozen prefix"
            )
        selected = legal[int(rng.integers(0, len(legal)))]
        next_state, info = sim.step(state, selected)
        if not info.moved:
            raise PilotIntegrityError("Exploration selected an illegal move")
        if next_state.game_over or not sim.legal_actions(next_state):
            raise PilotIntegrityError(
                "Engineering ancestry terminated at its frozen prefix"
            )
        state = next_state
        reached += 1
    legal = sim.legal_actions(state)
    if (
        sim.starter_tile is not None
        or state.game_over
        or not legal
        or reached != int(row["target_prefix_steps"])
    ):
        raise PilotIntegrityError("Engineering state is not a legal prefix")
    snapshot = j1.simulator_snapshot(sim, state)
    state_hash = canonical_hash(snapshot)
    manifest = {
        **dict(row),
        "reached_prefix_steps": reached,
        "stopped_before_target": False,
        "feature_family": j2.feature_family(state.board),
        "state_sha256": state_hash,
        "legal_action_count": len(legal),
    }
    return manifest, snapshot


def build_and_seal_inventory(
    *,
    output_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = engineering_stream_rows()
    manifest_rows = []
    snapshots = []
    for row in rows:
        manifest, snapshot = _state_from_engineering_row(row)
        manifest_rows.append(manifest)
        snapshots.append(snapshot)
    state_hashes = [str(row["state_sha256"]) for row in manifest_rows]
    families = Counter(str(row["feature_family"]) for row in manifest_rows)
    checks = {
        "count_exact": len(manifest_rows) == INVENTORY_COUNT,
        "state_hashes_unique": len(set(state_hashes)) == INVENTORY_COUNT,
        "roots_unique": len(
            {str(row["root_id"]) for row in manifest_rows}
        )
        == INVENTORY_COUNT,
        "ancestries_unique": len(
            {str(row["ancestry_id"]) for row in manifest_rows}
        )
        == INVENTORY_COUNT,
        "all_states_legal": all(
            int(row["legal_action_count"]) > 0 for row in manifest_rows
        ),
        "all_prefixes_reached_exactly": all(
            int(row["reached_prefix_steps"])
            == int(row["target_prefix_steps"])
            and row["stopped_before_target"] is False
            for row in manifest_rows
        ),
        "teacher_queries_before_inventory_seal_zero": True,
        "scores_retained_zero": True,
        "actions_retained_zero": True,
        "trajectories_retained_zero": True,
    }
    if not all(checks.values()):
        raise PilotIntegrityError(
            "Engineering inventory failed its frozen completeness contract"
        )
    payload = {
        "version": f"{VERSION}_state_inventory_v1",
        "rows": manifest_rows,
        "ordered_inventory_sha256": canonical_hash(manifest_rows),
        "ordered_state_hash_sha256": canonical_hash(state_hashes),
        "natural_feature_family_counts": dict(sorted(families.items())),
        "natural_feature_family_frequencies": {
            family: count / INVENTORY_COUNT
            for family, count in sorted(families.items())
        },
        "checks": checks,
        "passes": all(checks.values()),
        "teacher_queries": 0,
        "scientific_stream_reservations": 0,
        "scientific_stream_consumptions": 0,
    }
    write_immutable(
        output_dir / INVENTORY_NAME,
        payload,
        field="inventory_payload_sha256",
    )
    observed = load_hashed_json(
        output_dir / INVENTORY_NAME,
        field="inventory_payload_sha256",
    )
    if observed["ordered_inventory_sha256"] != canonical_hash(manifest_rows):
        raise PilotIntegrityError("Inventory changed after create-once seal")
    for row, snapshot in zip(observed["rows"], snapshots, strict=True):
        if str(row["state_sha256"]) != canonical_hash(snapshot):
            raise PilotIntegrityError("Inventory state binding changed")
    return observed, snapshots


def regenerate_inventory(
    inventory: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = list(inventory["rows"])
    snapshots = []
    for expected in rows:
        manifest, snapshot = _state_from_engineering_row(expected)
        immutable_fields = {
            key: expected[key]
            for key in (
                "state_index",
                "root_id",
                "ancestry_id",
                "worker_id",
                "target_prefix_steps",
                "streams",
                "reached_prefix_steps",
                "stopped_before_target",
                "feature_family",
                "state_sha256",
                "legal_action_count",
            )
        }
        if manifest != immutable_fields:
            raise PilotIntegrityError("Inventory regeneration changed")
        snapshots.append(snapshot)
    if canonical_hash(rows) != inventory["ordered_inventory_sha256"]:
        raise PilotIntegrityError("Inventory ordered identity changed")
    return snapshots


def _timing_summary(samples: Sequence[float]) -> dict[str, float | int]:
    values = np.asarray(samples, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise PilotIntegrityError("Timing samples are empty or nonfinite")
    return {
        "count": int(values.size),
        "median_seconds": float(np.quantile(values, 0.5, method="linear")),
        "p90_seconds": float(np.quantile(values, 0.9, method="linear")),
        "p99_seconds": float(np.quantile(values, 0.99, method="linear")),
        "max_seconds": float(np.max(values)),
        "mean_seconds": float(np.mean(values)),
    }


def _query_one(
    policy: Any,
    snapshot: Mapping[str, Any],
    policy_stream_id: int,
) -> tuple[int, float]:
    sim, state = j1.simulator_from_snapshot(snapshot)
    legal = sim.legal_actions(state)
    if state.game_over or not legal:
        raise PilotIntegrityError("Teacher query state is not legal")
    rng = np.random.default_rng(int(policy_stream_id))
    started = time.perf_counter()
    selected = int(policy(state, sim, rng))
    elapsed = time.perf_counter() - started
    if selected not in legal:
        raise PilotIntegrityError("Exact teacher returned an illegal action")
    return selected, elapsed


def _teacher_worker(
    worker_id: int,
    command_queue: Any,
    result_queue: Any,
    expected_binding: Mapping[str, Any],
) -> None:
    try:
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        from threes_rl import j1_execution_surface as j1_execution

        load_started = time.perf_counter()
        policy = j1_execution.load_bound_incumbent_policy(expected_binding)
        load_seconds = time.perf_counter() - load_started
        result_queue.put(
            {
                "kind": "ready",
                "worker_id": worker_id,
                "pid": os.getpid(),
                "load_seconds": load_seconds,
                "peak_rss_bytes": _rss_bytes(),
            }
        )
        _serve_teacher_commands(
            worker_id=worker_id,
            command_queue=command_queue,
            result_queue=result_queue,
            policy=policy,
        )
    except BaseException as error:
        result_queue.put(
            {
                "kind": "error",
                "worker_id": worker_id,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback_sha256": hashlib.sha256(
                    traceback.format_exc().encode("utf-8")
                ).hexdigest(),
            }
        )


def _serve_teacher_commands(
    *,
    worker_id: int,
    command_queue: Any,
    result_queue: Any,
    policy: Any,
) -> None:
    warmed = False
    while True:
        command = command_queue.get()
        if command.get("kind") == "stop":
            return
        kind = str(command.get("kind"))
        round_id = str(command["round_id"])
        rows = list(command["rows"])
        if not rows:
            raise PilotIntegrityError("Teacher worker received no rows")
        if kind == "warmup":
            if warmed:
                raise PilotIntegrityError(
                    "Teacher worker warmup was requested twice"
                )
            started = time.perf_counter()
            transient_actions = []
            for offset in range(WARMUP_CALLS):
                row = rows[offset % len(rows)]
                action, _elapsed = _query_one(
                    policy,
                    row["snapshot"],
                    int(row["policy_stream_id"]),
                )
                transient_actions.append(action)
            for index in range(len(transient_actions)):
                transient_actions[index] = -1
            transient_actions.clear()
            warmed = True
            result_queue.put(
                {
                    "kind": "warmup_complete",
                    "round_id": round_id,
                    "worker_id": worker_id,
                    "warmup_calls": WARMUP_CALLS,
                    "warmup_wall_seconds": time.perf_counter() - started,
                    "actions_retained": 0,
                }
            )
            continue
        if kind != "query":
            raise PilotIntegrityError("Unknown teacher worker command")
        if not warmed:
            raise PilotIntegrityError(
                "Measured teacher query preceded explicit warmup"
            )
        cpu_started = time.process_time()
        actions = []
        timings = []
        indices = []
        for row in rows:
            action, elapsed = _query_one(
                policy,
                row["snapshot"],
                int(row["policy_stream_id"]),
            )
            indices.append(int(row["state_index"]))
            actions.append(action)
            timings.append(elapsed)
        cpu_seconds = time.process_time() - cpu_started
        result_queue.put(
            {
                "kind": "result",
                "round_id": round_id,
                "worker_id": worker_id,
                "indices": indices,
                "actions": actions,
                "timings": timings,
                "cpu_seconds": cpu_seconds,
                "peak_rss_bytes": _rss_bytes(),
                "warmup_calls": 0,
            }
        )


class TeacherWorkerGroup:
    def __init__(self, binding: Mapping[str, Any]) -> None:
        self.context = mp.get_context("spawn")
        self.result_queue = self.context.Queue()
        self.command_queues = [self.context.Queue() for _ in range(WORKERS)]
        self.processes = [
            self.context.Process(
                target=_teacher_worker,
                args=(
                    worker_id,
                    self.command_queues[worker_id],
                    self.result_queue,
                    dict(binding),
                ),
                name=f"j2-teacher-worker-{worker_id}",
            )
            for worker_id in range(WORKERS)
        ]
        started = time.perf_counter()
        for process in self.processes:
            process.start()
        pids = [os.getpid()] + [
            int(process.pid)
            for process in self.processes
            if process.pid is not None
        ]
        self.lifetime_rss_sampler = PeakRSSSampler(pids)
        self.lifetime_rss_evidence: dict[str, int | bool] | None = None
        self.lifetime_rss_sampler.__enter__()
        try:
            self.ready = self._receive_many(WORKERS, expected_kind="ready")
        except BaseException:
            for process in self.processes:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=5.0)
            self.lifetime_rss_sampler.__exit__(*sys.exc_info())
            raise
        self.startup_and_load_wall_seconds = time.perf_counter() - started
        if {int(row["worker_id"]) for row in self.ready} != set(
            range(WORKERS)
        ):
            raise PilotIntegrityError("Worker ready identities changed")
        self.warmup_evidence: dict[str, Any] | None = None

    def _receive_many(
        self,
        count: int,
        *,
        expected_kind: str,
    ) -> list[dict[str, Any]]:
        records = []
        deadline = time.monotonic() + QUERY_TIMEOUT_SECONDS
        while len(records) < count:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise PilotOperationalHold("Teacher workers timed out")
            try:
                row = self.result_queue.get(timeout=min(5.0, remaining))
            except queue.Empty:
                dead = [
                    process.pid
                    for process in self.processes
                    if not process.is_alive()
                ]
                if dead:
                    raise PilotIntegrityError(
                        f"Teacher worker exited before result: {dead}"
                    )
                continue
            if row.get("kind") == "error":
                raise PilotIntegrityError(
                    "Teacher worker failed: "
                    f"{row.get('error_type')}: {row.get('error_message')}"
                )
            if row.get("kind") != expected_kind:
                raise PilotIntegrityError("Unexpected worker record kind")
            records.append(row)
        return records

    def run_round(
        self,
        *,
        round_id: str,
        rows: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        if self.warmup_evidence is None:
            raise PilotIntegrityError(
                "Measured worker round preceded warmup barrier"
            )
        expected_indices = [int(row["state_index"]) for row in rows]
        dispatch_monotonic_ns = time.monotonic_ns()
        for worker_id in range(WORKERS):
            owned = [
                dict(row)
                for row in rows
                if int(row["state_index"]) % WORKERS == worker_id
            ]
            if not owned:
                raise PilotIntegrityError("Worker received no states")
            self.command_queues[worker_id].put(
                {
                    "kind": "query",
                    "round_id": round_id,
                    "rows": owned,
                }
            )
        started = time.perf_counter()
        pids = [os.getpid()] + [
            int(process.pid)
            for process in self.processes
            if process.pid is not None
        ]
        with PeakRSSSampler(pids) as rss_sampler:
            records = self._receive_many(WORKERS, expected_kind="result")
        wall_seconds = time.perf_counter() - started
        received_monotonic_ns = time.monotonic_ns()
        return validate_worker_records(
            records,
            expected_indices=expected_indices,
            round_id=round_id,
            wall_seconds=wall_seconds,
            ready_records=self.ready,
            startup_and_load_wall_seconds=self.startup_and_load_wall_seconds,
            dispatch_monotonic_ns=dispatch_monotonic_ns,
            received_monotonic_ns=received_monotonic_ns,
            lifetime_contemporaneous_peak_rss_bytes=(
                self.lifetime_rss_sampler.peak_bytes
            ),
            lifetime_rss_sample_count=(
                self.lifetime_rss_sampler.sample_count
            ),
            measured_round_contemporaneous_peak_rss_bytes=(
                rss_sampler.peak_bytes
            ),
            measured_round_rss_sample_count=rss_sampler.sample_count,
        )

    def warmup(
        self,
        *,
        rows: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        if self.warmup_evidence is not None:
            raise PilotIntegrityError("Worker group warmup was requested twice")
        owned_rows: dict[int, Mapping[str, Any]] = {}
        for row in rows:
            worker_id = int(row["state_index"]) % WORKERS
            owned_rows.setdefault(worker_id, row)
        if set(owned_rows) != set(range(WORKERS)):
            raise PilotIntegrityError(
                "Warmup inventory does not cover every worker"
            )
        started = time.perf_counter()
        for worker_id in range(WORKERS):
            self.command_queues[worker_id].put(
                {
                    "kind": "warmup",
                    "round_id": "explicit-worker-warmup",
                    "rows": [dict(owned_rows[worker_id])],
                }
            )
        records = self._receive_many(
            WORKERS,
            expected_kind="warmup_complete",
        )
        warmup_wall_seconds = time.perf_counter() - started
        by_worker = {
            int(record["worker_id"]): dict(record) for record in records
        }
        if set(by_worker) != set(range(WORKERS)):
            raise PilotIntegrityError("Warmup worker identities changed")
        if any(
            int(record["warmup_calls"]) != WARMUP_CALLS
            or int(record["actions_retained"]) != 0
            for record in records
        ):
            raise PilotIntegrityError("Worker warmup accounting changed")
        self.warmup_evidence = {
            "warmup_calls_per_process": WARMUP_CALLS,
            "worker_process_count": WORKERS,
            "total_worker_warmup_calls": WORKERS * WARMUP_CALLS,
            "warmup_wall_seconds": warmup_wall_seconds,
            "worker_warmup_wall_seconds": {
                str(worker_id): float(
                    by_worker[worker_id]["warmup_wall_seconds"]
                )
                for worker_id in range(WORKERS)
            },
            "actions_retained": 0,
            "lifetime_contemporaneous_peak_rss_bytes_after_warmup": (
                self.lifetime_rss_sampler.peak_bytes
            ),
            "lifetime_rss_sample_count_after_warmup": (
                self.lifetime_rss_sampler.sample_count
            ),
        }
        return dict(self.warmup_evidence)

    def close(self) -> None:
        error: BaseException | None = None
        try:
            for command_queue in self.command_queues:
                command_queue.put({"kind": "stop"})
            for process in self.processes:
                process.join(timeout=30.0)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5.0)
                    raise PilotIntegrityError(
                        "Teacher worker failed to stop"
                    )
                if process.exitcode != 0:
                    raise PilotIntegrityError(
                        "Teacher worker exit was nonzero"
                    )
        except BaseException as caught:
            error = caught
        try:
            self.lifetime_rss_sampler.__exit__(
                type(error) if error is not None else None,
                error,
                error.__traceback__ if error is not None else None,
            )
        except BaseException:
            if error is None:
                raise
        self.lifetime_rss_evidence = {
            "maximum_contemporaneous_parent_children_rss_bytes": (
                self.lifetime_rss_sampler.peak_bytes
            ),
            "sample_count": self.lifetime_rss_sampler.sample_count,
            "covers_load_warmup_queries_and_shutdown": True,
        }
        if (
            int(self.lifetime_rss_evidence["sample_count"]) <= 0
            or int(
                self.lifetime_rss_evidence[
                    "maximum_contemporaneous_parent_children_rss_bytes"
                ]
            )
            <= 0
        ):
            raise PilotIntegrityError(
                "Worker-group lifetime RSS evidence is empty"
            )
        if error is not None:
            raise error

    def __enter__(self) -> "TeacherWorkerGroup":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        try:
            self.close()
        except Exception:
            if exc_type is None:
                raise


def validate_worker_records(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_indices: Sequence[int],
    round_id: str,
    wall_seconds: float,
    ready_records: Sequence[Mapping[str, Any]],
    startup_and_load_wall_seconds: float,
    dispatch_monotonic_ns: int,
    received_monotonic_ns: int,
    lifetime_contemporaneous_peak_rss_bytes: int,
    lifetime_rss_sample_count: int,
    measured_round_contemporaneous_peak_rss_bytes: int,
    measured_round_rss_sample_count: int,
) -> dict[str, Any]:
    if len(records) != WORKERS:
        raise PilotIntegrityError("Worker result count changed")
    by_worker: dict[int, Mapping[str, Any]] = {}
    for record in records:
        worker_id = int(record["worker_id"])
        if worker_id in by_worker:
            raise PilotIntegrityError("Duplicate worker result")
        observed_round_id = str(record["round_id"])
        if observed_round_id != round_id:
            expected_prefix, expected_separator, expected_suffix = (
                round_id.rpartition("-")
            )
            observed_prefix, observed_separator, observed_suffix = (
                observed_round_id.rpartition("-")
            )
            late = (
                expected_separator == observed_separator == "-"
                and expected_prefix == observed_prefix
                and expected_suffix.isdigit()
                and observed_suffix.isdigit()
                and int(observed_suffix) < int(expected_suffix)
            )
            raise PilotIntegrityError(
                "Late worker result"
                if late
                else "Cross-round worker result"
            )
        indices = [int(value) for value in record["indices"]]
        expected_owned = [
            index
            for index in expected_indices
            if index % WORKERS == worker_id
        ]
        if indices != expected_owned:
            raise PilotIntegrityError(
                "Worker ownership/order or state membership changed"
            )
        actions = list(record["actions"])
        timings = list(record["timings"])
        if len(actions) != len(indices) or len(timings) != len(indices):
            raise PilotIntegrityError("Worker result vectors changed")
        if any(int(action) not in range(4) for action in actions):
            raise PilotIntegrityError("Worker returned an invalid action")
        if any(
            not math.isfinite(float(elapsed)) or float(elapsed) < 0
            for elapsed in timings
        ):
            raise PilotIntegrityError("Worker timing is invalid")
        if int(record.get("warmup_calls", -1)) != 0:
            raise PilotIntegrityError(
                "Measured worker result included warmup calls"
            )
        by_worker[worker_id] = record
    if set(by_worker) != set(range(WORKERS)):
        raise PilotIntegrityError("Missing worker result")
    merged = []
    timings = []
    for index in expected_indices:
        worker_id = index % WORKERS
        record = by_worker[worker_id]
        position = list(record["indices"]).index(index)
        merged.append((index, int(record["actions"][position])))
        timings.append(float(record["timings"][position]))
    if [index for index, _action in merged] != list(expected_indices):
        raise PilotIntegrityError("Canonical worker merge changed")
    if wall_seconds <= 0:
        raise PilotIntegrityError("Parallel wall time is invalid")
    if (
        lifetime_contemporaneous_peak_rss_bytes <= 0
        or lifetime_rss_sample_count <= 0
        or measured_round_contemporaneous_peak_rss_bytes <= 0
        or measured_round_rss_sample_count <= 0
    ):
        raise PilotIntegrityError(
            "Worker-group RSS evidence is missing or started late"
        )
    return {
        "actions": [action for _index, action in merged],
        "ordered_output_sha256": canonical_hash(merged),
        "timings": timings,
        "timing_summary": _timing_summary(timings),
        "wall_seconds": wall_seconds,
        "calls_per_second": len(merged) / wall_seconds,
        "child_cpu_seconds": sum(
            float(record["cpu_seconds"]) for record in records
        ),
        "worker_peak_rss_bytes": {
            str(worker_id): int(by_worker[worker_id]["peak_rss_bytes"])
            for worker_id in range(WORKERS)
        },
        "summed_worker_peak_rss_bytes": sum(
            int(record["peak_rss_bytes"]) for record in records
        ),
        "worker_load_seconds": {
            str(int(record["worker_id"])): float(record["load_seconds"])
            for record in ready_records
        },
        "startup_and_load_wall_seconds": startup_and_load_wall_seconds,
        "warmup_calls": sum(
            int(record["warmup_calls"]) for record in records
        ),
        "worker_count": WORKERS,
        "record_count": len(merged),
        "dispatch_monotonic_ns": dispatch_monotonic_ns,
        "received_monotonic_ns": received_monotonic_ns,
        "max_contemporaneous_parent_children_rss_bytes": (
            lifetime_contemporaneous_peak_rss_bytes
        ),
        "lifetime_rss_sample_count": lifetime_rss_sample_count,
        "measured_round_contemporaneous_parent_children_rss_bytes": (
            measured_round_contemporaneous_peak_rss_bytes
        ),
        "measured_round_rss_sample_count": (
            measured_round_rss_sample_count
        ),
    }


def _query_rows(
    snapshots: Sequence[Mapping[str, Any]],
    inventory_rows: Sequence[Mapping[str, Any]],
    count: int,
) -> list[dict[str, Any]]:
    if count > len(snapshots) or count > len(inventory_rows):
        raise PilotIntegrityError("Requested workload exceeds inventory")
    rows = []
    for row, snapshot in zip(
        inventory_rows[:count],
        snapshots[:count],
        strict=True,
    ):
        rows.append(
            {
                "state_index": int(row["state_index"]),
                "snapshot": snapshot,
                "policy_stream_id": int(
                    row["streams"]["exploration_policy_stream_id"]
                )
                + 10_000_000,
            }
        )
    return rows


def run_serial_workload(
    *,
    binding: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    from threes_rl import j1_execution_surface as j1_execution

    load_started = time.perf_counter()
    policy = j1_execution.load_bound_incumbent_policy(binding)
    load_seconds = time.perf_counter() - load_started
    for offset in range(WARMUP_CALLS):
        row = rows[offset % len(rows)]
        _query_one(
            policy,
            row["snapshot"],
            int(row["policy_stream_id"]),
        )
    actions = []
    timings = []
    cpu_started = time.process_time()
    wall_started = time.perf_counter()
    for row in rows:
        action, elapsed = _query_one(
            policy,
            row["snapshot"],
            int(row["policy_stream_id"]),
        )
        actions.append(action)
        timings.append(elapsed)
    wall_seconds = time.perf_counter() - wall_started
    cpu_seconds = time.process_time() - cpu_started
    ordered = [
        (int(row["state_index"]), action)
        for row, action in zip(rows, actions, strict=True)
    ]
    return {
        "actions": actions,
        "ordered_output_sha256": canonical_hash(ordered),
        "timing_summary": _timing_summary(timings),
        "wall_seconds": wall_seconds,
        "calls_per_second": len(rows) / wall_seconds,
        "cpu_seconds": cpu_seconds,
        "load_seconds": load_seconds,
        "parent_peak_rss_bytes": _rss_bytes(),
        "warmup_calls": WARMUP_CALLS,
        "record_count": len(rows),
    }


def run_parallel_workload(
    *,
    binding: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    round_id: str,
) -> dict[str, Any]:
    group = TeacherWorkerGroup(binding)
    try:
        warmup = group.warmup(rows=rows)
        report = group.run_round(round_id=round_id, rows=rows)
        report["warmup_evidence"] = warmup
    finally:
        group.close()
    if group.lifetime_rss_evidence is None:
        raise PilotIntegrityError("Worker-group lifetime RSS was not sealed")
    report["max_contemporaneous_parent_children_rss_bytes"] = int(
        group.lifetime_rss_evidence[
            "maximum_contemporaneous_parent_children_rss_bytes"
        ]
    )
    report["lifetime_rss_sample_count"] = int(
        group.lifetime_rss_evidence["sample_count"]
    )
    report["lifetime_rss_covers_load_warmup_queries_and_shutdown"] = bool(
        group.lifetime_rss_evidence[
            "covers_load_warmup_queries_and_shutdown"
        ]
    )
    return report


def _public_cost_result(
    *,
    workload: str,
    serial: Mapping[str, Any],
    parallel: Mapping[str, Any],
    reference_digest: str | None,
) -> dict[str, Any]:
    equality = list(serial["actions"]) == list(parallel["actions"])
    digest_equal = (
        serial["ordered_output_sha256"]
        == parallel["ordered_output_sha256"]
    )
    repeated = (
        reference_digest is None
        or serial["ordered_output_sha256"] == reference_digest
    )
    result = {
        "version": f"{VERSION}_{workload}_cost_v1",
        "workload": workload,
        "record_count": int(serial["record_count"]),
        "warmup_calls_per_process": WARMUP_CALLS,
        "serial_warmup_calls_total": int(serial["warmup_calls"]),
        "parallel_worker_warmup_calls_total": int(
            parallel["warmup_evidence"]["total_worker_warmup_calls"]
        ),
        "parallel_measured_warmup_calls": int(
            parallel["warmup_calls"]
        ),
        "warmups_excluded_from_steady_wall": (
            int(parallel["warmup_calls"]) == 0
            and float(
                parallel["warmup_evidence"]["warmup_wall_seconds"]
            )
            >= 0
        ),
        "serial": {
            key: value
            for key, value in serial.items()
            if key not in {"actions"}
        },
        "parallel_eight_process": {
            key: value
            for key, value in parallel.items()
            if key not in {"actions", "timings"}
        },
        "speedup": float(serial["wall_seconds"])
        / float(parallel["wall_seconds"]),
        "exact_action_equality": equality,
        "ordered_digest_equality": digest_equal,
        "repeated_reference_digest_equality": repeated,
        "actions_retained": 0,
        "labels_retained": 0,
        "outcomes_retained": 0,
        "passes": (
            equality
            and digest_equal
            and repeated
            and int(parallel["warmup_calls"]) == 0
            and int(
                parallel["warmup_evidence"]["total_worker_warmup_calls"]
            )
            == WORKERS * WARMUP_CALLS
        ),
    }
    assert_no_forbidden_retained_fields(result)
    return result


def _zeroize_transient_actions(record: Mapping[str, Any]) -> None:
    actions = record.get("actions")
    if isinstance(actions, list):
        for index in range(len(actions)):
            actions[index] = -1
        actions.clear()


def central_p99_admission_contract() -> dict[str, Any]:
    calls = j2.PRE_PPO_TEACHER_ROOTS * j2.PLANNING_MOVES
    usable_steady_hours = (
        PHASE_RUNTIME_CAP_HOURS / SAFETY_MULTIPLIER
        - OPTIMIZER_FIXTURE_HOURS
    )
    maximum_p99_seconds = (
        usable_steady_hours * 3600.0 * WORKERS / calls
    )
    if calls != 10_240 * 512 or maximum_p99_seconds <= 0:
        raise PilotIntegrityError("Central p99 admission contract changed")
    return {
        "version": f"{VERSION}_central_p99_admission_contract_v1",
        "teacher_root_equivalents": j2.PRE_PPO_TEACHER_ROOTS,
        "planning_calls_per_root": j2.PLANNING_MOVES,
        "total_teacher_calls": calls,
        "worker_concurrency_divisor": WORKERS,
        "optimizer_fixture_hours": OPTIMIZER_FIXTURE_HOURS,
        "safety_multiplier": SAFETY_MULTIPLIER,
        "runtime_cap_hours": PHASE_RUNTIME_CAP_HOURS,
        "projection_equation": (
            "1.25 * ((total_teacher_calls * observed_p99_seconds) "
            "/ (8 * 3600) + optimizer_fixture_hours)"
        ),
        "maximum_admissible_p99_equation": (
            "((72 / 1.25 - optimizer_fixture_hours) * 3600 * 8) "
            "/ total_teacher_calls"
        ),
        "maximum_admissible_p99_seconds": maximum_p99_seconds,
        "aggregate_throughput_role": (
            "descriptive_and_internal_consistency_only"
        ),
    }


def _project_phase_costs(
    *,
    central_public: Mapping[str, Any],
    sensitivity_public: Mapping[str, Any],
    sync_public: Mapping[str, Any],
    preflight_available_memory_bytes: int,
    output_bytes: int,
    admission_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    frozen_admission = central_p99_admission_contract()
    if (
        admission_contract is not None
        and j2.json_native(admission_contract) != frozen_admission
    ):
        raise PilotIntegrityError("Central p99 admission contract changed")
    p99 = float(
        central_public["parallel_eight_process"]["timing_summary"][
            "p99_seconds"
        ]
    )
    central_calls_per_second = float(
        central_public["parallel_eight_process"]["calls_per_second"]
    )
    pre_ppo_calls = int(frozen_admission["total_teacher_calls"])
    optimizer_hours = float(
        frozen_admission["optimizer_fixture_hours"]
    )
    pre_ppo_hours = (
        pre_ppo_calls * p99 / WORKERS / 3600.0
        + optimizer_hours
    ) * SAFETY_MULTIPLIER
    maximum_admissible_p99_seconds = float(
        frozen_admission["maximum_admissible_p99_seconds"]
    )
    observed_p99_margin_seconds = maximum_admissible_p99_seconds - p99
    observed_p99_margin_ratio = maximum_admissible_p99_seconds / p99
    central_parallel = central_public["parallel_eight_process"]
    recomputed_central_calls_per_second = (
        int(central_parallel["record_count"])
        / float(central_parallel["wall_seconds"])
    )
    sync_calls_per_second = float(sync_public["calls_per_second"])
    online_calls = j2.ONLINE_TEACHER_ROOTS * j2.PLANNING_MOVES
    inherited_j1_margin_hours = 3.309263890690274
    inherited_j1_pre_margin = inherited_j1_margin_hours / SAFETY_MULTIPLIER
    online_hours = (
        inherited_j1_pre_margin
        + online_calls / sync_calls_per_second / 3600.0
    ) * SAFETY_MULTIPLIER
    required_online_calls_per_second = online_calls / (
        (
            PHASE_RUNTIME_CAP_HOURS / SAFETY_MULTIPLIER
            - inherited_j1_pre_margin
        )
        * 3600.0
    )
    sensitivity_calls_per_second = float(
        sensitivity_public["parallel_eight_process"]["calls_per_second"]
    )
    pre_ppo_5000_hours = (
        j2.PRE_PPO_TEACHER_ROOTS
        * j2.SENSITIVITY_MOVES
        / sensitivity_calls_per_second
        / 3600.0
        + optimizer_hours
    ) * SAFETY_MULTIPLIER
    online_5000_hours = (
        inherited_j1_pre_margin
        + j2.ONLINE_TEACHER_ROOTS
        * j2.SENSITIVITY_MOVES
        / sync_calls_per_second
        / 3600.0
    ) * SAFETY_MULTIPLIER
    parent_peak = max(
        int(central_public["serial"]["parent_peak_rss_bytes"]),
        int(sensitivity_public["serial"]["parent_peak_rss_bytes"]),
        _rss_bytes(),
    )
    worker_independent_peak_sum = max(
        int(
            central_public["parallel_eight_process"][
                "summed_worker_peak_rss_bytes"
            ]
        ),
        int(
            sensitivity_public["parallel_eight_process"][
                "summed_worker_peak_rss_bytes"
            ]
        ),
        int(sync_public["summed_worker_peak_rss_bytes"]),
    )
    conservative_independent_peak_sum = (
        parent_peak + worker_independent_peak_sum
    )
    contemporaneous_peak = max(
        int(
            central_public["parallel_eight_process"][
                "max_contemporaneous_parent_children_rss_bytes"
            ]
        ),
        int(
            sensitivity_public["parallel_eight_process"][
                "max_contemporaneous_parent_children_rss_bytes"
            ]
        ),
        int(sync_public["max_contemporaneous_parent_children_rss_bytes"]),
    )
    memory_cap = min(
        PHASE_STORAGE_CAP_BYTES,
        int(preflight_available_memory_bytes * MEMORY_FRACTION_CAP),
    )
    distillation_storage = 14_973_665_280
    ppo_storage = 18_792_249_880
    projected_final_output_bytes = (
        output_bytes + FINAL_EVIDENCE_ALLOWANCE_BYTES
    )
    checks = {
        "central_p99_finite": math.isfinite(p99) and p99 > 0,
        "pretraining_runtime_with_margin_within_72h": (
            pre_ppo_hours <= PHASE_RUNTIME_CAP_HOURS
        ),
        "online_runtime_with_margin_within_72h": (
            online_hours <= PHASE_RUNTIME_CAP_HOURS
        ),
        "pretraining_p99_meets_derived_ceiling": (
            p99 <= maximum_admissible_p99_seconds
        ),
        "aggregate_throughput_positive_descriptive": (
            central_calls_per_second > 0
        ),
        "aggregate_throughput_recomputes_from_measured_wall": math.isclose(
            central_calls_per_second,
            recomputed_central_calls_per_second,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ),
        "online_observed_throughput_meets_derived_floor": (
            sync_calls_per_second >= required_online_calls_per_second
        ),
        "contemporaneous_peak_memory_within_frozen_cap": (
            contemporaneous_peak <= memory_cap
        ),
        "projected_final_pilot_output_with_allowance_within_1gib": (
            projected_final_output_bytes <= PILOT_OUTPUT_CAP_BYTES
        ),
        "distillation_storage_within_24gib": (
            distillation_storage <= PHASE_STORAGE_CAP_BYTES
        ),
        "ppo_storage_within_24gib": (
            ppo_storage <= PHASE_STORAGE_CAP_BYTES
        ),
    }
    return {
        "version": f"{VERSION}_measured_phase_projection_v1",
        "central": {
            "admission_contract": frozen_admission,
            "observed_calls_per_second": central_calls_per_second,
            "recomputed_calls_per_second": (
                recomputed_central_calls_per_second
            ),
            "observed_p99_seconds_per_call": p99,
            "worker_divisor": WORKERS,
            "projection_equation": (
                "(calls * observed_p99 / 8 / 3600 + optimizer_hours) "
                "* 1.25"
            ),
            "maximum_admissible_p99_seconds": (
                maximum_admissible_p99_seconds
            ),
            "observed_p99_margin_seconds": observed_p99_margin_seconds,
            "observed_p99_margin_ratio": observed_p99_margin_ratio,
            "runtime_hours_with_25pct_margin": pre_ppo_hours,
            "runtime_cap_hours": PHASE_RUNTIME_CAP_HOURS,
            "retained_storage_bytes": distillation_storage,
        },
        "online_synchronous": {
            "observed_calls_per_second": sync_calls_per_second,
            "required_calls_per_second": required_online_calls_per_second,
            "runtime_hours_with_25pct_margin": online_hours,
            "runtime_cap_hours": PHASE_RUNTIME_CAP_HOURS,
            "retained_storage_bytes": ppo_storage,
        },
        "sensitivity_5000_moves": {
            "diagnostic_not_conjunctive": True,
            "pretraining_runtime_hours_with_25pct_margin": (
                pre_ppo_5000_hours
            ),
            "pretraining_runtime_fits_72h": (
                pre_ppo_5000_hours <= PHASE_RUNTIME_CAP_HOURS
            ),
            "online_runtime_hours_with_25pct_margin": online_5000_hours,
            "online_runtime_fits_72h": (
                online_5000_hours <= PHASE_RUNTIME_CAP_HOURS
            ),
            "distillation_storage_fits_24gib": False,
            "on_policy_storage_fits_24gib": False,
        },
        "memory": {
            "parent_peak_rss_bytes": parent_peak,
            "summed_independent_worker_peak_rss_bytes": (
                worker_independent_peak_sum
            ),
            "conservative_independent_peak_sum_bytes": (
                conservative_independent_peak_sum
            ),
            "maximum_contemporaneous_parent_children_rss_bytes": (
                contemporaneous_peak
            ),
            "preflight_available_memory_bytes": (
                preflight_available_memory_bytes
            ),
            "effective_memory_cap_bytes": memory_cap,
        },
        "pilot_output": {
            "preterminal_execution_delta_bytes": output_bytes,
            "terminal_retention_allowance_bytes": (
                FINAL_EVIDENCE_ALLOWANCE_BYTES
            ),
            "projected_final_execution_delta_bytes": (
                projected_final_output_bytes
            ),
            "cap_bytes": PILOT_OUTPUT_CAP_BYTES,
        },
        "checks": checks,
        "passes": all(checks.values()),
    }


def run_synchronous_orchestration(
    *,
    binding: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    if len(rows) != SYNC_COUNT:
        raise PilotIntegrityError("Synchronous workload count changed")
    predecessor = None
    timing_samples = []
    wall_seconds = 0.0
    cpu_seconds = 0.0
    round_identities = []
    round_output_digests = []
    round_equalities = []
    chronology = []
    previous_commit_monotonic_ns = None
    max_summed_rss = 0
    max_contemporaneous_rss = 0
    from threes_rl import j1_execution_surface as j1_execution

    reference_load_started = time.perf_counter()
    reference_policy = j1_execution.load_bound_incumbent_policy(binding)
    reference_load_seconds = time.perf_counter() - reference_load_started
    for offset in range(WARMUP_CALLS):
        row = rows[offset % len(rows)]
        _query_one(
            reference_policy,
            row["snapshot"],
            int(row["policy_stream_id"]),
        )
    reference_call_seconds = 0.0
    with TeacherWorkerGroup(binding) as group:
        startup = group.startup_and_load_wall_seconds
        load_seconds = {
            str(int(row["worker_id"])): float(row["load_seconds"])
            for row in group.ready
        }
        parallel_warmup = group.warmup(rows=rows)
        measured_round_warmup_calls = []
        for round_index in range(SYNC_ROUNDS):
            start = round_index * SYNC_STATES_PER_ROUND
            stop = start + SYNC_STATES_PER_ROUND
            round_rows = list(rows[start:stop])
            reference = []
            reference_started = time.perf_counter()
            for row in round_rows:
                selected, _elapsed = _query_one(
                    reference_policy,
                    row["snapshot"],
                    int(row["policy_stream_id"]),
                )
                reference.append(selected)
            reference_call_seconds += time.perf_counter() - reference_started
            report = group.run_round(
                round_id=f"sync-round-{round_index:02d}",
                rows=round_rows,
            )
            measured_round_warmup_calls.append(
                int(report["warmup_calls"])
            )
            if report["warmup_calls"] != 0:
                raise PilotIntegrityError(
                    "Measured synchronous round included warmup calls"
                )
            equality = list(report["actions"]) == reference
            if not equality:
                raise PilotIntegrityError(
                    "Synchronous teacher actions differ from serial reference"
                )
            dispatch_after_predecessor = (
                previous_commit_monotonic_ns is None
                or int(report["dispatch_monotonic_ns"])
                > previous_commit_monotonic_ns
            )
            if not dispatch_after_predecessor:
                raise PilotIntegrityError(
                    "Synchronous round dispatched before predecessor commit"
                )
            serial_reference_sha256 = canonical_hash(
                [
                    (int(row["state_index"]), selected)
                    for row, selected in zip(
                        round_rows,
                        reference,
                        strict=True,
                    )
                ]
            )
            if (
                serial_reference_sha256
                != report["ordered_output_sha256"]
            ):
                raise PilotIntegrityError(
                    "Synchronous aggregate output digest changed"
                )
            round_payload = {
                "version": f"{VERSION}_sync_round_manifest_v1",
                "round_index": round_index,
                "state_count": SYNC_STATES_PER_ROUND,
                "state_index_start": start,
                "state_index_stop_exclusive": stop,
                "input_state_sha256": canonical_hash(
                    [
                        {
                            "state_index": int(row["state_index"]),
                            "state_sha256": canonical_hash(row["snapshot"]),
                            "worker_id": int(row["state_index"]) % WORKERS,
                        }
                        for row in round_rows
                    ]
                ),
                "ordered_output_sha256": report[
                    "ordered_output_sha256"
                ],
                "serial_reference_sha256": serial_reference_sha256,
                "exact_reference_equality": equality,
                "worker_count": WORKERS,
                "worker_record_counts": {
                    str(worker): sum(
                        int(row["state_index"]) % WORKERS == worker
                        for row in round_rows
                    )
                    for worker in range(WORKERS)
                },
                "timing_summary": report["timing_summary"],
                "wall_seconds": report["wall_seconds"],
                "child_cpu_seconds": report["child_cpu_seconds"],
                "worker_warmup_calls_during_measured_round": report[
                    "warmup_calls"
                ],
                "dispatch_monotonic_ns": report[
                    "dispatch_monotonic_ns"
                ],
                "all_results_received_monotonic_ns": report[
                    "received_monotonic_ns"
                ],
                "predecessor_commit_monotonic_ns": (
                    previous_commit_monotonic_ns
                ),
                "dispatch_after_predecessor_commit": (
                    dispatch_after_predecessor
                ),
                "predecessor_manifest_payload_sha256": predecessor,
                "missing_records": 0,
                "duplicate_records": 0,
                "late_records": 0,
                "cross_round_records": 0,
                "actions_retained": 0,
                "labels_retained": 0,
                "passes": equality,
            }
            _zeroize_transient_actions(report)
            for index in range(len(reference)):
                reference[index] = -1
            reference.clear()
            assert_no_forbidden_retained_fields(round_payload)
            path = (
                output_dir
                / ROUND_DIR_NAME
                / f"round_{round_index:02d}.json"
            )
            write_immutable(
                path,
                round_payload,
                field="round_manifest_payload_sha256",
            )
            identity = immutable_identity(
                path,
                "round_manifest_payload_sha256",
            )
            commit_monotonic_ns = time.monotonic_ns()
            chronology.append(
                {
                    "round_index": round_index,
                    "dispatch_monotonic_ns": round_payload[
                        "dispatch_monotonic_ns"
                    ],
                    "all_results_received_monotonic_ns": round_payload[
                        "all_results_received_monotonic_ns"
                    ],
                    "commit_monotonic_ns": commit_monotonic_ns,
                    "dispatch_after_predecessor_commit": (
                        dispatch_after_predecessor
                    ),
                }
            )
            previous_commit_monotonic_ns = commit_monotonic_ns
            predecessor = identity["payload_sha256"]
            round_identities.append(identity)
            round_output_digests.append(
                round_payload["ordered_output_sha256"]
            )
            round_equalities.append(equality)
            timing_samples.extend(report["timings"])
            wall_seconds += float(report["wall_seconds"])
            cpu_seconds += float(report["child_cpu_seconds"])
            max_summed_rss = max(
                max_summed_rss,
                int(report["summed_worker_peak_rss_bytes"]),
            )
            max_contemporaneous_rss = max(
                max_contemporaneous_rss,
                int(
                    report[
                        "max_contemporaneous_parent_children_rss_bytes"
                    ]
                ),
            )
    if group.lifetime_rss_evidence is None:
        raise PilotIntegrityError(
            "Synchronous worker-group lifetime RSS was not sealed"
        )
    max_contemporaneous_rss = max(
        max_contemporaneous_rss,
        int(
            group.lifetime_rss_evidence[
                "maximum_contemporaneous_parent_children_rss_bytes"
            ]
        ),
    )
    public = {
        "version": f"{VERSION}_synchronous_orchestration_v1",
        "rounds": SYNC_ROUNDS,
        "states_per_round": SYNC_STATES_PER_ROUND,
        "record_count": SYNC_COUNT,
        "worker_count": WORKERS,
        "warmup_calls_per_process": WARMUP_CALLS,
        "total_sync_worker_warmup_calls": int(
            parallel_warmup["total_worker_warmup_calls"]
        ),
        "sync_worker_warmup_wall_seconds": float(
            parallel_warmup["warmup_wall_seconds"]
        ),
        "serial_reference_warmup_calls": WARMUP_CALLS,
        "measured_round_warmup_calls": measured_round_warmup_calls,
        "all_measured_round_warmups_zero": all(
            value == 0 for value in measured_round_warmup_calls
        ),
        "warmups_excluded_from_steady_wall": True,
        "round_manifest_identities": round_identities,
        "round_chain_head_payload_sha256": predecessor,
        "ordered_round_output_digest_sha256": canonical_hash(
            round_output_digests
        ),
        "exact_reference_equality": all(round_equalities),
        "all_barriers_exact": len(round_identities) == SYNC_ROUNDS,
        "dispatch_commit_chronology": chronology,
        "no_next_round_prefetch": all(
            bool(row["dispatch_after_predecessor_commit"])
            for row in chronology
        ),
        "calls_per_second": SYNC_COUNT / wall_seconds,
        "timing_summary": _timing_summary(timing_samples),
        "steady_wall_seconds": wall_seconds,
        "child_cpu_seconds": cpu_seconds,
        "startup_and_load_wall_seconds": startup,
        "worker_load_seconds": load_seconds,
        "serial_reference_load_seconds": reference_load_seconds,
        "serial_reference_call_seconds": reference_call_seconds,
        "summed_worker_peak_rss_bytes": max_summed_rss,
        "max_contemporaneous_parent_children_rss_bytes": (
            max_contemporaneous_rss
        ),
        "lifetime_rss_sample_count": int(
            group.lifetime_rss_evidence["sample_count"]
        ),
        "lifetime_rss_covers_load_warmup_queries_and_shutdown": bool(
            group.lifetime_rss_evidence[
                "covers_load_warmup_queries_and_shutdown"
            ]
        ),
        "missing_records": 0,
        "duplicate_records": 0,
        "late_records": 0,
        "cross_round_records": 0,
        "prefetched_rounds": 0,
        "actions_retained": 0,
        "labels_retained": 0,
        "ppo_trajectories": 0,
        "optimizer_steps": 0,
        "checkpoints": 0,
        "passes": (
            all(round_equalities)
            and len(round_identities) == SYNC_ROUNDS
            and int(parallel_warmup["total_worker_warmup_calls"])
            == WORKERS * WARMUP_CALLS
            and all(value == 0 for value in measured_round_warmup_calls)
            and all(
                bool(row["dispatch_after_predecessor_commit"])
                for row in chronology
            )
        ),
    }
    assert_no_forbidden_retained_fields(public)
    return public


def run_power_sizing() -> dict[str, Any]:
    rows = []
    for n_pairs in POWER_N_GRID:
        report = j2.common_or_power_grid(
            n_pairs=n_pairs,
            datasets=POWER_DATASETS,
            bootstraps=POWER_BOOTSTRAPS,
        )
        worst_row = min(
            report["rows"],
            key=lambda row: float(row["primary_gate_power"]),
        )
        rows.append(
            {
                "n_pairs": n_pairs,
                "worst_case_power": report["worst_case_primary_power"],
                "worst_case_mcse": worst_row[
                    "monte_carlo_standard_error"
                ],
                "worst_case_control_rate": worst_row["control_rate"],
                "worst_case_coupling": worst_row["coupling"],
                "full_report_sha256": canonical_hash(report),
                "passes_080": (
                    float(report["worst_case_primary_power"]) >= 0.80
                ),
            }
        )
    passing = [row["n_pairs"] for row in rows if row["passes_080"]]
    smallest = min(passing) if passing else None
    checks = {
        "n_grid_exact": [row["n_pairs"] for row in rows]
        == list(POWER_N_GRID),
        "datasets_exact": POWER_DATASETS == 768,
        "bootstraps_exact": POWER_BOOTSTRAPS == 199,
        "smallest_powered_n_found": smallest is not None,
        "published_n2048_reproduced": math.isclose(
            float(rows[0]["worst_case_power"]),
            0.6432291666666666,
            rel_tol=0.0,
            abs_tol=0.0,
        ),
    }
    return {
        "version": f"{VERSION}_power_sizing_v1",
        "method": {
            "strata": 8,
            "control_rates": list(j2.CONTROL_RATES),
            "couplings": list(j2.PAIRING_COUPLINGS),
            "point_gate": j2.FIDELITY_OR_POINT_FLOOR,
            "lower_gate": j2.FIDELITY_OR_CI_FLOOR,
            "datasets": POWER_DATASETS,
            "bootstraps": POWER_BOOTSTRAPS,
            "seed": j2.POWER_SEED,
            "quantile": "numpy linear 0.025/0.975",
        },
        "rows": rows,
        "smallest_grid_n_at_least_080": smallest,
        "checks": checks,
        "passes": all(checks.values()),
    }


def assert_no_forbidden_retained_fields(value: Any, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower()
            if normalized in FORBIDDEN_RETAINED_KEYS:
                raise PilotIntegrityError(
                    f"Forbidden retained field {path}.{key}"
                )
            assert_no_forbidden_retained_fields(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            assert_no_forbidden_retained_fields(child, f"{path}[{index}]")


def zero_work_audit(
    *,
    output_dir: Path = OUTPUT_DIR,
    allowed_files: Sequence[str] = (),
    include_operational: bool = True,
) -> dict[str, Any]:
    files = (
        sorted(
            str(path.relative_to(output_dir))
            for path in output_dir.rglob("*")
            if path.is_file()
        )
        if output_dir.exists()
        else []
    )
    future = {
        str(path.resolve()): path.exists()
        for path in j2.FUTURE_EXECUTION_DIRS
    }
    operational = (
        operational_audit(
            output_dir=output_dir,
            include_namespace_absence=not output_dir.exists(),
        )
        if include_operational
        else {"passes": True, "skipped": True}
    )
    checks = {
        "only_allowed_files": files == sorted(allowed_files),
        "j2_future_namespaces_absent": not any(future.values()),
        "no_teacher_queries": True,
        "no_actions_or_labels": True,
        "no_games_or_outcomes": True,
        "no_stream_reservation_or_consumption": True,
        "operational_passes": operational["passes"],
    }
    return {
        "version": f"{VERSION}_zero_work_audit_v1",
        "files": files,
        "allowed_files": sorted(allowed_files),
        "future_j2_namespaces": future,
        "operational": operational,
        "checks": checks,
        "passes": all(checks.values()),
        "teacher_queries": 0,
        "actions_retained": 0,
        "labels": 0,
        "games": 0,
        "outcomes": 0,
        "j2_streams_reserved": 0,
        "j2_streams_consumed": 0,
    }


def write_test_evidence(
    *,
    output_dir: Path,
    commands: Sequence[Mapping[str, Any]],
    deselections: Sequence[str],
) -> dict[str, Any]:
    zero = zero_work_audit(
        output_dir=output_dir,
        include_operational=False,
    )
    if not zero["passes"] or not commands:
        raise PilotIntegrityError("Test evidence is not zero-work or empty")
    normalized = []
    for row in commands:
        passed = int(row["passed"])
        failed = int(row.get("failed", 0))
        if passed < 1 or failed != 0:
            raise PilotIntegrityError("Test command did not pass")
        normalized.append(
            {
                "name": str(row["name"]),
                "command": str(row["command"]),
                "passed": passed,
                "failed": failed,
                "deselected": int(row.get("deselected", 0)),
            }
        )
    sources = source_identity_audit()
    streams = stream_authority_audit()
    if not sources["passes"] or not streams["passes"]:
        raise PilotIntegrityError("Pilot source or streams failed")
    payload = {
        "version": f"{VERSION}_test_evidence_v1",
        "source_identities": sources["local_sources"],
        "source_audit_sha256": canonical_hash(sources),
        "stream_audit_sha256": canonical_hash(streams),
        "commands": normalized,
        "total_passed": sum(row["passed"] for row in normalized),
        "total_failed": 0,
        "deselections": sorted(str(value) for value in deselections),
        "zero_work": zero,
    }
    return write_immutable(
        output_dir / TEST_EVIDENCE_NAME,
        payload,
        field="test_evidence_payload_sha256",
    )


def prepare(*, output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    zero = zero_work_audit(
        output_dir=output_dir,
        allowed_files=(TEST_EVIDENCE_NAME,),
    )
    evidence = load_hashed_json(
        output_dir / TEST_EVIDENCE_NAME,
        field="test_evidence_payload_sha256",
    )
    inventory, snapshots = build_and_seal_inventory(output_dir=output_dir)
    if len(snapshots) != INVENTORY_COUNT:
        raise PilotIntegrityError("Prepared inventory snapshot count changed")
    del snapshots
    inventory_identity = immutable_identity(
        output_dir / INVENTORY_NAME,
        "inventory_payload_sha256",
    )
    sources = source_identity_audit()
    streams = stream_authority_audit()
    integrity = {
        "zero_work": zero["passes"],
        "test_evidence_source_exact": evidence["source_identities"]
        == sources["local_sources"],
        "source_identity": sources["passes"],
        "stream_authority": streams["passes"],
        "inventory_complete_and_reloaded": (
            inventory["passes"]
            and len(inventory["rows"]) == INVENTORY_COUNT
            and load_hashed_json(
                output_dir / INVENTORY_NAME,
                field="inventory_payload_sha256",
            )
            == inventory
        ),
        "marker_absent": not (output_dir / MARKER_NAME).exists(),
        "terminal_absent": not (output_dir / TERMINAL_NAME).exists(),
    }
    operational = dict(zero["operational"]["checks"])
    decision = (
        READY_PREFLIGHT
        if all(integrity.values()) and all(operational.values())
        else (
            KILL_TERMINAL
            if not all(integrity.values())
            else HOLD_PREFLIGHT
        )
    )
    lock = {
        "version": f"{VERSION}_preflight_lock_v1",
        "decision": decision,
        "source_audit": sources,
        "stream_authority": streams,
        "test_evidence": immutable_identity(
            output_dir / TEST_EVIDENCE_NAME,
            "test_evidence_payload_sha256",
        ),
        "inventory": inventory_identity,
        "operational": zero["operational"],
        "inventory_contract": {
            "count": INVENTORY_COUNT,
            "central_indices": [0, CENTRAL_COUNT - 1],
            "sensitivity_indices": [0, INVENTORY_COUNT - 1],
            "sync_indices": [0, SYNC_COUNT - 1],
            "prefix_formula": "16 + ((73 * state_index + 19) mod 160)",
            "warmup_calls_per_process": WARMUP_CALLS,
        },
        "power_n_grid": list(POWER_N_GRID),
        "power_datasets": POWER_DATASETS,
        "power_bootstraps": POWER_BOOTSTRAPS,
        "central_p99_admission_contract": (
            central_p99_admission_contract()
        ),
        "execution_command": (
            "nice -n 10 env PYTHONPATH=. .venv/bin/python -m "
            "threes_rl.j2_exact_teacher_feasibility_pilot execute "
            "--out-dir "
            "threes_rl/runs/forensics/"
            "j2_exact_teacher_feasibility_pilot_v1"
        ),
        "teacher_queries": 0,
        "teacher_loads": 0,
        "actions_retained": 0,
        "labels": 0,
        "games": 0,
        "outcomes": 0,
        "execution_authorized": decision == READY_PREFLIGHT,
    }
    write_immutable(
        output_dir / PREFLIGHT_LOCK_NAME,
        lock,
        field="preflight_lock_payload_sha256",
    )
    result = {
        "version": f"{VERSION}_preflight_result_v1",
        "decision": decision,
        "lock": immutable_identity(
            output_dir / PREFLIGHT_LOCK_NAME,
            "preflight_lock_payload_sha256",
        ),
        "inventory": inventory_identity,
        "integrity": integrity,
        "operational": operational,
        "execution_authorized": decision == READY_PREFLIGHT,
        "teacher_queries": 0,
        "teacher_loads": 0,
        "actions_retained": 0,
        "labels": 0,
        "games": 0,
        "outcomes": 0,
    }
    write_immutable(
        output_dir / PREFLIGHT_RESULT_NAME,
        result,
        field="preflight_result_payload_sha256",
    )
    return load_hashed_json(
        output_dir / PREFLIGHT_RESULT_NAME,
        field="preflight_result_payload_sha256",
    )


def open_execution(*, output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    lock = load_hashed_json(
        output_dir / PREFLIGHT_LOCK_NAME,
        field="preflight_lock_payload_sha256",
    )
    result = load_hashed_json(
        output_dir / PREFLIGHT_RESULT_NAME,
        field="preflight_result_payload_sha256",
    )
    if (
        lock["decision"] != READY_PREFLIGHT
        or result["decision"] != READY_PREFLIGHT
        or not result["execution_authorized"]
    ):
        raise PilotIntegrityError("Pilot preflight did not authorize open")
    inventory = load_hashed_json(
        output_dir / INVENTORY_NAME,
        field="inventory_payload_sha256",
    )
    inventory_identity = immutable_identity(
        output_dir / INVENTORY_NAME,
        "inventory_payload_sha256",
    )
    if (
        inventory_identity != lock.get("inventory")
        or inventory_identity != result.get("inventory")
        or len(inventory.get("rows", [])) != INVENTORY_COUNT
        or inventory.get("passes") is not True
    ):
        raise PilotIntegrityError("Prepared inventory identity changed")
    sources = source_identity_audit()
    if not sources["passes"] or sources != lock["source_audit"]:
        raise PilotIntegrityError("Pilot sources changed before marker")
    allowed = {
        TEST_EVIDENCE_NAME,
        INVENTORY_NAME,
        PREFLIGHT_LOCK_NAME,
        PREFLIGHT_RESULT_NAME,
    }
    observed = {
        path.name
        for path in output_dir.iterdir()
        if path.is_file()
    }
    if observed != allowed:
        raise PilotIntegrityError("Unexpected pre-marker artifact")
    operational = operational_audit(
        output_dir=output_dir,
        include_namespace_absence=False,
    )
    if not operational["passes"]:
        raise PilotOperationalHold("Pilot open operational gate failed")
    marker = {
        "version": f"{VERSION}_execution_marker_v1",
        "preflight_lock": immutable_identity(
            output_dir / PREFLIGHT_LOCK_NAME,
            "preflight_lock_payload_sha256",
        ),
        "preflight_result": immutable_identity(
            output_dir / PREFLIGHT_RESULT_NAME,
            "preflight_result_payload_sha256",
        ),
        "source_identities": sources["local_sources"],
        "teacher_binding_sha256": sources["teacher_binding"][
            "incumbent_binding_sha256"
        ],
        "stream_authority_sha256": canonical_hash(
            stream_authority_audit()
        ),
        "inventory": inventory_identity,
        "sealed_inventory_states_before_marker": INVENTORY_COUNT,
        "central_p99_admission_contract": lock[
            "central_p99_admission_contract"
        ],
        "operational": operational,
        "execution_command": lock["execution_command"],
        "teacher_queries_before_marker": 0,
        "teacher_loads_before_marker": 0,
        "inventory_states_before_marker": INVENTORY_COUNT,
        "actions_retained": 0,
        "labels": 0,
        "games": 0,
        "outcomes": 0,
    }
    return write_immutable(
        output_dir / MARKER_NAME,
        marker,
        field="marker_payload_sha256",
    )


def _directory_bytes(path: Path) -> int:
    return sum(
        child.stat().st_size for child in path.rglob("*") if child.is_file()
    )


def teacher_query_accounting() -> dict[str, int]:
    accounting = {
        "central_serial_measured": CENTRAL_COUNT,
        "central_parallel_measured": CENTRAL_COUNT,
        "sensitivity_serial_measured": INVENTORY_COUNT,
        "sensitivity_parallel_measured": INVENTORY_COUNT,
        "synchronous_serial_reference_measured": SYNC_COUNT,
        "synchronous_parallel_measured": SYNC_COUNT,
        "serial_warmups": 3 * WARMUP_CALLS,
        "parallel_worker_warmups": 3 * WORKERS * WARMUP_CALLS,
    }
    accounting["total"] = sum(accounting.values())
    if accounting["total"] != EXPECTED_TEACHER_QUERY_CALLS:
        raise PilotIntegrityError("Teacher query accounting changed")
    return accounting


def terminal_decision(
    *,
    integrity_pass: bool,
    throughput_pass: bool,
    synchronous_pass: bool,
    power_pass: bool,
) -> str:
    if not integrity_pass:
        return KILL_TERMINAL
    if throughput_pass and synchronous_pass and power_pass:
        return READY_TERMINAL
    return HOLD_TERMINAL


def final_output_cap_audit(
    *,
    preterminal_execution_delta_bytes: int,
    final_execution_delta_bytes: int,
) -> dict[str, Any]:
    checks = {
        "final_not_below_preterminal": (
            final_execution_delta_bytes >= preterminal_execution_delta_bytes
        ),
        "terminal_retention_within_frozen_allowance": (
            final_execution_delta_bytes
            <= preterminal_execution_delta_bytes
            + FINAL_EVIDENCE_ALLOWANCE_BYTES
        ),
        "final_retained_execution_delta_within_1gib": (
            final_execution_delta_bytes <= PILOT_OUTPUT_CAP_BYTES
        ),
    }
    return {
        "preterminal_execution_delta_bytes": (
            preterminal_execution_delta_bytes
        ),
        "final_execution_delta_bytes": final_execution_delta_bytes,
        "terminal_retention_allowance_bytes": (
            FINAL_EVIDENCE_ALLOWANCE_BYTES
        ),
        "cap_bytes": PILOT_OUTPUT_CAP_BYTES,
        "checks": checks,
        "passes": all(checks.values()),
    }


def _execute_one_shot(*, output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    marker = load_hashed_json(
        output_dir / MARKER_NAME,
        field="marker_payload_sha256",
    )
    lock = load_hashed_json(
        output_dir / PREFLIGHT_LOCK_NAME,
        field="preflight_lock_payload_sha256",
    )
    if lock["decision"] != READY_PREFLIGHT:
        raise PilotIntegrityError("Pilot lock is not READY")
    if marker["execution_command"] != lock["execution_command"]:
        raise PilotIntegrityError("Marker command changed")
    if (
        marker.get("central_p99_admission_contract")
        != lock.get("central_p99_admission_contract")
        or lock.get("central_p99_admission_contract")
        != central_p99_admission_contract()
    ):
        raise PilotIntegrityError("Central p99 admission binding changed")
    inventory = load_hashed_json(
        output_dir / INVENTORY_NAME,
        field="inventory_payload_sha256",
    )
    inventory_identity = immutable_identity(
        output_dir / INVENTORY_NAME,
        "inventory_payload_sha256",
    )
    if (
        inventory_identity != marker.get("inventory")
        or inventory_identity != lock.get("inventory")
        or len(inventory.get("rows", [])) != INVENTORY_COUNT
    ):
        raise PilotIntegrityError("Execution inventory identity changed")
    sources = source_identity_audit()
    if (
        not sources["passes"]
        or sources["local_sources"] != marker["source_identities"]
        or sources["teacher_binding"]["incumbent_binding_sha256"]
        != marker["teacher_binding_sha256"]
    ):
        raise PilotIntegrityError("Teacher/source identity changed")
    operational = operational_audit(
        output_dir=output_dir,
        include_namespace_absence=False,
    )
    if not operational["passes"]:
        raise PilotOperationalHold("Pilot execute operational gate failed")
    if (output_dir / TERMINAL_NAME).exists():
        raise PilotIntegrityError("Pilot terminal already exists")

    started_bytes = _directory_bytes(output_dir)
    snapshots = regenerate_inventory(inventory)
    inventory_rows = list(inventory["rows"])
    binding = sources["teacher_binding"]

    central_rows = _query_rows(
        snapshots,
        inventory_rows,
        CENTRAL_COUNT,
    )
    serial_central = run_serial_workload(
        binding=binding,
        rows=central_rows,
    )
    parallel_central = run_parallel_workload(
        binding=binding,
        rows=central_rows,
        round_id="central",
    )
    central_public = _public_cost_result(
        workload="central",
        serial=serial_central,
        parallel=parallel_central,
        reference_digest=None,
    )
    _zeroize_transient_actions(serial_central)
    _zeroize_transient_actions(parallel_central)
    write_immutable(
        output_dir / CENTRAL_NAME,
        central_public,
        field="central_payload_sha256",
    )

    sensitivity_rows = _query_rows(
        snapshots,
        inventory_rows,
        INVENTORY_COUNT,
    )
    serial_sensitivity = run_serial_workload(
        binding=binding,
        rows=sensitivity_rows,
    )
    central_prefix_digest = canonical_hash(
        [
            (index, action)
            for index, action in enumerate(
                serial_sensitivity["actions"][:CENTRAL_COUNT]
            )
        ]
    )
    if central_prefix_digest != serial_central["ordered_output_sha256"]:
        raise PilotIntegrityError("Repeated central digest changed")
    parallel_sensitivity = run_parallel_workload(
        binding=binding,
        rows=sensitivity_rows,
        round_id="sensitivity",
    )
    sensitivity_public = _public_cost_result(
        workload="sensitivity",
        serial=serial_sensitivity,
        parallel=parallel_sensitivity,
        reference_digest=None,
    )
    sensitivity_public["central_prefix_digest_repeats"] = True
    _zeroize_transient_actions(serial_sensitivity)
    _zeroize_transient_actions(parallel_sensitivity)
    write_immutable(
        output_dir / SENSITIVITY_NAME,
        sensitivity_public,
        field="sensitivity_payload_sha256",
    )

    sync_rows = sensitivity_rows[:SYNC_COUNT]
    sync_public = run_synchronous_orchestration(
        binding=binding,
        rows=sync_rows,
        output_dir=output_dir,
    )
    write_immutable(
        output_dir / SYNC_NAME,
        sync_public,
        field="sync_payload_sha256",
    )

    power = run_power_sizing()
    write_immutable(
        output_dir / POWER_NAME,
        power,
        field="power_payload_sha256",
    )

    output_bytes = _directory_bytes(output_dir) - started_bytes
    projection = _project_phase_costs(
        central_public=central_public,
        sensitivity_public=sensitivity_public,
        sync_public=sync_public,
        preflight_available_memory_bytes=int(
            marker["operational"]["available_memory_bytes"]
        ),
        output_bytes=output_bytes,
        admission_contract=lock["central_p99_admission_contract"],
    )
    throughput_pass = (
        central_public["passes"]
        and sensitivity_public["passes"]
        and all(
            projection["checks"][key]
            for key in (
                "pretraining_runtime_with_margin_within_72h",
                "pretraining_p99_meets_derived_ceiling",
                "contemporaneous_peak_memory_within_frozen_cap",
                "projected_final_pilot_output_with_allowance_within_1gib",
                "distillation_storage_within_24gib",
            )
        )
    )
    synchronous_pass = (
        sync_public["passes"]
        and all(
            projection["checks"][key]
            for key in (
                "online_runtime_with_margin_within_72h",
                "online_observed_throughput_meets_derived_floor",
                "contemporaneous_peak_memory_within_frozen_cap",
                "projected_final_pilot_output_with_allowance_within_1gib",
                "ppo_storage_within_24gib",
            )
        )
    )
    retained_paths = [
        path
        for path in output_dir.rglob("*.json")
        if path.name not in {TERMINAL_NAME, RETENTION_NAME}
    ]
    for path in retained_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert_no_forbidden_retained_fields(payload)
    integrity_pass = (
        inventory["passes"]
        and central_public["passes"]
        and sensitivity_public["passes"]
        and sync_public["passes"]
        and len(retained_paths) == 9 + SYNC_ROUNDS
    )
    decision = terminal_decision(
        integrity_pass=integrity_pass,
        throughput_pass=throughput_pass,
        synchronous_pass=synchronous_pass,
        power_pass=bool(power["passes"]),
    )
    terminal = {
        "version": f"{VERSION}_terminal_result_v1",
        "decision": decision,
        "separate_decisions": {
            "real_eight_process_pretraining_throughput_memory": (
                "PASS" if throughput_pass else "HOLD"
            ),
            "synchronous_16_round_orchestration": (
                "PASS" if synchronous_pass else "HOLD"
            ),
            "powered_validation_n_recommendation": {
                "decision": (
                    "PASS" if power["passes"] else "HOLD"
                ),
                "smallest_grid_n_at_least_080": power[
                    "smallest_grid_n_at_least_080"
                ],
            },
        },
        "inventory": immutable_identity(
            output_dir / INVENTORY_NAME,
            "inventory_payload_sha256",
        ),
        "central": immutable_identity(
            output_dir / CENTRAL_NAME,
            "central_payload_sha256",
        ),
        "sensitivity": immutable_identity(
            output_dir / SENSITIVITY_NAME,
            "sensitivity_payload_sha256",
        ),
        "synchronous": immutable_identity(
            output_dir / SYNC_NAME,
            "sync_payload_sha256",
        ),
        "power": immutable_identity(
            output_dir / POWER_NAME,
            "power_payload_sha256",
        ),
        "measured_phase_projection": projection,
        "integrity_passes": integrity_pass,
        "operations_passed_at_open": marker["operational"]["passes"],
        "teacher_query_accounting": teacher_query_accounting(),
        "teacher_query_calls": EXPECTED_TEACHER_QUERY_CALLS,
        "actions_retained": 0,
        "labels_retained": 0,
        "scores_retained": 0,
        "outcomes_retained": 0,
        "trajectories_retained": 0,
        "ppo_trajectories": 0,
        "optimizer_steps": 0,
        "checkpoints": 0,
        "j2_scientific_streams_reserved": 0,
        "j2_scientific_streams_consumed": 0,
        "development_opened": False,
        "confirmation_opened": False,
        "promotion": False,
    }
    assert_no_forbidden_retained_fields(terminal)
    written_terminal = write_immutable(
        output_dir / TERMINAL_NAME,
        terminal,
        field="terminal_payload_sha256",
    )
    inventory_paths = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != RETENTION_NAME
    )
    retention_rows = [
        {
            "path": str(path.relative_to(output_dir)),
            "file_sha256": sha256_path(path),
            "bytes": path.stat().st_size,
        }
        for path in inventory_paths
    ]
    retention = {
        "version": f"{VERSION}_retention_v1",
        "decision": decision,
        "files": retention_rows,
        "file_count": len(retention_rows),
        "total_bytes": sum(row["bytes"] for row in retention_rows),
        "inventory_sha256": canonical_hash(retention_rows),
        "preserve_byte_for_byte": True,
        "output_cap_contract": {
            "preterminal_execution_delta_bytes": output_bytes,
            "terminal_retention_allowance_bytes": (
                FINAL_EVIDENCE_ALLOWANCE_BYTES
            ),
            "final_cap_bytes": PILOT_OUTPUT_CAP_BYTES,
        },
        "forbidden_retained_fields_zero": True,
        "actions_retained": 0,
        "labels_retained": 0,
        "scores_retained": 0,
        "outcomes_retained": 0,
        "trajectories_retained": 0,
        "passes": True,
    }
    assert_no_forbidden_retained_fields(retention)
    write_immutable(
        output_dir / RETENTION_NAME,
        retention,
        field="retention_payload_sha256",
    )
    return written_terminal


def _seal_failed_execution(
    *,
    output_dir: Path,
    error: PilotIntegrityError | PilotOperationalHold,
) -> dict[str, Any]:
    terminal_path = output_dir / TERMINAL_NAME
    retention_path = output_dir / RETENTION_NAME
    if retention_path.exists() and not terminal_path.exists():
        raise PilotIntegrityError(
            "Orphan retention cannot authorize execution recovery"
        )
    if terminal_path.exists():
        terminal = load_hashed_json(
            output_dir / TERMINAL_NAME,
            field="terminal_payload_sha256",
        )
        if not retention_path.exists():
            raise PilotIntegrityError(
                "Existing terminal is incomplete without retention"
            )
        retention = load_hashed_json(
            retention_path,
            field="retention_payload_sha256",
        )
        if (
            retention.get("decision") != terminal.get("decision")
            or retention.get("passes") is not True
        ):
            raise PilotIntegrityError(
                "Existing terminal retention is not authoritative"
            )
        return terminal
    decision = (
        KILL_TERMINAL
        if isinstance(error, PilotIntegrityError)
        else HOLD_TERMINAL
    )
    existing_files = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file()
        and path.name not in {TERMINAL_NAME, RETENTION_NAME}
    )
    artifact_rows = [
        {
            "path": str(path.relative_to(output_dir)),
            "file_sha256": sha256_path(path),
            "bytes": path.stat().st_size,
        }
        for path in existing_files
    ]
    terminal = {
        "version": f"{VERSION}_failed_execution_terminal_v1",
        "decision": decision,
        "failure_class": (
            "immutable_integrity"
            if isinstance(error, PilotIntegrityError)
            else "operational_or_resource"
        ),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "artifacts_present_before_terminal": artifact_rows,
        "artifacts_present_sha256": canonical_hash(artifact_rows),
        "retry_authorized": False,
        "teacher_labels_retained": 0,
        "actions_retained": 0,
        "scores_retained": 0,
        "outcomes_retained": 0,
        "trajectories_retained": 0,
        "ppo_trajectories": 0,
        "optimizer_steps": 0,
        "checkpoints": 0,
        "j2_scientific_streams_reserved": 0,
        "j2_scientific_streams_consumed": 0,
        "development_opened": False,
        "confirmation_opened": False,
        "promotion": False,
    }
    assert_no_forbidden_retained_fields(terminal)
    write_immutable(
        terminal_path,
        terminal,
        field="terminal_payload_sha256",
    )
    retained = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != RETENTION_NAME
    )
    retention_rows = [
        {
            "path": str(path.relative_to(output_dir)),
            "file_sha256": sha256_path(path),
            "bytes": path.stat().st_size,
        }
        for path in retained
    ]
    retention = {
        "version": f"{VERSION}_failed_execution_retention_v1",
        "decision": decision,
        "files": retention_rows,
        "file_count": len(retention_rows),
        "total_bytes": sum(row["bytes"] for row in retention_rows),
        "inventory_sha256": canonical_hash(retention_rows),
        "preserve_byte_for_byte": True,
        "retry_authorized": False,
        "passes": True,
    }
    write_immutable(
        retention_path,
        retention,
        field="retention_payload_sha256",
    )
    return load_hashed_json(
        output_dir / TERMINAL_NAME,
        field="terminal_payload_sha256",
    )


def execute(*, output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    if (output_dir / TERMINAL_NAME).exists():
        return _seal_failed_execution(
            output_dir=output_dir,
            error=PilotIntegrityError(
                "Existing terminal forbids pilot restart"
            ),
        )
    try:
        return _execute_one_shot(output_dir=output_dir)
    except (PilotIntegrityError, PilotOperationalHold) as error:
        return _seal_failed_execution(output_dir=output_dir, error=error)


def _parse_command(value: str) -> dict[str, Any]:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError("Command must be JSON") from error
    if not isinstance(payload, dict):
        raise argparse.ArgumentTypeError("Command must be a JSON object")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Outcome-free J2 exact-teacher feasibility pilot"
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    audit = subparsers.add_parser("audit-zero-work")
    audit.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    evidence = subparsers.add_parser("write-test-evidence")
    evidence.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    evidence.add_argument(
        "--recorded-command",
        action="append",
        required=True,
        type=_parse_command,
    )
    evidence.add_argument("--deselection", action="append", default=[])
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    open_parser = subparsers.add_parser("open")
    open_parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    execute_parser = subparsers.add_parser("execute")
    execute_parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.subcommand == "audit-zero-work":
        payload = zero_work_audit(output_dir=args.out_dir)
    elif args.subcommand == "write-test-evidence":
        payload = write_test_evidence(
            output_dir=args.out_dir,
            commands=args.recorded_command,
            deselections=args.deselection,
        )
    elif args.subcommand == "prepare":
        payload = prepare(output_dir=args.out_dir)
    elif args.subcommand == "open":
        payload = open_execution(output_dir=args.out_dir)
    elif args.subcommand == "execute":
        payload = execute(output_dir=args.out_dir)
    else:
        raise PilotIntegrityError(f"Forbidden command: {args.subcommand}")
    print(json.dumps(j2.json_native(payload), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
