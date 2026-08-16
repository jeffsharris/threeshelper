"""Outcome-free J1b deterministic Torch-runtime repair preflight.

This module intentionally imports only the Python standard library at module
load. The clean scientific runtime path configures PyTorch before importing any
J1 parent module.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


VERSION = "j1b_operational_repair_preflight_v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "threes_rl" / "runs"
CHARTER_PATH = (
    REPO_ROOT / "threes_rl" / "J1B_OPERATIONAL_REPAIR_PREFLIGHT_CHARTER.md"
)
AMENDMENT_A1_PATH = (
    REPO_ROOT
    / "threes_rl"
    / "J1B_OPERATIONAL_REPAIR_PREFLIGHT_AMENDMENT_A1.md"
)
RUNNER_PATH = REPO_ROOT / "threes_rl" / "j1b_operational_repair_preflight.py"
TEST_PATH = REPO_ROOT / "tests" / "test_rl_j1b_operational_repair_preflight.py"
READINESS_DIR = (
    RUNS_ROOT / "forensics" / "j1b_operational_repair_readiness_v1"
)
FUTURE_EXECUTION_ROOT = RUNS_ROOT / "forensics" / "j1b_execution_v1"

TEST_EVIDENCE_NAME = "J1B_TEST_EVIDENCE.json"
ROOT_CAUSE_AUDIT_NAME = "J1B_GENESIS_ROOT_CAUSE_AUDIT.json"
DENYLIST_NAME = "J1B_PROTECTED_STREAM_DENYLIST.json"
MANIFEST_NAME = "J1B_PROSPECTIVE_TRAINING_MANIFEST.json"
RUNTIME_AUDIT_NAME = "J1B_RUNTIME_ORCHESTRATION_AUDIT.json"
PROJECTION_NAME = "J1B_RUNTIME_STORAGE_PROJECTION.json"
SCHEMA_NAME = "J1B_SCHEMA.json"
READINESS_LOCK_NAME = "J1B_READINESS_LOCK.json"
READINESS_RESULT_NAME = "J1B_READINESS_RESULT.json"

EXPECTED_CHARTER_SHA256 = (
    "a426801fc3015051ea51517e925a7d1c2e556718e2551ee480b802c8a7422cc1"
)
EXPECTED_AMENDMENT_A1_SHA256 = (
    "64de3de37bff6a08bd95da217dc52d2f4bb58fbf99d28bede263a44d0aa2eb9c"
)
PRE_A1_HISTORY_PATH = (
    RUNS_ROOT
    / "forensics/j1b_operational_repair_preseal_history_v1/"
    "J1B_TEST_EVIDENCE_PRE_A1.json"
)
PRE_A1_HISTORY_FILE_SHA256 = (
    "d2f6333bd4fdbe584fbf231141a24c01256dcc9ebe0f57c2691e19a8f046bddf"
)
PRE_A1_HISTORY_PAYLOAD_SHA256 = (
    "b462c0b46afaa478caeb66c622799eb1e7a533673439a89fe0e60650a448e25e"
)
PARENT_SOURCE_IDENTITIES = {
    "threes_rl/J1_EXECUTION_SURFACE_CHARTER.md":
        "468cc181c32a934fcbc64bb4cadc22758bd0fc46870f0f120f9ac6008ddb696a",
    "threes_rl/j1_execution_surface.py":
        "d4367d95aba05ec592310008bae21e7de90905fa1268601dd60cc8fcb2b6f2bd",
    "tests/test_rl_j1_execution_surface.py":
        "cb696e4502d61abd7a24d5781d7c15e2dd8a0ffed538480ecbd2a27434a339cf",
    "threes_rl/J1_NORMAL_START_JOINT_POLICY_VALUE_PROPOSAL.md":
        "26b225c282fb4b58e11484210cf1f45de273714b1b35054f8670081032980bb2",
    "threes_rl/J1A_OUTCOME_FREE_COST_POWER_AMENDMENT.md":
        "d738a55bb438ee87d59d2433466e813cfd0a9fb5f041cbc3cc807d4bbafa2e11",
    "threes_rl/j1_joint_policy_value.py":
        "55d9e3206c2905509466c4962006e6cf3426f76647af6d2e60afe674b80c9bfe",
    "threes_rl/j1a_cost_power_preflight.py":
        "27ffb3825d60bd8ca4ec0646f976e325c2a7c5f00a077aea3803544531fe6a98",
    "tests/test_rl_j1_joint_policy_value.py":
        "e6b169f2d629021f96315380a3cf0ff6eece94a30e5027b1ace4d741499fbfa4",
    "tests/test_rl_j1a_cost_power_preflight.py":
        "898f25aa4ed109db2c9fc27b4bba9d7e9641dc57834e4e02d7a8242df195eb59",
}

PARENT_READINESS_DIR = (
    RUNS_ROOT / "forensics" / "j1_execution_surface_readiness_v1"
)
PARENT_READINESS_IDENTITIES = {
    "J1_EXECUTION_TEST_EVIDENCE.json":
        "465d1c4a00f91e3e614cd496ad3260236ecbb3106dc0b69e3a12a38380ff89b6",
    "J1_PROSPECTIVE_MANIFEST.json":
        "2aee68a08325cdbc5e42153942079c1375163f2b88217bf407e64fd95f096dce",
    "J1_EXECUTION_READINESS_LOCK.json":
        "e7f648eb04d7d197a9a2391352f82af5df6a12f7868ced8c8e9559703adb9fdc",
    "J1_EXECUTION_RUNTIME_STORAGE_PROJECTION.json":
        "92dfc49a8f0830a4b39c627d9257e4a20b4ca504019c455b3b2b1eb05a959f20",
    "J1_EXECUTION_READINESS_RESULT.json":
        "ba3e9d67c64b89cf583c2ad1778b073a6a702c003bf1a895c164d6f9f984d4f6",
    "J1_EXECUTION_SCHEMA.json":
        "0c9bd38e5cbccd840bbc4aed575b6e1dd95aa9516ed1f9431b87ff5f93d13730",
}
PARENT_READINESS_PAYLOADS = {
    "J1_EXECUTION_TEST_EVIDENCE.json": (
        "test_evidence_payload_sha256",
        "3982fb9bace0fe1ac73610804445df19c53bcf0944bb1ba81a70ec9cdc3738d7",
    ),
    "J1_PROSPECTIVE_MANIFEST.json": (
        "prospective_manifest_payload_sha256",
        "de0046a2121138659dd2fd0bb46a48081d80842c5d24334d1a683dbf0a9a7093",
    ),
    "J1_EXECUTION_READINESS_LOCK.json": (
        "readiness_lock_payload_sha256",
        "70c83f640632ec034b346cda355c875f79cc002409474d537ac67a6ab7c975cc",
    ),
    "J1_EXECUTION_RUNTIME_STORAGE_PROJECTION.json": (
        "projection_payload_sha256",
        "60e9697e82409e5ea930b7b07d2ab042ca3b28ecebff4bc6c2058f8b04e9f6ce",
    ),
    "J1_EXECUTION_READINESS_RESULT.json": (
        "readiness_result_payload_sha256",
        "af5525b35ec5d5c0deab88d1ec00d8215fbb4dc14abb2aaa8dc9aa70b27d556c",
    ),
    "J1_EXECUTION_SCHEMA.json": (
        "schema_sha256",
        "d58082c26ecdcca641531c71198307bc65b997e5c27dd647008b96e6ca6ac681",
    ),
}

ORIGINAL_EXECUTION_DIR = (
    RUNS_ROOT / "forensics" / "j1_execution_v1" / "training"
)
ORIGINAL_EXECUTION_IDENTITIES = {
    "commit_journals/000000000_aeebad4a796fcc2e15dc.json":
        "6ec867d1db215be0c095e19f5c9c399502bd48b971508a0e477d69a66a12e656",
    "commit_heads/000000000_aeebad4a796fcc2e15dc.json":
        "585cb70637dfaf398ef640c118e37de1061c53777cd04162f34aec2062deaa50",
    "phase_lock.json":
        "df2c9a27c9ae05212ba92518121413be548946ebed93a3dd91cfbedc12120c0c",
    "execution_opened.json":
        "fe0f53d92c9f21352da86ba318af74f9d1f3362c84d5010fa98d31e1d5d45079",
    "terminal_result.json":
        "21092fb34631eb0eaf48811caa814ff4d05abbb23c9bc5add85eefd93a8959d3",
    "retention_manifest.json":
        "dc339aafdbe32859d07c591a36c9088afa53f5be30412f3340049ca18994ceb0",
    "stream_consumption_opened.json":
        "0e278b1b920430a9efa729f20cb8a91e74de28df5d8f6e9c791c48229d335cb3",
    "writer_owner.json":
        "ad4b70a5615612cc0e64ded4af0f314f71c1033d5f0ca3889ced2d3e97d5762b",
    "stream_reservation.json":
        "025755cc0cf0b93991fd7b49fec62442aad1d90780046c82428c23cf139f3f73",
    "phase_lock_result.json":
        "6a9175f30635cbedc7a91afc7ed2578dacce77a151cea0e3522c68d05a0af926",
    "commit_states/000000000_aeebad4a796fcc2e15dc.bin":
        "7b4b6878f3aacafa547a63cd3c4ee7b68cfc501e86443edd51f844e93e543a08",
    "commit_head.json":
        "2ffa6f3bf7b5772718d7a3e8d683a15412387f4f0259f40c88baa370e698f774",
    "root_manifest.json":
        "479487701230af128c4a1cff3aea49a29f59330efcb9ce7eafb9324abc455f0c",
    "commit_records/000000000_aeebad4a796fcc2e15dc.json":
        "38dce0fcabfe3bef4c880cf649868a57db808847280448e7b821a430293258ce",
}
ORIGINAL_TERMINAL_PAYLOAD_SHA256 = (
    "9bcc81d217141fdfa801d1fca606c356720e4ac5c0e2a26f9d1ab688ca93dbcf"
)
ORIGINAL_RETENTION_PAYLOAD_SHA256 = (
    "11cc89c6a6fe41ff74c472e3fa0b61d179e1cedfa4755cc4f13fe7ced44018b2"
)
ORIGINAL_RETENTION_INVENTORY_SHA256 = (
    "7233c65745a9ae7258dbb165b60f4ae55c1cf60376819b80bb9e0be17d677471"
)
ORIGINAL_MARKER_FILE_SHA256 = (
    "fe0f53d92c9f21352da86ba318af74f9d1f3362c84d5010fa98d31e1d5d45079"
)
ORIGINAL_PHASE_LOCK_FILE_SHA256 = (
    "df2c9a27c9ae05212ba92518121413be548946ebed93a3dd91cfbedc12120c0c"
)
ORIGINAL_EXECUTE_COMMAND = (
    "nice -n 10 env PYTHONPATH=. .venv/bin/python -m "
    "threes_rl.j1_execution_surface execute --phase training "
    "--execution-root threes_rl/runs/forensics/j1_execution_v1 "
    "--readiness-dir "
    "threes_rl/runs/forensics/j1_execution_surface_readiness_v1 --jobs 1"
)

PARENT_DENYLIST_PATH = (
    RUNS_ROOT
    / "forensics/j1_implementation_preflight_v1/"
    "J1_PROTECTED_ID_DENYLIST.json"
)
PARENT_DENYLIST_FILE_SHA256 = (
    "0a7be318ebe5281a11ded38f3bbde29745ccb7c3a969585de1788df468fbd763"
)
PARENT_DENYLIST_PAYLOAD_SHA256 = (
    "22731c89df661419d7ca2bcffdb86240f2ad8974b00e765dd715cf8f4e675add"
)

TRAIN_ROOTS = 16_384
FRESH_STREAMS = {
    "logical": 213_000_016_384,
    "deck": 214_000_016_384,
    "slot": 215_000_016_384,
    "candidate_policy": 216_000_016_384,
}
ORIGINAL_CONSUMED_STREAMS = {
    "logical": (213_000_000_000, 213_000_016_383),
    "deck": (214_000_000_000, 214_000_016_383),
    "slot": (215_000_000_000, 215_000_016_383),
    "candidate_policy": (216_000_000_000, 216_000_016_383),
}
J1B_PHASE_NONCE = EXPECTED_CHARTER_SHA256
EXPECTED_PARAMETER_COUNT = 411_656
EXPECTED_LEGACY_INTEROP_THREADS = 12

ZERO_WORK = {
    "j1b_phase_locks": 0,
    "j1b_markers": 0,
    "j1b_owners": 0,
    "j1b_streams_reserved": 0,
    "j1b_streams_consumed": 0,
    "j1b_genesis_commits": 0,
    "normal_start_games": 0,
    "scientific_labels": 0,
    "scientific_optimizer_steps": 0,
    "scientific_checkpoints": 0,
    "development_content_reads": 0,
    "confirmation_content_reads": 0,
    "score_or_policy_outcomes": 0,
    "human_session_reads": 0,
    "incumbent_changes": 0,
    "dashboard_changes": 0,
    "promotion_actions": 0,
}


class J1bIntegrityError(RuntimeError):
    """An immutable J1b identity or fail-closed contract changed."""


class J1bOperationalHold(RuntimeError):
    """A mutable runtime, process, service, or storage gate failed."""


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def canonical_json_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def payload_with_hash(
    payload: Mapping[str, Any],
    field: str,
) -> dict[str, Any]:
    body = dict(payload)
    body.pop(field, None)
    body[field] = canonical_json_hash(body)
    return body


def verify_payload_hash(payload: Mapping[str, Any], field: str) -> bool:
    body = dict(payload)
    observed = body.pop(field, None)
    return (
        isinstance(observed, str)
        and observed == canonical_json_hash(body)
    )


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise J1bIntegrityError(f"Expected JSON object: {path}")
    return payload


def write_immutable_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    field: str,
) -> dict[str, Any]:
    body = payload_with_hash(payload, field)
    serialized = (json.dumps(body, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    if not verify_payload_hash(json.loads(serialized), field):
        raise J1bIntegrityError(f"JSON reload instability: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != serialized:
            raise J1bIntegrityError(
                f"Immutable artifact collision changed bytes: {path}"
            )
        raise FileExistsError(f"Immutable artifact exists: {path}")
    temporary = path.with_name(
        f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}"
    )
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            if path.read_bytes() != serialized:
                raise J1bIntegrityError(
                    f"Concurrent immutable artifact mismatch: {path}"
                ) from error
            raise FileExistsError(
                f"Immutable artifact won by another writer: {path}"
            ) from error
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    observed = load_json(path)
    if not verify_payload_hash(observed, field):
        raise J1bIntegrityError(f"Written payload is invalid: {path}")
    return observed


def artifact_identity(path: Path, *, payload_field: str) -> dict[str, Any]:
    payload = load_json(path)
    if not verify_payload_hash(payload, payload_field):
        raise J1bIntegrityError(f"Artifact payload is invalid: {path}")
    return {
        "path": str(path.resolve()),
        "file_sha256": sha256_path(path),
        "payload_field": payload_field,
        "payload_sha256": payload[payload_field],
    }


def _hash_audit(
    root: Path,
    expected: Mapping[str, str],
) -> dict[str, Any]:
    rows = {}
    for relative, expected_sha in expected.items():
        path = root / relative
        observed = sha256_path(path) if path.is_file() else None
        rows[relative] = {
            "expected_sha256": expected_sha,
            "observed_sha256": observed,
            "matches": observed == expected_sha,
        }
    return {
        "rows": rows,
        "passes": all(row["matches"] for row in rows.values()),
    }


def parent_identity_audit() -> dict[str, Any]:
    sources = _hash_audit(REPO_ROOT, PARENT_SOURCE_IDENTITIES)
    readiness_files = _hash_audit(
        PARENT_READINESS_DIR,
        PARENT_READINESS_IDENTITIES,
    )
    readiness_payloads = {}
    for name, (field, expected) in PARENT_READINESS_PAYLOADS.items():
        path = PARENT_READINESS_DIR / name
        try:
            payload = load_json(path)
            observed = payload.get(field)
            stable = verify_payload_hash(payload, field)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            observed = None
            stable = False
        readiness_payloads[name] = {
            "field": field,
            "expected_sha256": expected,
            "observed_sha256": observed,
            "reload_stable": stable,
            "matches": stable and observed == expected,
        }
    payloads_pass = all(
        row["matches"] for row in readiness_payloads.values()
    )
    return {
        "sources": sources,
        "readiness_files": readiness_files,
        "readiness_payloads": readiness_payloads,
        "checks": {
            "parent_sources_exact": sources["passes"],
            "parent_readiness_files_exact": readiness_files["passes"],
            "parent_readiness_payloads_exact": payloads_pass,
        },
        "passes": sources["passes"] and readiness_files["passes"]
        and payloads_pass,
    }


def pre_a1_history_audit() -> dict[str, Any]:
    try:
        payload = load_json(PRE_A1_HISTORY_PATH)
        payload_stable = verify_payload_hash(
            payload,
            "test_evidence_payload_sha256",
        )
        observed_payload = payload.get("test_evidence_payload_sha256")
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        payload_stable = False
        observed_payload = None
    observed_file = (
        sha256_path(PRE_A1_HISTORY_PATH)
        if PRE_A1_HISTORY_PATH.is_file()
        else None
    )
    checks = {
        "amendment_a1_exact": (
            sha256_path(AMENDMENT_A1_PATH)
            == EXPECTED_AMENDMENT_A1_SHA256
        ),
        "historical_file_exact": (
            observed_file == PRE_A1_HISTORY_FILE_SHA256
        ),
        "historical_payload_exact": (
            payload_stable
            and observed_payload == PRE_A1_HISTORY_PAYLOAD_SHA256
        ),
        "historical_namespace_separate": (
            PRE_A1_HISTORY_PATH.parent != READINESS_DIR
        ),
    }
    return {
        "path": str(PRE_A1_HISTORY_PATH.resolve()),
        "file_sha256": observed_file,
        "payload_sha256": observed_payload,
        "checks": checks,
        "passes": all(checks.values()),
    }


def original_execution_identity_audit() -> dict[str, Any]:
    files = _hash_audit(
        ORIGINAL_EXECUTION_DIR,
        ORIGINAL_EXECUTION_IDENTITIES,
    )
    actual_files = sorted(
        str(path.relative_to(ORIGINAL_EXECUTION_DIR))
        for path in ORIGINAL_EXECUTION_DIR.rglob("*")
        if path.is_file()
    )
    terminal = load_json(ORIGINAL_EXECUTION_DIR / "terminal_result.json")
    retention = load_json(ORIGINAL_EXECUTION_DIR / "retention_manifest.json")
    checks = {
        "all_original_files_exact": files["passes"],
        "exact_file_set": (
            actual_files == sorted(ORIGINAL_EXECUTION_IDENTITIES)
        ),
        "terminal_payload_exact": (
            verify_payload_hash(
                terminal,
                "terminal_result_payload_sha256",
            )
            and terminal.get("terminal_result_payload_sha256")
            == ORIGINAL_TERMINAL_PAYLOAD_SHA256
        ),
        "terminal_hold_exact": (
            terminal.get("decision") == "HOLD_J1_OPERATIONAL"
            and terminal.get("failure_class") == "operational"
            and terminal.get("error_type") == "J1ExecutionOperationalHold"
        ),
        "retention_payload_exact": (
            verify_payload_hash(retention, "retention_payload_sha256")
            and retention.get("retention_payload_sha256")
            == ORIGINAL_RETENTION_PAYLOAD_SHA256
        ),
        "retention_inventory_exact": (
            retention.get("file_inventory_sha256")
            == ORIGINAL_RETENTION_INVENTORY_SHA256
        ),
    }
    return {
        "files": files,
        "actual_files": actual_files,
        "terminal": {
            "decision": terminal.get("decision"),
            "file_sha256": sha256_path(
                ORIGINAL_EXECUTION_DIR / "terminal_result.json"
            ),
            "payload_sha256": terminal.get(
                "terminal_result_payload_sha256"
            ),
        },
        "retention": {
            "file_sha256": sha256_path(
                ORIGINAL_EXECUTION_DIR / "retention_manifest.json"
            ),
            "payload_sha256": retention.get("retention_payload_sha256"),
            "file_inventory_sha256": retention.get(
                "file_inventory_sha256"
            ),
        },
        "checks": checks,
        "passes": all(checks.values()),
    }


def establish_torch_runtime(
    *,
    torch_module: Any | None = None,
) -> dict[str, Any]:
    torch = (
        importlib.import_module("torch")
        if torch_module is None
        else torch_module
    )
    try:
        torch.set_num_interop_threads(1)
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
    except Exception as error:
        raise J1bOperationalHold(
            "PyTorch runtime configuration could not be established"
        ) from error
    checks = {
        "one_torch_interop_thread": (
            int(torch.get_num_interop_threads()) == 1
        ),
        "one_torch_intraop_thread": int(torch.get_num_threads()) == 1,
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
    }
    if not all(checks.values()):
        raise J1bOperationalHold(
            "PyTorch runtime configuration did not verify exactly"
        )
    return {
        "torch_num_interop_threads": int(
            torch.get_num_interop_threads()
        ),
        "torch_num_threads": int(torch.get_num_threads()),
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "checks": checks,
        "passes": True,
    }


def guarded_runtime_entrypoint(
    *,
    phase_dir: Path,
    torch_module: Any | None = None,
    parent_loader: Any | None = None,
    model_initializer: Any | None = None,
    operational_audit: Any | None = None,
    after_guard: Any | None = None,
) -> dict[str, Any]:
    ordering = []
    ordering.append("configure_torch_runtime")
    runtime = establish_torch_runtime(torch_module=torch_module)
    ordering.append("import_parent")
    parent = (
        importlib.import_module("threes_rl.j1_execution_surface")
        if parent_loader is None
        else parent_loader()
    )
    ordering.append("initialize_frozen_model_optimizer")
    if model_initializer is None:
        model, optimizer = parent.j1.initialize_model_optimizer()
    else:
        model, optimizer = model_initializer(parent)
    ordering.append("first_unchanged_operational_guard")
    audit = (
        parent.default_phase_operational_audit(
            phase_dir=phase_dir,
            phase="training",
            active_seconds=0.0,
            require_target_disk=True,
        )
        if operational_audit is None
        else operational_audit(parent, phase_dir)
    )
    if audit.get("passes") is not True:
        raise J1bOperationalHold(
            "The unchanged first training operational guard did not pass"
        )
    ordering.append("guard_passed_before_scientific_artifacts")
    after = None
    if after_guard is not None:
        ordering.append("after_guard_callback")
        after = after_guard(parent, model, optimizer)
    return {
        "runtime": runtime,
        "operational_audit": audit,
        "ordering": ordering,
        "after_guard": after,
        "passes": True,
    }


def _root_cause_probe() -> dict[str, Any]:
    pre_import = {
        "torch_loaded": "torch" in sys.modules,
        "parent_loaded": "threes_rl.j1_execution_surface" in sys.modules,
    }
    parent = importlib.import_module("threes_rl.j1_execution_surface")
    model, _optimizer = parent.j1.initialize_model_optimizer()
    runtime = {
        "torch_num_interop_threads": int(
            parent.torch.get_num_interop_threads()
        ),
        "torch_num_threads": int(parent.torch.get_num_threads()),
        "deterministic_algorithms": bool(
            parent.torch.are_deterministic_algorithms_enabled()
        ),
    }
    boundary = parent.verify_commit_boundary(
        phase_dir=ORIGINAL_EXECUTION_DIR,
        phase="training",
        marker_file_sha256=ORIGINAL_MARKER_FILE_SHA256,
        phase_lock_file_sha256=ORIGINAL_PHASE_LOCK_FILE_SHA256,
        command=ORIGINAL_EXECUTE_COMMAND,
        execution_mode="scientific",
    )
    state = parent.load_atomic_binary(Path(boundary["state_path"]))
    resource = dict(state.get("resource_clock", {}))
    zero_counts = {
        "completed_roots": len(state.get("all_completed_root_ids", [])),
        "attempts_started": int(resource.get("attempts_started", -1)),
        "attempts_finished": int(resource.get("attempts_finished", -1)),
        "attempts_abandoned": int(resource.get("attempts_abandoned", -1)),
        "optimizer_steps": len(state.get("optimizer_step_ids", [])),
        "round_aggregates": len(state.get("round_aggregates", [])),
    }
    checks = {
        "fresh_process_before_parent_import": not any(pre_import.values()),
        "genesis_sequence_zero": (
            boundary.get("sequence") == 0
            and boundary.get("unit_id") == "genesis"
        ),
        "full_chain_verified": (
            boundary.get("chain_audit", {}).get(
                "full_predecessor_chain_verified"
            )
            is True
        ),
        "all_scientific_counts_zero": all(
            value == 0 for value in zero_counts.values()
        ),
        "legacy_deterministic_true": (
            runtime["deterministic_algorithms"] is True
        ),
        "legacy_intraop_one": runtime["torch_num_threads"] == 1,
        "legacy_interop_twelve": (
            runtime["torch_num_interop_threads"]
            == EXPECTED_LEGACY_INTEROP_THREADS
        ),
        "frozen_parameter_count": (
            parent.j1.parameter_count(model) == EXPECTED_PARAMETER_COUNT
        ),
    }
    return {
        "version": f"{VERSION}_root_cause_probe_v1",
        "pre_import": pre_import,
        "runtime": runtime,
        "zero_counts": zero_counts,
        "commit_boundary": {
            "sequence": boundary.get("sequence"),
            "unit_id": boundary.get("unit_id"),
            "chain_audit": boundary.get("chain_audit"),
            "state_file_sha256": boundary.get("state_file_sha256"),
        },
        "model": {
            "parameter_count": parent.j1.parameter_count(model),
            "schema_sha256": parent.j1.model_schema_sha256(),
            "initial_state_sha256": parent.j1.stable_hash(
                model.state_dict()
            ),
        },
        "checks": checks,
        "passes": all(checks.values()),
    }


def _runtime_probe(
    *,
    phase_dir: Path,
    future_execution_root: Path,
) -> dict[str, Any]:
    pre_import = {
        "torch_loaded": "torch" in sys.modules,
        "parent_loaded": "threes_rl.j1_execution_surface" in sys.modules,
    }
    future_absent_before = not future_execution_root.exists()

    def after_guard(parent: Any, model: Any, _optimizer: Any) -> dict[str, Any]:
        return {
            "parameter_count": parent.j1.parameter_count(model),
            "schema_sha256": parent.j1.model_schema_sha256(),
            "initial_state_sha256": parent.j1.stable_hash(
                model.state_dict()
            ),
        }

    result = guarded_runtime_entrypoint(
        phase_dir=phase_dir,
        after_guard=after_guard,
    )
    future_absent_after = not future_execution_root.exists()
    checks = {
        "fresh_process_before_torch_or_parent_import": (
            not any(pre_import.values())
        ),
        "future_execution_absent_before": future_absent_before,
        "future_execution_absent_after": future_absent_after,
        "interop_configured_before_parent_import": (
            result["ordering"][:2]
            == ["configure_torch_runtime", "import_parent"]
        ),
        "first_real_operational_guard_passed": (
            result["operational_audit"].get("passes") is True
        ),
        "frozen_parameter_count": (
            result["after_guard"]["parameter_count"]
            == EXPECTED_PARAMETER_COUNT
        ),
        "no_owner_reservation_consumption_or_genesis": (
            not future_execution_root.exists()
        ),
    }
    return {
        "version": f"{VERSION}_runtime_probe_v1",
        "pre_import": pre_import,
        "runtime": result["runtime"],
        "operational_audit": result["operational_audit"],
        "ordering": result["ordering"],
        "model": result["after_guard"],
        "scientific_artifacts": {
            "owners": 0,
            "stream_reservations": 0,
            "stream_consumptions": 0,
            "genesis_commits": 0,
            "games": 0,
            "optimizer_steps": 0,
        },
        "checks": checks,
        "passes": all(checks.values()),
    }


def _run_json_subprocess(
    arguments: Sequence[str],
    *,
    nice_10: bool,
) -> dict[str, Any]:
    command = [sys.executable, "-m", "threes_rl.j1b_operational_repair_preflight"]
    command.extend(arguments)
    if nice_10:
        nice_path = shutil.which("nice")
        if nice_path is None:
            return {
                "command": command,
                "returncode": None,
                "stdout": "",
                "stderr": "nice executable is unavailable",
                "payload": None,
                "passes": False,
            }
        command = [nice_path, "-n", "10", *command]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        payload = json.loads(completed.stdout)
    except (json.JSONDecodeError, TypeError):
        payload = None
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "payload": payload,
        "passes": (
            completed.returncode == 0
            and isinstance(payload, dict)
            and payload.get("passes") is True
        ),
    }


def iter_fresh_stream_rows() -> Iterable[dict[str, Any]]:
    for row_index in range(TRAIN_ROOTS):
        row = {
            "phase": "training",
            "partition": "train",
            "row_index": row_index,
            "block": row_index % 8,
            "logical_stream_id": FRESH_STREAMS["logical"] + row_index,
            "deck_stream_id": FRESH_STREAMS["deck"] + row_index,
            "slot_stream_id": FRESH_STREAMS["slot"] + row_index,
            "candidate_policy_stream_id": (
                FRESH_STREAMS["candidate_policy"] + row_index
            ),
            "control_policy_stream_id": None,
            "arm_count": 1,
            "starter_tile": None,
        }
        row["row_commitment_sha256"] = canonical_json_hash(row)
        yield row


def ordered_rows_hash(rows: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(canonical_json_bytes(row))
        digest.update(b"\n")
    return digest.hexdigest()


def j1b_root_commitment() -> dict[str, Any]:
    rows_sha = ordered_rows_hash(iter_fresh_stream_rows())
    payload = {
        "version": f"{VERSION}_training_marker_root_commitment_v1",
        "root_identity_version": "accepted_j1_marker_payload_root_v1",
        "phase": "training",
        "partition": "train",
        "phase_nonce": J1B_PHASE_NONCE,
        "canonical_rows_sha256": rows_sha,
        "row_count": TRAIN_ROOTS,
        "j1b_charter_file_sha256": EXPECTED_CHARTER_SHA256,
        "parent_readiness_result_file_sha256":
            PARENT_READINESS_IDENTITIES[
                "J1_EXECUTION_READINESS_RESULT.json"
            ],
        "spent_j1_terminal_file_sha256":
            ORIGINAL_EXECUTION_IDENTITIES["terminal_result.json"],
        "operational_activation_fields_excluded": [
            "activation_command",
            "activation_hostname",
            "activation_opened_at",
            "service_evidence",
            "process_evidence",
            "storage_evidence",
        ],
        "streams_reserved": 0,
        "streams_consumed": 0,
    }
    return payload_with_hash(payload, "marker_payload_sha256")


def root_id_for_commitment(
    commitment: Mapping[str, Any],
    row: Mapping[str, Any],
) -> str:
    if not verify_payload_hash(commitment, "marker_payload_sha256"):
        raise J1bIntegrityError("J1b root commitment hash is invalid")
    if (
        commitment.get("phase") != row.get("phase")
        or commitment.get("partition") != row.get("partition")
    ):
        raise J1bIntegrityError("J1b root commitment/row mismatch")
    return canonical_json_hash(
        {
            "marker_payload_sha256": commitment[
                "marker_payload_sha256"
            ],
            "partition": commitment["partition"],
            "row": int(row["row_index"]),
            "logical_stream_id": int(row["logical_stream_id"]),
            "deck_stream_id": int(row["deck_stream_id"]),
            "slot_stream_id": int(row["slot_stream_id"]),
        }
    )


def prospective_training_manifest() -> dict[str, Any]:
    commitment = j1b_root_commitment()
    rows = []
    for base in iter_fresh_stream_rows():
        root_id = root_id_for_commitment(commitment, base)
        rows.append(
            {
                **base,
                "root_id": root_id,
                "ancestry_id": root_id,
            }
        )
    root_ids = [row["root_id"] for row in rows]
    ancestry_ids = [row["ancestry_id"] for row in rows]
    stream_fields = (
        "logical_stream_id",
        "deck_stream_id",
        "slot_stream_id",
        "candidate_policy_stream_id",
    )
    stream_ids = [
        int(row[field])
        for row in rows
        for field in stream_fields
    ]
    checks = {
        "row_count_exact": len(rows) == TRAIN_ROOTS,
        "root_ids_unique": len(set(root_ids)) == TRAIN_ROOTS,
        "ancestries_unique": len(set(ancestry_ids)) == TRAIN_ROOTS,
        "one_root_per_ancestry": root_ids == ancestry_ids,
        "starter_none": all(row["starter_tile"] is None for row in rows),
        "one_arm": all(row["arm_count"] == 1 for row in rows),
        "stream_role_ids_globally_unique": (
            len(stream_ids) == len(set(stream_ids)) == TRAIN_ROOTS * 4
        ),
        "fresh_bases_exact": all(
            int(rows[0][f"{role}_stream_id"])
            == FRESH_STREAMS[role]
            for role in FRESH_STREAMS
        ),
        "fresh_ends_exact": all(
            int(rows[-1][f"{role}_stream_id"])
            == FRESH_STREAMS[role] + TRAIN_ROOTS - 1
            for role in FRESH_STREAMS
        ),
    }
    payload = {
        "version": f"{VERSION}_prospective_training_manifest_v1",
        "phase": "training",
        "partition": "train",
        "root_commitment": commitment,
        "rows": rows,
        "canonical_rows_sha256": ordered_rows_hash(rows),
        "role_counts": {
            "roots": TRAIN_ROOTS,
            "ancestries": TRAIN_ROOTS,
            "logical": TRAIN_ROOTS,
            "deck": TRAIN_ROOTS,
            "slot": TRAIN_ROOTS,
            "candidate_policy": TRAIN_ROOTS,
        },
        "checks": checks,
        "streams_reserved": 0,
        "streams_consumed": 0,
        "passes": all(checks.values()),
    }
    return payload_with_hash(
        payload,
        "prospective_manifest_payload_sha256",
    )


def _intervals_overlap(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> bool:
    return not (
        int(left["end_inclusive"]) < int(right["start"])
        or int(right["end_inclusive"]) < int(left["start"])
    )


def protected_stream_denylist(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    parent = load_json(PARENT_DENYLIST_PATH)
    if (
        sha256_path(PARENT_DENYLIST_PATH)
        != PARENT_DENYLIST_FILE_SHA256
        or not verify_payload_hash(parent, "denylist_payload_sha256")
        or parent.get("denylist_payload_sha256")
        != PARENT_DENYLIST_PAYLOAD_SHA256
    ):
        raise J1bIntegrityError("Parent protected denylist identity changed")
    denied = [
        {
            "kind": "historical_ceiling",
            "role": "all",
            "start": int(
                parent["historical_denied_interval"]["start"]
            ),
            "end_inclusive": int(
                parent["historical_denied_interval"]["end_inclusive"]
            ),
        }
    ]
    for row in parent["prospective_stream_contract"][
        "prospective_intervals"
    ]:
        denied.append(
            {
                "kind": "accepted_parent_prospective_interval",
                "role": str(row["stream_role"]),
                "partition": str(row["partition"]),
                "start": int(row["base"]),
                "end_inclusive": int(row["end_inclusive"]),
            }
        )
    fresh = [
        {
            "kind": "j1b_fresh_training_interval",
            "role": role,
            "partition": "train",
            "start": base,
            "end_inclusive": base + TRAIN_ROOTS - 1,
        }
        for role, base in FRESH_STREAMS.items()
    ]
    role_alias = {
        "candidate_policy": "candidate_policy",
        "logical": "logical",
        "deck": "deck",
        "slot": "slot",
    }
    collision_rows = []
    for fresh_row in fresh:
        for denied_row in denied:
            role_matches = (
                denied_row["role"] == "all"
                or denied_row["role"] == role_alias[fresh_row["role"]]
            )
            if role_matches and _intervals_overlap(fresh_row, denied_row):
                collision_rows.append(
                    {
                        "fresh": fresh_row,
                        "denied": denied_row,
                    }
                )
    original_manifest = load_json(
        ORIGINAL_EXECUTION_DIR / "root_manifest.json"
    )
    original_rows = original_manifest["rows"]
    actual_old = {
        role: {
            int(row[f"{role}_stream_id"]) for row in original_rows
        }
        for role in FRESH_STREAMS
    }
    actual_new = {
        role: {
            int(row[f"{role}_stream_id"])
            for row in manifest["rows"]
        }
        for role in FRESH_STREAMS
    }
    original_actual_intersections = {
        role: len(actual_old[role] & actual_new[role])
        for role in FRESH_STREAMS
    }
    checks = {
        "parent_denylist_exact": True,
        "parent_protected_payloads_not_parsed": (
            parent.get("protected_payloads_parsed") is False
        ),
        "historical_schema_discovery_not_used": (
            parent.get("historical_schema_discovery_used") is False
        ),
        "fresh_ranges_contiguous": all(
            row["end_inclusive"] - row["start"] + 1 == TRAIN_ROOTS
            for row in fresh
        ),
        "fresh_ranges_start_after_spent_parent_prefix": all(
            FRESH_STREAMS[role]
            == ORIGINAL_CONSUMED_STREAMS[role][1] + 1
            for role in FRESH_STREAMS
        ),
        "zero_denied_interval_collisions": not collision_rows,
        "zero_actual_original_row_collisions": all(
            value == 0 for value in original_actual_intersections.values()
        ),
        "manifest_unreserved": manifest.get("streams_reserved") == 0,
        "manifest_unconsumed": manifest.get("streams_consumed") == 0,
    }
    payload = {
        "version": f"{VERSION}_protected_stream_denylist_v1",
        "method": (
            "exact accepted compact intervals plus exact spent J1 root "
            "manifest stream identities; no schema-heterogeneous scan"
        ),
        "parent_denylist": {
            "path": str(PARENT_DENYLIST_PATH.resolve()),
            "file_sha256": PARENT_DENYLIST_FILE_SHA256,
            "payload_sha256": PARENT_DENYLIST_PAYLOAD_SHA256,
        },
        "spent_j1_manifest": {
            "path": str(
                (ORIGINAL_EXECUTION_DIR / "root_manifest.json").resolve()
            ),
            "file_sha256": ORIGINAL_EXECUTION_IDENTITIES[
                "root_manifest.json"
            ],
            "payload_sha256": original_manifest.get(
                "root_manifest_payload_sha256"
            ),
            "row_count": len(original_rows),
        },
        "denied_intervals": denied,
        "fresh_intervals": fresh,
        "collision_rows": collision_rows,
        "original_actual_intersections": original_actual_intersections,
        "checks": checks,
        "passes": all(checks.values()),
    }
    return payload_with_hash(payload, "denylist_payload_sha256")


def runtime_storage_projection() -> dict[str, Any]:
    parent_path = (
        PARENT_READINESS_DIR
        / "J1_EXECUTION_RUNTIME_STORAGE_PROJECTION.json"
    )
    parent = load_json(parent_path)
    checks = {
        "parent_file_exact": (
            sha256_path(parent_path)
            == PARENT_READINESS_IDENTITIES[
                "J1_EXECUTION_RUNTIME_STORAGE_PROJECTION.json"
            ]
        ),
        "parent_payload_exact": (
            verify_payload_hash(parent, "projection_payload_sha256")
            and parent.get("projection_payload_sha256")
            == PARENT_READINESS_PAYLOADS[
                "J1_EXECUTION_RUNTIME_STORAGE_PROJECTION.json"
            ][1]
        ),
        "parent_projection_passes": parent.get("passes") is True,
        "training_roots_unchanged": TRAIN_ROOTS == 16_384,
        "training_runtime_cap_unchanged": (
            parent["training"]["central"]["runtime_cap_hours"] == 72.0
        ),
        "training_storage_cap_unchanged": (
            parent["training"]["central"]["storage"]["cap_gib"] == 24.0
        ),
        "fixed_margin_unchanged": parent.get("safety_multiplier") == 1.25,
        "no_retiming": True,
    }
    payload = {
        "version": f"{VERSION}_runtime_storage_projection_v1",
        "method": (
            "byte-identical accepted parent bounded-fixture projection; "
            "J1b changes only pre-import Torch runtime orchestration"
        ),
        "parent_projection": {
            "path": str(parent_path.resolve()),
            "file_sha256": sha256_path(parent_path),
            "payload_sha256": parent.get("projection_payload_sha256"),
        },
        "training_central": parent["training"]["central"],
        "training_sensitivity_5000_moves":
            parent["training"]["sensitivity_5000_moves"],
        "retirement_contract": parent["retirement_contract"],
        "checks": checks,
        "zero_work": ZERO_WORK,
        "passes": all(checks.values()),
    }
    return payload_with_hash(payload, "projection_payload_sha256")


def schema_payload() -> dict[str, Any]:
    payload = {
        "version": f"{VERSION}_schema_v1",
        "sole_semantic_repair": {
            "before_parent_import": True,
            "torch_num_interop_threads": 1,
            "torch_num_threads": 1,
            "deterministic_algorithms": True,
            "guard_unchanged": True,
        },
        "scientific_contract": {
            "parameter_count": EXPECTED_PARAMETER_COUNT,
            "training_roots": TRAIN_ROOTS,
            "rounds": 64,
            "roots_per_round": 256,
            "synchronous_envs": 16,
            "starter_tile": None,
            "alternate_seed": False,
            "scientific_change": False,
        },
        "fresh_streams": {
            role: {
                "start": base,
                "end_inclusive": base + TRAIN_ROOTS - 1,
                "rows": TRAIN_ROOTS,
            }
            for role, base in FRESH_STREAMS.items()
        },
        "future_commands_in_this_version": [],
        "permitted_commands": [
            "write-test-evidence",
            "prepare",
        ],
        "hidden_read_only_probe_commands": [
            "_root-cause-probe",
            "_runtime-probe",
        ],
        "future_execution_root": str(FUTURE_EXECUTION_ROOT.resolve()),
        "zero_work": ZERO_WORK,
    }
    return payload_with_hash(payload, "schema_payload_sha256")


def write_test_evidence(
    *,
    readiness_dir: Path,
    py_compile_command: str,
    focused_command: str,
    focused_passed: int,
    parent_execution_command: str,
    parent_execution_passed: int,
    parent_j1_command: str,
    parent_j1_passed: int,
    parent_j1a_command: str,
    parent_j1a_passed: int,
    applicable_command: str,
    applicable_passed: int,
    documented_deselections: Sequence[str],
) -> dict[str, Any]:
    if FUTURE_EXECUTION_ROOT.exists():
        raise J1bIntegrityError(
            "Future J1b execution namespace must remain absent"
        )
    if readiness_dir.exists() and any(readiness_dir.iterdir()):
        raise FileExistsError(
            "J1b readiness namespace is not fresh for test evidence"
        )
    source_identities = {
        "charter_file_sha256": sha256_path(CHARTER_PATH),
        "amendment_a1_file_sha256": sha256_path(AMENDMENT_A1_PATH),
        "runner_file_sha256": sha256_path(RUNNER_PATH),
        "test_file_sha256": sha256_path(TEST_PATH),
    }
    if source_identities["charter_file_sha256"] != EXPECTED_CHARTER_SHA256:
        raise J1bIntegrityError("Frozen J1b charter identity changed")
    if (
        source_identities["amendment_a1_file_sha256"]
        != EXPECTED_AMENDMENT_A1_SHA256
    ):
        raise J1bIntegrityError("Frozen J1b A1 amendment identity changed")
    payload = {
        "version": f"{VERSION}_test_evidence_v1",
        "source_identities": source_identities,
        "commands": [
            {
                "name": "py_compile",
                "command": py_compile_command,
                "passed": True,
            },
            {
                "name": "focused_j1b",
                "command": focused_command,
                "passed": focused_passed,
            },
            {
                "name": "parent_execution_surface",
                "command": parent_execution_command,
                "passed": parent_execution_passed,
            },
            {
                "name": "parent_j1",
                "command": parent_j1_command,
                "passed": parent_j1_passed,
            },
            {
                "name": "parent_j1a",
                "command": parent_j1a_command,
                "passed": parent_j1a_passed,
            },
            {
                "name": "applicable_regressions",
                "command": applicable_command,
                "passed": applicable_passed,
                "documented_deselections": list(
                    documented_deselections
                ),
            },
        ],
        "documented_deselection_count": len(documented_deselections),
        "scientific_test_work": 0,
        "zero_work": ZERO_WORK,
        "passes": True,
    }
    return write_immutable_json(
        readiness_dir / TEST_EVIDENCE_NAME,
        payload,
        field="test_evidence_payload_sha256",
    )


def _test_evidence_audit(readiness_dir: Path) -> dict[str, Any]:
    path = readiness_dir / TEST_EVIDENCE_NAME
    payload = load_json(path)
    source = payload.get("source_identities", {})
    checks = {
        "payload_stable": verify_payload_hash(
            payload,
            "test_evidence_payload_sha256",
        ),
        "charter_exact": (
            source.get("charter_file_sha256")
            == sha256_path(CHARTER_PATH)
            == EXPECTED_CHARTER_SHA256
        ),
        "amendment_a1_exact": (
            source.get("amendment_a1_file_sha256")
            == sha256_path(AMENDMENT_A1_PATH)
            == EXPECTED_AMENDMENT_A1_SHA256
        ),
        "runner_exact": (
            source.get("runner_file_sha256") == sha256_path(RUNNER_PATH)
        ),
        "tests_exact": (
            source.get("test_file_sha256") == sha256_path(TEST_PATH)
        ),
        "all_commands_passed": all(
            (
                row.get("passed") is True
                if row.get("name") == "py_compile"
                else int(row.get("passed", -1)) > 0
            )
            for row in payload.get("commands", [])
        ),
        "zero_scientific_work": (
            payload.get("zero_work") == ZERO_WORK
            and payload.get("scientific_test_work") == 0
        ),
    }
    return {
        "identity": {
            "path": str(path.resolve()),
            "file_sha256": sha256_path(path),
            "payload_sha256": payload.get(
                "test_evidence_payload_sha256"
            ),
        },
        "checks": checks,
        "passes": all(checks.values()),
    }


def _readiness_paths(readiness_dir: Path) -> dict[str, Path]:
    return {
        "evidence": readiness_dir / TEST_EVIDENCE_NAME,
        "root_cause": readiness_dir / ROOT_CAUSE_AUDIT_NAME,
        "denylist": readiness_dir / DENYLIST_NAME,
        "manifest": readiness_dir / MANIFEST_NAME,
        "runtime": readiness_dir / RUNTIME_AUDIT_NAME,
        "projection": readiness_dir / PROJECTION_NAME,
        "schema": readiness_dir / SCHEMA_NAME,
        "lock": readiness_dir / READINESS_LOCK_NAME,
        "result": readiness_dir / READINESS_RESULT_NAME,
    }


def prepare_readiness(
    *,
    readiness_dir: Path = READINESS_DIR,
    future_execution_root: Path = FUTURE_EXECUTION_ROOT,
    subprocess_runner: Any = _run_json_subprocess,
) -> dict[str, Any]:
    paths = _readiness_paths(readiness_dir)
    if not paths["evidence"].is_file():
        raise J1bIntegrityError("Immutable J1b test evidence is missing")
    unexpected = sorted(
        path.name
        for path in readiness_dir.iterdir()
        if path != paths["evidence"]
    )
    if unexpected:
        raise J1bIntegrityError(
            f"J1b readiness namespace is not zero-work fresh: {unexpected}"
        )
    if future_execution_root.exists():
        raise J1bIntegrityError(
            "Future J1b execution namespace exists before readiness"
        )

    evidence = _test_evidence_audit(readiness_dir)
    pre_a1_history = pre_a1_history_audit()
    parents = parent_identity_audit()
    original_before = original_execution_identity_audit()

    root_cause_run = subprocess_runner(
        ["_root-cause-probe"],
        nice_10=False,
    )
    with tempfile.TemporaryDirectory(
        prefix="j1b_runtime_probe_",
    ) as temporary:
        runtime_run = subprocess_runner(
            [
                "_runtime-probe",
                "--phase-dir",
                temporary,
                "--future-execution-root",
                str(future_execution_root),
            ],
            nice_10=True,
        )

    manifest = prospective_training_manifest()
    denylist = protected_stream_denylist(manifest)
    projection = runtime_storage_projection()
    schema = schema_payload()
    original_after = original_execution_identity_audit()

    root_payload = (
        root_cause_run.get("payload")
        if isinstance(root_cause_run.get("payload"), dict)
        else {}
    )
    runtime_payload = (
        runtime_run.get("payload")
        if isinstance(runtime_run.get("payload"), dict)
        else {}
    )
    root_cause_artifact = {
        "version": f"{VERSION}_genesis_root_cause_audit_v1",
        "subprocess": {
            "command": root_cause_run.get("command"),
            "returncode": root_cause_run.get("returncode"),
            "stderr": root_cause_run.get("stderr"),
        },
        "observed": root_payload,
        "original_execution_identity_before": original_before,
        "original_execution_identity_after": original_after,
        "checks": {
            "subprocess_passed": root_cause_run.get("passes") is True,
            "root_cause_probe_passed": root_payload.get("passes") is True,
            "original_exact_before": original_before["passes"],
            "original_exact_after": original_after["passes"],
            "original_unchanged_during_preflight": (
                original_before == original_after
            ),
        },
    }
    root_cause_artifact["passes"] = all(
        root_cause_artifact["checks"].values()
    )
    root_cause_artifact = payload_with_hash(
        root_cause_artifact,
        "root_cause_audit_payload_sha256",
    )

    runtime_artifact = {
        "version": f"{VERSION}_runtime_orchestration_audit_v1",
        "subprocess": {
            "command": runtime_run.get("command"),
            "returncode": runtime_run.get("returncode"),
            "stderr": runtime_run.get("stderr"),
        },
        "observed": runtime_payload,
        "checks": {
            "clean_nice10_subprocess_passed": (
                runtime_run.get("passes") is True
            ),
            "runtime_probe_passed": runtime_payload.get("passes") is True,
            "interop_one": (
                runtime_payload.get("runtime", {}).get(
                    "torch_num_interop_threads"
                )
                == 1
            ),
            "intraop_one": (
                runtime_payload.get("runtime", {}).get(
                    "torch_num_threads"
                )
                == 1
            ),
            "deterministic_algorithms": (
                runtime_payload.get("runtime", {}).get(
                    "deterministic_algorithms"
                )
                is True
            ),
            "first_real_guard_passed": (
                runtime_payload.get("checks", {}).get(
                    "first_real_operational_guard_passed"
                )
                is True
            ),
            "no_scientific_artifacts": (
                runtime_payload.get("scientific_artifacts")
                == {
                    "owners": 0,
                    "stream_reservations": 0,
                    "stream_consumptions": 0,
                    "genesis_commits": 0,
                    "games": 0,
                    "optimizer_steps": 0,
                }
            ),
        },
    }
    runtime_artifact["passes"] = all(runtime_artifact["checks"].values())
    runtime_artifact = payload_with_hash(
        runtime_artifact,
        "runtime_audit_payload_sha256",
    )

    integrity_checks = {
        "test_evidence": evidence["passes"],
        "pre_a1_history": pre_a1_history["passes"],
        "parent_identities": parents["passes"],
        "spent_execution_exact": original_after["passes"],
        "root_cause": root_cause_artifact["passes"],
        "manifest": manifest["passes"],
        "denylist": denylist["passes"],
        "projection": projection["passes"],
        "schema": verify_payload_hash(schema, "schema_payload_sha256"),
        "future_execution_absent": not future_execution_root.exists(),
        "zero_work_exact": all(value == 0 for value in ZERO_WORK.values()),
    }
    operational_checks = {
        "runtime_orchestration": runtime_artifact["passes"],
        "first_guard_services": (
            runtime_payload.get("operational_audit", {})
            .get("checks", {})
            .get("services_healthy")
            is True
        ),
        "first_guard_process": (
            runtime_payload.get("operational_audit", {})
            .get("checks", {})
            .get("one_heavy_process")
            is True
        ),
        "first_guard_disk_hard": (
            runtime_payload.get("operational_audit", {})
            .get("checks", {})
            .get("free_disk_hard_floor")
            is True
        ),
        "first_guard_disk_target": (
            runtime_payload.get("operational_audit", {})
            .get("checks", {})
            .get("free_disk_target")
            is True
        ),
        "nice_at_least_10": (
            runtime_payload.get("operational_audit", {})
            .get("checks", {})
            .get("nice_at_least_10")
            is True
        ),
    }
    if not all(integrity_checks.values()):
        decision = "KILL_J1B_PREFLIGHT_INTEGRITY"
    elif not all(operational_checks.values()):
        decision = "HOLD_J1B_OPERATIONAL_REPAIR_PREFLIGHT"
    else:
        decision = "READY_J1B_OPERATIONAL_REPAIR_PREFLIGHT"

    written = {}
    artifact_specs = (
        (
            "root_cause",
            root_cause_artifact,
            "root_cause_audit_payload_sha256",
        ),
        ("denylist", denylist, "denylist_payload_sha256"),
        (
            "manifest",
            manifest,
            "prospective_manifest_payload_sha256",
        ),
        ("runtime", runtime_artifact, "runtime_audit_payload_sha256"),
        ("projection", projection, "projection_payload_sha256"),
        ("schema", schema, "schema_payload_sha256"),
    )
    for name, payload, field in artifact_specs:
        body = dict(payload)
        body.pop(field, None)
        write_immutable_json(paths[name], body, field=field)
        written[name] = artifact_identity(
            paths[name],
            payload_field=field,
        )
    written["evidence"] = artifact_identity(
        paths["evidence"],
        payload_field="test_evidence_payload_sha256",
    )

    lock_payload = {
        "version": f"{VERSION}_readiness_lock_v1",
        "decision": decision,
        "charter_file_sha256": sha256_path(CHARTER_PATH),
        "amendment_a1_file_sha256": sha256_path(AMENDMENT_A1_PATH),
        "runner_file_sha256": sha256_path(RUNNER_PATH),
        "test_file_sha256": sha256_path(TEST_PATH),
        "parent_source_identities": PARENT_SOURCE_IDENTITIES,
        "parent_readiness_identities": PARENT_READINESS_IDENTITIES,
        "spent_j1_execution_identities": ORIGINAL_EXECUTION_IDENTITIES,
        "pre_a1_historical_evidence": pre_a1_history,
        "artifacts": written,
        "fresh_stream_contract": {
            role: {
                "start": base,
                "end_inclusive": base + TRAIN_ROOTS - 1,
                "rows": TRAIN_ROOTS,
            }
            for role, base in FRESH_STREAMS.items()
        },
        "integrity_checks": integrity_checks,
        "operational_checks": operational_checks,
        "future_execution_root": str(future_execution_root.resolve()),
        "future_execution_root_absent": not future_execution_root.exists(),
        "zero_work": ZERO_WORK,
    }
    write_immutable_json(
        paths["lock"],
        lock_payload,
        field="readiness_lock_payload_sha256",
    )
    lock_identity = artifact_identity(
        paths["lock"],
        payload_field="readiness_lock_payload_sha256",
    )

    result_payload = {
        "version": f"{VERSION}_readiness_result_v1",
        "decision": decision,
        "continue": (
            "research-lead review of J1b training execution"
            if decision == "READY_J1B_OPERATIONAL_REPAIR_PREFLIGHT"
            else False
        ),
        "hold": "all J1b phase execution and all J1 development/confirmation",
        "kill": (
            "J1b exact preflight integrity"
            if decision == "KILL_J1B_PREFLIGHT_INTEGRITY"
            else "historical kills unchanged; J1/J1b not scientifically killed"
        ),
        "promote": False,
        "readiness_lock": lock_identity,
        "integrity_checks": integrity_checks,
        "operational_checks": operational_checks,
        "fresh_manifest": written["manifest"],
        "root_cause_audit": written["root_cause"],
        "runtime_orchestration_audit": written["runtime"],
        "zero_work": ZERO_WORK,
    }
    write_immutable_json(
        paths["result"],
        result_payload,
        field="readiness_result_payload_sha256",
    )
    result_identity = artifact_identity(
        paths["result"],
        payload_field="readiness_result_payload_sha256",
    )
    return {
        "decision": decision,
        "readiness_dir": str(readiness_dir.resolve()),
        "readiness_lock": lock_identity,
        "readiness_result": result_identity,
        "artifacts": written,
        "integrity_checks": integrity_checks,
        "operational_checks": operational_checks,
        "zero_work": ZERO_WORK,
        "passes": decision == "READY_J1B_OPERATIONAL_REPAIR_PREFLIGHT",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    evidence = subparsers.add_parser("write-test-evidence")
    evidence.add_argument("--readiness-dir", type=Path, default=READINESS_DIR)
    evidence.add_argument("--py-compile-command", required=True)
    evidence.add_argument("--focused-command", required=True)
    evidence.add_argument("--focused-passed", type=int, required=True)
    evidence.add_argument("--parent-execution-command", required=True)
    evidence.add_argument("--parent-execution-passed", type=int, required=True)
    evidence.add_argument("--parent-j1-command", required=True)
    evidence.add_argument("--parent-j1-passed", type=int, required=True)
    evidence.add_argument("--parent-j1a-command", required=True)
    evidence.add_argument("--parent-j1a-passed", type=int, required=True)
    evidence.add_argument("--applicable-command", required=True)
    evidence.add_argument("--applicable-passed", type=int, required=True)
    evidence.add_argument("--deselection", action="append", default=[])

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--readiness-dir", type=Path, default=READINESS_DIR)
    prepare.add_argument(
        "--future-execution-root",
        type=Path,
        default=FUTURE_EXECUTION_ROOT,
    )

    root_probe = subparsers.add_parser("_root-cause-probe")
    root_probe.set_defaults(hidden_probe=True)

    runtime_probe = subparsers.add_parser("_runtime-probe")
    runtime_probe.add_argument("--phase-dir", type=Path, required=True)
    runtime_probe.add_argument(
        "--future-execution-root",
        type=Path,
        required=True,
    )
    runtime_probe.set_defaults(hidden_probe=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.subcommand == "write-test-evidence":
        result = write_test_evidence(
            readiness_dir=args.readiness_dir,
            py_compile_command=args.py_compile_command,
            focused_command=args.focused_command,
            focused_passed=args.focused_passed,
            parent_execution_command=args.parent_execution_command,
            parent_execution_passed=args.parent_execution_passed,
            parent_j1_command=args.parent_j1_command,
            parent_j1_passed=args.parent_j1_passed,
            parent_j1a_command=args.parent_j1a_command,
            parent_j1a_passed=args.parent_j1a_passed,
            applicable_command=args.applicable_command,
            applicable_passed=args.applicable_passed,
            documented_deselections=args.deselection,
        )
    elif args.subcommand == "prepare":
        result = prepare_readiness(
            readiness_dir=args.readiness_dir,
            future_execution_root=args.future_execution_root,
        )
    elif args.subcommand == "_root-cause-probe":
        result = _root_cause_probe()
    elif args.subcommand == "_runtime-probe":
        result = _runtime_probe(
            phase_dir=args.phase_dir,
            future_execution_root=args.future_execution_root,
        )
    else:
        raise AssertionError(f"Unhandled subcommand: {args.subcommand}")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("passes") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
