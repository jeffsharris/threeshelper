"""Frozen O3 fresh-root acquisition and support-only allocation gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from threes_rl import g1r_acquire as history
from threes_rl import g1r_acquire_v2_qd5 as policy_source
from threes_rl import o3_p0_preflight as p0
from threes_rl.eval import EvalJob, EvalStreamIds, iter_eval_job_outputs
from threes_rl.g1r_qd_admission_v2 import _heavy_process_audit
from threes_rl.o1_geometry_option import air_safe, anchor_safe
from threes_rl.o3_designated_pair_option import (
    TRAIN_TARGETS,
    root_option_eligible,
    schema_sha256,
    select_designated_pair,
)
from threes_rl.replay_provenance import (
    ORIGIN_FRESH,
    direct_root_fields,
    replay_provenance,
)
from threes_rl.restart_manifest import canonical_ancestry_id, state_signature
from threes_rl.run_artifacts import write_json
from threes_rl.s3_power_preflight import sha256_path
from threes_rl.sim import DIRECTION_NAMES, ThreesSim
from threes_rl.train_td import state_from_replay_payload


VERSION = "o3_event_acquisition_v1"
ROOT = Path("threes_rl/runs")
CHARTER_PATH = Path("threes_rl/O3_EVENT_ACQUISITION_EXECUTION_CHARTER.md")
COURSE_CHARTER_PATH = p0.CHARTER_PATH
RUNNER_PATH = Path("threes_rl/o3_event_acquire.py")
TEST_PATH = Path("tests/test_rl_o3_event_acquire.py")
TEST_EVIDENCE_PATH = (
    ROOT / "forensics/o3_event_acquisition_test_evidence.json"
)
OUTPUT_DIR = ROOT / "forensics/o3_event_acquisition_v1"
MARKER_PATH = OUTPUT_DIR / "O3_ACQUISITION_OPENED.json"
RESULT_PATH = OUTPUT_DIR / "O3_ACQUISITION_RESULT.json"
ATTEMPT_PATH = OUTPUT_DIR / "attempts.jsonl"
COMPLETION_PATH = OUTPUT_DIR / "completed_games.jsonl"
RUNTIME_PATH = OUTPUT_DIR / "runtime_state.json"
REPLAY_DIR = OUTPUT_DIR / "source_replays"
SUPPORT_PATH = OUTPUT_DIR / "O3_SUPPORT_SCAN.json"
SELECTED_PATH = OUTPUT_DIR / "O3_SELECTED_ROOTS.json"

P0_DIR = p0.OUTPUT_DIR
P0_ARTIFACTS = {
    "marker": (
        P0_DIR / p0.MARKER_NAME,
        "4b8133ea1e8f237debbbdb90bb682214d04560de4756e0e6e639c4df9e6e63d1",
        "opened_payload_sha256",
        "e7b19680ba7b340690400abeeaa222d92267b15204c3dbfa3461f9df71762955",
    ),
    "result": (
        P0_DIR / p0.RESULT_NAME,
        "9ced80be3e2a784372f50fd2a99b0b41bdcc98920820796daa03a8db1640ced5",
        "result_payload_sha256",
        "6c67d285e9c2e3315a362089b9a8c3798affb58f032b810d15db8b894cd9f99e",
    ),
    "streams": (
        P0_DIR / p0.STREAM_MANIFEST_NAME,
        "94e7b0dfe83e568b4e9686dd3ee44cc70739c0312349fe36a05bb6df80c77225",
        "payload_sha256",
        "27e3200e88d31d4f38a921965b631f264aa43f0ef02cb380f41b0c04d8455d1b",
    ),
    "partitions": (
        P0_DIR / p0.PARTITION_MANIFEST_NAME,
        "d08770f8363463a993e2cb360dbcd440de30a2a951f7b76a282374e30d4ea182",
        "payload_sha256",
        "f63ba67aab2416221c698b93f26cbf19599343eb57921a19770cdd9d3130af44",
    ),
    "policies": (
        P0_DIR / p0.POLICY_MANIFEST_NAME,
        "2b498ce5bc22f54f6286e114f3212758e911a1ac7a651da2c3095db42dea0e60",
        "payload_sha256",
        "6c09df4c8e0e0d3720720e05d58cda8459dea9296d050b751db6b115705deb9c",
    ),
    "power": (
        P0_DIR / p0.POWER_MANIFEST_NAME,
        "96e84bd9c0c2d34bc202988db8253ab7b5a9538ddb5fad1c3f5bc065341267d6",
        "payload_sha256",
        "1f683659754d72929530fe830bde7557921fab5c138bffb6e32a67009ce7580c",
    ),
    "collision": (
        P0_DIR / p0.COLLISION_MANIFEST_NAME,
        "0fba84dc9278df892491295df4f54a07a3bb9fcb2b84bb16b7083c5174627a0c",
        "payload_sha256",
        "d50732c0143a2bb09cb08377be9f622cdb971d060b875b65ccbaffe703929739",
    ),
}
P0_TEST_EVIDENCE_FILE_SHA256 = (
    "5c9de58258e778428897f210b0701b2c9d34cc05ca79482dc7b3c048100085a1"
)
P0_TEST_EVIDENCE_PAYLOAD_SHA256 = (
    "a1a7f8e60db8e5139e7d3eccd255f071274a3fb7db03a649e3d8b914c331c61a"
)

FAMILY_ORDER = p0.O3_FAMILY_ORDER
O3_TO_G1R = p0.O3_TO_G1R
G1R_SPEC_BY_FAMILY = dict(policy_source.FAMILY_SLATE)
POLICY_SPECS = {
    o3_family: G1R_SPEC_BY_FAMILY[g1r_family]
    for o3_family, g1r_family in O3_TO_G1R.items()
}
ROOTS_PER_FAMILY = p0.ROOTS_PER_FAMILY
TOTAL_ROOTS = p0.ACQUISITION_ROOTS
ROLE_ORDER = ("train", "development", "untouched_mechanism")
TARGET_ORDER = (192, 96, 48)
TARGET_COUNTS = p0.TARGET_SELECTED_COUNTS
MIN_FAMILY_COUNTS = {
    "train": 4,
    "development": 2,
    "untouched_mechanism": 8,
}
MAX_FAMILY_SHARE = 0.40
FROZEN_JOBS = 1
MINIMUM_NICE = 10
STARTER_TILE = 1536
MAX_MOVES = 5000
CHUNK_SIZE = len(FAMILY_ORDER)
ACTIVE_RUNTIME_LIMIT = p0.ACQUISITION_ACTIVE_SECONDS
BYTE_LIMIT = p0.ACQUISITION_BYTE_LIMIT
MIN_FREE_GIB = p0.MIN_FREE_GIB
TARGET_FREE_GIB = p0.TARGET_FREE_GIB
ATTEMPT_TERMINAL_STATUSES = {
    "completed",
    "completed_recovered",
    "interrupted_no_replay",
}
O2_FORBIDDEN_DIRS = (
    ROOT / "forensics/o2_yield_pilot_v1",
    ROOT / "forensics/o2_yield_pilot_scan_recovery_v1",
)
DEPENDENCY_PATHS = (
    COURSE_CHARTER_PATH,
    Path("threes_rl/o3_designated_pair_option.py"),
    Path("threes_rl/o3_power_contract.py"),
    Path("threes_rl/o3_p0_preflight.py"),
    Path("tests/test_rl_o3_designated_pair_option.py"),
    Path("tests/test_rl_o3_p0_preflight.py"),
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
    raw = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _write_immutable_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    self_hash_field: str,
) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"Immutable artifact already exists: {path}")
    body = dict(payload)
    body[self_hash_field] = canonical_json_hash(body)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"Atomic temporary already exists: {temporary}")
    temporary.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
    return body


def _verify_self_hash(payload: Mapping[str, Any], field: str) -> bool:
    body = dict(payload)
    embedded = body.pop(field, None)
    return isinstance(embedded, str) and embedded == canonical_json_hash(body)


def _artifact_identity(path: Path, self_hash_field: str) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not _verify_self_hash(payload, self_hash_field):
        raise ValueError(f"Artifact self hash mismatch: {path}")
    return {
        "path": str(path),
        "file_sha256": sha256_path(path),
        "payload_sha256": payload[self_hash_field],
    }


def _sealed_artifact_audit() -> dict[str, Any]:
    rows = {}
    for name, (path, expected_file, field, expected_payload) in P0_ARTIFACTS.items():
        actual_file = sha256_path(path)
        payload = json.loads(path.read_text())
        body = dict(payload)
        embedded = body.pop(field, None)
        computed = canonical_json_hash(body)
        pre_serialization_reproduction = None
        if name == "partitions":
            # The frozen P0 payload contains integer target-map keys. JSON
            # round-tripping turns them into strings and changes sort order,
            # so reproduce the original pre-serialization hash explicitly.
            pre_serialization_reproduction = canonical_json_hash(
                {
                    "version": f"{p0.VERSION}_partitions",
                    **p0.partition_plan(p0.future_stream_rows()),
                    "outcomes_opened": False,
                }
            )
        checks = {
            "file_exact": actual_file == expected_file,
            "embedded_payload_exact": embedded == expected_payload,
            "payload_identity_exact": (
                computed == expected_payload
                if name != "partitions"
                else pre_serialization_reproduction == expected_payload
            ),
        }
        rows[name] = {
            "path": str(path),
            "file_sha256": actual_file,
            "payload_sha256": expected_payload,
            "roundtrip_computed_payload_sha256": computed,
            "pre_serialization_reproduction_sha256": (
                pre_serialization_reproduction
            ),
            "checks": checks,
            "passes": all(checks.values()),
        }
    p0_tests = json.loads(p0.TEST_EVIDENCE_PATH.read_text())
    tests_body = dict(p0_tests)
    embedded = tests_body.pop("test_evidence_payload_sha256", None)
    tests_checks = {
        "file_exact": sha256_path(p0.TEST_EVIDENCE_PATH)
        == P0_TEST_EVIDENCE_FILE_SHA256,
        "embedded_payload_exact": embedded == P0_TEST_EVIDENCE_PAYLOAD_SHA256,
        "computed_payload_exact": canonical_json_hash(tests_body)
        == P0_TEST_EVIDENCE_PAYLOAD_SHA256,
    }
    rows["test_evidence"] = {
        "path": str(p0.TEST_EVIDENCE_PATH),
        "file_sha256": sha256_path(p0.TEST_EVIDENCE_PATH),
        "payload_sha256": canonical_json_hash(tests_body),
        "checks": tests_checks,
        "passes": all(tests_checks.values()),
    }
    result = json.loads((P0_DIR / p0.RESULT_NAME).read_text())
    checks = {
        "all_p0_artifacts_exact": all(row["passes"] for row in rows.values()),
        "p0_ready": result.get("decision") == "READY_O3_EVENT_ACQUISITION",
        "p0_acquisition_authorized": bool(
            result.get("acquisition_authorized_by_result")
        ),
        "p0_zero_work": all(
            int(value) == 0 or value is False
            for value in result.get("zero_work", {}).values()
        ),
    }
    return {"artifacts": rows, "checks": checks, "passes": all(checks.values())}


def _dependency_manifest() -> dict[str, Any]:
    rows = [
        {
            "path": str(path),
            "bytes": int(path.stat().st_size),
            "sha256": sha256_path(path),
        }
        for path in DEPENDENCY_PATHS
    ]
    return {
        "rows": rows,
        "manifest_sha256": canonical_json_hash(rows),
    }


def _load_test_evidence() -> dict[str, Any]:
    payload = json.loads(TEST_EVIDENCE_PATH.read_text())
    if not _verify_self_hash(payload, "test_evidence_payload_sha256"):
        raise ValueError("O3 acquisition test evidence payload mismatch")
    expected = {
        "course_charter_sha256": sha256_path(COURSE_CHARTER_PATH),
        "execution_charter_sha256": sha256_path(CHARTER_PATH),
        "runner_sha256": sha256_path(RUNNER_PATH),
        "tests_sha256": sha256_path(TEST_PATH),
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("O3 acquisition test evidence source mismatch")
    if not payload.get("passes"):
        raise ValueError("O3 acquisition test evidence is not passing")
    return payload


def seal_test_evidence(
    *,
    focused_passed: int,
    regression_passed: int,
    commands: Sequence[str],
) -> dict[str, Any]:
    if focused_passed <= 0 or regression_passed <= 0 or not commands:
        raise ValueError("Passing test counts and commands are required")
    return _write_immutable_json(
        TEST_EVIDENCE_PATH,
        {
            "version": f"{VERSION}_test_evidence",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "course_charter_sha256": sha256_path(COURSE_CHARTER_PATH),
            "execution_charter_sha256": sha256_path(CHARTER_PATH),
            "runner_sha256": sha256_path(RUNNER_PATH),
            "tests_sha256": sha256_path(TEST_PATH),
            "focused_tests_passed": int(focused_passed),
            "regression_tests_passed": int(regression_passed),
            "commands": list(commands),
            "passes": True,
            "games_generated": 0,
            "streams_consumed": 0,
            "support_content_opened": 0,
            "labels_generated": 0,
            "models_fit": 0,
            "policy_outcomes_inspected": False,
            "o2_row_level_content_read": False,
        },
        self_hash_field="test_evidence_payload_sha256",
    )


def acquisition_rows() -> list[dict[str, Any]]:
    payload = json.loads((P0_DIR / p0.STREAM_MANIFEST_NAME).read_text())
    if not _verify_self_hash(payload, "payload_sha256"):
        raise ValueError("O3 P0 stream manifest payload mismatch")
    rows = [
        dict(row)
        for row in payload["rows"]
        if row.get("purpose") == "acquisition"
    ]
    checks = {
        "exact_count": len(rows) == TOTAL_ROOTS,
        "equal_families": Counter(str(row["family"]) for row in rows)
        == {family: ROOTS_PER_FAMILY for family in FAMILY_ORDER},
        "family_order": tuple(
            dict.fromkeys(str(row["family"]) for row in rows)
        )
        == FAMILY_ORDER,
        "roles_exact": Counter(str(row["role"]) for row in rows)
        == p0.ROLE_COUNTS,
        "stream_rows_exact": rows == p0.acquisition_rows(),
    }
    if not all(checks.values()):
        raise ValueError(f"O3 acquisition stream manifest mismatch: {checks}")
    return rows


def round_robin_chunks(rows: Sequence[Mapping[str, Any]]) -> list[list[dict[str, Any]]]:
    by_key = {
        (str(row["family"]), int(row["game_index"])): dict(row)
        for row in rows
    }
    if len(by_key) != TOTAL_ROOTS:
        raise ValueError("O3 acquisition rows are not unique")
    chunks = []
    for game_index in range(ROOTS_PER_FAMILY):
        chunk = [
            by_key[(family, game_index)]
            for family in FAMILY_ORDER
        ]
        chunks.append(chunk)
    if len(chunks) != ROOTS_PER_FAMILY or any(
        len(chunk) != CHUNK_SIZE for chunk in chunks
    ):
        raise ValueError("O3 acquisition chunk construction failed")
    return chunks


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


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
    matched = []
    excluded = []
    for path in sorted(scan_root.rglob("*")):
        if not path.is_file() or path.suffix not in {".json", ".jsonl", ".csv"}:
            continue
        classification = None
        if _is_within(path, out_dir):
            classification = "current_o3_acquisition_namespace"
        elif _is_within(path, P0_DIR):
            classification = "immutable_o3_p0_reservation_namespace"
        elif any(_is_within(path, directory) for directory in O2_FORBIDDEN_DIRS):
            classification = "immutable_o2_content_forbidden_unread"
        if classification is not None:
            excluded.append(
                {
                    "path": str(path),
                    "classification": classification,
                    "bytes": int(path.stat().st_size),
                    "sha256": sha256_path(path),
                }
            )
            continue
        values = history._scan_history_file(path)
        if not values:
            continue
        for field, items in values.items():
            found[field].update(items)
        matched.append(
            {
                "path": str(path),
                "bytes": int(path.stat().st_size),
                "sha256": sha256_path(path),
                "counts": {
                    field: len(items)
                    for field, items in sorted(values.items())
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
        "requested_stream_ids_internally_unique": len(flat) == len(set(flat)),
        "zero_external_collisions": not any(collisions.values()),
        "p0_reservation_namespace_excluded": any(
            row["classification"] == "immutable_o3_p0_reservation_namespace"
            for row in excluded
        ),
        "o2_content_excluded_unread": any(
            row["classification"] == "immutable_o2_content_forbidden_unread"
            for row in excluded
        ),
    }
    return {
        "matched_source_count": len(matched),
        "matched_sources_sha256": canonical_json_hash(matched),
        "matched_sources": matched,
        "excluded_source_count": len(excluded),
        "excluded_sources_sha256": canonical_json_hash(excluded),
        "excluded_sources": excluded,
        "collisions": collisions,
        "checks": checks,
        "passes": all(checks.values()),
    }


def _load_policies() -> tuple[dict[str, Any], dict[str, Any]]:
    p0_policies = json.loads((P0_DIR / p0.POLICY_MANIFEST_NAME).read_text())
    if not _verify_self_hash(p0_policies, "payload_sha256"):
        raise ValueError("O3 P0 policy artifact payload mismatch")
    current_lock, loaded = policy_source._policy_lock()
    if current_lock["policy_lock_sha256"] != p0_policies["policy_lock_sha256"]:
        raise ValueError("O3 collector policy payloads changed after P0")
    policies = {
        o3_family: loaded[g1r_family]
        for o3_family, g1r_family in O3_TO_G1R.items()
    }
    checks = {
        "five_policies_loaded": len(policies) == len(FAMILY_ORDER),
        "family_order_exact": tuple(policies) == FAMILY_ORDER,
        "policy_lock_exact": True,
        "signatures_bound_from_p0": p0_policies["signature_sha256"]
        == p0.EXPECTED_SIGNATURES,
    }
    return policies, {
        "policy_lock_sha256": current_lock["policy_lock_sha256"],
        "policy_source_manifest_sha256": current_lock[
            "source_manifest_sha256"
        ],
        "family_order": list(policies),
        "signature_sha256": p0_policies["signature_sha256"],
        "checks": checks,
        "passes": all(checks.values()),
    }


def _bound_commands(out_dir: Path) -> dict[str, str]:
    prefix = (
        "zsh -ic 'no-secrets nice -n 10 env PYTHONPATH=. "
        ".venv/bin/python -m threes_rl.o3_event_acquire"
    )
    suffix = f" --out-dir {out_dir} --jobs 1'"
    return {
        "open": f"{prefix} open{suffix}",
        "execute": f"{prefix} execute{suffix}",
    }


def _marker_identity(out_dir: Path) -> dict[str, Any]:
    tests = _load_test_evidence()
    p0_audit = _sealed_artifact_audit()
    if not p0_audit["passes"]:
        raise ValueError("O3 P0 immutable input audit failed")
    rows = acquisition_rows()
    dependency = _dependency_manifest()
    return {
        "version": VERSION,
        "bound_out_dir": str(out_dir.resolve()),
        "course_charter_sha256": sha256_path(COURSE_CHARTER_PATH),
        "execution_charter_sha256": sha256_path(CHARTER_PATH),
        "runner_sha256": sha256_path(RUNNER_PATH),
        "tests_sha256": sha256_path(TEST_PATH),
        "test_evidence_file_sha256": sha256_path(TEST_EVIDENCE_PATH),
        "test_evidence_payload_sha256": tests[
            "test_evidence_payload_sha256"
        ],
        "schema_sha256": schema_sha256(),
        "p0_artifacts": {
            name: {
                "file_sha256": row["file_sha256"],
                "payload_sha256": row["payload_sha256"],
            }
            for name, row in p0_audit["artifacts"].items()
        },
        "dependency_manifest_sha256": dependency["manifest_sha256"],
        "dependency_manifest": dependency["rows"],
        "family_order": list(FAMILY_ORDER),
        "roots_per_family": ROOTS_PER_FAMILY,
        "total_roots": TOTAL_ROOTS,
        "stream_manifest_sha256": canonical_json_hash(rows),
        "jobs": FROZEN_JOBS,
        "minimum_nice": MINIMUM_NICE,
        "active_runtime_limit_seconds": ACTIVE_RUNTIME_LIMIT,
        "byte_limit": BYTE_LIMIT,
        "minimum_free_gib": MIN_FREE_GIB,
        "target_free_gib": TARGET_FREE_GIB,
        "commands": _bound_commands(out_dir),
    }


def _operational_audit(out_dir: Path) -> dict[str, Any]:
    free_gib = shutil.disk_usage(out_dir.parent).free / 1024**3
    used = history._directory_bytes(out_dir) if out_dir.exists() else 0
    heavy = _heavy_process_audit()
    services = history.service_health()
    checks = {
        "nice_at_least_10": history.current_nice() >= MINIMUM_NICE,
        "one_heavy_process": heavy["passes"],
        "free_disk_above_100_gib": free_gib > MIN_FREE_GIB,
        "output_below_28_gib": used < BYTE_LIMIT,
        "services_dashboard_top_three": services["passes"],
    }
    return {
        "nice": history.current_nice(),
        "free_gib": free_gib,
        "output_bytes": used,
        "heavy_process_audit": heavy,
        "services": services,
        "checks": checks,
        "passes": all(checks.values()),
    }


def open_execution(
    *,
    out_dir: Path = OUTPUT_DIR,
    jobs: int = FROZEN_JOBS,
) -> dict[str, Any]:
    if out_dir.resolve() != OUTPUT_DIR.resolve():
        raise ValueError("O3 acquisition output directory is immutable")
    if jobs != FROZEN_JOBS:
        raise ValueError("O3 acquisition freezes jobs=1")
    if out_dir.exists():
        raise FileExistsError(f"O3 acquisition namespace exists: {out_dir}")
    if history.current_nice() < MINIMUM_NICE:
        raise ValueError("O3 acquisition open requires nice >=10")
    p0_audit = _sealed_artifact_audit()
    tests = _load_test_evidence()
    _policies, policy_audit = _load_policies()
    rows = acquisition_rows()
    collision = collision_audit(rows, out_dir=out_dir)
    operations = _operational_audit(out_dir)
    checks = {
        "p0_ready_and_exact": p0_audit["passes"],
        "tests_exact": bool(tests["passes"]),
        "policies_exact": policy_audit["passes"],
        "stream_collision_free": collision["passes"],
        "operations_pass": operations["passes"],
        "free_disk_target_met": operations["free_gib"] > TARGET_FREE_GIB,
        "zero_prior_namespace": True,
        "zero_games_streams_support_labels_models_outcomes": True,
    }
    if not all(checks.values()):
        raise ValueError(f"O3 acquisition open failed: {checks}")
    marker = {
        **_marker_identity(out_dir),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "decision": "O3_ACQUISITION_OPENED_ZERO_WORK",
        "policy_audit": policy_audit,
        "collision_audit": collision,
        "operations": operations,
        "checks": checks,
        "zero_work": {
            "games": 0,
            "streams_consumed": 0,
            "replays": 0,
            "support_states_opened": 0,
            "labels": 0,
            "fits": 0,
            "policy_outcomes": 0,
            "score_inspection": 0,
            "dashboard_changes": 0,
            "incumbent_changes": 0,
        },
    }
    return _write_immutable_json(
        MARKER_PATH,
        marker,
        self_hash_field="opened_payload_sha256",
    )


def _load_marker(
    *,
    out_dir: Path,
    jobs: int,
) -> dict[str, Any]:
    if out_dir.resolve() != OUTPUT_DIR.resolve() or jobs != FROZEN_JOBS:
        raise ValueError("O3 acquisition execute identity mismatch")
    if not MARKER_PATH.is_file():
        raise FileNotFoundError("O3 acquisition marker is missing")
    marker = json.loads(MARKER_PATH.read_text())
    if not _verify_self_hash(marker, "opened_payload_sha256"):
        raise ValueError("O3 acquisition marker payload mismatch")
    current = _marker_identity(out_dir)
    for key, value in current.items():
        if marker.get(key) != value:
            raise ValueError(f"O3 acquisition marker binding changed: {key}")
    if RESULT_PATH.exists():
        raise FileExistsError("O3 acquisition already has a terminal result")
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


def _stream_key(row: Mapping[str, Any]) -> tuple[str, int]:
    return str(row["family"]), int(row["game_index"])


def _attempt_id(
    row: Mapping[str, Any],
    attempt_number: int,
) -> str:
    return hashlib.sha256(
        (
            "O3-acquisition-attempt-v1|"
            f"{row['family']}|{row['game_index']}|{attempt_number}|"
            f"{row['logical_seed']}|{row['deck_stream_id']}|"
            f"{row['slot_stream_id']}|{row['policy_stream_id']}"
        ).encode("ascii")
    ).hexdigest()


def _append_attempt_event(
    row: Mapping[str, Any],
    *,
    attempt_number: int,
    status: str,
    chunk_index: int,
) -> dict[str, Any]:
    if status not in {"opened", *ATTEMPT_TERMINAL_STATUSES}:
        raise ValueError(f"Unknown O3 attempt status: {status}")
    event = {
        "attempt_id": _attempt_id(row, attempt_number),
        "attempt_number": int(attempt_number),
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
    }
    history._append_jsonl_row(ATTEMPT_PATH, event)
    return event


def _load_attempt_ledger(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int], list[dict[str, Any]]]:
    expected = {_stream_key(row): dict(row) for row in rows}
    grouped: dict[tuple[str, int], dict[int, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    if ATTEMPT_PATH.is_file():
        for line in ATTEMPT_PATH.read_text().splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            key = (str(event["family"]), int(event["game_index"]))
            if key not in expected:
                raise ValueError(f"Attempt event outside frozen manifest: {key}")
            row = expected[key]
            for field in (
                "family_index",
                "logical_seed",
                "deck_stream_id",
                "slot_stream_id",
                "policy_stream_id",
            ):
                if int(event[field]) != int(row[field]):
                    raise ValueError(f"Attempt stream mismatch for {key}: {field}")
            grouped[key][int(event["attempt_number"])].append(event)
    result: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for key, attempts in grouped.items():
        ordered = []
        for number in sorted(attempts):
            if number != len(ordered):
                raise ValueError(f"Attempt numbering gap for {key}")
            events = attempts[number]
            statuses = [str(event["status"]) for event in events]
            if not statuses or statuses[0] != "opened" or len(statuses) > 2:
                raise ValueError(f"Malformed attempt lifecycle for {key}: {statuses}")
            if len(statuses) == 2 and statuses[1] not in ATTEMPT_TERMINAL_STATUSES:
                raise ValueError(f"Malformed terminal attempt for {key}: {statuses}")
            ordered.append(
                {
                    "attempt_number": number,
                    "attempt_id": events[0]["attempt_id"],
                    "statuses": statuses,
                }
            )
        result[key] = ordered
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
    stream_row: Mapping[str, Any],
) -> dict[str, Any]:
    provenance = replay_provenance(dict(replay), replay_path)
    expected_seed = int(stream_row["logical_seed"])
    streams = replay.get("rng_streams")
    if not isinstance(streams, dict):
        raise ValueError("O3 replay lacks split-stream metadata")
    checks = {
        "fresh_replay": provenance.get("replay_origin") == ORIGIN_FRESH,
        "fresh_root": provenance.get("root_origin") == ORIGIN_FRESH,
        "reset_invariant": bool(provenance.get("replay_reset_invariant")),
        "root_seed": int(provenance.get("root_seed")) == expected_seed,
        "replay_seed": int(replay.get("seed")) == expected_seed,
        "starter": int(replay.get("starter_tile")) == STARTER_TILE,
        "deck_stream": int(streams.get("deck_stream_id"))
        == int(stream_row["deck_stream_id"]),
        "slot_stream": int(streams.get("slot_stream_id"))
        == int(stream_row["slot_stream_id"]),
        "policy_stream": int(streams.get("policy_stream_id"))
        == int(stream_row["policy_stream_id"]),
        "dashboard_ineligible": replay.get("dashboard_eligible") is False,
    }
    if not all(checks.values()):
        raise ValueError(f"O3 replay provenance mismatch: {checks}")
    return {
        "family": str(stream_row["family"]),
        "family_index": int(stream_row["family_index"]),
        "game_index": int(stream_row["game_index"]),
        "role": str(stream_row["role"]),
        "planned_root_id": str(stream_row["planned_root_id"]),
        "logical_seed": expected_seed,
        "deck_stream_id": int(stream_row["deck_stream_id"]),
        "slot_stream_id": int(stream_row["slot_stream_id"]),
        "policy_stream_id": int(stream_row["policy_stream_id"]),
        "source_replay": str(replay_path),
        "source_replay_sha256": sha256_path(replay_path),
        "root_cluster": canonical_ancestry_id(dict(replay), replay_path),
        "complete": True,
        "dashboard_eligible": False,
    }


def _store_output(
    output: Any,
    *,
    stream_row: Mapping[str, Any],
) -> dict[str, Any]:
    replay = output.replay
    if not isinstance(replay, dict):
        raise ValueError("O3 evaluator omitted replay capture")
    replay_path = _replay_path(stream_row)
    if replay_path.exists():
        raise FileExistsError(f"O3 replay already exists: {replay_path}")
    replay.update(
        direct_root_fields(
            origin=ORIGIN_FRESH,
            seed=int(stream_row["logical_seed"]),
            policy=str(stream_row["family"]),
            replay_path=replay_path,
            first_score=None,
        )
    )
    replay["behavior_family"] = str(stream_row["family"])
    replay["nominal_family"] = str(stream_row["family"])
    replay["acquisition_policy_spec"] = POLICY_SPECS[str(stream_row["family"])]
    replay["o3_event_acquisition"] = True
    replay["planned_root_id"] = str(stream_row["planned_root_id"])
    replay["acquisition_role"] = str(stream_row["role"])
    replay["dashboard_eligible"] = False
    _write_immutable_json(
        replay_path,
        replay,
        self_hash_field="o3_replay_payload_sha256",
    )
    stored = json.loads(replay_path.read_text())
    return _completion_from_replay(
        stored,
        replay_path=replay_path,
        stream_row=stream_row,
    )


def _load_completions() -> dict[tuple[str, int], dict[str, Any]]:
    completed: dict[tuple[str, int], dict[str, Any]] = {}
    if not COMPLETION_PATH.is_file():
        return completed
    for line in COMPLETION_PATH.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (str(row["family"]), int(row["game_index"]))
        if key in completed:
            raise ValueError(f"Duplicate O3 completion row: {key}")
        completed[key] = row
    return completed


def _verify_existing_completions(
    completions: Mapping[tuple[str, int], Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    expected = {_stream_key(row): dict(row) for row in rows}
    roots = set()
    for key, completion in completions.items():
        row = expected.get(key)
        if row is None:
            raise ValueError(f"Completion outside frozen manifest: {key}")
        replay_path = Path(str(completion["source_replay"]))
        if replay_path != _replay_path(row) or not replay_path.is_file():
            raise ValueError(f"Missing or mislocated O3 replay: {key}")
        if sha256_path(replay_path) != completion["source_replay_sha256"]:
            raise ValueError(f"O3 replay hash changed: {key}")
        replay = json.loads(replay_path.read_text())
        restored = _completion_from_replay(
            replay,
            replay_path=replay_path,
            stream_row=row,
        )
        if restored != dict(completion):
            raise ValueError(f"O3 completion no longer matches replay: {key}")
        root = str(completion["root_cluster"])
        if root in roots:
            raise ValueError(f"Duplicate O3 ancestry in completions: {root}")
        roots.add(root)


def _recover_existing_evidence(
    rows: Sequence[Mapping[str, Any]],
    completions: dict[tuple[str, int], dict[str, Any]],
) -> dict[tuple[str, int], list[dict[str, Any]]]:
    attempts = _load_attempt_ledger(rows)
    for row in rows:
        key = _stream_key(row)
        replay_path = _replay_path(row)
        lifecycle = attempts.get(key, [])
        if key in completions:
            if lifecycle and len(lifecycle[-1]["statuses"]) == 1:
                _append_attempt_event(
                    row,
                    attempt_number=int(lifecycle[-1]["attempt_number"]),
                    status="completed_recovered",
                    chunk_index=int(row["game_index"]),
                )
                lifecycle[-1]["statuses"].append("completed_recovered")
            continue
        if replay_path.is_file():
            replay = json.loads(replay_path.read_text())
            completion = _completion_from_replay(
                replay,
                replay_path=replay_path,
                stream_row=row,
            )
            history._append_jsonl_row(COMPLETION_PATH, completion)
            completions[key] = completion
            if not lifecycle or len(lifecycle[-1]["statuses"]) != 1:
                raise ValueError(f"Orphan replay lacks open attempt: {key}")
            _append_attempt_event(
                row,
                attempt_number=int(lifecycle[-1]["attempt_number"]),
                status="completed_recovered",
                chunk_index=int(row["game_index"]),
            )
            lifecycle[-1]["statuses"].append("completed_recovered")
    return attempts


def _open_attempt(
    row: Mapping[str, Any],
    *,
    chunk_index: int,
    attempts: dict[tuple[str, int], list[dict[str, Any]]],
) -> int:
    key = _stream_key(row)
    lifecycles = attempts.setdefault(key, [])
    if lifecycles and len(lifecycles[-1]["statuses"]) == 1:
        if _replay_path(row).exists():
            raise ValueError("Retained replay must be recovered before retry")
        _append_attempt_event(
            row,
            attempt_number=int(lifecycles[-1]["attempt_number"]),
            status="interrupted_no_replay",
            chunk_index=chunk_index,
        )
        lifecycles[-1]["statuses"].append("interrupted_no_replay")
    attempt_number = len(lifecycles)
    opened = _append_attempt_event(
        row,
        attempt_number=attempt_number,
        status="opened",
        chunk_index=chunk_index,
    )
    lifecycles.append(
        {
            "attempt_number": attempt_number,
            "attempt_id": opened["attempt_id"],
            "statuses": ["opened"],
        }
    )
    return attempt_number


def _guard_execution(
    runtime: Mapping[str, Any],
    *,
    out_dir: Path = OUTPUT_DIR,
) -> dict[str, Any]:
    operations = _operational_audit(out_dir)
    checks = dict(operations["checks"])
    checks["active_runtime_below_144h"] = (
        float(runtime["active_runtime_seconds"]) <= ACTIVE_RUNTIME_LIMIT
    )
    if not all(checks.values()):
        raise history.AcquisitionPause(
            "HOLD_O3_ACQUISITION_INTEGRITY",
            f"O3 acquisition operational guard failed: {checks}",
        )
    return {**operations, "checks": checks, "passes": True}


def collect_all(
    marker: Mapping[str, Any],
    *,
    jobs: int,
    policies: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    del marker
    rows = acquisition_rows()
    chunks = round_robin_chunks(rows)
    completions = _load_completions()
    _verify_existing_completions(completions, rows)
    attempts = _recover_existing_evidence(rows, completions)
    _verify_existing_completions(completions, rows)
    runtime = _runtime_state()
    REPLAY_DIR.mkdir(exist_ok=True)
    active_policies: dict[str, Any] = {} if policies is None else policies

    for chunk_index, chunk in enumerate(chunks):
        pending = [row for row in chunk if _stream_key(row) not in completions]
        if not pending:
            continue
        _guard_execution(runtime)
        for row in pending:
            family = str(row["family"])
            if family not in active_policies:
                g1r_family = O3_TO_G1R[family]
                active_policies[family] = policy_source.load_policy(
                    g1r_family,
                    POLICY_SPECS[family],
                )
            attempt_number = _open_attempt(
                row,
                chunk_index=chunk_index,
                attempts=attempts,
            )
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
                        policy_name=POLICY_SPECS[family],
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
                raise ValueError("O3 evaluator returned an invalid game batch")
            completion = _store_output(outputs[0], stream_row=row)
            history._append_jsonl_row(COMPLETION_PATH, completion)
            completions[_stream_key(row)] = completion
            _append_attempt_event(
                row,
                attempt_number=attempt_number,
                status="completed",
                chunk_index=chunk_index,
            )
            attempts[_stream_key(row)][-1]["statuses"].append("completed")
        runtime["chunks_completed"] = int(runtime["chunks_completed"]) + 1
        runtime["games_completed"] = len(completions)
        write_json(RUNTIME_PATH, runtime)
        if len(completions) % 100 == 0:
            print(
                json.dumps(
                    {
                        "phase": "o3_acquisition",
                        "completed": len(completions),
                        "total": TOTAL_ROOTS,
                        "active_runtime_seconds": runtime[
                            "active_runtime_seconds"
                        ],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    expected = {_stream_key(row) for row in rows}
    if set(completions) != expected:
        raise ValueError("O3 acquisition did not complete the exact manifest")
    result = [completions[key] for key in sorted(expected)]
    roots = [str(row["root_cluster"]) for row in result]
    checks = {
        "exact_20500": len(result) == TOTAL_ROOTS,
        "equal_families": Counter(row["family"] for row in result)
        == {family: ROOTS_PER_FAMILY for family in FAMILY_ORDER},
        "unique_ancestries": len(roots) == len(set(roots)),
        "all_complete": all(row["complete"] for row in result),
        "all_replays_retained": all(
            Path(row["source_replay"]).is_file() for row in result
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"O3 acquisition completion integrity failed: {checks}")
    return result


def attempt_ledger_audit(
    rows: Sequence[Mapping[str, Any]],
    completions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    attempts = _load_attempt_ledger(rows)
    expected = {_stream_key(row) for row in rows}
    completed = {
        (str(row["family"]), int(row["game_index"])) for row in completions
    }
    statuses = Counter()
    opened = 0
    final_completed = 0
    for lifecycles in attempts.values():
        opened += len(lifecycles)
        for lifecycle in lifecycles:
            statuses.update(lifecycle["statuses"])
        if lifecycles and lifecycles[-1]["statuses"] in (
            ["opened", "completed"],
            ["opened", "completed_recovered"],
        ):
            final_completed += 1
    checks = {
        "every_root_attempted": set(attempts) == expected,
        "every_root_completed": completed == expected,
        "one_final_completion_per_root": final_completed == len(expected),
        "all_attempts_paired": all(
            len(lifecycle["statuses"]) == 2
            for lifecycles in attempts.values()
            for lifecycle in lifecycles
        ),
        "no_hidden_retries": opened
        == statuses["completed"]
        + statuses["completed_recovered"]
        + statuses["interrupted_no_replay"],
    }
    return {
        "attempt_rows": sum(statuses.values()),
        "attempts_opened": opened,
        "retries": opened - len(expected),
        "status_counts": dict(sorted(statuses.items())),
        "file_sha256": sha256_path(ATTEMPT_PATH),
        "checks": checks,
        "passes": all(checks.values()),
    }


def _candidate_hash(
    *,
    role: str,
    target: int,
    family: str,
    root: str,
    frame: int,
    state_hash: str,
) -> str:
    return hashlib.sha256(
        "|".join(
            (
                "O3-event-root-v1",
                role,
                str(target),
                family,
                root,
                str(frame),
                state_hash,
            )
        ).encode("ascii")
    ).hexdigest()


def _descriptive_stage(pair: Any) -> int:
    if pair.safe_merge_actions:
        return 3
    if int(pair.manhattan) == 1:
        return 2
    if int(pair.chebyshev) == 1:
        return 1
    return 0


def scan_support(
    completions: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(completions) != TOTAL_ROOTS:
        raise ValueError("O3 support scan requires all 20,500 completions")
    roots = [str(row["root_cluster"]) for row in completions]
    if len(roots) != len(set(roots)):
        raise ValueError("O3 support scan requires unique ancestries")
    best: dict[tuple[str, int], dict[str, Any]] = {}
    provenance_rows = []
    frames_scanned = 0
    for completion in sorted(
        completions,
        key=lambda row: (int(row["family_index"]), int(row["game_index"])),
    ):
        path = Path(str(completion["source_replay"]))
        if not path.is_file() or sha256_path(path) != completion[
            "source_replay_sha256"
        ]:
            raise ValueError(f"O3 retained replay changed: {path}")
        replay = json.loads(path.read_text())
        provenance = replay_provenance(replay, path)
        root = canonical_ancestry_id(replay, path)
        checks = {
            "root_matches": root == completion["root_cluster"],
            "fresh_replay": provenance.get("replay_origin") == ORIGIN_FRESH,
            "fresh_root": provenance.get("root_origin") == ORIGIN_FRESH,
            "reset_invariant": bool(provenance.get("replay_reset_invariant")),
            "seed": int(provenance.get("root_seed"))
            == int(completion["logical_seed"]),
        }
        if not all(checks.values()):
            raise ValueError(f"O3 support provenance failed: {checks}")
        provenance_rows.append(
            {
                "root_cluster": root,
                "family": completion["family"],
                "role": completion["role"],
                "source_replay": str(path),
                "source_replay_sha256": completion["source_replay_sha256"],
                "checks": checks,
            }
        )
        validator = ThreesSim.from_stream_ids(
            deck_stream_id=int(completion["deck_stream_id"]),
            slot_stream_id=int(completion["slot_stream_id"]),
            starter_tile=STARTER_TILE,
        )
        frames = replay.get("frames")
        if not isinstance(frames, list) or not frames:
            raise ValueError(f"O3 replay has no frames: {path}")
        for fallback, frame in enumerate(frames):
            if not isinstance(frame, dict):
                raise ValueError(f"O3 malformed frame: {path}:{fallback}")
            payload = frame.get("state")
            if not isinstance(payload, dict):
                raise ValueError(f"O3 malformed state: {path}:{fallback}")
            if bool(payload.get("game_over")):
                continue
            state = state_from_replay_payload(payload)
            legal = validator.legal_actions(state)
            expected_names = [DIRECTION_NAMES[action] for action in legal]
            if payload.get("legal_actions") != expected_names:
                raise ValueError(f"O3 legal-action mismatch: {path}:{fallback}")
            if not (
                anchor_safe(state.board, STARTER_TILE)
                and air_safe(state.board)
                and len(legal) >= 2
            ):
                continue
            frame_index = int(frame.get("index", fallback))
            state_hash = state_signature(payload, STARTER_TILE)
            frames_scanned += 1
            for target in TARGET_ORDER:
                pair = select_designated_pair(
                    state.board,
                    STARTER_TILE,
                    requested_target=target,
                    allowed_targets=TRAIN_TARGETS,
                )
                if pair is None or pair.safe_merge_actions:
                    continue
                if not root_option_eligible(
                    state,
                    validator,
                    STARTER_TILE,
                    allowed_targets=(target,),
                ):
                    continue
                selection = _candidate_hash(
                    role=str(completion["role"]),
                    target=target,
                    family=str(completion["family"]),
                    root=root,
                    frame=frame_index,
                    state_hash=state_hash,
                )
                row = {
                    "root_cluster": root,
                    "family": str(completion["family"]),
                    "family_index": int(completion["family_index"]),
                    "game_index": int(completion["game_index"]),
                    "role": str(completion["role"]),
                    "target": int(target),
                    "frame_index": frame_index,
                    "state_sha1": state_hash,
                    "selection_sha256": selection,
                    "pair": [list(coord) for coord in pair.coordinates],
                    "pair_manhattan": int(pair.manhattan),
                    "pair_chebyshev": int(pair.chebyshev),
                    "pair_blockers": int(pair.blocker_count),
                    "descriptive_stage": _descriptive_stage(pair),
                    "empty_count": int(np.count_nonzero(state.board == 0)),
                    "legal_count": len(legal),
                    "source_replay": str(path),
                    "source_replay_sha256": completion[
                        "source_replay_sha256"
                    ],
                }
                key = (root, target)
                prior = best.get(key)
                candidate_key = (selection, frame_index, state_hash)
                if prior is None or candidate_key < (
                    prior["selection_sha256"],
                    int(prior["frame_index"]),
                    prior["state_sha1"],
                ):
                    best[key] = row
    candidates = sorted(
        best.values(),
        key=lambda row: (
            ROLE_ORDER.index(str(row["role"])),
            TARGET_ORDER.index(int(row["target"])),
            row["selection_sha256"],
            row["root_cluster"],
        ),
    )
    audit = {
        "completion_roots": len(completions),
        "unique_roots": len(set(roots)),
        "frames_scanned": frames_scanned,
        "candidate_rows": len(candidates),
        "candidate_roots": len({row["root_cluster"] for row in candidates}),
        "candidate_counts_by_role_target": {
            f"{role}/T{target}": sum(
                row["role"] == role and int(row["target"]) == target
                for row in candidates
            )
            for role in ROLE_ORDER
            for target in TARGET_ORDER
        },
        "provenance_rows": provenance_rows,
        "provenance_manifest_sha256": canonical_json_hash(provenance_rows),
        "score_field_access": (
            "reset/root provenance score may be read only inside "
            "replay_provenance; final/future score is not accessed or used"
        ),
        "recorded_actions_accessed": False,
        "final_or_future_milestone_or_max_fields_accessed": False,
        "policy_outcomes_compared": False,
        "stage_used_only_descriptively": True,
        "passes": len(provenance_rows) == TOTAL_ROOTS,
    }
    return candidates, audit


def allocate_candidates(
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    selected = []
    used_roots: set[str] = set()
    deficits = []
    for role_index, role in enumerate(ROLE_ORDER):
        role_selected = []
        for target_index, target in enumerate(TARGET_ORDER):
            quota = int(TARGET_COUNTS[role][target])
            family_start = (role_index + target_index) % len(FAMILY_ORDER)
            family_cycle = (
                FAMILY_ORDER[family_start:] + FAMILY_ORDER[:family_start]
            )
            queues = {
                family: sorted(
                    (
                        dict(row)
                        for row in candidates
                        if row["role"] == role
                        and int(row["target"]) == target
                        and row["family"] == family
                    ),
                    key=lambda row: (
                        row["selection_sha256"],
                        row["root_cluster"],
                    ),
                )
                for family in FAMILY_ORDER
            }
            cursor = {family: 0 for family in FAMILY_ORDER}
            picked = 0
            while picked < quota:
                progress = False
                for family in family_cycle:
                    queue = queues[family]
                    while (
                        cursor[family] < len(queue)
                        and queue[cursor[family]]["root_cluster"] in used_roots
                    ):
                        cursor[family] += 1
                    if cursor[family] >= len(queue):
                        continue
                    row = queue[cursor[family]]
                    cursor[family] += 1
                    used_roots.add(str(row["root_cluster"]))
                    role_selected.append(row)
                    picked += 1
                    progress = True
                    if picked == quota:
                        break
                if not progress:
                    break
            if picked != quota:
                deficits.append(
                    {
                        "role": role,
                        "target": target,
                        "required": quota,
                        "selected": picked,
                    }
                )
        selected.extend(role_selected)
    per_role = {}
    for role in ROLE_ORDER:
        rows = [row for row in selected if row["role"] == role]
        counts = Counter(str(row["family"]) for row in rows)
        total = int(p0.SELECTED_COUNTS[role])
        represented = {family: count for family, count in counts.items() if count}
        checks = {
            "exact_role_total": len(rows) == total,
            "target_counts_exact": Counter(int(row["target"]) for row in rows)
            == TARGET_COUNTS[role],
            "at_least_four_families": len(represented) >= 4,
            "family_share_at_most_40pct": (
                max(counts.values(), default=0) / total <= MAX_FAMILY_SHARE
            ),
            "represented_family_minimum": all(
                count >= MIN_FAMILY_COUNTS[role]
                for count in represented.values()
            ),
        }
        per_role[role] = {
            "selected": len(rows),
            "target_counts": dict(
                sorted(Counter(int(row["target"]) for row in rows).items())
            ),
            "family_counts": dict(sorted(counts.items())),
            "max_family_share": max(counts.values(), default=0) / total,
            "descriptive_stage_counts": dict(
                sorted(
                    Counter(int(row["descriptive_stage"]) for row in rows).items()
                )
            ),
            "checks": checks,
            "passes": all(checks.values()),
        }
    root_counts = Counter(str(row["root_cluster"]) for row in selected)
    checks = {
        "no_allocator_deficits": not deficits,
        "one_state_per_whole_ancestry": all(
            count == 1 for count in root_counts.values()
        ),
        "exact_total_320": len(selected) == sum(p0.SELECTED_COUNTS.values()),
        "all_role_gates_pass": all(row["passes"] for row in per_role.values()),
        "stage_not_a_gate": True,
    }
    compact = [
        dict(row)
        for row in sorted(
            selected,
            key=lambda row: (
                ROLE_ORDER.index(str(row["role"])),
                TARGET_ORDER.index(int(row["target"])),
                row["selection_sha256"],
            ),
        )
    ]
    return {
        "selection_rule": (
            "rare-first targets 192,96,48; role+target rotated family "
            "round robin; candidate SHA order; one state per ancestry"
        ),
        "selected": compact,
        "selected_manifest_sha256": canonical_json_hash(compact),
        "per_role": per_role,
        "deficits": deficits,
        "checks": checks,
        "passes": all(checks.values()),
    }


def _terminal_zero_forbidden_work() -> dict[str, Any]:
    return {
        "labels_generated": 0,
        "models_fit": 0,
        "option_rollouts": 0,
        "h10_h20_h40_outcomes": 0,
        "normal_start_policy_evaluations": 0,
        "policy_outcomes_compared": False,
        "scores_inspected_or_reported": False,
        "recorded_actions_inspected_or_reported": False,
        "dashboard_changes": 0,
        "incumbent_changes": 0,
        "o2_row_level_content_read": False,
    }


def _seal_terminal_error(error: Exception) -> dict[str, Any]:
    result = {
        "version": VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "decision": "HOLD_O3_ACQUISITION_INTEGRITY",
        "continue": "NONE",
        "hold": [
            "acquisition_retry",
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
        "marker": _artifact_identity(MARKER_PATH, "opened_payload_sha256"),
        "attempt_rows": (
            len([line for line in ATTEMPT_PATH.read_text().splitlines() if line])
            if ATTEMPT_PATH.is_file()
            else 0
        ),
        "completion_rows": len(_load_completions()),
        "runtime": _runtime_state(),
        "output_bytes": (
            history._directory_bytes(OUTPUT_DIR) if OUTPUT_DIR.exists() else 0
        ),
        "zero_forbidden_work": _terminal_zero_forbidden_work(),
        "dashboard_eligible": False,
    }
    return _write_immutable_json(
        RESULT_PATH,
        result,
        self_hash_field="result_payload_sha256",
    )


def execute(
    *,
    out_dir: Path = OUTPUT_DIR,
    jobs: int = FROZEN_JOBS,
) -> dict[str, Any]:
    marker = _load_marker(out_dir=out_dir, jobs=jobs)
    try:
        if not _sealed_artifact_audit()["passes"]:
            raise ValueError("O3 P0 immutable inputs changed before acquisition")
        policies, policy_audit = _load_policies()
        if not policy_audit["passes"]:
            raise ValueError("O3 policy lock changed before acquisition")
        rows = acquisition_rows()
        collision = collision_audit(rows, out_dir=out_dir)
        if not collision["passes"]:
            raise ValueError("O3 acquisition stream collision appeared")
        _guard_execution(_runtime_state(), out_dir=out_dir)
        completions = collect_all(marker, jobs=jobs, policies=policies)
        _guard_execution(_runtime_state(), out_dir=out_dir)
        attempts = attempt_ledger_audit(rows, completions)
        if not attempts["passes"]:
            raise ValueError("O3 acquisition attempt ledger integrity failed")
        candidates, support_audit = scan_support(completions)
        allocation = allocate_candidates(candidates)
        support = _write_immutable_json(
            SUPPORT_PATH,
            {
                "version": f"{VERSION}_support",
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
        terminal_collision = collision_audit(rows, out_dir=out_dir)
        terminal_operations = _guard_execution(
            _runtime_state(), out_dir=out_dir
        )
        decision = (
            "READY_O3_OPTION_TRAINING"
            if allocation["passes"] and support_audit["passes"]
            else "HOLD_O3_DATA_OR_POWER"
        )
        result = {
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
            "attempt_ledger": attempts,
            "completion_rows": len(completions),
            "completion_manifest_sha256": canonical_json_hash(
                sorted(
                    (dict(row) for row in completions),
                    key=lambda row: (
                        int(row["family_index"]),
                        int(row["game_index"]),
                    ),
                )
            ),
            "games_by_family": dict(
                sorted(Counter(row["family"] for row in completions).items())
            ),
            "unique_ancestries": len(
                {row["root_cluster"] for row in completions}
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
                "support_payload_sha256": support[
                    "support_payload_sha256"
                ],
                "selected_payload_sha256": selected[
                    "selected_payload_sha256"
                ],
            },
            "runtime": _runtime_state(),
            "output_bytes": history._directory_bytes(out_dir),
            "terminal_collision": {
                "matched_source_count": terminal_collision[
                    "matched_source_count"
                ],
                "matched_sources_sha256": terminal_collision[
                    "matched_sources_sha256"
                ],
                "collisions": terminal_collision["collisions"],
                "passes": terminal_collision["passes"],
            },
            "terminal_operations": terminal_operations,
            "zero_forbidden_work": _terminal_zero_forbidden_work(),
            "dashboard_eligible": False,
        }
        return _write_immutable_json(
            RESULT_PATH,
            result,
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
        result = open_execution(out_dir=args.out_dir, jobs=args.jobs)
    else:
        result = execute(out_dir=args.out_dir, jobs=args.jobs)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
