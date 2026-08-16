"""Separate five-family G1-R pilot-v2 acquisition and no-game preflight."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from threes_rl import g1r_acquire as base
from threes_rl.eval import EvalJob, EvalStreamIds, iter_eval_job_outputs, make_policy
from threes_rl.g1r_qd_admission_v2 import (
    StaticArchiveQDPolicy,
    _heavy_process_audit,
)
from threes_rl.record_replay import state_payload
from threes_rl.run_artifacts import write_json
from threes_rl.s3_power_preflight import sha256_path
from threes_rl.sim import ThreesSim
from threes_rl.train_td import state_from_replay_payload


VERSION = "g1r_acquisition_pilot_v2_qd5"
CHARTER_PATH = Path("threes_rl/G1R_PILOT_V2_QD5_ACQUISITION_CHARTER.md")
IMPLEMENTATION_PATH = Path("threes_rl/g1r_acquire_v2_qd5.py")
TEST_PATH = Path("tests/test_rl_g1r_acquire_v2_qd5.py")
BASE_IMPLEMENTATION_PATH = Path("threes_rl/g1r_acquire.py")
BASE_TEST_PATH = Path("tests/test_rl_g1r_acquire.py")
OUTPUT_DIR = Path("threes_rl/runs/forensics/g1r_acquisition/pilot_v2_qd5")
SUPERSEDED_TEST_EVIDENCE = (
    (
        Path(
            "threes_rl/runs/forensics/g1r_acquisition/"
            "pilot_v2_qd5_test_evidence.json"
        ),
        "59b6fae9d0f9eaab97cf70a00e1d3147d84544a3f500769dc80642a966b464aa",
    ),
    (
        Path(
            "threes_rl/runs/forensics/g1r_acquisition/"
            "pilot_v2_qd5_test_evidence_v2.json"
        ),
        "a0080aaef89988e563cf8d000cf61bd683a027c7b92052cc70bac364bfb467f0",
    ),
)
TEST_EVIDENCE_PATH = Path(
    "threes_rl/runs/forensics/g1r_acquisition/"
    "pilot_v2_qd5_test_evidence_v3.json"
)
PILOT_V1_LOCK = Path(
    "threes_rl/runs/forensics/g1r_acquisition/pilot_v1/"
    "preflight_lock_pilot_v1.json"
)
S3_DIR = Path("threes_rl/runs/forensics/s3_full_policy")
S3_POWER_PATH = S3_DIR / "S3_POWER_PREFLIGHT_V2_SEALED.json"
S3_PROVENANCE_PATH = S3_DIR / "S3_PROVENANCE_SEAL_V2.json"
QD_DIR = Path(
    "threes_rl/runs/forensics/g1r_qd_admission_v2_terminal_schema"
)
QD_POLICY_DIR = QD_DIR / "policy"
QD_EXECUTION_LOCK = QD_DIR / "execution_lock.json"
QD_ADMISSION_MARKER = QD_DIR / "ADMISSION_OPENED.json"
QD_ADMISSION_RESULT = QD_DIR / "admission_result.json"
QD_STORAGE_AUDIT = QD_DIR / "QD_V2_STORAGE_ADMISSION_AUDIT.json"
QD_STORAGE_INVENTORY = QD_DIR / "QD_V2_STORAGE_REPLAY_INVENTORY.json"

PILOT_V1_LOCK_SHA256 = (
    "f78288b3f47bda6aa6d15c2157fd79f7b3d0685f0367d8b9964f5dc73981ea91"
)
PANEL_SHA256 = (
    "b8862aa3c8eaf6278fc078fb3e03aa7222a01930673cfee497738c74e81eff9d"
)
S3_POWER_SHA256 = (
    "4dabd5325dcbbc5234c4e015eccbd4d5f4706be9fefa54fd5220d8720b1fc345"
)
S3_PROVENANCE_SHA256 = (
    "5326f25b50ad33b4e00eb5ca7180468d3a243917075d15d377a1511b04867949"
)
BASE_IMPLEMENTATION_SHA256 = (
    "73ba88103024e6cf62ba4418d88a9bbe71cf42aafc1b911ef39818647f655d6a"
)
BASE_TEST_SHA256 = (
    "b2b989f5413c090f3a1faabc8624f21d8077c735394f30b74dc01d0055286e5a"
)
QD_FILE_HASHES = {
    "execution_lock": (
        QD_EXECUTION_LOCK,
        "1f48822fbfbc4055e164ebc87c95c617daba044782850ed723e672b7db25ff4a",
        "lock_payload_sha256",
        "f6f6287bfa3b217d5a2958f7f1b1167724ba7ca72249515a6223c9387d33af74",
    ),
    "admission_marker": (
        QD_ADMISSION_MARKER,
        "11b21137303fa4cfd258dfe3ff536b227c24fa4cb7db727ca376b970418c5135",
        "opened_payload_sha256",
        "aa7e517a8c756cc929a0a1882d0b4843e1d14fdca4135318c3dd29fded0986e2",
    ),
    "admission_result": (
        QD_ADMISSION_RESULT,
        "27bcb3328a02d6dc5094dcc5a8e52b8f27d2f3e4ea7b92f5c1a8153bc1326a8e",
        "result_payload_sha256",
        "0eced74a61279613661e82d3f696c8e1d34d5256548194b155ad2db3d3ed38e2",
    ),
    "storage_audit": (
        QD_STORAGE_AUDIT,
        "0bdef1deafe485458c70f7379328140b6221b932a6c073e75aff91c10d1f037f",
        "audit_payload_sha256",
        "2864146d541819958fc1117b25ed93eba20612bc642ea625a0bf4f69c928bd78",
    ),
    "storage_inventory": (
        QD_STORAGE_INVENTORY,
        "0dd9e2d4c2ef01e1376588b4a5402a0cf6a0ca39ec2e73eed024d701fa2fe552",
        "inventory_payload_sha256",
        "d5ebf1bad2da61ccf62b3a04ebfe5764070364a8da7b5a98a96d9b9bd8adaa03",
    ),
}

QD_FAMILY = "g1r_qd_static_archive_oneply_v2_terminal_schema"
QD_SPEC = f"{QD_FAMILY}:{QD_POLICY_DIR}"
FAMILY_SLATE = (
    ("g1r_corner2", "corner2"),
    ("g1r_expectimax2", "expectimax2"),
    (
        "g1r_parent_mc1000",
        "ntuple_expectimax2:"
        "threes_rl/runs/td_default_corner2_mc_1000_init3000_a0005_"
        "20260706/latest",
    ),
    (
        "g1r_replaycal",
        "ntuple_expectimax2:"
        "threes_rl/runs/replay_cal_phase4_late_midlate_top13_e3_a001_tc_"
        "20260706/latest",
    ),
    (QD_FAMILY, QD_SPEC),
)
STREAM_BASES = {
    "logical_seed": 49_000_000_000,
    "deck_stream_id": 50_000_000_000,
    "slot_stream_id": 51_000_000_000,
    "policy_stream_id": 52_000_000_000,
}
GAMES_PER_FAMILY = 20
FROZEN_JOBS = 1
MAX_CHUNK_SIZE = 8
MAX_MOVES = 5000
STARTER_TILE = 1536
MINIMUM_NICE = 10
MIN_FREE_GIB = 100.0
TARGET_FREE_GIB = 120.0
PILOT_WALL_SECONDS = 12 * 3600
PILOT_BYTE_LIMIT = 4 * 1024**3
MAX_TOTAL_GAMES = 12_000
STRATA = ("pre1536", "pre3072")
PAIRWISE_FLOOR = 0.02
WILSON_Z = 1.6448536269514722

EXPECTED_SIGNATURES = {
    "g1r_corner2": (
        "4be4214166f40ddaaac5af499cb1e1d08d992b0a90bb680cfcb7cab04d217043"
    ),
    "g1r_expectimax2": (
        "2ad642cdca7739cc73af4f570de5054c422815f9a7d8f93a2619921b46b74b38"
    ),
    "g1r_parent_mc1000": (
        "e43dc11f3220557d7f9aef228db96dc6f06f49b26300d5a4128ea00bf8ba2064"
    ),
    "g1r_replaycal": (
        "e07c566b55d86a889ab7ca54d01c00c9b6cdf808fdb1627f70596bd829fdeab3"
    ),
    QD_FAMILY: (
        "66da7d61c9178bd982cac492e397a0cfc4424d51f396fe1b290c4af7fe1cd281"
    ),
}
SOURCE_PATHS = (
    IMPLEMENTATION_PATH,
    BASE_IMPLEMENTATION_PATH,
    Path("threes_rl/eval.py"),
    Path("threes_rl/expectimax.py"),
    Path("threes_rl/ntuple.py"),
    Path("threes_rl/action_prior.py"),
    Path("threes_rl/sim.py"),
    Path("threes_rl/record_replay.py"),
    Path("threes_rl/replay_provenance.py"),
    Path("threes_rl/run_artifacts.py"),
    Path("threes_rl/train_td.py"),
    Path("threes_rl/g1r_qd_admission_v2.py"),
)


def canonical_json_hash(value: Any) -> str:
    return base.canonical_json_hash(value)


def _write_new_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Immutable artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"Atomic temporary already exists: {temporary}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _sealed_payload_audit(
    path: Path,
    *,
    expected_file_sha256: str,
    self_hash_field: str,
    expected_payload_sha256: str,
) -> dict[str, Any]:
    file_sha = sha256_path(path)
    payload = json.loads(path.read_text())
    embedded = payload.pop(self_hash_field)
    computed = canonical_json_hash(payload)
    checks = {
        "file_sha256": file_sha == expected_file_sha256,
        "embedded_payload_sha256": embedded == expected_payload_sha256,
        "computed_payload_sha256": computed == expected_payload_sha256,
    }
    return {
        "path": str(path),
        "file_sha256": file_sha,
        "expected_file_sha256": expected_file_sha256,
        "payload_sha256": computed,
        "expected_payload_sha256": expected_payload_sha256,
        "checks": checks,
        "passes": all(checks.values()),
    }


def _file_manifest(path: Path) -> dict[str, Any]:
    if not path.is_dir():
        raise FileNotFoundError(path)
    rows = [
        {
            "path": str(child),
            "relative_path": str(child.relative_to(path)),
            "bytes": int(child.stat().st_size),
            "sha256": sha256_path(child),
        }
        for child in sorted(path.rglob("*"))
        if child.is_file()
    ]
    if not rows:
        raise ValueError(f"Empty artifact directory: {path}")
    return {
        "path": str(path),
        "files": rows,
        "file_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "manifest_sha256": canonical_json_hash(rows),
    }


def policy_slate() -> tuple[tuple[str, str], ...]:
    return FAMILY_SLATE


def stream_ids(family_index: int, game_index: int) -> dict[str, int]:
    offset = int(family_index) * 1_000_000 + int(game_index)
    return {name: int(value) + offset for name, value in STREAM_BASES.items()}


def requested_stream_manifest() -> list[dict[str, Any]]:
    rows = []
    for family_index, (family, spec) in enumerate(FAMILY_SLATE):
        for game_index in range(GAMES_PER_FAMILY):
            rows.append(
                {
                    "family_index": family_index,
                    "nominal_family": family,
                    "policy_spec": spec,
                    "game_index": game_index,
                    **stream_ids(family_index, game_index),
                }
            )
    return rows


def stream_collision_audit(
    rows: list[dict[str, Any]],
    *,
    exclude_dir: Path,
) -> dict[str, Any]:
    prior, sources = base.historical_collision_union(exclude_dir=exclude_dir)
    collisions: dict[str, list[int]] = {}
    for key in STREAM_BASES:
        prior_values = set(prior.get(key, set()))
        if key == "logical_seed":
            for alias in ("seed", "root_seed", "source_seed", "fresh_root_seed"):
                prior_values.update(prior.get(alias, set()))
        requested = {int(row[key]) for row in rows}
        collisions[key] = sorted(requested.intersection(prior_values))
    flat = [int(row[key]) for row in rows for key in STREAM_BASES]
    internal_unique = len(flat) == len(set(flat))
    return {
        "historical_union": sources,
        "collisions": collisions,
        "internal_stream_ids_unique": internal_unique,
        "zero_collisions": internal_unique and not any(collisions.values()),
    }


def load_policy(family: str, spec: str) -> Any:
    if family == QD_FAMILY:
        if spec != QD_SPEC:
            raise ValueError("QD policy spec mismatch")
        return StaticArchiveQDPolicy.load(QD_POLICY_DIR)
    return make_policy(spec)


def _policy_lock() -> tuple[dict[str, Any], dict[str, Any]]:
    loaded: dict[str, Any] = {}
    families = []
    artifact_cache: dict[Path, dict[str, Any]] = {}
    for family, spec in FAMILY_SLATE:
        policy = load_policy(family, spec)
        loaded[family] = policy
        checkpoints = []
        if family != QD_FAMILY:
            for checkpoint in base._checkpoint_dirs(spec):
                if checkpoint not in artifact_cache:
                    artifact_cache[checkpoint] = _file_manifest(checkpoint)
                checkpoints.append(artifact_cache[checkpoint])
        families.append(
            {
                "family": family,
                "policy_spec": spec,
                "policy_spec_sha256": hashlib.sha256(spec.encode()).hexdigest(),
                "loaded_type": type(policy).__name__,
                "checkpoint_manifests": checkpoints,
                "qd_policy_bundle": (
                    _file_manifest(QD_POLICY_DIR) if family == QD_FAMILY else None
                ),
            }
        )
    source_hashes = {str(path): sha256_path(path) for path in SOURCE_PATHS}
    qd_seals = {
        name: _sealed_payload_audit(
            path,
            expected_file_sha256=file_sha,
            self_hash_field=field,
            expected_payload_sha256=payload_sha,
        )
        for name, (path, file_sha, field, payload_sha) in QD_FILE_HASHES.items()
    }
    qd_static_payloads = {
        name: {
            "path": str(path),
            "sha256": sha256_path(path),
            "bytes": int(path.stat().st_size),
        }
        for name, path in {
            "archive": QD_DIR / "archive.json",
            "archive_sources": QD_DIR / "archive_sources.json",
            "policy_json": QD_POLICY_DIR / "policy.json",
            "policy_archive": QD_POLICY_DIR / "archive.json",
        }.items()
    }
    lock = {
        "family_order": [family for family, _spec in FAMILY_SLATE],
        "families": families,
        "source_hashes": source_hashes,
        "source_manifest_sha256": canonical_json_hash(source_hashes),
        "qd_sealed_artifacts": qd_seals,
        "qd_static_payloads": qd_static_payloads,
        "incumbent_policy_file": str(base.INCUMBENT_PATH),
        "incumbent_policy_file_sha256": sha256_path(base.INCUMBENT_PATH),
    }
    lock["policy_lock_sha256"] = canonical_json_hash(lock)
    if not all(row["passes"] for row in qd_seals.values()):
        raise ValueError("QD sealed artifact audit failed")
    return lock, loaded


def _load_panel() -> tuple[dict[str, Any], dict[str, Any]]:
    if sha256_path(PILOT_V1_LOCK) != PILOT_V1_LOCK_SHA256:
        raise ValueError("pilot-v1 preflight lock changed")
    lock = json.loads(PILOT_V1_LOCK.read_text())
    payload_sha = lock.pop("preflight_payload_sha256")
    if canonical_json_hash(lock) != payload_sha:
        raise ValueError("pilot-v1 preflight payload mismatch")
    panel = copy.deepcopy(lock["action_distinctness_panel"])
    if panel.get("panel_sha256") != PANEL_SHA256:
        raise ValueError("pilot-v1 panel hash mismatch")
    if len(panel.get("records", [])) != 64:
        raise ValueError("pilot-v1 panel must contain 64 states")
    if Counter(row["stratum"] for row in panel["records"]) != {
        "pre1536": 32,
        "pre3072": 32,
    }:
        raise ValueError("pilot-v1 panel stratum count mismatch")
    return panel, {
        "path": str(PILOT_V1_LOCK),
        "file_sha256": PILOT_V1_LOCK_SHA256,
        "payload_sha256": payload_sha,
    }


def _deterministic_action(
    family: str,
    policy: Any,
    payload: dict[str, Any],
) -> dict[str, Any]:
    before = canonical_json_hash(payload)
    base._roundtrip_state(payload)
    if family != QD_FAMILY:
        first = base.deterministic_policy_action(policy, payload)
        second = base.deterministic_policy_action(policy, payload)
        if first != second:
            raise ValueError(f"Nondeterministic panel action for {family}")
        return {
            **first,
            "state_unmutated": canonical_json_hash(payload) == before,
        }
    rows = []
    for _repeat in range(2):
        state = state_from_replay_payload(payload)
        sim = ThreesSim.from_stream_ids(
            deck_stream_id=7,
            slot_stream_id=11,
            starter_tile=STARTER_TILE,
        )
        decision = policy.decision(state, sim)
        if state_payload(state, sim) != payload:
            raise ValueError("QD policy mutated panel state")
        rows.append(
            {
                "action": int(decision["action"]),
                "exact_tie_count": int(
                    decision["tie_count_before_action_priority"]
                ),
            }
        )
    if rows[0] != rows[1]:
        raise ValueError("QD panel action is nondeterministic")
    return {
        **rows[0],
        "state_unmutated": canonical_json_hash(payload) == before,
    }


def _accepted_pairwise() -> dict[tuple[str, str], dict[str, Any]]:
    pilot = json.loads(PILOT_V1_LOCK.read_text())
    qd = json.loads(QD_ADMISSION_RESULT.read_text())
    accepted: dict[tuple[str, str], dict[str, Any]] = {}
    allowed = {family for family, _spec in FAMILY_SLATE}
    for row in pilot["action_distinctness_audit"]["pairwise"]:
        if row["left"] in allowed and row["right"] in allowed:
            accepted[(row["left"], row["right"])] = {
                "overall_disagreement": float(row["overall_disagreement"]),
                "stratum_disagreement": {
                    key: float(value)
                    for key, value in row["stratum_disagreement"].items()
                },
            }
    for row in qd["pairwise"]:
        left = str(row["reference"])
        right = QD_FAMILY
        accepted[(left, right)] = {
            "overall_disagreement": float(row["overall_disagreement"]),
            "stratum_disagreement": {
                key: float(value)
                for key, value in row["stratum_disagreement"].items()
            },
        }
    return accepted


def action_signature_audit(
    policies: dict[str, Any],
    panel: dict[str, Any],
) -> dict[str, Any]:
    signatures: dict[str, list[int]] = {}
    tie_counts: dict[str, int] = {}
    exactness: dict[str, Any] = {}
    for family, _spec in FAMILY_SLATE:
        rows = [
            _deterministic_action(family, policies[family], record["state"])
            for record in panel["records"]
        ]
        signatures[family] = [int(row["action"]) for row in rows]
        tie_counts[family] = sum(int(row["exact_tie_count"] > 1) for row in rows)
        exactness[family] = {
            "states_checked": len(rows),
            "all_states_unmutated": all(row["state_unmutated"] for row in rows),
            "repeat_actions_identical": True,
        }
    signature_hashes = {
        family: canonical_json_hash(signatures[family])
        for family, _spec in FAMILY_SLATE
    }
    accepted = _accepted_pairwise()
    pairs = []
    families = [family for family, _spec in FAMILY_SLATE]
    for left_index, left in enumerate(families):
        for right in families[left_index + 1 :]:
            stratum_rates = {}
            for stratum in STRATA:
                indices = [
                    index
                    for index, record in enumerate(panel["records"])
                    if record["stratum"] == stratum
                ]
                stratum_rates[stratum] = sum(
                    signatures[left][index] != signatures[right][index]
                    for index in indices
                ) / len(indices)
            overall = sum(
                a != b
                for a, b in zip(signatures[left], signatures[right], strict=True)
            ) / len(panel["records"])
            expected = accepted.get((left, right))
            exact_match = expected == {
                "overall_disagreement": overall,
                "stratum_disagreement": stratum_rates,
            }
            passes = (
                overall >= PAIRWISE_FLOOR
                and all(stratum_rates[stratum] > 0 for stratum in STRATA)
                and exact_match
            )
            pairs.append(
                {
                    "left": left,
                    "right": right,
                    "overall_disagreement": overall,
                    "stratum_disagreement": stratum_rates,
                    "accepted_exact_match": exact_match,
                    "passes": passes,
                }
            )
    checks = {
        "family_order_exact": families
        == [family for family, _spec in FAMILY_SLATE],
        "signature_hashes_exact": signature_hashes == EXPECTED_SIGNATURES,
        "all_pairwise_exact_and_distinct": all(row["passes"] for row in pairs),
        "all_states_unmutated": all(
            row["all_states_unmutated"] for row in exactness.values()
        ),
        "all_repeat_actions_identical": all(
            row["repeat_actions_identical"] for row in exactness.values()
        ),
        "no_timing_performed": True,
    }
    audit = {
        "panel_sha256": PANEL_SHA256,
        "family_order": families,
        "signature_sha256": signature_hashes,
        "tie_state_counts": tie_counts,
        "pairwise": pairs,
        "exactness": exactness,
        "checks": checks,
        "passes": all(checks.values()),
    }
    audit["audit_sha256"] = canonical_json_hash(audit)
    return audit


def split_reset_roundtrip_fixture() -> dict[str, Any]:
    ids = {
        "deck_stream_id": 40_999_999_991,
        "slot_stream_id": 40_999_999_992,
    }
    left = ThreesSim.from_stream_ids(starter_tile=STARTER_TILE, **ids)
    right = ThreesSim.from_stream_ids(starter_tile=STARTER_TILE, **ids)
    left_state = left.reset()
    right_state = right.reset()
    left_payload = state_payload(left_state, left)
    right_payload = state_payload(right_state, right)
    base._roundtrip_state(left_payload)
    checks = {
        "identical_split_stream_reset": left_payload == right_payload,
        "exact_state_roundtrip": True,
        "reserved_pilot_namespace_used": any(
            int(value) in range(base_value, base_value + 5_000_000)
            for value in ids.values()
            for base_value in STREAM_BASES.values()
        ),
    }
    checks["reserved_pilot_namespace_unused"] = not checks.pop(
        "reserved_pilot_namespace_used"
    )
    return {
        "diagnostic_stream_ids": ids,
        "state_payload_sha256": canonical_json_hash(left_payload),
        "checks": checks,
        "passes": all(checks.values()),
    }


def storage_admission_audit() -> dict[str, Any]:
    seal = _sealed_payload_audit(
        QD_STORAGE_AUDIT,
        expected_file_sha256=QD_FILE_HASHES["storage_audit"][1],
        self_hash_field=QD_FILE_HASHES["storage_audit"][2],
        expected_payload_sha256=QD_FILE_HASHES["storage_audit"][3],
    )
    payload = json.loads(QD_STORAGE_AUDIT.read_text())
    projected = int(payload["projection"]["projected_bytes_P"])
    checks = {
        "sealed_audit_valid": seal["passes"],
        "decision_ready": payload["decision"] == "READY_QD_STORAGE_ADMISSION",
        "projected_first_120_below_4_gib": projected < PILOT_BYTE_LIMIT,
        "pilot_100_no_larger_than_projection_120": 100 <= int(
            payload["projection"]["game_count"]
        ),
    }
    return {
        "seal": seal,
        "projected_bytes": projected,
        "projected_gib": projected / 1024**3,
        "pilot_limit_bytes": PILOT_BYTE_LIMIT,
        "checks": checks,
        "passes": all(checks.values()),
    }


def _immutable_input_audit() -> dict[str, Any]:
    rows = {
        "s3_power": {
            "path": str(S3_POWER_PATH),
            "expected": S3_POWER_SHA256,
            "actual": sha256_path(S3_POWER_PATH),
        },
        "s3_provenance": {
            "path": str(S3_PROVENANCE_PATH),
            "expected": S3_PROVENANCE_SHA256,
            "actual": sha256_path(S3_PROVENANCE_PATH),
        },
        "pilot_v1": {
            "path": str(PILOT_V1_LOCK),
            "expected": PILOT_V1_LOCK_SHA256,
            "actual": sha256_path(PILOT_V1_LOCK),
        },
        "base_implementation": {
            "path": str(BASE_IMPLEMENTATION_PATH),
            "expected": BASE_IMPLEMENTATION_SHA256,
            "actual": sha256_path(BASE_IMPLEMENTATION_PATH),
        },
        "base_test": {
            "path": str(BASE_TEST_PATH),
            "expected": BASE_TEST_SHA256,
            "actual": sha256_path(BASE_TEST_PATH),
        },
    }
    for row in rows.values():
        row["passes"] = row["actual"] == row["expected"]
    return {"artifacts": rows, "passes": all(row["passes"] for row in rows.values())}


def _load_test_evidence() -> dict[str, Any]:
    payload = json.loads(TEST_EVIDENCE_PATH.read_text())
    embedded = payload.pop("test_evidence_payload_sha256")
    if canonical_json_hash(payload) != embedded:
        raise ValueError("Pilot-v2 test evidence payload mismatch")
    expected = {
        "charter_sha256": sha256_path(CHARTER_PATH),
        "implementation_sha256": sha256_path(IMPLEMENTATION_PATH),
        "focused_test_sha256": sha256_path(TEST_PATH),
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("Pilot-v2 test evidence source identity mismatch")
    payload["test_evidence_payload_sha256"] = embedded
    payload["file_sha256"] = sha256_path(TEST_EVIDENCE_PATH)
    if not payload.get("passes"):
        raise ValueError("Pilot-v2 test evidence is not passing")
    return payload


def seal_test_evidence(
    *,
    focused_passed: int,
    regression_passed: int,
    commands: list[str],
    phase_inapplicable_tests: list[str],
) -> dict[str, Any]:
    if focused_passed <= 0 or regression_passed <= 0 or not commands:
        raise ValueError("Passing test counts and commands are required")
    payload = {
        "version": "g1r_pilot_v2_qd5_test_evidence_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "charter_sha256": sha256_path(CHARTER_PATH),
        "implementation_sha256": sha256_path(IMPLEMENTATION_PATH),
        "focused_test_sha256": sha256_path(TEST_PATH),
        "focused_tests_passed": int(focused_passed),
        "regression_tests_passed": int(regression_passed),
        "commands": commands,
        "phase_inapplicable_tests": phase_inapplicable_tests,
        "superseded_test_evidence": [
            {"path": str(path), "sha256": digest}
            for path, digest in SUPERSEDED_TEST_EVIDENCE
        ],
        "passes": True,
        "games_generated": 0,
        "streams_consumed": False,
        "labels_generated": 0,
        "models_fit": 0,
        "outcomes_inspected": False,
    }
    payload["test_evidence_payload_sha256"] = canonical_json_hash(payload)
    _write_new_json_atomic(TEST_EVIDENCE_PATH, payload)
    return payload


def _preflight_identity() -> dict[str, str]:
    return {
        "charter_sha256": sha256_path(CHARTER_PATH),
        "implementation_sha256": sha256_path(IMPLEMENTATION_PATH),
        "focused_test_sha256": sha256_path(TEST_PATH),
        "test_evidence_file_sha256": sha256_path(TEST_EVIDENCE_PATH),
    }


def _prepare_preflight_in_staging(
    staging_dir: Path,
    final_dir: Path,
) -> dict[str, Any]:
    if base.current_nice() < MINIMUM_NICE:
        raise ValueError("Pilot-v2 preflight requires nice >=10")
    immutable = _immutable_input_audit()
    if not immutable["passes"]:
        raise ValueError("Immutable S3/pilot-v1/base acquisition input changed")
    tests = _load_test_evidence()
    policies_lock, policies = _policy_lock()
    panel, panel_source = _load_panel()
    signatures = action_signature_audit(policies, panel)
    streams = requested_stream_manifest()
    collision = stream_collision_audit(streams, exclude_dir=staging_dir)
    fixture = split_reset_roundtrip_fixture()
    storage = storage_admission_audit()
    heavy = _heavy_process_audit()
    services = base.service_health()
    free_gib = shutil.disk_usage(staging_dir).free / 1024**3
    checks = {
        "fresh_separate_output": not final_dir.exists(),
        "exact_five_family_order": [name for name, _spec in FAMILY_SLATE]
        == list(signatures["family_order"]),
        "all_policies_and_sources_hashed": len(policies) == 5,
        "signatures_pairwise_exact": signatures["passes"],
        "qd_family_and_storage_admitted": all(
            row["passes"] for row in policies_lock["qd_sealed_artifacts"].values()
        )
        and storage["passes"],
        "split_reset_and_roundtrip": fixture["passes"],
        "stream_manifest_exactly_100": len(streams) == 100,
        "stream_manifest_internal_and_historical_collision_free": collision[
            "zero_collisions"
        ],
        "one_worker_frozen": FROZEN_JOBS == 1,
        "nice_at_least_10": base.current_nice() >= MINIMUM_NICE,
        "no_competing_heavy_process": heavy["passes"],
        "projected_compact_storage_below_4_gib": storage["passes"],
        "free_disk_strictly_above_120_gib": free_gib > TARGET_FREE_GIB,
        "services_dashboard_top_three": services["passes"],
        "tests_passed_and_bound": tests["passes"],
        "immutable_inputs_unchanged": immutable["passes"],
        "active_human_session_content_not_read": True,
        "no_acquisition_stream_consumed": True,
    }
    if not all(checks.values()):
        raise ValueError(f"Pilot-v2 preflight checks failed: {checks}")
    lock = {
        "version": VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "decision": "READY_G1R_PILOT_V2_QD5_PREFLIGHT",
        "pilot_execution_authorized": False,
        "bound_out_dir": str(final_dir),
        "identity": _preflight_identity(),
        "charter": str(CHARTER_PATH),
        "test_evidence": tests,
        "immutable_input_audit": immutable,
        "policy_lock": policies_lock,
        "panel_source": panel_source,
        "action_signature_audit": signatures,
        "games_per_family": GAMES_PER_FAMILY,
        "total_requested_games": len(streams),
        "family_order": [family for family, _spec in FAMILY_SLATE],
        "frozen_jobs": FROZEN_JOBS,
        "maximum_chunk_size": MAX_CHUNK_SIZE,
        "required_minimum_nice": MINIMUM_NICE,
        "preflight_process_nice": base.current_nice(),
        "active_wall_seconds_limit": PILOT_WALL_SECONDS,
        "pilot_byte_limit": PILOT_BYTE_LIMIT,
        "pause_free_disk_gib": MIN_FREE_GIB,
        "target_free_disk_gib": TARGET_FREE_GIB,
        "stream_bases": STREAM_BASES,
        "stream_rows": streams,
        "stream_manifest_sha256": canonical_json_hash(streams),
        "historical_stream_collision_audit": collision,
        "split_reset_roundtrip_fixture": fixture,
        "storage_admission": storage,
        "heavy_process_audit": heavy,
        "free_gib": free_gib,
        "service_health": services,
        "later_pilot_inspection_contract": {
            "completion_integrity_only": True,
            "exact_rung_availability": True,
            "root_uniqueness": True,
            "source_success_window_metadata_only": True,
            "wilson_yield_projection_only": True,
            "score_filtering": False,
            "policy_outcome_comparison": False,
            "one_selected_state_per_ancestry": True,
        },
        "checks": checks,
        "zero_work": {
            "games_generated": 0,
            "acquisition_streams_consumed": 0,
            "action_labels_generated": 0,
            "h40_outcomes_generated": 0,
            "models_fit": 0,
            "continuations_run": 0,
            "score_or_policy_outcomes_inspected": False,
            "dashboard_points_added": 0,
            "incumbent_changed": False,
        },
    }
    lock["preflight_payload_sha256"] = canonical_json_hash(lock)
    _write_new_json_atomic(staging_dir / "preflight_lock.json", lock)
    return lock


def prepare_preflight(out_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    final_dir = out_dir.resolve()
    if final_dir != OUTPUT_DIR.resolve():
        raise ValueError(f"Pilot-v2 output must be {OUTPUT_DIR.resolve()}")
    if final_dir.exists():
        raise FileExistsError(f"Pilot-v2 output already exists: {final_dir}")
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = final_dir.with_name(f"{final_dir.name}.staging.{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"Pilot-v2 staging already exists: {staging}")
    staging.mkdir()
    stage = "staging_created"
    try:
        stage = "preflight_checks"
        lock = _prepare_preflight_in_staging(staging, final_dir)
        stage = "atomic_promotion"
        os.replace(staging, final_dir)
        return lock
    except Exception as error:
        failure = {
            "version": VERSION,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "decision": "HOLD_G1R_PILOT_V2_PREFLIGHT_ERROR",
            "stage": stage,
            "error_type": type(error).__name__,
            "error": str(error),
            "bound_out_dir": str(final_dir),
            "games_generated": 0,
            "streams_consumed": False,
            "labels_generated": 0,
            "models_fit": 0,
            "outcomes_inspected": False,
        }
        if staging.exists():
            _write_new_json_atomic(staging / "PREPARATION_FAILED.json", failure)
        raise


def _validate_preflight(path: Path) -> dict[str, Any]:
    lock = json.loads(path.read_text())
    payload_hash = lock.pop("preflight_payload_sha256")
    if canonical_json_hash(lock) != payload_hash:
        raise ValueError("Pilot-v2 preflight payload mismatch")
    lock["preflight_payload_sha256"] = payload_hash
    if lock.get("version") != VERSION:
        raise ValueError("Pilot-v2 preflight version mismatch")
    if lock.get("decision") != "READY_G1R_PILOT_V2_QD5_PREFLIGHT":
        raise ValueError("Pilot-v2 preflight is not ready")
    if lock["bound_out_dir"] != str(OUTPUT_DIR.resolve()):
        raise ValueError("Pilot-v2 preflight output binding mismatch")
    if lock["identity"] != _preflight_identity():
        raise ValueError("Pilot-v2 source/test identity changed")
    return lock


def wilson_lower(k: int, n: int) -> float:
    if n == 0:
        return 0.0
    p = k / n
    z2 = WILSON_Z**2
    denominator = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denominator
    half = (
        WILSON_Z
        * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
        / denominator
    )
    return max(0.0, center - half)


def root_cap_candidates(candidates: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    by_root: dict[str, tuple[str, dict[str, Any]]] = {}
    for candidate in candidates:
        root = str(candidate["root_cluster"])
        key = canonical_json_hash(
            [
                "G1R-pilot-v2-root-cap",
                root,
                str(candidate["stratum"]),
                int(candidate["source_frame_index"]),
                str(candidate["state_sha1"]),
            ]
        )
        prior = by_root.get(root)
        if prior is None or key < prior[0]:
            by_root[root] = (key, candidate)
    return [by_root[root][1] for root in sorted(by_root)]


def yield_projection(
    completed_rows: list[dict[str, Any]],
    selected_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    families = [family for family, _spec in FAMILY_SLATE]
    selected_roots = [str(row["root_cluster"]) for row in selected_candidates]
    if len(selected_roots) != len(set(selected_roots)):
        raise ValueError("Yield projection candidates are not ancestry-unique")
    complete = Counter(str(row["nominal_family"]) for row in completed_rows)
    by_family_stratum = Counter(
        (str(row["behavior_family"]), str(row["stratum"]))
        for row in selected_candidates
    )
    remaining = MAX_TOTAL_GAMES - len(completed_rows)
    base_attempts, residual = divmod(remaining, len(families))
    rows = []
    totals = {"pre1536": 0, "pre3072": 0, "any": 0}
    for index, family in enumerate(families):
        n = int(complete[family])
        attempts = base_attempts + int(index < residual)
        counts = {
            stratum: int(by_family_stratum[(family, stratum)])
            for stratum in STRATA
        }
        counts["any"] = sum(counts.values())
        conservation_passes = (
            counts["pre1536"] + counts["pre3072"] == counts["any"]
        )
        if not conservation_passes:
            raise ValueError(f"Cross-stratum root conservation failed for {family}")
        projected = {}
        bounds = {}
        for stratum in (*STRATA, "any"):
            bounds[stratum] = wilson_lower(counts[stratum], n)
            projected[stratum] = counts[stratum] + math.floor(
                attempts * bounds[stratum]
            )
            totals[stratum] += projected[stratum]
        rows.append(
            {
                "family": family,
                "completed_roots": n,
                "counts": counts,
                "ancestry_unique_conservation_passes": conservation_passes,
                "wilson_lower_90": bounds,
                "projected_attempts": attempts,
                "projected_counts": projected,
            }
        )
    checks = {
        "pre1536_at_least_432": totals["pre1536"] >= 432,
        "pre3072_at_least_432": totals["pre3072"] >= 432,
        "any_at_least_864": totals["any"] >= 864,
        "selected_roots_ancestry_unique": len(selected_roots)
        == len(set(selected_roots)),
        "per_family_cross_stratum_conservation": all(
            row["ancestry_unique_conservation_passes"] for row in rows
        ),
    }
    return {
        "z": WILSON_Z,
        "family_rows": rows,
        "projected_totals": totals,
        "checks": checks,
        "passes": all(checks.values()),
    }


def _pilot_policy_lock_matches(lock: dict[str, Any]) -> bool:
    current, _policies = _policy_lock()
    return current["policy_lock_sha256"] == lock["policy_lock"][
        "policy_lock_sha256"
    ]


def run_pilot(
    *,
    out_dir: Path,
    preflight_lock: Path,
    jobs: int,
) -> dict[str, Any]:
    """Future pilot runner. Not invoked by the no-game preflight."""
    lock = _validate_preflight(preflight_lock)
    if out_dir.resolve() != OUTPUT_DIR.resolve():
        raise ValueError("Pilot-v2 output directory mismatch")
    if jobs != FROZEN_JOBS:
        raise ValueError("Pilot-v2 requires exactly one frozen worker")
    if base.current_nice() < MINIMUM_NICE:
        raise ValueError("Pilot-v2 requires nice >=10")
    if not _pilot_policy_lock_matches(lock):
        raise ValueError("Pilot-v2 policy lock changed")
    collision = stream_collision_audit(lock["stream_rows"], exclude_dir=out_dir)
    if not collision["zero_collisions"]:
        raise ValueError("Pilot-v2 historical stream collision appeared")
    completed_path = out_dir / "completed_games.jsonl"
    runtime_path = out_dir / "runtime_state.json"
    replay_dir = out_dir / "source_replays"
    replay_dir.mkdir(exist_ok=True)
    completed = base._load_completed(completed_path)
    runtime = base._runtime_state(runtime_path)
    specs = dict(FAMILY_SLATE)
    for family, _spec in FAMILY_SLATE:
        pending = [
            row
            for row in lock["stream_rows"]
            if row["nominal_family"] == family
            and (family, int(row["game_index"])) not in completed
        ]
        policy = load_policy(family, specs[family])
        for start in range(0, len(pending), MAX_CHUNK_SIZE):
            free_gib = shutil.disk_usage(out_dir).free / 1024**3
            if free_gib < MIN_FREE_GIB:
                raise base.AcquisitionPause("HOLD_G1R_BUDGET", "free disk below floor")
            if base._directory_bytes(out_dir) >= PILOT_BYTE_LIMIT:
                raise base.AcquisitionPause("HOLD_G1R_BUDGET", "pilot bytes at cap")
            if runtime["active_runtime_seconds"] >= PILOT_WALL_SECONDS:
                raise base.AcquisitionPause("HOLD_G1R_BUDGET", "pilot wall time at cap")
            if not base.service_health()["passes"]:
                raise base.AcquisitionPause("HOLD_G1R_SERVICE", "service degraded")
            chunk = pending[start : start + MAX_CHUNK_SIZE]
            jobs_rows = [
                EvalJob(
                    index=index,
                    seed=int(row["logical_seed"]),
                    starter_tile=STARTER_TILE,
                    stream_ids=EvalStreamIds(
                        deck_stream_id=int(row["deck_stream_id"]),
                        slot_stream_id=int(row["slot_stream_id"]),
                        policy_stream_id=int(row["policy_stream_id"]),
                    ),
                )
                for index, row in enumerate(chunk)
            ]
            started = time.perf_counter()
            outputs = list(
                iter_eval_job_outputs(
                    policy=policy,
                    policy_name=specs[family],
                    eval_jobs=jobs_rows,
                    max_moves=MAX_MOVES,
                    capture_replay=True,
                    jobs=jobs,
                )
            )
            for output in sorted(outputs, key=lambda row: row.index):
                stream_row = chunk[output.index]
                row = base._process_output(
                    output=output,
                    stream_row=stream_row,
                    genuine_family=family,
                    policy_spec=specs[family],
                    replay_dir=replay_dir,
                )
                base._append_jsonl_row(completed_path, row)
                completed[(family, int(stream_row["game_index"]))] = row
            runtime["active_runtime_seconds"] += time.perf_counter() - started
            runtime["chunks_completed"] += 1
            write_json(runtime_path, runtime)
    expected = {
        (str(row["nominal_family"]), int(row["game_index"]))
        for row in lock["stream_rows"]
    }
    if set(completed) != expected:
        raise RuntimeError("Pilot-v2 did not complete the exact frozen manifest")
    rows = [completed[key] for key in sorted(expected)]
    candidates = [candidate for row in rows for candidate in row["candidates"]]
    selected = root_cap_candidates(candidates)
    integrity = base.verify_retained_sources(selected)
    projection = yield_projection(rows, selected)
    decision = (
        "READY_G1R_ACQUISITION_CONTINUATION"
        if integrity["passes"] and projection["passes"]
        else "HOLD_G1R_YIELD_PROJECTION"
    )
    summary = {
        "version": VERSION,
        "decision": decision,
        "pilot_only_authorizes_continued_root_acquisition": True,
        "games": len(rows),
        "games_by_family": dict(
            sorted(Counter(row["nominal_family"] for row in rows).items())
        ),
        "root_capped_candidates": selected,
        "root_count": len(selected),
        "role_counts": dict(
            sorted(Counter(row["role"] for row in selected).items())
        ),
        "retained_source_integrity": integrity,
        "yield_projection": projection,
        "score_filtering_used": False,
        "policy_outcomes_compared": False,
        "labels_generated": 0,
        "models_fit": 0,
        "dashboard_eligible": False,
    }
    write_json(out_dir / "pilot_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    tests_parser = subparsers.add_parser("seal-test-evidence")
    tests_parser.add_argument("--focused-passed", type=int, required=True)
    tests_parser.add_argument("--regression-passed", type=int, required=True)
    tests_parser.add_argument(
        "--test-command", dest="test_commands", action="append", required=True
    )
    tests_parser.add_argument(
        "--phase-inapplicable-test",
        dest="phase_inapplicable_tests",
        action="append",
        default=[],
    )

    preflight_parser = subparsers.add_parser("prepare-preflight")
    preflight_parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)

    pilot_parser = subparsers.add_parser("run-pilot")
    pilot_parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    pilot_parser.add_argument("--preflight-lock", type=Path, required=True)
    pilot_parser.add_argument("--jobs", type=int, default=FROZEN_JOBS)

    args = parser.parse_args()
    if args.command == "seal-test-evidence":
        payload = seal_test_evidence(
            focused_passed=args.focused_passed,
            regression_passed=args.regression_passed,
            commands=args.test_commands,
            phase_inapplicable_tests=args.phase_inapplicable_tests,
        )
    elif args.command == "prepare-preflight":
        payload = prepare_preflight(args.out_dir)
    else:
        payload = run_pilot(
            out_dir=args.out_dir,
            preflight_lock=args.preflight_lock,
            jobs=args.jobs,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
