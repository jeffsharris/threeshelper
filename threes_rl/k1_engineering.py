"""Freeze and run the one-shot K1 compiled exact-kernel engineering gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from threes_rl import c2_cost_admission as c2
from threes_rl import g1r_acquire as history
from threes_rl.c1_search_optimization import clone_batched
from threes_rl.eval import (
    EvalJob,
    EvalStreamIds,
    iter_eval_job_outputs,
    make_policy,
    max_tile_excluding_initial_starter,
)
from threes_rl.expectimax import NtupleExpectimaxPolicy
from threes_rl.g1r_qd_admission_v2 import _heavy_process_audit
from threes_rl.k1_compiled_kernel import (
    COMPILE_FLAGS,
    COMPILER,
    COMPILER_SHA256,
    SDK_ROOT,
    NativeKernel,
    build_native_kernel,
    clone_k1,
    sha256_path,
)
from threes_rl.r2a_adaptive_expectimax import (
    EMPTY_TRIGGER,
    MARGIN_TRIGGER,
    choose_action,
    milestone_for_built_max,
    normalized_margin,
)
from threes_rl.record_replay import state_payload
from threes_rl.replay_provenance import (
    ORIGIN_FRESH,
    direct_root_fields,
    initial_reset_diagnostics,
)
from threes_rl.sim import DIRECTION_NAMES, ThreesSim, score_board, simulate_base_move
from threes_rl.train_td import state_from_replay_payload


VERSION = "k1_compiled_kernel_engineering_v1"
OUTPUT_DIR = Path("threes_rl/runs/forensics/k1_compiled_kernel_v1")
CHARTER_PATH = Path("threes_rl/K1_COMPILED_EXACT_KERNEL_CHARTER.md")
AMENDMENT_PATHS = (
    Path("threes_rl/K1_COMPILED_EXACT_KERNEL_CHARTER_AMENDMENT_A1.md"),
    Path("threes_rl/K1_COMPILED_EXACT_KERNEL_CHARTER_AMENDMENT_A2.md"),
    Path("threes_rl/K1_COMPILED_EXACT_KERNEL_CHARTER_AMENDMENT_A3.md"),
    Path("threes_rl/K1_COMPILED_EXACT_KERNEL_CHARTER_AMENDMENT_A4.md"),
    Path("threes_rl/K1_COMPILED_EXACT_KERNEL_CHARTER_AMENDMENT_A5.md"),
)
DESIGN_PREFLIGHT_PATHS = (
    Path("threes_rl/runs/forensics/K1_TOOLCHAIN_DESIGN_PREFLIGHT.json"),
    Path("threes_rl/runs/forensics/K1_TOOLCHAIN_DESIGN_PREFLIGHT_A1.json"),
    Path("threes_rl/runs/forensics/K1_TOOLCHAIN_DESIGN_PREFLIGHT_A2.json"),
    Path("threes_rl/runs/forensics/K1_TOOLCHAIN_DESIGN_PREFLIGHT_A3.json"),
    Path("threes_rl/runs/forensics/K1_TOOLCHAIN_DESIGN_PREFLIGHT_A4.json"),
    Path("threes_rl/runs/forensics/K1_TOOLCHAIN_DESIGN_PREFLIGHT_A5.json"),
)
NATIVE_SOURCE_PATH = Path("threes_rl/k1_exact_kernel.c")
WRAPPER_PATH = Path("threes_rl/k1_compiled_kernel.py")
RUNNER_PATH = Path("threes_rl/k1_engineering.py")
TEST_PATH = Path("tests/test_rl_k1_compiled_kernel.py")
TEST_EVIDENCE_PATH = Path(
    "threes_rl/runs/forensics/k1_compiled_kernel_test_evidence_a5.json"
)
INCUMBENT_PATH = Path("threes_rl/current_incumbent_policy.txt")

FAMILY_SLATE = (
    ("k1_corner2", "corner2"),
    (
        "k1_parent_mc1000",
        "ntuple_expectimax2:"
        "threes_rl/runs/td_default_corner2_mc_1000_init3000_a0005_"
        "20260706/latest",
    ),
    (
        "k1_replaycal",
        "ntuple_expectimax2:"
        "threes_rl/runs/replay_cal_phase4_late_midlate_top13_e3_a001_tc_"
        "20260706/latest",
    ),
)
C2_FAMILY_NAMES = {
    "k1_corner2": "c2_corner2",
    "k1_parent_mc1000": "c2_parent_mc1000",
    "k1_replaycal": "c2_replaycal",
}
STREAM_BASES = {
    "logical_seed": 73_000_000_000,
    "deck_stream_id": 74_000_000_000,
    "slot_stream_id": 75_000_000_000,
    "policy_stream_id": 76_000_000_000,
}
GAMES_PER_FAMILY = 36
TOTAL_GAMES = GAMES_PER_FAMILY * len(FAMILY_SLATE)
MAX_MOVES = 5000
STARTER_TILE = 1536
MAX_CHUNK_SIZE = 6
FROZEN_JOBS = 1
MINIMUM_NICE = 10
ACTIVE_WALL_SECONDS = 12 * 3600
BYTE_LIMIT = 4 * 1024**3
MIN_FREE_GIB = 100.0
TARGET_FREE_GIB = 120.0
STATES_PER_ROOT = 4
ROOTS_PER_FAMILY = {
    "fresh_equivalence": 4,
    "engineering_validation": 4,
    "untouched_runtime_gate": 4,
}
TIMED_REPEATS = 5
LEAF_TOLERANCE = 1e-9
VALUE_TOLERANCE = 1e-8
EXPECTED_TOP_THREE = (263670, 261369, 258561)
COLLISION_SCAN_ROOT = Path("threes_rl/runs")
INTERNAL_COLLISION_ALLOWLIST = {
    Path("threes_rl/runs/forensics/K1_TOOLCHAIN_DESIGN_PREFLIGHT_A4.json"):
        "6e806ed4d9a7c62b9d674228af5d113216fd8dd6f77d3caddbdfb143866e4799",
    Path(
        "threes_rl/runs/forensics/"
        "k1_compiled_kernel_v1.staging.11781"
    ):
        "56c7405477df5c093ff4831ab2dde6e38dad156dcb10d6c7d4d16302192f505e",
}

PRIOR_ROOT_SOURCE_PATHS = (
    *c2.PRIOR_ROOT_SOURCE_PATHS,
    Path(
        "threes_rl/runs/forensics/c2_cost_admission_v1/"
        "C2_CORPUS_MANIFEST.json"
    ),
)

IMMUTABLE_SOURCE_PATHS = (
    CHARTER_PATH,
    *AMENDMENT_PATHS,
    *DESIGN_PREFLIGHT_PATHS,
    NATIVE_SOURCE_PATH,
    WRAPPER_PATH,
    RUNNER_PATH,
    TEST_PATH,
    TEST_EVIDENCE_PATH,
    INCUMBENT_PATH,
    Path("threes_rl/c1_search_optimization.py"),
    Path("threes_rl/r2a_adaptive_expectimax.py"),
    Path("threes_rl/c2_cost_admission.py"),
    Path("threes_rl/eval.py"),
    Path("threes_rl/expectimax.py"),
    Path("threes_rl/ntuple.py"),
    Path("threes_rl/sim.py"),
    Path("threes_rl/replay_provenance.py"),
    Path("threes_rl/train_td.py"),
    Path("threes_rl/runs/forensics/c1_search/C1_RUNTIME_GATE.json"),
    Path(
        "threes_rl/runs/forensics/c2_cost_admission_v1/"
        "C2_TERMINAL_RESULT.json"
    ),
    Path(
        "threes_rl/runs/forensics/c2_cost_admission_v1/"
        "C2_CORPUS_MANIFEST.json"
    ),
)


class EngineeringFault(RuntimeError):
    pass


def canonical_json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def payload_with_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result.pop("canonical_payload_sha256", None)
    result["canonical_payload_sha256"] = canonical_json_hash(result)
    return result


def verify_payload_hash(payload: Mapping[str, Any]) -> bool:
    raw = dict(payload)
    expected = raw.pop("canonical_payload_sha256", None)
    return isinstance(expected, str) and canonical_json_hash(raw) == expected


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def directory_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(
        child.stat().st_size
        for child in path.rglob("*")
        if child.is_file()
    )


def free_gib(path: Path) -> float:
    target = path if path.exists() else path.parent
    return shutil.disk_usage(target).free / 1024**3


def file_manifest(path: Path) -> dict[str, Any]:
    if path.is_file():
        rows = [{
            "path": str(path),
            "relative_path": path.name,
            "byte_size": path.stat().st_size,
            "sha256": sha256_path(path),
        }]
    elif path.is_dir():
        rows = [
            {
                "path": str(child),
                "relative_path": str(child.relative_to(path)),
                "byte_size": child.stat().st_size,
                "sha256": sha256_path(child),
            }
            for child in sorted(path.rglob("*"))
            if child.is_file()
        ]
    else:
        raise FileNotFoundError(path)
    if not rows:
        raise ValueError(f"Empty K1 artifact manifest: {path}")
    return {
        "path": str(path),
        "files": rows,
        "file_count": len(rows),
        "total_bytes": sum(int(row["byte_size"]) for row in rows),
        "manifest_sha256": canonical_json_hash(rows),
    }


def incumbent_spec() -> str:
    return c2.incumbent_spec()


def stream_ids(family_index: int, game_index: int) -> dict[str, int]:
    offset = int(family_index) * 1_000_000 + int(game_index)
    return {key: int(base + offset) for key, base in STREAM_BASES.items()}


def requested_stream_manifest() -> list[dict[str, Any]]:
    return [
        {
            "family_index": family_index,
            "behavior_family": family,
            "policy_spec": spec,
            "game_index": game_index,
            **stream_ids(family_index, game_index),
        }
        for family_index, (family, spec) in enumerate(FAMILY_SLATE)
        for game_index in range(GAMES_PER_FAMILY)
    ]


def _recursive_root_values(value: Any, *, parent_key: str | None = None) -> set[str]:
    return c2._recursive_root_values(value, parent_key=parent_key)


def build_exclusion_manifest() -> dict[str, Any]:
    roots: set[str] = set()
    sources = []
    for path in PRIOR_ROOT_SOURCE_PATHS:
        if not path.is_file():
            raise FileNotFoundError(f"Missing K1 exclusion source: {path}")
        payload = json.loads(path.read_text())
        source_roots = _recursive_root_values(payload)
        roots.update(source_roots)
        sources.append({
            "path": str(path),
            "sha256": sha256_path(path),
            "byte_size": path.stat().st_size,
            "root_tokens": len(source_roots),
        })
    requested = {
        f"fresh:{row['logical_seed']}:{STARTER_TILE}"
        for row in requested_stream_manifest()
    }
    collisions = sorted(requested.intersection(roots))
    return payload_with_hash({
        "version": "k1_root_exclusion_v1",
        "sources": sources,
        "source_manifest_sha256": canonical_json_hash(sources),
        "root_tokens": sorted(roots),
        "root_token_count": len(roots),
        "requested_root_count": len(requested),
        "requested_root_collisions": collisions,
        "c2_untouched_explicitly_excluded": True,
        "passes": not collisions,
    })


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def internal_collision_allowlist_manifest() -> dict[str, Any]:
    forbidden_names = {
        "K1_EXECUTION_OPENED.json",
        "K1_TERMINAL_RESULT.json",
        "libk1_exact.dylib",
        "completed_games.jsonl",
        "K1_CORPUS_MANIFEST.json",
        "K1_FRESH_ENGINEERING_GATE.json",
    }
    rows = []
    for path, expected_manifest in INTERNAL_COLLISION_ALLOWLIST.items():
        current = file_manifest(path)
        files = [
            path if path.is_file() else path / str(row["relative_path"])
            for row in current["files"]
        ]
        zero_work_attested = False
        for source in files:
            if source.suffix != ".json":
                continue
            payload = json.loads(source.read_text())
            zero_work = payload.get("zero_work")
            if zero_work is True:
                zero_work_attested = True
            elif isinstance(zero_work, dict) and all(
                int(value) == 0 for value in zero_work.values()
            ):
                zero_work_attested = True
        checks = {
            "manifest_exact":
                current["manifest_sha256"] == expected_manifest,
            "zero_work_attested": zero_work_attested,
            "forbidden_work_absent":
                not forbidden_names.intersection(source.name for source in files),
        }
        if not all(checks.values()):
            raise EngineeringFault(
                f"K1 internal collision namespace changed: {path}: {checks}"
            )
        rows.append({
            "path": str(path),
            "resolved_path": str(path.resolve()),
            "manifest_sha256": current["manifest_sha256"],
            "file_count": current["file_count"],
            "total_bytes": current["total_bytes"],
            "checks": checks,
        })
    return payload_with_hash({
        "version": "k1_internal_collision_allowlist_v1",
        "rows": rows,
        "rows_sha256": canonical_json_hash(rows),
        "passes": True,
    })


def _scan_collision_sources(
    *,
    out_dir: Path,
) -> dict[str, Any]:
    internal = internal_collision_allowlist_manifest()
    internal_roots = [
        Path(str(row["resolved_path"])) for row in internal["rows"]
    ]
    prior: dict[str, set[int]] = defaultdict(set)
    immutable_sources = []
    live_sources = []
    for path in sorted(COLLISION_SCAN_ROOT.rglob("*")):
        if (
            not path.is_file()
            or path.suffix not in {".json", ".jsonl", ".csv"}
            or _is_within(path, out_dir)
            or any(_is_within(path, root) for root in internal_roots)
        ):
            continue
        values = history._scan_history_file(path)
        if not values:
            continue
        for key, items in values.items():
            prior[key].update(items)
        row = {
            "path": str(path),
            "sha256": sha256_path(path),
            "byte_size": path.stat().st_size,
            "counts": {
                key: len(items) for key, items in sorted(values.items())
            },
        }
        target = (
            live_sources
            if path in c2.LIVE_COLLISION_PATHS
            else immutable_sources
        )
        target.append(row)
    return {
        "prior": prior,
        "immutable_sources": immutable_sources,
        "live_sources": live_sources,
        "immutable_source_count": len(immutable_sources),
        "immutable_inventory_sha256": canonical_json_hash(immutable_sources),
        "live_source_count": len(live_sources),
        "live_paths": sorted(str(row["path"]) for row in live_sources),
        "internal_allowlist": internal,
    }


def build_stream_collision_manifest(
    rows: Sequence[Mapping[str, Any]],
    *,
    out_dir: Path,
) -> dict[str, Any]:
    scan = _scan_collision_sources(out_dir=out_dir)
    collisions: dict[str, list[int]] = {}
    for key in STREAM_BASES:
        prior_values = set(scan["prior"].get(key, set()))
        if key == "logical_seed":
            for alias in ("seed", "root_seed", "source_seed", "fresh_root_seed"):
                prior_values.update(scan["prior"].get(alias, set()))
        collisions[key] = sorted(
            {int(row[key]) for row in rows}.intersection(prior_values)
        )
    all_ids = [int(row[key]) for row in rows for key in STREAM_BASES]
    checks = {
        "exact_108_rows": len(rows) == TOTAL_GAMES,
        "exact_432_internal_ids": len(all_ids) == 4 * TOTAL_GAMES,
        "internal_ids_unique": len(all_ids) == len(set(all_ids)),
        "zero_historical_collisions": not any(collisions.values()),
    }
    return payload_with_hash({
        "version": "k1_stream_collision_v1",
        "requested_rows_sha256": canonical_json_hash(list(rows)),
        "stream_bases": STREAM_BASES,
        "collisions": collisions,
        "immutable_sources": scan["immutable_sources"],
        "immutable_source_count": scan["immutable_source_count"],
        "immutable_inventory_sha256": scan["immutable_inventory_sha256"],
        "live_sources": scan["live_sources"],
        "live_source_count": scan["live_source_count"],
        "live_paths": scan["live_paths"],
        "internal_allowlist": scan["internal_allowlist"],
        "checks": checks,
        "passes": all(checks.values()),
    })


def revalidate_stream_collision_manifest(
    manifest: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    out_dir: Path,
) -> dict[str, Any]:
    if not verify_payload_hash(manifest):
        raise ValueError("K1 collision manifest payload hash mismatch")
    current = build_stream_collision_manifest(rows, out_dir=out_dir)
    checks = {
        "requested_rows_exact":
            current["requested_rows_sha256"] == manifest["requested_rows_sha256"],
        "immutable_inventory_exact":
            current["immutable_inventory_sha256"]
            == manifest["immutable_inventory_sha256"]
            and current["immutable_sources"] == manifest["immutable_sources"],
        "live_paths_exact": current["live_paths"] == manifest["live_paths"],
        "internal_allowlist_exact":
            current["internal_allowlist"] == manifest["internal_allowlist"],
        "zero_collisions": current["passes"],
    }
    return {"checks": checks, "passes": all(checks.values()), "current": current}


def build_policy_lock() -> tuple[dict[str, Any], dict[str, Any]]:
    c2_lock, c2_loaded = c2.build_policy_lock()
    accepted = c2.accepted_family_audit()
    loaded = {
        family: c2_loaded[C2_FAMILY_NAMES[family]]
        for family, _spec in FAMILY_SLATE
    }
    rows = []
    for family, spec in FAMILY_SLATE:
        source_family = C2_FAMILY_NAMES[family]
        source_row = next(
            row for row in c2_lock["families"]
            if row["behavior_family"] == source_family
        )
        rows.append({
            "behavior_family": family,
            "policy_spec": spec,
            "policy_spec_sha256": hashlib.sha256(spec.encode()).hexdigest(),
            "accepted_signature_sha256":
                source_row["action_signature_sha256"],
            "checkpoint_manifests": source_row["checkpoint_manifests"],
        })
    payload = {
        "version": "k1_policy_lock_v1",
        "families": rows,
        "family_order": [family for family, _spec in FAMILY_SLATE],
        "c2_policy_lock_sha256": c2_lock["policy_lock_sha256"],
        "accepted_family_audit": accepted,
        "incumbent_spec": incumbent_spec(),
        "incumbent_spec_sha256": hashlib.sha256(
            incumbent_spec().encode()
        ).hexdigest(),
        "incumbent_file_sha256": sha256_path(INCUMBENT_PATH),
    }
    payload["policy_lock_sha256"] = canonical_json_hash(payload)
    return payload, loaded


def revalidate_policy_lock(expected: Mapping[str, Any]) -> dict[str, Any]:
    current, _loaded = build_policy_lock()
    return {
        "expected_sha256": expected["policy_lock_sha256"],
        "current_sha256": current["policy_lock_sha256"],
        "exact": current == expected,
    }


def source_lock() -> dict[str, Any]:
    rows = [
        {
            "path": str(path),
            "sha256": sha256_path(path),
            "byte_size": path.stat().st_size,
        }
        for path in IMMUTABLE_SOURCE_PATHS
    ]
    return payload_with_hash({
        "version": "k1_source_lock_v1",
        "sources": rows,
        "source_manifest_sha256": canonical_json_hash(rows),
        "c1_source_sha256": sha256_path(
            Path("threes_rl/c1_search_optimization.py")
        ),
        "c2_terminal_sha256": sha256_path(
            Path(
                "threes_rl/runs/forensics/c2_cost_admission_v1/"
                "C2_TERMINAL_RESULT.json"
            )
        ),
    })


def build_spec() -> dict[str, Any]:
    return payload_with_hash({
        "version": "k1_build_spec_v1",
        "compiler": str(COMPILER),
        "compiler_sha256": sha256_path(COMPILER),
        "sdk_root": str(SDK_ROOT),
        "sdk_exists": SDK_ROOT.is_dir(),
        "flags": list(COMPILE_FLAGS),
        "native_source": str(NATIVE_SOURCE_PATH),
        "native_source_sha256": sha256_path(NATIVE_SOURCE_PATH),
        "wrapper": str(WRAPPER_PATH),
        "wrapper_sha256": sha256_path(WRAPPER_PATH),
        "exports": [
            "k1_eval_composite",
            "k1_base_move",
            "k1_score_board",
            "k1_post_spawn_rows",
            "k1_kernel_abi_version",
        ],
        "chance_limit": 8,
        "value_node_budget": 2048,
        "approximation": False,
    })


def corpus_plan() -> dict[str, Any]:
    partitions = [
        {
            "partition": partition,
            "roots_per_family": count,
            "roots": count * len(FAMILY_SLATE),
            "states": count * len(FAMILY_SLATE) * STATES_PER_ROOT,
        }
        for partition, count in ROOTS_PER_FAMILY.items()
    ]
    return payload_with_hash({
        "version": "k1_corpus_plan_v1",
        "games_per_family": GAMES_PER_FAMILY,
        "total_games": TOTAL_GAMES,
        "required_roots_per_family": sum(ROOTS_PER_FAMILY.values()),
        "states_per_root": STATES_PER_ROOT,
        "partitions": partitions,
        "family_share_each_partition": 1.0 / len(FAMILY_SLATE),
        "partition_before_timing": True,
        "score_or_outcome_selection": False,
    })


def _load_test_evidence() -> dict[str, Any]:
    payload = json.loads(TEST_EVIDENCE_PATH.read_text())
    if not verify_payload_hash(payload):
        raise ValueError("K1 test evidence payload hash mismatch")
    checks = {
        "runner": payload.get("runner_sha256") == sha256_path(RUNNER_PATH),
        "native": payload.get("native_source_sha256")
        == sha256_path(NATIVE_SOURCE_PATH),
        "wrapper": payload.get("wrapper_sha256") == sha256_path(WRAPPER_PATH),
        "tests": payload.get("test_sha256") == sha256_path(TEST_PATH),
        "all_passed": payload.get("all_passed") is True,
    }
    if not all(checks.values()):
        raise ValueError(f"K1 test evidence mismatch: {checks}")
    return payload


def _operational_audit(path: Path) -> dict[str, Any]:
    disk = free_gib(path)
    heavy = _heavy_process_audit()
    services = history.service_health()
    checks = {
        "nice_at_least_10": history.current_nice() >= MINIMUM_NICE,
        "no_competing_heavy_process": bool(heavy["passes"]),
        "disk_above_100_gib": disk >= MIN_FREE_GIB,
        "disk_above_120_gib_target": disk >= TARGET_FREE_GIB,
        "services_dashboard_top_three": bool(services["passes"]),
        "dashboard_top_three_exact":
            tuple(services.get("dashboard_top_scores", ()))
            == EXPECTED_TOP_THREE,
    }
    return {
        "nice": history.current_nice(),
        "free_gib": disk,
        "heavy_process_audit": heavy,
        "service_health": services,
        "checks": checks,
        "passes": all(checks.values()),
    }


def _commands(out_dir: Path) -> dict[str, str]:
    lock = out_dir / "K1_PREFLIGHT_LOCK.json"
    prefix = (
        "zsh -ic 'no-secrets nice -n 10 env PYTHONPATH=. "
        ".venv/bin/python -m threes_rl.k1_engineering"
    )
    suffix = f" --out-dir {out_dir} --preflight-lock {lock} --jobs 1'"
    return {
        "open": f"{prefix} open{suffix}",
        "execute": f"{prefix} execute{suffix}",
    }


def run_preflight(*, out_dir: Path, jobs: int) -> dict[str, Any]:
    if out_dir.resolve() != OUTPUT_DIR.resolve():
        raise ValueError("K1 output directory does not match frozen namespace")
    if jobs != FROZEN_JOBS:
        raise ValueError("K1 jobs must equal one")
    if out_dir.exists():
        raise FileExistsError("K1 output already exists")
    staging = out_dir.with_name(f"{out_dir.name}.staging.{os.getpid()}")
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    try:
        rows = requested_stream_manifest()
        stream_payload = payload_with_hash({
            "version": "k1_stream_manifest_v1",
            "rows": rows,
            "rows_sha256": canonical_json_hash(rows),
        })
        collision = build_stream_collision_manifest(rows, out_dir=staging)
        exclusion = build_exclusion_manifest()
        policy, _loaded = build_policy_lock()
        source = source_lock()
        build = build_spec()
        plan = corpus_plan()
        tests = _load_test_evidence()
        operations = _operational_audit(staging)
        manifests = {
            "K1_STREAM_MANIFEST.json": stream_payload,
            "K1_COLLISION_MANIFEST.json": collision,
            "K1_EXCLUSION_MANIFEST.json": exclusion,
            "K1_POLICY_LOCK.json": payload_with_hash(policy),
            "K1_SOURCE_LOCK.json": source,
            "K1_BUILD_SPEC.json": build,
            "K1_CORPUS_PLAN.json": plan,
        }
        for name, payload in manifests.items():
            atomic_write_json(staging / name, payload)
        checks = {
            "tests_pass": bool(tests["all_passed"]),
            "source_lock": source["c1_source_sha256"]
            == "c12852cc7dcc8211d8ecc47ccf8c5598d6055a5f12a9bcec497dc47715e0e789",
            "c2_lock": source["c2_terminal_sha256"]
            == "ac1e3b490a6ab7d498cacfdd1157ce68020ebe8459e7b654ac487fa28eb3cb9f",
            "compiler": build["compiler_sha256"] == COMPILER_SHA256,
            "sdk": build["sdk_exists"],
            "policy": policy["accepted_family_audit"]["passes"],
            "collision": collision["passes"],
            "exclusion": exclusion["passes"],
            "operations": operations["passes"],
            "no_library": not (staging / "libk1_exact.dylib").exists(),
            "zero_work": True,
        }
        if not all(checks.values()):
            raise EngineeringFault(f"K1 preflight failed: {checks}")
        manifest_bindings = {
            name: {
                "file_sha256": sha256_path(staging / name),
                "payload_sha256": payload["canonical_payload_sha256"],
            }
            for name, payload in manifests.items()
        }
        lock = payload_with_hash({
            "version": "k1_execution_preflight_v1",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "decision": "READY_K1_ENGINEERING_EXECUTION",
            "bound_out_dir": str(out_dir.resolve()),
            "jobs": jobs,
            "commands": _commands(out_dir),
            "charter_sha256": sha256_path(CHARTER_PATH),
            "amendment_sha256": {
                path.name: sha256_path(path) for path in AMENDMENT_PATHS
            },
            "design_preflight_sha256": {
                path.name: sha256_path(path) for path in DESIGN_PREFLIGHT_PATHS
            },
            "runner_sha256": sha256_path(RUNNER_PATH),
            "native_source_sha256": sha256_path(NATIVE_SOURCE_PATH),
            "wrapper_sha256": sha256_path(WRAPPER_PATH),
            "test_sha256": sha256_path(TEST_PATH),
            "test_evidence_file_sha256": sha256_path(TEST_EVIDENCE_PATH),
            "test_evidence_payload_sha256":
                tests["canonical_payload_sha256"],
            "manifest_bindings": manifest_bindings,
            "stream_rows": TOTAL_GAMES,
            "operations": operations,
            "checks": checks,
            "zero_work": {
                "compiled_libraries": 0,
                "fresh_games": 0,
                "streams_consumed": 0,
                "timings": 0,
                "depth3_results": 0,
                "policy_outcomes": 0,
                "scores_inspected": 0,
            },
        })
        atomic_write_json(staging / "K1_PREFLIGHT_LOCK.json", lock)
        os.replace(staging, out_dir)
        return {
            "decision": lock["decision"],
            "lock_path": str(out_dir / "K1_PREFLIGHT_LOCK.json"),
            "lock_file_sha256": sha256_path(
                out_dir / "K1_PREFLIGHT_LOCK.json"
            ),
            "lock_payload_sha256": lock["canonical_payload_sha256"],
            "stream_rows": TOTAL_GAMES,
            "historical_source_count": collision["immutable_source_count"],
            "historical_source_sha256":
                collision["immutable_inventory_sha256"],
            "operations": operations,
            "zero_work": lock["zero_work"],
        }
    except Exception:
        failure = {
            "version": "k1_preflight_failure_v1",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "decision": "HOLD_K1_ENGINEERING_FAULT",
            "stage": "preflight",
            "error": traceback.format_exc(),
            "zero_work": True,
        }
        atomic_write_json(staging / "K1_PREFLIGHT_FAILURE.json", failure)
        raise


def _load_lock(out_dir: Path, lock_path: Path) -> dict[str, Any]:
    if out_dir.resolve() != OUTPUT_DIR.resolve():
        raise ValueError("K1 output mismatch")
    expected = out_dir / "K1_PREFLIGHT_LOCK.json"
    if lock_path.resolve() != expected.resolve():
        raise ValueError("K1 lock path mismatch")
    payload = json.loads(expected.read_text())
    if not verify_payload_hash(payload):
        raise ValueError("K1 preflight payload hash mismatch")
    if payload["bound_out_dir"] != str(out_dir.resolve()):
        raise ValueError("K1 preflight bound output mismatch")
    return payload


def _revalidate_files(out_dir: Path, lock: Mapping[str, Any]) -> None:
    direct = {
        CHARTER_PATH: lock["charter_sha256"],
        RUNNER_PATH: lock["runner_sha256"],
        NATIVE_SOURCE_PATH: lock["native_source_sha256"],
        WRAPPER_PATH: lock["wrapper_sha256"],
        TEST_PATH: lock["test_sha256"],
        TEST_EVIDENCE_PATH: lock["test_evidence_file_sha256"],
    }
    for path, expected in direct.items():
        if sha256_path(path) != expected:
            raise EngineeringFault(f"K1 immutable file changed: {path}")
    for path in AMENDMENT_PATHS:
        if sha256_path(path) != lock["amendment_sha256"][path.name]:
            raise EngineeringFault(f"K1 amendment changed: {path}")
    for path in DESIGN_PREFLIGHT_PATHS:
        if (
            sha256_path(path)
            != lock["design_preflight_sha256"][path.name]
        ):
            raise EngineeringFault(f"K1 design preflight changed: {path}")
    for name, binding in lock["manifest_bindings"].items():
        if sha256_path(out_dir / name) != binding["file_sha256"]:
            raise EngineeringFault(f"K1 manifest changed: {name}")


def _collision_revalidation(
    out_dir: Path,
    lock: Mapping[str, Any],
) -> dict[str, Any]:
    stream = json.loads((out_dir / "K1_STREAM_MANIFEST.json").read_text())
    collision = json.loads((out_dir / "K1_COLLISION_MANIFEST.json").read_text())
    return revalidate_stream_collision_manifest(
        collision,
        stream["rows"],
        out_dir=out_dir,
    )


def seal_execution_opened(
    *,
    out_dir: Path,
    preflight_lock: Path,
    jobs: int,
) -> dict[str, Any]:
    lock = _load_lock(out_dir, preflight_lock)
    if jobs != FROZEN_JOBS:
        raise ValueError("K1 jobs mismatch")
    marker_path = out_dir / "K1_EXECUTION_OPENED.json"
    terminal_path = out_dir / "K1_TERMINAL_RESULT.json"
    if marker_path.exists() or terminal_path.exists():
        raise FileExistsError("K1 execution already opened or terminal")
    forbidden = (
        out_dir / "libk1_exact.dylib",
        out_dir / "completed_games.jsonl",
        out_dir / "K1_CORPUS_MANIFEST.json",
        out_dir / "K1_FRESH_TIMINGS.json",
    )
    if any(path.exists() for path in forbidden):
        raise EngineeringFault("K1 work exists before marker")
    _revalidate_files(out_dir, lock)
    collision = _collision_revalidation(out_dir, lock)
    operations = _operational_audit(out_dir)
    if not collision["passes"] or not operations["passes"]:
        raise EngineeringFault("K1 open revalidation failed")
    marker = payload_with_hash({
        "version": "k1_execution_opened_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "decision": "K1_EXECUTION_OPENED",
        "execution_opened": True,
        "bound_out_dir": str(out_dir.resolve()),
        "preflight_lock_file_sha256": sha256_path(preflight_lock),
        "preflight_lock_payload_sha256": lock["canonical_payload_sha256"],
        "execute_command": lock["commands"]["execute"],
        "jobs": jobs,
        "operations": operations,
        "collision_revalidation": collision["checks"],
        "zero_work": {
            "compiled_libraries": 0,
            "fresh_games": 0,
            "streams_consumed": 0,
            "timings": 0,
            "depth3_results": 0,
            "policy_outcomes": 0,
            "scores_inspected": 0,
        },
    })
    atomic_write_json(marker_path, marker)
    return {
        "decision": marker["decision"],
        "marker_path": str(marker_path),
        "marker_file_sha256": sha256_path(marker_path),
        "marker_payload_sha256": marker["canonical_payload_sha256"],
        "execute_command": marker["execute_command"],
        "zero_work": marker["zero_work"],
    }


def _load_marker(out_dir: Path, lock: Mapping[str, Any]) -> dict[str, Any]:
    path = out_dir / "K1_EXECUTION_OPENED.json"
    if not path.is_file():
        raise EngineeringFault("K1 marker missing")
    marker = json.loads(path.read_text())
    checks = {
        "payload": verify_payload_hash(marker),
        "opened": marker.get("execution_opened") is True,
        "preflight_file": marker.get("preflight_lock_file_sha256")
        == sha256_path(out_dir / "K1_PREFLIGHT_LOCK.json"),
        "preflight_payload": marker.get("preflight_lock_payload_sha256")
        == lock["canonical_payload_sha256"],
        "command": marker.get("execute_command") == lock["commands"]["execute"],
        "jobs": int(marker.get("jobs", -1)) == FROZEN_JOBS,
    }
    if not all(checks.values()):
        raise EngineeringFault(f"K1 marker mismatch: {checks}")
    return marker


def _root_ancestry(logical_seed: int) -> str:
    return f"fresh:{int(logical_seed)}:{STARTER_TILE}"


def _state_sim(payload: Mapping[str, Any]) -> tuple[Any, ThreesSim]:
    state = state_from_replay_payload(dict(payload))
    sim = ThreesSim.from_stream_ids(
        deck_stream_id=1,
        slot_stream_id=2,
        starter_tile=STARTER_TILE,
    )
    if state_payload(state, sim) != payload:
        raise EngineeringFault("K1 state failed exact round trip")
    return state, sim


def _state_hash(payload: Mapping[str, Any]) -> str:
    return canonical_json_hash(payload)


def _frame_candidate(
    frame: Mapping[str, Any],
    *,
    root: str,
) -> dict[str, Any] | None:
    payload = frame.get("state")
    if not isinstance(payload, dict):
        return None
    state, sim = _state_sim(payload)
    if state.game_over:
        return None
    legal = sim.legal_actions(state)
    if not legal:
        return None
    built_max = max_tile_excluding_initial_starter(
        state.board,
        STARTER_TILE,
    )
    if milestone_for_built_max(built_max) is None:
        return None
    return {
        "root_ancestry": root,
        "frame_index": int(frame.get("index", -1)),
        "state": payload,
        "state_sha256": _state_hash(payload),
        "empty_count": int(np.count_nonzero(state.board == 0)),
        "built_max": int(built_max),
        "legal_actions": [DIRECTION_NAMES[action] for action in legal],
    }


def _incumbent_metadata(
    row: Mapping[str, Any],
    incumbent: NtupleExpectimaxPolicy,
) -> dict[str, Any]:
    state, sim = _state_sim(row["state"])
    values = incumbent.action_values(state, sim)
    if not values or not all(
        math.isfinite(float(value)) for _action, value in values
    ):
        raise EngineeringFault("K1 incumbent values are invalid")
    margin = normalized_margin(values)
    return {
        "incumbent_margin": float(margin),
        "trigger_reasons": {
            "low_empty": int(row["empty_count"]) <= EMPTY_TRIGGER,
            "low_margin": float(margin) <= MARGIN_TRIGGER,
        },
        "incumbent_legal_actions": [
            DIRECTION_NAMES[action] for action, _value in values
        ],
    }


def extract_selected_states(
    replay: Mapping[str, Any],
    *,
    family: str,
    stream_row: Mapping[str, Any],
    incumbent: NtupleExpectimaxPolicy,
) -> list[dict[str, Any]]:
    root = _root_ancestry(int(stream_row["logical_seed"]))
    candidates = [
        candidate
        for frame in replay.get("frames", [])
        if (candidate := _frame_candidate(frame, root=root)) is not None
    ]
    low_empty = [
        row for row in candidates
        if int(row["empty_count"]) <= EMPTY_TRIGGER
    ]
    high_empty = sorted(
        (
            row for row in candidates
            if int(row["empty_count"]) > EMPTY_TRIGGER
        ),
        key=lambda row: (
            hashlib.sha256(
                (
                    "K1-high-empty-v1|"
                    f"{root}|{row['frame_index']}|{row['state_sha256']}"
                ).encode()
            ).hexdigest(),
            int(row["frame_index"]),
        ),
    )[:8]
    eligible = [dict(row) for row in low_empty]
    for row in high_empty:
        metadata = _incumbent_metadata(row, incumbent)
        if metadata["trigger_reasons"]["low_margin"]:
            eligible.append({**row, **metadata})
    eligible.sort(key=lambda row: int(row["frame_index"]))
    if len(eligible) < STATES_PER_ROOT:
        return []
    buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(eligible):
        bucket = min(3, STATES_PER_ROOT * index // len(eligible))
        buckets[bucket].append(row)
    if set(buckets) != set(range(STATES_PER_ROOT)):
        raise EngineeringFault("K1 temporal buckets are incomplete")
    selected = []
    for bucket in range(STATES_PER_ROOT):
        row = min(
            buckets[bucket],
            key=lambda item: (
                hashlib.sha256(
                    (
                        "K1-state-v1|"
                        f"{root}|{item['frame_index']}|{item['state_sha256']}"
                    ).encode()
                ).hexdigest(),
                int(item["frame_index"]),
            ),
        )
        metadata = (
            {
                "incumbent_margin": row["incumbent_margin"],
                "trigger_reasons": row["trigger_reasons"],
                "incumbent_legal_actions": row["incumbent_legal_actions"],
            }
            if "incumbent_margin" in row
            else _incumbent_metadata(row, incumbent)
        )
        if not any(metadata["trigger_reasons"].values()):
            raise EngineeringFault("K1 selected non-trigger state")
        selected.append({
            **row,
            **metadata,
            "behavior_family": family,
            "family_index": int(stream_row["family_index"]),
            "game_index": int(stream_row["game_index"]),
            "logical_seed": int(stream_row["logical_seed"]),
            "deck_stream_id": int(stream_row["deck_stream_id"]),
            "slot_stream_id": int(stream_row["slot_stream_id"]),
            "policy_stream_id": int(stream_row["policy_stream_id"]),
            "selection_bucket": bucket,
        })
    return selected


def _load_completed(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    rows = {}
    if not path.is_file():
        return rows
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            key = (str(row["behavior_family"]), int(row["game_index"]))
            if key in rows:
                raise EngineeringFault(f"Duplicate K1 completion: {key}")
            rows[key] = row
    return rows


def _runtime_state(path: Path) -> dict[str, Any]:
    if path.is_file():
        return json.loads(path.read_text())
    return {
        "active_runtime_seconds": 0.0,
        "chunks_completed": 0,
        "phase": "build",
    }


def _execution_guard(out_dir: Path, runtime: Mapping[str, Any]) -> dict[str, Any]:
    disk = free_gib(out_dir)
    used = directory_bytes(out_dir)
    services = history.service_health()
    heavy = _heavy_process_audit()
    checks = {
        "nice": history.current_nice() >= MINIMUM_NICE,
        "disk": disk >= MIN_FREE_GIB,
        "output": used < BYTE_LIMIT,
        "runtime": float(runtime.get("active_runtime_seconds", 0.0))
        < ACTIVE_WALL_SECONDS,
        "services": services["passes"],
        "contention": heavy["passes"],
    }
    if not all(checks.values()):
        raise EngineeringFault(f"K1 execution guard failed: {checks}")
    return {
        "checks": checks,
        "free_gib": disk,
        "output_bytes": used,
        "services": services,
        "heavy": heavy,
        "passes": True,
    }


def _acquisition_chunks(rows: Sequence[Mapping[str, Any]]) -> list[list[dict]]:
    by_family = {
        family: sorted(
            (
                dict(row) for row in rows
                if row["behavior_family"] == family
            ),
            key=lambda row: int(row["game_index"]),
        )
        for family, _spec in FAMILY_SLATE
    }
    chunks = []
    for start in range(0, GAMES_PER_FAMILY, MAX_CHUNK_SIZE):
        for family, _spec in FAMILY_SLATE:
            chunks.append(by_family[family][start:start + MAX_CHUNK_SIZE])
    return chunks


def _process_game(
    *,
    output: Any,
    stream_row: Mapping[str, Any],
    family: str,
    spec: str,
    incumbent: NtupleExpectimaxPolicy,
    retained_count: int,
    replay_dir: Path,
    state_dir: Path,
) -> dict[str, Any]:
    replay = output.replay
    if replay is None:
        raise EngineeringFault("K1 replay missing")
    seed = int(stream_row["logical_seed"])
    replay.update(direct_root_fields(
        origin=ORIGIN_FRESH,
        seed=seed,
        policy=family,
        first_score=int(replay["frames"][0]["state"]["score"]),
    ))
    replay["behavior_family"] = family
    replay["acquisition_policy_spec"] = spec
    replay["dashboard_eligible"] = False
    reset = initial_reset_diagnostics(replay)
    if not reset["is_reset_start"]:
        raise EngineeringFault(f"K1 reset invariant failed: {reset}")
    complete = bool(replay.get("game_over", False))
    selected = []
    if complete and retained_count < sum(ROOTS_PER_FAMILY.values()):
        selected = extract_selected_states(
            replay,
            family=family,
            stream_row=stream_row,
            incumbent=incumbent,
        )
    replay_path = replay_dir / (
        f"{family}_game_{int(stream_row['game_index']):05d}_seed_{seed}.json"
    )
    state_path = state_dir / (
        f"{family}_game_{int(stream_row['game_index']):05d}_seed_{seed}.json"
    )
    if selected:
        atomic_write_json(replay_path, replay)
        replay_sha = sha256_path(replay_path)
        for row in selected:
            row["source_replay"] = str(replay_path)
            row["source_replay_sha256"] = replay_sha
        state_record = payload_with_hash({
            "version": "k1_selected_root_states_v1",
            "root_ancestry": _root_ancestry(seed),
            "behavior_family": family,
            "game_index": int(stream_row["game_index"]),
            "source_replay": str(replay_path),
            "source_replay_sha256": replay_sha,
            "states": selected,
        })
        atomic_write_json(state_path, state_record)
    return {
        "behavior_family": family,
        "family_index": int(stream_row["family_index"]),
        "game_index": int(stream_row["game_index"]),
        "policy_spec_sha256": hashlib.sha256(spec.encode()).hexdigest(),
        "logical_seed": seed,
        "deck_stream_id": int(stream_row["deck_stream_id"]),
        "slot_stream_id": int(stream_row["slot_stream_id"]),
        "policy_stream_id": int(stream_row["policy_stream_id"]),
        "root_ancestry": _root_ancestry(seed),
        "complete": complete,
        "move_count": int(output.result.moves),
        "qualifying_root": bool(selected),
        "selected_state_count": len(selected),
        "source_replay": str(replay_path) if selected else None,
        "source_replay_sha256":
            sha256_path(replay_path) if selected else None,
        "selected_states": str(state_path) if selected else None,
        "selected_states_sha256":
            sha256_path(state_path) if selected else None,
        "score_inspected": False,
        "dashboard_eligible": False,
    }


def run_acquisition(
    *,
    out_dir: Path,
    lock: Mapping[str, Any],
    jobs: int,
) -> dict[str, Any]:
    stream = json.loads((out_dir / "K1_STREAM_MANIFEST.json").read_text())
    rows = [dict(row) for row in stream["rows"]]
    collision = _collision_revalidation(out_dir, lock)
    if not collision["passes"]:
        raise EngineeringFault("K1 stream collision appeared")
    completed_path = out_dir / "completed_games.jsonl"
    runtime_path = out_dir / "runtime_state.json"
    replay_dir = out_dir / "source_replays"
    state_dir = out_dir / "selected_states"
    replay_dir.mkdir(exist_ok=True)
    state_dir.mkdir(exist_ok=True)
    completed = _load_completed(completed_path)
    runtime = _runtime_state(runtime_path)
    expected_policy = json.loads((out_dir / "K1_POLICY_LOCK.json").read_text())
    expected_raw = dict(expected_policy)
    expected_raw.pop("canonical_payload_sha256", None)
    if not revalidate_policy_lock(expected_raw)["exact"]:
        raise EngineeringFault("K1 policy lock changed")
    incumbent = make_policy(incumbent_spec())
    if not isinstance(incumbent, NtupleExpectimaxPolicy):
        raise EngineeringFault("K1 incumbent type mismatch")
    policies = {family: make_policy(spec) for family, spec in FAMILY_SLATE}
    specs = dict(FAMILY_SLATE)
    for chunk in _acquisition_chunks(rows):
        pending = [
            row for row in chunk
            if (str(row["behavior_family"]), int(row["game_index"]))
            not in completed
        ]
        if not pending:
            continue
        _execution_guard(out_dir, runtime)
        family = str(pending[0]["behavior_family"])
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
            for index, row in enumerate(pending)
        ]
        started = time.perf_counter()
        outputs = list(iter_eval_job_outputs(
            policy=policies[family],
            policy_name=specs[family],
            eval_jobs=jobs_rows,
            max_moves=MAX_MOVES,
            capture_replay=True,
            jobs=jobs,
        ))
        for output in sorted(outputs, key=lambda value: value.index):
            row = pending[int(output.index)]
            retained = sum(
                bool(item["qualifying_root"])
                for (row_family, _index), item in completed.items()
                if row_family == family
            )
            completion = _process_game(
                output=output,
                stream_row=row,
                family=family,
                spec=specs[family],
                incumbent=incumbent,
                retained_count=retained,
                replay_dir=replay_dir,
                state_dir=state_dir,
            )
            append_jsonl(completed_path, completion)
            completed[(family, int(row["game_index"]))] = completion
        runtime["active_runtime_seconds"] = (
            float(runtime["active_runtime_seconds"])
            + time.perf_counter() - started
        )
        runtime["chunks_completed"] = int(runtime["chunks_completed"]) + 1
        runtime["phase"] = "acquisition"
        atomic_write_json(runtime_path, runtime)
    expected = {
        (str(row["behavior_family"]), int(row["game_index"]))
        for row in rows
    }
    if set(completed) != expected:
        raise EngineeringFault("K1 acquisition incomplete")
    ordered = [completed[key] for key in sorted(expected)]
    return {
        "rows": ordered,
        "completed": completed,
        "runtime": runtime,
        "all_complete": all(bool(row["complete"]) for row in ordered),
        "qualifying_by_family": {
            family: sum(
                bool(row["qualifying_root"])
                for row in ordered
                if row["behavior_family"] == family
            )
            for family, _spec in FAMILY_SLATE
        },
    }


def _partition_for_index(index: int) -> str:
    cursor = 0
    for partition, count in ROOTS_PER_FAMILY.items():
        if cursor <= index < cursor + count:
            return partition
        cursor += count
    raise ValueError(f"K1 qualifying index out of range: {index}")


def build_corpus(
    *,
    out_dir: Path,
    acquisition: Mapping[str, Any],
) -> dict[str, Any]:
    exclusion = json.loads((out_dir / "K1_EXCLUSION_MANIFEST.json").read_text())
    prior_roots = set(str(root) for root in exclusion["root_tokens"])
    roots = []
    states = []
    for family, _spec in FAMILY_SLATE:
        qualifying = [
            row for row in acquisition["rows"]
            if row["behavior_family"] == family and row["qualifying_root"]
        ]
        qualifying.sort(key=lambda row: int(row["game_index"]))
        if len(qualifying) != sum(ROOTS_PER_FAMILY.values()):
            raise EngineeringFault(
                f"K1 expected exactly 12 qualifying roots for {family}, "
                f"got {len(qualifying)}"
            )
        for index, completion in enumerate(qualifying):
            if completion["root_ancestry"] in prior_roots:
                raise EngineeringFault("K1 root overlaps prior branch")
            state_file = Path(str(completion["selected_states"]))
            if sha256_path(state_file) != completion["selected_states_sha256"]:
                raise EngineeringFault("K1 selected-state hash changed")
            payload = json.loads(state_file.read_text())
            if not verify_payload_hash(payload):
                raise EngineeringFault("K1 selected-state payload mismatch")
            if len(payload["states"]) != STATES_PER_ROOT:
                raise EngineeringFault("K1 selected-state count mismatch")
            partition = _partition_for_index(index)
            root = str(completion["root_ancestry"])
            roots.append({
                "root_ancestry": root,
                "behavior_family": family,
                "game_index": int(completion["game_index"]),
                "partition": partition,
                "source_replay": completion["source_replay"],
                "source_replay_sha256":
                    completion["source_replay_sha256"],
                "selected_states": str(state_file),
                "selected_states_sha256":
                    completion["selected_states_sha256"],
            })
            for row in payload["states"]:
                state, sim = _state_sim(row["state"])
                if not any(row["trigger_reasons"].values()):
                    raise EngineeringFault("K1 corpus has non-trigger state")
                record = {
                    **row,
                    "partition": partition,
                    "record_id": hashlib.sha256(
                        (
                            "K1-record-v1|"
                            f"{root}|{row['frame_index']}|"
                            f"{row['state_sha256']}"
                        ).encode()
                    ).hexdigest()[:24],
                    "restore_exact": state_payload(state, sim) == row["state"],
                }
                states.append(record)
    root_ids = [str(row["root_ancestry"]) for row in roots]
    summary = {
        partition: {
            "roots": sum(row["partition"] == partition for row in roots),
            "states": sum(row["partition"] == partition for row in states),
            "families": dict(Counter(
                row["behavior_family"]
                for row in roots
                if row["partition"] == partition
            )),
        }
        for partition in ROOTS_PER_FAMILY
    }
    checks = {
        "all_108_games_complete": acquisition["all_complete"],
        "exact_36_roots": len(roots) == 36 == len(set(root_ids)),
        "exact_144_states": len(states) == 144,
        "restore_exact": all(row["restore_exact"] for row in states),
        "zero_prior_overlap": not set(root_ids).intersection(prior_roots),
        "partitions_exact": all(
            summary[partition]["roots"] == 12
            and summary[partition]["states"] == 48
            and set(summary[partition]["families"]) == {
                family for family, _spec in FAMILY_SLATE
            }
            and max(summary[partition]["families"].values()) / 12 <= 0.40
            for partition in ROOTS_PER_FAMILY
        ),
    }
    payload = payload_with_hash({
        "version": "k1_fresh_corpus_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "decision": "K1_CORPUS_READY" if all(checks.values()) else "FAIL",
        "roots": roots,
        "states": states,
        "root_manifest_sha256": canonical_json_hash(roots),
        "state_manifest_sha256": canonical_json_hash(states),
        "partition_summary": summary,
        "checks": checks,
        "score_inspected": False,
        "policy_outcomes_compared": False,
        "dashboard_eligible": False,
    })
    if not all(checks.values()):
        raise EngineeringFault(f"K1 corpus failed: {checks}")
    atomic_write_json(out_dir / "K1_CORPUS_MANIFEST.json", payload)
    return payload


def _selection_seed(record_id: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"K1-tie-v1|{record_id}".encode()).digest()[:8],
        "little",
    )


def _values_exact(
    left: Sequence[tuple[int, float]],
    right: Sequence[tuple[int, float]],
) -> tuple[bool, float]:
    if [int(action) for action, _value in left] != [
        int(action) for action, _value in right
    ]:
        return False, float("inf")
    maximum = max(
        (
            abs(float(left_value) - float(right_value))
            for (_left_action, left_value), (_right_action, right_value)
            in zip(left, right)
        ),
        default=0.0,
    )
    return maximum <= VALUE_TOLERANCE, float(maximum)


def _fresh_exactness_row(
    *,
    record: Mapping[str, Any],
    incumbent: NtupleExpectimaxPolicy,
    reference: Any,
    compiled: Any,
) -> dict[str, Any]:
    state = state_from_replay_payload(record["state"])
    starter = STARTER_TILE
    board = np.asarray(state.board, dtype=np.int32)
    transition_exact = True
    for action in range(4):
        native = compiled.native_kernel.base_move(board, action)
        python = simulate_base_move(board, action)
        transition_exact = (
            transition_exact
            and np.array_equal(native[0], python[0])
            and native[1] == tuple(python[1])
        )
    score_exact = compiled.native_kernel.score_board(board) == score_board(board)
    leaf_native = float(compiled.native_kernel.evaluate_many([board])[0])
    leaf_reference = float(
        compiled.native_kernel.reference_leaf.evaluate_many([board])[0]
    )
    reference.clear_decision_caches()
    compiled.clear_decision_caches()
    sim_reference = ThreesSim.from_stream_ids(
        deck_stream_id=1,
        slot_stream_id=2,
        starter_tile=starter,
    )
    sim_compiled = ThreesSim.from_stream_ids(
        deck_stream_id=1,
        slot_stream_id=2,
        starter_tile=starter,
    )
    expected = reference.adaptive_values(state, sim_reference)
    actual = compiled.adaptive_values(state, sim_compiled)
    depth2_exact, depth2_max = _values_exact(
        expected["depth2"],
        actual["depth2"],
    )
    depth3_exact, depth3_max = _values_exact(
        expected["depth3"],
        actual["depth3"],
    )
    seed = _selection_seed(str(record["record_id"]))
    expected_action = choose_action(
        incumbent,
        expected["depth3"],
        seed,
    )
    actual_action = choose_action(
        incumbent,
        actual["depth3"],
        seed,
    )
    return {
        "record_id": record["record_id"],
        "root_ancestry": record["root_ancestry"],
        "behavior_family": record["behavior_family"],
        "partition": record["partition"],
        "transition_exact": transition_exact,
        "score_exact": score_exact,
        "leaf_difference": abs(leaf_native - leaf_reference),
        "leaf_exact": abs(leaf_native - leaf_reference) <= LEAF_TOLERANCE,
        "depth2_exact": depth2_exact,
        "depth2_max_difference": depth2_max,
        "depth3_exact": depth3_exact,
        "depth3_max_difference": depth3_max,
        "action_exact": expected_action == actual_action,
        "compiled_activity": (
            actual["compiled_calls"]["leaf"]
            + actual["compiled_calls"]["post_spawn"]
        ) > 0,
        "reference_fallback": bool(expected["reference_fallback"]),
        "compiled_fallback": bool(actual["reference_fallback"]),
        "fallback_exact":
            bool(expected["reference_fallback"])
            == bool(actual["reference_fallback"]),
    }


def _timed_partition(
    *,
    records: Sequence[Mapping[str, Any]],
    incumbent: NtupleExpectimaxPolicy,
    compiled: Any,
    expected_rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    baseline = make_policy(incumbent_spec())
    rows = []
    for record_index, record in enumerate(records):
        state = state_from_replay_payload(record["state"])
        sim = ThreesSim.from_stream_ids(
            deck_stream_id=1,
            slot_stream_id=2,
            starter_tile=STARTER_TILE,
        )
        baseline.action_values(state, sim)
        compiled.clear_decision_caches()
        compiled.adaptive_values(state, sim)
        baseline_times = []
        compiled_times = []
        repeat_exact = True
        compiled_active = True
        for repeat in range(TIMED_REPEATS):
            order = (
                ("baseline", "compiled")
                if (record_index + repeat) % 2 == 0
                else ("compiled", "baseline")
            )
            values: dict[str, Any] = {}
            for arm in order:
                sim = ThreesSim.from_stream_ids(
                    deck_stream_id=1,
                    slot_stream_id=2,
                    starter_tile=STARTER_TILE,
                )
                if arm == "baseline":
                    started = time.perf_counter_ns()
                    values[arm] = baseline.action_values(state, sim)
                    baseline_times.append(
                        (time.perf_counter_ns() - started) / 1e9
                    )
                else:
                    compiled.clear_decision_caches()
                    started = time.perf_counter_ns()
                    values[arm] = compiled.adaptive_values(state, sim)
                    compiled_times.append(
                        (time.perf_counter_ns() - started) / 1e9
                    )
            expected = expected_rows[str(record["record_id"])]
            depth2_exact, _ = _values_exact(
                expected["depth2_values"],
                values["compiled"]["depth2"],
            )
            depth3_exact, _ = _values_exact(
                expected["depth3_values"],
                values["compiled"]["depth3"],
            )
            baseline_exact, _ = _values_exact(
                expected["depth2_values"],
                values["baseline"],
            )
            repeat_exact = (
                repeat_exact
                and depth2_exact
                and depth3_exact
                and baseline_exact
            )
            compiled_active = compiled_active and (
                values["compiled"]["compiled_calls"]["leaf"]
                + values["compiled"]["compiled_calls"]["post_spawn"]
            ) > 0
        baseline_median = float(np.median(baseline_times))
        compiled_median = float(np.median(compiled_times))
        rows.append({
            "record_id": record["record_id"],
            "root_ancestry": record["root_ancestry"],
            "behavior_family": record["behavior_family"],
            "partition": record["partition"],
            "baseline_seconds": baseline_times,
            "compiled_seconds": compiled_times,
            "baseline_median_seconds": baseline_median,
            "compiled_median_seconds": compiled_median,
            "compiled_over_depth2":
                compiled_median / max(baseline_median, 1e-12),
            "repeat_exact": repeat_exact,
            "compiled_activity": compiled_active,
        })
    return {"rows": rows}


def _runtime_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ratios = np.asarray(
        [float(row["compiled_over_depth2"]) for row in rows],
        dtype=np.float64,
    )
    absolute = np.asarray(
        [float(row["compiled_median_seconds"]) for row in rows],
        dtype=np.float64,
    )
    baseline = np.asarray(
        [float(row["baseline_median_seconds"]) for row in rows],
        dtype=np.float64,
    )
    families = sorted({
        str(row["behavior_family"])
        for row in rows if row["compiled_activity"]
    })
    return {
        "records": len(rows),
        "roots": len({str(row["root_ancestry"]) for row in rows}),
        "families": dict(Counter(str(row["behavior_family"]) for row in rows)),
        "ratio_median": float(np.median(ratios)),
        "ratio_p90": float(np.quantile(ratios, 0.90)),
        "ratio_p99": float(np.quantile(ratios, 0.99)),
        "ratio_max": float(np.max(ratios)),
        "absolute_median_seconds": float(np.median(absolute)),
        "absolute_p90_seconds": float(np.quantile(absolute, 0.90)),
        "absolute_p99_seconds": float(np.quantile(absolute, 0.99)),
        "absolute_max_seconds": float(np.max(absolute)),
        "baseline_median_seconds": float(np.median(baseline)),
        "baseline_p10_seconds": float(np.quantile(baseline, 0.10)),
        "baseline_min_seconds": float(np.min(baseline)),
        "activity_fraction": float(np.mean([
            bool(row["compiled_activity"]) for row in rows
        ])),
        "activity_families": families,
        "all_repeat_exact": all(bool(row["repeat_exact"]) for row in rows),
    }


def runtime_gate_checks(
    summary: Mapping[str, Any],
    *,
    exactness_checks: Mapping[str, bool],
) -> dict[str, bool]:
    return {
        "ratio_median_le_3": float(summary["ratio_median"]) <= 3.0,
        "ratio_p90_le_5": float(summary["ratio_p90"]) <= 5.0,
        "ratio_p99_le_8": float(summary["ratio_p99"]) <= 8.0,
        "ratio_max_le_12": float(summary["ratio_max"]) <= 12.0,
        "absolute_p99_lt_2_5":
            float(summary["absolute_p99_seconds"]) < 2.5,
        "zero_mismatch": all(exactness_checks.values())
        and bool(summary["all_repeat_exact"]),
        "activity_100pct": float(summary["activity_fraction"]) == 1.0,
        "activity_three_families":
            len(summary["activity_families"]) == 3,
    }


def run_fresh_gate(
    *,
    out_dir: Path,
    corpus: Mapping[str, Any],
    library_path: Path,
) -> dict[str, Any]:
    incumbent = make_policy(incumbent_spec())
    if not isinstance(incumbent, NtupleExpectimaxPolicy):
        raise EngineeringFault("K1 incumbent type mismatch")
    reference = clone_batched(incumbent)
    compiled = clone_k1(incumbent, library_path)
    exactness_rows = []
    expected_rows = {}
    for record in corpus["states"]:
        row = _fresh_exactness_row(
            record=record,
            incumbent=incumbent,
            reference=reference,
            compiled=compiled,
        )
        exactness_rows.append(row)
        state = state_from_replay_payload(record["state"])
        sim = ThreesSim.from_stream_ids(
            deck_stream_id=1,
            slot_stream_id=2,
            starter_tile=STARTER_TILE,
        )
        reference.clear_decision_caches()
        values = reference.adaptive_values(state, sim)
        expected_rows[str(record["record_id"])] = {
            "depth2_values": values["depth2"],
            "depth3_values": values["depth3"],
        }
    exactness_checks = {
        "all_transition_exact": all(
            row["transition_exact"] for row in exactness_rows
        ),
        "all_score_exact": all(row["score_exact"] for row in exactness_rows),
        "all_leaf_exact": all(row["leaf_exact"] for row in exactness_rows),
        "all_depth2_exact": all(row["depth2_exact"] for row in exactness_rows),
        "all_depth3_exact": all(row["depth3_exact"] for row in exactness_rows),
        "all_action_exact": all(row["action_exact"] for row in exactness_rows),
        "all_fallback_exact": all(
            row["fallback_exact"] for row in exactness_rows
        ),
        "all_compiled_active": all(
            row["compiled_activity"] for row in exactness_rows
        ),
    }
    validation_records = [
        row for row in corpus["states"]
        if row["partition"] == "engineering_validation"
    ]
    gate_records = [
        row for row in corpus["states"]
        if row["partition"] == "untouched_runtime_gate"
    ]
    validation = _timed_partition(
        records=validation_records,
        incumbent=incumbent,
        compiled=compiled,
        expected_rows=expected_rows,
    )
    gate = _timed_partition(
        records=gate_records,
        incumbent=incumbent,
        compiled=compiled,
        expected_rows=expected_rows,
    )
    validation_summary = _runtime_summary(validation["rows"])
    gate_summary = _runtime_summary(gate["rows"])
    gate_checks = runtime_gate_checks(
        gate_summary,
        exactness_checks=exactness_checks,
    )
    decision = (
        "READY_K1_FULL_POLICY_PREFLIGHT"
        if all(gate_checks.values())
        else "KILL_K1_COMPILED_KERNEL"
    )
    report = payload_with_hash({
        "version": "k1_fresh_engineering_gate_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "decision": decision,
        "exactness_checks": exactness_checks,
        "exactness_rows": exactness_rows,
        "validation": {
            "summary": validation_summary,
            "rows": validation["rows"],
        },
        "untouched_runtime_gate": {
            "summary": gate_summary,
            "checks": gate_checks,
            "rows": gate["rows"],
        },
        "compiler_binding": compiled.native_kernel.binding_manifest,
        "policy_outcomes": False,
        "scores_inspected": False,
        "dashboard_eligible": False,
    })
    atomic_write_json(out_dir / "K1_FRESH_ENGINEERING_GATE.json", report)
    return report


def _artifact_manifest(out_dir: Path) -> dict[str, Any]:
    excluded = {"K1_TERMINAL_RESULT.json"}
    rows = [
        {
            "relative_path": str(path.relative_to(out_dir)),
            "byte_size": path.stat().st_size,
            "sha256": sha256_path(path),
        }
        for path in sorted(out_dir.rglob("*"))
        if path.is_file() and path.name not in excluded
    ]
    return {
        "files": rows,
        "file_count": len(rows),
        "total_bytes": sum(int(row["byte_size"]) for row in rows),
        "manifest_sha256": canonical_json_hash(rows),
    }


def _seal_terminal(
    *,
    out_dir: Path,
    marker: Mapping[str, Any],
    decision: str,
    stage: str,
    summary: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    terminal_path = out_dir / "K1_TERMINAL_RESULT.json"
    if terminal_path.exists():
        raise FileExistsError("K1 terminal already exists")
    operations = _operational_audit(out_dir)
    payload = payload_with_hash({
        "version": "k1_terminal_result_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "decision": decision,
        "stage": stage,
        "summary": dict(summary),
        "marker_file_sha256": sha256_path(
            out_dir / "K1_EXECUTION_OPENED.json"
        ),
        "marker_payload_sha256": marker["canonical_payload_sha256"],
        "operations": operations,
        "artifact_manifest": _artifact_manifest(out_dir),
        "forbidden_work": {
            "policy_outcomes": 0,
            "game_scores_inspected": 0,
            "h10_h20_h40_outcomes": 0,
            "new_labels": 0,
            "training_models": 0,
            "incumbent_changes": 0,
            "dashboard_changes": 0,
            "human_actions_used": 0,
        },
        "state": {
            "CONTINUE":
                "future_full_policy_preflight" if decision
                == "READY_K1_FULL_POLICY_PREFLIGHT" else "none",
            "HOLD": "policy_evaluation_human_training_ground_PROMOTE",
            "KILL": {
                "C1": "permanent",
                "C2": "permanent",
                "G3": "permanent",
                "G4": "permanent",
                "K1": decision == "KILL_K1_COMPILED_KERNEL",
            },
            "PROMOTE": False,
        },
        **(dict(extra) if extra else {}),
    })
    atomic_write_json(terminal_path, payload)
    return {
        "decision": decision,
        "terminal_path": str(terminal_path),
        "terminal_file_sha256": sha256_path(terminal_path),
        "terminal_payload_sha256": payload["canonical_payload_sha256"],
        "stage": stage,
        "summary": dict(summary),
        "operations": operations,
        "state": payload["state"],
        "forbidden_work": payload["forbidden_work"],
    }


def run_execution(
    *,
    out_dir: Path,
    preflight_lock: Path,
    jobs: int,
) -> dict[str, Any]:
    lock = _load_lock(out_dir, preflight_lock)
    marker = _load_marker(out_dir, lock)
    if (out_dir / "K1_TERMINAL_RESULT.json").exists():
        raise FileExistsError("K1 terminal already sealed")
    stage = "start"
    runtime_path = out_dir / "runtime_state.json"
    runtime = _runtime_state(runtime_path)
    try:
        _revalidate_files(out_dir, lock)
        if not _collision_revalidation(out_dir, lock)["passes"]:
            raise EngineeringFault("K1 collision revalidation failed")
        _execution_guard(out_dir, runtime)
        stage = "build"
        library_path = out_dir / "libk1_exact.dylib"
        build_manifest_path = out_dir / "K1_BUILD_MANIFEST.json"
        if not library_path.exists():
            started = time.perf_counter()
            build = payload_with_hash(build_native_kernel(library_path))
            atomic_write_json(build_manifest_path, build)
            runtime["active_runtime_seconds"] += time.perf_counter() - started
            runtime["phase"] = "build_complete"
            atomic_write_json(runtime_path, runtime)
        else:
            build = json.loads(build_manifest_path.read_text())
            if (
                not verify_payload_hash(build)
                or build["library_sha256"] != sha256_path(library_path)
            ):
                raise EngineeringFault("K1 resumed build mismatch")
        incumbent = make_policy(incumbent_spec())
        if not isinstance(incumbent, NtupleExpectimaxPolicy):
            raise EngineeringFault("K1 incumbent type mismatch")
        kernel = NativeKernel(library_path, incumbent)
        binding = payload_with_hash(kernel.binding_manifest)
        atomic_write_json(out_dir / "K1_NATIVE_BINDING.json", binding)
        stage = "acquisition"
        acquisition = run_acquisition(
            out_dir=out_dir,
            lock=lock,
            jobs=jobs,
        )
        runtime = acquisition["runtime"]
        if not acquisition["all_complete"]:
            raise EngineeringFault("K1 games were not all complete")
        stage = "corpus"
        corpus = build_corpus(out_dir=out_dir, acquisition=acquisition)
        runtime["phase"] = "fresh_gate"
        atomic_write_json(runtime_path, runtime)
        _execution_guard(out_dir, runtime)
        stage = "fresh_engineering_gate"
        started = time.perf_counter()
        gate = run_fresh_gate(
            out_dir=out_dir,
            corpus=corpus,
            library_path=library_path,
        )
        runtime["active_runtime_seconds"] += time.perf_counter() - started
        runtime["phase"] = "terminal"
        atomic_write_json(runtime_path, runtime)
        summary = gate["untouched_runtime_gate"]["summary"]
        return _seal_terminal(
            out_dir=out_dir,
            marker=marker,
            decision=gate["decision"],
            stage=stage,
            summary=summary,
            extra={
                "build_file_sha256": sha256_path(build_manifest_path),
                "library_sha256": sha256_path(library_path),
                "binding_file_sha256": sha256_path(
                    out_dir / "K1_NATIVE_BINDING.json"
                ),
                "corpus_file_sha256": sha256_path(
                    out_dir / "K1_CORPUS_MANIFEST.json"
                ),
                "gate_file_sha256": sha256_path(
                    out_dir / "K1_FRESH_ENGINEERING_GATE.json"
                ),
                "gate_payload_sha256": gate["canonical_payload_sha256"],
                "validation_summary": gate["validation"]["summary"],
                "gate_checks":
                    gate["untouched_runtime_gate"]["checks"],
                "active_runtime_seconds":
                    runtime["active_runtime_seconds"],
            },
        )
    except Exception:
        if (out_dir / "K1_TERMINAL_RESULT.json").exists():
            raise
        return _seal_terminal(
            out_dir=out_dir,
            marker=marker,
            decision="HOLD_K1_ENGINEERING_FAULT",
            stage=stage,
            summary={"error": str(traceback.format_exc().splitlines()[-1])},
            extra={"traceback": traceback.format_exc()},
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--out-dir", type=Path, required=True)
    preflight.add_argument("--jobs", type=int, required=True)
    for name in ("open", "execute"):
        command = subparsers.add_parser(name)
        command.add_argument("--out-dir", type=Path, required=True)
        command.add_argument("--preflight-lock", type=Path, required=True)
        command.add_argument("--jobs", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "preflight":
        result = run_preflight(out_dir=args.out_dir, jobs=args.jobs)
    elif args.command == "open":
        result = seal_execution_opened(
            out_dir=args.out_dir,
            preflight_lock=args.preflight_lock,
            jobs=args.jobs,
        )
    else:
        result = run_execution(
            out_dir=args.out_dir,
            preflight_lock=args.preflight_lock,
            jobs=args.jobs,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
