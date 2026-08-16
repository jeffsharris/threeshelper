"""Freeze and run the one-shot C2 deterministic cost-admission assay."""

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
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import lsq_linear
from scipy.stats import spearmanr

from threes_rl import g1r_acquire as history
from threes_rl.c1_search_optimization import (
    BatchedPersistentPolicy,
    _values_close,
    clone_batched,
)
from threes_rl.eval import (
    EvalJob,
    EvalStreamIds,
    iter_eval_job_outputs,
    make_policy,
    max_tile_excluding_initial_starter,
)
from threes_rl.expectimax import NtupleExpectimaxPolicy, _state_key
from threes_rl.g1r_qd_admission_v2 import _heavy_process_audit
from threes_rl.r2a_adaptive_expectimax import (
    CHANCE_LIMIT,
    EMPTY_TRIGGER,
    MARGIN_TRIGGER,
    NODE_BUDGET,
    choose_action,
    milestone_for_built_max,
    normalized_margin,
)
from threes_rl.record_replay import state_payload
from threes_rl.replay_provenance import ORIGIN_FRESH, direct_root_fields
from threes_rl.s3_power_preflight import sha256_path
from threes_rl.sim import DIRECTION_NAMES, ThreesSim
from threes_rl.train_td import state_from_replay_payload


VERSION = "c2_cost_admission_v1"
CHARTER_PATH = Path("threes_rl/C2_COST_ADMISSION_EXECUTION_CHARTER.md")
IMPLEMENTATION_PATH = Path("threes_rl/c2_cost_admission.py")
TEST_PATH = Path("tests/test_rl_c2_cost_admission.py")
TEST_EVIDENCE_PATH = Path(
    "threes_rl/runs/forensics/c2_cost_admission_test_evidence.json"
)
OUTPUT_DIR = Path("threes_rl/runs/forensics/c2_cost_admission_v1")
INCUMBENT_PATH = Path("threes_rl/current_incumbent_policy.txt")
G1R_PREFLIGHT_PATH = Path(
    "threes_rl/runs/forensics/g1r_acquisition/pilot_v2_qd5/preflight_lock.json"
)

FAMILY_SLATE = (
    ("c2_corner2", "corner2"),
    (
        "c2_parent_mc1000",
        "ntuple_expectimax2:"
        "threes_rl/runs/td_default_corner2_mc_1000_init3000_a0005_"
        "20260706/latest",
    ),
    (
        "c2_replaycal",
        "ntuple_expectimax2:"
        "threes_rl/runs/replay_cal_phase4_late_midlate_top13_e3_a001_tc_"
        "20260706/latest",
    ),
)
G1R_FAMILY_NAMES = {
    "c2_corner2": "g1r_corner2",
    "c2_parent_mc1000": "g1r_parent_mc1000",
    "c2_replaycal": "g1r_replaycal",
}
EXPECTED_SIGNATURES = {
    "c2_corner2":
        "4be4214166f40ddaaac5af499cb1e1d08d992b0a90bb680cfcb7cab04d217043",
    "c2_parent_mc1000":
        "e43dc11f3220557d7f9aef228db96dc6f06f49b26300d5a4128ea00bf8ba2064",
    "c2_replaycal":
        "e07c566b55d86a889ab7ca54d01c00c9b6cdf808fdb1627f70596bd829fdeab3",
}
EXPECTED_PAIRWISE = {
    ("c2_corner2", "c2_parent_mc1000"): (0.53125, 0.59375, 0.46875),
    ("c2_corner2", "c2_replaycal"): (0.515625, 0.59375, 0.4375),
    ("c2_parent_mc1000", "c2_replaycal"): (0.15625, 0.28125, 0.03125),
}

GAMES_PER_FAMILY = 72
TOTAL_GAMES = len(FAMILY_SLATE) * GAMES_PER_FAMILY
MAX_MOVES = 5000
STARTER_TILE = 1536
MAX_CHUNK_SIZE = 6
FROZEN_JOBS = 1
MINIMUM_NICE = 10
ACTIVE_WALL_SECONDS = 6 * 3600
BYTE_LIMIT = 4 * 1024**3
MIN_FREE_GIB = 100.0
TARGET_FREE_GIB = 120.0
STREAM_BASES = {
    "logical_seed": 65_000_000_000,
    "deck_stream_id": 66_000_000_000,
    "slot_stream_id": 67_000_000_000,
    "policy_stream_id": 68_000_000_000,
}

ROOTS_PER_FAMILY = {
    "cost_fit": 6,
    "engineering_validation": 2,
    "untouched_runtime_gate": 4,
}
STATES_PER_ROOT = 4
TIMED_REPEATS = 3
VALUE_TOLERANCE = 1e-9

ABSOLUTE_LOAD_SECONDS = 2.0
RELATIVE_LOAD_RATIO = 6.0
UPPER_MULTIPLIER = 1.25
UPPER_OFFSET = 0.10
ADMISSION_THRESHOLD = 1.0
L2_LAMBDA = 0.001
LSQ_TOLERANCE = 1e-12
LSQ_MAX_ITER = 10_000

FEATURE_NAMES = (
    "legal_actions",
    "empty_fraction",
    "preview_red",
    "preview_blue",
    "preview_gray",
    "preview_bonus",
    "preview_candidate_fraction",
    "low_margin_pressure",
    "low_empty_pressure",
    "action_calls",
    "value_lookups",
    "unique_value_states",
    "chance_calls",
    "chance_outcomes",
    "afterstate_lookups",
    "unique_afterstates",
    "base_move_calls",
    "unique_base_moves",
    "legal_lookup_calls",
    "cheap_depth2_pressure",
)

PRIOR_ROOT_SOURCE_PATHS = (
    Path("threes_rl/runs/forensics/c1_search/C1_CORPUS.json"),
    Path("threes_rl/runs/forensics/r2a_adaptive/R2A_ROOT_MANIFEST.json"),
    Path(
        "threes_rl/runs/forensics/g1_relational/"
        "G1_EXISTING_CORPUS_PREFLIGHT_V5_AUTHORITATIVE.json"
    ),
    Path(
        "threes_rl/runs/forensics/g2_scale_equivariant_relational_hazard/"
        "G2_ROOT_MANIFEST.json"
    ),
    Path(
        "threes_rl/runs/forensics/g2_fresh_transfer_acquisition_v1/"
        "G2_TRANSFER_ACQUISITION_RESULT.json"
    ),
    Path("threes_rl/runs/forensics/g3_e0_label_fit_v4/E0_RECORD_MANIFEST.json"),
    Path(
        "threes_rl/runs/forensics/g4_conditional_pairwise_v1/"
        "G4_PAIR_MANIFEST.json"
    ),
    Path(
        "threes_rl/runs/forensics/g4_conditional_pairwise_v2/"
        "G4_V2_FOLD_MANIFEST.json"
    ),
    Path(
        "threes_rl/runs/forensics/s3_full_policy/S3_PROVENANCE_SEAL_V2.json"
    ),
)

IMMUTABLE_SOURCE_PATHS = (
    CHARTER_PATH,
    IMPLEMENTATION_PATH,
    TEST_PATH,
    INCUMBENT_PATH,
    G1R_PREFLIGHT_PATH,
    Path("threes_rl/C1_SEARCH_OPTIMIZATION_PREREGISTRATION.md"),
    Path("threes_rl/C1_TAIL_MECHANISM_AUDIT.md"),
    Path("threes_rl/C2_COST_ADMISSION_PROPOSAL.md"),
    Path("threes_rl/c1_search_optimization.py"),
    Path("threes_rl/r2a_adaptive_expectimax.py"),
    Path("threes_rl/eval.py"),
    Path("threes_rl/expectimax.py"),
    Path("threes_rl/ntuple.py"),
    Path("threes_rl/sim.py"),
    Path("threes_rl/replay_provenance.py"),
    Path("threes_rl/train_td.py"),
    Path("threes_rl/runs/forensics/c1_search/C1_PREFLIGHT_LOCK.json"),
    Path("threes_rl/runs/forensics/c1_search/C1_CORPUS.json"),
    Path("threes_rl/runs/forensics/c1_search/C1_BYTEKEY_LEAF_BENCHMARK.json"),
    Path("threes_rl/runs/forensics/c1_search/C1_RUNTIME_GATE.json"),
    Path("threes_rl/runs/forensics/c1_search/C1_STOP_GO.json"),
)

ROOT_KEYS = {
    "root",
    "root_cluster",
    "root_ancestry",
    "ancestry",
    "ancestry_id",
    "canonical_root",
    "source_root",
}
LIVE_COLLISION_PATHS = (
    Path("threes_rl/runs/dashboard/dashboard.json"),
    Path("threes_rl/runs/dashboard/score_trends.json"),
)


class EngineeringFault(RuntimeError):
    """A genuine execution/integrity fault, not a scientific gate failure."""


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
    expected = str(payload.get("canonical_payload_sha256", ""))
    unhashed = dict(payload)
    unhashed.pop("canonical_payload_sha256", None)
    return bool(expected) and canonical_json_hash(unhashed) == expected


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
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
        raise ValueError(f"Empty artifact manifest: {path}")
    return {
        "path": str(path),
        "files": rows,
        "file_count": len(rows),
        "total_bytes": sum(int(row["byte_size"]) for row in rows),
        "manifest_sha256": canonical_json_hash(rows),
    }


def incumbent_spec() -> str:
    lines = [
        line.strip()
        for line in INCUMBENT_PATH.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(lines) != 1:
        raise ValueError("Expected exactly one incumbent policy line")
    return lines[0]


def stream_ids(family_index: int, game_index: int) -> dict[str, int]:
    offset = int(family_index) * 1_000_000 + int(game_index)
    return {name: int(base + offset) for name, base in STREAM_BASES.items()}


def requested_stream_manifest() -> list[dict[str, Any]]:
    rows = []
    for family_index, (family, spec) in enumerate(FAMILY_SLATE):
        for game_index in range(GAMES_PER_FAMILY):
            rows.append({
                "family_index": family_index,
                "behavior_family": family,
                "policy_spec": spec,
                "game_index": game_index,
                **stream_ids(family_index, game_index),
            })
    return rows


def _recursive_root_values(value: Any, *, parent_key: str | None = None) -> set[str]:
    roots: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ROOT_KEYS and isinstance(item, (str, int)):
                roots.add(str(item))
            roots.update(_recursive_root_values(item, parent_key=str(key)))
    elif isinstance(value, list):
        for item in value:
            roots.update(_recursive_root_values(item, parent_key=parent_key))
    elif (
        isinstance(value, str)
        and (value.startswith("fresh:") or value.startswith("human:"))
        and parent_key is not None
    ):
        roots.add(value)
    return roots


def build_exclusion_manifest() -> dict[str, Any]:
    source_rows = []
    roots: set[str] = set()
    for path in PRIOR_ROOT_SOURCE_PATHS:
        if not path.is_file():
            raise FileNotFoundError(f"Missing C2 exclusion source: {path}")
        payload = json.loads(path.read_text())
        source_roots = _recursive_root_values(payload)
        roots.update(source_roots)
        source_rows.append({
            "path": str(path),
            "sha256": sha256_path(path),
            "byte_size": path.stat().st_size,
            "root_tokens": len(source_roots),
        })
    requested_roots = {
        f"fresh:{row['logical_seed']}:{STARTER_TILE}"
        for row in requested_stream_manifest()
    }
    collisions = sorted(requested_roots.intersection(roots))
    return payload_with_hash({
        "version": "c2_root_exclusion_v1",
        "sources": source_rows,
        "source_manifest_sha256": canonical_json_hash(source_rows),
        "root_tokens": sorted(roots),
        "root_token_count": len(roots),
        "requested_root_count": len(requested_roots),
        "requested_root_collisions": collisions,
        "passes": not collisions,
    })


def _scan_collision_sources(
    *,
    out_dir: Path,
) -> dict[str, Any]:
    prior, history_payload = history.historical_collision_union(exclude_dir=out_dir)
    immutable_sources = []
    live_sources = []
    for row in history_payload["matched_sources"]:
        path = Path(str(row["path"]))
        target = live_sources if path in LIVE_COLLISION_PATHS else immutable_sources
        target.append(row)
    return {
        "prior": prior,
        "immutable_sources": immutable_sources,
        "live_sources": live_sources,
        "immutable_source_count": len(immutable_sources),
        "immutable_inventory_sha256": canonical_json_hash(immutable_sources),
        "live_source_count": len(live_sources),
        "live_paths": sorted(str(row["path"]) for row in live_sources),
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
        requested = {int(row[key]) for row in rows}
        collisions[key] = sorted(requested.intersection(prior_values))
    all_ids = [int(row[key]) for row in rows for key in STREAM_BASES]
    checks = {
        "exact_216_rows": len(rows) == TOTAL_GAMES,
        "exact_864_internal_ids": len(all_ids) == 4 * TOTAL_GAMES,
        "internal_ids_unique": len(all_ids) == len(set(all_ids)),
        "zero_historical_collisions": not any(collisions.values()),
        "live_paths_recognized": set(scan["live_paths"]).issubset(
            {str(path) for path in LIVE_COLLISION_PATHS}
        ),
    }
    return payload_with_hash({
        "version": "c2_stream_collision_v1",
        "requested_rows_sha256": canonical_json_hash(list(rows)),
        "stream_bases": STREAM_BASES,
        "collisions": collisions,
        "immutable_sources": scan["immutable_sources"],
        "immutable_source_count": scan["immutable_source_count"],
        "immutable_inventory_sha256": scan["immutable_inventory_sha256"],
        "live_sources": scan["live_sources"],
        "live_source_count": scan["live_source_count"],
        "live_paths": scan["live_paths"],
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
        raise ValueError("C2 collision manifest payload hash mismatch")
    current = build_stream_collision_manifest(rows, out_dir=out_dir)
    checks = {
        "requested_rows_exact":
            current["requested_rows_sha256"] == manifest["requested_rows_sha256"],
        "immutable_inventory_exact":
            current["immutable_inventory_sha256"]
            == manifest["immutable_inventory_sha256"]
            and current["immutable_sources"] == manifest["immutable_sources"],
        "live_paths_exact": current["live_paths"] == manifest["live_paths"],
        "zero_collisions": current["passes"],
    }
    return {
        "checks": checks,
        "passes": all(checks.values()),
        "current": current,
    }


def build_policy_lock() -> tuple[dict[str, Any], dict[str, Any]]:
    source_paths = (
        Path("threes_rl/eval.py"),
        Path("threes_rl/expectimax.py"),
        Path("threes_rl/ntuple.py"),
        Path("threes_rl/sim.py"),
        Path("threes_rl/c1_search_optimization.py"),
        Path("threes_rl/r2a_adaptive_expectimax.py"),
    )
    source_hashes = {str(path): sha256_path(path) for path in source_paths}
    loaded = {}
    family_rows = []
    for family, spec in FAMILY_SLATE:
        policy = make_policy(spec)
        loaded[family] = policy
        checkpoint_manifests = []
        if ":threes_rl/runs/" in spec:
            checkpoint = Path(spec.split(":", 1)[1])
            checkpoint_manifests.append(file_manifest(checkpoint))
        family_rows.append({
            "behavior_family": family,
            "policy_spec": spec,
            "policy_spec_sha256": hashlib.sha256(spec.encode()).hexdigest(),
            "loaded_type": type(policy).__name__,
            "action_signature_sha256": EXPECTED_SIGNATURES[family],
            "checkpoint_manifests": checkpoint_manifests,
        })
    lock = {
        "families": family_rows,
        "family_order": [family for family, _spec in FAMILY_SLATE],
        "policy_sources": source_hashes,
        "policy_source_manifest_sha256": canonical_json_hash(source_hashes),
        "incumbent_policy_file": str(INCUMBENT_PATH),
        "incumbent_policy_file_sha256": sha256_path(INCUMBENT_PATH),
        "incumbent_spec": incumbent_spec(),
        "incumbent_spec_sha256": hashlib.sha256(incumbent_spec().encode()).hexdigest(),
    }
    lock["policy_lock_sha256"] = canonical_json_hash(lock)
    return lock, loaded


def revalidate_policy_lock(expected: Mapping[str, Any]) -> dict[str, Any]:
    current, _loaded = build_policy_lock()
    return {
        "expected_sha256": expected["policy_lock_sha256"],
        "current_sha256": current["policy_lock_sha256"],
        "exact": current == expected,
    }


def accepted_family_audit() -> dict[str, Any]:
    if sha256_path(G1R_PREFLIGHT_PATH) != (
        "0d50edaae52e9a6f6291c4b397fd03c9d7d8651b28bb9dbd05b53c8718ee22ad"
    ):
        raise ValueError("Accepted G1-R preflight changed")
    lock = json.loads(G1R_PREFLIGHT_PATH.read_text())
    signature_rows = lock["action_signature_audit"]["signature_sha256"]
    signature_checks = {
        family: signature_rows.get(G1R_FAMILY_NAMES[family])
        == EXPECTED_SIGNATURES[family]
        for family, _spec in FAMILY_SLATE
    }
    pair_rows = {
        (str(row["left"]), str(row["right"])): row
        for row in lock["action_signature_audit"]["pairwise"]
    }
    pair_checks = {}
    pairwise = []
    for pair, expected in EXPECTED_PAIRWISE.items():
        left = G1R_FAMILY_NAMES[pair[0]]
        right = G1R_FAMILY_NAMES[pair[1]]
        row = pair_rows.get((left, right)) or pair_rows.get((right, left))
        if row is None:
            pair_checks["|".join(pair)] = False
            continue
        observed = (
            float(row["overall_disagreement"]),
            float(row["stratum_disagreement"]["pre1536"]),
            float(row["stratum_disagreement"]["pre3072"]),
        )
        passes = observed == expected and bool(row["passes"])
        pair_checks["|".join(pair)] = passes
        pairwise.append({
            "left": pair[0],
            "right": pair[1],
            "overall": observed[0],
            "pre1536": observed[1],
            "pre3072": observed[2],
            "passes": passes,
        })
    checks = {
        "signatures_exact": all(signature_checks.values()),
        "pairwise_exact": all(pair_checks.values()),
        "three_genuine_families": len(FAMILY_SLATE) == 3,
    }
    return {
        "source": str(G1R_PREFLIGHT_PATH),
        "source_sha256": sha256_path(G1R_PREFLIGHT_PATH),
        "signature_checks": signature_checks,
        "pair_checks": pair_checks,
        "pairwise": pairwise,
        "checks": checks,
        "passes": all(checks.values()),
    }


def feature_schema_payload() -> dict[str, Any]:
    formulas = (
        "root_legal_count/4",
        "empty_count/16",
        "1[preview.kind==red]",
        "1[preview.kind==blue]",
        "1[preview.kind==gray]",
        "1[preview.kind==bonus]",
        "len(preview.candidates)/3",
        "clip((0.02-margin)/0.02,0,1)",
        "clip((3-empty_count)/3,0,1)",
        "clip(log1p(action_calls)/log1p(64),0,1)",
        "clip(log1p(value_lookups)/log1p(4096),0,1)",
        "clip(log1p(unique_value_states)/log1p(2048),0,1)",
        "clip(log1p(chance_calls)/log1p(2048),0,1)",
        "clip(log1p(chance_outcomes)/log1p(8192),0,1)",
        "clip(log1p(afterstate_lookups)/log1p(16384),0,1)",
        "clip(log1p(unique_afterstates)/log1p(8192),0,1)",
        "clip(log1p(base_move_calls)/log1p(32768),0,1)",
        "clip(log1p(unique_base_moves)/log1p(16384),0,1)",
        "clip(log1p(legal_lookup_calls)/log1p(4096),0,1)",
        "1/(1+(unique_value_states+unique_afterstates+"
        "chance_outcomes/4+unique_base_moves/4)/256)",
    )
    payload = {
        "version": "c2_cost_feature_schema_v1",
        "names": list(FEATURE_NAMES),
        "formulas": list(formulas),
        "width": len(FEATURE_NAMES),
        "all_inputs_available_after_depth2": True,
        "wall_clock_input": False,
        "future_outcome_input": False,
        "normalization_is_fixed": True,
    }
    payload["schema_sha256"] = canonical_json_hash(payload)
    return payload


def _normalized_log_count(value: int, maximum: int) -> float:
    if value < 0:
        raise ValueError("C2 counter cannot be negative")
    return float(np.clip(math.log1p(value) / math.log1p(maximum), 0.0, 1.0))


def cost_features(
    *,
    state: Any,
    values: Sequence[tuple[int, float]],
    counters: Mapping[str, int],
) -> np.ndarray:
    empty_count = int(np.count_nonzero(np.asarray(state.board) == 0))
    margin = float(normalized_margin(list(values)))
    preview_kind = str(state.preview.kind)
    if preview_kind not in {"red", "blue", "gray", "bonus"}:
        raise ValueError(f"Unsupported C2 preview kind: {preview_kind}")
    candidates = tuple(int(value) for value in state.preview.candidates)
    if len(candidates) > 3:
        raise ValueError("C2 preview candidate count exceeds three")
    work_units = (
        int(counters["unique_value_states"])
        + int(counters["unique_afterstates"])
        + int(counters["chance_outcomes"]) / 4.0
        + int(counters["unique_base_moves"]) / 4.0
    )
    features = np.asarray([
        len(values) / 4.0,
        empty_count / 16.0,
        float(preview_kind == "red"),
        float(preview_kind == "blue"),
        float(preview_kind == "gray"),
        float(preview_kind == "bonus"),
        len(candidates) / 3.0,
        float(np.clip((MARGIN_TRIGGER - margin) / MARGIN_TRIGGER, 0.0, 1.0)),
        float(np.clip((EMPTY_TRIGGER - empty_count) / EMPTY_TRIGGER, 0.0, 1.0)),
        _normalized_log_count(int(counters["action_calls"]), 64),
        _normalized_log_count(int(counters["value_lookups"]), 4096),
        _normalized_log_count(int(counters["unique_value_states"]), 2048),
        _normalized_log_count(int(counters["chance_calls"]), 2048),
        _normalized_log_count(int(counters["chance_outcomes"]), 8192),
        _normalized_log_count(int(counters["afterstate_lookups"]), 16384),
        _normalized_log_count(int(counters["unique_afterstates"]), 8192),
        _normalized_log_count(int(counters["base_move_calls"]), 32768),
        _normalized_log_count(int(counters["unique_base_moves"]), 16384),
        _normalized_log_count(int(counters["legal_lookup_calls"]), 4096),
        1.0 / (1.0 + work_units / 256.0),
    ], dtype=np.float64)
    if features.shape != (len(FEATURE_NAMES),):
        raise AssertionError("C2 feature width changed")
    if not np.all(np.isfinite(features)) or np.any(features < 0.0):
        raise ValueError("C2 features must be finite and nonnegative")
    return features


class InstrumentedC2Policy(BatchedPersistentPolicy):
    """C1-exact search with outcome-free depth-2 complexity counters."""

    def reset_c2_counters(self) -> None:
        self.c2_counts = Counter()
        self.c2_unique_values: set[tuple[Any, ...]] = set()
        self.c2_unique_afterstates: set[bytes] = set()
        self.c2_unique_base_moves: set[tuple[bytes, int]] = set()

    def _value(self, state: Any, sim: ThreesSim, depth: int) -> float:
        self.c2_counts["value_lookups"] += 1
        self.c2_unique_values.add(_state_key(state, depth))
        return super()._value(state, sim, depth)

    def _afterstate_value(self, board: np.ndarray) -> float:
        self.c2_counts["afterstate_lookups"] += 1
        self.c2_unique_afterstates.add(self._fast_board_key(board))
        return super()._afterstate_value(board)

    def _base_move(
        self,
        board: np.ndarray,
        action: int,
    ) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
        self.c2_counts["base_move_calls"] += 1
        self.c2_unique_base_moves.add((self._fast_board_key(board), int(action)))
        return super()._base_move(board, action)

    def _legal_actions(self, state: Any, sim: ThreesSim) -> tuple[int, ...]:
        self.c2_counts["legal_lookup_calls"] += 1
        return super()._legal_actions(state, sim)

    def _action_value(self, state: Any, sim: ThreesSim, action: int, depth: int) -> float:
        self.c2_counts["action_calls"] += 1
        return super()._action_value(state, sim, action, depth)

    def _transition_outcomes(
        self,
        state: Any,
        sim: ThreesSim,
        action: int,
        *,
        include_next_preview: bool,
    ):
        self.c2_counts["chance_calls"] += 1
        outcomes = super()._transition_outcomes(
            state,
            sim,
            action,
            include_next_preview=include_next_preview,
        )
        self.c2_counts["chance_outcomes"] += len(outcomes)
        return outcomes

    def depth2_probe(self, state: Any, sim: ThreesSim) -> dict[str, Any]:
        self.clear_decision_caches()
        self.reset_c2_counters()
        self.depth = 2
        self.chance_limit = None
        self.node_budget = 1_000_000_000
        self._action_cache.clear()
        self.expanded_value_nodes = 0
        self.budget_cutoffs = 0
        started = time.perf_counter()
        values = self._manual_root_values(state, sim, 2)
        elapsed = time.perf_counter() - started
        depth2_cache = dict(self._cache)
        counters = {
            **{key: int(value) for key, value in self.c2_counts.items()},
            "unique_value_states": len(self.c2_unique_values),
            "unique_afterstates": len(self.c2_unique_afterstates),
            "unique_base_moves": len(self.c2_unique_base_moves),
        }
        for key in (
            "action_calls",
            "value_lookups",
            "chance_calls",
            "chance_outcomes",
            "afterstate_lookups",
            "base_move_calls",
            "legal_lookup_calls",
        ):
            counters.setdefault(key, 0)
        return {
            "values": values,
            "elapsed_s": float(elapsed),
            "depth2_cache": depth2_cache,
            "counters": counters,
            "features": cost_features(state=state, values=values, counters=counters),
        }

    def exact_depth3_from_probe(
        self,
        state: Any,
        sim: ThreesSim,
        probe: Mapping[str, Any],
    ) -> dict[str, Any]:
        self.depth = 3
        self.chance_limit = CHANCE_LIMIT
        self.node_budget = NODE_BUDGET
        self.expanded_value_nodes = 0
        self.budget_cutoffs = 0
        self._cache = dict(probe["depth2_cache"])
        self._action_cache.clear()
        self.prefilled_value_keys = set(self._cache)
        self.reused_value_keys = set()
        started = time.perf_counter()
        values = self._manual_root_values(state, sim, 3)
        elapsed = time.perf_counter() - started
        effective_nodes = self.expanded_value_nodes + len(self.reused_value_keys)
        fallback = bool(self.budget_cutoffs or effective_nodes >= NODE_BUDGET)
        if fallback:
            self._cache.clear()
            self._action_cache.clear()
            self.prefilled_value_keys = set()
            self.reused_value_keys = set()
            self.expanded_value_nodes = 0
            self.budget_cutoffs = 0
            started = time.perf_counter()
            values = self._manual_root_values(state, sim, 3)
            elapsed = time.perf_counter() - started
            effective_nodes = self.expanded_value_nodes
        self.prefilled_value_keys = set()
        return {
            "values": values,
            "elapsed_s": float(elapsed),
            "combined_s": float(probe["elapsed_s"]) + float(elapsed),
            "expanded_value_nodes": int(self.expanded_value_nodes),
            "effective_value_nodes": int(effective_nodes),
            "budget_cutoffs": int(self.budget_cutoffs),
            "reference_fallback": bool(fallback),
        }

    def cost_admitted_values(
        self,
        state: Any,
        sim: ThreesSim,
        model: Mapping[str, Any],
    ) -> dict[str, Any]:
        started = time.perf_counter()
        probe = self.depth2_probe(state, sim)
        raw_prediction = predict_cost(model, probe["features"])
        upper = conservative_upper(raw_prediction)
        empties = int(np.count_nonzero(np.asarray(state.board) == 0))
        built_max = max_tile_excluding_initial_starter(state.board, STARTER_TILE)
        margin = normalized_margin(probe["values"])
        eligible = (
            milestone_for_built_max(built_max) is not None
            and (empties <= EMPTY_TRIGGER or margin <= MARGIN_TRIGGER)
        )
        admitted = bool(eligible and upper <= ADMISSION_THRESHOLD)
        if admitted:
            deep = self.exact_depth3_from_probe(state, sim, probe)
            values = deep["values"]
        else:
            deep = None
            values = probe["values"]
        elapsed = time.perf_counter() - started
        return {
            "values": values,
            "depth2_values": probe["values"],
            "features": probe["features"],
            "counters": probe["counters"],
            "prediction": float(raw_prediction),
            "upper_load": float(upper),
            "eligible": bool(eligible),
            "admitted": admitted,
            "elapsed_s": float(elapsed),
            "deep": deep,
        }


def clone_instrumented(base: NtupleExpectimaxPolicy) -> InstrumentedC2Policy:
    return InstrumentedC2Policy(
        base.checkpoint,
        depth=3,
        chance_limit=CHANCE_LIMIT,
        blend_specs=list(base.blend_specs),
        phase_blend_specs=list(base.phase_blend_specs),
        bonus_specs=list(base.bonus_specs),
        tie_margin=base.tie_margin,
        tie_breaker=base.tie_breaker,
        ensemble_mode=base.ensemble_mode,
        geometry_weight=base.geometry_weight,
        geometry_min_tile=base.geometry_min_tile,
        node_budget=NODE_BUDGET,
    )


def fit_cost_model(
    features: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
) -> dict[str, Any]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES):
        raise ValueError("C2 model feature width mismatch")
    if y.shape != (len(x),) or w.shape != (len(x),):
        raise ValueError("C2 model target/weight shape mismatch")
    if (
        not np.all(np.isfinite(x))
        or not np.all(np.isfinite(y))
        or not np.all(np.isfinite(w))
        or np.any(x < 0)
        or np.any(y < 0)
        or np.any(w <= 0)
    ):
        raise ValueError("C2 model inputs must be finite and nonnegative")
    design = np.column_stack([np.ones(len(x), dtype=np.float64), x])
    weighted_design = design * np.sqrt(w)[:, None]
    weighted_target = y * np.sqrt(w)
    ridge = np.sqrt(L2_LAMBDA) * np.eye(design.shape[1], dtype=np.float64)
    ridge[0, 0] = 0.0
    augmented_x = np.vstack([weighted_design, ridge])
    augmented_y = np.concatenate([weighted_target, np.zeros(design.shape[1])])
    result = lsq_linear(
        augmented_x,
        augmented_y,
        bounds=(0.0, np.inf),
        tol=LSQ_TOLERANCE,
        max_iter=LSQ_MAX_ITER,
        lsmr_tol=LSQ_TOLERANCE,
        verbose=0,
    )
    coefficients = np.asarray(result.x[1:], dtype=np.float64)
    intercept = float(result.x[0])
    predictions = intercept + x @ coefficients
    model = {
        "version": "c2_nonnegative_cost_model_v1",
        "feature_schema_sha256": feature_schema_payload()["schema_sha256"],
        "feature_names": list(FEATURE_NAMES),
        "intercept": intercept,
        "coefficients": coefficients.tolist(),
        "l2_lambda": L2_LAMBDA,
        "solver": "scipy.optimize.lsq_linear",
        "solver_success": bool(result.success),
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "solver_iterations": (
            None if result.nit is None else int(result.nit)
        ),
        "cost": float(result.cost),
        "optimality": float(result.optimality),
        "training_prediction_sha256": hashlib.sha256(
            np.asarray(predictions, dtype="<f8").tobytes()
        ).hexdigest(),
    }
    model["model_sha256"] = canonical_json_hash(model)
    return model


def validate_cost_model_payload(model: Mapping[str, Any]) -> None:
    if model.get("version") != "c2_nonnegative_cost_model_v1":
        raise ValueError("C2 model version mismatch")
    if model.get("feature_schema_sha256") != feature_schema_payload()["schema_sha256"]:
        raise ValueError("C2 model schema mismatch")
    if tuple(model.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("C2 model feature order mismatch")
    coefficients = np.asarray(model.get("coefficients"), dtype=np.float64)
    intercept = float(model.get("intercept"))
    if coefficients.shape != (len(FEATURE_NAMES),):
        raise ValueError("C2 model coefficient width mismatch")
    if (
        not np.all(np.isfinite(coefficients))
        or np.any(coefficients < 0)
        or not math.isfinite(intercept)
        or intercept < 0
    ):
        raise ValueError("C2 model parameters must be finite and nonnegative")
    unhashed = dict(model)
    expected = str(unhashed.pop("model_sha256", ""))
    if not expected or canonical_json_hash(unhashed) != expected:
        raise ValueError("C2 model hash mismatch")


def predict_cost(model: Mapping[str, Any], features: np.ndarray) -> float:
    validate_cost_model_payload(model)
    x = np.asarray(features, dtype=np.float64)
    if x.shape != (len(FEATURE_NAMES),) or not np.all(np.isfinite(x)):
        raise ValueError("C2 prediction feature mismatch")
    coefficients = np.asarray(model["coefficients"], dtype=np.float64)
    return float(max(0.0, float(model["intercept"]) + float(x @ coefficients)))


def conservative_upper(prediction: float) -> float:
    if not math.isfinite(prediction) or prediction < 0:
        raise ValueError("C2 cost prediction must be finite and nonnegative")
    return float(UPPER_MULTIPLIER * prediction + UPPER_OFFSET)


def safety_load(*, depth2_s: float, combined_s: float) -> float:
    if (
        not math.isfinite(depth2_s)
        or not math.isfinite(combined_s)
        or depth2_s <= 0
        or combined_s < 0
    ):
        raise ValueError("C2 timing target must be finite and positive")
    return float(max(
        combined_s / ABSOLUTE_LOAD_SECONDS,
        (combined_s / depth2_s) / RELATIVE_LOAD_RATIO,
    ))


def corpus_plan_payload(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    partitions = []
    cursor = 0
    for partition, count in ROOTS_PER_FAMILY.items():
        partitions.append({
            "partition": partition,
            "qualifying_root_indices_per_family": list(range(cursor, cursor + count)),
            "roots_per_family": count,
            "states_per_root": STATES_PER_ROOT,
        })
        cursor += count
    return payload_with_hash({
        "version": "c2_corpus_plan_v1",
        "family_order": [family for family, _spec in FAMILY_SLATE],
        "games_per_family": GAMES_PER_FAMILY,
        "total_games": TOTAL_GAMES,
        "starter_tile": STARTER_TILE,
        "max_moves": MAX_MOVES,
        "partitions": partitions,
        "required_qualifying_roots_per_family": cursor,
        "state_selection": {
            "higher_empty_screen_limit": 8,
            "higher_empty_hash_namespace": "C2-high-empty-screen-v1",
            "temporal_buckets": 4,
            "state_hash_namespace": "C2-state-v1",
            "states_per_root": STATES_PER_ROOT,
        },
        "requested_stream_rows_sha256": canonical_json_hash(list(rows)),
        "score_filtering": False,
        "future_outcome_filtering": False,
        "one_whole_game_per_ancestry": True,
        "partition_before_timing": True,
    })


def _immutable_source_lock() -> dict[str, Any]:
    rows = []
    for path in IMMUTABLE_SOURCE_PATHS:
        if not path.is_file():
            raise FileNotFoundError(f"Missing C2 immutable source: {path}")
        rows.append({
            "path": str(path),
            "sha256": sha256_path(path),
            "byte_size": path.stat().st_size,
        })
    expected = {
        "threes_rl/c1_search_optimization.py":
            "c12852cc7dcc8211d8ecc47ccf8c5598d6055a5f12a9bcec497dc47715e0e789",
        "threes_rl/r2a_adaptive_expectimax.py":
            "ece2a1fc34ea759168d2722ca3a82a212649de97b47400a27b2d0b2055d6d4f6",
        "threes_rl/runs/forensics/c1_search/C1_CORPUS.json":
            "a31ed8d151d41871eecfcb86d9967cd31e82350cee1539a9d0bb7846d7b218af",
        "threes_rl/runs/forensics/c1_search/C1_BYTEKEY_LEAF_BENCHMARK.json":
            "78d8d5bd9cb7c6a667e6b09be2bffe50749d3f0e108a0c723a6944ab0f75cf17",
        "threes_rl/runs/forensics/c1_search/C1_RUNTIME_GATE.json":
            "2f76415d097d47da2749be58dbf3a16dd22d30d7b0670e3e874361af744c1a0f",
        "threes_rl/current_incumbent_policy.txt":
            "d85a91576b8dc0ad80c2ed041dd1a0d62498eac9edb48445cb73233bb5454dd4",
    }
    by_path = {row["path"]: row["sha256"] for row in rows}
    checks = {
        path: by_path.get(path) == expected_hash
        for path, expected_hash in expected.items()
    }
    if not all(checks.values()):
        raise ValueError(f"C2 inherited lock mismatch: {checks}")
    payload = {
        "sources": rows,
        "source_count": len(rows),
        "source_manifest_sha256": canonical_json_hash(rows),
        "inherited_lock_checks": checks,
    }
    payload["source_lock_sha256"] = canonical_json_hash(payload)
    return payload


def _existing_state_roundtrip_fixture() -> dict[str, Any]:
    corpus = json.loads(
        Path("threes_rl/runs/forensics/c1_search/C1_CORPUS.json").read_text()
    )
    record = corpus["splits"]["profile"][0]
    payload = dict(record["state"])
    restored = state_from_replay_payload(payload)
    sim = ThreesSim.from_stream_ids(
        deck_stream_id=1,
        slot_stream_id=2,
        starter_tile=int(record.get("starter_tile", STARTER_TILE)),
    )
    reproduced = state_payload(restored, sim)
    return {
        "record_id": str(record["record_id"]),
        "state_sha256": canonical_json_hash(payload),
        "restored_sha256": canonical_json_hash(reproduced),
        "exact": reproduced == payload,
        "new_stream_consumed": False,
    }


def _load_test_evidence() -> dict[str, Any]:
    if not TEST_EVIDENCE_PATH.is_file():
        raise FileNotFoundError("C2 test evidence is missing")
    payload = json.loads(TEST_EVIDENCE_PATH.read_text())
    if not verify_payload_hash(payload):
        raise ValueError("C2 test evidence payload hash mismatch")
    if payload.get("implementation_sha256") != sha256_path(IMPLEMENTATION_PATH):
        raise ValueError("C2 test evidence implementation hash mismatch")
    if payload.get("test_sha256") != sha256_path(TEST_PATH):
        raise ValueError("C2 test evidence test hash mismatch")
    if payload.get("charter_sha256") != sha256_path(CHARTER_PATH):
        raise ValueError("C2 test evidence charter hash mismatch")
    if not payload.get("all_passed"):
        raise ValueError("C2 tests did not pass")
    return payload


def _preflight_commands(out_dir: Path) -> dict[str, str]:
    lock_path = out_dir / "C2_PREFLIGHT_LOCK.json"
    prefix = (
        "zsh -ic 'no-secrets nice -n 10 env PYTHONPATH=. "
        ".venv/bin/python -m threes_rl.c2_cost_admission"
    )
    suffix = (
        f" --out-dir {out_dir} --preflight-lock {lock_path} --jobs 1'"
    )
    return {
        "open": f"{prefix} open{suffix}",
        "execute": f"{prefix} execute{suffix}",
    }


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
    }
    return {
        "nice": history.current_nice(),
        "free_gib": disk,
        "heavy_process_audit": heavy,
        "service_health": services,
        "checks": checks,
        "passes": all(checks.values()),
    }


def run_preflight(*, out_dir: Path, jobs: int) -> dict[str, Any]:
    if out_dir.resolve() != OUTPUT_DIR.resolve():
        raise ValueError("C2 output directory does not match frozen namespace")
    if jobs != FROZEN_JOBS:
        raise ValueError("C2 preflight requires exactly one worker")
    if out_dir.exists():
        raise FileExistsError("C2 output directory must be fresh")
    staging = out_dir.with_name(f"{out_dir.name}.staging.{os.getpid()}")
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    try:
        rows = requested_stream_manifest()
        stream_payload = payload_with_hash({
            "version": "c2_stream_manifest_v1",
            "rows": rows,
            "row_count": len(rows),
            "row_manifest_sha256": canonical_json_hash(rows),
            "streams_consumed": False,
        })
        exclusion = build_exclusion_manifest()
        collision = build_stream_collision_manifest(rows, out_dir=staging)
        plan = corpus_plan_payload(rows)
        schema = payload_with_hash(feature_schema_payload())
        policy_lock, _loaded = build_policy_lock()
        family_audit = accepted_family_audit()
        source_lock = _immutable_source_lock()
        fixture = _existing_state_roundtrip_fixture()
        tests = _load_test_evidence()
        operations = _operational_audit(staging)
        projected_bytes = int(math.ceil(1.25 * (
            directory_bytes(Path("threes_rl/runs/forensics/g1r_acquisition/pilot_v2_qd5"))
            * (TOTAL_GAMES / 100.0)
            + 512 * 1024**2
        )))
        commands = _preflight_commands(out_dir)
        checks = {
            "fresh_output_namespace": not out_dir.exists(),
            "exact_three_family_order":
                [family for family, _spec in FAMILY_SLATE]
                == ["c2_corner2", "c2_parent_mc1000", "c2_replaycal"],
            "genuine_family_audit": family_audit["passes"],
            "policy_lock_complete": bool(policy_lock["policy_lock_sha256"]),
            "immutable_sources_locked": bool(source_lock["source_lock_sha256"]),
            "root_exclusions_collision_free": exclusion["passes"],
            "exact_216_stream_rows": len(rows) == TOTAL_GAMES,
            "stream_collision_free": collision["passes"],
            "corpus_plan_exact": plan["required_qualifying_roots_per_family"] == 12,
            "feature_schema_exact": schema["width"] == len(FEATURE_NAMES) == 20,
            "existing_state_roundtrip": fixture["exact"],
            "tests_passed_and_bound": bool(tests["all_passed"]),
            "projected_storage_below_4_gib": projected_bytes < BYTE_LIMIT,
            "operations_pass": operations["passes"],
            "zero_c2_games": True,
            "zero_c2_timings": True,
            "zero_exact_depth3_results": True,
            "zero_models": True,
            "zero_admissions": True,
            "zero_policy_outcomes": True,
            "zero_score_inspection": True,
        }
        decision = (
            "READY_C2_ENGINEERING_EXECUTION"
            if all(checks.values())
            else "HOLD_C2_ENGINEERING_FAULT"
        )
        named_payloads = {
            "C2_STREAM_MANIFEST.json": stream_payload,
            "C2_EXCLUSION_MANIFEST.json": exclusion,
            "C2_COLLISION_SOURCE_MANIFEST.json": collision,
            "C2_CORPUS_PLAN.json": plan,
            "C2_FEATURE_SCHEMA.json": schema,
            "C2_POLICY_LOCK.json": payload_with_hash(policy_lock),
        }
        for name, payload in named_payloads.items():
            atomic_write_json(staging / name, payload)
        manifest_bindings = {
            name: {
                "file_sha256": sha256_path(staging / name),
                "payload_sha256": payload["canonical_payload_sha256"],
            }
            for name, payload in named_payloads.items()
        }
        lock = payload_with_hash({
            "version": "c2_cost_admission_preflight_v1",
            "decision": decision,
            "bound_out_dir": str(out_dir.resolve()),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "charter": {
                "path": str(CHARTER_PATH),
                "sha256": sha256_path(CHARTER_PATH),
            },
            "implementation": {
                "path": str(IMPLEMENTATION_PATH),
                "sha256": sha256_path(IMPLEMENTATION_PATH),
            },
            "tests": {
                "path": str(TEST_PATH),
                "sha256": sha256_path(TEST_PATH),
                "evidence_path": str(TEST_EVIDENCE_PATH),
                "evidence_file_sha256": sha256_path(TEST_EVIDENCE_PATH),
                "evidence_payload_sha256": tests["canonical_payload_sha256"],
            },
            "manifest_bindings": manifest_bindings,
            "policy_lock_sha256": policy_lock["policy_lock_sha256"],
            "source_lock": source_lock,
            "family_audit": family_audit,
            "state_roundtrip_fixture": fixture,
            "feature_schema_sha256": schema["schema_sha256"],
            "commands": commands,
            "resources": {
                "jobs": FROZEN_JOBS,
                "minimum_nice": MINIMUM_NICE,
                "active_wall_seconds": ACTIVE_WALL_SECONDS,
                "byte_limit": BYTE_LIMIT,
                "min_free_gib": MIN_FREE_GIB,
                "target_free_gib": TARGET_FREE_GIB,
                "max_chunk_size": MAX_CHUNK_SIZE,
                "projected_bytes": projected_bytes,
            },
            "operations": operations,
            "checks": checks,
            "zero_work": {
                "games": 0,
                "streams_consumed": 0,
                "timings": 0,
                "depth3_results": 0,
                "models": 0,
                "admissions": 0,
                "policy_outcomes": 0,
                "scores_inspected": 0,
            },
            "execution_authorized_by_preflight": decision
            == "READY_C2_ENGINEERING_EXECUTION",
            "dashboard_eligible": False,
            "promotable": False,
        })
        atomic_write_json(staging / "C2_PREFLIGHT_LOCK.json", lock)
        if decision != "READY_C2_ENGINEERING_EXECUTION":
            return {
                "decision": decision,
                "staging": str(staging),
                "lock_file_sha256": sha256_path(staging / "C2_PREFLIGHT_LOCK.json"),
                "lock_payload_sha256": lock["canonical_payload_sha256"],
                "checks": checks,
            }
        os.replace(staging, out_dir)
        return {
            "decision": decision,
            "out_dir": str(out_dir),
            "lock_path": str(out_dir / "C2_PREFLIGHT_LOCK.json"),
            "lock_file_sha256": sha256_path(out_dir / "C2_PREFLIGHT_LOCK.json"),
            "lock_payload_sha256": lock["canonical_payload_sha256"],
            "manifest_bindings": manifest_bindings,
            "policy_lock_sha256": policy_lock["policy_lock_sha256"],
            "source_lock_sha256": source_lock["source_lock_sha256"],
            "feature_schema_sha256": schema["schema_sha256"],
            "stream_rows": len(rows),
            "historical_source_count": collision["immutable_source_count"],
            "historical_source_sha256": collision["immutable_inventory_sha256"],
            "operations": operations,
            "zero_work": lock["zero_work"],
        }
    except Exception as error:
        failure = payload_with_hash({
            "version": "c2_preflight_failure_v1",
            "decision": "HOLD_C2_ENGINEERING_FAULT",
            "stage": "preflight",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "zero_work": {
                "games": 0,
                "streams_consumed": 0,
                "timings": 0,
                "depth3_results": 0,
                "models": 0,
                "admissions": 0,
                "policy_outcomes": 0,
                "scores_inspected": 0,
            },
        })
        atomic_write_json(staging / "PREFLIGHT_FAILURE.json", failure)
        raise


def _load_preflight(lock_path: Path, out_dir: Path) -> dict[str, Any]:
    if out_dir.resolve() != OUTPUT_DIR.resolve():
        raise ValueError("C2 output directory mismatch")
    if lock_path.resolve() != (out_dir / "C2_PREFLIGHT_LOCK.json").resolve():
        raise ValueError("C2 preflight lock path mismatch")
    lock = json.loads(lock_path.read_text())
    if not verify_payload_hash(lock):
        raise ValueError("C2 preflight payload hash mismatch")
    if lock.get("decision") != "READY_C2_ENGINEERING_EXECUTION":
        raise ValueError("C2 preflight is not READY")
    if Path(str(lock["bound_out_dir"])).resolve() != out_dir.resolve():
        raise ValueError("C2 lock is bound to another output directory")
    exact_paths = {
        CHARTER_PATH: lock["charter"]["sha256"],
        IMPLEMENTATION_PATH: lock["implementation"]["sha256"],
        TEST_PATH: lock["tests"]["sha256"],
        TEST_EVIDENCE_PATH: lock["tests"]["evidence_file_sha256"],
    }
    for path, expected in exact_paths.items():
        if sha256_path(path) != expected:
            raise ValueError(f"C2 bound source changed: {path}")
    for name, binding in lock["manifest_bindings"].items():
        path = out_dir / name
        if sha256_path(path) != binding["file_sha256"]:
            raise ValueError(f"C2 bound manifest changed: {name}")
        payload = json.loads(path.read_text())
        if (
            not verify_payload_hash(payload)
            or payload["canonical_payload_sha256"] != binding["payload_sha256"]
        ):
            raise ValueError(f"C2 bound manifest payload changed: {name}")
    source_lock = _immutable_source_lock()
    if source_lock != lock["source_lock"]:
        raise ValueError("C2 immutable source lock changed")
    policy_payload = json.loads((out_dir / "C2_POLICY_LOCK.json").read_text())
    policy_unhashed = dict(policy_payload)
    policy_unhashed.pop("canonical_payload_sha256", None)
    policy_check = revalidate_policy_lock(policy_unhashed)
    if not policy_check["exact"]:
        raise ValueError("C2 policy artifacts changed")
    collision_payload = json.loads(
        (out_dir / "C2_COLLISION_SOURCE_MANIFEST.json").read_text()
    )
    stream_payload = json.loads((out_dir / "C2_STREAM_MANIFEST.json").read_text())
    collision = revalidate_stream_collision_manifest(
        collision_payload,
        stream_payload["rows"],
        out_dir=out_dir,
    )
    if not collision["passes"]:
        raise ValueError(f"C2 stream collision revalidation failed: {collision['checks']}")
    return lock


def _zero_execution_work(out_dir: Path) -> dict[str, Any]:
    paths = {
        "marker": out_dir / "C2_EXECUTION_OPENED.json",
        "terminal": out_dir / "C2_TERMINAL_RESULT.json",
        "completed_games": out_dir / "completed_games.jsonl",
        "corpus_manifest": out_dir / "C2_CORPUS_MANIFEST.json",
        "fit_timings": out_dir / "fit_timings.jsonl",
        "validation_timings": out_dir / "validation_timings.jsonl",
        "cost_model": out_dir / "C2_COST_MODEL.json",
        "validation_report": out_dir / "C2_VALIDATION_REPORT.json",
        "gate_timings": out_dir / "gate_timings.jsonl",
    }
    return {
        "existing": {
            name: path.exists() for name, path in paths.items()
        },
        "passes": not any(path.exists() for path in paths.values()),
    }


def seal_execution_opened(
    *,
    out_dir: Path,
    preflight_lock: Path,
    jobs: int,
) -> dict[str, Any]:
    if jobs != FROZEN_JOBS:
        raise ValueError("C2 open requires exactly one worker")
    lock = _load_preflight(preflight_lock, out_dir)
    marker_path = out_dir / "C2_EXECUTION_OPENED.json"
    terminal_path = out_dir / "C2_TERMINAL_RESULT.json"
    if marker_path.exists() or terminal_path.exists():
        raise FileExistsError("C2 execution has already opened or terminated")
    zero_work = _zero_execution_work(out_dir)
    if not zero_work["passes"]:
        raise ValueError(f"C2 open found prior execution work: {zero_work}")
    operations = _operational_audit(out_dir)
    if not operations["passes"]:
        raise EngineeringFault(f"C2 open operational audit failed: {operations['checks']}")
    marker = payload_with_hash({
        "version": "c2_execution_opened_v1",
        "admission_opened": True,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "preflight_lock_path": str(preflight_lock),
        "preflight_lock_file_sha256": sha256_path(preflight_lock),
        "preflight_lock_payload_sha256": lock["canonical_payload_sha256"],
        "charter_sha256": lock["charter"]["sha256"],
        "implementation_sha256": lock["implementation"]["sha256"],
        "test_sha256": lock["tests"]["sha256"],
        "test_evidence_file_sha256": lock["tests"]["evidence_file_sha256"],
        "manifest_bindings": lock["manifest_bindings"],
        "source_lock_sha256": lock["source_lock"]["source_lock_sha256"],
        "policy_lock_sha256": lock["policy_lock_sha256"],
        "feature_schema_sha256": lock["feature_schema_sha256"],
        "execute_command": lock["commands"]["execute"],
        "jobs": FROZEN_JOBS,
        "minimum_nice": MINIMUM_NICE,
        "resource_limits": lock["resources"],
        "operations": operations,
        "zero_work_before_marker": {
            "games": 0,
            "streams_consumed": 0,
            "timings": 0,
            "depth3_results": 0,
            "models": 0,
            "admissions": 0,
            "policy_outcomes": 0,
            "scores_inspected": 0,
        },
        "dashboard_eligible": False,
        "promotable": False,
    })
    atomic_write_json(marker_path, marker)
    return {
        "decision": "C2_EXECUTION_OPENED",
        "marker_path": str(marker_path),
        "marker_file_sha256": sha256_path(marker_path),
        "marker_payload_sha256": marker["canonical_payload_sha256"],
        "execute_command": lock["commands"]["execute"],
        "zero_work": marker["zero_work_before_marker"],
        "operations": operations,
    }


def _load_marker(out_dir: Path, lock: Mapping[str, Any]) -> dict[str, Any]:
    path = out_dir / "C2_EXECUTION_OPENED.json"
    if not path.is_file():
        raise ValueError("C2 execution marker is missing")
    marker = json.loads(path.read_text())
    if not verify_payload_hash(marker):
        raise ValueError("C2 marker payload hash mismatch")
    checks = {
        "opened": marker.get("admission_opened") is True,
        "preflight_file":
            marker.get("preflight_lock_file_sha256")
            == sha256_path(out_dir / "C2_PREFLIGHT_LOCK.json"),
        "preflight_payload":
            marker.get("preflight_lock_payload_sha256")
            == lock["canonical_payload_sha256"],
        "charter": marker.get("charter_sha256") == lock["charter"]["sha256"],
        "implementation":
            marker.get("implementation_sha256")
            == lock["implementation"]["sha256"],
        "tests": marker.get("test_sha256") == lock["tests"]["sha256"],
        "execute_command":
            marker.get("execute_command") == lock["commands"]["execute"],
        "jobs": int(marker.get("jobs", -1)) == FROZEN_JOBS,
    }
    if not all(checks.values()):
        raise ValueError(f"C2 marker mismatch: {checks}")
    return marker


def _load_completed_games(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    rows = {}
    if not path.is_file():
        return rows
    with path.open() as handle:
        for line in handle:
            payload = json.loads(line)
            key = (str(payload["behavior_family"]), int(payload["game_index"]))
            if key in rows:
                raise ValueError(f"Duplicate C2 completion row: {key}")
            rows[key] = payload
    return rows


def _state_hash(payload: Mapping[str, Any]) -> str:
    return canonical_json_hash(payload)


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
        raise ValueError("C2 state failed exact round trip")
    return state, sim


def _frame_base_candidate(
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
    built_max = max_tile_excluding_initial_starter(state.board, STARTER_TILE)
    if milestone_for_built_max(built_max) is None:
        return None
    frame_index = int(frame.get("index", -1))
    state_sha = _state_hash(payload)
    return {
        "root_ancestry": root,
        "frame_index": frame_index,
        "state": payload,
        "state_sha256": state_sha,
        "empty_count": int(np.count_nonzero(state.board == 0)),
        "built_max": int(built_max),
        "legal_actions": [DIRECTION_NAMES[action] for action in legal],
    }


def _incumbent_values_for_candidate(
    candidate: Mapping[str, Any],
    incumbent: NtupleExpectimaxPolicy,
) -> dict[str, Any]:
    state, sim = _state_sim(candidate["state"])
    values = incumbent.action_values(state, sim)
    if not values or not all(math.isfinite(float(value)) for _action, value in values):
        raise ValueError("C2 incumbent values are missing or nonfinite")
    margin = normalized_margin(values)
    return {
        "incumbent_margin": float(margin),
        "trigger_reasons": {
            "low_empty": int(candidate["empty_count"]) <= EMPTY_TRIGGER,
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
    base_candidates = [
        candidate
        for frame in replay.get("frames", [])
        if (
            candidate := _frame_base_candidate(frame, root=root)
        ) is not None
    ]
    low_empty = [
        row for row in base_candidates
        if int(row["empty_count"]) <= EMPTY_TRIGGER
    ]
    high_empty = sorted(
        (
            row for row in base_candidates
            if int(row["empty_count"]) > EMPTY_TRIGGER
        ),
        key=lambda row: (
            hashlib.sha256(
                (
                    "C2-high-empty-screen-v1|"
                    f"{root}|{row['frame_index']}|{row['state_sha256']}"
                ).encode()
            ).hexdigest(),
            int(row["frame_index"]),
        ),
    )[:8]
    eligible = [dict(row) for row in low_empty]
    for row in high_empty:
        incumbent_metadata = _incumbent_values_for_candidate(row, incumbent)
        if float(incumbent_metadata["incumbent_margin"]) <= MARGIN_TRIGGER:
            eligible.append({**row, **incumbent_metadata})
    eligible.sort(key=lambda row: int(row["frame_index"]))
    if len(eligible) < STATES_PER_ROOT:
        return []
    buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(eligible):
        bucket = min(3, (STATES_PER_ROOT * index) // len(eligible))
        buckets[bucket].append(row)
    if set(buckets) != set(range(STATES_PER_ROOT)):
        raise ValueError("C2 temporal state buckets are incomplete")
    selected = []
    for bucket in range(STATES_PER_ROOT):
        row = min(
            buckets[bucket],
            key=lambda item: (
                hashlib.sha256(
                    (
                        "C2-state-v1|"
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
            else _incumbent_values_for_candidate(row, incumbent)
        )
        if not any(metadata["trigger_reasons"].values()):
            raise AssertionError("C2 selected state is not trigger-eligible")
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
    if len({row["frame_index"] for row in selected}) != STATES_PER_ROOT:
        raise ValueError("C2 selected duplicate frames")
    return selected


def _process_game_output(
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
        raise EngineeringFault("C2 replay capture unexpectedly missing")
    logical_seed = int(stream_row["logical_seed"])
    replay.update(direct_root_fields(
        origin=ORIGIN_FRESH,
        seed=logical_seed,
        policy=family,
        first_score=int(replay["frames"][0]["state"]["score"]),
    ))
    replay["behavior_family"] = family
    replay["acquisition_policy_spec"] = spec
    replay["dashboard_eligible"] = False
    # replay_provenance owns the reset invariant; import locally to avoid
    # broadening the public acquisition surface.
    from threes_rl.replay_provenance import initial_reset_diagnostics
    reset = initial_reset_diagnostics(replay)
    if not reset["is_reset_start"]:
        raise EngineeringFault(f"C2 fresh reset failed: {reset}")
    selected = []
    complete = bool(replay.get("game_over", False))
    if complete and retained_count < sum(ROOTS_PER_FAMILY.values()):
        selected = extract_selected_states(
            replay,
            family=family,
            stream_row=stream_row,
            incumbent=incumbent,
        )
    replay_path = replay_dir / (
        f"{family}_game_{int(stream_row['game_index']):05d}_"
        f"seed_{logical_seed}.json"
    )
    state_path = state_dir / (
        f"{family}_game_{int(stream_row['game_index']):05d}_"
        f"seed_{logical_seed}.json"
    )
    if selected:
        atomic_write_json(replay_path, dict(replay))
        replay_sha = sha256_path(replay_path)
        for row in selected:
            row["source_replay"] = str(replay_path)
            row["source_replay_sha256"] = replay_sha
        state_payload_record = payload_with_hash({
            "version": "c2_selected_root_states_v1",
            "root_ancestry": _root_ancestry(logical_seed),
            "behavior_family": family,
            "game_index": int(stream_row["game_index"]),
            "source_replay": str(replay_path),
            "source_replay_sha256": replay_sha,
            "states": selected,
        })
        atomic_write_json(state_path, state_payload_record)
    return {
        "behavior_family": family,
        "family_index": int(stream_row["family_index"]),
        "game_index": int(stream_row["game_index"]),
        "policy_spec_sha256": hashlib.sha256(spec.encode()).hexdigest(),
        "logical_seed": logical_seed,
        "deck_stream_id": int(stream_row["deck_stream_id"]),
        "slot_stream_id": int(stream_row["slot_stream_id"]),
        "policy_stream_id": int(stream_row["policy_stream_id"]),
        "root_ancestry": _root_ancestry(logical_seed),
        "complete": complete,
        "move_count": int(output.result.moves),
        "qualifying_root": bool(selected),
        "selected_state_count": len(selected),
        "source_replay": str(replay_path) if selected else None,
        "source_replay_sha256": sha256_path(replay_path) if selected else None,
        "selected_states": str(state_path) if selected else None,
        "selected_states_sha256": sha256_path(state_path) if selected else None,
        "score_inspected": False,
        "dashboard_eligible": False,
    }


def _runtime_state(path: Path) -> dict[str, Any]:
    if path.is_file():
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise ValueError("C2 runtime state is malformed")
        return payload
    return {
        "active_runtime_seconds": 0.0,
        "chunks_completed": 0,
        "phase": "acquisition",
    }


def _execution_guard(
    *,
    out_dir: Path,
    runtime: Mapping[str, Any],
    check_contention: bool = True,
) -> dict[str, Any]:
    disk = free_gib(out_dir)
    used = directory_bytes(out_dir)
    services = history.service_health()
    heavy = _heavy_process_audit() if check_contention else {"passes": True}
    checks = {
        "nice_at_least_10": history.current_nice() >= MINIMUM_NICE,
        "disk_above_100_gib": disk >= MIN_FREE_GIB,
        "output_below_4_gib": used < BYTE_LIMIT,
        "active_wall_below_limit":
            float(runtime.get("active_runtime_seconds", 0.0))
            < ACTIVE_WALL_SECONDS,
        "services_healthy": bool(services["passes"]),
        "no_competing_heavy_process": bool(heavy["passes"]),
    }
    if not all(checks.values()):
        raise EngineeringFault(f"C2 execution guard failed: {checks}")
    return {
        "checks": checks,
        "passes": True,
        "free_gib": disk,
        "output_bytes": used,
        "services": services,
        "heavy_process_audit": heavy,
    }


def _acquisition_order(rows: Sequence[Mapping[str, Any]]) -> list[list[dict[str, Any]]]:
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


def run_fresh_acquisition(
    *,
    out_dir: Path,
    lock: Mapping[str, Any],
    jobs: int,
) -> dict[str, Any]:
    stream_manifest = json.loads(
        (out_dir / "C2_STREAM_MANIFEST.json").read_text()
    )
    rows = [dict(row) for row in stream_manifest["rows"]]
    collision_manifest = json.loads(
        (out_dir / "C2_COLLISION_SOURCE_MANIFEST.json").read_text()
    )
    collision = revalidate_stream_collision_manifest(
        collision_manifest,
        rows,
        out_dir=out_dir,
    )
    if not collision["passes"]:
        raise EngineeringFault("C2 stream collision appeared before acquisition")
    completed_path = out_dir / "completed_games.jsonl"
    runtime_path = out_dir / "runtime_state.json"
    replay_dir = out_dir / "source_replays"
    state_dir = out_dir / "selected_states"
    replay_dir.mkdir(exist_ok=True)
    state_dir.mkdir(exist_ok=True)
    completed = _load_completed_games(completed_path)
    runtime = _runtime_state(runtime_path)
    policy_lock_payload = json.loads((out_dir / "C2_POLICY_LOCK.json").read_text())
    raw_policy_lock = dict(policy_lock_payload)
    raw_policy_lock.pop("canonical_payload_sha256", None)
    policy_check = revalidate_policy_lock(raw_policy_lock)
    if not policy_check["exact"]:
        raise EngineeringFault("C2 acquisition policy lock changed")
    incumbent = make_policy(incumbent_spec())
    if not isinstance(incumbent, NtupleExpectimaxPolicy):
        raise EngineeringFault("C2 incumbent is not n-tuple expectimax")
    policies = {
        family: make_policy(spec) for family, spec in FAMILY_SLATE
    }
    specs = dict(FAMILY_SLATE)
    for chunk in _acquisition_order(rows):
        pending = [
            row for row in chunk
            if (str(row["behavior_family"]), int(row["game_index"]))
            not in completed
        ]
        if not pending:
            continue
        _execution_guard(out_dir=out_dir, runtime=runtime)
        family = str(pending[0]["behavior_family"])
        if any(str(row["behavior_family"]) != family for row in pending):
            raise AssertionError("C2 acquisition chunk crossed families")
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
            stream_row = pending[int(output.index)]
            retained_count = sum(
                bool(row["qualifying_root"])
                for (row_family, _game_index), row in completed.items()
                if row_family == family
            )
            completion = _process_game_output(
                output=output,
                stream_row=stream_row,
                family=family,
                spec=specs[family],
                incumbent=incumbent,
                retained_count=retained_count,
                replay_dir=replay_dir,
                state_dir=state_dir,
            )
            append_jsonl(completed_path, completion)
            completed[(family, int(stream_row["game_index"]))] = completion
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
        raise EngineeringFault(
            f"C2 acquisition incomplete: {len(completed)} != {len(expected)}"
        )
    ordered = [completed[key] for key in sorted(expected)]
    return {
        "completed": completed,
        "rows": ordered,
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


def _partition_for_qualifying_index(index: int) -> str:
    cursor = 0
    for partition, count in ROOTS_PER_FAMILY.items():
        if cursor <= index < cursor + count:
            return partition
        cursor += count
    raise ValueError(f"C2 qualifying-root index out of range: {index}")


def build_corpus_manifest(
    *,
    out_dir: Path,
    acquisition: Mapping[str, Any],
) -> dict[str, Any]:
    exclusion = json.loads((out_dir / "C2_EXCLUSION_MANIFEST.json").read_text())
    if not verify_payload_hash(exclusion):
        raise EngineeringFault("C2 exclusion manifest hash mismatch")
    prior_roots = set(str(root) for root in exclusion["root_tokens"])
    selected_roots = []
    selected_states = []
    source_failures = []
    for family, _spec in FAMILY_SLATE:
        qualifying = sorted(
            (
                row for row in acquisition["rows"]
                if row["behavior_family"] == family
                and bool(row["qualifying_root"])
            ),
            key=lambda row: int(row["game_index"]),
        )
        if len(qualifying) != sum(ROOTS_PER_FAMILY.values()):
            continue
        for index, completion in enumerate(qualifying):
            partition = _partition_for_qualifying_index(index)
            replay_path = Path(str(completion["source_replay"]))
            state_path = Path(str(completion["selected_states"]))
            if (
                not replay_path.is_file()
                or not state_path.is_file()
                or sha256_path(replay_path) != completion["source_replay_sha256"]
                or sha256_path(state_path) != completion["selected_states_sha256"]
            ):
                source_failures.append({
                    "root": completion["root_ancestry"],
                    "reason": "missing_or_hash_changed",
                })
                continue
            replay = json.loads(replay_path.read_text())
            source = json.loads(state_path.read_text())
            if not verify_payload_hash(source):
                source_failures.append({
                    "root": completion["root_ancestry"],
                    "reason": "selected_state_payload_hash",
                })
                continue
            if (
                str(replay.get("root_origin")) != ORIGIN_FRESH
                or int(replay.get("root_seed")) != int(completion["logical_seed"])
                or int(replay.get("root_frame_index")) != 0
                or str(completion["root_ancestry"]) in prior_roots
            ):
                source_failures.append({
                    "root": completion["root_ancestry"],
                    "reason": "root_provenance_or_overlap",
                })
                continue
            frames = {
                int(frame["index"]): frame["state"]
                for frame in replay["frames"]
                if isinstance(frame, dict) and isinstance(frame.get("state"), dict)
            }
            if len(source["states"]) != STATES_PER_ROOT:
                source_failures.append({
                    "root": completion["root_ancestry"],
                    "reason": "state_count",
                })
                continue
            root_states = []
            for row in source["states"]:
                frame_index = int(row["frame_index"])
                if (
                    frame_index not in frames
                    or frames[frame_index] != row["state"]
                    or _state_hash(row["state"]) != row["state_sha256"]
                ):
                    source_failures.append({
                        "root": completion["root_ancestry"],
                        "frame_index": frame_index,
                        "reason": "frame_or_state_mismatch",
                    })
                    break
                _state_sim(row["state"])
                record_id = hashlib.sha256(
                    (
                        "C2-record-v1|"
                        f"{completion['root_ancestry']}|{frame_index}|"
                        f"{row['state_sha256']}"
                    ).encode()
                ).hexdigest()[:24]
                compact = {
                    **row,
                    "record_id": record_id,
                    "partition": partition,
                    "source_replay": str(replay_path),
                    "source_replay_sha256": sha256_path(replay_path),
                    "selected_states_source": str(state_path),
                    "selected_states_source_sha256": sha256_path(state_path),
                }
                root_states.append(compact)
            else:
                selected_states.extend(root_states)
                selected_roots.append({
                    "root_ancestry": completion["root_ancestry"],
                    "behavior_family": family,
                    "game_index": int(completion["game_index"]),
                    "partition": partition,
                    "source_replay": str(replay_path),
                    "source_replay_sha256": sha256_path(replay_path),
                    "state_count": len(root_states),
                })
    partition_roots = {
        partition: [
            row for row in selected_roots if row["partition"] == partition
        ]
        for partition in ROOTS_PER_FAMILY
    }
    partition_states = {
        partition: [
            row for row in selected_states if row["partition"] == partition
        ]
        for partition in ROOTS_PER_FAMILY
    }
    all_roots = [str(row["root_ancestry"]) for row in selected_roots]
    checks = {
        "all_216_games_complete":
            len(acquisition["rows"]) == TOTAL_GAMES
            and bool(acquisition["all_complete"]),
        "exact_12_qualifying_roots_each_family":
            all(
                acquisition["qualifying_by_family"].get(family) == 12
                for family, _spec in FAMILY_SLATE
            ),
        "source_integrity": not source_failures,
        "exact_36_unique_roots":
            len(all_roots) == 36 and len(set(all_roots)) == 36,
        "exact_144_states": len(selected_states) == 144,
        "exact_partition_root_counts":
            {partition: len(rows) for partition, rows in partition_roots.items()}
            == {
                "cost_fit": 18,
                "engineering_validation": 6,
                "untouched_runtime_gate": 12,
            },
        "exact_partition_state_counts":
            {partition: len(rows) for partition, rows in partition_states.items()}
            == {
                "cost_fit": 72,
                "engineering_validation": 24,
                "untouched_runtime_gate": 48,
            },
        "each_partition_three_families": all(
            len({row["behavior_family"] for row in rows}) == 3
            for rows in partition_roots.values()
        ),
        "gate_max_family_share_le_40pct": max(
            Counter(
                row["behavior_family"]
                for row in partition_roots["untouched_runtime_gate"]
            ).values(),
            default=0,
        ) / max(1, len(partition_roots["untouched_runtime_gate"])) <= 0.40,
        "zero_prior_root_overlap": not (set(all_roots) & prior_roots),
        "zero_cross_partition_root_overlap": len(set(all_roots)) == len(all_roots),
        "one_partition_per_root": all(
            len({
                row["partition"] for row in selected_states
                if row["root_ancestry"] == root
            }) == 1
            for root in all_roots
        ),
    }
    decision = "C2_CORPUS_READY" if all(checks.values()) else "C2_CORPUS_KILL"
    return payload_with_hash({
        "version": "c2_fresh_corpus_v1",
        "decision": decision,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "checks": checks,
        "source_failures": source_failures,
        "qualifying_by_family": acquisition["qualifying_by_family"],
        "roots": selected_roots,
        "states": selected_states,
        "root_manifest_sha256": canonical_json_hash(selected_roots),
        "state_manifest_sha256": canonical_json_hash(selected_states),
        "partition_summary": {
            partition: {
                "roots": len(partition_roots[partition]),
                "states": len(partition_states[partition]),
                "families": dict(Counter(
                    row["behavior_family"]
                    for row in partition_roots[partition]
                )),
            }
            for partition in ROOTS_PER_FAMILY
        },
        "score_inspected": False,
        "policy_outcomes_compared": False,
        "dashboard_eligible": False,
    })


def _selection_seed(record: Mapping[str, Any]) -> int:
    digest = hashlib.sha256(
        f"C2-selection-v1|{record['record_id']}".encode()
    ).hexdigest()
    return int(digest[:16], 16) & ((1 << 63) - 1)


def _values_hash(values: Sequence[tuple[int, float]]) -> str:
    payload = [
        [int(action), float(value)] for action, value in values
    ]
    return canonical_json_hash(payload)


def _median(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("C2 median requires values")
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _measure_cost_state(
    *,
    record: Mapping[str, Any],
    state_index: int,
    base: NtupleExpectimaxPolicy,
    oracle: BatchedPersistentPolicy,
    instrumented: InstrumentedC2Policy,
) -> dict[str, Any]:
    state, sim = _state_sim(record["state"])
    probe = instrumented.depth2_probe(state, sim)
    plain_warm = base.action_values(state, sim)
    depth2_close, depth2_difference = _values_close(
        {DIRECTION_NAMES[action]: value for action, value in plain_warm},
        probe["values"],
    )
    oracle.clear_decision_caches()
    oracle_warm = oracle.adaptive_values(state, sim)
    oracle_depth2_close, oracle_depth2_difference = _values_close(
        {DIRECTION_NAMES[action]: value for action, value in plain_warm},
        oracle_warm["depth2"],
    )
    depth2_times = []
    combined_times = []
    internal_combined_times = []
    order_rows = []
    for repeat in range(TIMED_REPEATS):
        results: dict[str, Any] = {}

        def run_depth2() -> None:
            started = time.perf_counter()
            results["depth2_values"] = base.action_values(state, sim)
            results["depth2_s"] = time.perf_counter() - started

        def run_oracle() -> None:
            oracle.clear_decision_caches()
            started = time.perf_counter()
            results["oracle"] = oracle.adaptive_values(state, sim)
            results["combined_s"] = time.perf_counter() - started

        order = (
            (run_depth2, run_oracle)
            if (state_index + repeat) % 2 == 0
            else (run_oracle, run_depth2)
        )
        for operation in order:
            operation()
        plain_close, _plain_difference = _values_close(
            {
                DIRECTION_NAMES[action]: value
                for action, value in results["depth2_values"]
            },
            results["oracle"]["depth2"],
        )
        if not plain_close:
            raise EngineeringFault("C2 timing oracle depth2 mismatch")
        depth2_times.append(float(results["depth2_s"]))
        combined_times.append(float(results["combined_s"]))
        internal_combined_times.append(float(results["oracle"]["combined_s"]))
        order_rows.append([operation.__name__ for operation in order])
    depth2_s = _median(depth2_times)
    combined_s = _median(combined_times)
    ratio = combined_s / max(depth2_s, 1e-12)
    seed = _selection_seed(record)
    plain_action = choose_action(base, plain_warm, seed)
    probe_action = choose_action(instrumented, probe["values"], seed)
    return {
        "record_id": str(record["record_id"]),
        "root_ancestry": str(record["root_ancestry"]),
        "behavior_family": str(record["behavior_family"]),
        "partition": str(record["partition"]),
        "state_sha256": str(record["state_sha256"]),
        "state_index": int(state_index),
        "feature_schema_sha256": feature_schema_payload()["schema_sha256"],
        "features": np.asarray(probe["features"], dtype=np.float64).tolist(),
        "counters": probe["counters"],
        "depth2_s_repeats": depth2_times,
        "exact_c1_combined_s_repeats": combined_times,
        "exact_c1_internal_combined_s_repeats": internal_combined_times,
        "interleave_orders": order_rows,
        "depth2_s": depth2_s,
        "exact_c1_combined_s": combined_s,
        "exact_c1_over_depth2": float(ratio),
        "safety_load": safety_load(depth2_s=depth2_s, combined_s=combined_s),
        "depth2_values_exact": bool(depth2_close and oracle_depth2_close),
        "depth2_max_abs_difference": float(max(
            depth2_difference, oracle_depth2_difference
        )),
        "depth2_action_match": plain_action == probe_action,
        "plain_value_hash": _values_hash(plain_warm),
        "probe_value_hash": _values_hash(probe["values"]),
        "oracle_depth3_value_hash": _values_hash(oracle_warm["depth3"]),
        "score_inspected": False,
    }


def _load_jsonl_by_record(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    if not path.is_file():
        return rows
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            record_id = str(row["record_id"])
            if record_id in rows:
                raise ValueError(f"Duplicate C2 timing row: {record_id}")
            rows[record_id] = row
    return rows


def measure_cost_partition(
    *,
    records: Sequence[Mapping[str, Any]],
    path: Path,
    runtime: dict[str, Any],
    out_dir: Path,
) -> list[dict[str, Any]]:
    existing = _load_jsonl_by_record(path)
    template = make_policy(incumbent_spec())
    base = make_policy(incumbent_spec())
    if not isinstance(template, NtupleExpectimaxPolicy) or not isinstance(
        base, NtupleExpectimaxPolicy
    ):
        raise EngineeringFault("C2 timing incumbent type mismatch")
    oracle = clone_batched(template)
    instrumented = clone_instrumented(template)
    for state_index, record in enumerate(records):
        record_id = str(record["record_id"])
        if record_id in existing:
            continue
        _execution_guard(out_dir=out_dir, runtime=runtime)
        started = time.perf_counter()
        row = _measure_cost_state(
            record=record,
            state_index=state_index,
            base=base,
            oracle=oracle,
            instrumented=instrumented,
        )
        runtime["active_runtime_seconds"] = (
            float(runtime["active_runtime_seconds"])
            + time.perf_counter() - started
        )
        append_jsonl(path, row)
        existing[record_id] = row
    if set(existing) != {str(record["record_id"]) for record in records}:
        raise EngineeringFault("C2 timing partition completion mismatch")
    return [existing[str(record["record_id"])] for record in records]


def _fit_weights(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    family_roots: dict[str, set[str]] = defaultdict(set)
    root_counts: Counter[str] = Counter()
    for row in rows:
        family_roots[str(row["behavior_family"])].add(str(row["root_ancestry"]))
        root_counts[str(row["root_ancestry"])] += 1
    family_count = len(family_roots)
    weights = []
    for row in rows:
        family = str(row["behavior_family"])
        root = str(row["root_ancestry"])
        weight = (
            1.0 / family_count
            / len(family_roots[family])
            / root_counts[root]
        )
        weights.append(weight)
    array = np.asarray(weights, dtype=np.float64)
    return array * (len(array) / np.sum(array))


def fit_and_seal_model(
    *,
    rows: Sequence[Mapping[str, Any]],
    out_dir: Path,
) -> dict[str, Any]:
    model_path = out_dir / "C2_COST_MODEL.json"
    if model_path.is_file():
        model = json.loads(model_path.read_text())
        validate_cost_model_payload(model)
        return model
    x = np.asarray([row["features"] for row in rows], dtype=np.float64)
    y = np.asarray([row["safety_load"] for row in rows], dtype=np.float64)
    weights = _fit_weights(rows)
    model = fit_cost_model(x, y, weights)
    repeated = fit_cost_model(x, y, weights)
    deterministic = (
        np.allclose(
            np.asarray(model["coefficients"]),
            np.asarray(repeated["coefficients"]),
            rtol=0.0,
            atol=1e-12,
        )
        and abs(float(model["intercept"]) - float(repeated["intercept"])) <= 1e-12
    )
    model.update({
        "fit_rows": len(rows),
        "fit_roots": len({row["root_ancestry"] for row in rows}),
        "fit_families": dict(Counter(row["behavior_family"] for row in rows)),
        "fit_record_manifest_sha256": canonical_json_hash([
            str(row["record_id"]) for row in rows
        ]),
        "fit_target_sha256": hashlib.sha256(
            np.asarray(y, dtype="<f8").tobytes()
        ).hexdigest(),
        "fit_weight_sha256": hashlib.sha256(
            np.asarray(weights, dtype="<f8").tobytes()
        ).hexdigest(),
        "deterministic_refit": bool(deterministic),
        "no_calibration": True,
        "promotable": False,
    })
    unhashed = dict(model)
    unhashed.pop("model_sha256", None)
    model["model_sha256"] = canonical_json_hash(unhashed)
    validate_cost_model_payload(model)
    atomic_write_json(model_path, model)
    return model


def _root_balanced_values(
    rows: Sequence[Mapping[str, Any]],
    value_key: str,
) -> list[float]:
    by_root: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_root[str(row["root_ancestry"])].append(float(row[value_key]))
    return [
        float(np.mean(values))
        for _root, values in sorted(by_root.items())
    ]


def validation_report(
    *,
    timing_rows: Sequence[Mapping[str, Any]],
    model: Mapping[str, Any],
) -> dict[str, Any]:
    rows = []
    for row in timing_rows:
        prediction = predict_cost(model, np.asarray(row["features"], dtype=np.float64))
        upper = conservative_upper(prediction)
        actual = float(row["safety_load"])
        admitted = upper <= ADMISSION_THRESHOLD
        rows.append({
            "record_id": row["record_id"],
            "root_ancestry": row["root_ancestry"],
            "behavior_family": row["behavior_family"],
            "prediction": prediction,
            "upper_load": upper,
            "actual_load": actual,
            "absolute_error": abs(prediction - actual),
            "upper_covers": actual <= upper,
            "admitted": admitted,
            "safe_absolute": float(row["exact_c1_combined_s"]) <= 2.5,
            "safe_relative": float(row["exact_c1_over_depth2"]) <= 8.0,
            "depth2_exact": bool(
                row["depth2_values_exact"] and row["depth2_action_match"]
            ),
        })
    root_predictions = _root_balanced_values(rows, "prediction")
    root_actual = _root_balanced_values(rows, "actual_load")
    correlation_result = spearmanr(root_predictions, root_actual)
    correlation = float(correlation_result.statistic)
    root_errors = _root_balanced_values(rows, "absolute_error")
    root_coverage = _root_balanced_values(rows, "upper_covers")
    admitted_rows = [row for row in rows if row["admitted"]]
    activity_families = sorted({
        str(row["behavior_family"]) for row in admitted_rows
    })
    checks = {
        "solver_success": bool(model["solver_success"]),
        "finite_nonnegative_parameters": bool(
            math.isfinite(float(model["intercept"]))
            and float(model["intercept"]) >= 0.0
            and np.all(np.isfinite(model["coefficients"]))
            and np.all(np.asarray(model["coefficients"]) >= 0.0)
        ),
        "deterministic_refit": bool(model["deterministic_refit"]),
        "depth2_exact": all(row["depth2_exact"] for row in rows),
        "spearman_ge_0_25": math.isfinite(correlation) and correlation >= 0.25,
        "root_mean_abs_error_le_0_35": float(np.mean(root_errors)) <= 0.35,
        "root_p90_abs_error_le_0_75": float(np.quantile(root_errors, 0.90)) <= 0.75,
        "upper_coverage_ge_90pct": float(np.mean(root_coverage)) >= 0.90,
        "activity_ge_15pct": len(admitted_rows) / max(1, len(rows)) >= 0.15,
        "activity_all_3_families": len(activity_families) == 3,
        "admitted_absolute_safe": all(row["safe_absolute"] for row in admitted_rows),
        "admitted_relative_safe": all(row["safe_relative"] for row in admitted_rows),
    }
    return payload_with_hash({
        "version": "c2_validation_report_v1",
        "decision": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "summary": {
            "records": len(rows),
            "roots": len({row["root_ancestry"] for row in rows}),
            "families": dict(Counter(row["behavior_family"] for row in rows)),
            "root_spearman": correlation,
            "root_mean_abs_error": float(np.mean(root_errors)),
            "root_p90_abs_error": float(np.quantile(root_errors, 0.90)),
            "root_upper_coverage": float(np.mean(root_coverage)),
            "admitted_records": len(admitted_rows),
            "activity_fraction": len(admitted_rows) / max(1, len(rows)),
            "activity_families": activity_families,
        },
        "rows": rows,
        "model_sha256": model["model_sha256"],
        "score_inspected": False,
        "dashboard_eligible": False,
    })


def _measure_gate_state(
    *,
    record: Mapping[str, Any],
    state_index: int,
    model: Mapping[str, Any],
    base: NtupleExpectimaxPolicy,
    oracle: BatchedPersistentPolicy,
    candidate: InstrumentedC2Policy,
) -> dict[str, Any]:
    state, sim = _state_sim(record["state"])
    plain_warm = base.action_values(state, sim)
    candidate_warm = candidate.cost_admitted_values(state, sim, model)
    warm_depth2_close, warm_depth2_difference = _values_close(
        {DIRECTION_NAMES[action]: value for action, value in plain_warm},
        candidate_warm["depth2_values"],
    )
    oracle_match = True
    oracle_difference = 0.0
    oracle_action_match = True
    seed = _selection_seed(record)
    plain_action = choose_action(base, plain_warm, seed)
    if candidate_warm["admitted"]:
        oracle.clear_decision_caches()
        oracle_values = oracle.adaptive_values(state, sim)["depth3"]
        oracle_match, oracle_difference = _values_close(
            {DIRECTION_NAMES[action]: value for action, value in oracle_values},
            candidate_warm["values"],
        )
        oracle_action = choose_action(oracle, oracle_values, seed)
        candidate_action = choose_action(candidate, candidate_warm["values"], seed)
        oracle_action_match = oracle_action == candidate_action
    else:
        rejected_match, rejected_difference = _values_close(
            {DIRECTION_NAMES[action]: value for action, value in plain_warm},
            candidate_warm["values"],
        )
        oracle_match = rejected_match
        oracle_difference = rejected_difference
        candidate_action = choose_action(candidate, candidate_warm["values"], seed)
        oracle_action_match = candidate_action == plain_action

    depth2_times = []
    candidate_times = []
    admissions = []
    result_hashes = []
    order_rows = []
    for repeat in range(TIMED_REPEATS):
        results: dict[str, Any] = {}

        def run_depth2() -> None:
            started = time.perf_counter()
            results["plain_values"] = base.action_values(state, sim)
            results["depth2_s"] = time.perf_counter() - started

        def run_candidate() -> None:
            started = time.perf_counter()
            results["candidate"] = candidate.cost_admitted_values(
                state, sim, model
            )
            results["candidate_s"] = time.perf_counter() - started

        order = (
            (run_depth2, run_candidate)
            if (state_index + repeat) % 2 == 0
            else (run_candidate, run_depth2)
        )
        for operation in order:
            operation()
        depth2_times.append(float(results["depth2_s"]))
        candidate_times.append(float(results["candidate_s"]))
        admissions.append(bool(results["candidate"]["admitted"]))
        result_hashes.append(_values_hash(results["candidate"]["values"]))
        order_rows.append([operation.__name__ for operation in order])
    if len(set(admissions)) != 1 or admissions[0] != bool(candidate_warm["admitted"]):
        raise EngineeringFault("C2 admission was nondeterministic")
    if len(set(result_hashes)) != 1:
        raise EngineeringFault("C2 returned values were nondeterministic")
    depth2_s = _median(depth2_times)
    candidate_s = _median(candidate_times)
    ratio = candidate_s / max(depth2_s, 1e-12)
    actual_load = safety_load(depth2_s=depth2_s, combined_s=candidate_s)
    exact = bool(
        warm_depth2_close
        and oracle_match
        and oracle_action_match
        and candidate_warm["eligible"]
    )
    return {
        "record_id": str(record["record_id"]),
        "root_ancestry": str(record["root_ancestry"]),
        "behavior_family": str(record["behavior_family"]),
        "partition": str(record["partition"]),
        "state_sha256": str(record["state_sha256"]),
        "state_index": int(state_index),
        "prediction": float(candidate_warm["prediction"]),
        "upper_load": float(candidate_warm["upper_load"]),
        "eligible": bool(candidate_warm["eligible"]),
        "admitted": bool(candidate_warm["admitted"]),
        "depth2_s_repeats": depth2_times,
        "c2_s_repeats": candidate_times,
        "interleave_orders": order_rows,
        "depth2_s": depth2_s,
        "c2_s": candidate_s,
        "c2_over_depth2": float(ratio),
        "actual_load": actual_load,
        "absolute_error": abs(
            float(candidate_warm["prediction"]) - actual_load
        ),
        "upper_covers": actual_load <= float(candidate_warm["upper_load"]),
        "value_action_equivalence": exact,
        "depth2_max_abs_difference": float(warm_depth2_difference),
        "oracle_max_abs_difference": float(oracle_difference),
        "result_value_sha256": result_hashes[0],
        "admission_repeat_exact": True,
        "score_inspected": False,
    }


def measure_runtime_gate(
    *,
    records: Sequence[Mapping[str, Any]],
    path: Path,
    model: Mapping[str, Any],
    runtime: dict[str, Any],
    out_dir: Path,
) -> list[dict[str, Any]]:
    existing = _load_jsonl_by_record(path)
    template = make_policy(incumbent_spec())
    base = make_policy(incumbent_spec())
    if not isinstance(template, NtupleExpectimaxPolicy) or not isinstance(
        base, NtupleExpectimaxPolicy
    ):
        raise EngineeringFault("C2 gate incumbent type mismatch")
    oracle = clone_batched(template)
    candidate = clone_instrumented(template)
    for state_index, record in enumerate(records):
        record_id = str(record["record_id"])
        if record_id in existing:
            continue
        _execution_guard(out_dir=out_dir, runtime=runtime)
        started = time.perf_counter()
        row = _measure_gate_state(
            record=record,
            state_index=state_index,
            model=model,
            base=base,
            oracle=oracle,
            candidate=candidate,
        )
        runtime["active_runtime_seconds"] = (
            float(runtime["active_runtime_seconds"])
            + time.perf_counter() - started
        )
        append_jsonl(path, row)
        existing[record_id] = row
    if set(existing) != {str(record["record_id"]) for record in records}:
        raise EngineeringFault("C2 gate completion mismatch")
    return [existing[str(record["record_id"])] for record in records]


def runtime_gate_report(
    *,
    rows: Sequence[Mapping[str, Any]],
    model: Mapping[str, Any],
) -> dict[str, Any]:
    ratios = np.asarray(
        [float(row["c2_over_depth2"]) for row in rows], dtype=np.float64
    )
    c2_times = np.asarray([float(row["c2_s"]) for row in rows], dtype=np.float64)
    depth2_times = np.asarray(
        [float(row["depth2_s"]) for row in rows], dtype=np.float64
    )
    admitted = [row for row in rows if bool(row["admitted"])]
    admitted_families = sorted({
        str(row["behavior_family"]) for row in admitted
    })
    admitted_root_errors = _root_balanced_values(admitted, "absolute_error")
    admitted_root_coverage = _root_balanced_values(admitted, "upper_covers")
    family_counts = Counter(str(row["behavior_family"]) for row in rows)
    checks = {
        "median_ratio_le_3x": float(np.median(ratios)) <= 3.0,
        "p90_ratio_le_5x": float(np.quantile(ratios, 0.90)) <= 5.0,
        "p99_ratio_le_8x": float(np.quantile(ratios, 0.99)) <= 8.0,
        "max_ratio_le_12x": float(np.max(ratios)) <= 12.0,
        "absolute_p99_lt_2_5s": float(np.quantile(c2_times, 0.99)) < 2.5,
        "zero_value_action_mismatch":
            all(bool(row["value_action_equivalence"]) for row in rows),
        "activity_ge_15pct": len(admitted) / max(1, len(rows)) >= 0.15,
        "activity_all_3_families": len(admitted_families) == 3,
        "max_family_share_le_40pct":
            max(family_counts.values(), default=0) / max(1, len(rows)) <= 0.40,
        "admitted_upper_coverage_ge_90pct": (
            bool(admitted_root_coverage)
            and float(np.mean(admitted_root_coverage)) >= 0.90
        ),
        "admitted_root_mean_abs_error_le_0_35": (
            bool(admitted_root_errors)
            and float(np.mean(admitted_root_errors)) <= 0.35
        ),
        "all_states_eligible": all(bool(row["eligible"]) for row in rows),
        "admission_deterministic":
            all(bool(row["admission_repeat_exact"]) for row in rows),
    }
    return payload_with_hash({
        "version": "c2_runtime_gate_report_v1",
        "decision": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "summary": {
            "states": len(rows),
            "roots": len({row["root_ancestry"] for row in rows}),
            "families": dict(family_counts),
            "admitted_states": len(admitted),
            "activity_fraction": len(admitted) / max(1, len(rows)),
            "admitted_families": admitted_families,
            "ratio_median": float(np.median(ratios)),
            "ratio_p90": float(np.quantile(ratios, 0.90)),
            "ratio_p99": float(np.quantile(ratios, 0.99)),
            "ratio_max": float(np.max(ratios)),
            "depth2_absolute_median_s": float(np.median(depth2_times)),
            "depth2_absolute_p99_s": float(np.quantile(depth2_times, 0.99)),
            "depth2_absolute_min_s": float(np.min(depth2_times)),
            "c2_absolute_median_s": float(np.median(c2_times)),
            "c2_absolute_p90_s": float(np.quantile(c2_times, 0.90)),
            "c2_absolute_p99_s": float(np.quantile(c2_times, 0.99)),
            "c2_absolute_max_s": float(np.max(c2_times)),
            "admitted_root_upper_coverage": (
                float(np.mean(admitted_root_coverage))
                if admitted_root_coverage else None
            ),
            "admitted_root_mean_abs_error": (
                float(np.mean(admitted_root_errors))
                if admitted_root_errors else None
            ),
        },
        "model_sha256": model["model_sha256"],
        "rows": list(rows),
        "score_inspected": False,
        "policy_outcomes": False,
        "dashboard_eligible": False,
    })


def _artifact_hashes(out_dir: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(out_dir.rglob("*")):
        if not path.is_file() or path.name == "C2_TERMINAL_RESULT.json":
            continue
        rows.append({
            "path": str(path),
            "relative_path": str(path.relative_to(out_dir)),
            "byte_size": path.stat().st_size,
            "sha256": sha256_path(path),
        })
    return {
        "files": rows,
        "file_count": len(rows),
        "total_bytes": sum(int(row["byte_size"]) for row in rows),
        "manifest_sha256": canonical_json_hash(rows),
    }


def _seal_terminal(
    *,
    out_dir: Path,
    decision: str,
    stage: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    terminal_path = out_dir / "C2_TERMINAL_RESULT.json"
    if terminal_path.exists():
        raise FileExistsError("C2 terminal already exists")
    operations = _operational_audit(out_dir)
    terminal = payload_with_hash({
        "version": "c2_cost_admission_terminal_v1",
        "decision": decision,
        "stage": stage,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **dict(payload),
        "artifact_manifest": _artifact_hashes(out_dir),
        "operations": operations,
        "forbidden_work": {
            "game_scores_inspected": 0,
            "policy_outcomes": 0,
            "h10_h20_h40_outcomes": 0,
            "new_labels": 0,
            "training_models": 0,
            "incumbent_changes": 0,
            "dashboard_changes": 0,
            "g3_transfer_access": 0,
        },
        "state": {
            "CONTINUE": (
                "fresh_full_policy_proposal_only"
                if decision == "READY_C2_FULL_POLICY_PREFLIGHT"
                else "none"
            ),
            "HOLD": "policy_evaluation_C2_successor_C2_human_PROMOTE",
            "KILL": {
                "C1": "permanent",
                "G3": "permanent",
                "G4": "permanent",
                "C2_cost_admission": decision == "KILL_C2_COST_ADMISSION",
            },
            "PROMOTE": False,
        },
        "dashboard_eligible": False,
        "promotable": False,
    })
    atomic_write_json(terminal_path, terminal)
    return {
        "decision": decision,
        "stage": stage,
        "terminal_path": str(terminal_path),
        "terminal_file_sha256": sha256_path(terminal_path),
        "terminal_payload_sha256": terminal["canonical_payload_sha256"],
        "summary": terminal.get("summary"),
        "operations": operations,
        "forbidden_work": terminal["forbidden_work"],
        "state": terminal["state"],
    }


def execute(*, out_dir: Path, preflight_lock: Path, jobs: int) -> dict[str, Any]:
    if jobs != FROZEN_JOBS:
        raise ValueError("C2 execute requires exactly one worker")
    lock = _load_preflight(preflight_lock, out_dir)
    marker = _load_marker(out_dir, lock)
    terminal_path = out_dir / "C2_TERMINAL_RESULT.json"
    if terminal_path.exists():
        raise FileExistsError("C2 execution already terminated")
    runtime_path = out_dir / "runtime_state.json"
    runtime = _runtime_state(runtime_path)
    try:
        _execution_guard(out_dir=out_dir, runtime=runtime)
        acquisition = run_fresh_acquisition(
            out_dir=out_dir,
            lock=lock,
            jobs=jobs,
        )
        runtime = acquisition["runtime"]
        runtime["phase"] = "corpus_seal"
        atomic_write_json(runtime_path, runtime)
        corpus_path = out_dir / "C2_CORPUS_MANIFEST.json"
        if corpus_path.is_file():
            corpus = json.loads(corpus_path.read_text())
            if not verify_payload_hash(corpus):
                raise EngineeringFault("C2 corpus manifest hash mismatch")
        else:
            corpus = build_corpus_manifest(
                out_dir=out_dir,
                acquisition=acquisition,
            )
            atomic_write_json(corpus_path, corpus)
        if corpus["decision"] != "C2_CORPUS_READY":
            return _seal_terminal(
                out_dir=out_dir,
                decision="KILL_C2_COST_ADMISSION",
                stage="fresh_corpus_yield",
                payload={
                    "marker_file_sha256": sha256_path(
                        out_dir / "C2_EXECUTION_OPENED.json"
                    ),
                    "marker_payload_sha256": marker["canonical_payload_sha256"],
                    "corpus_file_sha256": sha256_path(corpus_path),
                    "corpus_payload_sha256": corpus["canonical_payload_sha256"],
                    "corpus_checks": corpus["checks"],
                    "summary": {
                        "games": len(acquisition["rows"]),
                        "qualifying_by_family": acquisition[
                            "qualifying_by_family"
                        ],
                    },
                },
            )
        by_partition = {
            partition: [
                row for row in corpus["states"]
                if row["partition"] == partition
            ]
            for partition in ROOTS_PER_FAMILY
        }

        runtime["phase"] = "fit_timing"
        atomic_write_json(runtime_path, runtime)
        fit_rows = measure_cost_partition(
            records=by_partition["cost_fit"],
            path=out_dir / "fit_timings.jsonl",
            runtime=runtime,
            out_dir=out_dir,
        )
        runtime["phase"] = "model_fit"
        atomic_write_json(runtime_path, runtime)
        model = fit_and_seal_model(rows=fit_rows, out_dir=out_dir)

        runtime["phase"] = "validation_timing"
        atomic_write_json(runtime_path, runtime)
        validation_rows = measure_cost_partition(
            records=by_partition["engineering_validation"],
            path=out_dir / "validation_timings.jsonl",
            runtime=runtime,
            out_dir=out_dir,
        )
        report_path = out_dir / "C2_VALIDATION_REPORT.json"
        if report_path.is_file():
            validation = json.loads(report_path.read_text())
            if not verify_payload_hash(validation):
                raise EngineeringFault("C2 validation report hash mismatch")
        else:
            validation = validation_report(
                timing_rows=validation_rows,
                model=model,
            )
            atomic_write_json(report_path, validation)
        if validation["decision"] != "PASS":
            return _seal_terminal(
                out_dir=out_dir,
                decision="KILL_C2_COST_ADMISSION",
                stage="engineering_validation",
                payload={
                    "marker_file_sha256": sha256_path(
                        out_dir / "C2_EXECUTION_OPENED.json"
                    ),
                    "marker_payload_sha256": marker["canonical_payload_sha256"],
                    "corpus_file_sha256": sha256_path(corpus_path),
                    "corpus_payload_sha256": corpus["canonical_payload_sha256"],
                    "model_file_sha256": sha256_path(
                        out_dir / "C2_COST_MODEL.json"
                    ),
                    "model_sha256": model["model_sha256"],
                    "validation_file_sha256": sha256_path(report_path),
                    "validation_payload_sha256":
                        validation["canonical_payload_sha256"],
                    "validation_checks": validation["checks"],
                    "summary": validation["summary"],
                },
            )

        runtime["phase"] = "untouched_runtime_gate"
        atomic_write_json(runtime_path, runtime)
        gate_rows = measure_runtime_gate(
            records=by_partition["untouched_runtime_gate"],
            path=out_dir / "gate_timings.jsonl",
            model=model,
            runtime=runtime,
            out_dir=out_dir,
        )
        gate_path = out_dir / "C2_RUNTIME_GATE_REPORT.json"
        gate = runtime_gate_report(rows=gate_rows, model=model)
        atomic_write_json(gate_path, gate)
        decision = (
            "READY_C2_FULL_POLICY_PREFLIGHT"
            if gate["decision"] == "PASS"
            else "KILL_C2_COST_ADMISSION"
        )
        runtime["phase"] = "terminal"
        atomic_write_json(runtime_path, runtime)
        return _seal_terminal(
            out_dir=out_dir,
            decision=decision,
            stage="untouched_runtime_gate",
            payload={
                "marker_file_sha256": sha256_path(
                    out_dir / "C2_EXECUTION_OPENED.json"
                ),
                "marker_payload_sha256": marker["canonical_payload_sha256"],
                "corpus_file_sha256": sha256_path(corpus_path),
                "corpus_payload_sha256": corpus["canonical_payload_sha256"],
                "model_file_sha256": sha256_path(
                    out_dir / "C2_COST_MODEL.json"
                ),
                "model_sha256": model["model_sha256"],
                "validation_file_sha256": sha256_path(report_path),
                "validation_payload_sha256":
                    validation["canonical_payload_sha256"],
                "runtime_gate_file_sha256": sha256_path(gate_path),
                "runtime_gate_payload_sha256":
                    gate["canonical_payload_sha256"],
                "validation_checks": validation["checks"],
                "runtime_gate_checks": gate["checks"],
                "summary": {
                    "corpus": corpus["partition_summary"],
                    "validation": validation["summary"],
                    "runtime_gate": gate["summary"],
                    "active_runtime_seconds":
                        float(runtime["active_runtime_seconds"]),
                    "output_bytes": directory_bytes(out_dir),
                },
            },
        )
    except EngineeringFault as error:
        if terminal_path.exists():
            raise
        return _seal_terminal(
            out_dir=out_dir,
            decision="HOLD_C2_ENGINEERING_FAULT",
            stage=str(runtime.get("phase", "unknown")),
            payload={
                "marker_file_sha256": sha256_path(
                    out_dir / "C2_EXECUTION_OPENED.json"
                ),
                "marker_payload_sha256": marker["canonical_payload_sha256"],
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "summary": {
                    "active_runtime_seconds":
                        float(runtime.get("active_runtime_seconds", 0.0)),
                    "output_bytes": directory_bytes(out_dir),
                },
            },
        )
    except Exception as error:
        if terminal_path.exists():
            raise
        return _seal_terminal(
            out_dir=out_dir,
            decision="HOLD_C2_ENGINEERING_FAULT",
            stage=str(runtime.get("phase", "unknown")),
            payload={
                "marker_file_sha256": sha256_path(
                    out_dir / "C2_EXECUTION_OPENED.json"
                ),
                "marker_payload_sha256": marker["canonical_payload_sha256"],
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "summary": {
                    "active_runtime_seconds":
                        float(runtime.get("active_runtime_seconds", 0.0)),
                    "output_bytes": directory_bytes(out_dir),
                },
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "open", "execute"):
        command = subparsers.add_parser(name)
        command.add_argument("--out-dir", type=Path, required=True)
        command.add_argument("--jobs", type=int, default=FROZEN_JOBS)
        if name != "preflight":
            command.add_argument("--preflight-lock", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "preflight":
        result = run_preflight(out_dir=args.out_dir, jobs=args.jobs)
    elif args.command == "open":
        result = seal_execution_opened(
            out_dir=args.out_dir,
            preflight_lock=args.preflight_lock,
            jobs=args.jobs,
        )
    else:
        result = execute(
            out_dir=args.out_dir,
            preflight_lock=args.preflight_lock,
            jobs=args.jobs,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
