"""Generate fresh normal-start replay roots under the frozen G1-R slate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from threes_rl.eval import (
    EvalJob,
    EvalStreamIds,
    iter_eval_job_outputs,
    make_policy,
    max_tile_excluding_initial_starter,
)
from threes_rl.r15a_context_inventory import deterministic_key
from threes_rl.record_replay import state_payload
from threes_rl.replay_provenance import ORIGIN_FRESH, direct_root_fields
from threes_rl.restart_manifest import state_signature
from threes_rl.run_artifacts import write_json
from threes_rl.s3_power_preflight import sha256_path
from threes_rl.sim import ThreesSim
from threes_rl.train_td import state_from_replay_payload


VERSION = "g1r_acquisition_v2"
CHARTER_PATH = Path("threes_rl/G1R_NATURAL_ROOT_ACQUISITION_CHARTER.md")
AUTHORITATIVE_G1_V5_PATH = Path(
    "threes_rl/runs/forensics/g1_relational/"
    "G1_EXISTING_CORPUS_PREFLIGHT_V5_AUTHORITATIVE.json"
)
DIAGNOSTIC_INVENTORY_PATH = Path(
    "threes_rl/runs/forensics/r15a_context_a1/"
    "r15a_natural_state_inventory_a1_20260711.json"
)
INCUMBENT_PATH = Path("threes_rl/current_incumbent_policy.txt")
TEST_PATH = Path("tests/test_rl_g1r_acquire.py")
DASHBOARD_PATH = Path("threes_rl/runs/dashboard/dashboard.json")
MAX_MOVES = 5000
STARTER_TILE = 1536
MAX_JOBS = 2
MINIMUM_NICE = 10
MAX_TOTAL_GAMES = 12_000
MIN_FREE_GIB = 100.0
TARGET_FREE_GIB = 120.0
PANEL_PER_STRATUM = 32
MIN_PAIRWISE_DISAGREEMENT = 0.02
MIN_GENUINE_FAMILIES = 5
MIN_TEST_FAMILIES = 3
MAX_FAMILY_SHARE = 0.40
MIN_CELL_COUNT = 10
STREAM_BASES = {
    "logical_seed": 41_000_000_000,
    "deck_stream_id": 42_000_000_000,
    "slot_stream_id": 43_000_000_000,
    "policy_stream_id": 44_000_000_000,
}
PARTITION_TARGETS = {
    "test": {"pre1536": 256, "pre3072": 256},
    "validation": {"pre1536": 48, "pre3072": 48},
    "train": {"pre1536": 128, "pre3072": 128},
}
PARTITION_ORDER = ("test", "validation", "train")
STRATA = ("pre1536", "pre3072")
ROLES = ("source_success_window", "source_control")
POLICY_SOURCE_PATHS = (
    Path("threes_rl/eval.py"),
    Path("threes_rl/expectimax.py"),
    Path("threes_rl/ntuple.py"),
    Path("threes_rl/action_prior.py"),
    Path("threes_rl/sim.py"),
)
HISTORY_KEYS = (
    "logical_seed",
    "deck_stream_id",
    "slot_stream_id",
    "policy_stream_id",
    "seed",
    "root_seed",
    "source_seed",
)
HISTORY_PATTERN = re.compile(
    rb'"('
    + b"|".join(key.encode() for key in HISTORY_KEYS)
    + rb')"\s*:\s*([0-9]+)'
)
FRESH_ROOT_PATTERN = re.compile(rb"fresh:([0-9]+):1536")
CHECKPOINT_PATTERN = re.compile(r"(threes_rl/runs/[^:|]+/latest)")


class AcquisitionPause(RuntimeError):
    """A bounded acquisition pause with a frozen, reportable reason."""

    def __init__(self, decision: str, reason: str) -> None:
        super().__init__(reason)
        self.decision = decision
        self.reason = reason


def canonical_json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def current_nice() -> int:
    return int(os.getpriority(os.PRIO_PROCESS, 0))


def incumbent_spec() -> str:
    lines = [
        line.strip()
        for line in INCUMBENT_PATH.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(lines) != 1:
        raise ValueError("Expected exactly one incumbent policy line")
    return lines[0]


def policy_slate() -> tuple[tuple[str, str], ...]:
    return (
        ("g1r_corner2", "corner2"),
        ("g1r_expectimax2", "expectimax2"),
        (
            "g1r_parent_mc1000",
            "ntuple_expectimax2:"
            "threes_rl/runs/td_default_corner2_mc_1000_init3000_a0005_"
            "20260706/latest",
        ),
        (
            "g1r_student1",
            "ntuple_expectimax2:"
            "threes_rl/runs/td_default_student1_nstep_tc_50_from_mc1000_"
            "20260706/latest",
        ),
        (
            "g1r_replaycal",
            "ntuple_expectimax2:"
            "threes_rl/runs/replay_cal_phase4_late_midlate_top13_e3_a001_tc_"
            "20260706/latest",
        ),
        ("g1r_incumbent_depth2", incumbent_spec()),
    )


def stream_ids(family_index: int, game_index: int) -> dict[str, int]:
    offset = int(family_index) * 1_000_000 + int(game_index)
    return {name: base + offset for name, base in STREAM_BASES.items()}


def requested_stream_manifest(
    games_per_family: int,
    *,
    representative_families: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    selected = (
        set(representative_families)
        if representative_families is not None
        else {family for family, _policy in policy_slate()}
    )
    rows = []
    for family_index, (family, policy) in enumerate(policy_slate()):
        if family not in selected:
            continue
        for game_index in range(games_per_family):
            rows.append(
                {
                    "family_index": family_index,
                    "nominal_family": family,
                    "policy": policy,
                    "game_index": game_index,
                    **stream_ids(family_index, game_index),
                }
            )
    return rows


def _scan_history_file(path: Path) -> dict[str, set[int]]:
    found: dict[str, set[int]] = defaultdict(set)
    if not path.is_file() or path.suffix not in {".json", ".jsonl", ".csv"}:
        return found
    overlap = b""
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(4 * 1024 * 1024)
            if not chunk:
                break
            chunk = overlap + chunk
            for key, value in HISTORY_PATTERN.findall(chunk):
                found[key.decode()].add(int(value))
            for value in FRESH_ROOT_PATTERN.findall(chunk):
                found["fresh_root_seed"].add(int(value))
            overlap = chunk[-256:]
    return found


def historical_collision_union(
    *,
    exclude_dir: Path | None = None,
) -> tuple[dict[str, set[int]], dict[str, Any]]:
    found: dict[str, set[int]] = defaultdict(set)
    matched_sources: list[dict[str, Any]] = []
    root = Path("threes_rl/runs")
    exclude_resolved = exclude_dir.resolve() if exclude_dir is not None else None
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in {".json", ".jsonl", ".csv"}:
            continue
        if exclude_resolved is not None:
            try:
                path.resolve().relative_to(exclude_resolved)
                continue
            except ValueError:
                pass
        values = _scan_history_file(path)
        if not values:
            continue
        for key, items in values.items():
            found[key].update(items)
        matched_sources.append(
            {
                "path": str(path),
                "sha256": sha256_path(path),
                "byte_size": path.stat().st_size,
                "counts": {
                    key: len(items) for key, items in sorted(values.items())
                },
            }
        )
    return found, {
        "scan_root": str(root),
        "matched_source_count": len(matched_sources),
        "matched_sources": matched_sources,
        "matched_sources_sha256": canonical_json_hash(matched_sources),
        "value_counts": {
            key: len(items) for key, items in sorted(found.items())
        },
    }


def stream_collision_audit(
    rows: list[dict[str, Any]],
    *,
    exclude_dir: Path | None = None,
) -> dict[str, Any]:
    prior, sources = historical_collision_union(exclude_dir=exclude_dir)
    collisions: dict[str, list[int]] = {}
    for key in STREAM_BASES:
        requested = {int(row[key]) for row in rows}
        prior_values = set(prior.get(key, set()))
        if key == "logical_seed":
            for alias in ("seed", "root_seed", "source_seed", "fresh_root_seed"):
                prior_values.update(prior.get(alias, set()))
        collisions[key] = sorted(requested.intersection(prior_values))
    internal_values = [
        int(row[key]) for row in rows for key in STREAM_BASES
    ]
    internal_unique = len(internal_values) == len(set(internal_values))
    return {
        "historical_union": sources,
        "collisions": collisions,
        "zero_collisions": internal_unique and not any(collisions.values()),
        "internal_stream_ids_unique": internal_unique,
    }


def _checkpoint_dirs(spec: str) -> tuple[Path, ...]:
    return tuple(
        sorted({Path(match) for match in CHECKPOINT_PATTERN.findall(spec)})
    )


def _directory_artifact_manifest(path: Path) -> dict[str, Any]:
    if not path.is_dir():
        raise FileNotFoundError(f"Missing policy checkpoint: {path}")
    files = []
    for child in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        files.append(
            {
                "path": str(child),
                "relative_path": str(child.relative_to(path)),
                "byte_size": child.stat().st_size,
                "sha256": sha256_path(child),
            }
        )
    if not files:
        raise ValueError(f"Policy checkpoint contains no artifacts: {path}")
    return {
        "path": str(path),
        "files": files,
        "manifest_sha256": canonical_json_hash(files),
        "total_bytes": sum(row["byte_size"] for row in files),
    }


def load_and_lock_policies() -> tuple[dict[str, Any], dict[str, Any]]:
    source_hashes = {
        str(path): sha256_path(path) for path in POLICY_SOURCE_PATHS
    }
    artifact_cache: dict[Path, dict[str, Any]] = {}
    loaded: dict[str, Any] = {}
    families = []
    for family, spec in policy_slate():
        policy = make_policy(spec)
        loaded[family] = policy
        artifacts = []
        for checkpoint in _checkpoint_dirs(spec):
            if checkpoint not in artifact_cache:
                artifact_cache[checkpoint] = _directory_artifact_manifest(checkpoint)
            artifacts.append(artifact_cache[checkpoint])
        families.append(
            {
                "nominal_family": family,
                "resolved_policy_spec": spec,
                "resolved_policy_spec_sha256": hashlib.sha256(spec.encode()).hexdigest(),
                "loaded_type": type(policy).__name__,
                "checkpoint_artifacts": artifacts,
            }
        )
    lock = {
        "incumbent_policy_file": str(INCUMBENT_PATH),
        "incumbent_policy_file_sha256": sha256_path(INCUMBENT_PATH),
        "resolved_incumbent_spec": incumbent_spec(),
        "resolved_incumbent_spec_sha256": hashlib.sha256(
            incumbent_spec().encode()
        ).hexdigest(),
        "policy_implementation_sources": source_hashes,
        "policy_implementation_source_manifest_sha256": canonical_json_hash(
            source_hashes
        ),
        "families": families,
    }
    lock["policy_lock_sha256"] = canonical_json_hash(lock)
    return lock, loaded


def _roundtrip_state(payload: dict[str, Any]) -> None:
    validator = ThreesSim.from_stream_ids(
        deck_stream_id=1,
        slot_stream_id=2,
        starter_tile=STARTER_TILE,
    )
    restored = state_from_replay_payload(payload)
    if state_payload(restored, validator) != payload:
        raise ValueError("State payload failed exact simulator round trip")


def build_distinctness_panel() -> dict[str, Any]:
    inventory = json.loads(DIAGNOSTIC_INVENTORY_PATH.read_text())
    by_stratum_root: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in inventory["selected_records"]:
        if record.get("root_origin") != "fresh":
            continue
        if int(record.get("starter_tile", STARTER_TILE)) != STARTER_TILE:
            continue
        payload = record["state"]
        board = np.asarray(payload["board"], dtype=np.int32)
        built_max = max_tile_excluding_initial_starter(board, STARTER_TILE)
        if built_max == 768:
            stratum = "pre1536"
        elif built_max == 1536:
            stratum = "pre3072"
        else:
            continue
        _roundtrip_state(payload)
        root = str(record["root_cluster"])
        candidate = {
            "stratum": stratum,
            "root_cluster": root,
            "record_id": str(record["record_id"]),
            "source_replay": str(record["source_replay"]),
            "source_replay_sha256": str(record["source_replay_sha256"]),
            "source_frame_index": int(record["source_frame_index"]),
            "state": payload,
        }
        prior = by_stratum_root[stratum].get(root)
        if prior is None or deterministic_key(
            "G1R-family-panel-state-v1",
            stratum,
            root,
            candidate["record_id"],
        ) < deterministic_key(
            "G1R-family-panel-state-v1",
            stratum,
            root,
            prior["record_id"],
        ):
            by_stratum_root[stratum][root] = candidate
    records = []
    for stratum in STRATA:
        available = sorted(
            by_stratum_root[stratum].values(),
            key=lambda row: deterministic_key(
                "G1R-family-panel-v1",
                stratum,
                row["root_cluster"],
                row["record_id"],
            ),
        )
        if len(available) < PANEL_PER_STRATUM:
            raise ValueError(
                f"Distinctness panel has only {len(available)} {stratum} roots"
            )
        records.extend(available[:PANEL_PER_STRATUM])
    panel = {
        "version": "g1r_action_distinctness_panel_v1",
        "source_inventory": str(DIAGNOSTIC_INVENTORY_PATH),
        "source_inventory_sha256": sha256_path(DIAGNOSTIC_INVENTORY_PATH),
        "selection_outcomes_used": False,
        "diagnostic_only": True,
        "tie_rule": (
            "evaluate every legal action with the frozen policy calculation; "
            "choose the lowest simulator action index among exact maximum values"
        ),
        "records": records,
    }
    panel["panel_sha256"] = canonical_json_hash(panel)
    return panel


def _policy_action_values(policy: Any, state: Any, sim: ThreesSim) -> list[tuple[int, float]]:
    public = getattr(policy, "action_values", None)
    if callable(public):
        return [(int(action), float(value)) for action, value in public(state, sim)]
    for cache_name in ("_cache", "_action_cache", "_eval_cache"):
        cache = getattr(policy, cache_name, None)
        if cache is not None:
            cache.clear()
    legal = sim.legal_actions(state)
    return [
        (int(action), float(policy._action_value(state, sim, action, policy.depth)))
        for action in legal
    ]


def deterministic_policy_action(policy: Any, payload: dict[str, Any]) -> dict[str, Any]:
    state = state_from_replay_payload(payload)
    sim = ThreesSim.from_stream_ids(
        deck_stream_id=7,
        slot_stream_id=11,
        starter_tile=STARTER_TILE,
    )
    values = _policy_action_values(policy, state, sim)
    if not values:
        raise ValueError("Distinctness panel state has no legal action")
    best = max(value for _action, value in values)
    tied = sorted(action for action, value in values if value == best)
    return {
        "action": int(tied[0]),
        "exact_tie_count": len(tied),
    }


def _connected_components(
    families: list[str],
    alias_edges: set[tuple[str, str]],
) -> list[list[str]]:
    parent = {family: family for family in families}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left, right in alias_edges:
        union(left, right)
    components: dict[str, list[str]] = defaultdict(list)
    for family in families:
        components[find(family)].append(family)
    slate_order = {family: index for index, (family, _spec) in enumerate(policy_slate())}
    return sorted(
        (
            sorted(component, key=lambda family: slate_order[family])
            for component in components.values()
        ),
        key=lambda component: slate_order[component[0]],
    )


def audit_policy_distinctness(
    policies: dict[str, Any],
    panel: dict[str, Any],
) -> dict[str, Any]:
    signatures: dict[str, list[int]] = {}
    tie_counts: dict[str, int] = {}
    for family, _spec in policy_slate():
        rows = [
            deterministic_policy_action(policies[family], record["state"])
            for record in panel["records"]
        ]
        signatures[family] = [int(row["action"]) for row in rows]
        tie_counts[family] = sum(int(row["exact_tie_count"] > 1) for row in rows)
    pairs = []
    alias_edges: set[tuple[str, str]] = set()
    families = [family for family, _spec in policy_slate()]
    for left_index, left in enumerate(families):
        for right in families[left_index + 1 :]:
            strata_rates = {}
            for stratum in STRATA:
                indices = [
                    index
                    for index, record in enumerate(panel["records"])
                    if record["stratum"] == stratum
                ]
                disagreements = sum(
                    signatures[left][index] != signatures[right][index]
                    for index in indices
                )
                strata_rates[stratum] = disagreements / len(indices)
            overall = sum(
                left_action != right_action
                for left_action, right_action in zip(
                    signatures[left], signatures[right], strict=True
                )
            ) / len(panel["records"])
            passes = (
                overall >= MIN_PAIRWISE_DISAGREEMENT
                and all(strata_rates[stratum] > 0.0 for stratum in STRATA)
            )
            if not passes:
                alias_edges.add((left, right))
            pairs.append(
                {
                    "left": left,
                    "right": right,
                    "overall_disagreement": overall,
                    "stratum_disagreement": strata_rates,
                    "passes_floor": passes,
                }
            )
    components = _connected_components(families, alias_edges)
    representative_by_family = {
        family: component[0]
        for component in components
        for family in component
    }
    representatives = [component[0] for component in components]
    audit = {
        "panel_sha256": panel["panel_sha256"],
        "minimum_pairwise_disagreement": MIN_PAIRWISE_DISAGREEMENT,
        "requires_nonzero_in_each_stratum": True,
        "action_signature_sha256": {
            family: canonical_json_hash(signature)
            for family, signature in signatures.items()
        },
        "tie_state_counts": tie_counts,
        "pairwise": pairs,
        "alias_components": components,
        "representative_by_nominal_family": representative_by_family,
        "representative_families": representatives,
        "genuine_family_count": len(components),
        "passes": len(components) >= MIN_GENUINE_FAMILIES,
    }
    audit["audit_sha256"] = canonical_json_hash(audit)
    return audit


def source_role(
    frames: list[dict[str, Any]],
    frame_position: int,
    target: int,
) -> str:
    for frame in frames[frame_position + 1 : frame_position + 41]:
        board = np.asarray(frame["state"]["board"], dtype=np.int32)
        if max_tile_excluding_initial_starter(board, STARTER_TILE) >= target:
            return "source_success_window"
    return "source_control"


def extract_candidates(
    replay: dict[str, Any],
    *,
    family: str,
    replay_path: Path,
) -> list[dict[str, Any]]:
    frames = replay["frames"]
    root = f"{family}:fresh:{int(replay['seed'])}:{STARTER_TILE}"
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for position, frame in enumerate(frames):
        payload = frame["state"]
        if payload.get("game_over"):
            continue
        board = np.asarray(payload["board"], dtype=np.int32)
        built_max = max_tile_excluding_initial_starter(board, STARTER_TILE)
        if built_max == 768:
            stratum, target = "pre1536", 1536
        elif built_max == 1536:
            stratum, target = "pre3072", 3072
        else:
            continue
        _roundtrip_state(payload)
        state_hash = state_signature(payload, STARTER_TILE)
        by_stratum[stratum].append(
            {
                "record_id": deterministic_key(
                    "G1R-candidate-v1",
                    root,
                    stratum,
                    int(frame["index"]),
                    state_hash,
                )[:20],
                "root_cluster": root,
                "root_seed": int(replay["seed"]),
                "behavior_family": family,
                "stratum": stratum,
                "role": source_role(frames, position, target),
                "source_frame_index": int(frame["index"]),
                "state_sha1": state_hash,
                "source_replay": str(replay_path),
                "source_replay_sha256": None,
                "state": payload,
            }
        )
    selected = []
    for stratum, rows in sorted(by_stratum.items()):
        selected.append(
            min(
                rows,
                key=lambda row: deterministic_key(
                    "G1R-state-v1",
                    row["root_cluster"],
                    stratum,
                    row["source_frame_index"],
                    row["state_sha1"],
                ),
            )
        )
    return selected


def _allocation_key(partition: str, candidate: dict[str, Any]) -> str:
    return deterministic_key(
        "G1R-allocate-v1",
        partition,
        candidate["root_cluster"],
        candidate["stratum"],
        candidate["role"],
        candidate["record_id"],
    )


def _dedupe_candidates(
    candidates: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_root_stratum: dict[tuple[str, str], dict[str, Any]] = {}
    for candidate in candidates:
        key = (str(candidate["root_cluster"]), str(candidate["stratum"]))
        prior = by_root_stratum.get(key)
        if prior is None or _allocation_key("dedupe", candidate) < _allocation_key(
            "dedupe", prior
        ):
            by_root_stratum[key] = candidate
    return list(by_root_stratum.values())


def _pick_candidates(
    *,
    available: list[dict[str, Any]],
    count: int,
    partition: str,
    used_roots: set[str],
    family_counts: Counter[str],
    family_cap: int,
    stratum: str,
    role: str | None,
) -> list[dict[str, Any]]:
    selected = []
    while len(selected) < count:
        eligible = [
            candidate
            for candidate in available
            if candidate["root_cluster"] not in used_roots
            and candidate["stratum"] == stratum
            and (role is None or candidate["role"] == role)
            and family_counts[candidate["behavior_family"]] < family_cap
        ]
        if not eligible:
            break
        candidate = min(
            eligible,
            key=lambda row: (
                family_counts[row["behavior_family"]],
                _allocation_key(partition, row),
            ),
        )
        selected.append(candidate)
        used_roots.add(str(candidate["root_cluster"]))
        family_counts[str(candidate["behavior_family"])] += 1
    return selected


def allocate_partition_manifest(
    candidates: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    available = _dedupe_candidates(candidates)
    used_roots: set[str] = set()
    assignments: list[dict[str, Any]] = []
    deficits = []
    for partition in PARTITION_ORDER:
        targets = PARTITION_TARGETS[partition]
        partition_total = sum(targets.values())
        family_cap = math.floor(MAX_FAMILY_SHARE * partition_total)
        family_counts: Counter[str] = Counter()
        partition_rows: list[dict[str, Any]] = []
        for stratum in STRATA:
            for role in ROLES:
                picked = _pick_candidates(
                    available=available,
                    count=MIN_CELL_COUNT,
                    partition=partition,
                    used_roots=used_roots,
                    family_counts=family_counts,
                    family_cap=family_cap,
                    stratum=stratum,
                    role=role,
                )
                partition_rows.extend(picked)
                if len(picked) < MIN_CELL_COUNT:
                    deficits.append(
                        {
                            "partition": partition,
                            "stratum": stratum,
                            "role": role,
                            "needed": MIN_CELL_COUNT,
                            "selected": len(picked),
                        }
                    )
        for stratum in STRATA:
            have = sum(row["stratum"] == stratum for row in partition_rows)
            needed = targets[stratum] - have
            picked = _pick_candidates(
                available=available,
                count=max(0, needed),
                partition=partition,
                used_roots=used_roots,
                family_counts=family_counts,
                family_cap=family_cap,
                stratum=stratum,
                role=None,
            )
            partition_rows.extend(picked)
            if len(picked) < max(0, needed):
                deficits.append(
                    {
                        "partition": partition,
                        "stratum": stratum,
                        "role": "any",
                        "needed": max(0, needed),
                        "selected": len(picked),
                    }
                )
        assignments.extend(
            {
                **row,
                "partition": partition,
                "allocation_sha256": _allocation_key(partition, row),
            }
            for row in partition_rows
        )
    root_counts = Counter(row["root_cluster"] for row in assignments)
    per_partition = {}
    for partition in PARTITION_ORDER:
        rows = [row for row in assignments if row["partition"] == partition]
        total = sum(PARTITION_TARGETS[partition].values())
        family_counts = Counter(row["behavior_family"] for row in rows)
        per_partition[partition] = {
            "roots": len(rows),
            "stratum_counts": dict(
                sorted(Counter(row["stratum"] for row in rows).items())
            ),
            "role_cell_counts": {
                f"{stratum}/{role}": sum(
                    row["stratum"] == stratum and row["role"] == role
                    for row in rows
                )
                for stratum in STRATA
                for role in ROLES
            },
            "family_counts": dict(sorted(family_counts.items())),
            "family_share_max": (
                max(family_counts.values(), default=0) / len(rows) if rows else 0.0
            ),
            "family_cap_count": math.floor(MAX_FAMILY_SHARE * total),
        }
    overall_families = {
        row["behavior_family"] for row in assignments
    }
    test_families = {
        row["behavior_family"]
        for row in assignments
        if row["partition"] == "test"
    }
    structure_checks = {
        "exact_partition_sizes": all(
            per_partition[partition]["roots"]
            == sum(PARTITION_TARGETS[partition].values())
            for partition in PARTITION_ORDER
        ),
        "exact_stratum_targets": all(
            per_partition[partition]["stratum_counts"].get(stratum, 0)
            == PARTITION_TARGETS[partition][stratum]
            for partition in PARTITION_ORDER
            for stratum in STRATA
        ),
        "all_role_cells_at_least_10": all(
            count >= MIN_CELL_COUNT
            for partition in per_partition.values()
            for count in partition["role_cell_counts"].values()
        ),
        "family_cap_each_partition": all(
            partition["family_share_max"] <= MAX_FAMILY_SHARE
            for partition in per_partition.values()
        ),
        "at_least_5_genuine_families": len(overall_families)
        >= MIN_GENUINE_FAMILIES,
        "at_least_3_test_families": len(test_families) >= MIN_TEST_FAMILIES,
        "one_state_per_root": all(count == 1 for count in root_counts.values()),
        "zero_cross_partition_root_overlap": len(root_counts) == len(assignments),
        "no_allocator_deficits": not deficits,
    }
    compact_assignments = [
        {
            "partition": row["partition"],
            "record_id": row["record_id"],
            "root_cluster": row["root_cluster"],
            "behavior_family": row["behavior_family"],
            "stratum": row["stratum"],
            "role": row["role"],
            "state_sha1": row["state_sha1"],
            "source_replay": row["source_replay"],
            "source_replay_sha256": row["source_replay_sha256"],
            "source_frame_index": row["source_frame_index"],
            "allocation_sha256": row["allocation_sha256"],
        }
        for row in assignments
    ]
    return {
        "version": "g1r_partition_allocator_v1",
        "targets": PARTITION_TARGETS,
        "partition_order": PARTITION_ORDER,
        "stratum_order": STRATA,
        "role_order": ROLES,
        "role_cell_minimum": MIN_CELL_COUNT,
        "max_family_share": MAX_FAMILY_SHARE,
        "candidate_records_after_root_stratum_dedupe": len(available),
        "assignments": compact_assignments,
        "assignment_manifest_sha256": canonical_json_hash(compact_assignments),
        "per_partition": per_partition,
        "deficits": deficits,
        "structure_checks": structure_checks,
        "ready": all(structure_checks.values()),
    }


def service_health() -> dict[str, Any]:
    with urllib.request.urlopen(
        "http://127.0.0.1:8765/threes_rl/runs/dashboard/index.html",
        timeout=3,
    ) as response:
        dashboard_http_status = int(response.status)
    with urllib.request.urlopen(
        "http://127.0.0.1:8770/api/health",
        timeout=3,
    ) as response:
        advisor_health = json.loads(response.read())
        advisor_http_status = int(response.status)
    dashboard = json.loads(DASHBOARD_PATH.read_text())
    top_scores = [int(value) for value in dashboard["global_top_scores"][:3]]
    top_replays = dashboard["global_top_replays"][:3]
    protected_exist = all(
        Path(row["json"]).is_file() and Path(row["html"]).is_file()
        for row in top_replays
    )
    checks = {
        "dashboard_http_200": dashboard_http_status == 200,
        "advisor_http_200": advisor_http_status == 200,
        "advisor_status_ok": advisor_health.get("status") == "ok",
        "dashboard_record_263670": bool(top_scores) and top_scores[0] == 263670,
        "protected_top_three_exist": protected_exist and len(top_replays) == 3,
    }
    return {
        "checks": checks,
        "passes": all(checks.values()),
        "dashboard_top_scores": top_scores,
        "dashboard_sha256": sha256_path(DASHBOARD_PATH),
        "advisor": {
            "status": advisor_health.get("status"),
            "advisor": advisor_health.get("advisor"),
        },
    }


def replay_roundtrip_fixture() -> dict[str, Any]:
    sim = ThreesSim.from_stream_ids(
        deck_stream_id=45_999_999_901,
        slot_stream_id=45_999_999_902,
        starter_tile=STARTER_TILE,
    )
    state = sim.reset()
    payloads = []
    for _ in range(8):
        payload = state_payload(state, sim)
        _roundtrip_state(payload)
        payloads.append(payload)
        legal = sim.legal_actions(state)
        if not legal:
            break
        state, _info = sim.step(state, legal[0])
    return {
        "stream_ids": {
            "deck_stream_id": 45_999_999_901,
            "slot_stream_id": 45_999_999_902,
        },
        "states_checked": len(payloads),
        "payload_sha256": canonical_json_hash(payloads),
        "passes": bool(payloads),
    }


def _lock_identity() -> dict[str, str]:
    return {
        "charter_sha256": sha256_path(CHARTER_PATH),
        "authoritative_g1_v5_sha256": sha256_path(AUTHORITATIVE_G1_V5_PATH),
        "acquisition_implementation_sha256": sha256_path(Path(__file__)),
        "acquisition_test_sha256": sha256_path(TEST_PATH),
    }


def create_preflight_lock(
    *,
    out_dir: Path,
    games_per_family: int,
    lock_name: str,
    frozen_jobs: int,
) -> dict[str, Any]:
    if games_per_family <= 0:
        raise ValueError("games_per_family must be positive")
    if not 1 <= frozen_jobs <= 2:
        raise ValueError("G1-R preflight freezes one or two reduced-priority workers")
    out_dir.mkdir(parents=True, exist_ok=True)
    lock_path = out_dir / f"preflight_lock_{lock_name}.json"
    if lock_path.exists():
        raise FileExistsError(f"Preflight locks are immutable: {lock_path}")
    free_gib = shutil.disk_usage(out_dir).free / (1024**3)
    policies_lock, policies = load_and_lock_policies()
    panel = build_distinctness_panel()
    distinctness = audit_policy_distinctness(policies, panel)
    representatives = distinctness["representative_families"]
    stream_rows = requested_stream_manifest(
        games_per_family,
        representative_families=representatives,
    )
    if len(stream_rows) > MAX_TOTAL_GAMES:
        raise ValueError(
            f"Requested {len(stream_rows)} games exceeds {MAX_TOTAL_GAMES}"
        )
    collision = stream_collision_audit(stream_rows, exclude_dir=out_dir)
    completed_subset = completed_rows_subset_audit(
        completed_path=out_dir / "completed_games.jsonl",
        stream_rows=stream_rows,
        representative_map=distinctness["representative_by_nominal_family"],
    )
    services = service_health()
    fixture = replay_roundtrip_fixture()
    checks = {
        "genuine_family_count_at_least_5": distinctness["passes"],
        "all_policies_loaded_and_artifacts_hashed": len(policies)
        == len(policy_slate()),
        "zero_historical_or_internal_stream_collisions": collision[
            "zero_collisions"
        ],
        "replay_roundtrip_fixture": fixture["passes"],
        "free_disk_above_100_gib": free_gib >= MIN_FREE_GIB,
        "services_and_dashboard_truth": services["passes"],
        "bounded_game_manifest": len(stream_rows) <= MAX_TOTAL_GAMES,
        "completed_rows_exact_stream_spec_subset": completed_subset["passes"],
    }
    lock = {
        "version": VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "lock_name": lock_name,
        "bound_out_dir": str(out_dir.resolve()),
        "frozen_jobs": frozen_jobs,
        "required_minimum_nice": MINIMUM_NICE,
        "preflight_process_nice": current_nice(),
        "identity": _lock_identity(),
        "charter": str(CHARTER_PATH),
        "authoritative_g1_v5": str(AUTHORITATIVE_G1_V5_PATH),
        "policy_locks": policies_lock,
        "action_distinctness_panel": panel,
        "action_distinctness_audit": distinctness,
        "games_per_genuine_family": games_per_family,
        "representative_families": representatives,
        "max_moves": MAX_MOVES,
        "starter_tile": STARTER_TILE,
        "stream_rows": stream_rows,
        "stream_manifest_sha256": canonical_json_hash(stream_rows),
        "stream_collision_audit": collision,
        "completed_rows_subset_audit": completed_subset,
        "replay_roundtrip_fixture": fixture,
        "service_health": services,
        "free_gib": free_gib,
        "target_free_gib": TARGET_FREE_GIB,
        "checks": checks,
        "preflight_ready": all(checks.values()),
        "dashboard_eligible": False,
        "labels_generated": False,
        "models_fit": False,
    }
    lock["preflight_payload_sha256"] = canonical_json_hash(lock)
    write_json(lock_path, lock)
    return lock


def _validate_preflight_lock(path: Path) -> dict[str, Any]:
    lock = json.loads(path.read_text())
    if not lock.get("preflight_ready"):
        raise ValueError(f"Preflight is not READY: {path}")
    if lock["identity"] != _lock_identity():
        raise ValueError("G1-R charter/code/test/V5 identity changed after preflight")
    payload_hash = lock.pop("preflight_payload_sha256")
    if canonical_json_hash(lock) != payload_hash:
        raise ValueError("G1-R preflight payload hash mismatch")
    lock["preflight_payload_sha256"] = payload_hash
    return lock


def completed_rows_subset_audit(
    *,
    completed_path: Path,
    stream_rows: list[dict[str, Any]],
    representative_map: dict[str, str],
) -> dict[str, Any]:
    completed = _load_completed(completed_path)
    requested = {
        (str(row["nominal_family"]), int(row["game_index"])): row
        for row in stream_rows
    }
    mismatches = []
    for key, completed_row in sorted(completed.items()):
        requested_row = requested.get(key)
        if requested_row is None:
            mismatches.append({"key": list(key), "reason": "not_in_requested_manifest"})
            continue
        expected = {
            "genuine_family": representative_map[str(requested_row["nominal_family"])],
            "logical_seed": int(requested_row["logical_seed"]),
            "deck_stream_id": int(requested_row["deck_stream_id"]),
            "slot_stream_id": int(requested_row["slot_stream_id"]),
            "policy_stream_id": int(requested_row["policy_stream_id"]),
            "policy_spec_sha256": hashlib.sha256(
                str(requested_row["policy"]).encode()
            ).hexdigest(),
        }
        row_mismatches = {
            field: {
                "expected": value,
                "actual": completed_row.get(field),
            }
            for field, value in expected.items()
            if completed_row.get(field) != value
        }
        if row_mismatches:
            mismatches.append(
                {"key": list(key), "reason": "field_mismatch", "fields": row_mismatches}
            )
    return {
        "completed_rows": len(completed),
        "requested_rows": len(stream_rows),
        "mismatches": mismatches,
        "passes": not mismatches,
    }


def _load_completed(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    completed = {}
    if not path.is_file():
        return completed
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            key = (str(row["nominal_family"]), int(row["game_index"]))
            if key in completed:
                raise ValueError(f"Duplicate acquisition checkpoint row: {key}")
            completed[key] = row
    return completed


def _append_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _directory_bytes(path: Path) -> int:
    return sum(
        child.stat().st_size
        for child in path.rglob("*")
        if child.is_file()
    )


def _budget_limits(lock: dict[str, Any]) -> tuple[float, int]:
    is_pilot = int(lock["games_per_genuine_family"]) == 20
    return (
        12 * 3600 if is_pilot else 72 * 3600,
        4 * 1024**3 if is_pilot else 20 * 1024**3,
    )


def _runtime_state(path: Path) -> dict[str, Any]:
    if path.is_file():
        return json.loads(path.read_text())
    return {"active_runtime_seconds": 0.0, "chunks_completed": 0}


def _execution_guard(
    *,
    out_dir: Path,
    lock: dict[str, Any],
    runtime: dict[str, Any],
) -> None:
    wall_limit, byte_limit = _budget_limits(lock)
    free_gib = shutil.disk_usage(out_dir).free / (1024**3)
    if free_gib < MIN_FREE_GIB:
        raise AcquisitionPause(
            "HOLD_G1R_BUDGET",
            f"Free disk {free_gib:.2f} GiB is below {MIN_FREE_GIB:.0f} GiB",
        )
    used = _directory_bytes(out_dir)
    if used >= byte_limit:
        raise AcquisitionPause(
            "HOLD_G1R_BUDGET",
            f"Acquisition directory {used} bytes reached {byte_limit}",
        )
    if float(runtime["active_runtime_seconds"]) >= wall_limit:
        raise AcquisitionPause(
            "HOLD_G1R_BUDGET",
            f"Active runtime reached {wall_limit} seconds",
        )
    try:
        health = service_health()
    except Exception as error:
        raise AcquisitionPause(
            "HOLD_G1R_SERVICE",
            f"Service-health check raised {type(error).__name__}: {error}",
        ) from error
    if not health["passes"]:
        raise AcquisitionPause(
            "HOLD_G1R_SERVICE",
            f"Service health degraded: {health['checks']}",
        )


def _process_output(
    *,
    output: Any,
    stream_row: dict[str, Any],
    genuine_family: str,
    policy_spec: str,
    replay_dir: Path,
) -> dict[str, Any]:
    replay = output.replay
    if replay is None:
        raise RuntimeError("G1-R replay capture unexpectedly missing")
    logical_seed = int(stream_row["logical_seed"])
    replay.update(
        direct_root_fields(
            origin=ORIGIN_FRESH,
            seed=logical_seed,
            policy=genuine_family,
            first_score=int(replay["frames"][0]["state"]["score"]),
        )
    )
    replay["behavior_family"] = genuine_family
    replay["nominal_family"] = stream_row["nominal_family"]
    replay["acquisition_policy_spec"] = policy_spec
    replay["dashboard_eligible"] = False
    probe = extract_candidates(
        replay,
        family=genuine_family,
        replay_path=Path("pending"),
    )
    replay_path = replay_dir / (
        f"{stream_row['nominal_family']}_game_"
        f"{int(stream_row['game_index']):05d}_seed_{logical_seed}.json"
    )
    candidates = []
    if probe:
        replay_path.write_text(
            json.dumps(replay, separators=(",", ":"), sort_keys=True) + "\n"
        )
        candidates = extract_candidates(
            replay,
            family=genuine_family,
            replay_path=replay_path,
        )
        replay_hash = sha256_path(replay_path)
        for candidate in candidates:
            candidate["source_replay_sha256"] = replay_hash
    return {
        "nominal_family": str(stream_row["nominal_family"]),
        "genuine_family": genuine_family,
        "policy_spec_sha256": hashlib.sha256(policy_spec.encode()).hexdigest(),
        "family_index": int(stream_row["family_index"]),
        "game_index": int(stream_row["game_index"]),
        "logical_seed": logical_seed,
        "deck_stream_id": int(stream_row["deck_stream_id"]),
        "slot_stream_id": int(stream_row["slot_stream_id"]),
        "policy_stream_id": int(stream_row["policy_stream_id"]),
        "completed_moves": int(output.result.moves),
        "replay_retained": bool(candidates),
        "source_replay": str(replay_path) if candidates else None,
        "candidates": candidates,
        "dashboard_eligible": False,
    }


def verify_retained_sources(
    candidates: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    failures = []
    checked_replays: dict[str, dict[str, Any]] = {}
    checked_states = 0
    for candidate in candidates:
        source_path = str(candidate["source_replay"])
        expected_hash = str(candidate["source_replay_sha256"])
        if source_path not in checked_replays:
            path = Path(source_path)
            if not path.is_file():
                failures.append(
                    {"source_replay": source_path, "reason": "missing_replay"}
                )
                continue
            actual_hash = sha256_path(path)
            if actual_hash != expected_hash:
                failures.append(
                    {
                        "source_replay": source_path,
                        "reason": "hash_mismatch",
                        "expected": expected_hash,
                        "actual": actual_hash,
                    }
                )
                continue
            checked_replays[source_path] = json.loads(path.read_text())
        replay = checked_replays.get(source_path)
        if replay is None:
            continue
        frame_index = int(candidate["source_frame_index"])
        frames = [
            frame
            for frame in replay["frames"]
            if int(frame["index"]) == frame_index
        ]
        if len(frames) != 1:
            failures.append(
                {
                    "source_replay": source_path,
                    "frame_index": frame_index,
                    "reason": "frame_lookup_not_unique",
                }
            )
            continue
        payload = frames[0]["state"]
        try:
            _roundtrip_state(payload)
        except ValueError as error:
            failures.append(
                {
                    "source_replay": source_path,
                    "frame_index": frame_index,
                    "reason": "roundtrip_failure",
                    "detail": str(error),
                }
            )
            continue
        actual_state_hash = state_signature(payload, STARTER_TILE)
        if actual_state_hash != candidate["state_sha1"] or payload != candidate["state"]:
            failures.append(
                {
                    "source_replay": source_path,
                    "frame_index": frame_index,
                    "reason": "state_payload_mismatch",
                }
            )
            continue
        checked_states += 1
    return {
        "checked_replays": len(checked_replays),
        "checked_states": checked_states,
        "failures": failures,
        "passes": not failures,
    }


def _write_summary(
    *,
    out_dir: Path,
    lock: dict[str, Any],
    rows: list[dict[str, Any]],
    runtime: dict[str, Any],
    decision_override: str | None = None,
    stop_reason: str | None = None,
) -> dict[str, Any]:
    candidates = [
        candidate for row in rows for candidate in row["candidates"]
    ]
    allocator = allocate_partition_manifest(candidates)
    source_integrity = (
        verify_retained_sources(candidates)
        if allocator["ready"]
        else {
            "checked_replays": 0,
            "checked_states": 0,
            "failures": [],
            "passes": None,
            "status": "deferred_until_allocator_ready",
        }
    )
    terminal_collision = (
        stream_collision_audit(lock["stream_rows"], exclude_dir=out_dir)
        if allocator["ready"] and source_integrity["passes"]
        else {
            "zero_collisions": None,
            "status": "deferred_until_allocator_and_source_integrity_ready",
        }
    )
    decision = (
        decision_override
        or (
            "READY_G1R_ROOTS"
            if (
                allocator["ready"]
                and source_integrity["passes"]
                and terminal_collision["zero_collisions"]
            )
            else (
                "HOLD_G1R_INTEGRITY"
                if allocator["ready"]
                else "CONTINUE_G1R"
            )
        )
    )
    summary = {
        "version": VERSION,
        "decision": decision,
        "stop_reason": stop_reason,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "preflight_payload_sha256": lock["preflight_payload_sha256"],
        "games": len(rows),
        "active_runtime_seconds": float(runtime["active_runtime_seconds"]),
        "chunks_completed": int(runtime["chunks_completed"]),
        "games_by_nominal_family": dict(
            sorted(Counter(row["nominal_family"] for row in rows).items())
        ),
        "eligible_roots_by_family_stratum": {
            family: {
                stratum: len(
                    {
                        candidate["root_cluster"]
                        for candidate in candidates
                        if candidate["behavior_family"] == family
                        and candidate["stratum"] == stratum
                    }
                )
                for stratum in STRATA
            }
            for family in lock["representative_families"]
        },
        "role_counts": dict(
            sorted(Counter(candidate["role"] for candidate in candidates).items())
        ),
        "candidate_records": len(candidates),
        "candidate_roots": len(
            {candidate["root_cluster"] for candidate in candidates}
        ),
        "partition_allocator": allocator,
        "retained_source_integrity": source_integrity,
        "terminal_historical_stream_collision_audit": terminal_collision,
        "stream_manifest_sha256": lock["stream_manifest_sha256"],
        "output_bytes": _directory_bytes(out_dir),
        "free_gib": shutil.disk_usage(out_dir).free / (1024**3),
        "dashboard_eligible": False,
        "labels_generated": False,
        "models_fit": False,
    }
    write_json(out_dir / "summary.json", summary)
    write_json(
        out_dir / "candidate_inventory.json",
        {
            "version": VERSION,
            "preflight_payload_sha256": lock["preflight_payload_sha256"],
            "records": candidates,
        },
    )
    if decision == "READY_G1R_ROOTS":
        write_json(out_dir / "ready_partition_manifest.json", allocator)
    return summary


def run_acquisition(
    *,
    out_dir: Path,
    preflight_lock: Path,
    jobs: int,
) -> dict[str, Any]:
    if not 1 <= jobs <= MAX_JOBS:
        raise ValueError(f"jobs must be between 1 and {MAX_JOBS}")
    lock = _validate_preflight_lock(preflight_lock)
    if str(out_dir.resolve()) != lock["bound_out_dir"]:
        raise ValueError(
            f"Preflight is bound to {lock['bound_out_dir']}, not {out_dir.resolve()}"
        )
    if jobs != int(lock["frozen_jobs"]):
        raise ValueError(
            f"Runtime jobs={jobs} differs from frozen jobs={lock['frozen_jobs']}"
        )
    nice_value = current_nice()
    if nice_value < int(lock["required_minimum_nice"]):
        raise ValueError(
            f"Runtime nice={nice_value} is below frozen minimum "
            f"{lock['required_minimum_nice']}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / "completed_games.jsonl"
    runtime_path = out_dir / "runtime_state.json"
    completed = _load_completed(checkpoint)
    runtime = _runtime_state(runtime_path)
    replay_dir = out_dir / "source_replays"
    replay_dir.mkdir(exist_ok=True)
    representative_map = lock["action_distinctness_audit"][
        "representative_by_nominal_family"
    ]
    current_policy_lock, _loaded_policies = load_and_lock_policies()
    policy_lock_matches = (
        current_policy_lock["policy_lock_sha256"]
        == lock["policy_locks"]["policy_lock_sha256"]
    )
    current_collision = stream_collision_audit(
        lock["stream_rows"],
        exclude_dir=out_dir,
    )
    subset = completed_rows_subset_audit(
        completed_path=checkpoint,
        stream_rows=lock["stream_rows"],
        representative_map=representative_map,
    )
    resume_checks = {
        "bound_out_dir_matches": True,
        "frozen_jobs_matches": True,
        "reduced_priority_verified": nice_value
        >= int(lock["required_minimum_nice"]),
        "policy_lock_matches": policy_lock_matches,
        "zero_current_historical_stream_collisions": current_collision[
            "zero_collisions"
        ],
        "completed_rows_exact_stream_spec_subset": subset["passes"],
    }
    resume_audit = {
        "version": VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "preflight_payload_sha256": lock["preflight_payload_sha256"],
        "checks": resume_checks,
        "current_policy_lock_sha256": current_policy_lock["policy_lock_sha256"],
        "runtime_nice": nice_value,
        "required_minimum_nice": int(lock["required_minimum_nice"]),
        "historical_stream_collision_audit": current_collision,
        "completed_rows_subset_audit": subset,
        "passes": all(resume_checks.values()),
    }
    resume_audit["resume_audit_sha256"] = canonical_json_hash(resume_audit)
    write_json(
        out_dir / f"resume_integrity_{time.time_ns()}.json",
        resume_audit,
    )
    if not resume_audit["passes"]:
        raise ValueError(f"G1-R resume integrity failed: {resume_checks}")
    specs = dict(policy_slate())
    stream_rows = lock["stream_rows"]
    if len(stream_rows) > MAX_TOTAL_GAMES:
        raise ValueError("Preflight stream manifest exceeds total game budget")
    try:
        for family in lock["representative_families"]:
            pending = [
                row
                for row in stream_rows
                if row["nominal_family"] == family
                and (family, int(row["game_index"])) not in completed
            ]
            policy_spec = specs[family]
            policy = make_policy(policy_spec)
            chunk_size = max(1, min(8, jobs * 2))
            for chunk_start in range(0, len(pending), chunk_size):
                _execution_guard(out_dir=out_dir, lock=lock, runtime=runtime)
                chunk = pending[chunk_start : chunk_start + chunk_size]
                eval_jobs = [
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
                        policy_name=policy_spec,
                        eval_jobs=eval_jobs,
                        max_moves=MAX_MOVES,
                        capture_replay=True,
                        jobs=jobs,
                    )
                )
                outputs.sort(key=lambda output: output.index)
                for output in outputs:
                    stream_row = chunk[output.index]
                    nominal = str(stream_row["nominal_family"])
                    genuine = str(representative_map[nominal])
                    row = _process_output(
                        output=output,
                        stream_row=stream_row,
                        genuine_family=genuine,
                        policy_spec=policy_spec,
                        replay_dir=replay_dir,
                    )
                    _append_jsonl_row(checkpoint, row)
                    completed[(nominal, int(stream_row["game_index"]))] = row
                runtime["active_runtime_seconds"] = float(
                    runtime["active_runtime_seconds"]
                ) + (time.perf_counter() - started)
                runtime["chunks_completed"] = int(runtime["chunks_completed"]) + 1
                write_json(runtime_path, runtime)
                _execution_guard(out_dir=out_dir, lock=lock, runtime=runtime)
    except AcquisitionPause as pause:
        rows = [
            completed[key]
            for key in sorted(completed, key=lambda item: (item[0], item[1]))
        ]
        return _write_summary(
            out_dir=out_dir,
            lock=lock,
            rows=rows,
            runtime=runtime,
            decision_override=pause.decision,
            stop_reason=pause.reason,
        )
    expected_keys = {
        (str(row["nominal_family"]), int(row["game_index"]))
        for row in stream_rows
    }
    missing = expected_keys.difference(completed)
    if missing:
        raise RuntimeError(f"Acquisition ended with missing games: {sorted(missing)[:5]}")
    rows = [
        completed[key]
        for key in sorted(expected_keys, key=lambda item: (item[0], item[1]))
    ]
    return _write_summary(
        out_dir=out_dir,
        lock=lock,
        rows=rows,
        runtime=runtime,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--games-per-family", type=int, default=20)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--lock-name", default="pilot_v1")
    parser.add_argument("--preflight-lock", type=Path)
    args = parser.parse_args()
    if args.preflight_only:
        payload = create_preflight_lock(
            out_dir=args.out_dir,
            games_per_family=args.games_per_family,
            lock_name=args.lock_name,
            frozen_jobs=args.jobs,
        )
    else:
        if args.preflight_lock is None:
            parser.error("--preflight-lock is required for acquisition")
        try:
            os.nice(10)
        except OSError:
            pass
        payload = run_acquisition(
            out_dir=args.out_dir,
            preflight_lock=args.preflight_lock,
            jobs=args.jobs,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
