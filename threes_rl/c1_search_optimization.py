"""Freeze and profile exact C1 optimization corpora against killed R2a."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import tracemalloc
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from threes_rl.eval import make_policy
from threes_rl.expectimax import NtupleExpectimaxPolicy, _state_key
from threes_rl.ntuple import (
    NUM_RANKS,
    RANK_BY_VALUE,
    SYMMETRY_CELL_PERMS,
    NtupleValue,
    StagedNtupleValue,
)
from threes_rl.r15a_context_labels import sha256_path
from threes_rl.r2a_adaptive_expectimax import (
    CHANCE_LIMIT,
    EMPTY_TRIGGER,
    MARGIN_TRIGGER,
    NODE_BUDGET,
    NodeBudgetedNtuplePolicy,
    choose_action,
    clone_depth3,
    milestone_for_built_max,
    normalized_margin,
)
from threes_rl.run_artifacts import write_json
from threes_rl.sim import DIRECTION_NAMES, TERMINAL_TILE, ThreesSim, score_board, simulate_base_move
from threes_rl.train_td import state_from_replay_payload


VERSION = "c1_search_optimization_v1"
SOURCE_MANIFEST_SHA256 = "8604778696164fdabd5ab653c933b0b543ca1d20a8fde1d78b6e7da2994d794a"
POLICY_FILE_SHA256 = "d85a91576b8dc0ad80c2ed041dd1a0d62498eac9edb48445cb73233bb5454dd4"
SPLIT_SIZES = {"profile": 12, "equivalence": 24, "runtime_gate": 48}
TIMED_REPEATS = 3
TOLERANCE = 1e-9


def deterministic_key(*parts: object) -> str:
    return hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def _candidate_priority(record: dict[str, Any]) -> tuple[Any, ...]:
    metadata = record["context_metadata"]
    return (
        int(metadata["empty_count"]),
        deterministic_key("C1-state", record["record_id"]),
    )


def _allocate_split(
    by_family: dict[str, list[dict[str, Any]]],
    size: int,
) -> list[dict[str, Any]]:
    selected = []
    families = sorted(by_family)
    while len(selected) < size:
        added = False
        for family in families:
            if not by_family[family]:
                continue
            selected.append(by_family[family].pop(0))
            added = True
            if len(selected) == size:
                break
        if not added:
            break
    return selected


def freeze_corpus(source_manifest_path: Path, policy_file: Path, r2a_roots_path: Path) -> dict[str, Any]:
    if sha256_path(source_manifest_path) != SOURCE_MANIFEST_SHA256:
        raise ValueError("C1 source manifest changed")
    if sha256_path(policy_file) != POLICY_FILE_SHA256:
        raise ValueError("C1 incumbent policy changed")
    source = json.loads(source_manifest_path.read_text())
    r2a = json.loads(r2a_roots_path.read_text())
    excluded_roots = {str(root["root_cluster"]) for root in r2a["roots"]}
    by_root: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in source["selected_records"]:
        if record.get("partition") != "train":
            continue
        root = str(record["root_cluster"])
        if root in excluded_roots:
            continue
        built_max = int(record["context_metadata"]["built_max"])
        if milestone_for_built_max(built_max) is None:
            continue
        by_root[root].append(dict(record))
    candidates = [min(rows, key=_candidate_priority) for rows in by_root.values()]

    policy_spec = policy_file.read_text().splitlines()[-1]
    base = make_policy(policy_spec)
    if not isinstance(base, NtupleExpectimaxPolicy):
        raise TypeError("C1 incumbent is not n-tuple expectimax")
    triggered = []
    for record in candidates:
        starter = int(record.get("starter_tile", 1536))
        state = state_from_replay_payload(record["state"])
        sim = ThreesSim.from_stream_ids(deck_stream_id=1, slot_stream_id=2, starter_tile=starter)
        values = base.action_values(state, sim)
        margin = normalized_margin(values)
        empties = int(record["context_metadata"]["empty_count"])
        if empties > EMPTY_TRIGGER and margin > MARGIN_TRIGGER:
            continue
        record["incumbent_margin"] = margin
        record["incumbent_action_values"] = {DIRECTION_NAMES[action]: float(value) for action, value in values}
        record["trigger_reasons"] = {
            "low_empty": empties <= EMPTY_TRIGGER,
            "low_margin": margin <= MARGIN_TRIGGER,
        }
        triggered.append(record)
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in triggered:
        by_family[str(record["behavior_family"])].append(record)
    for family, rows in by_family.items():
        rows.sort(key=lambda row: deterministic_key("C1-root", family, row["root_cluster"]))
    split_names = list(SPLIT_SIZES)
    anchor_families = sorted(by_family, key=lambda family: (-len(by_family[family]), family))[:3]
    if len(anchor_families) < 3 or any(len(by_family[family]) < len(split_names) for family in anchor_families):
        raise ValueError("C1 lacks three families with enough roots for all splits")
    splits = {name: [] for name in split_names}
    for split in split_names:
        for family in anchor_families:
            splits[split].append(by_family[family].pop(0))
    used_roots: set[str] = set()
    for split, size in SPLIT_SIZES.items():
        selected = [*splits[split], *_allocate_split(by_family, size - len(splits[split]))]
        if len(selected) != size:
            raise ValueError(f"C1 {split} source shortage: {len(selected)} != {size}")
        split_roots = {str(row["root_cluster"]) for row in selected}
        if used_roots & split_roots:
            raise RuntimeError("C1 split root overlap")
        used_roots.update(split_roots)
        splits[split] = selected
    checks = {
        "split_sizes_exact": all(len(splits[name]) == size for name, size in SPLIT_SIZES.items()),
        "split_root_overlap_zero": sum(len(rows) for rows in splits.values()) == len(used_roots),
        "r2a_root_overlap_zero": not (used_roots & excluded_roots),
        "each_split_min_3_families": all(len({row["behavior_family"] for row in rows}) >= 3 for rows in splits.values()),
        "all_states_triggered": all(
            int(row["context_metadata"]["empty_count"]) <= EMPTY_TRIGGER
            or float(row["incumbent_margin"]) <= MARGIN_TRIGGER
            for rows in splits.values() for row in rows
        ),
    }
    return {
        "manifest_version": VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "decision": "C1_CORPUS_READY" if all(checks.values()) else "HOLD",
        "engineering_only": True,
        "dashboard_eligible": False,
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "r2a_roots": str(r2a_roots_path),
        "r2a_roots_sha256": sha256_path(r2a_roots_path),
        "incumbent_policy_file": str(policy_file),
        "incumbent_policy_file_sha256": POLICY_FILE_SHA256,
        "incumbent_policy": policy_spec,
        "config": {
            "node_budget": NODE_BUDGET,
            "chance_limit": CHANCE_LIMIT,
            "empty_trigger": EMPTY_TRIGGER,
            "margin_trigger": MARGIN_TRIGGER,
            "timed_repeats": TIMED_REPEATS,
            "relative_value_tolerance": TOLERANCE,
        },
        "checks": checks,
        "split_summary": {
            name: {
                "roots": len(rows),
                "families": dict(Counter(str(row["behavior_family"]) for row in rows)),
                "empty_counts": dict(Counter(str(row["context_metadata"]["empty_count"]) for row in rows)),
                "low_margin": sum(float(row["incumbent_margin"]) <= MARGIN_TRIGGER for row in rows),
            }
            for name, rows in splits.items()
        },
        "splits": splits,
    }


class ProfiledReferencePolicy(NodeBudgetedNtuplePolicy):
    def _reset_profile(self) -> None:
        self.profile = Counter()
        self.profile_time = Counter()
        self.unique_value_keys: set[tuple] = set()

    def action_values(self, state: Any, sim: ThreesSim) -> list[tuple[int, float]]:
        self._reset_profile()
        started = time.perf_counter()
        result = super().action_values(state, sim)
        self.profile_time["total"] += time.perf_counter() - started
        return result

    def _value(self, state: Any, sim: ThreesSim, depth: int) -> float:
        started = time.perf_counter()
        key = _state_key(state, depth)
        self.profile["transposition_lookups"] += 1
        if key in self._cache:
            self.profile["transposition_hits"] += 1
        else:
            self.unique_value_keys.add(key)
        result = super()._value(state, sim, depth)
        self.profile_time["value_inclusive"] += time.perf_counter() - started
        return result

    def _afterstate_value(self, board: np.ndarray) -> float:
        started = time.perf_counter()
        key = tuple(int(value) for value in np.asarray(board, dtype=np.int32).reshape(-1))
        self.profile["afterstate_lookups"] += 1
        if key in self._afterstate_cache:
            self.profile["afterstate_hits"] += 1
        result = super()._afterstate_value(board)
        self.profile_time["leaf"] += time.perf_counter() - started
        return result

    def _transition_outcomes(self, state: Any, sim: ThreesSim, action: int, *, include_next_preview: bool):
        started = time.perf_counter()
        outcomes = super()._transition_outcomes(state, sim, action, include_next_preview=include_next_preview)
        self.profile["chance_calls"] += 1
        self.profile["chance_outcomes"] += len(outcomes)
        self.profile_time["chance"] += time.perf_counter() - started
        return outcomes


def clone_profiled(base: NtupleExpectimaxPolicy, depth: int) -> ProfiledReferencePolicy:
    return ProfiledReferencePolicy(
        base.checkpoint,
        depth=depth,
        chance_limit=CHANCE_LIMIT if depth == 3 else base.chance_limit,
        blend_specs=list(base.blend_specs),
        phase_blend_specs=list(base.phase_blend_specs),
        bonus_specs=list(base.bonus_specs),
        tie_margin=base.tie_margin,
        tie_breaker=base.tie_breaker,
        ensemble_mode=base.ensemble_mode,
        geometry_weight=base.geometry_weight,
        geometry_min_tile=base.geometry_min_tile,
        node_budget=NODE_BUDGET if depth == 3 else 1_000_000_000,
    )


def _timed_values(policy: NtupleExpectimaxPolicy, state: Any, sim: ThreesSim) -> tuple[list[tuple[int, float]], float, int]:
    tracemalloc.start()
    started = time.perf_counter()
    values = policy.action_values(state, sim)
    elapsed = time.perf_counter() - started
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return values, elapsed, int(peak)


def profile_reference(corpus_path: Path) -> dict[str, Any]:
    corpus = json.loads(corpus_path.read_text())
    rows = corpus["splits"]["profile"]
    base_template = make_policy(corpus["incumbent_policy"])
    if not isinstance(base_template, NtupleExpectimaxPolicy):
        raise TypeError("C1 incumbent is not n-tuple expectimax")
    base = clone_profiled(base_template, 2)
    deep = clone_profiled(base_template, 3)
    measurements = []
    for row_index, row in enumerate(rows):
        starter = int(row.get("starter_tile", 1536))
        state = state_from_replay_payload(row["state"])
        sim = ThreesSim.from_stream_ids(deck_stream_id=1, slot_stream_id=2, starter_tile=starter)
        base.action_values(state, sim)
        deep.action_values(state, sim)
        for repeat in range(TIMED_REPEATS):
            base_values, base_time, base_peak = _timed_values(base, state, sim)
            base_profile = {"counts": dict(base.profile), "times": dict(base.profile_time), "unique_value_states": len(base.unique_value_keys)}
            deep_values, deep_time, deep_peak = _timed_values(deep, state, sim)
            deep_profile = {"counts": dict(deep.profile), "times": dict(deep.profile_time), "unique_value_states": len(deep.unique_value_keys)}
            measurements.append(
                {
                    "record_id": row["record_id"],
                    "root_cluster": row["root_cluster"],
                    "behavior_family": row["behavior_family"],
                    "repeat": repeat,
                    "depth2_s": base_time,
                    "depth3_s": deep_time,
                    "combined_s": base_time + deep_time,
                    "depth3_over_depth2": deep_time / max(base_time, 1e-12),
                    "combined_over_depth2": (base_time + deep_time) / max(base_time, 1e-12),
                    "depth2_peak_python_bytes": base_peak,
                    "depth3_peak_python_bytes": deep_peak,
                    "depth2_values": {DIRECTION_NAMES[action]: value for action, value in base_values},
                    "depth3_values": {DIRECTION_NAMES[action]: value for action, value in deep_values},
                    "depth2_profile": base_profile,
                    "depth3_profile": deep_profile,
                    "depth3_expanded_nodes": deep.expanded_value_nodes,
                    "depth3_budget_cutoffs": deep.budget_cutoffs,
                }
            )
    ratios = np.asarray([row["combined_over_depth2"] for row in measurements])
    aggregate_counts = Counter()
    aggregate_times = Counter()
    for row in measurements:
        aggregate_counts.update(row["depth3_profile"]["counts"])
        aggregate_times.update(row["depth3_profile"]["times"])
    total = float(aggregate_times["total"])
    chance = float(aggregate_times["chance"])
    leaf = float(aggregate_times["leaf"])
    return {
        "profile_version": VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "engineering_only": True,
        "score_outcomes_inspected": False,
        "corpus": str(corpus_path),
        "corpus_sha256": sha256_path(corpus_path),
        "measurements": measurements,
        "summary": {
            "states": len(rows),
            "timed_measurements": len(measurements),
            "combined_ratio_median": float(np.median(ratios)),
            "combined_ratio_p90": float(np.quantile(ratios, 0.90)),
            "combined_ratio_p99": float(np.quantile(ratios, 0.99)),
            "combined_ratio_max": float(np.max(ratios)),
            "aggregate_depth3_counts": dict(aggregate_counts),
            "aggregate_depth3_times_s": dict(aggregate_times),
            "aggregate_player_other_s": max(0.0, total - chance - leaf),
            "transposition_hit_rate": aggregate_counts["transposition_hits"] / max(1, aggregate_counts["transposition_lookups"]),
            "afterstate_hit_rate": aggregate_counts["afterstate_hits"] / max(1, aggregate_counts["afterstate_lookups"]),
            "median_depth3_expanded_nodes": median(row["depth3_expanded_nodes"] for row in measurements),
            "total_depth3_budget_cutoffs": sum(row["depth3_budget_cutoffs"] for row in measurements),
            "median_depth3_peak_python_bytes": median(row["depth3_peak_python_bytes"] for row in measurements),
        },
        "dashboard_eligible": False,
    }


def capture_reference(corpus_path: Path, splits: list[str]) -> dict[str, Any]:
    corpus = json.loads(corpus_path.read_text())
    template = make_policy(corpus["incumbent_policy"])
    if not isinstance(template, NtupleExpectimaxPolicy):
        raise TypeError("C1 incumbent is not n-tuple expectimax")
    base = make_policy(corpus["incumbent_policy"])
    if not isinstance(base, NtupleExpectimaxPolicy):
        raise TypeError("C1 incumbent is not n-tuple expectimax")
    deep = clone_depth3(template)
    rows = []
    for split in splits:
        for record in corpus["splits"][split]:
            starter = int(record.get("starter_tile", 1536))
            state = state_from_replay_payload(record["state"])
            sim = ThreesSim.from_stream_ids(deck_stream_id=1, slot_stream_id=2, starter_tile=starter)
            selection_seed = int(deterministic_key("C1-selection", record["record_id"])[:16], 16) & ((1 << 63) - 1)
            depth2 = base.action_values(state, sim)
            depth3 = deep.action_values(state, sim)
            rows.append(
                {
                    "split": split,
                    "record_id": record["record_id"],
                    "root_cluster": record["root_cluster"],
                    "selection_seed": selection_seed,
                    "legal_actions": [DIRECTION_NAMES[action] for action, _value in depth2],
                    "depth2_values": {DIRECTION_NAMES[action]: value for action, value in depth2},
                    "depth3_values": {DIRECTION_NAMES[action]: value for action, value in depth3},
                    "depth2_action": DIRECTION_NAMES[choose_action(base, depth2, selection_seed)],
                    "depth3_action": DIRECTION_NAMES[choose_action(deep, depth3, selection_seed)],
                    "depth3_expanded_nodes": deep.expanded_value_nodes,
                    "depth3_budget_cutoffs": deep.budget_cutoffs,
                }
            )
    return {
        "reference_version": "c1_reference_values_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "corpus": str(corpus_path),
        "corpus_sha256": sha256_path(corpus_path),
        "splits": splits,
        "rows": rows,
        "engineering_only": True,
        "score_outcomes_inspected": False,
        "dashboard_eligible": False,
    }


class IterativeReusePolicy(NodeBudgetedNtuplePolicy):
    """Reference-equivalent adaptive search with safe depth-2 cache reuse."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.prefilled_value_keys: set[tuple] = set()
        self.reused_value_keys: set[tuple] = set()
        self.reference_fallbacks = 0

    def _value(self, state: Any, sim: ThreesSim, depth: int) -> float:
        key = _state_key(state, depth)
        if key in self.prefilled_value_keys:
            self.reused_value_keys.add(key)
        return super()._value(state, sim, depth)

    def _manual_root_values(self, state: Any, sim: ThreesSim, depth: int) -> list[tuple[int, float]]:
        legal = self._legal_actions(state, sim)
        return [(int(action), float(self._action_value(state, sim, action, depth))) for action in legal]

    def adaptive_values(self, state: Any, sim: ThreesSim) -> dict[str, Any]:
        self.depth = 2
        self.chance_limit = None
        self.node_budget = 1_000_000_000
        started = time.perf_counter()
        depth2 = super().action_values(state, sim)
        depth2_s = time.perf_counter() - started
        depth2_cache = dict(self._cache)

        self.depth = 3
        self.chance_limit = CHANCE_LIMIT
        self.node_budget = NODE_BUDGET
        self.expanded_value_nodes = 0
        self.budget_cutoffs = 0
        self._cache = depth2_cache
        self._action_cache.clear()
        self.prefilled_value_keys = set(depth2_cache)
        self.reused_value_keys = set()
        started = time.perf_counter()
        depth3 = self._manual_root_values(state, sim, 3)
        depth3_s = time.perf_counter() - started
        effective_nodes = self.expanded_value_nodes + len(self.reused_value_keys)
        fallback = bool(self.budget_cutoffs or effective_nodes >= NODE_BUDGET)
        if fallback:
            self.reference_fallbacks += 1
            self._cache.clear()
            self._action_cache.clear()
            self.prefilled_value_keys = set()
            self.reused_value_keys = set()
            self.expanded_value_nodes = 0
            self.budget_cutoffs = 0
            started = time.perf_counter()
            depth3 = self._manual_root_values(state, sim, 3)
            depth3_s = time.perf_counter() - started
            effective_nodes = self.expanded_value_nodes
        self.prefilled_value_keys = set()
        return {
            "depth2": depth2,
            "depth3": depth3,
            "depth2_s": depth2_s,
            "depth3_s": depth3_s,
            "combined_s": depth2_s + depth3_s,
            "reused_value_states": len(self.reused_value_keys),
            "expanded_value_nodes": self.expanded_value_nodes,
            "effective_value_nodes": effective_nodes,
            "budget_cutoffs": self.budget_cutoffs,
            "reference_fallback": fallback,
        }


def clone_iterative(base: NtupleExpectimaxPolicy) -> IterativeReusePolicy:
    return IterativeReusePolicy(
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


def _values_close(reference: dict[str, float], candidate: list[tuple[int, float]]) -> tuple[bool, float]:
    candidate_map = {DIRECTION_NAMES[action]: float(value) for action, value in candidate}
    if set(reference) != set(candidate_map):
        return False, float("inf")
    maximum = 0.0
    for action, expected in reference.items():
        observed = candidate_map[action]
        difference = abs(observed - float(expected))
        maximum = max(maximum, difference)
        if difference > TOLERANCE * max(1.0, abs(float(expected))):
            return False, maximum
    return True, maximum


def benchmark_iterative(corpus_path: Path, reference_path: Path) -> dict[str, Any]:
    corpus = json.loads(corpus_path.read_text())
    reference = json.loads(reference_path.read_text())
    if reference["corpus_sha256"] != sha256_path(corpus_path):
        raise ValueError("C1 equivalence reference corpus mismatch")
    reference_rows = {str(row["record_id"]): row for row in reference["rows"]}
    template = make_policy(corpus["incumbent_policy"])
    if not isinstance(template, NtupleExpectimaxPolicy):
        raise TypeError("C1 incumbent is not n-tuple expectimax")
    base = make_policy(corpus["incumbent_policy"])
    deep = clone_depth3(template)
    optimized = clone_iterative(template)
    equivalence_rows = []
    for record in corpus["splits"]["equivalence"]:
        starter = int(record.get("starter_tile", 1536))
        state = state_from_replay_payload(record["state"])
        sim = ThreesSim.from_stream_ids(deck_stream_id=1, slot_stream_id=2, starter_tile=starter)
        result = optimized.adaptive_values(state, sim)
        expected = reference_rows[str(record["record_id"])]
        depth2_close, depth2_max = _values_close(expected["depth2_values"], result["depth2"])
        depth3_close, depth3_max = _values_close(expected["depth3_values"], result["depth3"])
        seed = int(expected["selection_seed"])
        depth2_action = DIRECTION_NAMES[choose_action(optimized, result["depth2"], seed)]
        depth3_action = DIRECTION_NAMES[choose_action(optimized, result["depth3"], seed)]
        equivalence_rows.append(
            {
                "record_id": record["record_id"],
                "depth2_close": depth2_close,
                "depth3_close": depth3_close,
                "depth2_max_abs_difference": depth2_max,
                "depth3_max_abs_difference": depth3_max,
                "depth2_action_match": depth2_action == expected["depth2_action"],
                "depth3_action_match": depth3_action == expected["depth3_action"],
                "reference_fallback": result["reference_fallback"],
                "reused_value_states": result["reused_value_states"],
            }
        )

    measurements = []
    for record in corpus["splits"]["profile"]:
        starter = int(record.get("starter_tile", 1536))
        state = state_from_replay_payload(record["state"])
        sim = ThreesSim.from_stream_ids(deck_stream_id=1, slot_stream_id=2, starter_tile=starter)
        base.action_values(state, sim)
        deep.action_values(state, sim)
        optimized.adaptive_values(state, sim)
        for repeat in range(TIMED_REPEATS):
            started = time.perf_counter()
            base_values = base.action_values(state, sim)
            base_s = time.perf_counter() - started
            started = time.perf_counter()
            deep_values = deep.action_values(state, sim)
            deep_s = time.perf_counter() - started
            optimized_result = optimized.adaptive_values(state, sim)
            base_close, _base_max = _values_close(
                {DIRECTION_NAMES[action]: value for action, value in base_values},
                optimized_result["depth2"],
            )
            deep_close, _deep_max = _values_close(
                {DIRECTION_NAMES[action]: value for action, value in deep_values},
                optimized_result["depth3"],
            )
            measurements.append(
                {
                    "record_id": record["record_id"],
                    "repeat": repeat,
                    "reference_depth2_s": base_s,
                    "reference_depth3_s": deep_s,
                    "reference_combined_s": base_s + deep_s,
                    "optimized_depth2_s": optimized_result["depth2_s"],
                    "optimized_depth3_s": optimized_result["depth3_s"],
                    "optimized_combined_s": optimized_result["combined_s"],
                    "optimized_over_depth2": optimized_result["combined_s"] / max(base_s, 1e-12),
                    "reference_over_depth2": (base_s + deep_s) / max(base_s, 1e-12),
                    "speedup": (base_s + deep_s) / max(optimized_result["combined_s"], 1e-12),
                    "depth2_close": base_close,
                    "depth3_close": deep_close,
                    "reused_value_states": optimized_result["reused_value_states"],
                    "reference_fallback": optimized_result["reference_fallback"],
                }
            )
    ratios = np.asarray([row["optimized_over_depth2"] for row in measurements])
    equivalence_pass = all(
        row["depth2_close"] and row["depth3_close"]
        and row["depth2_action_match"] and row["depth3_action_match"]
        for row in equivalence_rows
    )
    return {
        "benchmark_version": "c1_iterative_reuse_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "coherent_step": "a_iterative_depth_independent_and_depth_keyed_reuse",
        "engineering_only": True,
        "score_outcomes_inspected": False,
        "corpus": str(corpus_path),
        "corpus_sha256": sha256_path(corpus_path),
        "reference": str(reference_path),
        "reference_sha256": sha256_path(reference_path),
        "equivalence_pass": equivalence_pass,
        "equivalence_rows": equivalence_rows,
        "measurements": measurements,
        "summary": {
            "optimized_ratio_median": float(np.median(ratios)),
            "optimized_ratio_p90": float(np.quantile(ratios, 0.90)),
            "optimized_ratio_p99": float(np.quantile(ratios, 0.99)),
            "optimized_ratio_max": float(np.max(ratios)),
            "median_speedup": float(np.median([row["speedup"] for row in measurements])),
            "median_reused_value_states": median(row["reused_value_states"] for row in measurements),
            "reference_fallbacks": sum(row["reference_fallback"] for row in measurements),
        },
        "retained": equivalence_pass,
        "dashboard_eligible": False,
    }


def freeze_adjacency_supplement(corpus_path: Path) -> dict[str, Any]:
    corpus = json.loads(corpus_path.read_text())
    sequences = []
    for record in corpus["splits"]["profile"]:
        replay = json.loads(Path(str(record["source_replay"])).read_text())
        start = int(record["source_frame_index"])
        frames = {
            int(frame.get("index", -1)): frame
            for frame in replay.get("frames", [])
            if isinstance(frame, dict) and isinstance(frame.get("state"), dict)
        }
        states = []
        for offset in (0, 1, 2):
            frame = frames.get(start + offset)
            if frame is None:
                continue
            states.append(
                {
                    "offset": offset,
                    "frame_index": start + offset,
                    "state": frame["state"],
                }
            )
        if len(states) >= 2:
            sequences.append(
                {
                    "record_id": record["record_id"],
                    "root_cluster": record["root_cluster"],
                    "behavior_family": record["behavior_family"],
                    "starter_tile": record.get("starter_tile", 1536),
                    "source_replay": record["source_replay"],
                    "source_replay_sha256": record["source_replay_sha256"],
                    "states": states,
                }
            )
    return {
        "manifest_version": "c1_adjacent_profile_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "corpus": str(corpus_path),
        "corpus_sha256": sha256_path(corpus_path),
        "engineering_only": True,
        "score_outcomes_inspected": False,
        "sequences": sequences,
        "summary": {
            "root_sequences": len(sequences),
            "states": sum(len(sequence["states"]) for sequence in sequences),
            "families": dict(Counter(str(sequence["behavior_family"]) for sequence in sequences)),
        },
        "dashboard_eligible": False,
    }


def _prune_dict(mapping: dict[Any, Any], limit: int) -> None:
    excess = len(mapping) - limit
    if excess <= 0:
        return
    for key in list(mapping)[:excess]:
        mapping.pop(key, None)


class PersistentIterativePolicy(IterativeReusePolicy):
    """Exact bounded cross-decision cache reuse for fixed C1 parameters."""

    VALUE_CACHE_LIMIT = 50_000
    BOARD_CACHE_LIMIT = 100_000

    def adaptive_values(self, state: Any, sim: ThreesSim) -> dict[str, Any]:
        self.depth = 2
        self.chance_limit = None
        self.node_budget = 1_000_000_000
        self._action_cache.clear()
        self.expanded_value_nodes = 0
        self.budget_cutoffs = 0
        started = time.perf_counter()
        depth2 = self._manual_root_values(state, sim, 2)
        depth2_s = time.perf_counter() - started
        depth2_cache = dict(self._cache)

        self.depth = 3
        self.chance_limit = CHANCE_LIMIT
        self.node_budget = NODE_BUDGET
        self.expanded_value_nodes = 0
        self.budget_cutoffs = 0
        self._cache = depth2_cache
        self._action_cache.clear()
        self.prefilled_value_keys = set(depth2_cache)
        self.reused_value_keys = set()
        started = time.perf_counter()
        depth3 = self._manual_root_values(state, sim, 3)
        depth3_s = time.perf_counter() - started
        effective_nodes = self.expanded_value_nodes + len(self.reused_value_keys)
        fallback = bool(self.budget_cutoffs or effective_nodes >= NODE_BUDGET)
        if fallback:
            self.reference_fallbacks += 1
            self._cache.clear()
            self._action_cache.clear()
            self.prefilled_value_keys = set()
            self.reused_value_keys = set()
            self.expanded_value_nodes = 0
            self.budget_cutoffs = 0
            started = time.perf_counter()
            depth3 = self._manual_root_values(state, sim, 3)
            depth3_s = time.perf_counter() - started
            effective_nodes = self.expanded_value_nodes
        self.prefilled_value_keys = set()
        _prune_dict(self._cache, self.VALUE_CACHE_LIMIT)
        for mapping in (
            self._afterstate_cache,
            self._post_spawn_cache,
            self._score_cache,
            self._legal_cache,
            self._base_move_cache,
        ):
            _prune_dict(mapping, self.BOARD_CACHE_LIMIT)
        return {
            "depth2": depth2,
            "depth3": depth3,
            "depth2_s": depth2_s,
            "depth3_s": depth3_s,
            "combined_s": depth2_s + depth3_s,
            "reused_value_states": len(self.reused_value_keys),
            "expanded_value_nodes": self.expanded_value_nodes,
            "effective_value_nodes": effective_nodes,
            "budget_cutoffs": self.budget_cutoffs,
            "reference_fallback": fallback,
            "cache_sizes": {
                "value": len(self._cache),
                "afterstate": len(self._afterstate_cache),
                "post_spawn": len(self._post_spawn_cache),
                "base_move": len(self._base_move_cache),
            },
        }


def clone_persistent(base: NtupleExpectimaxPolicy) -> PersistentIterativePolicy:
    return PersistentIterativePolicy(
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


def benchmark_persistent(corpus_path: Path, adjacency_path: Path, reference_path: Path) -> dict[str, Any]:
    corpus = json.loads(corpus_path.read_text())
    adjacency = json.loads(adjacency_path.read_text())
    reference = json.loads(reference_path.read_text())
    if adjacency["corpus_sha256"] != sha256_path(corpus_path) or reference["corpus_sha256"] != sha256_path(corpus_path):
        raise ValueError("C1 persistent benchmark input mismatch")
    template = make_policy(corpus["incumbent_policy"])
    if not isinstance(template, NtupleExpectimaxPolicy):
        raise TypeError("C1 incumbent is not n-tuple expectimax")
    base = make_policy(corpus["incumbent_policy"])
    deep = clone_depth3(template)
    measurements = []
    for sequence in adjacency["sequences"]:
        optimized = clone_persistent(template)
        starter = int(sequence.get("starter_tile", 1536))
        for state_row in sequence["states"]:
            state = state_from_replay_payload(state_row["state"])
            sim = ThreesSim.from_stream_ids(deck_stream_id=1, slot_stream_id=2, starter_tile=starter)
            started = time.perf_counter()
            base_values = base.action_values(state, sim)
            base_s = time.perf_counter() - started
            started = time.perf_counter()
            deep_values = deep.action_values(state, sim)
            deep_s = time.perf_counter() - started
            result = optimized.adaptive_values(state, sim)
            base_close, _ = _values_close({DIRECTION_NAMES[a]: v for a, v in base_values}, result["depth2"])
            deep_close, _ = _values_close({DIRECTION_NAMES[a]: v for a, v in deep_values}, result["depth3"])
            measurements.append(
                {
                    "record_id": sequence["record_id"],
                    "offset": state_row["offset"],
                    "reference_depth2_s": base_s,
                    "reference_depth3_s": deep_s,
                    "optimized_combined_s": result["combined_s"],
                    "optimized_over_depth2": result["combined_s"] / max(base_s, 1e-12),
                    "speedup": (base_s + deep_s) / max(result["combined_s"], 1e-12),
                    "depth2_close": base_close,
                    "depth3_close": deep_close,
                    "reference_fallback": result["reference_fallback"],
                    "reused_value_states": result["reused_value_states"],
                    "cache_sizes": result["cache_sizes"],
                }
            )
    ratios = np.asarray([row["optimized_over_depth2"] for row in measurements])
    exact = all(row["depth2_close"] and row["depth3_close"] for row in measurements)
    return {
        "benchmark_version": "c1_persistent_reuse_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "coherent_step": "b_bounded_cross_decision_exact_reuse",
        "engineering_only": True,
        "score_outcomes_inspected": False,
        "corpus": str(corpus_path),
        "adjacency": str(adjacency_path),
        "reference": str(reference_path),
        "equivalence_pass": exact,
        "measurements": measurements,
        "summary": {
            "states": len(measurements),
            "optimized_ratio_median": float(np.median(ratios)),
            "optimized_ratio_p90": float(np.quantile(ratios, 0.90)),
            "optimized_ratio_p99": float(np.quantile(ratios, 0.99)),
            "optimized_ratio_max": float(np.max(ratios)),
            "median_speedup": float(np.median([row["speedup"] for row in measurements])),
            "median_reused_value_states": median(row["reused_value_states"] for row in measurements),
            "reference_fallbacks": sum(row["reference_fallback"] for row in measurements),
        },
        "retained": exact,
        "dashboard_eligible": False,
    }


class VectorizedCompositeLeaf:
    """Batch exact-order n-tuple feature lookup for the frozen incumbent leaf."""

    def __init__(self, policy: NtupleExpectimaxPolicy) -> None:
        if policy.ensemble_mode != "blend" or policy.geometry_weight != 0.0:
            raise ValueError("C1 vectorized leaf supports the frozen blend incumbent only")
        self.policy = policy
        self.models = [
            policy.value_model,
            *(model for _path, model, _weight in policy.blend_models),
            *(model for _path, model, _weight, _gate in policy.phase_blend_models),
            *(model for _path, model, _weight, _gate in policy.bonus_models),
        ]
        pattern_sets = [tuple(getattr(model, "patterns", getattr(model, "_patterns", ()))) for model in self.models]
        if not pattern_sets or any(patterns != pattern_sets[0] for patterns in pattern_sets):
            raise ValueError("C1 vectorized leaf requires identical n-tuple patterns")
        self.patterns = pattern_sets[0]
        self.cells = [
            np.asarray(
                [perm[np.asarray(pattern, dtype=np.intp)] for perm in SYMMETRY_CELL_PERMS],
                dtype=np.intp,
            )
            for pattern in self.patterns
        ]
        self.powers = [
            np.asarray([NUM_RANKS ** (len(pattern) - index - 1) for index in range(len(pattern))], dtype=np.int64)
            for pattern in self.patterns
        ]
        self.pattern_groups = {}
        for length in sorted({len(pattern) for pattern in self.patterns}):
            pattern_indices = [index for index, pattern in enumerate(self.patterns) if len(pattern) == length]
            self.pattern_groups[length] = {
                "pattern_indices": pattern_indices,
                "cells": np.stack([self.cells[index] for index in pattern_indices], axis=0),
                "powers": self.powers[pattern_indices[0]],
            }
        self.rank_lut = np.zeros(max(RANK_BY_VALUE) + 1, dtype=np.int16)
        for value, rank in RANK_BY_VALUE.items():
            if value < len(self.rank_lut):
                self.rank_lut[value] = int(rank)

    def _indices(self, boards: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        flat = np.asarray(boards, dtype=np.int32).reshape(len(boards), 16)
        if int(np.max(flat, initial=0)) >= len(self.rank_lut) or int(np.min(flat, initial=0)) < 0:
            raise ValueError("C1 vectorized leaf encountered unsupported tile value")
        ranks = self.rank_lut[flat]
        indices: list[np.ndarray | None] = [None] * len(self.patterns)
        for group in self.pattern_groups.values():
            grouped = np.sum(
                ranks[:, group["cells"]] * group["powers"][None, None, None, :],
                axis=3,
                dtype=np.int64,
            )
            for local_index, pattern_index in enumerate(group["pattern_indices"]):
                indices[pattern_index] = grouped[:, local_index, :]
        if any(value is None for value in indices):
            raise AssertionError("C1 pattern index extraction was incomplete")
        typed_indices = [value for value in indices if value is not None]
        return ranks, typed_indices

    def _phase_indices(self, boards: np.ndarray) -> np.ndarray:
        flat = np.asarray(boards, dtype=np.int32).reshape(len(boards), 16).copy()
        starter = 1536
        for row in flat:
            if row[0] == starter:
                row[0] = 0
                continue
            for index, value in enumerate(row):
                if value == starter:
                    row[index] = 0
                    break
        built = np.max(flat, axis=1, initial=0)
        return np.where(built < 384, 0, np.where(built < 1536, 1, np.where(built < 3072, 2, 3))).astype(np.int64)

    def _model_values(
        self,
        model: Any,
        boards: np.ndarray,
        indices: list[np.ndarray],
        phase_indices: np.ndarray,
    ) -> np.ndarray:
        batch = len(boards)
        features = np.empty((batch, len(SYMMETRY_CELL_PERMS), len(self.patterns)), dtype=np.float64)
        if isinstance(model, NtupleValue):
            for pattern_index, table_indices in enumerate(indices):
                features[:, :, pattern_index] = model.tables[pattern_index][table_indices]
        elif isinstance(model, StagedNtupleValue) and not model.promotion_enabled:
            if model.stage_mode != "phase4" or model.starter_tile != 1536:
                return np.asarray([float(model.value(board)) for board in boards], dtype=np.float64)
            stages = phase_indices
            for stage_index in range(len(model.stages)):
                mask = stages == stage_index
                if not bool(np.any(mask)):
                    continue
                stage = model.stages[stage_index]
                if stage is None:
                    features[mask] = 0.0
                    continue
                for pattern_index, table_indices in enumerate(indices):
                    features[mask, :, pattern_index] = stage.tables[pattern_index][table_indices[mask]]
        else:
            return np.asarray([float(model.value(board)) for board in boards], dtype=np.float64)
        ordered = features.reshape(batch, -1)
        return np.cumsum(ordered, axis=1, dtype=np.float64)[:, -1]

    def evaluate_many(self, boards: list[np.ndarray]) -> np.ndarray:
        if not boards:
            return np.empty(0, dtype=np.float64)
        board_array = np.asarray(boards, dtype=np.int32)
        _ranks, indices = self._indices(board_array)
        phase_indices = self._phase_indices(board_array)
        model_values = {
            id(model): self._model_values(model, board_array, indices, phase_indices)
            for model in self.models
        }
        output = np.empty(len(boards), dtype=np.float64)
        for index, board in enumerate(board_array):
            base = model_values[id(self.policy.value_model)][index]
            active_phase = [
                (model, weight)
                for _path, model, weight, gate in self.policy.phase_blend_models
                if gate == "all" or (isinstance(gate, int) and int(phase_indices[index]) >= gate)
            ]
            value = (self.policy.base_weight - sum(weight for _model, weight in active_phase)) * base
            for _path, model, weight in self.policy.blend_models:
                if weight > 0.0:
                    value += weight * model_values[id(model)][index]
            for model, weight in active_phase:
                if weight > 0.0:
                    value += weight * model_values[id(model)][index]
            for _path, model, weight, gate in self.policy.bonus_models:
                if weight != 0.0 and (gate == "all" or (isinstance(gate, int) and int(phase_indices[index]) >= gate)):
                    value += weight * model_values[id(model)][index]
            output[index] = value
        return output


class BatchedPersistentPolicy(PersistentIterativePolicy):
    """C1 step C: exact memoized chance expansion and batched leaf boards."""

    CHANCE_CACHE_LIMIT = 50_000

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.vectorized_leaf = VectorizedCompositeLeaf(self)
        self._chance_outcome_cache: dict[tuple[Any, ...], Any] = {}
        self.chance_cache_hits = 0
        self.chance_cache_misses = 0

    @staticmethod
    def _fast_board_key(board: np.ndarray) -> bytes:
        return np.asarray(board, dtype=np.int32).reshape(16).tobytes()

    def _score_board(self, board: np.ndarray) -> int:
        key = self._fast_board_key(board)
        cached = self._score_cache.get(key)
        if cached is not None:
            return cached
        value = score_board(board)
        self._score_cache[key] = value
        return value

    def _base_move(self, board: np.ndarray, action: int) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
        key = (self._fast_board_key(board), int(action))
        cached = self._base_move_cache.get(key)
        if cached is not None:
            return cached
        shifted, eligible = simulate_base_move(board, int(action))
        value = (shifted, tuple((int(row), int(col)) for row, col in eligible))
        self._base_move_cache[key] = value
        return value

    def _legal_actions(self, state: Any, sim: ThreesSim) -> tuple[int, ...]:
        if state.game_over:
            return ()
        key = self._fast_board_key(state.board)
        cached = self._legal_cache.get(key)
        if cached is not None:
            return cached
        if int(np.max(state.board, initial=0)) >= TERMINAL_TILE:
            legal = ()
        else:
            legal = tuple(action for action in range(4) if self._base_move(state.board, action)[1])
        self._legal_cache[key] = legal
        return legal

    def _afterstate_value(self, board: np.ndarray) -> float:
        key = self._fast_board_key(board)
        cached = self._afterstate_cache.get(key)
        if cached is not None:
            return cached
        value = float(self.vectorized_leaf.evaluate_many([np.asarray(board, dtype=np.int32)])[0])
        self._afterstate_cache[key] = value
        return value

    def _post_spawn_state_value(self, state: Any, sim: ThreesSim) -> float:
        if state.game_over:
            return 0.0
        key = self._fast_board_key(state.board)
        cached = self._post_spawn_cache.get(key)
        if cached is not None:
            return cached
        legal = self._legal_actions(state, sim)
        if not legal:
            self._post_spawn_cache[key] = 0.0
            return 0.0
        before_score = self._score_board(state.board)
        rows = []
        missing_keys = []
        missing_boards = []
        for action in legal:
            shifted, eligible = self._base_move(state.board, action)
            if not eligible:
                continue
            board_key = self._fast_board_key(shifted)
            if board_key not in self._afterstate_cache:
                missing_keys.append(board_key)
                missing_boards.append(shifted)
            rows.append((shifted, self._score_board(shifted) - before_score))
        if missing_boards:
            values = self.vectorized_leaf.evaluate_many(missing_boards)
            for board_key, value in zip(missing_keys, values):
                self._afterstate_cache[board_key] = float(value)
        value = max(
            float(score_delta + self._afterstate_cache[self._fast_board_key(shifted)])
            for shifted, score_delta in rows
        )
        self._post_spawn_cache[key] = value
        return value

    def _transition_outcomes(self, state: Any, sim: ThreesSim, action: int, *, include_next_preview: bool):
        key = (
            _state_key(state, 0),
            int(action),
            bool(include_next_preview),
            CHANCE_LIMIT,
            NODE_BUDGET,
        )
        cached = self._chance_outcome_cache.get(key)
        if cached is not None:
            self.chance_cache_hits += 1
            return cached
        self.chance_cache_misses += 1
        outcomes = super()._transition_outcomes(state, sim, action, include_next_preview=include_next_preview)
        self._chance_outcome_cache[key] = outcomes
        _prune_dict(self._chance_outcome_cache, self.CHANCE_CACHE_LIMIT)
        return outcomes

    def adaptive_values(self, state: Any, sim: ThreesSim) -> dict[str, Any]:
        before_hits = self.chance_cache_hits
        before_misses = self.chance_cache_misses
        result = super().adaptive_values(state, sim)
        result["chance_cache_hits"] = self.chance_cache_hits - before_hits
        result["chance_cache_misses"] = self.chance_cache_misses - before_misses
        result["cache_sizes"]["chance"] = len(self._chance_outcome_cache)
        return result

    def clear_decision_caches(self) -> None:
        self._cache.clear()
        self._action_cache.clear()
        self._afterstate_cache.clear()
        self._post_spawn_cache.clear()
        self._score_cache.clear()
        self._legal_cache.clear()
        self._base_move_cache.clear()
        self._chance_outcome_cache.clear()


def clone_batched(base: NtupleExpectimaxPolicy) -> BatchedPersistentPolicy:
    return BatchedPersistentPolicy(
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


def benchmark_batched(corpus_path: Path, reference_path: Path) -> dict[str, Any]:
    corpus = json.loads(corpus_path.read_text())
    reference = json.loads(reference_path.read_text())
    if reference["corpus_sha256"] != sha256_path(corpus_path):
        raise ValueError("C1 batched benchmark input mismatch")
    reference_rows = {str(row["record_id"]): row for row in reference["rows"]}
    template = make_policy(corpus["incumbent_policy"])
    if not isinstance(template, NtupleExpectimaxPolicy):
        raise TypeError("C1 incumbent is not n-tuple expectimax")
    optimized = clone_batched(template)
    equivalence_rows = []
    for record in corpus["splits"]["equivalence"]:
        optimized.clear_decision_caches()
        starter = int(record.get("starter_tile", 1536))
        state = state_from_replay_payload(record["state"])
        sim = ThreesSim.from_stream_ids(deck_stream_id=1, slot_stream_id=2, starter_tile=starter)
        result = optimized.adaptive_values(state, sim)
        expected = reference_rows[str(record["record_id"])]
        depth2_close, depth2_max = _values_close(expected["depth2_values"], result["depth2"])
        depth3_close, depth3_max = _values_close(expected["depth3_values"], result["depth3"])
        seed = int(expected["selection_seed"])
        equivalence_rows.append(
            {
                "record_id": record["record_id"],
                "depth2_close": depth2_close,
                "depth3_close": depth3_close,
                "depth2_max_abs_difference": depth2_max,
                "depth3_max_abs_difference": depth3_max,
                "depth2_action_match": DIRECTION_NAMES[choose_action(optimized, result["depth2"], seed)] == expected["depth2_action"],
                "depth3_action_match": DIRECTION_NAMES[choose_action(optimized, result["depth3"], seed)] == expected["depth3_action"],
                "reference_fallback": result["reference_fallback"],
            }
        )
    base = make_policy(corpus["incumbent_policy"])
    deep = clone_depth3(template)
    measurements = []
    for record in corpus["splits"]["profile"]:
        starter = int(record.get("starter_tile", 1536))
        state = state_from_replay_payload(record["state"])
        sim = ThreesSim.from_stream_ids(deck_stream_id=1, slot_stream_id=2, starter_tile=starter)
        base.action_values(state, sim)
        deep.action_values(state, sim)
        optimized.clear_decision_caches()
        optimized.adaptive_values(state, sim)
        for repeat in range(TIMED_REPEATS):
            optimized.clear_decision_caches()
            started = time.perf_counter()
            base_values = base.action_values(state, sim)
            base_s = time.perf_counter() - started
            started = time.perf_counter()
            deep_values = deep.action_values(state, sim)
            deep_s = time.perf_counter() - started
            result = optimized.adaptive_values(state, sim)
            base_close, _ = _values_close({DIRECTION_NAMES[a]: v for a, v in base_values}, result["depth2"])
            deep_close, _ = _values_close({DIRECTION_NAMES[a]: v for a, v in deep_values}, result["depth3"])
            measurements.append(
                {
                    "record_id": record["record_id"],
                    "repeat": repeat,
                    "reference_depth2_s": base_s,
                    "reference_depth3_s": deep_s,
                    "optimized_combined_s": result["combined_s"],
                    "optimized_over_depth2": result["combined_s"] / max(base_s, 1e-12),
                    "speedup": (base_s + deep_s) / max(result["combined_s"], 1e-12),
                    "depth2_close": base_close,
                    "depth3_close": deep_close,
                    "chance_cache_hits": result["chance_cache_hits"],
                    "chance_cache_misses": result["chance_cache_misses"],
                    "reference_fallback": result["reference_fallback"],
                }
            )
    exact = all(
        row["depth2_close"] and row["depth3_close"]
        and row["depth2_action_match"] and row["depth3_action_match"]
        for row in equivalence_rows
    ) and all(row["depth2_close"] and row["depth3_close"] for row in measurements)
    ratios = np.asarray([row["optimized_over_depth2"] for row in measurements])
    return {
        "benchmark_version": "c1_bytekey_grouped_leaf_v4",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "coherent_step": "c_memoized_chance_and_batched_leaf_expansion",
        "engineering_only": True,
        "score_outcomes_inspected": False,
        "equivalence_pass": exact,
        "equivalence_rows": equivalence_rows,
        "measurements": measurements,
        "summary": {
            "optimized_ratio_median": float(np.median(ratios)),
            "optimized_ratio_p90": float(np.quantile(ratios, 0.90)),
            "optimized_ratio_p99": float(np.quantile(ratios, 0.99)),
            "optimized_ratio_max": float(np.max(ratios)),
            "median_speedup": float(np.median([row["speedup"] for row in measurements])),
            "total_chance_cache_hits": sum(row["chance_cache_hits"] for row in measurements),
            "total_chance_cache_misses": sum(row["chance_cache_misses"] for row in measurements),
            "reference_fallbacks": sum(row["reference_fallback"] for row in measurements),
        },
        "retained": exact,
        "dashboard_eligible": False,
    }


def _runtime_gate_checks(ratios: np.ndarray, *, exact: bool, deterministic: bool) -> dict[str, bool]:
    return {
        "equivalence": bool(exact),
        "determinism": bool(deterministic),
        "median_le_3x": bool(np.median(ratios) <= 3.0),
        "p90_le_5x": bool(np.quantile(ratios, 0.90) <= 5.0),
        "p99_le_8x": bool(np.quantile(ratios, 0.99) <= 8.0),
        "max_le_12x": bool(np.max(ratios) <= 12.0),
    }


def run_runtime_gate(corpus_path: Path) -> dict[str, Any]:
    """Open the frozen gate once and interleave reference/optimized timing."""
    corpus = json.loads(corpus_path.read_text())
    rows = corpus["splits"]["runtime_gate"]
    template = make_policy(corpus["incumbent_policy"])
    if not isinstance(template, NtupleExpectimaxPolicy):
        raise TypeError("C1 incumbent is not n-tuple expectimax")
    base = make_policy(corpus["incumbent_policy"])
    deep = clone_depth3(template)
    optimized = clone_batched(template)
    measurements = []
    for row_index, record in enumerate(rows):
        starter = int(record.get("starter_tile", 1536))
        state = state_from_replay_payload(record["state"])
        sim = ThreesSim.from_stream_ids(deck_stream_id=1, slot_stream_id=2, starter_tile=starter)
        results: dict[str, Any] = {}

        def run_base() -> None:
            started = time.perf_counter()
            results["base_values"] = base.action_values(state, sim)
            results["base_s"] = time.perf_counter() - started

        def run_deep() -> None:
            started = time.perf_counter()
            results["deep_values"] = deep.action_values(state, sim)
            results["deep_s"] = time.perf_counter() - started

        def run_optimized() -> None:
            optimized.clear_decision_caches()
            results["optimized"] = optimized.adaptive_values(state, sim)

        order = (run_base, run_deep, run_optimized) if row_index % 2 == 0 else (run_optimized, run_deep, run_base)
        for operation in order:
            operation()

        optimized_result = results["optimized"]
        base_values = results["base_values"]
        deep_values = results["deep_values"]
        base_close, base_max = _values_close(
            {DIRECTION_NAMES[action]: value for action, value in base_values},
            optimized_result["depth2"],
        )
        deep_close, deep_max = _values_close(
            {DIRECTION_NAMES[action]: value for action, value in deep_values},
            optimized_result["depth3"],
        )
        selection_seed = int(deterministic_key("C1-runtime-selection", record["record_id"])[:16], 16)
        base_action = choose_action(base, base_values, selection_seed)
        deep_action = choose_action(deep, deep_values, selection_seed)
        optimized_base_action = choose_action(optimized, optimized_result["depth2"], selection_seed)
        optimized_deep_action = choose_action(optimized, optimized_result["depth3"], selection_seed)
        ratio = float(optimized_result["combined_s"] / max(float(results["base_s"]), 1e-12))
        measurements.append(
            {
                "record_id": record["record_id"],
                "root_cluster": record["root_cluster"],
                "behavior_family": record["behavior_family"],
                "interleave_order": [operation.__name__ for operation in order],
                "reference_depth2_s": float(results["base_s"]),
                "reference_depth3_s": float(results["deep_s"]),
                "optimized_depth2_s": float(optimized_result["depth2_s"]),
                "optimized_depth3_s": float(optimized_result["depth3_s"]),
                "optimized_combined_s": float(optimized_result["combined_s"]),
                "optimized_over_depth2": ratio,
                "depth2_close": base_close,
                "depth3_close": deep_close,
                "depth2_max_abs_difference": base_max,
                "depth3_max_abs_difference": deep_max,
                "depth2_action_match": base_action == optimized_base_action,
                "depth3_action_match": deep_action == optimized_deep_action,
                "reference_fallback": bool(optimized_result["reference_fallback"]),
                "budget_cutoffs": int(optimized_result["budget_cutoffs"]),
            }
        )

    ratios = np.asarray([row["optimized_over_depth2"] for row in measurements], dtype=np.float64)
    exact = all(
        row["depth2_close"] and row["depth3_close"]
        and row["depth2_action_match"] and row["depth3_action_match"]
        for row in measurements
    )
    deterministic = not any(row["reference_fallback"] or row["budget_cutoffs"] for row in measurements)
    checks = _runtime_gate_checks(ratios, exact=exact, deterministic=deterministic)
    summary = {
        "roots": len(measurements),
        "families": dict(Counter(row["behavior_family"] for row in measurements)),
        "optimized_ratio_median": float(np.median(ratios)),
        "optimized_ratio_p90": float(np.quantile(ratios, 0.90)),
        "optimized_ratio_p99": float(np.quantile(ratios, 0.99)),
        "optimized_ratio_max": float(np.max(ratios)),
        "reference_fallbacks": sum(row["reference_fallback"] for row in measurements),
        "budget_cutoffs": sum(row["budget_cutoffs"] for row in measurements),
    }
    return {
        "gate_version": "c1_runtime_gate_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "engineering_only": True,
        "score_outcomes_inspected": False,
        "opened_once": True,
        "corpus": str(corpus_path),
        "corpus_sha256": sha256_path(corpus_path),
        "implementation": "c1_bytekey_grouped_leaf_v4",
        "checks": checks,
        "summary": summary,
        "measurements": measurements,
        "decision": "PASS" if all(checks.values()) else "FAIL",
        "dashboard_eligible": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze_parser = subparsers.add_parser("freeze-corpus")
    freeze_parser.add_argument("--source-manifest", type=Path, required=True)
    freeze_parser.add_argument("--policy-file", type=Path, required=True)
    freeze_parser.add_argument("--r2a-roots", type=Path, required=True)
    freeze_parser.add_argument("--out", type=Path, required=True)
    profile_parser = subparsers.add_parser("profile-reference")
    profile_parser.add_argument("--corpus", type=Path, required=True)
    profile_parser.add_argument("--out", type=Path, required=True)
    capture_parser = subparsers.add_parser("capture-reference")
    capture_parser.add_argument("--corpus", type=Path, required=True)
    capture_parser.add_argument("--split", action="append", required=True)
    capture_parser.add_argument("--out", type=Path, required=True)
    iterative_parser = subparsers.add_parser("benchmark-iterative")
    iterative_parser.add_argument("--corpus", type=Path, required=True)
    iterative_parser.add_argument("--reference", type=Path, required=True)
    iterative_parser.add_argument("--out", type=Path, required=True)
    adjacency_parser = subparsers.add_parser("freeze-adjacency")
    adjacency_parser.add_argument("--corpus", type=Path, required=True)
    adjacency_parser.add_argument("--out", type=Path, required=True)
    persistent_parser = subparsers.add_parser("benchmark-persistent")
    persistent_parser.add_argument("--corpus", type=Path, required=True)
    persistent_parser.add_argument("--adjacency", type=Path, required=True)
    persistent_parser.add_argument("--reference", type=Path, required=True)
    persistent_parser.add_argument("--out", type=Path, required=True)
    batched_parser = subparsers.add_parser("benchmark-batched")
    batched_parser.add_argument("--corpus", type=Path, required=True)
    batched_parser.add_argument("--reference", type=Path, required=True)
    batched_parser.add_argument("--out", type=Path, required=True)
    runtime_parser = subparsers.add_parser("runtime-gate")
    runtime_parser.add_argument("--corpus", type=Path, required=True)
    runtime_parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze-corpus":
        payload = freeze_corpus(args.source_manifest, args.policy_file, args.r2a_roots)
        write_json(args.out, payload)
        printable = {"decision": payload["decision"], "checks": payload["checks"], "splits": payload["split_summary"]}
    elif args.command == "profile-reference":
        payload = profile_reference(args.corpus)
        write_json(args.out, payload)
        printable = payload["summary"]
    elif args.command == "capture-reference":
        payload = capture_reference(args.corpus, args.split)
        write_json(args.out, payload)
        printable = {"rows": len(payload["rows"]), "splits": payload["splits"]}
    elif args.command == "benchmark-iterative":
        payload = benchmark_iterative(args.corpus, args.reference)
        write_json(args.out, payload)
        printable = {"equivalence_pass": payload["equivalence_pass"], "retained": payload["retained"], "summary": payload["summary"]}
    elif args.command == "freeze-adjacency":
        payload = freeze_adjacency_supplement(args.corpus)
        write_json(args.out, payload)
        printable = payload["summary"]
    elif args.command == "benchmark-persistent":
        payload = benchmark_persistent(args.corpus, args.adjacency, args.reference)
        write_json(args.out, payload)
        printable = {"equivalence_pass": payload["equivalence_pass"], "retained": payload["retained"], "summary": payload["summary"]}
    elif args.command == "benchmark-batched":
        payload = benchmark_batched(args.corpus, args.reference)
        write_json(args.out, payload)
        printable = {"equivalence_pass": payload["equivalence_pass"], "retained": payload["retained"], "summary": payload["summary"]}
    else:
        payload = run_runtime_gate(args.corpus)
        write_json(args.out, payload)
        printable = {"decision": payload["decision"], "checks": payload["checks"], "summary": payload["summary"]}
    print(json.dumps(printable, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
