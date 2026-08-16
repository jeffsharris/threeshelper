"""Outcome-free root availability and clustered power preflight for S3."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from threes_rl.eval import max_tile_excluding_initial_starter
from threes_rl.phase0_replay_coverage_inventory import replay_action_signature
from threes_rl.r15a_context_inventory import (
    coalesced_behavior_family,
    deterministic_key,
)
from threes_rl.replay_provenance import GENUINE_ROOT_ORIGINS, replay_provenance
from threes_rl.restart_manifest import canonical_ancestry_id, state_signature
from threes_rl.run_artifacts import write_json
from threes_rl.sim import ThreesSim
from threes_rl.train_td import state_from_replay_payload


VERSION = "s3_power_preflight_v2"
CHARTER_PATH = Path("threes_rl/S3_FULL_POLICY_UTILITY_CHARTER.md")
SOURCE_INVENTORY_PATH = Path(
    "threes_rl/runs/forensics/r15a_context_a1/"
    "r15a_natural_state_inventory_a1_20260711.json"
)
HISTORICAL_LABELS_PATH = Path(
    "threes_rl/runs/forensics/r15a_context_a2/natural_labels/labels.jsonl"
)
R2A_ROOTS_PATH = Path(
    "threes_rl/runs/forensics/r2a_adaptive/R2A_ROOT_MANIFEST.json"
)
C1_CORPUS_PATH = Path("threes_rl/runs/forensics/c1_search/C1_CORPUS.json")
C1_RUNTIME_PATH = Path(
    "threes_rl/runs/forensics/c1_search/C1_RUNTIME_GATE.json"
)

ROOT_DESIGNS = (96, 128, 160, 192)
REPEAT_DESIGNS = (4, 6, 8)
ODDS_RATIOS = (1.25, 1.50, 2.00)
SIMULATION_DRAWS = 10_000
BOOTSTRAP_REPLICATES = 199
FAMILY_CAP = 0.40
COMPACT_STORAGE_LIMIT_BYTES = 10 * 1024**3
FREE_DISK_MIN_BYTES = 100 * 1024**3
FREE_DISK_TARGET_BYTES = 120 * 1024**3
DISK_CHECK_PATH = Path("threes_rl/runs")
STRATA = ("pre1536", "pre3072")
MILESTONE_FIELD = {
    "pre1536": "reached_1536",
    "pre3072": "reached_3072",
}
TARGET_BY_STRATUM = {"pre1536": 1536, "pre3072": 3072}

PRIOR_GATE_DIRECTORIES = (
    "bridge_reachability",
    "endgame_action_labels",
    "first_action_afterstates",
    "first_action_milestone",
    "first_action_path_forensics",
    "first_action_support_preservation",
    "frontier_compare",
    "frontier_selection",
    "label_stability",
    "mcts_rollout_gate",
    "post3072_frontier",
    "pre_nearfail_support_action_labels",
    "promotion_labels",
    "rare_event_frontier",
    "reachability",
    "reachability_screen",
    "selective_rollout_gate",
    "support_accumulation_frontier",
    "support_chain_gate",
    "support_chain_start_labels",
    "support_loss_action_labels",
    "support_preservation_frontier",
    "transition_reachability_audit",
)
FRESH_ROOT_PATTERN = re.compile(rb"fresh:[0-9]+:(?:1536|null|None)")
ROOT_ANCESTRY_SEED_PATTERN = re.compile(
    rb'"ancestry_key"\s*:\s*"root:fresh:[^"\r\n]+?:([0-9]+):0"'
)
ROOT_SEED_PATTERN = re.compile(rb'"root_seed"\s*:\s*([0-9]+)')


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def stratum_for_built_max(value: int) -> str | None:
    if int(value) == 768:
        return "pre1536"
    if int(value) == 1536:
        return "pre3072"
    return None


def beta_binomial_rho(root_means: np.ndarray, repeats: int) -> float:
    if len(root_means) < 2:
        return 0.0
    probability = float(np.mean(root_means))
    if probability <= 0.0 or probability >= 1.0:
        return 0.0
    observed_variance = float(np.var(root_means, ddof=1))
    rho = (
        repeats * observed_variance / (probability * (1.0 - probability)) - 1.0
    ) / max(1, repeats - 1)
    return float(np.clip(rho, 0.0, 0.99))


def odds_shift(probability: np.ndarray, odds_ratio: float) -> np.ndarray:
    clipped = np.clip(probability, 1e-12, 1.0 - 1e-12)
    odds = clipped / (1.0 - clipped)
    shifted = odds_ratio * odds
    return shifted / (1.0 + shifted)


def standardized_log_odds(
    control: dict[str, np.ndarray],
    treatment: dict[str, np.ndarray],
) -> float:
    effects = []
    for stratum in STRATA:
        control_mean = float(np.mean(control[stratum]))
        treatment_mean = float(np.mean(treatment[stratum]))
        if control_mean <= 0.0 or control_mean >= 1.0:
            count = float(np.sum(control[stratum]))
            control_mean = (count + 0.5) / (control[stratum].size + 1.0)
        if treatment_mean <= 0.0 or treatment_mean >= 1.0:
            count = float(np.sum(treatment[stratum]))
            treatment_mean = (count + 0.5) / (treatment[stratum].size + 1.0)
        effects.append(
            math.log(treatment_mean / (1.0 - treatment_mean))
            - math.log(control_mean / (1.0 - control_mean))
        )
    return float(np.mean(effects))


def historical_control_calibration(
    source_inventory: dict[str, Any],
    labels_path: Path,
) -> dict[str, Any]:
    relevant: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in source_inventory["selected_records"]:
        if record.get("partition") == "human_diagnostic":
            continue
        stratum = stratum_for_built_max(
            int(record["context_metadata"]["built_max"])
        )
        if stratum is not None:
            relevant[(str(record["root_cluster"]), stratum)].append(record)

    selected: dict[str, dict[str, Any]] = {}
    for (root, stratum), records in relevant.items():
        record = min(
            records,
            key=lambda row: deterministic_key(
                "S3-historical-control", row["record_id"]
            ),
        )
        selected[str(record["record_id"])] = {
            "root": root,
            "stratum": stratum,
            "family": str(record["behavior_family"]),
        }

    outcomes: dict[str, dict[str, list[int]]] = {
        stratum: defaultdict(list) for stratum in STRATA
    }
    terminal: dict[str, list[int]] = {stratum: [] for stratum in STRATA}
    block_outcomes: dict[str, dict[str, dict[str, list[int]]]] = {
        stratum: defaultdict(lambda: defaultdict(list)) for stratum in STRATA
    }
    with labels_path.open() as handle:
        for line in handle:
            task = json.loads(line)
            metadata = selected.get(str(task["record_id"]))
            if metadata is None:
                continue
            horizon_row = next(
                row for row in task["rows"] if int(row["horizon"]) == 40
            )
            stratum = metadata["stratum"]
            root = metadata["root"]
            value = int(horizon_row[MILESTONE_FIELD[stratum]])
            outcomes[stratum][root].append(value)
            terminal[stratum].append(int(horizon_row["terminal"]))
            block_outcomes[stratum][root][str(task["block"])].append(value)

    calibration: dict[str, Any] = {}
    for stratum in STRATA:
        roots = outcomes[stratum]
        repeat_counts = Counter(len(values) for values in roots.values())
        if len(repeat_counts) != 1:
            raise ValueError(
                f"Historical repeat counts vary for {stratum}: {repeat_counts}"
            )
        repeats = next(iter(repeat_counts))
        means = np.asarray(
            [np.mean(values) for _root, values in sorted(roots.items())],
            dtype=np.float64,
        )
        probability = float(np.mean(means))
        rho = beta_binomial_rho(means, repeats)
        concentration = (1.0 / rho - 1.0) if rho > 0.0 else 1_000_000.0
        calibration[stratum] = {
            "roots": len(roots),
            "repeats_per_root": repeats,
            "root_equal_base_rate": probability,
            "root_mean_variance": float(np.var(means, ddof=1)),
            "beta_binomial_rho": rho,
            "beta_alpha": probability * concentration,
            "beta_beta": (1.0 - probability) * concentration,
            "terminal_rate": float(np.mean(terminal[stratum])),
            "roots_with_any_success": int(np.count_nonzero(means > 0.0)),
            "families": dict(
                sorted(
                    Counter(
                        metadata["family"]
                        for metadata in selected.values()
                        if metadata["stratum"] == stratum
                    ).items()
                )
            ),
            "block_rate": {
                block: float(
                    np.mean(
                        [
                            np.mean(blocks[block])
                            for blocks in block_outcomes[stratum].values()
                            if block in blocks
                        ]
                    )
                )
                for block in ("A", "B")
            },
        }
    return {
        "selection": (
            "one deterministic relevant A2 state per whole ancestry and stratum; "
            "selection uses record ID only"
        ),
        "labels": str(labels_path),
        "labels_sha256": sha256_path(labels_path),
        "strata": calibration,
    }


def _scan_fresh_ids(path: Path) -> set[str]:
    roots: set[str] = set()
    if not path.is_file() or path.suffix not in {".json", ".jsonl"}:
        return roots
    overlap = b""
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(4 * 1024 * 1024)
            if not chunk:
                break
            chunk = overlap + chunk
            roots.update(
                match.decode("utf-8")
                for match in FRESH_ROOT_PATTERN.findall(chunk)
            )
            roots.update(
                f"fresh:{int(seed)}:1536"
                for seed in ROOT_ANCESTRY_SEED_PATTERN.findall(chunk)
            )
            roots.update(
                f"fresh:{int(seed)}:1536"
                for seed in ROOT_SEED_PATTERN.findall(chunk)
            )
            overlap = chunk[-64:]
    return roots


def prior_exclusion_catalog() -> dict[str, Any]:
    roots_by_source: dict[str, set[str]] = defaultdict(set)

    r2a = _json(R2A_ROOTS_PATH)
    roots_by_source["R2a"] = {
        str(record["root_cluster"]) for record in r2a["roots"]
    }
    c1 = _json(C1_CORPUS_PATH)
    roots_by_source["C1"] = {
        str(record["root_cluster"])
        for records in c1["splits"].values()
        for record in records
    }

    for directory_name in PRIOR_GATE_DIRECTORIES:
        directory = Path("threes_rl/runs/forensics") / directory_name
        for path in directory.rglob("*.json"):
            roots_by_source[f"prior_gate:{directory_name}"].update(
                _scan_fresh_ids(path)
            )
        for path in directory.rglob("*.jsonl"):
            roots_by_source[f"prior_gate:{directory_name}"].update(
                _scan_fresh_ids(path)
            )

    eval_manifest_paths = (
        Path(
            "threes_rl/runs/eval_manifests/"
            "r1_split_streams_d0_64_d1_192_c_512_20260709.json"
        ),
        Path(
            "threes_rl/runs/eval_manifests/"
            "r1b_split_streams_d2_512_20260709.json"
        ),
    )
    for path in eval_manifest_paths:
        manifest = _json(path)
        for rows in manifest["blocks"].values():
            for row in rows:
                roots_by_source["R1_R1b_eval"].add(
                    f"fresh:{int(row['logical_seed'])}:{row.get('starter_tile', 1536)}"
                )
    pre_c = Path(
        "threes_rl/runs/eval_manifests/"
        "r1b_pre_c_diagnostic_21roots_16repeats_20260710.json"
    )
    roots_by_source["R1b_pre_C"].update(_scan_fresh_ids(pre_c))

    parent_config = _json(
        Path(
            "threes_rl/runs/td_default_corner2_mc_1000_init3000_a0005_"
            "20260706/config.json"
        )
    )
    parent_seed = int(parent_config["seed"])
    for game_index in range(1, int(parent_config["games"]) + 1):
        seed = parent_seed + 1_000_003 * game_index
        roots_by_source["incumbent_parent_training"].add(
            f"fresh:{seed}:{parent_config.get('starter_tile', 1536)}"
        )

    component_configs = (
        Path(
            "threes_rl/runs/td_default_student1_nstep_tc_50_from_mc1000_"
            "20260706/config.json"
        ),
        Path(
            "threes_rl/runs/replay_cal_phase4_late_midlate_top13_e3_a001_tc_"
            "20260706/config.json"
        ),
    )
    for config_path in component_configs:
        config = _json(config_path)
        for key in ("start_state_replays", "replay_json"):
            for replay_text in config.get(key, []):
                replay_path = Path(replay_text)
                if not replay_path.is_file():
                    continue
                replay = _json(replay_path)
                roots_by_source["incumbent_component_sources"].add(
                    canonical_ancestry_id(replay, replay_path)
                )

    union = set().union(*roots_by_source.values())
    return {
        "roots": union,
        "counts": {
            source: len(roots) for source, roots in sorted(roots_by_source.items())
        },
        "union_count": len(union),
        "sources": {
            source: sorted(roots) for source, roots in sorted(roots_by_source.items())
        },
    }


def _source_success_role(
    frames: list[Any],
    start_position: int,
    starter_tile: int | None,
    target: int,
) -> str:
    for frame in frames[start_position + 1 : start_position + 41]:
        if not isinstance(frame, dict):
            continue
        state = frame.get("state")
        if not isinstance(state, dict):
            continue
        board = np.asarray(state.get("board"), dtype=np.int32)
        if board.shape != (4, 4):
            continue
        if max_tile_excluding_initial_starter(board, starter_tile) >= target:
            return "source_success_window"
    return "source_control"


def natural_root_candidates(
    source_inventory: dict[str, Any],
    excluded_roots: set[str],
) -> dict[str, Any]:
    replay_signatures: set[tuple[str, str]] = set()
    replay_candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    counts: Counter[str] = Counter()

    for text in source_inventory["source_paths"]:
        path = Path(text)
        counts["source_paths"] += 1
        if not path.is_file():
            counts["missing_source"] += 1
            continue
        try:
            replay = _json(path)
        except (OSError, json.JSONDecodeError, ValueError):
            counts["invalid_replay"] += 1
            continue
        provenance = replay_provenance(replay, path)
        if (
            provenance.get("replay_origin") not in GENUINE_ROOT_ORIGINS
            or provenance.get("root_origin") not in GENUINE_ROOT_ORIGINS
            or not provenance.get("replay_reset_invariant")
        ):
            counts["non_natural"] += 1
            continue
        family = coalesced_behavior_family(replay, path)
        if family == "human_observed":
            counts["human_excluded"] += 1
            continue
        replay_key = (family, replay_action_signature(replay))
        if replay_key in replay_signatures:
            counts["duplicate_replay_copy"] += 1
            continue
        replay_signatures.add(replay_key)
        root = canonical_ancestry_id(replay, path)
        if root in excluded_roots:
            counts["excluded_prior_root"] += 1
            continue
        starter_value = replay.get("starter_tile", 1536)
        starter_tile = None if starter_value is None else int(starter_value)
        validator = ThreesSim.from_stream_ids(
            deck_stream_id=1,
            slot_stream_id=2,
            starter_tile=starter_tile,
        )
        frames = replay.get("frames")
        if not isinstance(frames, list):
            counts["missing_frames"] += 1
            continue
        replay_hash = sha256_path(path)
        candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for fallback_index, frame in enumerate(frames):
            if not isinstance(frame, dict) or not isinstance(frame.get("state"), dict):
                continue
            payload = frame["state"]
            if bool(payload.get("game_over")):
                continue
            board = np.asarray(payload.get("board"), dtype=np.int32)
            if board.shape != (4, 4):
                continue
            built_max = max_tile_excluding_initial_starter(board, starter_tile)
            stratum = stratum_for_built_max(built_max)
            if stratum is None:
                continue
            try:
                state = state_from_replay_payload(payload)
            except (KeyError, TypeError, ValueError):
                counts["invalid_state_restore"] += 1
                continue
            legal = validator.legal_actions(state)
            if not legal:
                continue
            frame_index = int(frame.get("index", fallback_index))
            state_hash = state_signature(payload, starter_tile)
            preview = payload.get("preview") or {}
            cycle = payload.get("tile_cycle") or {}
            candidates[stratum].append(
                {
                    "record_id": deterministic_key(
                        "S3-candidate", root, frame_index, state_hash
                    )[:20],
                    "root_cluster": root,
                    "behavior_family": family,
                    "stratum": stratum,
                    "source_replay": str(path),
                    "source_replay_sha256": replay_hash,
                    "source_frame_index": frame_index,
                    "source_seed": replay.get("seed"),
                    "state_sha1": state_hash,
                    "starter_tile": starter_tile,
                    "empty_count": int(np.count_nonzero(board == 0)),
                    "legal_count": len(legal),
                    "preview_kind": str(preview.get("kind")),
                    "large_pending": bool(cycle.get("large_pending")),
                    "role": _source_success_role(
                        frames,
                        fallback_index,
                        starter_tile,
                        TARGET_BY_STRATUM[stratum],
                    ),
                    "state": payload,
                }
            )
        if not candidates:
            counts["natural_replay_without_target_state"] += 1
            continue
        replay_candidates[root].append(
            {
                "family": family,
                "path": str(path),
                "strata": {
                    stratum: min(
                        rows,
                        key=lambda row: deterministic_key(
                            "S3-state-choice", row["record_id"]
                        ),
                    )
                    for stratum, rows in candidates.items()
                },
                "target_frame_count": sum(len(rows) for rows in candidates.values()),
            }
        )
        counts["eligible_natural_replays"] += 1

    selected: list[dict[str, Any]] = []
    alias_count = 0
    for root, replay_rows in replay_candidates.items():
        representative = min(
            replay_rows,
            key=lambda row: (
                -len(row["strata"]),
                -int(row["target_frame_count"]),
                deterministic_key(
                    "S3-replay-representative", root, row["family"], row["path"]
                ),
            ),
        )
        alias_count += len(replay_rows) - 1
        selected.extend(representative["strata"].values())
    counts["cross_family_or_policy_aliases_removed"] = alias_count

    return {
        "records": sorted(
            selected,
            key=lambda row: (row["root_cluster"], row["stratum"]),
        ),
        "counts": dict(sorted(counts.items())),
    }


def _selection_attempt(
    records: list[dict[str, Any]],
    roots_required: int,
    attempt: int,
) -> list[dict[str, Any]] | None:
    per_stratum = roots_required // 2
    family_limit = int(math.floor(FAMILY_CAP * roots_required + 1e-12))
    by_stratum_family: dict[str, dict[str, list[dict[str, Any]]]] = {
        stratum: defaultdict(list) for stratum in STRATA
    }
    for record in records:
        by_stratum_family[record["stratum"]][record["behavior_family"]].append(
            record
        )
    for stratum in STRATA:
        for family, rows in by_stratum_family[stratum].items():
            rows.sort(
                key=lambda row: deterministic_key(
                    "S3-feasibility", attempt, stratum, family, row["record_id"]
                )
            )

    selected: list[dict[str, Any]] = []
    selected_roots: set[str] = set()
    family_counts: Counter[str] = Counter()
    stratum_counts: Counter[str] = Counter()
    while any(stratum_counts[stratum] < per_stratum for stratum in STRATA):
        progressed = False
        for stratum in STRATA:
            if stratum_counts[stratum] >= per_stratum:
                continue
            families = sorted(
                by_stratum_family[stratum],
                key=lambda family: (
                    family_counts[family],
                    deterministic_key("S3-family-order", attempt, stratum, family),
                ),
            )
            for family in families:
                if family_counts[family] >= family_limit:
                    continue
                queue = by_stratum_family[stratum][family]
                while queue and queue[0]["root_cluster"] in selected_roots:
                    queue.pop(0)
                if not queue:
                    continue
                row = queue.pop(0)
                selected.append(row)
                selected_roots.add(row["root_cluster"])
                family_counts[family] += 1
                stratum_counts[stratum] += 1
                progressed = True
                break
        if not progressed:
            return None
    if len(family_counts) < 3:
        return None
    if max(family_counts.values(), default=0) > family_limit:
        return None
    return selected


def feasible_designs(
    records: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[int, list[dict[str, Any]]]]:
    by_stratum = {
        stratum: [record for record in records if record["stratum"] == stratum]
        for stratum in STRATA
    }
    root_families: dict[str, dict[str, set[str]]] = {
        stratum: defaultdict(set) for stratum in STRATA
    }
    for stratum, rows in by_stratum.items():
        for row in rows:
            root_families[stratum][row["behavior_family"]].add(
                row["root_cluster"]
            )
    selections: dict[int, list[dict[str, Any]]] = {}
    checks = {}
    for root_count in ROOT_DESIGNS:
        roots_per_stratum_required = root_count // 2
        available_roots_by_stratum = {
            stratum: len(
                {row["root_cluster"] for row in by_stratum[stratum]}
            )
            for stratum in STRATA
        }
        pool_family_shares = {
            stratum: {
                family: (
                    len(roots) / available_roots_by_stratum[stratum]
                    if available_roots_by_stratum[stratum]
                    else 0.0
                )
                for family, roots in sorted(root_families[stratum].items())
            }
            for stratum in STRATA
        }
        selection = None
        for attempt in range(200):
            selection = _selection_attempt(records, root_count, attempt)
            if selection is not None:
                break
        if selection is not None:
            selections[root_count] = selection
        checks[str(root_count)] = {
            "feasible": selection is not None,
            "root_count_feasible": all(
                count >= roots_per_stratum_required
                for count in available_roots_by_stratum.values()
            ),
            "available_pool_family_cap_pass": all(
                max(shares.values(), default=0.0) <= FAMILY_CAP
                for shares in pool_family_shares.values()
            ),
            "roots_per_stratum_required": roots_per_stratum_required,
            "available_roots_by_stratum": available_roots_by_stratum,
            "available_root_families_by_stratum": {
                stratum: {
                    family: len(roots)
                    for family, roots in sorted(root_families[stratum].items())
                }
                for stratum in STRATA
            },
            "available_root_family_shares_by_stratum": pool_family_shares,
            "selected_family_counts": (
                dict(
                    sorted(
                        Counter(
                            row["behavior_family"] for row in selection
                        ).items()
                    )
                )
                if selection is not None
                else {}
            ),
        }
    return {"designs": checks}, selections


def independent_coherence_checks(
    power_rows: list[dict[str, Any]],
    feasibility: dict[str, Any],
    free_disk_bytes: int,
    selected_design: dict[str, Any] | None,
) -> dict[str, Any]:
    designs = feasibility["designs"]
    return {
        "power_or_1_50_ge_80pct": any(
            row["power"]["1.50"]["conditional_independent"][
                "power_lower_ci_gt_1"
            ]
            >= 0.80
            for row in power_rows
        ),
        "root_count_at_least_48_each_stratum": any(
            bool(details["root_count_feasible"])
            and int(details["roots_per_stratum_required"]) >= 48
            for details in designs.values()
        ),
        "available_pool_family_cap_40pct": all(
            bool(details["available_pool_family_cap_pass"])
            for details in designs.values()
        ),
        "joint_root_family_selection_feasible": any(
            bool(details["feasible"]) for details in designs.values()
        ),
        "compact_storage_below_10gib": all(
            int(row["runtime_storage"]["compact_storage_bytes"])
            < COMPACT_STORAGE_LIMIT_BYTES
            for row in power_rows
        ),
        "free_disk_above_100gib": free_disk_bytes >= FREE_DISK_MIN_BYTES,
        "selected_coherent_design": selected_design is not None,
    }


def _bootstrap_weights(
    rng: np.random.Generator,
    roots_per_stratum: int,
    replicates: int,
) -> np.ndarray:
    indices = rng.integers(
        0,
        roots_per_stratum,
        size=(replicates, roots_per_stratum),
    )
    weights = np.zeros(
        (replicates, roots_per_stratum),
        dtype=np.float64,
    )
    for index, row in enumerate(indices):
        weights[index] = np.bincount(
            row, minlength=roots_per_stratum
        ) / roots_per_stratum
    return weights


def simulate_power(
    calibration: dict[str, Any],
    roots: int,
    repeats: int,
    odds_ratio: float,
    *,
    draws: int = SIMULATION_DRAWS,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    coupling: str = "conditional_independent",
    seed: int = 20260725,
) -> dict[str, Any]:
    if roots % 2:
        raise ValueError("S3 root count must be even")
    if coupling not in {"conditional_independent", "shared_uniform"}:
        raise ValueError(f"Unsupported coupling: {coupling}")
    roots_per_stratum = roots // 2
    rng = np.random.default_rng(
        seed + roots * 10_000 + repeats * 100 + int(round(odds_ratio * 100))
        + (1 if coupling == "shared_uniform" else 0)
    )
    weights = {
        stratum: _bootstrap_weights(
            rng, roots_per_stratum, bootstrap_replicates
        )
        for stratum in STRATA
    }
    significant = 0
    useful_significant = 0
    estimates: list[float] = []
    chunk_size = 100
    for start in range(0, draws, chunk_size):
        chunk = min(chunk_size, draws - start)
        arm_rates: dict[str, dict[str, np.ndarray]] = {
            "control": {},
            "treatment": {},
        }
        for stratum in STRATA:
            details = calibration["strata"][stratum]
            root_probability = rng.beta(
                float(details["beta_alpha"]),
                float(details["beta_beta"]),
                size=(chunk, roots_per_stratum),
            )
            treatment_probability = odds_shift(root_probability, odds_ratio)
            if coupling == "shared_uniform":
                uniforms = rng.random(
                    (chunk, roots_per_stratum, repeats)
                )
                control = np.mean(
                    uniforms < root_probability[:, :, None], axis=2
                )
                treatment = np.mean(
                    uniforms < treatment_probability[:, :, None], axis=2
                )
            else:
                control = rng.binomial(repeats, root_probability) / repeats
                treatment = (
                    rng.binomial(repeats, treatment_probability) / repeats
                )
            arm_rates["control"][stratum] = control
            arm_rates["treatment"][stratum] = treatment

        observed_effects = np.zeros(chunk, dtype=np.float64)
        bootstrap_effects = np.zeros(
            (chunk, bootstrap_replicates), dtype=np.float64
        )
        for stratum in STRATA:
            control = arm_rates["control"][stratum]
            treatment = arm_rates["treatment"][stratum]
            control_mean = np.mean(control, axis=1)
            treatment_mean = np.mean(treatment, axis=1)
            control_boundary = (control_mean <= 0.0) | (control_mean >= 1.0)
            treatment_boundary = (
                (treatment_mean <= 0.0) | (treatment_mean >= 1.0)
            )
            control_mean[control_boundary] = (
                np.sum(control[control_boundary], axis=1) * repeats + 0.5
            ) / (roots_per_stratum * repeats + 1.0)
            treatment_mean[treatment_boundary] = (
                np.sum(treatment[treatment_boundary], axis=1) * repeats + 0.5
            ) / (roots_per_stratum * repeats + 1.0)
            observed_effects += 0.5 * (
                np.log(treatment_mean / (1.0 - treatment_mean))
                - np.log(control_mean / (1.0 - control_mean))
            )

            control_boot = control @ weights[stratum].T
            treatment_boot = treatment @ weights[stratum].T
            control_boot = np.clip(
                control_boot,
                0.5 / (roots_per_stratum * repeats + 1.0),
                1.0 - 0.5 / (roots_per_stratum * repeats + 1.0),
            )
            treatment_boot = np.clip(
                treatment_boot,
                0.5 / (roots_per_stratum * repeats + 1.0),
                1.0 - 0.5 / (roots_per_stratum * repeats + 1.0),
            )
            bootstrap_effects += 0.5 * (
                np.log(treatment_boot / (1.0 - treatment_boot))
                - np.log(control_boot / (1.0 - control_boot))
            )

        lower = np.quantile(bootstrap_effects, 0.025, axis=1)
        passes = lower > 0.0
        significant += int(np.count_nonzero(passes))
        useful_significant += int(
            np.count_nonzero(
                passes & (np.exp(observed_effects) >= 1.25)
            )
        )
        estimates.extend(observed_effects.tolist())

    power = significant / draws
    return {
        "roots": roots,
        "repeats_per_arm_root": repeats,
        "stream_blocks": {
            "A": repeats // 2,
            "B": repeats - repeats // 2,
        },
        "true_common_odds_ratio": odds_ratio,
        "coupling": coupling,
        "simulation_draws": draws,
        "ancestry_bootstrap_replicates": bootstrap_replicates,
        "power_lower_ci_gt_1": power,
        "power_pass_point_and_ci": useful_significant / draws,
        "monte_carlo_standard_error": math.sqrt(
            max(power * (1.0 - power), 0.0) / draws
        ),
        "median_estimated_odds_ratio": float(
            np.exp(np.median(np.asarray(estimates)))
        ),
    }


def mde_for_design(
    calibration: dict[str, Any],
    roots: int,
    repeats: int,
) -> dict[str, Any]:
    low = 1.0
    high = 4.0
    evaluations = []
    for iteration in range(9):
        odds_ratio = (low + high) / 2.0
        result = simulate_power(
            calibration,
            roots,
            repeats,
            odds_ratio,
            draws=SIMULATION_DRAWS,
            bootstrap_replicates=BOOTSTRAP_REPLICATES,
            coupling="conditional_independent",
            seed=20260825 + iteration * 100_000,
        )
        evaluations.append(
            {
                "odds_ratio": odds_ratio,
                "power": result["power_lower_ci_gt_1"],
            }
        )
        if result["power_lower_ci_gt_1"] >= 0.80:
            high = odds_ratio
        else:
            low = odds_ratio
    absolute_effects = {}
    for stratum in STRATA:
        base = float(
            calibration["strata"][stratum]["root_equal_base_rate"]
        )
        shifted = float(odds_shift(np.asarray([base]), high)[0])
        absolute_effects[stratum] = shifted - base
    return {
        "design": {"roots": roots, "repeats_per_arm_root": repeats},
        "mde_common_odds_ratio_80pct_power": high,
        "stratum_absolute_effects_at_mde": absolute_effects,
        "search_evaluations": evaluations,
    }


def runtime_and_storage_estimate(
    roots: int,
    repeats: int,
    source_inventory: dict[str, Any],
) -> dict[str, Any]:
    c1 = _json(C1_RUNTIME_PATH)
    optimized = np.asarray(
        [
            float(row["optimized_combined_s"])
            for row in c1["measurements"]
        ],
        dtype=np.float64,
    )
    label_estimate = source_inventory["future_label_estimate"]
    control_trajectory_seconds = (
        float(label_estimate["estimated_wall_seconds_at_measured_h0_throughput"])
        / float(label_estimate["trajectories"])
    )
    median_treatment_seconds = 40.0 * float(np.median(optimized))
    p90_treatment_seconds = 40.0 * float(np.quantile(optimized, 0.90))
    trajectories_per_arm = roots * repeats
    compact_storage_bytes = 8_192 * 2 * trajectories_per_arm
    return {
        "trajectories_per_arm": trajectories_per_arm,
        "total_trajectories": 2 * trajectories_per_arm,
        "control_h40_seconds_from_historical_labels": control_trajectory_seconds,
        "treatment_h40_seconds_all_40_decisions_triggered_median": median_treatment_seconds,
        "treatment_h40_seconds_all_40_decisions_triggered_p90": p90_treatment_seconds,
        "cpu_hours_median_upper_bound": (
            trajectories_per_arm
            * (control_trajectory_seconds + median_treatment_seconds)
            / 3600.0
        ),
        "cpu_hours_p90_upper_bound": (
            trajectories_per_arm
            * (control_trajectory_seconds + p90_treatment_seconds)
            / 3600.0
        ),
        "compact_storage_bytes": compact_storage_bytes,
        "compact_storage_gib": compact_storage_bytes / (1024**3),
    }


def run_preflight() -> dict[str, Any]:
    source_inventory = _json(SOURCE_INVENTORY_PATH)
    calibration = historical_control_calibration(
        source_inventory, HISTORICAL_LABELS_PATH
    )
    exclusions = prior_exclusion_catalog()
    candidates = natural_root_candidates(
        source_inventory, exclusions["roots"]
    )
    feasibility, selections = feasible_designs(candidates["records"])

    power_rows = []
    for roots in ROOT_DESIGNS:
        for repeats in REPEAT_DESIGNS:
            row = {
                "roots": roots,
                "repeats_per_arm_root": repeats,
                "root_availability_feasible": roots in selections,
                "runtime_storage": runtime_and_storage_estimate(
                    roots, repeats, source_inventory
                ),
                "power": {},
            }
            for odds_ratio in ODDS_RATIOS:
                row["power"][f"{odds_ratio:.2f}"] = {
                    coupling: simulate_power(
                        calibration,
                        roots,
                        repeats,
                        odds_ratio,
                        coupling=coupling,
                    )
                    for coupling in (
                        "conditional_independent",
                        "shared_uniform",
                    )
                }
            power_rows.append(row)

    disk_usage = shutil.disk_usage(DISK_CHECK_PATH)
    coherent = [
        row
        for row in power_rows
        if row["root_availability_feasible"]
        and row["power"]["1.50"]["conditional_independent"][
            "power_lower_ci_gt_1"
        ]
        >= 0.80
        and row["roots"] // 2 >= 48
        and row["runtime_storage"]["compact_storage_bytes"]
        < COMPACT_STORAGE_LIMIT_BYTES
        and disk_usage.free >= FREE_DISK_MIN_BYTES
    ]
    coherent.sort(key=lambda row: (row["roots"], row["repeats_per_arm_root"]))
    selected_design = coherent[0] if coherent else None
    best_power_design = max(
        power_rows,
        key=lambda row: row["power"]["1.50"]["conditional_independent"][
            "power_lower_ci_gt_1"
        ],
    )
    mde = mde_for_design(
        calibration,
        int(best_power_design["roots"]),
        int(best_power_design["repeats_per_arm_root"]),
    )
    decision = (
        "READY_FOR_S3_OUTCOMES"
        if selected_design is not None
        else "HOLD_UNDERPOWERED_PREFLIGHT"
    )
    return {
        "preflight_version": VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "decision": decision,
        "outcomes_generated": False,
        "treatment_outcomes_inspected": False,
        "selection_uses_s3_outcome": False,
        "dashboard_eligible": False,
        "locks": {
            "charter": str(CHARTER_PATH),
            "charter_sha256": sha256_path(CHARTER_PATH),
            "source_inventory": str(SOURCE_INVENTORY_PATH),
            "source_inventory_sha256": sha256_path(SOURCE_INVENTORY_PATH),
            "historical_labels": str(HISTORICAL_LABELS_PATH),
            "historical_labels_sha256": sha256_path(HISTORICAL_LABELS_PATH),
            "r2a_roots_sha256": sha256_path(R2A_ROOTS_PATH),
            "c1_corpus_sha256": sha256_path(C1_CORPUS_PATH),
            "c1_runtime_sha256": sha256_path(C1_RUNTIME_PATH),
        },
        "power_contract": {
            "draws_per_candidate_effect": SIMULATION_DRAWS,
            "ancestry_bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "primary_coupling": (
                "conditional-independent arm residuals with shared latent "
                "root propensity; conservative because cross-arm endpoint "
                "correlation is not identifiable from historical control-only data"
            ),
            "optimistic_sensitivity": (
                "shared-uniform monotone arm coupling; never sufficient by itself"
            ),
            "coherence_power": 0.80,
            "coherence_common_odds_ratio": 1.50,
        },
        "storage_headroom": {
            "path": str(DISK_CHECK_PATH),
            "free_bytes": disk_usage.free,
            "free_gib": disk_usage.free / (1024**3),
            "minimum_bytes": FREE_DISK_MIN_BYTES,
            "minimum_gib": FREE_DISK_MIN_BYTES / (1024**3),
            "target_bytes": FREE_DISK_TARGET_BYTES,
            "target_gib": FREE_DISK_TARGET_BYTES / (1024**3),
            "minimum_pass": disk_usage.free >= FREE_DISK_MIN_BYTES,
            "target_pass": disk_usage.free >= FREE_DISK_TARGET_BYTES,
        },
        "historical_control_calibration": calibration,
        "exclusion_catalog": {
            "counts": exclusions["counts"],
            "union_count": exclusions["union_count"],
            "root_list_sha256": hashlib.sha256(
                "\n".join(sorted(exclusions["roots"])).encode("utf-8")
            ).hexdigest(),
        },
        "candidate_catalog": {
            "counts": candidates["counts"],
            "records": len(candidates["records"]),
            "unique_roots": len(
                {row["root_cluster"] for row in candidates["records"]}
            ),
            "records_by_stratum": dict(
                sorted(
                    Counter(
                        row["stratum"] for row in candidates["records"]
                    ).items()
                )
            ),
            "roots_by_family_and_stratum": {
                stratum: {
                    family: len(
                        {
                            row["root_cluster"]
                            for row in candidates["records"]
                            if row["stratum"] == stratum
                            and row["behavior_family"] == family
                        }
                    )
                    for family in sorted(
                        {
                            row["behavior_family"]
                            for row in candidates["records"]
                            if row["stratum"] == stratum
                        }
                    )
                }
                for stratum in STRATA
            },
            "role_counts": dict(
                sorted(
                    Counter(row["role"] for row in candidates["records"]).items()
                )
            ),
        },
        "availability": feasibility,
        "power_designs": power_rows,
        "best_candidate_mde": mde,
        "selected_design": selected_design,
        "coherence_checks": independent_coherence_checks(
            power_rows,
            feasibility,
            disk_usage.free,
            selected_design,
        ),
        "outcome_free_remedy": (
            None
            if selected_design is not None
            else {
                "required": True,
                "next_step": (
                    "Acquire or expose additional independent natural roots and "
                    "redesign the estimator around a less extreme but still direct "
                    "progression endpoint before any S3 outcome."
                ),
                "do_not": (
                    "Do not run the underpowered assay, relax the family cap, "
                    "increase repeats in place of roots, or convert this hold to a kill."
                ),
            }
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = run_preflight()
    write_json(args.out, payload)
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "candidate_catalog": payload["candidate_catalog"],
                "availability": payload["availability"],
                "best_candidate_mde": payload["best_candidate_mde"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
