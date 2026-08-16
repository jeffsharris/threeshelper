"""Small rare-event frontier pilot for support-ladder transitions.

This is a diagnostic runner, not a policy wrapper.  It starts from state-record
artifacts with explicit root provenance, branches every legal first action, and
uses common-random-number stochastic continuations to estimate whether any
action increases the probability of reaching a support-ladder milestone within a
short horizon.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import numpy as np

from threes_rl.eval import make_policy, max_tile_excluding_initial_starter
from threes_rl.record_replay import state_payload
from threes_rl.replay_provenance import GENUINE_ROOT_ORIGINS, provenance_fields_from_record
from threes_rl.run_artifacts import safe_name, write_json
from threes_rl.sim import DIRECTION_NAMES, SimState, ThreesSim, direction_index, score_board
from threes_rl.support_ladder_window_reservoir import milestone_specs, raw_ladder_features
from threes_rl.swing_label import state_features
from threes_rl.train_td import copy_state, state_from_replay_payload

EXTRA_FRONTIER_TARGETS = ("reached_1536", "reached_3072", "reached_6144")
FIRST_ACTION_MODES = ("all", "top-two", "recorded", "recorded-plus-top-two")


@dataclass
class FrontierCase:
    id: str
    state: SimState
    starter_tile: int | None
    source_replay: str | None
    source_seed: int | None
    source_frame_index: int | None
    source_policy: str | None
    root_origin: str
    root_replay: str | None
    root_seed: int | None
    root_frame_index: int | None
    root_policy: str | None
    root_policy_family: str | None
    ancestry_key: str | None
    features: dict[str, Any]
    raw: dict[str, Any]


def _int_or_none(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _load_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text())
    records = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError(f"{path} does not contain records[]")
    out: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        if isinstance(record, dict):
            row = dict(record)
            row.setdefault("_source_json", str(path))
            row.setdefault("_record_index", idx)
            out.append(row)
    return out


def load_records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        records.extend(_load_records(Path(path)))
    return records


def _record_starter(record: dict[str, Any], default: int | None) -> int | None:
    value = record.get("starter_tile", default)
    return None if value is None else int(value)


def _case_id(record: dict[str, Any]) -> str:
    if record.get("id") is not None:
        return str(record["id"])
    raw = json.dumps(
        {
            "source_replay": record.get("source_replay"),
            "source_frame_index": record.get("source_frame_index", record.get("frame_index")),
            "state": record.get("state"),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return safe_name(f"frontier_case_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:12]}")


def case_from_record(record: dict[str, Any], *, default_starter_tile: int | None) -> FrontierCase | None:
    state_payload_dict = record.get("state")
    if not isinstance(state_payload_dict, dict):
        return None
    state = state_from_replay_payload(state_payload_dict)
    if state.game_over:
        return None
    starter_tile = _record_starter(record, default_starter_tile)
    sim = ThreesSim(np.random.default_rng(_int_or_none(record.get("source_seed", record.get("seed"))) or 0), starter_tile=starter_tile)
    features = dict(state_features(state, sim, starter_tile))
    recorded_action = _recorded_action_from_record(record)
    if recorded_action is not None:
        features["recorded_action"] = recorded_action
    raw = dict(raw_ladder_features(state.board, starter_tile))
    provenance = provenance_fields_from_record(record, record.get("_source_json"))
    return FrontierCase(
        id=_case_id(record),
        state=copy_state(state),
        starter_tile=starter_tile,
        source_replay=provenance["source_replay"],
        source_seed=provenance["source_seed"],
        source_frame_index=provenance["source_frame_index"],
        source_policy=provenance["source_policy"],
        root_origin=str(provenance["root_origin"]),
        root_replay=provenance["root_replay"],
        root_seed=provenance["root_seed"],
        root_frame_index=provenance["root_frame_index"],
        root_policy=provenance["root_policy"],
        root_policy_family=provenance["root_policy_family"],
        ancestry_key=provenance["ancestry_key"],
        features=features,
        raw=raw,
    )


def _recorded_action_from_record(record: dict[str, Any]) -> str | None:
    for key in ("source_next_action", "recorded_action", "next_action"):
        value = record.get(key)
        if value is not None:
            return str(value)
    return None


def load_cases(
    records: Iterable[dict[str, Any]],
    *,
    default_starter_tile: int | None,
    root_origins: set[str],
    case_ids: set[str] | None = None,
) -> tuple[list[FrontierCase], dict[str, int]]:
    cases: list[FrontierCase] = []
    rejected: Counter[str] = Counter()
    for record in records:
        case = case_from_record(record, default_starter_tile=default_starter_tile)
        if case is None:
            rejected["bad_record"] += 1
            continue
        if case.root_origin not in root_origins:
            rejected[f"root_origin:{case.root_origin}"] += 1
            continue
        if case_ids is not None and case.id not in case_ids:
            rejected["case_id_filter"] += 1
            continue
        cases.append(case)
    if case_ids is not None:
        found = {case.id for case in cases}
        rejected["case_id_missing"] += len(case_ids - found)
    return cases, dict(rejected)


def _diversity_key(case: FrontierCase) -> tuple[str, str, str, int]:
    root = case.root_replay or case.ancestry_key or case.source_replay or case.id
    support = f"dup{case.raw.get('raw_highest_duplicate_tile', 0)}_adj{case.raw.get('raw_highest_adjacent_pair_tile', 0)}"
    air_bucket = str(int(case.features.get("empty_count", 0)) // 2)
    return (str(root), str(case.features.get("stratum")), support, int(air_bucket))


def select_diverse_cases(cases: list[FrontierCase], *, max_starts: int, seed: int) -> list[FrontierCase]:
    if max_starts <= 0 or len(cases) <= max_starts:
        return list(cases)
    rng = np.random.default_rng(seed)
    shuffled = [cases[int(idx)] for idx in rng.permutation(len(cases))]
    selected: list[FrontierCase] = []
    selected_ids: set[str] = set()
    seen: set[tuple[str, str, str, int]] = set()
    for case in shuffled:
        key = _diversity_key(case)
        if key in seen:
            continue
        selected.append(case)
        selected_ids.add(case.id)
        seen.add(key)
        if len(selected) >= max_starts:
            return selected
    for case in shuffled:
        if case.id not in selected_ids:
            selected.append(case)
            selected_ids.add(case.id)
            if len(selected) >= max_starts:
                break
    return selected


def _ranked_policy_actions(policy: object, case: FrontierCase, sim: ThreesSim, rng: np.random.Generator) -> list[int]:
    legal = [int(action) for action in sim.legal_actions(case.state)]
    if not legal:
        return []
    if hasattr(policy, "action_values"):
        rows = [(int(action), float(value)) for action, value in policy.action_values(case.state, sim)]
        legal_set = set(legal)
        rows = [(action, value) for action, value in rows if action in legal_set]
        rows.sort(key=lambda item: (-float(item[1]), int(item[0])))
        return [int(action) for action, _value in rows]
    action = int(policy(case.state, sim, rng))
    if action in legal:
        return [action]
    return legal[:1]


def _recorded_action_index(case: FrontierCase, legal_actions: list[int]) -> int | None:
    name = case.features.get("recorded_action")
    if name is None:
        return None
    try:
        action = int(direction_index(str(name)))
    except ValueError:
        return None
    return action if action in set(legal_actions) else None


def _dedupe_actions(actions: list[int]) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for action in actions:
        if int(action) in seen:
            continue
        out.append(int(action))
        seen.add(int(action))
    return out


def select_first_actions(
    *,
    policy: object,
    case: FrontierCase,
    sim: ThreesSim,
    mode: str,
    rng: np.random.Generator,
) -> list[int]:
    if mode not in FIRST_ACTION_MODES:
        raise ValueError(f"Unsupported first-action mode: {mode}")
    legal = [int(action) for action in sim.legal_actions(case.state)]
    if mode == "all" or not legal:
        return legal
    ranked = _ranked_policy_actions(policy, case, sim, rng)
    recorded = _recorded_action_index(case, legal)
    if mode == "top-two":
        return _dedupe_actions(ranked[:2])
    if mode == "recorded":
        return [] if recorded is None else [recorded]
    if mode == "recorded-plus-top-two":
        actions = ([] if recorded is None else [recorded]) + ranked[:2]
        return _dedupe_actions(actions)
    raise AssertionError(f"Unhandled first-action mode: {mode}")


def target_reached(state: SimState, starter_tile: int | None, target: str) -> bool:
    if target == "reached_1536":
        return max_tile_excluding_initial_starter(state.board, starter_tile) >= 1536
    if target == "reached_3072":
        return max_tile_excluding_initial_starter(state.board, starter_tile) >= 3072
    if target == "reached_6144":
        return max_tile_excluding_initial_starter(state.board, starter_tile) >= 6144
    specs = milestone_specs()
    if target not in specs or target == "first_3072":
        raise ValueError(f"Unsupported frontier target: {target}")
    return bool(specs[target].predicate(raw_ladder_features(state.board, starter_tile)))


def supported_frontier_targets() -> list[str]:
    return [name for name in milestone_specs() if name != "first_3072"] + list(EXTRA_FRONTIER_TARGETS)


def _state_digest(state: SimState) -> str:
    raw = json.dumps(
        {
            "board": np.asarray(state.board, dtype=int).tolist(),
            "preview": state.preview.label,
            "tile_cycle": {
                "small_counts": state.small_counts,
                "small_pos": state.small_pos,
                "span_small_pos": state.span_small_pos,
                "large_pending": state.large_pending,
            },
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def frontier_progress_key(
    *,
    policy_name: str,
    target: str,
    horizon: int,
    case: FrontierCase,
    first_action: int,
    repeat_index: int,
    seed: int,
) -> str:
    raw = json.dumps(
        {
            "version": 1,
            "policy": str(policy_name),
            "target": str(target),
            "horizon": int(horizon),
            "seed": int(seed),
            "first_action": DIRECTION_NAMES[int(first_action)],
            "repeat_index": int(repeat_index),
            "case": {
                "id": case.id,
                "source_replay": case.source_replay,
                "source_frame_index": case.source_frame_index,
                "root_origin": case.root_origin,
                "root_replay": case.root_replay,
                "root_seed": case.root_seed,
                "root_frame_index": case.root_frame_index,
                "starter_tile": case.starter_tile,
                "move_count": int(case.state.move_count),
                "preview": case.state.preview.label,
                "board": [int(value) for value in np.asarray(case.state.board, dtype=np.int32).reshape(-1)],
            },
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _load_frontier_progress(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "version": 1,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "updated_at": None,
            "entries": {},
        }
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a frontier progress object")
    entries = payload.setdefault("entries", {})
    if not isinstance(entries, dict):
        raise ValueError(f"{path} has invalid frontier entries")
    payload.setdefault("version", 1)
    return payload


def _write_frontier_progress(path: Path, progress: dict[str, Any]) -> None:
    progress["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, progress)


def _progress_entry_for_rollout(
    *,
    key: str,
    policy_name: str,
    horizon: int,
    rollout: dict[str, Any],
    frontier_record: dict[str, Any],
) -> dict[str, Any]:
    return {
        "key": key,
        "policy": str(policy_name),
        "target": str(rollout["target"]),
        "horizon": int(horizon),
        "seed": int(rollout["seed"]),
        "first_action": str(rollout["first_action"]),
        "repeat_index": int(rollout["repeat_index"]),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "rollout": rollout,
        "frontier_record": frontier_record,
    }


def _rollout_from_progress_entry(entry: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    rollout = entry.get("rollout")
    frontier_record = entry.get("frontier_record")
    if not isinstance(rollout, dict) or not isinstance(frontier_record, dict):
        raise ValueError("invalid frontier progress entry")
    return rollout, frontier_record


def rollout_branch(
    *,
    case: FrontierCase,
    policy: object,
    first_action: int,
    repeat_index: int,
    seed: int,
    horizon: int,
    target: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    sim = ThreesSim(np.random.default_rng(seed), starter_tile=case.starter_tile)
    policy_rng = np.random.default_rng(seed + 37)
    state = copy_state(case.state)
    start_score = int(score_board(state.board))
    start_move = int(state.move_count)
    reached_at: int | None = 0 if target_reached(state, case.starter_tile, target) else None
    invalid = False
    actions: list[str] = []

    for step_idx in range(int(horizon)):
        before = state
        action = int(first_action) if step_idx == 0 else int(policy(before, sim, policy_rng))
        if action not in sim.legal_actions(before):
            if step_idx == 0:
                invalid = True
                break
            legal = sim.legal_actions(before)
            if not legal:
                break
            action = int(legal[0])
        state, info = sim.step(before, action)
        if not info.moved:
            legal = sim.legal_actions(before)
            if not legal:
                break
            action = int(legal[0])
            state, info = sim.step(before, action)
            if not info.moved:
                break
        actions.append(DIRECTION_NAMES[action])
        if reached_at is None and target_reached(state, case.starter_tile, target):
            reached_at = int(state.move_count - start_move)
            break
        if state.game_over:
            break

    final_score = int(score_board(state.board))
    final_features = state_features(state, sim, case.starter_tile)
    final_raw = raw_ladder_features(state.board, case.starter_tile)
    rollout = {
        "case_id": case.id,
        "target": target,
        "first_action": DIRECTION_NAMES[int(first_action)],
        "repeat_index": int(repeat_index),
        "seed": int(seed),
        "invalid_first_action": bool(invalid),
        "target_reached": reached_at is not None,
        "moves_to_target": reached_at,
        "moves_delta": int(state.move_count - start_move),
        "score_delta": int(final_score - start_score),
        "final_score": final_score,
        "final_max_tile_excl_starter": int(max_tile_excluding_initial_starter(state.board, case.starter_tile)),
        "final_empty_count": int(final_features["empty_count"]),
        "final_raw_highest_duplicate_tile": int(final_raw["raw_highest_duplicate_tile"]),
        "final_raw_highest_adjacent_pair_tile": int(final_raw["raw_highest_adjacent_pair_tile"]),
        "game_over": bool(state.game_over),
        "actions": actions,
        "root_origin": case.root_origin,
        "root_replay": case.root_replay,
        "root_seed": case.root_seed,
        "root_policy_family": case.root_policy_family,
        "ancestry_key": case.ancestry_key,
        "source_replay": case.source_replay,
        "source_frame_index": case.source_frame_index,
        "start_move_count": int(start_move),
        "start_score": int(start_score),
        "start_phase": str(case.features["phase"]),
        "start_stratum": str(case.features["stratum"]),
        "start_raw_highest_duplicate_tile": int(case.raw["raw_highest_duplicate_tile"]),
        "start_raw_highest_adjacent_pair_tile": int(case.raw["raw_highest_adjacent_pair_tile"]),
    }
    frontier_record = {
        "id": safe_name(f"frontier_{case.id}_{DIRECTION_NAMES[int(first_action)]}_r{repeat_index}_{_state_digest(state)}"),
        "kind": "rare_event_frontier_state",
        "source_case_id": case.id,
        "target": target,
        "first_action": DIRECTION_NAMES[int(first_action)],
        "repeat_index": int(repeat_index),
        "target_reached": reached_at is not None,
        "starter_tile": case.starter_tile,
        "source_replay": case.source_replay,
        "source_seed": case.source_seed,
        "source_frame_index": case.source_frame_index,
        "source_policy": case.source_policy,
        "root_origin": case.root_origin,
        "root_replay": case.root_replay,
        "root_seed": case.root_seed,
        "root_frame_index": case.root_frame_index,
        "root_policy": case.root_policy,
        "root_policy_family": case.root_policy_family,
        "ancestry_key": case.ancestry_key,
        "move_count": int(state.move_count),
        "score": final_score,
        "score_delta": int(final_score - start_score),
        "max_tile_excl_starter": int(max_tile_excluding_initial_starter(state.board, case.starter_tile)),
        "features": {**final_features, **final_raw},
        "state": state_payload(state, sim),
    }
    return rollout, frontier_record


def summarize_rollouts(
    rollouts: list[dict[str, Any]],
    *,
    cases_total: int,
    cases_selected: int,
    target: str,
    horizon: int,
    repeats: int,
    policy_name: str,
    rejected: dict[str, int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for rollout in rollouts:
        groups[(str(rollout["case_id"]), str(rollout["first_action"]))].append(rollout)
    action_rows: list[dict[str, Any]] = []
    for (case_id, action), rows in sorted(groups.items()):
        valid = [row for row in rows if not row.get("invalid_first_action")]
        hits = [row for row in valid if row.get("target_reached")]
        deltas = [int(row["score_delta"]) for row in valid]
        action_rows.append(
            {
                "case_id": case_id,
                "first_action": action,
                "rollouts": len(rows),
                "valid_rollouts": len(valid),
                "target_hits": len(hits),
                "target_rate": len(hits) / len(valid) if valid else 0.0,
                "mean_score_delta": float(mean(deltas)) if deltas else 0.0,
                "max_score_delta": max(deltas) if deltas else 0,
                "root_replay": rows[0].get("root_replay") if rows else None,
                "root_policy_family": rows[0].get("root_policy_family") if rows else None,
                "start_stratum": rows[0].get("start_stratum") if rows else None,
            }
        )
    best_by_case: dict[str, dict[str, Any]] = {}
    for row in action_rows:
        current = best_by_case.get(str(row["case_id"]))
        if current is None or (float(row["target_rate"]), float(row["mean_score_delta"])) > (
            float(current["target_rate"]),
            float(current["mean_score_delta"]),
        ):
            best_by_case[str(row["case_id"])] = row
    valid_rollouts = [row for row in rollouts if not row.get("invalid_first_action")]
    hits = [row for row in valid_rollouts if row.get("target_reached")]
    return (
        {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "policy": policy_name,
            "target": target,
            "horizon": int(horizon),
            "repeats_per_action": int(repeats),
            "cases_total": int(cases_total),
            "cases_selected": int(cases_selected),
            "rollouts": len(rollouts),
            "valid_rollouts": len(valid_rollouts),
            "target_hits": len(hits),
            "target_rate": len(hits) / len(valid_rollouts) if valid_rollouts else 0.0,
            "roots_selected": len({str(row.get("root_replay")) for row in rollouts if row.get("root_replay") is not None}),
            "ancestries_selected": len({str(row.get("ancestry_key")) for row in rollouts if row.get("ancestry_key") is not None}),
            "by_root_origin": dict(Counter(str(row.get("root_origin", "unknown")) for row in rollouts)),
            "by_root_policy_family": dict(Counter(str(row.get("root_policy_family", "unknown")) for row in rollouts)),
            "cases_with_any_hit": len({str(row["case_id"]) for row in hits}),
            "best_action_mean_target_rate": float(mean(float(row["target_rate"]) for row in best_by_case.values()))
            if best_by_case
            else 0.0,
            "rejected": rejected,
        },
        action_rows,
    )


def run_frontier(
    *,
    records_json: list[Path],
    policy_name: str,
    target: str,
    horizon: int,
    repeats: int,
    max_starts: int,
    seed: int,
    root_origins: set[str],
    case_ids: set[str] | None,
    default_starter_tile: int | None,
    out_dir: Path,
    checkpoint_rollouts: bool = False,
    progress_json: Path | None = None,
    progress_every: int = 0,
    first_action_mode: str = "all",
) -> dict[str, Any]:
    if first_action_mode not in FIRST_ACTION_MODES:
        raise ValueError(f"Unsupported first-action mode: {first_action_mode}")
    records = load_records(records_json)
    cases, rejected = load_cases(
        records,
        default_starter_tile=default_starter_tile,
        root_origins=root_origins,
        case_ids=case_ids,
    )
    selected = select_diverse_cases(cases, max_starts=max_starts, seed=seed)
    if not selected:
        raise ValueError("No frontier cases matched the requested filters")
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = progress_json
    if progress_path is None and checkpoint_rollouts:
        progress_path = out_dir / "frontier_progress.json"
    progress_payload: dict[str, Any] | None = None
    progress_entries: dict[str, Any] | None = None
    if progress_path is not None:
        progress_path = Path(progress_path)
        progress_payload = _load_frontier_progress(progress_path)
        progress_entries_obj = progress_payload.setdefault("entries", {})
        if not isinstance(progress_entries_obj, dict):
            raise ValueError(f"{progress_path} has invalid frontier entries")
        progress_entries = progress_entries_obj

    policy = make_policy(policy_name)
    case_actions: list[tuple[int, FrontierCase, list[int]]] = []
    for case_idx, case in enumerate(selected):
        sim = ThreesSim(np.random.default_rng(seed + case_idx), starter_tile=case.starter_tile)
        action_rng = np.random.default_rng(seed + case_idx + 17)
        case_actions.append(
            (
                case_idx,
                case,
                select_first_actions(
                    policy=policy,
                    case=case,
                    sim=sim,
                    mode=first_action_mode,
                    rng=action_rng,
                ),
            )
        )
    total_work = sum(len(legal_actions) * int(repeats) for _, _, legal_actions in case_actions)
    rollouts: list[dict[str, Any]] = []
    frontier_records: list[dict[str, Any]] = []
    ran_rollouts = 0
    resumed_rollouts = 0
    for case_idx, case, legal_actions in case_actions:
        for repeat_idx in range(int(repeats)):
            crn_seed = int(seed) + case_idx * 100_003 + repeat_idx * 997
            for action in legal_actions:
                progress_key = frontier_progress_key(
                    policy_name=policy_name,
                    target=target,
                    horizon=horizon,
                    case=case,
                    first_action=int(action),
                    repeat_index=repeat_idx,
                    seed=crn_seed,
                )
                if progress_entries is not None:
                    entry = progress_entries.get(progress_key)
                    if isinstance(entry, dict):
                        rollout, frontier_record = _rollout_from_progress_entry(entry)
                        rollouts.append(rollout)
                        frontier_records.append(frontier_record)
                        resumed_rollouts += 1
                        if progress_every > 0 and len(rollouts) % int(progress_every) == 0:
                            hits = sum(1 for row in rollouts if row.get("target_reached") and not row.get("invalid_first_action"))
                            print(
                                "frontier_progress "
                                f"{len(rollouts)}/{total_work} "
                                f"hits={hits} ran={ran_rollouts} resumed={resumed_rollouts}",
                                flush=True,
                            )
                        continue

                rollout, frontier_record = rollout_branch(
                    case=case,
                    policy=policy,
                    first_action=int(action),
                    repeat_index=repeat_idx,
                    seed=crn_seed,
                    horizon=horizon,
                    target=target,
                )
                rollouts.append(rollout)
                frontier_records.append(frontier_record)
                ran_rollouts += 1
                if progress_entries is not None and progress_payload is not None and progress_path is not None:
                    progress_entries[progress_key] = _progress_entry_for_rollout(
                        key=progress_key,
                        policy_name=policy_name,
                        horizon=horizon,
                        rollout=rollout,
                        frontier_record=frontier_record,
                    )
                    _write_frontier_progress(progress_path, progress_payload)
                if progress_every > 0 and len(rollouts) % int(progress_every) == 0:
                    hits = sum(1 for row in rollouts if row.get("target_reached") and not row.get("invalid_first_action"))
                    print(
                        "frontier_progress "
                        f"{len(rollouts)}/{total_work} "
                        f"hits={hits} ran={ran_rollouts} resumed={resumed_rollouts}",
                        flush=True,
                    )

    summary, action_rows = summarize_rollouts(
        rollouts,
        cases_total=len(cases),
        cases_selected=len(selected),
        target=target,
        horizon=horizon,
        repeats=repeats,
        policy_name=policy_name,
        rejected=rejected,
    )
    summary["rollouts_planned"] = int(total_work)
    summary["rollouts_ran"] = int(ran_rollouts)
    summary["rollouts_resumed"] = int(resumed_rollouts)
    summary["checkpoint_rollouts"] = progress_path is not None
    summary["first_action_mode"] = str(first_action_mode)
    if progress_path is not None:
        summary["frontier_progress_json"] = str(progress_path)
    payload = {
        "version": 1,
        "kind": "rare_event_frontier",
        "records_json": [str(path) for path in records_json],
        "case_id_filter": sorted(case_ids) if case_ids is not None else None,
        "first_action_mode": str(first_action_mode),
        "summary": summary,
        "action_summary": action_rows,
        "rollouts": rollouts,
        "frontier_records": frontier_records,
    }
    write_json(out_dir / "rare_event_frontier.json", payload)
    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "action_summary.json", action_rows)
    write_json(out_dir / "rollouts.json", rollouts)
    write_json(out_dir / "frontier_records.json", frontier_records)
    return payload


def parse_root_origins(text: str | None) -> set[str]:
    if text is None or not text.strip():
        return set(GENUINE_ROOT_ORIGINS)
    origins = {part.strip() for part in text.split(",") if part.strip()}
    if not origins:
        raise ValueError("at least one root origin is required")
    return origins


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-json", type=Path, action="append", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument(
        "--target",
        choices=supported_frontier_targets(),
        default="raw_duplicate_1536",
    )
    parser.add_argument("--horizon", type=int, default=40)
    parser.add_argument("--repeats-per-action", type=int, default=8)
    parser.add_argument("--max-starts", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--root-origin", help="Comma-separated root origins; defaults to fresh,human.")
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Only run the named source case id. May be repeated.",
    )
    parser.add_argument("--starter", default="1536")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--first-action-mode", choices=FIRST_ACTION_MODES, default="all")
    parser.add_argument(
        "--checkpoint-rollouts",
        action="store_true",
        help="Checkpoint each completed rollout branch and reuse matching completed entries on rerun.",
    )
    parser.add_argument(
        "--progress-json",
        type=Path,
        help="Explicit resumable frontier progress JSON path; implies checkpoint/resume behavior.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("threes_rl/runs/forensics/rare_event_frontier/latest"))
    args = parser.parse_args()
    starter_text = args.starter.strip().lower()
    default_starter = None if starter_text == "none" else int(starter_text)
    payload = run_frontier(
        records_json=args.records_json,
        policy_name=args.policy,
        target=args.target,
        horizon=args.horizon,
        repeats=args.repeats_per_action,
        max_starts=args.max_starts,
        seed=args.seed,
        root_origins=parse_root_origins(args.root_origin),
        case_ids=set(args.case_id) if args.case_id else None,
        default_starter_tile=default_starter,
        out_dir=args.out_dir,
        checkpoint_rollouts=args.checkpoint_rollouts,
        progress_json=args.progress_json,
        progress_every=args.progress_every,
        first_action_mode=args.first_action_mode,
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"json={args.out_dir / 'rare_event_frontier.json'}")
    print(f"frontier_records={args.out_dir / 'frontier_records.json'}")


if __name__ == "__main__":
    main()
