"""Scan replay states for root-action differences between two policies."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np

from threes_rl.continue_from_replays import StartCase, collect_start_cases, select_start_cases
from threes_rl.eval import make_policy
from threes_rl.ntuple import corner_risk_bucket_for_board
from threes_rl.run_artifacts import write_json
from threes_rl.sim import DIRECTION_NAMES, ThreesSim
from threes_rl.train_td import parse_phase_filter


@dataclass
class DivergenceRecord:
    index: int
    source_replay: str
    source_seed: int | None
    frame_index: int
    phase: str
    corner_risk: str
    stratum: str
    start_score: int
    start_max_tile_excl_starter: int
    base_action: str
    candidate_action: str
    changed: bool
    base_top_two: list[str]
    candidate_top_two: list[str]
    top_two_changed: bool
    base_margin: float | None
    candidate_margin: float | None
    max_abs_value_delta: float | None
    mean_abs_value_delta: float | None
    base_values: dict[str, float] | None = None
    candidate_values: dict[str, float] | None = None


def _action_values(policy: object, state, sim: ThreesSim, rng: np.random.Generator) -> list[tuple[int, float]]:
    if hasattr(policy, "action_values"):
        return [(int(action), float(value)) for action, value in policy.action_values(state, sim)]
    action = int(policy(state, sim, rng))
    return [(action, 0.0)]


def _best_action(values: list[tuple[int, float]]) -> int:
    if not values:
        return 0
    best_value = max(value for _action, value in values)
    return min(action for action, value in values if value == best_value)


def _ranked_actions(values: list[tuple[int, float]]) -> list[int]:
    return [
        int(action)
        for action, _value in sorted(
            values,
            key=lambda item: (-float(item[1]), int(item[0])),
        )
    ]


def _named_actions(actions: list[int]) -> list[str]:
    return [DIRECTION_NAMES[int(action)] for action in actions]


def _margin(values: list[tuple[int, float]]) -> float | None:
    if len(values) < 2:
        return None
    sorted_values = sorted((float(value) for _action, value in values), reverse=True)
    return sorted_values[0] - sorted_values[1]


def _value_deltas(
    base_values: list[tuple[int, float]],
    candidate_values: list[tuple[int, float]],
) -> tuple[float | None, float | None]:
    base = {int(action): float(value) for action, value in base_values}
    candidate = {int(action): float(value) for action, value in candidate_values}
    shared = sorted(set(base) & set(candidate))
    if not shared:
        return None, None
    deltas = [abs(candidate[action] - base[action]) for action in shared]
    return max(deltas), mean(deltas)


def _named_values(values: list[tuple[int, float]]) -> dict[str, float]:
    return {DIRECTION_NAMES[int(action)]: float(value) for action, value in values}


def _case_corner_risk(case: StartCase) -> str:
    return corner_risk_bucket_for_board(case.state.board, starter_tile=case.starter_tile)


def _case_bucket_key(case: StartCase, sample_mode: str) -> str:
    if sample_mode == "phase_balanced":
        return case.phase
    if sample_mode == "stratum_balanced":
        return f"{case.phase}/{_case_corner_risk(case)}"
    raise ValueError(f"Unsupported balanced sample mode: {sample_mode}")


def select_scan_cases(
    cases: list[StartCase],
    *,
    max_states: int,
    seed: int,
    sample_mode: str = "flat",
) -> list[StartCase]:
    if max_states <= 0 or len(cases) <= max_states:
        return list(cases)
    if sample_mode == "flat":
        return select_start_cases(cases, max_starts=max_states, seed=seed)
    if sample_mode not in ("phase_balanced", "stratum_balanced"):
        raise ValueError(f"Unsupported sample mode: {sample_mode}")

    rng = np.random.default_rng(int(seed))
    buckets: dict[str, list[StartCase]] = {}
    for case in cases:
        buckets.setdefault(_case_bucket_key(case, sample_mode), []).append(case)
    for phase_cases in buckets.values():
        rng.shuffle(phase_cases)

    ordered_phases = sorted(buckets)
    selected: list[StartCase] = []
    cursor = {phase: 0 for phase in ordered_phases}
    while len(selected) < int(max_states):
        added = False
        for phase in ordered_phases:
            idx = cursor[phase]
            phase_cases = buckets[phase]
            if idx >= len(phase_cases):
                continue
            selected.append(phase_cases[idx])
            cursor[phase] += 1
            added = True
            if len(selected) >= int(max_states):
                break
        if not added:
            break
    return selected


def scan_cases(
    *,
    base_policy: object,
    candidate_policy: object,
    cases: list[StartCase],
    include_values: bool = False,
) -> list[DivergenceRecord]:
    records: list[DivergenceRecord] = []
    for idx, case in enumerate(cases):
        sim = ThreesSim(np.random.default_rng(10_000 + idx), starter_tile=case.starter_tile)
        base_rng = np.random.default_rng(20_000 + idx)
        candidate_rng = np.random.default_rng(20_000 + idx)
        base_values = _action_values(base_policy, case.state, sim, base_rng)
        candidate_values = _action_values(candidate_policy, case.state, sim, candidate_rng)
        base_action = _best_action(base_values)
        candidate_action = _best_action(candidate_values)
        base_top_two_actions = _ranked_actions(base_values)[:2]
        candidate_top_two_actions = _ranked_actions(candidate_values)[:2]
        top_two_changed = (
            len(base_top_two_actions) >= 2
            and len(candidate_top_two_actions) >= 2
            and base_top_two_actions != candidate_top_two_actions
        )
        max_delta, mean_delta = _value_deltas(base_values, candidate_values)
        corner_risk = _case_corner_risk(case)
        records.append(
            DivergenceRecord(
                index=idx,
                source_replay=case.source_replay,
                source_seed=case.source_seed,
                frame_index=int(case.frame_index),
                phase=case.phase,
                corner_risk=corner_risk,
                stratum=f"{case.phase}/{corner_risk}",
                start_score=int(case.start_score),
                start_max_tile_excl_starter=int(case.start_max_tile_excl_starter),
                base_action=DIRECTION_NAMES[base_action],
                candidate_action=DIRECTION_NAMES[candidate_action],
                changed=base_action != candidate_action,
                base_top_two=_named_actions(base_top_two_actions),
                candidate_top_two=_named_actions(candidate_top_two_actions),
                top_two_changed=top_two_changed,
                base_margin=_margin(base_values),
                candidate_margin=_margin(candidate_values),
                max_abs_value_delta=max_delta,
                mean_abs_value_delta=mean_delta,
                base_values=_named_values(base_values) if include_values else None,
                candidate_values=_named_values(candidate_values) if include_values else None,
            )
        )
    return records


def summarize_records(records: list[DivergenceRecord]) -> dict[str, object]:
    if not records:
        return {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "states_scanned": 0,
            "changed_actions": 0,
            "changed_fraction": 0.0,
            "changed_top_two": 0,
            "changed_top_two_fraction": 0.0,
        }
    base_margins = [record.base_margin for record in records if record.base_margin is not None]
    value_deltas = [record.max_abs_value_delta for record in records if record.max_abs_value_delta is not None]
    by_phase: dict[str, dict[str, object]] = {}
    by_stratum: dict[str, dict[str, object]] = {}
    for record in records:
        phase = by_phase.setdefault(record.phase, {"states": 0, "changed": 0, "top_two_changed": 0})
        phase["states"] = int(phase["states"]) + 1
        phase["changed"] = int(phase["changed"]) + int(record.changed)
        phase["top_two_changed"] = int(phase["top_two_changed"]) + int(record.top_two_changed)
        stratum = by_stratum.setdefault(record.stratum, {"states": 0, "changed": 0, "top_two_changed": 0})
        stratum["states"] = int(stratum["states"]) + 1
        stratum["changed"] = int(stratum["changed"]) + int(record.changed)
        stratum["top_two_changed"] = int(stratum["top_two_changed"]) + int(record.top_two_changed)
    for phase in by_phase.values():
        states = int(phase["states"])
        phase["changed_fraction"] = 0.0 if states == 0 else float(phase["changed"]) / float(states)
        phase["top_two_changed_fraction"] = 0.0 if states == 0 else float(phase["top_two_changed"]) / float(states)
    for stratum in by_stratum.values():
        states = int(stratum["states"])
        stratum["changed_fraction"] = 0.0 if states == 0 else float(stratum["changed"]) / float(states)
        stratum["top_two_changed_fraction"] = 0.0 if states == 0 else float(stratum["top_two_changed"]) / float(states)
    changed = [record for record in records if record.changed]
    top_two_changed = [record for record in records if record.top_two_changed]
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "states_scanned": len(records),
        "changed_actions": len(changed),
        "changed_fraction": len(changed) / len(records),
        "changed_top_two": len(top_two_changed),
        "changed_top_two_fraction": len(top_two_changed) / len(records),
        "mean_base_margin": float(mean(base_margins)) if base_margins else None,
        "median_base_margin": float(median(base_margins)) if base_margins else None,
        "max_abs_value_delta": float(max(value_deltas)) if value_deltas else None,
        "mean_max_abs_value_delta": float(mean(value_deltas)) if value_deltas else None,
        "by_phase": by_phase,
        "by_stratum": by_stratum,
        "changed_records": [asdict(record) for record in changed[:25]],
        "top_two_changed_records": [asdict(record) for record in top_two_changed[:25]],
    }


def run_from_args(args: argparse.Namespace) -> dict[str, object]:
    replay_paths = [path for group in args.replay_json for path in group]
    phase_filter = set(parse_phase_filter(args.phase_filter)) if args.phase_filter else None
    starter_text = args.starter.strip().lower()
    default_starter = None if starter_text == "none" else int(starter_text)
    cases = collect_start_cases(
        replay_paths,
        min_tile=args.start_state_min_tile,
        phase_filter=phase_filter,
        default_starter_tile=default_starter,
    )
    selected = select_scan_cases(cases, max_states=args.max_states, seed=args.seed, sample_mode=args.sample_mode)
    base_policy = make_policy(args.base_policy)
    candidate_policy = make_policy(args.candidate_policy)
    records = scan_cases(
        base_policy=base_policy,
        candidate_policy=candidate_policy,
        cases=selected,
        include_values=bool(args.include_values),
    )
    summary = summarize_records(records)
    summary.update(
        {
            "base_policy": args.base_policy,
            "candidate_policy": args.candidate_policy,
            "source_replays": [str(path) for path in replay_paths],
            "start_cases_total": len(cases),
            "max_states": int(args.max_states),
            "sample_mode": args.sample_mode,
            "start_state_min_tile": int(args.start_state_min_tile),
            "phase_filter": sorted(phase_filter) if phase_filter else None,
        }
    )
    if hasattr(base_policy, "summary_stats"):
        summary["base_policy_stats"] = base_policy.summary_stats()
    if hasattr(candidate_policy, "summary_stats"):
        summary["candidate_policy_stats"] = candidate_policy.summary_stats()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "summary.json", summary)
    write_json(args.out_dir / "records.json", [asdict(record) for record in records])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-policy", required=True)
    parser.add_argument("--candidate-policy", required=True)
    parser.add_argument(
        "--replay-json",
        type=Path,
        nargs="+",
        action="append",
        required=True,
        help="Replay JSONs or high-board reservoir JSONs with records[].state.",
    )
    parser.add_argument("--starter", default="1536")
    parser.add_argument("--start-state-min-tile", type=int, default=1536)
    parser.add_argument("--phase-filter", help="Comma-separated phase names/aliases for scanned replay states.")
    parser.add_argument("--max-states", type=int, default=40)
    parser.add_argument(
        "--sample-mode",
        choices=["flat", "phase_balanced", "stratum_balanced"],
        default="flat",
        help="Select replay states uniformly, by phase, or by phase/corner-risk stratum.",
    )
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--include-values", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("threes_rl/runs/forensics/policy_divergence/latest"))
    args = parser.parse_args()
    summary = run_from_args(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"summary={args.out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
