"""Label pre-near-failure support states with paired first-action rollouts."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from html import escape
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import numpy as np

from threes_rl.eval import make_policy
from threes_rl.rare_event_frontier import (
    FIRST_ACTION_MODES,
    FrontierCase,
    case_from_record,
    load_records,
    parse_root_origins,
    select_diverse_cases,
    select_first_actions,
)
from threes_rl.run_artifacts import write_json
from threes_rl.sim import DIRECTION_NAMES, SimState, ThreesSim, direction_index, score_board
from threes_rl.support_accumulation_frontier import _raw
from threes_rl.train_td import copy_state


ENDPOINT_ORDER = (
    "support_h5",
    "support_h10",
    "support_one768_h5",
    "support_one768_h10",
    "raw2_h20",
    "raw2_h40",
)


def _parse_int_list(text: str, *, default: tuple[int, ...]) -> tuple[int, ...]:
    if text is None or not str(text).strip():
        return default
    values: list[int] = []
    seen: set[int] = set()
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value < 0:
            raise ValueError(f"expected non-negative integer, got {value}")
        if value not in seen:
            values.append(value)
            seen.add(value)
    if not values:
        raise ValueError("at least one value is required")
    return tuple(values)


def _recorded_action(record: dict[str, Any], legal_actions: Iterable[int]) -> int | None:
    value = record.get("source_next_action") or record.get("recorded_action") or record.get("next_action")
    if value is None:
        return None
    try:
        action = int(direction_index(str(value)))
    except (ValueError, TypeError):
        return None
    legal = {int(action) for action in legal_actions}
    return action if action in legal else None


def _support_present(raw: dict[str, Any]) -> bool:
    return int(raw.get("raw_count_384", 0) or 0) > 0 or int(raw.get("raw_count_192", 0) or 0) > 0


def _snapshot(state: SimState, starter_tile: int | None, start_score: int, start_move: int) -> dict[str, Any]:
    raw = _raw(state, starter_tile)
    masked_1536 = int(raw.get("masked_count_1536", 0) or 0)
    support_present = _support_present(raw) and masked_1536 == 0
    support_one768 = support_present and int(raw.get("raw_count_768", 0) or 0) == 1
    raw2 = int(raw.get("raw_count_768", 0) or 0) >= 2 and masked_1536 == 0
    return {
        **raw,
        "score": int(score_board(state.board)),
        "score_delta": int(score_board(state.board) - start_score),
        "move_count": int(state.move_count),
        "moves_delta": int(state.move_count - start_move),
        "game_over": bool(state.game_over),
        "support_present_no1536": bool(support_present),
        "support_one768_no1536": bool(support_one768),
        "raw_two_768_no1536": bool(raw2),
    }


def rollout_branch(
    *,
    case: FrontierCase,
    policy: object,
    first_action: int,
    repeat_index: int,
    seed_block: int,
    sim_seed: int,
    policy_seed: int,
    horizons: tuple[int, ...],
) -> dict[str, Any]:
    sim = ThreesSim(np.random.default_rng(int(sim_seed)), starter_tile=case.starter_tile)
    policy_rng = np.random.default_rng(int(policy_seed))
    state = copy_state(case.state)
    start_score = int(score_board(state.board))
    start_move = int(state.move_count)
    horizon_set = {int(horizon) for horizon in horizons}
    max_horizon = max(horizon_set)
    snapshots: dict[str, dict[str, Any]] = {}
    actions: list[str] = []
    invalid = False

    for step_idx in range(max_horizon):
        if state.game_over or not sim.legal_actions(state):
            break
        action = int(first_action) if step_idx == 0 else int(policy(state, sim, policy_rng))
        legal = sim.legal_actions(state)
        if action not in legal:
            if step_idx == 0:
                invalid = True
                break
            action = int(legal[0])
        state, info = sim.step(state, action)
        if not info.moved:
            legal = sim.legal_actions(state)
            if not legal:
                break
            action = int(legal[0])
            state, info = sim.step(state, action)
            if not info.moved:
                break
        actions.append(DIRECTION_NAMES[int(action)])
        completed = step_idx + 1
        if completed in horizon_set:
            snapshots[str(completed)] = _snapshot(state, case.starter_tile, start_score, start_move)

    final_snapshot = _snapshot(state, case.starter_tile, start_score, start_move)
    for horizon in horizons:
        snapshots.setdefault(str(int(horizon)), dict(final_snapshot))

    return {
        "case_id": case.id,
        "root_seed": case.root_seed,
        "root_origin": case.root_origin,
        "root_replay": case.root_replay,
        "ancestry_key": case.ancestry_key,
        "source_replay": case.source_replay,
        "source_seed": case.source_seed,
        "source_frame_index": case.source_frame_index,
        "window_offset": case.features.get("window_offset"),
        "start_move_count": int(start_move),
        "start_score": int(start_score),
        "first_action": DIRECTION_NAMES[int(first_action)],
        "first_action_index": int(first_action),
        "repeat_index": int(repeat_index),
        "seed_block": int(seed_block),
        "sim_seed": int(sim_seed),
        "policy_seed": int(policy_seed),
        "invalid_first_action": bool(invalid),
        "moves_delta": int(final_snapshot["moves_delta"]),
        "score_delta": int(final_snapshot["score_delta"]),
        "game_over": bool(final_snapshot["game_over"]),
        "actions": actions,
        "horizon_metrics": snapshots,
    }


def _rate(rows: list[dict[str, Any]], horizon: int, key: str) -> float:
    valid = [row for row in rows if not row.get("invalid_first_action")]
    if not valid:
        return 0.0
    return sum(bool((row.get("horizon_metrics") or {}).get(str(horizon), {}).get(key)) for row in valid) / float(
        len(valid)
    )


def _mean_metric(rows: list[dict[str, Any]], horizon: int, key: str) -> float:
    values = [
        float((row.get("horizon_metrics") or {}).get(str(horizon), {}).get(key, 0.0))
        for row in rows
        if not row.get("invalid_first_action")
    ]
    return float(mean(values)) if values else 0.0


def _endpoint_rates(rows: list[dict[str, Any]], *, support_horizons: tuple[int, ...], raw2_horizons: tuple[int, ...]) -> dict[str, float]:
    out: dict[str, float] = {}
    for horizon in support_horizons:
        out[f"support_h{int(horizon)}"] = _rate(rows, int(horizon), "support_present_no1536")
        out[f"support_one768_h{int(horizon)}"] = _rate(rows, int(horizon), "support_one768_no1536")
    for horizon in raw2_horizons:
        out[f"raw2_h{int(horizon)}"] = _rate(rows, int(horizon), "raw_two_768_no1536")
    return out


def _action_summary(
    *,
    case: FrontierCase,
    action_name: str,
    seed_block: int,
    rows: list[dict[str, Any]],
    support_horizons: tuple[int, ...],
    raw2_horizons: tuple[int, ...],
) -> dict[str, Any]:
    valid = [row for row in rows if not row.get("invalid_first_action")]
    max_horizon = max(tuple(support_horizons) + tuple(raw2_horizons))
    return {
        "case_id": case.id,
        "root_seed": case.root_seed,
        "window_offset": case.features.get("window_offset"),
        "first_action": action_name,
        "seed_block": int(seed_block),
        "rollouts": len(rows),
        "valid_rollouts": len(valid),
        **_endpoint_rates(rows, support_horizons=support_horizons, raw2_horizons=raw2_horizons),
        "mean_score_delta_hmax": _mean_metric(rows, int(max_horizon), "score_delta"),
        "terminal_rate_hmax": sum(bool(row.get("game_over")) for row in valid) / float(len(valid)) if valid else 0.0,
    }


def _best_unique_action(rows: list[dict[str, Any]], endpoint: str) -> str | None:
    if not rows:
        return None
    values = [(str(row["first_action"]), float(row.get(endpoint, 0.0)), float(row.get("mean_score_delta_hmax", 0.0))) for row in rows]
    best_rate = max(rate for _action, rate, _score in values)
    tied = [(action, score) for action, rate, score in values if rate == best_rate]
    if len(tied) != 1:
        return None
    return tied[0][0]


def _best_action(rows: list[dict[str, Any]], endpoint: str) -> str | None:
    if not rows:
        return None
    return str(
        max(
            rows,
            key=lambda row: (
                float(row.get(endpoint, 0.0)),
                float(row.get("mean_score_delta_hmax", 0.0)),
                str(row.get("first_action")),
            ),
        )["first_action"]
    )


def _case_summaries(
    *,
    case: FrontierCase,
    record: dict[str, Any],
    action_rows: list[dict[str, Any]],
    base_action_name: str | None,
    support_endpoint: str,
    raw2_endpoint: str,
) -> list[dict[str, Any]]:
    rows_by_block: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in action_rows:
        rows_by_block[int(row["seed_block"])].append(row)
    out: list[dict[str, Any]] = []
    for block, rows in sorted(rows_by_block.items()):
        by_action = {str(row["first_action"]): row for row in rows}
        support_best = _best_action(rows, support_endpoint)
        raw2_best = _best_action(rows, raw2_endpoint)
        support_best_unique = _best_unique_action(rows, support_endpoint)
        raw2_best_unique = _best_unique_action(rows, raw2_endpoint)
        base_row = by_action.get(str(base_action_name)) if base_action_name is not None else None
        support_row = by_action.get(str(support_best)) if support_best is not None else None
        raw2_row = by_action.get(str(raw2_best)) if raw2_best is not None else None
        out.append(
            {
                "case_id": case.id,
                "root_seed": case.root_seed,
                "window_offset": case.features.get("window_offset"),
                "support_bucket": record.get("support_bucket"),
                "base_action": base_action_name,
                "seed_block": int(block),
                "support_endpoint": support_endpoint,
                "raw2_endpoint": raw2_endpoint,
                "support_best_action": support_best,
                "support_best_unique": support_best_unique,
                "raw2_best_action": raw2_best,
                "raw2_best_unique": raw2_best_unique,
                "support_best_rate": float(support_row.get(support_endpoint, 0.0)) if support_row else 0.0,
                "base_support_rate": float(base_row.get(support_endpoint, 0.0)) if base_row else None,
                "support_gain_vs_base": float(support_row.get(support_endpoint, 0.0)) - float(base_row.get(support_endpoint, 0.0))
                if support_row and base_row
                else None,
                "support_best_raw2_rate": float(support_row.get(raw2_endpoint, 0.0)) if support_row else 0.0,
                "base_raw2_rate": float(base_row.get(raw2_endpoint, 0.0)) if base_row else None,
                "support_best_raw2_gain_vs_base": float(support_row.get(raw2_endpoint, 0.0)) - float(base_row.get(raw2_endpoint, 0.0))
                if support_row and base_row
                else None,
                "raw2_best_rate": float(raw2_row.get(raw2_endpoint, 0.0)) if raw2_row else 0.0,
                "raw2_oracle_gain_vs_base": float(raw2_row.get(raw2_endpoint, 0.0)) - float(base_row.get(raw2_endpoint, 0.0))
                if raw2_row and base_row
                else None,
            }
        )
    return out


def _cluster_bootstrap_ci(values_by_root: dict[str, float], *, seed: int, resamples: int = 2000) -> dict[str, float | None]:
    if not values_by_root:
        return {"mean": None, "ci_low": None, "ci_high": None, "roots": 0}
    roots = sorted(values_by_root)
    arr = np.asarray([float(values_by_root[root]) for root in roots], dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    means = []
    for _ in range(int(resamples)):
        sample = arr[rng.integers(0, len(arr), size=len(arr))]
        means.append(float(sample.mean()))
    return {
        "mean": float(arr.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "roots": int(len(roots)),
    }


def _winner_stability(case_rows: list[dict[str, Any]], endpoints: list[str]) -> dict[str, Any]:
    by_case: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in case_rows:
        by_case[str(row["case_id"])][int(row["seed_block"])] = row
    out: dict[str, Any] = {}
    for endpoint in endpoints:
        field = "support_best_unique" if endpoint.startswith("support") else "raw2_best_unique"
        if endpoint.startswith("support") and endpoint not in {row.get("support_endpoint") for row in case_rows}:
            continue
        if endpoint.startswith("raw2") and endpoint not in {row.get("raw2_endpoint") for row in case_rows}:
            continue
        non_tie = 0
        stable = 0
        by_actions: Counter[str] = Counter()
        for block_rows in by_case.values():
            actions = [str(row.get(field)) for row in block_rows.values() if row.get(field)]
            if len(actions) < 2:
                continue
            non_tie += 1
            if len(set(actions)) == 1:
                stable += 1
                by_actions[actions[0]] += 1
        out[endpoint] = {
            "non_tie_cases": int(non_tie),
            "stable_cases": int(stable),
            "stable_fraction": stable / float(non_tie) if non_tie else None,
            "stable_action_counts": dict(by_actions),
        }
    return out


def summarize(
    *,
    records_total: int,
    cases_total: int,
    cases_selected: int,
    rejected: dict[str, int],
    action_rows: list[dict[str, Any]],
    case_rows: list[dict[str, Any]],
    rollouts: list[dict[str, Any]],
    support_horizons: tuple[int, ...],
    raw2_horizons: tuple[int, ...],
    support_endpoint: str,
    raw2_endpoint: str,
    seed: int,
) -> dict[str, Any]:
    offsets = Counter(str(row.get("window_offset")) for row in case_rows)
    roots = {str(row.get("root_seed")) for row in case_rows if row.get("root_seed") is not None}
    root_values_by_block: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    root_values_all: dict[str, list[float]] = defaultdict(list)
    support_gains_all: list[float] = []
    oracle_gains_all: list[float] = []
    for row in case_rows:
        root = str(row.get("root_seed") or row.get("case_id"))
        block = int(row["seed_block"])
        diff = row.get("support_best_raw2_gain_vs_base")
        oracle = row.get("raw2_oracle_gain_vs_base")
        if diff is not None:
            root_values_by_block[block][root].append(float(diff))
            root_values_all[root].append(float(diff))
            support_gains_all.append(float(diff))
        if oracle is not None:
            oracle_gains_all.append(float(oracle))

    root_mean_by_block = {
        block: {root: float(mean(values)) for root, values in values_by_root.items()}
        for block, values_by_root in root_values_by_block.items()
    }
    root_mean_all = {root: float(mean(values)) for root, values in root_values_all.items()}
    block_cis = {
        str(block): _cluster_bootstrap_ci(values, seed=seed + int(block) * 101)
        for block, values in sorted(root_mean_by_block.items())
    }
    all_ci = _cluster_bootstrap_ci(root_mean_all, seed=seed + 999)
    block_signs = {
        str(block): 1 if (ci["mean"] or 0.0) > 0 else -1 if (ci["mean"] or 0.0) < 0 else 0
        for block, ci in block_cis.items()
    }

    endpoints = [f"support_h{h}" for h in support_horizons] + [f"support_one768_h{h}" for h in support_horizons] + [
        f"raw2_h{h}" for h in raw2_horizons
    ]
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "records_total": int(records_total),
        "cases_total": int(cases_total),
        "cases_selected": int(cases_selected),
        "unique_roots": int(len(roots)),
        "case_rows": int(len(case_rows)),
        "action_rows": int(len(action_rows)),
        "rollouts": int(len(rollouts)),
        "valid_rollouts": int(sum(1 for row in rollouts if not row.get("invalid_first_action"))),
        "offset_counts": dict(offsets),
        "seed_blocks": sorted({int(row["seed_block"]) for row in action_rows}),
        "support_horizons": [int(v) for v in support_horizons],
        "raw2_horizons": [int(v) for v in raw2_horizons],
        "support_endpoint_for_mediation": support_endpoint,
        "raw2_endpoint_for_mediation": raw2_endpoint,
        "support_best_raw2_gain_vs_base": {
            "mean_over_case_blocks": float(mean(support_gains_all)) if support_gains_all else None,
            "positive_case_blocks": int(sum(1 for value in support_gains_all if value > 0.0)),
            "negative_case_blocks": int(sum(1 for value in support_gains_all if value < 0.0)),
            "zero_case_blocks": int(sum(1 for value in support_gains_all if value == 0.0)),
            "root_cluster_ci_all_blocks": all_ci,
            "root_cluster_ci_by_block": block_cis,
            "block_mean_signs": block_signs,
        },
        "raw2_oracle_gain_vs_base": {
            "mean_over_case_blocks": float(mean(oracle_gains_all)) if oracle_gains_all else None,
            "positive_case_blocks": int(sum(1 for value in oracle_gains_all if value > 0.0)),
            "negative_case_blocks": int(sum(1 for value in oracle_gains_all if value < 0.0)),
            "zero_case_blocks": int(sum(1 for value in oracle_gains_all if value == 0.0)),
        },
        "winner_stability": _winner_stability(case_rows, endpoints),
        "by_start_support_bucket": dict(Counter(str(row.get("support_bucket")) for row in case_rows)),
        "rejected": rejected,
    }


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    gain = summary["support_best_raw2_gain_vs_base"]
    ci = gain["root_cluster_ci_all_blocks"]
    lines = [
        "# Pre-Near-Failure Support Action Labels",
        "",
        f"created_at: `{summary['created_at']}`",
        "",
        f"- cases selected: `{summary['cases_selected']}` from `{summary['unique_roots']}` roots",
        f"- offsets: `{summary['offset_counts']}`",
        f"- rollouts: `{summary['valid_rollouts']}` valid / `{summary['rollouts']}` total",
        f"- mediation endpoints: `{summary['support_endpoint_for_mediation']}` -> `{summary['raw2_endpoint_for_mediation']}`",
        "",
        "Support-best action raw2 lift vs base:",
        f"- mean over case-blocks: `{gain['mean_over_case_blocks']}`",
        f"- root-cluster CI all blocks: mean `{ci['mean']}`, 95% CI `[{ci['ci_low']}, {ci['ci_high']}]`, roots `{ci['roots']}`",
        f"- block signs: `{gain['block_mean_signs']}`",
        "",
        "Raw2 oracle gain vs base:",
        f"- `{summary['raw2_oracle_gain_vs_base']}`",
        "",
        "Winner stability:",
    ]
    for endpoint, row in summary["winner_stability"].items():
        lines.append(f"- `{endpoint}`: `{row}`")
    path.write_text("\n".join(lines) + "\n")


def _write_html(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    case_rows = payload.get("case_summary", [])

    def cell(value: object) -> str:
        return escape(str(value))

    rows = []
    for row in case_rows[:200]:
        rows.append(
            "<tr>"
            f"<td>{cell(row.get('root_seed'))}</td>"
            f"<td>{cell(row.get('window_offset'))}</td>"
            f"<td>{cell(row.get('seed_block'))}</td>"
            f"<td>{cell(row.get('base_action'))}</td>"
            f"<td>{cell(row.get('support_best_action'))}</td>"
            f"<td>{cell(row.get('raw2_best_action'))}</td>"
            f"<td>{float(row.get('support_best_raw2_gain_vs_base') or 0.0):+.2f}</td>"
            f"<td>{float(row.get('raw2_oracle_gain_vs_base') or 0.0):+.2f}</td>"
            "</tr>"
        )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Pre-Near-Failure Support Action Labels</title>
  <style>
    :root {{ color-scheme: dark; --bg:#101318; --panel:#171d24; --line:#34404d; --text:#edf2f7; --muted:#aab6c2; --gold:#f2c14e; }}
    body {{ margin:0; background:var(--bg); color:var(--text); font-family:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ width:min(1180px, calc(100vw - 32px)); margin:0 auto; padding:24px 0 40px; }}
    h1 {{ margin:0 0 8px; font-size:24px; }}
    .muted {{ color:var(--muted); }}
    .cards {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(150px, 1fr)); gap:10px; margin:18px 0; }}
    .card, .panel {{ border:1px solid var(--line); background:var(--panel); border-radius:8px; padding:12px; }}
    .label {{ color:var(--muted); font-size:11px; text-transform:uppercase; }}
    .value {{ margin-top:4px; color:var(--gold); font-size:22px; font-weight:800; }}
    table {{ width:100%; border-collapse:collapse; font-size:12px; }}
    th, td {{ border-bottom:1px solid var(--line); padding:7px 8px; text-align:right; }}
    th:first-child, td:first-child {{ text-align:left; }}
    pre {{ white-space:pre-wrap; overflow-wrap:anywhere; color:var(--muted); }}
  </style>
</head>
<body>
<main>
  <h1>Pre-Near-Failure Support Action Labels</h1>
  <p class="muted">Paired common-random-number first-action labels for support preservation and second-768 acquisition.</p>
  <section class="cards">
    <div class="card"><div class="label">Cases</div><div class="value">{cell(summary.get('cases_selected'))}</div></div>
    <div class="card"><div class="label">Roots</div><div class="value">{cell(summary.get('unique_roots'))}</div></div>
    <div class="card"><div class="label">Rollouts</div><div class="value">{cell(summary.get('valid_rollouts'))}</div></div>
    <div class="card"><div class="label">Mean Raw2 Lift</div><div class="value">{cell(summary.get('support_best_raw2_gain_vs_base', {}).get('mean_over_case_blocks'))}</div></div>
  </section>
  <section class="panel"><pre>{cell(json.dumps(summary, indent=2, sort_keys=True))}</pre></section>
  <section class="panel" style="margin-top:14px;">
    <table><thead><tr><th>Root</th><th>Offset</th><th>Block</th><th>Base</th><th>Support Best</th><th>Raw2 Best</th><th>Support->Raw2</th><th>Oracle</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
  </section>
</main>
</body>
</html>
"""
    path.write_text(html)


def _eligible_record(record: dict[str, Any], offsets: set[int] | None) -> bool:
    if offsets is None:
        return True
    try:
        return int(record.get("window_offset")) in offsets
    except (TypeError, ValueError):
        return False


def run_labeling(
    *,
    records_json: list[Path],
    policy_name: str,
    support_horizons: tuple[int, ...],
    raw2_horizons: tuple[int, ...],
    repeats_per_block: int,
    seed_blocks: tuple[int, ...],
    max_starts: int,
    seed: int,
    offsets: set[int] | None,
    root_origins: set[str],
    default_starter_tile: int | None,
    first_action_mode: str,
    base_action_mode: str,
    out_dir: Path,
    progress_every: int = 25,
) -> dict[str, Any]:
    if first_action_mode not in FIRST_ACTION_MODES:
        raise ValueError(f"Unsupported first-action mode: {first_action_mode}")
    if base_action_mode not in {"recorded", "actor"}:
        raise ValueError(f"Unsupported base action mode: {base_action_mode}")
    records = [record for record in load_records(records_json) if _eligible_record(record, offsets)]
    pairs: list[tuple[dict[str, Any], FrontierCase]] = []
    rejected: Counter[str] = Counter()
    for record in records:
        case = case_from_record(record, default_starter_tile=default_starter_tile)
        if case is None:
            rejected["bad_record"] += 1
            continue
        if case.root_origin not in root_origins:
            rejected[f"root_origin:{case.root_origin}"] += 1
            continue
        case.features["window_offset"] = record.get("window_offset")
        pairs.append((record, case))
    selected_cases = select_diverse_cases([case for _record, case in pairs], max_starts=max_starts, seed=seed)
    selected_ids = {case.id for case in selected_cases}
    selected = [(record, case) for record, case in pairs if case.id in selected_ids]
    if not selected:
        raise ValueError("No support action-label cases matched the requested filters")

    horizons = tuple(sorted(set(tuple(support_horizons) + tuple(raw2_horizons))))
    max_support_h = max(int(h) for h in support_horizons)
    max_raw2_h = max(int(h) for h in raw2_horizons)
    support_endpoint = f"support_h{max_support_h}"
    raw2_endpoint = f"raw2_h{max_raw2_h}"
    policy = make_policy(policy_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    action_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    rollouts: list[dict[str, Any]] = []
    planned = 0
    action_plan: list[tuple[int, dict[str, Any], FrontierCase, list[int], str | None]] = []
    for case_idx, (record, case) in enumerate(selected):
        sim = ThreesSim(np.random.default_rng(seed + case_idx), starter_tile=case.starter_tile)
        legal = [int(action) for action in sim.legal_actions(case.state)]
        actions = select_first_actions(
            policy=policy,
            case=case,
            sim=sim,
            mode=first_action_mode,
            rng=np.random.default_rng(seed + case_idx + 17),
        )
        actor_action = int(policy(case.state, sim, np.random.default_rng(seed + case_idx + 31))) if legal else None
        if actor_action not in legal if actor_action is not None else False:
            actor_action = legal[0] if legal else None
        recorded = _recorded_action(record, legal)
        base_action = recorded if base_action_mode == "recorded" and recorded is not None else actor_action
        base_action_name = DIRECTION_NAMES[int(base_action)] if base_action is not None else None
        if base_action is not None and int(base_action) not in set(actions):
            actions = list(actions) + [int(base_action)]
        action_plan.append((case_idx, record, case, actions, base_action_name))
        planned += len(actions) * int(repeats_per_block) * len(seed_blocks)

    completed = 0
    for case_idx, record, case, actions, base_action_name in action_plan:
        case_action_rows: list[dict[str, Any]] = []
        for block_idx, seed_block in enumerate(seed_blocks):
            block_action_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for repeat_idx in range(int(repeats_per_block)):
                crn_seed = int(seed_block) + case_idx * 1_000_003 + repeat_idx * 10_007
                policy_seed = int(seed_block) + case_idx * 917_503 + repeat_idx * 7_919 + 37
                for action in actions:
                    row = rollout_branch(
                        case=case,
                        policy=policy,
                        first_action=int(action),
                        repeat_index=repeat_idx,
                        seed_block=block_idx,
                        sim_seed=crn_seed,
                        policy_seed=policy_seed,
                        horizons=horizons,
                    )
                    rollouts.append(row)
                    block_action_rows[DIRECTION_NAMES[int(action)]].append(row)
                    completed += 1
                    if progress_every > 0 and completed % int(progress_every) == 0:
                        raw2_hits = sum(
                            bool((item.get("horizon_metrics") or {}).get(str(max_raw2_h), {}).get("raw_two_768_no1536"))
                            for item in rollouts
                            if not item.get("invalid_first_action")
                        )
                        print(f"support_label_progress {completed}/{planned} raw2_h{max_raw2_h}_hits={raw2_hits}", flush=True)
            for action_name, rows in sorted(block_action_rows.items()):
                summary = _action_summary(
                    case=case,
                    action_name=action_name,
                    seed_block=block_idx,
                    rows=rows,
                    support_horizons=support_horizons,
                    raw2_horizons=raw2_horizons,
                )
                action_rows.append(summary)
                case_action_rows.append(summary)
        case_rows.extend(
            _case_summaries(
                case=case,
                record=record,
                action_rows=case_action_rows,
                base_action_name=base_action_name,
                support_endpoint=support_endpoint,
                raw2_endpoint=raw2_endpoint,
            )
        )

    summary = summarize(
        records_total=len(records),
        cases_total=len(pairs),
        cases_selected=len(selected),
        rejected=dict(rejected),
        action_rows=action_rows,
        case_rows=case_rows,
        rollouts=rollouts,
        support_horizons=support_horizons,
        raw2_horizons=raw2_horizons,
        support_endpoint=support_endpoint,
        raw2_endpoint=raw2_endpoint,
        seed=seed,
    )
    summary.update(
        {
            "policy": policy_name,
            "first_action_mode": first_action_mode,
            "base_action_mode": base_action_mode,
            "repeats_per_block": int(repeats_per_block),
            "seed_block_values": [int(value) for value in seed_blocks],
            "max_starts": int(max_starts),
        }
    )
    payload = {
        "version": 1,
        "kind": "pre_nearfail_support_action_labels",
        "records_json": [str(path) for path in records_json],
        "summary": summary,
        "case_summary": case_rows,
        "action_summary": action_rows,
        "rollouts": rollouts,
    }
    write_json(out_dir / "pre_nearfail_support_action_labels.json", payload)
    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "case_summary.json", case_rows)
    write_json(out_dir / "action_summary.json", action_rows)
    write_json(out_dir / "rollouts.json", rollouts)
    _write_report(out_dir / "report.md", payload)
    _write_html(out_dir / "pre_nearfail_support_action_labels.html", payload)
    return {
        **payload,
        "json": str(out_dir / "pre_nearfail_support_action_labels.json"),
        "report": str(out_dir / "report.md"),
        "html": str(out_dir / "pre_nearfail_support_action_labels.html"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-json", type=Path, action="append", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--support-horizons", default="5,10")
    parser.add_argument("--raw2-horizons", default="20,40")
    parser.add_argument("--repeats-per-block", type=int, default=4)
    parser.add_argument("--seed-blocks", default="20260731,20260801")
    parser.add_argument("--max-starts", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--offsets", default="5,10")
    parser.add_argument("--root-origin", help="Comma-separated root origins; defaults to fresh,human.")
    parser.add_argument("--starter", default="1536")
    parser.add_argument("--first-action-mode", choices=FIRST_ACTION_MODES, default="all")
    parser.add_argument("--base-action-mode", choices=("recorded", "actor"), default="recorded")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--out-dir", type=Path, default=Path("threes_rl/runs/forensics/pre_nearfail_support_action_labels/latest"))
    args = parser.parse_args()
    starter_text = args.starter.strip().lower()
    default_starter = None if starter_text == "none" else int(starter_text)
    offsets = set(_parse_int_list(args.offsets, default=(5, 10))) if args.offsets.strip().lower() != "all" else None
    payload = run_labeling(
        records_json=args.records_json,
        policy_name=args.policy,
        support_horizons=_parse_int_list(args.support_horizons, default=(5, 10)),
        raw2_horizons=_parse_int_list(args.raw2_horizons, default=(20, 40)),
        repeats_per_block=args.repeats_per_block,
        seed_blocks=_parse_int_list(args.seed_blocks, default=(20260731, 20260801)),
        max_starts=args.max_starts,
        seed=args.seed,
        offsets=offsets,
        root_origins=parse_root_origins(args.root_origin),
        default_starter_tile=default_starter,
        first_action_mode=args.first_action_mode,
        base_action_mode=args.base_action_mode,
        out_dir=args.out_dir,
        progress_every=args.progress_every,
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"json={payload['json']}")
    print(f"report={payload['report']}")
    print(f"html={payload['html']}")


if __name__ == "__main__":
    main()
