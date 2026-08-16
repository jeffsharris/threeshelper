"""Offline MCTS first-action gate for pre-1536 failure states.

This diagnostic compares a budgeted stochastic UCT first action against the
incumbent first action on independent common-random-number continuations. It is
not a policy wrapper and does not make normal-start claims.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import numpy as np

from threes_rl.eval import make_policy
from threes_rl.rare_event_frontier import (
    FrontierCase,
    load_cases,
    load_records,
    parse_root_origins,
    rollout_branch,
    supported_frontier_targets,
    target_reached,
)
from threes_rl.run_artifacts import write_json
from threes_rl.selective_rollout_gate import (
    _action_name,
    _cluster_bootstrap_ci,
    _dedupe_actions,
    _load_gate_progress,
    _mean,
    _progress_entry,
    _record_progress,
    _write_gate_progress,
    gate_progress_key,
    incumbent_action,
    summarize_action_rows,
)
from threes_rl.sim import DIRECTION_NAMES, SimState, ThreesSim, score_board
from threes_rl.train_td import copy_state


@dataclass
class UctStats:
    visits: int = 0
    total: float = 0.0

    @property
    def mean(self) -> float:
        return self.total / self.visits if self.visits else 0.0


@dataclass
class UctNode:
    visits: int
    legal_actions: tuple[int, ...]
    untried: list[int]
    stats: dict[int, UctStats] = field(default_factory=dict)


def _state_key(state: SimState) -> tuple[Any, ...]:
    return (
        tuple(int(value) for value in np.asarray(state.board, dtype=np.int32).reshape(-1)),
        (state.preview.kind, state.preview.value, tuple(state.preview.candidates)),
        tuple(sorted((str(key), int(value)) for key, value in state.small_counts.items())),
        int(state.small_pos),
        int(state.small_seen_total),
        int(state.span_small_pos),
        bool(state.large_pending),
        int(state.max_tile),
    )


def _leaf_value(policy: object, state: SimState, sim: ThreesSim) -> float:
    if state.game_over or not sim.legal_actions(state):
        return 0.0
    if hasattr(policy, "_post_spawn_state_value"):
        return float(policy._post_spawn_state_value(state, sim))  # noqa: SLF001
    if hasattr(policy, "action_values"):
        values = list(policy.action_values(state, sim))
        return max((float(value) for _action, value in values), default=0.0)
    return float(score_board(state.board))


def _uct_action(
    node: UctNode,
    *,
    rng: np.random.Generator,
    exploration: float,
) -> int:
    if node.untried:
        idx = int(rng.integers(len(node.untried)))
        return int(node.untried.pop(idx))
    log_parent = math.log(max(1, node.visits))
    best_score: float | None = None
    best_actions: list[int] = []
    for action in node.legal_actions:
        stats = node.stats.setdefault(int(action), UctStats())
        if stats.visits <= 0:
            score = float("inf")
        else:
            score = stats.mean + float(exploration) * math.sqrt(log_parent / stats.visits)
        if best_score is None or score > best_score:
            best_score = score
            best_actions = [int(action)]
        elif score == best_score:
            best_actions.append(int(action))
    return int(best_actions[int(rng.integers(len(best_actions)))])


def mcts_action(
    *,
    policy: object,
    case: FrontierCase,
    simulations: int,
    depth: int,
    seed: int,
    exploration: float,
    select_by: str = "visits",
    target: str | None = None,
    reward_mode: str = "value",
    target_bonus: float = 100000.0,
    score_weight: float = 1.0,
    survival_bonus: float = 0.0,
    leaf_weight: float = 1.0,
) -> tuple[int | None, dict[str, Any]]:
    if select_by not in {"visits", "mean"}:
        raise ValueError(f"Unsupported MCTS selection rule: {select_by}")
    if reward_mode not in {"value", "target"}:
        raise ValueError(f"Unsupported MCTS reward mode: {reward_mode}")
    root_sim = ThreesSim(np.random.default_rng(seed), starter_tile=case.starter_tile)
    root_state = copy_state(case.state)
    root_legal = tuple(int(action) for action in root_sim.legal_actions(root_state))
    if not root_legal:
        return None, {"reason": "no_legal_actions"}

    nodes: dict[tuple[Any, ...], UctNode] = {}
    root_key = _state_key(root_state)
    nodes[root_key] = UctNode(visits=0, legal_actions=root_legal, untried=list(root_legal))

    for sim_idx in range(int(simulations)):
        rng = np.random.default_rng(int(seed) + 1_000_003 + int(sim_idx) * 9_973)
        sim = ThreesSim(rng, starter_tile=case.starter_tile)
        state = copy_state(root_state)
        path: list[tuple[UctNode, int, float]] = []
        cumulative = 0.0
        hit_target = bool(target and target_reached(state, case.starter_tile, target))

        for _depth_idx in range(int(depth)):
            key = _state_key(state)
            legal = tuple(int(action) for action in sim.legal_actions(state))
            if not legal:
                break
            node = nodes.get(key)
            if node is None:
                node = UctNode(visits=0, legal_actions=legal, untried=list(legal))
                nodes[key] = node
            action = _uct_action(node, rng=rng, exploration=exploration)
            path.append((node, int(action), float(cumulative)))
            next_state, info = sim.step(state, int(action))
            if not info.moved:
                break
            cumulative += float(info.score_delta)
            state = next_state
            if target and target_reached(state, case.starter_tile, target):
                hit_target = True
                if reward_mode == "target":
                    break
            if state.game_over:
                break

        if reward_mode == "target" and float(leaf_weight) == 0.0:
            leaf_value = 0.0
        else:
            leaf_value = _leaf_value(policy, state, root_sim)
        if reward_mode == "target":
            final_return = (
                float(score_weight) * float(cumulative)
                + (float(target_bonus) if hit_target else 0.0)
                + (float(survival_bonus) if not state.game_over else 0.0)
                + float(leaf_weight) * float(leaf_value)
            )
        else:
            final_return = float(cumulative) + float(leaf_weight) * float(leaf_value)
        for node, action, cumulative_before in path:
            stats = node.stats.setdefault(int(action), UctStats())
            stats.visits += 1
            score_baseline = float(score_weight) * float(cumulative_before) if reward_mode == "target" else float(cumulative_before)
            stats.total += final_return - score_baseline
            node.visits += 1

    root = nodes[root_key]
    action_rows: list[dict[str, Any]] = []
    for action in root.legal_actions:
        stats = root.stats.setdefault(int(action), UctStats())
        action_rows.append(
            {
                "first_action": _action_name(action),
                "visits": int(stats.visits),
                "mean_value": float(stats.mean),
                "total_value": float(stats.total),
            }
        )
    action_rows.sort(key=lambda row: (-int(row["visits"]), -float(row["mean_value"]), str(row["first_action"])))
    if not action_rows:
        return int(root_legal[0]), {"reason": "fallback_first_legal"}
    visited_rows = [row for row in action_rows if int(row["visits"]) > 0] or action_rows
    if select_by == "mean":
        best = max(
            visited_rows,
            key=lambda row: (float(row["mean_value"]), int(row["visits"]), -DIRECTION_NAMES.index(str(row["first_action"]))),
        )
    else:
        best = max(
            visited_rows,
            key=lambda row: (int(row["visits"]), float(row["mean_value"]), -DIRECTION_NAMES.index(str(row["first_action"]))),
        )
    return DIRECTION_NAMES.index(str(best["first_action"])), {
        "reason": f"mcts_{select_by}",
        "select_by": str(select_by),
        "simulations": int(simulations),
        "depth": int(depth),
        "exploration": float(exploration),
        "reward_mode": str(reward_mode),
        "target": target,
        "target_bonus": float(target_bonus),
        "score_weight": float(score_weight),
        "survival_bonus": float(survival_bonus),
        "leaf_weight": float(leaf_weight),
        "nodes": int(len(nodes)),
        "root_visits": int(root.visits),
        "root_actions": action_rows,
    }


def _load_case_ids(paths: Iterable[Path]) -> set[str]:
    ids: set[str] = set()
    for path in paths:
        payload = json.loads(Path(path).read_text())
        rows = payload.get("case_summary") if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise ValueError(f"{path} does not contain a case summary list")
        for row in rows:
            if isinstance(row, dict) and row.get("case_id") is not None:
                ids.add(str(row["case_id"]))
    return ids


def _case_eval_rows(
    *,
    case: FrontierCase,
    base_action: str,
    selected_action: str,
    mcts_info: dict[str, Any],
    eval_action_rows: list[dict[str, Any]],
    eval_blocks: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for block_idx in range(int(eval_blocks)):
        block_name = f"eval_{block_idx}"
        by_action = {
            str(row["first_action"]): row
            for row in eval_action_rows
            if str(row.get("block_name")) == block_name and str(row.get("case_id")) == case.id
        }
        selected_row = by_action.get(selected_action, {})
        base_row = by_action.get(base_action, {})
        selected_rate = float(selected_row.get("target_rate", 0.0))
        base_rate = float(base_row.get("target_rate", 0.0))
        selected_score = float(selected_row.get("mean_score_delta", 0.0))
        base_score = float(base_row.get("mean_score_delta", 0.0))
        selected_survival = 1.0 - float(selected_row.get("terminal_rate", 0.0))
        base_survival = 1.0 - float(base_row.get("terminal_rate", 0.0))
        rows.append(
            {
                "case_id": case.id,
                "root_seed": case.root_seed,
                "root_origin": case.root_origin,
                "root_replay": case.root_replay,
                "ancestry_key": case.ancestry_key,
                "source_replay": case.source_replay,
                "source_frame_index": case.source_frame_index,
                "base_action": base_action,
                "selected_action": selected_action,
                "changed_action": selected_action != base_action,
                "eval_block": int(block_idx),
                "eval_selected_target_rate": selected_rate,
                "eval_base_target_rate": base_rate,
                "eval_target_lift_vs_base": selected_rate - base_rate,
                "eval_selected_mean_score_delta": selected_score,
                "eval_base_mean_score_delta": base_score,
                "eval_score_lift_vs_base": selected_score - base_score,
                "eval_selected_survival_rate": selected_survival,
                "eval_base_survival_rate": base_survival,
                "eval_survival_lift_vs_base": selected_survival - base_survival,
                "mcts_info": mcts_info,
            }
        )
    return rows


def _root_values(case_summary: list[dict[str, Any]], key: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in case_summary:
        root = str(row.get("root_seed") or row.get("ancestry_key") or row.get("case_id"))
        grouped[root].append(float(row.get(key, 0.0)))
    return {root: float(mean(values)) for root, values in grouped.items()}


def summarize_mcts_gate(
    *,
    records_total: int,
    cases_total: int,
    cases_selected: int,
    rejected: dict[str, int],
    target: str,
    horizon: int,
    eval_repeats: int,
    eval_blocks: int,
    policy_name: str,
    simulations: int,
    depth: int,
    exploration: float,
    mcts_select_by: str,
    reward_mode: str,
    target_bonus: float,
    score_weight: float,
    survival_bonus: float,
    leaf_weight: float,
    eval_rollouts: list[dict[str, Any]],
    case_summary: list[dict[str, Any]],
    seed: int,
    min_promotion_roots: int,
) -> dict[str, Any]:
    roots = {str(row.get("root_seed") or row.get("ancestry_key") or row.get("case_id")) for row in case_summary}
    changed_cases = {str(row["case_id"]) for row in case_summary if bool(row.get("changed_action"))}
    target_lifts = [float(row.get("eval_target_lift_vs_base", 0.0)) for row in case_summary]
    score_lifts = [float(row.get("eval_score_lift_vs_base", 0.0)) for row in case_summary]
    survival_lifts = [float(row.get("eval_survival_lift_vs_base", 0.0)) for row in case_summary]
    target_ci = _cluster_bootstrap_ci(_root_values(case_summary, "eval_target_lift_vs_base"), seed=int(seed) + 99_999)
    score_ci = _cluster_bootstrap_ci(_root_values(case_summary, "eval_score_lift_vs_base"), seed=int(seed) + 199_999)
    survival_ci = _cluster_bootstrap_ci(_root_values(case_summary, "eval_survival_lift_vs_base"), seed=int(seed) + 299_999)
    block_root_values: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in case_summary:
        root = str(row.get("root_seed") or row.get("ancestry_key") or row.get("case_id"))
        block_root_values[int(row.get("eval_block", 0))][root].append(float(row.get("eval_target_lift_vs_base", 0.0)))
    block_cis = {
        str(block): _cluster_bootstrap_ci(
            {root: float(mean(values)) for root, values in values_by_root.items()},
            seed=int(seed) + 10_000 + int(block) * 101,
        )
        for block, values_by_root in sorted(block_root_values.items())
    }
    block_positive = {
        block: bool((ci.get("mean") is not None) and float(ci["mean"]) > 0.0)
        for block, ci in block_cis.items()
    }
    target_ci_excludes_zero = target_ci.get("ci_low") is not None and float(target_ci["ci_low"]) > 0.0
    score_ci_excludes_zero = score_ci.get("ci_low") is not None and float(score_ci["ci_low"]) > 0.0
    survival_ci_excludes_zero = survival_ci.get("ci_low") is not None and float(survival_ci["ci_low"]) >= 0.0
    screen_passed = (
        len(roots) >= int(min_promotion_roots)
        and bool(block_positive)
        and all(block_positive.values())
        and bool(target_ci_excludes_zero)
        and bool(score_ci.get("mean") is not None and float(score_ci["mean"]) > 0.0)
        and bool(survival_ci.get("mean") is not None and float(survival_ci["mean"]) >= 0.0)
    )
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "policy": policy_name,
        "target": target,
        "horizon": int(horizon),
        "mcts_simulations": int(simulations),
        "mcts_depth": int(depth),
        "mcts_exploration": float(exploration),
        "mcts_select_by": str(mcts_select_by),
        "mcts_reward_mode": str(reward_mode),
        "mcts_target_bonus": float(target_bonus),
        "mcts_score_weight": float(score_weight),
        "mcts_survival_bonus": float(survival_bonus),
        "mcts_leaf_weight": float(leaf_weight),
        "eval_repeats_per_action": int(eval_repeats),
        "eval_blocks": int(eval_blocks),
        "records_total": int(records_total),
        "cases_total": int(cases_total),
        "cases_selected": int(cases_selected),
        "unique_roots": int(len(roots)),
        "changed_cases": int(len(changed_cases)),
        "eval_rollouts": int(len(eval_rollouts)),
        "valid_eval_rollouts": int(sum(1 for row in eval_rollouts if not row.get("invalid_first_action"))),
        "case_blocks": int(len(case_summary)),
        "mean_eval_target_lift_vs_base": _mean(target_lifts),
        "mean_eval_score_lift_vs_base": _mean(score_lifts),
        "mean_eval_survival_lift_vs_base": _mean(survival_lifts),
        "positive_target_case_blocks": int(sum(1 for value in target_lifts if value > 0.0)),
        "negative_target_case_blocks": int(sum(1 for value in target_lifts if value < 0.0)),
        "zero_target_case_blocks": int(sum(1 for value in target_lifts if value == 0.0)),
        "target_root_cluster_ci_all_blocks": target_ci,
        "score_root_cluster_ci_all_blocks": score_ci,
        "survival_root_cluster_ci_all_blocks": survival_ci,
        "target_root_cluster_ci_by_block": block_cis,
        "block_mean_positive": block_positive,
        "promotion_screen": {
            "min_roots": int(min_promotion_roots),
            "enough_roots": len(roots) >= int(min_promotion_roots),
            "all_block_target_means_positive": all(block_positive.values()) if block_positive else False,
            "target_ci_excludes_zero": bool(target_ci_excludes_zero),
            "score_ci_excludes_zero": bool(score_ci_excludes_zero),
            "survival_ci_nonnegative": bool(survival_ci_excludes_zero),
            "passed": bool(screen_passed),
        },
        "selected_action_counts": dict(Counter(str(row.get("selected_action")) for row in case_summary if int(row.get("eval_block", 0)) == 0)),
        "base_action_counts": dict(Counter(str(row.get("base_action")) for row in case_summary if int(row.get("eval_block", 0)) == 0)),
        "rejected": rejected,
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    target_ci = summary["target_root_cluster_ci_all_blocks"]
    score_ci = summary["score_root_cluster_ci_all_blocks"]
    survival_ci = summary["survival_root_cluster_ci_all_blocks"]
    screen = summary["promotion_screen"]
    lines = [
        "# MCTS Rollout Gate",
        "",
        f"created_at: `{summary['created_at']}`",
        "",
        f"- target: `{summary['target']}` at h`{summary['horizon']}`",
        f"- roots: `{summary['unique_roots']}` / cases: `{summary['cases_selected']}`",
        f"- MCTS: `{summary['mcts_simulations']}` simulations, depth `{summary['mcts_depth']}`, exploration `{summary['mcts_exploration']}`",
        f"- reward: `{summary['mcts_reward_mode']}`, select `{summary['mcts_select_by']}`, target bonus `{summary['mcts_target_bonus']}`",
        f"- changed cases: `{summary['changed_cases']}`",
        f"- mean target lift: `{summary['mean_eval_target_lift_vs_base']}`",
        f"- target CI: mean `{target_ci['mean']}`, 95% CI `[{target_ci['ci_low']}, {target_ci['ci_high']}]`",
        f"- score CI: mean `{score_ci['mean']}`, 95% CI `[{score_ci['ci_low']}, {score_ci['ci_high']}]`",
        f"- survival CI: mean `{survival_ci['mean']}`, 95% CI `[{survival_ci['ci_low']}, {survival_ci['ci_high']}]`",
        f"- promotion screen passed: `{screen['passed']}`",
        "",
        "This is an offline diagnostic. It does not promote a policy by itself.",
    ]
    path.write_text("\n".join(lines) + "\n")


def write_html(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    case_rows = payload.get("case_summary", [])

    def cell(value: object) -> str:
        return escape(str(value))

    rows = []
    for row in case_rows[:240]:
        rows.append(
            "<tr>"
            f"<td>{cell(row.get('root_seed'))}</td>"
            f"<td>{cell(row.get('eval_block'))}</td>"
            f"<td>{cell(row.get('base_action'))}</td>"
            f"<td>{cell(row.get('selected_action'))}</td>"
            f"<td>{float(row.get('eval_target_lift_vs_base', 0.0)):+.2f}</td>"
            f"<td>{float(row.get('eval_score_lift_vs_base', 0.0)):+.0f}</td>"
            f"<td>{float(row.get('eval_survival_lift_vs_base', 0.0)):+.2f}</td>"
            "</tr>"
        )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MCTS Rollout Gate</title>
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
  <h1>MCTS Rollout Gate</h1>
  <p class="muted">Budgeted UCT first action versus incumbent action on independent CRN evaluation blocks.</p>
  <section class="cards">
    <div class="card"><div class="label">Roots</div><div class="value">{cell(summary.get('unique_roots'))}</div></div>
    <div class="card"><div class="label">Changed</div><div class="value">{cell(summary.get('changed_cases'))}</div></div>
    <div class="card"><div class="label">Target Lift</div><div class="value">{float(summary.get('mean_eval_target_lift_vs_base') or 0.0):+.2f}</div></div>
    <div class="card"><div class="label">Screen</div><div class="value">{cell(summary.get('promotion_screen', {}).get('passed'))}</div></div>
  </section>
  <section class="panel"><pre>{escape(json.dumps(summary, indent=2, sort_keys=True))}</pre></section>
  <section class="panel" style="margin-top:14px;">
    <table><thead><tr><th>Root</th><th>Block</th><th>Base</th><th>MCTS</th><th>Target Lift</th><th>Score Lift</th><th>Survival Lift</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
  </section>
</main>
</body>
</html>
"""
    path.write_text(html)


def run_gate(
    *,
    records_json: list[Path],
    policy_name: str,
    target: str,
    horizon: int,
    eval_repeats: int,
    eval_blocks: int,
    max_starts: int,
    seed: int,
    root_origins: set[str],
    case_ids: set[str] | None,
    default_starter_tile: int | None,
    simulations: int,
    depth: int,
    exploration: float,
    mcts_select_by: str,
    out_dir: Path,
    reward_mode: str = "value",
    target_bonus: float = 100000.0,
    score_weight: float = 1.0,
    survival_bonus: float = 0.0,
    leaf_weight: float = 1.0,
    progress_every: int = 50,
    min_promotion_roots: int = 20,
    checkpoint_rollouts: bool = False,
    progress_json: Path | None = None,
) -> dict[str, Any]:
    if target not in supported_frontier_targets():
        raise ValueError(f"Unsupported target: {target}")
    records = load_records(records_json)
    cases, rejected = load_cases(
        records,
        default_starter_tile=default_starter_tile,
        root_origins=root_origins,
        case_ids=case_ids,
    )
    if max_starts > 0:
        from threes_rl.rare_event_frontier import select_diverse_cases

        selected = select_diverse_cases(cases, max_starts=max_starts, seed=seed)
    else:
        selected = list(cases)
    if not selected:
        raise ValueError("No MCTS gate cases matched the requested filters")
    policy = make_policy(policy_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = progress_json
    if progress_path is None and checkpoint_rollouts:
        progress_path = out_dir / "mcts_gate_progress.json"
    progress_payload: dict[str, Any] | None = None
    progress_entries: dict[str, Any] | None = None
    if progress_path is not None:
        progress_path = Path(progress_path)
        progress_payload = _load_gate_progress(progress_path)
        entries = progress_payload.setdefault("entries", {})
        if not isinstance(entries, dict):
            raise ValueError("invalid MCTS gate progress entries")
        progress_entries = entries

    case_setup: list[dict[str, Any]] = []
    for case_idx, case in enumerate(selected):
        base = incumbent_action(policy, case, seed=int(seed) + case_idx + 101)
        selected_action, info = mcts_action(
            policy=policy,
            case=case,
            simulations=simulations,
            depth=depth,
            seed=int(seed) + case_idx * 313,
            exploration=exploration,
            select_by=mcts_select_by,
            target=target,
            reward_mode=reward_mode,
            target_bonus=target_bonus,
            score_weight=score_weight,
            survival_bonus=survival_bonus,
            leaf_weight=leaf_weight,
        )
        if base is None or selected_action is None:
            continue
        case_setup.append(
            {
                "case_idx": case_idx,
                "case": case,
                "base": int(base),
                "selected": int(selected_action),
                "mcts_info": info,
            }
        )
        if progress_every > 0 and len(case_setup) % int(progress_every) == 0:
            print(f"mcts_gate_selection {len(case_setup)}/{len(selected)}", flush=True)
    if not case_setup:
        raise ValueError("No MCTS gate cases had legal actions")

    eval_rollouts: list[dict[str, Any]] = []
    planned = sum(
        len(_dedupe_actions([int(item["selected"]), int(item["base"])])) * int(eval_repeats) * int(eval_blocks)
        for item in case_setup
    )
    progress = {"completed": 0, "planned": int(planned), "hits": 0, "ran": 0, "resumed": 0}
    for block_idx in range(int(eval_blocks)):
        block_seed = int(seed) + 2_000_000 + int(block_idx) * 1_000_003
        for item in case_setup:
            case_idx = int(item["case_idx"])
            case = item["case"]
            actions = _dedupe_actions([int(item["selected"]), int(item["base"])])
            for repeat_idx in range(int(eval_repeats)):
                crn_seed = int(block_seed) + int(case_idx) * 100_003 + int(repeat_idx) * 997
                for action in actions:
                    block_name = f"eval_{block_idx}"
                    progress_key = gate_progress_key(
                        policy_name=policy_name,
                        target=target,
                        horizon=horizon,
                        case=case,
                        first_action=int(action),
                        repeat_index=int(repeat_idx),
                        seed=int(crn_seed),
                        block_name=block_name,
                    )
                    if progress_entries is not None:
                        entry = progress_entries.get(progress_key)
                        if isinstance(entry, dict) and isinstance(entry.get("rollout"), dict):
                            rollout = dict(entry["rollout"])
                            rollout["block_name"] = block_name
                            rollout["policy"] = str(policy_name)
                            eval_rollouts.append(rollout)
                            progress["resumed"] = int(progress.get("resumed", 0)) + 1
                            _record_progress(progress, rollout, progress_every=progress_every, block_name=block_name)
                            continue
                    rollout, _frontier_record = rollout_branch(
                        case=case,
                        policy=policy,
                        first_action=int(action),
                        repeat_index=int(repeat_idx),
                        seed=int(crn_seed),
                        horizon=int(horizon),
                        target=target,
                    )
                    rollout["block_name"] = block_name
                    rollout["policy"] = str(policy_name)
                    eval_rollouts.append(rollout)
                    progress["ran"] = int(progress.get("ran", 0)) + 1
                    if progress_entries is not None and progress_payload is not None and progress_path is not None:
                        progress_entries[progress_key] = _progress_entry(
                            key=progress_key,
                            policy_name=policy_name,
                            target=target,
                            horizon=horizon,
                            block_name=block_name,
                            rollout=rollout,
                        )
                        _write_gate_progress(progress_path, progress_payload)
                    _record_progress(progress, rollout, progress_every=progress_every, block_name=block_name)

    eval_action_summary = summarize_action_rows(eval_rollouts)
    case_summary: list[dict[str, Any]] = []
    mcts_action_summary: list[dict[str, Any]] = []
    for item in case_setup:
        case = item["case"]
        base_name = _action_name(int(item["base"]))
        selected_name = _action_name(int(item["selected"]))
        mcts_info = dict(item["mcts_info"])
        mcts_action_summary.append(
            {
                "case_id": case.id,
                "root_seed": case.root_seed,
                "base_action": base_name,
                "selected_action": selected_name,
                "changed_action": selected_name != base_name,
                **mcts_info,
            }
        )
        case_summary.extend(
            _case_eval_rows(
                case=case,
                base_action=base_name,
                selected_action=selected_name,
                mcts_info=mcts_info,
                eval_action_rows=eval_action_summary,
                eval_blocks=eval_blocks,
            )
        )

    summary = summarize_mcts_gate(
        records_total=len(records),
        cases_total=len(cases),
        cases_selected=len(case_setup),
        rejected=rejected,
        target=target,
        horizon=horizon,
        eval_repeats=eval_repeats,
        eval_blocks=eval_blocks,
        policy_name=policy_name,
        simulations=simulations,
        depth=depth,
        exploration=exploration,
        mcts_select_by=mcts_select_by,
        reward_mode=reward_mode,
        target_bonus=target_bonus,
        score_weight=score_weight,
        survival_bonus=survival_bonus,
        leaf_weight=leaf_weight,
        eval_rollouts=eval_rollouts,
        case_summary=case_summary,
        seed=seed,
        min_promotion_roots=min_promotion_roots,
    )
    summary["rollouts_planned"] = int(progress.get("planned", 0))
    summary["rollouts_ran"] = int(progress.get("ran", 0))
    summary["rollouts_resumed"] = int(progress.get("resumed", 0))
    summary["checkpoint_rollouts"] = progress_path is not None
    if progress_path is not None:
        summary["mcts_gate_progress_json"] = str(progress_path)
    payload = {
        "version": 1,
        "kind": "mcts_rollout_gate",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "records_json": [str(path) for path in records_json],
        "case_id_filter": sorted(case_ids) if case_ids is not None else None,
        "summary": summary,
        "mcts_action_summary": mcts_action_summary,
        "eval_action_summary": eval_action_summary,
        "case_summary": case_summary,
        "eval_rollouts": eval_rollouts,
    }
    write_json(out_dir / "mcts_rollout_gate.json", payload)
    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "mcts_action_summary.json", mcts_action_summary)
    write_json(out_dir / "eval_action_summary.json", eval_action_summary)
    write_json(out_dir / "case_summary.json", case_summary)
    write_json(out_dir / "eval_rollouts.json", eval_rollouts)
    write_report(out_dir / "report.md", payload)
    write_html(out_dir / "mcts_rollout_gate.html", payload)
    payload["json"] = str(out_dir / "mcts_rollout_gate.json")
    payload["summary_json"] = str(out_dir / "summary.json")
    payload["report"] = str(out_dir / "report.md")
    payload["html"] = str(out_dir / "mcts_rollout_gate.html")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-json", type=Path, action="append", required=True)
    parser.add_argument("--case-summary-json", type=Path, action="append", default=[])
    parser.add_argument("--policy", required=True)
    parser.add_argument("--target", choices=supported_frontier_targets(), default="reached_1536")
    parser.add_argument("--horizon", type=int, default=40)
    parser.add_argument("--eval-repeats-per-action", type=int, default=4)
    parser.add_argument("--eval-blocks", type=int, default=2)
    parser.add_argument("--max-starts", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--root-origin", help="Comma-separated root origins; defaults to fresh,human.")
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--starter", default="1536")
    parser.add_argument("--mcts-simulations", type=int, default=64)
    parser.add_argument("--mcts-depth", type=int, default=12)
    parser.add_argument("--mcts-exploration", type=float, default=10000.0)
    parser.add_argument("--mcts-select", choices=("visits", "mean"), default="visits")
    parser.add_argument("--mcts-reward-mode", choices=("value", "target"), default="value")
    parser.add_argument("--mcts-target-bonus", type=float, default=100000.0)
    parser.add_argument("--mcts-score-weight", type=float, default=1.0)
    parser.add_argument("--mcts-survival-bonus", type=float, default=0.0)
    parser.add_argument("--mcts-leaf-weight", type=float, default=1.0)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--min-promotion-roots", type=int, default=20)
    parser.add_argument(
        "--checkpoint-rollouts",
        action="store_true",
        help="Checkpoint each completed evaluation branch and reuse matching completed entries on rerun.",
    )
    parser.add_argument(
        "--progress-json",
        type=Path,
        help="Explicit resumable MCTS gate progress JSON path; implies checkpoint/resume behavior.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("threes_rl/runs/forensics/mcts_rollout_gate/latest"))
    args = parser.parse_args()
    starter_text = str(args.starter).strip().lower()
    default_starter = None if starter_text == "none" else int(starter_text)
    case_ids = set(args.case_id) if args.case_id else set()
    if args.case_summary_json:
        case_ids.update(_load_case_ids(args.case_summary_json))
    payload = run_gate(
        records_json=args.records_json,
        policy_name=args.policy,
        target=args.target,
        horizon=args.horizon,
        eval_repeats=args.eval_repeats_per_action,
        eval_blocks=args.eval_blocks,
        max_starts=args.max_starts,
        seed=args.seed,
        root_origins=parse_root_origins(args.root_origin),
        case_ids=case_ids or None,
        default_starter_tile=default_starter,
        simulations=args.mcts_simulations,
        depth=args.mcts_depth,
        exploration=args.mcts_exploration,
        mcts_select_by=args.mcts_select,
        out_dir=args.out_dir,
        reward_mode=args.mcts_reward_mode,
        target_bonus=args.mcts_target_bonus,
        score_weight=args.mcts_score_weight,
        survival_bonus=args.mcts_survival_bonus,
        leaf_weight=args.mcts_leaf_weight,
        progress_every=args.progress_every,
        min_promotion_roots=args.min_promotion_roots,
        checkpoint_rollouts=args.checkpoint_rollouts,
        progress_json=args.progress_json,
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"json={payload['json']}")
    print(f"report={payload['report']}")
    print(f"html={payload['html']}")


if __name__ == "__main__":
    main()
