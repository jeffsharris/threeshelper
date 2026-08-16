"""Offline selective-rollout gate for one-768 failure roots.

This diagnostic is intentionally not a policy wrapper. It asks whether a small
pilot rollout block can select a first action that improves an independently
evaluated milestone endpoint against the incumbent first action.
"""

from __future__ import annotations

import argparse
import hashlib
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
    FrontierCase,
    load_cases,
    load_records,
    parse_root_origins,
    rollout_branch,
    select_diverse_cases,
    select_first_actions,
    supported_frontier_targets,
)
from threes_rl.run_artifacts import write_json
from threes_rl.sim import DIRECTION_NAMES, ThreesSim


def _action_name(action: int) -> str:
    return DIRECTION_NAMES[int(action)]


def _action_index(name: str) -> int:
    return DIRECTION_NAMES.index(str(name))


def _dedupe_actions(actions: Iterable[int]) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for action in actions:
        value = int(action)
        if value in seen:
            continue
        out.append(value)
        seen.add(value)
    return out


def _mean(values: Iterable[float]) -> float:
    rows = list(values)
    return float(mean(rows)) if rows else 0.0


def gate_progress_key(
    *,
    policy_name: str,
    target: str,
    horizon: int,
    case: FrontierCase,
    first_action: int,
    repeat_index: int,
    seed: int,
    block_name: str,
) -> str:
    raw = json.dumps(
        {
            "version": 1,
            "policy": str(policy_name),
            "target": str(target),
            "horizon": int(horizon),
            "block_name": str(block_name),
            "seed": int(seed),
            "first_action": _action_name(int(first_action)),
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


def _load_gate_progress(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "version": 1,
            "kind": "selective_rollout_gate_progress",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "updated_at": None,
            "entries": {},
        }
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a gate progress object")
    entries = payload.setdefault("entries", {})
    if not isinstance(entries, dict):
        raise ValueError(f"{path} has invalid progress entries")
    payload.setdefault("version", 1)
    payload.setdefault("kind", "selective_rollout_gate_progress")
    return payload


def _write_gate_progress(path: Path, payload: dict[str, Any]) -> None:
    payload["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, payload)


def _progress_entry(
    *,
    key: str,
    policy_name: str,
    target: str,
    horizon: int,
    block_name: str,
    rollout: dict[str, Any],
) -> dict[str, Any]:
    return {
        "key": str(key),
        "policy": str(policy_name),
        "target": str(target),
        "horizon": int(horizon),
        "block_name": str(block_name),
        "seed": int(rollout["seed"]),
        "first_action": str(rollout["first_action"]),
        "repeat_index": int(rollout["repeat_index"]),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "rollout": rollout,
    }


def _record_progress(progress: dict[str, int], rollout: dict[str, Any], *, progress_every: int, block_name: str) -> None:
    progress["completed"] += 1
    progress["hits"] = int(progress.get("hits", 0)) + int(bool(rollout.get("target_reached")))
    if progress_every > 0 and progress["completed"] % int(progress_every) == 0:
        print(
            "selective_gate_progress "
            f"{progress['completed']}/{progress['planned']} "
            f"block={block_name} hits={progress['hits']} "
            f"ran={progress.get('ran', 0)} resumed={progress.get('resumed', 0)}",
            flush=True,
        )


def _cluster_bootstrap_ci(values_by_root: dict[str, float], *, seed: int, resamples: int = 2000) -> dict[str, Any]:
    if not values_by_root:
        return {"mean": None, "ci_low": None, "ci_high": None, "roots": 0}
    roots = sorted(values_by_root)
    arr = np.asarray([float(values_by_root[root]) for root in roots], dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    samples = []
    for _ in range(int(resamples)):
        sampled = arr[rng.integers(0, len(arr), size=len(arr))]
        samples.append(float(sampled.mean()))
    return {
        "mean": float(arr.mean()),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "roots": int(len(roots)),
    }


def incumbent_action(policy: object, case: FrontierCase, *, seed: int) -> int | None:
    sim = ThreesSim(np.random.default_rng(int(seed)), starter_tile=case.starter_tile)
    ranked = select_first_actions(
        policy=policy,
        case=case,
        sim=sim,
        mode="top-two",
        rng=np.random.default_rng(int(seed) + 17),
    )
    if ranked:
        return int(ranked[0])
    legal = select_first_actions(
        policy=policy,
        case=case,
        sim=sim,
        mode="all",
        rng=np.random.default_rng(int(seed) + 23),
    )
    return int(legal[0]) if legal else None


def legal_actions_for_case(policy: object, case: FrontierCase, *, seed: int) -> list[int]:
    sim = ThreesSim(np.random.default_rng(int(seed)), starter_tile=case.starter_tile)
    return select_first_actions(
        policy=policy,
        case=case,
        sim=sim,
        mode="all",
        rng=np.random.default_rng(int(seed) + 31),
    )


def _rollouts_for_actions(
    *,
    case: FrontierCase,
    case_index: int,
    policy: object,
    policy_name: str,
    target: str,
    horizon: int,
    actions: list[int],
    repeats: int,
    seed: int,
    block_name: str,
    progress: dict[str, int],
    progress_every: int,
    progress_payload: dict[str, Any] | None = None,
    progress_path: Path | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    progress_entries: dict[str, Any] | None = None
    if progress_payload is not None:
        entries = progress_payload.setdefault("entries", {})
        if not isinstance(entries, dict):
            raise ValueError("invalid gate progress entries")
        progress_entries = entries
    for repeat_idx in range(int(repeats)):
        crn_seed = int(seed) + int(case_index) * 100_003 + int(repeat_idx) * 997
        for action in actions:
            key = gate_progress_key(
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
                entry = progress_entries.get(key)
                if isinstance(entry, dict) and isinstance(entry.get("rollout"), dict):
                    rollout = dict(entry["rollout"])
                    rollout["block_name"] = str(block_name)
                    rollout["policy"] = str(policy_name)
                    rows.append(rollout)
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
            rollout["block_name"] = str(block_name)
            rollout["policy"] = str(policy_name)
            rows.append(rollout)
            progress["ran"] = int(progress.get("ran", 0)) + 1
            if progress_entries is not None and progress_payload is not None and progress_path is not None:
                progress_entries[key] = _progress_entry(
                    key=key,
                    policy_name=policy_name,
                    target=target,
                    horizon=horizon,
                    block_name=block_name,
                    rollout=rollout,
                )
                _write_gate_progress(progress_path, progress_payload)
            _record_progress(progress, rollout, progress_every=progress_every, block_name=block_name)
    return rows


def summarize_action_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["block_name"]), str(row["case_id"]), str(row["first_action"]))].append(row)
    out: list[dict[str, Any]] = []
    for (block_name, case_id, action), action_rows in sorted(groups.items()):
        valid = [row for row in action_rows if not bool(row.get("invalid_first_action"))]
        hits = [row for row in valid if bool(row.get("target_reached"))]
        score_deltas = [float(row.get("score_delta", 0.0)) for row in valid]
        moves_to_target = [float(row["moves_to_target"]) for row in hits if row.get("moves_to_target") is not None]
        first = action_rows[0] if action_rows else {}
        out.append(
            {
                "block_name": block_name,
                "case_id": case_id,
                "root_seed": first.get("root_seed"),
                "root_replay": first.get("root_replay"),
                "ancestry_key": first.get("ancestry_key"),
                "first_action": action,
                "rollouts": len(action_rows),
                "valid_rollouts": len(valid),
                "target_hits": len(hits),
                "target_rate": len(hits) / float(len(valid)) if valid else 0.0,
                "mean_score_delta": _mean(score_deltas),
                "mean_moves_to_target": _mean(moves_to_target),
                "terminal_rate": sum(bool(row.get("game_over")) for row in valid) / float(len(valid)) if valid else 0.0,
            }
        )
    return out


def _rows_by_action(action_rows: list[dict[str, Any]], *, block_name: str, case_id: str) -> dict[str, dict[str, Any]]:
    return {
        str(row["first_action"]): row
        for row in action_rows
        if str(row.get("block_name")) == str(block_name) and str(row.get("case_id")) == str(case_id)
    }


def choose_pilot_action(action_rows: list[dict[str, Any]], *, base_action: str) -> tuple[str, dict[str, Any]]:
    if not action_rows:
        return base_action, {"reason": "no_pilot_rows", "target_rate_tie": True}
    max_rate = max(float(row.get("target_rate", 0.0)) for row in action_rows)
    tied = [row for row in action_rows if float(row.get("target_rate", 0.0)) == max_rate]
    base_row = next((row for row in tied if str(row.get("first_action")) == str(base_action)), None)
    if base_row is not None:
        return str(base_action), {
            "reason": "incumbent_tied_best_target_rate",
            "target_rate_tie": len(tied) > 1,
            "max_target_rate": float(max_rate),
        }
    winner = max(
        tied,
        key=lambda row: (
            float(row.get("mean_score_delta", 0.0)),
            -_action_index(str(row.get("first_action"))),
        ),
    )
    return str(winner["first_action"]), {
        "reason": "strict_target_rate_gain",
        "target_rate_tie": len(tied) > 1,
        "max_target_rate": float(max_rate),
    }


def _case_eval_rows(
    *,
    case: FrontierCase,
    base_action: str,
    selected_action: str,
    pilot_rows: list[dict[str, Any]],
    eval_action_rows: list[dict[str, Any]],
    eval_blocks: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pilot_by_action = {str(row["first_action"]): row for row in pilot_rows}
    selected_pilot = pilot_by_action.get(selected_action, {})
    base_pilot = pilot_by_action.get(base_action, {})
    for block_idx in range(int(eval_blocks)):
        block_name = f"eval_{block_idx}"
        by_action = _rows_by_action(eval_action_rows, block_name=block_name, case_id=case.id)
        selected_row = by_action.get(selected_action, {})
        base_row = by_action.get(base_action, {})
        selected_rate = float(selected_row.get("target_rate", 0.0))
        base_rate = float(base_row.get("target_rate", 0.0))
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
                "pilot_selected_target_rate": float(selected_pilot.get("target_rate", 0.0)),
                "pilot_base_target_rate": float(base_pilot.get("target_rate", 0.0)),
                "pilot_target_lift_vs_base": float(selected_pilot.get("target_rate", 0.0))
                - float(base_pilot.get("target_rate", 0.0)),
                "eval_selected_target_rate": selected_rate,
                "eval_base_target_rate": base_rate,
                "eval_target_lift_vs_base": selected_rate - base_rate,
                "eval_selected_mean_score_delta": float(selected_row.get("mean_score_delta", 0.0)),
                "eval_base_mean_score_delta": float(base_row.get("mean_score_delta", 0.0)),
            }
        )
    return rows


def summarize(
    *,
    records_total: int,
    cases_total: int,
    cases_selected: int,
    rejected: dict[str, int],
    target: str,
    horizon: int,
    pilot_repeats: int,
    eval_repeats: int,
    eval_blocks: int,
    policy_name: str,
    pilot_rollouts: list[dict[str, Any]],
    eval_rollouts: list[dict[str, Any]],
    case_summary: list[dict[str, Any]],
    seed: int,
    min_promotion_roots: int,
) -> dict[str, Any]:
    roots = {str(row.get("root_seed") or row.get("ancestry_key") or row.get("case_id")) for row in case_summary}
    changed_cases = {str(row["case_id"]) for row in case_summary if bool(row.get("changed_action"))}
    block_root_values: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    all_root_values: dict[str, list[float]] = defaultdict(list)
    for row in case_summary:
        root = str(row.get("root_seed") or row.get("ancestry_key") or row.get("case_id"))
        block = int(row.get("eval_block", 0))
        lift = float(row.get("eval_target_lift_vs_base", 0.0))
        block_root_values[block][root].append(lift)
        all_root_values[root].append(lift)
    block_cis = {
        str(block): _cluster_bootstrap_ci(
            {root: float(mean(values)) for root, values in values_by_root.items()},
            seed=int(seed) + 10_000 + int(block) * 101,
        )
        for block, values_by_root in sorted(block_root_values.items())
    }
    all_ci = _cluster_bootstrap_ci(
        {root: float(mean(values)) for root, values in all_root_values.items()},
        seed=int(seed) + 99_999,
    )
    block_positive = {
        block: bool((ci.get("mean") is not None) and float(ci["mean"]) > 0.0)
        for block, ci in block_cis.items()
    }
    all_ci_excludes_zero = all_ci.get("ci_low") is not None and float(all_ci["ci_low"]) > 0.0
    screen_passed = (
        len(roots) >= int(min_promotion_roots)
        and bool(block_positive)
        and all(block_positive.values())
        and bool(all_ci_excludes_zero)
    )
    eval_lifts = [float(row.get("eval_target_lift_vs_base", 0.0)) for row in case_summary]
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "policy": policy_name,
        "target": target,
        "horizon": int(horizon),
        "pilot_repeats_per_action": int(pilot_repeats),
        "eval_repeats_per_action": int(eval_repeats),
        "eval_blocks": int(eval_blocks),
        "records_total": int(records_total),
        "cases_total": int(cases_total),
        "cases_selected": int(cases_selected),
        "unique_roots": int(len(roots)),
        "changed_cases": int(len(changed_cases)),
        "pilot_rollouts": int(len(pilot_rollouts)),
        "eval_rollouts": int(len(eval_rollouts)),
        "valid_pilot_rollouts": int(sum(1 for row in pilot_rollouts if not row.get("invalid_first_action"))),
        "valid_eval_rollouts": int(sum(1 for row in eval_rollouts if not row.get("invalid_first_action"))),
        "case_blocks": int(len(case_summary)),
        "mean_eval_target_lift_vs_base": _mean(eval_lifts),
        "positive_case_blocks": int(sum(1 for value in eval_lifts if value > 0.0)),
        "negative_case_blocks": int(sum(1 for value in eval_lifts if value < 0.0)),
        "zero_case_blocks": int(sum(1 for value in eval_lifts if value == 0.0)),
        "root_cluster_ci_all_blocks": all_ci,
        "root_cluster_ci_by_block": block_cis,
        "block_mean_positive": block_positive,
        "promotion_screen": {
            "min_roots": int(min_promotion_roots),
            "enough_roots": len(roots) >= int(min_promotion_roots),
            "all_block_means_positive": all(block_positive.values()) if block_positive else False,
            "all_blocks_ci": all_ci,
            "all_blocks_ci_excludes_zero": bool(all_ci_excludes_zero),
            "passed": bool(screen_passed),
        },
        "selected_action_counts": dict(Counter(str(row.get("selected_action")) for row in case_summary if int(row.get("eval_block", 0)) == 0)),
        "base_action_counts": dict(Counter(str(row.get("base_action")) for row in case_summary if int(row.get("eval_block", 0)) == 0)),
        "rejected": rejected,
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    screen = summary["promotion_screen"]
    ci = summary["root_cluster_ci_all_blocks"]
    lines = [
        "# Selective Rollout Gate",
        "",
        f"created_at: `{summary['created_at']}`",
        "",
        f"- target: `{summary['target']}` at h`{summary['horizon']}`",
        f"- roots: `{summary['unique_roots']}` / cases: `{summary['cases_selected']}`",
        f"- changed cases: `{summary['changed_cases']}`",
        f"- mean eval lift: `{summary['mean_eval_target_lift_vs_base']}`",
        f"- root-cluster CI all blocks: mean `{ci['mean']}`, 95% CI `[{ci['ci_low']}, {ci['ci_high']}]`",
        f"- block mean positive: `{summary['block_mean_positive']}`",
        f"- promotion screen passed: `{screen['passed']}`",
        "",
        "Promotion screen:",
        f"- enough roots: `{screen['enough_roots']}` (minimum `{screen['min_roots']}`)",
        f"- all block means positive: `{screen['all_block_means_positive']}`",
        f"- all-block CI excludes zero: `{screen['all_blocks_ci_excludes_zero']}`",
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
            f"<td>{float(row.get('pilot_target_lift_vs_base', 0.0)):+.2f}</td>"
            f"<td>{float(row.get('eval_target_lift_vs_base', 0.0)):+.2f}</td>"
            f"<td>{float(row.get('eval_base_target_rate', 0.0)):.0%}</td>"
            f"<td>{float(row.get('eval_selected_target_rate', 0.0)):.0%}</td>"
            "</tr>"
        )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Selective Rollout Gate</title>
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
  <h1>Selective Rollout Gate</h1>
  <p class="muted">Pilot-selected first action versus incumbent action on independent CRN evaluation blocks.</p>
  <section class="cards">
    <div class="card"><div class="label">Roots</div><div class="value">{cell(summary.get('unique_roots'))}</div></div>
    <div class="card"><div class="label">Changed</div><div class="value">{cell(summary.get('changed_cases'))}</div></div>
    <div class="card"><div class="label">Mean Lift</div><div class="value">{float(summary.get('mean_eval_target_lift_vs_base') or 0.0):+.2f}</div></div>
    <div class="card"><div class="label">Screen</div><div class="value">{cell(summary.get('promotion_screen', {}).get('passed'))}</div></div>
  </section>
  <section class="panel"><pre>{escape(json.dumps(summary, indent=2, sort_keys=True))}</pre></section>
  <section class="panel" style="margin-top:14px;">
    <table><thead><tr><th>Root</th><th>Block</th><th>Base</th><th>Selected</th><th>Pilot Lift</th><th>Eval Lift</th><th>Base Hit</th><th>Selected Hit</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
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
    pilot_repeats: int,
    eval_repeats: int,
    eval_blocks: int,
    max_starts: int,
    seed: int,
    root_origins: set[str],
    case_ids: set[str] | None,
    default_starter_tile: int | None,
    out_dir: Path,
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
    selected = select_diverse_cases(cases, max_starts=max_starts, seed=seed)
    if not selected:
        raise ValueError("No gate cases matched the requested filters")
    policy = make_policy(policy_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = progress_json
    if progress_path is None and checkpoint_rollouts:
        progress_path = out_dir / "gate_progress.json"
    progress_payload: dict[str, Any] | None = None
    if progress_path is not None:
        progress_path = Path(progress_path)
        progress_payload = _load_gate_progress(progress_path)

    planned = 0
    case_setup: list[dict[str, Any]] = []
    for case_idx, case in enumerate(selected):
        legal = legal_actions_for_case(policy, case, seed=int(seed) + case_idx)
        base = incumbent_action(policy, case, seed=int(seed) + case_idx + 101)
        if base is None or not legal:
            continue
        planned += len(legal) * int(pilot_repeats)
        case_setup.append({"case_idx": case_idx, "case": case, "legal": legal, "base": int(base)})
    if not case_setup:
        raise ValueError("No gate cases had legal incumbent actions")

    pilot_rollouts: list[dict[str, Any]] = []
    eval_rollouts: list[dict[str, Any]] = []
    progress = {"completed": 0, "planned": int(planned), "hits": 0, "ran": 0, "resumed": 0}
    pilot_summaries: dict[str, dict[str, Any]] = {}
    selected_actions: dict[str, str] = {}
    base_actions: dict[str, str] = {}

    for item in case_setup:
        case_idx = int(item["case_idx"])
        case = item["case"]
        legal = [int(action) for action in item["legal"]]
        base_action = int(item["base"])
        rows = _rollouts_for_actions(
            case=case,
            case_index=case_idx,
            policy=policy,
            policy_name=policy_name,
            target=target,
            horizon=horizon,
            actions=legal,
            repeats=pilot_repeats,
            seed=seed,
            block_name="pilot",
            progress=progress,
            progress_every=progress_every,
            progress_payload=progress_payload,
            progress_path=progress_path,
        )
        pilot_rollouts.extend(rows)
        action_rows = summarize_action_rows(rows)
        base_name = _action_name(base_action)
        selected_name, choice = choose_pilot_action(action_rows, base_action=base_name)
        pilot_summaries[case.id] = {"choice": choice, "action_rows": action_rows}
        selected_actions[case.id] = selected_name
        base_actions[case.id] = base_name

    eval_planned = 0
    for item in case_setup:
        case = item["case"]
        selected_name = selected_actions[case.id]
        base_name = base_actions[case.id]
        eval_actions = _dedupe_actions([_action_index(selected_name), _action_index(base_name)])
        eval_planned += len(eval_actions) * int(eval_repeats) * int(eval_blocks)
    progress["planned"] = progress["completed"] + int(eval_planned)

    for block_idx in range(int(eval_blocks)):
        block_seed = int(seed) + 1_000_000 + int(block_idx) * 1_000_003
        for item in case_setup:
            case_idx = int(item["case_idx"])
            case = item["case"]
            selected_name = selected_actions[case.id]
            base_name = base_actions[case.id]
            eval_actions = _dedupe_actions([_action_index(selected_name), _action_index(base_name)])
            eval_rollouts.extend(
                _rollouts_for_actions(
                    case=case,
                    case_index=case_idx,
                    policy=policy,
                    policy_name=policy_name,
                    target=target,
                    horizon=horizon,
                    actions=eval_actions,
                    repeats=eval_repeats,
                    seed=block_seed,
                    block_name=f"eval_{block_idx}",
                    progress=progress,
                    progress_every=progress_every,
                    progress_payload=progress_payload,
                    progress_path=progress_path,
                )
            )

    pilot_action_summary = [row for item in pilot_summaries.values() for row in item["action_rows"]]
    eval_action_summary = summarize_action_rows(eval_rollouts)
    case_summary: list[dict[str, Any]] = []
    for item in case_setup:
        case = item["case"]
        case_summary.extend(
            _case_eval_rows(
                case=case,
                base_action=base_actions[case.id],
                selected_action=selected_actions[case.id],
                pilot_rows=pilot_summaries[case.id]["action_rows"],
                eval_action_rows=eval_action_summary,
                eval_blocks=eval_blocks,
            )
        )
        for row in case_summary:
            if row["case_id"] == case.id:
                row["pilot_choice"] = pilot_summaries[case.id]["choice"]

    summary = summarize(
        records_total=len(records),
        cases_total=len(cases),
        cases_selected=len(case_setup),
        rejected=rejected,
        target=target,
        horizon=horizon,
        pilot_repeats=pilot_repeats,
        eval_repeats=eval_repeats,
        eval_blocks=eval_blocks,
        policy_name=policy_name,
        pilot_rollouts=pilot_rollouts,
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
        summary["gate_progress_json"] = str(progress_path)
    payload = {
        "version": 1,
        "kind": "selective_rollout_gate",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "records_json": [str(path) for path in records_json],
        "case_id_filter": sorted(case_ids) if case_ids is not None else None,
        "summary": summary,
        "pilot_action_summary": pilot_action_summary,
        "eval_action_summary": eval_action_summary,
        "case_summary": case_summary,
        "pilot_rollouts": pilot_rollouts,
        "eval_rollouts": eval_rollouts,
    }
    write_json(out_dir / "selective_rollout_gate.json", payload)
    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "pilot_action_summary.json", pilot_action_summary)
    write_json(out_dir / "eval_action_summary.json", eval_action_summary)
    write_json(out_dir / "case_summary.json", case_summary)
    write_json(out_dir / "pilot_rollouts.json", pilot_rollouts)
    write_json(out_dir / "eval_rollouts.json", eval_rollouts)
    write_report(out_dir / "report.md", payload)
    write_html(out_dir / "selective_rollout_gate.html", payload)
    payload["json"] = str(out_dir / "selective_rollout_gate.json")
    payload["summary_json"] = str(out_dir / "summary.json")
    payload["report"] = str(out_dir / "report.md")
    payload["html"] = str(out_dir / "selective_rollout_gate.html")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-json", type=Path, action="append", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--target", choices=supported_frontier_targets(), default="reached_1536")
    parser.add_argument("--horizon", type=int, default=40)
    parser.add_argument("--pilot-repeats-per-action", type=int, default=4)
    parser.add_argument("--eval-repeats-per-action", type=int, default=8)
    parser.add_argument("--eval-blocks", type=int, default=2)
    parser.add_argument("--max-starts", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--root-origin", help="Comma-separated root origins; defaults to fresh,human.")
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--starter", default="1536")
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--min-promotion-roots", type=int, default=20)
    parser.add_argument(
        "--checkpoint-rollouts",
        action="store_true",
        help="Checkpoint each completed branch rollout and reuse matching completed entries on rerun.",
    )
    parser.add_argument(
        "--progress-json",
        type=Path,
        help="Explicit resumable gate progress JSON path; implies checkpoint/resume behavior.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("threes_rl/runs/forensics/selective_rollout_gate/latest"))
    args = parser.parse_args()
    starter_text = str(args.starter).strip().lower()
    default_starter = None if starter_text == "none" else int(starter_text)
    payload = run_gate(
        records_json=args.records_json,
        policy_name=args.policy,
        target=args.target,
        horizon=args.horizon,
        pilot_repeats=args.pilot_repeats_per_action,
        eval_repeats=args.eval_repeats_per_action,
        eval_blocks=args.eval_blocks,
        max_starts=args.max_starts,
        seed=args.seed,
        root_origins=parse_root_origins(args.root_origin),
        case_ids=set(args.case_id) if args.case_id else None,
        default_starter_tile=default_starter,
        out_dir=args.out_dir,
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
