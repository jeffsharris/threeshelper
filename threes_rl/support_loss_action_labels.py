"""Label first actions in support-loss windows with short CRN continuations."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from html import escape
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np

from threes_rl.eval import make_policy
from threes_rl.nearfail_support_audit import support_bucket
from threes_rl.rare_event_frontier import (
    FrontierCase,
    load_cases,
    load_records,
    parse_root_origins,
    select_diverse_cases,
    select_first_actions,
)
from threes_rl.run_artifacts import write_json
from threes_rl.sim import DIRECTION_NAMES, SimState, ThreesSim, score_board
from threes_rl.support_accumulation_frontier import _raw
from threes_rl.train_td import copy_state

SUPPORTED_TARGETS = ("raw_one_384_with_one_768_no_1536",)
FIRST_ACTION_MODES = ("all", "top-two", "recorded", "recorded-plus-top-two")


def target_reached(state: SimState, starter_tile: int | None, target: str) -> bool:
    raw = _raw(state, starter_tile)
    if target == "raw_one_384_with_one_768_no_1536":
        return (
            int(raw["masked_count_1536"]) == 0
            and int(raw["raw_count_768"]) >= 1
            and int(raw["raw_count_384"]) >= 1
        )
    raise ValueError(f"Unsupported support-loss label target: {target}")


def _support_present(raw: dict[str, Any]) -> bool:
    return int(raw.get("raw_count_384", 0)) > 0 or int(raw.get("raw_count_192", 0)) > 0


def _raw_summary(state: SimState, starter_tile: int | None) -> dict[str, Any]:
    raw = _raw(state, starter_tile)
    return {
        "raw_count_768": int(raw["raw_count_768"]),
        "raw_count_384": int(raw["raw_count_384"]),
        "raw_count_192": int(raw["raw_count_192"]),
        "raw_has_adjacent_768": bool(raw["raw_has_adjacent_768"]),
        "raw_has_adjacent_384": bool(raw["raw_has_adjacent_384"]),
        "raw_has_adjacent_192": bool(raw["raw_has_adjacent_192"]),
        "raw_highest_duplicate_tile": int(raw["raw_highest_duplicate_tile"]),
        "raw_highest_adjacent_pair_tile": int(raw["raw_highest_adjacent_pair_tile"]),
        "masked_count_1536": int(raw["masked_count_1536"]),
        "empty_count": int(raw["empty_count"]),
        "support_bucket": support_bucket(raw),
        "support_present": _support_present(raw),
    }


def _mean(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def _median(values: list[float]) -> float:
    return float(median(values)) if values else 0.0


def rollout_branch(
    *,
    case: FrontierCase,
    policy: object,
    first_action: int,
    repeat_index: int,
    crn_seed: int,
    horizon: int,
    target: str,
) -> dict[str, Any]:
    sim = ThreesSim(np.random.default_rng(int(crn_seed)), starter_tile=case.starter_tile)
    policy_rng = np.random.default_rng(int(crn_seed) + 37)
    state = copy_state(case.state)
    start_score = int(score_board(state.board))
    start_move = int(state.move_count)
    start_raw = _raw_summary(state, case.starter_tile)
    reached_at: int | None = 0 if target_reached(state, case.starter_tile, target) else None
    target_raw: dict[str, Any] | None = dict(start_raw) if reached_at == 0 else None
    support_present_ever = bool(start_raw["support_present"])
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
        actions.append(DIRECTION_NAMES[int(action)])
        raw = _raw_summary(state, case.starter_tile)
        support_present_ever = support_present_ever or bool(raw["support_present"])
        if reached_at is None and target_reached(state, case.starter_tile, target):
            reached_at = int(state.move_count - start_move)
            target_raw = dict(raw)
        if state.game_over:
            break

    final_raw = _raw_summary(state, case.starter_tile)
    return {
        "case_id": case.id,
        "target": target,
        "first_action": DIRECTION_NAMES[int(first_action)],
        "repeat_index": int(repeat_index),
        "crn_seed": int(crn_seed),
        "invalid_first_action": bool(invalid),
        "target_reached": reached_at is not None,
        "moves_to_target": reached_at,
        "target_raw": target_raw,
        "support_present_ever": bool(support_present_ever),
        "final_support_present": bool(final_raw["support_present"]),
        "final_support_bucket": final_raw["support_bucket"],
        "final_raw_count_768": int(final_raw["raw_count_768"]),
        "final_raw_count_384": int(final_raw["raw_count_384"]),
        "final_raw_count_192": int(final_raw["raw_count_192"]),
        "final_masked_count_1536": int(final_raw["masked_count_1536"]),
        "moves_delta": int(state.move_count - start_move),
        "score_delta": int(score_board(state.board) - start_score),
        "game_over": bool(state.game_over),
        "actions": actions,
        "start_raw": start_raw,
        "source_replay": case.source_replay,
        "source_seed": case.source_seed,
        "source_frame_index": case.source_frame_index,
        "root_origin": case.root_origin,
        "root_replay": case.root_replay,
        "root_seed": case.root_seed,
        "root_policy_family": case.root_policy_family,
        "ancestry_key": case.ancestry_key,
    }


def _rate(rows: list[dict[str, Any]], key: str) -> float:
    return 0.0 if not rows else sum(bool(row.get(key)) for row in rows) / float(len(rows))


def _action_summary(case_id: str, action_name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if not bool(row.get("invalid_first_action"))]
    deltas = [float(row.get("score_delta", 0.0)) for row in valid]
    return {
        "case_id": case_id,
        "first_action": action_name,
        "repeats": len(rows),
        "valid_repeats": len(valid),
        "p_target": _rate(valid, "target_reached"),
        "p_final_support": _rate(valid, "final_support_present"),
        "p_support_ever": _rate(valid, "support_present_ever"),
        "terminal_rate": _rate(valid, "game_over"),
        "mean_moves_to_target": _mean([float(row["moves_to_target"]) for row in valid if row.get("moves_to_target") is not None]),
        "mean_score_delta": _mean(deltas),
        "median_score_delta": _median(deltas),
        "max_score_delta": int(max(deltas)) if deltas else 0,
    }


def _winner(action_rows: list[dict[str, Any]]) -> str | None:
    if not action_rows:
        return None
    return str(
        max(
            action_rows,
            key=lambda row: (
                float(row.get("p_target", 0.0)),
                float(row.get("p_final_support", 0.0)),
                float(row.get("p_support_ever", 0.0)),
                float(row.get("mean_score_delta", 0.0)),
                str(row.get("first_action")),
            ),
        )["first_action"]
    )


def summarize_case(case: FrontierCase, *, base_action_name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_action: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_action.setdefault(str(row["first_action"]), []).append(row)
    action_rows = [
        _action_summary(case.id, action, action_rows)
        for action, action_rows in sorted(by_action.items())
    ]
    winner = _winner(action_rows)
    by_name = {str(row["first_action"]): row for row in action_rows}
    winner_row = by_name.get(str(winner), {})
    base_row = by_name.get(base_action_name, {})
    return {
        "case_id": case.id,
        "source_replay": case.source_replay,
        "source_seed": case.source_seed,
        "source_frame_index": case.source_frame_index,
        "root_seed": case.root_seed,
        "root_policy_family": case.root_policy_family,
        "base_action": base_action_name,
        "winner": winner,
        "winner_target_rate": float(winner_row.get("p_target", 0.0)),
        "base_target_rate": float(base_row.get("p_target", 0.0)),
        "target_gain_vs_base": float(winner_row.get("p_target", 0.0)) - float(base_row.get("p_target", 0.0)),
        "winner_final_support_rate": float(winner_row.get("p_final_support", 0.0)),
        "base_final_support_rate": float(base_row.get("p_final_support", 0.0)),
        "final_support_gain_vs_base": float(winner_row.get("p_final_support", 0.0))
        - float(base_row.get("p_final_support", 0.0)),
        "start_raw": dict(case.raw),
        "start_features": dict(case.features),
        "action_results": action_rows,
    }


def summarize(rollouts: list[dict[str, Any]], labels: list[dict[str, Any]], *, ran: int) -> dict[str, Any]:
    valid = [row for row in rollouts if not bool(row.get("invalid_first_action"))]
    target_gains = [float(label.get("target_gain_vs_base", 0.0)) for label in labels if float(label.get("target_gain_vs_base", 0.0)) > 0.0]
    support_gains = [
        float(label.get("final_support_gain_vs_base", 0.0))
        for label in labels
        if float(label.get("final_support_gain_vs_base", 0.0)) > 0.0
    ]
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "labels": len(labels),
        "rollouts": len(rollouts),
        "valid_rollouts": len(valid),
        "rollouts_ran": int(ran),
        "p_target": _rate(valid, "target_reached"),
        "p_final_support": _rate(valid, "final_support_present"),
        "starts_with_any_target": len({row["case_id"] for row in valid if row.get("target_reached")}),
        "starts_with_any_final_support": len({row["case_id"] for row in valid if row.get("final_support_present")}),
        "winner_counts": dict(Counter(str(label.get("winner")) for label in labels)),
        "base_action_counts": dict(Counter(str(label.get("base_action")) for label in labels)),
        "positive_target_gains": len(target_gains),
        "max_target_gain_vs_base": max(target_gains) if target_gains else 0.0,
        "mean_positive_target_gain_vs_base": _mean(target_gains),
        "positive_final_support_gains": len(support_gains),
        "max_final_support_gain_vs_base": max(support_gains) if support_gains else 0.0,
        "mean_positive_final_support_gain_vs_base": _mean(support_gains),
    }


def write_html(path: Path, payload: dict[str, Any]) -> None:
    summary = payload.get("summary", {})
    labels = payload.get("labels", [])

    def cell(value: object) -> str:
        return escape(str(value))

    rows = []
    for label in labels if isinstance(labels, list) else []:
        rows.append(
            "<tr>"
            f"<td>{cell(label.get('root_seed'))}</td>"
            f"<td>{cell(label.get('source_frame_index'))}</td>"
            f"<td>{cell(label.get('base_action'))}</td>"
            f"<td>{cell(label.get('winner'))}</td>"
            f"<td>{float(label.get('winner_target_rate', 0.0)):.0%}</td>"
            f"<td>{float(label.get('target_gain_vs_base', 0.0)):.0%}</td>"
            f"<td>{float(label.get('winner_final_support_rate', 0.0)):.0%}</td>"
            f"<td><pre>{cell(json.dumps(label.get('action_results'), sort_keys=True))}</pre></td>"
            "</tr>"
        )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Support-Loss Action Labels</title>
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
    th, td {{ border-bottom:1px solid var(--line); padding:6px 8px; text-align:right; vertical-align:top; }}
    th:first-child, td:first-child {{ text-align:left; }}
    pre {{ white-space:pre-wrap; overflow-wrap:anywhere; margin:0; color:var(--muted); }}
  </style>
</head>
<body>
  <main>
    <h1>Support-Loss Action Labels</h1>
    <p class="muted">Forced legal first actions with common-random-number continuations.</p>
    <section class="cards">
      <div class="card"><div class="label">Labels</div><div class="value">{cell(summary.get('labels', 0))}</div></div>
      <div class="card"><div class="label">Rollouts</div><div class="value">{cell(summary.get('valid_rollouts', 0))}</div></div>
      <div class="card"><div class="label">Target</div><div class="value">{float(summary.get('p_target', 0.0)):.0%}</div></div>
      <div class="card"><div class="label">Final Support</div><div class="value">{float(summary.get('p_final_support', 0.0)):.0%}</div></div>
    </section>
    <section class="panel"><pre>{escape(json.dumps(summary, indent=2, sort_keys=True))}</pre></section>
    <section class="panel" style="margin-top:14px;">
      <table><thead><tr><th>Root</th><th>Frame</th><th>Base</th><th>Winner</th><th>Winner Target</th><th>Target Gain</th><th>Winner Support</th><th>Actions</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
    </section>
  </main>
</body>
</html>
"""
    path.write_text(html)


def run_labels(
    *,
    records_json: list[Path],
    policy_name: str,
    target: str,
    horizon: int,
    repeats: int,
    max_starts: int,
    seed: int,
    root_origins: set[str],
    default_starter_tile: int | None,
    first_action_mode: str,
    out_dir: Path,
    progress_every: int = 50,
) -> dict[str, Any]:
    if target not in SUPPORTED_TARGETS:
        raise ValueError(f"Unsupported target: {target}")
    if first_action_mode not in FIRST_ACTION_MODES:
        raise ValueError(f"Unsupported first action mode: {first_action_mode}")
    records = load_records(records_json)
    cases, rejected = load_cases(records, default_starter_tile=default_starter_tile, root_origins=root_origins)
    selected = select_diverse_cases(cases, max_starts=max_starts, seed=seed)
    if not selected:
        raise ValueError("No start cases matched the requested filters")
    policy = make_policy(policy_name)
    rollouts: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    ran = 0
    total = 0
    for case in selected:
        sim = ThreesSim(np.random.default_rng(seed), starter_tile=case.starter_tile)
        actions = select_first_actions(policy=policy, case=case, sim=sim, mode=first_action_mode, rng=np.random.default_rng(seed))
        total += len(actions) * int(repeats)
    completed = 0
    for case_idx, case in enumerate(selected):
        sim_for_actions = ThreesSim(np.random.default_rng(seed + case_idx), starter_tile=case.starter_tile)
        actions = select_first_actions(
            policy=policy,
            case=case,
            sim=sim_for_actions,
            mode=first_action_mode,
            rng=np.random.default_rng(seed + case_idx + 17),
        )
        if not actions:
            continue
        base_action = case.features.get("recorded_action")
        if base_action is None:
            ranked = select_first_actions(
                policy=policy,
                case=case,
                sim=sim_for_actions,
                mode="top-two",
                rng=np.random.default_rng(seed + case_idx + 23),
            )
            base_action = DIRECTION_NAMES[int(ranked[0])] if ranked else DIRECTION_NAMES[int(actions[0])]
        base_action_name = str(base_action)
        case_rows: list[dict[str, Any]] = []
        for repeat_idx in range(int(repeats)):
            crn_seed = int(seed) + case_idx * 1_000_003 + repeat_idx * 10_007
            for action in actions:
                row = rollout_branch(
                    case=case,
                    policy=policy,
                    first_action=int(action),
                    repeat_index=repeat_idx,
                    crn_seed=crn_seed,
                    horizon=horizon,
                    target=target,
                )
                rollouts.append(row)
                case_rows.append(row)
                ran += 1
                completed += 1
                if progress_every > 0 and completed % int(progress_every) == 0:
                    hits = sum(bool(item.get("target_reached")) and not bool(item.get("invalid_first_action")) for item in rollouts)
                    support = sum(bool(item.get("final_support_present")) and not bool(item.get("invalid_first_action")) for item in rollouts)
                    print(f"support_loss_progress {completed}/{total} target={hits} final_support={support}", flush=True)
        labels.append(summarize_case(case, base_action_name=base_action_name, rows=case_rows))
    action_summary = [row for label in labels for row in label.get("action_results", [])]
    summary = summarize(rollouts, labels, ran=ran)
    summary.update(
        {
            "source_records": len(records),
            "cases_total": len(cases),
            "cases_selected": len(selected),
            "rejected": rejected,
            "target": target,
            "horizon": int(horizon),
            "repeats_per_action": int(repeats),
            "first_action_mode": first_action_mode,
            "policy": policy_name,
        }
    )
    payload = {
        "version": 1,
        "kind": "support_loss_action_labels",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "records_json": [str(path) for path in records_json],
        "policy": policy_name,
        "target": target,
        "summary": summary,
        "labels": labels,
        "action_summary": action_summary,
        "rollouts": rollouts,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    payload["json"] = str(out_dir / "support_loss_action_labels.json")
    payload["records_json"] = str(out_dir / "records.json")
    payload["summary_json"] = str(out_dir / "summary.json")
    payload["action_summary_json"] = str(out_dir / "action_summary.json")
    payload["html"] = str(out_dir / "support_loss_action_labels.html")
    write_json(out_dir / "support_loss_action_labels.json", payload)
    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "records.json", rollouts)
    write_json(out_dir / "action_summary.json", action_summary)
    write_html(out_dir / "support_loss_action_labels.html", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-json", type=Path, action="append", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--target", choices=SUPPORTED_TARGETS, default="raw_one_384_with_one_768_no_1536")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--repeats-per-action", type=int, default=8)
    parser.add_argument("--max-starts", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--root-origin", help="Comma-separated root origins; defaults to fresh,human.")
    parser.add_argument("--starter", default="1536")
    parser.add_argument("--first-action-mode", choices=FIRST_ACTION_MODES, default="all")
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--out-dir", type=Path, default=Path("threes_rl/runs/forensics/support_loss_action_labels/latest"))
    args = parser.parse_args()
    starter_text = args.starter.strip().lower()
    default_starter = None if starter_text == "none" else int(starter_text)
    payload = run_labels(
        records_json=args.records_json,
        policy_name=args.policy,
        target=args.target,
        horizon=args.horizon,
        repeats=args.repeats_per_action,
        max_starts=args.max_starts,
        seed=args.seed,
        root_origins=parse_root_origins(args.root_origin),
        default_starter_tile=default_starter,
        first_action_mode=args.first_action_mode,
        out_dir=args.out_dir,
        progress_every=args.progress_every,
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"json={payload['json']}")
    print(f"records={payload['records_json']}")
    print(f"html={payload['html']}")


if __name__ == "__main__":
    main()
