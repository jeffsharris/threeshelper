"""Audit first actions for preserving 768 support through first 1536 creation."""

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
from threes_rl.rare_event_frontier import load_cases, load_records, parse_root_origins, select_diverse_cases
from threes_rl.run_artifacts import write_json
from threes_rl.sim import DIRECTION_NAMES, SimState, ThreesSim, score_board
from threes_rl.support_ladder_window_reservoir import raw_ladder_features
from threes_rl.train_td import copy_state


def _empty_count(state: SimState) -> int:
    return int(np.count_nonzero(np.asarray(state.board, dtype=np.int32) == 0))


def _support_with_1536(raw: dict[str, Any]) -> bool:
    return bool(raw.get("raw_has_adjacent_768")) and int(raw.get("raw_count_1536", 0)) >= 1


def _raw_snapshot(state: SimState, starter_tile: int | None) -> dict[str, Any]:
    raw = raw_ladder_features(state.board, starter_tile)
    return {
        "raw_count_768": int(raw["raw_count_768"]),
        "raw_count_1536": int(raw["raw_count_1536"]),
        "raw_count_3072": int(raw["raw_count_3072"]),
        "raw_has_adjacent_768": bool(raw["raw_has_adjacent_768"]),
        "raw_has_adjacent_1536": bool(raw["raw_has_adjacent_1536"]),
        "raw_highest_duplicate_tile": int(raw["raw_highest_duplicate_tile"]),
        "raw_highest_adjacent_pair_tile": int(raw["raw_highest_adjacent_pair_tile"]),
        "empty_count": _empty_count(state),
    }


def _mean_or_none(values: list[int | float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return float(mean(clean)) if clean else None


def _median_or_none(values: list[int | float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return float(median(clean)) if clean else None


def rollout_branch(
    *,
    case: object,
    policy: object,
    first_action: int,
    repeat_index: int,
    seed: int,
    horizon: int,
) -> dict[str, Any]:
    sim = ThreesSim(np.random.default_rng(seed), starter_tile=getattr(case, "starter_tile"))
    policy_rng = np.random.default_rng(seed + 37)
    state = copy_state(getattr(case, "state"))
    start_score = int(score_board(state.board))
    start_move = int(state.move_count)
    start_raw = _raw_snapshot(state, getattr(case, "starter_tile"))
    first_1536: dict[str, Any] | None = None
    support_ever = _support_with_1536(start_raw)
    moves_to_support_ever: int | None = 0 if support_ever else None
    invalid = False
    actions: list[str] = []

    if int(start_raw["raw_count_1536"]) >= 1:
        first_1536 = {
            **start_raw,
            "moves_to_one1536": 0,
            "score_delta_at_one1536": 0,
        }

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
        raw = _raw_snapshot(state, getattr(case, "starter_tile"))
        moves_delta = int(state.move_count - start_move)
        score_delta = int(score_board(state.board) - start_score)
        if first_1536 is None and int(raw["raw_count_1536"]) >= 1:
            first_1536 = {
                **raw,
                "moves_to_one1536": moves_delta,
                "score_delta_at_one1536": score_delta,
            }
        if not support_ever and _support_with_1536(raw):
            support_ever = True
            moves_to_support_ever = moves_delta
        if state.game_over:
            break

    final_raw = _raw_snapshot(state, getattr(case, "starter_tile"))
    first = first_1536 or {}
    return {
        "case_id": str(getattr(case, "id")),
        "first_action": DIRECTION_NAMES[int(first_action)],
        "repeat_index": int(repeat_index),
        "seed": int(seed),
        "invalid_first_action": bool(invalid),
        "one1536_reached": first_1536 is not None,
        "moves_to_one1536": first.get("moves_to_one1536"),
        "support_preserved_at_one1536": bool(first.get("raw_has_adjacent_768", False)) if first_1536 is not None else False,
        "support_with_1536_ever": bool(support_ever),
        "moves_to_support_with_1536": moves_to_support_ever,
        "raw_count_768_at_one1536": first.get("raw_count_768"),
        "raw_highest_adjacent_pair_tile_at_one1536": first.get("raw_highest_adjacent_pair_tile"),
        "empty_count_at_one1536": first.get("empty_count"),
        "score_delta_at_one1536": first.get("score_delta_at_one1536"),
        "moves_delta": int(state.move_count - start_move),
        "score_delta": int(score_board(state.board) - start_score),
        "game_over": bool(state.game_over),
        "actions": actions,
        "start_raw_count_768": int(start_raw["raw_count_768"]),
        "start_raw_count_1536": int(start_raw["raw_count_1536"]),
        "start_raw_has_adjacent_768": bool(start_raw["raw_has_adjacent_768"]),
        "start_empty_count": int(start_raw["empty_count"]),
        "final_raw_count_768": int(final_raw["raw_count_768"]),
        "final_raw_count_1536": int(final_raw["raw_count_1536"]),
        "final_raw_has_adjacent_768": bool(final_raw["raw_has_adjacent_768"]),
        "final_raw_highest_adjacent_pair_tile": int(final_raw["raw_highest_adjacent_pair_tile"]),
        "final_empty_count": int(final_raw["empty_count"]),
        "source_replay": getattr(case, "source_replay"),
        "source_seed": getattr(case, "source_seed"),
        "source_frame_index": getattr(case, "source_frame_index"),
        "root_origin": getattr(case, "root_origin"),
        "root_replay": getattr(case, "root_replay"),
        "root_seed": getattr(case, "root_seed"),
        "root_policy_family": getattr(case, "root_policy_family"),
        "ancestry_key": getattr(case, "ancestry_key"),
    }


def _rate(rows: list[dict[str, Any]], key: str) -> float:
    return 0.0 if not rows else sum(bool(row.get(key)) for row in rows) / float(len(rows))


def _action_summary(case_id: str, action_name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = [int(row["score_delta"]) for row in rows]
    return {
        "case_id": case_id,
        "first_action": action_name,
        "repeats": len(rows),
        "p_one1536": _rate(rows, "one1536_reached"),
        "p_support_preserved_at_one1536": _rate(rows, "support_preserved_at_one1536"),
        "p_support_with_1536_ever": _rate(rows, "support_with_1536_ever"),
        "mean_moves_to_one1536": _mean_or_none([row.get("moves_to_one1536") for row in rows]),
        "median_moves_to_one1536": _median_or_none([row.get("moves_to_one1536") for row in rows]),
        "mean_score_delta": float(mean(deltas)) if deltas else 0.0,
        "median_score_delta": float(median(deltas)) if deltas else 0.0,
        "max_score_delta": int(max(deltas)) if deltas else 0,
        "terminal_rate": _rate(rows, "game_over"),
    }


def _winner(action_rows: list[dict[str, Any]]) -> str | None:
    if not action_rows:
        return None
    return str(
        max(
            action_rows,
            key=lambda row: (
                float(row.get("p_support_preserved_at_one1536", 0.0)),
                float(row.get("p_support_with_1536_ever", 0.0)),
                float(row.get("p_one1536", 0.0)),
                float(row.get("mean_score_delta", 0.0)),
                str(row.get("first_action")),
            ),
        )["first_action"]
    )


def summarize_case(*, case: object, base_action_name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_action: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_action.setdefault(str(row["first_action"]), []).append(row)
    action_rows = [_action_summary(str(getattr(case, "id")), action, action_rows) for action, action_rows in sorted(by_action.items())]
    winner = _winner(action_rows)
    row_by_action = {str(row["first_action"]): row for row in action_rows}
    winner_row = row_by_action.get(str(winner), {})
    base_row = row_by_action.get(base_action_name, {})
    return {
        "case_id": str(getattr(case, "id")),
        "source_replay": getattr(case, "source_replay"),
        "source_seed": getattr(case, "source_seed"),
        "source_frame_index": getattr(case, "source_frame_index"),
        "root_seed": getattr(case, "root_seed"),
        "root_policy_family": getattr(case, "root_policy_family"),
        "base_action": base_action_name,
        "winner": winner,
        "winner_support_preserved_rate": float(winner_row.get("p_support_preserved_at_one1536", 0.0)),
        "base_support_preserved_rate": float(base_row.get("p_support_preserved_at_one1536", 0.0)),
        "support_preserved_gain_vs_base": float(winner_row.get("p_support_preserved_at_one1536", 0.0))
        - float(base_row.get("p_support_preserved_at_one1536", 0.0)),
        "winner_one1536_rate": float(winner_row.get("p_one1536", 0.0)),
        "base_one1536_rate": float(base_row.get("p_one1536", 0.0)),
        "start_features": dict(getattr(case, "features")),
        "start_raw": dict(getattr(case, "raw")),
        "action_results": action_rows,
    }


def summarize(rollouts: list[dict[str, Any]], labels: list[dict[str, Any]], *, ran: int) -> dict[str, Any]:
    positive_gains = [
        float(label.get("support_preserved_gain_vs_base", 0.0))
        for label in labels
        if float(label.get("support_preserved_gain_vs_base", 0.0)) > 0.0
    ]
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "labels": len(labels),
        "rollouts": len(rollouts),
        "rollouts_ran": int(ran),
        "start_cases_with_adjacent_768": sum(bool((label.get("start_raw") or {}).get("raw_has_adjacent_768")) for label in labels),
        "p_one1536": _rate(rollouts, "one1536_reached"),
        "p_support_preserved_at_one1536": _rate(rollouts, "support_preserved_at_one1536"),
        "p_support_with_1536_ever": _rate(rollouts, "support_with_1536_ever"),
        "starts_with_any_one1536": len({row["case_id"] for row in rollouts if row.get("one1536_reached")}),
        "starts_with_any_support_preserved": len({row["case_id"] for row in rollouts if row.get("support_preserved_at_one1536")}),
        "starts_with_any_support_with_1536_ever": len({row["case_id"] for row in rollouts if row.get("support_with_1536_ever")}),
        "winner_counts": dict(Counter(str(label.get("winner")) for label in labels)),
        "base_action_counts": dict(Counter(str(label.get("base_action")) for label in labels)),
        "positive_support_preserved_gains": len(positive_gains),
        "max_support_preserved_gain_vs_base": max(positive_gains) if positive_gains else 0.0,
        "mean_positive_support_preserved_gain_vs_base": float(mean(positive_gains)) if positive_gains else 0.0,
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
            f"<td>{cell(label.get('source_seed'))}</td>"
            f"<td>{cell(label.get('source_frame_index'))}</td>"
            f"<td>{cell(label.get('base_action'))}</td>"
            f"<td>{cell(label.get('winner'))}</td>"
            f"<td>{float(label.get('winner_support_preserved_rate', 0.0)):.0%}</td>"
            f"<td>{float(label.get('support_preserved_gain_vs_base', 0.0)):.0%}</td>"
            f"<td><pre>{cell(json.dumps(label.get('action_results'), sort_keys=True))}</pre></td>"
            "</tr>"
        )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>First-Action Support Preservation Audit</title>
  <style>
    :root {{ color-scheme: dark; --bg: #101318; --panel: #171d24; --line: #34404d; --text: #edf2f7; --muted: #aab6c2; --gold: #f2c14e; }}
    body {{ margin: 0; background: var(--bg); color: var(--text); font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ width: min(1180px, calc(100vw - 32px)); margin: 0 auto; padding: 24px 0 40px; }}
    h1 {{ margin: 0 0 8px; font-size: 24px; }}
    .muted {{ color: var(--muted); }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 18px 0; }}
    .card, .panel {{ border: 1px solid var(--line); background: var(--panel); border-radius: 8px; padding: 12px; }}
    .label {{ color: var(--muted); font-size: 11px; text-transform: uppercase; }}
    .value {{ margin-top: 4px; color: var(--gold); font-size: 22px; font-weight: 800; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 6px 8px; text-align: right; vertical-align: top; }}
    th:first-child, td:first-child {{ text-align: left; }}
    pre {{ white-space: pre-wrap; overflow-wrap: anywhere; margin: 0; color: var(--muted); }}
  </style>
</head>
<body>
  <main>
    <h1>First-Action Support Preservation Audit</h1>
    <p class="muted">Forced legal first actions, then frozen actor continuation, scored at first raw 1536 creation.</p>
    <section class="cards">
      <div class="card"><div class="label">Labels</div><div class="value">{cell(summary.get('labels', 0))}</div></div>
      <div class="card"><div class="label">Rollouts</div><div class="value">{cell(summary.get('rollouts', 0))}</div></div>
      <div class="card"><div class="label">One 1536</div><div class="value">{float(summary.get('p_one1536', 0.0)):.0%}</div></div>
      <div class="card"><div class="label">Support Preserved</div><div class="value">{float(summary.get('p_support_preserved_at_one1536', 0.0)):.0%}</div></div>
    </section>
    <section class="panel"><pre>{escape(json.dumps(summary, indent=2, sort_keys=True))}</pre></section>
    <section class="panel" style="margin-top: 14px;">
      <table><thead><tr><th>Seed</th><th>Frame</th><th>Base</th><th>Winner</th><th>Winner Preserve</th><th>Gain</th><th>Actions</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
    </section>
  </main>
</body>
</html>
"""
    path.write_text(html)


def run_audit(
    *,
    records_json: list[Path],
    policy_name: str,
    horizon: int,
    repeats: int,
    max_starts: int,
    seed: int,
    root_origins: set[str],
    default_starter_tile: int | None,
    require_start_adjacent_768: bool,
    out_dir: Path,
    progress_every: int = 25,
) -> dict[str, Any]:
    records = load_records(records_json)
    cases, rejected = load_cases(records, default_starter_tile=default_starter_tile, root_origins=root_origins)
    if require_start_adjacent_768:
        kept = [case for case in cases if bool(case.raw.get("raw_has_adjacent_768"))]
        rejected["missing_start_adjacent_768"] = len(cases) - len(kept)
        cases = kept
    selected = select_diverse_cases(cases, max_starts=max_starts, seed=seed)
    if not selected:
        raise ValueError("No start cases matched the requested filters")

    policy = make_policy(policy_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    rollouts: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    ran = 0
    completed = 0
    total = 0
    for case in selected:
        sim_for_legal = ThreesSim(np.random.default_rng(seed), starter_tile=case.starter_tile)
        total += len(sim_for_legal.legal_actions(case.state)) * int(repeats)

    for case_idx, case in enumerate(selected):
        sim_for_legal = ThreesSim(np.random.default_rng(seed + case_idx), starter_tile=case.starter_tile)
        legal_actions = sim_for_legal.legal_actions(case.state)
        if not legal_actions:
            continue
        base_action = int(policy(case.state, sim_for_legal, np.random.default_rng(seed + case_idx + 37)))
        if base_action not in legal_actions:
            base_action = int(legal_actions[0])
        base_action_name = DIRECTION_NAMES[base_action]
        case_rows: list[dict[str, Any]] = []
        for repeat_idx in range(int(repeats)):
            repeat_seed = int(seed) + case_idx * 1_000_003 + repeat_idx * 10_007
            for action in legal_actions:
                row = rollout_branch(
                    case=case,
                    policy=policy,
                    first_action=int(action),
                    repeat_index=repeat_idx,
                    seed=repeat_seed + int(action) * 503,
                    horizon=horizon,
                )
                rollouts.append(row)
                case_rows.append(row)
                ran += 1
                completed += 1
                if progress_every > 0 and completed % int(progress_every) == 0:
                    print(
                        "support_progress "
                        f"{completed}/{total} "
                        f"one1536={sum(bool(item.get('one1536_reached')) for item in rollouts)} "
                        f"preserved={sum(bool(item.get('support_preserved_at_one1536')) for item in rollouts)}",
                        flush=True,
                    )
        labels.append(summarize_case(case=case, base_action_name=base_action_name, rows=case_rows))

    summary = summarize(rollouts, labels, ran=ran)
    summary.update(
        {
            "source_records": len(records),
            "cases_total": len(cases),
            "cases_selected": len(selected),
            "rejected": rejected,
            "horizon": int(horizon),
            "repeats_per_action": int(repeats),
            "require_start_adjacent_768": bool(require_start_adjacent_768),
        }
    )
    action_rows = [row for label in labels for row in label.get("action_results", [])]
    payload = {
        "version": 1,
        "kind": "first_action_support_preservation_audit",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "records_json": [str(path) for path in records_json],
        "policy": policy_name,
        "summary": summary,
        "labels": labels,
        "action_summary": action_rows,
        "rollouts": rollouts,
    }
    payload["json"] = str(out_dir / "first_action_support_preservation_audit.json")
    payload["records_json"] = str(out_dir / "records.json")
    payload["action_summary_json"] = str(out_dir / "action_summary.json")
    payload["summary_json"] = str(out_dir / "summary.json")
    payload["html"] = str(out_dir / "first_action_support_preservation_audit.html")
    write_json(out_dir / "first_action_support_preservation_audit.json", payload)
    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "records.json", rollouts)
    write_json(out_dir / "action_summary.json", action_rows)
    write_html(out_dir / "first_action_support_preservation_audit.html", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-json", type=Path, action="append", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--horizon", type=int, default=40)
    parser.add_argument("--repeats-per-action", type=int, default=4)
    parser.add_argument("--max-starts", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--root-origin", help="Comma-separated root origins; defaults to fresh,human.")
    parser.add_argument("--starter", default="1536")
    parser.add_argument("--require-start-adjacent-768", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--out-dir", type=Path, default=Path("threes_rl/runs/forensics/first_action_support_preservation/latest"))
    args = parser.parse_args()
    starter_text = args.starter.strip().lower()
    default_starter = None if starter_text == "none" else int(starter_text)
    payload = run_audit(
        records_json=args.records_json,
        policy_name=args.policy,
        horizon=args.horizon,
        repeats=args.repeats_per_action,
        max_starts=args.max_starts,
        seed=args.seed,
        root_origins=parse_root_origins(args.root_origin),
        default_starter_tile=default_starter,
        require_start_adjacent_768=bool(args.require_start_adjacent_768),
        out_dir=args.out_dir,
        progress_every=args.progress_every,
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"json={payload['json']}")
    print(f"records={payload['records_json']}")
    print(f"html={payload['html']}")


if __name__ == "__main__":
    main()
