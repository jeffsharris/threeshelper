"""Audit which forced first actions reach support-ladder milestones."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from html import escape
from pathlib import Path
from statistics import mean, median
from typing import Any
from uuid import uuid4

import numpy as np

from threes_rl.continue_from_replays import _flatten_paths, collect_start_cases, select_start_cases
from threes_rl.endgame_action_labels import (
    _action_record_from_progress_entry,
    _load_action_progress,
    _progress_entry_for_action_record,
    _write_action_progress,
    action_continuation_progress_key,
    run_first_action_continuation,
)
from threes_rl.eval import make_policy
from threes_rl.run_artifacts import safe_name, write_json
from threes_rl.sim import DIRECTION_NAMES, ThreesSim
from threes_rl.support_chain_progression import analyze_replay
from threes_rl.train_td import parse_phase_filter


def _milestone_present(analysis: dict[str, Any], name: str) -> bool:
    milestones = analysis.get("milestones")
    return isinstance(milestones, dict) and milestones.get(name) is not None


def _record_metrics(record: object) -> dict[str, Any]:
    replay = getattr(record, "replay")
    analysis = analyze_replay_payload(replay)
    return {
        "raw_duplicate_1536": _milestone_present(analysis, "first_raw_duplicate_1536"),
        "raw_adjacent_1536": _milestone_present(analysis, "first_raw_adjacent_pair_1536"),
        "second_3072": analysis.get("second_3072_event") is not None,
        "reached_6144": analysis.get("first_6144") is not None,
        "max_raw_count_1536": analysis.get("post_3072_max_raw_count_1536"),
        "max_raw_duplicate_tile": analysis.get("post_3072_max_raw_duplicate_tile"),
        "max_raw_adjacent_pair_tile": analysis.get("post_3072_max_raw_adjacent_pair_tile"),
    }


def analyze_replay_payload(replay: dict[str, Any]) -> dict[str, Any]:
    tmp_path = Path("/tmp") / f"first_action_milestone_{uuid4().hex}.json"
    write_json(tmp_path, replay)
    try:
        return analyze_replay(tmp_path)
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass


def _record_payload(record: object, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "start_id": str(getattr(record, "start_id")),
        "action": int(getattr(record, "action")),
        "action_name": str(getattr(record, "action_name")),
        "repeat_index": int(getattr(record, "repeat_index")),
        "seed": int(getattr(record, "seed")),
        "score": int(getattr(record, "score")),
        "score_delta": int(getattr(record, "score_delta")),
        "moves_delta": int(getattr(record, "moves_delta")),
        "max_tile": int(getattr(record, "max_tile")),
        "max_tile_excl_starter": int(getattr(record, "max_tile_excl_starter")),
        "source_replay": getattr(record.start_case, "source_replay"),
        "source_seed": getattr(record.start_case, "source_seed"),
        "source_frame_index": int(getattr(record.start_case, "frame_index")),
        "start_score": int(getattr(record.start_case, "start_score")),
        "start_max_tile_excl_starter": int(getattr(record.start_case, "start_max_tile_excl_starter")),
        **metrics,
    }


def _rate(records: list[dict[str, Any]], key: str) -> float:
    return 0.0 if not records else sum(bool(row.get(key)) for row in records) / float(len(records))


def _winner_for_action_rows(rows: list[dict[str, Any]], target: str) -> str | None:
    if not rows:
        return None
    target_rate = f"p_{target}"
    return str(
        max(
            rows,
            key=lambda row: (
                float(row.get(target_rate, 0.0)),
                float(row.get("p_raw_adjacent_1536", 0.0)),
                float(row.get("p_raw_duplicate_1536", 0.0)),
                float(row.get("p_second_3072", 0.0)),
                float(row.get("p_reached_6144", 0.0)),
                float(row.get("mean_delta", 0.0)),
                str(row.get("action")),
            ),
        )["action"]
    )


def summarize_start(
    *,
    start_case: object,
    base_action_name: str,
    rows: list[dict[str, Any]],
    target: str,
) -> dict[str, Any]:
    by_action: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_action.setdefault(str(row["action_name"]), []).append(row)
    action_rows = []
    for action_name, action_records in sorted(by_action.items()):
        deltas = [int(row["score_delta"]) for row in action_records]
        action_rows.append(
            {
                "action": action_name,
                "repeats": len(action_records),
                "p_raw_duplicate_1536": _rate(action_records, "raw_duplicate_1536"),
                "p_raw_adjacent_1536": _rate(action_records, "raw_adjacent_1536"),
                "p_second_3072": _rate(action_records, "second_3072"),
                "p_reached_6144": _rate(action_records, "reached_6144"),
                "mean_delta": float(mean(deltas)),
                "median_delta": float(median(deltas)),
                "max_delta": int(max(deltas)),
                "max_tile_excl_starter": int(max(int(row["max_tile_excl_starter"]) for row in action_records)),
            }
        )
    winner = _winner_for_action_rows(action_rows, target)
    row_by_action = {str(row["action"]): row for row in action_rows}
    winner_row = row_by_action.get(str(winner), {})
    base_row = row_by_action.get(base_action_name, {})
    return {
        "id": str(getattr(start_case, "id")),
        "source_replay": str(getattr(start_case, "source_replay")),
        "source_seed": getattr(start_case, "source_seed"),
        "source_frame_index": int(getattr(start_case, "frame_index")),
        "start_score": int(getattr(start_case, "start_score")),
        "start_max_tile_excl_starter": int(getattr(start_case, "start_max_tile_excl_starter")),
        "phase": str(getattr(start_case, "phase")),
        "base_action": base_action_name,
        "winner": winner,
        "winner_target_rate": float(winner_row.get(f"p_{target}", 0.0)),
        "base_target_rate": float(base_row.get(f"p_{target}", 0.0)),
        "target_rate_gain_vs_base": float(winner_row.get(f"p_{target}", 0.0))
        - float(base_row.get(f"p_{target}", 0.0)),
        "action_results": action_rows,
    }


def summarize(records: list[dict[str, Any]], labels: list[dict[str, Any]], *, ran: int, resumed: int) -> dict[str, Any]:
    positive_gains = [float(label.get("target_rate_gain_vs_base", 0.0)) for label in labels if float(label.get("target_rate_gain_vs_base", 0.0)) > 0.0]
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "labels": len(labels),
        "continuations": len(records),
        "continuations_ran": int(ran),
        "continuations_resumed": int(resumed),
        "starts_with_any_raw_duplicate_1536": len({row["start_id"] for row in records if row.get("raw_duplicate_1536")}),
        "starts_with_any_raw_adjacent_1536": len({row["start_id"] for row in records if row.get("raw_adjacent_1536")}),
        "p_any_raw_duplicate_1536": _rate(records, "raw_duplicate_1536"),
        "p_any_raw_adjacent_1536": _rate(records, "raw_adjacent_1536"),
        "p_any_second_3072": _rate(records, "second_3072"),
        "p_any_6144": _rate(records, "reached_6144"),
        "winner_counts": dict(Counter(str(label.get("winner")) for label in labels)),
        "base_action_counts": dict(Counter(str(label.get("base_action")) for label in labels)),
        "positive_target_rate_gains": len(positive_gains),
        "max_target_rate_gain_vs_base": max(positive_gains) if positive_gains else 0.0,
        "mean_positive_target_rate_gain_vs_base": float(mean(positive_gains)) if positive_gains else 0.0,
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
            f"<td>{float(label.get('winner_target_rate', 0.0)):.0%}</td>"
            f"<td>{float(label.get('target_rate_gain_vs_base', 0.0)):.0%}</td>"
            f"<td><pre>{cell(json.dumps(label.get('action_results'), sort_keys=True))}</pre></td>"
            "</tr>"
        )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>First-Action Milestone Audit</title>
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
    <h1>First-Action Milestone Audit</h1>
    <p class="muted">Forced legal first actions, then frozen actor continuation, scored by support-ladder milestones.</p>
    <section class="cards">
      <div class="card"><div class="label">Labels</div><div class="value">{cell(summary.get('labels', 0))}</div></div>
      <div class="card"><div class="label">Continuations</div><div class="value">{cell(summary.get('continuations', 0))}</div></div>
      <div class="card"><div class="label">Raw Dup 1536</div><div class="value">{float(summary.get('p_any_raw_duplicate_1536', 0.0)):.0%}</div></div>
      <div class="card"><div class="label">Raw Adj 1536</div><div class="value">{float(summary.get('p_any_raw_adjacent_1536', 0.0)):.0%}</div></div>
    </section>
    <section class="panel"><pre>{escape(json.dumps(summary, indent=2, sort_keys=True))}</pre></section>
    <section class="panel" style="margin-top: 14px;">
      <table><thead><tr><th>Seed</th><th>Frame</th><th>Base</th><th>Winner</th><th>Winner Target</th><th>Gain</th><th>Actions</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
    </section>
  </main>
</body>
</html>
"""
    path.write_text(html)


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    replay_paths = _flatten_paths(args.replay_json)
    phase_filter = set(parse_phase_filter(args.phase_filter)) if args.phase_filter else None
    starter_text = args.starter.strip().lower()
    default_starter = None if starter_text == "none" else int(starter_text)
    start_cases = collect_start_cases(
        replay_paths,
        min_tile=args.start_state_min_tile,
        phase_filter=phase_filter,
        default_starter_tile=default_starter,
    )
    selected_cases = select_start_cases(start_cases, max_starts=args.max_starts, seed=args.seed)
    if not selected_cases:
        raise ValueError("No start cases matched the requested filters")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = getattr(args, "continuation_progress_json", None)
    if progress_path is None and bool(getattr(args, "checkpoint_continuations", False)):
        progress_path = args.out_dir / "action_milestone_progress.json"
    progress_payload: dict[str, Any] | None = None
    progress_entries: dict[str, Any] | None = None
    if progress_path is not None:
        progress_path = Path(progress_path)
        progress_payload = _load_action_progress(progress_path)
        progress_entries_obj = progress_payload.setdefault("entries", {})
        if not isinstance(progress_entries_obj, dict):
            raise ValueError(f"{progress_path} has invalid action-continuation entries")
        progress_entries = progress_entries_obj

    policy = make_policy(args.policy)
    all_records: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    ran = 0
    resumed = 0
    completed = 0
    total = 0
    for case in selected_cases:
        sim_for_legal = ThreesSim(np.random.default_rng(args.seed), starter_tile=case.starter_tile)
        total += len(sim_for_legal.legal_actions(case.state)) * int(args.repeats)

    for case_idx, start_case in enumerate(selected_cases):
        sim_for_legal = ThreesSim(np.random.default_rng(int(args.seed) + case_idx), starter_tile=start_case.starter_tile)
        legal_actions = sim_for_legal.legal_actions(start_case.state)
        if not legal_actions:
            continue
        base_action = int(policy(start_case.state, sim_for_legal, np.random.default_rng(int(args.seed) + case_idx + 37)))
        base_action_name = DIRECTION_NAMES[base_action]
        case_rows: list[dict[str, Any]] = []
        for repeat_idx in range(int(args.repeats)):
            repeat_seed = int(args.seed) + case_idx * 1_000_003 + repeat_idx * 10_007
            for action in legal_actions:
                progress_key = action_continuation_progress_key(
                    policy_name=args.policy,
                    start_case=start_case,
                    action=int(action),
                    repeat_index=repeat_idx,
                    seed=repeat_seed,
                    max_moves=args.max_moves,
                )
                if progress_entries is not None and isinstance(progress_entries.get(progress_key), dict):
                    record = _action_record_from_progress_entry(progress_entries[progress_key], start_case)
                    resumed += 1
                else:
                    record = run_first_action_continuation(
                        policy=policy,
                        policy_name=args.policy,
                        start_case=start_case,
                        first_action=int(action),
                        repeat_index=repeat_idx,
                        seed=repeat_seed,
                        max_moves=args.max_moves,
                    )
                    ran += 1
                    if progress_entries is not None and progress_payload is not None and progress_path is not None:
                        progress_entries[progress_key] = _progress_entry_for_action_record(
                            key=progress_key,
                            policy_name=args.policy,
                            seed=repeat_seed,
                            max_moves=args.max_moves,
                            record=record,
                        )
                        _write_action_progress(progress_path, progress_payload)
                metrics = _record_metrics(record)
                payload = _record_payload(record, metrics)
                all_records.append(payload)
                case_rows.append(payload)
                completed += 1
                if args.progress_every > 0 and completed % int(args.progress_every) == 0:
                    print(
                        "progress "
                        f"{completed}/{total} "
                        f"raw_dup1536={sum(bool(row.get('raw_duplicate_1536')) for row in all_records)} "
                        f"raw_adj1536={sum(bool(row.get('raw_adjacent_1536')) for row in all_records)}",
                        flush=True,
                    )
        labels.append(
            summarize_start(
                start_case=start_case,
                base_action_name=base_action_name,
                rows=case_rows,
                target=args.target,
            )
        )

    summary = summarize(all_records, labels, ran=ran, resumed=resumed)
    if progress_path is not None:
        summary["continuation_progress_json"] = str(progress_path)
    payload = {
        "version": 1,
        "kind": "first_action_milestone_audit",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "policy": args.policy,
        "source_replays": [str(path) for path in replay_paths],
        "start_cases_total": len(start_cases),
        "selected_start_cases": len(selected_cases),
        "repeats": int(args.repeats),
        "max_moves": int(args.max_moves),
        "target": args.target,
        "summary": summary,
        "labels": labels,
        "records": all_records,
    }
    payload["json"] = str(args.out_dir / "first_action_milestone_audit.json")
    payload["records_json"] = str(args.out_dir / "records.json")
    payload["html"] = str(args.out_dir / "first_action_milestone_audit.html")
    write_json(args.out_dir / "first_action_milestone_audit.json", payload)
    write_json(args.out_dir / "records.json", all_records)
    write_html(args.out_dir / "first_action_milestone_audit.html", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--replay-json", type=Path, nargs="+", action="append", required=True)
    parser.add_argument("--starter", default="1536")
    parser.add_argument("--start-state-min-tile", type=int, default=3072)
    parser.add_argument("--phase-filter", help="Comma-separated phase names/aliases for start states.")
    parser.add_argument("--max-starts", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--max-moves", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument(
        "--target",
        choices=["raw_duplicate_1536", "raw_adjacent_1536", "second_3072", "reached_6144"],
        default="raw_adjacent_1536",
    )
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--checkpoint-continuations", action="store_true")
    parser.add_argument("--continuation-progress-json", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = run_from_args(args)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"json={payload['json']}")
    print(f"records={payload['records_json']}")
    print(f"html={payload['html']}")


if __name__ == "__main__":
    main()
