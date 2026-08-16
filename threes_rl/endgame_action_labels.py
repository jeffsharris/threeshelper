"""Label first-action choices from high-board replay states."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from collections import Counter
from dataclasses import dataclass
from html import escape
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np

from threes_rl.continue_from_replays import StartCase, _flatten_paths, collect_start_cases, select_start_cases
from threes_rl.eval import make_policy, max_tile_excluding_initial_starter
from threes_rl.record_replay import preview_payload, state_payload, write_html
from threes_rl.run_artifacts import safe_name, write_json
from threes_rl.sim import DIRECTION_NAMES, SimState, ThreesSim, score_board
from threes_rl.swing_label import state_features
from threes_rl.train_td import copy_state, parse_phase_filter


@dataclass
class ActionContinuation:
    start_id: str
    action: int
    action_name: str
    repeat_index: int
    seed: int
    score: int
    score_delta: int
    moves_delta: int
    max_tile: int
    max_tile_excl_starter: int
    advanced_to_6144: bool
    replay: dict[str, Any]
    start_case: StartCase


def _clone_state(state: SimState) -> SimState:
    return copy_state(state)


def action_continuation_progress_key(
    *,
    policy_name: str,
    start_case: StartCase,
    action: int,
    repeat_index: int,
    seed: int,
    max_moves: int,
) -> str:
    raw = json.dumps(
        {
            "version": 1,
            "policy": str(policy_name),
            "seed": int(seed),
            "repeat_index": int(repeat_index),
            "first_action": int(action),
            "max_moves": int(max_moves),
            "start": {
                "id": start_case.id,
                "source_replay": start_case.source_replay,
                "source_seed": start_case.source_seed,
                "source_frame_index": int(start_case.frame_index),
                "starter_tile": start_case.starter_tile,
                "start_score": int(start_case.start_score),
                "start_max_tile_excl_starter": int(start_case.start_max_tile_excl_starter),
                "move_count": int(start_case.state.move_count),
                "board": [int(value) for value in np.asarray(start_case.state.board, dtype=np.int32).reshape(-1)],
                "preview": {
                    "kind": start_case.state.preview.kind,
                    "value": start_case.state.preview.value,
                    "candidates": [int(value) for value in start_case.state.preview.candidates],
                },
                "tile_cycle": {
                    "small_counts": {
                        str(key): int(value)
                        for key, value in sorted(start_case.state.small_counts.items())
                    },
                    "small_pos": int(start_case.state.small_pos),
                    "small_seen_total": int(start_case.state.small_seen_total),
                    "span_small_pos": int(start_case.state.span_small_pos),
                    "large_pending": bool(start_case.state.large_pending),
                    "max_tile": int(start_case.state.max_tile),
                },
            },
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _load_action_progress(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "version": 1,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "updated_at": None,
            "entries": {},
        }
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not an action-continuation progress object")
    entries = payload.setdefault("entries", {})
    if not isinstance(entries, dict):
        raise ValueError(f"{path} has invalid action-continuation entries")
    payload.setdefault("version", 1)
    return payload


def _write_action_progress(path: Path, progress: dict[str, Any]) -> None:
    progress["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, progress)


def parse_corner_risk_filter(text: str | None) -> set[str] | None:
    if text is None or not text.strip():
        return None
    aliases = {
        "low": "low_corner_risk",
        "medium": "medium_corner_risk",
        "med": "medium_corner_risk",
        "high": "high_corner_risk",
    }
    values: set[str] = set()
    allowed = {"low_corner_risk", "medium_corner_risk", "high_corner_risk"}
    for part in text.split(","):
        raw = part.strip().lower()
        if not raw:
            continue
        normalized = aliases.get(raw, raw)
        if normalized not in allowed:
            raise ValueError(f"Unsupported corner-risk bucket: {part}")
        values.add(normalized)
    return values or None


def _features_for_start_case(start_case: StartCase) -> dict[str, Any]:
    sim = ThreesSim(np.random.default_rng(0), starter_tile=start_case.starter_tile)
    return state_features(start_case.state, sim, start_case.starter_tile)


def _step_with_fallback(sim: ThreesSim, state: SimState, action: int) -> tuple[SimState, object, int] | None:
    next_state, info = sim.step(state, int(action))
    if info.moved:
        return next_state, info, int(action)
    legal = sim.legal_actions(state)
    if not legal:
        return None
    next_state, info = sim.step(state, int(legal[0]))
    if not info.moved:
        return None
    return next_state, info, int(legal[0])


def run_first_action_continuation(
    *,
    policy: object,
    policy_name: str,
    start_case: StartCase,
    first_action: int,
    repeat_index: int,
    seed: int,
    max_moves: int,
) -> ActionContinuation:
    sim_seed = int(seed)
    sim = ThreesSim(np.random.default_rng(sim_seed), starter_tile=start_case.starter_tile)
    policy_rng = np.random.default_rng(sim_seed + 37)
    state = _clone_state(start_case.state)
    start_score = int(score_board(state.board))
    start_move_count = int(state.move_count)
    frames: list[dict[str, Any]] = [{"index": 0, "state": state_payload(state, sim), "move": None}]

    forced = True
    while not state.game_over and state.move_count - start_move_count < int(max_moves):
        before = state
        action = int(first_action) if forced else int(policy(before, sim, policy_rng))
        forced = False
        stepped = _step_with_fallback(sim, before, action)
        if stepped is None:
            break
        state, info, actual_action = stepped
        frames.append(
            {
                "index": len(frames),
                "state": state_payload(state, sim),
                "move": {
                    "action": DIRECTION_NAMES[actual_action],
                    "requested_action": DIRECTION_NAMES[action],
                    "preview_used": preview_payload(before),
                    "inserted_value": info.inserted_value,
                    "inserted_pos": list(info.inserted_pos) if info.inserted_pos is not None else None,
                    "eligible_positions": [list(pos) for pos in info.eligible_positions],
                    "merge_score_delta": int(info.merge_score_delta),
                    "score_delta": int(info.score_delta),
                    "terminal_merge": bool(info.terminal_merge),
                    "score_before": int(score_board(before.board)),
                    "score_after": int(score_board(state.board)),
                    "max_tile_before": int(before.max_tile),
                    "max_tile_after": int(state.max_tile),
                },
            }
        )

    final_score = int(score_board(state.board))
    max_tile_excl_starter = max_tile_excluding_initial_starter(state.board, start_case.starter_tile)
    action_name = DIRECTION_NAMES[int(first_action)]
    replay = {
        "policy": policy_name,
        "label_policy": policy_name,
        "seed": int(sim_seed),
        "starter_tile": start_case.starter_tile,
        "source_replay": start_case.source_replay,
        "source_seed": start_case.source_seed,
        "source_frame_index": int(start_case.frame_index),
        "first_action": action_name,
        "repeat_index": int(repeat_index),
        "start_score": int(start_score),
        "start_max_tile_excl_starter": int(start_case.start_max_tile_excl_starter),
        "start_phase": start_case.phase,
        "max_moves": int(max_moves),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "final_score": int(final_score),
        "final_score_delta": int(final_score - start_score),
        "final_moves": int(state.move_count),
        "final_moves_delta": int(state.move_count - start_move_count),
        "final_max_tile": int(state.max_tile),
        "final_max_tile_excl_starter": int(max_tile_excl_starter),
        "game_over": bool(state.game_over),
        "frames": frames,
    }
    return ActionContinuation(
        start_id=start_case.id,
        action=int(first_action),
        action_name=action_name,
        repeat_index=int(repeat_index),
        seed=int(sim_seed),
        score=int(final_score),
        score_delta=int(final_score - start_score),
        moves_delta=int(state.move_count - start_move_count),
        max_tile=int(state.max_tile),
        max_tile_excl_starter=int(max_tile_excl_starter),
        advanced_to_6144=bool(max_tile_excl_starter >= 6144),
        replay=replay,
        start_case=start_case,
    )


def action_continuation_record_payload(record: ActionContinuation) -> dict[str, object]:
    return {
        "start_id": record.start_id,
        "action": int(record.action),
        "action_name": record.action_name,
        "repeat_index": int(record.repeat_index),
        "seed": int(record.seed),
        "score": int(record.score),
        "score_delta": int(record.score_delta),
        "moves_delta": int(record.moves_delta),
        "max_tile": int(record.max_tile),
        "max_tile_excl_starter": int(record.max_tile_excl_starter),
        "advanced_to_6144": bool(record.advanced_to_6144),
        "source_replay": record.start_case.source_replay,
        "source_seed": record.start_case.source_seed,
        "source_frame_index": int(record.start_case.frame_index),
        "start_score": int(record.start_case.start_score),
        "start_max_tile_excl_starter": int(record.start_case.start_max_tile_excl_starter),
        "start_phase": record.start_case.phase,
    }


def _progress_entry_for_action_record(
    *,
    key: str,
    policy_name: str,
    seed: int,
    max_moves: int,
    record: ActionContinuation,
) -> dict[str, Any]:
    return {
        "key": key,
        "policy": str(policy_name),
        "seed": int(seed),
        "repeat_index": int(record.repeat_index),
        "first_action": int(record.action),
        "max_moves": int(max_moves),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "record": action_continuation_record_payload(record),
        "replay": record.replay,
    }


def _action_record_from_progress_entry(entry: dict[str, Any], start_case: StartCase) -> ActionContinuation:
    record_payload = entry.get("record")
    replay = entry.get("replay")
    if not isinstance(record_payload, dict) or not isinstance(replay, dict):
        raise ValueError("invalid action-continuation progress entry")
    action_name = str(record_payload.get("action_name", DIRECTION_NAMES[int(record_payload["action"])]))
    return ActionContinuation(
        start_id=str(record_payload.get("start_id", start_case.id)),
        action=int(record_payload["action"]),
        action_name=action_name,
        repeat_index=int(record_payload["repeat_index"]),
        seed=int(record_payload["seed"]),
        score=int(record_payload["score"]),
        score_delta=int(record_payload["score_delta"]),
        moves_delta=int(record_payload["moves_delta"]),
        max_tile=int(record_payload["max_tile"]),
        max_tile_excl_starter=int(record_payload["max_tile_excl_starter"]),
        advanced_to_6144=bool(record_payload["advanced_to_6144"]),
        replay=dict(replay),
        start_case=start_case,
    )


def _winner_for_rows(rows: list[dict[str, Any]]) -> str:
    return str(
        max(
            rows,
            key=lambda row: (
                float(row["p6144"]),
                float(row["mean_delta"]),
                float(row["median_delta"]),
                str(row["action"]),
            ),
        )["action"]
    )


def _bootstrap_winner_fraction(
    by_action: dict[str, list[ActionContinuation]],
    winner: str,
    rng: np.random.Generator,
    resamples: int = 200,
) -> float:
    if winner not in by_action:
        return 0.0
    repeat_count = min(len(records) for records in by_action.values())
    if repeat_count <= 0:
        return 0.0
    kept = 0
    action_names = sorted(by_action)
    for _ in range(int(resamples)):
        indices = rng.integers(0, repeat_count, size=repeat_count)
        rows = []
        for action_name in action_names:
            records = by_action[action_name]
            sample = [records[int(idx)] for idx in indices]
            rows.append(
                {
                    "action": action_name,
                    "p6144": sum(record.advanced_to_6144 for record in sample) / float(len(sample)),
                    "mean_delta": float(mean(record.score_delta for record in sample)),
                    "median_delta": float(median(record.score_delta for record in sample)),
                }
            )
        if _winner_for_rows(rows) == winner:
            kept += 1
    return kept / float(resamples)


def summarize_start_label(
    *,
    start_case: StartCase,
    records: list[ActionContinuation],
    base_action_name: str,
    stability_threshold: float,
) -> dict[str, Any]:
    by_action: dict[str, list[ActionContinuation]] = {}
    for record in records:
        by_action.setdefault(record.action_name, []).append(record)

    action_rows: list[dict[str, Any]] = []
    for action_name, action_records in sorted(by_action.items()):
        deltas = [record.score_delta for record in action_records]
        action_rows.append(
            {
                "action": action_name,
                "repeats": len(action_records),
                "mean_delta": float(mean(deltas)),
                "median_delta": float(median(deltas)),
                "max_delta": int(max(deltas)),
                "p6144": sum(record.advanced_to_6144 for record in action_records) / float(len(action_records)),
                "max_tile_excl_starter": int(max(record.max_tile_excl_starter for record in action_records)),
            }
        )
    winner = _winner_for_rows(action_rows)
    digest = hashlib.sha1(start_case.id.encode("utf-8")).hexdigest()
    bootstrap_fraction = _bootstrap_winner_fraction(
        by_action,
        winner,
        np.random.default_rng(int(digest[:8], 16)),
    )
    row_by_action = {str(row["action"]): row for row in action_rows}
    winner_delta = float(row_by_action[winner]["mean_delta"])
    base_delta = float(row_by_action.get(base_action_name, {"mean_delta": winner_delta})["mean_delta"])
    features = _features_for_start_case(start_case)
    return {
        "id": start_case.id,
        "source_replay": start_case.source_replay,
        "source_seed": start_case.source_seed,
        "source_frame_index": int(start_case.frame_index),
        "start_score": int(start_case.start_score),
        "start_max_tile_excl_starter": int(start_case.start_max_tile_excl_starter),
        "phase": start_case.phase,
        "features": features,
        "base_action": base_action_name,
        "winner": winner,
        "stable": bool(bootstrap_fraction >= float(stability_threshold)),
        "bootstrap_winner_fraction": float(bootstrap_fraction),
        "winner_mean_delta": winner_delta,
        "base_mean_delta": base_delta,
        "oracle_regret": max(0.0, winner_delta - base_delta),
        "winner_p6144": float(row_by_action[winner]["p6144"]),
        "action_results": action_rows,
    }


def write_top_replays(out_dir: Path, records: list[ActionContinuation], keep: int) -> dict[str, list[dict[str, object]]]:
    def write_group(name: str, chosen: list[ActionContinuation]) -> list[dict[str, object]]:
        top_dir = out_dir / name
        if top_dir.exists():
            shutil.rmtree(top_dir)
        top_dir.mkdir(parents=True, exist_ok=True)
        manifest: list[dict[str, object]] = []
        for rank, record in enumerate(chosen, start=1):
            game_dir = top_dir / (
                f"rank_{rank:02d}_delta_{record.score_delta}_score_{record.score}_"
                f"action_{safe_name(record.action_name, max_length=12)}_seed_{record.seed}"
            )
            game_dir.mkdir(parents=True, exist_ok=True)
            json_path = game_dir / "replay.json"
            html_path = game_dir / "replay.html"
            write_json(json_path, record.replay)
            write_html(html_path, record.replay)
            manifest.append(
                {
                    "rank": rank,
                    "seed": int(record.seed),
                    "action": record.action_name,
                    "score": int(record.score),
                    "score_delta": int(record.score_delta),
                    "moves_delta": int(record.moves_delta),
                    "max_tile": int(record.max_tile),
                    "max_tile_excl_starter": int(record.max_tile_excl_starter),
                    "start_id": record.start_id,
                    "start_score": int(record.start_case.start_score),
                    "start_max_tile_excl_starter": int(record.start_case.start_max_tile_excl_starter),
                    "source_replay": record.start_case.source_replay,
                    "source_frame_index": int(record.start_case.frame_index),
                    "html": str(html_path),
                    "json": str(json_path),
                }
            )
        write_json(top_dir / "manifest.json", manifest)
        return manifest

    keep_count = max(0, int(keep))
    by_delta = sorted(records, key=lambda record: (record.score_delta, record.score, record.moves_delta), reverse=True)[:keep_count]
    by_score = sorted(records, key=lambda record: (record.score, record.score_delta, record.moves_delta), reverse=True)[:keep_count]
    return {
        "top_delta_games": write_group("top_delta_games", by_delta),
        "top_games": write_group("top_games", by_score),
    }


def summarize_labels(labels: list[dict[str, Any]], records: list[ActionContinuation]) -> dict[str, Any]:
    stable = [label for label in labels if label.get("stable")]
    stable_disagreements = [
        label
        for label in stable
        if label.get("winner") != label.get("base_action")
    ]
    regrets = [float(label.get("oracle_regret", 0.0)) for label in labels]
    positive_regrets = [regret for regret in regrets if regret > 0.0]
    return {
        "labels": len(labels),
        "continuations": len(records),
        "stable_labels": len(stable),
        "stable_label_rate": float(len(stable) / len(labels)) if labels else 0.0,
        "stable_oracle_disagreements": len(stable_disagreements),
        "stable_oracle_disagreement_rate": float(len(stable_disagreements) / len(stable)) if stable else 0.0,
        "oracle_positive_regrets": len(positive_regrets),
        "mean_positive_oracle_regret": float(mean(positive_regrets)) if positive_regrets else 0.0,
        "median_positive_oracle_regret": float(median(positive_regrets)) if positive_regrets else 0.0,
        "max_positive_oracle_regret": float(max(positive_regrets)) if positive_regrets else 0.0,
        "p_any_action_6144": sum(record.advanced_to_6144 for record in records) / float(len(records)) if records else 0.0,
        "winner_counts": dict(Counter(str(label.get("winner")) for label in labels)),
    }


def write_report_html(path: Path, payload: dict[str, Any]) -> None:
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
            f"<td>{float(label.get('bootstrap_winner_fraction', 0.0)):.0%}</td>"
            f"<td>{float(label.get('oracle_regret', 0.0)):.0f}</td>"
            f"<td>{float(label.get('winner_mean_delta', 0.0)):.0f}</td>"
            f"<td>{float(label.get('winner_p6144', 0.0)):.0%}</td>"
            "</tr>"
        )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Endgame Action Labels</title>
  <style>
    :root {{ color-scheme: dark; --bg: #101113; --panel: #191d21; --line: #364047; --ink: #f1f5f0; --muted: #a9b3ad; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--ink); font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ width: min(1120px, calc(100vw - 32px)); margin: 0 auto; padding: 24px 0 36px; }}
    h1, h2, p {{ margin: 0; }}
    h1 {{ font-size: 24px; }}
    h2 {{ font-size: 16px; margin-bottom: 8px; }}
    .muted {{ color: var(--muted); margin-top: 6px; }}
    .cards {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin: 16px 0; }}
    .card, .panel {{ border: 1px solid var(--line); background: var(--panel); border-radius: 8px; padding: 12px; }}
    .label {{ color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .08em; }}
    .value {{ margin-top: 4px; font-size: 23px; font-weight: 800; font-variant-numeric: tabular-nums; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; font-variant-numeric: tabular-nums; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 7px 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    pre {{ white-space: pre-wrap; overflow-wrap: anywhere; color: var(--muted); font-size: 12px; }}
  </style>
</head>
<body>
  <main>
    <h1>Endgame Action Labels</h1>
    <p class="muted">All legal first actions from sampled 3072 replay states, then rollout with the configured policy.</p>
    <section class="cards">
      <div class="card"><div class="label">Labels</div><div class="value">{cell(summary.get('labels', 0))}</div></div>
      <div class="card"><div class="label">Stable Rate</div><div class="value">{float(summary.get('stable_label_rate', 0.0)):.0%}</div></div>
      <div class="card"><div class="label">Stable Flips</div><div class="value">{cell(summary.get('stable_oracle_disagreements', 0))}</div></div>
      <div class="card"><div class="label">Any 6144</div><div class="value">{float(summary.get('p_any_action_6144', 0.0)):.0%}</div></div>
    </section>
    <section class="panel">
      <h2>Labels</h2>
      <table><thead><tr><th>Seed</th><th>Frame</th><th>Base</th><th>Winner</th><th>Bootstrap</th><th>Regret</th><th>Winner Mean</th><th>Winner 6144</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
    </section>
    <section class="panel" style="margin-top: 14px;">
      <h2>Summary JSON</h2>
      <pre>{escape(json.dumps(summary, indent=2, sort_keys=True))}</pre>
    </section>
  </main>
</body>
</html>
"""
    path.write_text(html)


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    replay_paths = _flatten_paths(args.replay_json)
    phase_filter = set(parse_phase_filter(args.phase_filter)) if args.phase_filter else None
    corner_risk_filter = parse_corner_risk_filter(getattr(args, "corner_risk_filter", None))
    starter_text = args.starter.strip().lower()
    default_starter = None if starter_text == "none" else int(starter_text)
    start_cases = collect_start_cases(
        replay_paths,
        min_tile=args.start_state_min_tile,
        phase_filter=phase_filter,
        default_starter_tile=default_starter,
    )
    if corner_risk_filter is not None:
        start_cases = [
            case
            for case in start_cases
            if str(_features_for_start_case(case).get("corner_risk")) in corner_risk_filter
        ]
    selected_cases = select_start_cases(start_cases, max_starts=args.max_starts, seed=args.seed)
    if not selected_cases:
        raise ValueError("No start cases matched the requested filters")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_continuations = bool(getattr(args, "checkpoint_continuations", False))
    progress_path = getattr(args, "continuation_progress_json", None)
    if progress_path is None and checkpoint_continuations:
        progress_path = args.out_dir / "action_continuation_progress.json"
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
    all_records: list[ActionContinuation] = []
    labels: list[dict[str, Any]] = []
    total_work = sum(len(ThreesSim(np.random.default_rng(args.seed), starter_tile=case.starter_tile).legal_actions(case.state)) for case in selected_cases)
    total_work *= int(args.repeats)
    completed = 0
    ran_records = 0
    resumed_records = 0
    for case_idx, start_case in enumerate(selected_cases):
        sim_for_legal = ThreesSim(np.random.default_rng(int(args.seed) + case_idx), starter_tile=start_case.starter_tile)
        legal_actions = sim_for_legal.legal_actions(start_case.state)
        if not legal_actions:
            continue
        base_action = int(policy(start_case.state, sim_for_legal, np.random.default_rng(int(args.seed) + case_idx + 37)))
        base_action_name = DIRECTION_NAMES[base_action]
        case_records: list[ActionContinuation] = []
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
                if progress_entries is not None:
                    entry = progress_entries.get(progress_key)
                    if isinstance(entry, dict):
                        record = _action_record_from_progress_entry(entry, start_case)
                        all_records.append(record)
                        case_records.append(record)
                        completed += 1
                        resumed_records += 1
                        if args.progress_every > 0 and completed % int(args.progress_every) == 0:
                            print(
                                "progress "
                                f"{completed}/{total_work} "
                                f"best_delta={max(item.score_delta for item in all_records)} "
                                f"any_6144={sum(item.advanced_to_6144 for item in all_records)} "
                                "resumed=1",
                                flush=True,
                            )
                        continue

                record = run_first_action_continuation(
                    policy=policy,
                    policy_name=args.policy,
                    start_case=start_case,
                    first_action=int(action),
                    repeat_index=repeat_idx,
                    seed=repeat_seed,
                    max_moves=args.max_moves,
                )
                all_records.append(record)
                case_records.append(record)
                ran_records += 1
                completed += 1
                if progress_entries is not None and progress_payload is not None and progress_path is not None:
                    progress_entries[progress_key] = _progress_entry_for_action_record(
                        key=progress_key,
                        policy_name=args.policy,
                        seed=repeat_seed,
                        max_moves=args.max_moves,
                        record=record,
                    )
                    _write_action_progress(progress_path, progress_payload)
                if args.progress_every > 0 and completed % int(args.progress_every) == 0:
                    print(
                        "progress "
                        f"{completed}/{total_work} "
                        f"best_delta={max(item.score_delta for item in all_records)} "
                        f"any_6144={sum(item.advanced_to_6144 for item in all_records)}",
                        flush=True,
                    )
        labels.append(
            summarize_start_label(
                start_case=start_case,
                records=case_records,
                base_action_name=base_action_name,
                stability_threshold=args.stability_threshold,
            )
        )

    summary = summarize_labels(labels, all_records)
    summary["continuations_ran"] = int(ran_records)
    summary["continuations_resumed"] = int(resumed_records)
    summary["checkpoint_continuations"] = progress_path is not None
    if progress_path is not None:
        summary["continuation_progress_json"] = str(progress_path)
    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "policy": args.policy,
        "source_replays": [str(path) for path in replay_paths],
        "start_cases_total": len(start_cases),
        "selected_start_cases": len(selected_cases),
        "start_state_min_tile": int(args.start_state_min_tile),
        "phase_filter": sorted(phase_filter) if phase_filter is not None else None,
        "corner_risk_filter": sorted(corner_risk_filter) if corner_risk_filter is not None else None,
        "repeats": int(args.repeats),
        "max_moves": int(args.max_moves),
        "stability_threshold": float(args.stability_threshold),
        "labels": labels,
        "summary": summary,
    }
    payload.update(write_top_replays(args.out_dir, all_records, args.keep_top_games))
    write_json(args.out_dir / "endgame_action_labels.json", payload)
    write_report_html(args.out_dir / "endgame_action_labels.html", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--replay-json", type=Path, nargs="+", action="append", required=True)
    parser.add_argument("--starter", default="1536")
    parser.add_argument("--start-state-min-tile", type=int, default=3072)
    parser.add_argument("--phase-filter", help="Comma-separated phase names/aliases for start states.")
    parser.add_argument("--corner-risk-filter", help="Comma-separated corner-risk buckets; aliases include low, medium, high.")
    parser.add_argument("--max-starts", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--max-moves", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--stability-threshold", type=float, default=0.70)
    parser.add_argument("--keep-top-games", type=int, default=3)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument(
        "--checkpoint-continuations",
        action="store_true",
        help="Checkpoint each completed action continuation and reuse matching completed entries on rerun.",
    )
    parser.add_argument(
        "--continuation-progress-json",
        type=Path,
        help="Explicit resumable action-continuation progress JSON path; implies checkpoint/resume behavior.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("threes_rl/runs/forensics/endgame_action_labels/latest"))
    args = parser.parse_args()
    payload = run_from_args(args)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"json={args.out_dir / 'endgame_action_labels.json'}")
    print(f"html={args.out_dir / 'endgame_action_labels.html'}")


if __name__ == "__main__":
    main()
