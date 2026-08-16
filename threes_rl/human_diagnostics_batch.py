"""Batch-discover and process observed human Threes sessions."""

from __future__ import annotations

import argparse
import glob
import json
import shlex
import time
from html import escape
from pathlib import Path
from typing import Any

import numpy as np

from threes_rl.eval import max_tile_excluding_initial_starter
from threes_rl.human_diagnostics_pipeline import run_pipeline
from threes_rl.run_artifacts import safe_name, write_json


DEFAULT_EVENTS_GLOB = "datasets/human_watch/*/events.jsonl"
DEFAULT_OUT_ROOT = Path("threes_rl/runs/human_diagnostics")
DEFAULT_POLICY_FILE = Path("threes_rl/current_incumbent_policy.txt")


def discover_events(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()
    for pattern in patterns:
        for text in glob.glob(pattern):
            path = Path(text)
            key = str(path.expanduser().resolve(strict=False))
            if key in seen or not path.is_file():
                continue
            seen.add(key)
            paths.append(path)
    return sorted(paths, key=lambda path: str(path))


def _session_id(events_path: Path) -> str:
    parent = events_path.parent.name.strip()
    if parent and parent not in {".", ".."}:
        return safe_name(parent)
    return safe_name(events_path.stem)


def _manifest_path(out_dir: Path) -> Path:
    return out_dir / "human_diagnostics_manifest.json"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _imported_game_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    imports = manifest.get("imports", [])
    if not isinstance(imports, list):
        return rows
    for imported in imports:
        if not isinstance(imported, dict):
            continue
        replays = imported.get("replays", [])
        if not isinstance(replays, list):
            continue
        for replay in replays:
            if isinstance(replay, dict):
                rows.append(replay)
    return rows


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _replay_milestone_summary(path: Path) -> dict[str, Any]:
    replay = _load_json(path)
    starter_tile = replay.get("starter_tile", 1536)
    if starter_tile is not None:
        starter_tile = _int_or_none(starter_tile)
    frames = replay.get("frames")
    if not isinstance(frames, list):
        frames = []

    highest_excl = 0
    first_moves: dict[str, int | None] = {"1536": None, "3072": None, "6144": None}
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        state = frame.get("state")
        if not isinstance(state, dict):
            continue
        board = state.get("board")
        if board is None:
            continue
        move_count = _int_or_none(state.get("move_count"))
        try:
            max_excl = max_tile_excluding_initial_starter(np.asarray(board, dtype=np.int32), starter_tile)
        except (TypeError, ValueError):
            continue
        highest_excl = max(highest_excl, int(max_excl))
        for threshold in (1536, 3072, 6144):
            key = str(threshold)
            if first_moves[key] is None and max_excl >= threshold:
                first_moves[key] = move_count

    return {
        "highest_max_tile_excl_starter": int(highest_excl),
        "reached_nonstarter_1536": bool(highest_excl >= 1536),
        "reached_3072": bool(highest_excl >= 3072),
        "reached_6144": bool(highest_excl >= 6144),
        "first_nonstarter_1536_move": first_moves["1536"],
        "first_3072_move": first_moves["3072"],
        "first_6144_move": first_moves["6144"],
    }


def _manifest_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    rows = _imported_game_rows(manifest)
    scores = [int(row["final_score"]) for row in rows if row.get("final_score") is not None]
    max_tiles = [int(row["final_max_tile"]) for row in rows if row.get("final_max_tile") is not None]
    milestone_rows: list[dict[str, Any]] = []
    missing_replays = 0
    for row in rows:
        replay_json = row.get("json")
        if not replay_json:
            missing_replays += 1
            continue
        replay_path = Path(str(replay_json))
        if not replay_path.exists():
            missing_replays += 1
            continue
        milestone_rows.append(_replay_milestone_summary(replay_path))
    max_excl_values = [
        int(row["highest_max_tile_excl_starter"])
        for row in milestone_rows
        if row.get("highest_max_tile_excl_starter") is not None
    ]
    return {
        "games_seen": int(manifest.get("games_seen") or 0),
        "games_imported": int(manifest.get("games_imported") or 0),
        "games_skipped": int(manifest.get("games_skipped") or 0),
        "high_score": max(scores) if scores else None,
        "highest_final_max_tile": max(max_tiles) if max_tiles else None,
        "highest_max_tile_excl_starter": max(max_excl_values) if max_excl_values else None,
        "games_reaching_nonstarter_1536": sum(1 for row in milestone_rows if row.get("reached_nonstarter_1536")),
        "games_reaching_3072": sum(1 for row in milestone_rows if row.get("reached_3072")),
        "games_reaching_6144": sum(1 for row in milestone_rows if row.get("reached_6144")),
        "replay_json_missing": int(missing_replays),
    }


def _pipeline_args(args: argparse.Namespace, *, events_path: Path, out_dir: Path) -> argparse.Namespace:
    policy_file = None
    if not bool(getattr(args, "no_policy", False)):
        configured = getattr(args, "policy_file", None)
        if configured is not None and Path(configured).exists():
            policy_file = Path(configured)
    return argparse.Namespace(
        events_jsonl=[[events_path]],
        out_dir=out_dir,
        starter=args.starter,
        min_valid_moves=args.min_valid_moves,
        no_replay_html=args.no_replay_html,
        policy=None,
        policy_file=policy_file,
        min_tile=args.min_tile,
        phase_filter=args.phase_filter,
        corner_risk_filter=args.corner_risk_filter,
        reservoir_first_per=args.reservoir_first_per,
        reservoir_max_records=args.reservoir_max_records,
        reservoir_max_per_stratum=args.reservoir_max_per_stratum,
        reservoir_sort_by=args.reservoir_sort_by,
        no_transition_windows=args.no_transition_windows,
        transition_targets=args.transition_targets,
        transition_window_size=args.transition_window_size,
        transition_max_records=args.transition_max_records,
        transition_no_failures=args.transition_no_failures,
        no_support_ladder=args.no_support_ladder,
        support_ladder_targets=args.support_ladder_targets,
        support_ladder_window_size=args.support_ladder_window_size,
        support_ladder_max_records=args.support_ladder_max_records,
        support_ladder_no_failures=args.support_ladder_no_failures,
        sample_mode=args.sample_mode,
        margin_threshold=args.margin_threshold,
        min_top_value=args.min_top_value,
        scan_max_samples=args.scan_max_samples,
        scan_max_per_stratum=args.scan_max_per_stratum,
        scan_first_per=args.scan_first_per,
        max_state_records=args.max_state_records,
        scan_phase_filter=args.scan_phase_filter,
        scan_corner_risk_filter=args.scan_corner_risk_filter,
        anchor_min_tile=args.anchor_min_tile,
        geometry_min_tile=args.geometry_min_tile,
        geometry_min_delta=args.geometry_min_delta,
        action_value_cache=None,
        agreement_min_tile=args.agreement_min_tile,
        agreement_phase_filter=args.agreement_phase_filter,
        agreement_corner_risk_filter=args.agreement_corner_risk_filter,
        agreement_max_records=args.agreement_max_records,
        agreement_high_confidence_margin=args.agreement_high_confidence_margin,
        no_agreement_report=args.no_agreement_report,
        progress_every_state_records=args.progress_every_state_records,
        label_seed=args.label_seed,
        stability_threshold=args.stability_threshold,
        label_workers=args.label_workers,
    )


def _next_command(args: argparse.Namespace) -> str:
    command = [
        ".venv/bin/python",
        "-m",
        "threes_rl.human_diagnostics_batch",
        "--run",
        "--out-root",
        str(args.out_root),
    ]
    for pattern in args.events_glob:
        command.extend(["--events-glob", pattern])
    if bool(getattr(args, "no_policy", False)):
        command.append("--no-policy")
    elif getattr(args, "policy_file", None) is not None:
        command.extend(["--policy-file", str(args.policy_file)])
    return shlex.join(command)


def write_report_html(path: Path, payload: dict[str, Any]) -> None:
    sessions = payload.get("sessions", [])
    if not isinstance(sessions, list):
        sessions = []
    rows = []
    for session in sessions:
        if not isinstance(session, dict):
            continue
        rows.append(
            "<tr>"
            f"<td>{escape(str(session.get('session_id', '')))}</td>"
            f"<td>{escape(str(session.get('status', '')))}</td>"
            f"<td>{escape(str(session.get('games_imported', 0)))}</td>"
            f"<td>{escape(str(session.get('high_score', '')))}</td>"
            f"<td>{escape(str(session.get('highest_final_max_tile', '')))}</td>"
            f"<td>{escape(str(session.get('highest_max_tile_excl_starter', '')))}</td>"
            f"<td>{escape(str(session.get('games_reaching_nonstarter_1536', 0)))}</td>"
            f"<td>{escape(str(session.get('games_reaching_3072', 0)))}</td>"
            f"<td><code>{escape(str(session.get('events_jsonl', '')))}</code></td>"
            "</tr>"
        )
    totals = payload.get("totals", {})
    if not isinstance(totals, dict):
        totals = {}
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Threes Human Data Inbox</title>
  <style>
    :root {{ color-scheme: dark; --bg: #101214; --panel: #171c20; --line: #344049; --ink: #edf3ee; --muted: #aab4ad; --gold: #e4bd4b; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--ink); font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ width: min(1180px, calc(100vw - 32px)); margin: 0 auto; padding: 24px 0 42px; }}
    h1 {{ margin: 0 0 6px; font-size: 24px; }}
    h2 {{ margin: 18px 0 8px; font-size: 17px; }}
    .muted {{ color: var(--muted); }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 10px; margin: 16px 0; }}
    .metric, .panel {{ border: 1px solid var(--line); background: var(--panel); border-radius: 8px; padding: 12px; }}
    .metric b {{ display: block; color: var(--gold); font-size: 22px; margin-top: 4px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 7px 8px; text-align: right; vertical-align: top; }}
    th:first-child, td:first-child, th:last-child, td:last-child {{ text-align: left; }}
    code {{ white-space: pre-wrap; overflow-wrap: anywhere; color: var(--gold); }}
  </style>
</head>
<body>
  <main>
    <h1>Human Data Inbox</h1>
    <p class="muted">Status: {escape(str(payload.get('status', '')))}. Target intake is at least five independent games reaching non-starter 1536 and one or more reaching 3072.</p>
    <section class="grid">
      <div class="metric"><span class="muted">Sessions</span><b>{escape(str(totals.get('sessions', 0)))}</b></div>
      <div class="metric"><span class="muted">Pending</span><b>{escape(str(totals.get('pending_sessions', 0)))}</b></div>
      <div class="metric"><span class="muted">Processed</span><b>{escape(str(totals.get('processed_sessions', 0)))}</b></div>
      <div class="metric"><span class="muted">Imported Games</span><b>{escape(str(totals.get('games_imported', 0)))}</b></div>
      <div class="metric"><span class="muted">Best Human Score</span><b>{escape(str(totals.get('high_score', '') or ''))}</b></div>
      <div class="metric"><span class="muted">Non-Starter 1536</span><b>{escape(str(totals.get('games_reaching_nonstarter_1536', 0)))}/5</b></div>
      <div class="metric"><span class="muted">Reached 3072</span><b>{escape(str(totals.get('games_reaching_3072', 0)))}/1</b></div>
    </section>
    <section class="panel">
      <h2>Sessions</h2>
      <table><thead><tr><th>Session</th><th>Status</th><th>Imported</th><th>High</th><th>Final Max</th><th>Max Excl Starter</th><th>Non-Starter 1536</th><th>3072</th><th>Events</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
    </section>
    <section class="panel" style="margin-top: 14px;">
      <h2>Next Command</h2>
      <code>{escape(str(payload.get('next_command', '')))}</code>
    </section>
  </main>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html)


def run_batch(args: argparse.Namespace) -> dict[str, Any]:
    events_paths = discover_events(args.events_glob)
    sessions: list[dict[str, Any]] = []
    processed = 0
    pending = 0
    current = 0
    for events_path in events_paths:
        session_id = _session_id(events_path)
        out_dir = Path(args.out_root) / session_id
        manifest_path = _manifest_path(out_dir)
        manifest = _load_json(manifest_path)
        is_current = manifest_path.exists() and manifest_path.stat().st_mtime >= events_path.stat().st_mtime
        status = "current" if is_current and not args.force else "pending"
        summary = _manifest_summary(manifest)
        if args.run and (args.force or not is_current):
            result = run_pipeline(_pipeline_args(args, events_path=events_path, out_dir=out_dir))
            manifest = result
            summary = _manifest_summary(result)
            status = "processed"
            processed += 1
        elif status == "current":
            current += 1
        else:
            pending += 1
        sessions.append(
            {
                "session_id": session_id,
                "events_jsonl": str(events_path),
                "events_mtime": events_path.stat().st_mtime,
                "out_dir": str(out_dir),
                "manifest_json": str(manifest_path),
                "manifest_exists": manifest_path.exists(),
                "status": status,
                **summary,
            }
        )
    high_scores = [int(session["high_score"]) for session in sessions if session.get("high_score") is not None]
    max_excl_values = [
        int(session["highest_max_tile_excl_starter"])
        for session in sessions
        if session.get("highest_max_tile_excl_starter") is not None
    ]
    games_reaching_nonstarter_1536 = sum(
        int(session.get("games_reaching_nonstarter_1536") or 0) for session in sessions
    )
    games_reaching_3072 = sum(int(session.get("games_reaching_3072") or 0) for session in sessions)
    games_reaching_6144 = sum(int(session.get("games_reaching_6144") or 0) for session in sessions)
    target_nonstarter_1536 = 5
    target_3072 = 1
    status = "waiting_for_human_data"
    if sessions:
        status = "processed" if processed else ("current" if current == len(sessions) else "ready_to_process")
    payload = {
        "version": 1,
        "kind": "threes_human_diagnostics_batch",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "mode": "run" if args.run else "dry_run",
        "status": status,
        "events_glob": list(args.events_glob),
        "out_root": str(args.out_root),
        "policy_file": str(args.policy_file) if getattr(args, "policy_file", None) is not None else None,
        "policy_enabled": not bool(getattr(args, "no_policy", False)),
        "sessions": sessions,
        "totals": {
            "sessions": len(sessions),
            "pending_sessions": sum(1 for session in sessions if session.get("status") == "pending"),
            "current_sessions": sum(1 for session in sessions if session.get("status") == "current"),
            "processed_sessions": sum(1 for session in sessions if session.get("status") == "processed"),
            "games_seen": sum(int(session.get("games_seen") or 0) for session in sessions),
            "games_imported": sum(int(session.get("games_imported") or 0) for session in sessions),
            "games_skipped": sum(int(session.get("games_skipped") or 0) for session in sessions),
            "high_score": max(high_scores) if high_scores else None,
            "highest_max_tile_excl_starter": max(max_excl_values) if max_excl_values else None,
            "games_reaching_nonstarter_1536": games_reaching_nonstarter_1536,
            "games_reaching_3072": games_reaching_3072,
            "games_reaching_6144": games_reaching_6144,
            "replay_json_missing": sum(int(session.get("replay_json_missing") or 0) for session in sessions),
        },
        "target_intake": {
            "independent_games_reaching_nonstarter_1536": target_nonstarter_1536,
            "independent_games_reaching_3072": target_3072,
            "current_games_reaching_nonstarter_1536": games_reaching_nonstarter_1536,
            "current_games_reaching_3072": games_reaching_3072,
            "remaining_games_reaching_nonstarter_1536": max(0, target_nonstarter_1536 - games_reaching_nonstarter_1536),
            "remaining_games_reaching_3072": max(0, target_3072 - games_reaching_3072),
            "ready_for_human_root_labeling": bool(
                games_reaching_nonstarter_1536 >= target_nonstarter_1536 and games_reaching_3072 >= target_3072
            ),
            "note": (
                "Counts are computed from replay frames using max tile excluding the fixed "
                "starter, so final_max_tile alone cannot satisfy the intake target."
            ),
        },
        "next_command": _next_command(args),
    }
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    json_path = out_root / "human_diagnostics_batch.json"
    html_path = out_root / "human_diagnostics_batch.html"
    payload["json"] = str(json_path)
    payload["html"] = str(html_path)
    write_json(json_path, payload)
    write_report_html(html_path, payload)
    write_json(json_path, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events-glob", action="append", default=[])
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--policy-file", type=Path, default=DEFAULT_POLICY_FILE)
    parser.add_argument("--no-policy", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--starter", default="1536")
    parser.add_argument("--min-valid-moves", type=int, default=1)
    parser.add_argument("--no-replay-html", action="store_true")
    parser.add_argument("--min-tile", type=int, default=1536)
    parser.add_argument("--phase-filter", action="append", default=["late,endgame"])
    parser.add_argument("--corner-risk-filter", action="append", default=["medium,high"])
    parser.add_argument("--reservoir-first-per", default="none")
    parser.add_argument("--reservoir-max-records", type=int, default=0)
    parser.add_argument("--reservoir-max-per-stratum", type=int, default=0)
    parser.add_argument("--reservoir-sort-by", default="max_tile")
    parser.add_argument("--no-transition-windows", action="store_true")
    parser.add_argument("--transition-targets", default="1536,3072,6144")
    parser.add_argument("--transition-window-size", type=int, default=40)
    parser.add_argument("--transition-max-records", type=int, default=0)
    parser.add_argument("--transition-no-failures", action="store_true")
    parser.add_argument("--no-support-ladder", action="store_true")
    parser.add_argument(
        "--support-ladder-targets",
        default=(
            "raw_duplicate_768,raw_adjacent_768,raw_one_1536,"
            "raw_duplicate_1536,raw_near_adjacent_1536,raw_adjacent_1536,second_3072"
        ),
    )
    parser.add_argument("--support-ladder-window-size", type=int, default=40)
    parser.add_argument("--support-ladder-max-records", type=int, default=0)
    parser.add_argument("--support-ladder-no-failures", action="store_true")
    parser.add_argument("--sample-mode", choices=["top-two", "anchor-risk", "geometry-risk"], default="top-two")
    parser.add_argument("--margin-threshold", type=float, default=0.002)
    parser.add_argument("--min-top-value", type=float, default=5000.0)
    parser.add_argument("--scan-max-samples", type=int, default=24)
    parser.add_argument("--scan-max-per-stratum", type=int, default=4)
    parser.add_argument("--scan-first-per", default="replay-stratum")
    parser.add_argument("--max-state-records", type=int, default=0)
    parser.add_argument("--scan-phase-filter", action="append", default=["late,endgame"])
    parser.add_argument("--scan-corner-risk-filter", action="append", default=["medium,high"])
    parser.add_argument("--anchor-min-tile", type=int, default=1536)
    parser.add_argument("--geometry-min-tile", type=int, default=1536)
    parser.add_argument("--geometry-min-delta", type=float, default=1.0)
    parser.add_argument("--agreement-min-tile", type=int)
    parser.add_argument("--agreement-phase-filter", action="append")
    parser.add_argument("--agreement-corner-risk-filter", action="append")
    parser.add_argument("--agreement-max-records", type=int, default=0)
    parser.add_argument("--agreement-high-confidence-margin", type=float, default=0.01)
    parser.add_argument("--no-agreement-report", action="store_true")
    parser.add_argument("--progress-every-state-records", type=int, default=0)
    parser.add_argument("--label-seed", type=int, default=20260706)
    parser.add_argument("--stability-threshold", type=float, default=0.70)
    parser.add_argument("--label-workers", type=int, default=1)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not args.events_glob:
        args.events_glob = [DEFAULT_EVENTS_GLOB]
    payload = run_batch(args)
    print(
        json.dumps(
            {
                "mode": payload["mode"],
                "status": payload["status"],
                "totals": payload["totals"],
                "target_intake": payload["target_intake"],
                "json": payload["json"],
                "html": payload["html"],
                "next_command": payload["next_command"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
