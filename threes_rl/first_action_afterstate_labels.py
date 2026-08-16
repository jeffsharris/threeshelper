"""Extract labeled first-action afterstates from continuation checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from html import escape
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from threes_rl.compare_replays import state_from_payload
from threes_rl.first_action_milestone_audit import analyze_replay_payload
from threes_rl.run_artifacts import safe_name, write_json
from threes_rl.sim import ThreesSim
from threes_rl.swing_label import state_features


MILESTONE_TARGETS = {
    "raw_duplicate_1536": ("milestones", "first_raw_duplicate_1536"),
    "raw_adjacent_1536": ("milestones", "first_raw_adjacent_pair_1536"),
    "second_3072": ("second_3072_event",),
    "reached_6144": ("first_6144",),
}


def _flatten_paths(path_groups: list[list[Path]] | None) -> list[Path]:
    if not path_groups:
        return []
    return [path for group in path_groups for path in group]


def _load_entries(paths: Iterable[Path]) -> list[tuple[str, dict[str, Any], str]]:
    loaded: list[tuple[str, dict[str, Any], str]] = []
    for path in paths:
        payload = json.loads(Path(path).read_text())
        entries = payload.get("entries") if isinstance(payload, dict) else None
        if not isinstance(entries, dict):
            raise ValueError(f"{path} does not contain entries{{}}")
        for key, entry in entries.items():
            if isinstance(entry, dict):
                loaded.append((str(key), entry, str(path)))
    return loaded


def _value_at_path(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = payload
    for part in path:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _target_success(analysis: dict[str, Any], target: str) -> bool:
    path = MILESTONE_TARGETS[target]
    return _value_at_path(analysis, path) is not None


def _digest(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _starter_from_replay(replay: dict[str, Any]) -> int | None:
    value = replay.get("starter_tile", 1536)
    return None if value is None else int(value)


def _label_record(
    *,
    entry_key: str,
    entry: dict[str, Any],
    source_json: str,
    target: str,
    after_frame_index: int,
) -> dict[str, Any] | None:
    replay = entry.get("replay")
    record = entry.get("record")
    if not isinstance(replay, dict) or not isinstance(record, dict):
        return None
    frames = replay.get("frames")
    if not isinstance(frames, list) or len(frames) <= int(after_frame_index):
        return None
    frame = frames[int(after_frame_index)]
    if not isinstance(frame, dict) or not isinstance(frame.get("state"), dict):
        return None
    move = frame.get("move") if isinstance(frame.get("move"), dict) else {}
    state_payload = frame["state"]
    state = state_from_payload(state_payload)
    starter_tile = _starter_from_replay(replay)
    sim = ThreesSim(np.random.default_rng(0), starter_tile=starter_tile)
    features = state_features(state, sim, starter_tile)
    analysis = analyze_replay_payload(replay)
    success = _target_success(analysis, target)
    first_action = str(record.get("action_name", replay.get("first_action", move.get("requested_action", "unknown"))))
    source_replay = str(record.get("source_replay", replay.get("source_replay", source_json)))
    source_seed = record.get("source_seed", replay.get("source_seed"))
    source_frame_index = record.get("source_frame_index", replay.get("source_frame_index"))
    start_id = str(record.get("start_id", replay.get("start_case_id", "unknown_start")))
    label_id = safe_name(
        "first_action_afterstate_"
        f"{target}_{'success' if success else 'failure'}_"
        f"{start_id}_{first_action}_r{record.get('repeat_index', entry.get('repeat_index', 0))}_"
        f"{_digest({'key': entry_key, 'source': source_json, 'target': target, 'after': after_frame_index})}",
        max_length=180,
    )
    return {
        "id": label_id,
        "kind": "first_action_afterstate_label",
        "outcome": "success" if success else "failure",
        "target_milestone": target,
        "target_tile": 6144,
        "state": state_payload,
        "features": features,
        "score_minus_starter": features.get("score_minus_starter"),
        "empty_count": features.get("empty_count"),
        "legal_count": len(state_payload.get("legal_actions", [])) if isinstance(state_payload, dict) else None,
        "preview": features.get("preview"),
        "large_pending": features.get("large_pending"),
        "corner_risk": features.get("corner_risk"),
        "stratum": features.get("stratum"),
        "source_replay": source_replay,
        "source_seed": source_seed,
        "source_frame_index": source_frame_index,
        "source_next_action": first_action,
        "first_action": first_action,
        "actual_action": move.get("action"),
        "requested_action": move.get("requested_action"),
        "repeat_index": record.get("repeat_index", entry.get("repeat_index")),
        "continuation_seed": record.get("seed", entry.get("seed")),
        "continuation_score_delta": record.get("score_delta"),
        "continuation_moves_delta": record.get("moves_delta"),
        "continuation_max_tile_excl_starter": record.get("max_tile_excl_starter"),
        "continuation_reached_6144": _target_success(analysis, "reached_6144"),
        "continuation_raw_duplicate_1536": _target_success(analysis, "raw_duplicate_1536"),
        "continuation_raw_adjacent_1536": _target_success(analysis, "raw_adjacent_1536"),
        "continuation_second_3072": _target_success(analysis, "second_3072"),
        "start_case_id": start_id,
        "start_score": record.get("start_score", replay.get("start_score")),
        "start_max_tile_excl_starter": record.get("start_max_tile_excl_starter", replay.get("start_max_tile_excl_starter")),
        "starter_tile": starter_tile,
        "after_frame_index": int(after_frame_index),
        "progress_key": entry_key,
        "progress_json": source_json,
    }


def summarize(records: list[dict[str, Any]], *, target: str, paths: list[Path]) -> dict[str, Any]:
    successes = [record for record in records if str(record.get("outcome")) == "success"]
    by_action = Counter(str(record.get("first_action")) for record in records)
    success_by_action = Counter(str(record.get("first_action")) for record in successes)
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "target_milestone": target,
        "progress_json": [str(path) for path in paths],
        "records": len(records),
        "successes": len(successes),
        "failures": len(records) - len(successes),
        "success_rate": float(len(successes) / len(records)) if records else 0.0,
        "source_starts": len({str(record.get("start_case_id")) for record in records}),
        "source_replays": len({str(record.get("source_replay")) for record in records}),
        "by_action": dict(by_action),
        "success_by_action": dict(success_by_action),
        "success_rate_by_action": {
            action: float(success_by_action[action] / count) if count else 0.0
            for action, count in sorted(by_action.items())
        },
    }


def write_html(path: Path, payload: dict[str, Any]) -> None:
    summary = payload.get("summary", {})
    records = payload.get("records", [])

    def cell(value: object) -> str:
        return escape(str(value))

    rows = []
    for record in records[:200] if isinstance(records, list) else []:
        rows.append(
            "<tr>"
            f"<td>{cell(record.get('outcome'))}</td>"
            f"<td>{cell(record.get('source_seed'))}</td>"
            f"<td>{cell(record.get('source_frame_index'))}</td>"
            f"<td>{cell(record.get('first_action'))}</td>"
            f"<td>{cell(record.get('preview'))}</td>"
            f"<td>{cell(record.get('empty_count'))}</td>"
            f"<td>{cell(record.get('continuation_score_delta'))}</td>"
            f"<td>{cell(record.get('start_case_id'))}</td>"
            "</tr>"
        )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>First-Action Afterstate Labels</title>
  <style>
    :root {{ color-scheme: dark; --bg: #101318; --panel: #171d24; --line: #34404d; --text: #edf2f7; --muted: #aab6c2; --gold: #f2c14e; }}
    body {{ margin: 0; background: var(--bg); color: var(--text); font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ width: min(1120px, calc(100vw - 32px)); margin: 0 auto; padding: 24px 0 40px; }}
    h1 {{ margin: 0 0 8px; font-size: 24px; }}
    .muted {{ color: var(--muted); }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 18px 0; }}
    .card, .panel {{ border: 1px solid var(--line); background: var(--panel); border-radius: 8px; padding: 12px; }}
    .label {{ color: var(--muted); font-size: 11px; text-transform: uppercase; }}
    .value {{ margin-top: 4px; color: var(--gold); font-size: 22px; font-weight: 800; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; font-variant-numeric: tabular-nums; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 7px 8px; text-align: right; }}
    th:first-child, td:first-child, th:last-child, td:last-child {{ text-align: left; }}
    pre {{ white-space: pre-wrap; overflow-wrap: anywhere; color: var(--muted); }}
  </style>
</head>
<body>
  <main>
    <h1>First-Action Afterstate Labels</h1>
    <p class="muted">Immediate post-forced-action states labeled by later support-chain milestone completion.</p>
    <section class="cards">
      <div class="card"><div class="label">Records</div><div class="value">{cell(summary.get('records', 0))}</div></div>
      <div class="card"><div class="label">Success Rate</div><div class="value">{float(summary.get('success_rate', 0.0)):.0%}</div></div>
      <div class="card"><div class="label">Starts</div><div class="value">{cell(summary.get('source_starts', 0))}</div></div>
      <div class="card"><div class="label">Replays</div><div class="value">{cell(summary.get('source_replays', 0))}</div></div>
    </section>
    <section class="panel"><pre>{escape(json.dumps(summary, indent=2, sort_keys=True))}</pre></section>
    <section class="panel" style="margin-top: 14px;">
      <table><thead><tr><th>Outcome</th><th>Seed</th><th>Frame</th><th>Action</th><th>Preview</th><th>Empty</th><th>Delta</th><th>Start</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
    </section>
  </main>
</body>
</html>
"""
    path.write_text(html)


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    progress_paths = _flatten_paths(args.progress_json)
    entries = _load_entries(progress_paths)
    records = []
    skipped = Counter()
    for key, entry, source_json in entries:
        record = _label_record(
            entry_key=key,
            entry=entry,
            source_json=source_json,
            target=args.target,
            after_frame_index=args.after_frame_index,
        )
        if record is None:
            skipped["invalid_entry"] += 1
            continue
        records.append(record)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(records, target=args.target, paths=progress_paths)
    summary["skipped"] = dict(skipped)
    payload = {
        "version": 1,
        "kind": "first_action_afterstate_labels",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "target_milestone": args.target,
        "after_frame_index": int(args.after_frame_index),
        "summary": summary,
        "records": records,
    }
    payload["json"] = str(args.out_dir / "first_action_afterstate_labels.json")
    payload["records_json"] = str(args.out_dir / "records.json")
    payload["html"] = str(args.out_dir / "first_action_afterstate_labels.html")
    write_json(args.out_dir / "first_action_afterstate_labels.json", payload)
    write_json(args.out_dir / "records.json", records)
    write_html(args.out_dir / "first_action_afterstate_labels.html", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress-json", type=Path, nargs="+", action="append", required=True)
    parser.add_argument(
        "--target",
        choices=sorted(MILESTONE_TARGETS),
        default="raw_adjacent_1536",
    )
    parser.add_argument("--after-frame-index", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = run_from_args(args)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"json={payload['json']}")
    print(f"records={payload['records_json']}")
    print(f"html={payload['html']}")


if __name__ == "__main__":
    main()
