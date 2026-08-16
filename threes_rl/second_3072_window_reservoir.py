"""Extract windows before creating the second built 3072 tile."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import time
from collections import Counter
from html import escape
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

import numpy as np

from threes_rl.compare_replays import state_from_payload
from threes_rl.geometry_forensics import geometry_features
from threes_rl.replay_provenance import replay_provenance
from threes_rl.run_artifacts import safe_name, write_json
from threes_rl.sim import ThreesSim
from threes_rl.swing_label import state_features


def _flatten_paths(path_groups: list[list[Path]] | None) -> list[Path]:
    if not path_groups:
        return []
    return [path for group in path_groups for path in group]


def _glob_paths(patterns: list[str] | None) -> list[Path]:
    if not patterns:
        return []
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(Path(match) for match in glob.glob(pattern, recursive=True))
    return sorted(path for path in paths if path.is_file())


def _int_or_none(value: object) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _starter_from_replay(replay: dict[str, Any], default: int | None) -> int | None:
    value = replay.get("starter_tile", default)
    return None if value is None else int(value)


def _move_action(frame: object) -> str | None:
    if not isinstance(frame, dict):
        return None
    move = frame.get("move")
    if not isinstance(move, dict):
        return None
    action = move.get("action")
    return str(action) if action is not None else None


def _record_id(
    *,
    source_replay: Path,
    seed: int | None,
    frame_index: int,
    outcome: str,
) -> str:
    raw = json.dumps(
        {
            "source_replay": str(source_replay),
            "seed": seed,
            "frame_index": frame_index,
            "target_event": "second_3072",
            "outcome": outcome,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    return safe_name(
        f"{source_replay.stem}_seed{seed}_second3072_{outcome}_frame{int(frame_index)}_{digest}",
        max_length=128,
    )


def _frame_rows(
    replay_path: Path,
    replay: dict[str, Any],
    *,
    default_starter_tile: int | None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    frames = replay.get("frames", [])
    if not isinstance(frames, list):
        return [], {"bad_replay": 1}
    starter_tile = _starter_from_replay(replay, default_starter_tile)
    seed = _int_or_none(replay.get("seed"))
    sim = ThreesSim(np.random.default_rng(seed or 0), starter_tile=starter_tile)
    rows: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    for frame_pos, frame in enumerate(frames):
        if not isinstance(frame, dict):
            continue
        payload = frame.get("state")
        if not isinstance(payload, dict):
            rejected["missing_state"] += 1
            continue
        try:
            state = state_from_payload(payload)
        except (TypeError, ValueError):
            rejected["bad_state"] += 1
            continue
        board = np.asarray(state.board, dtype=np.int32)
        features = state_features(state, sim, starter_tile)
        geo = geometry_features(board, starter_tile)
        rows.append(
            {
                "frame_position": int(frame_pos),
                "frame_index": int(frame.get("index", frame_pos)),
                "state": state,
                "state_payload": payload,
                "features": features,
                "geometry": geo,
                "starter_tile": starter_tile,
                "source_seed": seed,
                "source_next_action": _move_action(frames[frame_pos + 1]) if frame_pos + 1 < len(frames) else None,
                "game_over": bool(state.game_over),
            }
        )
    return rows, dict(rejected)


def _second_3072_event(rows: list[dict[str, Any]]) -> tuple[int, str] | None:
    for idx, row in enumerate(rows):
        geo = row["geometry"]
        if int(geo["count_3072"]) >= 2:
            return idx, "visible_two_3072"
    for idx, row in enumerate(rows):
        features = row["features"]
        if int(features["max_tile_excl_starter"]) >= 6144:
            return idx, "direct_6144"
    return None


def _first_3072_position(rows: list[dict[str, Any]]) -> int | None:
    for idx, row in enumerate(rows):
        features = row["features"]
        geo = row["geometry"]
        if int(features["max_tile_excl_starter"]) >= 3072 or int(geo["count_3072"]) >= 1:
            return idx
    return None


def _record_from_row(
    row: dict[str, Any],
    *,
    replay_path: Path,
    replay: dict[str, Any],
    outcome: str,
    event_kind: str | None,
    window_start_position: int,
    window_end_position: int,
    event_position: int | None,
    terminal_position: int | None,
) -> dict[str, Any]:
    state = row["state"]
    features = dict(row["features"])
    geometry = dict(row["geometry"])
    frame_position = int(row["frame_position"])
    frame_index = int(row["frame_index"])
    seed = _int_or_none(replay.get("seed"))
    provenance = replay_provenance(replay, replay_path)
    starter_tile = row.get("starter_tile", _starter_from_replay(replay, 1536))
    if starter_tile is not None:
        starter_tile = int(starter_tile)
    legal_count = int(len(ThreesSim(np.random.default_rng(seed or 0), starter_tile=starter_tile).legal_actions(state)))
    moves_to_event = None if event_position is None else int(event_position - frame_position)
    moves_to_terminal = None if terminal_position is None else int(terminal_position - frame_position)
    merged_features = {
        **features,
        **{f"geometry_{key}": value for key, value in geometry.items()},
        "legal_count": legal_count,
        "target_event": "second_3072",
        "event_kind": event_kind,
    }
    return {
        "id": _record_id(
            source_replay=replay_path,
            seed=seed,
            frame_index=frame_index,
            outcome=outcome,
        ),
        "kind": "second_3072_window_state",
        "target_event": "second_3072",
        "target_tile": 6144,
        "outcome": outcome,
        "event_kind": event_kind,
        "moves_to_event": moves_to_event,
        "moves_to_promotion": moves_to_event,
        "moves_to_terminal": moves_to_terminal,
        "window_start_position": int(window_start_position),
        "window_end_position": int(window_end_position),
        "event_frame_position": None if event_position is None else int(event_position),
        "terminal_frame_position": None if terminal_position is None else int(terminal_position),
        "source_replay": str(replay_path),
        "source_origin": provenance["replay_origin"],
        "source_policy": replay.get("policy"),
        "source_policy_family": provenance["source_policy_family"],
        "source_seed": seed,
        "seed": seed,
        "root_origin": provenance["root_origin"],
        "root_replay": provenance["root_replay"],
        "root_seed": provenance["root_seed"],
        "root_frame_index": provenance["root_frame_index"],
        "root_move_count": provenance["root_move_count"],
        "root_score": provenance["root_score"],
        "root_policy": provenance["root_policy"],
        "root_policy_family": provenance["root_policy_family"],
        "root_is_genuine": provenance["root_is_genuine"],
        "ancestry_key": provenance["ancestry_key"],
        "starter_tile": starter_tile,
        "source_frame_index": frame_index,
        "frame_position": frame_position,
        "source_next_action": row.get("source_next_action"),
        "move_count": int(state.move_count),
        "score": int(row["state_payload"].get("score", 0)),
        "score_minus_starter": int(features["score_minus_starter"]),
        "max_tile": int(features["max_tile"]),
        "max_tile_excl_starter": int(features["max_tile_excl_starter"]),
        "phase": str(features["phase"]),
        "corner_risk": str(features["corner_risk"]),
        "stratum": str(features["stratum"]),
        "empty_count": int(features["empty_count"]),
        "legal_count": legal_count,
        "preview": str(features["preview"]),
        "large_pending": bool(features["large_pending"]),
        "count_1536": int(geometry["count_1536"]),
        "count_3072": int(geometry["count_3072"]),
        "count_6144": int(geometry["count_6144"]),
        "highest_duplicate_tile": int(geometry["highest_duplicate_tile"]),
        "highest_adjacent_pair_tile": int(geometry["highest_adjacent_pair_tile"]),
        "adjacent_same_max": bool(geometry["adjacent_same_max"]),
        "adjacent_half_max": bool(geometry["adjacent_half_max"]),
        "features": merged_features,
        "state": row["state_payload"],
    }


def _success_records(
    rows: list[dict[str, Any]],
    *,
    replay_path: Path,
    replay: dict[str, Any],
    window_size: int,
) -> list[dict[str, Any]]:
    event = _second_3072_event(rows)
    if event is None:
        return []
    event_position, event_kind = event
    start = max(0, int(event_position) - int(window_size))
    end = int(event_position) - 1
    if end < start:
        return []
    records = []
    for row in rows[start : end + 1]:
        if row.get("source_next_action") is None or bool(row.get("game_over")):
            continue
        records.append(
            _record_from_row(
                row,
                replay_path=replay_path,
                replay=replay,
                outcome="success",
                event_kind=event_kind,
                window_start_position=start,
                window_end_position=end,
                event_position=event_position,
                terminal_position=None,
            )
        )
    return records


def _failure_records(
    rows: list[dict[str, Any]],
    *,
    replay_path: Path,
    replay: dict[str, Any],
    window_size: int,
) -> list[dict[str, Any]]:
    if _second_3072_event(rows) is not None:
        return []
    first_3072 = _first_3072_position(rows)
    if first_3072 is None:
        return []
    candidates = [
        row
        for row in rows[int(first_3072) :]
        if int(row["features"]["max_tile_excl_starter"]) >= 3072
        and int(row["features"]["max_tile_excl_starter"]) < 6144
        and row.get("source_next_action") is not None
        and not bool(row.get("game_over"))
    ]
    if not candidates:
        return []
    window_rows = candidates[-int(window_size) :]
    start = int(window_rows[0]["frame_position"])
    end = int(window_rows[-1]["frame_position"])
    terminal_position = int(rows[-1]["frame_position"]) if rows else None
    return [
        _record_from_row(
            row,
            replay_path=replay_path,
            replay=replay,
            outcome="failure",
            event_kind=None,
            window_start_position=start,
            window_end_position=end,
            event_position=None,
            terminal_position=terminal_position,
        )
        for row in window_rows
    ]


def collect_second_3072_records(
    replay_paths: Iterable[Path],
    *,
    window_size: int = 40,
    include_failures: bool = True,
    default_starter_tile: int | None = 1536,
    max_records: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    paths = [Path(path) for path in replay_paths]
    replays_scanned = 0
    for replay_path in paths:
        try:
            replay = json.loads(replay_path.read_text())
        except (OSError, json.JSONDecodeError):
            rejected["bad_replay"] += 1
            continue
        if not isinstance(replay, dict):
            rejected["bad_replay"] += 1
            continue
        replays_scanned += 1
        rows, row_rejected = _frame_rows(replay_path, replay, default_starter_tile=default_starter_tile)
        rejected.update(row_rejected)
        if not rows:
            rejected["no_frames"] += 1
            continue
        records.extend(_success_records(rows, replay_path=replay_path, replay=replay, window_size=window_size))
        if include_failures:
            records.extend(_failure_records(rows, replay_path=replay_path, replay=replay, window_size=window_size))
    if max_records > 0:
        records = records[: int(max_records)]
    summary = summarize_records(
        records,
        source_replay_paths=[str(path) for path in paths],
        replays_scanned=replays_scanned,
        window_size=window_size,
        include_failures=include_failures,
        max_records=max_records,
        rejected=dict(rejected),
    )
    return records, summary


def summarize_records(
    records: list[dict[str, Any]],
    *,
    source_replay_paths: list[str],
    replays_scanned: int,
    window_size: int,
    include_failures: bool,
    max_records: int,
    rejected: dict[str, int],
) -> dict[str, Any]:
    scores = [int(record["score_minus_starter"]) for record in records]
    moves_to_event = [int(record["moves_to_event"]) for record in records if record.get("moves_to_event") is not None]
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_replays": len(source_replay_paths),
        "replays_scanned": int(replays_scanned),
        "target_event": "second_3072",
        "target_tile": 6144,
        "window_size": int(window_size),
        "include_failures": bool(include_failures),
        "max_records": int(max_records),
        "records": len(records),
        "by_outcome": dict(Counter(str(record.get("outcome")) for record in records)),
        "by_event_kind": dict(Counter(str(record.get("event_kind")) for record in records if record.get("event_kind"))),
        "by_stratum": dict(Counter(str(record.get("stratum", "unknown")) for record in records)),
        "source_replays_with_records": len({str(record.get("source_replay")) for record in records}),
        "score_minus_starter": {
            "mean": float(mean(scores)) if scores else 0.0,
            "median": float(median(scores)) if scores else 0.0,
            "max": int(max(scores)) if scores else 0,
        },
        "moves_to_event": {
            "mean": float(mean(moves_to_event)) if moves_to_event else None,
            "median": float(median(moves_to_event)) if moves_to_event else None,
            "min": min(moves_to_event) if moves_to_event else None,
            "max": max(moves_to_event) if moves_to_event else None,
        },
        "max_highest_duplicate_tile": max((int(record.get("highest_duplicate_tile") or 0) for record in records), default=0),
        "max_highest_adjacent_pair_tile": max((int(record.get("highest_adjacent_pair_tile") or 0) for record in records), default=0),
        "source_replay_paths": source_replay_paths,
        "rejected": rejected,
    }


def write_html(path: Path, payload: dict[str, Any]) -> None:
    summary = payload.get("summary", {})
    records = payload.get("records", [])

    def cell(value: object) -> str:
        return escape(str(value))

    rows = []
    for record in records[:300] if isinstance(records, list) else []:
        if not isinstance(record, dict):
            continue
        rows.append(
            "<tr>"
            f"<td>{cell(record.get('outcome'))}</td>"
            f"<td>{cell(record.get('event_kind'))}</td>"
            f"<td>{cell(record.get('moves_to_event'))}</td>"
            f"<td>{cell(record.get('moves_to_terminal'))}</td>"
            f"<td>{cell(record.get('move_count'))}</td>"
            f"<td>{cell(record.get('stratum'))}</td>"
            f"<td>{cell(record.get('count_3072'))}</td>"
            f"<td>{cell(record.get('highest_duplicate_tile'))}</td>"
            f"<td>{cell(record.get('highest_adjacent_pair_tile'))}</td>"
            f"<td>{cell(record.get('source_next_action'))}</td>"
            f"<td>{cell(record.get('source_replay'))}</td>"
            "</tr>"
        )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Second 3072 Windows</title>
  <style>
    :root {{ color-scheme: dark; --bg: #101318; --panel: #171d24; --line: #34404d; --text: #edf2f7; --muted: #aab6c2; --gold: #f2c14e; }}
    body {{ margin: 0; background: var(--bg); color: var(--text); font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ width: min(1240px, calc(100vw - 32px)); margin: 0 auto; padding: 24px 0 40px; }}
    h1 {{ margin: 0 0 8px; font-size: 24px; }}
    .muted {{ color: var(--muted); }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 18px 0; }}
    .card {{ border: 1px solid var(--line); background: var(--panel); border-radius: 8px; padding: 12px; }}
    .label {{ color: var(--muted); font-size: 11px; text-transform: uppercase; }}
    .value {{ margin-top: 4px; color: var(--gold); font-size: 22px; font-weight: 800; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 7px 8px; text-align: right; vertical-align: top; }}
    th:nth-child(6), td:nth-child(6), th:nth-child(11), td:nth-child(11) {{ text-align: left; }}
    td:nth-child(11) {{ max-width: 360px; overflow-wrap: anywhere; color: var(--muted); }}
    pre {{ white-space: pre-wrap; overflow-wrap: anywhere; color: var(--muted); }}
  </style>
</head>
<body>
  <main>
    <h1>Second 3072 Windows</h1>
    <p class="muted">Windows before first visible two-3072 board, with matched one-3072 failure controls.</p>
    <section class="cards">
      <div class="card"><div class="label">Records</div><div class="value">{cell(summary.get('records', 0))}</div></div>
      <div class="card"><div class="label">Source Replays</div><div class="value">{cell(summary.get('source_replays_with_records', 0))}</div></div>
      <div class="card"><div class="label">Success</div><div class="value">{cell((summary.get('by_outcome') or {}).get('success', 0))}</div></div>
      <div class="card"><div class="label">Failure</div><div class="value">{cell((summary.get('by_outcome') or {}).get('failure', 0))}</div></div>
    </section>
    <table><thead><tr><th>Outcome</th><th>Event</th><th>To Event</th><th>To Terminal</th><th>Move</th><th>Stratum</th><th>3072 Count</th><th>Duplicate</th><th>Adjacent Pair</th><th>Next Action</th><th>Replay</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
    <h2>Summary JSON</h2>
    <pre>{escape(json.dumps(summary, indent=2, sort_keys=True))}</pre>
  </main>
</body>
</html>
"""
    path.write_text(html)


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    replay_paths = _flatten_paths(args.replay_json) + _glob_paths(args.replay_glob)
    starter_text = args.starter.strip().lower()
    default_starter = None if starter_text == "none" else int(starter_text)
    records, summary = collect_second_3072_records(
        replay_paths,
        window_size=args.window_size,
        include_failures=not args.no_failures,
        default_starter_tile=default_starter,
        max_records=args.max_records,
    )
    payload = {
        "version": 1,
        "kind": "second_3072_window_reservoir",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "summary": summary,
        "records": records,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload["json"] = str(args.out_dir / "second_3072_windows.json")
    payload["records_json"] = str(args.out_dir / "records.json")
    payload["html"] = str(args.out_dir / "second_3072_windows.html")
    write_json(args.out_dir / "second_3072_windows.json", payload)
    write_json(args.out_dir / "records.json", records)
    write_html(args.out_dir / "second_3072_windows.html", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-json", type=Path, nargs="+", action="append", default=[])
    parser.add_argument("--replay-glob", action="append", default=[])
    parser.add_argument("--window-size", type=int, default=40)
    parser.add_argument("--no-failures", action="store_true")
    parser.add_argument("--starter", default="1536")
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("threes_rl/runs/forensics/second_3072_windows/latest"))
    args = parser.parse_args()
    payload = run_from_args(args)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"json={payload['json']}")
    print(f"records={payload['records_json']}")
    print(f"html={payload['html']}")


if __name__ == "__main__":
    main()
