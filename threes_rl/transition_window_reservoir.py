"""Extract pre-promotion transition windows from replay JSON files."""

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
from threes_rl.replay_provenance import replay_provenance
from threes_rl.run_artifacts import safe_name, write_json
from threes_rl.sim import ThreesSim
from threes_rl.support_ladder_window_reservoir import raw_ladder_features
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


def parse_targets(text: str | None) -> list[int]:
    raw = text or "1536,3072,6144"
    targets: list[int] = []
    seen: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        target = int(part)
        if target <= 0:
            raise ValueError(f"target tiles must be positive: {target}")
        if target not in seen:
            targets.append(target)
            seen.add(target)
    if not targets:
        raise ValueError("at least one target tile is required")
    return targets


def _int_or_none(value: object) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _starter_from_replay(replay: dict[str, Any], default: int | None) -> int | None:
    value = replay.get("starter_tile", default)
    if value is None:
        return None
    return int(value)


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
    target_tile: int,
    outcome: str,
) -> str:
    raw = json.dumps(
        {
            "source_replay": str(source_replay),
            "seed": seed,
            "frame_index": frame_index,
            "target_tile": int(target_tile),
            "outcome": outcome,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    return safe_name(
        f"{source_replay.stem}_seed{seed}_target{int(target_tile)}_{outcome}_frame{int(frame_index)}_{digest}",
        max_length=128,
    )


def _frame_rows(
    replay_path: Path,
    replay: dict[str, Any],
    *,
    default_starter_tile: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    frames = replay.get("frames", [])
    if not isinstance(frames, list):
        return [], {"bad_replay": 1}
    starter_tile = _starter_from_replay(replay, default_starter_tile)
    seed = _int_or_none(replay.get("seed"))
    sim_seed = seed if seed is not None else 0
    sim = ThreesSim(np.random.default_rng(int(sim_seed)), starter_tile=starter_tile)
    rows: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    for frame_pos, frame in enumerate(frames):
        if not isinstance(frame, dict):
            continue
        state_payload = frame.get("state")
        if not isinstance(state_payload, dict):
            rejected["missing_state"] += 1
            continue
        try:
            state = state_from_payload(state_payload)
        except (TypeError, ValueError):
            rejected["bad_state"] += 1
            continue
        features = state_features(state, sim, starter_tile)
        rows.append(
            {
                "frame_position": int(frame_pos),
                "frame_index": int(frame.get("index", frame_pos)),
                "state": state,
                "state_payload": state_payload,
                "features": features,
                "starter_tile": starter_tile,
                "source_seed": seed,
                "source_next_action": _move_action(frames[frame_pos + 1]) if frame_pos + 1 < len(frames) else None,
                "game_over": bool(state.game_over),
            }
        )
    return rows, dict(rejected)


def _promotion_index(rows: list[dict[str, Any]], target_tile: int) -> int | None:
    previous_max = 0
    for idx, row in enumerate(rows):
        built_max = int(row["features"]["max_tile_excl_starter"])
        if previous_max < int(target_tile) <= built_max:
            return idx
        previous_max = max(previous_max, built_max)
    return None


def _record_from_row(
    row: dict[str, Any],
    *,
    replay_path: Path,
    replay: dict[str, Any],
    target_tile: int,
    outcome: str,
    window_start_position: int,
    window_end_position: int,
    promotion_position: int | None,
    terminal_position: int | None,
    control_tile: int | None,
) -> dict[str, Any]:
    features = dict(row["features"])
    state = row["state"]
    frame_position = int(row["frame_position"])
    frame_index = int(row["frame_index"])
    seed = _int_or_none(replay.get("seed"))
    provenance = replay_provenance(replay, replay_path)
    starter_tile = row.get("starter_tile", _starter_from_replay(replay, 1536))
    if starter_tile is not None:
        starter_tile = int(starter_tile)
    raw = raw_ladder_features(state.board, starter_tile)
    legal_count = int(len(ThreesSim(np.random.default_rng(seed or 0), starter_tile=starter_tile).legal_actions(state)))
    moves_to_promotion = None if promotion_position is None else int(promotion_position - frame_position)
    moves_to_terminal = None if terminal_position is None else int(terminal_position - frame_position)
    return {
        "id": _record_id(
            source_replay=replay_path,
            seed=seed,
            frame_index=frame_index,
            target_tile=target_tile,
            outcome=outcome,
        ),
        "kind": "transition_window_state",
        "target_tile": int(target_tile),
        "outcome": outcome,
        "control_tile": None if control_tile is None else int(control_tile),
        "moves_to_promotion": moves_to_promotion,
        "moves_to_terminal": moves_to_terminal,
        "window_start_position": int(window_start_position),
        "window_end_position": int(window_end_position),
        "promotion_frame_position": None if promotion_position is None else int(promotion_position),
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
        "raw_count_768": int(raw["raw_count_768"]),
        "raw_count_1536": int(raw["raw_count_1536"]),
        "raw_count_3072": int(raw["raw_count_3072"]),
        "raw_highest_duplicate_tile": int(raw["raw_highest_duplicate_tile"]),
        "raw_highest_adjacent_pair_tile": int(raw["raw_highest_adjacent_pair_tile"]),
        "raw_has_adjacent_768": bool(raw["raw_has_adjacent_768"]),
        "raw_has_adjacent_1536": bool(raw["raw_has_adjacent_1536"]),
        "raw_has_near_adjacent_1536": bool(raw["raw_has_near_adjacent_1536"]),
        "features": {**features, **raw, "legal_count": legal_count},
        "state": row["state_payload"],
    }


def _success_records_for_target(
    rows: list[dict[str, Any]],
    *,
    replay_path: Path,
    replay: dict[str, Any],
    target_tile: int,
    window_size: int,
) -> list[dict[str, Any]]:
    promotion = _promotion_index(rows, target_tile)
    if promotion is None:
        return []
    start = max(0, int(promotion) - int(window_size))
    end = int(promotion) - 1
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
                target_tile=target_tile,
                outcome="success",
                window_start_position=start,
                window_end_position=end,
                promotion_position=promotion,
                terminal_position=None,
                control_tile=None,
            )
        )
    return records


def _failure_records_for_target(
    rows: list[dict[str, Any]],
    *,
    replay_path: Path,
    replay: dict[str, Any],
    target_tile: int,
    window_size: int,
) -> list[dict[str, Any]]:
    if _promotion_index(rows, target_tile) is not None:
        return []
    control_tile = max(1, int(target_tile) // 2)
    candidates = [
        row
        for row in rows
        if int(row["features"]["max_tile_excl_starter"]) >= control_tile
        and int(row["features"]["max_tile_excl_starter"]) < int(target_tile)
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
            target_tile=target_tile,
            outcome="failure",
            window_start_position=start,
            window_end_position=end,
            promotion_position=None,
            terminal_position=terminal_position,
            control_tile=control_tile,
        )
        for row in window_rows
    ]


def collect_transition_window_records(
    replay_paths: Iterable[Path],
    *,
    targets: Iterable[int],
    window_size: int = 40,
    include_failures: bool = True,
    default_starter_tile: int | None = 1536,
    max_records: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    source_paths = [Path(path) for path in replay_paths]
    target_list = [int(target) for target in targets]
    replays_scanned = 0
    for replay_path in source_paths:
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
        for target in target_list:
            records.extend(
                _success_records_for_target(
                    rows,
                    replay_path=replay_path,
                    replay=replay,
                    target_tile=target,
                    window_size=window_size,
                )
            )
            if include_failures:
                records.extend(
                    _failure_records_for_target(
                        rows,
                        replay_path=replay_path,
                        replay=replay,
                        target_tile=target,
                        window_size=window_size,
                    )
                )
    if max_records > 0:
        records = records[: int(max_records)]
    summary = summarize_records(
        records,
        source_replay_paths=[str(path) for path in source_paths],
        replays_scanned=replays_scanned,
        targets=target_list,
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
    targets: list[int],
    window_size: int,
    include_failures: bool,
    max_records: int,
    rejected: dict[str, int],
) -> dict[str, Any]:
    by_target_outcome: dict[str, dict[str, int]] = {}
    by_stratum: Counter[str] = Counter()
    scores = [int(record["score_minus_starter"]) for record in records]
    moves_to_promotion = [
        int(record["moves_to_promotion"])
        for record in records
        if record.get("moves_to_promotion") is not None
    ]
    for record in records:
        target_key = str(record.get("target_tile"))
        outcome = str(record.get("outcome"))
        bucket = by_target_outcome.setdefault(target_key, {"success": 0, "failure": 0})
        bucket[outcome] = int(bucket.get(outcome, 0)) + 1
        by_stratum[str(record.get("stratum", "unknown"))] += 1
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_replays": len(source_replay_paths),
        "replays_scanned": int(replays_scanned),
        "targets": [int(target) for target in targets],
        "window_size": int(window_size),
        "include_failures": bool(include_failures),
        "max_records": int(max_records),
        "records": len(records),
        "by_target_outcome": by_target_outcome,
        "by_stratum": dict(by_stratum),
        "score_minus_starter": {
            "mean": float(mean(scores)) if scores else 0.0,
            "median": float(median(scores)) if scores else 0.0,
            "max": int(max(scores)) if scores else 0,
        },
        "moves_to_promotion": {
            "mean": float(mean(moves_to_promotion)) if moves_to_promotion else None,
            "median": float(median(moves_to_promotion)) if moves_to_promotion else None,
            "min": min(moves_to_promotion) if moves_to_promotion else None,
            "max": max(moves_to_promotion) if moves_to_promotion else None,
        },
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
            f"<td>{cell(record.get('target_tile'))}</td>"
            f"<td>{cell(record.get('outcome'))}</td>"
            f"<td>{cell(record.get('moves_to_promotion'))}</td>"
            f"<td>{cell(record.get('moves_to_terminal'))}</td>"
            f"<td>{cell(record.get('move_count'))}</td>"
            f"<td>{cell(record.get('stratum'))}</td>"
            f"<td>{cell(record.get('empty_count'))}</td>"
            f"<td>{cell(record.get('legal_count'))}</td>"
            f"<td>{cell(record.get('source_next_action'))}</td>"
            f"<td>{cell(record.get('source_replay'))}</td>"
            "</tr>"
        )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Threes Transition Windows</title>
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
    th:nth-child(6), td:nth-child(6), th:nth-child(10), td:nth-child(10) {{ text-align: left; }}
    td:nth-child(10) {{ max-width: 360px; overflow-wrap: anywhere; color: var(--muted); }}
    pre {{ white-space: pre-wrap; overflow-wrap: anywhere; color: var(--muted); }}
  </style>
</head>
<body>
  <main>
    <h1>Transition Windows</h1>
    <p class="muted">Pre-promotion success windows plus matched one-step-below failure controls.</p>
    <section class="cards">
      <div class="card"><div class="label">Records</div><div class="value">{cell(summary.get('records', 0))}</div></div>
      <div class="card"><div class="label">Source Replays</div><div class="value">{cell(summary.get('source_replays', 0))}</div></div>
      <div class="card"><div class="label">Targets</div><div class="value">{cell(summary.get('targets', []))}</div></div>
      <div class="card"><div class="label">Window</div><div class="value">{cell(summary.get('window_size', 0))}</div></div>
    </section>
    <table><thead><tr><th>Target</th><th>Outcome</th><th>To Promo</th><th>To Terminal</th><th>Move</th><th>Stratum</th><th>Empty</th><th>Legal</th><th>Next Action</th><th>Replay</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
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
    records, summary = collect_transition_window_records(
        replay_paths,
        targets=parse_targets(args.targets),
        window_size=args.window_size,
        include_failures=not args.no_failures,
        default_starter_tile=default_starter,
        max_records=args.max_records,
    )
    payload = {
        "version": 1,
        "kind": "transition_window_reservoir",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "summary": summary,
        "records": records,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload["json"] = str(args.out_dir / "transition_windows.json")
    payload["records_json"] = str(args.out_dir / "records.json")
    payload["html"] = str(args.out_dir / "transition_windows.html")
    write_json(args.out_dir / "transition_windows.json", payload)
    write_json(args.out_dir / "records.json", records)
    write_html(args.out_dir / "transition_windows.html", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-json", type=Path, nargs="+", action="append", default=[])
    parser.add_argument("--replay-glob", action="append", default=[])
    parser.add_argument("--targets", default="1536,3072,6144")
    parser.add_argument("--window-size", type=int, default=40)
    parser.add_argument("--no-failures", action="store_true")
    parser.add_argument("--starter", default="1536")
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("threes_rl/runs/forensics/transition_windows/latest"))
    args = parser.parse_args()
    payload = run_from_args(args)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"json={payload['json']}")
    print(f"records={payload['records_json']}")
    print(f"html={payload['html']}")


if __name__ == "__main__":
    main()
