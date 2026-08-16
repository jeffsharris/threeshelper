"""Extract reusable high-board start states from replay JSON files."""

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
from threes_rl.swing_label import (
    CORNER_RISK_BUCKETS,
    PHASE_BUCKETS,
    parse_filter_values,
    state_features,
)


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
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _starter_from_replay(replay: dict[str, Any], default: int | None) -> int | None:
    value = replay.get("starter_tile", default)
    if value is None:
        return None
    return int(value)


def _next_action(frames: list[object], frame_pos: int) -> str | None:
    if frame_pos + 1 >= len(frames):
        return None
    next_frame = frames[frame_pos + 1]
    if not isinstance(next_frame, dict):
        return None
    move = next_frame.get("move")
    if not isinstance(move, dict):
        return None
    action = move.get("action")
    return str(action) if action is not None else None


def _record_id(source_replay: Path, seed: int | None, frame_index: int, features: dict[str, Any]) -> str:
    raw = json.dumps(
        {
            "source_replay": str(source_replay),
            "seed": seed,
            "frame_index": frame_index,
            "stratum": features.get("stratum"),
            "score_minus_starter": features.get("score_minus_starter"),
            "max_tile_excl_starter": features.get("max_tile_excl_starter"),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    return safe_name(
        f"{source_replay.stem}_seed{seed}_frame{frame_index}_{features.get('stratum')}_{digest}",
        max_length=120,
    )


def _scope_key(record: dict[str, Any], first_per: str) -> tuple[object, ...] | None:
    if first_per == "none":
        return None
    source = str(record.get("source_replay"))
    features = record.get("features")
    if not isinstance(features, dict):
        features = {}
    if first_per == "replay":
        return (source,)
    if first_per == "replay-phase":
        return (source, str(features.get("phase")))
    if first_per == "replay-stratum":
        return (source, str(features.get("stratum")))
    raise ValueError(f"Unsupported first_per mode: {first_per}")


def _sort_key(record: dict[str, Any], sort_by: str) -> tuple[object, ...]:
    if sort_by == "source":
        return (int(record.get("source_order", 0)), int(record.get("source_frame_index", 0)))
    if sort_by == "score":
        return (
            -int(record.get("score_minus_starter", 0)),
            -int(record.get("max_tile_excl_starter", 0)),
            int(record.get("source_order", 0)),
            int(record.get("source_frame_index", 0)),
        )
    if sort_by == "max_tile":
        return (
            -int(record.get("max_tile_excl_starter", 0)),
            -int(record.get("score_minus_starter", 0)),
            int(record.get("source_order", 0)),
            int(record.get("source_frame_index", 0)),
        )
    if sort_by == "move":
        return (-int(record.get("move_count", 0)), int(record.get("source_order", 0)))
    raise ValueError(f"Unsupported sort_by: {sort_by}")


def collect_reservoir_records(
    replay_paths: Iterable[Path],
    *,
    min_tile: int,
    phase_filter: set[str] | None = None,
    corner_risk_filter: set[str] | None = None,
    default_starter_tile: int | None = 1536,
    first_per: str = "none",
    max_records: int = 0,
    max_per_stratum: int = 0,
    sort_by: str = "source",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if first_per not in ("none", "replay", "replay-phase", "replay-stratum"):
        raise ValueError(f"Unsupported first_per mode: {first_per}")
    if sort_by not in ("source", "score", "max_tile", "move"):
        raise ValueError(f"Unsupported sort_by: {sort_by}")

    accepted: list[dict[str, Any]] = []
    seen_scopes: set[tuple[object, ...]] = set()
    stratum_counts: Counter[str] = Counter()
    rejected: Counter[str] = Counter()
    scanned_frames = 0
    candidate_frames = 0
    replay_count = 0
    resolved_paths = [Path(path) for path in replay_paths]

    for source_order, replay_path in enumerate(resolved_paths):
        try:
            replay = json.loads(replay_path.read_text())
        except (OSError, json.JSONDecodeError):
            rejected["bad_replay"] += 1
            continue
        if not isinstance(replay, dict):
            rejected["bad_replay"] += 1
            continue
        replay_count += 1
        starter_tile = _starter_from_replay(replay, default_starter_tile)
        seed = _int_or_none(replay.get("seed"))
        frames = replay.get("frames", [])
        if not isinstance(frames, list):
            rejected["bad_replay"] += 1
            continue
        provenance = replay_provenance(replay, replay_path)
        sim_seed = seed if seed is not None else source_order
        sim = ThreesSim(np.random.default_rng(int(sim_seed)), starter_tile=starter_tile)
        for frame_pos, frame in enumerate(frames):
            if not isinstance(frame, dict):
                continue
            state_payload = frame.get("state")
            if not isinstance(state_payload, dict):
                continue
            scanned_frames += 1
            state = state_from_payload(state_payload)
            if state.game_over:
                rejected["game_over"] += 1
                continue
            features = state_features(state, sim, starter_tile)
            if int(features["max_tile_excl_starter"]) < int(min_tile):
                rejected["below_min_tile"] += 1
                continue
            if phase_filter is not None and str(features.get("phase")) not in phase_filter:
                rejected["phase_filter"] += 1
                continue
            if corner_risk_filter is not None and str(features.get("corner_risk")) not in corner_risk_filter:
                rejected["corner_risk_filter"] += 1
                continue

            candidate_frames += 1
            frame_index = int(frame.get("index", frame_pos))
            record = {
                "id": _record_id(replay_path, seed, frame_index, features),
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
                "source_order": int(source_order),
                "source_frame_index": int(frame_index),
                "frame_position": int(frame_pos),
                "source_next_action": _next_action(frames, frame_pos),
                "starter_tile": starter_tile,
                "move_count": int(state.move_count),
                "score": int(state_payload.get("score", 0)),
                "score_minus_starter": int(features["score_minus_starter"]),
                "max_tile": int(features["max_tile"]),
                "max_tile_excl_starter": int(features["max_tile_excl_starter"]),
                "phase": str(features["phase"]),
                "corner_risk": str(features["corner_risk"]),
                "stratum": str(features["stratum"]),
                "features": features,
                "state": state_payload,
            }

            scope = _scope_key(record, first_per)
            if scope is not None and scope in seen_scopes:
                rejected["scope_seen"] += 1
                continue
            stratum = str(record["stratum"])
            if max_per_stratum > 0 and stratum_counts[stratum] >= int(max_per_stratum):
                rejected["bucket_full"] += 1
                continue
            accepted.append(record)
            stratum_counts[stratum] += 1
            if scope is not None:
                seen_scopes.add(scope)

    accepted.sort(key=lambda record: _sort_key(record, sort_by))
    records = accepted[: int(max_records)] if int(max_records) > 0 else accepted
    summary = summarize_records(
        records,
        source_replay_paths=[str(path) for path in resolved_paths],
        scanned_replays=replay_count,
        scanned_frames=scanned_frames,
        candidate_frames=candidate_frames,
        accepted_before_cap=len(accepted),
        rejected=dict(rejected),
    )
    summary.update(
        {
            "min_tile": int(min_tile),
            "phase_filter": sorted(phase_filter) if phase_filter is not None else None,
            "corner_risk_filter": sorted(corner_risk_filter) if corner_risk_filter is not None else None,
            "first_per": first_per,
            "max_records": int(max_records),
            "max_per_stratum": int(max_per_stratum),
            "sort_by": sort_by,
        }
    )
    return records, summary


def summarize_records(
    records: list[dict[str, Any]],
    *,
    source_replay_paths: list[str],
    scanned_replays: int,
    scanned_frames: int,
    candidate_frames: int,
    accepted_before_cap: int,
    rejected: dict[str, int],
) -> dict[str, Any]:
    scores = [int(record.get("score_minus_starter", 0)) for record in records]
    maxes = [int(record.get("max_tile_excl_starter", 0)) for record in records]
    moves = [int(record.get("move_count", 0)) for record in records]
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_replays": len(source_replay_paths),
        "scanned_replays": int(scanned_replays),
        "scanned_frames": int(scanned_frames),
        "candidate_frames": int(candidate_frames),
        "accepted_before_cap": int(accepted_before_cap),
        "records": len(records),
        "source_replay_paths": source_replay_paths,
        "phase_counts": dict(Counter(str(record.get("phase", "unknown")) for record in records)),
        "corner_risk_counts": dict(Counter(str(record.get("corner_risk", "unknown")) for record in records)),
        "strata": dict(Counter(str(record.get("stratum", "unknown")) for record in records)),
        "max_tile_thresholds": {
            f">={threshold}": sum(1 for value in maxes if value >= threshold)
            for threshold in (768, 1536, 3072, 6144)
        },
        "score_minus_starter": {
            "high": max(scores) if scores else 0,
            "mean": float(mean(scores)) if scores else 0.0,
            "median": float(median(scores)) if scores else 0.0,
        },
        "move_count": {
            "max": max(moves) if moves else 0,
            "mean": float(mean(moves)) if moves else 0.0,
            "median": float(median(moves)) if moves else 0.0,
        },
        "rejected": rejected,
    }


def write_report_html(path: Path, payload: dict[str, Any]) -> None:
    summary = payload.get("summary", {})
    records = payload.get("records", [])
    if not isinstance(summary, dict):
        summary = {}
    if not isinstance(records, list):
        records = []
    rows = []
    for record in records[:200]:
        rows.append(
            "<tr>"
            f"<td>{escape(str(record.get('id', '')))}</td>"
            f"<td>{escape(str(record.get('stratum', '')))}</td>"
            f"<td>{escape(str(record.get('max_tile_excl_starter', '')))}</td>"
            f"<td>{escape(str(record.get('score_minus_starter', '')))}</td>"
            f"<td>{escape(str(record.get('move_count', '')))}</td>"
            f"<td>{escape(str(record.get('source_next_action', '')))}</td>"
            f"<td>{escape(str(record.get('source_replay', '')))}</td>"
            f"<td>{escape(str(record.get('source_frame_index', '')))}</td>"
            "</tr>"
        )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Threes High-Board Reservoir</title>
  <style>
    :root {{ color-scheme: dark; --bg: #101318; --panel: #171d24; --line: #34404d; --text: #edf2f7; --muted: #aab6c2; }}
    body {{ margin: 0; background: var(--bg); color: var(--text); font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ width: min(1180px, calc(100vw - 32px)); margin: 0 auto; padding: 24px 0 40px; }}
    h1 {{ margin: 0 0 6px; font-size: 24px; letter-spacing: 0; }}
    .muted {{ color: var(--muted); }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 10px; margin: 18px 0; }}
    .metric {{ border: 1px solid var(--line); background: var(--panel); border-radius: 8px; padding: 12px; }}
    .metric strong {{ display: block; font-size: 22px; margin-top: 4px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 7px 8px; text-align: right; vertical-align: top; }}
    th:first-child, td:first-child, th:nth-child(7), td:nth-child(7) {{ text-align: left; }}
    td:nth-child(7) {{ max-width: 360px; overflow-wrap: anywhere; color: var(--muted); }}
  </style>
</head>
<body>
  <main>
    <h1>High-Board Reservoir</h1>
    <div class="muted">Reusable late-game states extracted from replay frames.</div>
    <section class="grid">
      <div class="metric"><span class="muted">Records</span><strong>{escape(str(summary.get('records', 0)))}</strong></div>
      <div class="metric"><span class="muted">Candidate Frames</span><strong>{escape(str(summary.get('candidate_frames', 0)))}</strong></div>
      <div class="metric"><span class="muted">Scanned Frames</span><strong>{escape(str(summary.get('scanned_frames', 0)))}</strong></div>
      <div class="metric"><span class="muted">Built 3072+</span><strong>{escape(str(summary.get('max_tile_thresholds', {}).get('>=3072', 0) if isinstance(summary.get('max_tile_thresholds'), dict) else 0))}</strong></div>
    </section>
    <table>
      <thead><tr><th>ID</th><th>Stratum</th><th>Max</th><th>Score-Base</th><th>Move</th><th>Next</th><th>Replay</th><th>Frame</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </main>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html)


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    replay_paths = _flatten_paths(args.replay_json) + _glob_paths(args.replay_glob)
    if not replay_paths:
        raise ValueError("No replay JSONs matched")
    phase_filter = parse_filter_values(
        args.phase_filter,
        allowed=PHASE_BUCKETS,
        aliases={
            "early": "early_lt384",
            "mid": "mid_384_768",
            "middle": "mid_384_768",
            "late": "late_1536",
            "endgame": "endgame_3072p",
        },
        label="phase",
    )
    corner_risk_filter = parse_filter_values(
        args.corner_risk_filter,
        allowed=CORNER_RISK_BUCKETS,
        aliases={
            "low": "low_corner_risk",
            "medium": "medium_corner_risk",
            "med": "medium_corner_risk",
            "high": "high_corner_risk",
        },
        label="corner risk",
    )
    starter_text = args.default_starter.strip().lower()
    default_starter = None if starter_text == "none" else int(starter_text)
    records, summary = collect_reservoir_records(
        replay_paths,
        min_tile=args.min_tile,
        phase_filter=phase_filter,
        corner_risk_filter=corner_risk_filter,
        default_starter_tile=default_starter,
        first_per=args.first_per,
        max_records=args.max_records,
        max_per_stratum=args.max_per_stratum,
        sort_by=args.sort_by,
    )
    payload = {
        "version": 1,
        "kind": "threes_high_board_reservoir",
        "records": records,
        "summary": summary,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "reservoir.json"
    records_path = args.out_dir / "records.json"
    html_path = args.out_dir / "reservoir.html"
    write_json(json_path, payload)
    write_json(records_path, records)
    write_report_html(html_path, payload)
    payload["json"] = str(json_path)
    payload["records_json"] = str(records_path)
    payload["html"] = str(html_path)
    write_json(json_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-json", type=Path, nargs="+", action="append")
    parser.add_argument("--replay-glob", action="append")
    parser.add_argument("--min-tile", type=int, default=1536)
    parser.add_argument("--phase-filter", action="append")
    parser.add_argument("--corner-risk-filter", action="append")
    parser.add_argument("--default-starter", default="1536")
    parser.add_argument(
        "--first-per",
        choices=["none", "replay", "replay-phase", "replay-stratum"],
        default="none",
    )
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--max-per-stratum", type=int, default=0)
    parser.add_argument("--sort-by", choices=["source", "score", "max_tile", "move"], default="source")
    parser.add_argument("--out-dir", type=Path, default=Path("threes_rl/runs/forensics/high_board_reservoir/latest"))
    args = parser.parse_args()
    payload = run_from_args(args)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"json={payload['json']}")
    print(f"html={payload['html']}")


if __name__ == "__main__":
    main()
