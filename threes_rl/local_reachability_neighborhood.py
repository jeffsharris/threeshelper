"""Audit reachability-model neighborhoods around selected replay starts."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from html import escape
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

import numpy as np

from threes_rl.compare_replays import state_from_payload
from threes_rl.continue_from_replays import _starter_from_record
from threes_rl.high_board_reservoir import _record_id as high_board_record_id
from threes_rl.replay_policy_agreement import parse_phase_filters
from threes_rl.run_artifacts import write_json
from threes_rl.sim import ThreesSim
from threes_rl.support_ladder_window_reservoir import raw_ladder_features
from threes_rl.swing_label import state_features
from threes_rl.transition_reachability_audit import build_rows, load_records
from threes_rl.transition_reachability_score import (
    _fit_model,
    _model_summary,
    _predict,
    _score_summary,
    _scored_row,
    candidate_pairs,
)


def _flatten_paths(path_groups: list[list[Path]] | None) -> list[Path]:
    if not path_groups:
        return []
    return [path for group in path_groups for path in group]


def _load_record_payload(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    records = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError(f"{path} does not contain records[]")
    return [record for record in records if isinstance(record, dict)]


def load_anchor_records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    anchors: list[dict[str, Any]] = []
    for path in paths:
        for idx, record in enumerate(_load_record_payload(Path(path))):
            anchors.append({**record, "_anchor_json": str(path), "_anchor_record_index": int(idx)})
    return anchors


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


def _move_action(frame: dict[str, Any] | None) -> str | None:
    if not isinstance(frame, dict):
        return None
    move = frame.get("move")
    if not isinstance(move, dict):
        return None
    action = move.get("action")
    return str(action) if action is not None else None


def _frame_position(frames: list[Any], frame_index: int) -> int | None:
    for pos, frame in enumerate(frames):
        if not isinstance(frame, dict):
            continue
        if int(frame.get("index", pos)) == int(frame_index):
            return pos
    return None


def _anchor_id(anchor: dict[str, Any], fallback: int) -> str:
    value = anchor.get("id")
    if value is not None:
        return str(value)
    return f"anchor_{fallback}"


def _record_from_frame(
    *,
    replay_path: Path,
    replay: dict[str, Any],
    frames: list[Any],
    frame_pos: int,
    anchor: dict[str, Any],
    anchor_index: int,
    anchor_frame_index: int,
    default_starter_tile: int | None,
) -> dict[str, Any] | None:
    frame = frames[frame_pos]
    if not isinstance(frame, dict):
        return None
    state_payload = frame.get("state")
    if not isinstance(state_payload, dict):
        return None
    state = state_from_payload(state_payload)
    if state.game_over:
        return None
    starter_tile = _starter_from_record(anchor, _starter_from_replay(replay, default_starter_tile))
    seed = _int_or_none(replay.get("seed", anchor.get("source_seed", anchor.get("seed"))))
    sim = ThreesSim(np.random.default_rng(seed if seed is not None else anchor_index), starter_tile=starter_tile)
    features = dict(state_features(state, sim, starter_tile))
    raw = raw_ladder_features(state.board, starter_tile)
    legal_count = len(sim.legal_actions(state))
    merged_features = {
        **features,
        **raw,
        "legal_count": int(legal_count),
        "anchor_id": _anchor_id(anchor, anchor_index),
        "anchor_frame_index": int(anchor_frame_index),
        "frame_offset": int(int(frame.get("index", frame_pos)) - int(anchor_frame_index)),
    }
    frame_index = int(frame.get("index", frame_pos))
    record = {
        "id": high_board_record_id(replay_path, seed, frame_index, features),
        "kind": "local_reachability_neighborhood_state",
        "anchor_id": _anchor_id(anchor, anchor_index),
        "anchor_frame_index": int(anchor_frame_index),
        "anchor_source_replay": str(anchor.get("source_replay", replay_path)),
        "frame_offset": int(frame_index - int(anchor_frame_index)),
        "source_replay": str(replay_path),
        "source_policy": replay.get("policy"),
        "source_seed": seed,
        "seed": seed,
        "source_frame_index": int(frame_index),
        "frame_position": int(frame_pos),
        "source_next_action": _move_action(frames[frame_pos + 1] if frame_pos + 1 < len(frames) else None),
        "starter_tile": starter_tile,
        "move_count": int(state.move_count),
        "score": int(state_payload.get("score", 0)),
        "score_minus_starter": int(features["score_minus_starter"]),
        "max_tile": int(features["max_tile"]),
        "max_tile_excl_starter": int(features["max_tile_excl_starter"]),
        "phase": str(features["phase"]),
        "corner_risk": str(features["corner_risk"]),
        "stratum": str(features["stratum"]),
        "empty_count": int(features["empty_count"]),
        "legal_count": int(legal_count),
        "preview": str(features["preview"]),
        "large_pending": bool(features["large_pending"]),
        "raw_count_768": int(raw["raw_count_768"]),
        "raw_count_1536": int(raw["raw_count_1536"]),
        "raw_count_3072": int(raw["raw_count_3072"]),
        "raw_count_6144": int(raw["raw_count_6144"]),
        "raw_highest_duplicate_tile": int(raw["raw_highest_duplicate_tile"]),
        "raw_highest_adjacent_pair_tile": int(raw["raw_highest_adjacent_pair_tile"]),
        "raw_has_adjacent_768": bool(raw["raw_has_adjacent_768"]),
        "raw_has_adjacent_1536": bool(raw["raw_has_adjacent_1536"]),
        "features": merged_features,
        "state": state_payload,
    }
    return record


def collect_neighborhood_records(
    anchors: list[dict[str, Any]],
    *,
    radius: int,
    min_tile: int,
    phase_filter: set[str] | None,
    default_starter_tile: int | None = 1536,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    scanned = 0
    for anchor_index, anchor in enumerate(anchors):
        source = anchor.get("source_replay")
        if source is None:
            rejected["missing_source_replay"] += 1
            continue
        replay_path = Path(str(source))
        try:
            replay = json.loads(replay_path.read_text())
        except (OSError, json.JSONDecodeError):
            rejected["bad_replay"] += 1
            continue
        frames = replay.get("frames")
        if not isinstance(frames, list):
            rejected["bad_replay"] += 1
            continue
        frame_index = _int_or_none(anchor.get("source_frame_index", anchor.get("frame_index")))
        if frame_index is None:
            rejected["missing_anchor_frame"] += 1
            continue
        frame_pos = _frame_position(frames, frame_index)
        if frame_pos is None:
            rejected["anchor_frame_not_found"] += 1
            continue
        start = max(0, int(frame_pos) - int(radius))
        end = min(len(frames) - 1, int(frame_pos) + int(radius))
        for pos in range(start, end + 1):
            scanned += 1
            record = _record_from_frame(
                replay_path=replay_path,
                replay=replay,
                frames=frames,
                frame_pos=pos,
                anchor=anchor,
                anchor_index=anchor_index,
                anchor_frame_index=frame_index,
                default_starter_tile=default_starter_tile,
            )
            if record is None:
                rejected["bad_frame"] += 1
                continue
            if int(record["max_tile_excl_starter"]) < int(min_tile):
                rejected["below_min_tile"] += 1
                continue
            if phase_filter is not None and str(record["phase"]) not in phase_filter:
                rejected["phase_filter"] += 1
                continue
            records.append(record)
    summary = {
        "anchors": len(anchors),
        "radius": int(radius),
        "min_tile": int(min_tile),
        "phase_filter": sorted(phase_filter) if phase_filter is not None else None,
        "scanned_frames": int(scanned),
        "records": len(records),
        "rejected": dict(rejected),
    }
    return records, summary


def score_records(
    records: list[dict[str, Any]],
    *,
    train_paths: Iterable[Path],
    target_tile: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    train_records = load_records(list(train_paths))
    train_rows = build_rows(train_records, target_tile=target_tile)
    model = _fit_model(train_rows)
    pairs = candidate_pairs(records)
    probs = _predict(model, [row for _record, row in pairs])
    scored_records: list[dict[str, Any]] = []
    scored_rows: list[dict[str, Any]] = []
    for rank, ((record, row), prob) in enumerate(zip(pairs, probs), start=1):
        scored_row = _scored_row(record, row, prob)
        scored_row.update(
            {
                "anchor_id": record.get("anchor_id"),
                "anchor_frame_index": record.get("anchor_frame_index"),
                "frame_offset": record.get("frame_offset"),
            }
        )
        scored_rows.append(scored_row)
        scored_record = {**record, "reachability_prob": float(prob)}
        scored_records.append(scored_record)
    scored_rows.sort(key=lambda item: float(item["reachability_prob"]), reverse=True)
    for rank, row in enumerate(scored_rows, start=1):
        row["rank"] = int(rank)
    scored_by_id = {str(row.get("id")): row for row in scored_rows}
    for record in scored_records:
        row = scored_by_id.get(str(record.get("id")))
        record["reachability_rank"] = int(row["rank"]) if row is not None else None
        record["reachability_model"] = "transition_reachability_logistic"
    scored_records.sort(key=lambda item: float(item.get("reachability_prob", 0.0)), reverse=True)
    model_payload = {
        "train_records": len(train_records),
        "train_rows": len(train_rows),
        "train_successes": int(sum(int(row["y"]) for row in train_rows)),
        "train_failures": int(len(train_rows) - sum(int(row["y"]) for row in train_rows)),
        "model": _model_summary(model),
    }
    return scored_records, {"summary": _score_summary(scored_rows), "top_rows": scored_rows[:32]}, model_payload


def anchor_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("anchor_id"))].append(record)
    rows: list[dict[str, Any]] = []
    for anchor_id, items in grouped.items():
        probs = [float(item.get("reachability_prob", 0.0)) for item in items]
        anchor_items = [item for item in items if int(item.get("frame_offset", 999999)) == 0]
        top = max(items, key=lambda item: float(item.get("reachability_prob", 0.0)))
        rows.append(
            {
                "anchor_id": anchor_id,
                "source_replay": top.get("source_replay"),
                "anchor_frame_index": top.get("anchor_frame_index"),
                "records": len(items),
                "mean_prob": float(mean(probs)) if probs else 0.0,
                "median_prob": float(median(probs)) if probs else 0.0,
                "max_prob": float(max(probs)) if probs else 0.0,
                "anchor_prob": float(anchor_items[0].get("reachability_prob", 0.0)) if anchor_items else None,
                "top_frame_index": top.get("source_frame_index"),
                "top_frame_offset": top.get("frame_offset"),
                "top_score_minus_starter": top.get("score_minus_starter"),
                "top_empty_count": top.get("empty_count"),
                "top_preview": top.get("preview"),
                "top_raw_count_768": top.get("raw_count_768"),
                "top_raw_count_1536": top.get("raw_count_1536"),
            }
        )
    rows.sort(key=lambda row: (float(row["max_prob"]), float(row["mean_prob"])), reverse=True)
    return rows


def by_offset_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for record in records:
        grouped[int(record.get("frame_offset", 0))].append(float(record.get("reachability_prob", 0.0)))
    rows = [
        {
            "frame_offset": offset,
            "records": len(values),
            "mean_prob": float(mean(values)),
            "median_prob": float(median(values)),
            "max_prob": float(max(values)),
        }
        for offset, values in grouped.items()
    ]
    rows.sort(key=lambda row: int(row["frame_offset"]))
    return rows


def dedupe_source_frames(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for record in records:
        key = (str(record.get("source_replay")), int(record.get("source_frame_index", -1)))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


def select_export_records(
    records: list[dict[str, Any]],
    *,
    top_n: int = 0,
    source_replay_limit: int = 0,
    source_seed_limit: int = 0,
    frame_min_gap: int = 0,
) -> list[dict[str, Any]]:
    limit = len(records) if int(top_n) <= 0 else int(top_n)
    selected: list[dict[str, Any]] = []
    replay_counts: Counter[str] = Counter()
    seed_counts: Counter[str] = Counter()
    replay_frames: dict[str, list[int]] = defaultdict(list)
    for record in records:
        replay = str(record.get("source_replay", "unknown"))
        seed = str(record.get("source_seed", "unknown"))
        frame = int(record.get("source_frame_index", -1))
        if source_replay_limit > 0 and replay_counts[replay] >= int(source_replay_limit):
            continue
        if source_seed_limit > 0 and seed_counts[seed] >= int(source_seed_limit):
            continue
        if frame_min_gap > 0 and any(abs(frame - prior) < int(frame_min_gap) for prior in replay_frames[replay]):
            continue
        selected.append(record)
        replay_counts[replay] += 1
        seed_counts[seed] += 1
        replay_frames[replay].append(frame)
        if len(selected) >= limit:
            break
    return selected


def write_html(path: Path, payload: dict[str, Any]) -> None:
    anchors = payload.get("by_anchor", [])
    top = payload.get("top_records", [])

    def cell(value: object) -> str:
        return escape(str(value))

    anchor_rows = []
    for row in anchors if isinstance(anchors, list) else []:
        anchor_rows.append(
            "<tr>"
            f"<td>{cell(row.get('anchor_id'))}</td>"
            f"<td>{cell(row.get('anchor_frame_index'))}</td>"
            f"<td>{cell(row.get('records'))}</td>"
            f"<td>{float(row.get('anchor_prob') or 0.0):.3f}</td>"
            f"<td>{float(row.get('max_prob', 0.0)):.3f}</td>"
            f"<td>{cell(row.get('top_frame_index'))}</td>"
            f"<td>{cell(row.get('top_frame_offset'))}</td>"
            f"<td>{cell(row.get('top_empty_count'))}</td>"
            f"<td>{cell(row.get('top_preview'))}</td>"
            "</tr>"
        )
    top_rows = []
    for row in top if isinstance(top, list) else []:
        top_rows.append(
            "<tr>"
            f"<td>{cell(row.get('reachability_rank'))}</td>"
            f"<td>{float(row.get('reachability_prob', 0.0)):.3f}</td>"
            f"<td>{cell(row.get('anchor_frame_index'))}</td>"
            f"<td>{cell(row.get('source_frame_index'))}</td>"
            f"<td>{cell(row.get('frame_offset'))}</td>"
            f"<td>{cell(row.get('score_minus_starter'))}</td>"
            f"<td>{cell(row.get('empty_count'))}</td>"
            f"<td>{cell(row.get('preview'))}</td>"
            f"<td>{cell(row.get('raw_count_768'))}</td>"
            f"<td>{cell(row.get('raw_count_1536'))}</td>"
            "</tr>"
        )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Local Reachability Neighborhood</title>
  <style>
    :root {{ color-scheme: dark; --bg: #101318; --panel: #171d24; --line: #34404d; --text: #edf2f7; --muted: #aab6c2; --gold: #f2c14e; }}
    body {{ margin: 0; background: var(--bg); color: var(--text); font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ width: min(1180px, calc(100vw - 32px)); margin: 0 auto; padding: 24px 0 40px; }}
    h1 {{ margin: 0 0 8px; font-size: 24px; }}
    h2 {{ margin: 22px 0 10px; font-size: 16px; }}
    .muted {{ color: var(--muted); }}
    .panel {{ border: 1px solid var(--line); background: var(--panel); border-radius: 8px; padding: 12px; margin-top: 12px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; font-variant-numeric: tabular-nums; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 7px 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; max-width: 360px; overflow-wrap: anywhere; }}
    pre {{ white-space: pre-wrap; overflow-wrap: anywhere; color: var(--muted); }}
  </style>
</head>
<body>
  <main>
    <h1>Local Reachability Neighborhood</h1>
    <p class="muted">Frames near selected anchors scored by the transition reachability model.</p>
    <section class="panel"><pre>{escape(json.dumps(payload.get('summary', {}), indent=2, sort_keys=True))}</pre></section>
    <h2>By Anchor</h2>
    <section class="panel"><table><thead><tr><th>Anchor</th><th>Frame</th><th>Rows</th><th>Anchor Prob</th><th>Max Prob</th><th>Top Frame</th><th>Offset</th><th>Empty</th><th>Preview</th></tr></thead><tbody>{''.join(anchor_rows)}</tbody></table></section>
    <h2>Top Frames</h2>
    <section class="panel"><table><thead><tr><th>Rank</th><th>Prob</th><th>Anchor Frame</th><th>Frame</th><th>Offset</th><th>Score - Starter</th><th>Empty</th><th>Preview</th><th>Raw 768</th><th>Raw 1536</th></tr></thead><tbody>{''.join(top_rows)}</tbody></table></section>
  </main>
</body>
</html>
"""
    path.write_text(html)


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    anchor_paths = _flatten_paths(args.anchor_json)
    train_paths = _flatten_paths(args.train_json)
    anchors = load_anchor_records(anchor_paths)
    phase_filter = parse_phase_filters(args.phase_filter)
    default_starter = None if args.default_starter == "none" else int(args.default_starter)
    records, collect_summary = collect_neighborhood_records(
        anchors,
        radius=args.radius,
        min_tile=args.min_tile,
        phase_filter=phase_filter,
        default_starter_tile=default_starter,
    )
    scored_records, score_payload, model_payload = score_records(
        records,
        train_paths=train_paths,
        target_tile=args.target_tile,
    )
    unique_scored_records = dedupe_source_frames(scored_records)
    selected_records = select_export_records(
        unique_scored_records,
        top_n=args.export_top_n,
        source_replay_limit=args.export_source_replay_limit,
        source_seed_limit=args.export_source_seed_limit,
        frame_min_gap=args.export_frame_min_gap,
    )
    payload = {
        "kind": "local_reachability_neighborhood",
        "version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "anchor_json": [str(path) for path in anchor_paths],
        "train_json": [str(path) for path in train_paths],
        "target_tile": int(args.target_tile),
        "summary": {
            **collect_summary,
            "unique_records": len(unique_scored_records),
            "selected_records": len(selected_records),
            **score_payload["summary"],
            **{key: value for key, value in model_payload.items() if key != "model"},
        },
        "export_selection": {
            "export_top_n": int(args.export_top_n),
            "source_replay_limit": int(args.export_source_replay_limit),
            "source_seed_limit": int(args.export_source_seed_limit),
            "frame_min_gap": int(args.export_frame_min_gap),
        },
        "model": model_payload["model"],
        "by_anchor": anchor_summary(scored_records),
        "by_offset": by_offset_summary(scored_records),
        "top_records": selected_records[: int(args.top_n)],
        "records": unique_scored_records,
        "selected_records": selected_records,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload["json"] = str(args.out_dir / "local_reachability_neighborhood.json")
    payload["records_json"] = str(args.out_dir / "records.json")
    payload["selected_records_json"] = str(args.out_dir / "selected_records.json")
    payload["html"] = str(args.out_dir / "local_reachability_neighborhood.html")
    write_json(args.out_dir / "local_reachability_neighborhood.json", payload)
    write_json(args.out_dir / "records.json", unique_scored_records)
    write_json(args.out_dir / "selected_records.json", selected_records)
    write_html(args.out_dir / "local_reachability_neighborhood.html", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-json", type=Path, nargs="+", action="append", required=True)
    parser.add_argument("--train-json", type=Path, nargs="+", action="append", required=True)
    parser.add_argument("--target-tile", type=int, default=1536)
    parser.add_argument("--radius", type=int, default=20)
    parser.add_argument("--min-tile", type=int, default=3072)
    parser.add_argument("--phase-filter", action="append", default=["endgame"])
    parser.add_argument("--default-starter", default="1536")
    parser.add_argument("--top-n", type=int, default=32)
    parser.add_argument("--export-top-n", type=int, default=0)
    parser.add_argument("--export-source-replay-limit", type=int, default=0)
    parser.add_argument("--export-source-seed-limit", type=int, default=0)
    parser.add_argument("--export-frame-min-gap", type=int, default=0)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("threes_rl/runs/forensics/local_reachability_neighborhood/latest"),
    )
    args = parser.parse_args()
    payload = run_from_args(args)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"json={payload['json']}")
    print(f"records={payload['records_json']}")
    print(f"selected_records={payload['selected_records_json']}")
    print(f"html={payload['html']}")


if __name__ == "__main__":
    main()
