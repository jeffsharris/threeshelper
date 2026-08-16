"""Extract support-material windows before selected near-failure states."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import Counter, defaultdict
from html import escape
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import numpy as np

from threes_rl.compare_replays import state_from_payload
from threes_rl.nearfail_support_audit import support_bucket
from threes_rl.rare_event_frontier import load_records, parse_root_origins
from threes_rl.record_replay import state_payload
from threes_rl.run_artifacts import safe_name, write_json
from threes_rl.sim import ThreesSim, score_board
from threes_rl.support_accumulation_frontier import _raw


def parse_offsets(text: str | None) -> list[int]:
    raw = text or "40,20,10,5,0"
    offsets: list[int] = []
    seen: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        offset = int(part)
        if offset < 0:
            raise ValueError(f"offset must be non-negative: {offset}")
        if offset not in seen:
            offsets.append(offset)
            seen.add(offset)
    if not offsets:
        raise ValueError("at least one offset is required")
    return offsets


def _int_or_none(value: object) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _record_id(record: dict[str, Any]) -> str:
    if record.get("id") is not None:
        return str(record["id"])
    raw = json.dumps(
        {
            "source_replay": record.get("source_replay"),
            "source_frame_index": record.get("source_frame_index", record.get("frame_index")),
            "root_seed": record.get("root_seed"),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return safe_name(f"nearfail_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:12]}")


def _move_action(frame: object) -> str | None:
    if not isinstance(frame, dict):
        return None
    move = frame.get("move")
    if not isinstance(move, dict):
        return None
    value = move.get("action")
    return str(value) if value is not None else None


def _frames(replay: dict[str, Any]) -> list[dict[str, Any]]:
    frames = replay.get("frames")
    return frames if isinstance(frames, list) else []


def _frame_position(frames: list[dict[str, Any]], frame_index: int | None) -> int | None:
    if frame_index is None:
        return None
    for pos, frame in enumerate(frames):
        if isinstance(frame, dict) and _int_or_none(frame.get("index")) == int(frame_index):
            return int(pos)
    if 0 <= int(frame_index) < len(frames):
        return int(frame_index)
    return None


def _starter(record: dict[str, Any], replay: dict[str, Any], default_starter_tile: int | None) -> int | None:
    value = record.get("starter_tile", replay.get("starter_tile", default_starter_tile))
    return None if value is None else int(value)


def _state_payload_from_frame(frame: dict[str, Any]) -> dict[str, Any] | None:
    payload = frame.get("state")
    return payload if isinstance(payload, dict) else None


def _row_for_frame(
    *,
    record: dict[str, Any],
    replay_path: Path,
    replay: dict[str, Any],
    frames: list[dict[str, Any]],
    frame_position: int,
    selected_position: int,
    offset: int,
    default_starter_tile: int | None,
) -> dict[str, Any]:
    frame = frames[frame_position]
    payload = _state_payload_from_frame(frame)
    if payload is None:
        raise ValueError("missing state payload")
    starter_tile = _starter(record, replay, default_starter_tile)
    state = state_from_payload(payload)
    raw = _raw(state, starter_tile)
    seed = _int_or_none(record.get("source_seed", replay.get("seed")))
    sim = ThreesSim(np.random.default_rng(seed or 0), starter_tile=starter_tile)
    legal = sim.legal_actions(state)
    nearfail_id = _record_id(record)
    row_id = safe_name(f"{nearfail_id}_pre{int(offset)}_{hashlib.sha1(str(replay_path).encode('utf-8')).hexdigest()[:8]}")
    source_frame_index = _int_or_none(frame.get("index", frame_position))
    return {
        "id": row_id,
        "kind": "pre_nearfail_support_window_state",
        "nearfail_record_id": nearfail_id,
        "window_offset": int(offset),
        "selected_frame_position": int(selected_position),
        "source_frame_position": int(frame_position),
        "source_frame_index": source_frame_index,
        "source_next_action": _move_action(frames[frame_position + 1]) if frame_position + 1 < len(frames) else None,
        "source_replay": str(replay_path),
        "source_seed": seed,
        "source_policy": record.get("source_policy", replay.get("policy")),
        "source_policy_family": record.get("source_policy_family"),
        "root_origin": record.get("root_origin"),
        "root_replay": record.get("root_replay"),
        "root_seed": record.get("root_seed"),
        "root_frame_index": record.get("root_frame_index"),
        "root_policy": record.get("root_policy"),
        "root_policy_family": record.get("root_policy_family"),
        "ancestry_key": record.get("ancestry_key"),
        "starter_tile": starter_tile,
        "state": state_payload(state, sim),
        "features": {
            **raw,
            "support_bucket": support_bucket(raw),
            "legal_count": int(len(legal)),
            "score": int(score_board(state.board)),
            "move_count": int(state.move_count),
            "max_tile": int(state.max_tile),
        },
        "score": int(score_board(state.board)),
        "move_count": int(state.move_count),
        "raw_count_768": int(raw["raw_count_768"]),
        "raw_count_384": int(raw["raw_count_384"]),
        "raw_count_192": int(raw["raw_count_192"]),
        "raw_has_adjacent_384": bool(raw["raw_has_adjacent_384"]),
        "raw_has_adjacent_192": bool(raw["raw_has_adjacent_192"]),
        "raw_has_adjacent_768": bool(raw["raw_has_adjacent_768"]),
        "masked_count_1536": int(raw["masked_count_1536"]),
        "support_bucket": support_bucket(raw),
        "empty_count": int(raw["empty_count"]),
        "legal_count": int(len(legal)),
    }


def _case_final_bucket(record: dict[str, Any], default_starter_tile: int | None) -> str | None:
    payload = record.get("state")
    if not isinstance(payload, dict):
        return None
    state = state_from_payload(payload)
    starter_tile = None if record.get("starter_tile", default_starter_tile) is None else int(record.get("starter_tile", default_starter_tile))
    return support_bucket(_raw(state, starter_tile))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "nearfail_record_id",
        "root_seed",
        "window_offset",
        "source_frame_index",
        "source_next_action",
        "support_bucket",
        "raw_count_768",
        "raw_count_384",
        "raw_count_192",
        "raw_has_adjacent_384",
        "raw_has_adjacent_192",
        "raw_has_adjacent_768",
        "masked_count_1536",
        "empty_count",
        "legal_count",
        "score",
        "move_count",
        "source_replay",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def summarize(rows: list[dict[str, Any]], *, selected_cases: int, skipped: Counter[str], offsets: list[int]) -> dict[str, Any]:
    by_offset: dict[str, Counter[str]] = {str(offset): Counter() for offset in offsets}
    raw_768_by_offset: dict[str, Counter[str]] = {str(offset): Counter() for offset in offsets}
    support_material_by_offset: dict[str, Counter[str]] = {str(offset): Counter() for offset in offsets}
    support_present_by_offset: Counter[str] = Counter()
    no_support_at_zero = {
        row["nearfail_record_id"]
        for row in rows
        if int(row["window_offset"]) == 0 and row.get("support_bucket") == "no_384_192"
    }
    case_offset_support: dict[str, dict[int, bool]] = defaultdict(dict)
    for row in rows:
        offset = int(row["window_offset"])
        bucket = str(row.get("support_bucket"))
        by_offset[str(offset)][bucket] += 1
        raw_count_384 = int(row.get("raw_count_384", 0))
        raw_count_192 = int(row.get("raw_count_192", 0))
        raw_768_by_offset[str(offset)][str(int(row.get("raw_count_768", 0)))] += 1
        if raw_count_384 > 0:
            support_material_by_offset[str(offset)]["has_384"] += 1
        if raw_count_192 > 0:
            support_material_by_offset[str(offset)]["has_192"] += 1
        if raw_count_384 == 0 and raw_count_192 == 0:
            support_material_by_offset[str(offset)]["has_neither"] += 1
        has_support = raw_count_384 > 0 or raw_count_192 > 0
        if has_support:
            support_present_by_offset[str(offset)] += 1
        case_offset_support[str(row["nearfail_record_id"])][offset] = has_support
    lost_support_by_offset: dict[str, int] = {}
    for offset in offsets:
        if int(offset) == 0:
            continue
        lost_support_by_offset[str(offset)] = sum(
            1 for case_id in no_support_at_zero if case_offset_support.get(case_id, {}).get(int(offset), False)
        )
    nearest_support_before_zero: Counter[str] = Counter()
    positive_offsets = sorted((int(offset) for offset in offsets if int(offset) > 0))
    for case_id in no_support_at_zero:
        supported = [
            offset for offset in positive_offsets if case_offset_support.get(case_id, {}).get(int(offset), False)
        ]
        nearest_support_before_zero[str(min(supported)) if supported else "none"] += 1
    score_by_offset: dict[str, float | None] = {}
    for offset in offsets:
        values = [float(row["score"]) for row in rows if int(row["window_offset"]) == int(offset)]
        score_by_offset[str(offset)] = float(mean(values)) if values else None
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "selected_cases": int(selected_cases),
        "rows": len(rows),
        "unique_roots": len({row.get("root_seed") for row in rows if row.get("root_seed") is not None}),
        "offsets": [int(offset) for offset in offsets],
        "skipped": dict(skipped),
        "by_offset_support_bucket": {offset: dict(counter) for offset, counter in by_offset.items()},
        "raw_768_by_offset": {offset: dict(counter) for offset, counter in raw_768_by_offset.items()},
        "support_material_by_offset": {offset: dict(counter) for offset, counter in support_material_by_offset.items()},
        "support_present_by_offset": {offset: int(support_present_by_offset.get(offset, 0)) for offset in map(str, offsets)},
        "no_support_at_zero_cases": len(no_support_at_zero),
        "lost_support_by_offset": lost_support_by_offset,
        "nearest_support_before_zero": dict(nearest_support_before_zero),
        "mean_score_by_offset": score_by_offset,
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    offsets = [int(offset) for offset in summary["offsets"]]
    lines = [
        "# Pre-Near-Failure Support Window Audit",
        "",
        f"created_at: `{summary['created_at']}`",
        "",
        f"- selected cases: `{summary['selected_cases']}`",
        f"- emitted rows: `{summary['rows']}`",
        f"- unique roots: `{summary['unique_roots']}`",
        f"- skipped: `{summary['skipped']}`",
        "",
        "| offset | support-present rows | material | raw 768 counts | support buckets | mean score |",
        "| ---: | ---: | --- | --- | --- | ---: |",
    ]
    for offset in offsets:
        buckets = summary["by_offset_support_bucket"].get(str(offset), {})
        material = summary["support_material_by_offset"].get(str(offset), {})
        raw_768 = summary["raw_768_by_offset"].get(str(offset), {})
        lines.append(
            f"| {offset} | {summary['support_present_by_offset'].get(str(offset), 0)} | "
            f"`{material}` | `{raw_768}` | `{buckets}` | {summary['mean_score_by_offset'].get(str(offset))} |"
        )
    if summary.get("lost_support_by_offset"):
        lines += ["", "Loss relative to offset `0` no-support cases:", ""]
        for offset, count in summary["lost_support_by_offset"].items():
            lines.append(f"- offset `{offset}` had support in `{count}` cases that ended with no `192/384` support at offset `0`.")
    if summary.get("nearest_support_before_zero"):
        lines += ["", f"Nearest observed support before offset `0`: `{summary['nearest_support_before_zero']}`"]
    path.write_text("\n".join(lines) + "\n")


def write_html(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]

    def cell(value: object) -> str:
        return escape(str(value))

    rows = []
    for offset in summary["offsets"]:
        buckets = summary["by_offset_support_bucket"].get(str(offset), {})
        rows.append(
            "<tr>"
            f"<td>{cell(offset)}</td>"
            f"<td>{cell(summary['support_present_by_offset'].get(str(offset), 0))}</td>"
            f"<td>{cell(summary['support_material_by_offset'].get(str(offset), {}))}</td>"
            f"<td>{cell(summary['raw_768_by_offset'].get(str(offset), {}))}</td>"
            f"<td>{cell(buckets)}</td>"
            f"<td>{cell(summary['mean_score_by_offset'].get(str(offset)))}</td>"
            "</tr>"
        )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Pre-Near-Failure Support Window Audit</title>
  <style>
    :root {{ color-scheme: dark; --bg:#101318; --panel:#171d24; --line:#34404d; --text:#edf2f7; --muted:#aab6c2; --gold:#f2c14e; }}
    body {{ margin:0; background:var(--bg); color:var(--text); font-family:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ width:min(1100px, calc(100vw - 32px)); margin:0 auto; padding:24px 0 40px; }}
    h1 {{ margin:0 0 8px; font-size:24px; }}
    .muted {{ color:var(--muted); }}
    .cards {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(150px, 1fr)); gap:10px; margin:18px 0; }}
    .card, .panel {{ border:1px solid var(--line); background:var(--panel); border-radius:8px; padding:12px; }}
    .label {{ color:var(--muted); font-size:11px; text-transform:uppercase; }}
    .value {{ margin-top:4px; color:var(--gold); font-size:22px; font-weight:800; }}
    table {{ width:100%; border-collapse:collapse; font-size:12px; margin-top:12px; }}
    th, td {{ border-bottom:1px solid var(--line); padding:8px; text-align:right; vertical-align:top; }}
    th:nth-child(3), td:nth-child(3), th:nth-child(4), td:nth-child(4), th:nth-child(5), td:nth-child(5) {{ text-align:left; overflow-wrap:anywhere; }}
  </style>
</head>
<body>
<main>
  <h1>Pre-Near-Failure Support Window Audit</h1>
  <p class="muted">Raw 192/384 support material before selected one-768 near-failure states.</p>
  <section class="cards">
    <div class="card"><div class="label">Cases</div><div class="value">{cell(summary['selected_cases'])}</div></div>
    <div class="card"><div class="label">Rows</div><div class="value">{cell(summary['rows'])}</div></div>
    <div class="card"><div class="label">Roots</div><div class="value">{cell(summary['unique_roots'])}</div></div>
    <div class="card"><div class="label">No-Support At 0</div><div class="value">{cell(summary['no_support_at_zero_cases'])}</div></div>
  </section>
  <section class="panel">
    <table><thead><tr><th>Offset</th><th>Support Rows</th><th>Material</th><th>Raw 768</th><th>Support Buckets</th><th>Mean Score</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
  </section>
</main>
</body>
</html>
"""
    path.write_text(html)


def run_audit(
    *,
    records_json: list[Path],
    out_dir: Path,
    offsets: list[int],
    final_bucket: str | None = None,
    root_origins: set[str] | None = None,
    default_starter_tile: int | None = 1536,
) -> dict[str, Any]:
    records = load_records(records_json)
    replay_cache: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    selected_cases = 0
    root_origins = root_origins or {"fresh", "human"}
    for record in records:
        if str(record.get("root_origin", "")) not in root_origins:
            skipped["root_origin"] += 1
            continue
        if final_bucket:
            bucket = _case_final_bucket(record, default_starter_tile)
            if bucket != final_bucket:
                skipped[f"final_bucket:{bucket}"] += 1
                continue
        replay_value = record.get("source_replay")
        if not replay_value:
            skipped["missing_source_replay"] += 1
            continue
        replay_path = Path(str(replay_value))
        if not replay_path.exists():
            skipped["missing_replay_file"] += 1
            continue
        replay = replay_cache.get(str(replay_path))
        if replay is None:
            replay = json.loads(replay_path.read_text())
            replay_cache[str(replay_path)] = replay
        frames = _frames(replay)
        selected_position = _frame_position(frames, _int_or_none(record.get("source_frame_index", record.get("frame_index"))))
        if selected_position is None:
            skipped["missing_selected_frame"] += 1
            continue
        selected_cases += 1
        for offset in offsets:
            pos = int(selected_position) - int(offset)
            if pos < 0:
                skipped[f"offset_before_replay:{offset}"] += 1
                continue
            try:
                rows.append(
                    _row_for_frame(
                        record=record,
                        replay_path=replay_path,
                        replay=replay,
                        frames=frames,
                        frame_position=pos,
                        selected_position=int(selected_position),
                        offset=int(offset),
                        default_starter_tile=default_starter_tile,
                    )
                )
            except (TypeError, ValueError, KeyError):
                skipped[f"bad_frame:{offset}"] += 1
    summary = summarize(rows, selected_cases=selected_cases, skipped=skipped, offsets=offsets)
    payload = {
        "kind": "pre_nearfail_support_window_audit",
        "records_json": [str(path) for path in records_json],
        "final_bucket": final_bucket,
        "summary": summary,
        "records": rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    payload["json"] = str(out_dir / "pre_nearfail_support_windows.json")
    payload["records_csv"] = str(out_dir / "pre_nearfail_support_windows.csv")
    payload["summary_json"] = str(out_dir / "summary.json")
    payload["report"] = str(out_dir / "report.md")
    payload["html"] = str(out_dir / "pre_nearfail_support_windows.html")
    write_json(out_dir / "pre_nearfail_support_windows.json", payload)
    write_json(out_dir / "summary.json", summary)
    _write_csv(out_dir / "pre_nearfail_support_windows.csv", rows)
    write_report(out_dir / "report.md", payload)
    write_html(out_dir / "pre_nearfail_support_windows.html", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-json", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("threes_rl/runs/forensics/pre_nearfail_support_windows/latest"))
    parser.add_argument("--offsets", default="40,20,10,5,0")
    parser.add_argument("--final-bucket", help="Only include selected states whose final support bucket matches this value.")
    parser.add_argument("--root-origin", help="Comma-separated root origins; defaults to fresh,human.")
    parser.add_argument("--starter", default="1536")
    args = parser.parse_args()
    starter_text = args.starter.strip().lower()
    starter = None if starter_text == "none" else int(starter_text)
    payload = run_audit(
        records_json=args.records_json,
        out_dir=args.out_dir,
        offsets=parse_offsets(args.offsets),
        final_bucket=args.final_bucket,
        root_origins=parse_root_origins(args.root_origin),
        default_starter_tile=starter,
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"json={payload['json']}")
    print(f"report={payload['report']}")
    print(f"html={payload['html']}")


if __name__ == "__main__":
    main()
