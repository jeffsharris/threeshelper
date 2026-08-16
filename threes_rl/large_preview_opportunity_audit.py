"""Audit large-preview opportunities in continuation replays."""

from __future__ import annotations

import argparse
import glob
import json
import time
from collections import Counter
from html import escape
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

from threes_rl.run_artifacts import write_json
from threes_rl.support_chain_progression import analyze_replay


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


def _preview_payload(frame: dict[str, Any]) -> dict[str, Any]:
    state = frame.get("state") if isinstance(frame, dict) else None
    preview = state.get("preview") if isinstance(state, dict) else None
    return preview if isinstance(preview, dict) else {}


def _preview_label(frame: dict[str, Any]) -> str:
    preview = _preview_payload(frame)
    return str(preview.get("label", preview.get("kind", "unknown")))


def _preview_candidates(frame: dict[str, Any]) -> list[int]:
    preview = _preview_payload(frame)
    values = preview.get("candidates", [])
    if not isinstance(values, list):
        return []
    candidates: list[int] = []
    for value in values:
        try:
            candidates.append(int(value))
        except (TypeError, ValueError):
            continue
    return candidates


def _is_large_preview(frame: dict[str, Any]) -> bool:
    preview = _preview_payload(frame)
    if str(preview.get("label", "")).lower() == "large_candidates":
        return True
    if str(preview.get("kind", "")).lower() == "bonus":
        return True
    return bool(_preview_candidates(frame))


def _move_action(frame: dict[str, Any]) -> str | None:
    move = frame.get("move") if isinstance(frame, dict) else None
    if not isinstance(move, dict):
        return None
    action = move.get("action")
    return str(action) if action is not None else None


def _move_inserted_value(frame: dict[str, Any]) -> int | None:
    move = frame.get("move") if isinstance(frame, dict) else None
    if not isinstance(move, dict) or move.get("inserted_value") is None:
        return None
    return int(move["inserted_value"])


def _find_first_large(frames: list[dict[str, Any]], *, min_candidate: int = 0) -> dict[str, Any] | None:
    for idx, frame in enumerate(frames):
        candidates = _preview_candidates(frame)
        if not _is_large_preview(frame):
            continue
        if min_candidate > 0 and max(candidates, default=0) < int(min_candidate):
            continue
        return {
            "frame_position": int(idx),
            "frame_index": int(frame.get("index", idx)),
            "label": _preview_label(frame),
            "candidates": candidates,
            "max_candidate": max(candidates, default=0),
        }
    return None


def _milestone_present(analysis: dict[str, Any], name: str) -> bool:
    milestones = analysis.get("milestones")
    if not isinstance(milestones, dict):
        return False
    return milestones.get(name) is not None


def audit_replay(path: Path) -> dict[str, Any]:
    replay = json.loads(Path(path).read_text())
    if not isinstance(replay, dict):
        raise ValueError(f"{path} is not a replay object")
    frames = [frame for frame in replay.get("frames", []) if isinstance(frame, dict)]
    analysis = analyze_replay(Path(path))
    after_first = frames[1] if len(frames) > 1 else {}
    after_first_candidates = _preview_candidates(after_first)
    first_large = _find_first_large(frames[1:])
    first_large_ge_768 = _find_first_large(frames[1:], min_candidate=768)
    return {
        "source_replay": str(path),
        "seed": replay.get("seed"),
        "start_case_id": replay.get("start_case_id"),
        "source_seed": replay.get("source_seed"),
        "source_frame_index": replay.get("source_frame_index"),
        "start_score": replay.get("start_score"),
        "final_score": replay.get("final_score"),
        "score_delta": replay.get("final_score_delta"),
        "final_max_tile_excl_starter": replay.get("final_max_tile_excl_starter"),
        "reached_6144": analysis.get("first_6144") is not None,
        "reached_second_3072": analysis.get("second_3072_event") is not None,
        "raw_duplicate_1536": _milestone_present(analysis, "first_raw_duplicate_1536"),
        "raw_adjacent_1536": _milestone_present(analysis, "first_raw_adjacent_pair_1536"),
        "masked_adjacent_1536": _milestone_present(analysis, "first_masked_adjacent_pair_1536"),
        "first_action": _move_action(frames[1]) if len(frames) > 1 else None,
        "first_inserted_value": _move_inserted_value(frames[1]) if len(frames) > 1 else None,
        "preview_after_first_label": _preview_label(after_first) if after_first else None,
        "preview_after_first_candidates": after_first_candidates,
        "preview_after_first_max_candidate": max(after_first_candidates, default=0),
        "preview_after_first_large": _is_large_preview(after_first) if after_first else False,
        "preview_after_first_ge_768": max(after_first_candidates, default=0) >= 768,
        "first_large_preview": first_large,
        "first_large_ge_768": first_large_ge_768,
        "first_large_frame_after_start": None if first_large is None else int(first_large["frame_position"]),
        "first_large_ge_768_frame_after_start": None
        if first_large_ge_768 is None
        else int(first_large_ge_768["frame_position"]),
        "max_raw_count_1536": analysis.get("post_3072_max_raw_count_1536"),
        "max_raw_adjacent_pair_tile": analysis.get("post_3072_max_raw_adjacent_pair_tile"),
        "max_raw_duplicate_tile": analysis.get("post_3072_max_raw_duplicate_tile"),
    }


def _rate(count: int, total: int) -> float:
    return 0.0 if total <= 0 else float(count) / float(total)


def _bucket_summary(records: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        bucket = str(record.get(key))
        buckets.setdefault(bucket, []).append(record)
    summary: dict[str, dict[str, Any]] = {}
    for bucket, rows in sorted(buckets.items(), key=lambda item: item[0]):
        summary[bucket] = {
            "records": len(rows),
            "raw_duplicate_1536": sum(bool(row.get("raw_duplicate_1536")) for row in rows),
            "raw_adjacent_1536": sum(bool(row.get("raw_adjacent_1536")) for row in rows),
            "reached_second_3072": sum(bool(row.get("reached_second_3072")) for row in rows),
            "reached_6144": sum(bool(row.get("reached_6144")) for row in rows),
            "high_score_delta": max((int(row.get("score_delta") or 0) for row in rows), default=0),
        }
    return summary


def summarize(records: list[dict[str, Any]], *, source_count: int) -> dict[str, Any]:
    first_large_frames = [
        int(record["first_large_frame_after_start"])
        for record in records
        if record.get("first_large_frame_after_start") is not None
    ]
    first_large_ge_768_frames = [
        int(record["first_large_ge_768_frame_after_start"])
        for record in records
        if record.get("first_large_ge_768_frame_after_start") is not None
    ]
    candidate_maxes = [int(record.get("preview_after_first_max_candidate") or 0) for record in records]
    total = len(records)
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_replays": int(source_count),
        "records": total,
        "start_cases": len({str(record.get("start_case_id")) for record in records}),
        "reached_6144": sum(bool(record.get("reached_6144")) for record in records),
        "reached_second_3072": sum(bool(record.get("reached_second_3072")) for record in records),
        "raw_duplicate_1536": sum(bool(record.get("raw_duplicate_1536")) for record in records),
        "raw_adjacent_1536": sum(bool(record.get("raw_adjacent_1536")) for record in records),
        "preview_after_first_large": sum(bool(record.get("preview_after_first_large")) for record in records),
        "preview_after_first_ge_768": sum(bool(record.get("preview_after_first_ge_768")) for record in records),
        "first_large_ge_768_seen": sum(record.get("first_large_ge_768") is not None for record in records),
        "rates": {
            "reached_6144": _rate(sum(bool(record.get("reached_6144")) for record in records), total),
            "raw_duplicate_1536": _rate(sum(bool(record.get("raw_duplicate_1536")) for record in records), total),
            "preview_after_first_large": _rate(sum(bool(record.get("preview_after_first_large")) for record in records), total),
            "preview_after_first_ge_768": _rate(sum(bool(record.get("preview_after_first_ge_768")) for record in records), total),
            "first_large_ge_768_seen": _rate(sum(record.get("first_large_ge_768") is not None for record in records), total),
        },
        "first_large_frame_after_start": {
            "min": min(first_large_frames) if first_large_frames else None,
            "median": float(median(first_large_frames)) if first_large_frames else None,
            "mean": float(mean(first_large_frames)) if first_large_frames else None,
            "max": max(first_large_frames) if first_large_frames else None,
        },
        "first_large_ge_768_frame_after_start": {
            "min": min(first_large_ge_768_frames) if first_large_ge_768_frames else None,
            "median": float(median(first_large_ge_768_frames)) if first_large_ge_768_frames else None,
            "mean": float(mean(first_large_ge_768_frames)) if first_large_ge_768_frames else None,
            "max": max(first_large_ge_768_frames) if first_large_ge_768_frames else None,
        },
        "preview_after_first_label": dict(Counter(str(record.get("preview_after_first_label")) for record in records)),
        "preview_after_first_max_candidate": dict(Counter(str(value) for value in candidate_maxes)),
        "by_preview_after_first_label": _bucket_summary(records, "preview_after_first_label"),
        "by_preview_after_first_max_candidate": _bucket_summary(records, "preview_after_first_max_candidate"),
    }


def write_html(path: Path, payload: dict[str, Any]) -> None:
    summary = payload.get("summary", {})
    records = payload.get("records", [])

    def cell(value: object) -> str:
        return escape(str(value))

    rows = []
    for record in records[:400] if isinstance(records, list) else []:
        if not isinstance(record, dict):
            continue
        rows.append(
            "<tr>"
            f"<td>{cell(record.get('seed'))}</td>"
            f"<td>{cell(record.get('start_case_id'))}</td>"
            f"<td>{cell(record.get('score_delta'))}</td>"
            f"<td>{cell(record.get('final_max_tile_excl_starter'))}</td>"
            f"<td>{cell(record.get('preview_after_first_label'))}</td>"
            f"<td>{cell(record.get('preview_after_first_candidates'))}</td>"
            f"<td>{cell(record.get('first_large_ge_768_frame_after_start'))}</td>"
            f"<td>{cell(record.get('raw_duplicate_1536'))}</td>"
            f"<td>{cell(record.get('raw_adjacent_1536'))}</td>"
            f"<td>{cell(record.get('reached_6144'))}</td>"
            "</tr>"
        )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Large Preview Opportunity Audit</title>
  <style>
    :root {{ color-scheme: dark; --bg: #101318; --panel: #171d24; --line: #34404d; --text: #edf2f7; --muted: #aab6c2; --gold: #f2c14e; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--text); font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ width: min(1180px, calc(100vw - 32px)); margin: 0 auto; padding: 24px 0 40px; }}
    h1 {{ margin: 0 0 8px; font-size: 24px; letter-spacing: 0; }}
    .muted {{ color: var(--muted); }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 18px 0; }}
    .card {{ border: 1px solid var(--line); background: var(--panel); border-radius: 8px; padding: 12px; }}
    .label {{ color: var(--muted); font-size: 11px; text-transform: uppercase; }}
    .value {{ margin-top: 4px; color: var(--gold); font-size: 22px; font-weight: 800; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 6px 8px; text-align: right; vertical-align: top; }}
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
    pre {{ white-space: pre-wrap; overflow-wrap: anywhere; color: var(--muted); }}
  </style>
</head>
<body>
  <main>
    <h1>Large Preview Opportunity Audit</h1>
    <p class="muted">Continuation replay audit for large-preview timing and support-ladder outcomes.</p>
    <section class="cards">
      <div class="card"><div class="label">Records</div><div class="value">{cell(summary.get('records', 0))}</div></div>
      <div class="card"><div class="label">Starts</div><div class="value">{cell(summary.get('start_cases', 0))}</div></div>
      <div class="card"><div class="label">First Large >=768</div><div class="value">{cell(summary.get('first_large_ge_768_seen', 0))}</div></div>
      <div class="card"><div class="label">Raw Dup 1536</div><div class="value">{cell(summary.get('raw_duplicate_1536', 0))}</div></div>
      <div class="card"><div class="label">6144</div><div class="value">{cell(summary.get('reached_6144', 0))}</div></div>
    </section>
    <pre>{escape(json.dumps(summary, indent=2, sort_keys=True))}</pre>
    <table>
      <thead><tr><th>Seed</th><th>Start</th><th>Delta</th><th>Max</th><th>Preview After First</th><th>Candidates</th><th>First Large >=768</th><th>Raw Dup 1536</th><th>Raw Adj 1536</th><th>6144</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </main>
</body>
</html>
"""
    path.write_text(html)


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    paths = _flatten_paths(args.replay_json) + _glob_paths(args.replay_glob)
    records = [audit_replay(path) for path in paths]
    summary = summarize(records, source_count=len(paths))
    payload = {
        "version": 1,
        "kind": "large_preview_opportunity_audit",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "summary": summary,
        "records": records,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload["json"] = str(args.out_dir / "large_preview_opportunity_audit.json")
    payload["records_json"] = str(args.out_dir / "records.json")
    payload["html"] = str(args.out_dir / "large_preview_opportunity_audit.html")
    write_json(args.out_dir / "large_preview_opportunity_audit.json", payload)
    write_json(args.out_dir / "records.json", records)
    write_html(args.out_dir / "large_preview_opportunity_audit.html", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-json", type=Path, nargs="+", action="append")
    parser.add_argument("--replay-glob", action="append")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = run_from_args(args)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"json={payload['json']}")
    print(f"records={payload['records_json']}")
    print(f"html={payload['html']}")


if __name__ == "__main__":
    main()
