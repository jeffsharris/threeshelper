"""Report support-chain milestone completion rates for continuation batches."""

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


STAGES = (
    ("raw_duplicate_768", "first_raw_duplicate_768"),
    ("raw_adjacent_768", "first_raw_adjacent_pair_768"),
    ("raw_duplicate_1536", "first_raw_duplicate_1536"),
    ("raw_adjacent_1536", "first_raw_adjacent_pair_1536"),
    ("second_3072", "second_3072_event"),
    ("reached_6144", "first_6144"),
)


def _flatten_paths(path_groups: list[list[Path]] | None) -> list[Path]:
    if not path_groups:
        return []
    return [path for group in path_groups for path in group]


def _glob_paths(patterns: list[str] | None) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns or []:
        paths.extend(Path(match) for match in glob.glob(pattern, recursive=True))
    return sorted(path for path in paths if path.is_file())


def _load_records(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    payload = json.loads(Path(path).read_text())
    records = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError(f"{path} does not contain records[]")
    return [record for record in records if isinstance(record, dict)]


def _float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: object, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _milestone(analysis: dict[str, Any], key: str) -> dict[str, Any] | None:
    if key in ("second_3072_event", "first_6144"):
        value = analysis.get(key)
    else:
        milestones = analysis.get("milestones")
        value = milestones.get(key) if isinstance(milestones, dict) else None
    return value if isinstance(value, dict) else None


def _stage_present(analysis: dict[str, Any], analysis_key: str) -> bool:
    return _milestone(analysis, analysis_key) is not None


def _stage_frames_after_first_3072(analysis: dict[str, Any], analysis_key: str) -> int | None:
    milestone = _milestone(analysis, analysis_key)
    if not isinstance(milestone, dict) or milestone.get("frames_after_first_3072") is None:
        return None
    return int(milestone["frames_after_first_3072"])


def _load_replay(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a replay object")
    return payload


def row_for_replay(path: Path) -> dict[str, Any]:
    replay = _load_replay(path)
    analysis = analyze_replay(path)
    final_features = analysis.get("final_features")
    fallback_max = (
        _int(final_features.get("max_tile_excl_starter"))
        if isinstance(final_features, dict)
        else 0
    )
    row: dict[str, Any] = {
        "replay_json": str(path),
        "start_case_id": str(replay.get("start_case_id", Path(path).stem)),
        "source_replay": str(replay.get("source_replay", "")),
        "source_seed": replay.get("source_seed"),
        "source_frame_index": replay.get("source_frame_index"),
        "seed": replay.get("seed", analysis.get("seed")),
        "score": _int(replay.get("final_score", analysis.get("final_score"))),
        "score_delta": _int(replay.get("final_score_delta")),
        "moves_delta": _int(replay.get("final_moves_delta")),
        "max_tile_excl_starter": _int(replay.get("final_max_tile_excl_starter"), fallback_max),
        "outcome": str(analysis.get("outcome")),
    }
    for stage_name, analysis_key in STAGES:
        row[stage_name] = _stage_present(analysis, analysis_key)
        row[f"{stage_name}_frames_after_first_3072"] = _stage_frames_after_first_3072(analysis, analysis_key)
    return row


def rows_for_replays(paths: Iterable[Path]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    for path in paths:
        try:
            rows.append(row_for_replay(Path(path)))
        except (OSError, json.JSONDecodeError, ValueError, TypeError):
            rejected["bad_replay"] += 1
    return rows, dict(rejected)


def _rate(rows: list[dict[str, Any]], key: str) -> float:
    return 0.0 if not rows else sum(bool(row.get(key)) for row in rows) / float(len(rows))


def _stats(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": float(mean(values)),
        "median": float(median(values)),
        "min": int(min(values)),
        "max": int(max(values)),
    }


def _stage_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    previous_count: int | None = None
    for stage_name, _analysis_key in STAGES:
        count = sum(bool(row.get(stage_name)) for row in rows)
        values = [
            int(row[f"{stage_name}_frames_after_first_3072"])
            for row in rows
            if row.get(f"{stage_name}_frames_after_first_3072") is not None
        ]
        summary[stage_name] = {
            "count": int(count),
            "rate": float(count / len(rows)) if rows else 0.0,
            "conditional_from_previous": None
            if previous_count is None
            else (float(count / previous_count) if previous_count else 0.0),
            "frames_after_first_3072": _stats(values),
        }
        previous_count = int(count)
    return summary


def _start_metadata(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for record in records:
        record_id = record.get("id")
        if record_id is None:
            continue
        metadata[str(record_id)] = {
            "reachability_prob": _float(record.get("reachability_prob")),
            "reachability_rank": _float(record.get("reachability_rank")),
            "anchor_id": record.get("anchor_id"),
            "anchor_frame_index": record.get("anchor_frame_index"),
            "frame_offset": _float(record.get("frame_offset")),
            "corner_risk": (record.get("features") or {}).get("corner_risk") if isinstance(record.get("features"), dict) else record.get("corner_risk"),
            "preview": (record.get("features") or {}).get("preview") if isinstance(record.get("features"), dict) else record.get("preview"),
            "empty_count": _float((record.get("features") or {}).get("empty_count") if isinstance(record.get("features"), dict) else record.get("empty_count")),
            "raw_count_768": _float(record.get("raw_count_768")),
            "raw_count_1536": _float(record.get("raw_count_1536")),
        }
    return metadata


def summarize_by_start(rows: list[dict[str, Any]], *, start_metadata: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("start_case_id")), []).append(row)
    start_rows: list[dict[str, Any]] = []
    for start_id, group in grouped.items():
        deltas = [int(row.get("score_delta", 0)) for row in group]
        item: dict[str, Any] = {
            "start_case_id": start_id,
            "continuations": len(group),
            "source_replay": group[0].get("source_replay"),
            "source_seed": group[0].get("source_seed"),
            "source_frame_index": group[0].get("source_frame_index"),
            "high_score_delta": int(max(deltas)) if deltas else 0,
            "mean_score_delta": float(mean(deltas)) if deltas else 0.0,
            "median_score_delta": float(median(deltas)) if deltas else 0.0,
        }
        for stage_name, _analysis_key in STAGES:
            item[f"p_{stage_name}"] = _rate(group, stage_name)
            values = [
                int(row[f"{stage_name}_frames_after_first_3072"])
                for row in group
                if row.get(f"{stage_name}_frames_after_first_3072") is not None
            ]
            item[f"{stage_name}_frames_after_first_3072"] = _stats(values)
        if start_metadata and start_id in start_metadata:
            item.update(start_metadata[start_id])
        start_rows.append(item)
    start_rows.sort(
        key=lambda row: (
            float(row.get("p_reached_6144", 0.0)),
            float(row.get("p_raw_adjacent_1536", 0.0)),
            float(row.get("p_raw_adjacent_768", 0.0)),
            int(row.get("high_score_delta", 0)),
        ),
        reverse=True,
    )
    return start_rows


def summarize(rows: list[dict[str, Any]], *, rejected: dict[str, int]) -> dict[str, Any]:
    deltas = [int(row.get("score_delta", 0)) for row in rows]
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "continuations": len(rows),
        "starts": len({str(row.get("start_case_id")) for row in rows}),
        "rejected": rejected,
        "stage_summary": _stage_summary(rows),
        "high_score": max((_int(row.get("score")) for row in rows), default=0),
        "high_score_delta": max(deltas, default=0),
        "mean_score_delta": float(mean(deltas)) if deltas else 0.0,
        "median_score_delta": float(median(deltas)) if deltas else 0.0,
        "max_tile_excl_starter_dist": dict(Counter(str(row.get("max_tile_excl_starter", "unknown")) for row in rows)),
    }


def write_html(path: Path, payload: dict[str, Any]) -> None:
    summary = payload.get("summary", {})
    stage_summary = summary.get("stage_summary", {}) if isinstance(summary, dict) else {}
    start_rows = payload.get("start_rows", [])

    def cell(value: object) -> str:
        return escape(str(value))

    def percent_or_blank(value: object) -> str:
        if value is None:
            return ""
        return f"{float(value):.1%}"

    stage_rows = []
    for stage_name, _analysis_key in STAGES:
        item = stage_summary.get(stage_name, {}) if isinstance(stage_summary, dict) else {}
        stage_rows.append(
            "<tr>"
            f"<td>{cell(stage_name)}</td>"
            f"<td>{cell(item.get('count', 0))}</td>"
            f"<td>{float(item.get('rate', 0.0)):.1%}</td>"
            f"<td>{percent_or_blank(item.get('conditional_from_previous'))}</td>"
            f"<td>{cell((item.get('frames_after_first_3072') or {}).get('median'))}</td>"
            "</tr>"
        )
    start_table_rows = []
    for row in start_rows if isinstance(start_rows, list) else []:
        start_table_rows.append(
            "<tr>"
            f"<td>{cell(row.get('source_seed'))}</td>"
            f"<td>{cell(row.get('source_frame_index'))}</td>"
            f"<td>{cell(row.get('continuations'))}</td>"
            f"<td>{float(row.get('p_raw_adjacent_768', 0.0)):.1%}</td>"
            f"<td>{float(row.get('p_raw_adjacent_1536', 0.0)):.1%}</td>"
            f"<td>{float(row.get('p_second_3072', 0.0)):.1%}</td>"
            f"<td>{float(row.get('p_reached_6144', 0.0)):.1%}</td>"
            f"<td>{cell(row.get('high_score_delta'))}</td>"
            f"<td>{cell(row.get('reachability_rank'))}</td>"
            f"<td>{float(row.get('reachability_prob', 0.0)):.3f}</td>"
            "</tr>"
        )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Support-Chain Gate Report</title>
  <style>
    :root {{ color-scheme: dark; --bg:#101318; --panel:#171d24; --line:#34404d; --text:#edf2f7; --muted:#aab6c2; --gold:#f2c14e; }}
    body {{ margin:0; background:var(--bg); color:var(--text); font-family:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ width:min(1180px, calc(100vw - 32px)); margin:0 auto; padding:24px 0 40px; }}
    h1 {{ margin:0 0 8px; font-size:24px; }}
    .muted {{ color:var(--muted); }}
    .cards {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(150px, 1fr)); gap:10px; margin:18px 0; }}
    .card, .panel {{ border:1px solid var(--line); background:var(--panel); border-radius:8px; padding:12px; }}
    .label {{ color:var(--muted); font-size:11px; text-transform:uppercase; }}
    .value {{ margin-top:4px; color:var(--gold); font-size:22px; font-weight:800; }}
    table {{ width:100%; border-collapse:collapse; font-size:12px; font-variant-numeric:tabular-nums; }}
    th, td {{ border-bottom:1px solid var(--line); padding:7px 8px; text-align:right; vertical-align:top; }}
    th:first-child, td:first-child {{ text-align:left; }}
    pre {{ white-space:pre-wrap; overflow-wrap:anywhere; color:var(--muted); }}
  </style>
</head>
<body>
  <main>
    <h1>Support-Chain Gate Report</h1>
    <p class="muted">Continuation-batch milestone completion rates for the path toward 6144.</p>
    <section class="cards">
      <div class="card"><div class="label">Continuations</div><div class="value">{cell(summary.get('continuations', 0))}</div></div>
      <div class="card"><div class="label">Starts</div><div class="value">{cell(summary.get('starts', 0))}</div></div>
      <div class="card"><div class="label">6144 Rate</div><div class="value">{float((stage_summary.get('reached_6144') or {}).get('rate', 0.0)):.1%}</div></div>
      <div class="card"><div class="label">High Delta</div><div class="value">{cell(summary.get('high_score_delta', 0))}</div></div>
    </section>
    <section class="panel">
      <table><thead><tr><th>Stage</th><th>Count</th><th>Rate</th><th>From Prev</th><th>Median +Frames</th></tr></thead><tbody>{''.join(stage_rows)}</tbody></table>
    </section>
    <section class="panel" style="margin-top:14px;">
      <table><thead><tr><th>Seed</th><th>Frame</th><th>N</th><th>Adj 768</th><th>Adj 1536</th><th>2x3072</th><th>6144</th><th>High Delta</th><th>Reach Rank</th><th>Reach Prob</th></tr></thead><tbody>{''.join(start_table_rows)}</tbody></table>
    </section>
    <section class="panel" style="margin-top:14px;"><pre>{escape(json.dumps(summary, indent=2, sort_keys=True))}</pre></section>
  </main>
</body>
</html>
"""
    path.write_text(html)


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    replay_paths = _flatten_paths(args.replay_json) + _glob_paths(args.replay_glob)
    if not replay_paths:
        raise ValueError("No replay JSONs matched")
    rows, rejected = rows_for_replays(replay_paths)
    metadata = _start_metadata(_load_records(args.start_json))
    start_rows = summarize_by_start(rows, start_metadata=metadata)
    payload = {
        "version": 1,
        "kind": "support_chain_gate_report",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_replays": [str(path) for path in replay_paths],
        "summary": summarize(rows, rejected=rejected),
        "rows": rows,
        "start_rows": start_rows,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload["json"] = str(args.out_dir / "support_chain_gate_report.json")
    payload["html"] = str(args.out_dir / "support_chain_gate_report.html")
    write_json(args.out_dir / "support_chain_gate_report.json", payload)
    write_html(args.out_dir / "support_chain_gate_report.html", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-json", type=Path, nargs="+", action="append", default=[])
    parser.add_argument("--replay-glob", action="append", default=[])
    parser.add_argument("--start-json", type=Path, help="Optional start-record JSON for reachability metadata.")
    parser.add_argument("--out-dir", type=Path, default=Path("threes_rl/runs/forensics/support_chain_gate/latest"))
    args = parser.parse_args()
    payload = run_from_args(args)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"json={payload['json']}")
    print(f"html={payload['html']}")


if __name__ == "__main__":
    main()
