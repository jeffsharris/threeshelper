"""Report scorer calibration against continuation-screen outcomes."""

from __future__ import annotations

import argparse
import json
import time
from html import escape
from pathlib import Path
from statistics import mean, median
from typing import Any

from threes_rl.run_artifacts import write_json


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


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


def _auc(labels: list[int], scores: list[float]) -> float:
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return 0.0
    pairs = sorted(zip(scores, labels), key=lambda item: item[0])
    rank_sum = 0.0
    idx = 0
    while idx < len(pairs):
        end = idx + 1
        while end < len(pairs) and pairs[end][0] == pairs[idx][0]:
            end += 1
        avg_rank = (idx + 1 + end) / 2.0
        for pair_idx in range(idx, end):
            if pairs[pair_idx][1] == 1:
                rank_sum += avg_rank
        idx = end
    return float((rank_sum - pos * (pos + 1) / 2.0) / (pos * neg))


def _top_bucket(rows: list[dict[str, Any]], k: int) -> dict[str, Any]:
    bucket = rows[: min(int(k), len(rows))]
    continuations = sum(_int(row.get("continuations")) for row in bucket)
    hits = sum(_int(row.get("hits")) for row in bucket)
    return {
        "k": int(k),
        "starts": len(bucket),
        "continuations": int(continuations),
        "hits": int(hits),
        "hit_starts": sum(1 for row in bucket if _int(row.get("hits")) > 0),
        "hit_rate": float(hits / continuations) if continuations else 0.0,
        "hit_start_rate": (
            sum(1 for row in bucket if _int(row.get("hits")) > 0) / len(bucket)
            if bucket
            else 0.0
        ),
    }


def _score_context(score_json: Path | None) -> dict[str, Any] | None:
    if score_json is None:
        return None
    payload = _load_json(score_json)
    return {
        "score_json": str(score_json),
        "raw_candidates": payload.get("raw_candidates"),
        "filtered_candidates": payload.get("filtered_candidates"),
        "unique_candidates": payload.get("unique_candidates"),
        "summary": payload.get("summary"),
        "top_selection": payload.get("top_selection"),
    }


def rows_from_gate(gate_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in gate_payload.get("start_rows", []):
        if not isinstance(raw, dict):
            continue
        continuations = _int(raw.get("continuations"))
        p6144 = _float(raw.get("p_reached_6144"))
        hits = int(round(p6144 * continuations))
        rows.append(
            {
                "start_case_id": str(raw.get("start_case_id")),
                "source_replay": raw.get("source_replay"),
                "source_seed": raw.get("source_seed"),
                "source_frame_index": raw.get("source_frame_index"),
                "reachability_rank": _float(raw.get("reachability_rank"), 1e9),
                "reachability_prob": _float(raw.get("reachability_prob")),
                "continuations": continuations,
                "hits": hits,
                "p_reached_6144": p6144,
                "p_raw_duplicate_768": _float(raw.get("p_raw_duplicate_768")),
                "p_raw_adjacent_768": _float(raw.get("p_raw_adjacent_768")),
                "p_raw_duplicate_1536": _float(raw.get("p_raw_duplicate_1536")),
                "p_raw_adjacent_1536": _float(raw.get("p_raw_adjacent_1536")),
                "p_second_3072": _float(raw.get("p_second_3072")),
                "high_score_delta": _int(raw.get("high_score_delta")),
                "mean_score_delta": _float(raw.get("mean_score_delta")),
                "median_score_delta": _float(raw.get("median_score_delta")),
            }
        )
    rows.sort(key=lambda row: (float(row.get("reachability_rank", 1e9)), -float(row.get("reachability_prob", 0.0))))
    return rows


def summarize(rows: list[dict[str, Any]], *, target_tile: int = 6144) -> dict[str, Any]:
    continuations = sum(_int(row.get("continuations")) for row in rows)
    hits = sum(_int(row.get("hits")) for row in rows)
    hit_rows = [row for row in rows if _int(row.get("hits")) > 0]
    miss_rows = [row for row in rows if _int(row.get("hits")) <= 0]
    labels = [1 if _int(row.get("hits")) > 0 else 0 for row in rows]
    probs = [_float(row.get("reachability_prob")) for row in rows]
    ranks_hit = [_float(row.get("reachability_rank")) for row in hit_rows]
    ranks_miss = [_float(row.get("reachability_rank")) for row in miss_rows]
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "target_tile": int(target_tile),
        "starts": len(rows),
        "continuations": int(continuations),
        "hits": int(hits),
        "hit_rate": float(hits / continuations) if continuations else 0.0,
        "hit_starts": len(hit_rows),
        "hit_start_rate": float(len(hit_rows) / len(rows)) if rows else 0.0,
        "stable_failure_starts": len(miss_rows),
        "stable_success_starts": sum(
            1
            for row in rows
            if _int(row.get("continuations")) > 0 and _int(row.get("hits")) == _int(row.get("continuations"))
        ),
        "start_auc_by_prob": _auc(labels, probs),
        "mean_prob_hit_starts": float(mean(_float(row.get("reachability_prob")) for row in hit_rows)) if hit_rows else 0.0,
        "mean_prob_miss_starts": float(mean(_float(row.get("reachability_prob")) for row in miss_rows)) if miss_rows else 0.0,
        "median_rank_hit_starts": float(median(ranks_hit)) if ranks_hit else 0.0,
        "median_rank_miss_starts": float(median(ranks_miss)) if ranks_miss else 0.0,
        "top_buckets": [_top_bucket(rows, k) for k in (3, 5, 10, 12, 24, 36) if rows],
        "dead_top_starts": [row for row in rows if _int(row.get("hits")) == 0][:8],
        "productive_starts": sorted(
            hit_rows,
            key=lambda row: (
                _float(row.get("p_reached_6144")),
                _int(row.get("high_score_delta")),
                -_float(row.get("reachability_rank")),
            ),
            reverse=True,
        )[:8],
    }


def write_html(path: Path, payload: dict[str, Any]) -> None:
    summary = payload.get("summary", {})
    rows = payload.get("rows", [])

    def cell(value: object) -> str:
        return escape(str(value))

    table_rows = []
    for row in rows if isinstance(rows, list) else []:
        table_rows.append(
            "<tr>"
            f"<td>{float(row.get('reachability_rank', 0.0)):.0f}</td>"
            f"<td>{float(row.get('reachability_prob', 0.0)):.3f}</td>"
            f"<td>{cell(row.get('source_seed'))}</td>"
            f"<td>{cell(row.get('source_frame_index'))}</td>"
            f"<td>{cell(row.get('continuations'))}</td>"
            f"<td>{cell(row.get('hits'))}</td>"
            f"<td>{float(row.get('p_reached_6144', 0.0)):.0%}</td>"
            f"<td>{float(row.get('p_raw_adjacent_1536', 0.0)):.0%}</td>"
            f"<td>{cell(row.get('high_score_delta'))}</td>"
            f"<td>{cell(row.get('start_case_id'))}</td>"
            "</tr>"
        )

    bucket_rows = []
    for bucket in summary.get("top_buckets", []) if isinstance(summary, dict) else []:
        bucket_rows.append(
            "<tr>"
            f"<td>{cell(bucket.get('k'))}</td>"
            f"<td>{cell(bucket.get('starts'))}</td>"
            f"<td>{cell(bucket.get('hits'))}</td>"
            f"<td>{float(bucket.get('hit_rate', 0.0)):.1%}</td>"
            f"<td>{cell(bucket.get('hit_starts'))}</td>"
            f"<td>{float(bucket.get('hit_start_rate', 0.0)):.1%}</td>"
            "</tr>"
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Reachability Screen Report</title>
  <style>
    :root {{ color-scheme: dark; --bg:#101318; --panel:#171d24; --line:#34404d; --text:#edf2f7; --muted:#aab6c2; --gold:#f2c14e; }}
    body {{ margin:0; background:var(--bg); color:var(--text); font-family:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ width:min(1220px, calc(100vw - 32px)); margin:0 auto; padding:24px 0 40px; }}
    h1 {{ margin:0 0 8px; font-size:24px; }}
    .muted {{ color:var(--muted); }}
    .cards {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(150px, 1fr)); gap:10px; margin:18px 0; }}
    .card, .panel {{ border:1px solid var(--line); background:var(--panel); border-radius:8px; padding:12px; }}
    .label {{ color:var(--muted); font-size:11px; text-transform:uppercase; }}
    .value {{ margin-top:4px; color:var(--gold); font-size:22px; font-weight:800; }}
    table {{ width:100%; border-collapse:collapse; font-size:12px; font-variant-numeric:tabular-nums; }}
    th, td {{ border-bottom:1px solid var(--line); padding:7px 8px; text-align:right; vertical-align:top; }}
    th:last-child, td:last-child {{ text-align:left; max-width:360px; overflow-wrap:anywhere; }}
    pre {{ white-space:pre-wrap; overflow-wrap:anywhere; color:var(--muted); }}
  </style>
</head>
<body>
  <main>
    <h1>Reachability Screen Report</h1>
    <p class="muted">Scorer ranking compared with same-start continuation outcomes.</p>
    <section class="cards">
      <div class="card"><div class="label">Starts</div><div class="value">{cell(summary.get('starts', 0))}</div></div>
      <div class="card"><div class="label">6144 Rate</div><div class="value">{float(summary.get('hit_rate', 0.0)):.1%}</div></div>
      <div class="card"><div class="label">Hit Starts</div><div class="value">{cell(summary.get('hit_starts', 0))}</div></div>
      <div class="card"><div class="label">AUC</div><div class="value">{float(summary.get('start_auc_by_prob', 0.0)):.3f}</div></div>
    </section>
    <section class="panel">
      <table><thead><tr><th>Top K</th><th>Starts</th><th>Hits</th><th>Hit Rate</th><th>Hit Starts</th><th>Hit Start Rate</th></tr></thead><tbody>{''.join(bucket_rows)}</tbody></table>
    </section>
    <section class="panel" style="margin-top:14px;">
      <table><thead><tr><th>Rank</th><th>Prob</th><th>Seed</th><th>Frame</th><th>N</th><th>Hits</th><th>6144</th><th>Adj 1536</th><th>High Delta</th><th>Start</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table>
    </section>
    <section class="panel" style="margin-top:14px;"><pre>{escape(json.dumps(summary, indent=2, sort_keys=True))}</pre></section>
  </main>
</body>
</html>
"""
    path.write_text(html)


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    gate_payload = _load_json(args.gate_json)
    rows = rows_from_gate(gate_payload)
    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "gate_json": str(args.gate_json),
        "score_context": _score_context(args.score_json),
        "summary": summarize(rows, target_tile=args.target_tile),
        "rows": rows,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload["json"] = str(args.out_dir / "reachability_screen_report.json")
    payload["html"] = str(args.out_dir / "reachability_screen_report.html")
    write_json(args.out_dir / "reachability_screen_report.json", payload)
    write_html(args.out_dir / "reachability_screen_report.html", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-json", type=Path, required=True)
    parser.add_argument("--score-json", type=Path)
    parser.add_argument("--target-tile", type=int, default=6144)
    parser.add_argument("--out-dir", type=Path, default=Path("threes_rl/runs/forensics/reachability_screen/latest"))
    args = parser.parse_args()
    payload = run_from_args(args)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"json={payload['json']}")
    print(f"html={payload['html']}")


if __name__ == "__main__":
    main()
