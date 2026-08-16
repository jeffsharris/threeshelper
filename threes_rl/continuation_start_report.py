"""Summarize continuation outcomes by start state."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from html import escape
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

import numpy as np

from threes_rl.run_artifacts import write_json


NUMERIC_FEATURES = (
    "reachability_prob",
    "reachability_rank",
    "frame_offset",
    "score_minus_starter",
    "empty_count",
    "legal_count",
    "top_left",
    "top_left_is_max",
    "raw_count_768",
    "raw_count_1536",
    "raw_count_3072",
    "raw_has_adjacent_768",
    "raw_has_adjacent_1536",
    "safe_smalls_until_large_possible",
)


def _load_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text())
    records = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError(f"{path} does not contain records[]")
    return [record for record in records if isinstance(record, dict)]


def load_records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        records.extend(_load_records(Path(path)))
    return records


def _float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _board(record: dict[str, Any]) -> np.ndarray:
    state = record.get("state")
    if isinstance(state, dict) and isinstance(state.get("board"), list):
        return np.asarray(state["board"], dtype=np.int32)
    return np.zeros((4, 4), dtype=np.int32)


def _feature(record: dict[str, Any], name: str, default: object = None) -> object:
    features = record.get("features")
    if isinstance(features, dict) and name in features:
        return features.get(name)
    return record.get(name, default)


def _start_feature_row(record: dict[str, Any]) -> dict[str, Any]:
    board = _board(record)
    top_left = int(board[0, 0]) if board.shape == (4, 4) else 0
    board_max = int(board.max(initial=0))
    return {
        "start_case_id": str(record.get("id")),
        "source_replay": str(record.get("source_replay", "")),
        "source_seed": record.get("source_seed", record.get("seed")),
        "source_frame_index": record.get("source_frame_index"),
        "reachability_prob": _float(record.get("reachability_prob")),
        "reachability_rank": _float(record.get("reachability_rank")),
        "anchor_id": record.get("anchor_id"),
        "anchor_frame_index": record.get("anchor_frame_index"),
        "frame_offset": _float(record.get("frame_offset")),
        "synthetic_kind": record.get("synthetic_kind"),
        "corner_risk": str(_feature(record, "corner_risk", "unknown")),
        "preview": str(_feature(record, "preview", "unknown")),
        "score_minus_starter": _float(_feature(record, "score_minus_starter")),
        "empty_count": _float(_feature(record, "empty_count")),
        "legal_count": _float(record.get("legal_count", _feature(record, "legal_count"))),
        "top_left": float(top_left),
        "top_left_is_max": 1.0 if top_left == board_max and board_max > 0 else 0.0,
        "raw_count_768": _float(record.get("raw_count_768", _feature(record, "raw_count_768", np.count_nonzero(board == 768)))),
        "raw_count_1536": _float(
            record.get("raw_count_1536", _feature(record, "raw_count_1536", np.count_nonzero(board == 1536)))
        ),
        "raw_count_3072": _float(
            record.get("raw_count_3072", _feature(record, "raw_count_3072", np.count_nonzero(board == 3072)))
        ),
        "raw_has_adjacent_768": 1.0 if bool(record.get("raw_has_adjacent_768", _feature(record, "raw_has_adjacent_768", False))) else 0.0,
        "raw_has_adjacent_1536": 1.0 if bool(record.get("raw_has_adjacent_1536", _feature(record, "raw_has_adjacent_1536", False))) else 0.0,
        "safe_smalls_until_large_possible": _float(_feature(record, "safe_smalls_until_large_possible")),
    }


def build_rows(
    starts: list[dict[str, Any]],
    continuations: list[dict[str, Any]],
    *,
    target_tile: int,
) -> list[dict[str, Any]]:
    by_start: dict[str, list[dict[str, Any]]] = {}
    for record in continuations:
        start_id = record.get("start_case_id")
        if start_id is None:
            continue
        by_start.setdefault(str(start_id), []).append(record)

    rows: list[dict[str, Any]] = []
    for start in starts:
        start_id = str(start.get("id"))
        records = by_start.get(start_id, [])
        deltas = [int(record.get("score_delta", 0)) for record in records]
        hits = [record for record in records if int(record.get("max_tile_excl_starter", 0)) >= int(target_tile)]
        row = _start_feature_row(start)
        row.update(
            {
                "continuations": len(records),
                "hits": len(hits),
                "hit_rate": float(len(hits) / len(records)) if records else 0.0,
                "mean_score_delta": float(mean(deltas)) if deltas else 0.0,
                "median_score_delta": float(median(deltas)) if deltas else 0.0,
                "high_score_delta": int(max(deltas)) if deltas else 0,
                "max_tile_excl_starter_dist": dict(
                    Counter(str(record.get("max_tile_excl_starter", "unknown")) for record in records)
                ),
            }
        )
        rows.append(row)
    rows.sort(key=lambda row: (float(row["hit_rate"]), float(row["median_score_delta"])), reverse=True)
    return rows


def summarize(rows: list[dict[str, Any]], *, target_tile: int) -> dict[str, Any]:
    continuations = sum(int(row.get("continuations", 0)) for row in rows)
    hits = sum(int(row.get("hits", 0)) for row in rows)
    by_kind: dict[str, dict[str, Any]] = {}
    for row in rows:
        kind = str(row.get("synthetic_kind", "unknown"))
        item = by_kind.setdefault(
            kind,
            {
                "starts": 0,
                "continuations": 0,
                "hits": 0,
                "stable_success_starts": 0,
                "stable_failure_starts": 0,
                "mixed_starts": 0,
                "high_score_delta": 0,
            },
        )
        item["starts"] += 1
        item["continuations"] += int(row.get("continuations", 0))
        item["hits"] += int(row.get("hits", 0))
        if int(row.get("continuations", 0)) <= 0:
            item["high_score_delta"] = max(int(item["high_score_delta"]), int(row.get("high_score_delta", 0)))
            continue
        hit_rate = float(row.get("hit_rate", 0.0))
        if hit_rate >= 1.0:
            item["stable_success_starts"] += 1
        elif hit_rate <= 0.0:
            item["stable_failure_starts"] += 1
        else:
            item["mixed_starts"] += 1
        item["high_score_delta"] = max(int(item["high_score_delta"]), int(row.get("high_score_delta", 0)))
    for item in by_kind.values():
        item["hit_rate"] = (
            float(int(item["hits"]) / int(item["continuations"])) if int(item["continuations"]) else 0.0
        )
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "target_tile": int(target_tile),
        "starts": len(rows),
        "continuations": int(continuations),
        "hits": int(hits),
        "hit_rate": float(hits / continuations) if continuations else 0.0,
        "stable_success_starts": sum(1 for row in rows if row.get("continuations") and float(row.get("hit_rate", 0.0)) >= 1.0),
        "stable_failure_starts": sum(1 for row in rows if row.get("continuations") and float(row.get("hit_rate", 0.0)) <= 0.0),
        "mixed_starts": sum(1 for row in rows if 0.0 < float(row.get("hit_rate", 0.0)) < 1.0),
        "by_synthetic_kind": dict(Counter(str(row.get("synthetic_kind", "unknown")) for row in rows)),
        "outcome_by_synthetic_kind": by_kind,
        "by_corner_risk": dict(Counter(str(row.get("corner_risk", "unknown")) for row in rows)),
        "high_score_delta": int(max((int(row.get("high_score_delta", 0)) for row in rows), default=0)),
    }


def feature_split(rows: list[dict[str, Any]], *, threshold: float) -> dict[str, Any]:
    sampled = [row for row in rows if int(row.get("continuations", 0)) > 0]
    high = [row for row in sampled if float(row.get("hit_rate", 0.0)) >= float(threshold)]
    low = [row for row in sampled if float(row.get("hit_rate", 0.0)) < float(threshold)]
    numeric: list[dict[str, Any]] = []
    for feature in NUMERIC_FEATURES:
        hi_values = [_float(row.get(feature)) for row in high]
        lo_values = [_float(row.get(feature)) for row in low]
        numeric.append(
            {
                "feature": feature,
                "high_mean": float(mean(hi_values)) if hi_values else 0.0,
                "low_mean": float(mean(lo_values)) if lo_values else 0.0,
                "high_counts": dict(Counter(str(value) for value in hi_values)),
                "low_counts": dict(Counter(str(value) for value in lo_values)),
            }
        )
    return {
        "threshold": float(threshold),
        "high_starts": len(high),
        "low_starts": len(low),
        "numeric": numeric,
    }


def _sampled_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if int(row.get("continuations", 0)) > 0]


def _outcome_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    continuations = sum(int(row.get("continuations", 0)) for row in rows)
    hits = sum(int(row.get("hits", 0)) for row in rows)
    hit_starts = sum(1 for row in rows if int(row.get("hits", 0)) > 0)
    stable_success = sum(
        1
        for row in rows
        if int(row.get("continuations", 0)) > 0
        and int(row.get("hits", 0)) == int(row.get("continuations", 0))
    )
    stable_failure = sum(
        1
        for row in rows
        if int(row.get("continuations", 0)) > 0 and int(row.get("hits", 0)) == 0
    )
    mixed = sum(
        1
        for row in rows
        if 0 < int(row.get("hits", 0)) < int(row.get("continuations", 0))
    )
    probs = [_float(row.get("reachability_prob")) for row in rows]
    ranks = [_float(row.get("reachability_rank")) for row in rows if _float(row.get("reachability_rank")) > 0]
    medians = [_float(row.get("median_score_delta")) for row in rows]
    highs = [int(row.get("high_score_delta", 0)) for row in rows]
    hit_rates = [_float(row.get("hit_rate")) for row in rows]
    return {
        "starts": len(rows),
        "continuations": int(continuations),
        "hits": int(hits),
        "hit_rate": float(hits / continuations) if continuations else 0.0,
        "hit_starts": int(hit_starts),
        "hit_start_rate": float(hit_starts / len(rows)) if rows else 0.0,
        "stable_success_starts": int(stable_success),
        "mixed_starts": int(mixed),
        "stable_failure_starts": int(stable_failure),
        "mean_reachability_prob": float(mean(probs)) if probs else 0.0,
        "median_reachability_prob": float(median(probs)) if probs else 0.0,
        "min_reachability_prob": float(min(probs)) if probs else 0.0,
        "max_reachability_prob": float(max(probs)) if probs else 0.0,
        "min_reachability_rank": float(min(ranks)) if ranks else 0.0,
        "max_reachability_rank": float(max(ranks)) if ranks else 0.0,
        "median_hit_rate": float(median(hit_rates)) if hit_rates else 0.0,
        "mean_median_score_delta": float(mean(medians)) if medians else 0.0,
        "median_score_delta": float(median(medians)) if medians else 0.0,
        "high_score_delta": int(max(highs)) if highs else 0,
    }


def _parse_probability_edges(text: str | None) -> list[float]:
    if not text:
        return [0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.97, 1.000001]
    edges = sorted({float(part.strip()) for part in text.split(",") if part.strip()})
    if len(edges) < 2:
        raise ValueError("--probability-bands must include at least two comma-separated edges")
    if edges[0] > 0.0:
        edges.insert(0, 0.0)
    if edges[-1] < 1.0:
        edges.append(1.000001)
    else:
        edges[-1] = max(edges[-1], 1.000001)
    return edges


def probability_band_summary(rows: list[dict[str, Any]], *, edges: list[float] | None = None) -> list[dict[str, Any]]:
    sampled = _sampled_rows(rows)
    band_edges = edges or _parse_probability_edges(None)
    bands: list[dict[str, Any]] = []
    for lo, hi in zip(band_edges[:-1], band_edges[1:]):
        bucket = [
            row
            for row in sampled
            if float(lo) <= _float(row.get("reachability_prob")) < float(hi)
        ]
        item = {
            "probability_min": float(lo),
            "probability_max": 1.0 if hi > 1.0 else float(hi),
            "label": f"[{float(lo):.3f}, {1.0 if hi > 1.0 else float(hi):.3f})",
            **_outcome_summary(bucket),
        }
        bands.append(item)
    bands.reverse()
    return bands


def rank_bucket_summary(rows: list[dict[str, Any]], *, bucket_size: int = 12) -> list[dict[str, Any]]:
    size = max(1, int(bucket_size))

    def rank_sort_key(row: dict[str, Any]) -> tuple[float, float]:
        rank = _float(row.get("reachability_rank"))
        return (rank if rank > 0 else 1e12, -_float(row.get("reachability_prob")))

    sampled = sorted(
        _sampled_rows(rows),
        key=rank_sort_key,
    )
    buckets: list[dict[str, Any]] = []
    for idx in range(0, len(sampled), size):
        bucket = sampled[idx : idx + size]
        if not bucket:
            continue
        ranks = [_float(row.get("reachability_rank")) for row in bucket if _float(row.get("reachability_rank")) > 0]
        item = {
            "bucket": len(buckets) + 1,
            "rank_start": int(min(ranks)) if ranks else idx + 1,
            "rank_end": int(max(ranks)) if ranks else idx + len(bucket),
            **_outcome_summary(bucket),
        }
        buckets.append(item)
    return buckets


def write_html(path: Path, payload: dict[str, Any]) -> None:
    summary = payload.get("summary", {})
    rows = payload.get("rows", [])
    split = payload.get("feature_split", {})
    probability_bands = payload.get("probability_bands", [])
    rank_buckets = payload.get("rank_buckets", [])

    def cell(value: object) -> str:
        return escape(str(value))

    table_rows = []
    for row in rows if isinstance(rows, list) else []:
        table_rows.append(
            "<tr>"
            f"<td>{cell(row.get('start_case_id'))}</td>"
            f"<td>{cell(row.get('source_seed'))}</td>"
            f"<td>{cell(row.get('source_frame_index'))}</td>"
            f"<td>{float(row.get('reachability_prob', 0.0)):.3f}</td>"
            f"<td>{float(row.get('hit_rate', 0.0)):.3f}</td>"
            f"<td>{cell(row.get('hits'))}/{cell(row.get('continuations'))}</td>"
            f"<td>{float(row.get('median_score_delta', 0.0)):.0f}</td>"
            f"<td>{cell(row.get('top_left'))}</td>"
            f"<td>{cell(row.get('empty_count'))}</td>"
            f"<td>{cell(row.get('raw_count_768'))}</td>"
            f"<td>{cell(row.get('raw_count_1536'))}</td>"
            "</tr>"
        )
    band_rows = []
    for row in probability_bands if isinstance(probability_bands, list) else []:
        band_rows.append(
            "<tr>"
            f"<td>{cell(row.get('label'))}</td>"
            f"<td>{cell(row.get('starts'))}</td>"
            f"<td>{cell(row.get('hits'))}/{cell(row.get('continuations'))}</td>"
            f"<td>{float(row.get('hit_rate', 0.0)):.1%}</td>"
            f"<td>{cell(row.get('stable_success_starts'))}</td>"
            f"<td>{cell(row.get('mixed_starts'))}</td>"
            f"<td>{cell(row.get('stable_failure_starts'))}</td>"
            f"<td>{float(row.get('median_score_delta', 0.0)):.0f}</td>"
            "</tr>"
        )
    rank_rows = []
    for row in rank_buckets if isinstance(rank_buckets, list) else []:
        rank_rows.append(
            "<tr>"
            f"<td>{cell(row.get('rank_start'))}-{cell(row.get('rank_end'))}</td>"
            f"<td>{cell(row.get('starts'))}</td>"
            f"<td>{cell(row.get('hits'))}/{cell(row.get('continuations'))}</td>"
            f"<td>{float(row.get('hit_rate', 0.0)):.1%}</td>"
            f"<td>{cell(row.get('stable_success_starts'))}</td>"
            f"<td>{cell(row.get('mixed_starts'))}</td>"
            f"<td>{cell(row.get('stable_failure_starts'))}</td>"
            f"<td>{float(row.get('median_score_delta', 0.0)):.0f}</td>"
            "</tr>"
        )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Continuation Start Report</title>
  <style>
    :root {{ color-scheme: dark; --bg: #101318; --panel: #171d24; --line: #34404d; --text: #edf2f7; --muted: #aab6c2; --gold: #f2c14e; }}
    body {{ margin: 0; background: var(--bg); color: var(--text); font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ width: min(1180px, calc(100vw - 32px)); margin: 0 auto; padding: 24px 0 40px; }}
    h1 {{ margin: 0 0 8px; font-size: 24px; }}
    .muted {{ color: var(--muted); }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 18px 0; }}
    .card, .panel {{ border: 1px solid var(--line); background: var(--panel); border-radius: 8px; padding: 12px; }}
    .label {{ color: var(--muted); font-size: 11px; text-transform: uppercase; }}
    .value {{ margin-top: 4px; color: var(--gold); font-size: 22px; font-weight: 800; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; font-variant-numeric: tabular-nums; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 7px 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; max-width: 360px; overflow-wrap: anywhere; }}
    pre {{ white-space: pre-wrap; overflow-wrap: anywhere; color: var(--muted); }}
  </style>
</head>
<body>
  <main>
    <h1>Continuation Start Report</h1>
    <p class="muted">Continuation outcomes grouped by exact start state.</p>
    <section class="cards">
      <div class="card"><div class="label">Starts</div><div class="value">{cell(summary.get('starts', 0))}</div></div>
      <div class="card"><div class="label">Continuations</div><div class="value">{cell(summary.get('continuations', 0))}</div></div>
      <div class="card"><div class="label">Hit Rate</div><div class="value">{float(summary.get('hit_rate', 0.0)):.0%}</div></div>
      <div class="card"><div class="label">Stable Success</div><div class="value">{cell(summary.get('stable_success_starts', 0))}</div></div>
    </section>
    <section class="panel">
      <table><thead><tr><th>Prob Band</th><th>Starts</th><th>Hits</th><th>Hit Rate</th><th>Stable</th><th>Mixed</th><th>Dead</th><th>Median Delta</th></tr></thead><tbody>{''.join(band_rows)}</tbody></table>
    </section>
    <section class="panel" style="margin-top:14px;">
      <table><thead><tr><th>Rank Band</th><th>Starts</th><th>Hits</th><th>Hit Rate</th><th>Stable</th><th>Mixed</th><th>Dead</th><th>Median Delta</th></tr></thead><tbody>{''.join(rank_rows)}</tbody></table>
    </section>
    <section class="panel" style="margin-top:14px;">
      <table><thead><tr><th>Start</th><th>Seed</th><th>Frame</th><th>Reach Prob</th><th>Hit Rate</th><th>Hits</th><th>Median Delta</th><th>Top Left</th><th>Empty</th><th>Raw 768</th><th>Raw 1536</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table>
    </section>
    <section class="panel" style="margin-top:14px;"><pre>{escape(json.dumps(split, indent=2, sort_keys=True))}</pre></section>
  </main>
</body>
</html>
"""
    path.write_text(html)


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    starts = load_records([path for group in args.start_json for path in group])
    continuations = load_records([path for group in args.continuation_json for path in group])
    rows = build_rows(starts, continuations, target_tile=args.target_tile)
    probability_edges = _parse_probability_edges(args.probability_bands)
    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "start_json": [str(path) for group in args.start_json for path in group],
        "continuation_json": [str(path) for group in args.continuation_json for path in group],
        "target_tile": int(args.target_tile),
        "summary": summarize(rows, target_tile=args.target_tile),
        "feature_split": feature_split(rows, threshold=args.hit_rate_threshold),
        "probability_edges": probability_edges,
        "probability_bands": probability_band_summary(rows, edges=probability_edges),
        "rank_buckets": rank_bucket_summary(rows, bucket_size=args.rank_bucket_size),
        "rows": rows,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload["json"] = str(args.out_dir / "continuation_start_report.json")
    payload["html"] = str(args.out_dir / "continuation_start_report.html")
    write_json(args.out_dir / "continuation_start_report.json", payload)
    write_html(args.out_dir / "continuation_start_report.html", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-json", type=Path, nargs="+", action="append", required=True)
    parser.add_argument("--continuation-json", type=Path, nargs="+", action="append", required=True)
    parser.add_argument("--target-tile", type=int, default=6144)
    parser.add_argument("--hit-rate-threshold", type=float, default=0.5)
    parser.add_argument(
        "--probability-bands",
        help="Comma-separated probability edges for band calibration, e.g. 0,.5,.75,.9,.97,1.",
    )
    parser.add_argument("--rank-bucket-size", type=int, default=12)
    parser.add_argument("--out-dir", type=Path, default=Path("threes_rl/runs/forensics/continuation_start_report/latest"))
    args = parser.parse_args()
    payload = run_from_args(args)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"json={payload['json']}")
    print(f"html={payload['html']}")


if __name__ == "__main__":
    main()
