"""Sample root-capped transition-window records at fixed move offsets."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from html import escape
from pathlib import Path
from typing import Any, Iterable

from threes_rl.run_artifacts import write_json


def _flatten_paths(path_groups: list[list[Path]] | None) -> list[Path]:
    if not path_groups:
        return []
    return [path for group in path_groups for path in group]


def _load_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    records = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError(f"{path} does not contain records[]")
    out: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        if isinstance(record, dict):
            row = dict(record)
            row.setdefault("_source_json", str(path))
            row.setdefault("_record_index", idx)
            out.append(row)
    return out


def load_records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        records.extend(_load_records(Path(path)))
    return records


def parse_offsets(text: str | None) -> set[int]:
    raw = text or "10,20,40"
    offsets = {int(part.strip()) for part in raw.split(",") if part.strip()}
    if not offsets:
        raise ValueError("at least one offset is required")
    if any(offset <= 0 for offset in offsets):
        raise ValueError(f"offsets must be positive: {sorted(offsets)}")
    return offsets


def _text_filter(text: str | None) -> set[str] | None:
    if text is None or not text.strip() or text.strip().lower() == "all":
        return None
    return {part.strip().lower() for part in text.split(",") if part.strip()}


def _int_filter(text: str | None) -> set[int] | None:
    if text is None or not text.strip() or text.strip().lower() == "all":
        return None
    return {int(part.strip()) for part in text.split(",") if part.strip()}


def _int_or_none(value: object) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def record_offset(record: dict[str, Any]) -> int | None:
    promotion = _int_or_none(record.get("moves_to_promotion"))
    if promotion is not None:
        return promotion
    return _int_or_none(record.get("moves_to_terminal"))


def root_cluster_key(record: dict[str, Any]) -> str:
    return str(
        record.get("root_replay")
        or record.get("ancestry_key")
        or record.get("source_replay")
        or record.get("root_seed")
        or record.get("source_seed")
        or record.get("id")
        or "unknown"
    )


def _sort_key(record: dict[str, Any]) -> tuple[object, ...]:
    return (
        int(record.get("target_tile", 0) or 0),
        str(record.get("outcome", "")),
        int(record.get("sample_offset", 0) or 0),
        str(record.get("root_seed", "")),
        int(record.get("source_frame_index", record.get("frame_position", 0)) or 0),
        str(record.get("id", "")),
    )


def sample_offset_records(
    records: Iterable[dict[str, Any]],
    *,
    offsets: set[int],
    target_filter: set[int] | None = None,
    outcome_filter: set[str] | None = None,
    max_per_root_offset: int = 1,
    max_records: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rejected: Counter[str] = Counter()
    candidates: list[dict[str, Any]] = []
    for record in records:
        target = _int_or_none(record.get("target_tile"))
        if target_filter is not None and target not in target_filter:
            rejected["target_filter"] += 1
            continue
        outcome = str(record.get("outcome", "")).lower()
        if outcome_filter is not None and outcome not in outcome_filter:
            rejected["outcome_filter"] += 1
            continue
        offset = record_offset(record)
        if offset not in offsets:
            rejected["offset_filter"] += 1
            continue
        row = {key: value for key, value in record.items() if not str(key).startswith("_")}
        row["sample_offset"] = int(offset)
        row["sample_horizon"] = int(offset)
        row["root_cluster_key"] = root_cluster_key(row)
        candidates.append(row)

    candidates.sort(key=_sort_key)
    selected: list[dict[str, Any]] = []
    counts: Counter[tuple[str, int, str, int]] = Counter()
    for row in candidates:
        key = (
            str(row["root_cluster_key"]),
            int(row.get("target_tile", 0) or 0),
            str(row.get("outcome", "")),
            int(row["sample_offset"]),
        )
        if max_per_root_offset > 0 and counts[key] >= int(max_per_root_offset):
            rejected["max_per_root_offset"] += 1
            continue
        selected.append(row)
        counts[key] += 1
        if max_records > 0 and len(selected) >= int(max_records):
            break
    return selected, dict(rejected)


def summarize_records(
    records: list[dict[str, Any]],
    *,
    source_paths: list[str],
    source_records: int,
    offsets: set[int],
    target_filter: set[int] | None,
    outcome_filter: set[str] | None,
    max_per_root_offset: int,
    max_records: int,
    rejected: dict[str, int],
) -> dict[str, Any]:
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_paths": source_paths,
        "source_records": int(source_records),
        "records": len(records),
        "offsets": sorted(offsets),
        "target_filter": sorted(target_filter) if target_filter is not None else None,
        "outcome_filter": sorted(outcome_filter) if outcome_filter is not None else None,
        "max_per_root_offset": int(max_per_root_offset),
        "max_records": int(max_records),
        "by_target_outcome_offset": dict(
            Counter(
                f"{record.get('target_tile')}:{record.get('outcome')}:{record.get('sample_offset')}"
                for record in records
            )
        ),
        "by_outcome": dict(Counter(str(record.get("outcome")) for record in records)),
        "by_offset": dict(Counter(str(record.get("sample_offset")) for record in records)),
        "root_clusters": len({str(record.get("root_cluster_key")) for record in records}),
        "root_seeds": len({str(record.get("root_seed")) for record in records if record.get("root_seed") is not None}),
        "by_root_origin": dict(Counter(str(record.get("root_origin", "unknown")) for record in records)),
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
            f"<td>{cell(record.get('sample_offset'))}</td>"
            f"<td>{cell(record.get('root_seed'))}</td>"
            f"<td>{cell(record.get('move_count'))}</td>"
            f"<td>{cell(record.get('score_minus_starter'))}</td>"
            f"<td>{cell(record.get('raw_count_768'))}</td>"
            f"<td>{cell(record.get('raw_count_1536'))}</td>"
            f"<td>{cell(record.get('source_next_action'))}</td>"
            f"<td>{cell(record.get('source_replay'))}</td>"
            "</tr>"
        )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Transition Offset Sample</title>
  <style>
    :root {{ color-scheme: dark; --bg:#101318; --panel:#171d24; --line:#34404d; --text:#edf2f7; --muted:#aab6c2; --gold:#f2c14e; }}
    body {{ margin:0; background:var(--bg); color:var(--text); font-family:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ width:min(1180px, calc(100vw - 32px)); margin:0 auto; padding:24px 0 40px; }}
    h1 {{ margin:0 0 8px; font-size:24px; }}
    .muted {{ color:var(--muted); }}
    .cards {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(150px, 1fr)); gap:10px; margin:18px 0; }}
    .card {{ border:1px solid var(--line); background:var(--panel); border-radius:8px; padding:12px; }}
    .label {{ color:var(--muted); font-size:11px; text-transform:uppercase; }}
    .value {{ margin-top:4px; color:var(--gold); font-size:22px; font-weight:800; }}
    table {{ width:100%; border-collapse:collapse; font-size:12px; font-variant-numeric:tabular-nums; }}
    th, td {{ border-bottom:1px solid var(--line); padding:7px 8px; text-align:right; vertical-align:top; }}
    th:last-child, td:last-child {{ text-align:left; max-width:360px; overflow-wrap:anywhere; color:var(--muted); }}
    pre {{ white-space:pre-wrap; overflow-wrap:anywhere; color:var(--muted); }}
  </style>
</head>
<body>
  <main>
    <h1>Transition Offset Sample</h1>
    <p class="muted">Root-capped records sampled at fixed moves before promotion or terminal failure.</p>
    <section class="cards">
      <div class="card"><div class="label">Records</div><div class="value">{cell(summary.get('records', 0))}</div></div>
      <div class="card"><div class="label">Root Clusters</div><div class="value">{cell(summary.get('root_clusters', 0))}</div></div>
      <div class="card"><div class="label">Offsets</div><div class="value">{cell(summary.get('offsets', []))}</div></div>
      <div class="card"><div class="label">Targets</div><div class="value">{cell(summary.get('target_filter'))}</div></div>
    </section>
    <table><thead><tr><th>Target</th><th>Outcome</th><th>Offset</th><th>Root</th><th>Move</th><th>Score - Starter</th><th>768s</th><th>1536s</th><th>Next</th><th>Replay</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
    <h2>Summary JSON</h2>
    <pre>{escape(json.dumps(summary, indent=2, sort_keys=True))}</pre>
  </main>
</body>
</html>
"""
    path.write_text(html)


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    paths = _flatten_paths(args.records_json)
    source_records = load_records(paths)
    offsets = parse_offsets(args.offsets)
    target_filter = _int_filter(args.target)
    outcome_filter = _text_filter(args.outcome)
    records, rejected = sample_offset_records(
        source_records,
        offsets=offsets,
        target_filter=target_filter,
        outcome_filter=outcome_filter,
        max_per_root_offset=args.max_per_root_offset,
        max_records=args.max_records,
    )
    summary = summarize_records(
        records,
        source_paths=[str(path) for path in paths],
        source_records=len(source_records),
        offsets=offsets,
        target_filter=target_filter,
        outcome_filter=outcome_filter,
        max_per_root_offset=args.max_per_root_offset,
        max_records=args.max_records,
        rejected=rejected,
    )
    payload = {
        "version": 1,
        "kind": "transition_offset_sample",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "summary": summary,
        "records": records,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload["json"] = str(args.out_dir / "transition_offset_sample.json")
    payload["records_json"] = str(args.out_dir / "records.json")
    payload["html"] = str(args.out_dir / "transition_offset_sample.html")
    write_json(args.out_dir / "transition_offset_sample.json", payload)
    write_json(args.out_dir / "records.json", records)
    write_json(args.out_dir / "summary.json", summary)
    write_html(args.out_dir / "transition_offset_sample.html", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-json", type=Path, nargs="+", action="append", required=True)
    parser.add_argument("--offsets", default="10,20,40")
    parser.add_argument("--target", default="3072")
    parser.add_argument("--outcome", default="all")
    parser.add_argument("--max-per-root-offset", type=int, default=1)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("threes_rl/runs/forensics/transition_offset_samples/latest"))
    args = parser.parse_args()
    payload = run_from_args(args)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"json={payload['json']}")
    print(f"records={payload['records_json']}")
    print(f"html={payload['html']}")


if __name__ == "__main__":
    main()
