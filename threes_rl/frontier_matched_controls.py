"""Build root-matched frontier control corpora from a ranked state pool."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from html import escape
from pathlib import Path
from typing import Any, Iterable

from threes_rl.run_artifacts import write_json

SUPPORTED_MODES = ("top", "bottom", "random")
SUPPORTED_GROUP_KEYS = ("root_seed", "ancestry_key", "source_replay")


def _extract_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = (
            payload.get("records")
            or payload.get("target_records")
            or payload.get("selected_records")
            or payload.get("frontier_records")
        )
    else:
        records = None
    if not isinstance(records, list):
        raise ValueError("JSON payload does not contain a records list")
    return [record for record in records if isinstance(record, dict)]


def load_records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(Path(path).read_text())
        for idx, record in enumerate(_extract_records(payload)):
            row = dict(record)
            row.setdefault("_source_json", str(path))
            row.setdefault("_record_index", int(idx))
            records.append(row)
    return records


def record_id(record: dict[str, Any]) -> str:
    value = record.get("id")
    if value is not None:
        return str(value)
    return f"{record.get('_source_json', 'unknown')}:{record.get('_record_index', 'unknown')}"


def group_key(record: dict[str, Any], key: str) -> str:
    if key not in SUPPORTED_GROUP_KEYS:
        raise ValueError(f"Unsupported group key: {key}")
    value = record.get(key)
    if value is not None:
        return str(value)
    if key == "root_seed":
        return str(record.get("root_replay") or record.get("ancestry_key") or record_id(record))
    if key == "ancestry_key":
        return str(record.get("root_replay") or record.get("root_seed") or record_id(record))
    return str(record.get("root_seed") or record.get("ancestry_key") or record_id(record))


def rank_tuple(record: dict[str, Any]) -> tuple[float, ...]:
    for owner_name in ("frontier", "frontier_selector"):
        owner = record.get(owner_name)
        if isinstance(owner, dict) and isinstance(owner.get("rank_tuple"), list):
            out: list[float] = []
            for value in owner["rank_tuple"]:
                try:
                    out.append(float(value))
                except (TypeError, ValueError):
                    out.append(0.0)
            return tuple(out)
    return tuple()


def reference_counts(records: Iterable[dict[str, Any]], *, group_by: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        counts[group_key(record, group_by)] += 1
    return counts


def select_matched_records(
    pool_records: Iterable[dict[str, Any]],
    reference_records: Iterable[dict[str, Any]],
    *,
    mode: str,
    group_by: str = "root_seed",
    seed: int = 0,
    exclude_reference: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"Unsupported mode: {mode}")
    ref_records = list(reference_records)
    ref_counts = reference_counts(ref_records, group_by=group_by)
    ref_ids = {record_id(record) for record in ref_records}
    by_group: dict[str, list[dict[str, Any]]] = {key: [] for key in ref_counts}
    skipped: Counter[str] = Counter()
    seen_ids: set[str] = set()
    for record in pool_records:
        rid = record_id(record)
        if rid in seen_ids:
            skipped["duplicate_id"] += 1
            continue
        seen_ids.add(rid)
        if exclude_reference and rid in ref_ids:
            skipped["reference_record"] += 1
            continue
        key = group_key(record, group_by)
        if key not in by_group:
            skipped["unmatched_group"] += 1
            continue
        by_group[key].append(dict(record))

    rng = random.Random(int(seed))
    selected: list[dict[str, Any]] = []
    shortages: dict[str, int] = {}
    selected_counts: Counter[str] = Counter()
    for key in sorted(ref_counts):
        candidates = by_group[key]
        if mode == "random":
            ranked = list(candidates)
            rng.shuffle(ranked)
        else:
            ranked = sorted(
                candidates,
                key=lambda row: (rank_tuple(row), int(row.get("score_delta") or 0), record_id(row)),
                reverse=(mode == "top"),
            )
        want = int(ref_counts[key])
        chosen = ranked[:want]
        if len(chosen) < want:
            shortages[key] = want - len(chosen)
        for local_rank, row in enumerate(chosen, start=1):
            control = {name: value for name, value in row.items() if not str(name).startswith("_")}
            control["matched_control"] = {
                "mode": mode,
                "group_by": group_by,
                "group_key": key,
                "group_rank": int(local_rank),
                "reference_count_for_group": want,
                "rank_tuple": list(rank_tuple(row)),
                "exclude_reference": bool(exclude_reference),
                "seed": int(seed),
            }
            selected.append(control)
            selected_counts[key] += 1
    selected.sort(key=lambda row: (str((row.get("matched_control") or {}).get("group_key")), int((row.get("matched_control") or {}).get("group_rank", 0))))
    summary = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "mode": mode,
        "group_by": group_by,
        "seed": int(seed),
        "exclude_reference": bool(exclude_reference),
        "pool_records": len(seen_ids),
        "reference_records": len(ref_records),
        "selected_records": len(selected),
        "reference_counts": dict(ref_counts),
        "selected_counts": dict(selected_counts),
        "shortages": shortages,
        "skipped": dict(skipped),
        "unique_groups": len(ref_counts),
    }
    return selected, summary


def write_html(path: Path, payload: dict[str, Any]) -> None:
    summary = payload.get("summary", {})
    records = payload.get("records", [])

    def cell(value: object) -> str:
        return escape(str(value))

    rows = []
    for record in records[:100] if isinstance(records, list) else []:
        control = record.get("matched_control") if isinstance(record, dict) else {}
        if not isinstance(control, dict):
            control = {}
        features = record.get("features") if isinstance(record, dict) else {}
        if not isinstance(features, dict):
            features = {}
        rows.append(
            "<tr>"
            f"<td>{cell(control.get('group_key'))}</td>"
            f"<td>{cell(control.get('group_rank'))}</td>"
            f"<td>{cell(record.get('id'))}</td>"
            f"<td>{cell(record.get('move_count'))}</td>"
            f"<td>{cell(features.get('empty_count'))}</td>"
            f"<td>{cell(features.get('raw_768_adjacent_pairs'))}</td>"
            f"<td>{cell(features.get('raw_768_air_neighbors'))}</td>"
            f"<td>{cell(control.get('rank_tuple'))}</td>"
            "</tr>"
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Matched Frontier Controls</title>
  <style>
    :root {{ color-scheme: dark; --bg:#101318; --panel:#171d24; --line:#34404d; --text:#edf2f7; --muted:#aab6c2; --gold:#f2c14e; }}
    body {{ margin:0; background:var(--bg); color:var(--text); font-family:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ width:min(1160px, calc(100vw - 32px)); margin:0 auto; padding:24px 0 40px; }}
    h1 {{ margin:0 0 8px; font-size:24px; }}
    .muted {{ color:var(--muted); }}
    .cards {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(150px, 1fr)); gap:10px; margin:18px 0; }}
    .card, .panel {{ border:1px solid var(--line); background:var(--panel); border-radius:8px; padding:12px; }}
    .label {{ color:var(--muted); font-size:11px; text-transform:uppercase; }}
    .value {{ margin-top:4px; color:var(--gold); font-size:22px; font-weight:800; }}
    table {{ width:100%; border-collapse:collapse; font-size:12px; margin-top:12px; }}
    th, td {{ border-bottom:1px solid var(--line); padding:6px 8px; text-align:right; vertical-align:top; }}
    th:nth-child(3), td:nth-child(3), th:last-child, td:last-child {{ text-align:left; overflow-wrap:anywhere; }}
    pre {{ white-space:pre-wrap; overflow-wrap:anywhere; margin:0; color:var(--muted); }}
  </style>
</head>
<body>
  <main>
    <h1>Matched Frontier Controls</h1>
    <p class="muted">Control states matched to a reference corpus by root/source group counts.</p>
    <section class="cards">
      <div class="card"><div class="label">Mode</div><div class="value">{cell(summary.get('mode'))}</div></div>
      <div class="card"><div class="label">Selected</div><div class="value">{cell(summary.get('selected_records', 0))}</div></div>
      <div class="card"><div class="label">Groups</div><div class="value">{cell(summary.get('unique_groups', 0))}</div></div>
      <div class="card"><div class="label">Shortages</div><div class="value">{cell(len(summary.get('shortages', {})))}</div></div>
    </section>
    <section class="panel"><pre>{escape(json.dumps(summary, indent=2, sort_keys=True))}</pre></section>
    <section class="panel" style="margin-top:14px;">
      <table>
        <thead><tr><th>Group</th><th>Rank</th><th>ID</th><th>Move</th><th>Air</th><th>Adj Pairs</th><th>Air Neighbors</th><th>Rank Tuple</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </section>
  </main>
</body>
</html>
"""
    path.write_text(html)


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    pool_records = load_records(args.pool_json)
    ref_records = load_records(args.reference_json)
    selected, summary = select_matched_records(
        pool_records,
        ref_records,
        mode=args.mode,
        group_by=args.group_by,
        seed=args.seed,
        exclude_reference=not args.include_reference,
    )
    payload = {
        "version": 1,
        "kind": "frontier_matched_controls",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "summary": summary,
        "records": selected,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload["json"] = str(args.out_dir / "matched_controls.json")
    payload["records_json"] = str(args.out_dir / "records.json")
    payload["summary_json"] = str(args.out_dir / "summary.json")
    payload["html"] = str(args.out_dir / "matched_controls.html")
    write_json(args.out_dir / "matched_controls.json", payload)
    write_json(args.out_dir / "records.json", selected)
    write_json(args.out_dir / "summary.json", summary)
    write_html(args.out_dir / "matched_controls.html", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-json", type=Path, nargs="+", required=True)
    parser.add_argument("--reference-json", type=Path, nargs="+", required=True)
    parser.add_argument("--mode", choices=SUPPORTED_MODES, required=True)
    parser.add_argument("--group-by", choices=SUPPORTED_GROUP_KEYS, default="root_seed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-reference", action="store_true")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = run_from_args(args)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"records={payload['records_json']}")
    print(f"html={payload['html']}")


if __name__ == "__main__":
    main()
