"""Select frontier records using cheap rollout-screen conversion estimates."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from html import escape
from pathlib import Path
from typing import Any, Iterable

from threes_rl.frontier_matched_controls import group_key, record_id
from threes_rl.run_artifacts import write_json

SUPPORTED_MODES = ("top", "bottom", "random")
SUPPORTED_SCORE_FIELDS = ("best_action_rate", "target_rate", "target_hits", "valid_rollouts")
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


def load_action_summary(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(Path(path).read_text())
        action_summary = payload.get("action_summary") if isinstance(payload, dict) else None
        if not isinstance(action_summary, list):
            raise ValueError(f"{path} does not contain action_summary[]")
        rows.extend(dict(row) for row in action_summary if isinstance(row, dict))
    return rows


def aggregate_action_summary(action_rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_case: dict[str, dict[str, Any]] = {}
    for row in action_rows:
        case_id = row.get("case_id")
        if case_id is None:
            continue
        key = str(case_id)
        item = by_case.setdefault(
            key,
            {
                "target_hits": 0,
                "valid_rollouts": 0,
                "rollouts": 0,
                "best_action": None,
                "best_action_rate": 0.0,
                "best_action_hits": 0,
                "best_action_rollouts": 0,
                "actions": [],
            },
        )
        hits = int(row.get("target_hits", 0) or 0)
        valid = int(row.get("valid_rollouts", row.get("rollouts", 0)) or 0)
        rollouts = int(row.get("rollouts", valid) or 0)
        rate = float(row.get("target_rate", 0.0) or 0.0)
        action = str(row.get("first_action", "unknown"))
        item["target_hits"] += hits
        item["valid_rollouts"] += valid
        item["rollouts"] += rollouts
        item["actions"].append(
            {
                "first_action": action,
                "target_hits": hits,
                "valid_rollouts": valid,
                "target_rate": rate,
                "mean_score_delta": float(row.get("mean_score_delta", 0.0) or 0.0),
            }
        )
        if rate > float(item["best_action_rate"]):
            item["best_action"] = action
            item["best_action_rate"] = rate
            item["best_action_hits"] = hits
            item["best_action_rollouts"] = valid
    for item in by_case.values():
        valid = int(item.get("valid_rollouts", 0) or 0)
        hits = int(item.get("target_hits", 0) or 0)
        item["target_rate"] = float(hits / valid) if valid else 0.0
        item["positive"] = hits > 0
    return by_case


def _score(stats: dict[str, Any], field: str) -> float:
    if field not in SUPPORTED_SCORE_FIELDS:
        raise ValueError(f"Unsupported score field: {field}")
    return float(stats.get(field, 0.0) or 0.0)


def _reference_counts(records: Iterable[dict[str, Any]], *, group_by: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        counts[group_key(record, group_by)] += 1
    return counts


def _ranked(
    records: list[dict[str, Any]],
    *,
    mode: str,
    score_field: str,
    seed: int,
) -> list[dict[str, Any]]:
    if mode == "random":
        rows = list(records)
        random.Random(int(seed)).shuffle(rows)
        return rows
    return sorted(
        records,
        key=lambda row: (
            _score(row["rollout_selector"]["screen"], score_field),
            int(row.get("score_delta") or 0),
            record_id(row),
        ),
        reverse=(mode == "top"),
    )


def select_records(
    records: Iterable[dict[str, Any]],
    *,
    scores: dict[str, dict[str, Any]],
    mode: str,
    score_field: str = "best_action_rate",
    group_by: str = "root_seed",
    max_records: int = 0,
    max_per_group: int = 0,
    reference_records: Iterable[dict[str, Any]] | None = None,
    seed: int = 0,
    exclude_reference: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"Unsupported mode: {mode}")
    if group_by not in SUPPORTED_GROUP_KEYS:
        raise ValueError(f"Unsupported group key: {group_by}")
    if score_field not in SUPPORTED_SCORE_FIELDS:
        raise ValueError(f"Unsupported score field: {score_field}")

    ref_records = list(reference_records or [])
    ref_counts = _reference_counts(ref_records, group_by=group_by) if ref_records else None
    ref_ids = {record_id(record) for record in ref_records}

    scored: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    seen_ids: set[str] = set()
    for record in records:
        rid = record_id(record)
        if rid in seen_ids:
            skipped["duplicate_id"] += 1
            continue
        seen_ids.add(rid)
        if exclude_reference and rid in ref_ids:
            skipped["reference_record"] += 1
            continue
        stats = scores.get(rid)
        if stats is None:
            skipped["missing_screen"] += 1
            continue
        key = group_key(record, group_by)
        if ref_counts is not None and key not in ref_counts:
            skipped["unmatched_group"] += 1
            continue
        row = {name: value for name, value in record.items() if not str(name).startswith("_")}
        row["rollout_selector"] = {
            "mode": mode,
            "score_field": score_field,
            "group_by": group_by,
            "group_key": key,
            "screen": stats,
            "seed": int(seed),
        }
        scored.append(row)

    selected: list[dict[str, Any]] = []
    selected_counts: Counter[str] = Counter()
    shortages: dict[str, int] = {}
    if ref_counts is not None:
        by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in scored:
            by_group[str(row["rollout_selector"]["group_key"])].append(row)
        for key in sorted(ref_counts):
            ranked = _ranked(by_group.get(key, []), mode=mode, score_field=score_field, seed=seed)
            want = int(ref_counts[key])
            chosen = ranked[:want]
            if len(chosen) < want:
                shortages[key] = want - len(chosen)
            for local_rank, row in enumerate(chosen, start=1):
                row["rollout_selector"]["group_rank"] = int(local_rank)
                row["rollout_selector"]["reference_count_for_group"] = want
                selected.append(row)
                selected_counts[key] += 1
    else:
        ranked = _ranked(scored, mode=mode, score_field=score_field, seed=seed)
        for row in ranked:
            key = str(row["rollout_selector"]["group_key"])
            if max_per_group > 0 and selected_counts[key] >= int(max_per_group):
                skipped["max_per_group"] += 1
                continue
            row["rollout_selector"]["rank"] = len(selected) + 1
            selected.append(row)
            selected_counts[key] += 1
            if max_records > 0 and len(selected) >= int(max_records):
                break

    selected.sort(
        key=lambda row: (
            str(row["rollout_selector"]["group_key"]),
            int(row["rollout_selector"].get("group_rank", row["rollout_selector"].get("rank", 0))),
        )
    )
    screen_scores = [_score(row["rollout_selector"]["screen"], score_field) for row in selected]
    summary = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "mode": mode,
        "score_field": score_field,
        "group_by": group_by,
        "seed": int(seed),
        "pool_records": len(seen_ids),
        "screened_records": len(scored),
        "reference_records": len(ref_records),
        "selected_records": len(selected),
        "selected_counts": dict(selected_counts),
        "reference_counts": dict(ref_counts or {}),
        "shortages": shortages,
        "skipped": dict(skipped),
        "mean_selected_score": float(sum(screen_scores) / len(screen_scores)) if screen_scores else 0.0,
        "max_selected_score": float(max(screen_scores)) if screen_scores else 0.0,
        "min_selected_score": float(min(screen_scores)) if screen_scores else 0.0,
    }
    return selected, summary


def write_html(path: Path, payload: dict[str, Any]) -> None:
    summary = payload.get("summary", {})
    records = payload.get("records", [])

    def cell(value: object) -> str:
        return escape(str(value))

    rows = []
    for record in records[:120] if isinstance(records, list) else []:
        selector = record.get("rollout_selector") if isinstance(record, dict) else {}
        if not isinstance(selector, dict):
            selector = {}
        screen = selector.get("screen") if isinstance(selector.get("screen"), dict) else {}
        rows.append(
            "<tr>"
            f"<td>{cell(selector.get('group_key'))}</td>"
            f"<td>{cell(selector.get('rank', selector.get('group_rank')))}</td>"
            f"<td>{cell(record.get('id'))}</td>"
            f"<td>{cell(screen.get('best_action_rate'))}</td>"
            f"<td>{cell(screen.get('target_rate'))}</td>"
            f"<td>{cell(screen.get('target_hits'))}</td>"
            f"<td>{cell(screen.get('valid_rollouts'))}</td>"
            f"<td>{cell(screen.get('best_action'))}</td>"
            "</tr>"
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Rollout Frontier Selector</title>
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
    table {{ width:100%; border-collapse:collapse; font-size:12px; margin-top:12px; }}
    th, td {{ border-bottom:1px solid var(--line); padding:6px 8px; text-align:right; vertical-align:top; }}
    th:nth-child(3), td:nth-child(3) {{ text-align:left; overflow-wrap:anywhere; }}
    pre {{ white-space:pre-wrap; overflow-wrap:anywhere; margin:0; color:var(--muted); }}
  </style>
</head>
<body>
  <main>
    <h1>Rollout Frontier Selector</h1>
    <p class="muted">Frontier states selected by cheap continuation-screen statistics.</p>
    <section class="cards">
      <div class="card"><div class="label">Mode</div><div class="value">{cell(summary.get('mode'))}</div></div>
      <div class="card"><div class="label">Selected</div><div class="value">{cell(summary.get('selected_records', 0))}</div></div>
      <div class="card"><div class="label">Mean Score</div><div class="value">{cell(round(float(summary.get('mean_selected_score', 0.0)), 3))}</div></div>
      <div class="card"><div class="label">Field</div><div class="value">{cell(summary.get('score_field'))}</div></div>
    </section>
    <section class="panel"><pre>{escape(json.dumps(summary, indent=2, sort_keys=True))}</pre></section>
    <section class="panel" style="margin-top:14px;">
      <table>
        <thead><tr><th>Group</th><th>Rank</th><th>ID</th><th>Best Rate</th><th>Target Rate</th><th>Hits</th><th>Rollouts</th><th>Best Action</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </section>
  </main>
</body>
</html>
"""
    path.write_text(html)


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    records = load_records(args.records_json)
    scores = aggregate_action_summary(load_action_summary(args.frontier_json))
    reference_records = load_records(args.reference_json) if args.reference_json else None
    selected, summary = select_records(
        records,
        scores=scores,
        mode=args.mode,
        score_field=args.score_field,
        group_by=args.group_by,
        max_records=args.max_records,
        max_per_group=args.max_per_group,
        reference_records=reference_records,
        seed=args.seed,
        exclude_reference=not args.include_reference,
    )
    payload = {
        "version": 1,
        "kind": "frontier_rollout_selector",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "records_json": [str(path) for path in args.records_json],
        "frontier_json": [str(path) for path in args.frontier_json],
        "summary": summary,
        "records": selected,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload["json"] = str(args.out_dir / "rollout_selection.json")
    payload["records_json_out"] = str(args.out_dir / "records.json")
    payload["summary_json"] = str(args.out_dir / "summary.json")
    payload["html"] = str(args.out_dir / "rollout_selection.html")
    write_json(args.out_dir / "rollout_selection.json", payload)
    write_json(args.out_dir / "records.json", selected)
    write_json(args.out_dir / "summary.json", summary)
    write_html(args.out_dir / "rollout_selection.html", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-json", type=Path, nargs="+", required=True)
    parser.add_argument("--frontier-json", type=Path, nargs="+", required=True)
    parser.add_argument("--reference-json", type=Path, nargs="+")
    parser.add_argument("--mode", choices=SUPPORTED_MODES, required=True)
    parser.add_argument("--score-field", choices=SUPPORTED_SCORE_FIELDS, default="best_action_rate")
    parser.add_argument("--group-by", choices=SUPPORTED_GROUP_KEYS, default="root_seed")
    parser.add_argument("--max-records", type=int, default=16)
    parser.add_argument("--max-per-group", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-reference", action="store_true")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = run_from_args(args)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"records={payload['records_json_out']}")
    print(f"html={payload['html']}")


if __name__ == "__main__":
    main()
