"""Filter state-record JSON corpora for focused diagnostics."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from html import escape
from pathlib import Path
from typing import Any, Iterable

from threes_rl.replay_policy_agreement import parse_corner_risk_filters, parse_phase_filters
from threes_rl.run_artifacts import write_json


def _flatten_paths(path_groups: list[list[Path]] | None) -> list[Path]:
    if not path_groups:
        return []
    return [path for group in path_groups for path in group]


def _load_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    records = payload.get("records", []) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError(f"{path} does not contain a records list")
    return [record for record in records if isinstance(record, dict)]


def load_records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        for idx, record in enumerate(_load_records(Path(path))):
            records.append({**record, "_source_json": str(path), "_record_index": int(idx)})
    return records


def parse_int_filter(text: str | None) -> set[int] | None:
    if text is None or not text.strip():
        return None
    values: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if part:
            values.add(int(part))
    return values or None


def parse_text_filter(text: str | None) -> set[str] | None:
    if text is None or not text.strip():
        return None
    values = {part.strip().lower() for part in text.split(",") if part.strip()}
    return values or None


def parse_id_filter(text: str | None) -> set[str] | None:
    if text is None or not text.strip():
        return None
    values = {part.strip() for part in text.split(",") if part.strip()}
    return values or None


def _record_phase(record: dict[str, Any]) -> str:
    features = record.get("features")
    if isinstance(features, dict) and features.get("phase") is not None:
        return str(features["phase"])
    return str(record.get("phase", "unknown"))


def _record_corner_risk(record: dict[str, Any]) -> str:
    features = record.get("features")
    if isinstance(features, dict) and features.get("corner_risk") is not None:
        return str(features["corner_risk"])
    return str(record.get("corner_risk", "unknown"))


def _record_stratum(record: dict[str, Any]) -> str:
    features = record.get("features")
    if isinstance(features, dict) and features.get("stratum") is not None:
        return str(features["stratum"])
    return str(record.get("stratum", f"{_record_phase(record)}/{_record_corner_risk(record)}"))


def filter_records(
    records: Iterable[dict[str, Any]],
    *,
    id_filter: set[str] | None = None,
    synthetic_kind_filter: set[str] | None = None,
    target_filter: set[int] | None = None,
    milestone_filter: set[str] | None = None,
    outcome_filter: set[str] | None = None,
    phase_filter: set[str] | None = None,
    corner_risk_filter: set[str] | None = None,
    max_records: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    accepted: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    for record in records:
        if id_filter is not None and str(record.get("id")) not in id_filter:
            rejected["id_filter"] += 1
            continue
        synthetic_kind = record.get("synthetic_kind")
        if synthetic_kind_filter is not None and (
            synthetic_kind is None or str(synthetic_kind).lower() not in synthetic_kind_filter
        ):
            rejected["synthetic_kind_filter"] += 1
            continue
        target = record.get("target_tile")
        if target_filter is not None and (target is None or int(target) not in target_filter):
            rejected["target_filter"] += 1
            continue
        milestone = record.get("target_milestone")
        if milestone_filter is not None and (milestone is None or str(milestone).lower() not in milestone_filter):
            rejected["milestone_filter"] += 1
            continue
        outcome = record.get("outcome")
        if outcome_filter is not None and (outcome is None or str(outcome).lower() not in outcome_filter):
            rejected["outcome_filter"] += 1
            continue
        phase = _record_phase(record)
        if phase_filter is not None and phase not in phase_filter:
            rejected["phase_filter"] += 1
            continue
        corner_risk = _record_corner_risk(record)
        if corner_risk_filter is not None and corner_risk not in corner_risk_filter:
            rejected["corner_risk_filter"] += 1
            continue
        accepted.append(record)
        if max_records > 0 and len(accepted) >= int(max_records):
            break
    return accepted, dict(rejected)


def summarize(
    records: list[dict[str, Any]],
    *,
    source_paths: list[str],
    source_count: int,
    rejected: dict[str, int],
    id_filter: set[str] | None,
    synthetic_kind_filter: set[str] | None,
    target_filter: set[int] | None,
    milestone_filter: set[str] | None,
    outcome_filter: set[str] | None,
    phase_filter: set[str] | None,
    corner_risk_filter: set[str] | None,
    max_records: int,
) -> dict[str, Any]:
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_paths": source_paths,
        "source_records": int(source_count),
        "records": len(records),
        "id_filter": sorted(id_filter) if id_filter is not None else None,
        "synthetic_kind_filter": sorted(synthetic_kind_filter) if synthetic_kind_filter is not None else None,
        "target_filter": sorted(target_filter) if target_filter is not None else None,
        "milestone_filter": sorted(milestone_filter) if milestone_filter is not None else None,
        "outcome_filter": sorted(outcome_filter) if outcome_filter is not None else None,
        "phase_filter": sorted(phase_filter) if phase_filter is not None else None,
        "corner_risk_filter": sorted(corner_risk_filter) if corner_risk_filter is not None else None,
        "max_records": int(max_records),
        "by_target_outcome": _target_outcome_counts(records),
        "by_milestone_outcome": _milestone_outcome_counts(records),
        "by_target": dict(Counter(str(record.get("target_tile", "unknown")) for record in records)),
        "by_milestone": dict(Counter(str(record.get("target_milestone", "unknown")) for record in records)),
        "by_outcome": dict(Counter(str(record.get("outcome", "unknown")) for record in records)),
        "by_synthetic_kind": dict(Counter(str(record.get("synthetic_kind", "unknown")) for record in records)),
        "by_stratum": dict(Counter(_record_stratum(record) for record in records)),
        "rejected": rejected,
    }


def _target_outcome_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(f"{record.get('target_tile')}:{record.get('outcome')}" for record in records))


def _milestone_outcome_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(f"{record.get('target_milestone', 'unknown')}:{record.get('outcome')}" for record in records))


def write_html(path: Path, payload: dict[str, Any]) -> None:
    summary = payload.get("summary", {})

    def cell(value: object) -> str:
        return escape(str(value))

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Filtered State Records</title>
  <style>
    :root {{ color-scheme: dark; --bg: #101318; --panel: #171d24; --line: #34404d; --text: #edf2f7; --muted: #aab6c2; --gold: #f2c14e; }}
    body {{ margin: 0; background: var(--bg); color: var(--text); font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ width: min(1000px, calc(100vw - 32px)); margin: 0 auto; padding: 24px 0 40px; }}
    h1 {{ margin: 0 0 8px; font-size: 24px; }}
    .muted {{ color: var(--muted); }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 18px 0; }}
    .card {{ border: 1px solid var(--line); background: var(--panel); border-radius: 8px; padding: 12px; }}
    .label {{ color: var(--muted); font-size: 11px; text-transform: uppercase; }}
    .value {{ margin-top: 4px; color: var(--gold); font-size: 22px; font-weight: 800; }}
    pre {{ white-space: pre-wrap; overflow-wrap: anywhere; color: var(--muted); }}
  </style>
</head>
<body>
  <main>
    <h1>Filtered State Records</h1>
    <p class="muted">Focused state-record corpus for diagnostic scans.</p>
    <section class="cards">
      <div class="card"><div class="label">Records</div><div class="value">{cell(summary.get('records', 0))}</div></div>
      <div class="card"><div class="label">Source Records</div><div class="value">{cell(summary.get('source_records', 0))}</div></div>
      <div class="card"><div class="label">Targets</div><div class="value">{cell(summary.get('target_filter'))}</div></div>
      <div class="card"><div class="label">Outcomes</div><div class="value">{cell(summary.get('outcome_filter'))}</div></div>
    </section>
    <pre>{escape(json.dumps(summary, indent=2, sort_keys=True))}</pre>
  </main>
</body>
</html>
"""
    path.write_text(html)


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    paths = _flatten_paths(args.state_json)
    source_records = load_records(paths)
    phase_filter = parse_phase_filters(args.phase_filter)
    corner_filter = parse_corner_risk_filters(args.corner_risk_filter)
    id_filter = parse_id_filter(args.id)
    synthetic_kind_filter = parse_text_filter(args.synthetic_kind)
    target_filter = parse_int_filter(args.target)
    milestone_filter = parse_text_filter(args.milestone)
    outcome_filter = parse_text_filter(args.outcome)
    records, rejected = filter_records(
        source_records,
        id_filter=id_filter,
        synthetic_kind_filter=synthetic_kind_filter,
        target_filter=target_filter,
        milestone_filter=milestone_filter,
        outcome_filter=outcome_filter,
        phase_filter=phase_filter,
        corner_risk_filter=corner_filter,
        max_records=args.max_records,
    )
    summary = summarize(
        records,
        source_paths=[str(path) for path in paths],
        source_count=len(source_records),
        rejected=rejected,
        id_filter=id_filter,
        synthetic_kind_filter=synthetic_kind_filter,
        target_filter=target_filter,
        milestone_filter=milestone_filter,
        outcome_filter=outcome_filter,
        phase_filter=phase_filter,
        corner_risk_filter=corner_filter,
        max_records=args.max_records,
    )
    payload = {
        "version": 1,
        "kind": "filtered_state_records",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "summary": summary,
        "records": records,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload["json"] = str(args.out_dir / "filtered_state_records.json")
    payload["records_json"] = str(args.out_dir / "records.json")
    payload["html"] = str(args.out_dir / "filtered_state_records.html")
    write_json(args.out_dir / "filtered_state_records.json", payload)
    write_json(args.out_dir / "records.json", records)
    write_html(args.out_dir / "filtered_state_records.html", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-json", type=Path, nargs="+", action="append", required=True)
    parser.add_argument("--id", help="Comma-separated record ids to keep.")
    parser.add_argument("--synthetic-kind", help="Comma-separated synthetic_kind values to keep.")
    parser.add_argument("--target", help="Comma-separated target_tile values to keep.")
    parser.add_argument("--milestone", help="Comma-separated target_milestone values to keep.")
    parser.add_argument("--outcome", help="Comma-separated outcomes to keep, e.g. success,failure.")
    parser.add_argument("--phase-filter", action="append")
    parser.add_argument("--corner-risk-filter", action="append")
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("threes_rl/runs/forensics/state_records/latest_filter"))
    args = parser.parse_args()
    payload = run_from_args(args)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"json={payload['json']}")
    print(f"records={payload['records_json']}")
    print(f"html={payload['html']}")


if __name__ == "__main__":
    main()
