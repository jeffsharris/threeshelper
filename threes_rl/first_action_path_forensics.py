"""Join first-action continuation path diagnostics into a compact report."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from html import escape
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

from threes_rl.run_artifacts import write_json


def _load_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text())
    records = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError(f"{path} does not contain records[]")
    return [record for record in records if isinstance(record, dict)]


def _load_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text())
    cases = payload.get("cases") if isinstance(payload, dict) else payload
    if not isinstance(cases, list):
        raise ValueError(f"{path} does not contain cases[]")
    return [case for case in cases if isinstance(case, dict)]


def _milestone_frame(case: dict[str, Any], key: str) -> int | None:
    if key in ("second_3072_event", "first_6144"):
        milestone = case.get(key)
    else:
        milestones = case.get("milestones")
        milestone = milestones.get(key) if isinstance(milestones, dict) else None
    if not isinstance(milestone, dict) or milestone.get("frames_after_first_3072") is None:
        return None
    return int(milestone["frames_after_first_3072"])


def _int(value: object, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _rate(count: int, total: int) -> float:
    return 0.0 if total <= 0 else float(count) / float(total)


def _stats(values: Iterable[int | None]) -> dict[str, Any]:
    cleaned = [int(value) for value in values if value is not None]
    if not cleaned:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(cleaned),
        "mean": float(mean(cleaned)),
        "median": float(median(cleaned)),
        "min": int(min(cleaned)),
        "max": int(max(cleaned)),
    }


def _source_key(record: dict[str, Any]) -> str:
    return f"{record.get('source_seed')}:{record.get('source_frame_index')}"


def _join_records(large_records: list[dict[str, Any]], support_cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    large_by_path = {str(record.get("source_replay")): record for record in large_records}
    rows: list[dict[str, Any]] = []
    for case in support_cases:
        replay_path = str(case.get("source_replay"))
        large = large_by_path.get(replay_path, {})
        first_large = large.get("first_large_ge_768")
        max_candidate = 0
        if isinstance(first_large, dict):
            max_candidate = _int(first_large.get("max_candidate"))
        rows.append(
            {
                "source_replay": replay_path,
                "seed": case.get("seed", large.get("seed")),
                "source_seed": large.get("source_seed"),
                "source_frame_index": large.get("source_frame_index"),
                "source_key": _source_key(large),
                "first_action": large.get("first_action"),
                "first_inserted_value": large.get("first_inserted_value"),
                "score_delta": _int(large.get("score_delta")),
                "final_max_tile_excl_starter": _int(large.get("final_max_tile_excl_starter")),
                "outcome": str(case.get("outcome")),
                "success": str(case.get("outcome")) == "success",
                "reached_6144": bool(large.get("reached_6144") or case.get("first_6144") is not None),
                "raw_duplicate_1536": _milestone_frame(case, "first_raw_duplicate_1536") is not None,
                "raw_adjacent_1536": _milestone_frame(case, "first_raw_adjacent_pair_1536") is not None,
                "second_3072": case.get("second_3072_event") is not None,
                "raw_duplicate_1536_frames_after_first_3072": _milestone_frame(case, "first_raw_duplicate_1536"),
                "raw_adjacent_1536_frames_after_first_3072": _milestone_frame(case, "first_raw_adjacent_pair_1536"),
                "second_3072_frames_after_first_3072": _milestone_frame(case, "second_3072_event"),
                "first_6144_frames_after_first_3072": _milestone_frame(case, "first_6144"),
                "first_large_ge_768": large.get("first_large_ge_768") is not None,
                "first_large_ge_768_frame_after_start": large.get("first_large_ge_768_frame_after_start"),
                "first_large_ge_768_max_candidate": max_candidate,
                "preview_after_first_label": large.get("preview_after_first_label"),
            }
        )
    return rows


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    return {
        "records": total,
        "success": sum(bool(row.get("success")) for row in rows),
        "reached_6144": sum(bool(row.get("reached_6144")) for row in rows),
        "raw_duplicate_1536": sum(bool(row.get("raw_duplicate_1536")) for row in rows),
        "raw_adjacent_1536": sum(bool(row.get("raw_adjacent_1536")) for row in rows),
        "second_3072": sum(bool(row.get("second_3072")) for row in rows),
        "first_large_ge_768": sum(bool(row.get("first_large_ge_768")) for row in rows),
        "rates": {
            "success": _rate(sum(bool(row.get("success")) for row in rows), total),
            "reached_6144": _rate(sum(bool(row.get("reached_6144")) for row in rows), total),
            "raw_adjacent_1536": _rate(sum(bool(row.get("raw_adjacent_1536")) for row in rows), total),
            "first_large_ge_768": _rate(sum(bool(row.get("first_large_ge_768")) for row in rows), total),
        },
        "raw_duplicate_1536_frames_after_first_3072": _stats(
            row.get("raw_duplicate_1536_frames_after_first_3072") for row in rows
        ),
        "raw_adjacent_1536_frames_after_first_3072": _stats(
            row.get("raw_adjacent_1536_frames_after_first_3072") for row in rows
        ),
        "second_3072_frames_after_first_3072": _stats(row.get("second_3072_frames_after_first_3072") for row in rows),
        "first_6144_frames_after_first_3072": _stats(row.get("first_6144_frames_after_first_3072") for row in rows),
        "first_large_ge_768_frame_after_start": _stats(row.get("first_large_ge_768_frame_after_start") for row in rows),
    }


def _group_summary(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get(key)), []).append(row)
    out: list[dict[str, Any]] = []
    for group_key, group in sorted(groups.items(), key=lambda item: item[0]):
        summary = _summarize_rows(group)
        deltas = [_int(row.get("score_delta")) for row in group]
        summary.update(
            {
                key: group_key,
                "mean_score_delta": float(mean(deltas)) if deltas else 0.0,
                "median_score_delta": float(median(deltas)) if deltas else 0.0,
                "high_score_delta": int(max(deltas)) if deltas else 0,
            }
        )
        out.append(summary)
    return out


def _source_action_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row.get("source_key")), str(row.get("first_action"))), []).append(row)
    out: list[dict[str, Any]] = []
    for (source_key, action), group in sorted(groups.items(), key=lambda item: item[0]):
        summary = _summarize_rows(group)
        deltas = [_int(row.get("score_delta")) for row in group]
        first = group[0] if group else {}
        summary.update(
            {
                "source_key": source_key,
                "source_seed": first.get("source_seed"),
                "source_frame_index": first.get("source_frame_index"),
                "first_action": action,
                "mean_score_delta": float(mean(deltas)) if deltas else 0.0,
                "median_score_delta": float(median(deltas)) if deltas else 0.0,
                "high_score_delta": int(max(deltas)) if deltas else 0,
            }
        )
        out.append(summary)
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "overall": _summarize_rows(rows),
        "by_outcome": _group_summary(rows, "outcome"),
        "by_first_action": _group_summary(rows, "first_action"),
        "by_source_key": _group_summary(rows, "source_key"),
        "by_source_action": _source_action_summary(rows),
        "preview_after_first_label": dict(Counter(str(row.get("preview_after_first_label")) for row in rows)),
    }


def write_html(path: Path, payload: dict[str, Any]) -> None:
    summary = payload.get("summary", {})
    overall = summary.get("overall", {}) if isinstance(summary, dict) else {}
    source_actions = summary.get("by_source_action", []) if isinstance(summary, dict) else []

    def cell(value: object) -> str:
        return escape("" if value is None else str(value))

    rows = []
    for row in source_actions if isinstance(source_actions, list) else []:
        if not isinstance(row, dict):
            continue
        rows.append(
            "<tr>"
            f"<td>{cell(row.get('source_key'))}</td>"
            f"<td>{cell(row.get('first_action'))}</td>"
            f"<td>{cell(row.get('records'))}</td>"
            f"<td>{cell(row.get('success'))}</td>"
            f"<td>{cell(row.get('reached_6144'))}</td>"
            f"<td>{cell(row.get('raw_adjacent_1536'))}</td>"
            f"<td>{cell(row.get('first_large_ge_768'))}</td>"
            f"<td>{cell(round(float((row.get('rates') or {}).get('success', 0.0)), 3))}</td>"
            f"<td>{cell(round(float(row.get('mean_score_delta', 0.0)), 1))}</td>"
            "</tr>"
        )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>First-Action Path Forensics</title>
  <style>
    :root {{ color-scheme: dark; --bg: #101318; --panel: #171d24; --line: #34404d; --text: #edf2f7; --muted: #aab6c2; --gold: #f2c14e; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--text); font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ width: min(1180px, calc(100vw - 32px)); margin: 0 auto; padding: 24px 0 40px; }}
    h1 {{ margin: 0 0 8px; font-size: 24px; letter-spacing: 0; }}
    .muted {{ color: var(--muted); }}
    .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 18px 0; }}
    .metric {{ border: 1px solid var(--line); border-radius: 8px; background: var(--panel); padding: 12px; }}
    .label {{ color: var(--muted); font-size: 12px; }}
    .value {{ margin-top: 6px; font-size: 24px; font-weight: 700; color: var(--gold); }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
  </style>
</head>
<body>
  <main>
    <h1>First-Action Path Forensics</h1>
    <p class="muted">Joined large-preview and support-chain diagnostics for forced first-action continuations.</p>
    <section class="summary">
      <div class="metric"><div class="label">Records</div><div class="value">{cell(overall.get('records'))}</div></div>
      <div class="metric"><div class="label">Success</div><div class="value">{cell(overall.get('success'))}</div></div>
      <div class="metric"><div class="label">Reached 6144</div><div class="value">{cell(overall.get('reached_6144'))}</div></div>
      <div class="metric"><div class="label">Raw Adjacent 1536</div><div class="value">{cell(overall.get('raw_adjacent_1536'))}</div></div>
      <div class="metric"><div class="label">Large >= 768</div><div class="value">{cell(overall.get('first_large_ge_768'))}</div></div>
    </section>
    <table>
      <thead><tr><th>Source</th><th>Action</th><th>N</th><th>Success</th><th>6144</th><th>Adj 1536</th><th>Large >=768</th><th>P(success)</th><th>Mean delta</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </main>
</body>
</html>
"""
    path.write_text(html)


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    large_records = _load_records(args.large_preview_records)
    support_cases = _load_cases(args.support_chain_json)
    rows = _join_records(large_records, support_cases)
    payload = {
        "version": 1,
        "kind": "first_action_path_forensics",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "large_preview_records": str(args.large_preview_records),
        "support_chain_json": str(args.support_chain_json),
        "summary": summarize(rows),
        "records": rows,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload["json"] = str(args.out_dir / "first_action_path_forensics.json")
    payload["records_json"] = str(args.out_dir / "records.json")
    payload["html"] = str(args.out_dir / "first_action_path_forensics.html")
    write_json(args.out_dir / "first_action_path_forensics.json", payload)
    write_json(args.out_dir / "records.json", rows)
    write_html(args.out_dir / "first_action_path_forensics.html", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--large-preview-records", type=Path, required=True)
    parser.add_argument("--support-chain-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = run_from_args(args)
    print(json.dumps(payload["summary"]["overall"], indent=2, sort_keys=True))
    print(f"json={payload['json']}")
    print(f"records={payload['records_json']}")
    print(f"html={payload['html']}")


if __name__ == "__main__":
    main()
