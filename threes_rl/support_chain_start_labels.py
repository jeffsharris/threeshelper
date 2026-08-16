"""Join support-chain gate outcomes onto start-state records."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from html import escape
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from threes_rl.run_artifacts import safe_name, write_json
from threes_rl.support_chain_gate_report import STAGES


STAGE_NAMES = tuple(stage for stage, _analysis_key in STAGES)
SOURCE_KEY_PREFIX = "__source_frame__:"


def _flatten_paths(path_groups: list[list[Path]] | None) -> list[Path]:
    if not path_groups:
        return []
    return [path for group in path_groups for path in group]


def _load_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text())
    records = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError(f"{path} does not contain records[]")
    return [record for record in records if isinstance(record, dict)]


def load_start_records(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in paths:
        for record in _load_records(Path(path)):
            record_id = record.get("id")
            if record_id is not None:
                records[str(record_id)] = record
            source_key = _source_frame_key(record)
            if source_key is not None and source_key not in records:
                records[source_key] = record
    return records


def _source_frame_key(record: dict[str, Any]) -> str | None:
    source_replay = record.get("source_replay")
    frame_index = record.get("source_frame_index", record.get("frame_index"))
    if source_replay is None or frame_index is None:
        return None
    try:
        frame = int(frame_index)
    except (TypeError, ValueError):
        return None
    return f"{SOURCE_KEY_PREFIX}{source_replay}::{frame}"


def _find_start_record(
    start_records: dict[str, dict[str, Any]],
    row: dict[str, Any],
) -> dict[str, Any] | None:
    start_id = row.get("start_case_id")
    if start_id is not None:
        record = start_records.get(str(start_id))
        if record is not None:
            return record
    source_key = _source_frame_key(row)
    if source_key is not None:
        return start_records.get(source_key)
    return None


def _load_gate(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a support-chain gate report")
    return payload


def load_gate_rows(paths: Iterable[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    start_rows: list[dict[str, Any]] = []
    for path in paths:
        payload = _load_gate(Path(path))
        raw_rows = payload.get("rows")
        if isinstance(raw_rows, list):
            for idx, row in enumerate(raw_rows):
                if isinstance(row, dict):
                    rows.append({**row, "_gate_json": str(path), "_gate_row_index": int(idx)})
        raw_start_rows = payload.get("start_rows")
        if isinstance(raw_start_rows, list):
            for idx, row in enumerate(raw_start_rows):
                if isinstance(row, dict):
                    start_rows.append({**row, "_gate_json": str(path), "_gate_start_index": int(idx)})
    return rows, start_rows


def _clean_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if not str(key).startswith("_")}


def _digest(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]


def _label_id(start_id: str, *, target_stage: str, outcome: str, source_key: dict[str, Any]) -> str:
    return safe_name(
        f"support_chain_{target_stage}_{outcome}_{start_id}_{_digest(source_key)}",
        max_length=160,
    )


def _rate(row: dict[str, Any], target_stage: str) -> float:
    try:
        return float(row.get(f"p_{target_stage}", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _bool_outcome(row: dict[str, Any], target_stage: str) -> bool:
    return bool(row.get(target_stage))


def _record_with_label(
    start_record: dict[str, Any],
    *,
    target_stage: str,
    outcome: str,
    target_rate: float | None,
    source: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    start_id = str(start_record.get("id"))
    record = _clean_record(start_record)
    record.update(
        {
            "id": _label_id(
                start_id,
                target_stage=target_stage,
                outcome=outcome,
                source_key={
                    "mode": mode,
                    "target_stage": target_stage,
                    "start_id": start_id,
                    "gate_json": source.get("_gate_json"),
                    "gate_row_index": source.get("_gate_row_index"),
                    "gate_start_index": source.get("_gate_start_index"),
                    "replay_json": source.get("replay_json"),
                },
            ),
            "base_start_id": start_id,
            "kind": "support_chain_start_label",
            "outcome": outcome,
            "target_milestone": target_stage,
            "target_tile": 6144,
            "label_mode": mode,
            "label_target_stage": target_stage,
            "label_target_rate": target_rate,
            "label_gate_json": source.get("_gate_json"),
            "label_replay_json": source.get("replay_json"),
            "label_score_delta": source.get("score_delta", source.get("high_score_delta")),
            "label_max_tile_excl_starter": source.get("max_tile_excl_starter"),
        }
    )
    return record


def build_continuation_labels(
    rows: list[dict[str, Any]],
    *,
    start_records: dict[str, dict[str, Any]],
    target_stage: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    labels: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    for row in rows:
        start_record = _find_start_record(start_records, row)
        if start_record is None:
            rejected["missing_start"] += 1
            continue
        success = _bool_outcome(row, target_stage)
        labels.append(
            _record_with_label(
                start_record,
                target_stage=target_stage,
                outcome="success" if success else "failure",
                target_rate=1.0 if success else 0.0,
                source=row,
                mode="continuation",
            )
        )
    return labels, dict(rejected)


def build_start_labels(
    start_rows: list[dict[str, Any]],
    *,
    start_records: dict[str, dict[str, Any]],
    target_stage: str,
    success_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    labels: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    seen: set[tuple[str, str]] = set()
    for row in start_rows:
        start_id = str(row.get("start_case_id"))
        key = (str(row.get("_gate_json")), start_id)
        if key in seen:
            rejected["duplicate_start"] += 1
            continue
        seen.add(key)
        start_record = _find_start_record(start_records, row)
        if start_record is None:
            rejected["missing_start"] += 1
            continue
        target_rate = _rate(row, target_stage)
        labels.append(
            _record_with_label(
                start_record,
                target_stage=target_stage,
                outcome="success" if target_rate >= float(success_threshold) else "failure",
                target_rate=target_rate,
                source=row,
                mode="start_threshold",
            )
        )
    return labels, dict(rejected)


def summarize(labels: list[dict[str, Any]], *, target_stage: str, mode: str, rejected: dict[str, int]) -> dict[str, Any]:
    rates = [
        float(label.get("label_target_rate"))
        for label in labels
        if label.get("label_target_rate") is not None
    ]
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "target_stage": target_stage,
        "mode": mode,
        "records": len(labels),
        "successes": sum(str(label.get("outcome")) == "success" for label in labels),
        "failures": sum(str(label.get("outcome")) == "failure" for label in labels),
        "success_rate": (
            sum(str(label.get("outcome")) == "success" for label in labels) / len(labels)
            if labels
            else 0.0
        ),
        "source_starts": len({str(label.get("base_start_id")) for label in labels}),
        "by_base_start": dict(Counter(str(label.get("base_start_id")) for label in labels)),
        "by_outcome": dict(Counter(str(label.get("outcome")) for label in labels)),
        "mean_target_rate": float(mean(rates)) if rates else 0.0,
        "rejected": rejected,
    }


def write_html(path: Path, payload: dict[str, Any]) -> None:
    summary = payload.get("summary", {})

    def cell(value: object) -> str:
        return escape(str(value))

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Support-Chain Start Labels</title>
  <style>
    :root {{ color-scheme: dark; --bg:#101318; --panel:#171d24; --line:#34404d; --text:#edf2f7; --muted:#aab6c2; --gold:#f2c14e; }}
    body {{ margin:0; background:var(--bg); color:var(--text); font-family:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ width:min(980px, calc(100vw - 32px)); margin:0 auto; padding:24px 0 40px; }}
    h1 {{ margin:0 0 8px; font-size:24px; }}
    .muted {{ color:var(--muted); }}
    .cards {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(150px, 1fr)); gap:10px; margin:18px 0; }}
    .card, .panel {{ border:1px solid var(--line); background:var(--panel); border-radius:8px; padding:12px; }}
    .label {{ color:var(--muted); font-size:11px; text-transform:uppercase; }}
    .value {{ margin-top:4px; color:var(--gold); font-size:22px; font-weight:800; }}
    pre {{ white-space:pre-wrap; overflow-wrap:anywhere; color:var(--muted); }}
  </style>
</head>
<body>
  <main>
    <h1>Support-Chain Start Labels</h1>
    <p class="muted">Start-state labels derived from support-chain gate outcomes.</p>
    <section class="cards">
      <div class="card"><div class="label">Records</div><div class="value">{cell(summary.get('records', 0))}</div></div>
      <div class="card"><div class="label">Starts</div><div class="value">{cell(summary.get('source_starts', 0))}</div></div>
      <div class="card"><div class="label">Success Rate</div><div class="value">{float(summary.get('success_rate', 0.0)):.1%}</div></div>
      <div class="card"><div class="label">Target</div><div class="value">{cell(summary.get('target_stage'))}</div></div>
    </section>
    <section class="panel"><pre>{escape(json.dumps(summary, indent=2, sort_keys=True))}</pre></section>
  </main>
</body>
</html>
"""
    path.write_text(html)


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    if args.target_stage not in STAGE_NAMES:
        raise ValueError(f"Unsupported target stage: {args.target_stage}")
    start_records = load_start_records(_flatten_paths(args.start_json))
    rows, start_rows = load_gate_rows(_flatten_paths(args.gate_json))
    if args.mode == "continuation":
        labels, rejected = build_continuation_labels(
            rows,
            start_records=start_records,
            target_stage=args.target_stage,
        )
    else:
        labels, rejected = build_start_labels(
            start_rows,
            start_records=start_records,
            target_stage=args.target_stage,
            success_threshold=args.success_threshold,
        )
    payload = {
        "version": 1,
        "kind": "support_chain_start_labels",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "gate_json": [str(path) for path in _flatten_paths(args.gate_json)],
        "start_json": [str(path) for path in _flatten_paths(args.start_json)],
        "target_stage": args.target_stage,
        "mode": args.mode,
        "success_threshold": float(args.success_threshold),
        "summary": summarize(labels, target_stage=args.target_stage, mode=args.mode, rejected=rejected),
        "records": labels,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload["json"] = str(args.out_dir / "support_chain_start_labels.json")
    payload["records_json"] = str(args.out_dir / "records.json")
    payload["html"] = str(args.out_dir / "support_chain_start_labels.html")
    write_json(args.out_dir / "support_chain_start_labels.json", payload)
    write_json(args.out_dir / "records.json", labels)
    write_html(args.out_dir / "support_chain_start_labels.html", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-json", type=Path, nargs="+", action="append", required=True)
    parser.add_argument("--start-json", type=Path, nargs="+", action="append", required=True)
    parser.add_argument("--target-stage", choices=STAGE_NAMES, default="raw_adjacent_1536")
    parser.add_argument("--mode", choices=["continuation", "start-threshold"], default="continuation")
    parser.add_argument("--success-threshold", type=float, default=0.25)
    parser.add_argument("--out-dir", type=Path, default=Path("threes_rl/runs/forensics/support_chain_start_labels/latest"))
    args = parser.parse_args()
    payload = run_from_args(args)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"json={payload['json']}")
    print(f"records={payload['records_json']}")
    print(f"html={payload['html']}")


if __name__ == "__main__":
    main()
