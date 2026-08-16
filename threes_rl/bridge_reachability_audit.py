"""Audit bridge states against downstream support-preserving conversion labels."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from threes_rl.run_artifacts import write_json
from threes_rl.transition_reachability_audit import (
    build_rows,
    feature_summary,
    grouped_logistic_probe,
    load_records,
    summarize,
    write_html,
)


def _records_from_path(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text())
    records = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError(f"{path} does not contain records[]")
    return [dict(record) for record in records if isinstance(record, dict)]


def _load_action_summary(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text())
    action_summary = payload.get("action_summary")
    if not isinstance(action_summary, list):
        raise ValueError(f"{path} does not contain action_summary[]")
    return [dict(row) for row in action_summary if isinstance(row, dict)]


def _aggregate_actions(action_rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
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
                "actions": [],
                "best_action": None,
                "best_action_rate": 0.0,
                "best_action_hits": 0,
                "best_action_rollouts": 0,
            },
        )
        hits = int(row.get("target_hits", 0) or 0)
        valid = int(row.get("valid_rollouts", row.get("rollouts", 0)) or 0)
        rate = float(row.get("target_rate", 0.0) or 0.0)
        action = str(row.get("first_action", "unknown"))
        item["target_hits"] += hits
        item["valid_rollouts"] += valid
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
        valid = int(item.get("valid_rollouts", 0))
        hits = int(item.get("target_hits", 0))
        item["target_rate"] = float(hits / valid) if valid else 0.0
        item["positive"] = hits > 0
    return by_case


def label_bridge_records(
    records: list[dict[str, Any]],
    *,
    action_summary: list[dict[str, Any]],
    min_best_action_rate: float = 0.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels = _aggregate_actions(action_summary)
    labeled: list[dict[str, Any]] = []
    missing = 0
    for record in records:
        case_id = str(record.get("id"))
        label = labels.get(case_id)
        if label is None:
            missing += 1
            continue
        positive = bool(label.get("positive")) and float(label.get("best_action_rate", 0.0)) >= float(min_best_action_rate)
        row = dict(record)
        features = dict(row.get("features") or {})
        if features.get("legal_count") is not None:
            row["legal_count"] = int(features["legal_count"])
        row["outcome"] = "success" if positive else "failure"
        row["bridge_label"] = label
        row["target_tile"] = row.get("target_tile", 1536)
        row["source_group"] = str(row.get("root_replay") or row.get("ancestry_key") or row.get("source_replay") or case_id)
        labeled.append(row)
    summary = {
        "records": len(records),
        "labeled_records": len(labeled),
        "missing_labels": int(missing),
        "positive_records": sum(1 for record in labeled if record.get("outcome") == "success"),
        "negative_records": sum(1 for record in labeled if record.get("outcome") == "failure"),
        "min_best_action_rate": float(min_best_action_rate),
        "by_root_seed_outcome": {
            str(key): int(value)
            for key, value in Counter((record.get("root_seed"), record.get("outcome")) for record in labeled).items()
        },
    }
    return labeled, summary


def run_audit(
    *,
    records_json: list[Path],
    frontier_json: Path | list[Path],
    out_dir: Path,
    min_best_action_rate: float = 0.0,
    group_by: str = "source-replay",
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for path in records_json:
        records.extend(_records_from_path(Path(path)))
    frontier_paths = [frontier_json] if isinstance(frontier_json, Path) else list(frontier_json)
    action_summary: list[dict[str, Any]] = []
    for path in frontier_paths:
        action_summary.extend(_load_action_summary(Path(path)))
    labeled, label_summary = label_bridge_records(
        records,
        action_summary=action_summary,
        min_best_action_rate=min_best_action_rate,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    labeled_path = out_dir / "labeled_bridge_records.json"
    write_json(labeled_path, labeled)
    rows = build_rows(load_records([labeled_path]), target_tile=None, group_by=group_by)
    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "records_json": [str(path) for path in records_json],
        "frontier_json": [str(path) for path in frontier_paths],
        "group_by": group_by,
        "label_summary": label_summary,
        "summary": summarize(rows),
        "feature_summary": feature_summary(rows),
        "grouped_logistic_probe": grouped_logistic_probe(rows),
        "rows": rows,
    }
    payload["json"] = str(out_dir / "bridge_reachability_audit.json")
    payload["html"] = str(out_dir / "bridge_reachability_audit.html")
    payload["labeled_records_json"] = str(labeled_path)
    write_json(out_dir / "bridge_reachability_audit.json", payload)
    write_html(out_dir / "bridge_reachability_audit.html", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-json", type=Path, action="append", required=True)
    parser.add_argument("--frontier-json", type=Path, action="append", required=True)
    parser.add_argument("--min-best-action-rate", type=float, default=0.0)
    parser.add_argument(
        "--group-by",
        choices=("auto", "source-group", "source-replay", "original-replay"),
        default="source-replay",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("threes_rl/runs/forensics/bridge_reachability/latest"),
    )
    args = parser.parse_args()
    payload = run_audit(
        records_json=args.records_json,
        frontier_json=args.frontier_json,
        out_dir=args.out_dir,
        min_best_action_rate=args.min_best_action_rate,
        group_by=args.group_by,
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(json.dumps(payload["label_summary"], indent=2, sort_keys=True))
    print(json.dumps(payload["grouped_logistic_probe"], indent=2, sort_keys=True))
    print(f"json={payload['json']}")
    print(f"html={payload['html']}")


if __name__ == "__main__":
    main()
