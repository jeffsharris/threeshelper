"""Audit raw duplicate-1536 geometry against adjacent-1536 rollout labels."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from threes_rl.run_artifacts import write_json


def _extract_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = payload.get("records") or payload.get("target_records") or payload.get("frontier_records")
    else:
        records = None
    if not isinstance(records, list):
        raise ValueError("JSON payload does not contain records[]")
    return [dict(record) for record in records if isinstance(record, dict)]


def load_records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(Path(path).read_text())
        records.extend(_extract_records(payload))
    return records


def _extract_action_summary(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = payload.get("action_summary")
    else:
        rows = None
    if not isinstance(rows, list):
        raise ValueError("JSON payload does not contain action_summary[]")
    return [dict(row) for row in rows if isinstance(row, dict)]


def load_action_summary(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(_extract_action_summary(json.loads(Path(path).read_text())))
    return rows


def aggregate_action_summary(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_case: dict[str, dict[str, Any]] = {}
    for row in rows:
        case_id = row.get("case_id")
        if case_id is None:
            continue
        case = by_case.setdefault(
            str(case_id),
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
        rollouts = int(row.get("valid_rollouts", row.get("rollouts", 0)) or 0)
        rate = float(row.get("target_rate", 0.0) or 0.0)
        action = str(row.get("first_action", "unknown"))
        case["target_hits"] += hits
        case["valid_rollouts"] += rollouts
        case["actions"].append(
            {
                "first_action": action,
                "target_hits": hits,
                "valid_rollouts": rollouts,
                "target_rate": rate,
                "mean_score_delta": float(row.get("mean_score_delta", 0.0) or 0.0),
            }
        )
        if rate > float(case["best_action_rate"]):
            case["best_action"] = action
            case["best_action_rate"] = rate
            case["best_action_hits"] = hits
            case["best_action_rollouts"] = rollouts
    for case in by_case.values():
        rollouts = int(case.get("valid_rollouts", 0) or 0)
        hits = int(case.get("target_hits", 0) or 0)
        case["target_rate"] = float(hits / rollouts) if rollouts else 0.0
        case["positive"] = hits > 0
    return by_case


def _positions(board: np.ndarray, tile: int) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for r, c in zip(*np.where(board == int(tile)), strict=False)]


def _line_blockers(board: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> list[int]:
    ar, ac = a
    br, bc = b
    if ar == br:
        lo, hi = sorted((ac, bc))
        return [int(board[ar, c]) for c in range(lo + 1, hi)]
    if ac == bc:
        lo, hi = sorted((ar, br))
        return [int(board[r, ac]) for r in range(lo + 1, hi)]
    return []


def pair_geometry(board_payload: Any, *, tile: int = 1536) -> dict[str, Any]:
    board = np.asarray(board_payload, dtype=np.int32)
    positions = _positions(board, tile)
    if len(positions) < 2:
        return {
            "tile": int(tile),
            "count": len(positions),
            "positions": [[r, c] for r, c in positions],
            "has_pair": False,
            "relation": "missing_pair",
            "min_distance": None,
            "same_line": False,
            "line_blockers": [],
            "has_3072_between": False,
        }

    def pair_key(pair: tuple[tuple[int, int], tuple[int, int]]) -> tuple[int, int, tuple[tuple[int, int], tuple[int, int]]]:
        a, b = pair
        dist = abs(a[0] - b[0]) + abs(a[1] - b[1])
        same_line = int(not (a[0] == b[0] or a[1] == b[1]))
        return (dist, same_line, tuple(sorted(pair)))

    a, b = min(combinations(positions, 2), key=pair_key)
    dist = abs(a[0] - b[0]) + abs(a[1] - b[1])
    same_row = a[0] == b[0]
    same_col = a[1] == b[1]
    blockers = _line_blockers(board, a, b)
    if dist == 1:
        relation = "adjacent"
    elif same_row:
        relation = f"same_row_gap{abs(a[1] - b[1]) - 1}"
    elif same_col:
        relation = f"same_col_gap{abs(a[0] - b[0]) - 1}"
    elif abs(a[0] - b[0]) == 1 and abs(a[1] - b[1]) == 1:
        relation = "diagonal_touch"
    else:
        relation = "diagonal_far"
    return {
        "tile": int(tile),
        "count": len(positions),
        "positions": [[r, c] for r, c in sorted(positions)],
        "has_pair": True,
        "pair": [[int(a[0]), int(a[1])], [int(b[0]), int(b[1])]],
        "relation": relation,
        "min_distance": int(dist),
        "same_line": bool(same_row or same_col),
        "same_row": bool(same_row),
        "same_col": bool(same_col),
        "line_blockers": blockers,
        "line_blocker_signature": ",".join(str(value) for value in blockers) if blockers else "",
        "has_3072_between": any(value == 3072 for value in blockers),
    }


def _record_id(record: dict[str, Any]) -> str:
    value = record.get("id")
    if value is not None:
        return str(value)
    return str(record.get("source_replay") or record.get("root_replay") or id(record))


def _state(record: dict[str, Any]) -> dict[str, Any]:
    state = record.get("state")
    return state if isinstance(state, dict) else {}


def build_rows(records: Iterable[dict[str, Any]], labels: dict[str, dict[str, Any]], *, tile: int = 1536) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        state = _state(record)
        board = state.get("board")
        if board is None:
            continue
        geometry = pair_geometry(board, tile=tile)
        label = labels.get(_record_id(record), {})
        features = record.get("features") if isinstance(record.get("features"), dict) else {}
        legal_mask = state.get("legal_mask") if isinstance(state.get("legal_mask"), list) else []
        row = {
            "id": _record_id(record),
            "root_seed": str(record.get("root_seed")),
            "root_origin": record.get("root_origin"),
            "root_policy_family": record.get("root_policy_family"),
            "source_replay": record.get("source_replay"),
            "root_replay": record.get("root_replay"),
            "move_count": int(state.get("move_count", record.get("move_count", 0)) or 0),
            "score": int(state.get("score", record.get("score", 0)) or 0),
            "game_over": bool(state.get("game_over", False)),
            "legal_count": int(sum(1 for value in legal_mask if value)),
            "empty_count": int(features.get("empty_count", 0) or 0),
            "preview": features.get("preview") or (state.get("preview") or {}).get("label"),
            "large_pending": bool(features.get("large_pending", False)),
            "safe_smalls_until_large_possible": features.get("safe_smalls_until_large_possible"),
            "top_left": features.get("top_left"),
            "corner_risk": features.get("corner_risk"),
            "target_hits": int(label.get("target_hits", 0) or 0),
            "valid_rollouts": int(label.get("valid_rollouts", 0) or 0),
            "target_rate": float(label.get("target_rate", 0.0) or 0.0),
            "best_action": label.get("best_action"),
            "best_action_rate": float(label.get("best_action_rate", 0.0) or 0.0),
            "positive": bool(label.get("positive", False)),
            "label_missing": _record_id(record) not in labels,
            **{f"geom_{key}": value for key, value in geometry.items()},
        }
        rows.append(row)
    return rows


def _rate(hits: int, rollouts: int) -> float:
    return float(hits / rollouts) if rollouts else 0.0


def _group_summary(rows: Iterable[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key))].append(row)
    out: list[dict[str, Any]] = []
    for value in sorted(grouped):
        items = grouped[value]
        hits = sum(int(row.get("target_hits", 0) or 0) for row in items)
        rollouts = sum(int(row.get("valid_rollouts", 0) or 0) for row in items)
        best_rates = [float(row.get("best_action_rate", 0.0) or 0.0) for row in items if not row.get("label_missing")]
        out.append(
            {
                key: value,
                "records": len(items),
                "labeled_records": sum(0 if row.get("label_missing") else 1 for row in items),
                "positive_records": sum(1 for row in items if row.get("positive")),
                "target_hits": hits,
                "valid_rollouts": rollouts,
                "target_rate": _rate(hits, rollouts),
                "mean_best_action_rate": float(sum(best_rates) / len(best_rates)) if best_rates else 0.0,
                "start_adjacent": sum(1 for row in items if row.get("geom_relation") == "adjacent"),
                "nonadjacent": sum(1 for row in items if row.get("geom_relation") != "adjacent"),
                "relations": dict(Counter(str(row.get("geom_relation")) for row in items)),
                "distances": dict(Counter(str(row.get("geom_min_distance")) for row in items)),
                "with_3072_between": sum(1 for row in items if row.get("geom_has_3072_between")),
            }
        )
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    hits = sum(int(row.get("target_hits", 0) or 0) for row in rows)
    rollouts = sum(int(row.get("valid_rollouts", 0) or 0) for row in rows)
    labeled = [row for row in rows if not row.get("label_missing")]
    best_rates = [float(row.get("best_action_rate", 0.0) or 0.0) for row in labeled]
    return {
        "records": len(rows),
        "labeled_records": len(labeled),
        "missing_labels": len(rows) - len(labeled),
        "positive_records": sum(1 for row in labeled if row.get("positive")),
        "target_hits": hits,
        "valid_rollouts": rollouts,
        "target_rate": _rate(hits, rollouts),
        "mean_best_action_rate": float(sum(best_rates) / len(best_rates)) if best_rates else 0.0,
        "start_adjacent": sum(1 for row in rows if row.get("geom_relation") == "adjacent"),
        "nonadjacent": sum(1 for row in rows if row.get("geom_relation") != "adjacent"),
        "by_root": _group_summary(rows, "root_seed"),
        "by_relation": _group_summary(rows, "geom_relation"),
        "by_distance": _group_summary(rows, "geom_min_distance"),
    }


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def write_report(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    lines = [
        "# Duplicate-1536 Geometry Audit",
        "",
        f"- records: `{summary['records']}`",
        f"- labeled records: `{summary['labeled_records']}`",
        f"- positives: `{summary['positive_records']}`",
        f"- hits / rollouts: `{summary['target_hits']} / {summary['valid_rollouts']}` (`{_pct(summary['target_rate'])}`)",
        f"- start-adjacent: `{summary['start_adjacent']}`; non-adjacent: `{summary['nonadjacent']}`",
        "",
        "## By Root",
        "",
        "| root | records | labeled | positives | hits / rollouts | hit rate | mean best-action | start adjacent | relations | distances | 3072-between |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |",
    ]
    for row in summary["by_root"]:
        lines.append(
            f"| {row['root_seed']} | {row['records']} | {row['labeled_records']} | {row['positive_records']} | "
            f"{row['target_hits']} / {row['valid_rollouts']} | {_pct(row['target_rate'])} | "
            f"{_pct(row['mean_best_action_rate'])} | {row['start_adjacent']} | "
            f"`{json.dumps(row['relations'], sort_keys=True)}` | "
            f"`{json.dumps(row['distances'], sort_keys=True)}` | {row['with_3072_between']} |"
        )
    lines.extend(
        [
            "",
            "## By Relation",
            "",
            "| relation | records | labeled | positives | hits / rollouts | hit rate | mean best-action |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["by_relation"]:
        lines.append(
            f"| {row['geom_relation']} | {row['records']} | {row['labeled_records']} | {row['positive_records']} | "
            f"{row['target_hits']} / {row['valid_rollouts']} | {_pct(row['target_rate'])} | "
            f"{_pct(row['mean_best_action_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Lowest-Rate Rows",
            "",
            "| root | id | relation | distance | blockers | hits / rollouts | best action | best rate |",
            "| --- | --- | --- | ---: | --- | ---: | --- | ---: |",
        ]
    )
    sorted_rows = sorted(
        (row for row in payload["rows"] if not row.get("label_missing")),
        key=lambda row: (float(row.get("best_action_rate", 0.0)), float(row.get("target_rate", 0.0)), str(row.get("id"))),
    )
    for row in sorted_rows[:20]:
        lines.append(
            f"| {row['root_seed']} | `{row['id']}` | {row['geom_relation']} | "
            f"{row.get('geom_min_distance')} | `{row.get('geom_line_blocker_signature', '')}` | "
            f"{row['target_hits']} / {row['valid_rollouts']} | {row.get('best_action')} | {_pct(row['best_action_rate'])} |"
        )
    missing = [row for row in payload["rows"] if row.get("label_missing")]
    if missing:
        by_root = Counter(str(row.get("root_seed")) for row in missing)
        lines.extend(["", "## Missing Labels", "", f"`{len(missing)}` records had no rollout label: `{json.dumps(dict(sorted(by_root.items())), sort_keys=True)}`."])
    path.write_text("\n".join(lines) + "\n")


def run_audit(
    *,
    records_json: list[Path],
    frontier_json: list[Path],
    out_dir: Path,
    tile: int = 1536,
) -> dict[str, Any]:
    records = load_records(records_json)
    labels = aggregate_action_summary(load_action_summary(frontier_json))
    rows = build_rows(records, labels, tile=tile)
    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "records_json": [str(path) for path in records_json],
        "frontier_json": [str(path) for path in frontier_json],
        "tile": int(tile),
        "summary": summarize(rows),
        "rows": rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    payload["json"] = str(out_dir / "duplicate1536_geometry_audit.json")
    payload["report"] = str(out_dir / "report.md")
    write_json(out_dir / "duplicate1536_geometry_audit.json", payload)
    write_report(out_dir / "report.md", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-json", type=Path, action="append", required=True)
    parser.add_argument("--frontier-json", type=Path, action="append", required=True)
    parser.add_argument("--tile", type=int, default=1536)
    parser.add_argument("--out-dir", type=Path, default=Path("threes_rl/runs/forensics/duplicate1536_geometry/latest"))
    args = parser.parse_args()
    payload = run_audit(
        records_json=args.records_json,
        frontier_json=args.frontier_json,
        out_dir=args.out_dir,
        tile=args.tile,
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"json={payload['json']}")
    print(f"report={payload['report']}")


if __name__ == "__main__":
    main()
