"""Select live rare-event frontier states for the next milestone audit."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from html import escape
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from threes_rl.run_artifacts import write_json

SUPPORTED_NEXT_TARGETS = (
    "raw_duplicate_768",
    "raw_adjacent_768",
    "raw_one_1536",
    "raw_adjacent_768_with_1536",
    "raw_duplicate_1536",
    "raw_adjacent_1536",
    "second_3072",
    "reached_6144",
)
SUPPORTED_RANK_PROFILES = ("support", "air_survival")


def _load_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text())
    if isinstance(payload, dict):
        records = payload.get("frontier_records") or payload.get("records") or payload.get("selected_records")
    else:
        records = payload
    if not isinstance(records, list):
        raise ValueError(f"{path} does not contain a frontier/state records list")
    out: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        if isinstance(record, dict):
            row = dict(record)
            row.setdefault("_source_json", str(path))
            row.setdefault("_record_index", int(idx))
            out.append(row)
    return out


def load_records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        records.extend(_load_records(Path(path)))
    return records


def _feature(record: dict[str, Any], name: str, default: float = 0.0) -> float:
    features = record.get("features")
    value = features.get(name) if isinstance(features, dict) and name in features else record.get(name, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _state_dict(record: dict[str, Any]) -> dict[str, Any]:
    state = record.get("state")
    return state if isinstance(state, dict) else {}


def legal_count(record: dict[str, Any]) -> int:
    state = _state_dict(record)
    legal = state.get("legal_actions")
    if isinstance(legal, list):
        return len(legal)
    legal_mask = state.get("legal_mask")
    if isinstance(legal_mask, list):
        return sum(bool(value) for value in legal_mask)
    return int(_feature(record, "legal_count", 0.0))


def is_live_record(record: dict[str, Any]) -> bool:
    state = _state_dict(record)
    return not bool(state.get("game_over", record.get("game_over", False))) and legal_count(record) > 0


def root_key(record: dict[str, Any]) -> str:
    selector = record.get("frontier_selector")
    if isinstance(selector, dict) and selector.get("root_key") is not None:
        return str(selector["root_key"])
    return str(
        record.get("root_replay")
        or record.get("ancestry_key")
        or record.get("source_replay")
        or record.get("root_seed")
        or record.get("id")
        or "unknown"
    )


def source_key(record: dict[str, Any]) -> str:
    selector = record.get("frontier_selector")
    if isinstance(selector, dict) and selector.get("source_key") is not None:
        return str(selector["source_key"])
    return str(record.get("source_replay") or record.get("root_replay") or record.get("id") or "unknown")


def rank_tuple(record: dict[str, Any], *, next_target: str, rank_profile: str = "support") -> tuple[float, ...]:
    if rank_profile not in SUPPORTED_RANK_PROFILES:
        raise ValueError(f"Unsupported rank profile: {rank_profile}")
    empty_count = _feature(record, "empty_count")
    legal = float(legal_count(record))
    score_delta = float(record.get("score_delta") or 0.0)
    raw_count_1536 = _feature(record, "raw_count_1536")
    raw_count_3072 = _feature(record, "raw_count_3072")
    raw_dup = _feature(record, "raw_highest_duplicate_tile")
    raw_adj = _feature(record, "raw_highest_adjacent_pair_tile")
    max_tile = _feature(record, "max_tile_excl_starter", _feature(record, "max_tile"))
    if next_target == "raw_duplicate_768":
        raw_count_768 = _feature(record, "raw_count_768")
        if rank_profile == "air_survival":
            return (
                empty_count / 16.0,
                legal / 4.0,
                raw_count_768,
                raw_dup / 768.0,
                raw_adj / 768.0,
                score_delta / 200000.0,
            )
        return (raw_count_768, raw_dup / 768.0, raw_adj / 768.0, empty_count / 16.0, score_delta / 200000.0)
    if next_target == "raw_adjacent_768":
        raw_count_768 = _feature(record, "raw_count_768")
        has_adj_768 = 1.0 if raw_adj >= 768 else 0.0
        if rank_profile == "air_survival":
            return (
                empty_count / 16.0,
                legal / 4.0,
                has_adj_768,
                raw_count_768,
                raw_adj / 768.0,
                raw_dup / 768.0,
                score_delta / 200000.0,
            )
        return (has_adj_768, raw_count_768, raw_adj / 768.0, raw_dup / 768.0, empty_count / 16.0, score_delta / 200000.0)
    if next_target == "raw_one_1536":
        raw_count_768 = _feature(record, "raw_count_768")
        if rank_profile == "air_survival":
            return (
                empty_count / 16.0,
                legal / 4.0,
                raw_count_1536,
                raw_count_768,
                raw_dup / 1536.0,
                raw_adj / 768.0,
                score_delta / 200000.0,
            )
        return (raw_count_1536, raw_count_768, raw_dup / 1536.0, raw_adj / 768.0, empty_count / 16.0, score_delta / 200000.0)
    if next_target == "raw_adjacent_768_with_1536":
        raw_count_768 = _feature(record, "raw_count_768")
        has_adj_768 = 1.0 if raw_adj >= 768 else 0.0
        if rank_profile == "air_survival":
            return (
                empty_count / 16.0,
                legal / 4.0,
                has_adj_768,
                raw_count_1536,
                raw_count_768,
                raw_adj / 768.0,
                raw_dup / 1536.0,
                score_delta / 200000.0,
            )
        return (
            has_adj_768,
            raw_count_1536,
            raw_count_768,
            raw_adj / 768.0,
            raw_dup / 1536.0,
            empty_count / 16.0,
            score_delta / 200000.0,
        )
    if next_target == "raw_duplicate_1536":
        if rank_profile == "air_survival":
            return (
                empty_count / 16.0,
                legal / 4.0,
                raw_count_1536,
                raw_dup / 1536.0,
                raw_adj / 1536.0,
                score_delta / 200000.0,
            )
        return (raw_count_1536, raw_dup / 1536.0, raw_adj / 1536.0, empty_count / 16.0, score_delta / 200000.0)
    if next_target == "raw_adjacent_1536":
        if rank_profile == "air_survival":
            return (
                empty_count / 16.0,
                legal / 4.0,
                1.0 if raw_adj >= 1536 else 0.0,
                raw_count_1536,
                raw_adj / 1536.0,
                raw_dup / 1536.0,
                score_delta / 200000.0,
            )
        return (
            1.0 if raw_adj >= 1536 else 0.0,
            raw_count_1536,
            raw_adj / 1536.0,
            raw_dup / 1536.0,
            empty_count / 16.0,
            score_delta / 200000.0,
        )
    if next_target == "second_3072":
        if rank_profile == "air_survival":
            return (
                empty_count / 16.0,
                legal / 4.0,
                1.0 if raw_count_1536 >= 2 else 0.0,
                1.0 if raw_adj >= 1536 else 0.0,
                raw_count_3072,
                raw_dup / 3072.0,
                score_delta / 200000.0,
            )
        return (
            1.0 if raw_count_1536 >= 2 else 0.0,
            1.0 if raw_adj >= 1536 else 0.0,
            raw_count_3072,
            raw_dup / 3072.0,
            empty_count / 16.0,
            score_delta / 200000.0,
        )
    if next_target == "reached_6144":
        if rank_profile == "air_survival":
            return (
                empty_count / 16.0,
                legal / 4.0,
                1.0 if max_tile >= 3072 else 0.0,
                raw_count_3072,
                1.0 if raw_adj >= 3072 else 0.0,
                raw_dup / 3072.0,
                score_delta / 200000.0,
            )
        return (
            1.0 if max_tile >= 3072 else 0.0,
            raw_count_3072,
            1.0 if raw_adj >= 3072 else 0.0,
            raw_dup / 3072.0,
            empty_count / 16.0,
            score_delta / 200000.0,
        )
    raise ValueError(f"Unsupported next target: {next_target}")


def _parse_min_feature_filters(values: list[str] | None) -> dict[str, float]:
    filters: dict[str, float] = {}
    for raw in values or []:
        if "=" not in raw:
            raise ValueError(f"Expected name=value feature filter, got {raw!r}")
        name, value = raw.split("=", 1)
        filters[name.strip()] = float(value)
    return {name: value for name, value in filters.items() if name}


def _parse_max_feature_filters(values: list[str] | None) -> dict[str, float]:
    filters: dict[str, float] = {}
    for raw in values or []:
        if "=" not in raw:
            raise ValueError(f"Expected name=value feature filter, got {raw!r}")
        name, value = raw.split("=", 1)
        filters[name.strip()] = float(value)
    return {name: value for name, value in filters.items() if name}


def _passes_min_features(record: dict[str, Any], filters: dict[str, float]) -> bool:
    return all(_feature(record, name) >= threshold for name, threshold in filters.items())


def _passes_max_features(record: dict[str, Any], filters: dict[str, float]) -> bool:
    return all(_feature(record, name) <= threshold for name, threshold in filters.items())


def _target_reached_matches(record: dict[str, Any], mode: str) -> bool:
    if mode == "all":
        return True
    reached = bool(record.get("target_reached", False))
    return reached if mode == "true" else not reached


def select_frontier_records(
    records: Iterable[dict[str, Any]],
    *,
    next_target: str,
    target_reached_filter: str = "all",
    live_only: bool = True,
    min_features: dict[str, float] | None = None,
    max_features: dict[str, float] | None = None,
    root_filter: set[str] | None = None,
    rank_order: str = "top",
    rank_profile: str = "support",
    max_records: int = 0,
    max_per_root: int = 1,
    max_per_source: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    if rank_order not in ("top", "bottom"):
        raise ValueError(f"Unsupported rank_order: {rank_order}")
    if rank_profile not in SUPPORTED_RANK_PROFILES:
        raise ValueError(f"Unsupported rank_profile: {rank_profile}")
    rejected: Counter[str] = Counter()
    candidates: list[dict[str, Any]] = []
    min_features = min_features or {}
    max_features = max_features or {}
    for record in records:
        if live_only and not is_live_record(record):
            rejected["not_live"] += 1
            continue
        if not _target_reached_matches(record, target_reached_filter):
            rejected["target_reached_filter"] += 1
            continue
        if not _passes_min_features(record, min_features):
            rejected["min_feature_filter"] += 1
            continue
        if not _passes_max_features(record, max_features):
            rejected["max_feature_filter"] += 1
            continue
        root = root_key(record)
        if root_filter is not None and root not in root_filter:
            rejected["root_filter"] += 1
            continue
        row = {key: value for key, value in record.items() if not str(key).startswith("_")}
        ranking = rank_tuple(record, next_target=next_target, rank_profile=rank_profile)
        row["frontier_selector"] = {
            "next_target": next_target,
            "rank_profile": rank_profile,
            "rank_tuple": list(ranking),
            "root_key": root,
            "source_key": source_key(record),
            "legal_count": legal_count(record),
            "live": is_live_record(record),
            "rank_order": rank_order,
        }
        candidates.append(row)

    candidates.sort(
        key=lambda row: (
            tuple(row["frontier_selector"]["rank_tuple"]),
            int(row.get("score_delta") or 0),
            str(row.get("id")),
        ),
        reverse=True,
    )
    if rank_order == "bottom":
        candidates = list(reversed(candidates))
    selected: list[dict[str, Any]] = []
    root_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    for row in candidates:
        selector = row["frontier_selector"]
        root = str(selector["root_key"])
        source = str(selector["source_key"])
        if max_per_root > 0 and root_counts[root] >= int(max_per_root):
            rejected["max_per_root"] += 1
            continue
        if max_per_source > 0 and source_counts[source] >= int(max_per_source):
            rejected["max_per_source"] += 1
            continue
        selected.append(row)
        root_counts[root] += 1
        source_counts[source] += 1
        if max_records > 0 and len(selected) >= int(max_records):
            break

    for rank, row in enumerate(selected, start=1):
        row["frontier_selector"]["rank"] = int(rank)
    return selected, candidates, dict(rejected)


def _summary(
    *,
    source_paths: list[str],
    source_count: int,
    selected: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    rejected: dict[str, int],
    next_target: str,
    target_reached_filter: str,
    live_only: bool,
    min_features: dict[str, float],
    max_features: dict[str, float],
    root_filter: set[str] | None,
    rank_order: str,
    rank_profile: str,
    max_records: int,
    max_per_root: int,
    max_per_source: int,
) -> dict[str, Any]:
    rank_heads = [tuple(row["frontier_selector"]["rank_tuple"]) for row in selected]
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_paths": source_paths,
        "source_records": int(source_count),
        "candidate_records": len(candidates),
        "selected_records": len(selected),
        "next_target": next_target,
        "target_reached_filter": target_reached_filter,
        "live_only": bool(live_only),
        "min_features": min_features,
        "max_features": max_features,
        "root_filter_count": len(root_filter) if root_filter is not None else None,
        "rank_order": rank_order,
        "rank_profile": rank_profile,
        "max_records": int(max_records),
        "max_per_root": int(max_per_root),
        "max_per_source": int(max_per_source),
        "unique_roots": len({str(row["frontier_selector"]["root_key"]) for row in selected}),
        "unique_sources": len({str(row["frontier_selector"]["source_key"]) for row in selected}),
        "by_root_origin": dict(Counter(str(row.get("root_origin", "unknown")) for row in selected)),
        "by_root_policy_family": dict(Counter(str(row.get("root_policy_family", "unknown")) for row in selected)),
        "mean_empty_count": float(mean(_feature(row, "empty_count") for row in selected)) if selected else 0.0,
        "best_rank_tuple": list(rank_heads[0]) if rank_heads else [],
        "rejected": rejected,
    }


def load_root_filter(paths: Iterable[Path]) -> set[str]:
    roots: set[str] = set()
    for record in load_records(paths):
        roots.add(root_key(record))
    return roots


def write_html(path: Path, payload: dict[str, Any]) -> None:
    summary = payload.get("summary", {})
    selected = payload.get("selected_records", [])

    def cell(value: object) -> str:
        return escape(str(value))

    rows = []
    for record in selected[:100] if isinstance(selected, list) else []:
        selector = record.get("frontier_selector", {})
        features = record.get("features", {})
        if not isinstance(selector, dict):
            selector = {}
        if not isinstance(features, dict):
            features = {}
        rows.append(
            "<tr>"
            f"<td>{cell(selector.get('rank'))}</td>"
            f"<td>{cell(record.get('id'))}</td>"
            f"<td>{cell(record.get('root_seed'))}</td>"
            f"<td>{cell(record.get('root_policy_family'))}</td>"
            f"<td>{cell(features.get('empty_count'))}</td>"
            f"<td>{cell(features.get('raw_count_1536'))}</td>"
            f"<td>{cell(features.get('raw_count_3072'))}</td>"
            f"<td>{cell(features.get('raw_highest_duplicate_tile'))}</td>"
            f"<td>{cell(features.get('raw_highest_adjacent_pair_tile'))}</td>"
            f"<td>{cell(selector.get('rank_tuple'))}</td>"
            "</tr>"
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Frontier Selector</title>
  <style>
    :root {{ color-scheme: dark; --bg: #101318; --panel: #171d24; --line: #34404d; --text: #edf2f7; --muted: #aab6c2; --gold: #f2c14e; }}
    body {{ margin: 0; background: var(--bg); color: var(--text); font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ width: min(1180px, calc(100vw - 32px)); margin: 0 auto; padding: 24px 0 40px; }}
    h1 {{ margin: 0 0 8px; font-size: 24px; }}
    .muted {{ color: var(--muted); }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 18px 0; }}
    .card {{ border: 1px solid var(--line); background: var(--panel); border-radius: 8px; padding: 12px; }}
    .label {{ color: var(--muted); font-size: 11px; text-transform: uppercase; }}
    .value {{ margin-top: 4px; color: var(--gold); font-size: 22px; font-weight: 800; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 7px 8px; text-align: right; vertical-align: top; }}
    th:nth-child(2), td:nth-child(2), th:nth-child(4), td:nth-child(4), th:last-child, td:last-child {{ text-align: left; overflow-wrap: anywhere; }}
  </style>
</head>
<body>
  <main>
    <h1>Frontier Selector</h1>
    <p class="muted">Live rare-event frontier states ranked for the next milestone.</p>
    <section class="cards">
      <div class="card"><div class="label">Selected</div><div class="value">{cell(summary.get('selected_records', 0))}</div></div>
      <div class="card"><div class="label">Candidates</div><div class="value">{cell(summary.get('candidate_records', 0))}</div></div>
      <div class="card"><div class="label">Roots</div><div class="value">{cell(summary.get('unique_roots', 0))}</div></div>
      <div class="card"><div class="label">Next Target</div><div class="value">{cell(summary.get('next_target'))}</div></div>
    </section>
    <table>
      <thead><tr><th>Rank</th><th>ID</th><th>Root</th><th>Family</th><th>Air</th><th>1536s</th><th>3072s</th><th>Dup</th><th>Adj</th><th>Rank Tuple</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </main>
</body>
</html>
"""
    path.write_text(html)


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    paths = [Path(path) for group in args.state_json for path in group]
    source_records = load_records(paths)
    min_features = _parse_min_feature_filters(args.min_feature)
    max_features = _parse_max_feature_filters(args.max_feature)
    root_filter_paths = [Path(path) for group in (args.root_filter_json or []) for path in group]
    root_filter = load_root_filter(root_filter_paths) if root_filter_paths else None
    selected, candidates, rejected = select_frontier_records(
        source_records,
        next_target=args.next_target,
        target_reached_filter=args.target_reached,
        live_only=not args.include_terminal,
        min_features=min_features,
        max_features=max_features,
        root_filter=root_filter,
        rank_order=args.rank_order,
        rank_profile=args.rank_profile,
        max_records=args.max_records,
        max_per_root=args.max_per_root,
        max_per_source=args.max_per_source,
    )
    summary = _summary(
        source_paths=[str(path) for path in paths],
        source_count=len(source_records),
        selected=selected,
        candidates=candidates,
        rejected=rejected,
        next_target=args.next_target,
        target_reached_filter=args.target_reached,
        live_only=not args.include_terminal,
        min_features=min_features,
        max_features=max_features,
        root_filter=root_filter,
        rank_order=args.rank_order,
        rank_profile=args.rank_profile,
        max_records=args.max_records,
        max_per_root=args.max_per_root,
        max_per_source=args.max_per_source,
    )
    payload = {
        "version": 1,
        "kind": "frontier_selector",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "summary": summary,
        "selected_records": selected,
        "candidate_records": candidates,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload["json"] = str(args.out_dir / "frontier_selection.json")
    payload["records_json"] = str(args.out_dir / "records.json")
    payload["html"] = str(args.out_dir / "frontier_selection.html")
    write_json(args.out_dir / "frontier_selection.json", payload)
    write_json(args.out_dir / "records.json", selected)
    write_html(args.out_dir / "frontier_selection.html", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-json", type=Path, nargs="+", action="append", required=True)
    parser.add_argument("--next-target", choices=SUPPORTED_NEXT_TARGETS, required=True)
    parser.add_argument("--target-reached", choices=["all", "true", "false"], default="all")
    parser.add_argument("--include-terminal", action="store_true")
    parser.add_argument("--min-feature", action="append", help="Require a feature to be at least a value, e.g. empty_count=2.")
    parser.add_argument("--max-feature", action="append", help="Require a feature to be at most a value, e.g. raw_count_1536=1.")
    parser.add_argument("--root-filter-json", type=Path, nargs="+", action="append")
    parser.add_argument("--rank-order", choices=["top", "bottom"], default="top")
    parser.add_argument("--rank-profile", choices=SUPPORTED_RANK_PROFILES, default="support")
    parser.add_argument("--max-records", type=int, default=16)
    parser.add_argument("--max-per-root", type=int, default=1)
    parser.add_argument("--max-per-source", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("threes_rl/runs/forensics/frontier_selection/latest"))
    args = parser.parse_args()
    payload = run_from_args(args)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"json={payload['json']}")
    print(f"records={payload['records_json']}")
    print(f"html={payload['html']}")


if __name__ == "__main__":
    main()
