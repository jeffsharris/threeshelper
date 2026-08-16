"""Audit support material in root-capped near-failure state pools."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import numpy as np

from threes_rl.rare_event_frontier import case_from_record, load_records
from threes_rl.run_artifacts import write_json
from threes_rl.sim import ThreesSim
from threes_rl.support_accumulation_frontier import _raw


def support_bucket(raw: dict[str, Any]) -> str:
    if int(raw.get("raw_count_768", 0)) != 1:
        return "not_one_768"
    if bool(raw.get("raw_has_adjacent_384", False)):
        return "adjacent_384"
    if int(raw.get("raw_count_384", 0)) >= 2:
        return "duplicate_384"
    if int(raw.get("raw_count_384", 0)) == 1:
        return "one_384"
    if bool(raw.get("raw_has_adjacent_192", False)):
        return "adjacent_192"
    if int(raw.get("raw_count_192", 0)) >= 2:
        return "duplicate_192"
    if int(raw.get("raw_count_192", 0)) == 1:
        return "one_192"
    return "no_384_192"


def _load_pool_records(paths: Iterable[Path], *, default_starter_tile: int | None) -> tuple[list[dict[str, Any]], Counter[str]]:
    rows: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    for record in load_records(paths):
        case = case_from_record(record, default_starter_tile=default_starter_tile)
        if case is None:
            rejected["bad_case"] += 1
            continue
        raw = _raw(case.state, case.starter_tile)
        sim = ThreesSim(np.random.default_rng(case.source_seed or 0), starter_tile=case.starter_tile)
        legal_count = len(sim.legal_actions(case.state))
        source_path = str(record.get("_source_json", ""))
        pool = Path(source_path).parent.name if source_path else "unknown"
        row = {
            "pool": pool,
            "source_json": source_path,
            "id": case.id,
            "root_seed": case.root_seed,
            "root_origin": case.root_origin,
            "root_policy_family": case.root_policy_family,
            "source_replay": case.source_replay,
            "source_frame_index": case.source_frame_index,
            "source_move_count": record.get("source_move_count", case.features.get("move_count")),
            "source_score": record.get("source_score", record.get("score", case.features.get("score_minus_starter", 0))),
            "final_score": record.get("final_score"),
            "starter_tile": case.starter_tile,
            "raw_count_768": int(raw["raw_count_768"]),
            "raw_has_adjacent_768": bool(raw["raw_has_adjacent_768"]),
            "raw_count_384": int(raw["raw_count_384"]),
            "raw_has_adjacent_384": bool(raw["raw_has_adjacent_384"]),
            "raw_count_192": int(raw["raw_count_192"]),
            "raw_has_adjacent_192": bool(raw["raw_has_adjacent_192"]),
            "raw_highest_duplicate_tile": int(raw["raw_highest_duplicate_tile"]),
            "raw_highest_adjacent_pair_tile": int(raw["raw_highest_adjacent_pair_tile"]),
            "masked_count_1536": int(raw["masked_count_1536"]),
            "empty_count": int(raw["empty_count"]),
            "legal_count": int(legal_count),
            "support_bucket": support_bucket(raw),
        }
        rows.append(row)
    return rows, rejected


def _counter(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key, "unknown")) for row in rows).items()))


def _numeric_values(rows: Iterable[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return values


def _group_stats(rows: list[dict[str, Any]], group_key: str) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get(group_key, "unknown")), []).append(row)
    stats: dict[str, dict[str, float | int]] = {}
    for key, group in sorted(grouped.items()):
        source_scores = _numeric_values(group, "source_score")
        final_scores = _numeric_values(group, "final_score")
        empty_counts = _numeric_values(group, "empty_count")
        stats[key] = {
            "records": len(group),
            "mean_source_score": mean(source_scores) if source_scores else 0.0,
            "mean_final_score": mean(final_scores) if final_scores else 0.0,
            "mean_empty_count": mean(empty_counts) if empty_counts else 0.0,
        }
    return stats


def summarize(rows: list[dict[str, Any]], *, rejected: Counter[str], source_paths: list[str]) -> dict[str, Any]:
    one_768 = [row for row in rows if int(row["raw_count_768"]) == 1 and int(row["masked_count_1536"]) == 0]
    has_384 = [row for row in one_768 if int(row["raw_count_384"]) > 0]
    has_192 = [row for row in one_768 if int(row["raw_count_192"]) > 0]
    no_384_192 = [row for row in one_768 if int(row["raw_count_384"]) == 0 and int(row["raw_count_192"]) == 0]
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_paths": source_paths,
        "records": len(rows),
        "unique_root_seeds": len({row.get("root_seed") for row in rows if row.get("root_seed") is not None}),
        "one_768_records": len(one_768),
        "one_768_with_384": len(has_384),
        "one_768_with_192": len(has_192),
        "one_768_no_384_192": len(no_384_192),
        "adjacent_384_records": sum(1 for row in one_768 if bool(row["raw_has_adjacent_384"])),
        "duplicate_384_records": sum(1 for row in one_768 if int(row["raw_count_384"]) >= 2),
        "adjacent_192_records": sum(1 for row in one_768 if bool(row["raw_has_adjacent_192"])),
        "duplicate_192_records": sum(1 for row in one_768 if int(row["raw_count_192"]) >= 2),
        "mean_empty_count": mean([int(row["empty_count"]) for row in rows]) if rows else 0.0,
        "by_pool": _counter(rows, "pool"),
        "by_support_bucket": _counter(rows, "support_bucket"),
        "by_raw_count_768": _counter(rows, "raw_count_768"),
        "by_raw_count_384": _counter(rows, "raw_count_384"),
        "by_raw_count_192": _counter(rows, "raw_count_192"),
        "support_bucket_stats": _group_stats(rows, "support_bucket"),
        "rejected": dict(rejected),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "pool",
        "root_seed",
        "source_frame_index",
        "source_move_count",
        "source_score",
        "final_score",
        "raw_count_768",
        "raw_has_adjacent_768",
        "raw_count_384",
        "raw_has_adjacent_384",
        "raw_count_192",
        "raw_has_adjacent_192",
        "raw_highest_duplicate_tile",
        "raw_highest_adjacent_pair_tile",
        "empty_count",
        "legal_count",
        "support_bucket",
        "source_replay",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def write_report(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    top_sparse = [
        row
        for row in sorted(
            rows,
            key=lambda row: (
                int(row["raw_count_384"]),
                int(row["raw_count_192"]),
                int(row["empty_count"]),
                int(row.get("source_score") or 0),
            ),
        )
        if row["support_bucket"] in {"no_384_192", "one_192", "duplicate_192"}
    ][:12]
    lines = [
        "# Near-Failure Support-Material Audit",
        "",
        f"Created: `{summary['created_at']}`",
        "",
        "## Summary",
        "",
        f"- Records: `{summary['records']}`",
        f"- Unique roots: `{summary['unique_root_seeds']}`",
        f"- One-768 records: `{summary['one_768_records']}`",
        f"- One-768 with raw 384: `{summary['one_768_with_384']}`",
        f"- One-768 with raw 192: `{summary['one_768_with_192']}`",
        f"- One-768 with no raw 384/192: `{summary['one_768_no_384_192']}`",
        f"- Adjacent / duplicate 384: `{summary['adjacent_384_records']}` / `{summary['duplicate_384_records']}`",
        f"- Adjacent / duplicate 192: `{summary['adjacent_192_records']}` / `{summary['duplicate_192_records']}`",
        "",
        "## Support Buckets",
        "",
        "| bucket | records |",
        "| --- | ---: |",
    ]
    for bucket, count in summary["by_support_bucket"].items():
        stats = summary["support_bucket_stats"].get(bucket, {})
        lines.append(
            f"| {bucket} | {count} | {float(stats.get('mean_source_score', 0.0)):.1f} | "
            f"{float(stats.get('mean_final_score', 0.0)):.1f} | {float(stats.get('mean_empty_count', 0.0)):.2f} |"
        )
    lines[lines.index("| bucket | records |")] = "| bucket | records | mean source score | mean final score | mean empty |"
    lines[lines.index("| --- | ---: |")] = "| --- | ---: | ---: | ---: | ---: |"
    lines.extend(
        [
            "",
            "## Pools",
            "",
            "| pool | records |",
            "| --- | ---: |",
        ]
    )
    for pool, count in summary["by_pool"].items():
        lines.append(f"| {pool} | {count} |")
    lines.extend(
        [
            "",
            "## Sparse Examples",
            "",
            "| root | bucket | raw384 | raw192 | empty | score | pool |",
            "| ---: | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in top_sparse:
        lines.append(
            f"| {row.get('root_seed')} | {row['support_bucket']} | {row['raw_count_384']} | "
            f"{row['raw_count_192']} | {row['empty_count']} | {row.get('source_score')} | {row['pool']} |"
        )
    path.write_text("\n".join(lines) + "\n")


def run_audit(
    *,
    records_json: list[Path],
    out_dir: Path,
    default_starter_tile: int | None = 1536,
) -> dict[str, Any]:
    rows, rejected = _load_pool_records(records_json, default_starter_tile=default_starter_tile)
    summary = summarize(rows, rejected=rejected, source_paths=[str(path) for path in records_json])
    payload = {"kind": "nearfail_support_audit", "summary": summary, "records": rows}
    out_dir.mkdir(parents=True, exist_ok=True)
    payload["json"] = str(out_dir / "support_audit.json")
    payload["records_csv"] = str(out_dir / "support_audit.csv")
    payload["report"] = str(out_dir / "report.md")
    write_json(out_dir / "support_audit.json", payload)
    write_json(out_dir / "summary.json", summary)
    write_csv(out_dir / "support_audit.csv", rows)
    write_report(out_dir / "report.md", summary, rows)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-json", type=Path, action="append", required=True)
    parser.add_argument("--starter", default="1536")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    starter_text = str(args.starter).strip().lower()
    default_starter = None if starter_text == "none" else int(starter_text)
    payload = run_audit(records_json=args.records_json, out_dir=args.out_dir, default_starter_tile=default_starter)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"report={payload['report']}")


if __name__ == "__main__":
    main()
