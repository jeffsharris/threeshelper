"""Cross-fit transition reachability scores by held-out source group."""

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
from threes_rl.transition_reachability_audit import _auc, row_from_record
from threes_rl.transition_reachability_score import (
    _candidate_key,
    _diversified_top_candidates,
    _fit_model,
    _parse_equal_feature_filters,
    _parse_max_feature_filters,
    _parse_min_feature_filters,
    _passes_equal_feature_filters,
    _passes_max_feature_filters,
    _passes_min_feature_filters,
    _predict,
    _record_with_score,
    _score_summary,
    _scored_row,
    load_records,
)


def _record_target_matches(record: dict[str, Any], target_tile: int | None) -> bool:
    if target_tile is None:
        return True
    try:
        return int(record.get("target_tile", -1)) == int(target_tile)
    except (TypeError, ValueError):
        return False


def _filtered_pairs(
    records: Iterable[dict[str, Any]],
    *,
    target_tile: int | None,
    group_by: str,
    min_filters: dict[str, float],
    max_filters: dict[str, float],
    equal_filters: dict[str, str],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for record in records:
        if not _record_target_matches(record, target_tile):
            continue
        row = row_from_record(record, require_outcome=True, group_by=group_by)
        if row is None:
            continue
        if (
            _passes_min_feature_filters(row, min_filters)
            and _passes_max_feature_filters(row, max_filters)
            and _passes_equal_feature_filters(row, equal_filters)
        ):
            pairs.append((record, row))
    return pairs


def _cap_train_pairs(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    cap_per_source_group_outcome: int,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    cap = int(cap_per_source_group_outcome)
    if cap <= 0:
        return pairs
    counts: Counter[tuple[str, str]] = Counter()
    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for record, row in pairs:
        key = (str(record.get("source_group", row.get("group_key", row.get("source_replay")))), str(row.get("outcome")))
        if counts[key] >= cap:
            continue
        counts[key] += 1
        selected.append((record, row))
    return selected


def _selected_records(
    *,
    selected: list[dict[str, Any]],
    unique_pairs: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]],
    scored: list[dict[str, Any]],
    reverse: bool = False,
) -> list[dict[str, Any]]:
    selected_keys = {
        (str(row.get("source_json")), int(row.get("record_index")))
        for row in selected
        if row.get("record_index") is not None
    }
    scored_by_key = {
        (str(row.get("source_json")), int(row.get("record_index"))): row
        for row in scored
        if row.get("record_index") is not None
    }
    rows = []
    for record, _row, _scored in unique_pairs:
        key = (str(record.get("_source_json")), int(record.get("_record_index", -1)))
        if key not in selected_keys:
            continue
        clean = _record_with_score(record, scored_by_key[key])
        clean["crossfit_group"] = scored_by_key[key].get("crossfit_group")
        clean["crossfit_train_records"] = scored_by_key[key].get("crossfit_train_records")
        rows.append(clean)
    rows.sort(key=lambda record: int(record["reachability_rank"]), reverse=reverse)
    return rows


def _crossfit_scores(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    cap_per_source_group_outcome: int,
) -> tuple[list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]], dict[str, Any]]:
    groups = sorted({str(row.get("group_key", row.get("source_replay"))) for _record, row in pairs})
    scored_pairs: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    skipped_records = 0
    fold_rows: list[dict[str, Any]] = []
    for group in groups:
        train_pairs = [(record, row) for record, row in pairs if str(row.get("group_key", row.get("source_replay"))) != group]
        test_pairs = [(record, row) for record, row in pairs if str(row.get("group_key", row.get("source_replay"))) == group]
        train_pairs = _cap_train_pairs(train_pairs, cap_per_source_group_outcome=cap_per_source_group_outcome)
        train_rows = [row for _record, row in train_pairs]
        test_rows = [row for _record, row in test_pairs]
        if len({int(row["y"]) for row in train_rows}) < 2 or not test_rows:
            skipped_records += len(test_rows)
            fold_rows.append(
                {
                    "group": group,
                    "test_records": len(test_rows),
                    "train_records": len(train_rows),
                    "train_successes": int(sum(int(row["y"]) for row in train_rows)),
                    "train_failures": int(len(train_rows) - sum(int(row["y"]) for row in train_rows)),
                    "skipped": True,
                }
            )
            continue
        model = _fit_model(train_rows)
        probs = _predict(model, test_rows)
        fold_rows.append(
            {
                "group": group,
                "test_records": len(test_rows),
                "train_records": len(train_rows),
                "train_successes": int(sum(int(row["y"]) for row in train_rows)),
                "train_failures": int(len(train_rows) - sum(int(row["y"]) for row in train_rows)),
                "skipped": False,
            }
        )
        for (record, row), prob in zip(test_pairs, probs):
            scored = _scored_row(record, row, prob)
            scored["outcome"] = row.get("outcome")
            scored["y"] = int(row["y"])
            scored["crossfit_group"] = group
            scored["crossfit_train_records"] = len(train_rows)
            scored_pairs.append((record, row, scored))
    y_true = [int(row["y"]) for _record, _row, row in scored_pairs]
    probs = [float(row["reachability_prob"]) for _record, _row, row in scored_pairs]
    accuracy = (
        sum((prob >= 0.5) == bool(y) for prob, y in zip(probs, y_true)) / len(y_true)
        if y_true
        else 0.0
    )
    summary = {
        "groups": len(groups),
        "folds": fold_rows,
        "scored_records": len(scored_pairs),
        "skipped_records": int(skipped_records),
        "auc": _auc(y_true, probs) if y_true else 0.0,
        "accuracy_at_0_5": float(accuracy),
        "mean_predicted_success": float(mean(probs)) if probs else 0.0,
        "median_predicted_success": float(median(probs)) if probs else 0.0,
    }
    return scored_pairs, summary


def _dedupe_and_rank(
    scored_pairs: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]],
) -> list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]:
    scored_pairs = sorted(scored_pairs, key=lambda item: float(item[2]["reachability_prob"]), reverse=True)
    unique: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    seen: set[str] = set()
    for record, row, scored in scored_pairs:
        key = _candidate_key(record, row)
        if key in seen:
            continue
        seen.add(key)
        unique.append((record, row, scored))
    for rank, (_record, _row, scored) in enumerate(unique, start=1):
        scored["rank"] = rank
    return unique


def write_html(path: Path, payload: dict[str, Any]) -> None:
    summary = payload.get("summary", {})
    top = payload.get("top_candidates", [])

    def cell(value: object) -> str:
        return escape(str(value))

    rows = []
    for row in top[:40] if isinstance(top, list) else []:
        rows.append(
            "<tr>"
            f"<td>{cell(row.get('rank'))}</td>"
            f"<td>{float(row.get('reachability_prob', 0.0)):.3f}</td>"
            f"<td>{cell(row.get('outcome'))}</td>"
            f"<td>{cell(row.get('source_seed'))}</td>"
            f"<td>{cell(row.get('source_frame_index'))}</td>"
            f"<td>{cell(row.get('source_replay'))}</td>"
            "</tr>"
        )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Cross-Fit Transition Reachability</title>
  <style>
    :root {{ color-scheme: dark; --bg:#101318; --panel:#171d24; --line:#34404d; --text:#edf2f7; --muted:#aab6c2; --gold:#f2c14e; }}
    body {{ margin:0; background:var(--bg); color:var(--text); font-family:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ width:min(1180px, calc(100vw - 32px)); margin:0 auto; padding:24px 0 40px; }}
    h1 {{ margin:0 0 8px; font-size:24px; }}
    .muted {{ color:var(--muted); }}
    .cards {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(150px, 1fr)); gap:10px; margin:18px 0; }}
    .card, .panel {{ border:1px solid var(--line); background:var(--panel); border-radius:8px; padding:12px; }}
    .label {{ color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.08em; }}
    .value {{ margin-top:4px; color:var(--gold); font-size:22px; font-weight:800; }}
    table {{ width:100%; border-collapse:collapse; font-size:12px; font-variant-numeric:tabular-nums; }}
    th, td {{ border-bottom:1px solid var(--line); padding:7px 8px; text-align:right; }}
    th:last-child, td:last-child {{ text-align:left; max-width:420px; overflow-wrap:anywhere; color:var(--muted); }}
    pre {{ white-space:pre-wrap; overflow-wrap:anywhere; color:var(--muted); }}
  </style>
</head>
<body>
  <main>
    <h1>Cross-Fit Transition Reachability</h1>
    <p class="muted">Out-of-fold reachability scores by held-out source group.</p>
    <section class="cards">
      <div class="card"><div class="label">Records</div><div class="value">{cell(summary.get('scored_records', 0))}</div></div>
      <div class="card"><div class="label">Groups</div><div class="value">{cell(summary.get('groups', 0))}</div></div>
      <div class="card"><div class="label">AUC</div><div class="value">{float(summary.get('auc', 0.0)):.3f}</div></div>
      <div class="card"><div class="label">Accuracy</div><div class="value">{float(summary.get('accuracy_at_0_5', 0.0)):.0%}</div></div>
    </section>
    <section class="panel">
      <table><thead><tr><th>Rank</th><th>Prob</th><th>Outcome</th><th>Seed</th><th>Frame</th><th>Replay</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
    </section>
    <section class="panel" style="margin-top:14px;"><pre>{escape(json.dumps(summary, indent=2, sort_keys=True))}</pre></section>
  </main>
</body>
</html>
"""
    path.write_text(html)


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    records = load_records([path for group in args.records_json for path in group])
    target_tile = None if bool(args.no_target_filter) else int(args.target_tile)
    min_filters = _parse_min_feature_filters(getattr(args, "candidate_min_feature", None))
    max_filters = _parse_max_feature_filters(getattr(args, "candidate_max_feature", None))
    equal_filters = _parse_equal_feature_filters(getattr(args, "candidate_feature_equals", None))
    pairs = _filtered_pairs(
        records,
        target_tile=target_tile,
        group_by=args.group_by,
        min_filters=min_filters,
        max_filters=max_filters,
        equal_filters=equal_filters,
    )
    scored_pairs, crossfit_summary = _crossfit_scores(
        pairs,
        cap_per_source_group_outcome=args.train_cap_per_source_group_outcome,
    )
    unique_pairs = _dedupe_and_rank(scored_pairs)
    scored = [scored for _record, _row, scored in unique_pairs]
    top_candidates = _diversified_top_candidates(
        scored,
        top_n=max(0, int(args.top_n)),
        source_replay_limit=max(0, int(args.top_source_replay_limit)),
        source_seed_limit=max(0, int(args.top_source_seed_limit)),
        frame_min_gap=max(0, int(args.top_frame_min_gap)),
    )
    bottom_candidates = _diversified_top_candidates(
        list(reversed(scored)),
        top_n=max(0, int(args.bottom_n)),
        source_replay_limit=max(0, int(args.top_source_replay_limit)),
        source_seed_limit=max(0, int(args.top_source_seed_limit)),
        frame_min_gap=max(0, int(args.top_frame_min_gap)),
    )
    top_records = _selected_records(selected=top_candidates, unique_pairs=unique_pairs, scored=scored)
    bottom_records = _selected_records(
        selected=bottom_candidates,
        unique_pairs=unique_pairs,
        scored=scored,
        reverse=True,
    )
    summary = {
        **crossfit_summary,
        **_score_summary(scored),
        "raw_records": len(records),
        "filtered_records": len(pairs),
        "unique_scored_records": len(scored),
        "target_tile": target_tile,
        "group_by": args.group_by,
        "train_cap_per_source_group_outcome": int(args.train_cap_per_source_group_outcome),
        "candidate_min_feature": min_filters,
        "candidate_max_feature": max_filters,
        "candidate_feature_equals": equal_filters,
        "top_selected": len(top_candidates),
        "bottom_selected": len(bottom_candidates),
    }
    payload = {
        "version": 1,
        "kind": "transition_reachability_crossfit",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "records_json": [str(path) for group in args.records_json for path in group],
        "summary": summary,
        "top_candidates": top_candidates,
        "bottom_candidates": bottom_candidates,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload["json"] = str(args.out_dir / "crossfit_scores.json")
    payload["html"] = str(args.out_dir / "crossfit_scores.html")
    payload["top_records_json"] = str(args.out_dir / "top_records.json")
    payload["bottom_records_json"] = str(args.out_dir / "bottom_records.json")
    write_json(args.out_dir / "crossfit_scores.json", payload)
    write_json(
        args.out_dir / "top_records.json",
        {
            "kind": "threes_transition_reachability_crossfit_top_records",
            "version": 1,
            "created_at": payload["created_at"],
            "source_score_json": payload["json"],
            "records": top_records,
        },
    )
    write_json(
        args.out_dir / "bottom_records.json",
        {
            "kind": "threes_transition_reachability_crossfit_bottom_records",
            "version": 1,
            "created_at": payload["created_at"],
            "source_score_json": payload["json"],
            "records": bottom_records,
        },
    )
    write_html(args.out_dir / "crossfit_scores.html", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-json", type=Path, nargs="+", action="append", required=True)
    parser.add_argument("--target-tile", type=int, default=6144)
    parser.add_argument("--no-target-filter", action="store_true")
    parser.add_argument(
        "--group-by",
        choices=("auto", "source-group", "source-replay", "original-replay"),
        default="original-replay",
    )
    parser.add_argument("--train-cap-per-source-group-outcome", type=int, default=0)
    parser.add_argument("--candidate-min-feature", action="append", default=[])
    parser.add_argument("--candidate-max-feature", action="append", default=[])
    parser.add_argument("--candidate-feature-equals", action="append", default=[])
    parser.add_argument("--top-n", type=int, default=32)
    parser.add_argument("--bottom-n", type=int, default=0)
    parser.add_argument("--top-source-replay-limit", type=int, default=0)
    parser.add_argument("--top-source-seed-limit", type=int, default=0)
    parser.add_argument("--top-frame-min-gap", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("threes_rl/runs/forensics/reachability_scores/latest_crossfit"))
    args = parser.parse_args()
    payload = run_from_args(args)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"json={payload['json']}")
    print(f"html={payload['html']}")
    print(f"top_records={payload['top_records_json']}")
    print(f"bottom_records={payload['bottom_records_json']}")


if __name__ == "__main__":
    main()
