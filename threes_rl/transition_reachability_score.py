"""Score unlabeled states with a transition-window reachability model."""

from __future__ import annotations

import argparse
import json
import time
from html import escape
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np

from threes_rl.run_artifacts import write_json
from threes_rl.transition_reachability_audit import (
    _design_matrix,
    _fit_logistic,
    _one_hot_maps,
    build_rows,
    load_records,
    row_from_record,
)


def _fit_model(train_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len({int(row["y"]) for row in train_rows}) < 2:
        raise ValueError("Training rows must contain both success and failure outcomes")
    one_hot = _one_hot_maps(train_rows)
    x_train, y_train, names, mean_vec, std_vec = _design_matrix(train_rows, one_hot=one_hot)
    weights = _fit_logistic(x_train, y_train)
    return {
        "one_hot": one_hot,
        "names": names,
        "mean_vec": mean_vec,
        "std_vec": std_vec,
        "weights": weights,
    }


def _predict(model: dict[str, Any], rows: list[dict[str, Any]]) -> list[float]:
    x, _y, _names, _mean, _std = _design_matrix(
        rows,
        one_hot=model["one_hot"],
        mean_vec=model["mean_vec"],
        std_vec=model["std_vec"],
    )
    logits = np.clip(x @ model["weights"], -30.0, 30.0)
    return [float(value) for value in 1.0 / (1.0 + np.exp(-logits))]


def candidate_pairs(records: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for record in records:
        row = row_from_record(record, require_outcome=False)
        if row is not None:
            pairs.append((record, row))
    return pairs


def _score_summary(scored: list[dict[str, Any]]) -> dict[str, Any]:
    probs = [float(row["reachability_prob"]) for row in scored]
    if not probs:
        return {
            "candidates": 0,
            "mean_reachability_prob": 0.0,
            "median_reachability_prob": 0.0,
            "max_reachability_prob": 0.0,
        }
    return {
        "candidates": len(scored),
        "mean_reachability_prob": float(mean(probs)),
        "median_reachability_prob": float(median(probs)),
        "max_reachability_prob": float(max(probs)),
        "p_ge_0_50": float(sum(prob >= 0.5 for prob in probs) / len(probs)),
        "p_ge_0_75": float(sum(prob >= 0.75 for prob in probs) / len(probs)),
        "p_ge_0_90": float(sum(prob >= 0.9 for prob in probs) / len(probs)),
    }


def _parse_min_feature_filters(values: list[str] | None) -> dict[str, float]:
    filters: dict[str, float] = {}
    for raw in values or []:
        if "=" not in raw:
            raise ValueError(f"Expected --candidate-min-feature name=value, got {raw!r}")
        name, value = raw.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Empty feature name in {raw!r}")
        filters[name] = float(value.strip())
    return filters


def _passes_min_feature_filters(row: dict[str, Any], filters: dict[str, float]) -> bool:
    for name, threshold in filters.items():
        try:
            value = float(row.get(name, 0.0))
        except (TypeError, ValueError):
            return False
        if value < float(threshold):
            return False
    return True


def _parse_max_feature_filters(values: list[str] | None) -> dict[str, float]:
    filters: dict[str, float] = {}
    for raw in values or []:
        if "=" not in raw:
            raise ValueError(f"Expected --candidate-max-feature name=value, got {raw!r}")
        name, value = raw.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Empty feature name in {raw!r}")
        filters[name] = float(value.strip())
    return filters


def _passes_max_feature_filters(row: dict[str, Any], filters: dict[str, float]) -> bool:
    for name, threshold in filters.items():
        try:
            value = float(row.get(name, 0.0))
        except (TypeError, ValueError):
            return False
        if value > float(threshold):
            return False
    return True


def _parse_equal_feature_filters(values: list[str] | None) -> dict[str, str]:
    filters: dict[str, str] = {}
    for raw in values or []:
        if "=" not in raw:
            raise ValueError(f"Expected --candidate-feature-equals name=value, got {raw!r}")
        name, value = raw.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Empty feature name in {raw!r}")
        filters[name] = value.strip()
    return filters


def _feature_value_equals(value: Any, expected: str) -> bool:
    try:
        return abs(float(value) - float(expected)) <= 1e-9
    except (TypeError, ValueError):
        return str(value).lower() == str(expected).lower()


def _passes_equal_feature_filters(row: dict[str, Any], filters: dict[str, str]) -> bool:
    for name, expected in filters.items():
        if not _feature_value_equals(row.get(name), expected):
            return False
    return True


def _group_summary(scored: list[dict[str, Any]], field: str, *, limit: int = 16) -> list[dict[str, Any]]:
    buckets: dict[str, list[float]] = {}
    for row in scored:
        buckets.setdefault(str(row.get(field, "unknown")), []).append(float(row["reachability_prob"]))
    rows = [
        {
            field: value,
            "records": len(values),
            "mean_prob": float(mean(values)),
            "median_prob": float(median(values)),
            "max_prob": float(max(values)),
        }
        for value, values in buckets.items()
    ]
    rows.sort(key=lambda row: (float(row["max_prob"]), float(row["mean_prob"]), int(row["records"])), reverse=True)
    return rows[: int(limit)]


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _diversified_top_candidates(
    scored: list[dict[str, Any]],
    *,
    top_n: int,
    source_replay_limit: int = 0,
    source_seed_limit: int = 0,
    frame_min_gap: int = 0,
) -> list[dict[str, Any]]:
    if top_n <= 0:
        return []
    if source_replay_limit <= 0 and source_seed_limit <= 0 and frame_min_gap <= 0:
        return scored[:top_n]

    selected: list[dict[str, Any]] = []
    replay_counts: dict[str, int] = {}
    seed_counts: dict[str, int] = {}
    replay_frames: dict[str, list[int]] = {}
    for row in scored:
        replay = str(row.get("source_replay", "unknown"))
        seed = str(row.get("source_seed", "unknown"))
        frame = _as_int(row.get("source_frame_index"))
        if source_replay_limit > 0 and replay_counts.get(replay, 0) >= source_replay_limit:
            continue
        if source_seed_limit > 0 and seed_counts.get(seed, 0) >= source_seed_limit:
            continue
        if frame_min_gap > 0 and frame is not None:
            prior_frames = replay_frames.get(replay, [])
            if any(abs(frame - prior) < frame_min_gap for prior in prior_frames):
                continue
        selected.append(row)
        replay_counts[replay] = replay_counts.get(replay, 0) + 1
        seed_counts[seed] = seed_counts.get(seed, 0) + 1
        if frame is not None:
            replay_frames.setdefault(replay, []).append(frame)
        if len(selected) >= top_n:
            break
    return selected


def _scored_row(record: dict[str, Any], row: dict[str, Any], prob: float) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "source_json": record.get("_source_json"),
        "record_index": record.get("_record_index"),
        "source_replay": row.get("source_replay"),
        "source_seed": row.get("source_seed"),
        "source_frame_index": row.get("source_frame_index"),
        "stratum": row.get("stratum"),
        "corner_risk": row.get("corner_risk"),
        "preview": row.get("preview"),
        "source_action": row.get("source_action"),
        "score_minus_starter": row.get("score_minus_starter"),
        "move_count": row.get("move_count"),
        "empty_count": row.get("empty_count"),
        "legal_count": row.get("legal_count"),
        "top_left": row.get("top_left"),
        "top_left_is_max": row.get("top_left_is_max"),
        "count_3072": row.get("count_3072"),
        "count_1536": row.get("count_1536"),
        "count_768": row.get("count_768"),
        "support_score": row.get("support_score"),
        "reachability_prob": float(prob),
    }


def _candidate_key(record: dict[str, Any], row: dict[str, Any]) -> str:
    state = record.get("state")
    if isinstance(state, dict):
        return json.dumps(state, sort_keys=True, separators=(",", ":"))
    return json.dumps(
        {
            "source_replay": row.get("source_replay"),
            "source_seed": row.get("source_seed"),
            "source_frame_index": row.get("source_frame_index"),
            "id": row.get("id"),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _record_with_score(record: dict[str, Any], scored_row: dict[str, Any]) -> dict[str, Any]:
    clean = {key: value for key, value in record.items() if not str(key).startswith("_")}
    clean["reachability_prob"] = float(scored_row["reachability_prob"])
    clean["reachability_rank"] = int(scored_row["rank"])
    clean["reachability_model"] = "transition_reachability_logistic"
    return clean


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
    records = []
    for record, _row, _scored_row_obj in unique_pairs:
        key = (str(record.get("_source_json")), int(record.get("_record_index", -1)))
        if key in selected_keys:
            records.append(_record_with_score(record, scored_by_key[key]))
    records.sort(key=lambda record: int(record["reachability_rank"]), reverse=reverse)
    return records


def _model_summary(model: dict[str, Any]) -> dict[str, Any]:
    top_weights = [
        {"feature": str(name), "weight": float(weight)}
        for name, weight in sorted(
            zip(model["names"], model["weights"]),
            key=lambda item: abs(float(item[1])),
            reverse=True,
        )[:20]
    ]
    return {
        "features": len(model["names"]),
        "top_weights": top_weights,
    }


def write_html(path: Path, payload: dict[str, Any]) -> None:
    top = payload.get("top_candidates", [])
    summary = payload.get("summary", {})

    def cell(value: object) -> str:
        return escape(str(value))

    rows = []
    for row in top if isinstance(top, list) else []:
        rows.append(
            "<tr>"
            f"<td>{cell(row.get('rank'))}</td>"
            f"<td>{float(row.get('reachability_prob', 0.0)):.3f}</td>"
            f"<td>{cell(row.get('source_seed'))}</td>"
            f"<td>{cell(row.get('source_frame_index'))}</td>"
            f"<td>{cell(row.get('stratum'))}</td>"
            f"<td>{cell(row.get('top_left'))}</td>"
            f"<td>{cell(row.get('empty_count'))}</td>"
            f"<td>{float(row.get('score_minus_starter', 0.0)):.0f}</td>"
            f"<td>{cell(row.get('source_replay'))}</td>"
            "</tr>"
        )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Transition Reachability Scores</title>
  <style>
    :root {{ color-scheme: dark; --bg: #101318; --panel: #171d24; --line: #34404d; --text: #edf2f7; --muted: #aab6c2; --gold: #f2c14e; }}
    body {{ margin: 0; background: var(--bg); color: var(--text); font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ width: min(1180px, calc(100vw - 32px)); margin: 0 auto; padding: 24px 0 40px; }}
    h1 {{ margin: 0 0 8px; font-size: 24px; }}
    .muted {{ color: var(--muted); }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 18px 0; }}
    .card, .panel {{ border: 1px solid var(--line); background: var(--panel); border-radius: 8px; padding: 12px; }}
    .label {{ color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .08em; }}
    .value {{ margin-top: 4px; color: var(--gold); font-size: 22px; font-weight: 800; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; font-variant-numeric: tabular-nums; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 7px 8px; text-align: right; }}
    th:first-child, td:first-child, th:last-child, td:last-child {{ text-align: left; }}
    td:last-child {{ max-width: 360px; overflow-wrap: anywhere; color: var(--muted); }}
    pre {{ white-space: pre-wrap; overflow-wrap: anywhere; color: var(--muted); }}
  </style>
</head>
<body>
  <main>
    <h1>Transition Reachability Scores</h1>
    <p class="muted">Unlabeled states ranked by a logistic model trained on labeled pre-promotion success/failure windows.</p>
    <section class="cards">
      <div class="card"><div class="label">Candidates</div><div class="value">{cell(summary.get('candidates', 0))}</div></div>
      <div class="card"><div class="label">Mean Prob</div><div class="value">{float(summary.get('mean_reachability_prob', 0.0)):.3f}</div></div>
      <div class="card"><div class="label">Median Prob</div><div class="value">{float(summary.get('median_reachability_prob', 0.0)):.3f}</div></div>
      <div class="card"><div class="label">Max Prob</div><div class="value">{float(summary.get('max_reachability_prob', 0.0)):.3f}</div></div>
    </section>
    <section class="panel">
      <table><thead><tr><th>Rank</th><th>Prob</th><th>Seed</th><th>Frame</th><th>Stratum</th><th>Top Left</th><th>Empty</th><th>Score - Starter</th><th>Source Replay</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
    </section>
    <section class="panel" style="margin-top:14px;"><pre>{escape(json.dumps(payload.get('model', {}), indent=2, sort_keys=True))}</pre></section>
  </main>
</body>
</html>
"""
    path.write_text(html)


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    train_records = load_records([path for group in args.train_json for path in group])
    candidate_records = load_records([path for group in args.candidate_json for path in group])
    train_rows = build_rows(train_records, target_tile=args.target_tile)
    pairs = candidate_pairs(candidate_records)
    min_feature_filters = _parse_min_feature_filters(getattr(args, "candidate_min_feature", None))
    max_feature_filters = _parse_max_feature_filters(getattr(args, "candidate_max_feature", None))
    equal_feature_filters = _parse_equal_feature_filters(getattr(args, "candidate_feature_equals", None))
    if min_feature_filters or max_feature_filters or equal_feature_filters:
        pairs = [
            (record, row)
            for record, row in pairs
            if _passes_min_feature_filters(row, min_feature_filters)
            and _passes_max_feature_filters(row, max_feature_filters)
            and _passes_equal_feature_filters(row, equal_feature_filters)
        ]
    candidate_rows = [row for _record, row in pairs]
    model = _fit_model(train_rows)
    probs = _predict(model, candidate_rows)
    scored_pairs = [
        (record, row, _scored_row(record, row, prob))
        for (record, row), prob in zip(pairs, probs)
    ]
    scored_pairs.sort(key=lambda item: float(item[2]["reachability_prob"]), reverse=True)
    unique_pairs: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    seen: set[str] = set()
    for record, row, scored_row in scored_pairs:
        key = _candidate_key(record, row)
        if key in seen:
            continue
        seen.add(key)
        unique_pairs.append((record, row, scored_row))
    scored = [scored_row for _record, _row, scored_row in unique_pairs]
    for rank, row in enumerate(scored, start=1):
        row["rank"] = rank

    top_n = max(0, int(args.top_n))
    top_source_replay_limit = max(0, int(getattr(args, "top_source_replay_limit", 0) or 0))
    top_source_seed_limit = max(0, int(getattr(args, "top_source_seed_limit", 0) or 0))
    top_frame_min_gap = max(0, int(getattr(args, "top_frame_min_gap", 0) or 0))
    top_candidates = _diversified_top_candidates(
        scored,
        top_n=top_n,
        source_replay_limit=top_source_replay_limit,
        source_seed_limit=top_source_seed_limit,
        frame_min_gap=top_frame_min_gap,
    )
    bottom_n = max(0, int(getattr(args, "bottom_n", 0) or 0))
    bottom_candidates = _diversified_top_candidates(
        list(reversed(scored)),
        top_n=bottom_n,
        source_replay_limit=top_source_replay_limit,
        source_seed_limit=top_source_seed_limit,
        frame_min_gap=top_frame_min_gap,
    )
    top_records = _selected_records(selected=top_candidates, unique_pairs=unique_pairs, scored=scored)
    bottom_records = _selected_records(
        selected=bottom_candidates,
        unique_pairs=unique_pairs,
        scored=scored,
        reverse=True,
    )

    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "train_json": [str(path) for group in args.train_json for path in group],
        "candidate_json": [str(path) for group in args.candidate_json for path in group],
        "target_tile": int(args.target_tile),
        "train_records": len(train_records),
        "train_rows": len(train_rows),
        "train_successes": int(sum(int(row["y"]) for row in train_rows)),
        "train_failures": int(len(train_rows) - sum(int(row["y"]) for row in train_rows)),
        "raw_candidates": len(candidate_records),
        "candidate_min_feature": min_feature_filters,
        "candidate_max_feature": max_feature_filters,
        "candidate_feature_equals": equal_feature_filters,
        "filtered_candidates": len(pairs),
        "unique_candidates": len(scored),
        "top_selection": {
            "top_n": top_n,
            "selected": len(top_candidates),
            "source_replay_limit": top_source_replay_limit,
            "source_seed_limit": top_source_seed_limit,
            "frame_min_gap": top_frame_min_gap,
        },
        "bottom_selection": {
            "bottom_n": bottom_n,
            "selected": len(bottom_candidates),
            "source_replay_limit": top_source_replay_limit,
            "source_seed_limit": top_source_seed_limit,
            "frame_min_gap": top_frame_min_gap,
        },
        "summary": _score_summary(scored),
        "by_stratum": _group_summary(scored, "stratum"),
        "by_source_replay": _group_summary(scored, "source_replay"),
        "model": _model_summary(model),
        "top_candidates": top_candidates,
        "bottom_candidates": bottom_candidates,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload["json"] = str(args.out_dir / "reachability_scores.json")
    payload["html"] = str(args.out_dir / "reachability_scores.html")
    payload["top_records_json"] = str(args.out_dir / "top_records.json")
    payload["bottom_records_json"] = str(args.out_dir / "bottom_records.json")
    write_json(args.out_dir / "reachability_scores.json", payload)
    write_json(
        args.out_dir / "top_records.json",
        {
            "kind": "threes_transition_reachability_top_records",
            "version": 1,
            "created_at": payload["created_at"],
            "source_score_json": payload["json"],
            "records": top_records,
        },
    )
    write_json(
        args.out_dir / "bottom_records.json",
        {
            "kind": "threes_transition_reachability_bottom_records",
            "version": 1,
            "created_at": payload["created_at"],
            "source_score_json": payload["json"],
            "records": bottom_records,
        },
    )
    write_html(args.out_dir / "reachability_scores.html", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-json", type=Path, nargs="+", action="append", required=True)
    parser.add_argument("--candidate-json", type=Path, nargs="+", action="append", required=True)
    parser.add_argument("--target-tile", type=int, default=6144)
    parser.add_argument("--top-n", type=int, default=32)
    parser.add_argument(
        "--bottom-n",
        type=int,
        default=0,
        help="Also write this many lowest-scored records to bottom_records.json.",
    )
    parser.add_argument(
        "--candidate-min-feature",
        action="append",
        default=[],
        help="Keep candidates only when a computed row feature is at least this value, e.g. raw_count_1536=1.",
    )
    parser.add_argument(
        "--candidate-max-feature",
        action="append",
        default=[],
        help="Keep candidates only when a computed row feature is at most this value, e.g. empty_count=1.",
    )
    parser.add_argument(
        "--candidate-feature-equals",
        action="append",
        default=[],
        help="Keep candidates only when a computed row feature equals this value, e.g. preview=red.",
    )
    parser.add_argument(
        "--top-source-replay-limit",
        type=int,
        default=0,
        help="Maximum selected top records from one source replay; 0 disables this diversity limit.",
    )
    parser.add_argument(
        "--top-source-seed-limit",
        type=int,
        default=0,
        help="Maximum selected top records from one source seed; 0 disables this diversity limit.",
    )
    parser.add_argument(
        "--top-frame-min-gap",
        type=int,
        default=0,
        help="Minimum frame distance between selected top records from the same source replay; 0 disables spacing.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("threes_rl/runs/forensics/reachability_scores/latest"))
    args = parser.parse_args()
    payload = run_from_args(args)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"json={payload['json']}")
    print(f"html={payload['html']}")
    print(f"top_records={payload['top_records_json']}")
    print(f"bottom_records={payload['bottom_records_json']}")


if __name__ == "__main__":
    main()
