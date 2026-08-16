"""Extract real support-ladder stage starts from state-record corpora."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from html import escape
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from threes_rl.compare_replays import state_from_payload
from threes_rl.eval import max_tile_excluding_initial_starter
from threes_rl.run_artifacts import safe_name, write_json
from threes_rl.sim import ThreesSim, score_board
from threes_rl.support_ladder_window_reservoir import raw_ladder_features
from threes_rl.swing_label import CORNER_RISK_BUCKETS, PHASE_BUCKETS, parse_filter_values, state_features


StagePredicate = Callable[[dict[str, Any]], bool]


STAGE_PREDICATES: dict[str, StagePredicate] = {
    "duplicate768": lambda raw: int(raw["raw_count_768"]) >= 2,
    "duplicate768_no_built1536": lambda raw: int(raw["raw_count_768"]) >= 2 and int(raw["masked_count_1536"]) == 0,
    "threeplus768_no_built1536": lambda raw: int(raw["raw_count_768"]) >= 3 and int(raw["masked_count_1536"]) == 0,
    "four768_no_built1536": lambda raw: int(raw["raw_count_768"]) >= 4 and int(raw["masked_count_1536"]) == 0,
    "duplicate768_no_adjacent": lambda raw: int(raw["raw_count_768"]) >= 2 and not bool(raw["raw_has_adjacent_768"]),
    "adjacent768": lambda raw: bool(raw["raw_has_adjacent_768"]),
    "adjacent768_no_built1536": lambda raw: bool(raw["raw_has_adjacent_768"]) and int(raw["masked_count_1536"]) == 0,
    "one1536_duplicate768_no_adjacent": lambda raw: int(raw["masked_count_1536"]) >= 1
    and int(raw["raw_count_768"]) >= 2
    and not bool(raw["raw_has_adjacent_768"]),
    "one1536_adjacent768": lambda raw: int(raw["masked_count_1536"]) >= 1 and bool(raw["raw_has_adjacent_768"]),
    "duplicate1536": lambda raw: int(raw["raw_count_1536"]) >= 2,
    "adjacent1536": lambda raw: bool(raw["raw_has_adjacent_1536"]),
}


def _flatten_paths(path_groups: list[list[Path]] | None) -> list[Path]:
    if not path_groups:
        return []
    return [path for group in path_groups for path in group]


def parse_stages(text: str | None) -> list[str]:
    raw = text or "duplicate768_no_adjacent"
    stages: list[str] = []
    seen: set[str] = set()
    for part in raw.split(","):
        stage = part.strip()
        if not stage:
            continue
        if stage not in STAGE_PREDICATES:
            raise ValueError(f"Unsupported support-ladder stage: {stage}")
        if stage not in seen:
            stages.append(stage)
            seen.add(stage)
    if not stages:
        raise ValueError("at least one stage is required")
    return stages


def _load_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    records = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError(f"{path} does not contain records[]")
    return [
        {**record, "_source_json": str(path), "_record_index": idx}
        for idx, record in enumerate(records)
        if isinstance(record, dict)
    ]


def load_records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        records.extend(_load_records(Path(path)))
    return records


def _record_starter(record: dict[str, Any], default: int | None) -> int | None:
    value = record.get("starter_tile", default)
    return None if value is None else int(value)


def _record_seed(record: dict[str, Any], fallback: int) -> int:
    value = record.get("source_seed", record.get("seed"))
    return int(value) if value is not None else int(fallback)


def _source_key(record: dict[str, Any]) -> str:
    replay = record.get("source_replay", record.get("_source_json", "unknown_replay"))
    seed = record.get("source_seed", record.get("seed", "unknown_seed"))
    return f"{replay}|{seed}"


def _state_key(state_payload_dict: dict[str, Any]) -> str:
    return json.dumps(
        {
            "board": state_payload_dict.get("board"),
            "preview": state_payload_dict.get("preview"),
            "tile_cycle": state_payload_dict.get("tile_cycle"),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _record_id(source_record: dict[str, Any], stage: str, raw: dict[str, Any]) -> str:
    source_replay = str(source_record.get("source_replay", source_record.get("_source_json", "")))
    seed = source_record.get("source_seed", source_record.get("seed", "unknown"))
    frame = source_record.get("source_frame_index", source_record.get("frame_index", "unknown"))
    state_payload_dict = source_record.get("state") if isinstance(source_record.get("state"), dict) else {}
    digest_raw = json.dumps(
        {
            "source_replay": source_replay,
            "seed": seed,
            "frame": frame,
            "stage": stage,
            "raw": raw,
            "state": _state_key(state_payload_dict),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.blake2s(digest_raw.encode("utf-8"), digest_size=6).hexdigest()
    return safe_name(f"stage_{stage}_seed{seed}_frame{frame}_{digest}", max_length=128)


def _sort_key(record: dict[str, Any], sort_by: str) -> tuple[object, ...]:
    if sort_by == "source":
        return (
            int(record.get("source_order", record.get("_record_index", 0))),
            int(record.get("source_frame_index", record.get("frame_index", 0))),
        )
    if sort_by == "score":
        return (
            -int(record.get("score_minus_starter", 0)),
            -int(record.get("raw_highest_adjacent_pair_tile", 0)),
            -int(record.get("raw_highest_duplicate_tile", 0)),
            int(record.get("_record_index", 0)),
        )
    if sort_by == "support":
        return (
            -int(record.get("raw_highest_adjacent_pair_tile", 0)),
            -int(record.get("raw_highest_duplicate_tile", 0)),
            -int(record.get("raw_count_1536", 0)),
            -int(record.get("raw_count_768", 0)),
            -int(record.get("score_minus_starter", 0)),
            int(record.get("_record_index", 0)),
        )
    raise ValueError(f"Unsupported sort_by: {sort_by}")


def _stage_record(
    *,
    source_record: dict[str, Any],
    stage: str,
    starter_tile: int | None,
    default_seed: int,
) -> dict[str, Any] | None:
    state_payload = source_record.get("state")
    if not isinstance(state_payload, dict):
        return None
    state = state_from_payload(state_payload)
    if state.game_over:
        return None
    seed = _record_seed(source_record, default_seed)
    sim = ThreesSim(np.random.default_rng(seed), starter_tile=starter_tile)
    features = state_features(state, sim, starter_tile)
    raw = raw_ladder_features(state.board, starter_tile)
    return {
        "id": _record_id(source_record, stage, raw),
        "kind": "support_ladder_stage_state",
        "stage": stage,
        "source_record_id": source_record.get("id"),
        "source_replay": str(source_record.get("source_replay", source_record.get("_source_json", "unknown"))),
        "source_origin": source_record.get("source_origin"),
        "source_policy": source_record.get("source_policy"),
        "source_policy_family": source_record.get("source_policy_family"),
        "source_seed": source_record.get("source_seed", source_record.get("seed")),
        "seed": source_record.get("source_seed", source_record.get("seed")),
        "root_origin": source_record.get("root_origin"),
        "root_replay": source_record.get("root_replay"),
        "root_seed": source_record.get("root_seed"),
        "root_frame_index": source_record.get("root_frame_index"),
        "root_move_count": source_record.get("root_move_count"),
        "root_score": source_record.get("root_score"),
        "root_policy": source_record.get("root_policy"),
        "root_policy_family": source_record.get("root_policy_family"),
        "root_is_genuine": source_record.get("root_is_genuine"),
        "ancestry_key": source_record.get("ancestry_key"),
        "source_order": int(source_record.get("source_order", source_record.get("_record_index", 0))),
        "source_frame_index": int(source_record.get("source_frame_index", source_record.get("frame_index", 0))),
        "frame_position": source_record.get("frame_position"),
        "source_next_action": source_record.get("source_next_action"),
        "starter_tile": starter_tile,
        "move_count": int(state.move_count),
        "score": int(score_board(state.board)),
        "score_minus_starter": int(features["score_minus_starter"]),
        "max_tile": int(features["max_tile"]),
        "max_tile_excl_starter": int(max_tile_excluding_initial_starter(state.board, starter_tile)),
        "phase": str(features["phase"]),
        "corner_risk": str(features["corner_risk"]),
        "stratum": str(features["stratum"]),
        "empty_count": int(features["empty_count"]),
        "legal_count": int(len(sim.legal_actions(state))),
        "preview": str(features["preview"]),
        "large_pending": bool(features["large_pending"]),
        "raw_count_768": int(raw["raw_count_768"]),
        "raw_count_1536": int(raw["raw_count_1536"]),
        "raw_count_3072": int(raw["raw_count_3072"]),
        "raw_count_6144": int(raw["raw_count_6144"]),
        "raw_highest_duplicate_tile": int(raw["raw_highest_duplicate_tile"]),
        "raw_highest_adjacent_pair_tile": int(raw["raw_highest_adjacent_pair_tile"]),
        "raw_has_adjacent_768": bool(raw["raw_has_adjacent_768"]),
        "raw_has_adjacent_1536": bool(raw["raw_has_adjacent_1536"]),
        "masked_count_1536": int(raw["masked_count_1536"]),
        "masked_count_3072": int(raw["masked_count_3072"]),
        "features": {**features, **raw, "legal_count": int(len(sim.legal_actions(state)))},
        "state": state_payload,
    }


def collect_stage_records(
    source_records: Iterable[dict[str, Any]],
    *,
    stages: Iterable[str],
    min_tile: int = 3072,
    phase_filter: set[str] | None = None,
    corner_risk_filter: set[str] | None = None,
    default_starter_tile: int | None = 1536,
    max_records: int = 0,
    max_per_source: int = 0,
    max_per_stage: int = 0,
    sort_by: str = "support",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    requested_stages = list(stages)
    candidates: list[tuple[str, str, dict[str, Any]]] = []
    records: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    stage_counts: Counter[str] = Counter()
    seen_states: set[str] = set()
    source_list = list(source_records)
    scanned = 0

    for idx, source_record in enumerate(source_list):
        starter_tile = _record_starter(source_record, default_starter_tile)
        source_key = _source_key(source_record)
        stage_record: dict[str, Any] | None = None
        matched_stage: str | None = None
        for stage in requested_stages:
            candidate = stage_record
            if candidate is None:
                candidate = _stage_record(
                    source_record=source_record,
                    stage=stage,
                    starter_tile=starter_tile,
                    default_seed=idx,
                )
                scanned += 1
            if candidate is None:
                rejected["bad_or_terminal_state"] += 1
                break
            if int(candidate["max_tile_excl_starter"]) < int(min_tile):
                rejected["below_min_tile"] += 1
                break
            if phase_filter is not None and str(candidate["phase"]) not in phase_filter:
                rejected["phase_filter"] += 1
                break
            if corner_risk_filter is not None and str(candidate["corner_risk"]) not in corner_risk_filter:
                rejected["corner_risk_filter"] += 1
                break
            raw = candidate["features"]
            if STAGE_PREDICATES[stage](raw):
                matched_stage = stage
                stage_record = candidate
                break
            stage_record = candidate
        if stage_record is None or matched_stage is None:
            rejected["stage_filter"] += 1
            continue
        stage_record["stage"] = matched_stage
        stage_record["id"] = _record_id(source_record, matched_stage, stage_record["features"])
        candidates.append((source_key, matched_stage, stage_record))

    candidates.sort(key=lambda item: _sort_key(item[2], sort_by))
    for source_key, matched_stage, stage_record in candidates:
        if max_per_source > 0 and source_counts[source_key] >= int(max_per_source):
            rejected["max_per_source"] += 1
            continue
        if max_per_stage > 0 and stage_counts[matched_stage] >= int(max_per_stage):
            rejected["max_per_stage"] += 1
            continue
        key = _state_key(stage_record["state"])
        if key in seen_states:
            rejected["duplicate_state"] += 1
            continue
        seen_states.add(key)
        records.append(stage_record)
        source_counts[source_key] += 1
        stage_counts[matched_stage] += 1
        if max_records > 0 and len(records) >= int(max_records):
            break

    summary = summarize_records(
        records,
        source_records=len(source_list),
        scanned_records=scanned,
        stages=requested_stages,
        min_tile=min_tile,
        phase_filter=phase_filter,
        corner_risk_filter=corner_risk_filter,
        max_records=max_records,
        max_per_source=max_per_source,
        max_per_stage=max_per_stage,
        sort_by=sort_by,
        rejected=dict(rejected),
    )
    return records, summary


def summarize_records(
    records: list[dict[str, Any]],
    *,
    source_records: int,
    scanned_records: int,
    stages: list[str],
    min_tile: int,
    phase_filter: set[str] | None,
    corner_risk_filter: set[str] | None,
    max_records: int,
    max_per_source: int,
    max_per_stage: int,
    sort_by: str,
    rejected: dict[str, int],
) -> dict[str, Any]:
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_records": int(source_records),
        "scanned_records": int(scanned_records),
        "records": len(records),
        "stages": stages,
        "min_tile": int(min_tile),
        "phase_filter": sorted(phase_filter) if phase_filter is not None else None,
        "corner_risk_filter": sorted(corner_risk_filter) if corner_risk_filter is not None else None,
        "max_records": int(max_records),
        "max_per_source": int(max_per_source),
        "max_per_stage": int(max_per_stage),
        "sort_by": sort_by,
        "by_stage": dict(Counter(str(record.get("stage")) for record in records)),
        "by_stratum": dict(Counter(str(record.get("stratum")) for record in records)),
        "source_replays": len({str(record.get("source_replay")) for record in records}),
        "source_seeds": len({str(record.get("source_seed")) for record in records}),
        "raw_adjacent_768": sum(bool(record.get("raw_has_adjacent_768")) for record in records),
        "raw_duplicate_768": sum(int(record.get("raw_count_768", 0)) >= 2 for record in records),
        "one_built_1536": sum(int(record.get("masked_count_1536", 0)) >= 1 for record in records),
        "raw_duplicate_1536": sum(int(record.get("raw_count_1536", 0)) >= 2 for record in records),
        "rejected": rejected,
    }


def write_html(path: Path, payload: dict[str, Any]) -> None:
    summary = payload.get("summary", {})
    records = payload.get("records", [])

    def cell(value: object) -> str:
        return escape(str(value))

    rows = []
    for record in records[:300] if isinstance(records, list) else []:
        if not isinstance(record, dict):
            continue
        rows.append(
            "<tr>"
            f"<td>{cell(record.get('stage'))}</td>"
            f"<td>{cell(record.get('source_seed'))}</td>"
            f"<td>{cell(record.get('source_frame_index'))}</td>"
            f"<td>{cell(record.get('stratum'))}</td>"
            f"<td>{cell(record.get('score_minus_starter'))}</td>"
            f"<td>{cell(record.get('raw_count_768'))}</td>"
            f"<td>{cell(record.get('raw_has_adjacent_768'))}</td>"
            f"<td>{cell(record.get('masked_count_1536'))}</td>"
            f"<td>{cell(record.get('raw_highest_adjacent_pair_tile'))}</td>"
            f"<td>{cell(record.get('source_replay'))}</td>"
            "</tr>"
        )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Support-Ladder Stage Reservoir</title>
  <style>
    :root {{ color-scheme: dark; --bg:#101318; --panel:#171d24; --line:#34404d; --text:#edf2f7; --muted:#aab6c2; --gold:#f2c14e; }}
    body {{ margin:0; background:var(--bg); color:var(--text); font-family:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ width:min(1240px, calc(100vw - 32px)); margin:0 auto; padding:24px 0 40px; }}
    h1 {{ margin:0 0 8px; font-size:24px; }}
    .muted {{ color:var(--muted); }}
    .cards {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(150px, 1fr)); gap:10px; margin:18px 0; }}
    .card {{ border:1px solid var(--line); background:var(--panel); border-radius:8px; padding:12px; }}
    .label {{ color:var(--muted); font-size:11px; text-transform:uppercase; }}
    .value {{ margin-top:4px; color:var(--gold); font-size:22px; font-weight:800; }}
    table {{ width:100%; border-collapse:collapse; font-size:12px; font-variant-numeric:tabular-nums; }}
    th, td {{ border-bottom:1px solid var(--line); padding:7px 8px; text-align:right; vertical-align:top; }}
    th:first-child, td:first-child, th:nth-child(4), td:nth-child(4), th:last-child, td:last-child {{ text-align:left; }}
    td:last-child {{ max-width:360px; overflow-wrap:anywhere; color:var(--muted); }}
    pre {{ white-space:pre-wrap; overflow-wrap:anywhere; color:var(--muted); }}
  </style>
</head>
<body>
  <main>
    <h1>Support-Ladder Stage Reservoir</h1>
    <p class="muted">Real state starts selected by raw support-ladder stage.</p>
    <section class="cards">
      <div class="card"><div class="label">Records</div><div class="value">{cell(summary.get('records', 0))}</div></div>
      <div class="card"><div class="label">Source Seeds</div><div class="value">{cell(summary.get('source_seeds', 0))}</div></div>
      <div class="card"><div class="label">Duplicate 768</div><div class="value">{cell(summary.get('raw_duplicate_768', 0))}</div></div>
      <div class="card"><div class="label">Built 1536</div><div class="value">{cell(summary.get('one_built_1536', 0))}</div></div>
    </section>
    <table><thead><tr><th>Stage</th><th>Seed</th><th>Frame</th><th>Stratum</th><th>Score - Starter</th><th>768 Count</th><th>Adj 768</th><th>Built 1536</th><th>Adj Pair</th><th>Source</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
    <h2>Summary JSON</h2>
    <pre>{escape(json.dumps(summary, indent=2, sort_keys=True))}</pre>
  </main>
</body>
</html>
"""
    path.write_text(html)


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    paths = _flatten_paths(args.state_json)
    source_records = load_records(paths)
    stages = parse_stages(args.stages)
    phase_filter = parse_filter_values(
        args.phase_filter,
        allowed=PHASE_BUCKETS,
        aliases={
            "early": "early_lt384",
            "mid": "mid_384_768",
            "middle": "mid_384_768",
            "late": "late_1536",
            "endgame": "endgame_3072p",
        },
        label="phase",
    )
    corner_risk_filter = parse_filter_values(
        args.corner_risk_filter,
        allowed=CORNER_RISK_BUCKETS,
        aliases={
            "low": "low_corner_risk",
            "medium": "medium_corner_risk",
            "med": "medium_corner_risk",
            "high": "high_corner_risk",
        },
        label="corner risk",
    )
    starter_text = str(args.starter).strip().lower()
    default_starter = None if starter_text == "none" else int(starter_text)
    records, summary = collect_stage_records(
        source_records,
        stages=stages,
        min_tile=args.min_tile,
        phase_filter=phase_filter,
        corner_risk_filter=corner_risk_filter,
        default_starter_tile=default_starter,
        max_records=args.max_records,
        max_per_source=args.max_per_source,
        max_per_stage=args.max_per_stage,
        sort_by=args.sort_by,
    )
    payload = {
        "version": 1,
        "kind": "support_ladder_stage_reservoir",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_json": [str(path) for path in paths],
        "summary": summary,
        "records": records,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload["json"] = str(args.out_dir / "support_ladder_stage_reservoir.json")
    payload["records_json"] = str(args.out_dir / "records.json")
    payload["html"] = str(args.out_dir / "support_ladder_stage_reservoir.html")
    write_json(args.out_dir / "support_ladder_stage_reservoir.json", payload)
    write_json(args.out_dir / "records.json", records)
    write_html(args.out_dir / "support_ladder_stage_reservoir.html", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-json", type=Path, nargs="+", action="append", required=True)
    parser.add_argument("--stages", default="duplicate768_no_adjacent")
    parser.add_argument("--min-tile", type=int, default=3072)
    parser.add_argument("--phase-filter", action="append")
    parser.add_argument("--corner-risk-filter", action="append")
    parser.add_argument("--starter", default="1536")
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--max-per-source", type=int, default=0)
    parser.add_argument("--max-per-stage", type=int, default=0)
    parser.add_argument("--sort-by", choices=["source", "score", "support"], default="support")
    parser.add_argument("--out-dir", type=Path, default=Path("threes_rl/runs/forensics/support_ladder_stage_reservoir/latest"))
    args = parser.parse_args()
    payload = run_from_args(args)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"json={payload['json']}")
    print(f"records={payload['records_json']}")
    print(f"html={payload['html']}")


if __name__ == "__main__":
    main()
