"""Extract exact support-chain milestone states from replay files."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import time
from collections import Counter
from html import escape
from pathlib import Path
from typing import Any, Callable

from threes_rl.run_artifacts import safe_name, write_json
from threes_rl.sim import score_board
from threes_rl.support_chain_progression import (
    _frame_board,
    _frame_features,
    _frame_move_count,
    _starter_from_replay,
)


MilestonePredicate = Callable[[dict[str, Any]], bool]

MILESTONE_PREDICATES: dict[str, MilestonePredicate] = {
    "raw_duplicate_768": lambda features: int(features["raw_highest_duplicate_tile"]) >= 768,
    "raw_adjacent_768": lambda features: bool(features["raw_has_adjacent_768"]),
    "raw_adjacent_768_no_1536": lambda features: bool(features["raw_has_adjacent_768"])
    and int(features["raw_count_1536"]) == 0,
    "raw_adjacent_768_with_1536": lambda features: bool(features["raw_has_adjacent_768"])
    and int(features["raw_count_1536"]) >= 1,
    "raw_three_768": lambda features: int(features["raw_count_768"]) >= 3,
    "raw_four_768": lambda features: int(features["raw_count_768"]) >= 4,
    "raw_four_768_no_1536": lambda features: int(features["raw_count_768"]) >= 4
    and int(features["raw_count_1536"]) == 0,
    "raw_one_1536": lambda features: int(features["raw_count_1536"]) >= 1,
    "raw_duplicate_1536": lambda features: int(features["raw_count_1536"]) >= 2,
    "raw_adjacent_1536": lambda features: bool(features["raw_has_adjacent_1536"]),
}


def _flatten_paths(path_groups: list[list[Path]] | None) -> list[Path]:
    if not path_groups:
        return []
    return [path for group in path_groups for path in group]


def _glob_paths(patterns: list[str] | None) -> list[Path]:
    if not patterns:
        return []
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(Path(match) for match in glob.glob(pattern, recursive=True))
    return sorted(path for path in paths if path.is_file())


def parse_milestones(text: str | None) -> list[str]:
    raw = text or "raw_adjacent_768"
    milestones: list[str] = []
    seen: set[str] = set()
    for part in raw.split(","):
        name = part.strip()
        if not name:
            continue
        if name not in MILESTONE_PREDICATES:
            raise ValueError(f"Unsupported milestone: {name}")
        if name not in seen:
            milestones.append(name)
            seen.add(name)
    if not milestones:
        raise ValueError("at least one milestone is required")
    return milestones


def _outcome_filter(text: str | None) -> set[str] | None:
    if text is None or not text.strip() or text.strip().lower() == "all":
        return None
    values = {part.strip().lower() for part in text.split(",") if part.strip()}
    bad = values - {"success", "failure"}
    if bad:
        raise ValueError(f"Unsupported outcome filter: {sorted(bad)}")
    return values


def _first_index(
    features_by_idx: list[dict[str, Any] | None],
    predicate: MilestonePredicate,
    start: int = 0,
    end: int | None = None,
) -> int | None:
    stop = len(features_by_idx) if end is None else min(len(features_by_idx), int(end))
    for idx in range(max(0, int(start)), stop):
        features = features_by_idx[idx]
        if features is not None and predicate(features):
            return idx
    return None


def _record_id(path: Path, replay: dict[str, Any], milestone: str, idx: int) -> str:
    raw = json.dumps(
        {
            "source_replay": str(path),
            "seed": replay.get("seed"),
            "milestone": milestone,
            "idx": int(idx),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.blake2s(raw.encode("utf-8"), digest_size=6).hexdigest()
    return safe_name(f"milestone_{milestone}_seed{replay.get('seed', 'unknown')}_frame{idx}_{digest}", max_length=128)


def _score_from_frame(frame: dict[str, Any]) -> int:
    state = frame.get("state") if isinstance(frame, dict) else None
    if isinstance(state, dict) and state.get("score") is not None:
        return int(state["score"])
    board = _frame_board(frame)
    return int(score_board(board)) if board is not None else 0


def extract_replay_records(
    path: Path,
    *,
    milestones: list[str],
    outcome_filter: set[str] | None,
    pre_event_only: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rejected: Counter[str] = Counter()
    replay = json.loads(Path(path).read_text())
    if not isinstance(replay, dict):
        rejected["bad_replay"] += 1
        return [], dict(rejected)
    starter_tile = _starter_from_replay(replay)
    frames = [frame for frame in replay.get("frames", []) if isinstance(frame, dict)]
    if not frames:
        rejected["no_frames"] += 1
        return [], dict(rejected)
    features_by_idx = [_frame_features(frame, starter_tile) for frame in frames]
    first_3072_idx = _first_index(features_by_idx, lambda features: int(features["max_tile_excl_starter"]) >= 3072)
    if first_3072_idx is None:
        rejected["no_first_3072"] += 1
        return [], dict(rejected)
    event_idx = _first_index(
        features_by_idx,
        lambda features: int(features["raw_count_3072"]) >= 2 or int(features["max_tile_excl_starter"]) >= 6144,
        start=first_3072_idx,
    )
    outcome = "success" if event_idx is not None else "failure"
    if outcome_filter is not None and outcome not in outcome_filter:
        rejected["outcome_filter"] += 1
        return [], dict(rejected)

    records: list[dict[str, Any]] = []
    search_end = event_idx if bool(pre_event_only) and event_idx is not None else None
    for milestone in milestones:
        idx = _first_index(features_by_idx, MILESTONE_PREDICATES[milestone], start=first_3072_idx, end=search_end)
        if idx is None:
            rejected[f"missing_{milestone}"] += 1
            continue
        frame = frames[idx]
        state_payload = frame.get("state")
        features = features_by_idx[idx]
        if not isinstance(state_payload, dict) or features is None:
            rejected["bad_state"] += 1
            continue
        record = {
            "id": _record_id(Path(path), replay, milestone, idx),
            "kind": "support_chain_milestone_state",
            "target_milestone": milestone,
            "outcome": outcome,
            "source_replay": str(path),
            "source_policy": replay.get("policy"),
            "source_seed": replay.get("seed"),
            "seed": replay.get("seed"),
            "source_frame_index": int(frame.get("index", idx)),
            "frame_position": int(idx),
            "starter_tile": starter_tile,
            "move_count": _frame_move_count(frame, idx),
            "score": _score_from_frame(frame),
            "score_minus_starter": int(features["score_minus_starter"]),
            "max_tile_excl_starter": int(features["max_tile_excl_starter"]),
            "empty_count": int(features["empty_count"]),
            "raw_count_768": int(features["raw_count_768"]),
            "raw_count_1536": int(features["raw_count_1536"]),
            "raw_count_3072": int(features["raw_count_3072"]),
            "raw_count_6144": int(features["raw_count_6144"]),
            "raw_highest_duplicate_tile": int(features["raw_highest_duplicate_tile"]),
            "raw_highest_adjacent_pair_tile": int(features["raw_highest_adjacent_pair_tile"]),
            "raw_has_adjacent_768": bool(features["raw_has_adjacent_768"]),
            "raw_has_adjacent_1536": bool(features["raw_has_adjacent_1536"]),
            "first_3072_frame_position": int(first_3072_idx),
            "frames_after_first_3072": int(idx) - int(first_3072_idx),
            "event_frame_position": event_idx,
            "frames_before_event": None if event_idx is None else int(event_idx) - int(idx),
            "features": features,
            "state": state_payload,
        }
        records.append(record)
    return records, dict(rejected)


def summarize(records: list[dict[str, Any]], *, source_replays: int, rejected: dict[str, int]) -> dict[str, Any]:
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_replays": int(source_replays),
        "records": len(records),
        "by_milestone": dict(Counter(str(record.get("target_milestone")) for record in records)),
        "by_outcome": dict(Counter(str(record.get("outcome")) for record in records)),
        "raw_count_768": dict(Counter(str(record.get("raw_count_768")) for record in records)),
        "raw_count_1536": dict(Counter(str(record.get("raw_count_1536")) for record in records)),
        "rejected": rejected,
    }


def write_html(path: Path, payload: dict[str, Any]) -> None:
    summary = payload.get("summary", {})
    records = payload.get("records", [])

    def cell(value: object) -> str:
        return escape(str(value))

    rows = []
    for record in records if isinstance(records, list) else []:
        rows.append(
            "<tr>"
            f"<td>{cell(record.get('target_milestone'))}</td>"
            f"<td>{cell(record.get('outcome'))}</td>"
            f"<td>{cell(record.get('source_seed'))}</td>"
            f"<td>{cell(record.get('move_count'))}</td>"
            f"<td>{cell(record.get('frames_after_first_3072'))}</td>"
            f"<td>{cell(record.get('frames_before_event'))}</td>"
            f"<td>{cell(record.get('raw_count_768'))}</td>"
            f"<td>{cell(record.get('raw_count_1536'))}</td>"
            f"<td>{cell(record.get('source_replay'))}</td>"
            "</tr>"
        )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Support-Chain Milestone Reservoir</title>
  <style>
    :root {{ color-scheme: dark; --bg:#101318; --panel:#171d24; --line:#34404d; --text:#edf2f7; --muted:#aab6c2; --gold:#f2c14e; }}
    body {{ margin:0; background:var(--bg); color:var(--text); font-family:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ width:min(1180px, calc(100vw - 32px)); margin:0 auto; padding:24px 0 40px; }}
    h1 {{ margin:0 0 8px; font-size:24px; }}
    .muted {{ color:var(--muted); }}
    .cards {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(150px, 1fr)); gap:10px; margin:18px 0; }}
    .card {{ border:1px solid var(--line); background:var(--panel); border-radius:8px; padding:12px; }}
    .label {{ color:var(--muted); font-size:11px; text-transform:uppercase; }}
    .value {{ margin-top:4px; color:var(--gold); font-size:22px; font-weight:800; }}
    table {{ width:100%; border-collapse:collapse; font-size:12px; font-variant-numeric:tabular-nums; }}
    th, td {{ border-bottom:1px solid var(--line); padding:7px 8px; text-align:right; vertical-align:top; }}
    th:last-child, td:last-child {{ text-align:left; max-width:420px; overflow-wrap:anywhere; color:var(--muted); }}
    pre {{ white-space:pre-wrap; overflow-wrap:anywhere; color:var(--muted); }}
  </style>
</head>
<body>
  <main>
    <h1>Support-Chain Milestone Reservoir</h1>
    <p class="muted">Exact replay states at first support-chain milestones.</p>
    <section class="cards">
      <div class="card"><div class="label">Records</div><div class="value">{cell(summary.get('records', 0))}</div></div>
      <div class="card"><div class="label">Replays</div><div class="value">{cell(summary.get('source_replays', 0))}</div></div>
      <div class="card"><div class="label">Milestones</div><div class="value">{cell(summary.get('by_milestone', {}))}</div></div>
    </section>
    <table><thead><tr><th>Milestone</th><th>Outcome</th><th>Seed</th><th>Move</th><th>Frames After 3072</th><th>Frames Before Event</th><th>Raw 768</th><th>Raw 1536</th><th>Replay</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
    <pre>{escape(json.dumps(summary, indent=2, sort_keys=True))}</pre>
  </main>
</body>
</html>
"""
    path.write_text(html)


def run(
    replay_paths: list[Path],
    *,
    milestones: list[str],
    outcome_filter: set[str] | None,
    out_dir: Path,
    pre_event_only: bool = False,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    for path in replay_paths:
        try:
            replay_records, replay_rejected = extract_replay_records(
                path,
                milestones=milestones,
                outcome_filter=outcome_filter,
                pre_event_only=pre_event_only,
            )
        except (OSError, json.JSONDecodeError, ValueError):
            rejected["bad_replay"] += 1
            continue
        records.extend(replay_records)
        rejected.update(replay_rejected)
    payload = {
        "version": 1,
        "kind": "support_chain_milestone_reservoir",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "milestones": milestones,
        "outcome_filter": sorted(outcome_filter) if outcome_filter is not None else None,
        "pre_event_only": bool(pre_event_only),
        "source_replays": [str(path) for path in replay_paths],
        "summary": summarize(records, source_replays=len(replay_paths), rejected=dict(rejected)),
        "records": records,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    payload["json"] = str(out_dir / "support_chain_milestone_reservoir.json")
    payload["records_json"] = str(out_dir / "records.json")
    payload["html"] = str(out_dir / "support_chain_milestone_reservoir.html")
    write_json(out_dir / "support_chain_milestone_reservoir.json", payload)
    write_json(out_dir / "records.json", records)
    write_html(out_dir / "support_chain_milestone_reservoir.html", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-json", type=Path, nargs="+", action="append", default=[])
    parser.add_argument("--replay-glob", action="append", default=[])
    parser.add_argument("--milestones", default="raw_adjacent_768")
    parser.add_argument("--outcome-filter", default="all")
    parser.add_argument(
        "--pre-event-only",
        action="store_true",
        help="For success replays, search milestone frames only before the second-3072/6144 event.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("threes_rl/runs/forensics/support_chain_milestones/latest"))
    args = parser.parse_args()
    replay_paths = _flatten_paths(args.replay_json) + _glob_paths(args.replay_glob)
    if not replay_paths:
        raise SystemExit("No replay JSONs matched.")
    payload = run(
        replay_paths,
        milestones=parse_milestones(args.milestones),
        outcome_filter=_outcome_filter(args.outcome_filter),
        out_dir=args.out_dir,
        pre_event_only=bool(args.pre_event_only),
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"json={payload['json']}")
    print(f"records={payload['records_json']}")
    print(f"html={payload['html']}")


if __name__ == "__main__":
    main()
