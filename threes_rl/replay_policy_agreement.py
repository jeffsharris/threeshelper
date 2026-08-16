"""Compare a policy's root choices with recorded replay actions."""

from __future__ import annotations

import argparse
import glob
import json
import time
from collections import Counter
from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

import numpy as np

from threes_rl.compare_replays import state_from_payload
from threes_rl.eval import make_policy
from threes_rl.run_artifacts import safe_name, write_json
from threes_rl.sim import DIRECTION_NAMES, ThreesSim, direction_index
from threes_rl.swing_label import (
    CORNER_RISK_BUCKETS,
    PHASE_BUCKETS,
    clear_policy_caches,
    parse_filter_values,
    state_features,
)

PHASE_ALIASES = {
    "early": "early_lt384",
    "mid": "mid_384_768",
    "middle": "mid_384_768",
    "late": "late_1536",
    "endgame": "endgame_3072p",
}
CORNER_RISK_ALIASES = {
    "low": "low_corner_risk",
    "medium": "medium_corner_risk",
    "med": "medium_corner_risk",
    "high": "high_corner_risk",
}


@dataclass
class AgreementRecord:
    id: str
    source_replay: str
    source_policy: str | None
    source_seed: int | None
    frame_index: int
    move_count: int
    phase: str
    corner_risk: str
    stratum: str
    score_minus_starter: int
    max_tile_excl_starter: int
    recorded_action: str
    policy_action: str
    action_match: bool
    recorded_rank: int | None
    recorded_in_top_two: bool
    policy_top_two: list[str]
    policy_margin: float | None
    normalized_policy_margin: float | None
    recorded_value: float | None
    best_value: float | None
    value_gap_to_recorded: float | None
    normalized_value_gap_to_recorded: float | None


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


def parse_phase_filters(values: list[str] | None) -> set[str] | None:
    return parse_filter_values(values, allowed=PHASE_BUCKETS, aliases=PHASE_ALIASES, label="phase")


def parse_corner_risk_filters(values: list[str] | None) -> set[str] | None:
    return parse_filter_values(
        values,
        allowed=CORNER_RISK_BUCKETS,
        aliases=CORNER_RISK_ALIASES,
        label="corner-risk",
    )


def _int_or_none(value: object) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _starter_from_replay(replay: dict[str, Any], default: int | None) -> int | None:
    value = replay.get("starter_tile", default)
    if value is None:
        return None
    return int(value)


def _move_action(frame: object) -> str | None:
    if not isinstance(frame, dict):
        return None
    move = frame.get("move")
    if not isinstance(move, dict):
        return None
    action = move.get("action")
    return str(action) if action is not None else None


def _load_state_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    records = payload.get("records", []) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError(f"{path} does not contain a records list")
    out: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        state_payload = record.get("state")
        if not isinstance(state_payload, dict):
            continue
        out.append({**record, "_source_json": str(path), "_record_index": int(idx)})
    return out


def _load_state_records_from_paths(paths: Iterable[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        records.extend(_load_state_records(Path(path)))
    return records


def _recorded_action_from_state_record(record: dict[str, Any]) -> str | None:
    for key in ("source_next_action", "recorded_action", "next_action"):
        value = record.get(key)
        if value is not None:
            return str(value)
    return None


def _starter_from_record(record: dict[str, Any], default: int | None) -> int | None:
    value = record.get("starter_tile", default)
    if value is None:
        return None
    return int(value)


def _ranked_action_rows(policy: object, state: Any, sim: ThreesSim) -> list[dict[str, Any]]:
    values: list[tuple[int, float]]
    if hasattr(policy, "action_values"):
        values = [(int(action), float(value)) for action, value in policy.action_values(state, sim)]
    elif hasattr(policy, "_action_value"):
        clear_policy_caches(policy)
        depth = policy._root_depth(state) if hasattr(policy, "_root_depth") else getattr(policy, "depth", 1)
        values = [
            (int(action), float(policy._action_value(state, sim, int(action), int(depth))))
            for action in sim.legal_actions(state)
        ]
    else:
        action = int(policy(state, sim, np.random.default_rng(0)))
        values = [(action, 0.0)]

    rows = [
        {"action": int(action), "name": DIRECTION_NAMES[int(action)], "value": float(value)}
        for action, value in values
    ]
    rows.sort(key=lambda row: (-float(row["value"]), int(row["action"])))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = int(rank)
    return rows


def _margin(rows: list[dict[str, Any]]) -> tuple[float | None, float | None]:
    if len(rows) < 2:
        return None, None
    gap = float(rows[0]["value"]) - float(rows[1]["value"])
    scale = max(1.0, abs(float(rows[0]["value"])), abs(float(rows[1]["value"])))
    return gap, gap / scale


def _recorded_rank(rows: list[dict[str, Any]], recorded_action: str) -> tuple[int | None, float | None]:
    for row in rows:
        if str(row["name"]) == recorded_action:
            return int(row["rank"]), float(row["value"])
    return None, None


def _record_id(source_replay: Path, seed: int | None, frame_index: int, recorded_action: str) -> str:
    return safe_name(
        f"{source_replay.stem}_seed{seed}_frame{frame_index}_{recorded_action}",
        max_length=120,
    )


def _agreement_record(
    *,
    policy: object,
    state_payload: dict[str, Any],
    recorded_action: str,
    source_replay: str,
    source_policy: str | None,
    source_seed: int | None,
    starter_tile: int | None,
    frame_index: int,
    fallback_id: str,
) -> tuple[AgreementRecord | None, str | None]:
    try:
        direction_index(recorded_action)
    except ValueError:
        return None, "bad_recorded_action"
    try:
        state = state_from_payload(state_payload)
    except (TypeError, ValueError):
        return None, "bad_state"
    sim_seed = source_seed if source_seed is not None else 0
    sim = ThreesSim(np.random.default_rng(int(sim_seed)), starter_tile=starter_tile)
    if state.game_over or not sim.legal_actions(state):
        return None, "game_over_or_no_legal"
    features = state_features(state, sim, starter_tile)
    rows = _ranked_action_rows(policy, state, sim)
    if not rows:
        return None, "no_policy_values"
    policy_action = str(rows[0]["name"])
    rank, recorded_value = _recorded_rank(rows, recorded_action)
    best_value = float(rows[0]["value"])
    value_gap = None if recorded_value is None else best_value - float(recorded_value)
    scale = max(1.0, abs(best_value), abs(float(recorded_value) if recorded_value is not None else 0.0))
    normalized_gap = None if value_gap is None else float(value_gap) / scale
    margin, normalized_margin = _margin(rows)
    return (
        AgreementRecord(
            id=fallback_id,
            source_replay=source_replay,
            source_policy=source_policy,
            source_seed=source_seed,
            frame_index=int(frame_index),
            move_count=int(state.move_count),
            phase=str(features["phase"]),
            corner_risk=str(features["corner_risk"]),
            stratum=str(features["stratum"]),
            score_minus_starter=int(features["score_minus_starter"]),
            max_tile_excl_starter=int(features["max_tile_excl_starter"]),
            recorded_action=recorded_action,
            policy_action=policy_action,
            action_match=policy_action == recorded_action,
            recorded_rank=rank,
            recorded_in_top_two=bool(rank is not None and rank <= 2),
            policy_top_two=[str(row["name"]) for row in rows[:2]],
            policy_margin=margin,
            normalized_policy_margin=normalized_margin,
            recorded_value=recorded_value,
            best_value=best_value,
            value_gap_to_recorded=value_gap,
            normalized_value_gap_to_recorded=normalized_gap,
        ),
        None,
    )


def _accept_features(
    features: dict[str, Any],
    *,
    min_tile: int,
    phase_filter: set[str] | None,
    corner_risk_filter: set[str] | None,
) -> str | None:
    if int(features["max_tile_excl_starter"]) < int(min_tile):
        return "below_min_tile"
    if phase_filter is not None and str(features.get("phase")) not in phase_filter:
        return "phase_filter"
    if corner_risk_filter is not None and str(features.get("corner_risk")) not in corner_risk_filter:
        return "corner_risk_filter"
    return None


def scan_replays(
    *,
    policy: object,
    policy_spec: str,
    replay_paths: Iterable[Path],
    min_tile: int = 1536,
    phase_filter: set[str] | None = None,
    corner_risk_filter: set[str] | None = None,
    default_starter_tile: int | None = 1536,
    max_records: int = 0,
    high_confidence_margin: float = 0.01,
) -> dict[str, Any]:
    records: list[AgreementRecord] = []
    rejected: Counter[str] = Counter()
    scanned_moves = 0
    replay_count = 0
    resolved_paths = [Path(path) for path in replay_paths]

    for source_order, replay_path in enumerate(resolved_paths):
        try:
            replay = json.loads(replay_path.read_text())
        except (OSError, json.JSONDecodeError):
            rejected["bad_replay"] += 1
            continue
        if not isinstance(replay, dict):
            rejected["bad_replay"] += 1
            continue
        replay_count += 1
        starter_tile = _starter_from_replay(replay, default_starter_tile)
        seed = _int_or_none(replay.get("seed"))
        sim_seed = seed if seed is not None else source_order
        sim = ThreesSim(np.random.default_rng(int(sim_seed)), starter_tile=starter_tile)
        frames = replay.get("frames", [])
        if not isinstance(frames, list):
            rejected["bad_replay"] += 1
            continue
        for frame_pos in range(max(0, len(frames) - 1)):
            before = frames[frame_pos]
            after = frames[frame_pos + 1]
            if not isinstance(before, dict):
                continue
            recorded_action = _move_action(after)
            if recorded_action is None:
                rejected["missing_recorded_action"] += 1
                continue
            state_payload = before.get("state")
            if not isinstance(state_payload, dict):
                rejected["missing_state"] += 1
                continue
            state = state_from_payload(state_payload)
            scanned_moves += 1
            features = state_features(state, sim, starter_tile)
            reject_reason = _accept_features(
                features,
                min_tile=min_tile,
                phase_filter=phase_filter,
                corner_risk_filter=corner_risk_filter,
            )
            if reject_reason is not None:
                rejected[reject_reason] += 1
                continue
            frame_index = int(before.get("index", frame_pos))
            record, record_reject = _agreement_record(
                policy=policy,
                state_payload=state_payload,
                recorded_action=recorded_action,
                source_replay=str(replay_path),
                source_policy=str(replay.get("policy")) if replay.get("policy") is not None else None,
                source_seed=seed,
                starter_tile=starter_tile,
                frame_index=frame_index,
                fallback_id=_record_id(replay_path, seed, frame_index, recorded_action),
            )
            if record_reject is not None:
                rejected[record_reject] += 1
                continue
            assert record is not None
            records.append(record)
            if max_records > 0 and len(records) >= int(max_records):
                break
        if max_records > 0 and len(records) >= int(max_records):
            break

    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "policy": policy_spec,
        "source_replays": [str(path) for path in resolved_paths],
        "summary": summarize_records(
            records,
            replay_count=replay_count,
            scanned_moves=scanned_moves,
            rejected=dict(rejected),
            min_tile=min_tile,
            phase_filter=phase_filter,
            corner_risk_filter=corner_risk_filter,
            max_records=max_records,
            high_confidence_margin=high_confidence_margin,
        ),
        "records": [asdict(record) for record in records],
    }
    return payload


def scan_state_records(
    *,
    policy: object,
    policy_spec: str,
    state_records: Iterable[dict[str, Any]],
    min_tile: int = 1536,
    phase_filter: set[str] | None = None,
    corner_risk_filter: set[str] | None = None,
    default_starter_tile: int | None = 1536,
    max_records: int = 0,
    high_confidence_margin: float = 0.01,
) -> dict[str, Any]:
    records: list[AgreementRecord] = []
    rejected: Counter[str] = Counter()
    scanned_records = 0
    source_jsons: set[str] = set()
    for record_idx, source_record in enumerate(state_records):
        if not isinstance(source_record, dict):
            continue
        source_json = source_record.get("_source_json")
        if source_json is not None:
            source_jsons.add(str(source_json))
        state_payload = source_record.get("state")
        if not isinstance(state_payload, dict):
            rejected["missing_state"] += 1
            continue
        recorded_action = _recorded_action_from_state_record(source_record)
        if recorded_action is None:
            rejected["missing_recorded_action"] += 1
            continue
        starter_tile = _starter_from_record(source_record, default_starter_tile)
        seed = _int_or_none(source_record.get("seed", source_record.get("source_seed")))
        try:
            state = state_from_payload(state_payload)
        except (TypeError, ValueError):
            rejected["bad_state"] += 1
            continue
        sim = ThreesSim(np.random.default_rng(int(seed if seed is not None else record_idx)), starter_tile=starter_tile)
        scanned_records += 1
        features = state_features(state, sim, starter_tile)
        reject_reason = _accept_features(
            features,
            min_tile=min_tile,
            phase_filter=phase_filter,
            corner_risk_filter=corner_risk_filter,
        )
        if reject_reason is not None:
            rejected[reject_reason] += 1
            continue
        source_replay = str(source_record.get("source_replay", source_json or "state_records"))
        frame_index = int(source_record.get("source_frame_index", source_record.get("frame_index", record_idx)))
        fallback_id = str(source_record.get("id") or _record_id(Path(source_replay), seed, frame_index, recorded_action))
        agreement, record_reject = _agreement_record(
            policy=policy,
            state_payload=state_payload,
            recorded_action=recorded_action,
            source_replay=source_replay,
            source_policy=str(source_record.get("source_policy")) if source_record.get("source_policy") is not None else None,
            source_seed=seed,
            starter_tile=starter_tile,
            frame_index=frame_index,
            fallback_id=fallback_id,
        )
        if record_reject is not None:
            rejected[record_reject] += 1
            continue
        assert agreement is not None
        records.append(agreement)
        if max_records > 0 and len(records) >= int(max_records):
            break

    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "policy": policy_spec,
        "source_state_records": sorted(source_jsons),
        "summary": summarize_records(
            records,
            replay_count=len({record.source_replay for record in records}),
            scanned_moves=scanned_records,
            rejected=dict(rejected),
            min_tile=min_tile,
            phase_filter=phase_filter,
            corner_risk_filter=corner_risk_filter,
            max_records=max_records,
            high_confidence_margin=high_confidence_margin,
        ),
        "records": [asdict(record) for record in records],
    }
    payload["summary"]["source_state_record_files"] = len(source_jsons)
    payload["summary"]["scanned_state_records"] = int(scanned_records)
    return payload


def _bucket_summary(records: list[AgreementRecord]) -> dict[str, Any]:
    if not records:
        return {"records": 0, "action_matches": 0, "action_match_rate": 0.0, "recorded_in_top_two": 0}
    matches = sum(1 for record in records if record.action_match)
    top_two = sum(1 for record in records if record.recorded_in_top_two)
    gaps = [
        float(record.normalized_value_gap_to_recorded)
        for record in records
        if record.normalized_value_gap_to_recorded is not None
    ]
    return {
        "records": len(records),
        "action_matches": matches,
        "action_match_rate": matches / len(records),
        "recorded_in_top_two": top_two,
        "recorded_in_top_two_rate": top_two / len(records),
        "mean_normalized_gap_to_recorded": float(mean(gaps)) if gaps else None,
        "median_normalized_gap_to_recorded": float(median(gaps)) if gaps else None,
    }


def summarize_records(
    records: list[AgreementRecord],
    *,
    replay_count: int,
    scanned_moves: int,
    rejected: dict[str, int],
    min_tile: int,
    phase_filter: set[str] | None,
    corner_risk_filter: set[str] | None,
    max_records: int,
    high_confidence_margin: float,
) -> dict[str, Any]:
    base = _bucket_summary(records)
    misses = [record for record in records if not record.action_match]
    high_confidence = [
        record
        for record in misses
        if record.normalized_value_gap_to_recorded is not None
        and record.normalized_value_gap_to_recorded >= float(high_confidence_margin)
    ]
    by_phase = {
        phase: _bucket_summary([record for record in records if record.phase == phase])
        for phase in sorted({record.phase for record in records})
    }
    by_stratum = {
        stratum: _bucket_summary([record for record in records if record.stratum == stratum])
        for stratum in sorted({record.stratum for record in records})
    }
    largest_misses = sorted(
        misses,
        key=lambda record: (
            float(record.normalized_value_gap_to_recorded or 0.0),
            float(record.value_gap_to_recorded or 0.0),
            int(record.score_minus_starter),
        ),
        reverse=True,
    )[:50]
    return {
        **base,
        "source_replays": int(replay_count),
        "scanned_moves": int(scanned_moves),
        "rejected": dict(rejected),
        "min_tile": int(min_tile),
        "phase_filter": sorted(phase_filter) if phase_filter is not None else None,
        "corner_risk_filter": sorted(corner_risk_filter) if corner_risk_filter is not None else None,
        "max_records": int(max_records),
        "misses": len(misses),
        "miss_rate": len(misses) / len(records) if records else 0.0,
        "high_confidence_margin": float(high_confidence_margin),
        "high_confidence_misses": len(high_confidence),
        "high_confidence_miss_rate": len(high_confidence) / len(records) if records else 0.0,
        "recorded_rank_counts": dict(Counter(str(record.recorded_rank) for record in records)),
        "by_phase": by_phase,
        "by_stratum": by_stratum,
        "largest_misses": [asdict(record) for record in largest_misses],
    }


def write_html(path: Path, payload: dict[str, Any]) -> None:
    summary = payload.get("summary", {})
    misses = summary.get("largest_misses", []) if isinstance(summary, dict) else []

    def cell(value: object) -> str:
        return escape(str(value))

    rows = []
    for record in misses if isinstance(misses, list) else []:
        if not isinstance(record, dict):
            continue
        rows.append(
            "<tr>"
            f"<td>{cell(record.get('source_seed'))}</td>"
            f"<td>{cell(record.get('move_count'))}</td>"
            f"<td>{cell(record.get('stratum'))}</td>"
            f"<td>{cell(record.get('score_minus_starter'))}</td>"
            f"<td>{cell(record.get('max_tile_excl_starter'))}</td>"
            f"<td>{cell(record.get('recorded_action'))}</td>"
            f"<td>{cell(record.get('policy_action'))}</td>"
            f"<td>{cell(record.get('recorded_rank'))}</td>"
            f"<td>{float(record.get('normalized_value_gap_to_recorded') or 0.0):.4f}</td>"
            f"<td>{cell(record.get('source_replay'))}</td>"
            "</tr>"
        )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Threes Replay Policy Agreement</title>
  <style>
    :root {{ color-scheme: dark; --bg: #101318; --panel: #171d24; --line: #34404d; --text: #edf2f7; --muted: #aab6c2; --gold: #f2c14e; }}
    body {{ margin: 0; background: var(--bg); color: var(--text); font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ width: min(1240px, calc(100vw - 32px)); margin: 0 auto; padding: 24px 0 40px; }}
    h1 {{ margin: 0 0 8px; font-size: 24px; }}
    h2 {{ margin: 18px 0 8px; font-size: 17px; }}
    .muted {{ color: var(--muted); }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 18px 0; }}
    .card {{ border: 1px solid var(--line); background: var(--panel); border-radius: 8px; padding: 12px; }}
    .label {{ color: var(--muted); font-size: 11px; text-transform: uppercase; }}
    .value {{ margin-top: 4px; color: var(--gold); font-size: 22px; font-weight: 800; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 7px 8px; text-align: right; vertical-align: top; }}
    th:nth-child(3), td:nth-child(3), th:nth-child(10), td:nth-child(10) {{ text-align: left; }}
    td:nth-child(10) {{ max-width: 330px; overflow-wrap: anywhere; color: var(--muted); }}
    pre {{ white-space: pre-wrap; overflow-wrap: anywhere; color: var(--muted); }}
  </style>
</head>
<body>
  <main>
    <h1>Replay Policy Agreement</h1>
    <p class="muted">Recorded replay actions compared with the supplied policy's root action ranking.</p>
    <section class="cards">
      <div class="card"><div class="label">Records</div><div class="value">{cell(summary.get('records', 0))}</div></div>
      <div class="card"><div class="label">Action Match</div><div class="value">{float(summary.get('action_match_rate') or 0.0):.1%}</div></div>
      <div class="card"><div class="label">Recorded In Top 2</div><div class="value">{float(summary.get('recorded_in_top_two_rate') or 0.0):.1%}</div></div>
      <div class="card"><div class="label">High-Conf Misses</div><div class="value">{cell(summary.get('high_confidence_misses', 0))}</div></div>
    </section>
    <h2>Largest Misses</h2>
    <table><thead><tr><th>Seed</th><th>Move</th><th>Stratum</th><th>Score - Starter</th><th>Max Excl</th><th>Recorded</th><th>Policy</th><th>Rank</th><th>Norm Gap</th><th>Replay</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
    <h2>Summary JSON</h2>
    <pre>{escape(json.dumps(summary, indent=2, sort_keys=True))}</pre>
  </main>
</body>
</html>
"""
    path.write_text(html)


def run(
    *,
    policy_spec: str,
    out_dir: Path,
    replay_paths: Iterable[Path] | None = None,
    state_records: Iterable[dict[str, Any]] | None = None,
    min_tile: int = 1536,
    phase_filter: set[str] | None = None,
    corner_risk_filter: set[str] | None = None,
    default_starter_tile: int | None = 1536,
    max_records: int = 0,
    high_confidence_margin: float = 0.01,
) -> dict[str, Any]:
    policy = make_policy(policy_spec)
    if state_records is not None:
        payload = scan_state_records(
            policy=policy,
            policy_spec=policy_spec,
            state_records=state_records,
            min_tile=min_tile,
            phase_filter=phase_filter,
            corner_risk_filter=corner_risk_filter,
            default_starter_tile=default_starter_tile,
            max_records=max_records,
            high_confidence_margin=high_confidence_margin,
        )
    else:
        payload = scan_replays(
            policy=policy,
            policy_spec=policy_spec,
            replay_paths=list(replay_paths or []),
            min_tile=min_tile,
            phase_filter=phase_filter,
            corner_risk_filter=corner_risk_filter,
            default_starter_tile=default_starter_tile,
            max_records=max_records,
            high_confidence_margin=high_confidence_margin,
        )
    if hasattr(policy, "summary_stats"):
        payload["policy_stats"] = policy.summary_stats()
    out_dir.mkdir(parents=True, exist_ok=True)
    payload["json"] = str(out_dir / "agreement.json")
    payload["html"] = str(out_dir / "agreement.html")
    write_json(out_dir / "agreement.json", payload)
    write_html(out_dir / "agreement.html", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--replay-json", type=Path, nargs="+", action="append", default=[])
    parser.add_argument("--replay-glob", action="append")
    parser.add_argument("--state-json", type=Path, nargs="+", action="append", default=[])
    parser.add_argument("--state-glob", action="append")
    parser.add_argument("--starter", default="1536")
    parser.add_argument("--min-tile", type=int, default=1536)
    parser.add_argument("--phase-filter", action="append")
    parser.add_argument("--corner-risk-filter", action="append")
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--high-confidence-margin", type=float, default=0.01)
    parser.add_argument("--out-dir", type=Path, default=Path("threes_rl/runs/forensics/replay_policy_agreement/latest"))
    args = parser.parse_args()
    starter_text = str(args.starter).strip().lower()
    starter_tile = None if starter_text == "none" else int(starter_text)
    phase_filter = parse_phase_filters(args.phase_filter)
    corner_filter = parse_corner_risk_filters(args.corner_risk_filter)
    replay_paths = _flatten_paths(args.replay_json) + _glob_paths(args.replay_glob)
    state_paths = _flatten_paths(args.state_json) + _glob_paths(args.state_glob)
    if bool(replay_paths) == bool(state_paths):
        raise ValueError("Provide exactly one of --replay-json/--replay-glob or --state-json/--state-glob")
    payload = run(
        policy_spec=args.policy,
        replay_paths=replay_paths,
        state_records=_load_state_records_from_paths(state_paths) if state_paths else None,
        out_dir=args.out_dir,
        min_tile=args.min_tile,
        phase_filter=phase_filter,
        corner_risk_filter=corner_filter,
        default_starter_tile=starter_tile,
        max_records=args.max_records,
        high_confidence_margin=args.high_confidence_margin,
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"json={payload['json']}")
    print(f"html={payload['html']}")


if __name__ == "__main__":
    main()
