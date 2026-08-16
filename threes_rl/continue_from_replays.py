"""Evaluate policies from high-board replay states."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

import numpy as np

from threes_rl.eval import GameResult, make_policy, max_tile_excluding_initial_starter, starter_baseline_score, summarize
from threes_rl.ntuple import PHASE4_NAMES, phase4_index_for_board
from threes_rl.record_replay import preview_payload, state_payload, write_html
from threes_rl.replay_provenance import (
    ORIGIN_CONTINUATION,
    ORIGIN_UNKNOWN,
    provenance_fields_from_record,
    replay_provenance,
)
from threes_rl.run_artifacts import write_json
from threes_rl.sim import DIRECTION_NAMES, SimState, ThreesSim, score_board
from threes_rl.train_td import copy_state, parse_phase_filter, state_from_replay_payload


@dataclass
class StartCase:
    id: str
    state: SimState
    starter_tile: int | None
    source_replay: str
    source_seed: int | None
    frame_index: int
    start_score: int
    start_max_tile_excl_starter: int
    phase: str
    source_origin: str = ORIGIN_UNKNOWN
    source_policy: str | None = None
    root_origin: str = ORIGIN_UNKNOWN
    root_replay: str | None = None
    root_seed: int | None = None
    root_frame_index: int | None = None
    root_move_count: int | None = None
    root_score: int | None = None
    root_policy: str | None = None
    root_policy_family: str | None = None
    ancestry_key: str | None = None


@dataclass
class ContinuationRecord:
    result: GameResult
    replay: dict[str, Any]
    start_case: StartCase
    repeat_index: int
    score_delta: int
    moves_delta: int


def _flatten_paths(path_groups: list[list[Path]] | None) -> list[Path]:
    if not path_groups:
        return []
    return [path for group in path_groups for path in group]


def _starter_from_replay(replay: dict[str, Any], default: int | None) -> int | None:
    value = replay.get("starter_tile", default)
    if value is None:
        return None
    return int(value)


def _int_or_none(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _starter_from_record(record: dict[str, Any], default: int | None) -> int | None:
    value = record.get("starter_tile", default)
    if value is None:
        return None
    return int(value)


def _record_source_replay(record: dict[str, Any], fallback: Path) -> str:
    source = record.get("source_replay")
    return str(source) if source is not None else str(fallback)


def _record_frame_index(record: dict[str, Any], fallback: int) -> int:
    value = record.get("source_frame_index", record.get("frame_index", fallback))
    return int(value)


def _start_case_from_payload(
    *,
    replay_path: Path,
    state_payload_dict: dict[str, Any],
    starter_tile: int | None,
    source_replay: str,
    source_seed: int | None,
    frame_index: int,
    provenance: dict[str, Any] | None,
    case_id: str | None,
    min_tile: int,
    phase_filter: set[str] | None,
) -> StartCase | None:
    state = state_from_replay_payload(state_payload_dict)
    if state.game_over:
        return None
    start_max = max_tile_excluding_initial_starter(state.board, starter_tile)
    if start_max < int(min_tile):
        return None
    phase = PHASE4_NAMES[phase4_index_for_board(state.board, starter_tile=starter_tile)]
    if phase_filter is not None and phase not in phase_filter:
        return None
    seed_label = "none" if source_seed is None else str(int(source_seed))
    return StartCase(
        id=case_id or f"{Path(replay_path).stem}_seed{seed_label}_frame{frame_index}",
        state=copy_state(state),
        starter_tile=starter_tile,
        source_replay=source_replay,
        source_seed=source_seed,
        frame_index=int(frame_index),
        start_score=int(score_board(state.board)),
        start_max_tile_excl_starter=int(start_max),
        phase=phase,
        source_origin=str((provenance or {}).get("source_origin", ORIGIN_UNKNOWN)),
        source_policy=(provenance or {}).get("source_policy"),
        root_origin=str((provenance or {}).get("root_origin", ORIGIN_UNKNOWN)),
        root_replay=(provenance or {}).get("root_replay"),
        root_seed=_int_or_none((provenance or {}).get("root_seed")),
        root_frame_index=_int_or_none((provenance or {}).get("root_frame_index")),
        root_move_count=_int_or_none((provenance or {}).get("root_move_count")),
        root_score=_int_or_none((provenance or {}).get("root_score")),
        root_policy=(provenance or {}).get("root_policy"),
        root_policy_family=(provenance or {}).get("root_policy_family"),
        ancestry_key=(provenance or {}).get("ancestry_key"),
    )


def collect_start_cases(
    replay_paths: Iterable[Path],
    *,
    min_tile: int,
    phase_filter: set[str] | None = None,
    default_starter_tile: int | None = 1536,
) -> list[StartCase]:
    cases: list[StartCase] = []
    for replay_path in replay_paths:
        replay = json.loads(Path(replay_path).read_text())
        starter_tile = _starter_from_replay(replay, default_starter_tile)
        seed_value = _int_or_none(replay.get("seed"))
        records = replay.get("records")
        if isinstance(records, list):
            for record_idx, record in enumerate(records):
                if not isinstance(record, dict) or not isinstance(record.get("state"), dict):
                    continue
                record_starter = _starter_from_record(record, starter_tile)
                record_provenance = provenance_fields_from_record(record, Path(replay_path))
                case = _start_case_from_payload(
                    replay_path=Path(replay_path),
                    state_payload_dict=record["state"],
                    starter_tile=record_starter,
                    source_replay=_record_source_replay(record, Path(replay_path)),
                    source_seed=_int_or_none(record.get("source_seed", record.get("seed", seed_value))),
                    frame_index=_record_frame_index(record, record_idx),
                    provenance=record_provenance,
                    case_id=str(record["id"]) if record.get("id") is not None else None,
                    min_tile=min_tile,
                    phase_filter=phase_filter,
                )
                if case is not None:
                    cases.append(case)
            continue
        frames = replay.get("frames", [])
        if not isinstance(frames, list):
            continue
        source_provenance = replay_provenance(replay, replay_path)
        frame_provenance = {
            "source_origin": source_provenance["replay_origin"],
            "source_policy": replay.get("policy"),
            "root_origin": source_provenance["root_origin"],
            "root_replay": source_provenance["root_replay"],
            "root_seed": source_provenance["root_seed"],
            "root_frame_index": source_provenance["root_frame_index"],
            "root_move_count": source_provenance["root_move_count"],
            "root_score": source_provenance["root_score"],
            "root_policy": source_provenance["root_policy"],
            "root_policy_family": source_provenance["root_policy_family"],
            "ancestry_key": source_provenance["ancestry_key"],
        }
        for frame_pos, frame in enumerate(frames):
            if not isinstance(frame, dict) or not isinstance(frame.get("state"), dict):
                continue
            frame_index = int(frame.get("index", frame_pos))
            case = _start_case_from_payload(
                replay_path=Path(replay_path),
                state_payload_dict=frame["state"],
                starter_tile=starter_tile,
                source_replay=str(replay_path),
                source_seed=seed_value,
                frame_index=frame_index,
                provenance=frame_provenance,
                case_id=None,
                min_tile=min_tile,
                phase_filter=phase_filter,
            )
            if case is not None:
                cases.append(case)
    return cases


def load_selection_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text())
    records = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError("--records-json must point to a records list or an object with records[]")
    return [record for record in records if isinstance(record, dict)]


def select_start_cases(cases: list[StartCase], *, max_starts: int, seed: int) -> list[StartCase]:
    if max_starts <= 0 or len(cases) <= max_starts:
        return list(cases)
    rng = np.random.default_rng(int(seed))
    indices = sorted(int(idx) for idx in rng.choice(len(cases), size=int(max_starts), replace=False))
    return [cases[idx] for idx in indices]


def select_start_cases_from_records(
    cases: list[StartCase],
    records: list[dict[str, Any]],
    *,
    records_filter: str = "all",
    max_starts: int = 0,
) -> tuple[list[StartCase], dict[str, int]]:
    if records_filter not in ("all", "changed", "top_two_changed"):
        raise ValueError(f"Unsupported records_filter: {records_filter}")
    case_by_key = {
        (str(case.source_replay), int(case.frame_index)): case
        for case in cases
    }
    selected: list[StartCase] = []
    seen_keys: set[tuple[str, int]] = set()
    requested = 0
    matched = 0
    skipped_filter = 0
    skipped_missing = 0
    skipped_duplicate = 0
    for record in records:
        if not isinstance(record, dict):
            continue
        if records_filter == "changed" and not bool(record.get("changed")):
            skipped_filter += 1
            continue
        if records_filter == "top_two_changed" and not bool(record.get("top_two_changed")):
            skipped_filter += 1
            continue
        requested += 1
        frame_index = record.get("frame_index", record.get("source_frame_index", -1))
        key = (str(record.get("source_replay")), int(frame_index))
        if key in seen_keys:
            skipped_duplicate += 1
            continue
        case = case_by_key.get(key)
        if case is None:
            skipped_missing += 1
            continue
        selected.append(case)
        seen_keys.add(key)
        matched += 1
        if max_starts > 0 and len(selected) >= int(max_starts):
            break
    return selected, {
        "records_requested": int(requested),
        "records_matched": int(matched),
        "records_skipped_filter": int(skipped_filter),
        "records_skipped_missing": int(skipped_missing),
        "records_skipped_duplicate": int(skipped_duplicate),
    }


def run_continuation(
    *,
    policy: object,
    policy_name: str,
    start_case: StartCase,
    repeat_index: int,
    seed: int,
    max_moves: int,
) -> ContinuationRecord:
    sim_seed = int(seed) + int(repeat_index) * 1_000_003 + int(start_case.frame_index) * 9_973
    sim = ThreesSim(np.random.default_rng(sim_seed), starter_tile=start_case.starter_tile)
    policy_rng = np.random.default_rng(sim_seed + 37)
    state = copy_state(start_case.state)
    start_score = int(score_board(state.board))
    start_move_count = int(state.move_count)
    frames: list[dict[str, Any]] = [{"index": 0, "state": state_payload(state, sim), "move": None}]

    while not state.game_over and state.move_count - start_move_count < int(max_moves):
        before = state
        action = int(policy(before, sim, policy_rng))
        state, info = sim.step(before, action)
        if not info.moved:
            legal = sim.legal_actions(before)
            if not legal:
                break
            action = int(legal[0])
            state, info = sim.step(before, action)
            if not info.moved:
                break
        frames.append(
            {
                "index": len(frames),
                "state": state_payload(state, sim),
                "move": {
                    "action": DIRECTION_NAMES[action],
                    "preview_used": preview_payload(before),
                    "inserted_value": info.inserted_value,
                    "inserted_pos": list(info.inserted_pos) if info.inserted_pos is not None else None,
                    "eligible_positions": [list(pos) for pos in info.eligible_positions],
                    "merge_score_delta": int(info.merge_score_delta),
                    "score_delta": int(info.score_delta),
                    "terminal_merge": bool(info.terminal_merge),
                    "score_before": int(score_board(before.board)),
                    "score_after": int(score_board(state.board)),
                    "max_tile_before": int(before.max_tile),
                    "max_tile_after": int(state.max_tile),
                },
            }
        )

    final_score = int(score_board(state.board))
    result = GameResult(
        seed=sim_seed,
        score=final_score,
        score_minus_starter=final_score - starter_baseline_score(start_case.starter_tile),
        moves=int(state.move_count),
        max_tile=int(state.max_tile),
        max_tile_excl_starter=max_tile_excluding_initial_starter(state.board, start_case.starter_tile),
        terminal_tile=bool(np.any(state.board == 12288)),
        starter_tile=start_case.starter_tile,
    )
    replay = {
        "policy": policy_name,
        "seed": int(sim_seed),
        "starter_tile": start_case.starter_tile,
        "replay_origin": ORIGIN_CONTINUATION,
        "start_case_id": start_case.id,
        "source_origin": start_case.source_origin,
        "source_replay": start_case.source_replay,
        "source_seed": start_case.source_seed,
        "source_frame_index": int(start_case.frame_index),
        "source_policy": start_case.source_policy,
        "root_origin": start_case.root_origin,
        "root_replay": start_case.root_replay,
        "root_seed": start_case.root_seed,
        "root_frame_index": start_case.root_frame_index,
        "root_move_count": start_case.root_move_count,
        "root_score": start_case.root_score,
        "root_policy": start_case.root_policy,
        "root_policy_family": start_case.root_policy_family,
        "ancestry_key": start_case.ancestry_key,
        "start_score": int(start_score),
        "start_max_tile_excl_starter": int(start_case.start_max_tile_excl_starter),
        "start_phase": start_case.phase,
        "max_moves": int(max_moves),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "final_score": int(final_score),
        "final_score_delta": int(final_score - start_score),
        "final_moves": int(state.move_count),
        "final_moves_delta": int(state.move_count - start_move_count),
        "final_max_tile": int(state.max_tile),
        "final_max_tile_excl_starter": int(result.max_tile_excl_starter),
        "game_over": bool(state.game_over),
        "frames": frames,
    }
    return ContinuationRecord(
        result=result,
        replay=replay,
        start_case=start_case,
        repeat_index=int(repeat_index),
        score_delta=int(final_score - start_score),
        moves_delta=int(state.move_count - start_move_count),
    )


def continuation_progress_key(
    *,
    policy_name: str,
    start_case: StartCase,
    repeat_index: int,
    seed: int,
    max_moves: int,
) -> str:
    raw = json.dumps(
        {
            "version": 1,
            "policy": str(policy_name),
            "seed": int(seed),
            "repeat_index": int(repeat_index),
            "max_moves": int(max_moves),
            "start": {
                "id": start_case.id,
                "source_replay": start_case.source_replay,
                "source_seed": start_case.source_seed,
                "source_frame_index": int(start_case.frame_index),
                "starter_tile": start_case.starter_tile,
                "start_score": int(start_case.start_score),
                "start_max_tile_excl_starter": int(start_case.start_max_tile_excl_starter),
                "move_count": int(start_case.state.move_count),
                "board": [int(value) for value in np.asarray(start_case.state.board, dtype=np.int32).reshape(-1)],
            },
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _load_continuation_progress(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "version": 1,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "updated_at": None,
            "entries": {},
        }
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a continuation progress object")
    entries = payload.setdefault("entries", {})
    if not isinstance(entries, dict):
        raise ValueError(f"{path} has invalid continuation entries")
    payload.setdefault("version", 1)
    return payload


def _write_continuation_progress(path: Path, progress: dict[str, Any]) -> None:
    progress["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, progress)


def _progress_entry_for_record(
    *,
    key: str,
    policy_name: str,
    seed: int,
    max_moves: int,
    record: ContinuationRecord,
) -> dict[str, Any]:
    return {
        "key": key,
        "policy": str(policy_name),
        "seed": int(seed),
        "repeat_index": int(record.repeat_index),
        "max_moves": int(max_moves),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "record": continuation_record_payload(record),
        "replay": record.replay,
    }


def _record_from_progress_entry(entry: dict[str, Any], start_case: StartCase) -> ContinuationRecord:
    record_payload = entry.get("record")
    replay = entry.get("replay")
    if not isinstance(record_payload, dict) or not isinstance(replay, dict):
        raise ValueError("invalid continuation progress entry")
    result = GameResult(
        seed=int(record_payload["seed"]),
        score=int(record_payload["score"]),
        score_minus_starter=int(record_payload["score_minus_starter"]),
        moves=int(record_payload["moves"]),
        max_tile=int(record_payload["max_tile"]),
        max_tile_excl_starter=int(record_payload["max_tile_excl_starter"]),
        terminal_tile=bool(record_payload["terminal_tile"]),
        starter_tile=record_payload.get("starter_tile", start_case.starter_tile),
    )
    return ContinuationRecord(
        result=result,
        replay=dict(replay),
        start_case=start_case,
        repeat_index=int(record_payload["repeat_index"]),
        score_delta=int(record_payload["score_delta"]),
        moves_delta=int(record_payload["moves_delta"]),
    )


def _write_manifest_records(top_dir: Path, records: list[ContinuationRecord]) -> list[dict[str, object]]:
    if top_dir.exists():
        shutil.rmtree(top_dir)
    top_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    for rank, record in enumerate(records, start=1):
        result = record.result
        game_dir = top_dir / f"rank_{rank:02d}_score_{result.score}_delta_{record.score_delta}_seed_{result.seed}"
        game_dir.mkdir(parents=True, exist_ok=True)
        json_path = game_dir / "replay.json"
        html_path = game_dir / "replay.html"
        write_json(json_path, record.replay)
        write_html(html_path, record.replay)
        manifest.append(
            {
                "rank": rank,
                "seed": int(result.seed),
                "starter_tile": result.starter_tile,
                "score": int(result.score),
                "score_delta": int(record.score_delta),
                "moves_delta": int(record.moves_delta),
                "max_tile": int(result.max_tile),
                "max_tile_excl_starter": int(result.max_tile_excl_starter),
                "start_score": int(record.start_case.start_score),
                "start_max_tile_excl_starter": int(record.start_case.start_max_tile_excl_starter),
                "start_case_id": record.start_case.id,
                "source_replay": record.start_case.source_replay,
                "source_frame_index": int(record.start_case.frame_index),
                "source_origin": record.start_case.source_origin,
                "root_origin": record.start_case.root_origin,
                "root_replay": record.start_case.root_replay,
                "root_frame_index": record.start_case.root_frame_index,
                "html": str(html_path),
                "json": str(json_path),
            }
        )
    write_json(top_dir / "manifest.json", manifest)
    return manifest


def write_top_continuations(run_dir: Path, records: list[ContinuationRecord], keep: int) -> dict[str, list[dict[str, object]]]:
    top_count = max(0, int(keep))
    by_score = sorted(records, key=lambda item: (item.result.score, item.score_delta, item.moves_delta), reverse=True)[:top_count]
    by_delta = sorted(records, key=lambda item: (item.score_delta, item.result.score, item.moves_delta), reverse=True)[:top_count]
    return {
        "top_games": _write_manifest_records(run_dir / "top_games", by_score),
        "top_delta_games": _write_manifest_records(run_dir / "top_delta_games", by_delta),
    }


def summarize_continuations(records: list[ContinuationRecord], *, start_cases_total: int) -> dict[str, object]:
    if not records:
        raise ValueError("No continuation records to summarize")
    base_summary = summarize([record.result for record in records], include_by_starter=False)
    deltas = sorted(record.score_delta for record in records)
    moves_delta = sorted(record.moves_delta for record in records)
    start_maxes = sorted(record.start_case.start_max_tile_excl_starter for record in records)
    base_summary.update(
        {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "continuations": len(records),
            "start_cases_total": int(start_cases_total),
            "high_score_delta": int(max(deltas)),
            "mean_score_delta": float(mean(deltas)),
            "median_score_delta": float(median(deltas)),
            "high_moves_delta": int(max(moves_delta)),
            "mean_moves_delta": float(mean(moves_delta)),
            "median_moves_delta": float(median(moves_delta)),
            "start_max_tile_excl_starter_dist": {
                f">={threshold}": sum(1 for value in start_maxes if value >= threshold) / len(start_maxes)
                for threshold in (768, 1536, 3072, 6144)
            },
            "p_advanced_to_6144": sum(record.result.max_tile_excl_starter >= 6144 for record in records) / len(records),
        }
    )
    return base_summary


def continuation_record_payload(record: ContinuationRecord) -> dict[str, object]:
    result = record.result
    start_case = record.start_case
    return {
        "seed": int(result.seed),
        "starter_tile": result.starter_tile,
        "repeat_index": int(record.repeat_index),
        "start_case_id": start_case.id,
        "source_origin": start_case.source_origin,
        "source_replay": start_case.source_replay,
        "source_seed": start_case.source_seed,
        "source_frame_index": int(start_case.frame_index),
        "source_policy": start_case.source_policy,
        "root_origin": start_case.root_origin,
        "root_replay": start_case.root_replay,
        "root_seed": start_case.root_seed,
        "root_frame_index": start_case.root_frame_index,
        "root_move_count": start_case.root_move_count,
        "root_score": start_case.root_score,
        "root_policy": start_case.root_policy,
        "root_policy_family": start_case.root_policy_family,
        "ancestry_key": start_case.ancestry_key,
        "start_phase": start_case.phase,
        "start_score": int(start_case.start_score),
        "start_max_tile_excl_starter": int(start_case.start_max_tile_excl_starter),
        "score": int(result.score),
        "score_minus_starter": int(result.score_minus_starter),
        "score_delta": int(record.score_delta),
        "moves": int(result.moves),
        "moves_delta": int(record.moves_delta),
        "max_tile": int(result.max_tile),
        "max_tile_excl_starter": int(result.max_tile_excl_starter),
        "terminal_tile": bool(result.terminal_tile),
    }


def run_from_args(args: argparse.Namespace) -> dict[str, object]:
    replay_paths = _flatten_paths(args.replay_json)
    phase_filter = set(parse_phase_filter(args.phase_filter)) if args.phase_filter else None
    starter_text = args.starter.strip().lower()
    default_starter = None if starter_text == "none" else int(starter_text)
    cases = collect_start_cases(
        replay_paths,
        min_tile=args.start_state_min_tile,
        phase_filter=phase_filter,
        default_starter_tile=default_starter,
    )
    records_json = getattr(args, "records_json", None)
    records_filter = str(getattr(args, "records_filter", "all"))
    record_selection_stats: dict[str, int] = {}
    if records_json is not None:
        records_payload = load_selection_records(Path(records_json))
        selected_cases, record_selection_stats = select_start_cases_from_records(
            cases,
            records_payload,
            records_filter=records_filter,
            max_starts=args.max_starts,
        )
    else:
        selected_cases = select_start_cases(cases, max_starts=args.max_starts, seed=args.seed)
    if not selected_cases:
        raise ValueError("No start cases matched the requested filters")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_continuations = bool(getattr(args, "checkpoint_continuations", False))
    progress_path = getattr(args, "continuation_progress_json", None)
    if progress_path is None and checkpoint_continuations:
        progress_path = args.out_dir / "continuation_progress.json"
    progress_payload: dict[str, Any] | None = None
    progress_entries: dict[str, Any] | None = None
    if progress_path is not None:
        progress_path = Path(progress_path)
        progress_payload = _load_continuation_progress(progress_path)
        progress_entries_obj = progress_payload.setdefault("entries", {})
        if not isinstance(progress_entries_obj, dict):
            raise ValueError(f"{progress_path} has invalid continuation entries")
        progress_entries = progress_entries_obj

    policy = make_policy(args.policy)
    records: list[ContinuationRecord] = []
    ran_records = 0
    resumed_records = 0
    for case_idx, start_case in enumerate(selected_cases):
        for repeat_idx in range(int(args.repeats_per_start)):
            run_seed = int(args.seed) + case_idx * 100_003
            progress_key = continuation_progress_key(
                policy_name=args.policy,
                start_case=start_case,
                repeat_index=repeat_idx,
                seed=run_seed,
                max_moves=args.max_moves,
            )
            if progress_entries is not None:
                entry = progress_entries.get(progress_key)
                if isinstance(entry, dict):
                    records.append(_record_from_progress_entry(entry, start_case))
                    resumed_records += 1
                    if args.progress_every > 0 and len(records) % int(args.progress_every) == 0:
                        print(
                            "progress "
                            f"{len(records)}/{len(selected_cases) * int(args.repeats_per_start)} "
                            f"best_delta={max(record.score_delta for record in records)} "
                            f"best_max_excl={max(record.result.max_tile_excl_starter for record in records)} "
                            "resumed=1",
                            flush=True,
                        )
                    continue

            record = run_continuation(
                policy=policy,
                policy_name=args.policy,
                start_case=start_case,
                repeat_index=repeat_idx,
                seed=run_seed,
                max_moves=args.max_moves,
            )
            records.append(record)
            ran_records += 1
            if progress_entries is not None and progress_payload is not None and progress_path is not None:
                progress_entries[progress_key] = _progress_entry_for_record(
                    key=progress_key,
                    policy_name=args.policy,
                    seed=run_seed,
                    max_moves=args.max_moves,
                    record=record,
                )
                _write_continuation_progress(progress_path, progress_payload)
            if args.progress_every > 0 and len(records) % int(args.progress_every) == 0:
                print(
                    "progress "
                    f"{len(records)}/{len(selected_cases) * int(args.repeats_per_start)} "
                    f"best_delta={max(record.score_delta for record in records)} "
                    f"best_max_excl={max(record.result.max_tile_excl_starter for record in records)}",
                    flush=True,
                )

    summary = summarize_continuations(records, start_cases_total=len(cases))
    summary["policy"] = args.policy
    summary["source_replays"] = [str(path) for path in replay_paths]
    summary["start_state_min_tile"] = int(args.start_state_min_tile)
    summary["phase_filter"] = sorted(phase_filter) if phase_filter is not None else None
    summary["max_starts"] = int(args.max_starts)
    summary["repeats_per_start"] = int(args.repeats_per_start)
    summary["continuations_ran"] = int(ran_records)
    summary["continuations_resumed"] = int(resumed_records)
    summary["checkpoint_continuations"] = progress_path is not None
    summary["by_source_origin"] = dict(Counter(str(record.start_case.source_origin) for record in records))
    summary["by_root_origin"] = dict(Counter(str(record.start_case.root_origin) for record in records))
    summary["unique_root_replays"] = len(
        {
            str(record.start_case.root_replay)
            for record in records
            if record.start_case.root_replay is not None
        }
    )
    summary["unique_ancestry_keys"] = len(
        {
            str(record.start_case.ancestry_key)
            for record in records
            if record.start_case.ancestry_key is not None
        }
    )
    if progress_path is not None:
        summary["continuation_progress_json"] = str(progress_path)
    if records_json is not None:
        summary["records_json"] = str(records_json)
        summary["records_filter"] = records_filter
        summary.update(record_selection_stats)
    if hasattr(policy, "summary_stats"):
        summary["policy_stats"] = policy.summary_stats()
    summary.update(write_top_continuations(args.out_dir, records, args.keep_top_games))
    write_json(args.out_dir / "summary.json", summary)
    write_json(args.out_dir / "records.json", [continuation_record_payload(record) for record in records])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True)
    parser.add_argument(
        "--replay-json",
        type=Path,
        nargs="+",
        action="append",
        required=True,
        help="Replay JSONs or high-board reservoir JSONs with records[].state.",
    )
    parser.add_argument("--starter", default="1536")
    parser.add_argument("--start-state-min-tile", type=int, default=3072)
    parser.add_argument("--phase-filter", help="Comma-separated phase names/aliases for start states.")
    parser.add_argument("--max-starts", type=int, default=40)
    parser.add_argument("--records-json", type=Path, help="Optional policy_divergence_scan records.json used to choose exact replay frames.")
    parser.add_argument(
        "--records-filter",
        choices=["all", "changed", "top_two_changed"],
        default="all",
        help="When --records-json is provided, choose all records, changed-action records, or changed-top-two records.",
    )
    parser.add_argument("--repeats-per-start", type=int, default=1)
    parser.add_argument("--max-moves", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--keep-top-games", type=int, default=3)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument(
        "--checkpoint-continuations",
        action="store_true",
        help="Checkpoint each completed continuation and reuse matching completed entries on rerun.",
    )
    parser.add_argument(
        "--continuation-progress-json",
        type=Path,
        help="Explicit resumable continuation progress JSON path; implies checkpoint/resume behavior.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("threes_rl/runs/continuations/latest"),
    )
    args = parser.parse_args()
    summary = run_from_args(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"summary={args.out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
