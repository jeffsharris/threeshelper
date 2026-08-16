"""Build a root-capped pool from retained pre-milestone failure replays."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from threes_rl.compare_replays import state_from_payload
from threes_rl.replay_provenance import ORIGIN_FRESH, policy_family, replay_provenance
from threes_rl.run_artifacts import safe_name, write_json
from threes_rl.sim import ThreesSim
from threes_rl.support_ladder_window_reservoir import raw_ladder_features
from threes_rl.swing_label import state_features


def _int_or_none(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict) or not isinstance(payload.get("replays"), list):
        raise ValueError(f"{path} does not contain replays[]")
    return payload


def _move_action(frame: object) -> str | None:
    if not isinstance(frame, dict):
        return None
    move = frame.get("move")
    if not isinstance(move, dict):
        return None
    action = move.get("action")
    return str(action) if action is not None else None


def _starter_from_replay(replay: dict[str, Any], manifest_entry: dict[str, Any], default: int | None) -> int | None:
    value = replay.get("starter_tile", manifest_entry.get("starter_tile", default))
    return None if value is None else int(value)


def _record_id(label: str, seed: int | None, frame_index: int, raw: dict[str, Any]) -> str:
    return safe_name(
        f"{label}_seed_{seed}_frame_{int(frame_index)}_raw768_{int(raw.get('raw_count_768', 0))}_"
        f"adj_{int(bool(raw.get('raw_has_adjacent_768')))}",
        max_length=128,
    )


def _selection_score(record: dict[str, Any]) -> tuple[int, int, int, int, int, int, int]:
    return (
        int(record.get("raw_count_768", 0) or 0),
        int(bool(record.get("raw_has_adjacent_768"))),
        int(record.get("empty_count", 0) or 0),
        int(record.get("legal_count", 0) or 0),
        int(record.get("source_score", 0) or 0),
        int(record.get("source_move_count", 0) or 0),
        int(record.get("source_frame_index", 0) or 0),
    )


def _candidate_record(
    *,
    label: str,
    replay_path: Path,
    replay: dict[str, Any],
    manifest_entry: dict[str, Any],
    frame: dict[str, Any],
    frame_position: int,
    next_action: str,
    default_starter_tile: int | None,
    threshold: int,
    root_origin: str,
) -> dict[str, Any] | None:
    state_payload = frame.get("state")
    if not isinstance(state_payload, dict):
        return None
    state = state_from_payload(state_payload)
    if state.game_over:
        return None
    starter_tile = _starter_from_replay(replay, manifest_entry, default_starter_tile)
    sim = ThreesSim(np.random.default_rng(_int_or_none(replay.get("seed")) or 0), starter_tile=starter_tile)
    features = dict(state_features(state, sim, starter_tile))
    raw = dict(raw_ladder_features(state.board, starter_tile))
    if int(raw.get("masked_count_1536", 0)) > 0:
        return None
    if int(features.get("max_tile_excl_starter", 0)) >= int(threshold):
        return None

    legal_count = len(sim.legal_actions(state))
    frame_index = int(frame.get("index", frame_position))
    seed = _int_or_none(replay.get("seed", manifest_entry.get("seed")))
    policy = str(replay.get("policy", ""))
    provenance = replay_provenance(replay, replay_path)
    root_score = None
    frames = replay.get("frames")
    if isinstance(frames, list) and frames:
        first_state = frames[0].get("state") if isinstance(frames[0], dict) else None
        if isinstance(first_state, dict):
            root_score = _int_or_none(first_state.get("score"))
    source_score = _int_or_none(state_payload.get("score")) or 0
    source_move_count = _int_or_none(state_payload.get("move_count"))
    if source_move_count is None:
        source_move_count = frame_index
    final_score = _int_or_none(replay.get("final_score", manifest_entry.get("score")))
    final_score_minus_starter = _int_or_none(manifest_entry.get("score_minus_starter"))
    root_replay = str(replay_path)
    root_frame_index = _int_or_none(provenance.get("root_frame_index"))
    if root_frame_index is None:
        root_frame_index = 0
    root_policy = policy or provenance.get("root_policy")
    ancestry_key = f"root:{root_origin}:{root_replay}:{seed}:{root_frame_index}"
    merged_features = {**features, **raw, "legal_count": int(legal_count)}
    return {
        "id": _record_id(label, seed, frame_index, raw),
        "kind": "pre_first_1536_nearfail_768_rootcap",
        "starter_tile": starter_tile,
        "replay_origin": root_origin,
        "source_origin": root_origin,
        "source_replay": str(replay_path),
        "source_seed": seed,
        "source_frame_index": frame_index,
        "source_move_count": int(source_move_count),
        "source_score": int(source_score),
        "source_next_action": next_action,
        "source_policy": policy or None,
        "source_policy_family": policy_family(policy),
        "root_origin": root_origin,
        "root_replay": root_replay,
        "root_seed": seed,
        "root_frame_index": int(root_frame_index),
        "root_move_count": 0,
        "root_score": root_score,
        "root_policy": root_policy,
        "root_policy_family": policy_family(root_policy),
        "ancestry_key": ancestry_key,
        "final_score": final_score,
        "final_score_minus_starter": final_score_minus_starter,
        "final_moves": _int_or_none(replay.get("final_moves", manifest_entry.get("moves"))),
        "max_tile_excl_starter": int(features["max_tile_excl_starter"]),
        "empty_count": int(features["empty_count"]),
        "legal_count": int(legal_count),
        "raw_count_768": int(raw["raw_count_768"]),
        "raw_count_1536": int(raw["raw_count_1536"]),
        "raw_count_3072": int(raw["raw_count_3072"]),
        "raw_has_adjacent_768": bool(raw["raw_has_adjacent_768"]),
        "raw_highest_duplicate_tile": int(raw["raw_highest_duplicate_tile"]),
        "raw_highest_adjacent_pair_tile": int(raw["raw_highest_adjacent_pair_tile"]),
        "masked_count_1536": int(raw["masked_count_1536"]),
        "masked_count_3072": int(raw["masked_count_3072"]),
        "masked_highest_duplicate_tile": int(raw["masked_highest_duplicate_tile"]),
        "masked_highest_adjacent_pair_tile": int(raw["masked_highest_adjacent_pair_tile"]),
        "features": merged_features,
        "state": state_payload,
    }


def _records_for_replay(
    *,
    label: str,
    manifest_entry: dict[str, Any],
    default_starter_tile: int | None,
    threshold: int,
    min_tile: int,
    root_origin: str,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    rejected: Counter[str] = Counter()
    replay_path = Path(str(manifest_entry.get("json", "")))
    if not replay_path.exists():
        rejected["missing_replay"] += 1
        return [], rejected
    try:
        replay = json.loads(replay_path.read_text())
    except (OSError, json.JSONDecodeError):
        rejected["bad_replay"] += 1
        return [], rejected
    frames = replay.get("frames")
    if not isinstance(frames, list):
        rejected["bad_replay"] += 1
        return [], rejected
    records: list[dict[str, Any]] = []
    for frame_position, frame in enumerate(frames):
        if not isinstance(frame, dict):
            continue
        if frame_position + 1 >= len(frames):
            rejected["no_next_action"] += 1
            continue
        next_action = _move_action(frames[frame_position + 1])
        if next_action is None:
            rejected["no_next_action"] += 1
            continue
        try:
            record = _candidate_record(
                label=label,
                replay_path=replay_path,
                replay=replay,
                manifest_entry=manifest_entry,
                frame=frame,
                frame_position=frame_position,
                next_action=next_action,
                default_starter_tile=default_starter_tile,
                threshold=threshold,
                root_origin=root_origin,
            )
        except (TypeError, ValueError):
            rejected["bad_frame"] += 1
            continue
        if record is None:
            rejected["not_candidate"] += 1
            continue
        if int(record["max_tile_excl_starter"]) < int(min_tile):
            rejected["below_min_tile"] += 1
            continue
        records.append(record)
    return records, rejected


def select_root_capped_records(
    manifest: dict[str, Any],
    *,
    label: str,
    default_starter_tile: int | None = 1536,
    threshold: int = 1536,
    min_tile: int | None = None,
    root_origin: str = ORIGIN_FRESH,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected_by_root: dict[str, dict[str, Any]] = {}
    rejected: Counter[str] = Counter()
    replay_records = manifest.get("replays", [])
    if min_tile is None:
        min_tile = int(manifest.get("min_tile", 0) or 0)
    for entry in replay_records:
        if not isinstance(entry, dict):
            rejected["bad_manifest_entry"] += 1
            continue
        records, local_rejected = _records_for_replay(
            label=label,
            manifest_entry=entry,
            default_starter_tile=default_starter_tile,
            threshold=threshold,
            min_tile=int(min_tile),
            root_origin=root_origin,
        )
        rejected.update(local_rejected)
        for record in records:
            key = str(record.get("root_seed") or record.get("ancestry_key") or record["id"])
            current = selected_by_root.get(key)
            if current is None or _selection_score(record) > _selection_score(current):
                selected_by_root[key] = record
    records = sorted(selected_by_root.values(), key=lambda row: (int(row.get("root_seed") or 0), str(row.get("id"))))
    raw_count_dist = Counter(str(record.get("raw_count_768", "unknown")) for record in records)
    raw_counts = [int(record.get("raw_count_768", 0) or 0) for record in records]
    summary = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_manifest": str(manifest.get("_manifest_path", "")),
        "source_run_dir": str(Path(str(manifest.get("_manifest_path", ""))).parents[2])
        if manifest.get("_manifest_path")
        else None,
        "qualified_games": int(manifest.get("qualified_games", 0) or 0),
        "retained_games": len([entry for entry in replay_records if isinstance(entry, dict)]),
        "records": len(records),
        "unique_root_seeds": len({record.get("root_seed") for record in records}),
        "root_origins": dict(Counter(str(record.get("root_origin", "unknown")) for record in records)),
        "min_tile": int(min_tile or 0),
        "min_start_raw_768": min(raw_counts) if raw_counts else 0,
        "threshold": int(threshold),
        "raw_count_768_dist": dict(sorted(raw_count_dist.items())),
        "adjacent_768_records": sum(1 for record in records if bool(record.get("raw_has_adjacent_768"))),
        "selection_rule": "highest raw_count_768, then adjacent 768, empty count, legal count, score, move_count, frame_index; one state per retained fresh seed",
        "rejects": dict(rejected),
    }
    return records, summary


def write_selected_states_csv(path: Path, records: Iterable[dict[str, Any]]) -> None:
    rows = list(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "seed",
                "frame_index",
                "move_count",
                "score",
                "final_score",
                "raw_count_768",
                "raw_has_adjacent_768",
                "raw_768_adjacent_pairs",
                "masked_count_1536",
                "empty_count",
                "legal_count",
                "source_next_action",
                "root_origin",
                "ancestry_key",
            ],
        )
        writer.writeheader()
        for record in rows:
            features = record.get("features") if isinstance(record.get("features"), dict) else {}
            writer.writerow(
                {
                    "seed": record.get("root_seed"),
                    "frame_index": record.get("source_frame_index"),
                    "move_count": record.get("source_move_count"),
                    "score": record.get("source_score"),
                    "final_score": record.get("final_score"),
                    "raw_count_768": record.get("raw_count_768"),
                    "raw_has_adjacent_768": record.get("raw_has_adjacent_768"),
                    "raw_768_adjacent_pairs": features.get("raw_768_adjacent_pairs", 0),
                    "masked_count_1536": record.get("masked_count_1536"),
                    "empty_count": record.get("empty_count"),
                    "legal_count": record.get("legal_count"),
                    "source_next_action": record.get("source_next_action"),
                    "root_origin": record.get("root_origin"),
                    "ancestry_key": record.get("ancestry_key"),
                }
            )


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    manifest = _load_manifest(args.manifest)
    manifest["_manifest_path"] = str(args.manifest)
    starter_text = str(args.starter).strip().lower()
    default_starter = None if starter_text == "none" else int(starter_text)
    records, summary = select_root_capped_records(
        manifest,
        label=args.label,
        default_starter_tile=default_starter,
        threshold=args.threshold,
        min_tile=args.min_tile if args.min_tile > 0 else None,
        root_origin=args.root_origin,
    )
    payload = {"version": 1, "kind": "pre_milestone_nearfail_root_pool", "records": records}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload["records_json"] = str(args.out_dir / "records.json")
    payload["summary_json"] = str(args.out_dir / "summary.json")
    payload["selection_csv"] = str(args.out_dir / "selected_states.csv")
    summary["records_json"] = payload["records_json"]
    summary["selection_csv"] = payload["selection_csv"]
    write_json(args.out_dir / "records.json", payload)
    write_json(args.out_dir / "summary.json", summary)
    write_selected_states_csv(args.out_dir / "selected_states.csv", records)
    return {**payload, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--threshold", type=int, default=1536)
    parser.add_argument("--min-tile", type=int, default=0)
    parser.add_argument("--starter", default="1536")
    parser.add_argument("--root-origin", default=ORIGIN_FRESH)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = run_from_args(args)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"records={payload['records_json']}")
    print(f"csv={payload['selection_csv']}")


if __name__ == "__main__":
    main()
