"""Build a provenance-safe, ancestry-grouped phase4 restart-state manifest."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from threes_rl.eval import max_tile_excluding_initial_starter
from threes_rl.phase0_oracle_corpus_audit import behavior_policy_family
from threes_rl.phase0_replay_coverage_inventory import DEFAULT_REPLAY_GLOBS, replay_action_signature
from threes_rl.replay_provenance import GENUINE_ROOT_ORIGINS, replay_provenance
from threes_rl.run_artifacts import write_json
from threes_rl.sim import ThreesSim
from threes_rl.train_td import state_from_replay_payload, start_state_phase_index


MANIFEST_VERSION = "phase4_ancestry_balanced_v1"
NEXT_STAGE_TILE = (384, 1536, 3072, 6144)


def glob_replays(patterns: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()
    for pattern in patterns:
        for text in glob.glob(pattern, recursive=True):
            path = Path(text)
            key = str(path.resolve(strict=False))
            if key in seen or not path.is_file():
                continue
            seen.add(key)
            paths.append(path)
    return sorted(paths, key=lambda path: str(path))


def canonical_ancestry_id(replay: dict[str, Any], path: Path) -> str:
    provenance = replay_provenance(replay, path)
    starter = replay.get("starter_tile", 1536)
    root_seed = provenance.get("root_seed")
    if root_seed is not None:
        return f"{provenance.get('root_origin')}:{int(root_seed)}:{starter}"
    signature = replay_action_signature(replay)
    return f"{provenance.get('root_origin')}:{signature}:{starter}"


def state_signature(state_payload: dict[str, Any], starter_tile: int | None) -> str:
    payload = {
        "starter_tile": starter_tile,
        "board": state_payload.get("board"),
        "preview": state_payload.get("preview"),
        "tile_cycle": state_payload.get("tile_cycle"),
        "move_count": state_payload.get("move_count"),
        "game_over": state_payload.get("game_over"),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def replay_final_built_max(replay: dict[str, Any], starter_tile: int | None) -> int:
    frames = replay.get("frames")
    if not isinstance(frames, list):
        return 0
    best = 0
    for frame in frames:
        state = frame.get("state") if isinstance(frame, dict) else None
        if not isinstance(state, dict):
            continue
        try:
            board = np.asarray(state.get("board"), dtype=np.int32)
        except (TypeError, ValueError):
            continue
        if board.shape == (4, 4):
            best = max(best, int(max_tile_excluding_initial_starter(board, starter_tile)))
    return best


def replay_behavior_family(replay: dict[str, Any], path: Path) -> str:
    provenance = replay_provenance(replay, path)
    return behavior_policy_family(
        {
            "root_policy": provenance.get("root_policy"),
            "root_policy_family": provenance.get("root_policy_family"),
            "source_policy": provenance.get("source_policy"),
            "source_policy_family": provenance.get("source_policy_family"),
            "source_replay": str(path),
        }
    )


def build_restart_manifest(paths: Iterable[Path]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    seen_states: set[tuple[str, int, str]] = set()
    replay_signatures: set[tuple[str, str]] = set()
    counts = Counter()
    stage_ancestries: dict[int, set[str]] = defaultdict(set)
    outcome_counts = Counter()
    family_counts = Counter()

    for path in paths:
        counts["source_files_scanned"] += 1
        try:
            replay = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            counts["invalid_replays"] += 1
            continue
        if not isinstance(replay, dict):
            counts["invalid_replays"] += 1
            continue
        provenance = replay_provenance(replay, path)
        if provenance.get("replay_origin") not in GENUINE_ROOT_ORIGINS or not provenance.get("replay_reset_invariant"):
            counts["non_normal_start_replays"] += 1
            continue
        family = replay_behavior_family(replay, path)
        action_signature = replay_action_signature(replay)
        replay_key = (family, action_signature)
        if replay_key in replay_signatures:
            counts["duplicate_replay_copies"] += 1
            continue
        replay_signatures.add(replay_key)
        counts["normal_start_replays"] += 1

        starter_value = replay.get("starter_tile", 1536)
        starter_tile = None if starter_value is None else int(starter_value)
        ancestry_id = canonical_ancestry_id(replay, path)
        final_built_max = replay_final_built_max(replay, starter_tile)
        validator = ThreesSim(np.random.default_rng(0), starter_tile=starter_tile)
        frames = replay.get("frames")
        if not isinstance(frames, list):
            continue
        for fallback_index, frame in enumerate(frames):
            if not isinstance(frame, dict):
                continue
            state_payload = frame.get("state")
            if not isinstance(state_payload, dict) or bool(state_payload.get("game_over")):
                continue
            try:
                state = state_from_replay_payload(state_payload)
            except (KeyError, TypeError, ValueError):
                counts["invalid_states"] += 1
                continue
            if not validator.legal_actions(state):
                counts["states_without_legal_actions"] += 1
                continue
            stage_idx = start_state_phase_index(state, starter_tile)
            signature = state_signature(state_payload, starter_tile)
            state_key = (ancestry_id, stage_idx, signature)
            if state_key in seen_states:
                counts["duplicate_states"] += 1
                continue
            seen_states.add(state_key)
            frame_index = int(frame.get("index", fallback_index))
            outcome = "success" if final_built_max >= NEXT_STAGE_TILE[stage_idx] else "failure"
            record_id = hashlib.sha1(
                f"{ancestry_id}:{stage_idx}:{signature}".encode("utf-8")
            ).hexdigest()[:20]
            records.append(
                {
                    "record_id": record_id,
                    "state": state_payload,
                    "starter_tile": starter_tile,
                    "phase4_stage_index": stage_idx,
                    "phase4_stage": ("early_lt384", "mid_384_768", "late_1536", "endgame_3072p")[stage_idx],
                    "trajectory_outcome": outcome,
                    "source_replay": str(path),
                    "source_seed": replay.get("seed"),
                    "source_frame_index": frame_index,
                    "source_policy": replay.get("policy"),
                    "source_origin": provenance.get("replay_origin"),
                    "root_origin": provenance.get("root_origin"),
                    "root_replay": provenance.get("root_replay") or str(path),
                    "root_seed": provenance.get("root_seed"),
                    "root_frame_index": provenance.get("root_frame_index", 0),
                    "root_move_count": provenance.get("root_move_count", 0),
                    "root_score": provenance.get("root_score"),
                    "root_policy": provenance.get("root_policy"),
                    "root_policy_family": provenance.get("root_policy_family"),
                    "behavior_family": family,
                    "ancestry_key": ancestry_id,
                }
            )
            stage_ancestries[stage_idx].add(ancestry_id)
            outcome_counts[(stage_idx, outcome)] += 1
            family_counts[family] += 1

    records.sort(key=lambda row: (int(row["phase4_stage_index"]), str(row["ancestry_key"]), str(row["record_id"])))
    stage_summary = {}
    for stage_idx, stage_name in enumerate(("early_lt384", "mid_384_768", "late_1536", "endgame_3072p")):
        stage_records = [row for row in records if int(row["phase4_stage_index"]) == stage_idx]
        ancestry_frame_counts = Counter(str(row["ancestry_key"]) for row in stage_records)
        stage_summary[stage_name] = {
            "states": len(stage_records),
            "unique_ancestries": len(stage_ancestries[stage_idx]),
            "max_ancestry_frame_share": (
                max(ancestry_frame_counts.values()) / len(stage_records) if stage_records else 0.0
            ),
            "success_states": int(outcome_counts[(stage_idx, "success")]),
            "failure_states": int(outcome_counts[(stage_idx, "failure")]),
        }
    return {
        "manifest_version": MANIFEST_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "sampling_contract": [
            "uniform eligible phase4 stage",
            "uniform root ancestry within stage",
            "uniform state within ancestry",
        ],
        "selection_uses_outcome": False,
        "counts": dict(counts),
        "stage_summary": stage_summary,
        "behavior_family_state_counts": dict(sorted(family_counts.items())),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-glob", action="append", default=[])
    parser.add_argument("--replay", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    patterns = args.replay_glob or DEFAULT_REPLAY_GLOBS
    paths = list(args.replay) + glob_replays(patterns)
    manifest = build_restart_manifest(paths)
    write_json(args.out, manifest)
    print(json.dumps({"out": str(args.out), "records": len(manifest["records"]), "stages": manifest["stage_summary"]}, indent=2))


if __name__ == "__main__":
    main()
