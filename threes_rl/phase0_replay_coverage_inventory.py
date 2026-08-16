"""Read-only replay coverage inventory for the phase-0 oracle gate.

This scans retained replay artifacts and asks whether existing normal-start
games can supply source-diverse h40 first-nonstarter-1536 roots.  It does not
run rollouts, labels, search, fitting, or normal-start evaluation.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import time
from collections import Counter, defaultdict
from html import escape
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from threes_rl.eval import max_tile_excluding_initial_starter
from threes_rl.phase0_oracle_corpus_audit import behavior_policy_family
from threes_rl.replay_provenance import GENUINE_ROOT_ORIGINS, replay_provenance
from threes_rl.run_artifacts import write_json


DEFAULT_REPLAY_GLOBS = [
    "threes_rl/runs/eval_artifacts/**/replay.json",
    "threes_rl/runs/replays/**/*.json",
    "threes_rl/runs/*/top_games/**/replay.json",
]


def _flatten_path_groups(path_groups: list[list[Path]] | None) -> list[Path]:
    if not path_groups:
        return []
    return [path for group in path_groups for path in group]


def _glob_replays(patterns: Iterable[str]) -> list[Path]:
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


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _move_action(frame: Any) -> str | None:
    if not isinstance(frame, dict):
        return None
    move = frame.get("move")
    if not isinstance(move, dict):
        return None
    action = move.get("action")
    return None if action is None else str(action)


def _frame_board(frame: Any) -> Any:
    if not isinstance(frame, dict):
        return None
    state = frame.get("state")
    if not isinstance(state, dict):
        return None
    return state.get("board")


def _frame_state(frame: Any) -> dict[str, Any] | None:
    if not isinstance(frame, dict):
        return None
    state = frame.get("state")
    return state if isinstance(state, dict) else None


def _starter_tile(replay: dict[str, Any]) -> int | None:
    value = replay.get("starter_tile", 1536)
    if value is None:
        return None
    return int(value)


def _max_excluding_starter(frame: Any, starter_tile: int | None) -> int | None:
    board = _frame_board(frame)
    if board is None:
        return None
    try:
        return int(max_tile_excluding_initial_starter(np.asarray(board, dtype=np.int32), starter_tile))
    except (TypeError, ValueError):
        return None


def _score_from_frame(frame: Any) -> int | None:
    state = _frame_state(frame)
    return None if state is None else _int_or_none(state.get("score"))


def _move_count_from_frame(frame: Any) -> int | None:
    state = _frame_state(frame)
    return None if state is None else _int_or_none(state.get("move_count"))


def _is_game_over(frame: Any) -> bool:
    state = _frame_state(frame)
    return bool(state.get("game_over")) if state is not None else False


def replay_action_signature(replay: dict[str, Any]) -> str:
    frames = replay.get("frames")
    actions: list[str | None] = []
    if isinstance(frames, list):
        actions = [_move_action(frame) for frame in frames]
    payload = {
        "policy": replay.get("policy"),
        "seed": replay.get("seed"),
        "starter_tile": replay.get("starter_tile"),
        "final_score": replay.get("final_score"),
        "final_moves": replay.get("final_moves"),
        "final_max_tile": replay.get("final_max_tile"),
        "actions": actions,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def first_nonstarter_promotion_index(
    replay: dict[str, Any],
    *,
    target_tile: int = 1536,
) -> int | None:
    frames = replay.get("frames")
    if not isinstance(frames, list):
        return None
    starter_tile = _starter_tile(replay)
    previous = 0
    for idx, frame in enumerate(frames):
        max_excl = _max_excluding_starter(frame, starter_tile)
        if max_excl is None:
            continue
        if previous < int(target_tile) <= int(max_excl):
            return int(idx)
        previous = max(previous, int(max_excl))
    return None


def success_extractable_h40(
    replay: dict[str, Any],
    *,
    target_tile: int = 1536,
    horizon: int = 40,
) -> tuple[bool, dict[str, Any]]:
    frames = replay.get("frames")
    if not isinstance(frames, list) or not frames:
        return False, {"reason": "missing_frames"}
    promotion = first_nonstarter_promotion_index(replay, target_tile=target_tile)
    if promotion is None:
        return False, {"reason": "no_promotion"}
    start = max(0, int(promotion) - int(horizon))
    for idx in range(start, int(promotion)):
        if idx + 1 >= len(frames):
            continue
        if _is_game_over(frames[idx]):
            continue
        action = _move_action(frames[idx + 1])
        if action is None:
            continue
        return True, {
            "reason": "ok",
            "frame_index": idx,
            "move_count": _move_count_from_frame(frames[idx]),
            "offset": int(promotion) - idx,
            "recorded_action": action,
            "score": _score_from_frame(frames[idx]),
        }
    return False, {"reason": "no_actionable_success_frame", "promotion_index": int(promotion)}


def failure_extractable_h40(
    replay: dict[str, Any],
    *,
    target_tile: int = 1536,
    horizon: int = 40,
) -> tuple[bool, dict[str, Any]]:
    frames = replay.get("frames")
    if not isinstance(frames, list) or not frames:
        return False, {"reason": "missing_frames"}
    if first_nonstarter_promotion_index(replay, target_tile=target_tile) is not None:
        return False, {"reason": "promoted"}
    starter_tile = _starter_tile(replay)
    control_tile = max(1, int(target_tile) // 2)
    terminal = len(frames) - 1
    for idx in range(max(0, terminal - int(horizon)), terminal):
        max_excl = _max_excluding_starter(frames[idx], starter_tile)
        if max_excl is None or int(max_excl) < control_tile or int(max_excl) >= int(target_tile):
            continue
        if _is_game_over(frames[idx]):
            continue
        action = _move_action(frames[idx + 1]) if idx + 1 < len(frames) else None
        if action is None:
            continue
        return True, {
            "reason": "ok",
            "frame_index": idx,
            "move_count": _move_count_from_frame(frames[idx]),
            "offset": int(terminal) - idx,
            "recorded_action": action,
            "score": _score_from_frame(frames[idx]),
            "max_tile_excl_starter": int(max_excl),
        }
    return False, {"reason": "no_actionable_failure_frame"}


def _outcome_for_replay(replay: dict[str, Any], *, target_tile: int) -> str:
    return "success" if first_nonstarter_promotion_index(replay, target_tile=target_tile) is not None else "failure"


def _matched_failure_possible(replay: dict[str, Any], *, target_tile: int) -> bool:
    if _outcome_for_replay(replay, target_tile=target_tile) != "failure":
        return False
    frames = replay.get("frames")
    if not isinstance(frames, list):
        return False
    starter_tile = _starter_tile(replay)
    control_tile = max(1, int(target_tile) // 2)
    for frame in frames:
        max_excl = _max_excluding_starter(frame, starter_tile)
        if max_excl is not None and control_tile <= int(max_excl) < int(target_tile):
            return True
    return False


def replay_inventory_row(path: Path, replay: dict[str, Any], *, target_tile: int, horizon: int) -> dict[str, Any]:
    provenance = replay_provenance(replay, path)
    outcome = _outcome_for_replay(replay, target_tile=target_tile)
    success_ok, success_info = success_extractable_h40(replay, target_tile=target_tile, horizon=horizon)
    failure_ok, failure_info = failure_extractable_h40(replay, target_tile=target_tile, horizon=horizon)
    extractable = success_ok if outcome == "success" else failure_ok
    extract_info = success_info if outcome == "success" else failure_info
    record_like = {
        "root_policy": provenance.get("root_policy"),
        "root_policy_family": provenance.get("root_policy_family"),
        "source_policy": provenance.get("source_policy"),
        "source_policy_family": provenance.get("source_policy_family"),
        "source_replay": str(path),
    }
    return {
        "path": str(path),
        "signature": replay_action_signature(replay),
        "policy": replay.get("policy"),
        "seed": replay.get("seed"),
        "starter_tile": replay.get("starter_tile"),
        "final_score": replay.get("final_score"),
        "final_moves": replay.get("final_moves"),
        "final_max_tile": replay.get("final_max_tile"),
        "origin": provenance["replay_origin"],
        "reset_invariant": provenance["replay_reset_invariant"],
        "reset_reason": provenance["replay_reset_reason"],
        "root_origin": provenance["root_origin"],
        "root_seed": provenance["root_seed"],
        "root_policy_family": provenance["root_policy_family"],
        "source_policy_family": provenance["source_policy_family"],
        "ancestry_key": provenance["ancestry_key"],
        "behavior_family": behavior_policy_family(record_like),
        "outcome": outcome,
        "matched_failure_possible": _matched_failure_possible(replay, target_tile=target_tile),
        "extractable_h40": bool(extractable),
        "extractable_reason": extract_info.get("reason"),
        "extractable_frame_index": extract_info.get("frame_index"),
        "extractable_move_count": extract_info.get("move_count"),
        "extractable_offset": extract_info.get("offset"),
        "extractable_recorded_action": extract_info.get("recorded_action"),
        "extractable_score": extract_info.get("score"),
    }


def _family_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[str(row.get("behavior_family", "unknown"))].append(row)
    out: dict[str, dict[str, Any]] = {}
    for family, family_rows in sorted(by_family.items()):
        extractable = [row for row in family_rows if row.get("extractable_h40")]
        success = [row for row in family_rows if row.get("outcome") == "success"]
        failure = [row for row in family_rows if row.get("outcome") == "failure"]
        extractable_success = [row for row in extractable if row.get("outcome") == "success"]
        extractable_failure = [row for row in extractable if row.get("outcome") == "failure"]
        out[family] = {
            "normal_start_replays": len(family_rows),
            "unique_ancestries": len({str(row.get("ancestry_key")) for row in family_rows}),
            "first_nonstarter_1536_success_replays": len(success),
            "matched_failure_replays": sum(1 for row in failure if row.get("matched_failure_possible")),
            "extractable_h40_roots": len(extractable),
            "extractable_h40_success_roots": len(extractable_success),
            "extractable_h40_failure_roots": len(extractable_failure),
            "root_seeds": len({row.get("root_seed") for row in family_rows if row.get("root_seed") is not None}),
            "example_replays": [row["path"] for row in family_rows[:5]],
        }
    return out


def _new_roots_needed_if_all_current_roots_kept_by_family(
    family_counts: dict[str, dict[str, Any]],
    *,
    max_family_share: float,
    min_roots: int,
    min_behavior_families: int,
) -> dict[str, dict[str, Any]]:
    current = {
        family: int(stats.get("extractable_h40_roots") or 0)
        for family, stats in family_counts.items()
        if int(stats.get("extractable_h40_roots") or 0) > 0
    }
    if current:
        largest_count = max(current.values())
    else:
        largest_count = 0
    required_peer = int(np.ceil(largest_count * (1.0 / float(max_family_share) - 1.0))) if largest_count else min_roots
    required_peer = max(required_peer, 1 if min_behavior_families > 1 else 0)
    out: dict[str, dict[str, Any]] = {}
    for family, stats in family_counts.items():
        have = int(stats.get("extractable_h40_roots") or 0)
        needed = max(0, required_peer - have)
        total_if_filled = sum(current.values()) + needed
        out[family] = {
            "have_extractable_h40_roots": have,
            "minimum_new_roots_to_pair_with_largest_family": needed,
            "would_meet_min_total_roots_after_fill": total_if_filled >= int(min_roots),
            "note": (
                "Assumes new roots are from this behavior family and existing largest-family roots remain in the corpus."
            ),
        }
    return out


def _outcome_feasible(selection: dict[str, int], family_counts: dict[str, dict[str, Any]], min_roots_per_outcome: int) -> bool:
    possible_success = 0
    possible_failure = 0
    for family, selected in selection.items():
        stats = family_counts.get(family, {})
        possible_success += min(int(selected), int(stats.get("extractable_h40_success_roots") or 0))
        possible_failure += min(int(selected), int(stats.get("extractable_h40_failure_roots") or 0))
    return possible_success >= int(min_roots_per_outcome) and possible_failure >= int(min_roots_per_outcome)


def _selection_summary(
    name: str,
    selection: dict[str, int],
    family_counts: dict[str, dict[str, Any]],
    *,
    max_family_share: float,
    min_roots: int,
    min_behavior_families: int,
    min_roots_per_outcome: int,
    note: str,
) -> dict[str, Any] | None:
    selection = {family: int(roots) for family, roots in selection.items() if int(roots) > 0}
    total = sum(selection.values())
    if not selection or total <= 0:
        return None
    largest_family, largest_roots = max(selection.items(), key=lambda item: item[1])
    largest_share = float(largest_roots / total)
    families = len(selection)
    success_capacity = sum(
        min(int(roots), int(family_counts.get(family, {}).get("extractable_h40_success_roots") or 0))
        for family, roots in selection.items()
    )
    failure_capacity = sum(
        min(int(roots), int(family_counts.get(family, {}).get("extractable_h40_failure_roots") or 0))
        for family, roots in selection.items()
    )
    checks = {
        "min_roots": total >= int(min_roots),
        "min_behavior_families": families >= int(min_behavior_families),
        "max_family_share": largest_share <= float(max_family_share),
        "success_controls_present": success_capacity >= int(min_roots_per_outcome),
        "failure_controls_present": failure_capacity >= int(min_roots_per_outcome),
    }
    return {
        "name": name,
        "selected_roots": total,
        "families": families,
        "largest_family": {"family": largest_family, "roots": largest_roots, "share": largest_share},
        "family_caps": dict(sorted(selection.items())),
        "success_capacity": int(success_capacity),
        "failure_capacity": int(failure_capacity),
        "checks": checks,
        "ready": all(checks.values()),
        "note": note,
    }


def _retained_downsample_options(
    family_counts: dict[str, dict[str, Any]],
    *,
    max_family_share: float,
    min_roots: int,
    min_behavior_families: int,
    min_roots_per_outcome: int,
) -> list[dict[str, Any]]:
    extractable = {
        family: int(stats.get("extractable_h40_roots") or 0)
        for family, stats in family_counts.items()
        if int(stats.get("extractable_h40_roots") or 0) > 0
    }
    if not extractable:
        return []

    options: list[dict[str, Any]] = []
    largest_family = max(extractable.items(), key=lambda item: item[1])[0]
    non_largest = {family: roots for family, roots in extractable.items() if family != largest_family}
    non_largest_total = sum(non_largest.values())
    if non_largest_total > 0:
        cap = int(non_largest_total * float(max_family_share) / max(1e-12, 1.0 - float(max_family_share)))
        cap = min(int(extractable[largest_family]), max(0, cap))
        selection = dict(non_largest)
        if cap > 0:
            selection[largest_family] = cap
        option = _selection_summary(
            "all_non_largest_plus_capped_largest",
            selection,
            family_counts,
            max_family_share=max_family_share,
            min_roots=min_roots,
            min_behavior_families=min_behavior_families,
            min_roots_per_outcome=min_roots_per_outcome,
            note="Uses all extractable roots outside the largest family and caps the largest family to the frozen share rule.",
        )
        if option is not None:
            options.append(option)

        non_largest_selection = dict(non_largest)
        if non_largest_selection:
            top_non_largest, top_roots = max(non_largest_selection.items(), key=lambda item: item[1])
            other_roots = non_largest_total - top_roots
            if other_roots > 0:
                cap = int(other_roots * float(max_family_share) / max(1e-12, 1.0 - float(max_family_share)))
                non_largest_selection[top_non_largest] = min(top_roots, max(0, cap))
            option = _selection_summary(
                "exclude_largest_family",
                non_largest_selection,
                family_counts,
                max_family_share=max_family_share,
                min_roots=min_roots,
                min_behavior_families=min_behavior_families,
                min_roots_per_outcome=min_roots_per_outcome,
                note="Excludes the dominant family entirely and caps the next-largest family only if needed.",
            )
            if option is not None:
                options.append(option)

    families = sorted(extractable.items(), key=lambda item: (-item[1], item[0]))
    for idx, (left, left_roots) in enumerate(families):
        for right, right_roots in families[idx + 1 :]:
            cap = min(left_roots, right_roots)
            option = _selection_summary(
                f"pair:{left}+{right}",
                {left: cap, right: cap},
                family_counts,
                max_family_share=max_family_share,
                min_roots=min_roots,
                min_behavior_families=min_behavior_families,
                min_roots_per_outcome=min_roots_per_outcome,
                note="Two-family retained subset with both families capped to equal root counts.",
            )
            if option is not None:
                options.append(option)

    ready_options = [option for option in options if option.get("ready")]
    other_options = [option for option in options if not option.get("ready")]
    ready_options.sort(key=lambda option: (-int(option["selected_roots"]), str(option["name"])))
    other_options.sort(key=lambda option: (-int(option["selected_roots"]), str(option["name"])))
    return ready_options[:10] + other_options[:5]


def inventory_replays(
    paths: list[Path],
    *,
    target_tile: int = 1536,
    horizon: int = 40,
    min_roots: int = 20,
    min_behavior_families: int = 2,
    max_family_share: float = 0.5,
    min_roots_per_outcome: int = 4,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    seen_signatures: set[str] = set()
    for path in paths:
        try:
            replay = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            rejected["bad_json"] += 1
            continue
        if not isinstance(replay, dict):
            rejected["bad_replay"] += 1
            continue
        row = replay_inventory_row(path, replay, target_tile=target_tile, horizon=horizon)
        if row["origin"] not in GENUINE_ROOT_ORIGINS or row["root_origin"] not in GENUINE_ROOT_ORIGINS:
            rejected["not_genuine_root"] += 1
            continue
        if not bool(row.get("reset_invariant")):
            rejected["not_reset_start"] += 1
            continue
        signature_key = f"{row['behavior_family']}:{row['signature']}"
        if signature_key in seen_signatures:
            rejected["duplicate_replay_signature"] += 1
            continue
        seen_signatures.add(signature_key)
        rows.append(row)

    family_counts = _family_counts(rows)
    extractable_by_family = {
        family: int(stats.get("extractable_h40_roots") or 0)
        for family, stats in family_counts.items()
        if int(stats.get("extractable_h40_roots") or 0) > 0
    }
    extractable_total = sum(extractable_by_family.values())
    largest_family = None
    largest_count = 0
    if extractable_by_family:
        largest_family, largest_count = max(extractable_by_family.items(), key=lambda item: item[1])
    behavior_families_with_extractable = len(extractable_by_family)
    largest_share = float(largest_count / extractable_total) if extractable_total else 0.0
    extractable_success_total = sum(
        int(stats.get("extractable_h40_success_roots") or 0) for stats in family_counts.values()
    )
    extractable_failure_total = sum(
        int(stats.get("extractable_h40_failure_roots") or 0) for stats in family_counts.values()
    )
    readiness = {
        "min_roots": extractable_total >= int(min_roots),
        "min_behavior_families": behavior_families_with_extractable >= int(min_behavior_families),
        "max_family_share": largest_share <= float(max_family_share) if extractable_total else False,
        "success_controls_present": extractable_success_total >= int(min_roots_per_outcome),
        "failure_controls_present": extractable_failure_total >= int(min_roots_per_outcome),
    }
    downsample_options = _retained_downsample_options(
        family_counts,
        max_family_share=max_family_share,
        min_roots=min_roots,
        min_behavior_families=min_behavior_families,
        min_roots_per_outcome=min_roots_per_outcome,
    )
    downsample_ready = any(bool(option.get("ready")) for option in downsample_options)
    return {
        "version": 1,
        "kind": "phase0_replay_coverage_inventory",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "mode": "dry_run_read_only",
        "target_tile": int(target_tile),
        "horizon": int(horizon),
        "source_files": len(paths),
        "normal_start_replays": len(rows),
        "deduped_replays": len(rows),
        "extractable_h40_roots": extractable_total,
        "extractable_h40_success_roots": extractable_success_total,
        "extractable_h40_failure_roots": extractable_failure_total,
        "behavior_families_with_extractable": behavior_families_with_extractable,
        "largest_extractable_family": {
            "family": largest_family,
            "roots": int(largest_count),
            "share": largest_share,
        },
        "family_counts": family_counts,
        "minimum_new_roots_if_all_current_roots_kept_by_family": _new_roots_needed_if_all_current_roots_kept_by_family(
            family_counts,
            max_family_share=max_family_share,
            min_roots=min_roots,
            min_behavior_families=min_behavior_families,
        ),
        "minimum_new_roots_by_family": _new_roots_needed_if_all_current_roots_kept_by_family(
            family_counts,
            max_family_share=max_family_share,
            min_roots=min_roots,
            min_behavior_families=min_behavior_families,
        ),
        "retained_downsample_options": downsample_options,
        "readiness_checks": readiness,
        "corpus_ready_if_using_retained_replays": all(readiness.values()),
        "corpus_selectable_from_retained_replays": downsample_ready,
        "minimum_new_roots_needed_if_downsample_allowed": 0 if downsample_ready else None,
        "rejected": dict(rejected),
        "rows_preview": rows[:200],
    }


def write_html(path: Path, payload: dict[str, Any]) -> None:
    def cell(value: object) -> str:
        return escape(str(value))

    family_rows = []
    family_counts = payload.get("family_counts", {})
    needs = payload.get("minimum_new_roots_if_all_current_roots_kept_by_family", {})
    if isinstance(family_counts, dict):
        for family, stats in sorted(family_counts.items()):
            need = needs.get(family, {}) if isinstance(needs, dict) else {}
            family_rows.append(
                "<tr>"
                f"<td>{cell(family)}</td>"
                f"<td>{cell(stats.get('normal_start_replays'))}</td>"
                f"<td>{cell(stats.get('unique_ancestries'))}</td>"
                f"<td>{cell(stats.get('first_nonstarter_1536_success_replays'))}</td>"
                f"<td>{cell(stats.get('matched_failure_replays'))}</td>"
                f"<td>{cell(stats.get('extractable_h40_roots'))}</td>"
                f"<td>{cell(stats.get('extractable_h40_success_roots'))}</td>"
                f"<td>{cell(stats.get('extractable_h40_failure_roots'))}</td>"
                f"<td>{cell(need.get('minimum_new_roots_to_pair_with_largest_family'))}</td>"
                "</tr>"
            )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Phase-0 Replay Coverage Inventory</title>
  <style>
    :root {{ color-scheme: dark; --bg:#101318; --panel:#171d24; --line:#34404d; --text:#edf2f7; --muted:#aab6c2; --gold:#f2c14e; }}
    body {{ margin:0; background:var(--bg); color:var(--text); font-family:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ width:min(1180px, calc(100vw - 32px)); margin:0 auto; padding:24px 0 40px; }}
    h1 {{ margin:0 0 8px; font-size:24px; }}
    .muted {{ color:var(--muted); }}
    .cards {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(160px, 1fr)); gap:10px; margin:18px 0; }}
    .card {{ border:1px solid var(--line); background:var(--panel); border-radius:8px; padding:12px; }}
    .label {{ color:var(--muted); font-size:11px; text-transform:uppercase; }}
    .value {{ margin-top:4px; color:var(--gold); font-size:22px; font-weight:800; }}
    table {{ width:100%; border-collapse:collapse; font-size:12px; font-variant-numeric:tabular-nums; margin:12px 0 20px; }}
    th, td {{ border-bottom:1px solid var(--line); padding:7px 8px; text-align:right; vertical-align:top; }}
    th:first-child, td:first-child {{ text-align:left; }}
    pre {{ white-space:pre-wrap; overflow-wrap:anywhere; color:var(--muted); }}
  </style>
</head>
<body>
  <main>
    <h1>Phase-0 Replay Coverage Inventory</h1>
    <p class="muted">Dry-run only: retained replay artifacts are counted, no rollout labels/search/training are executed.</p>
    <section class="cards">
      <div class="card"><div class="label">Raw Ready</div><div class="value">{cell(payload.get('corpus_ready_if_using_retained_replays'))}</div></div>
      <div class="card"><div class="label">Subset Ready</div><div class="value">{cell(payload.get('corpus_selectable_from_retained_replays'))}</div></div>
      <div class="card"><div class="label">Normal Starts</div><div class="value">{cell(payload.get('normal_start_replays'))}</div></div>
      <div class="card"><div class="label">Extractable Roots</div><div class="value">{cell(payload.get('extractable_h40_roots'))}</div></div>
      <div class="card"><div class="label">Families</div><div class="value">{cell(payload.get('behavior_families_with_extractable'))}</div></div>
    </section>
    <table><thead><tr><th>Family</th><th>Normal Starts</th><th>Ancestries</th><th>1536 Successes</th><th>Matched Failures</th><th>h40 Roots</th><th>h40 Success</th><th>h40 Failure</th><th>New Roots Needed If Keeping All Current</th></tr></thead><tbody>{''.join(family_rows)}</tbody></table>
    <h2>Retained Downsample Options</h2>
    <pre>{escape(json.dumps(payload.get('retained_downsample_options'), indent=2, sort_keys=True))}</pre>
    <h2>Summary JSON</h2>
    <pre>{escape(json.dumps({k: v for k, v in payload.items() if k != 'rows_preview'}, indent=2, sort_keys=True))}</pre>
  </main>
</body>
</html>
"""
    path.write_text(html)


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    paths = _flatten_path_groups(args.replay_json) + _glob_replays(args.replay_glob)
    payload = inventory_replays(
        paths,
        target_tile=args.target_tile,
        horizon=args.horizon,
        min_roots=args.min_roots,
        min_behavior_families=args.min_behavior_families,
        max_family_share=args.max_family_share,
        min_roots_per_outcome=args.min_roots_per_outcome,
    )
    payload["source_paths"] = [str(path) for path in paths]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload["json"] = str(args.out_dir / "phase0_replay_coverage_inventory.json")
    payload["summary_json"] = str(args.out_dir / "summary.json")
    payload["html"] = str(args.out_dir / "phase0_replay_coverage_inventory.html")
    write_json(args.out_dir / "phase0_replay_coverage_inventory.json", payload)
    write_json(args.out_dir / "summary.json", {key: value for key, value in payload.items() if key not in {"rows_preview", "source_paths"}})
    write_html(args.out_dir / "phase0_replay_coverage_inventory.html", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-json", type=Path, nargs="+", action="append", default=[])
    parser.add_argument("--replay-glob", action="append", default=list(DEFAULT_REPLAY_GLOBS))
    parser.add_argument("--target-tile", type=int, default=1536)
    parser.add_argument("--horizon", type=int, default=40)
    parser.add_argument("--min-roots", type=int, default=20)
    parser.add_argument("--min-behavior-families", type=int, default=2)
    parser.add_argument("--max-family-share", type=float, default=0.5)
    parser.add_argument("--min-roots-per-outcome", type=int, default=4)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("threes_rl/runs/forensics/phase0_replay_coverage_inventory/latest"),
    )
    args = parser.parse_args()
    payload = run_from_args(args)
    compact = {
        "corpus_ready_if_using_retained_replays": payload["corpus_ready_if_using_retained_replays"],
        "corpus_selectable_from_retained_replays": payload["corpus_selectable_from_retained_replays"],
        "normal_start_replays": payload["normal_start_replays"],
        "extractable_h40_roots": payload["extractable_h40_roots"],
        "extractable_h40_success_roots": payload["extractable_h40_success_roots"],
        "extractable_h40_failure_roots": payload["extractable_h40_failure_roots"],
        "behavior_families_with_extractable": payload["behavior_families_with_extractable"],
        "largest_extractable_family": payload["largest_extractable_family"],
        "family_counts": payload["family_counts"],
        "minimum_new_roots_if_all_current_roots_kept_by_family": payload[
            "minimum_new_roots_if_all_current_roots_kept_by_family"
        ],
        "retained_downsample_options": payload["retained_downsample_options"],
        "json": payload["json"],
        "html": payload["html"],
    }
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
