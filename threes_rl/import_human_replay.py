"""Import observed human tracker sessions into RL replay JSON."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from threes_rl.record_replay import preview_payload, state_payload, write_html
from threes_rl.replay_provenance import ORIGIN_HUMAN, direct_root_fields
from threes_rl.run_artifacts import safe_name, write_json
from threes_rl.sim import (
    DIRECTION_NAMES,
    SMALL_TILE_VALUES,
    SimState,
    ThreesSim,
    board_max_tile,
    label_for_insert_value,
    preview_from_label,
    score_board,
    simulate_base_move,
    tokens_to_board,
)


class HumanReplayImportError(ValueError):
    """Raised when an observed game cannot be converted safely."""


def parse_starter(text: str) -> int | None:
    value = text.strip().lower()
    return None if value in ("none", "null") else int(value)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with Path(path).open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                events.append(payload)
    return events


def split_games(events: Iterable[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    games: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for event in events:
        event_type = event.get("type")
        if event_type == "game_start":
            if current:
                games.append(current)
            current = [event]
            continue
        if event_type not in ("observed_move", "game_end"):
            continue
        if not current:
            current = []
        current.append(event)
        if event_type == "game_end":
            games.append(current)
            current = []
    if current:
        games.append(current)
    return games


def _board_from_tokens(tokens: object) -> np.ndarray:
    if not isinstance(tokens, list):
        raise HumanReplayImportError("missing board tokens")
    try:
        board = tokens_to_board(tokens)
    except (TypeError, ValueError) as exc:
        raise HumanReplayImportError(f"cannot parse board tokens: {exc}") from exc
    if board.shape != (4, 4):
        raise HumanReplayImportError(f"expected 4x4 board, got {board.shape}")
    return board


def _small_label_for_value(value: int) -> str | None:
    for label, small_value in SMALL_TILE_VALUES.items():
        if int(value) == int(small_value):
            return label
    return None


def _initial_state(start_event: dict[str, Any], *, starter_tile: int | None, sim: ThreesSim) -> SimState:
    board = _board_from_tokens(start_event.get("board") or start_event.get("before_board"))
    preview_label = str(start_event.get("preview_label") or start_event.get("before_preview_label") or "")
    if preview_label not in SMALL_TILE_VALUES:
        raise HumanReplayImportError(f"fresh-game preview must be small, got {preview_label!r}")

    small_counts = {"red": 4, "blue": 4, "gray": 4}
    small_seen_on_board = 0
    for value in board.reshape(-1).tolist():
        label = _small_label_for_value(int(value))
        if label is None:
            continue
        small_counts[label] -= 1
        small_seen_on_board += 1
        if small_counts[label] < 0:
            raise HumanReplayImportError(f"initial board overuses {label} tiles")
    if small_seen_on_board != 8:
        raise HumanReplayImportError(f"fresh-game import expects 8 small board tiles, got {small_seen_on_board}")
    if small_counts[preview_label] <= 0:
        raise HumanReplayImportError(f"initial preview {preview_label!r} is impossible from remaining bag")

    max_tile = board_max_tile(board)
    state = SimState(
        board=board,
        preview=preview_from_label(preview_label),
        small_counts=small_counts,
        small_pos=small_seen_on_board,
        small_seen_total=0,
        span_small_pos=0,
        large_pending=False,
        max_tile=max_tile,
        move_count=0,
        game_over=False,
    )
    state.game_over = max_tile >= 12288 or not sim.legal_actions(state)
    if starter_tile is not None and int(board[0, 0]) != int(starter_tile):
        raise HumanReplayImportError(
            f"starter_tile={starter_tile} expected at top-left, got {int(board[0, 0])}"
        )
    return state


def _bonus_preview_for_insert(
    *,
    sim: ThreesSim,
    max_tile: int,
    inserted_value: int | None,
    notes: list[dict[str, Any]],
    context: dict[str, Any],
):
    windows = sim.bonus_windows(max_tile)
    if not windows:
        notes.append({**context, "kind": "bonus_without_windows", "max_tile": int(max_tile)})
        return preview_from_label("large_candidates", ())
    if inserted_value is None:
        notes.append({**context, "kind": "bonus_window_unknown", "chosen": list(windows[0])})
        return preview_from_label("large_candidates", windows[0])
    containing = [window for window in windows if int(inserted_value) in window]
    if not containing:
        notes.append(
            {
                **context,
                "kind": "bonus_insert_outside_support",
                "inserted_value": int(inserted_value),
                "chosen": list(windows[0]),
            }
        )
        return preview_from_label("large_candidates", windows[0])

    def priority(window: tuple[int, int, int]) -> tuple[int, tuple[int, int, int]]:
        return (abs(window.index(int(inserted_value)) - 1), window)

    chosen = sorted(containing, key=priority)[0]
    if len(containing) > 1:
        notes.append(
            {
                **context,
                "kind": "bonus_window_ambiguous",
                "inserted_value": int(inserted_value),
                "chosen": list(chosen),
                "alternatives": [list(window) for window in containing],
            }
        )
    return preview_from_label("large_candidates", chosen)


def _preview_for_label(
    label: str,
    *,
    sim: ThreesSim,
    max_tile: int,
    inserted_value: int | None,
    notes: list[dict[str, Any]],
    context: dict[str, Any],
):
    if label == "large_candidates":
        return _bonus_preview_for_insert(
            sim=sim,
            max_tile=max_tile,
            inserted_value=inserted_value,
            notes=notes,
            context=context,
        )
    return preview_from_label(label)


def _event_is_valid(event: dict[str, Any]) -> bool:
    transition_check = event.get("transition_check")
    preview_check = event.get("preview_check")
    return (
        event.get("type") == "observed_move"
        and isinstance(transition_check, dict)
        and bool(transition_check.get("valid", False))
        and isinstance(preview_check, dict)
        and bool(preview_check.get("valid", True))
        and not bool(event.get("unknown_board"))
        and not bool(event.get("unknown_preview"))
    )


def _steps_for_event(event: dict[str, Any]) -> list[dict[str, Any]]:
    path = event.get("transition_path")
    if isinstance(path, list) and path:
        return [step for step in path if isinstance(step, dict)]
    transition_check = event.get("transition_check")
    if not isinstance(transition_check, dict):
        return []
    return [
        {
            "direction": event.get("direction"),
            "preview_label": event.get("before_preview_label"),
            "inserted_value": transition_check.get("inserted_value"),
            "inserted_pos": transition_check.get("inserted_pos"),
            "eligible_positions": transition_check.get("eligible_positions", []),
            "expected_values": transition_check.get("expected_values", []),
            "after_board": event.get("after_board"),
        }
    ]


def _as_pos(value: object) -> tuple[int, int] | None:
    if value is None:
        return None
    row, col = value  # type: ignore[misc]
    return int(row), int(col)


def _move_payload(
    *,
    before: SimState,
    after_board: np.ndarray,
    step: dict[str, Any],
) -> dict[str, Any]:
    action = str(step.get("direction"))
    if action not in DIRECTION_NAMES:
        raise HumanReplayImportError(f"unsupported direction {action!r}")
    shifted, computed_eligible = simulate_base_move(before.board, action)
    inserted_value = step.get("inserted_value")
    inserted_pos = _as_pos(step.get("inserted_pos"))
    eligible_positions = [
        _as_pos(pos) for pos in step.get("eligible_positions", []) if _as_pos(pos) is not None
    ]
    if not eligible_positions:
        eligible_positions = list(computed_eligible)

    terminal_merge = bool(np.any(shifted == 12288))
    if not terminal_merge:
        if inserted_value is None or inserted_pos is None:
            raise HumanReplayImportError("non-terminal observed move is missing inserted tile")
        candidate = shifted.copy()
        candidate[inserted_pos] = int(inserted_value)
        if not np.array_equal(candidate, after_board):
            raise HumanReplayImportError("observed after_board does not match direction/inserted tile")

    return {
        "action": action,
        "preview_used": preview_payload(before),
        "inserted_value": None if inserted_value is None else int(inserted_value),
        "inserted_pos": list(inserted_pos) if inserted_pos is not None else None,
        "eligible_positions": [list(pos) for pos in eligible_positions],
        "merge_score_delta": int(score_board(shifted) - score_board(before.board)),
        "score_delta": int(score_board(after_board) - score_board(before.board)),
        "terminal_merge": terminal_merge,
        "score_before": int(score_board(before.board)),
        "score_after": int(score_board(after_board)),
        "max_tile_before": int(before.max_tile),
        "max_tile_after": int(board_max_tile(after_board)),
    }


def _advance_state(
    *,
    sim: ThreesSim,
    before: SimState,
    after_board: np.ndarray,
    next_preview_label: str,
) -> SimState:
    counts, small_pos, small_seen_total, span_small_pos, large_pending = sim._consume_preview(
        before.small_counts,
        before.small_pos,
        before.small_seen_total,
        before.span_small_pos,
        before.large_pending,
        before.preview.label,
    )
    max_tile = board_max_tile(after_board)
    state = SimState(
        board=after_board.copy(),
        preview=preview_from_label(next_preview_label),
        small_counts=counts.copy(),
        small_pos=int(small_pos),
        small_seen_total=int(small_seen_total),
        span_small_pos=int(span_small_pos),
        large_pending=bool(large_pending),
        max_tile=int(max_tile),
        move_count=int(before.move_count) + 1,
        game_over=False,
    )
    state.game_over = max_tile >= 12288 or not sim.legal_actions(state)
    return state


def import_game(
    game_events: list[dict[str, Any]],
    *,
    source_events: Path,
    starter_tile: int | None = 1536,
    min_valid_moves: int = 1,
) -> dict[str, Any]:
    move_events = [event for event in game_events if event.get("type") == "observed_move"]
    if not move_events:
        raise HumanReplayImportError("game has no observed moves")
    start_event = next((event for event in game_events if event.get("type") == "game_start"), None)
    if start_event is None:
        first = move_events[0]
        start_event = {
            "board": first.get("before_board"),
            "preview_label": first.get("before_preview_label"),
            "game_index": first.get("game_index"),
        }

    sim = ThreesSim(np.random.default_rng(0), starter_tile=starter_tile)
    current = _initial_state(start_event, starter_tile=starter_tile, sim=sim)
    game_index = int(start_event.get("game_index") or move_events[0].get("game_index") or 1)
    frames: list[dict[str, Any]] = []
    pending_frame: dict[str, Any] | None = {"index": 0, "state": None, "move": None}
    preview_notes: list[dict[str, Any]] = []

    for event in move_events:
        if not _event_is_valid(event):
            raise HumanReplayImportError(
                f"invalid observed move at game={event.get('game_index')} move={event.get('move_index')}"
            )
        steps = _steps_for_event(event)
        if not steps:
            raise HumanReplayImportError(f"observed move has no transition path: move={event.get('move_index')}")

        for step_index, step in enumerate(steps):
            before_board = _board_from_tokens(event.get("before_board") if step_index == 0 else current.board.tolist())
            if not np.array_equal(before_board, current.board):
                raise HumanReplayImportError(f"state mismatch before move={event.get('move_index')}")
            expected_label = str(step.get("preview_label") or event.get("before_preview_label"))
            if expected_label != current.preview.label:
                raise HumanReplayImportError(
                    f"preview mismatch before move={event.get('move_index')}: "
                    f"expected {expected_label}, current {current.preview.label}"
                )

            inserted_value = step.get("inserted_value")
            current.preview = _preview_for_label(
                expected_label,
                sim=sim,
                max_tile=current.max_tile,
                inserted_value=None if inserted_value is None else int(inserted_value),
                notes=preview_notes,
                context={"move_count": int(current.move_count), "source_move": event.get("move_index")},
            )
            if pending_frame is not None:
                pending_frame["state"] = state_payload(current, sim)
                frames.append(pending_frame)
                pending_frame = None

            after_board = _board_from_tokens(step.get("after_board"))
            next_preview_label = (
                str(steps[step_index + 1].get("preview_label"))
                if step_index + 1 < len(steps)
                else str(event.get("after_preview_label"))
            )
            move = _move_payload(before=current, after_board=after_board, step=step)
            current = _advance_state(
                sim=sim,
                before=current,
                after_board=after_board,
                next_preview_label=next_preview_label,
            )
            pending_frame = {"index": len(frames), "state": None, "move": move}

    if pending_frame is not None:
        current.preview = _preview_for_label(
            current.preview.label,
            sim=sim,
            max_tile=current.max_tile,
            inserted_value=None,
            notes=preview_notes,
            context={"move_count": int(current.move_count), "source_move": "final"},
        )
        pending_frame["state"] = state_payload(current, sim)
        frames.append(pending_frame)

    if int(current.move_count) < int(min_valid_moves):
        raise HumanReplayImportError(f"only {current.move_count} valid moves")

    return {
        "policy": "human_observed",
        "seed": game_index,
        "starter_tile": starter_tile,
        "max_moves": int(current.move_count),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **direct_root_fields(
            origin=ORIGIN_HUMAN,
            seed=int(game_index),
            policy="human_observed",
            first_score=int(frames[0]["state"]["score"]),
        ),
        "final_score": int(score_board(current.board)),
        "final_moves": int(current.move_count),
        "final_max_tile": int(current.max_tile),
        "game_over": bool(current.game_over),
        "frames": frames,
        "human_import": {
            "source_events": str(source_events),
            "game_index": game_index,
            "preview_candidate_notes": preview_notes,
            "preview_candidate_note_count": len(preview_notes),
        },
    }


def import_events_file(
    events_path: Path,
    out_dir: Path,
    *,
    starter_tile: int | None = 1536,
    min_valid_moves: int = 1,
    write_replay_html: bool = True,
) -> dict[str, Any]:
    events_path = Path(events_path)
    out_dir = Path(out_dir)
    games = split_games(read_jsonl(events_path))
    replay_records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for game_pos, game_events in enumerate(games, start=1):
        game_index = next((event.get("game_index") for event in game_events if event.get("game_index") is not None), game_pos)
        try:
            replay = import_game(
                game_events,
                source_events=events_path,
                starter_tile=starter_tile,
                min_valid_moves=min_valid_moves,
            )
        except HumanReplayImportError as exc:
            skipped.append({"game_index": game_index, "reason": str(exc)})
            continue
        name = safe_name(f"human_game{int(game_index):03d}_moves{replay['final_moves']}_score{replay['final_score']}")
        replay_path = out_dir / f"{name}.json"
        html_path = out_dir / f"{name}.html"
        write_json(replay_path, replay)
        if write_replay_html:
            write_html(html_path, replay)
        replay_records.append(
            {
                "game_index": int(game_index),
                "json": str(replay_path),
                "html": str(html_path) if write_replay_html else None,
                "final_score": replay["final_score"],
                "final_moves": replay["final_moves"],
                "final_max_tile": replay["final_max_tile"],
                "preview_candidate_note_count": replay["human_import"]["preview_candidate_note_count"],
            }
        )

    summary = {
        "source_events": str(events_path),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "games_seen": len(games),
        "games_imported": len(replay_records),
        "games_skipped": len(skipped),
        "replays": replay_records,
        "skipped": skipped,
    }
    write_json(out_dir / "manifest.json", summary)
    return summary


def _flatten_paths(path_groups: list[list[Path]] | None) -> list[Path]:
    if not path_groups:
        return []
    return [path for group in path_groups for path in group]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events-jsonl", type=Path, nargs="+", action="append", required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("threes_rl/runs/human_replays/latest"))
    parser.add_argument("--starter", default="1536")
    parser.add_argument("--min-valid-moves", type=int, default=1)
    parser.add_argument("--no-html", action="store_true")
    args = parser.parse_args()

    manifests = []
    for path in _flatten_paths(args.events_jsonl):
        out_dir = args.out_dir / safe_name(path.parent.name or path.stem)
        manifests.append(
            import_events_file(
                path,
                out_dir,
                starter_tile=parse_starter(args.starter),
                min_valid_moves=args.min_valid_moves,
                write_replay_html=not args.no_html,
            )
        )
    combined = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "sources": [manifest["source_events"] for manifest in manifests],
        "games_seen": sum(int(manifest["games_seen"]) for manifest in manifests),
        "games_imported": sum(int(manifest["games_imported"]) for manifest in manifests),
        "games_skipped": sum(int(manifest["games_skipped"]) for manifest in manifests),
        "manifests": manifests,
    }
    write_json(args.out_dir / "manifest.json", combined)
    print(json.dumps(combined, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
