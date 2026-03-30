from typing import Dict, Optional

import mirroring_control as mc
import window_stream as ws
from state_hunt import (
    find_transition_paths,
    preview_check_from_snapshot,
    serialize_transition_step,
    valid_directions_for_transition,
)


def same_semantics(a: mc.FrameState, b: mc.FrameState) -> bool:
    return a.board == b.board and a.preview_label == b.preview_label


def seed_snapshot(frame: mc.FrameState) -> Optional[tuple]:
    err = ws._initial_state_error(frame.board, frame.preview_label)
    if err is not None:
        return None
    cycle = ws.TileCycle()
    ws.seed_tile_cycle_from_initial_state(cycle, frame.board, frame.preview_label)
    cycle.set_max_tile(ws.board_max_tile(frame.board))
    ok, _reason = ws.preview_possible(cycle, frame.preview_label)
    if ok:
        cycle.update(frame.preview_label)
    return cycle.snapshot()


def render_tracked_board(frame: mc.FrameState, snapshot: Optional[tuple]) -> str:
    if snapshot is None:
        return ws.format_board_with_preview(frame.board, frame.preview_label)
    cycle = ws.TileCycle()
    cycle.restore(snapshot)
    cycle.set_max_tile(ws.board_max_tile(frame.board))
    return ws.render_move_table(frame.board, frame.preview_label, cycle)


def build_move_event(
    before_frame: mc.FrameState,
    before_snapshot: Optional[tuple],
    after_frame: mc.FrameState,
    game_index: int,
    move_index_start: int,
    capture_id: int,
    max_recovery_depth: int,
) -> Dict[str, object]:
    matches = valid_directions_for_transition(
        before_frame.board,
        before_frame.preview_label,
        after_frame.board,
    )
    preview_check = preview_check_from_snapshot(
        before_snapshot,
        after_frame.board,
        after_frame.preview_label,
    )

    event: Dict[str, object] = {
        "type": "observed_move",
        "game_index": game_index,
        "move_index_start": move_index_start,
        "before_board": before_frame.board,
        "before_preview_label": before_frame.preview_label,
        "before_tile_cycle": before_snapshot,
        "after_capture_id": capture_id,
        "after_board": after_frame.board,
        "after_preview_label": after_frame.preview_label,
        "unknown_board": ws._board_has_unknowns(after_frame.board),
        "unknown_preview": after_frame.preview_label == "unknown",
        "preview_check": preview_check,
        "step_count": 1,
        "recovered_missed_moves": 0,
        "direction_sequence": [],
        "transition_path": [],
    }

    if matches:
        if len(matches) > 1:
            event["direction"] = None
            event["possible_directions"] = [direction for direction, _transition in matches]
            event["transition_check"] = {
                "valid": True,
                "reason": "ambiguous single-step direction",
                "eligible_positions": [],
                "expected_values": [],
                "inserted_value": None,
                "inserted_pos": None,
                "best_mismatch": None,
            }
            return event
        direction, transition = matches[0]
        event["direction"] = direction
        event["direction_sequence"] = [direction]
        event["transition_check"] = {
            "valid": True,
            "reason": transition.reason,
            "eligible_positions": transition.eligible_positions,
            "expected_values": transition.expected_values,
            "inserted_value": transition.inserted_value,
            "inserted_pos": transition.inserted_pos,
            "best_mismatch": transition.best_mismatch,
        }
        event["transition_path"] = [
            {
                "direction": direction,
                "preview_label": before_frame.preview_label,
                "inserted_value": transition.inserted_value,
                "inserted_pos": transition.inserted_pos,
                "eligible_positions": transition.eligible_positions,
                "expected_values": transition.expected_values,
                "after_board": after_frame.board,
            }
        ]
        return event

    paths = find_transition_paths(
        before_frame.board,
        before_frame.preview_label,
        before_snapshot,
        after_frame.board,
        after_frame.preview_label,
        max_depth=max_recovery_depth,
    )
    if len(paths) == 1:
        path = paths[0]
        event["direction"] = None
        event["possible_directions"] = []
        event["step_count"] = len(path.steps)
        event["recovered_missed_moves"] = max(0, len(path.steps) - 1)
        event["direction_sequence"] = [step.direction for step in path.steps]
        event["transition_path"] = [serialize_transition_step(step) for step in path.steps]
        event["preview_check"] = path.preview_check
        event["transition_check"] = {
            "valid": True,
            "reason": f"matched {len(path.steps)}-step transition path",
            "eligible_positions": [],
            "expected_values": [],
            "inserted_value": None,
            "inserted_pos": None,
            "best_mismatch": None,
        }
        return event

    event["direction"] = None
    event["possible_directions"] = [direction for direction, _transition in matches]
    event["transition_paths_considered"] = [
        [serialize_transition_step(step) for step in path.steps] for path in paths
    ]
    reason = "ambiguous multi-step path" if paths else "board does not match any legal move sequence"
    event["transition_check"] = {
        "valid": False,
        "reason": reason,
        "eligible_positions": [],
        "expected_values": [],
        "inserted_value": None,
        "inserted_pos": None,
        "best_mismatch": None,
    }
    return event
