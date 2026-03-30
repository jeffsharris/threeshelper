import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional

import mirroring_control as mc
import window_stream as ws
from state_hunt import HarnessRecorder, preview_check_from_snapshot, valid_directions_for_transition


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Observe a human-played Threes game and stop on the first invalid tracked state."
    )
    parser.add_argument("--window-id", type=int, help="Window ID to target directly.")
    parser.add_argument(
        "--auto-window-prefix",
        default="iPhone Mirroring",
        help="Automatically select the first window whose title starts with this prefix.",
    )
    parser.add_argument(
        "--capture-backend",
        choices=("quartz", "screencapture"),
        default="quartz",
        help="Capture backend used for board/preview parsing.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("datasets/human_watch"),
        help="Base directory for recorded observation runs.",
    )
    parser.add_argument(
        "--attach-current-game",
        action="store_true",
        help="Attach to the current visible game even if it is not a fresh initial board.",
    )
    parser.add_argument(
        "--poll",
        type=float,
        default=0.1,
        help="Polling interval in seconds while waiting for the board to settle.",
    )
    parser.add_argument(
        "--settle-frames",
        type=int,
        default=2,
        help="Number of identical settled samples required before a move is accepted.",
    )
    parser.add_argument(
        "--settle-threshold",
        type=float,
        default=0.15,
        help="Board signature delta threshold used to count a frame as settled.",
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=400,
        help="Maximum observed moves before stopping.",
    )
    parser.add_argument(
        "--idle-timeout",
        type=float,
        default=0.0,
        help="Optional timeout in seconds with no observed move before aborting (0 disables).",
    )
    return parser.parse_args()


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


def print_frame(frame: mc.FrameState, snapshot: Optional[tuple]) -> None:
    if snapshot is None:
        print(ws.format_board_with_preview(frame.board, frame.preview_label))
        return
    cycle = ws.TileCycle()
    cycle.restore(snapshot)
    cycle.set_max_tile(ws.board_max_tile(frame.board))
    print(ws.render_move_table(frame.board, frame.preview_label, cycle))


def build_move_event(
    before_frame: mc.FrameState,
    before_snapshot: Optional[tuple],
    after_frame: mc.FrameState,
    game_index: int,
    move_index: int,
    capture_id: int,
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
        "move_index": move_index,
        "before_board": before_frame.board,
        "before_preview_label": before_frame.preview_label,
        "before_tile_cycle": before_snapshot,
        "after_capture_id": capture_id,
        "after_board": after_frame.board,
        "after_preview_label": after_frame.preview_label,
        "unknown_board": ws._board_has_unknowns(after_frame.board),
        "unknown_preview": after_frame.preview_label == "unknown",
        "preview_check": preview_check,
    }
    if len(matches) == 1:
        direction, transition = matches[0]
        event["direction"] = direction
        event["transition_check"] = {
            "valid": True,
            "reason": transition.reason,
            "eligible_positions": transition.eligible_positions,
            "expected_values": transition.expected_values,
            "inserted_value": transition.inserted_value,
            "inserted_pos": transition.inserted_pos,
            "best_mismatch": transition.best_mismatch,
        }
    else:
        event["direction"] = None
        event["possible_directions"] = [direction for direction, _transition in matches]
        event["transition_check"] = {
            "valid": len(matches) > 0,
            "reason": "ambiguous direction" if matches else "board does not match any legal move",
            "eligible_positions": [],
            "expected_values": [],
            "inserted_value": None,
            "inserted_pos": None,
            "best_mismatch": None,
        }
    return event


def main() -> None:
    args = parse_args()
    window_id, window_info = mc.resolve_window(args.window_id, args.auto_window_prefix)
    recorder = HarnessRecorder(args.dataset_dir, window_info=window_info)

    print(f"Using window {window_id}", flush=True)
    print(f"Recording run to {recorder.session_dir}", flush=True)

    tracked = False
    game_index = 1
    move_index = 0
    last_move_ts = time.time()
    last_stable_frame: Optional[mc.FrameState] = None
    last_snapshot: Optional[tuple] = None
    last_capture_id: Optional[int] = None
    stable_state: Optional[mc.FrameState] = None
    stable_count = 0

    while True:
        state = mc.capture_screen_state(window_id, args.capture_backend)

        if state.scene in (mc.SCENE_SCREEN_OFF, mc.SCENE_PHONE_IN_USE):
            raise RuntimeError(f"Observation stopped on unready scene {state.scene!r}")

        if state.scene in (mc.SCENE_GAME_OVER, mc.SCENE_POSTGAME):
            scene_label = "gameover" if state.scene == mc.SCENE_GAME_OVER else "postgame"
            scene_path = recorder.record_scene(
                state,
                f"{scene_label}_game{game_index:03d}_move{move_index:04d}",
            )
            recorder.append_event(
                {
                    "type": "game_end",
                    "game_index": game_index,
                    "move_index": move_index,
                    "scene": state.scene,
                    "scene_capture": scene_path,
                }
            )
            print(f"Observed {state.scene} after {move_index} moves.", flush=True)
            print(f"Artifacts saved to {recorder.session_dir}", flush=True)
            return

        if state.scene != mc.SCENE_GAME or state.frame is None:
            stable_state = None
            stable_count = 0
            time.sleep(args.poll)
            continue

        frame = state.frame
        if stable_state is None:
            stable_state = frame
            stable_count = 1
            time.sleep(args.poll)
            continue

        diff = ws.board_signature_diff(stable_state.board_sig, frame.board_sig)
        if diff < args.settle_threshold and same_semantics(stable_state, frame):
            stable_count += 1
        else:
            stable_state = frame
            stable_count = 1
            time.sleep(args.poll)
            continue

        if stable_count < args.settle_frames:
            time.sleep(args.poll)
            continue

        settled = stable_state
        if not tracked:
            initial_error = ws._initial_state_error(settled.board, settled.preview_label)
            if initial_error is not None and not args.attach_current_game:
                time.sleep(args.poll)
                continue
            last_snapshot = seed_snapshot(settled)
            last_capture_id = recorder.record_game_state(settled, window_id, time.time())
            recorder.append_event(
                {
                    "type": "game_start",
                    "game_index": game_index,
                    "capture_id": last_capture_id,
                    "board": settled.board,
                    "preview_label": settled.preview_label,
                    "tile_cycle": last_snapshot,
                    "scene": state.scene,
                    "tracking_enabled": last_snapshot is not None,
                }
            )
            tracked = True
            last_stable_frame = settled
            last_move_ts = time.time()
            if last_snapshot is None and initial_error is not None:
                print(f"tracking disabled: {initial_error}", flush=True)
            print_frame(settled, last_snapshot)
            print(flush=True)
            time.sleep(args.poll)
            continue

        if last_stable_frame is None or same_semantics(last_stable_frame, settled):
            if args.idle_timeout > 0 and time.time() - last_move_ts > args.idle_timeout:
                print(f"Idle timeout reached with no observed move for {args.idle_timeout:.1f}s.", flush=True)
                print(f"Artifacts saved to {recorder.session_dir}", flush=True)
                return
            time.sleep(args.poll)
            continue

        move_index += 1
        capture_id = recorder.record_game_state(settled, window_id, time.time())
        event = build_move_event(
            last_stable_frame,
            last_snapshot,
            settled,
            game_index=game_index,
            move_index=move_index,
            capture_id=capture_id,
        )
        recorder.append_event(event)

        failure_reasons: List[str] = []
        transition_check = event["transition_check"]
        if not event["preview_check"].get("valid", True):
            failure_reasons.append(f"preview_invalid: {event['preview_check'].get('reason', '')}")
        if not transition_check.get("valid", True):
            failure_reasons.append(f"transition_invalid: {transition_check.get('reason', '')}")
        if event["unknown_board"]:
            failure_reasons.append("board_contains_unknowns")
        if event["unknown_preview"]:
            failure_reasons.append("preview_unknown")

        if failure_reasons:
            scene_path = recorder.record_scene(
                state,
                f"failure_game{game_index:03d}_move{move_index:04d}",
                extra=event,
            )
            recorder.append_failure(
                {
                    "game_index": game_index,
                    "move_index": move_index,
                    "reasons": failure_reasons,
                    "scene_capture": scene_path,
                    "event": event,
                }
            )
            print(f"Invalid state detected: {failure_reasons}", flush=True)
            print(f"Artifacts saved to {recorder.session_dir}", flush=True)
            return

        next_snapshot = event["preview_check"].get("next_snapshot")
        last_snapshot = next_snapshot if isinstance(next_snapshot, tuple) else next_snapshot
        last_stable_frame = settled
        last_capture_id = capture_id
        last_move_ts = time.time()

        direction = event["direction"] or ",".join(event.get("possible_directions", [])) or "?"
        print(f"observed move {move_index}: {direction}", flush=True)
        print_frame(settled, last_snapshot)
        print(flush=True)

        if move_index >= args.max_moves:
            print(f"Reached max observed moves ({args.max_moves}).", flush=True)
            print(f"Artifacts saved to {recorder.session_dir}", flush=True)
            return

        time.sleep(args.poll)


if __name__ == "__main__":
    main()
