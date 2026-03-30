import argparse
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import mirroring_control as mc
import window_stream as ws
from state_hunt import (
    HarnessRecorder,
    ordered_directions,
    preview_check_from_snapshot,
    validate_transition,
)


def build_player(
    args: argparse.Namespace,
    window_id: int,
    recorder: Optional[ws.DatasetRecorder],
) -> mc.AutoPlayer:
    return mc.AutoPlayer(
        window_id=window_id,
        backend=args.capture_backend,
        settle_frames=args.settle_frames,
        settle_poll=args.settle_poll,
        settle_threshold=args.settle_threshold,
        settle_timeout=args.settle_timeout,
        board_delta_threshold=args.board_delta_threshold,
        swipe_span_ratio=args.swipe_span_ratio,
        swipe_duration=args.swipe_duration,
        swipe_steps=args.swipe_steps,
        start_rel_x=args.start_rel_x,
        start_rel_y=args.start_rel_y,
        focus_delay=args.focus_delay,
        recorder=recorder,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play repeated Threes games and stop on the first invalid tracked state."
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
    parser.add_argument("--games", type=int, default=5, help="Number of games to play.")
    parser.add_argument(
        "--max-moves-per-game",
        type=int,
        default=400,
        help="Maximum successful moves to allow in a single game before aborting.",
    )
    parser.add_argument(
        "--move-order",
        choices=("random", "cycle"),
        default="random",
        help="Direction ordering strategy when searching for the next legal move.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for move ordering.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("datasets/state_hunt"),
        help="Base directory for recorded runs.",
    )
    parser.add_argument(
        "--start-from-title",
        action="store_true",
        help="Reset to the title screen before every game instead of retrying post-game directly.",
    )
    parser.add_argument(
        "--board-delta-threshold",
        type=float,
        default=0.3,
        help="Minimum board signature delta required to treat a swipe as a real move.",
    )
    parser.add_argument("--settle-frames", type=int, default=2)
    parser.add_argument("--settle-poll", type=float, default=0.1)
    parser.add_argument("--settle-threshold", type=float, default=0.15)
    parser.add_argument("--settle-timeout", type=float, default=1.5)
    parser.add_argument("--swipe-span-ratio", type=float, default=0.22)
    parser.add_argument("--swipe-duration", type=float, default=0.12)
    parser.add_argument("--swipe-steps", type=int, default=12)
    parser.add_argument("--start-rel-x", type=float, default=0.5)
    parser.add_argument("--start-rel-y", type=float, default=0.55)
    parser.add_argument("--focus-delay", type=float, default=0.2)
    parser.add_argument("--transition-delay", type=float, default=0.8)
    parser.add_argument(
        "--scene-timeout",
        type=float,
        default=3.5,
        help="Timeout in seconds for title/game/post-game transitions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    window_id, window_info = mc.resolve_window(args.window_id, args.auto_window_prefix)
    recorder = HarnessRecorder(args.dataset_dir, window_info=window_info)
    rng = random.Random(args.seed)

    print(f"Using window {window_id}", flush=True)
    print(f"Recording run to {recorder.session_dir}", flush=True)

    rotation = 0
    for game_index in range(1, args.games + 1):
        print(f"=== game {game_index} ===", flush=True)
        if args.start_from_title or game_index == 1:
            mc.ensure_title_scene(
                window_id,
                args.capture_backend,
                focus_delay=args.focus_delay,
                transition_delay=args.transition_delay,
                timeout=args.scene_timeout,
                poll=args.settle_poll,
            )
            mc.start_game_sequence(
                window_id,
                args.capture_backend,
                focus_delay=args.focus_delay,
                transition_delay=args.transition_delay,
                timeout=args.scene_timeout,
                poll=args.settle_poll,
            )
        else:
            mc.ensure_game_scene(
                window_id,
                args.capture_backend,
                focus_delay=args.focus_delay,
                transition_delay=args.transition_delay,
                timeout=args.scene_timeout,
                poll=args.settle_poll,
            )

        player = build_player(args, window_id, recorder.dataset)
        player.initialize()
        if player.current_state is None or player.current_state.scene != mc.SCENE_GAME or player.current_state.frame is None:
            raise RuntimeError("Failed to initialize player on a game screen")

        init_capture_id = recorder.record_game_state(player.current_state.frame, window_id, time.time())
        init_snapshot = player.tile_cycle.snapshot() if player.tile_cycle is not None else None
        recorder.append_event(
            {
                "type": "game_start",
                "game_index": game_index,
                "capture_id": init_capture_id,
                "board": player.current_state.frame.board,
                "preview_label": player.current_state.frame.preview_label,
                "tile_cycle": init_snapshot,
                "scene": player.current_state.scene,
            }
        )

        game_over = False
        last_capture_id = init_capture_id

        for move_index in range(1, args.max_moves_per_game + 1):
            directions, rotation = ordered_directions(args.move_order, rng, rotation)
            moved = False

            for attempt_index, direction in enumerate(directions, start=1):
                if player.current_state is None or player.current_state.frame is None:
                    raise RuntimeError("Lost current game state before move")
                before_frame = player.current_state.frame
                before_snapshot = player.tile_cycle.snapshot() if player.tile_cycle is not None else None
                before_capture_id = last_capture_id

                result = player.attempt_move(direction)
                after_state = player.current_state
                ts_event = time.time()

                event: Dict[str, object] = {
                    "type": "move_attempt",
                    "game_index": game_index,
                    "move_index": move_index,
                    "attempt_index": attempt_index,
                    "direction": direction,
                    "before_capture_id": before_capture_id,
                    "before_preview_label": before_frame.preview_label,
                    "before_board": before_frame.board,
                    "before_tile_cycle": before_snapshot,
                    "result": {
                        "changed": result.changed,
                        "scene": result.scene,
                        "board_delta": result.board_delta,
                        "game_over": result.game_over,
                    },
                }

                if result.scene == mc.SCENE_GAME and after_state is not None and after_state.frame is not None:
                    after_frame = after_state.frame
                    event.update(
                        {
                            "after_capture_id": recorder.dataset.last_capture_id if result.changed else None,
                            "after_preview_label": after_frame.preview_label,
                            "after_board": after_frame.board,
                            "unknown_board": ws._board_has_unknowns(after_frame.board),
                            "unknown_preview": after_frame.preview_label == "unknown",
                        }
                    )

                    failure_reasons: List[str] = []
                    if result.changed:
                        preview_check = preview_check_from_snapshot(
                            before_snapshot,
                            after_frame.board,
                            after_frame.preview_label,
                        )
                        transition = validate_transition(
                            before_frame.board,
                            direction,
                            before_frame.preview_label,
                            after_frame.board,
                        )
                        event["preview_check"] = preview_check
                        event["transition_check"] = {
                            "valid": transition.valid,
                            "reason": transition.reason,
                            "eligible_positions": transition.eligible_positions,
                            "expected_values": transition.expected_values,
                            "inserted_value": transition.inserted_value,
                            "inserted_pos": transition.inserted_pos,
                            "best_mismatch": transition.best_mismatch,
                        }
                        if not preview_check.get("valid", True):
                            failure_reasons.append(f"preview_invalid: {preview_check.get('reason', '')}")
                        if not transition.valid:
                            failure_reasons.append(f"transition_invalid: {transition.reason}")
                    else:
                        event["preview_check"] = {"valid": True, "reason": "no-op swipe"}
                        event["transition_check"] = {
                            "valid": True,
                            "reason": "no-op swipe",
                            "eligible_positions": [],
                            "expected_values": [],
                            "inserted_value": None,
                            "inserted_pos": None,
                            "best_mismatch": None,
                        }

                    if event["unknown_board"]:
                        failure_reasons.append("board_contains_unknowns")
                    if event["unknown_preview"]:
                        failure_reasons.append("preview_unknown")

                    recorder.append_event(event)

                    if failure_reasons:
                        scene_path = recorder.record_scene(
                            after_state,
                            f"failure_game{game_index:03d}_move{move_index:04d}",
                            extra=event,
                        )
                        failure = {
                            "game_index": game_index,
                            "move_index": move_index,
                            "direction": direction,
                            "reasons": failure_reasons,
                            "scene_capture": scene_path,
                            "event": event,
                        }
                        recorder.append_failure(failure)
                        print(f"Invalid state detected: {failure_reasons}", flush=True)
                        print(f"Artifacts saved to {recorder.session_dir}", flush=True)
                        return

                    if result.changed:
                        last_capture_id = recorder.dataset.last_capture_id or last_capture_id
                        moved = True
                        break
                    continue

                if result.scene in (mc.SCENE_GAME_OVER, mc.SCENE_POSTGAME) and after_state is not None:
                    scene_label = "gameover" if result.scene == mc.SCENE_GAME_OVER else "postgame"
                    scene_path = recorder.record_scene(
                        after_state,
                        f"{scene_label}_game{game_index:03d}_move{move_index:04d}",
                        extra=event,
                    )
                    event["scene_capture"] = scene_path
                    recorder.append_event(event)
                    moved = True
                    game_over = True
                    print(
                        f"Game {game_index} ended on {result.scene} after {move_index} moves.",
                        flush=True,
                    )
                    break

                scene_path = None
                if after_state is not None:
                    scene_path = recorder.record_scene(
                        after_state,
                        f"unexpected_game{game_index:03d}_move{move_index:04d}",
                        extra=event,
                    )
                event["scene_capture"] = scene_path
                recorder.append_event(event)
                recorder.append_failure(
                    {
                        "game_index": game_index,
                        "move_index": move_index,
                        "direction": direction,
                        "reasons": [f"unexpected_scene: {result.scene}"],
                        "event": event,
                    }
                )
                print(f"Unexpected scene after move: {result.scene}", flush=True)
                print(f"Artifacts saved to {recorder.session_dir}", flush=True)
                return

            if game_over:
                break
            if not moved:
                failure = {
                    "game_index": game_index,
                    "move_index": move_index,
                    "reasons": ["no_direction_changed_board"],
                }
                recorder.append_failure(failure)
                print("All four directions were no-ops without reaching a game-over scene.", flush=True)
                print(f"Artifacts saved to {recorder.session_dir}", flush=True)
                return
        else:
            failure = {
                "game_index": game_index,
                "reasons": [f"max_moves_reached:{args.max_moves_per_game}"],
            }
            recorder.append_failure(failure)
            print(f"Reached move cap in game {game_index} without finishing.", flush=True)
            print(f"Artifacts saved to {recorder.session_dir}", flush=True)
            return

    print(f"Completed {args.games} games without a tracked invalid state.", flush=True)
    print(f"Artifacts saved to {recorder.session_dir}", flush=True)


if __name__ == "__main__":
    main()
