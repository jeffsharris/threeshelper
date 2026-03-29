import argparse
import random
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

import Quartz

import window_stream as ws


APP_BUNDLE_ID = "com.apple.ScreenContinuity"
DIRECTIONS = ("up", "down", "left", "right")


@dataclass(frozen=True)
class WindowBounds:
    x: float
    y: float
    width: float
    height: float

    def point(self, rel_x: float, rel_y: float) -> Tuple[float, float]:
        return (self.x + self.width * rel_x, self.y + self.height * rel_y)


@dataclass
class FrameState:
    arr: np.ndarray
    board: List[List[str]]
    preview_label: str
    preview_debug: Dict[str, object]
    board_sig: np.ndarray


def activate_mirroring() -> None:
    subprocess.run(
        ["osascript", "-e", f'tell application id "{APP_BUNDLE_ID}" to activate'],
        check=True,
    )


def get_window_bounds(window_id: int) -> WindowBounds:
    info = Quartz.CGWindowListCopyWindowInfo(
        Quartz.kCGWindowListOptionIncludingWindow,
        window_id,
    )
    if not info:
        raise RuntimeError(f"Could not find window {window_id}")
    bounds = info[0].get("kCGWindowBounds")
    if not bounds:
        raise RuntimeError(f"Window {window_id} has no bounds")
    return WindowBounds(
        x=float(bounds["X"]),
        y=float(bounds["Y"]),
        width=float(bounds["Width"]),
        height=float(bounds["Height"]),
    )


def capture_window_image(window_id: int, backend: str) -> Image.Image:
    if backend == "quartz":
        image = Quartz.CGWindowListCreateImage(
            Quartz.CGRectNull,
            Quartz.kCGWindowListOptionIncludingWindow,
            window_id,
            Quartz.kCGWindowImageBoundsIgnoreFraming,
        )
        if image is None:
            raise RuntimeError("CGWindowListCreateImage returned None")
        width = Quartz.CGImageGetWidth(image)
        height = Quartz.CGImageGetHeight(image)
        bytes_per_row = Quartz.CGImageGetBytesPerRow(image)
        data = Quartz.CGDataProviderCopyData(Quartz.CGImageGetDataProvider(image))
        buf = bytes(data)
        return Image.frombuffer(
            "RGBA",
            (width, height),
            buf,
            "raw",
            "BGRA",
            bytes_per_row,
            1,
        ).convert("RGB")

    fd, path = tempfile.mkstemp(suffix=".png")
    try:
        subprocess.run(["screencapture", "-x", "-l", str(window_id), path], check=True)
        return Image.open(path).convert("RGB")
    finally:
        try:
            Path(path).unlink()
        except OSError:
            pass


def capture_frame(window_id: int, backend: str) -> FrameState:
    arr = np.array(capture_window_image(window_id, backend))
    board = ws.classify_board(arr)
    preview_label, preview_debug = ws.classify_array(arr)
    board_sig, _ = ws.board_signature(arr)
    return FrameState(
        arr=arr,
        board=board,
        preview_label=preview_label,
        preview_debug=preview_debug,
        board_sig=board_sig,
    )


def _post_mouse(event_type: int, point: Tuple[float, float]) -> None:
    event = Quartz.CGEventCreateMouseEvent(
        None,
        event_type,
        point,
        Quartz.kCGMouseButtonLeft,
    )
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)


def tap_window(
    window_id: int,
    rel_x: float,
    rel_y: float,
    focus_delay: float,
) -> None:
    activate_mirroring()
    time.sleep(focus_delay)
    bounds = get_window_bounds(window_id)
    point = bounds.point(rel_x, rel_y)
    _post_mouse(Quartz.kCGEventMouseMoved, point)
    time.sleep(0.01)
    _post_mouse(Quartz.kCGEventLeftMouseDown, point)
    time.sleep(0.02)
    _post_mouse(Quartz.kCGEventLeftMouseUp, point)


def drag_window(
    window_id: int,
    direction: str,
    span_ratio: float,
    duration: float,
    steps: int,
    start_rel_x: float,
    start_rel_y: float,
    focus_delay: float,
) -> None:
    if direction not in DIRECTIONS:
        raise ValueError(f"Unsupported direction: {direction}")
    activate_mirroring()
    time.sleep(focus_delay)
    bounds = get_window_bounds(window_id)
    start = bounds.point(start_rel_x, start_rel_y)

    rel_dx = 0.0
    rel_dy = 0.0
    if direction == "left":
        rel_dx = -span_ratio
    elif direction == "right":
        rel_dx = span_ratio
    elif direction == "up":
        rel_dy = -span_ratio
    elif direction == "down":
        rel_dy = span_ratio

    end = bounds.point(start_rel_x + rel_dx, start_rel_y + rel_dy)

    _post_mouse(Quartz.kCGEventMouseMoved, start)
    time.sleep(0.01)
    _post_mouse(Quartz.kCGEventLeftMouseDown, start)
    drag_delay = max(duration / max(1, steps), 0.001)
    for idx in range(1, max(steps, 1) + 1):
        t = idx / max(steps, 1)
        point = (
            start[0] + (end[0] - start[0]) * t,
            start[1] + (end[1] - start[1]) * t,
        )
        _post_mouse(Quartz.kCGEventLeftMouseDragged, point)
        time.sleep(drag_delay)
    _post_mouse(Quartz.kCGEventLeftMouseUp, end)


class AutoPlayer:
    def __init__(
        self,
        window_id: int,
        backend: str,
        settle_frames: int,
        settle_poll: float,
        settle_threshold: float,
        settle_timeout: float,
        board_delta_threshold: float,
        swipe_span_ratio: float,
        swipe_duration: float,
        swipe_steps: int,
        start_rel_x: float,
        start_rel_y: float,
        focus_delay: float,
        recorder: Optional[ws.DatasetRecorder] = None,
    ) -> None:
        self.window_id = window_id
        self.backend = backend
        self.settle_frames = settle_frames
        self.settle_poll = settle_poll
        self.settle_threshold = settle_threshold
        self.settle_timeout = settle_timeout
        self.board_delta_threshold = board_delta_threshold
        self.swipe_span_ratio = swipe_span_ratio
        self.swipe_duration = swipe_duration
        self.swipe_steps = swipe_steps
        self.start_rel_x = start_rel_x
        self.start_rel_y = start_rel_y
        self.focus_delay = focus_delay
        self.recorder = recorder

        self.tile_cycle: Optional[ws.TileCycle] = None
        self.current_frame: Optional[FrameState] = None

    def _render_state(self, frame: FrameState) -> str:
        if self.tile_cycle is None:
            return ws.format_board_with_preview(frame.board, frame.preview_label)
        return ws.render_move_table(frame.board, frame.preview_label, self.tile_cycle)

    def initialize(self) -> FrameState:
        frame = capture_frame(self.window_id, self.backend)
        err = ws._initial_state_error(frame.board, frame.preview_label)
        if err is None:
            self.tile_cycle = ws.TileCycle()
            ws.seed_tile_cycle_from_initial_state(self.tile_cycle, frame.board, frame.preview_label)
            ok, reason = ws.preview_possible(self.tile_cycle, frame.preview_label)
            if not ok:
                ws.print_error(f"preview '{frame.preview_label}' not possible at init: {reason}")
            self.tile_cycle.update(frame.preview_label)
        else:
            self.tile_cycle = None
            print(f"tracking disabled: {err}")
        self.current_frame = frame
        print(self._render_state(frame))
        print()
        return frame

    def _capture_after_settle(self) -> FrameState:
        frame = capture_frame(self.window_id, self.backend)
        if self.settle_frames <= 0:
            return frame
        prev_sig = frame.board_sig
        last_frame = frame
        stable = 0
        start = time.time()
        while time.time() - start < self.settle_timeout:
            time.sleep(self.settle_poll)
            current = capture_frame(self.window_id, self.backend)
            diff = ws.board_signature_diff(prev_sig, current.board_sig)
            prev_sig = current.board_sig
            last_frame = current
            if diff < self.settle_threshold:
                stable += 1
            else:
                stable = 0
            if stable >= self.settle_frames:
                return last_frame
        return last_frame

    def _record(self, frame: FrameState, ts_event: float) -> None:
        if not self.recorder:
            return
        self.recorder.record_capture(
            frame.arr,
            frame.board,
            frame.preview_label,
            frame.preview_debug,
            self.window_id,
            ts_event,
        )

    def attempt_move(self, direction: str) -> bool:
        if self.current_frame is None:
            raise RuntimeError("AutoPlayer must be initialized before moves")

        before = self.current_frame
        drag_window(
            self.window_id,
            direction,
            span_ratio=self.swipe_span_ratio,
            duration=self.swipe_duration,
            steps=self.swipe_steps,
            start_rel_x=self.start_rel_x,
            start_rel_y=self.start_rel_y,
            focus_delay=self.focus_delay,
        )
        after = self._capture_after_settle()
        diff = ws.board_signature_diff(before.board_sig, after.board_sig)
        ts_event = time.time()
        ts = time.strftime("%H:%M:%S", time.localtime(ts_event))

        if diff <= self.board_delta_threshold:
            print(f"[{ts}] swipe {direction}: no board change (boardΔ={diff:.3f})")
            return False

        self.current_frame = after
        if ws._board_has_unknowns(after.board) or after.preview_label == "unknown":
            ws.print_error("state contains unknown cells or preview label")
        if self.tile_cycle is not None:
            ok, reason = ws.preview_possible(self.tile_cycle, after.preview_label)
            if not ok:
                ws.print_error(f"preview '{after.preview_label}' not possible: {reason}")
            self.tile_cycle.update(after.preview_label)
        self._record(after, ts_event)
        print(f"[{ts}] swipe {direction} (boardΔ={diff:.3f})")
        print(self._render_state(after))
        print()
        return True

    def autoplay(
        self,
        max_moves: int,
        rng: random.Random,
        order_mode: str,
    ) -> int:
        moves_made = 0
        rotation = 0
        while moves_made < max_moves:
            directions = list(DIRECTIONS)
            if order_mode == "random":
                rng.shuffle(directions)
            elif order_mode == "cycle":
                directions = directions[rotation:] + directions[:rotation]
                rotation = (rotation + 1) % len(directions)

            moved = False
            for direction in directions:
                if self.attempt_move(direction):
                    moves_made += 1
                    moved = True
                    break

            if not moved:
                print("No swipe direction changed the board. Stopping autoplay.")
                break
        return moves_made


def resolve_window(window_id: Optional[int], auto_window_prefix: str) -> Tuple[int, Dict[str, object]]:
    if window_id is not None:
        return window_id, {"id": window_id, "app": "(provided id)", "title": ""}

    auto_pick = ws.find_window_by_prefix(auto_window_prefix or "")
    if auto_pick:
        wid, app_name, win_name = auto_pick
        return wid, {"id": wid, "app": app_name, "title": win_name}

    wid, app_name, win_name = ws.choose_window_interactive()
    return wid, {"id": wid, "app": app_name, "title": win_name}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Drive the iPhone Mirroring window with swipes and optional autoplay."
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
        "--tap-rel",
        nargs=2,
        type=float,
        metavar=("X", "Y"),
        help="Send one tap at relative window coordinates before exiting.",
    )
    parser.add_argument(
        "--swipe",
        choices=DIRECTIONS,
        help="Send one swipe gesture and print the resulting state.",
    )
    parser.add_argument(
        "--autoplay",
        action="store_true",
        help="Keep swiping until no move changes the board or --max-moves is reached.",
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=200,
        help="Maximum successful moves to make during autoplay.",
    )
    parser.add_argument(
        "--move-order",
        choices=("random", "cycle"),
        default="random",
        help="Direction ordering strategy during autoplay.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used when --move-order random is selected.",
    )
    parser.add_argument(
        "--board-delta-threshold",
        type=float,
        default=0.3,
        help="Minimum board signature delta required to treat a swipe as a real move.",
    )
    parser.add_argument(
        "--settle-frames",
        type=int,
        default=2,
        help="Number of stable board samples required after a swipe.",
    )
    parser.add_argument(
        "--settle-poll",
        type=float,
        default=0.1,
        help="Polling interval while waiting for the board to settle.",
    )
    parser.add_argument(
        "--settle-threshold",
        type=float,
        default=0.15,
        help="Board signature delta threshold used to count a frame as stable.",
    )
    parser.add_argument(
        "--settle-timeout",
        type=float,
        default=1.5,
        help="Maximum seconds to wait for the board to settle after a swipe.",
    )
    parser.add_argument(
        "--swipe-span-ratio",
        type=float,
        default=0.22,
        help="Swipe distance as a fraction of the window width/height.",
    )
    parser.add_argument(
        "--swipe-duration",
        type=float,
        default=0.12,
        help="Total time to spend dragging during a swipe gesture.",
    )
    parser.add_argument(
        "--swipe-steps",
        type=int,
        default=12,
        help="Number of drag interpolation steps per swipe.",
    )
    parser.add_argument(
        "--start-rel-x",
        type=float,
        default=0.5,
        help="Relative x position inside the mirrored window where swipes begin.",
    )
    parser.add_argument(
        "--start-rel-y",
        type=float,
        default=0.55,
        help="Relative y position inside the mirrored window where swipes begin.",
    )
    parser.add_argument(
        "--focus-delay",
        type=float,
        default=0.2,
        help="Delay after activating iPhone Mirroring before sending the gesture.",
    )
    parser.add_argument(
        "--record-dataset",
        type=str,
        help="Optional dataset directory for saving successful move captures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    window_id, window_info = resolve_window(args.window_id, args.auto_window_prefix)
    print(
        f"Using window {window_id} ({window_info.get('app', '')} - {window_info.get('title', '')})"
    )

    recorder = None
    if args.record_dataset:
        recorder = ws.DatasetRecorder(Path(args.record_dataset), window_info=window_info)
        print(f"Recording successful moves to {recorder.session_dir}")

    player = AutoPlayer(
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

    if args.tap_rel:
        tap_window(window_id, args.tap_rel[0], args.tap_rel[1], args.focus_delay)
        return

    player.initialize()
    ws.print_legend()

    if args.swipe:
        player.attempt_move(args.swipe)
        return

    if args.autoplay:
        rng = random.Random(args.seed)
        moves = player.autoplay(args.max_moves, rng=rng, order_mode=args.move_order)
        print(f"Autoplay finished after {moves} successful moves.")


if __name__ == "__main__":
    main()
