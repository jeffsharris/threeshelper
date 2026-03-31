#!/usr/bin/env python3

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.error import URLError
from urllib.request import urlopen

import mirroring_control as mc
import state_hunt as sh
from tracker_runtime import build_move_event, seed_snapshot


DEFAULT_DIRECTION_ORDER = ("left", "up", "right", "down")


def fetch_state(base_url: str, timeout: float) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/api/state?ts={time.time_ns()}"
    with urlopen(url, timeout=timeout) as response:
        return json.load(response)


def visible_board(state: Dict[str, Any]) -> Optional[List[List[str]]]:
    detected_board = state.get("detected", {}).get("board")
    if detected_board:
        return detected_board
    tracker_board = state.get("tracker", {}).get("board")
    if tracker_board:
        return tracker_board
    return None


def tracked_board(state: Dict[str, Any]) -> Optional[List[List[str]]]:
    tracker_board = state.get("tracker", {}).get("board")
    if tracker_board:
        return tracker_board
    return state.get("detected", {}).get("board")


def choose_direction(
    board: List[List[str]],
    preview_label: str,
    preferred_order: List[str],
) -> str:
    steps = sh.generate_transition_steps(board, preview_label)
    legal_directions: List[str] = []
    for step in steps:
        if step.direction not in legal_directions:
            legal_directions.append(step.direction)
    if not legal_directions:
        raise RuntimeError("No legal swipe direction is available for the current board.")
    for direction in preferred_order:
        if direction in legal_directions:
            return direction
    return legal_directions[0]


def wait_for_game_state(base_url: str, timeout: float, poll: float) -> Dict[str, Any]:
    deadline = time.time() + timeout
    last_error: Optional[str] = None
    while time.time() < deadline:
        try:
            state = fetch_state(base_url, timeout=max(1.0, poll * 4))
        except URLError as exc:
            last_error = str(exc)
            time.sleep(poll)
            continue
        tracker = state.get("tracker", {})
        if state.get("scene") == "game" and visible_board(state) and tracker.get("run_state") != "failure":
            return state
        last_error = tracker.get("message") or state.get("scene") or "unknown"
        time.sleep(poll)
    raise RuntimeError(f"Timed out waiting for a game board: {last_error or 'unknown error'}")


def wait_for_settled_game_state(
    base_url: str,
    timeout: float,
    poll: float,
    stable_polls: int = 4,
) -> Dict[str, Any]:
    deadline = time.time() + timeout
    stable = 0
    last_signature: Optional[str] = None
    last_state: Optional[Dict[str, Any]] = None
    while time.time() < deadline:
        state = wait_for_game_state(base_url, timeout=max(poll * 4, 1.0), poll=poll)
        tracker = state.get("tracker", {})
        board = visible_board(state)
        preview = (
            state.get("detected", {}).get("preview_label")
            or tracker.get("observed_preview", {}).get("label")
        )
        signature = json.dumps({"board": board, "preview": preview}, sort_keys=True)
        if tracker.get("run_state") == "settling":
            stable = 0
            last_signature = signature
            last_state = state
            time.sleep(poll)
            continue
        if signature == last_signature:
            stable += 1
        else:
            stable = 1
            last_signature = signature
        last_state = state
        if stable >= stable_polls:
            return state
        time.sleep(poll)
    raise RuntimeError("Timed out waiting for the board to settle before benchmarking.")


def percentile(values: List[float], fraction: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction))))
    return ordered[idx]


def event_failure_reasons(event: Dict[str, Any]) -> List[str]:
    reasons: List[str] = []
    transition_check = event.get("transition_check") or {}
    preview_check = event.get("preview_check") or {}
    if not preview_check.get("valid", True):
        reasons.append(f"preview_invalid: {preview_check.get('reason', '')}")
    if not transition_check.get("valid", True):
        reasons.append(f"transition_invalid: {transition_check.get('reason', '')}")
    if event.get("unknown_board"):
        reasons.append("board_contains_unknowns")
    if event.get("unknown_preview"):
        reasons.append("preview_unknown")
    return reasons


def capture_direct_frame(window_id: int, backend: str) -> mc.FrameState:
    return mc.capture_frame(window_id, backend)


def wait_for_settled_direct_frame(
    *,
    window_id: int,
    capture_backend: str,
    timeout: float,
    poll: float,
    stable_polls: int = 4,
) -> mc.FrameState:
    deadline = time.time() + timeout
    stable = 0
    last_signature: Optional[str] = None
    last_frame: Optional[mc.FrameState] = None
    while time.time() < deadline:
        frame = capture_direct_frame(window_id, capture_backend)
        signature = json.dumps({"board": frame.board, "preview": frame.preview_label}, sort_keys=True)
        if signature == last_signature:
            stable += 1
        else:
            stable = 1
            last_signature = signature
            last_frame = frame
        if stable >= stable_polls and last_frame is not None:
            return last_frame
        time.sleep(poll)
    raise RuntimeError("Timed out waiting for a stable direct frame before benchmarking.")


def benchmark_once(
    *,
    base_url: str,
    window_id: int,
    direction: Optional[str],
    direction_order: List[str],
    focus_delay: float,
    span_ratio: float,
    duration: float,
    steps: int,
    start_rel_x: float,
    start_rel_y: float,
    timeout: float,
    poll: float,
    input_method: str,
) -> Dict[str, Any]:
    before = wait_for_settled_game_state(base_url, timeout=timeout, poll=poll)
    before_visible_board = visible_board(before)
    before_tracked_board = tracked_board(before)
    assert before_visible_board is not None
    tracker = before.get("tracker", {})
    chosen_direction = direction
    if chosen_direction is None:
        direction_backend = str(before.get("backend") or "screen")
        if direction_backend != "screen":
            try:
                live_frame = capture_direct_frame(window_id, direction_backend)
                chosen_direction = choose_direction(
                    live_frame.board,
                    live_frame.preview_label,
                    direction_order,
                )
            except Exception:
                chosen_direction = None
        if chosen_direction is None:
            chosen_direction = choose_direction(
                before_visible_board,
                str(
                    before.get("detected", {}).get("preview_label")
                    or tracker.get("observed_preview", {}).get("label")
                    or "unknown"
                ),
                direction_order,
            )
    move_before = int(tracker.get("move_index") or 0)
    revision_before = int(before.get("revision") or 0)
    run_state_before = str(tracker.get("run_state") or "")
    t0 = time.perf_counter()
    if input_method == "arrow":
        mc.press_direction_key(window_id, chosen_direction, focus_delay=focus_delay)
    else:
        mc.drag_window(
            window_id,
            chosen_direction,
            span_ratio=span_ratio,
            duration=duration,
            steps=steps,
            start_rel_x=start_rel_x,
            start_rel_y=start_rel_y,
            focus_delay=focus_delay,
        )
    gesture_finished_ms = (time.perf_counter() - t0) * 1000.0

    first_revision_ms: Optional[float] = None
    first_visible_board_change_ms: Optional[float] = None
    first_tracked_board_change_ms: Optional[float] = None
    first_commit_ms: Optional[float] = None
    fast_confirmed = False
    final_state: Optional[Dict[str, Any]] = None
    while (time.perf_counter() - t0) < timeout:
        state = fetch_state(base_url, timeout=max(1.0, poll * 4))
        final_state = state
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        revision_now = int(state.get("revision") or 0)
        tracker_now = state.get("tracker", {})
        visible_board_now = visible_board(state)
        tracked_board_now = tracked_board(state)
        if first_revision_ms is None and revision_now != revision_before:
            first_revision_ms = elapsed_ms
        if first_visible_board_change_ms is None and visible_board_now != before_visible_board:
            first_visible_board_change_ms = elapsed_ms
        if first_tracked_board_change_ms is None and tracked_board_now != before_tracked_board:
            first_tracked_board_change_ms = elapsed_ms
        if first_commit_ms is None and int(tracker_now.get("move_index") or 0) != move_before:
            first_commit_ms = elapsed_ms
            fast_confirmed = bool((tracker_now.get("latest_event") or {}).get("fast_confirmed"))
        if first_commit_ms is not None and tracker_now.get("run_state") == "tracking":
            break
        time.sleep(poll)

    if final_state is None:
        raise RuntimeError("No state samples were collected after the gesture.")

    tracker_final = final_state.get("tracker", {})
    return {
        "direction": chosen_direction,
        "input_method": input_method,
        "move_before": move_before,
        "move_after": int(tracker_final.get("move_index") or 0),
        "run_state_before": run_state_before,
        "run_state_after": tracker_final.get("run_state"),
        "gesture_finished_ms": round(gesture_finished_ms, 1),
        "first_revision_ms": round(first_revision_ms, 1) if first_revision_ms is not None else None,
        "first_visible_board_change_ms": round(first_visible_board_change_ms, 1)
        if first_visible_board_change_ms is not None
        else None,
        "first_tracked_board_change_ms": round(first_tracked_board_change_ms, 1)
        if first_tracked_board_change_ms is not None
        else None,
        "first_commit_ms": round(first_commit_ms, 1) if first_commit_ms is not None else None,
        "fast_confirmed": fast_confirmed,
        "final_capture_elapsed_ms": final_state.get("capture_elapsed_ms"),
        "final_poll_target_ms": final_state.get("poll_target_ms"),
        "final_scene": final_state.get("scene"),
    }


def benchmark_once_direct(
    *,
    window_id: int,
    capture_backend: str,
    direction: Optional[str],
    direction_order: List[str],
    focus_delay: float,
    span_ratio: float,
    duration: float,
    steps: int,
    start_rel_x: float,
    start_rel_y: float,
    timeout: float,
    poll: float,
    input_method: str,
) -> Dict[str, Any]:
    before_frame = wait_for_settled_direct_frame(
        window_id=window_id,
        capture_backend=capture_backend,
        timeout=timeout,
        poll=poll,
    )
    chosen_direction = direction or choose_direction(
        before_frame.board,
        before_frame.preview_label,
        direction_order,
    )
    before_snapshot = seed_snapshot(before_frame)
    t0 = time.perf_counter()
    if input_method == "arrow":
        mc.press_direction_key(window_id, chosen_direction, focus_delay=focus_delay)
    else:
        mc.drag_window(
            window_id,
            chosen_direction,
            span_ratio=span_ratio,
            duration=duration,
            steps=steps,
            start_rel_x=start_rel_x,
            start_rel_y=start_rel_y,
            focus_delay=focus_delay,
        )
    gesture_finished_ms = (time.perf_counter() - t0) * 1000.0

    first_board_change_ms: Optional[float] = None
    first_legal_commit_ms: Optional[float] = None
    fast_confirmed = False
    final_frame = before_frame
    while (time.perf_counter() - t0) < timeout:
        frame = capture_direct_frame(window_id, capture_backend)
        final_frame = frame
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if first_board_change_ms is None and (
            frame.board != before_frame.board or frame.preview_label != before_frame.preview_label
        ):
            first_board_change_ms = elapsed_ms
        event = build_move_event(
            before_frame,
            before_snapshot,
            frame,
            game_index=0,
            move_index_start=1,
            capture_id=0,
            max_recovery_depth=2,
        )
        if first_legal_commit_ms is None and not event_failure_reasons(event):
            first_legal_commit_ms = elapsed_ms
            fast_confirmed = bool(event.get("fast_confirmed"))
            break
        time.sleep(poll)

    return {
        "capture_backend": capture_backend,
        "direction": chosen_direction,
        "input_method": input_method,
        "gesture_finished_ms": round(gesture_finished_ms, 1),
        "first_visible_board_change_ms": round(first_board_change_ms, 1)
        if first_board_change_ms is not None
        else None,
        "first_tracked_board_change_ms": round(first_legal_commit_ms, 1)
        if first_legal_commit_ms is not None
        else None,
        "first_commit_ms": round(first_legal_commit_ms, 1)
        if first_legal_commit_ms is not None
        else None,
        "fast_confirmed": fast_confirmed,
        "final_scene": "game",
        "final_preview": final_frame.preview_label,
    }


def summarize(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    def collect(key: str) -> List[float]:
        out: List[float] = []
        for sample in samples:
            value = sample.get(key)
            if isinstance(value, (int, float)):
                out.append(float(value))
        return out

    visible_board_change = collect("first_visible_board_change_ms")
    tracked_board_change = collect("first_tracked_board_change_ms")
    commit = collect("first_commit_ms")
    return {
        "samples": len(samples),
        "visible_board_change_ms": {
            "mean": round(statistics.mean(visible_board_change), 1) if visible_board_change else None,
            "p95": round(percentile(visible_board_change, 0.95), 1) if visible_board_change else None,
            "max": round(max(visible_board_change), 1) if visible_board_change else None,
        },
        "tracked_board_change_ms": {
            "mean": round(statistics.mean(tracked_board_change), 1) if tracked_board_change else None,
            "p95": round(percentile(tracked_board_change, 0.95), 1) if tracked_board_change else None,
            "max": round(max(tracked_board_change), 1) if tracked_board_change else None,
        },
        "commit_ms": {
            "mean": round(statistics.mean(commit), 1) if commit else None,
            "p95": round(percentile(commit, 0.95), 1) if commit else None,
            "max": round(max(commit), 1) if commit else None,
        },
        "fast_confirm_rate": round(
            sum(1 for sample in samples if sample.get("fast_confirmed")) / len(samples),
            3,
        )
        if samples
        else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark live Threes debug latency against the running dashboard.")
    parser.add_argument("--server-url", default="http://127.0.0.1:55777")
    parser.add_argument("--mode", choices=("server", "direct"), default="server")
    parser.add_argument("--window-prefix", default="iPhone Mirroring")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--direction", choices=mc.DIRECTIONS)
    parser.add_argument("--input-method", choices=("arrow", "drag"), default="arrow")
    parser.add_argument("--capture-backend", choices=("screen", "quartz", "screencapture"), default="screen")
    parser.add_argument("--direction-order", default="left,up,right,down")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--poll", type=float, default=0.02)
    parser.add_argument("--pause", type=float, default=0.2)
    parser.add_argument("--focus-delay", type=float, default=0.08)
    parser.add_argument("--swipe-span-ratio", type=float, default=0.22)
    parser.add_argument("--swipe-duration", type=float, default=0.12)
    parser.add_argument("--swipe-steps", type=int, default=12)
    parser.add_argument("--start-rel-x", type=float, default=0.5)
    parser.add_argument("--start-rel-y", type=float, default=0.72)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    direction_order = [item.strip() for item in args.direction_order.split(",") if item.strip()]
    window_id, _window_info = mc.resolve_window(None, args.window_prefix)
    samples: List[Dict[str, Any]] = []
    for index in range(args.samples):
        if args.mode == "direct":
            result = benchmark_once_direct(
                window_id=window_id,
                capture_backend=args.capture_backend,
                direction=args.direction,
                direction_order=direction_order,
                focus_delay=args.focus_delay,
                span_ratio=args.swipe_span_ratio,
                duration=args.swipe_duration,
                steps=args.swipe_steps,
                start_rel_x=args.start_rel_x,
                start_rel_y=args.start_rel_y,
                timeout=args.timeout,
                poll=args.poll,
                input_method=args.input_method,
            )
        else:
            result = benchmark_once(
                base_url=args.server_url,
                window_id=window_id,
                direction=args.direction,
                direction_order=direction_order,
                focus_delay=args.focus_delay,
                span_ratio=args.swipe_span_ratio,
                duration=args.swipe_duration,
                steps=args.swipe_steps,
                start_rel_x=args.start_rel_x,
                start_rel_y=args.start_rel_y,
                timeout=args.timeout,
                poll=args.poll,
                input_method=args.input_method,
            )
        result["sample_index"] = index + 1
        samples.append(result)
        print(json.dumps(result, indent=2))
        if index + 1 < args.samples:
            time.sleep(args.pause)

    summary = summarize(samples)
    output = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "mode": args.mode,
        "server_url": args.server_url,
        "window_id": window_id,
        "samples": samples,
        "summary": summary,
    }
    print(json.dumps({"summary": summary}, indent=2))

    out_path = args.out
    if out_path is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_path = Path("output/live_debug") / f"latency_benchmark_{stamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
