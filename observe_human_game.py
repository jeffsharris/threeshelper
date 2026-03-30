import argparse
import time
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional

import mirroring_control as mc
import window_stream as ws
from state_hunt import (
    HarnessRecorder,
    find_transition_paths,
    preview_check_from_snapshot,
    serialize_transition_step,
    valid_directions_for_transition,
)


DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="1">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Threes Human Observer</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #11111a;
      --panel: #1e1e2b;
      --border: #34364a;
      --text: #f5f5f7;
      --muted: #a4a7b5;
      --accent: #7ac7ff;
      --warn: #ffcf5c;
      --bad: #ff6e7d;
      --good: #8dd38c;
    }
    body {
      margin: 0;
      background: radial-gradient(circle at top, #222437 0%, var(--bg) 55%);
      color: var(--text);
      font-family: Menlo, Monaco, "Courier New", monospace;
    }
    main {
      max-width: 1100px;
      margin: 0 auto;
      padding: 20px;
      display: grid;
      grid-template-columns: 380px 1fr;
      gap: 20px;
    }
    .panel {
      background: rgba(30, 30, 43, 0.92);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 18px;
      box-shadow: 0 20px 50px rgba(0, 0, 0, 0.3);
    }
    h1, h2 {
      margin: 0 0 12px 0;
      font-size: 18px;
    }
    pre {
      margin: 0;
      white-space: pre-wrap;
      line-height: 1.35;
      font-size: 14px;
    }
    .meta {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }
    .ok { color: var(--good); }
    .warn { color: var(--warn); }
    .bad { color: var(--bad); }
    img {
      width: 100%;
      border-radius: 14px;
      border: 1px solid var(--border);
      background: #0c0c12;
      display: block;
    }
    .spacer {
      height: 12px;
    }
    @media (max-width: 860px) {
      main {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <main>
    <section class="panel">
      <h1>Live Tracker</h1>
      <div class="meta">Open <code>live_status.json</code> for raw state.</div>
      <div class="spacer"></div>
      <pre id="board">Waiting for status...</pre>
    </section>
    <section class="panel">
      <h2>Latest Capture</h2>
      <img id="capture" alt="Latest tracked board" src="">
      <div class="spacer"></div>
      <pre id="details"></pre>
    </section>
  </main>
  <script>
    const boardNode = document.getElementById("board");
    const detailNode = document.getElementById("details");
    const captureNode = document.getElementById("capture");

    function cssClass(state) {
      if (state.run_state === "failure") return "bad";
      if (state.recovered_missed_moves > 0) return "warn";
      if (state.run_state === "tracking") return "ok";
      return "";
    }

    async function refresh() {
      const response = await fetch("live_status.json?ts=" + Date.now(), { cache: "no-store" });
      const state = await response.json();
      boardNode.className = cssClass(state);
      boardNode.textContent = state.rendered_board || state.message || "Waiting for status...";
      const lines = [
        "state=" + (state.run_state || "?"),
        "scene=" + (state.scene || "?"),
        "game=" + (state.game_index ?? "?"),
        "move=" + (state.move_index ?? "?"),
        "message=" + (state.message || ""),
      ];
      if (state.direction_sequence && state.direction_sequence.length) {
        lines.push("moves=" + state.direction_sequence.join(" -> "));
      }
      if (state.recovered_missed_moves) {
        lines.push("recovered missed moves=" + state.recovered_missed_moves);
      }
      if (state.failure_reasons && state.failure_reasons.length) {
        lines.push("failure=" + state.failure_reasons.join("; "));
      }
      lines.push("session=" + (state.session_dir || ""));
      detailNode.textContent = lines.join("\\n");
      if (state.latest_board_overlay) {
        captureNode.src = state.latest_board_overlay + "?ts=" + Date.now();
      } else {
        captureNode.removeAttribute("src");
      }
    }

    refresh().catch((err) => {
      boardNode.textContent = "Dashboard error: " + err;
    });
    setInterval(() => refresh().catch(() => {}), 500);
  </script>
</body>
</html>
"""


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
    parser.add_argument(
        "--max-recovery-depth",
        type=int,
        default=2,
        help="Maximum move depth to search when a human move appears to skip an intermediate stable state.",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=0,
        help="Maximum number of games to observe before stopping (0 means keep running).",
    )
    parser.add_argument(
        "--dashboard-port",
        type=int,
        default=0,
        help="Local HTTP port for the live dashboard (0 picks a free port).",
    )
    parser.add_argument(
        "--no-open-dashboard",
        action="store_true",
        help="Do not auto-open the live dashboard in the default browser.",
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
    if len(matches) == 1:
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
        event["direction_sequence"] = [step.direction for step in path.steps]
        event["transition_path"] = [serialize_transition_step(step) for step in path.steps]
        event["step_count"] = len(path.steps)
        event["recovered_missed_moves"] = max(0, len(path.steps) - 1)
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


def start_dashboard_server(session_dir: Path, port: int) -> tuple[ThreadingHTTPServer, str]:
    dashboard_path = session_dir / "dashboard.html"
    dashboard_path.write_text(DASHBOARD_HTML)
    handler = partial(SimpleHTTPRequestHandler, directory=str(session_dir))
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    url = f"http://127.0.0.1:{server.server_address[1]}/dashboard.html"
    return server, url


def image_name(capture_id: Optional[int], suffix: str) -> Optional[str]:
    if capture_id is None:
        return None
    return f"{capture_id:06d}_{suffix}.png"


def status_payload(
    recorder: HarnessRecorder,
    *,
    run_state: str,
    message: str,
    scene: Optional[str],
    game_index: int,
    move_index: int,
    frame: Optional[mc.FrameState] = None,
    snapshot: Optional[tuple] = None,
    capture_id: Optional[int] = None,
    event: Optional[Dict[str, object]] = None,
    failure_reasons: Optional[List[str]] = None,
) -> Dict[str, object]:
    rendered_board = None
    board = None
    preview_label = None
    if frame is not None:
        board = frame.board
        preview_label = frame.preview_label
        if snapshot is None:
            rendered_board = ws.format_board_with_preview(frame.board, frame.preview_label)
        else:
            cycle = ws.TileCycle()
            cycle.restore(snapshot)
            cycle.set_max_tile(ws.board_max_tile(frame.board))
            rendered_board = ws.render_move_table(frame.board, frame.preview_label, cycle)

    latest_capture = capture_id
    if latest_capture is None:
        latest_capture = recorder.dataset.last_capture_id

    payload = {
        "run_state": run_state,
        "message": message,
        "scene": scene,
        "game_index": game_index,
        "move_index": move_index,
        "board": board,
        "preview_label": preview_label,
        "rendered_board": rendered_board,
        "session_dir": str(recorder.session_dir),
        "latest_capture_id": latest_capture,
        "latest_full": image_name(latest_capture, "full"),
        "latest_board_overlay": image_name(latest_capture, "board_overlay"),
        "latest_board": image_name(latest_capture, "board"),
        "latest_preview": image_name(latest_capture, "preview"),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "failure_reasons": failure_reasons or [],
        "direction_sequence": event.get("direction_sequence", []) if event else [],
        "recovered_missed_moves": event.get("recovered_missed_moves", 0) if event else 0,
        "event": event,
    }
    return payload


def main() -> None:
    args = parse_args()
    window_id, window_info = mc.resolve_window(args.window_id, args.auto_window_prefix)
    recorder = HarnessRecorder(args.dataset_dir, window_info=window_info)
    dashboard_server, dashboard_url = start_dashboard_server(recorder.session_dir, args.dashboard_port)
    try:
        print(f"Using window {window_id}", flush=True)
        print(f"Recording run to {recorder.session_dir}", flush=True)
        print(f"Live dashboard: {dashboard_url}", flush=True)
        if not args.no_open_dashboard:
            webbrowser.open(dashboard_url)
        recorder.write_status(
            status_payload(
                recorder,
                run_state="starting",
                message="Observer started. Waiting for the mirrored game window.",
                scene=None,
                game_index=1,
                move_index=0,
            )
        )

        tracked = False
        game_index = 1
        move_index = 0
        last_move_ts = time.time()
        last_stable_frame: Optional[mc.FrameState] = None
        last_snapshot: Optional[tuple] = None
        last_capture_id: Optional[int] = None
        stable_state: Optional[mc.FrameState] = None
        stable_count = 0
        last_status_key: Optional[tuple] = None

        while True:
            state = mc.capture_screen_state(window_id, args.capture_backend)

            status_key = (
                state.scene,
                tracked,
                move_index,
                game_index,
                last_capture_id,
            )
            if state.scene in (mc.SCENE_SCREEN_OFF, mc.SCENE_PHONE_IN_USE):
                if status_key != last_status_key:
                    recorder.write_status(
                        status_payload(
                            recorder,
                            run_state="waiting_for_device",
                            message=f"Waiting for mirrored device to become ready: {state.scene}",
                            scene=state.scene,
                            game_index=game_index,
                            move_index=move_index,
                            capture_id=last_capture_id,
                        )
                    )
                    last_status_key = status_key
                stable_state = None
                stable_count = 0
                time.sleep(args.poll)
                continue

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
                recorder.write_status(
                    status_payload(
                        recorder,
                        run_state="game_end",
                        message=f"Observed {state.scene} after {move_index} moves. Waiting for the next game.",
                        scene=state.scene,
                        game_index=game_index,
                        move_index=move_index,
                        capture_id=last_capture_id,
                    )
                )
                print(f"Observed {state.scene} after {move_index} moves.", flush=True)
                if args.max_games > 0 and game_index >= args.max_games:
                    print(f"Reached max observed games ({args.max_games}).", flush=True)
                    print(f"Artifacts saved to {recorder.session_dir}", flush=True)
                    return
                tracked = False
                game_index += 1
                move_index = 0
                last_stable_frame = None
                last_snapshot = None
                last_capture_id = None
                stable_state = None
                stable_count = 0
                last_move_ts = time.time()
                time.sleep(args.poll)
                continue

            if state.scene != mc.SCENE_GAME or state.frame is None:
                if status_key != last_status_key:
                    recorder.write_status(
                        status_payload(
                            recorder,
                            run_state="waiting_for_game",
                            message=f"Waiting for a game board. Current scene: {state.scene}.",
                            scene=state.scene,
                            game_index=game_index,
                            move_index=move_index,
                            capture_id=last_capture_id,
                        )
                    )
                    last_status_key = status_key
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
                    recorder.write_status(
                        status_payload(
                            recorder,
                            run_state="waiting_for_fresh_game",
                            message=f"Waiting for a fresh board: {initial_error}",
                            scene=state.scene,
                            game_index=game_index,
                            move_index=move_index,
                            frame=settled,
                            capture_id=last_capture_id,
                        )
                    )
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
                recorder.write_status(
                    status_payload(
                        recorder,
                        run_state="tracking",
                        message="Tracking live game state.",
                        scene=state.scene,
                        game_index=game_index,
                        move_index=move_index,
                        frame=settled,
                        snapshot=last_snapshot,
                        capture_id=last_capture_id,
                    )
                )
                last_status_key = None
                time.sleep(args.poll)
                continue

            if last_stable_frame is None or same_semantics(last_stable_frame, settled):
                if args.idle_timeout > 0 and time.time() - last_move_ts > args.idle_timeout:
                    recorder.write_status(
                        status_payload(
                            recorder,
                            run_state="idle_timeout",
                            message=f"Idle timeout reached with no observed move for {args.idle_timeout:.1f}s.",
                            scene=state.scene,
                            game_index=game_index,
                            move_index=move_index,
                            frame=last_stable_frame,
                            snapshot=last_snapshot,
                            capture_id=last_capture_id,
                        )
                    )
                    print(f"Idle timeout reached with no observed move for {args.idle_timeout:.1f}s.", flush=True)
                    print(f"Artifacts saved to {recorder.session_dir}", flush=True)
                    return
                time.sleep(args.poll)
                continue

            capture_id = recorder.record_game_state(settled, window_id, time.time())
            move_index_start = move_index + 1
            event = build_move_event(
                last_stable_frame,
                last_snapshot,
                settled,
                game_index=game_index,
                move_index_start=move_index_start,
                capture_id=capture_id,
                max_recovery_depth=args.max_recovery_depth,
            )
            move_index += int(event.get("step_count", 1))
            event["move_index"] = move_index
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
                recorder.write_status(
                    status_payload(
                        recorder,
                        run_state="failure",
                        message="Invalid tracked state detected.",
                        scene=state.scene,
                        game_index=game_index,
                        move_index=move_index,
                        frame=settled,
                        snapshot=last_snapshot,
                        capture_id=capture_id,
                        event=event,
                        failure_reasons=failure_reasons,
                    )
                )
                print(f"Invalid state detected: {failure_reasons}", flush=True)
                print(f"Artifacts saved to {recorder.session_dir}", flush=True)
                return

            next_snapshot = event["preview_check"].get("next_snapshot")
            last_snapshot = next_snapshot if isinstance(next_snapshot, tuple) else next_snapshot
            last_stable_frame = settled
            last_capture_id = capture_id
            last_move_ts = time.time()

            directions = event.get("direction_sequence", [])
            direction_text = " -> ".join(directions) if directions else event.get("direction") or "?"
            if event.get("recovered_missed_moves", 0) > 0:
                print(
                    f"observed moves {move_index_start}-{move_index}: {direction_text} "
                    f"(recovered {event['recovered_missed_moves']} skipped state)",
                    flush=True,
                )
            else:
                print(f"observed move {move_index}: {direction_text}", flush=True)
            print_frame(settled, last_snapshot)
            print(flush=True)
            recorder.write_status(
                status_payload(
                    recorder,
                    run_state="tracking",
                    message="Tracking live game state.",
                    scene=state.scene,
                    game_index=game_index,
                    move_index=move_index,
                    frame=settled,
                    snapshot=last_snapshot,
                    capture_id=capture_id,
                    event=event,
                )
            )
            last_status_key = None

            if move_index >= args.max_moves:
                recorder.write_status(
                    status_payload(
                        recorder,
                        run_state="max_moves",
                        message=f"Reached max observed moves ({args.max_moves}).",
                        scene=state.scene,
                        game_index=game_index,
                        move_index=move_index,
                        frame=settled,
                        snapshot=last_snapshot,
                        capture_id=capture_id,
                    )
                )
                print(f"Reached max observed moves ({args.max_moves}).", flush=True)
                print(f"Artifacts saved to {recorder.session_dir}", flush=True)
                return

            time.sleep(args.poll)
    finally:
        dashboard_server.shutdown()
        dashboard_server.server_close()


if __name__ == "__main__":
    main()
