import argparse
import io
import json
import os
import time
import traceback
import webbrowser
from collections import deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Deque, Dict, List, Optional

import mirroring_control as mc
import window_stream as ws
from PIL import Image, ImageDraw, ImageFont
from state_hunt import HarnessRecorder
from tracker_runtime import build_move_event, render_tracked_board, same_semantics, seed_snapshot


DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Threes Live Debug</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #111318;
      --panel: rgba(20, 25, 34, 0.92);
      --panel-2: rgba(27, 33, 44, 0.92);
      --border: #334054;
      --text: #f4f6f8;
      --muted: #95a2b7;
      --good: #7ee787;
      --warn: #ffd166;
      --bad: #ff6b6b;
      --accent: #7cc7ff;
      --accent-2: #9df0d1;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background:
        radial-gradient(circle at top left, rgba(124, 199, 255, 0.12), transparent 28%),
        radial-gradient(circle at top right, rgba(157, 240, 209, 0.08), transparent 22%),
        linear-gradient(180deg, #161c25 0%, var(--bg) 55%);
      color: var(--text);
      font-family: Menlo, Monaco, "SFMono-Regular", "Courier New", monospace;
    }
    header {
      max-width: 1600px;
      margin: 0 auto;
      padding: 22px 24px 10px;
    }
    h1 {
      margin: 0 0 8px 0;
      font-size: 28px;
      letter-spacing: 0.02em;
    }
    .subtitle {
      color: var(--muted);
      font-size: 14px;
      line-height: 1.5;
    }
    .layout {
      max-width: 1600px;
      margin: 0 auto;
      padding: 0 24px 24px;
      display: grid;
      grid-template-columns: minmax(420px, 1.25fr) minmax(360px, 1fr) minmax(320px, 0.95fr);
      gap: 18px;
      align-items: start;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 22px;
      padding: 18px;
      box-shadow: 0 18px 60px rgba(0, 0, 0, 0.28);
    }
    .panel.alt {
      background: var(--panel-2);
    }
    h2 {
      margin: 0 0 12px 0;
      font-size: 17px;
    }
    .image-wrap {
      position: relative;
      border-radius: 18px;
      overflow: hidden;
      border: 1px solid var(--border);
      background: #0a0d13;
      min-height: 320px;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    img {
      width: 100%;
      display: block;
      background: #0a0d13;
    }
    .placeholder {
      color: var(--muted);
      padding: 18px;
      text-align: center;
      line-height: 1.5;
      font-size: 14px;
    }
    .chips {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 14px;
    }
    .chip {
      border-radius: 999px;
      border: 1px solid var(--border);
      padding: 8px 12px;
      font-size: 12px;
      line-height: 1;
      color: var(--text);
      background: rgba(255, 255, 255, 0.04);
    }
    .chip.good { border-color: rgba(126, 231, 135, 0.45); color: var(--good); }
    .chip.warn { border-color: rgba(255, 209, 102, 0.45); color: var(--warn); }
    .chip.bad { border-color: rgba(255, 107, 107, 0.45); color: var(--bad); }
    .chip.accent { border-color: rgba(124, 199, 255, 0.45); color: var(--accent); }
    .stack {
      display: grid;
      gap: 18px;
    }
    pre {
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      line-height: 1.45;
      font-size: 13px;
      color: var(--text);
    }
    .meta {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.55;
    }
    .label {
      margin: 0 0 8px;
      color: var(--muted);
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .error {
      color: var(--bad);
    }
    .success {
      color: var(--good);
    }
    .section {
      border-top: 1px solid rgba(255, 255, 255, 0.06);
      padding-top: 14px;
      margin-top: 14px;
    }
    .mono-link {
      color: var(--accent);
      text-decoration: none;
    }
    .mono-link:hover {
      text-decoration: underline;
    }
    @media (max-width: 1280px) {
      .layout {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <header>
    <h1>Threes Live Debug</h1>
    <div class="subtitle">
      Live mirrored-phone capture, live board parsing, and live tracker state in one place.
      This page should stay up even when the phone is off-scene; the capture status will tell you why.
    </div>
  </header>
  <main class="layout">
    <section class="panel">
      <h2>Current Screen</h2>
      <div class="chips" id="chips"></div>
      <div class="image-wrap">
        <img id="fullImage" alt="Current mirrored screen">
        <div id="fullPlaceholder" class="placeholder">Waiting for the first capture...</div>
      </div>
      <div class="section">
        <div class="label">Recent Events</div>
        <pre id="events">Waiting for the capture loop...</pre>
      </div>
    </section>
    <section class="panel alt">
      <h2>Board Overlay</h2>
      <div class="image-wrap">
        <img id="boardImage" alt="Current board overlay">
        <div id="boardPlaceholder" class="placeholder">Board overlay will appear as soon as a game board is detected.</div>
      </div>
      <div class="section">
        <div class="label">Detected Board</div>
        <pre id="detectedBoard">Waiting for board detection...</pre>
      </div>
    </section>
    <section class="stack">
      <section class="panel">
        <h2>Tracker</h2>
        <div class="meta" id="trackerMeta">Waiting for tracker state...</div>
        <div class="section">
          <div class="label">Tracked Model</div>
          <pre id="trackedBoard">Waiting for the first settled frame...</pre>
        </div>
        <div class="section">
          <div class="label">Latest Event</div>
          <pre id="latestEvent">No events yet.</pre>
        </div>
      </section>
      <section class="panel">
        <h2>Issues</h2>
        <pre id="issues">No issues recorded.</pre>
      </section>
      <section class="panel">
        <h2>Session</h2>
        <pre id="sessionMeta">Loading...</pre>
      </section>
    </section>
  </main>
  <script>
    const chipsNode = document.getElementById("chips");
    const eventsNode = document.getElementById("events");
    const detectedBoardNode = document.getElementById("detectedBoard");
    const trackedBoardNode = document.getElementById("trackedBoard");
    const latestEventNode = document.getElementById("latestEvent");
    const issuesNode = document.getElementById("issues");
    const sessionMetaNode = document.getElementById("sessionMeta");
    const trackerMetaNode = document.getElementById("trackerMeta");
    const fullImageNode = document.getElementById("fullImage");
    const boardImageNode = document.getElementById("boardImage");
    const fullPlaceholderNode = document.getElementById("fullPlaceholder");
    const boardPlaceholderNode = document.getElementById("boardPlaceholder");

    let lastRevision = -1;
    let latestState = null;

    function chipClass(kind) {
      if (kind === "good") return "chip good";
      if (kind === "warn") return "chip warn";
      if (kind === "bad") return "chip bad";
      return "chip accent";
    }

    function renderChips(state) {
      const ageMs = Math.max(0, Date.now() - (state.captured_at_ms || Date.now()));
      const trackerClass =
        state.tracker.run_state === "failure" ? "bad" :
        state.tracker.run_state === "tracking" ? "good" :
        state.tracker.run_state === "settling" ? "warn" :
        "accent";
      const chips = [
        { label: "scene=" + (state.scene || "?"), kind: state.scene === "game" ? "good" : (state.scene === "phone_in_use" ? "warn" : "accent") },
        { label: "tracker=" + (state.tracker.run_state || "?"), kind: trackerClass },
        { label: "rev=" + (state.revision ?? "?"), kind: "accent" },
        { label: "age=" + ageMs + "ms", kind: ageMs > 1200 ? "bad" : (ageMs > 450 ? "warn" : "good") },
        { label: "capture=" + (state.capture_elapsed_ms ?? "?") + "ms", kind: "accent" },
        { label: "backend=" + (state.backend || "?"), kind: "accent" },
      ];
      chipsNode.innerHTML = chips.map((chip) => `<span class="${chipClass(chip.kind)}">${chip.label}</span>`).join("");
    }

    function swapImage(node, placeholderNode, path, rev, altText) {
      if (!path) {
        node.removeAttribute("src");
        placeholderNode.textContent = altText;
        placeholderNode.style.display = "block";
        return;
      }
      const revKey = String(rev) + ":" + path;
      if (node.dataset.revKey === revKey) {
        placeholderNode.style.display = "none";
        return;
      }
      const url = path + "?rev=" + encodeURIComponent(revKey);
      const preloader = new Image();
      preloader.onload = () => {
        node.src = url;
        node.dataset.revKey = revKey;
        placeholderNode.style.display = "none";
      };
      preloader.onerror = () => {
        placeholderNode.textContent = "Image update failed.";
        placeholderNode.style.display = "block";
      };
      preloader.src = url;
    }

    function renderState(state) {
      renderChips(state);
      detectedBoardNode.textContent = state.detected.rendered_board || state.detected.message || "No detected board.";
      trackedBoardNode.textContent = state.tracker.rendered_board || state.tracker.message || "No tracked state yet.";
      latestEventNode.textContent = state.tracker.latest_event_pretty || "No events yet.";
      issuesNode.textContent = (state.issues && state.issues.length) ? state.issues.join("\\n") : "No issues recorded.";
      eventsNode.textContent = (state.recent_events && state.recent_events.length) ? state.recent_events.join("\\n") : "No recent events yet.";
      trackerMetaNode.textContent = state.tracker.message || "Waiting for tracker state.";
      sessionMetaNode.textContent = [
        "session=" + (state.session_dir || ""),
        "window=" + (state.window_title || "?"),
        "window_id=" + (state.window_id ?? "?"),
        "updated=" + (state.updated_at || "?"),
        "raw=" + (state.images.full || ""),
      ].join("\\n");
      swapImage(
        fullImageNode,
        fullPlaceholderNode,
        state.images.full_annotated,
        state.revision,
        "Waiting for the first capture..."
      );
      swapImage(
        boardImageNode,
        boardPlaceholderNode,
        state.images.board_overlay,
        state.revision,
        "Board overlay will appear as soon as a game board is detected."
      );
    }

    async function refresh() {
      const response = await fetch("/api/state?ts=" + Date.now(), { cache: "no-store" });
      if (!response.ok) throw new Error("Failed to load live state");
      const state = await response.json();
      latestState = state;
      renderState(state);
      lastRevision = state.revision ?? lastRevision;
    }

    refresh().catch((error) => {
      issuesNode.textContent = "Dashboard error: " + error.message;
    });
    setInterval(() => refresh().catch(() => {}), 150);
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the live Threes debug dashboard.")
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
        help="Capture backend used for live parsing.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("datasets/live_debug"),
        help="Base directory for live debug sessions and recorded artifacts.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="HTTP bind host.")
    parser.add_argument("--port", type=int, default=55777, help="HTTP bind port.")
    parser.add_argument("--poll", type=float, default=0.18, help="Capture loop interval in seconds.")
    parser.add_argument("--settle-frames", type=int, default=2, help="Stable-frame count before accepting a move.")
    parser.add_argument("--settle-threshold", type=float, default=0.15, help="Board signature delta threshold.")
    parser.add_argument(
        "--max-recovery-depth",
        type=int,
        default=2,
        help="Maximum move depth to search when a human move skips an intermediate stable state.",
    )
    parser.add_argument(
        "--attach-current-game",
        action="store_true",
        help="Attach to the current board even if it is not a fresh game.",
    )
    parser.add_argument(
        "--no-open-browser",
        action="store_true",
        help="Do not auto-open the dashboard in the default browser.",
    )
    return parser.parse_args()


def _png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("Arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def draw_full_annotated(
    state: mc.ScreenState,
    display_frame: Optional[mc.FrameState],
) -> Image.Image:
    img = Image.fromarray(state.arr).convert("RGBA")
    draw = ImageDraw.Draw(img, "RGBA")
    board_roi, board_box = ws.find_board_roi(state.arr)
    preview_roi, preview_box = ws.find_preview_roi(state.arr)
    x0, y0, x1, y1 = board_box
    draw.rounded_rectangle((x0, y0, x1, y1), radius=18, outline=(255, 209, 102, 255), width=4)
    px0, py0, px1, py1 = preview_box
    draw.rounded_rectangle((px0, py0, px1, py1), radius=14, outline=(124, 199, 255, 255), width=3)

    label_font = _font(18)
    info_font = _font(15)
    scene_label = f"scene: {state.scene}"
    draw.rounded_rectangle((14, 14, 210, 58), radius=14, fill=(17, 19, 24, 220))
    draw.text((26, 24), scene_label, fill=(244, 246, 248, 255), font=label_font)
    if display_frame is not None:
        info = f"preview: {display_frame.preview_label}"
        draw.rounded_rectangle((14, 64, 220, 102), radius=12, fill=(17, 19, 24, 220))
        draw.text((26, 74), info, fill=(149, 162, 183, 255), font=info_font)
    return img.convert("RGB")


def draw_board_overlay(frame: mc.FrameState) -> tuple[Image.Image, Image.Image, Image.Image]:
    board_roi, _board_box = ws.find_board_roi(frame.arr)
    preview_roi, _preview_box = ws.find_preview_roi(frame.arr)
    board_img = Image.fromarray(board_roi).convert("RGBA")
    draw = ImageDraw.Draw(board_img, "RGBA")
    outer_boxes, _grid_meta = ws._board_cell_boxes(board_roi, inset_ratio=0.0)
    label_font = _font(20)
    coord_font = _font(12)
    for row, col, outer_box, _inner_box in outer_boxes:
        x0, y0, x1, y1 = outer_box
        token = frame.board[row][col]
        box_color = (126, 231, 135, 255) if token != ws.TOKEN_OTHER else (255, 107, 107, 255)
        fill_color = (126, 231, 135, 38) if token != ws.TOKEN_OTHER else (255, 107, 107, 48)
        draw.rounded_rectangle((x0, y0, x1, y1), radius=12, outline=box_color, fill=fill_color, width=3)
        draw.text((x0 + 6, y0 + 4), f"{row},{col}", fill=(244, 246, 248, 255), font=coord_font)
        draw.text((x0 + 6, y1 - 28), str(token), fill=(255, 209, 102, 255), font=label_font)
    return board_img.convert("RGB"), Image.fromarray(board_roi).convert("RGB"), Image.fromarray(preview_roi).convert("RGB")


@dataclass
class LiveImages:
    full_raw: bytes = b""
    full_annotated: bytes = b""
    board_overlay: Optional[bytes] = None
    board_raw: Optional[bytes] = None
    preview_raw: Optional[bytes] = None


@dataclass
class SharedState:
    revision: int = 0
    payload: Dict[str, object] = field(default_factory=dict)
    images: LiveImages = field(default_factory=LiveImages)


class LiveTrackerEngine:
    def __init__(
        self,
        recorder: HarnessRecorder,
        *,
        attach_current_game: bool,
        settle_frames: int,
        settle_threshold: float,
        max_recovery_depth: int,
    ) -> None:
        self.recorder = recorder
        self.attach_current_game = attach_current_game
        self.settle_frames = settle_frames
        self.settle_threshold = settle_threshold
        self.max_recovery_depth = max_recovery_depth
        self.events: Deque[str] = deque(maxlen=30)
        self.tracked = False
        self.game_index = 1
        self.move_index = 0
        self.last_snapshot: Optional[tuple] = None
        self.last_stable_frame: Optional[mc.FrameState] = None
        self.last_capture_id: Optional[int] = None
        self.stable_state: Optional[mc.FrameState] = None
        self.stable_count = 0
        self.run_state = "starting"
        self.message = "Waiting for the first capture."
        self.failure_reasons: List[str] = []
        self.latest_event: Optional[Dict[str, object]] = None

    def _append_event(self, text: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self.events.appendleft(f"[{stamp}] {text}")

    def _reset_tracking(self, *, new_game: bool = False) -> None:
        self.tracked = False
        self.last_snapshot = None
        self.last_stable_frame = None
        self.last_capture_id = None
        self.stable_state = None
        self.stable_count = 0
        if new_game:
            self.game_index += 1
            self.move_index = 0

    def observe(self, state: mc.ScreenState, window_id: Optional[int], ts_event: float) -> None:
        self.failure_reasons = []
        if state.scene in (mc.SCENE_SCREEN_OFF, mc.SCENE_PHONE_IN_USE):
            self.run_state = "waiting_for_device"
            self.message = f"Waiting for mirrored device: {state.scene}"
            self.stable_state = None
            self.stable_count = 0
            return

        if state.scene in (mc.SCENE_TITLE, mc.SCENE_MENU, mc.SCENE_END_CONFIRM):
            self.run_state = "scene_pause"
            self.message = f"Current scene is {state.scene}."
            self.stable_state = None
            self.stable_count = 0
            if state.scene == mc.SCENE_TITLE:
                self._reset_tracking()
            return

        if state.scene in (mc.SCENE_GAME_OVER, mc.SCENE_POSTGAME):
            if self.tracked or self.move_index > 0:
                self._append_event(f"Game {self.game_index} ended after {self.move_index} tracked moves.")
            self.run_state = "game_end"
            self.message = f"Observed {state.scene}."
            self.latest_event = {
                "type": "game_end",
                "game_index": self.game_index,
                "move_index": self.move_index,
                "scene": state.scene,
            }
            if window_id is not None:
                self.recorder.record_scene(state, f"{state.scene}_game{self.game_index:03d}_move{self.move_index:04d}")
            self._reset_tracking(new_game=True)
            return

        if state.scene != mc.SCENE_GAME or state.frame is None:
            self.run_state = "waiting_for_game"
            self.message = f"Waiting for a game board. Current scene: {state.scene}."
            self.stable_state = None
            self.stable_count = 0
            return

        frame = state.frame
        if self.stable_state is None:
            self.stable_state = frame
            self.stable_count = 1
            self.run_state = "settling"
            self.message = "Collecting the first stable game frame."
            return

        diff = ws.board_signature_diff(self.stable_state.board_sig, frame.board_sig)
        if diff < self.settle_threshold and same_semantics(self.stable_state, frame):
            self.stable_count += 1
        else:
            self.stable_state = frame
            self.stable_count = 1
            self.run_state = "settling"
            self.message = "Waiting for the board to settle after movement."
            return

        if self.stable_count < self.settle_frames:
            self.run_state = "settling"
            self.message = f"Waiting for stable confirmation ({self.stable_count}/{self.settle_frames})."
            return

        settled = self.stable_state
        if not self.tracked:
            initial_error = ws._initial_state_error(settled.board, settled.preview_label)
            if initial_error is not None and not self.attach_current_game:
                self.run_state = "waiting_for_fresh_game"
                self.message = initial_error
                return
            self.last_snapshot = seed_snapshot(settled)
            if window_id is not None:
                self.last_capture_id = self.recorder.record_game_state(settled, window_id, ts_event)
            self.latest_event = {
                "type": "game_start",
                "game_index": self.game_index,
                "capture_id": self.last_capture_id,
                "board": settled.board,
                "preview_label": settled.preview_label,
                "tile_cycle": self.last_snapshot,
                "scene": state.scene,
                "tracking_enabled": self.last_snapshot is not None,
            }
            self.recorder.append_event(self.latest_event)
            self.tracked = True
            self.last_stable_frame = settled
            self.run_state = "tracking"
            self.message = "Attached to the current game."
            self._append_event(f"Attached to game {self.game_index} at move {self.move_index}.")
            return

        if self.last_stable_frame is None or same_semantics(self.last_stable_frame, settled):
            self.run_state = "tracking"
            self.message = "Tracking live game state."
            return

        capture_id = self.last_capture_id
        if window_id is not None:
            capture_id = self.recorder.record_game_state(settled, window_id, ts_event)
        capture_id = capture_id or 0
        move_index_start = self.move_index + 1
        event = build_move_event(
            self.last_stable_frame,
            self.last_snapshot,
            settled,
            game_index=self.game_index,
            move_index_start=move_index_start,
            capture_id=capture_id,
            max_recovery_depth=self.max_recovery_depth,
        )
        self.move_index += int(event.get("step_count", 1))
        event["move_index"] = self.move_index
        self.recorder.append_event(event)
        self.latest_event = event

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
            self.failure_reasons = failure_reasons
            if window_id is not None:
                scene_path = self.recorder.record_scene(
                    state,
                    f"failure_game{self.game_index:03d}_move{self.move_index:04d}",
                    extra=event,
                )
                self.recorder.append_failure(
                    {
                        "game_index": self.game_index,
                        "move_index": self.move_index,
                        "reasons": failure_reasons,
                        "scene_capture": scene_path,
                        "event": event,
                    }
                )
            self.run_state = "failure"
            self.message = "Invalid tracked state detected. Re-seeding from the current board."
            self._append_event(f"Invalid state at move {self.move_index}: {'; '.join(failure_reasons)}")
            self.tracked = False
            self.last_stable_frame = None
            self.last_snapshot = None
            self.stable_state = None
            self.stable_count = 0
            return

        next_snapshot = event["preview_check"].get("next_snapshot")
        self.last_snapshot = next_snapshot if isinstance(next_snapshot, tuple) else next_snapshot
        self.last_stable_frame = settled
        self.last_capture_id = capture_id
        self.run_state = "tracking"
        self.message = "Tracking live game state."
        directions = event.get("direction_sequence", [])
        direction_text = " -> ".join(directions) if directions else event.get("direction") or "?"
        if event.get("recovered_missed_moves", 0) > 0:
            self._append_event(
                f"Moves {move_index_start}-{self.move_index}: {direction_text} "
                f"(recovered {event['recovered_missed_moves']} skipped state)"
            )
        else:
            self._append_event(f"Move {self.move_index}: {direction_text}")

    def payload(self) -> Dict[str, object]:
        rendered = None
        if self.last_stable_frame is not None:
            rendered = render_tracked_board(self.last_stable_frame, self.last_snapshot)
        latest_event_pretty = json.dumps(self.latest_event, indent=2) if self.latest_event else None
        return {
            "run_state": self.run_state,
            "message": self.message,
            "tracked": self.tracked,
            "game_index": self.game_index,
            "move_index": self.move_index,
            "rendered_board": rendered,
            "failure_reasons": self.failure_reasons,
            "latest_event": self.latest_event,
            "latest_event_pretty": latest_event_pretty,
            "last_capture_id": self.last_capture_id,
        }

    def recent_events(self) -> List[str]:
        return list(self.events)


class LiveDebugApp:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.window_id: Optional[int] = args.window_id
        self.window_title = ""
        self.window_bounds: Optional[Dict[str, object]] = None
        self.recorder: Optional[HarnessRecorder] = None
        self.shared = SharedState()
        self.lock = Lock()
        self.stop_event = Event()
        self.capture_thread: Optional[Thread] = None
        self.server: Optional[ThreadingHTTPServer] = None
        self.recent_events: Deque[str] = deque(maxlen=40)
        self.last_error: Optional[str] = None
        self.runtime_dir = Path("output/live_debug")
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.tracker: Optional[LiveTrackerEngine] = None
        self.session_dir: Optional[Path] = None

    def _append_event(self, text: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self.recent_events.appendleft(f"[{stamp}] {text}")

    def _ensure_window(self) -> None:
        current_window_id = self.window_id
        if self.window_id is not None and self.recorder is not None and self.window_title:
            return
        window_id, window_info = mc.resolve_window(self.window_id, self.args.auto_window_prefix)
        self.window_id = window_id
        self.window_title = str(window_info.get("title") or "")
        self.window_bounds = window_info
        if current_window_id != window_id or not self.recent_events:
            self._append_event(f"Attached to window {window_id}: {self.window_title}")
        if self.recorder is None:
            self.recorder = HarnessRecorder(self.args.dataset_dir, window_info=window_info)
            self.session_dir = self.recorder.session_dir
            self.tracker = LiveTrackerEngine(
                self.recorder,
                attach_current_game=self.args.attach_current_game,
                settle_frames=self.args.settle_frames,
                settle_threshold=self.args.settle_threshold,
                max_recovery_depth=self.args.max_recovery_depth,
            )

    def _refresh_window(self) -> None:
        self.window_id = None
        self._ensure_window()

    def _write_runtime_files(self, payload: Dict[str, object], images: LiveImages) -> None:
        if self.session_dir is None:
            return
        (self.session_dir / "live_status.json").write_text(json.dumps(payload, indent=2))
        (self.session_dir / "live_full.png").write_bytes(images.full_raw)
        (self.session_dir / "live_full_annotated.png").write_bytes(images.full_annotated)
        if images.board_overlay is not None:
            (self.session_dir / "live_board_overlay.png").write_bytes(images.board_overlay)
        if images.board_raw is not None:
            (self.session_dir / "live_board.png").write_bytes(images.board_raw)
        if images.preview_raw is not None:
            (self.session_dir / "live_preview.png").write_bytes(images.preview_raw)

    def _publish(self, payload: Dict[str, object], images: LiveImages) -> None:
        runtime = {
            "pid": os.getpid(),
            "port": self.args.port,
            "session_dir": str(self.session_dir) if self.session_dir else None,
            "updated_at": payload.get("updated_at"),
        }
        (self.runtime_dir / "runtime.json").write_text(json.dumps(runtime, indent=2))
        self._write_runtime_files(payload, images)
        with self.lock:
            self.shared.revision += 1
            payload["revision"] = self.shared.revision
            self.shared.payload = payload
            self.shared.images = images

    def _current_payload(self) -> Dict[str, object]:
        with self.lock:
            return dict(self.shared.payload)

    def _state_json(self) -> bytes:
        with self.lock:
            payload = dict(self.shared.payload)
            payload["revision"] = self.shared.revision
        return json.dumps(payload, indent=2).encode("utf-8")

    def _image_bytes(self, name: str) -> Optional[bytes]:
        with self.lock:
            images = self.shared.images
            if name == "full.png":
                return images.full_raw
            if name == "full_annotated.png":
                return images.full_annotated
            if name == "board_overlay.png":
                return images.board_overlay
            if name == "board.png":
                return images.board_raw
            if name == "preview.png":
                return images.preview_raw
        return None

    def capture_loop(self) -> None:
        while not self.stop_event.is_set():
            loop_start = time.perf_counter()
            captured_at_ms = int(time.time() * 1000)
            try:
                self._ensure_window()
                assert self.window_id is not None
                assert self.tracker is not None
                state = mc.capture_screen_state(self.window_id, self.args.capture_backend)
                self.tracker.observe(state, self.window_id, time.time())
                display_frame = state.frame or state.candidate_frame

                images = LiveImages()
                raw_image = Image.fromarray(state.arr).convert("RGB")
                images.full_raw = _png_bytes(raw_image)
                images.full_annotated = _png_bytes(draw_full_annotated(state, display_frame))
                if state.frame is not None:
                    board_overlay, board_raw, preview_raw = draw_board_overlay(state.frame)
                    images.board_overlay = _png_bytes(board_overlay)
                    images.board_raw = _png_bytes(board_raw)
                    images.preview_raw = _png_bytes(preview_raw)

                detected_board = display_frame.board if display_frame is not None else None
                detected_preview = display_frame.preview_label if display_frame is not None else None
                detected_rendered = (
                    ws.format_board_with_preview(display_frame.board, display_frame.preview_label)
                    if display_frame is not None
                    else None
                )
                issues: List[str] = []
                if display_frame is not None and ws._board_has_unknowns(display_frame.board):
                    issues.append("Detected board contains unknown cells.")
                if display_frame is not None and display_frame.preview_label == "unknown":
                    issues.append("Detected preview is unknown.")
                if self.tracker.failure_reasons:
                    issues.extend(self.tracker.failure_reasons)
                if self.last_error:
                    issues.append(self.last_error)

                elapsed_ms = int((time.perf_counter() - loop_start) * 1000)
                tracker_payload = self.tracker.payload()
                payload = {
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "captured_at_ms": captured_at_ms,
                    "capture_elapsed_ms": elapsed_ms,
                    "backend": self.args.capture_backend,
                    "window_id": self.window_id,
                    "window_title": self.window_title,
                    "scene": state.scene,
                    "scene_score": state.scene_score,
                    "scene_scores": state.scene_scores,
                    "session_dir": str(self.session_dir) if self.session_dir else None,
                    "detected": {
                        "board": detected_board,
                        "preview_label": detected_preview,
                        "rendered_board": detected_rendered,
                        "message": "No live board classification for the current scene." if detected_board is None else "",
                    },
                    "tracker": tracker_payload,
                    "issues": issues,
                    "recent_events": list(self.recent_events) + self.tracker.recent_events(),
                    "images": {
                        "full": "/frame/full.png",
                        "full_annotated": "/frame/full_annotated.png",
                        "board_overlay": "/frame/board_overlay.png" if images.board_overlay else None,
                        "board": "/frame/board.png" if images.board_raw else None,
                        "preview": "/frame/preview.png" if images.preview_raw else None,
                    },
                }
                self._publish(payload, images)
                self.last_error = None
            except Exception as exc:  # noqa: BLE001
                self.last_error = f"{type(exc).__name__}: {exc}"
                self._append_event(f"Capture error: {self.last_error}")
                payload = {
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "captured_at_ms": captured_at_ms,
                    "capture_elapsed_ms": int((time.perf_counter() - loop_start) * 1000),
                    "backend": self.args.capture_backend,
                    "window_id": self.window_id,
                    "window_title": self.window_title,
                    "scene": "error",
                    "scene_score": None,
                    "scene_scores": {},
                    "session_dir": str(self.session_dir) if self.session_dir else None,
                    "detected": {
                        "board": None,
                        "preview_label": None,
                        "rendered_board": None,
                        "message": self.last_error,
                    },
                    "tracker": self.tracker.payload() if self.tracker else {
                        "run_state": "error",
                        "message": self.last_error,
                        "tracked": False,
                        "game_index": 0,
                        "move_index": 0,
                        "rendered_board": None,
                        "failure_reasons": [self.last_error],
                        "latest_event": None,
                        "latest_event_pretty": traceback.format_exc(),
                        "last_capture_id": None,
                    },
                    "issues": [self.last_error],
                    "recent_events": list(self.recent_events),
                    "images": {
                        "full": "/frame/full.png",
                        "full_annotated": "/frame/full_annotated.png",
                        "board_overlay": None,
                        "board": None,
                        "preview": None,
                    },
                }
                self._publish(payload, LiveImages())
                time.sleep(max(self.args.poll, 0.3))
                self._refresh_window()
                continue

            sleep_for = max(0.02, self.args.poll - (time.perf_counter() - loop_start))
            self.stop_event.wait(sleep_for)

    def start_capture(self) -> None:
        self.capture_thread = Thread(target=self.capture_loop, daemon=True)
        self.capture_thread.start()

    def start_server(self) -> ThreadingHTTPServer:
        app = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path.startswith("/api/state"):
                    body = app._state_json()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_no_cache()
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if self.path == "/" or self.path.startswith("/dashboard.html"):
                    body = DASHBOARD_HTML.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_no_cache()
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if self.path.startswith("/frame/"):
                    name = self.path.split("?", 1)[0].split("/")[-1]
                    data = app._image_bytes(name)
                    if not data:
                        self.send_error(404, "Frame not ready")
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "image/png")
                    self.send_header("Content-Length", str(len(data)))
                    self.send_no_cache()
                    self.end_headers()
                    self.wfile.write(data)
                    return
                self.send_error(404, "Not found")

            def send_no_cache(self) -> None:
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")

            def log_message(self, _format: str, *_args) -> None:
                return

        self.server = ThreadingHTTPServer((self.args.host, self.args.port), Handler)
        return self.server

    def shutdown(self) -> None:
        self.stop_event.set()
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
        if self.capture_thread is not None:
            self.capture_thread.join(timeout=2.0)


def main() -> None:
    args = parse_args()
    app = LiveDebugApp(args)
    server = app.start_server()
    url = f"http://{args.host}:{server.server_address[1]}/dashboard.html"
    print(f"Live debug dashboard: {url}", flush=True)
    app.start_capture()
    if not args.no_open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.shutdown()


if __name__ == "__main__":
    main()
