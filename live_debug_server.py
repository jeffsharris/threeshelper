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
from PIL import Image
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
      --bg: #10151c;
      --panel: rgba(19, 24, 33, 0.94);
      --panel-2: rgba(24, 30, 40, 0.94);
      --border: #304055;
      --text: #f5f7fa;
      --muted: #92a0b2;
      --good: #83e59b;
      --warn: #ffd166;
      --bad: #ff7d7d;
      --accent: #86c7ff;
      --tile-empty: #1a1f2a;
      --tile-gray: #97a3b8;
      --tile-red: #ff5f82;
      --tile-blue: #63beff;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background:
        radial-gradient(circle at top left, rgba(134, 199, 255, 0.12), transparent 28%),
        linear-gradient(180deg, #18212c 0%, var(--bg) 58%);
      color: var(--text);
      font-family: Menlo, Monaco, "SFMono-Regular", "Courier New", monospace;
    }
    header {
      max-width: 1500px;
      margin: 0 auto;
      padding: 20px 24px 8px;
    }
    h1 {
      margin: 0 0 8px;
      font-size: 30px;
      letter-spacing: 0.02em;
    }
    .subtitle {
      color: var(--muted);
      font-size: 14px;
      line-height: 1.6;
      max-width: 980px;
    }
    .layout {
      max-width: 1500px;
      margin: 0 auto;
      padding: 8px 24px 20px;
      display: grid;
      grid-template-columns: minmax(420px, 1.05fr) minmax(420px, 0.95fr);
      gap: 18px;
      align-items: start;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 24px;
      padding: 16px;
      box-shadow: 0 18px 60px rgba(0, 0, 0, 0.24);
    }
    .panel.alt {
      background: var(--panel-2);
    }
    h2 {
      margin: 0 0 12px;
      font-size: 18px;
    }
    .board-header {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 16px;
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
    .chip.good { border-color: rgba(131, 229, 155, 0.45); color: var(--good); }
    .chip.warn { border-color: rgba(255, 209, 102, 0.45); color: var(--warn); }
    .chip.bad { border-color: rgba(255, 125, 125, 0.45); color: var(--bad); }
    .chip.accent { border-color: rgba(134, 199, 255, 0.45); color: var(--accent); }
    .image-wrap {
      border-radius: 18px;
      overflow: hidden;
      border: 1px solid var(--border);
      background: #0a0d13;
      min-height: 180px;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .screen-phone {
      width: min(100%, 300px);
      margin: 0 auto;
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
    .section {
      border-top: 1px solid rgba(255, 255, 255, 0.06);
      padding-top: 14px;
      margin-top: 14px;
    }
    .label {
      margin: 0 0 8px;
      color: var(--muted);
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .meta {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.6;
    }
    .subtle {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.55;
    }
    pre {
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      line-height: 1.45;
      font-size: 13px;
      color: var(--text);
    }
    .board-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(56px, 1fr));
      gap: 8px;
      margin-top: 12px;
      max-width: 460px;
    }
    .tile {
      aspect-ratio: 0.84;
      border-radius: 16px;
      border: 1px solid rgba(255, 255, 255, 0.08);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: clamp(20px, 3vw, 32px);
      font-weight: 700;
      letter-spacing: -0.04em;
      box-shadow: inset 0 -7px 0 rgba(0, 0, 0, 0.18);
    }
    .tile.empty {
      background: var(--tile-empty);
      color: rgba(255, 255, 255, 0.16);
      box-shadow: none;
    }
    .tile.gray {
      background: var(--tile-gray);
      color: #10151c;
    }
    .tile.red {
      background: var(--tile-red);
      color: #fff;
    }
    .tile.blue {
      background: var(--tile-blue);
      color: #fff;
    }
    .tile.mini {
      width: 30px;
      min-width: 30px;
      aspect-ratio: 0.76;
      border-radius: 8px;
      font-size: 14px;
      box-shadow: inset 0 -3px 0 rgba(0, 0, 0, 0.18);
    }
    .tile.slot {
      width: 15px;
      min-width: 15px;
      aspect-ratio: 0.76;
      border-radius: 4px;
      font-size: 8px;
      box-shadow: inset 0 -2px 0 rgba(0, 0, 0, 0.18);
    }
    .tile.cue {
      width: 38px;
      min-width: 38px;
      aspect-ratio: 0.76;
      border-radius: 10px;
      font-size: 17px;
      box-shadow: inset 0 -3px 0 rgba(0, 0, 0, 0.18);
    }
    .next-compact {
      min-width: 132px;
      padding: 8px 10px;
      border-radius: 16px;
      border: 1px solid rgba(255, 255, 255, 0.08);
      background: rgba(255, 255, 255, 0.04);
    }
    .next-compact-label {
      color: var(--muted);
      font-size: 11px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .observed-value {
      display: flex;
      justify-content: flex-start;
      gap: 6px;
      min-height: 42px;
      align-items: center;
      margin: 6px 0 4px;
    }
    .next-compact-note {
      color: var(--muted);
      font-size: 10px;
      line-height: 1.35;
    }
    .odds-group + .odds-group {
      margin-top: 14px;
      padding-top: 12px;
      border-top: 1px solid rgba(255, 255, 255, 0.06);
    }
    .odds-header {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 8px;
    }
    .aggregate-badge {
      color: var(--good);
      font-size: 12px;
      font-weight: 700;
      white-space: nowrap;
    }
    .odds-list {
      display: grid;
      gap: 8px;
    }
    .odds-row {
      display: grid;
      grid-template-columns: 34px 34px 1fr 48px;
      gap: 8px;
      align-items: center;
      font-size: 12px;
    }
    .slot-grid {
      width: 34px;
      min-width: 34px;
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 3px;
    }
    .slot-spacer {
      width: 34px;
      min-width: 34px;
      height: 1px;
    }
    .slot-cell {
      width: 15px;
      height: 15px;
      border-radius: 4px;
      border: 1px solid rgba(255, 255, 255, 0.08);
      background: rgba(255, 255, 255, 0.04);
    }
    .prob-tile {
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .prob-bar {
      height: 12px;
      border-radius: 999px;
      overflow: hidden;
      background: rgba(255, 255, 255, 0.07);
      border: 1px solid rgba(255, 255, 255, 0.06);
    }
    .prob-fill {
      height: 100%;
      border-radius: 999px;
      background: linear-gradient(90deg, rgba(134, 199, 255, 0.88), rgba(131, 229, 155, 0.88));
    }
    .prob-note {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.6;
      margin-top: 8px;
    }
    .issues.ok {
      color: var(--good);
    }
    @media (max-width: 1180px) {
      .layout {
        grid-template-columns: 1fr;
      }
    }
    @media (max-width: 720px) {
      .board-header {
        grid-template-columns: 1fr;
        display: grid;
      }
      .board-grid {
        gap: 8px;
      }
      .tile {
        font-size: 28px;
      }
    }
  </style>
</head>
<body>
  <header>
    <h1>Threes Live Debug</h1>
    <div class="subtitle">
      Live mirrored-phone capture, live board parsing, and live tracker state in one place.
      The right side is intentionally focused on the signals that matter: current board state and next-tile expectations.
    </div>
  </header>
  <main class="layout">
    <section class="panel alt">
      <div class="board-header">
        <div>
          <h2>Current Board</h2>
          <div class="meta" id="boardSource">Waiting for a board...</div>
        </div>
        <div class="next-compact">
          <div class="next-compact-label">Next Tile</div>
          <div class="observed-value" id="observedPreview"></div>
          <div class="next-compact-note" id="observedPreviewNote">Waiting for preview detection...</div>
        </div>
      </div>
      <div class="board-grid" id="boardGrid"></div>
    </section>
    <section class="panel">
      <h2>Coming Tile Probabilities</h2>
      <div class="subtle">
        After the visible cue is spent, this is the next-cue distribution from the remaining 12-tile bag plus any enabled big-tile bundles.
      </div>
      <div class="odds-group">
        <div class="odds-header">
          <div class="label">Tile Odds After The Visible Cue</div>
        </div>
        <div class="odds-list" id="smallProbabilities"></div>
      </div>
      <div class="odds-group">
        <div class="odds-header">
          <div class="label">Big Tile Chances</div>
          <div class="aggregate-badge" id="bigProbability"></div>
        </div>
        <div class="odds-list" id="bigProbabilities"></div>
        <div class="prob-note" id="probabilitiesNote">Waiting for probability model...</div>
      </div>
    </section>
    <section class="panel">
      <h2>Current Screen</h2>
      <div class="chips" id="chips"></div>
      <div class="image-wrap">
        <div class="screen-phone">
          <img id="fullImage" alt="Current mirrored screen">
          <div id="fullPlaceholder" class="placeholder">Waiting for the first capture...</div>
        </div>
      </div>
      <div class="section">
        <div class="label">Recent Events</div>
        <pre id="events">Waiting for the capture loop...</pre>
      </div>
    </section>
    <section class="panel">
      <h2>Tracker</h2>
      <div class="meta" id="trackerMeta">Waiting for tracker state...</div>
      <div class="section">
        <div class="label">Latest Event</div>
        <pre id="latestEvent">No events yet.</pre>
      </div>
    </section>
    <section class="panel">
      <h2>Issues</h2>
      <pre id="issues" class="issues ok">No issues recorded.</pre>
    </section>
    <section class="panel">
      <h2>Session</h2>
      <pre id="sessionMeta">Loading...</pre>
    </section>
  </main>
  <script>
    const chipsNode = document.getElementById("chips");
    const boardGridNode = document.getElementById("boardGrid");
    const boardSourceNode = document.getElementById("boardSource");
    const eventsNode = document.getElementById("events");
    const trackerMetaNode = document.getElementById("trackerMeta");
    const latestEventNode = document.getElementById("latestEvent");
    const issuesNode = document.getElementById("issues");
    const sessionMetaNode = document.getElementById("sessionMeta");
    const smallProbabilitiesNode = document.getElementById("smallProbabilities");
    const bigProbabilitiesNode = document.getElementById("bigProbabilities");
    const bigProbabilityNode = document.getElementById("bigProbability");
    const observedPreviewNode = document.getElementById("observedPreview");
    const observedPreviewNoteNode = document.getElementById("observedPreviewNote");
    const probabilitiesNoteNode = document.getElementById("probabilitiesNote");
    const fullImageNode = document.getElementById("fullImage");
    const fullPlaceholderNode = document.getElementById("fullPlaceholder");

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
      const sceneClass =
        state.scene === "game" ? "good" :
        (state.scene === "phone_in_use" || state.scene === "screen_off") ? "warn" :
        "accent";
      const ageClass = ageMs > 1200 ? "bad" : (ageMs > 450 ? "warn" : "good");
      const chips = [
        { label: "scene=" + (state.scene || "?"), kind: sceneClass },
        { label: "tracker=" + (state.tracker.run_state || "?"), kind: trackerClass },
        { label: "age=" + ageMs + "ms", kind: ageClass },
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

    function tileInfo(token) {
      if (token === "🟥") return { kind: "red", label: "2" };
      if (token === "🟦") return { kind: "blue", label: "1" };
      if (token === "·") return { kind: "empty", label: "" };
      if (token === "X") return { kind: "empty", label: "?" };
      return { kind: "gray", label: String(token || "") };
    }

    function tileInfoForValue(value) {
      if (value === 1) return { kind: "blue", label: "1" };
      if (value === 2) return { kind: "red", label: "2" };
      if (value === 3) return { kind: "gray", label: "3" };
      if (value === null || value === undefined) return { kind: "empty", label: "?" };
      return { kind: "gray", label: String(value) };
    }

    function tileMarkup(info, variant) {
      const variantClass = variant ? ` ${variant}` : "";
      return `<div class="tile ${info.kind}${variantClass}">${info.label}</div>`;
    }

    function renderBoard(board) {
      if (!board || !board.length) {
        boardGridNode.innerHTML = "";
        return false;
      }
      const html = [];
      for (const row of board) {
        for (const token of row) {
          const info = tileInfo(token);
          html.push(tileMarkup(info, ""));
        }
      }
      boardGridNode.innerHTML = html.join("");
      return true;
    }

    function renderObservedPreview(preview) {
      const values = (preview && preview.values) ? preview.values : [];
      const mode = preview && preview.mode ? preview.mode : "unknown";
      if (!values.length) {
        observedPreviewNode.innerHTML = "";
      } else if (mode === "bundle_generic") {
        observedPreviewNode.innerHTML = values.map(() => tileMarkup({ kind: "gray", label: "?" }, "cue")).join("");
      } else {
        observedPreviewNode.innerHTML = values
          .map((value) => tileMarkup(tileInfoForValue(value), "cue"))
          .join("");
      }
      observedPreviewNoteNode.textContent = (preview && preview.note) ? preview.note : "No preview data available.";
    }

    function renderBigProbability(expected) {
      const percent = expected && expected.big_tile_percent ? expected.big_tile_percent : 0;
      bigProbabilityNode.textContent = `Any big block next ${percent.toFixed(1)}%`;
    }

    function renderProbabilities(expected) {
      if (!expected || !expected.available) {
        smallProbabilitiesNode.innerHTML = "";
        bigProbabilitiesNode.innerHTML = "";
        bigProbabilityNode.innerHTML = "";
        probabilitiesNoteNode.textContent = (expected && expected.note) ? expected.note : "Probability model unavailable.";
        return;
      }
      renderBigProbability(expected);
      const smallBag = new Map((expected.small_bag || []).map((item) => [item.value, item]));
      const makeSlots = (item) => {
        if (!item) {
          return `<div class="slot-spacer" aria-hidden="true"></div>`;
        }
        const filled = Math.max(0, Math.min(4, item.remaining || 0));
        const filledMarkup = Array.from({ length: filled }, () => tileMarkup(tileInfoForValue(item.value), "slot")).join("");
        const emptyMarkup = Array.from({ length: 4 - filled }, () => '<div class="slot-cell"></div>').join("");
        return `<div class="slot-grid">${filledMarkup}${emptyMarkup}</div>`;
      };
      const makeRow = (item, slotItem) => {
        const width = Math.max(0, Math.min(100, item.percent || 0));
        const tile = tileMarkup(tileInfoForValue(item.value), "mini");
        return [
          `<div class="odds-row">`,
          `<div class="prob-tile">${tile}</div>`,
          makeSlots(slotItem),
          `<div class="prob-bar"><div class="prob-fill" style="width:${width}%"></div></div>`,
          `<div>${width.toFixed(1)}%</div>`,
          `</div>`
        ].join("");
      };
      const items = expected.items || [];
      const smallOrder = [3, 1, 2];
      const smallRows = smallOrder
        .map((value) => {
          const item = items.find((entry) => entry.value === value);
          if (!item) return "";
          return makeRow(item, smallBag.get(value));
        })
        .join("");
      const bigRows = items
        .filter((item) => (item.value || 0) > 3)
        .map((item) => makeRow(item, null))
        .join("");
      smallProbabilitiesNode.innerHTML = smallRows;
      bigProbabilitiesNode.innerHTML = bigRows;
      probabilitiesNoteNode.textContent = expected.note || "";
    }

    function renderState(state) {
      renderChips(state);
      trackerMetaNode.textContent = state.tracker.message || "Waiting for tracker state.";
      latestEventNode.textContent = state.tracker.latest_event_pretty || "No events yet.";
      const board = state.tracker.board || state.detected.board;
      if (renderBoard(board)) {
        boardSourceNode.textContent = state.scene === "game"
          ? "Live parsed board from the mirrored game."
          : "Last tracked board before the current non-game scene.";
      } else {
        boardSourceNode.textContent = state.detected.message || "No board currently available.";
      }
      renderObservedPreview(state.tracker.observed_preview);
      renderProbabilities(state.tracker.expected_next);
      const issues = (state.issues && state.issues.length) ? state.issues : [];
      issuesNode.className = issues.length ? "issues" : "issues ok";
      issuesNode.textContent = issues.length ? issues.join("\\n") : "No issues recorded.";
      eventsNode.textContent = (state.recent_events && state.recent_events.length)
        ? state.recent_events.join("\\n")
        : "No recent events yet.";
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
    }

    async function refresh() {
      const response = await fetch("/api/state?ts=" + Date.now(), { cache: "no-store" });
      if (!response.ok) throw new Error("Failed to load live state");
      const state = await response.json();
      renderState(state);
    }

    refresh().catch((error) => {
      issuesNode.className = "issues";
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
def draw_full_annotated(
    state: mc.ScreenState,
    display_frame: Optional[mc.FrameState],
) -> Image.Image:
    _ = display_frame
    return Image.fromarray(state.arr).convert("RGB")


@dataclass
class LiveImages:
    full_raw: bytes = b""
    full_annotated: bytes = b""


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
        self.issue_log: Deque[str] = deque(maxlen=12)
        self.tracked = False
        self.game_index = 1
        self.move_index = 0
        self.last_snapshot: Optional[tuple] = None
        self.last_stable_frame: Optional[mc.FrameState] = None
        self.last_visible_board_frame: Optional[mc.FrameState] = None
        self.last_visible_snapshot: Optional[tuple] = None
        self.last_capture_id: Optional[int] = None
        self.stable_state: Optional[mc.FrameState] = None
        self.stable_count = 0
        self.run_state = "starting"
        self.message = "Waiting for the first capture."
        self.failure_reasons: List[str] = []
        self.latest_event: Optional[Dict[str, object]] = None
        self.end_scene_active = False

    def _append_event(self, text: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self.events.appendleft(f"[{stamp}] {text}")

    def _remember_issue(self, text: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        entry = f"[{stamp}] {text}"
        if not self.issue_log or self.issue_log[0] != entry:
            self.issue_log.appendleft(entry)

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
        if state.scene not in (mc.SCENE_GAME_OVER, mc.SCENE_POSTGAME):
            self.end_scene_active = False

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
            self.run_state = "game_end"
            self.message = f"Observed {state.scene}."
            if not self.end_scene_active:
                if self.tracked or self.move_index > 0:
                    self._append_event(f"Game {self.game_index} ended after {self.move_index} tracked moves.")
                self.latest_event = {
                    "type": "game_end",
                    "game_index": self.game_index,
                    "move_index": self.move_index,
                    "scene": state.scene,
                }
                if window_id is not None:
                    self.recorder.record_scene(
                        state, f"{state.scene}_game{self.game_index:03d}_move{self.move_index:04d}"
                    )
                self.end_scene_active = True
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
        self.last_visible_board_frame = settled
        if self.last_snapshot is not None:
            self.last_visible_snapshot = self.last_snapshot
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
            if self.last_snapshot is None:
                self.message = (
                    "Attached mid-game. Exact next-tile probabilities are unavailable until the next fresh game."
                )
            else:
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
            self._remember_issue(f"Move {self.move_index}: {'; '.join(failure_reasons)}")
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

    def _display_frame(self) -> Optional[mc.FrameState]:
        return self.last_stable_frame or self.last_visible_board_frame

    def _display_snapshot(self) -> Optional[tuple]:
        return self.last_snapshot or self.last_visible_snapshot

    def _observed_preview_payload(self, frame: Optional[mc.FrameState]) -> Dict[str, object]:
        if frame is None:
            return {"label": None, "mode": "none", "values": [], "note": "No active game preview is visible."}
        max_tile = ws.board_max_tile(frame.board)
        label = frame.preview_label
        if label == "red":
            return {"label": label, "mode": "single", "values": [ws.SMALL_TILE_VALUES["red"]], "note": "Observed red cue."}
        if label == "blue":
            return {"label": label, "mode": "single", "values": [ws.SMALL_TILE_VALUES["blue"]], "note": "Observed blue cue."}
        if label == "gray":
            return {"label": label, "mode": "single", "values": [ws.SMALL_TILE_VALUES["gray"]], "note": "Observed gray cue."}
        if label == "large_candidates":
            return {
                "label": label,
                "mode": "bundle_generic",
                "values": [None, None, None],
                "note": f"Observed large-tile preview band. Current max tile is {max_tile}.",
            }
        return {"label": label, "mode": "unknown", "values": [], "note": "Observed preview cue."}

    def _probability_payload(self, frame: Optional[mc.FrameState]) -> Dict[str, object]:
        if frame is None:
            return {"available": False, "items": [], "note": "No active game board is visible.", "bonus_values": []}
        max_tile = ws.board_max_tile(frame.board)
        cycle = ws.TileCycle()
        cycle.set_max_tile(max_tile)
        bonus_values = cycle.bonus_values()
        bonus_cap = cycle.bonus_max_tile()
        locked_note = cycle.large_schedule_note()
        if self.last_snapshot is None:
            note = "Start tracking from a fresh game to compute exact cue odds after the visible tile is spent."
            return {
                "available": False,
                "items": [],
                "small_bag": [],
                "note": note,
                "bonus_values": bonus_values,
                "big_tile_percent": 0.0,
                "big_tile_note": f"No exact big-tile odds until the tracker is seeded from a fresh game. {locked_note}",
                "large_schedule_note": locked_note,
                "bonus_cap": bonus_cap,
                "max_tile": max_tile,
            }
        cycle.restore(self.last_snapshot)
        cycle.set_max_tile(max_tile)
        probs = cycle.probabilities()
        bonus_windows = cycle.bonus_windows()
        bonus_value_probs = cycle.bonus_value_probabilities()
        small_bag = [
            {
                "key": "gray",
                "value": ws.SMALL_TILE_VALUES["gray"],
                "remaining": cycle.small_counts.get("gray", 0),
                "base": 4,
                "probability": probs.get("gray", 0.0),
            },
            {
                "key": "blue",
                "value": ws.SMALL_TILE_VALUES["blue"],
                "remaining": cycle.small_counts.get("blue", 0),
                "base": 4,
                "probability": probs.get("blue", 0.0),
            },
            {
                "key": "red",
                "value": ws.SMALL_TILE_VALUES["red"],
                "remaining": cycle.small_counts.get("red", 0),
                "base": 4,
                "probability": probs.get("red", 0.0),
            },
        ]
        for bag_item in small_bag:
            bag_item["percent"] = round(bag_item["probability"] * 100.0, 1)
        items = [
            {"key": "blue", "value": ws.SMALL_TILE_VALUES["blue"], "probability": probs.get("blue", 0.0)},
            {"key": "red", "value": ws.SMALL_TILE_VALUES["red"], "probability": probs.get("red", 0.0)},
            {"key": "gray", "value": ws.SMALL_TILE_VALUES["gray"], "probability": probs.get("gray", 0.0)},
        ]
        large_prob = probs.get("large_candidates", 0.0)
        if large_prob > 0 and bonus_value_probs:
            for value in sorted(bonus_value_probs):
                items.append(
                    {
                        "key": f"bonus_{value}",
                        "value": value,
                        "probability": large_prob * bonus_value_probs[value],
                    }
                )
        items.sort(key=lambda item: int(item["value"]))
        for item in items:
            item["percent"] = round(item["probability"] * 100.0, 1)
        bonus_total = round(large_prob * 100.0, 1)
        note = "These odds are for the cue that should appear after the current visible tile is used. "
        note += f"Bag progress: {cycle.small_pos}/12 spent."
        bundle_items = []
        if bonus_windows:
            conditional_percent = round(100.0 / len(bonus_windows), 1)
            for idx, window in enumerate(bonus_windows):
                bundle_items.append(
                    {
                        "key": f"bundle_{idx}",
                        "values": window,
                        "conditional_percent": conditional_percent,
                        "overall_percent": round((large_prob / len(bonus_windows)) * 100.0, 1),
                    }
                )
            note += f" Big-tile support runs from 6 to {bonus_cap}."
        schedule_note = cycle.large_schedule_note()
        if large_prob <= 0:
            big_tile_note = schedule_note
        else:
            bundle_note = (
                f"grouped into {len(bonus_windows)} equal 3-tile bundles"
                if bonus_windows
                else "with no 3-tile bundle data available yet"
            )
            big_tile_note = (
                f"Any big block next: {bonus_total:.1f}%. "
                f"Allowed big tiles run from 6 to {bonus_cap}, {bundle_note}. "
                f"{schedule_note}"
            )
        return {
            "available": True,
            "items": items,
            "small_bag": small_bag,
            "note": note,
            "bonus_values": bonus_values,
            "bonus_bundles": bundle_items,
            "big_tile_percent": bonus_total,
            "big_tile_note": big_tile_note,
            "large_schedule_note": schedule_note,
            "bonus_cap": bonus_cap,
            "max_tile": max_tile,
        }

    def payload(self) -> Dict[str, object]:
        frame = self._display_frame()
        snapshot = self._display_snapshot()
        rendered = render_tracked_board(frame, snapshot) if frame is not None else None
        latest_event_pretty = json.dumps(self.latest_event, indent=2) if self.latest_event else None
        return {
            "run_state": self.run_state,
            "message": self.message,
            "tracked": self.tracked,
            "tracking_enabled": self.last_snapshot is not None,
            "game_index": self.game_index,
            "move_index": self.move_index,
            "rendered_board": rendered,
            "board": frame.board if frame is not None else None,
            "observed_preview": self._observed_preview_payload(frame),
            "expected_next": self._probability_payload(frame),
            "failure_reasons": self.failure_reasons,
            "issue_log": list(self.issue_log),
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

                images = LiveImages()
                raw_image = Image.fromarray(state.arr).convert("RGB")
                images.full_raw = _png_bytes(raw_image)
                images.full_annotated = _png_bytes(draw_full_annotated(state, state.frame))

                detected_board = state.frame.board if state.frame is not None else None
                detected_preview = state.frame.preview_label if state.frame is not None else None
                detected_rendered = (
                    ws.format_board_with_preview(state.frame.board, state.frame.preview_label)
                    if state.frame is not None
                    else None
                )
                issues: List[str] = []
                if state.frame is not None and ws._board_has_unknowns(state.frame.board):
                    issues.append("Detected board contains unknown cells.")
                if state.frame is not None and state.frame.preview_label == "unknown":
                    issues.append("Detected preview is unknown.")
                tracker_payload = self.tracker.payload()
                issues.extend(tracker_payload.get("issue_log", []))
                issues.extend(self.tracker.failure_reasons)
                if self.last_error:
                    issues.append(self.last_error)
                deduped_issues: List[str] = []
                seen_issues = set()
                for issue in issues:
                    if not issue or issue in seen_issues:
                        continue
                    deduped_issues.append(issue)
                    seen_issues.add(issue)

                elapsed_ms = int((time.perf_counter() - loop_start) * 1000)
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
                    "issues": deduped_issues,
                    "recent_events": list(self.recent_events) + self.tracker.recent_events(),
                    "images": {
                        "full": "/frame/full.png",
                        "full_annotated": "/frame/full_annotated.png",
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

        class ReusableThreadingHTTPServer(ThreadingHTTPServer):
            allow_reuse_address = True
            daemon_threads = True

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

        self.server = ReusableThreadingHTTPServer((self.args.host, self.args.port), Handler)
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
