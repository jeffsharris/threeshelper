import argparse
import json
import time
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Dict, List, Optional

import mirroring_control as mc
import window_stream as ws
from PIL import Image
from state_hunt import HarnessRecorder
from tracker_runtime import build_move_event, frame_with_board, render_tracked_board, same_semantics, seed_snapshot


DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
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
      <div class="meta"><a href="grid_editor.html" target="_blank" rel="noopener">Open Grid Editor</a></div>
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
    let lastCaptureKey = null;

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
        "updated=" + (state.updated_at || "?"),
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
      const displayImage = state.display_image || state.latest_board_overlay || state.latest_full || null;
      const captureKey = displayImage
        ? String(state.live_view_id || state.latest_capture_id || 0) + ":" + displayImage
        : null;
      if (displayImage && captureKey !== lastCaptureKey) {
        captureNode.src = displayImage + "?capture=" + encodeURIComponent(captureKey);
        lastCaptureKey = captureKey;
      } else {
        if (!displayImage) {
          captureNode.removeAttribute("src");
          lastCaptureKey = null;
        }
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


GRID_EDITOR_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Threes Grid Editor</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #12131d;
      --panel: #1f2131;
      --border: #3b3e55;
      --text: #f5f5f7;
      --muted: #a8acba;
      --vline: #ffcf5c;
      --hline: #7ac7ff;
      --saved: #8fd694;
    }
    body {
      margin: 0;
      background: radial-gradient(circle at top, #25283d 0%, var(--bg) 55%);
      color: var(--text);
      font-family: Menlo, Monaco, "Courier New", monospace;
    }
    main {
      max-width: 1280px;
      margin: 0 auto;
      padding: 20px;
      display: grid;
      grid-template-columns: 340px 1fr;
      gap: 20px;
    }
    .panel {
      background: rgba(31, 33, 49, 0.94);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 18px;
      box-shadow: 0 22px 50px rgba(0, 0, 0, 0.28);
    }
    h1, h2 {
      margin: 0 0 12px 0;
      font-size: 18px;
    }
    p, .meta, pre, textarea, button {
      font-size: 13px;
      line-height: 1.5;
    }
    .meta {
      color: var(--muted);
    }
    .controls {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin: 12px 0;
    }
    button {
      appearance: none;
      border: 1px solid var(--border);
      background: #2a2e46;
      color: var(--text);
      border-radius: 10px;
      padding: 10px 12px;
      cursor: pointer;
    }
    button:hover {
      background: #343955;
    }
    textarea {
      width: 100%;
      min-height: 220px;
      background: #12131d;
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 12px;
      box-sizing: border-box;
    }
    .stage-wrap {
      overflow: auto;
    }
    .stage {
      position: relative;
      display: inline-block;
      border-radius: 18px;
      border: 1px solid var(--border);
      overflow: hidden;
      background: #0f1018;
      min-width: 320px;
      min-height: 400px;
    }
    .stage img {
      display: block;
      max-width: none;
    }
    .line {
      position: absolute;
      z-index: 2;
      user-select: none;
    }
    .line.vertical {
      top: 0;
      bottom: 0;
      width: 4px;
      margin-left: -2px;
      background: var(--vline);
      cursor: ew-resize;
    }
    .line.horizontal {
      left: 0;
      right: 0;
      height: 4px;
      margin-top: -2px;
      background: var(--hline);
      cursor: ns-resize;
    }
    .line span {
      position: absolute;
      padding: 2px 6px;
      border-radius: 999px;
      background: rgba(0, 0, 0, 0.75);
      color: var(--text);
      font-size: 11px;
      white-space: nowrap;
    }
    .line.vertical span {
      top: 8px;
      left: 6px;
    }
    .line.horizontal span {
      top: 6px;
      left: 8px;
    }
    .saved {
      color: var(--saved);
    }
    @media (max-width: 980px) {
      main {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <main>
    <section class="panel">
      <h1>Grid Editor</h1>
      <p class="meta">Drag the column and row boundaries until each tile box is exactly right, then save. The saved JSON is written into this observer session so the tracker can use it later.</p>
      <div class="controls">
        <button id="reload">Reload Latest</button>
        <button id="save">Save Calibration</button>
      </div>
      <div id="status" class="meta">Loading latest capture...</div>
      <div class="meta">Vertical lines: `c0_left`, `c0_right`, ... `c3_left`, `c3_right`</div>
      <div class="meta">Horizontal lines: `r0_top`, `r0_bottom`, ... `r3_top`, `r3_bottom`</div>
      <h2>Saved JSON</h2>
      <textarea id="json" spellcheck="false"></textarea>
    </section>
    <section class="panel stage-wrap">
      <div id="stage" class="stage">
        <img id="board" alt="Latest board capture">
      </div>
    </section>
  </main>
  <script>
    const stage = document.getElementById("stage");
    const board = document.getElementById("board");
    const statusNode = document.getElementById("status");
    const jsonNode = document.getElementById("json");
    const reloadButton = document.getElementById("reload");
    const saveButton = document.getElementById("save");

    let currentState = null;
    let lines = null;
    let dragging = null;

    function median(values) {
      if (!values.length) return 0;
      const sorted = [...values].sort((a, b) => a - b);
      const mid = Math.floor(sorted.length / 2);
      if (sorted.length % 2) return sorted[mid];
      return (sorted[mid - 1] + sorted[mid]) / 2;
    }

    function clamp(value, min, max) {
      return Math.min(Math.max(value, min), max);
    }

    async function loadJson(path) {
      const response = await fetch(path + "?ts=" + Date.now(), { cache: "no-store" });
      if (!response.ok) throw new Error("Failed to load " + path);
      return response.json();
    }

    async function maybeLoadJson(path) {
      const response = await fetch(path + "?ts=" + Date.now(), { cache: "no-store" });
      if (!response.ok) return null;
      return response.json();
    }

    function deriveLinesFromMeta(meta) {
      const grouped = {};
      for (const stat of meta.cell_stats || []) {
        const outer = stat.outer_box || stat.box;
        const row = stat.row;
        const col = stat.col;
        grouped[row + "," + col] = outer;
      }
      const colLefts = [];
      const colRights = [];
      const rowTops = [];
      const rowBottoms = [];
      for (let col = 0; col < 4; col++) {
        const lefts = [];
        const rights = [];
        for (let row = 0; row < 4; row++) {
          const box = grouped[row + "," + col];
          if (!box) continue;
          lefts.push(box[0]);
          rights.push(box[2]);
        }
        colLefts.push(median(lefts));
        colRights.push(median(rights));
      }
      for (let row = 0; row < 4; row++) {
        const tops = [];
        const bottoms = [];
        for (let col = 0; col < 4; col++) {
          const box = grouped[row + "," + col];
          if (!box) continue;
          tops.push(box[1]);
          bottoms.push(box[3]);
        }
        rowTops.push(median(tops));
        rowBottoms.push(median(bottoms));
      }
      return { col_lefts: colLefts, col_rights: colRights, row_tops: rowTops, row_bottoms: rowBottoms };
    }

    function buildPayload() {
      if (!lines || !currentState) return {};
      const width = board.naturalWidth;
      const height = board.naturalHeight;
      const cellBoxes = [];
      for (let row = 0; row < 4; row++) {
        for (let col = 0; col < 4; col++) {
          cellBoxes.push({
            row,
            col,
            box: [
              lines.col_lefts[col],
              lines.row_tops[row],
              lines.col_rights[col],
              lines.row_bottoms[row],
            ],
          });
        }
      }
      return {
        capture_id: currentState.latest_capture_id,
        board_image: currentState.latest_board,
        meta_path: currentState.latest_meta,
        image_size: { width, height },
        col_lefts: lines.col_lefts,
        col_rights: lines.col_rights,
        row_tops: lines.row_tops,
        row_bottoms: lines.row_bottoms,
        normalized: {
          col_lefts: lines.col_lefts.map((value) => Number((value / width).toFixed(6))),
          col_rights: lines.col_rights.map((value) => Number((value / width).toFixed(6))),
          row_tops: lines.row_tops.map((value) => Number((value / height).toFixed(6))),
          row_bottoms: lines.row_bottoms.map((value) => Number((value / height).toFixed(6))),
        },
        cell_boxes: cellBoxes,
        saved_at: new Date().toISOString(),
      };
    }

    function renderJson() {
      jsonNode.value = JSON.stringify(buildPayload(), null, 2);
    }

    function clearLines() {
      stage.querySelectorAll(".line").forEach((node) => node.remove());
    }

    function createLine(axis, index, value, label) {
      const node = document.createElement("div");
      node.className = "line " + axis;
      node.dataset.axis = axis;
      node.dataset.index = String(index);
      const tag = document.createElement("span");
      tag.textContent = label;
      node.appendChild(tag);
      if (axis === "vertical") {
        node.style.left = value + "px";
      } else {
        node.style.top = value + "px";
      }
      node.addEventListener("mousedown", (event) => {
        dragging = { axis, index };
        event.preventDefault();
      });
      stage.appendChild(node);
    }

    function renderLines() {
      if (!lines) return;
      clearLines();
      lines.col_lefts.forEach((value, index) => createLine("vertical", index, value, "c" + index + "_left"));
      lines.col_rights.forEach((value, index) => createLine("vertical", index + 4, value, "c" + index + "_right"));
      lines.row_tops.forEach((value, index) => createLine("horizontal", index, value, "r" + index + "_top"));
      lines.row_bottoms.forEach((value, index) => createLine("horizontal", index + 4, value, "r" + index + "_bottom"));
      renderJson();
    }

    function orderedBounds(values, minGap, maxValue) {
      const out = [...values];
      for (let i = 1; i < out.length; i++) {
        out[i] = Math.max(out[i], out[i - 1] + minGap);
      }
      for (let i = out.length - 2; i >= 0; i--) {
        out[i] = Math.min(out[i], out[i + 1] - minGap);
      }
      return out.map((value) => clamp(value, 0, maxValue));
    }

    function moveLine(clientX, clientY) {
      if (!dragging || !lines) return;
      const rect = stage.getBoundingClientRect();
      if (dragging.axis === "vertical") {
        const x = clamp(clientX - rect.left, 0, board.naturalWidth);
        if (dragging.index < 4) {
          lines.col_lefts[dragging.index] = x;
          lines.col_lefts = orderedBounds(lines.col_lefts, 4, board.naturalWidth);
        } else {
          const idx = dragging.index - 4;
          lines.col_rights[idx] = x;
          lines.col_rights = orderedBounds(lines.col_rights, 4, board.naturalWidth);
        }
        for (let i = 0; i < 4; i++) {
          if (lines.col_rights[i] <= lines.col_lefts[i] + 8) {
            lines.col_rights[i] = lines.col_lefts[i] + 8;
          }
        }
        lines.col_rights = orderedBounds(lines.col_rights, 4, board.naturalWidth);
      } else {
        const y = clamp(clientY - rect.top, 0, board.naturalHeight);
        if (dragging.index < 4) {
          lines.row_tops[dragging.index] = y;
          lines.row_tops = orderedBounds(lines.row_tops, 4, board.naturalHeight);
        } else {
          const idx = dragging.index - 4;
          lines.row_bottoms[idx] = y;
          lines.row_bottoms = orderedBounds(lines.row_bottoms, 4, board.naturalHeight);
        }
        for (let i = 0; i < 4; i++) {
          if (lines.row_bottoms[i] <= lines.row_tops[i] + 8) {
            lines.row_bottoms[i] = lines.row_tops[i] + 8;
          }
        }
        lines.row_bottoms = orderedBounds(lines.row_bottoms, 4, board.naturalHeight);
      }
      renderLines();
    }

    async function loadLatest() {
      currentState = await loadJson("live_status.json");
      statusNode.textContent = currentState.message || "Loaded.";
      if (!currentState.latest_board || !currentState.latest_meta) {
        jsonNode.value = "";
        return;
      }
      await new Promise((resolve) => {
        board.onload = resolve;
        board.src = currentState.latest_board + "?ts=" + Date.now();
      });
      stage.style.width = board.naturalWidth + "px";
      stage.style.height = board.naturalHeight + "px";

      const saved = await maybeLoadJson("grid_calibration.json");
      if (saved && saved.board_image === currentState.latest_board) {
        lines = {
          col_lefts: saved.col_lefts,
          col_rights: saved.col_rights,
          row_tops: saved.row_tops,
          row_bottoms: saved.row_bottoms,
        };
        statusNode.innerHTML = '<span class="saved">Loaded saved calibration for this capture.</span>';
      } else {
        const meta = await loadJson(currentState.latest_meta);
        lines = deriveLinesFromMeta(meta);
      }
      renderLines();
    }

    async function saveCalibration() {
      const payload = buildPayload();
      const response = await fetch("save_grid", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!response.ok) {
        throw new Error("Save failed");
      }
      statusNode.innerHTML = '<span class="saved">Saved grid calibration to grid_calibration.json</span>';
      renderJson();
    }

    reloadButton.addEventListener("click", () => {
      loadLatest().catch((error) => {
        statusNode.textContent = "Load failed: " + error.message;
      });
    });

    saveButton.addEventListener("click", () => {
      saveCalibration().catch((error) => {
        statusNode.textContent = "Save failed: " + error.message;
      });
    });

    window.addEventListener("mousemove", (event) => moveLine(event.clientX, event.clientY));
    window.addEventListener("mouseup", () => {
      dragging = null;
    });

    loadLatest().catch((error) => {
      statusNode.textContent = "Load failed: " + error.message;
    });
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
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Exit immediately on the first invalid state instead of re-seeding from the current board.",
    )
    return parser.parse_args()


def print_frame(frame: mc.FrameState, snapshot: Optional[tuple]) -> None:
    print(render_tracked_board(frame, snapshot))


def start_dashboard_server(session_dir: Path, port: int) -> tuple[ThreadingHTTPServer, Thread, str]:
    dashboard_path = session_dir / "dashboard.html"
    dashboard_path.write_text(DASHBOARD_HTML)
    grid_editor_path = session_dir / "grid_editor.html"
    grid_editor_path.write_text(GRID_EDITOR_HTML)

    class SessionHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(session_dir), **kwargs)

        def end_headers(self) -> None:
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            super().end_headers()

        def do_POST(self) -> None:
            if self.path.rstrip("/") != "/save_grid":
                self.send_error(404, "File not found")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = self.rfile.read(length)
                data = json.loads(payload.decode("utf-8"))
                (session_dir / "grid_calibration.json").write_text(json.dumps(data, indent=2))
                repo_path = Path.cwd() / "board_grid_calibration.json"
                repo_path.write_text(json.dumps(data, indent=2))
                ws.clear_board_grid_calibration_cache()
            except Exception as exc:  # noqa: BLE001
                self.send_error(400, f"Could not save grid calibration: {exc}")
                return
            body = json.dumps({"ok": True}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", port), SessionHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}/dashboard.html"
    return server, thread, url


def image_name(capture_id: Optional[int], suffix: str) -> Optional[str]:
    if capture_id is None:
        return None
    return f"{capture_id:06d}_{suffix}.png"


def meta_name(capture_id: Optional[int]) -> Optional[str]:
    if capture_id is None:
        return None
    return f"{capture_id:06d}_meta.json"


def best_display_frame(state: Optional[mc.ScreenState]) -> Optional[mc.FrameState]:
    if state is None:
        return None
    return state.frame or state.candidate_frame


def frame_status_key(frame: Optional[mc.FrameState]) -> Optional[tuple]:
    if frame is None:
        return None
    return (
        tuple(tuple(row) for row in frame.board),
        frame.preview_label,
    )


def write_live_view(
    recorder: HarnessRecorder,
    state: mc.ScreenState,
    display_frame: Optional[mc.FrameState],
    view_id: int,
) -> Dict[str, object]:
    full_path = recorder.session_dir / "live_full.png"
    board_path = recorder.session_dir / "live_board.png"
    overlay_path = recorder.session_dir / "live_board_overlay.png"
    preview_path = recorder.session_dir / "live_preview.png"
    meta_path = recorder.session_dir / "live_meta.json"

    Image.fromarray(state.arr).save(full_path)
    live_info: Dict[str, object] = {
        "live_view_id": view_id,
        "display_image": full_path.name,
        "display_mode": "full",
        "live_full": full_path.name,
        "live_board": None,
        "live_board_overlay": None,
        "live_preview": None,
        "live_meta": meta_path.name,
    }

    meta: Dict[str, object] = {
        "scene": state.scene,
        "scene_score": state.scene_score,
        "scene_scores": state.scene_scores,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if state.frame is not None and display_frame is not None:
        board_roi, _board_box = ws.find_board_roi(state.arr)
        preview_roi, _preview_box = ws.find_preview_roi(state.arr)
        Image.fromarray(board_roi).save(board_path)
        Image.fromarray(preview_roi).save(preview_path)
        color_boxes, _grid_meta = ws._board_cell_boxes(
            board_roi, inset_ratio=ws.CLASSIFY_INSET_RATIO
        )
        overlay_boxes = [(row, col, outer_box) for row, col, outer_box, _inner_box in color_boxes]
        ws._draw_board_overlay(board_roi, overlay_boxes).save(overlay_path)
        live_info.update(
            {
                "display_image": overlay_path.name,
                "display_mode": "board_overlay",
                "live_board": board_path.name,
                "live_board_overlay": overlay_path.name,
                "live_preview": preview_path.name,
            }
        )
        meta.update(
            {
                "board": display_frame.board,
                "preview_label": display_frame.preview_label,
                "preview_debug": display_frame.preview_debug,
            }
        )
    meta_path.write_text(json.dumps(meta, indent=2))
    return live_info


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
    live_info: Optional[Dict[str, object]] = None,
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
        "latest_meta": meta_name(latest_capture),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "failure_reasons": failure_reasons or [],
        "direction_sequence": event.get("direction_sequence", []) if event else [],
        "recovered_missed_moves": event.get("recovered_missed_moves", 0) if event else 0,
        "event": event,
    }
    if live_info:
        payload.update(live_info)
    return payload


def main() -> None:
    args = parse_args()
    window_id, window_info = mc.resolve_window(args.window_id, args.auto_window_prefix)
    recorder = HarnessRecorder(args.dataset_dir, window_info=window_info)
    dashboard_server, dashboard_thread, dashboard_url = start_dashboard_server(
        recorder.session_dir,
        args.dashboard_port,
    )
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
        live_view_id = 0

        def publish_status(
            *,
            run_state: str,
            message: str,
            scene: Optional[str],
            game_index: int,
            move_index: int,
            state: Optional[mc.ScreenState] = None,
            frame: Optional[mc.FrameState] = None,
            snapshot: Optional[tuple] = None,
            capture_id: Optional[int] = None,
            event: Optional[Dict[str, object]] = None,
            failure_reasons: Optional[List[str]] = None,
        ) -> None:
            nonlocal live_view_id
            display_frame = frame if frame is not None else best_display_frame(state)
            live_info = None
            if state is not None:
                live_view_id += 1
                live_info = write_live_view(recorder, state, display_frame, live_view_id)
            recorder.write_status(
                status_payload(
                    recorder,
                    run_state=run_state,
                    message=message,
                    scene=scene,
                    game_index=game_index,
                    move_index=move_index,
                    frame=display_frame,
                    snapshot=snapshot,
                    capture_id=capture_id,
                    event=event,
                    failure_reasons=failure_reasons,
                    live_info=live_info,
                )
            )

        while True:
            state = mc.capture_screen_state(window_id, args.capture_backend)
            status_frame = best_display_frame(state)

            status_key = (
                state.scene,
                tracked,
                move_index,
                game_index,
                last_capture_id,
                frame_status_key(status_frame),
            )
            if state.scene in (mc.SCENE_SCREEN_OFF, mc.SCENE_PHONE_IN_USE):
                if status_key != last_status_key:
                    publish_status(
                        run_state="waiting_for_device",
                        message=f"Waiting for mirrored device to become ready: {state.scene}",
                        scene=state.scene,
                        game_index=game_index,
                        move_index=move_index,
                        state=state,
                        capture_id=last_capture_id,
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
                publish_status(
                    run_state="game_end",
                    message=f"Observed {state.scene} after {move_index} moves. Waiting for the next game.",
                    scene=state.scene,
                    game_index=game_index,
                    move_index=move_index,
                    state=state,
                    capture_id=last_capture_id,
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
                    publish_status(
                        run_state="waiting_for_game",
                        message=f"Waiting for a game board. Current scene: {state.scene}.",
                        scene=state.scene,
                        game_index=game_index,
                        move_index=move_index,
                        state=state,
                        capture_id=last_capture_id,
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
                    if status_key != last_status_key:
                        publish_status(
                            run_state="waiting_for_fresh_game",
                            message=f"Waiting for a fresh board: {initial_error}",
                            scene=state.scene,
                            game_index=game_index,
                            move_index=move_index,
                            state=state,
                            frame=settled,
                            capture_id=last_capture_id,
                        )
                        last_status_key = status_key
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
                publish_status(
                    run_state="tracking",
                    message="Tracking live game state.",
                    scene=state.scene,
                    game_index=game_index,
                    move_index=move_index,
                    state=state,
                    frame=settled,
                    snapshot=last_snapshot,
                    capture_id=last_capture_id,
                )
                last_status_key = None
                time.sleep(args.poll)
                continue

            if last_stable_frame is None or same_semantics(last_stable_frame, settled):
                if args.idle_timeout > 0 and time.time() - last_move_ts > args.idle_timeout:
                    publish_status(
                        run_state="idle_timeout",
                        message=f"Idle timeout reached with no observed move for {args.idle_timeout:.1f}s.",
                        scene=state.scene,
                        game_index=game_index,
                        move_index=move_index,
                        state=state,
                        frame=last_stable_frame,
                        snapshot=last_snapshot,
                        capture_id=last_capture_id,
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
                publish_status(
                    run_state="failure",
                    message="Invalid tracked state detected.",
                    scene=state.scene,
                    game_index=game_index,
                    move_index=move_index,
                    state=state,
                    frame=settled,
                    snapshot=last_snapshot,
                    capture_id=capture_id,
                    event=event,
                    failure_reasons=failure_reasons,
                )
                print(f"Invalid state detected: {failure_reasons}", flush=True)
                print(f"Artifacts saved to {recorder.session_dir}", flush=True)
                if args.stop_on_failure:
                    return
                print("Re-seeding observer from the current board and continuing.", flush=True)
                tracked = False
                last_stable_frame = None
                last_snapshot = None
                stable_state = None
                stable_count = 0
                last_move_ts = time.time()
                time.sleep(args.poll)
                continue

            next_snapshot = event["preview_check"].get("next_snapshot")
            last_snapshot = next_snapshot if isinstance(next_snapshot, tuple) else next_snapshot
            last_stable_frame = frame_with_board(settled, event.get("after_board"))
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
            repair = event.get("board_repair")
            if isinstance(repair, dict):
                repaired_cells = repair.get("repaired_cells") or []
                cell_text = ", ".join(
                    f"({cell['row']},{cell['col']}): {cell['observed']} -> {cell['expected']}"
                    for cell in repaired_cells
                    if isinstance(cell, dict)
                )
                if cell_text:
                    print(f"repaired board read at move {move_index}: {cell_text}", flush=True)
            print_frame(last_stable_frame, last_snapshot)
            print(flush=True)
            publish_status(
                run_state="tracking",
                message="Tracking live game state.",
                scene=state.scene,
                game_index=game_index,
                move_index=move_index,
                state=state,
                frame=settled,
                snapshot=last_snapshot,
                capture_id=capture_id,
                event=event,
            )
            last_status_key = None

            if move_index >= args.max_moves:
                publish_status(
                    run_state="max_moves",
                    message=f"Reached max observed moves ({args.max_moves}).",
                    scene=state.scene,
                    game_index=game_index,
                    move_index=move_index,
                    state=state,
                    frame=settled,
                    snapshot=last_snapshot,
                    capture_id=capture_id,
                )
                print(f"Reached max observed moves ({args.max_moves}).", flush=True)
                print(f"Artifacts saved to {recorder.session_dir}", flush=True)
                return

            time.sleep(args.poll)
    finally:
        dashboard_server.shutdown()
        dashboard_server.server_close()
        dashboard_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()
