Threes helper – preview detector
================================

Current RL research status, failures, protected artifacts, and harness commands
are summarized in [`threes_rl/RL_PROGRAM_HANDOFF.md`](threes_rl/RL_PROGRAM_HANDOFF.md).
The current disk/RAM allocation is itemized in
[`threes_rl/STORAGE_AUDIT_20260815.md`](threes_rl/STORAGE_AUDIT_20260815.md).

Environment
-----------

This repo now uses `uv` for Python environment management and applies a one-week package age gate via `uv.toml`.

```bash
cd /Users/jeffharris/code/threeshelper
uv venv
uv pip sync requirements.lock.txt
```

Run commands inside the activated environment or prefix them with `uv run`.

What’s here
-----------
- `preview_detector.py`: screenshot-based detector for the “next tile” preview.
- `window_stream.py`: capture/classification core for live mirrored gameplay.
- `mirroring_control.py`: direct iPhone Mirroring input driver and scene routing.
- `hunt_invalid_states.py`: repeated self-play harness that records artifacts and stops on the first invalid tracked state.
- `observe_human_game.py`: observe-only harness for user-driven play that validates each settled move without sending input.
- Uses Pillow + NumPy (already present in this environment).

How static detection works
--------------------------
- Finds a preview crop near the upper-center of the screenshot using a saturation mask.
- Inside that crop, looks at high-saturation pixels:
  - Flags `large_candidates` if the colored pixels are spread wide but very flat vertically (the three-option large-tile band).
  - Otherwise classifies the dominant color vs prototypes (`red`, `blue`, `gray`).

Run the detector on the provided screenshots
--------------------------------------------
Paths have a narrow no-break-space before “PM”, so use a glob-friendly call:
```bash
python3 - <<'PY'
import glob, subprocess
for path in glob.glob('Screenshot*PM.png'):
    print(f'=== {path}')
    subprocess.run(['python3', 'preview_detector.py', '--image', path, '--debug'], check=False)
PY
```

Single-image run:
```bash
python3 preview_detector.py --image "Screenshot 2025-11-19 at 4.12.40 PM.png" --debug
```

Live window polling (milestone 2, early)
----------------------------------------
`window_stream.py` captures a macOS window repeatedly and prints detected preview labels.

Default (arrow-key) trigger:
```bash
python3 window_stream.py
```
   - Captures on every arrow key press (up/left/down/right) and prints remaining small tiles plus `P(large)` using colored squares (🟥/🟦/⬜️).
   - Hotkeys:
     - `Z`: undo the last move (useful if an arrow key was a no-op).
     - `Q`: reset to a fresh game; the current visible preview becomes tile 1 of the new 12-tile batch.
   - Requires Quartz (pyobjc-framework-Quartz) and likely Input Monitoring/Accessibility permission for your terminal/IDE to see key events.
   - If you see “old” previews (race with animation), add a small delay before capture: `--arrow-delay 0.1` (default 0.3).
   - Auto-selects the first window whose title starts with `iPhone Mirroring`. Override with `--auto-window-prefix ""` to disable or another prefix to target a different window.

Dataset capture + labeling (arrow-key mode):
```bash
python3 window_stream.py --record-dataset datasets
```
   - Creates `datasets/session_YYYYMMDD_HHMMSS/` with full screenshots, board crops, preview crops, and JSON metadata.
   - Adds `*_board_overlay.png` with the inferred grid and per-cell stats in `*_meta.json` for debugging.
   - Label keys (global): `C`=correct, `X`=incorrect, `U`=undo label.
   - Use arrow keys to play; each arrow capture becomes a dataset entry you can label.
   - Capture waits for the board to settle (defaults: 2 stable frames, threshold 0.15, timeout 1.0s). Tune with `--settle-*` flags if needed.

Direct iPhone Mirroring control
-------------------------------
`mirroring_control.py` drives the `iPhone Mirroring` window directly with native macOS mouse events.

Print the current parsed board/preview:
```bash
python3 mirroring_control.py
```

Print the currently detected UI scene:
```bash
python3 mirroring_control.py --scene
```

Start a new game from the title screen:
```bash
python3 mirroring_control.py --start-game
```

Start another game from the post-game screen:
```bash
python3 mirroring_control.py --retry-game
```

Send one swipe:
```bash
python3 mirroring_control.py --swipe left
```

Self-play from the current board:
```bash
python3 mirroring_control.py --autoplay --max-moves 200
```

Start from title and immediately self-play:
```bash
python3 mirroring_control.py --start-game --autoplay --max-moves 200
```

Drive to a known title or active-game state:
```bash
python3 mirroring_control.py --ensure-title
python3 mirroring_control.py --ensure-game
```

Abort an in-progress game back to the title screen:
```bash
python3 mirroring_control.py --exit-game
```

Observation and state hunts
---------------------------
Run the dedicated live debug dashboard:
```bash
python3 live_debug_server.py --attach-current-game
```
   - Serves a persistent dashboard on `http://127.0.0.1:55777/dashboard.html` by default.
   - Continuously captures the mirrored phone, even when the app is not on a playable board.
   - Shows the live screen, parsed board state, visible next cue, following-cue probabilities, tracker state, issues, and recent events in one place.
   - Writes the latest live state into `datasets/live_debug/session_YYYYMMDD_HHMMSS/` so the current frame and JSON remain inspectable from the filesystem.
   - Defaults to `--capture-backend screen`, which captures the visible on-screen region of the iPhone Mirroring window instead of the app window buffer.
   - `screen` is the preferred live-debug backend when the mirrored phone is visible and unobscured; it materially reduces stale-frame latency compared with window-buffer capture.
   - If the window is occluded or off-screen, fall back to `--capture-backend quartz` or `--capture-backend screencapture`.
   - The live server now auto-falls back if the requested backend becomes slow or starts failing, instead of staying stuck on a degraded capture path.
   - The live board panel now uses the fastest candidate board read immediately, while the tracker still waits for a legality-valid committed move before advancing internal state.

Watch a human-played game from the next fresh board with a live dashboard:
```bash
python3 observe_human_game.py
```
   - Opens a local browser dashboard and writes machine-readable state to `live_status.json` in the session directory.
   - Keeps watching across game-over/post-game screens so you can restart and keep playing in the same session.
   - If you play faster than the settle loop, it searches for short legal multi-move paths before declaring an invalid state.

Attach to whatever game is currently visible, even if it is already mid-game:
```bash
python3 observe_human_game.py --attach-current-game
```

Useful observer flags:
```bash
python3 observe_human_game.py --no-open-dashboard
python3 observe_human_game.py --max-recovery-depth 2
python3 observe_human_game.py --max-games 3
```

After recording a strong human session, import it into the Threes RL
diagnostic pipeline:
```bash
.venv/bin/python -m threes_rl.human_diagnostics_pipeline \
  --events-jsonl datasets/human_watch/<session>/events.jsonl \
  --policy-file threes_rl/current_incumbent_policy.txt \
  --out-dir threes_rl/runs/human_diagnostics/<session>
```
Or scan the human-watch inbox and process every new or changed session:
```bash
.venv/bin/python -m threes_rl.human_diagnostics_batch --run
```
   - The pipeline writes imported replay JSON/HTML, high-board reservoir
     records, pre-promotion transition windows, human-root support-ladder
     windows, policy-agreement diagnostics, and a no-label top-two scan.
   - The batch command writes `threes_rl/runs/human_diagnostics/human_diagnostics_batch.html`
     and skips sessions whose diagnostics are already current unless
     `--force` is supplied.
   - Good research input is at least five independent human games reaching
     non-starter `1536`, with one or more reaching `3072`.
   - Treat these artifacts as diagnostics first; do not promote a policy
     change until direct milestone labels are stable on held-out human roots.

Latency benchmarking:
```bash
python3 benchmark_live_latency.py --mode direct --capture-backend screen --input-method arrow
python3 benchmark_live_latency.py --mode server --server-url http://127.0.0.1:55777 --input-method arrow
```
   - `direct` measures the raw capture + legality path without the live server in the middle.
   - `server` measures end-to-end dashboard/API latency against a running `live_debug_server.py`.
   - For latency-sensitive control, `screen` capture plus arrow-key input is the best current path.

Run repeated self-play and stop on the first tracked invalid state:
```bash
python3 hunt_invalid_states.py --games 5 --start-from-title --move-order cycle
```

Useful notes:
- Works against the live `iPhone Mirroring` window; the game does not need to exist as a local app target.
- The driver now distinguishes `title`, `game`, `game_over`, `postgame`, `menu`, and `end_confirm` scenes before deciding which transition routine to run.
- If the visible board does not look like a fresh game, tile-cycle probability tracking is disabled automatically and the script still plays.
- Transition validation is legality-first:
  - every move is one of `up/down/left/right`
  - tiles shift by one square at most
  - merges are only `1+2 -> 3` or identical `3+` tiles
  - a tile merges at most once per move
  - the inserted tile appears in one of the eligible freed edge slots for that swipe direction
- `--start-game` waits for a valid fresh board before seeding the tracker.
- `--retry-game` handles the full end-of-game path: `game_over -> swipe to summary -> Retry -> new game`.
- `--exit-game` follows the full abort path: `menu -> MAIN MENU -> END GAME -> title`.
- `--tap-rel X Y` sends a single tap at relative window coordinates, which is intended for future game-over/new-game flows.
- `--capture-backend screencapture` is available if Quartz capture ever disagrees with what the detector should see.
- `observe_human_game.py` waits for a real fresh board by default, validates each settled move, recovers short missed human sequences when possible, and records rewindable artifacts in `datasets/human_watch/`.
- `hunt_invalid_states.py` uses the same validator and artifact format for autonomous repeated-play runs in `datasets/state_hunt/`.
- `live_debug_server.py` keeps the fast path single-frame, but on an otherwise invalid move it performs a short recapture/reconfirm loop before dropping state.
- Invalid-state recovery is non-blocking in the live server now: ambiguous transitions are retried across future frames instead of sleeping inside the main capture loop.
- `benchmark_live_latency.py` now waits for a genuinely settled board before each sample so latency numbers are not polluted by the previous move still animating.
- Mid-game attach is guarded: the tracker will not attach to obviously bogus boards such as empty grids, unknown-cell reads, or boards missing the required 48+ anchor tile.
- The dashboard `Start New Game` action now has two paths:
  - inside Threes: exit current game and start fresh
  - outside Threes: go home, search for `Threes`, launch it, then start fresh

Gray tile labeling (build 3+ dataset):
```bash
python3 extract_gray_tiles.py --session session_YYYYMMDD_HHMMSS
```
   - Outputs `datasets/gray_labels/<session>/tiles/` with extracted gray tiles and `sheets/` contact sheets.
   - Fill `datasets/gray_labels/<session>/labels.csv` with numeric labels (3,6,12,24,48,96,192,384,768,1536).
   - Optional web labeler:
     ```bash
     python3 build_gray_labeler.py --labels-dir datasets/gray_labels/<session>
     ```
     Then open `datasets/gray_labels/<session>/labeler/index.html` and click “Save CSV”.

Train 3-detector from labeled tiles:
```bash
python3 train_three_detector.py --labels-dir datasets/gray_labels/<session>
```
   - Writes `three_detector.json` used by the live classifier.
   - Uses clustered 3-templates for better recall without false positives.

Optional polling mode (board diff):
```bash
python3 window_stream.py --poll --poll-seconds 0.4
```
   - Prefers CoreGraphics (Quartz) to enumerate windows. If Quartz is missing, install: `pip install pyobjc-framework-Quartz`.
   - Accessibility permission still needed for AppleScript fallback; Screen Recording permission needed for `screencapture`.
   - Prints when the board changes (low-res board fingerprint diff) or when the detected label changes. Add `--print-all` to log every poll.
   - Tuning: the board-change threshold defaults to `0.3`. If you get extra prints (noise) raise it, if moves are missed lower it: `--board-delta-threshold 0.2`.
   - Auto-selects the first window whose title starts with `iPhone Mirroring`. Override with `--auto-window-prefix ""` to disable or another prefix to target a different window.

Board snapshot mode (one-shot board + preview):
```bash
python3 window_stream.py --board-once
```
   - Captures once and prints a 4x4 board with colored squares for small tiles (🟥/🟦), blanks as `·`, all other/grays as `X`, and the preview label.

Tile dump (segmentation debug):
```bash
python3 window_stream.py --dump-tiles out_tiles
```
   - Captures once and writes the 16 board tiles (and preview crop) into the given directory so you can inspect segmentation.
   - If auto window pick fails, the script will list windows for manual selection; if that also fails, it will try to grab the frontmost window without prompting.

Default (arrow-key) trigger:
```bash
python3 window_stream.py
```
   - Captures on every arrow key press (up/left/down/right) and prints remaining small tiles plus `P(large)`.
   - Hotkeys:
     - `Z`: undo the last move (useful if an arrow key was a no-op).
     - `Q`: reset to a fresh game; the current visible preview becomes tile 1 of the new 12-tile batch.
   - Requires Quartz (pyobjc-framework-Quartz) and likely Input Monitoring/Accessibility permission for your terminal/IDE to see key events.
   - If you see “old” previews (race with animation), add a small delay before capture: `--arrow-delay 0.1` (default 0.3).

Optional: if you already know the window id (from `osascript`/`yabai`/`chunkwm`), pass `--window-id <id>` to skip the chooser.

Fallback if no windows are listed
---------------------------------
If the chooser prints “No windows found”, macOS did not expose windows via System Events (usually an Accessibility permission issue). The script will then ask you to press Enter and, within 3 seconds, click/focus the Threes window; it will capture the frontmost window after that delay.

Next phase
----------
- Run longer human-observed play sessions to catch deeper invalid states with full artifacts.
- Extend self-play beyond the current move-order search into strategy iteration and evaluation.
