Threes helper – preview detector
================================

What’s here
-----------
- `preview_detector.py`: screenshot-based detector for the “next tile” preview.
- `window_stream.py`: polls a macOS window and prints detected preview labels.
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
   - If you see “old” previews (race with animation), add a small delay before capture: `--arrow-delay 0.1` (default 0.1).
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

Gray tile labeling (build 3+ dataset):
```bash
python3 extract_gray_tiles.py --session session_YYYYMMDD_HHMMSS
```
   - Outputs `datasets/gray_labels/<session>/tiles/` with extracted gray tiles and `sheets/` contact sheets.
   - Fill `datasets/gray_labels/<session>/labels.csv` with numeric labels (3,6,12,24,48,96,192,384,768,1536).

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
   - If you see “old” previews (race with animation), add a small delay before capture: `--arrow-delay 0.1` (default 0.1).

Optional: if you already know the window id (from `osascript`/`yabai`/`chunkwm`), pass `--window-id <id>` to skip the chooser.

Fallback if no windows are listed
---------------------------------
If the chooser prints “No windows found”, macOS did not expose windows via System Events (usually an Accessibility permission issue). The script will then ask you to press Enter and, within 3 seconds, click/focus the Threes window; it will capture the frontmost window after that delay.

Next milestones
---------------
- Improve move/change detection (board diff or hotkey) to avoid fixed polling costs.
- Add the 12/24 counting engine and a minimal HUD.
