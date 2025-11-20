import argparse
import colorsys
import os
import queue
import subprocess
import tempfile
import time
import threading
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps

from preview_detector import (
    COLOR_PROTOTYPES,
    board_signature,
    board_signature_diff,
    classify_array,
    find_board_roi,
    find_preview_roi,
)

try:
    import Quartz

    HAVE_QUARTZ = True
except Exception:
    HAVE_QUARTZ = False

KEYCODES = {
    123: "left",
    124: "right",
    125: "down",
    126: "up",
    6: "undo",   # z
    12: "reset",  # q
}


class TileCycle:
    """
    Tracks small-tile pool (12 small tiles: 4 red/4 blue/4 gray) and large-tile pool
    (1 large per span of 24 small tiles). Provides next-tile probabilities.
    """

    def __init__(self) -> None:
        self.small_counts: Dict[str, int] = {"red": 4, "blue": 4, "gray": 4}
        self.small_pos = 0  # small tiles seen in current 12-span
        self.large_remaining = 1
        self.large_pos = 0  # small tiles seen in current 24-span

    def _reset_small(self) -> None:
        self.small_counts = {"red": 4, "blue": 4, "gray": 4}
        self.small_pos = 0

    def _reset_large(self) -> None:
        self.large_remaining = 1
        self.large_pos = 0

    def update(self, label: str) -> None:
        """
        Record an observed tile label (red/blue/gray/large_candidates/unknown).
        Advances cycle positions and updates remaining pools.
        Small/large spans advance only on non-large tiles (large tiles are inserted in addition).
        """
        is_large = label == "large_candidates"
        count_as_small = not is_large  # unknowns count as small to keep cycles moving.
        if count_as_small:
            self.small_pos += 1
            self.large_pos += 1

        if label in self.small_counts and self.small_counts[label] > 0:
            self.small_counts[label] -= 1

        if is_large and self.large_remaining > 0:
            self.large_remaining -= 1

        if self.small_pos >= 12:
            self._reset_small()
        if self.large_pos >= 24:
            self._reset_large()

    def probabilities(self) -> Dict[str, float]:
        """
        Return probabilities for next tile: red, blue, gray, large.
        Uses remaining counts divided by remaining slots in the current span.
        """
        small_slots_left = max(1, 12 - self.small_pos)
        large_slots_left = max(1, 24 - self.large_pos)

        probs: Dict[str, float] = {}
        for color, remaining in self.small_counts.items():
            probs[color] = remaining / small_slots_left

        probs["large_candidates"] = self.large_remaining / large_slots_left
        return probs

    def large_probability(self) -> float:
        """Probability that the next tile is large."""
        large_slots_left = max(1, 24 - self.large_pos)
        return self.large_remaining / large_slots_left

    def snapshot(self) -> Tuple[Dict[str, int], int, int, int]:
        return (self.small_counts.copy(), self.small_pos, self.large_remaining, self.large_pos)

    def restore(self, snapshot: Tuple[Dict[str, int], int, int, int]) -> None:
        counts, s_pos, large_rem, l_pos = snapshot
        self.small_counts = counts.copy()
        self.small_pos = s_pos
        self.large_remaining = large_rem
        self.large_pos = l_pos


def format_state(tile_cycle: TileCycle) -> str:
    """
    Return a human-readable small-pool listing plus large probability.
    Example: pool[3]: R R B | P(large)=4.2%
    """
    label_tokens = [
        ("red", "🟥"),
        ("blue", "🟦"),
        ("gray", "⬜️"),
    ]
    tokens: list[str] = []
    for key, tok in label_tokens:
        tokens.extend([tok] * max(0, tile_cycle.small_counts.get(key, 0)))
    pool_str = " ".join(tokens) if tokens else "(empty)"
    remaining = len(tokens)
    large_prob = tile_cycle.large_probability() * 100.0
    return f"pool[{remaining}]: {pool_str} | P(large)={large_prob:.1f}%"


# ---------- Board state classification ----------


SMALL_COLOR_MAP = {
    "red": "🟥",
    "blue": "🟦",
}
CELL_GRAY_TOKEN = "3"
TOKEN_EMPTY = "·"
TOKEN_OTHER = "X"
BOARD_COLOR_PROTOTYPES = {
    # Calibrated prototypes for board tiles (use center patch means).
    "red": np.array([231.84, 123.65, 141.79]),   # from provided red samples
    "blue": np.array([132.33, 198.81, 243.71]),  # from provided blue samples
}
RED_HUE_TARGET = 0.86  # calibrated from provided red tiles (mean hue)
RED_HUE_BAND = 0.12
THREE_MSE_THRESHOLD = 0.12
BLUE_HUE_TARGET = 0.52
THREE_TEMPLATE_FILES = [
    Path("out_tiles/tile_r0_c0.png"),
    Path("out_tiles/tile_r1_c1.png"),
    Path("out_tiles/tile_r2_c2.png"),
    Path("out_tiles/tile_r3_c0.png"),
]
THREE_MSE_THRESHOLD = 0.12


def cell_binarize(cell: np.ndarray, thresh: int = 140) -> np.ndarray:
    """Return a binarized 64x64 mask of the foreground glyph inside a tile."""
    try:
        resample = Image.Resampling.BILINEAR
    except AttributeError:
        resample = Image.BILINEAR
    gray = Image.fromarray(cell).convert("L")
    # crop margins to avoid tile borders
    w, h = gray.size
    margin_w = int(w * 0.15)
    margin_h = int(h * 0.15)
    gray = gray.crop((margin_w, margin_h, w - margin_w, h - margin_h))
    gray = gray.resize((64, 64), resample=resample)
    arr = np.array(gray, dtype=np.uint8)
    # Text is darker than background; invert threshold.
    mask = (arr < thresh).astype(np.uint8)
    return mask


@lru_cache(maxsize=1)
def three_template() -> Optional[np.ndarray]:
    """Average mask for '3' glyph from sample tiles if available."""
    masks = []
    for path in THREE_TEMPLATE_FILES:
        if path.exists():
            try:
                arr = np.array(Image.open(path).convert("RGB"))
                masks.append(cell_binarize(arr))
            except Exception:
                continue
    if not masks:
        return None
    return np.mean(masks, axis=0)


@lru_cache(maxsize=None)
def template_for_text(text: str) -> np.ndarray:
    """Render a text template mask at 64x64 for matching."""
    from PIL import ImageDraw, ImageFont

    canvas = Image.new("L", (64, 64), 0)
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("Arial.ttf", 38)
    except Exception:
        font = ImageFont.load_default()
    w, h = draw.textsize(text, font=font)
    x = (64 - w) // 2
    y = (64 - h) // 2
    draw.text((x, y), text, fill=255, font=font)
    return (np.array(canvas) > 0).astype(np.uint8)


NUMERIC_CANDIDATES = [
    "1",
    "2",
    "3",
    "6",
    "12",
    "24",
    "48",
    "96",
    "192",
    "384",
    "768",
    "1536",
]


def match_numeric(cell: np.ndarray) -> str:
    """Return the best-matching numeric label from templates."""
    mask = cell_binarize(cell)
    best = None
    best_score = float("inf")
    for cand in NUMERIC_CANDIDATES:
        tpl = template_for_text(cand)
        # Use mean squared error between masks.
        diff = (mask.astype(np.float32) - tpl.astype(np.float32)) ** 2
        score = diff.mean()
        if score < best_score:
            best_score = score
            best = cand
    return best or "?"


def classify_cell(cell: np.ndarray) -> str:
    """
    Classify a single tile cell into: red, blue, empty (·), or gray-as-X.
    No OCR is used.
    """
    # Trim a small margin to avoid borders.
    h, w, _ = cell.shape
    mh = int(h * 0.08)
    mw = int(w * 0.08)
    cell = cell[mh : h - mh, mw : w - mw]

    # Empty if overall very dark.
    if cell.reshape(-1, 3).mean() < 50:
        return TOKEN_EMPTY

    # Center patch mean color.
    ch, cw, _ = cell.shape
    patch = cell[int(ch * 0.2) : int(ch * 0.8), int(cw * 0.2) : int(cw * 0.8)]
    patch_mean = patch.reshape(-1, 3).mean(axis=0)

    # Prototype distance for red/blue.
    red_dist = float(np.linalg.norm(patch_mean - BOARD_COLOR_PROTOTYPES["red"]))
    blue_dist = float(np.linalg.norm(patch_mean - BOARD_COLOR_PROTOTYPES["blue"]))
    if red_dist < 60 and red_dist < blue_dist:
        return SMALL_COLOR_MAP["red"]
    if blue_dist < 40:
        return SMALL_COLOR_MAP["blue"]

    # Everything else is gray => X.
    return TOKEN_OTHER


def preprocess_for_ocr(cell: np.ndarray) -> Image.Image:
    """
    Prepare a cell image for tesseract:
    - center crop to avoid borders
    - grayscale, invert (dark text on light)
    - contrast/brightness normalize
    - upscale
    """
    h, w, _ = cell.shape
    mh = int(h * 0.1)
    mw = int(w * 0.1)
    cropped = cell[mh : h - mh, mw : w - mw]
    img = Image.fromarray(cropped).convert("L")
    img = ImageOps.invert(img)
    img = ImageOps.autocontrast(img)
    try:
        resample = Image.Resampling.BILINEAR  # Pillow >=9.1
    except AttributeError:
        resample = Image.BILINEAR
    img = img.resize((240, 240), resample=resample)
    return img


def ocr_cell(cell: np.ndarray) -> str:
    """Run tesseract CLI on the cell; return digits or TOKEN_OTHER."""
    img = preprocess_for_ocr(cell)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        img.save(tmp.name)
        tmp_path = tmp.name
    try:
        cmd = [
            "tesseract",
            tmp_path,
            "stdout",
            "--psm",
            "10",  # single character
            "-c",
            "tessedit_char_whitelist=0123456789",
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        text = out.decode(errors="ignore").strip()
        text = "".join(ch for ch in text if ch.isdigit())
        return text if text else TOKEN_OTHER
    except Exception:
        return TOKEN_OTHER
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
def classify_board(arr: np.ndarray) -> List[List[str]]:
    """
    Return a 4x4 grid of classified cells using fixed 1/4 splits inside the board ROI.
    A small inset is removed inside each cell to avoid tile borders.
    """
    roi, _box = find_board_roi(arr)
    h, w, _ = roi.shape
    # Tiles are slightly taller than wide; bias the vertical split spacing.
    cell_h = h / 4.0
    cell_w = w / 4.05  # small squeeze to reflect narrower width
    inset_y = 0.0
    inset_x = 0.0

    xs = [int(c * cell_w) for c in range(5)]
    ys = [int(r * cell_h) for r in range(5)]

    grid: List[List[str]] = []
    for r in range(4):
        row: List[str] = []
        y0, y1 = ys[r], ys[r + 1]
        y0i = int(y0 + inset_y)
        y1i = int(y1 - inset_y)
        for c in range(4):
            x0, x1 = xs[c], xs[c + 1]
            x0i = int(x0 + inset_x)
            x1i = int(x1 - inset_x)
            cell = roi[y0i:y1i, x0i:x1i]
            row.append(classify_cell(cell))
        grid.append(row)
    return grid


def segment_board_cells(arr: np.ndarray, inset_ratio: float = 0.0) -> List[Tuple[int, int, np.ndarray]]:
    """
    Return list of (row, col, cell_array) using fixed 1/4 splits inside the board ROI.
    """
    roi, _box = find_board_roi(arr)
    h, w, _ = roi.shape
    cell_h = h / 4.0
    cell_w = w / 4.05
    inset_y = cell_h * inset_ratio
    inset_x = cell_w * inset_ratio

    xs = [int(c * cell_w) for c in range(5)]
    ys = [int(r * cell_h) for r in range(5)]

    cells: List[Tuple[int, int, np.ndarray]] = []
    for r in range(4):
        y0, y1 = ys[r], ys[r + 1]
        y0i = int(y0 + inset_y)
        y1i = int(y1 - inset_y)
        for c in range(4):
            x0, x1 = xs[c], xs[c + 1]
            x0i = int(x0 + inset_x)
            x1i = int(x1 - inset_x)
            cell = roi[y0i:y1i, x0i:x1i]
            cells.append((r, c, cell))
    return cells


def format_board(board: List[List[str]]) -> str:
    return "\n".join(" ".join(row) for row in board)


def list_windows_cg() -> List[Tuple[int, str, str]]:
    """
    Enumerate on-screen windows using CoreGraphics (preferred).
    Returns list of (window_id, app_name, window_title), ordered front-to-back.
    """
    if not HAVE_QUARTZ:
        return []
    options = (
        Quartz.kCGWindowListOptionOnScreenOnly
        | Quartz.kCGWindowListExcludeDesktopElements
    )
    info = Quartz.CGWindowListCopyWindowInfo(options, Quartz.kCGNullWindowID)
    windows: List[Tuple[int, str, str]] = []
    for win in info:
        try:
            wid = int(win.get("kCGWindowNumber"))
            owner = win.get("kCGWindowOwnerName", "")
            name = win.get("kCGWindowName", "")
            windows.append((wid, owner, name))
        except Exception:
            continue
    return windows


def get_frontmost_window() -> Tuple[int, str, str]:
    """
    Return (window_id, app_name, window_title) of the frontmost window via CoreGraphics,
    falling back to System Events if needed.
    """
    # CoreGraphics frontmost is first entry.
    wins = list_windows_cg()
    if wins:
        return wins[0]

    # Fallback to AppleScript (may fail if app hides window ids).
    script = r'''
    tell application "System Events"
        set frontProc to first process whose frontmost is true
        set winId to id of first window of frontProc
        set winName to name of first window of frontProc
        set procName to name of frontProc
        return (winId as text) & tab & procName & tab & winName
    end tell
    '''
    out = subprocess.check_output(["osascript", "-e", script]).decode().strip()
    parts = out.split("\t", 2)
    if len(parts) != 3:
        raise RuntimeError(f"Could not parse window info: {out!r}")
    return int(parts[0]), parts[1], parts[2]


def list_windows() -> list[Tuple[int, str, str]]:
    """
    Return a list of (window_id, app_name, window_title) for all windows
    that report an id via System Events. Requires Accessibility permissions
    for the terminal/IDE.
    """
    script = r'''
    set outLines to {}
    tell application "System Events"
        set procs to application processes
        repeat with p in procs
            set procName to name of p
            try
                set wins to windows of p
                repeat with w in wins
                    try
                        set winId to id of w
                        set winName to name of w
                        set end of outLines to (winId as text) & tab & procName & tab & winName
                    end try
                end repeat
            end try
        end repeat
    end tell
    return outLines as text
    '''
    out = subprocess.check_output(["osascript", "-e", script]).decode().strip()
    windows: list[Tuple[int, str, str]] = []
    if not out:
        return windows
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        try:
            wid = int(parts[0])
        except ValueError:
            continue
        windows.append((wid, parts[1], parts[2]))
    return windows


def choose_window_interactive() -> Tuple[int, str, str]:
    """
    Prompt user with a numbered list of windows and return the chosen one.
    """
    if not HAVE_QUARTZ:
        print("Quartz (pyobjc) not available; install with: pip install pyobjc-framework-Quartz")
    wins = list_windows_cg()
    if not wins:
        try:
            wins = list_windows()
        except Exception as exc:  # noqa: BLE001
            print(f"Could not list windows via System Events: {exc}")
            wins = []
    if not wins:
        print(
            "No windows found via System Events. "
            "Ensure your terminal has Accessibility permission. "
            "Falling back to frontmost window capture."
        )
        try:
            input(
                "Press Enter, then within 3 seconds click/focus the Threes window. "
                "We'll grab the frontmost window after the delay..."
            )
        except EOFError:
            print("Input unavailable; attempting immediate frontmost window capture.")
        time.sleep(3.0)
        try:
            return get_frontmost_window()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("Could not capture frontmost window") from exc
    print("Select a window to monitor:")
    for idx, (wid, app, title) in enumerate(wins):
        display_title = title if title else "(untitled)"
        print(f"[{idx}] {display_title}   —   {app}   (id={wid})")
    while True:
        try:
            choice = input("Enter a number: ").strip()
            idx = int(choice)
            if 0 <= idx < len(wins):
                return wins[idx]
        except Exception:
            pass
        print("Invalid selection, try again.")


def find_window_by_prefix(prefix: str) -> Optional[Tuple[int, str, str]]:
    """Return the first window whose title starts with the given prefix."""
    if not prefix:
        return None
    wins = list_windows_cg()
    candidates = [(w, a, t) for (w, a, t) in wins if t and t.startswith(prefix)]
    if not candidates:
        try:
            wins = list_windows()
            candidates = [(w, a, t) for (w, a, t) in wins if t and t.startswith(prefix)]
        except Exception:
            candidates = []
    if candidates:
        return candidates[0]
    return None


def capture_window(window_id: int) -> Image.Image:
    """
    Capture a screenshot of the given window_id using macOS screencapture.
    Returns a PIL Image in RGB mode.
    """
    if HAVE_QUARTZ:
        try:
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
        except Exception:
            # Fall through to screencapture below.
            pass

    fd, path = tempfile.mkstemp(suffix=".png")
    try:
        subprocess.run(["screencapture", "-x", "-l", str(window_id), path], check=True)
        img = Image.open(path).convert("RGB")
    finally:
        try:
            import os
            os.remove(path)
        except OSError:
            pass
    return img


def stream_labels(
    window_id: int,
    poll_seconds: float,
    print_all: bool,
    board_delta_threshold: float,
) -> None:
    prev_label: Optional[str] = None
    prev_board_sig: Optional[np.ndarray] = None
    tile_cycle = TileCycle()
    while True:
        try:
            img = capture_window(window_id)
            arr = np.array(img)
            label, _debug = classify_array(arr)
            curr_board_sig, _box = board_signature(arr)
            board_delta = None
            if prev_board_sig is None:
                board_changed = True
            else:
                board_delta = board_signature_diff(prev_board_sig, curr_board_sig)
                board_changed = board_delta > board_delta_threshold
        except Exception as exc:  # noqa: BLE001
            print(f"[error] {exc}")
            time.sleep(poll_seconds)
            continue

        should_print = print_all or board_changed or label != prev_label
        if should_print:
            tile_cycle.update(label)
            state_str = format_state(tile_cycle)
            ts = time.strftime("%H:%M:%S")
            delta_str = f" boardΔ={board_delta:.3f}" if board_delta is not None else ""
            print(f"[{ts}]{delta_str} {state_str}")
            prev_label = label
        prev_board_sig = curr_board_sig
        time.sleep(poll_seconds)


def start_key_listener() -> "queue.Queue[Tuple[float, str]]":
    """
    Start a background listener for keydown events (arrows, z, q) via Quartz event tap.
    Returns a queue of (timestamp, key_name) entries.
    """
    if not HAVE_QUARTZ:
        raise RuntimeError("Arrow listener needs Quartz (pyobjc-framework-Quartz).")

    evt_mask = Quartz.CGEventMaskBit(Quartz.kCGEventKeyDown)
    q: "queue.Queue[Tuple[float, str]]" = queue.Queue()

    def handler(proxy, type_, event, _refcon):
        if type_ != Quartz.kCGEventKeyDown:
            return event
        keycode = int(Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventKeycode))
        key_name = KEYCODES.get(keycode)
        if key_name:
            q.put((time.time(), key_name))
        return event

    tap = Quartz.CGEventTapCreate(
        Quartz.kCGSessionEventTap,
        Quartz.kCGHeadInsertEventTap,
        Quartz.kCGEventTapOptionListenOnly,
        evt_mask,
        handler,
        None,
    )
    if not tap:
        raise RuntimeError("Could not create event tap (check Input Monitoring/Accessibility permissions).")
    run_loop_source = Quartz.CFMachPortCreateRunLoopSource(None, tap, 0)

    def run_loop():
        loop = Quartz.CFRunLoopGetCurrent()
        Quartz.CFRunLoopAddSource(loop, run_loop_source, Quartz.kCFRunLoopCommonModes)
        Quartz.CGEventTapEnable(tap, True)
        Quartz.CFRunLoopRun()

    threading.Thread(target=run_loop, daemon=True).start()
    return q


def dump_board_state(window_id: int) -> None:
    """
    Capture the current window and print the 4x4 board plus preview label.
    """
    try:
        arr = np.array(capture_window(window_id))
    except Exception as exc:  # noqa: BLE001
        print(f"[error] {exc}")
        return
    board = classify_board(arr)
    preview_label, _debug = classify_array(arr)
    print("Board state:")
    print(format_board(board))
    print(f"Preview: {preview_label}")


def dump_tiles(window_id: int, out_dir: str) -> None:
    """
    Capture once, dump 16 board tiles (no inset) and the preview crop to out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)
    try:
        arr = np.array(capture_window(window_id))
    except Exception as exc:  # noqa: BLE001
        print(f"[error] {exc}")
        return

    cells = segment_board_cells(arr, inset_ratio=0.0)
    for r, c, cell in cells:
        Image.fromarray(cell).save(os.path.join(out_dir, f"tile_r{r}_c{c}.png"))

    # Save preview crop.
    preview_roi, _ = find_preview_roi(arr)
    Image.fromarray(preview_roi).save(os.path.join(out_dir, "preview.png"))
    print(f"Dumped tiles to {out_dir}")


def stream_labels_on_keys(
    window_id: int,
    arrow_delay: float,
) -> None:
    """
    Capture/detect on arrow key presses; support undo (z) and reset (q).
    Requires Quartz event tap permissions.
    """
    events = start_key_listener()
    tile_cycle = TileCycle()
    history: list[Tuple[Dict[str, int], int, int, int]] = []

    def capture_and_update(ts_event: float, is_reset: bool = False, apply_delay: bool = False) -> None:
        nonlocal tile_cycle
        try:
            if apply_delay and arrow_delay > 0:
                time.sleep(arrow_delay)
            arr = np.array(capture_window(window_id))
            label, _debug = classify_array(arr)
        except Exception as exc:  # noqa: BLE001
            print(f"[error] {exc}")
            return
        if not is_reset:
            history.append(tile_cycle.snapshot())
        tile_cycle.update(label)
        ts = time.strftime("%H:%M:%S", time.localtime(ts_event))
        print(f"[{ts}] {format_state(tile_cycle)}")

    # Initial capture to seed the cycle with the first visible tile.
    capture_and_update(time.time(), is_reset=True, apply_delay=False)

    while True:
        ts_event, key = events.get()
        if key == "undo":
            if history:
                snapshot = history.pop()
                tile_cycle.restore(snapshot)
                ts = time.strftime("%H:%M:%S", time.localtime(ts_event))
                print(f"[{ts}] undo -> {format_state(tile_cycle)}")
            continue
        if key == "reset":
            tile_cycle = TileCycle()
            history.clear()
            capture_and_update(ts_event, is_reset=True, apply_delay=False)
            continue

        # Arrow key
        capture_and_update(ts_event, is_reset=False, apply_delay=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Continuously detect the Threes preview from a chosen window."
    )
    parser.add_argument(
        "--window-id",
        type=int,
        help="Window ID to monitor (otherwise the current frontmost window is used).",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=0.4,
        help="Polling interval between detections.",
    )
    parser.add_argument(
        "--print-all",
        action="store_true",
        help="Print every polling result (default only prints changes).",
    )
    parser.add_argument(
        "--board-delta-threshold",
        type=float,
        default=0.3,
        help="Mean absolute difference on the low-res board signature to flag a board change.",
    )
    parser.add_argument(
        "--arrow-delay",
        type=float,
        default=0.1,
        help="Delay (seconds) after an arrow key press before capture, to let the board update.",
    )
    parser.add_argument(
        "--poll",
        action="store_true",
        help="Use polling/board-change trigger instead of key-capture (default uses arrow keys).",
    )
    parser.add_argument(
        "--board-once",
        action="store_true",
        help="Capture once and print the 4x4 board plus preview, then exit.",
    )
    parser.add_argument(
        "--auto-window-prefix",
        default="iPhone Mirroring",
        help="Automatically select the first window whose title starts with this prefix. Leave empty to skip.",
    )
    parser.add_argument(
        "--dump-tiles",
        type=str,
        help="Dump the 16 board tiles (and preview) to this directory and exit.",
    )
    args = parser.parse_args()

    if args.window_id:
        window_id = args.window_id
        app_name = "(provided id)"
        win_name = ""
    else:
        auto_pick = find_window_by_prefix(args.auto_window_prefix or "")
        if auto_pick:
            window_id, app_name, win_name = auto_pick
            print(f"Auto-selected window {window_id} ({app_name} – {win_name})")
        else:
            window_id, app_name, win_name = choose_window_interactive()

    if args.dump_tiles:
        print(f"Capturing tiles once from window {window_id} ({app_name} – {win_name})")
        dump_tiles(window_id, args.dump_tiles)
        return
    if args.board_once:
        print(f"Capturing board once from window {window_id} ({app_name} – {win_name})")
        dump_board_state(window_id)
        return

    print(f"Monitoring window {window_id} ({app_name} – {win_name})")
    if args.poll:
        print(f"Polling every {args.poll_seconds}s. Ctrl+C to stop.")
        stream_labels(
            window_id,
            args.poll_seconds,
            args.print_all,
            args.board_delta_threshold,
        )
    else:
        print("Trigger: arrow-key captures (default). Ctrl+C to stop.")
        stream_labels_on_keys(
            window_id,
            args.arrow_delay,
        )


if __name__ == "__main__":
    main()
