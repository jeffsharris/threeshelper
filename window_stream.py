import argparse
import colorsys
import json
import os
import queue
from collections import deque
import subprocess
import tempfile
import time
import threading
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

import gray_hog as gh
try:
    import imagehash
    HAVE_IMAGEHASH = True
except Exception:
    HAVE_IMAGEHASH = False
    imagehash = None

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
    8: "label_correct",  # c
    7: "label_incorrect",  # x
    32: "label_undo",  # u
}


SMALL_BAG_SIZE = 12
BONUS_TRIGGER_TILE = 48
LARGE_DELAY_SMALLS = 20
LARGE_PREVIEW_OFFSET = 1
LARGE_DELAY_PREVIEWS = LARGE_DELAY_SMALLS + LARGE_PREVIEW_OFFSET
LARGE_SPAN_SMALLS = 20
SMALL_TILE_VALUES = {
    "red": 2,
    "blue": 1,
    "gray": 3,
}


class TileCycle:
    """
    Tracks the 12-tile small bag (4 red / 4 blue / 4 gray) plus the large-tile
    schedule used by Threes once bonus tiles are unlocked.
    """

    def __init__(self) -> None:
        self.small_counts: Dict[str, int] = {"red": 4, "blue": 4, "gray": 4}
        self.small_pos = 0  # small tiles consumed in the current 12-tile bag
        self.small_seen_total = 0  # total small tiles consumed since game start
        self.span_small_pos = 0  # small previews consumed inside the active 20-small large span
        self.large_pending = False  # whether the current large span still owes one big tile
        self.max_tile = 0

    def _reset_small(self) -> None:
        self.small_counts = {"red": 4, "blue": 4, "gray": 4}
        self.small_pos = 0

    def _start_new_large_span(self) -> None:
        self.span_small_pos = 0
        self.large_pending = True

    def set_max_tile(self, max_tile: int) -> None:
        self.max_tile = max(0, int(max_tile))

    def update(self, label: str) -> None:
        """
        Record an observed preview label.

        Red/blue/gray/unknown previews advance the 12-tile bag and the global
        small-preview counter. Large-candidate previews do not consume a small
        tile, but they do satisfy the pending large in the current 20-small span.
        """
        is_large = label == "large_candidates"
        count_as_small = not is_large

        if count_as_small:
            self.small_pos += 1
            self.small_seen_total += 1
            if self.small_seen_total == LARGE_DELAY_PREVIEWS:
                self._start_new_large_span()
            elif self.small_seen_total > LARGE_DELAY_PREVIEWS:
                self.span_small_pos += 1

        if label in self.small_counts and self.small_counts[label] > 0:
            self.small_counts[label] -= 1

        if is_large and self.large_pending:
            self.large_pending = False

        if self.small_pos >= SMALL_BAG_SIZE:
            self._reset_small()

        if self.small_seen_total >= LARGE_DELAY_PREVIEWS:
            self.span_small_pos = min(self.span_small_pos, LARGE_SPAN_SMALLS)
            if not self.large_pending and self.span_small_pos >= LARGE_SPAN_SMALLS:
                self._start_new_large_span()

    def probabilities(self) -> Dict[str, float]:
        """
        Return actual next-cue probabilities.

        Small-tile odds are scaled by the chance that the next cue is not a large
        bundle, so the returned probabilities sum to 1.0.
        """
        small_slots_left = max(1, SMALL_BAG_SIZE - self.small_pos)
        large_prob = self.large_probability()
        small_scale = max(0.0, 1.0 - large_prob)

        probs: Dict[str, float] = {}
        for color, remaining in self.small_counts.items():
            probs[color] = (remaining / small_slots_left) * small_scale

        probs["large_candidates"] = large_prob
        return probs

    def large_probability(self) -> float:
        """Probability that the next preview is the large-tile bundle cue."""
        if self.max_tile < BONUS_TRIGGER_TILE:
            return 0.0
        if self.small_seen_total < LARGE_DELAY_PREVIEWS:
            return 0.0
        if not self.large_pending:
            return 0.0
        remaining_slots = max(1, LARGE_SPAN_SMALLS + 1 - self.span_small_pos)
        return 1.0 / remaining_slots

    def bonus_max_tile(self) -> int:
        """
        Largest big tile currently allowed by the board state.

        Threes limits spontaneous bonus tiles to two levels below the largest tile
        already on the board, so a 1536 board caps at 384.
        """
        if self.max_tile < BONUS_TRIGGER_TILE:
            return 0
        return max(6, self.max_tile // 4)

    def bonus_values(self) -> List[int]:
        """
        Return the ordered support for bonus tiles at the current board max.

        If the largest tile on the board is M, the support spans
        6, 12, 24, ... up to M/4.
        """
        if self.max_tile < BONUS_TRIGGER_TILE:
            return []
        values: List[int] = []
        value = 6
        limit = self.bonus_max_tile()
        while value <= limit:
            values.append(value)
            value *= 2
        return values

    def bonus_windows(self) -> List[List[int]]:
        """
        Return all equally likely contiguous 3-value bonus bundles.
        """
        values = self.bonus_values()
        if len(values) < 3:
            return []
        return [values[idx : idx + 3] for idx in range(len(values) - 2)]

    def bonus_value_probabilities(self) -> Dict[int, float]:
        """
        Return the conditional probability of each bonus value once a bonus bundle
        has been selected.
        """
        windows = self.bonus_windows()
        if not windows:
            return {}
        hits: Dict[int, int] = {value: 0 for value in self.bonus_values()}
        total_slots = len(windows) * 3
        for window in windows:
            for value in window:
                hits[value] += 1
        return {value: count / total_slots for value, count in hits.items() if count > 0}

    def large_schedule_note(self) -> str:
        if self.max_tile < BONUS_TRIGGER_TILE:
            return f"Big tiles stay locked until a {BONUS_TRIGGER_TILE} tile is on the board."
        if self.small_seen_total < LARGE_DELAY_PREVIEWS:
            remaining = LARGE_DELAY_PREVIEWS - self.small_seen_total
            plural = "preview" if remaining == 1 else "previews"
            verb = "remains" if remaining == 1 else "remain"
            return f"Big tiles unlock after {LARGE_DELAY_PREVIEWS} small previews. {remaining} {plural} {verb}."
        if self.large_pending:
            remaining_slots = max(1, LARGE_SPAN_SMALLS + 1 - self.span_small_pos)
            return f"One big-tile bundle is still pending in the current {LARGE_SPAN_SMALLS}-small span ({remaining_slots} preview slots left)."
        remaining_smalls = max(0, LARGE_SPAN_SMALLS - self.span_small_pos)
        plural = "preview" if remaining_smalls == 1 else "previews"
        return f"This span already used its big tile. {remaining_smalls} more small {plural} until the next big-tile span."

    def snapshot(self) -> Tuple[Dict[str, int], int, int, int, bool, int]:
        return (
            self.small_counts.copy(),
            self.small_pos,
            self.small_seen_total,
            self.span_small_pos,
            self.large_pending,
            self.max_tile,
        )

    def restore(self, snapshot: Tuple[Dict[str, int], int, int, int, bool, int]) -> None:
        counts, s_pos, small_seen_total, v4, v5, v6 = snapshot
        self.small_counts = counts.copy()
        self.small_pos = s_pos
        self.small_seen_total = small_seen_total
        if isinstance(v6, int) and v6 > 0:
            self.span_small_pos = int(v4)
            self.large_pending = bool(v5)
            self.max_tile = int(v6)
            return

        # Best-effort compatibility for older simplified snapshots that only stored
        # the board max in the fourth slot.
        self.max_tile = max(0, int(v4))
        if self.small_seen_total < LARGE_DELAY_PREVIEWS:
            self.span_small_pos = 0
            self.large_pending = False
            return
        self.span_small_pos = min(max(0, self.small_seen_total - LARGE_DELAY_PREVIEWS), LARGE_SPAN_SMALLS)
        self.large_pending = True


def format_state(tile_cycle: TileCycle, show_debug: bool = True) -> str:
    """
    Return a human-readable small-pool listing plus large probability and move index.
    Example: move=9 | pool[3]: R R B | P(large)=4.2% | dbg span=2/20 pend=True
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
    move_idx = tile_cycle.small_seen_total
    parts = [f"move={move_idx}", f"pool[{remaining}]: {pool_str}", f"P(large)={large_prob:.1f}%"]
    if show_debug:
        bonus_vals = tile_cycle.bonus_values()
        vals_str = ",".join(str(v) for v in bonus_vals) if bonus_vals else "-"
        parts.append(
            f"dbg span={tile_cycle.span_small_pos}/{LARGE_SPAN_SMALLS} "
            f"pend={tile_cycle.large_pending} max={tile_cycle.max_tile} bonus={vals_str}"
        )
    return " | ".join(parts)


# ---------- Board state classification ----------


SMALL_COLOR_MAP = {
    "red": "🟥",
    "blue": "🟦",
}
CELL_GRAY_TOKEN = "3"
TOKEN_EMPTY = "·"
TOKEN_OTHER = "X"
BLANK_MEAN_THRESHOLD = 80.0  # calibrated from provided blank tiles (was 50)
# Use the same inset for gray digits and color checks to match training crops.
CLASSIFY_INSET_RATIO = 0.12
GRAY_INSET_RATIO = CLASSIFY_INSET_RATIO
UNKNOWN_TILE_SIZE = 96
BOARD_GRID_CALIBRATION_PATH = Path("board_grid_calibration.json")
_board_cell_outer_boxes_cache: Dict[
    Tuple[int, int, Optional[Tuple[Tuple[float, ...], Tuple[float, ...], Tuple[float, ...], Tuple[float, ...]]]],
    Tuple[List[Tuple[int, int, Tuple[int, int, int, int]]], Dict[str, object]],
] = {}
_cell_label_cache: Dict[bytes, str] = {}
BOARD_COLOR_PROTOTYPES = {
    "red": np.array([231.84, 123.65, 141.79]),
    "blue": np.array([132.33, 198.81, 243.71]),
}
RED_HUE_TARGET = 0.86  # calibrated from provided red tiles (mean hue)
RED_HUE_BAND = 0.12
BLUE_HUE_TARGET = 0.52
THREE_HASH_HEX = [
    # Current calibrated '3' tile hashes (phash with 8% margin crop + autocontrast).
    "f526999966663133",
    "f526999966e63131",
    "f526c99966e63131",
]
# Max Hamming distance to count as a '3' when compared to the above hashes.
THREE_HASH_THRESHOLD = 6
THREE_CONSENSUS_DIST_THRESHOLD = 0.35


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


_three_hash_cache: Optional[List[imagehash.ImageHash]] = None
_three_detector_cache: Optional[Dict[str, object]] = None
_gray_detector_cache: Optional[Dict[str, object]] = None
_board_grid_calibration_cache: Optional[Dict[str, object]] = None


def _prepare_image_for_hash(img: Image.Image, margin: float = 0.08) -> Image.Image:
    """Normalize a tile image before hashing to improve robustness."""
    img = img.convert("L")
    w, h = img.size
    mw = int(w * margin)
    mh = int(h * margin)
    img = img.crop((mw, mh, w - mw, h - mh))
    return ImageOps.autocontrast(img)


def load_three_detector(path: Path = Path("three_detector.json")) -> Optional[Dict[str, object]]:
    global _three_detector_cache
    if _three_detector_cache is not None:
        return _three_detector_cache
    if not path.exists():
        _three_detector_cache = None
        return None
    try:
        data = json.loads(path.read_text())
    except Exception:
        _three_detector_cache = None
        return None
    if "detectors" not in data:
        _three_detector_cache = None
        return None
    _three_detector_cache = data
    return data


def load_gray_detector(
    path: Path = Path("gray_tile_detector.json"),
) -> Optional[Dict[str, object]]:
    global _gray_detector_cache
    if _gray_detector_cache is not None:
        return _gray_detector_cache
    if not path.exists():
        _gray_detector_cache = None
        return None
    try:
        data = json.loads(path.read_text())
    except Exception:
        _gray_detector_cache = None
        return None
    if "detectors" not in data:
        _gray_detector_cache = None
        return None
    _gray_detector_cache = data
    return data


def _mask_from_tile(cell: np.ndarray, size: int = 64, margin: float = 0.12) -> np.ndarray:
    img = Image.fromarray(cell).convert("L")
    w, h = img.size
    mw = int(w * margin)
    mh = int(h * margin)
    img = img.crop((mw, mh, w - mw, h - mh))
    img = ImageOps.autocontrast(img)
    try:
        resample = Image.Resampling.BILINEAR
    except AttributeError:
        resample = Image.BILINEAR
    img = img.resize((size, size), resample=resample)
    arr = np.array(img, dtype=np.uint8)
    # Otsu threshold
    hist = np.bincount(arr.flatten(), minlength=256).astype(np.float64)
    total = arr.size
    sum_total = float((np.arange(256) * hist).sum())
    sum_b = 0.0
    w_b = 0.0
    max_var = 0.0
    threshold = 127
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > max_var:
            max_var = var_between
            threshold = t
    return (arr < threshold).astype(np.float32)


def is_three_by_template(cell: np.ndarray) -> bool:
    data = load_three_detector()
    if not data:
        return False
    detectors = data.get("detectors", [])
    for det in detectors:
        templates = det.get("templates")
        mean_mask = det.get("mean_mask")
        threshold = float(det.get("threshold", 0.0))
        margin = float(det.get("margin", 0.12))
        if templates:
            tmpl_arr = [np.array(t, dtype=np.float32) for t in templates]
            if not tmpl_arr:
                continue
            size = int(det.get("size", tmpl_arr[0].shape[0]))
            mask = _mask_from_tile(cell, size=size, margin=margin)
            dist = min(float(((mask - t) ** 2).mean()) for t in tmpl_arr)
            if dist <= threshold:
                return True
        elif mean_mask is not None:
            mean_arr = np.array(mean_mask, dtype=np.float32)
            size = int(det.get("size", mean_arr.shape[0] if mean_arr.ndim == 2 else 64))
            if mean_arr.ndim != 2:
                continue
            mask = _mask_from_tile(cell, size=size, margin=margin)
            dist = float(((mask - mean_arr) ** 2).mean())
            if dist <= threshold:
                return True
    return False


def classify_gray_tile(cell: np.ndarray) -> Optional[str]:
    """Return a numeric gray-tile label if confidently matched."""
    hog_model = gh.load_model()
    if hog_model:
        label = gh.predict_label(cell, hog_model, use_thresholds=True)
        if label:
            return label
    data = load_gray_detector()
    gap_threshold = float(data.get("gap_threshold", 0.0)) if data else 0.0
    best_label, best_dist, best_threshold, second_best = _gray_template_best_match(cell)
    if not best_label:
        return None
    gap = second_best - best_dist if second_best < float("inf") else best_dist
    if best_dist <= best_threshold and gap >= gap_threshold:
        return best_label
    return None


def _gray_template_best_match(
    cell: np.ndarray,
) -> Tuple[Optional[str], float, float, float]:
    """Return the best raw template match for a gray numeric tile."""
    data = load_gray_detector()
    if not data:
        return None, float("inf"), float("inf"), float("inf")
    detectors = data.get("detectors", [])
    best_label: Optional[str] = None
    best_dist = float("inf")
    best_threshold = float("inf")
    second_best = float("inf")
    for det in detectors:
        labels = det.get("labels", {})
        if not labels:
            continue
        thresholds = det.get("thresholds", {})
        margin = float(det.get("margin", 0.12))
        size = int(det.get("size", 64))
        mask = _mask_from_tile(cell, size=size, margin=margin)
        for label, templates in labels.items():
            tmpl_arr = [np.array(t, dtype=np.float32) for t in templates]
            if not tmpl_arr:
                continue
            dist = min(float(((mask - t) ** 2).mean()) for t in tmpl_arr)
            if dist < best_dist:
                second_best = best_dist
                best_dist = dist
                best_label = str(label)
                best_threshold = float(thresholds.get(label, float("inf")))
            elif dist < second_best:
                second_best = dist
    return best_label, best_dist, best_threshold, second_best


def is_three_by_consensus(cell: np.ndarray) -> bool:
    """
    Accept a gray tile as "3" when two independent raw detectors agree.
    This catches soft-missed 3s from newer mirrored-device captures without
    loosening the general numeric thresholds.
    """
    hog_model = gh.load_model()
    if not hog_model:
        return False
    raw_hog_label = gh.predict_label(cell, hog_model, use_thresholds=False)
    if raw_hog_label != CELL_GRAY_TOKEN:
        return False
    best_label, best_dist, _best_threshold, _second_best = _gray_template_best_match(cell)
    return best_label == CELL_GRAY_TOKEN and best_dist <= THREE_CONSENSUS_DIST_THRESHOLD


def load_three_hashes() -> List[imagehash.ImageHash]:
    """Load perceptual hashes for known '3' tiles."""
    if not HAVE_IMAGEHASH:
        return []
    global _three_hash_cache
    if _three_hash_cache is not None:
        return _three_hash_cache
    hashes: List[imagehash.ImageHash] = []
    # Primary source: hardcoded hashes so no files are required.
    for hx in THREE_HASH_HEX:
        try:
            hashes.append(imagehash.hex_to_hash(hx))
        except Exception:
            continue
    # Optional: also load from files if they happen to exist (helps when updating samples).
    for path in (
        Path("out_tiles/tile_r3_c2.png"),
        Path("out_tiles/tile_r2_c0.png"),
    ):
        if path.exists():
            try:
                img = _prepare_image_for_hash(Image.open(path))
                hashes.append(imagehash.phash(img))
            except Exception:
                continue
    _three_hash_cache = hashes
    return hashes


def is_three_by_hash(cell: np.ndarray) -> bool:
    """Return True if the cell matches a known '3' tile via perceptual hash."""
    refs = load_three_hashes()
    if not HAVE_IMAGEHASH or not refs:
        return False
    try:
        cell_img = Image.fromarray(cell)
        cell_hash = imagehash.phash(_prepare_image_for_hash(cell_img))
    except Exception:
        return False
    return any(cell_hash - ref <= THREE_HASH_THRESHOLD for ref in refs)


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


def _hue_dist(a: float, b: float) -> float:
    """Return circular hue distance for [0,1] hue values."""
    d = abs(a - b)
    return min(d, 1.0 - d)


def _rgb_to_hsv_mean(rgb: np.ndarray) -> Tuple[float, float, float]:
    r, g, b = (rgb / 255.0).tolist()
    return colorsys.rgb_to_hsv(r, g, b)


def _cell_cache_key(cell: np.ndarray, gray_cell: Optional[np.ndarray]) -> bytes:
    source = gray_cell if gray_cell is not None else cell
    try:
        resample = Image.Resampling.BILINEAR
    except AttributeError:
        resample = Image.BILINEAR
    gray_small = Image.fromarray(source).convert("L").resize((16, 16), resample=resample)
    gray_bytes = (np.array(gray_small, dtype=np.uint8) // 16).astype(np.uint8).tobytes()
    h, w, _ = cell.shape
    patch = cell[int(h * 0.25) : int(h * 0.75), int(w * 0.25) : int(w * 0.75)]
    patch_mean = patch.reshape(-1, 3).mean(axis=0)
    color_bytes = bytes(int(max(0, min(15, round(v / 16.0)))) for v in patch_mean)
    return color_bytes + gray_bytes


def _dynamic_blank_threshold(values: List[float], default: float) -> float:
    """Compute an Otsu-style threshold over cell brightness values."""
    if not values:
        return default
    uniq = sorted(set(values))
    if len(uniq) < 2:
        return default
    best_thr = default
    best_var = -1.0
    total = len(values)
    for i in range(len(uniq) - 1):
        thr = (uniq[i] + uniq[i + 1]) / 2.0
        low = [v for v in values if v <= thr]
        high = [v for v in values if v > thr]
        if not low or not high:
            continue
        w0 = len(low) / total
        w1 = len(high) / total
        m0 = float(np.mean(low))
        m1 = float(np.mean(high))
        var_between = w0 * w1 * (m0 - m1) ** 2
        if var_between > best_var:
            best_var = var_between
            best_thr = thr
    # Clamp to a sensible range to avoid extreme thresholds.
    return float(min(140.0, max(40.0, best_thr)))


def classify_cell(
    cell: np.ndarray,
    blank_threshold: float,
    color_prototypes: Optional[Dict[str, np.ndarray]] = None,
    hsv_prototypes: Optional[Dict[str, Tuple[float, float, float]]] = None,
    gray_cell: Optional[np.ndarray] = None,
) -> str:
    """
    Classify a single tile cell into: red, blue, empty (·), or gray-as-X.
    No OCR is used.
    """
    original = cell  # keep the untrimmed cell for hash-based matching
    gray_source = gray_cell if gray_cell is not None else original
    cache_key = _cell_cache_key(original, gray_source)
    cached = _cell_label_cache.get(cache_key)
    if cached is not None:
        return cached

    def remember(label: str) -> str:
        if len(_cell_label_cache) >= 4096:
            _cell_label_cache.clear()
        _cell_label_cache[cache_key] = label
        return label

    # Trim a small margin to avoid borders for color/blank checks.
    h, w, _ = cell.shape
    mh = int(h * 0.08)
    mw = int(w * 0.08)
    cell = cell[mh : h - mh, mw : w - mw]
    cell_mean = float(cell.reshape(-1, 3).mean())

    # Only very dark cells are immediately empty. Brighter cells may still be
    # gray-number tiles even when the adaptive blank threshold spikes.
    if cell_mean < min(blank_threshold, 80.0):
        return remember(TOKEN_EMPTY)

    # Center patch mean color.
    ch, cw, _ = cell.shape
    patch = cell[int(ch * 0.2) : int(ch * 0.8), int(cw * 0.2) : int(cw * 0.8)]
    patch_mean = patch.reshape(-1, 3).mean(axis=0)

    # Prototype distance for red/blue.
    protos = color_prototypes or BOARD_COLOR_PROTOTYPES
    hsv_protos = hsv_prototypes or {
        "red": _rgb_to_hsv_mean(protos["red"]),
        "blue": _rgb_to_hsv_mean(protos["blue"]),
    }
    red_dist = float(np.linalg.norm(patch_mean - protos["red"]))
    blue_dist = float(np.linalg.norm(patch_mean - protos["blue"]))
    h, s, v = _rgb_to_hsv_mean(patch_mean)
    red_h, red_s, _red_v = hsv_protos["red"]
    blue_h, blue_s, _blue_v = hsv_protos["blue"]
    red_h_dist = _hue_dist(h, red_h)
    blue_h_dist = _hue_dist(h, blue_h)

    # Require some saturation to classify colored tiles.
    if s > 0.16 and v > 0.2:
        if red_h_dist < 0.12 and red_dist < 80 and red_dist < blue_dist:
            return remember(SMALL_COLOR_MAP["red"])
        if blue_h_dist < 0.12 and blue_dist < 80 and blue_dist < red_dist:
            return remember(SMALL_COLOR_MAP["blue"])

    # Fallback to RGB-only distance checks.
    if red_dist < 60 and red_dist < blue_dist:
        return remember(SMALL_COLOR_MAP["red"])
    if blue_dist < 45:
        return remember(SMALL_COLOR_MAP["blue"])

    # Everything else is gray => attempt numeric label, then fallback to 3 detector.
    gray_label = classify_gray_tile(gray_source)
    if gray_label:
        return remember(gray_label)
    if is_three_by_consensus(gray_source):
        return remember(CELL_GRAY_TOKEN)
    if is_three_by_template(gray_source) or is_three_by_hash(gray_source):
        return remember(CELL_GRAY_TOKEN)
    if cell_mean < blank_threshold:
        return remember(TOKEN_EMPTY)
    return remember(TOKEN_OTHER)


def preprocess_for_ocr(cell: np.ndarray) -> Image.Image:
    """
    Prepare a cell image for tesseract:
    - center crop to avoid borders
    - grayscale, invert (dark text on light)
    - contrast/brightness normalize
    - upscale
    """
    h, w, _ = cell.shape
    mh = int(h * 0.08)
    mw = int(w * 0.08)
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


def _board_grid_params(
    roi_shape: Tuple[int, int, int], inset_ratio: float = 0.0
) -> Tuple[List[int], List[int], float, float]:
    h, w, _ = roi_shape
    # Tiles are slightly taller than wide; fixed splits with offsets to reduce drift.
    base_cell_h = h / 4.0
    base_cell_w = w / 4.05  # small squeeze to reflect narrower width
    offset_y = base_cell_h * 0.2  # lower the top row by ~20% of a tile height
    offset_x = base_cell_w * 0.1  # shift tiles right by ~10% of a tile width
    cell_h = (h - offset_y) / 4.0 * 0.95  # reduce per-row vertical step to ease overlap
    cell_w = (w - offset_x) / 4.05
    inset_y = cell_h * inset_ratio
    inset_x = cell_w * inset_ratio
    xs = [int(offset_x + c * cell_w) for c in range(5)]
    ys = [int(offset_y + r * cell_h) for r in range(5)]
    return xs, ys, inset_x, inset_y


def _gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))
    grad = np.zeros_like(gray)
    grad[:, 1:] += gx
    grad[1:, :] += gy
    return grad


def _grid_score(grad: np.ndarray, xs: List[int], ys: List[int]) -> float:
    h, w = grad.shape
    score = 0.0
    for x in xs:
        x0 = max(0, x - 1)
        x1 = min(w, x + 2)
        score += float(grad[:, x0:x1].mean())
    for y in ys:
        y0 = max(0, y - 1)
        y1 = min(h, y + 2)
        score += float(grad[y0:y1, :].mean())
    return score


def _refine_grid_params(
    roi: np.ndarray, inset_ratio: float = 0.0, search: int = 8
) -> Tuple[List[int], List[int], float, float, Dict[str, float]]:
    xs_base, ys_base, inset_x, inset_y = _board_grid_params(roi.shape, inset_ratio=inset_ratio)
    gray = roi.mean(axis=2).astype(np.float32)
    grad = _gradient_magnitude(gray)
    h, w = grad.shape
    best_score = -1.0
    best_dx = 0
    best_dy = 0
    for dx in range(-search, search + 1):
        for dy in range(-search, search + 1):
            xs = [x + dx for x in xs_base]
            ys = [y + dy for y in ys_base]
            if xs[0] < 1 or ys[0] < 1 or xs[-1] >= w - 1 or ys[-1] >= h - 1:
                continue
            score = _grid_score(grad, xs, ys)
            if score > best_score:
                best_score = score
                best_dx = dx
                best_dy = dy
    xs_ref = [x + best_dx for x in xs_base]
    ys_ref = [y + best_dy for y in ys_base]
    meta = {"dx": float(best_dx), "dy": float(best_dy), "score": float(best_score)}
    return xs_ref, ys_ref, inset_x, inset_y, meta


def load_board_grid_calibration(
    path: Path = BOARD_GRID_CALIBRATION_PATH,
) -> Optional[Dict[str, object]]:
    global _board_grid_calibration_cache
    if _board_grid_calibration_cache is not None:
        return _board_grid_calibration_cache
    if not path.exists():
        _board_grid_calibration_cache = None
        return None
    try:
        data = json.loads(path.read_text())
    except Exception:
        _board_grid_calibration_cache = None
        return None
    required = ("col_lefts", "col_rights", "row_tops", "row_bottoms", "normalized")
    if not all(key in data for key in required):
        _board_grid_calibration_cache = None
        return None
    normalized = data.get("normalized", {})
    required_norm = ("col_lefts", "col_rights", "row_tops", "row_bottoms")
    if not all(key in normalized for key in required_norm):
        _board_grid_calibration_cache = None
        return None
    _board_grid_calibration_cache = data
    return data


def clear_board_grid_calibration_cache() -> None:
    global _board_grid_calibration_cache
    _board_grid_calibration_cache = None
    _board_cell_outer_boxes_cache.clear()
    _cell_label_cache.clear()


def _estimate_board_background_color(roi: np.ndarray) -> np.ndarray:
    h, w, _ = roi.shape
    sample = max(8, min(h, w) // 12)
    corners = [
        roi[:sample, :sample],
        roi[:sample, w - sample :],
        roi[h - sample :, :sample],
        roi[h - sample :, w - sample :],
    ]
    return np.concatenate([patch.reshape(-1, 3) for patch in corners], axis=0).mean(axis=0)


def _component_boxes(mask: np.ndarray) -> List[Tuple[int, int, int, int, int, bool]]:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    boxes: List[Tuple[int, int, int, int, int, bool]] = []
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue
            q = deque([(y, x)])
            visited[y, x] = True
            min_x = max_x = x
            min_y = max_y = y
            area = 0
            touches_edge = False
            while q:
                cy, cx = q.popleft()
                area += 1
                if cy in (0, h - 1) or cx in (0, w - 1):
                    touches_edge = True
                min_x = min(min_x, cx)
                max_x = max(max_x, cx)
                min_y = min(min_y, cy)
                max_y = max(max_y, cy)
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        q.append((ny, nx))
            boxes.append((min_x, min_y, max_x, max_y, area, touches_edge))
    return boxes


def _detect_board_cell_outer_boxes(
    roi: np.ndarray,
) -> Optional[Tuple[List[Tuple[int, int, Tuple[int, int, int, int]]], Dict[str, object]]]:
    h, w, _ = roi.shape
    bg = _estimate_board_background_color(roi)
    dist = np.linalg.norm(roi.astype(np.float32) - bg.astype(np.float32), axis=2)
    min_w = max(24, int(w * 0.07))
    max_w = max(min_w + 1, int(w * 0.32))
    min_h = max(48, int(h * 0.12))
    max_h = max(min_h + 1, int(h * 0.30))
    min_area = int(min_w * min_h * 0.7)

    chosen_boxes: Optional[List[Tuple[int, int, int, int]]] = None
    chosen_threshold: Optional[float] = None
    threshold_candidates = (18.0, 16.0, 20.0, 14.0, 22.0, 24.0)
    best_distance = float("inf")

    for threshold in threshold_candidates:
        mask = dist > threshold
        boxes: List[Tuple[int, int, int, int]] = []
        for x0, y0, x1, y1, area, touches_edge in _component_boxes(mask):
            if touches_edge:
                continue
            bw = x1 - x0 + 1
            bh = y1 - y0 + 1
            if bw < min_w or bw > max_w or bh < min_h or bh > max_h or area < min_area:
                continue
            boxes.append((x0, y0, x1, y1))
        distance = abs(len(boxes) - 16)
        if len(boxes) == 16:
            chosen_boxes = boxes
            chosen_threshold = threshold
            break
        if boxes and distance < best_distance:
            chosen_boxes = boxes
            chosen_threshold = threshold
            best_distance = distance

    if chosen_boxes is None or len(chosen_boxes) != 16:
        return None

    x_centers = sorted((x0 + x1) / 2.0 for x0, _y0, x1, _y1 in chosen_boxes)
    y_centers = sorted((y0 + y1) / 2.0 for _x0, y0, _x1, y1 in chosen_boxes)
    col_centers = [sum(x_centers[i : i + 4]) / 4.0 for i in range(0, 16, 4)]
    row_centers = [sum(y_centers[i : i + 4]) / 4.0 for i in range(0, 16, 4)]

    assigned: List[Tuple[int, int, Tuple[int, int, int, int]]] = []
    seen = set()
    for x0, y0, x1, y1 in chosen_boxes:
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        col = min(range(4), key=lambda idx: abs(cx - col_centers[idx]))
        row = min(range(4), key=lambda idx: abs(cy - row_centers[idx]))
        if (row, col) in seen:
            return None
        seen.add((row, col))
        assigned.append((row, col, (x0, y0, x1, y1)))

    assigned.sort(key=lambda item: (item[0], item[1]))
    meta = {
        "method": "detected_components",
        "threshold": float(chosen_threshold or 0.0),
        "component_count": 16,
    }
    return assigned, meta


def _apply_inset(box: Tuple[int, int, int, int], inset_ratio: float) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    inset_x = int(round((x1 - x0) * inset_ratio))
    inset_y = int(round((y1 - y0) * inset_ratio))
    x0i = x0 + inset_x
    x1i = x1 - inset_x
    y0i = y0 + inset_y
    y1i = y1 - inset_y
    if x1i <= x0i:
        x0i, x1i = x0, x1
    if y1i <= y0i:
        y0i, y1i = y0, y1
    return x0i, y0i, x1i, y1i


def _manual_board_cell_outer_boxes(
    roi: np.ndarray,
) -> Optional[Tuple[List[Tuple[int, int, Tuple[int, int, int, int]]], Dict[str, object]]]:
    calibration = load_board_grid_calibration()
    if not calibration:
        return None
    normalized = calibration.get("normalized", {})
    try:
        col_lefts = [float(v) for v in normalized["col_lefts"]]
        col_rights = [float(v) for v in normalized["col_rights"]]
        row_tops = [float(v) for v in normalized["row_tops"]]
        row_bottoms = [float(v) for v in normalized["row_bottoms"]]
    except Exception:
        return None
    if not all(len(values) == 4 for values in (col_lefts, col_rights, row_tops, row_bottoms)):
        return None
    h, w, _ = roi.shape
    boxes: List[Tuple[int, int, Tuple[int, int, int, int]]] = []
    for row in range(4):
        for col in range(4):
            x0 = int(round(col_lefts[col] * w))
            x1 = int(round(col_rights[col] * w))
            y0 = int(round(row_tops[row] * h))
            y1 = int(round(row_bottoms[row] * h))
            if x1 <= x0 or y1 <= y0:
                return None
            boxes.append((row, col, (x0, y0, x1, y1)))
    meta = {"method": "manual_calibration"}
    return boxes, meta


def _board_layout_cache_key(
    roi: np.ndarray,
) -> Tuple[
    int,
    int,
    Optional[Tuple[Tuple[float, ...], Tuple[float, ...], Tuple[float, ...], Tuple[float, ...]]],
]:
    calibration = load_board_grid_calibration()
    normalized = calibration.get("normalized") if calibration else None
    signature = None
    if normalized:
        try:
            signature = (
                tuple(float(v) for v in normalized["col_lefts"]),
                tuple(float(v) for v in normalized["col_rights"]),
                tuple(float(v) for v in normalized["row_tops"]),
                tuple(float(v) for v in normalized["row_bottoms"]),
            )
        except Exception:
            signature = None
    h, w, _ = roi.shape
    return (h, w, signature)


def _fallback_board_cell_outer_boxes(
    roi: np.ndarray,
) -> Tuple[List[Tuple[int, int, Tuple[int, int, int, int]]], Dict[str, float]]:
    xs, ys, _inset_x, _inset_y, grid_meta = _refine_grid_params(roi, inset_ratio=0.0)
    boxes: List[Tuple[int, int, Tuple[int, int, int, int]]] = []
    for r in range(4):
        for c in range(4):
            boxes.append((r, c, (xs[c], ys[r], xs[c + 1], ys[r + 1])))
    meta = {"method": "fallback_grid", **grid_meta}
    return boxes, meta


def _board_cell_boxes(
    roi: np.ndarray,
    inset_ratio: float = 0.0,
) -> Tuple[List[Tuple[int, int, Tuple[int, int, int, int], Tuple[int, int, int, int]]], Dict[str, object]]:
    cache_key = _board_layout_cache_key(roi)
    cached = _board_cell_outer_boxes_cache.get(cache_key)
    if cached is not None:
        outer_boxes, meta = cached
    else:
        manual = _manual_board_cell_outer_boxes(roi)
        if manual is not None:
            outer_boxes, meta = manual
            _board_cell_outer_boxes_cache[cache_key] = (outer_boxes, meta)
        else:
            detected = _detect_board_cell_outer_boxes(roi)
            if detected is None:
                outer_boxes, meta = _fallback_board_cell_outer_boxes(roi)
            else:
                outer_boxes, meta = detected
                _board_cell_outer_boxes_cache[cache_key] = (outer_boxes, meta)
    boxes = []
    for r, c, outer_box in outer_boxes:
        boxes.append((r, c, outer_box, _apply_inset(outer_box, inset_ratio)))
    return boxes, meta


def classify_board(arr: np.ndarray) -> List[List[str]]:
    """
    Return a 4x4 grid of classified cells using fixed 1/4 splits inside the board ROI.
    A small inset is removed inside each cell to avoid tile borders.
    """
    roi, _box = find_board_roi(arr)
    color_boxes, _grid_meta = _board_cell_boxes(roi, inset_ratio=CLASSIFY_INSET_RATIO)
    gray_boxes, _grid_meta_g = _board_cell_boxes(roi, inset_ratio=GRAY_INSET_RATIO)
    gray_lookup = {(r, c): inner_box for r, c, _outer_box, inner_box in gray_boxes}

    # Precompute cell brightness values to derive an adaptive blank threshold.
    cells: List[Tuple[int, int, np.ndarray]] = []
    gray_cells: Dict[Tuple[int, int], np.ndarray] = {}
    cell_means: List[float] = []
    for r, c, _outer_box, inner_box in color_boxes:
        x0i, y0i, x1i, y1i = inner_box
        cell = roi[y0i:y1i, x0i:x1i]
        stats = _cell_stats(cell)
        cell_means.append(float(stats["trim_mean"]))
        cells.append((r, c, cell))
        x0g, y0g, x1g, y1g = gray_lookup[(r, c)]
        gray_cells[(r, c)] = roi[y0g:y1g, x0g:x1g]
    blank_threshold = _dynamic_blank_threshold(cell_means, BLANK_MEAN_THRESHOLD)

    grid: List[List[str]] = [["" for _ in range(4)] for _ in range(4)]
    for r, c, cell in cells:
        grid[r][c] = classify_cell(
            cell,
            blank_threshold=blank_threshold,
            gray_cell=gray_cells.get((r, c)),
        )
    return grid


def segment_board_cells(arr: np.ndarray, inset_ratio: float = 0.0) -> List[Tuple[int, int, np.ndarray]]:
    """
    Return list of (row, col, cell_array) using fixed 1/4 splits inside the board ROI.
    """
    roi, _box = find_board_roi(arr)
    boxes, _grid_meta = _board_cell_boxes(roi, inset_ratio=inset_ratio)

    cells: List[Tuple[int, int, np.ndarray]] = []
    for r, c, _outer_box, inner_box in boxes:
        x0i, y0i, x1i, y1i = inner_box
        cells.append((r, c, roi[y0i:y1i, x0i:x1i]))
    return cells


def segment_board_cells_with_boxes(
    arr: np.ndarray, inset_ratio: float = 0.0
) -> Tuple[List[Tuple[int, int, Tuple[int, int, int, int], np.ndarray]], np.ndarray]:
    """
    Return list of (row, col, (x0,y0,x1,y1), cell_array) plus the board ROI.
    """
    roi, _box = find_board_roi(arr)
    boxes_with_meta, _grid_meta = _board_cell_boxes(roi, inset_ratio=inset_ratio)
    cells: List[Tuple[int, int, Tuple[int, int, int, int], np.ndarray]] = []
    for r, c, _outer_box, inner_box in boxes_with_meta:
        x0i, y0i, x1i, y1i = inner_box
        cell = roi[y0i:y1i, x0i:x1i]
        cells.append((r, c, (x0i, y0i, x1i, y1i), cell))
    return cells, roi


POOL_GRAY_TOKEN = "⬜️"
WIDE_TOKENS = {SMALL_COLOR_MAP["red"], SMALL_COLOR_MAP["blue"], POOL_GRAY_TOKEN}


def _token_width(token: str) -> int:
    if token in WIDE_TOKENS:
        return 2
    return len(token)


def _board_cell_width(board: List[List[str]]) -> int:
    width = 2
    for row in board:
        for token in row:
            width = max(width, _token_width(token))
    return width


def _render_cell(token: str, cell_width: int) -> str:
    """Return a fixed-width representation of a cell using visual width."""
    width = _token_width(token)
    pad = max(0, cell_width - width)
    return f"{token}{' ' * pad}"


def _visual_width(text: str) -> int:
    """Compute visual width treating emoji tiles as width 2."""
    width = 0
    for ch in text:
        if ch in WIDE_TOKENS:
            width += 2
        else:
            width += 1
    return width


def _pad_visual(text: str, target: int) -> str:
    """Pad with spaces until visual width reaches target."""
    pad_count = max(0, target - _visual_width(text))
    return text + (" " * pad_count)


def _board_lines(board: List[List[str]]) -> List[str]:
    cell_width = _board_cell_width(board)
    return [" ".join(_render_cell(tok, cell_width) for tok in row) for row in board]


def format_board(board: List[List[str]]) -> str:
    return "\n".join(_board_lines(board))


def preview_token(label: str) -> str:
    """Render a token for the preview tile."""
    if label == "red":
        return SMALL_COLOR_MAP["red"]
    if label == "blue":
        return SMALL_COLOR_MAP["blue"]
    if label == "gray":
        return CELL_GRAY_TOKEN
    if label == "large_candidates":
        return TOKEN_OTHER
    return "?"


def format_board_with_preview(board: List[List[str]], preview_label: str) -> str:
    """Return a string with a centered preview token above the 4x4 grid."""
    board_lines = _board_lines(board)
    width = max(_visual_width(line) for line in board_lines) if board_lines else 0
    p_token = preview_token(preview_label)
    preview_line = _render_cell(p_token, _board_cell_width(board)).center(width)
    return "\n".join([preview_line] + board_lines)


def render_move_table(
    board: List[List[str]],
    preview_label: str,
    tile_cycle: "TileCycle",
) -> str:
    """
    Render a compact table with columns: board | preview | P(large) | remaining pool.
    """
    board_lines = _board_lines(board)
    board_w = max(_visual_width(line) for line in board_lines) if board_lines else 0

    preview_token_str = _render_cell(preview_token(preview_label), _board_cell_width(board))
    preview_w = max(_visual_width(preview_token_str), 2)
    preview_lines = [preview_token_str] + [" " * preview_w] * (max(1, len(board_lines)) - 1)

    move_line = f"move={tile_cycle.small_seen_total}"
    p_large = f"P(large)={tile_cycle.large_probability() * 100:.1f}%"
    max_line = f"max={tile_cycle.max_tile}"
    bonus_vals = tile_cycle.bonus_values()
    if bonus_vals:
        bonus_line = "bonus: " + ",".join(str(v) for v in bonus_vals)
    else:
        bonus_line = "bonus: off"
    bag_line = f"12-span: {tile_cycle.small_pos}/{SMALL_BAG_SIZE}"
    large_lines_raw = [move_line, p_large, max_line, f"{bonus_line} | {bag_line}"]
    large_w = max(_visual_width(line) for line in large_lines_raw)
    large_lines = large_lines_raw + [" " * large_w] * max(0, len(board_lines) - len(large_lines_raw))

    label_tokens = [
        ("blue", SMALL_COLOR_MAP["blue"]),
        ("red", SMALL_COLOR_MAP["red"]),
        ("gray", POOL_GRAY_TOKEN),
    ]
    rem_lines_raw: list[str] = []
    for key, tok in label_tokens:
        count = max(0, tile_cycle.small_counts.get(key, 0))
        line = " ".join([tok] * count) if count else ""
        rem_lines_raw.append(line)
    rem_w = max(_visual_width(line) for line in rem_lines_raw) if rem_lines_raw else 0
    rem_w = max(rem_w, 10)
    # Pad rem_lines to board height.
    rem_lines = rem_lines_raw + [""] * max(0, len(board_lines) - len(rem_lines_raw))

    # Normalize row count to board height.
    rows = max(len(board_lines), 1)
    out_lines = []
    for i in range(rows):
        b = board_lines[i] if i < len(board_lines) else " " * board_w
        p = preview_lines[i] if i < len(preview_lines) else " " * preview_w
        l = large_lines[i] if i < len(large_lines) else " " * large_w
        r = rem_lines[i] if i < len(rem_lines) else " " * rem_w
        out_lines.append(
            f"{_pad_visual(b, board_w)} | {_pad_visual(p, preview_w)} | {_pad_visual(l, large_w)} | {_pad_visual(r, rem_w)}"
        )
    return "\n".join(out_lines)


# ---------- Error formatting ----------

C_RED = "\033[31m"
C_RESET = "\033[0m"


def print_error(msg: str) -> None:
    """Print an error in red."""
    print(f"{C_RED}[error]{C_RESET} {msg}")


def print_legend() -> None:
    """Print a legend for board tokens."""
    labels: List[str] = []
    hog_model = gh.load_model()
    if hog_model:
        labels = [str(label) for label in hog_model.get("labels", [])]
    else:
        data = load_gray_detector()
        if data:
            for det in data.get("detectors", []):
                for label in det.get("labels", {}).keys():
                    if label not in labels:
                        labels.append(str(label))
    if labels:
        try:
            labels.sort(key=lambda x: int(x))
        except Exception:
            labels.sort()
        label_str = "/".join(labels)
        print(f"legend: 🟥=red, 🟦=blue, {label_str}=gray tiles, X=unknown, ·=empty")
    else:
        print("legend: 3=three, X=other gray, ·=empty")


def _iso_ts(ts: Optional[float] = None) -> str:
    stamp = ts if ts is not None else time.time()
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(stamp))


def _json_safe(obj: object) -> object:
    """Convert numpy scalars/arrays and tuples into JSON-serializable types."""
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _saturation_and_value(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    arr = arr.astype(float)
    mx = arr.max(axis=2)
    mn = arr.min(axis=2)
    diff = mx - mn
    sat = np.zeros_like(mx)
    nonzero = mx != 0
    sat[nonzero] = diff[nonzero] / mx[nonzero]
    return sat, mx


def _cell_stats(cell: np.ndarray) -> Dict[str, float]:
    """Compute stats aligned with classify_cell trimming and blank threshold."""
    h, w, _ = cell.shape
    mh = int(h * 0.08)
    mw = int(w * 0.08)
    trimmed = cell[mh : h - mh, mw : w - mw]
    mean_rgb = trimmed.reshape(-1, 3).mean(axis=0)
    mean_luma = float(0.2126 * mean_rgb[0] + 0.7152 * mean_rgb[1] + 0.0722 * mean_rgb[2])
    sat, val = _saturation_and_value(trimmed)
    stats = {
        "mean_rgb": mean_rgb.tolist(),
        "mean_luma": mean_luma,
        "mean_sat": float(sat.mean()),
        "mean_val": float(val.mean()),
        "trim_mean": float(trimmed.reshape(-1, 3).mean()),
    }
    return stats


def _draw_board_overlay(
    board_roi: np.ndarray,
    boxes: List[Tuple[int, int, Tuple[int, int, int, int]]],
) -> Image.Image:
    """Draw the inferred cell boxes on the board ROI for debugging."""
    img = Image.fromarray(board_roi).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("Arial.ttf", 12)
    except Exception:
        font = ImageFont.load_default()
    for r, c, box in boxes:
        x0, y0, x1, y1 = box
        draw.rectangle((x0, y0, x1, y1), outline=(255, 215, 0), width=2)
        draw.text((x0 + 2, y0 + 2), f"{r},{c}", fill=(255, 255, 255), font=font)
    return img


class DatasetRecorder:
    """Record captures and labels to a session folder."""

    def __init__(self, base_dir: Path, window_info: Optional[Dict[str, object]] = None) -> None:
        session = time.strftime("session_%Y%m%d_%H%M%S", time.localtime())
        self.session_dir = base_dir / session
        self.session_dir.mkdir(parents=True, exist_ok=False)
        self.manifest_path = self.session_dir / "manifest.jsonl"
        self.labels_path = self.session_dir / "labels.jsonl"
        self.unknown_tiles_dir = self.session_dir / "unknown_tiles"
        self.unknown_tiles_dir.mkdir(parents=True, exist_ok=True)
        self.unknown_index_path = self.session_dir / "unknown_tiles.jsonl"
        self.capture_idx = 0
        self.last_capture_id: Optional[int] = None
        self.last_unknown_count = 0
        meta = {
            "created": _iso_ts(),
            "window": window_info or {},
            "version": 1,
        }
        (self.session_dir / "session.json").write_text(json.dumps(_json_safe(meta), indent=2))

    def record_capture(
        self,
        arr: np.ndarray,
        board: List[List[str]],
        preview_label: str,
        preview_debug: Dict,
        window_id: int,
        ts_event: float,
    ) -> int:
        self.capture_idx += 1
        capture_id = self.capture_idx
        prefix = f"{capture_id:06d}"
        full_path = self.session_dir / f"{prefix}_full.png"
        board_path = self.session_dir / f"{prefix}_board.png"
        board_overlay_path = self.session_dir / f"{prefix}_board_overlay.png"
        preview_path = self.session_dir / f"{prefix}_preview.png"
        meta_path = self.session_dir / f"{prefix}_meta.json"

        Image.fromarray(arr).save(full_path)
        board_roi, board_box = find_board_roi(arr)
        preview_roi, preview_box = find_preview_roi(arr)
        Image.fromarray(board_roi).save(board_path)
        Image.fromarray(preview_roi).save(preview_path)

        color_boxes, grid_meta = _board_cell_boxes(
            board_roi, inset_ratio=CLASSIFY_INSET_RATIO
        )
        gray_boxes, _grid_meta_g = _board_cell_boxes(
            board_roi, inset_ratio=GRAY_INSET_RATIO
        )
        gray_lookup = {(r, c): inner_box for r, c, _outer_box, inner_box in gray_boxes}
        overlay_boxes: List[Tuple[int, int, Tuple[int, int, int, int]]] = []
        cell_stats: List[Dict[str, object]] = []
        cells: List[Tuple[int, int, np.ndarray]] = []
        for r, c, outer_box, inner_box in color_boxes:
            x0i, y0i, x1i, y1i = inner_box
            cell = board_roi[y0i:y1i, x0i:x1i]
            stats = _cell_stats(cell)
            stats.update({"row": r, "col": c, "box": list(inner_box), "outer_box": list(outer_box)})
            cell_stats.append(stats)
            overlay_boxes.append((r, c, outer_box))
            x0g, y0g, x1g, y1g = gray_lookup[(r, c)]
            cell_gray = board_roi[y0g:y1g, x0g:x1g]
            cells.append((r, c, cell_gray))
        overlay_img = _draw_board_overlay(board_roi, overlay_boxes)
        overlay_img.save(board_overlay_path)
        blank_threshold = _dynamic_blank_threshold(
            [float(stat["trim_mean"]) for stat in cell_stats], BLANK_MEAN_THRESHOLD
        )

        unknown_tiles: List[str] = []
        unknown_entries: List[Dict[str, object]] = []
        for r, c, cell in cells:
            if r >= len(board) or c >= len(board[r]):
                continue
            if board[r][c] != TOKEN_OTHER:
                continue
            tile_id = f"{capture_id:06d}_r{r}_c{c}"
            tile_path = self.unknown_tiles_dir / f"{tile_id}.png"
            tile_img = Image.fromarray(cell).resize(
                (UNKNOWN_TILE_SIZE, UNKNOWN_TILE_SIZE), Image.BILINEAR
            )
            tile_img.save(tile_path)
            unknown_tiles.append(tile_id)
            unknown_entries.append(
                {
                    "tile_id": tile_id,
                    "capture_id": capture_id,
                    "row": r,
                    "col": c,
                    "source": full_path.name,
                    "image": tile_path.name,
                }
            )

        if unknown_entries:
            with self.unknown_index_path.open("a") as f:
                for entry in unknown_entries:
                    f.write(json.dumps(_json_safe(entry)) + "\n")

        self.last_unknown_count = len(unknown_tiles)

        meta = {
            "id": capture_id,
            "ts_event": _iso_ts(ts_event),
            "ts_saved": _iso_ts(),
            "window_id": window_id,
            "label": None,
            "board": board,
            "preview_label": preview_label,
            "unknown_tiles": unknown_tiles,
            "unknown_count": len(unknown_tiles),
            "paths": {
                "full": full_path.name,
                "board": board_path.name,
                "board_overlay": board_overlay_path.name,
                "preview": preview_path.name,
            },
            "roi": {
                "board": board_box,
                "preview": preview_box,
            },
            "grid_meta": grid_meta,
            "blank_threshold": blank_threshold,
            "cell_stats": cell_stats,
            "preview_debug": preview_debug,
        }
        meta_path.write_text(json.dumps(_json_safe(meta), indent=2))
        manifest_entry = {
            "id": capture_id,
            "ts_event": meta["ts_event"],
            "meta": meta_path.name,
            "label": None,
        }
        with self.manifest_path.open("a") as f:
            f.write(json.dumps(_json_safe(manifest_entry)) + "\n")

        self.last_capture_id = capture_id
        return capture_id

    def set_label(self, label: Optional[str]) -> Optional[int]:
        if self.last_capture_id is None:
            return None
        meta_path = self.session_dir / f"{self.last_capture_id:06d}_meta.json"
        if not meta_path.exists():
            return None
        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            return None
        meta["label"] = label
        meta["label_ts"] = _iso_ts()
        meta_path.write_text(json.dumps(_json_safe(meta), indent=2))
        label_entry = {
            "id": self.last_capture_id,
            "label": label,
            "ts": meta["label_ts"],
        }
        with self.labels_path.open("a") as f:
            f.write(json.dumps(_json_safe(label_entry)) + "\n")
        return self.last_capture_id


def _count_board_small_tiles(board: List[List[str]]) -> Dict[str, int]:
    """Count small tiles on the 4x4 board (ignore empties and big/unknown 'X')."""
    counts = {"red": 0, "blue": 0, "gray": 0}
    for row in board:
        for cell in row:
            if cell == TOKEN_EMPTY:
                continue
            if cell == SMALL_COLOR_MAP["red"]:
                counts["red"] += 1
            elif cell == SMALL_COLOR_MAP["blue"]:
                counts["blue"] += 1
            elif cell == CELL_GRAY_TOKEN:
                counts["gray"] += 1
    return counts


def token_to_value(token: str) -> int:
    """Return the numeric value for a rendered board token."""
    if token == TOKEN_EMPTY:
        return 0
    if token == SMALL_COLOR_MAP["red"]:
        return SMALL_TILE_VALUES["red"]
    if token == SMALL_COLOR_MAP["blue"]:
        return SMALL_TILE_VALUES["blue"]
    try:
        return int(token)
    except (TypeError, ValueError):
        return 0


def board_max_tile(board: List[List[str]]) -> int:
    """Return the largest numeric tile currently visible on the board."""
    max_tile = 0
    for row in board:
        for cell in row:
            max_tile = max(max_tile, token_to_value(cell))
    return max_tile


def _board_has_unknowns(board: List[List[str]]) -> bool:
    """Return True if any board cell is unknown."""
    for row in board:
        for cell in row:
            if cell == TOKEN_OTHER:
                return True
    return False


def _initial_state_error(board: List[List[str]], preview_label: str) -> Optional[str]:
    """Return an error message if the initial board/preview state is invalid."""
    small_tokens = {SMALL_COLOR_MAP["red"], SMALL_COLOR_MAP["blue"], CELL_GRAY_TOKEN}
    small_count = 0
    large_count = 0
    for row in board:
        for cell in row:
            if cell == TOKEN_EMPTY:
                continue
            if cell in small_tokens:
                small_count += 1
            else:
                large_count += 1
    if preview_label not in ("red", "blue", "gray"):
        return (
            "initial state invalid: preview must be a small tile "
            f"(red/blue/gray), got {preview_label!r}"
        )
    if small_count != 8 or large_count != 1:
        return (
            "initial state invalid: expected 8 small tiles and 1 large tile on board, "
            f"got small={small_count}, large={large_count}"
        )
    return None


def preview_possible(tile_cycle: "TileCycle", preview_label: str) -> Tuple[bool, str]:
    """
    Return (is_possible, reason) for the current preview tile under the tile_cycle state.
    - red/blue/gray must have remaining count > 0.
    - large requires a 48+ tile on the board, which enables the independent bonus rule.
    """
    if preview_label in ("red", "blue", "gray"):
        remaining = tile_cycle.small_counts.get(preview_label, 0)
        if remaining > 0:
            return True, ""
        return False, f"no {preview_label} tiles remaining in pool"
    if preview_label == "large_candidates":
        if tile_cycle.large_probability() > 0 and tile_cycle.bonus_values():
            return True, ""
        return False, tile_cycle.large_schedule_note()
    return True, ""  # unknown preview labels are ignored


def seed_tile_cycle_from_initial_state(
    tile_cycle: "TileCycle", board: List[List[str]], preview_label: str
) -> None:
    """
    Seed the tile cycle to reflect the initial game load where 8 board tiles
    plus the preview (9th) are already known in the first batch of 12.
    """
    counts = _count_board_small_tiles(board)

    base = {"red": 4, "blue": 4, "gray": 4}
    # Clamp per-color to base (avoid negative remainder if detection overcounts a color).
    clamped = {k: min(counts.get(k, 0), base[k]) for k in base}
    # Ensure total seen does not exceed 9 (8 on board + 1 preview at new game start).
    total_seen = sum(clamped.values())
    if total_seen > 9:
        over = total_seen - 9
        # Trim from the most frequent colors first until we reach 9.
        for color in sorted(clamped, key=lambda k: clamped[k], reverse=True):
            if over <= 0:
                break
            drop = min(clamped[color], over)
            clamped[color] -= drop
            over -= drop

    tile_cycle.small_counts = {k: max(0, base[k] - clamped.get(k, 0)) for k in base}
    initial_seen = sum(clamped.values())
    # Board tiles have been consumed from the 12-span.
    tile_cycle.small_pos = initial_seen
    tile_cycle.small_seen_total = 0
    tile_cycle.span_small_pos = 0
    tile_cycle.large_pending = False
    tile_cycle.set_max_tile(board_max_tile(board))


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
            board = classify_board(arr)
            tile_cycle.set_max_tile(board_max_tile(board))
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
    tc = TileCycle()
    seed_tile_cycle_from_initial_state(tc, board, preview_label)
    tc.set_max_tile(board_max_tile(board))
    print(render_move_table(board, preview_label, tc))


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
    dataset_dir: Optional[str] = None,
    window_info: Optional[Dict[str, object]] = None,
    settle_threshold: float = 0.15,
    settle_frames: int = 2,
    settle_poll: float = 0.05,
    settle_timeout: float = 1.0,
) -> None:
    """
    Capture/detect on arrow key presses; support undo (z) and reset (q).
    Optional labeling keys: c=correct, x=incorrect, u=undo label.
    Requires Quartz event tap permissions.
    """
    events = start_key_listener()
    tile_cycle = TileCycle()
    history: list[Tuple[Dict[str, int], int, int, int, bool, int]] = []
    recorder = None
    initial_check = True
    prev_board: Optional[List[List[str]]] = None
    prev_preview_label: Optional[str] = None
    if dataset_dir:
        recorder = DatasetRecorder(Path(dataset_dir), window_info=window_info)
        print(f"Recording dataset to {recorder.session_dir}")
        print("Label keys: c=correct, x=incorrect, u=undo label")

    def initialize_cycle(ts_event: float) -> None:
        nonlocal tile_cycle
        nonlocal initial_check
        nonlocal prev_board
        nonlocal prev_preview_label
        try:
            arr = np.array(capture_window(window_id))
            board = classify_board(arr)
            preview_label, _debug = classify_array(arr)
        except Exception as exc:  # noqa: BLE001
            print_error(str(exc))
            return
        if initial_check:
            err = _initial_state_error(board, preview_label)
            if err:
                print_error(err)
                raise SystemExit(1)
            initial_check = False
        prev_board = board
        prev_preview_label = preview_label
        tile_cycle = TileCycle()
        seed_tile_cycle_from_initial_state(tile_cycle, board, preview_label)
        tile_cycle.set_max_tile(board_max_tile(board))
        history.clear()
        ok, reason = preview_possible(tile_cycle, preview_label)
        if not ok:
            print_error(f"preview '{preview_label}' not possible at init: {reason}")
        # Consume the preview so the displayed pool reflects tiles remaining after it.
        tile_cycle.update(preview_label)
        print(render_move_table(board, preview_label, tile_cycle))
        print()
        # No condensed debug line; table is the only state output (errors still printed).

    def capture_after_settle(apply_delay: bool) -> Optional[np.ndarray]:
        if apply_delay and arrow_delay > 0:
            time.sleep(arrow_delay)
        try:
            arr = np.array(capture_window(window_id))
        except Exception as exc:  # noqa: BLE001
            print_error(str(exc))
            return None
        if settle_frames <= 0:
            return arr
        try:
            prev_sig, _ = board_signature(arr)
        except Exception:
            return arr
        stable = 0
        start = time.time()
        last_arr = arr
        while time.time() - start < settle_timeout:
            time.sleep(settle_poll)
            try:
                curr_arr = np.array(capture_window(window_id))
                curr_sig, _ = board_signature(curr_arr)
                diff = board_signature_diff(prev_sig, curr_sig)
                prev_sig = curr_sig
                last_arr = curr_arr
                if diff < settle_threshold:
                    stable += 1
                else:
                    stable = 0
                if stable >= settle_frames:
                    return last_arr
            except Exception:
                continue
        return last_arr

    def capture_and_update(ts_event: float, apply_delay: bool = False) -> None:
        nonlocal tile_cycle
        nonlocal prev_board
        nonlocal prev_preview_label
        arr = capture_after_settle(apply_delay)
        if arr is None:
            return
        try:
            board = classify_board(arr)
            label, _debug = classify_array(arr)
        except Exception as exc:  # noqa: BLE001
            print_error(str(exc))
            return
        if prev_board is not None and prev_preview_label is not None:
            unknown_now = _board_has_unknowns(board) or label == "unknown"
            unknown_prev = _board_has_unknowns(prev_board) or prev_preview_label == "unknown"
            if not unknown_now and not unknown_prev:
                if board == prev_board and label == prev_preview_label:
                    ts = time.strftime("%H:%M:%S", time.localtime(ts_event))
                    print(f"[{ts}] skip: board+preview unchanged")
                    return
        prev_board = board
        prev_preview_label = label
        tile_cycle.set_max_tile(board_max_tile(board))
        ok, reason = preview_possible(tile_cycle, label)
        if not ok:
            print_error(f"preview '{label}' not possible: {reason}")
        history.append(tile_cycle.snapshot())
        tile_cycle.update(label)
        ts = time.strftime("%H:%M:%S", time.localtime(ts_event))
        capture_id = None
        if recorder:
            capture_id = recorder.record_capture(
                arr,
                board,
                label,
                _debug,
                window_id,
                ts_event,
            )
        print(f"[{ts}]")
        print(render_move_table(board, label, tile_cycle))
        if recorder and capture_id is not None:
            extra = ""
            if recorder.last_unknown_count:
                extra = f" unknown={recorder.last_unknown_count}"
            print(f"capture={capture_id} label=unlabeled{extra}")
        print()

    # Initial capture seeds the cycle with the 8 on-board + preview (9th) tiles.
    initialize_cycle(time.time())
    print_legend()

    while True:
        ts_event, key = events.get()
        if key == "undo":
            if history:
                snapshot = history.pop()
                tile_cycle.restore(snapshot)
                ts = time.strftime("%H:%M:%S", time.localtime(ts_event))
                print(f"[{ts}] undo")
            continue
        if key == "reset":
            initialize_cycle(ts_event)
            print_legend()
            continue
        if key in ("label_correct", "label_incorrect", "label_undo"):
            if not recorder:
                continue
            label = None
            if key == "label_correct":
                label = "correct"
            elif key == "label_incorrect":
                label = "incorrect"
            capture_id = recorder.set_label(label)
            ts = time.strftime("%H:%M:%S", time.localtime(ts_event))
            if capture_id is None:
                print_error("No capture available to label yet.")
            else:
                label_str = label if label is not None else "cleared"
                print(f"[{ts}] label={label_str} capture={capture_id}")
            continue

        # Arrow key
        capture_and_update(ts_event, apply_delay=True)


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
        default=0.3,
        help="Delay (seconds) after an arrow key press before capture, to let the board update.",
    )
    parser.add_argument(
        "--settle-threshold",
        type=float,
        default=0.15,
        help="Board signature diff threshold to consider the animation settled.",
    )
    parser.add_argument(
        "--settle-frames",
        type=int,
        default=2,
        help="Consecutive stable frames required before capture is accepted.",
    )
    parser.add_argument(
        "--settle-poll",
        type=float,
        default=0.05,
        help="Polling interval (seconds) while waiting for the board to settle.",
    )
    parser.add_argument(
        "--settle-timeout",
        type=float,
        default=1.0,
        help="Maximum time (seconds) to wait for the board to settle per move.",
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
    parser.add_argument(
        "--record-dataset",
        type=str,
        help="Record captures to a dataset directory (creates a session subfolder).",
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
        print_legend()
        return

    print(f"Monitoring window {window_id} ({app_name} – {win_name})")
    if args.record_dataset and args.poll:
        print_error("--record-dataset is supported only in arrow-key mode (omit --poll).")
        return
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
            dataset_dir=args.record_dataset,
            window_info={"id": window_id, "app": app_name, "title": win_name},
            settle_threshold=args.settle_threshold,
            settle_frames=args.settle_frames,
            settle_poll=args.settle_poll,
            settle_timeout=args.settle_timeout,
        )


if __name__ == "__main__":
    main()
