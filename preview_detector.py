import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image


# Prototype RGB colors for the three small-tile colors.
# Blue/gray are sampled from the provided screenshots; red comes from board tiles.
COLOR_PROTOTYPES = {
    "red": np.array([232.0, 109.0, 130.0]),
    "blue": np.array([118.0, 184.0, 236.0]),
    "gray": np.array([106.0, 167.0, 208.0]),
}
BOARD_REL_BOX = (0.10, 0.30, 0.90, 0.83)


def load_image(path: Path) -> np.ndarray:
    """Return an RGB numpy array."""
    with Image.open(path) as im:
        return np.array(im.convert("RGB"))


def saturation_and_value(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute saturation and value (max channel) for an RGB uint8 image."""
    arr = arr.astype(float)
    mx = arr.max(axis=2)
    mn = arr.min(axis=2)
    diff = mx - mn
    sat = np.zeros_like(mx)
    nonzero = mx != 0
    sat[nonzero] = diff[nonzero] / mx[nonzero]
    return sat, mx


def find_preview_roi(arr: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Estimate a crop around the "next" preview band.

    Steps:
    - Look at the upper third of the image near the center.
    - Compute a saturation mask to find a centroid of "interesting" pixels.
    - Center a fixed-size box (relative to the whole image) on that centroid.
    """
    h, w, _ = arr.shape
    sat, _ = saturation_and_value(arr)

    # Focus on the upper 30% and central 50% of the screen.
    ys = np.arange(h)[:, None]
    xs = np.arange(w)[None, :]
    top_mask = ys < h * 0.3
    center_mask = np.abs(xs - w / 2) < w * 0.25
    mask = (sat > 0.25) & top_mask & center_mask
    coords = np.argwhere(mask)
    if len(coords) == 0:
        # Fallback: center a box at a typical preview location.
        cx, cy = w / 2, h * 0.2
    else:
        cy, cx = coords.mean(axis=0)

    roi_w = w * 0.45
    roi_h = h * 0.15
    x0 = max(0, int(cx - roi_w / 2))
    x1 = min(w, int(cx + roi_w / 2))
    y0 = max(0, int(cy - roi_h / 2))
    y1 = min(h, int(cy + roi_h / 2))
    return arr[y0:y1, x0:x1], (x0, y0, x1, y1)


def dominant_mask(roi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a mask of high-saturation pixels and the associated coords."""
    sat, val = saturation_and_value(roi)
    mask = (sat > 0.35) & (val > 40)
    coords = np.argwhere(mask)
    return mask, coords, sat


def classify_roi(roi: np.ndarray) -> Tuple[str, Dict]:
    """
    Classify the preview ROI into one of:
    - "large_candidates" (three-option band)
    - "red", "blue", "gray" (small tiles)
    """
    mask, coords, _ = dominant_mask(roi)
    h, w, _ = roi.shape

    debug = {
        "roi_shape": roi.shape,
        "mask_count": int(len(coords)),
    }

    # If we found pixels with color, measure spread to flag the large-tile band.
    if len(coords) > 0:
        ys, xs = coords[:, 0], coords[:, 1]
        x_std_norm = xs.std() / w
        y_std_norm = ys.std() / h
        debug["x_std_norm"] = float(x_std_norm)
        debug["y_std_norm"] = float(y_std_norm)

        is_large = (
            len(coords) > 300
            and y_std_norm < 0.06
            and x_std_norm > 0.14
        )
        if is_large:
            return "large_candidates", debug

    # For color classification, use the mean of the mask; if empty, use center patch.
    if len(coords) > 0:
        mean_color = roi[mask].mean(axis=0)
    else:
        cy, cx = h // 2, w // 2
        patch = roi[int(cy - h * 0.1) : int(cy + h * 0.1), int(cx - w * 0.1) : int(cx + w * 0.1)]
        mean_color = patch.mean(axis=(0, 1))
    debug["mean_color"] = [float(x) for x in mean_color]

    # Nearest prototype.
    best_label = None
    best_dist = float("inf")
    for label, proto in COLOR_PROTOTYPES.items():
        dist = np.linalg.norm(mean_color - proto)
        debug[f"dist_{label}"] = float(dist)
        if dist < best_dist:
            best_dist = dist
            best_label = label

    return best_label or "unknown", debug


def classify_image(path: Path) -> Tuple[str, Dict]:
    arr = load_image(path)
    return classify_array(arr)


def classify_array(arr: np.ndarray) -> Tuple[str, Dict]:
    """
    Classify from an RGB numpy array of a full screenshot.
    Returns (label, debug_dict).
    """
    roi, box = find_preview_roi(arr)
    label, debug = classify_roi(roi)
    debug["roi_box"] = box
    return label, debug


def classify_pil(image: Image.Image) -> Tuple[str, Dict]:
    """Classify from a PIL Image."""
    return classify_array(np.array(image.convert("RGB")))


def find_board_roi(arr: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Return a crop around the main 4x4 board.

    The board position in the mirrored Threes layout is effectively fixed, while
    saturation-based tight boxes can drift downward when the top row is mostly
    empty. Use a stable relative crop instead.
    """
    h, w, _ = arr.shape
    x0 = int(w * BOARD_REL_BOX[0])
    x1 = int(w * BOARD_REL_BOX[2])
    y0 = int(h * BOARD_REL_BOX[1])
    y1 = int(h * BOARD_REL_BOX[3])
    return arr[y0:y1, x0:x1], (x0, y0, x1, y1)


def board_signature(arr: np.ndarray, size: int = 32) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Downsample the board ROI to a low-res, quantized signature for change detection.
    Returns (signature_array, roi_box).
    """
    roi, box = find_board_roi(arr)
    if roi.size == 0:
        raise ValueError("Empty board ROI")
    try:
        resample = Image.Resampling.BILINEAR  # Pillow >=9.1
    except AttributeError:  # pragma: no cover - Pillow <9.1
        resample = Image.BILINEAR
    small = Image.fromarray(roi).resize((size, size), resample=resample).convert("L")
    gray = np.array(small, dtype=np.uint8)
    # Quantize to reduce sensitivity to tiny noise.
    signature = (gray // 16).astype(np.uint8)
    return signature, box


def board_signature_diff(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """Mean absolute difference between two board signatures."""
    if sig_a.shape != sig_b.shape:
        raise ValueError("Board signatures must have the same shape")
    diff = np.abs(sig_a.astype(np.float32) - sig_b.astype(np.float32))
    return float(diff.mean())


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect the Threes preview tile from a screenshot.")
    parser.add_argument("--image", required=True, type=Path, help="Path to a screenshot image.")
    parser.add_argument("--debug", action="store_true", help="Print debug details as JSON.")
    args = parser.parse_args()

    label, debug = classify_image(args.image)
    print(f"Detected: {label}")
    if args.debug:
        print(json.dumps(debug, indent=2))


if __name__ == "__main__":
    main()
