import json
from pathlib import Path
import colorsys
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps


def _otsu_threshold(arr: np.ndarray) -> int:
    hist = np.bincount(arr.flatten(), minlength=256).astype(np.float64)
    total = arr.size
    sum_total = float(np.dot(np.arange(256), hist))
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
    return threshold


def yellow_ratio(cell: np.ndarray) -> float:
    """Estimate fraction of yellow digits based on bright pixels in HSV."""
    arr = cell.astype(np.float32) / 255.0
    bright = arr.mean(axis=2)
    thresh = np.percentile(bright, 95)
    mask = bright > thresh
    if mask.sum() == 0:
        return 0.0
    pixels = arr[mask]
    hsv = np.array([colorsys.rgb_to_hsv(*p) for p in pixels], dtype=np.float32)
    h = hsv[:, 0]
    s = hsv[:, 1]
    yellow = (h > 0.10) & (h < 0.18) & (s > 0.2)
    return float(yellow.mean())


def normalize_glyph(
    cell: np.ndarray,
    margin: float = 0.1,
    size: int = 32,
    pad_ratio: float = 0.08,
) -> np.ndarray:
    img = Image.fromarray(cell).convert("L")
    w, h = img.size
    mw = int(w * margin)
    mh = int(h * margin)
    img = img.crop((mw, mh, w - mw, h - mh))
    img = ImageOps.autocontrast(img)
    arr = np.array(img, dtype=np.float32)
    # Use edges to locate glyphs regardless of light/dark text.
    gx = np.zeros_like(arr)
    gy = np.zeros_like(arr)
    gx[:, 1:-1] = arr[:, 2:] - arr[:, :-2]
    gy[1:-1, :] = arr[2:, :] - arr[:-2, :]
    grad = np.hypot(gx, gy)
    thresh = np.percentile(grad, 85)
    edge_mask = grad > thresh
    coords = np.argwhere(edge_mask)
    if len(coords) <= 12:
        # Fallback to Otsu on intensity if edges are sparse.
        t = _otsu_threshold(arr.astype(np.uint8))
        mask = arr < t
        coords = np.argwhere(mask)
    if len(coords) > 12:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        pad = int(max(arr.shape) * pad_ratio)
        y0 = max(0, y0 - pad)
        x0 = max(0, x0 - pad)
        y1 = min(arr.shape[0], y1 + pad)
        x1 = min(arr.shape[1], x1 + pad)
        arr = arr[y0:y1, x0:x1]
    img = Image.fromarray(arr).resize((size, size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)
    if arr.max() > arr.min():
        arr = (arr - arr.min()) / (arr.max() - arr.min())
    else:
        arr = np.zeros_like(arr, dtype=np.float32)
    return arr


def hog_features(
    img: np.ndarray, cell_size: int = 8, bins: int = 9
) -> np.ndarray:
    h, w = img.shape
    gx = np.zeros_like(img, dtype=np.float32)
    gy = np.zeros_like(img, dtype=np.float32)
    gx[:, 1:-1] = img[:, 2:] - img[:, :-2]
    gy[1:-1, :] = img[2:, :] - img[:-2, :]
    mag = np.hypot(gx, gy)
    ang = (np.degrees(np.arctan2(gy, gx)) % 180.0).astype(np.float32)

    cells_y = h // cell_size
    cells_x = w // cell_size
    hist = np.zeros((cells_y, cells_x, bins), dtype=np.float32)
    bin_size = 180.0 / bins

    for cy in range(cells_y):
        for cx in range(cells_x):
            y0 = cy * cell_size
            x0 = cx * cell_size
            cell_mag = mag[y0 : y0 + cell_size, x0 : x0 + cell_size]
            cell_ang = ang[y0 : y0 + cell_size, x0 : x0 + cell_size]
            bin_idx = np.floor(cell_ang / bin_size).astype(int)
            bin_idx = np.clip(bin_idx, 0, bins - 1)
            for b in range(bins):
                hist[cy, cx, b] = cell_mag[bin_idx == b].sum()

    blocks = []
    for cy in range(cells_y - 1):
        for cx in range(cells_x - 1):
            block = hist[cy : cy + 2, cx : cx + 2, :].ravel()
            norm = np.linalg.norm(block) + 1e-6
            blocks.append(block / norm)
    if not blocks:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(blocks)


def _normalize_vec(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec) + 1e-6
    return vec / norm


def train_hog_model(
    samples: List[Tuple[np.ndarray, str]],
    margin: float = 0.1,
    size: int = 32,
    cell_size: int = 8,
    bins: int = 9,
    top_k: int = 3,
    score_threshold: Optional[float] = None,
    margin_threshold: Optional[float] = None,
) -> Dict[str, object]:
    feats_by_label: Dict[str, List[np.ndarray]] = {}
    yellow_by_label: Dict[str, List[float]] = {}
    sample_vectors: List[Dict[str, object]] = []
    for cell, label in samples:
        glyph = normalize_glyph(cell, margin=margin, size=size)
        feat = hog_features(glyph, cell_size=cell_size, bins=bins)
        vec = _normalize_vec(feat)
        feats_by_label.setdefault(label, []).append(vec)
        yellow_by_label.setdefault(label, []).append(yellow_ratio(cell))
        sample_vectors.append(
            {
                "label": label,
                "vec": vec.tolist(),
            }
        )

    mean_vectors: Dict[str, List[float]] = {}
    stats: Dict[str, Dict[str, float]] = {}

    for label, feats in feats_by_label.items():
        mat = np.stack(feats, axis=0)
        mean = _normalize_vec(mat.mean(axis=0))
        mean_vectors[label] = mean.tolist()
        yellow_vals = yellow_by_label.get(label, [])
        stats[label] = {"count": float(len(feats))}
        if yellow_vals:
            stats[label]["yellow_mean"] = float(np.mean(yellow_vals))

    # Evaluate thresholds from training data
    labels = list(mean_vectors.keys())
    mean_vecs = {k: np.array(v, dtype=np.float32) for k, v in mean_vectors.items()}
    scores = []
    margins = []
    for cell, label in samples:
        glyph = normalize_glyph(cell, margin=margin, size=size)
        feat = _normalize_vec(hog_features(glyph, cell_size=cell_size, bins=bins))
        sims = [(lab, float(np.dot(feat, mean_vecs[lab]))) for lab in labels]
        sims.sort(key=lambda x: x[1], reverse=True)
        if sims and sims[0][0] == label:
            scores.append(sims[0][1])
            if len(sims) > 1:
                margins.append(sims[0][1] - sims[1][1])

    score_p5 = float(np.percentile(scores, 5)) if scores else 0.0
    margin_p5 = float(np.percentile(margins, 5)) if margins else 0.0
    if score_threshold is None:
        score_threshold = score_p5
    if margin_threshold is None:
        margin_threshold = margin_p5

    model = {
        "labels": labels,
        "mean_vectors": mean_vectors,
        "samples": sample_vectors,
        "params": {
            "margin": margin,
            "size": size,
            "cell_size": cell_size,
            "bins": bins,
            "top_k": top_k,
        },
        "score_threshold": score_threshold,
        "margin_threshold": margin_threshold,
        "threshold_stats": {
            "score_p5": score_p5,
            "margin_p5": margin_p5,
        },
        "stats": stats,
    }
    # Labels with yellow digits (used to gate candidates).
    yellow_labels = [
        label for label, info in stats.items() if info.get("yellow_mean", 0.0) > 0.5
    ]
    model["yellow_labels"] = yellow_labels
    return model


def save_model(path: Path, model: Dict[str, object]) -> None:
    path.write_text(json.dumps(model, indent=2))


def load_model(path: Path = Path("gray_tile_hog.json")) -> Optional[Dict[str, object]]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    if "mean_vectors" not in data or "params" not in data:
        return None
    return data


def predict_label(
    cell: np.ndarray,
    model: Dict[str, object],
    use_thresholds: bool = True,
    top_k: int = 3,
) -> Optional[str]:
    params = model.get("params", {})
    margin = float(params.get("margin", 0.1))
    size = int(params.get("size", 32))
    cell_size = int(params.get("cell_size", 8))
    bins = int(params.get("bins", 9))
    top_k = int(params.get("top_k", top_k))

    glyph = normalize_glyph(cell, margin=margin, size=size)
    feat = hog_features(glyph, cell_size=cell_size, bins=bins)
    if feat.size == 0:
        return None
    feat = _normalize_vec(feat)

    samples = model.get("samples", [])
    if not samples:
        mean_vectors = {
            label: np.array(vec, dtype=np.float32)
            for label, vec in model.get("mean_vectors", {}).items()
        }
        if not mean_vectors:
            return None
        sims = [(lab, float(np.dot(feat, vec))) for lab, vec in mean_vectors.items()]
    else:
        is_yellow = yellow_ratio(cell) > 0.5
        yellow_labels = set(model.get("yellow_labels", []))
        sims = []
        for sample in samples:
            label = str(sample.get("label"))
            if yellow_labels:
                if is_yellow and label not in yellow_labels:
                    continue
                if not is_yellow and label in yellow_labels:
                    continue
            vec = np.array(sample.get("vec", []), dtype=np.float32)
            if vec.size == 0:
                continue
            sims.append((label, float(np.dot(feat, vec))))
    if not sims:
        return None
    sims.sort(key=lambda x: x[1], reverse=True)
    best_label, best_score = sims[0]
    second_score = sims[1][1] if len(sims) > 1 else -1.0

    if samples:
        # Use a weighted vote over the top-k neighbors to reduce tie bias.
        top = sims[: max(1, top_k)]
        totals: Dict[str, float] = {}
        for label, score in top:
            totals[label] = totals.get(label, 0.0) + score
        best_label = max(totals.items(), key=lambda x: x[1])[0]
        best_score = max(score for _label, score in top)
        second_score = 0.0
    if use_thresholds:
        score_threshold = float(model.get("score_threshold", 0.0))
        if best_score < score_threshold:
            return None
        if not samples:
            margin_threshold = float(model.get("margin_threshold", 0.0))
            if (best_score - second_score) < margin_threshold:
                return None
    return str(best_label)
