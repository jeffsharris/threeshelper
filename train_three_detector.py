import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageOps


def find_latest_labels(base_dir: Path) -> Path:
    labels_dir = base_dir / "gray_labels"
    sessions = sorted([p for p in labels_dir.iterdir() if p.is_dir()])
    if not sessions:
        raise FileNotFoundError(f"No gray_labels sessions found in {labels_dir}")
    return sessions[-1]


def load_labels(labels_path: Path) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    with labels_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tile_id = (row.get("tile_id") or "").strip()
            label = (row.get("label") or "").strip()
            if tile_id and label:
                labels[tile_id] = label
    return labels


def preprocess_tile(path: Path, size: int = 64, margin: float = 0.12) -> np.ndarray:
    img = Image.open(path).convert("L")
    w, h = img.size
    mw = int(w * margin)
    mh = int(h * margin)
    img = img.crop((mw, mh, w - mw, h - mh))
    img = ImageOps.autocontrast(img)
    img = img.resize((size, size), Image.BILINEAR)
    arr = np.array(img, dtype=np.uint8)
    return arr


def otsu_threshold(arr: np.ndarray) -> int:
    hist = np.bincount(arr.flatten(), minlength=256).astype(np.float64)
    total = arr.size
    sum_total = np.dot(np.arange(256), hist)
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


def mask_from_arr(arr: np.ndarray) -> np.ndarray:
    t = otsu_threshold(arr)
    # Dark digits on light background.
    return (arr < t).astype(np.float32)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(((a - b) ** 2).mean())


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.array(values, dtype=np.float32), p))


def kmeans(X: np.ndarray, k: int, iters: int = 20) -> np.ndarray:
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X), size=k, replace=False)
    centers = X[idx].copy()
    for _ in range(iters):
        dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        labels = dists.argmin(axis=1)
        for j in range(k):
            members = X[labels == j]
            if len(members) > 0:
                centers[j] = members.mean(axis=0)
    return centers


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a 3-tile detector from labeled gray tiles.")
    parser.add_argument(
        "--labels-dir",
        type=Path,
        help="Path to datasets/gray_labels/<session>. Defaults to latest.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("three_detector.json"),
        help="Output JSON path (default: three_detector.json).",
    )
    parser.add_argument(
        "--margins",
        type=str,
        default="0.08,0.12",
        help="Comma-separated crop margins to build multiple detectors.",
    )
    parser.add_argument(
        "--k-templates",
        type=int,
        default=5,
        help="Number of template clusters for 3 tiles.",
    )
    args = parser.parse_args()

    base_dir = Path("datasets")
    labels_dir = args.labels_dir or find_latest_labels(base_dir)
    labels_path = labels_dir / "labels.csv"
    tiles_dir = labels_dir / "tiles"

    labels = load_labels(labels_path)
    if not labels:
        raise RuntimeError(f"No labels found in {labels_path}")

    margins = [float(m.strip()) for m in args.margins.split(",") if m.strip()]
    detectors = []
    global_stats = {"three_count": 0, "other_count": 0}

    for margin in margins:
        three_masks: List[np.ndarray] = []
        other_masks: List[np.ndarray] = []
        for tile_id, label in labels.items():
            tile_path = tiles_dir / f"{tile_id}.png"
            if not tile_path.exists():
                continue
            arr = preprocess_tile(tile_path, margin=margin)
            mask = mask_from_arr(arr)
            if label == "3":
                three_masks.append(mask)
            else:
                other_masks.append(mask)

        if not three_masks:
            continue

        X = np.stack([m.reshape(-1) for m in three_masks], axis=0)
        k = min(args.k_templates, len(three_masks))
        centers = kmeans(X, k)
        templates = centers.reshape(k, three_masks[0].shape[0], three_masks[0].shape[1])

        def min_dist(mask: np.ndarray) -> float:
            return float(min(mse(mask, t) for t in templates))

        pos_dists = [min_dist(m) for m in three_masks]
        neg_dists = [min_dist(m) for m in other_masks]

        pos_p95 = percentile(pos_dists, 95)
        neg_p5 = percentile(neg_dists, 5) if neg_dists else pos_p95 * 1.5

        if pos_p95 < neg_p5:
            threshold = (pos_p95 + neg_p5) / 2.0
        else:
            threshold = pos_p95 * 1.05

        detectors.append(
            {
                "margin": margin,
                "size": templates.shape[1],
                "k": k,
                "threshold": threshold,
                "templates": [t.tolist() for t in templates],
                "stats": {
                    "three_count": len(three_masks),
                    "other_count": len(other_masks),
                    "pos_p50": percentile(pos_dists, 50),
                    "pos_p95": pos_p95,
                    "neg_p5": neg_p5,
                },
            }
        )
        global_stats["three_count"] = len(three_masks)
        global_stats["other_count"] = len(other_masks)

    if not detectors:
        raise RuntimeError("No '3' labels found; cannot train detector.")

    data = {
        "detectors": detectors,
        "labels_dir": str(labels_dir),
        "stats": global_stats,
    }

    args.out.write_text(json.dumps(data, indent=2))
    print(f"Wrote detector to {args.out}")
    print(json.dumps(data["stats"], indent=2))


if __name__ == "__main__":
    main()
