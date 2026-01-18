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
    return np.array(img, dtype=np.uint8)


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
    return (arr < t).astype(np.float32)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(((a - b) ** 2).mean())


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.array(values, dtype=np.float32), p))


def kmeans(X: np.ndarray, k: int, iters: int = 25) -> np.ndarray:
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
    parser = argparse.ArgumentParser(description="Train a gray-tile detector from labeled tiles.")
    parser.add_argument(
        "--labels-dir",
        type=Path,
        help="Path to datasets/gray_labels/<session>. Defaults to latest.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("gray_tile_detector.json"),
        help="Output JSON path (default: gray_tile_detector.json).",
    )
    parser.add_argument(
        "--margins",
        type=str,
        default="0.12",
        help="Comma-separated crop margins to build detectors.",
    )
    parser.add_argument(
        "--k-templates",
        type=int,
        default=4,
        help="Number of template clusters per label.",
    )
    parser.add_argument(
        "--gap-threshold",
        type=float,
        default=0.0,
        help="Minimum second-best gap to accept a label (default: 0.0).",
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
    stats: Dict[str, Dict[str, float]] = {}

    for margin in margins:
        label_masks: Dict[str, List[np.ndarray]] = {}
        for tile_id, label in labels.items():
            tile_path = tiles_dir / f"{tile_id}.png"
            if not tile_path.exists():
                continue
            arr = preprocess_tile(tile_path, margin=margin)
            mask = mask_from_arr(arr)
            label_masks.setdefault(label, []).append(mask)

        if not label_masks:
            continue

        label_templates: Dict[str, List[List[List[float]]]] = {}
        label_thresholds: Dict[str, float] = {}
        label_stats: Dict[str, Dict[str, float]] = {}

        for label, masks in label_masks.items():
            if not masks:
                continue
            X = np.stack([m.reshape(-1) for m in masks], axis=0)
            k = min(args.k_templates, len(masks))
            centers = kmeans(X, k)
            templates = centers.reshape(k, masks[0].shape[0], masks[0].shape[1])

            def min_dist(mask: np.ndarray) -> float:
                return float(min(mse(mask, t) for t in templates))

            pos_dists = [min_dist(m) for m in masks]
            pos_p95 = percentile(pos_dists, 95)
            threshold = pos_p95 * 1.1 + 0.003

            label_templates[label] = [t.tolist() for t in templates]
            label_thresholds[label] = threshold
            label_stats[label] = {
                "count": float(len(masks)),
                "pos_p50": percentile(pos_dists, 50),
                "pos_p95": pos_p95,
            }
            stats[label] = label_stats[label]

        detectors.append(
            {
                "margin": margin,
                "size": next(iter(label_masks.values()))[0].shape[0],
                "labels": label_templates,
                "thresholds": label_thresholds,
                "stats": label_stats,
            }
        )

    if not detectors:
        raise RuntimeError("No labeled tiles found; cannot train detector.")

    gap_threshold = max(0.0, args.gap_threshold)
    gap_stats: Dict[str, float] = {}
    if detectors:
        det = detectors[0]
        gap_values: List[float] = []
        margin = float(det.get("margin", 0.12))
        label_templates = det.get("labels", {})

        def min_dist(mask: np.ndarray, templates: List[List[List[float]]]) -> float:
            tmpl_arr = [np.array(t, dtype=np.float32) for t in templates]
            return float(min(mse(mask, t) for t in tmpl_arr))

        for tile_id, true_label in labels.items():
            tile_path = tiles_dir / f"{tile_id}.png"
            if not tile_path.exists() or true_label not in label_templates:
                continue
            arr = preprocess_tile(tile_path, margin=margin)
            mask = mask_from_arr(arr)
            dist_pairs: List[Tuple[str, float]] = []
            for label, templates in label_templates.items():
                if not templates:
                    continue
                dist_pairs.append((label, min_dist(mask, templates)))
            if len(dist_pairs) < 2:
                continue
            dist_pairs.sort(key=lambda x: x[1])
            best_label, best_dist = dist_pairs[0]
            second_dist = dist_pairs[1][1]
            if best_label == true_label:
                gap_values.append(second_dist - best_dist)
        if gap_values:
            gap_stats = {
                "gap_p10": percentile(gap_values, 10),
                "gap_p25": percentile(gap_values, 25),
                "gap_p50": percentile(gap_values, 50),
            }

    data = {
        "detectors": detectors,
        "gap_threshold": gap_threshold,
        "labels_dir": str(labels_dir),
        "stats": stats,
        "gap_stats": gap_stats,
    }

    args.out.write_text(json.dumps(data, indent=2))
    print(f"Wrote detector to {args.out}")
    print(json.dumps({"gap_threshold": gap_threshold, "labels": stats}, indent=2))


if __name__ == "__main__":
    main()
