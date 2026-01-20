import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

import gray_hog as gh


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


def load_samples(labels_path: Path, tiles_dir: Path) -> List[Tuple[np.ndarray, str]]:
    labels = load_labels(labels_path)
    samples: List[Tuple[np.ndarray, str]] = []
    for tile_id, label in labels.items():
        tile_path = tiles_dir / f"{tile_id}.png"
        if not tile_path.exists():
            continue
        arr = np.array(Image.open(tile_path).convert("RGB"))
        samples.append((arr, label))
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a HOG-based gray tile classifier.")
    parser.add_argument(
        "--labels-dir",
        type=Path,
        help="Path to datasets/gray_labels/<session>. Defaults to latest.",
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        action="append",
        help="Extra labels CSV to include (tile_id,label).",
    )
    parser.add_argument(
        "--tiles-dir",
        type=Path,
        action="append",
        help="Tiles directory for each extra labels CSV (same order).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("gray_tile_hog.json"),
        help="Output model path (default: gray_tile_hog.json).",
    )
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--cell-size", type=int, default=8)
    parser.add_argument("--bins", type=int, default=9)
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Top-k neighbors for the kNN vote (default: 3).",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.4,
        help="Minimum similarity score to accept a label (default: 0.4).",
    )
    parser.add_argument(
        "--margin-threshold",
        type=float,
        default=0.02,
        help="Minimum best-vs-second margin to accept a label (default: 0.02).",
    )
    args = parser.parse_args()

    base_dir = Path("datasets")
    labels_dir = args.labels_dir or find_latest_labels(base_dir)
    base_samples = load_samples(labels_dir / "labels.csv", labels_dir / "tiles")

    extra_samples: List[Tuple[np.ndarray, str]] = []
    if args.labels_csv or args.tiles_dir:
        if not args.labels_csv or not args.tiles_dir or len(args.labels_csv) != len(args.tiles_dir):
            raise RuntimeError("--labels-csv and --tiles-dir must be provided the same number of times")
        for labels_csv, tiles_dir in zip(args.labels_csv, args.tiles_dir):
            extra_samples.extend(load_samples(labels_csv, tiles_dir))

    samples = base_samples + extra_samples
    if not samples:
        raise RuntimeError("No labeled samples found.")

    model = gh.train_hog_model(
        samples,
        margin=args.margin,
        size=args.size,
        cell_size=args.cell_size,
        bins=args.bins,
        top_k=args.top_k,
        score_threshold=args.score_threshold,
        margin_threshold=args.margin_threshold,
    )
    gh.save_model(args.out, model)
    print(f"Wrote model to {args.out}")
    print(f"Labels: {sorted(model['labels'])}")
    print(f"score_threshold={model['score_threshold']:.3f} margin_threshold={model['margin_threshold']:.3f}")


if __name__ == "__main__":
    main()
