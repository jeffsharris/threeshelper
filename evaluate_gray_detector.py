import argparse
import csv
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

import window_stream as ws


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate gray-tile detector on labeled tiles.")
    parser.add_argument(
        "--labels-dir",
        type=Path,
        help="Path to datasets/gray_labels/<session>. Defaults to latest.",
    )
    args = parser.parse_args()

    base_dir = Path("datasets")
    labels_dir = args.labels_dir or find_latest_labels(base_dir)
    labels_path = labels_dir / "labels.csv"
    tiles_dir = labels_dir / "tiles"

    labels = load_labels(labels_path)
    if not labels:
        raise RuntimeError(f"No labels found in {labels_path}")

    total = 0
    correct = 0
    unknown = 0
    per_label: Dict[str, Dict[str, int]] = {}

    for tile_id, label in labels.items():
        tile_path = tiles_dir / f"{tile_id}.png"
        if not tile_path.exists():
            continue
        arr = np.array(Image.open(tile_path).convert("RGB"))
        pred = ws.classify_gray_tile(arr)
        total += 1
        per_label.setdefault(label, {"correct": 0, "total": 0, "unknown": 0})
        per_label[label]["total"] += 1
        if not pred:
            unknown += 1
            per_label[label]["unknown"] += 1
            continue
        if pred == label:
            correct += 1
            per_label[label]["correct"] += 1

    acc = (correct / total) * 100.0 if total else 0.0
    unk = (unknown / total) * 100.0 if total else 0.0
    print(f"Total: {total}")
    print(f"Correct: {correct} ({acc:.1f}%)")
    print(f"Unknown: {unknown} ({unk:.1f}%)")
    print("Per-label:")
    for label in sorted(per_label.keys(), key=lambda x: int(x) if x.isdigit() else x):
        stats = per_label[label]
        tot = stats["total"]
        corr = stats["correct"]
        unk_count = stats["unknown"]
        corr_pct = (corr / tot) * 100.0 if tot else 0.0
        unk_pct = (unk_count / tot) * 100.0 if tot else 0.0
        print(f"  {label}: {corr}/{tot} correct ({corr_pct:.1f}%), {unk_count} unknown ({unk_pct:.1f}%)")


if __name__ == "__main__":
    main()
