import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

from PIL import Image, ImageOps
import imagehash


def prep_image_for_hash(img: Image.Image, margin: float = 0.08) -> Image.Image:
    """Mirror the normalization used in the main script before hashing."""
    img = img.convert("L")
    w, h = img.size
    mw = int(w * margin)
    mh = int(h * margin)
    img = img.crop((mw, mh, w - mw, h - mh))
    return ImageOps.autocontrast(img)


def compute_hash(path: Path, margin: float) -> imagehash.ImageHash:
    return imagehash.phash(prep_image_for_hash(Image.open(path), margin=margin))


def format_distances(distances: Iterable[Tuple[str, int]]) -> str:
    return ", ".join(f"{name}:{dist}" for name, dist in distances)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute perceptual hashes for tiles and (optionally) distances to reference tiles."
    )
    parser.add_argument("images", nargs="+", help="Tile image paths to hash.")
    parser.add_argument(
        "--refs",
        nargs="+",
        default=[],
        help="Reference image paths to compare against (min distance is reported).",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.08,
        help="Crop margin ratio before hashing (default 0.08).",
    )
    args = parser.parse_args()

    image_paths = [Path(p) for p in args.images]
    ref_paths = [Path(p) for p in args.refs]

    ref_hashes: List[Tuple[str, imagehash.ImageHash]] = []
    if ref_paths:
        for ref in ref_paths:
            h = compute_hash(ref, args.margin)
            ref_hashes.append((ref.name, h))
        print("Reference hashes:")
        for name, h in ref_hashes:
            print(f"  {name}: {h}")
        if len(ref_hashes) > 1:
            print("Reference pairwise distances:")
            for i in range(len(ref_hashes)):
                for j in range(i + 1, len(ref_hashes)):
                    d = abs(ref_hashes[i][1] - ref_hashes[j][1])
                    print(f"  {ref_hashes[i][0]} <-> {ref_hashes[j][0]}: {d}")
        print()

    print("Image hashes:")
    for img_path in image_paths:
        h = compute_hash(img_path, args.margin)
        line = f"  {img_path.name}: {h}"
        if ref_hashes:
            dists = [(name, abs(h - ref_hash)) for name, ref_hash in ref_hashes]
            min_dist = min(d[1] for d in dists)
            line += f" | min_dist_to_refs={min_dist} ({format_distances(dists)})"
        print(line)


if __name__ == "__main__":
    main()
