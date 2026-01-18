import argparse
import colorsys
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import window_stream as ws


def find_latest_session(base_dir: Path) -> Path:
    sessions = sorted([p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith("session_")])
    if not sessions:
        raise FileNotFoundError(f"No sessions found in {base_dir}")
    return sessions[-1]


def load_meta_paths(session_dir: Path) -> List[Path]:
    return sorted(session_dir.glob("*_meta.json"))


def _get_font(size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype("Arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    try:
        box = draw.textbbox((0, 0), text, font=font)
        return box[2] - box[0], box[3] - box[1]
    except Exception:
        try:
            return draw.textsize(text, font=font)
        except Exception:
            return (0, 0)


def write_labels_csv(path: Path, tile_ids: List[str]) -> None:
    if path.exists():
        return
    with path.open("w") as f:
        f.write("tile_id,label\n")
        for tile_id in tile_ids:
            f.write(f"{tile_id},\n")


def make_contact_sheets(
    tiles: List[Tuple[str, Path]],
    out_dir: Path,
    tile_size: int,
    cols: int,
    rows: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    font = _get_font(12)
    pad = 10
    label_h = 16
    page_w = cols * tile_size + (cols + 1) * pad
    page_h = rows * (tile_size + label_h) + (rows + 1) * pad

    page_idx = 0
    for i in range(0, len(tiles), cols * rows):
        chunk = tiles[i : i + cols * rows]
        sheet = Image.new("RGB", (page_w, page_h), (20, 20, 28))
        draw = ImageDraw.Draw(sheet)
        for idx, (tile_id, path) in enumerate(chunk):
            r = idx // cols
            c = idx % cols
            x0 = pad + c * (tile_size + pad)
            y0 = pad + r * (tile_size + label_h + pad)
            try:
                tile_img = Image.open(path).convert("RGB")
            except Exception:
                continue
            sheet.paste(tile_img, (x0, y0))
            text = tile_id
            tw, th = _text_size(draw, text, font)
            tx = x0 + max(0, (tile_size - tw) // 2)
            ty = y0 + tile_size + 2
            draw.text((tx, ty), text, fill=(220, 220, 230), font=font)
        sheet_path = out_dir / f"sheet_{page_idx:03d}.png"
        sheet.save(sheet_path)
        page_idx += 1


def is_gray_candidate(cell: np.ndarray, blank_threshold: float = 60.0) -> bool:
    h, w, _ = cell.shape
    mh = int(h * 0.08)
    mw = int(w * 0.08)
    trimmed = cell[mh : h - mh, mw : w - mw]
    mean_rgb = trimmed.reshape(-1, 3).mean(axis=0)
    if trimmed.reshape(-1, 3).mean() < blank_threshold:
        return False
    red_dist = float(np.linalg.norm(mean_rgb - ws.BOARD_COLOR_PROTOTYPES["red"]))
    blue_dist = float(np.linalg.norm(mean_rgb - ws.BOARD_COLOR_PROTOTYPES["blue"]))
    # If strongly close to red/blue with sufficient saturation, skip.
    r, g, b = (mean_rgb / 255.0).tolist()
    _h, s, _v = colorsys.rgb_to_hsv(r, g, b)
    if s > 0.18 and (red_dist < 80 or blue_dist < 80):
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract gray tiles into labeled sheets.")
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=Path("datasets"),
        help="Base datasets directory (default: datasets).",
    )
    parser.add_argument(
        "--session",
        type=str,
        help="Session directory name or full path. Defaults to latest session.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Output directory (default: datasets/gray_labels/session_YYYYMMDD_HHMMSS).",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=96,
        help="Square size in pixels for extracted tile images.",
    )
    parser.add_argument(
        "--inset-ratio",
        type=float,
        default=0.12,
        help="Inset ratio inside each cell to avoid neighboring tiles.",
    )
    parser.add_argument(
        "--blank-threshold",
        type=float,
        default=60.0,
        help="Mean brightness threshold to treat a tile as non-empty.",
    )
    parser.add_argument(
        "--sheet-cols",
        type=int,
        default=10,
        help="Number of columns per contact sheet.",
    )
    parser.add_argument(
        "--sheet-rows",
        type=int,
        default=10,
        help="Number of rows per contact sheet.",
    )
    args = parser.parse_args()

    if args.session:
        session_path = Path(args.session)
        if not session_path.exists():
            session_path = args.datasets_dir / args.session
    else:
        session_path = find_latest_session(args.datasets_dir)

    if not session_path.exists():
        raise FileNotFoundError(f"Session not found: {session_path}")

    if args.out_dir:
        out_dir = args.out_dir
    else:
        session_name = session_path.name
        out_dir = args.datasets_dir / "gray_labels" / session_name

    tiles_dir = out_dir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    sheets_dir = out_dir / "sheets"
    index_path = out_dir / "index.jsonl"
    labels_path = out_dir / "labels.csv"

    tile_entries: List[Dict[str, object]] = []
    tile_paths: List[Tuple[str, Path]] = []

    for meta_path in load_meta_paths(session_path):
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue
        full_path = session_path / meta["paths"]["full"]
        try:
            full_img = Image.open(full_path).convert("RGB")
        except Exception:
            continue
        arr = np.array(full_img)
        grid = ws.classify_board(arr)
        cells, _roi = ws.segment_board_cells_with_boxes(arr, inset_ratio=args.inset_ratio)
        capture_id = int(meta.get("id", 0))
        for r, c, _box, cell in cells:
            token = grid[r][c] if r < len(grid) and c < len(grid[r]) else "?"
            if token in (ws.SMALL_COLOR_MAP["red"], ws.SMALL_COLOR_MAP["blue"], ws.TOKEN_EMPTY):
                continue
            if not is_gray_candidate(cell, blank_threshold=args.blank_threshold):
                continue
            tile_id = f"{capture_id:06d}_r{r}_c{c}"
            tile_path = tiles_dir / f"{tile_id}.png"
            tile_img = Image.fromarray(cell).resize((args.tile_size, args.tile_size), Image.BILINEAR)
            tile_img.save(tile_path)
            tile_entries.append(
                {
                    "tile_id": tile_id,
                    "session": session_path.name,
                    "capture_id": capture_id,
                    "row": r,
                    "col": c,
                    "source": str(full_path.name),
                    "image": str(tile_path.name),
                }
            )
            tile_paths.append((tile_id, tile_path))

    with index_path.open("w") as f:
        for entry in tile_entries:
            f.write(json.dumps(entry) + "\n")

    tile_ids = [tile_id for tile_id, _ in tile_paths]
    write_labels_csv(labels_path, tile_ids)
    make_contact_sheets(tile_paths, sheets_dir, args.tile_size, args.sheet_cols, args.sheet_rows)

    print(f"Extracted {len(tile_entries)} gray tiles to {out_dir}")
    print(f"Sheets: {sheets_dir}")
    print(f"Labels CSV: {labels_path}")


if __name__ == "__main__":
    main()
