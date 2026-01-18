import argparse
import base64
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI

import window_stream as ws


TOKEN_RED = ws.SMALL_COLOR_MAP["red"]
TOKEN_BLUE = ws.SMALL_COLOR_MAP["blue"]
TOKEN_GRAY = ws.CELL_GRAY_TOKEN
TOKEN_OTHER = ws.TOKEN_OTHER
TOKEN_EMPTY = ws.TOKEN_EMPTY

COLOR_BG = (24, 25, 36)
COLOR_GRID = (54, 58, 75)
COLOR_RED = (232, 109, 130)
COLOR_BLUE = (118, 184, 236)
COLOR_GRAY = (138, 148, 168)
COLOR_DARK = (35, 36, 50)
COLOR_TEXT = (235, 235, 240)


@dataclass
class Capture:
    capture_id: int
    meta_path: Path
    full_path: Path
    label: Optional[str]


def _read_json(path: Path) -> Dict:
    return json.loads(path.read_text())


def find_latest_session(base_dir: Path) -> Path:
    sessions = sorted([p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith("session_")])
    if not sessions:
        raise FileNotFoundError(f"No sessions found in {base_dir}")
    return sessions[-1]


def load_captures(session_dir: Path) -> List[Capture]:
    captures: List[Capture] = []
    for meta_path in sorted(session_dir.glob("*_meta.json")):
        meta = _read_json(meta_path)
        capture_id = int(meta.get("id"))
        label = meta.get("label")
        full_path = session_dir / meta["paths"]["full"]
        captures.append(
            Capture(
                capture_id=capture_id,
                meta_path=meta_path,
                full_path=full_path,
                label=label,
            )
        )
    return captures


def board_to_ascii(board: List[List[str]]) -> str:
    def conv(tok: str) -> str:
        if tok == TOKEN_RED:
            return "R"
        if tok == TOKEN_BLUE:
            return "B"
        if tok in (TOKEN_GRAY, TOKEN_OTHER):
            return "G"
        if tok == TOKEN_EMPTY:
            return "."
        return "?"

    lines = []
    for row in board:
        lines.append(" ".join(conv(tok) for tok in row))
    return "\n".join(lines)


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


def render_predicted_board(board: List[List[str]]) -> Image.Image:
    cell = 70
    gap = 8
    pad = 16
    header_h = 50
    board_w = cell * 4 + gap * 3
    board_h = cell * 4 + gap * 3
    width = board_w + pad * 2
    height = header_h + board_h + pad * 2
    img = Image.new("RGB", (width, height), COLOR_BG)
    draw = ImageDraw.Draw(img)

    draw.text((pad, pad), "predicted board", fill=COLOR_TEXT, font=_get_font(16))

    start_x = pad
    start_y = pad + header_h
    for r in range(4):
        for c in range(4):
            x0 = start_x + c * (cell + gap)
            y0 = start_y + r * (cell + gap)
            x1 = x0 + cell
            y1 = y0 + cell
            tok = board[r][c] if r < len(board) and c < len(board[r]) else "?"
            fill = COLOR_DARK
            if tok == TOKEN_RED:
                fill = COLOR_RED
            elif tok == TOKEN_BLUE:
                fill = COLOR_BLUE
            elif tok in (TOKEN_GRAY, TOKEN_OTHER):
                fill = COLOR_GRAY
            elif tok == TOKEN_EMPTY:
                fill = COLOR_DARK
            draw.rectangle((x0, y0, x1, y1), fill=fill, outline=COLOR_GRID, width=2)
    return img


def build_composite(board_crop: Image.Image, board_img: Image.Image, max_width: int) -> Image.Image:
    if board_crop.width > max_width:
        scale = max_width / float(board_crop.width)
        new_size = (max_width, int(board_crop.height * scale))
        board_crop = board_crop.resize(new_size, Image.LANCZOS)

    pad = 16
    width = board_crop.width + pad + board_img.width
    height = max(board_crop.height, board_img.height)
    composite = Image.new("RGB", (width, height), COLOR_BG)
    composite.paste(board_crop, (0, 0))
    composite.paste(board_img, (board_crop.width + pad, 0))
    return composite


def encode_image(path: Path) -> str:
    data = path.read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def build_prompt() -> str:
    return (
        "You are given a crop of the Threes 4x4 board. "
        "Extract the board into a 4x4 grid using tokens:\n"
        "- R for red tiles (2)\n"
        "- B for blue tiles (1)\n"
        "- G for any gray tile (3, 6, 12, 24, ...)\n"
        "- . for empty\n"
        "- ? if you are unsure\n"
        "Return rows top-to-bottom, columns left-to-right."
    )


def run_eval(
    session_dir: Path,
    model: str,
    max_width: int,
    limit: Optional[int],
    live: bool,
) -> Path:
    captures = load_captures(session_dir)
    labeled = [c for c in captures if c.label in ("correct", "incorrect")]
    if limit is not None:
        labeled = labeled[:limit]

    composites_dir = session_dir / "eval_composites"
    composites_dir.mkdir(exist_ok=True)

    out_jsonl = session_dir / "evaluation_results.jsonl"
    out_csv = session_dir / "evaluation_results.csv"

    client = OpenAI() if live else None

    total = 0
    matches = 0

    with out_jsonl.open("w") as fj, out_csv.open("w", newline="") as fc:
        writer = csv.writer(fc)
        writer.writerow(["id", "label", "is_correct", "match", "reason", "composite"])

        for cap in labeled:
            full_img = Image.open(cap.full_path).convert("RGB")
            arr = np.array(full_img)
            pred_board = ws.classify_board(arr)
            board_img = render_predicted_board(pred_board)
            board_roi, _ = ws.find_board_roi(arr)
            board_crop = Image.fromarray(board_roi).convert("RGB")
            composite = build_composite(board_crop, board_img, max_width=max_width)
            composite_path = composites_dir / f"{cap.capture_id:06d}_composite.png"
            composite.save(composite_path)

            prompt = build_prompt()

            is_correct = None
            reason = "dry-run"
            extracted = None
            if live and client is not None:
                image_url = encode_image(composite_path)
                response = client.responses.create(
                    model=model,
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": prompt},
                                {"type": "input_image", "image_url": image_url},
                            ],
                        }
                    ],
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "board_grid",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "grid": {
                                        "type": "array",
                                        "items": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                            "minItems": 4,
                                            "maxItems": 4,
                                        },
                                        "minItems": 4,
                                        "maxItems": 4,
                                    },
                                    "note": {"type": "string"},
                                },
                                "required": ["grid", "note"],
                                "additionalProperties": False,
                            },
                            "strict": True,
                        }
                    },
                    max_output_tokens=200,
                )
                try:
                    data = json.loads(response.output_text)
                    extracted = data.get("grid")
                    reason = data.get("note", "")
                except json.JSONDecodeError:
                    extracted = None
                    reason = response.output_text.strip() if response.output_text else ""

            pred_ascii = board_to_ascii(pred_board).splitlines()
            pred_grid = [line.split() for line in pred_ascii] if pred_ascii else []
            mismatches = 0
            unknowns = 0
            if extracted:
                for r in range(4):
                    for c in range(4):
                        exp = extracted[r][c]
                        got = pred_grid[r][c] if r < len(pred_grid) and c < len(pred_grid[r]) else "?"
                        if exp == "?":
                            unknowns += 1
                            continue
                        if exp != got:
                            mismatches += 1
            if extracted is None:
                is_correct = None
            else:
                is_correct = mismatches == 0 and unknowns <= 2

            expected = cap.label == "correct"
            match = (is_correct == expected) if is_correct is not None else False
            total += 1
            matches += int(match)

            record = {
                "id": cap.capture_id,
                "label": cap.label,
                "is_correct": is_correct,
                "match": match,
                "reason": reason,
                "mismatches": mismatches,
                "unknowns": unknowns,
                "extracted_grid": extracted,
                "pred_grid": pred_grid,
                "composite": composite_path.name,
            }
            fj.write(json.dumps(record) + "\n")
            writer.writerow([cap.capture_id, cap.label, is_correct, match, reason, composite_path.name])

    accuracy = matches / total if total else 0.0
    summary = {
        "total": total,
        "matches": matches,
        "accuracy": accuracy,
    }
    (session_dir / "evaluation_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Accuracy: {matches}/{total} = {accuracy:.1%}")
    return out_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate board classification against labeled captures using a vision judge."
    )
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
        "--model",
        type=str,
        default="gpt-4.1",
        help="OpenAI model for vision evaluation.",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=1000,
        help="Max width for the screenshot portion of composites.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of captures to evaluate.",
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

    live = os.environ.get("OPENAI_RUN_LIVE", "0") == "1"
    out_jsonl = run_eval(
        session_path,
        model=args.model,
        max_width=args.max_width,
        limit=args.limit,
        live=live,
    )
    print(f"Wrote evaluation to {out_jsonl}")
    if not live:
        print("Dry-run: set OPENAI_RUN_LIVE=1 to call the API.")


if __name__ == "__main__":
    main()
