import argparse
import base64
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI


TOKEN_RED = "\U0001F7E5"
TOKEN_BLUE = "\U0001F7E6"
TOKEN_GRAY = "3"
TOKEN_EMPTY = "\u00b7"
TOKEN_OTHER = "X"

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
    board: List[List[str]]
    preview_label: str
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
        full_path = session_dir / meta["paths"]["full"]
        captures.append(
            Capture(
                capture_id=capture_id,
                meta_path=meta_path,
                full_path=full_path,
                board=meta.get("board", []),
                preview_label=meta.get("preview_label", "unknown"),
                label=meta.get("label"),
            )
        )
    return captures


def board_to_ascii(board: List[List[str]]) -> str:
    def conv(tok: str) -> str:
        if tok == TOKEN_RED:
            return "R"
        if tok == TOKEN_BLUE:
            return "B"
        if tok == TOKEN_GRAY:
            return "G"
        if tok == TOKEN_OTHER:
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
            return draw.textsize(text, font=font)  # Pillow <10 fallback
        except Exception:
            return (0, 0)


def render_predicted_board(board: List[List[str]], preview_label: str) -> Image.Image:
    cell = 70
    gap = 8
    pad = 16
    header_h = 70
    board_w = cell * 4 + gap * 3
    board_h = cell * 4 + gap * 3
    width = board_w + pad * 2
    height = header_h + board_h + pad * 2
    img = Image.new("RGB", (width, height), COLOR_BG)
    draw = ImageDraw.Draw(img)

    # Header
    draw.text((pad, pad), f"predicted board", fill=COLOR_TEXT, font=_get_font(16))
    preview_color = COLOR_GRAY
    if preview_label == "red":
        preview_color = COLOR_RED
    elif preview_label == "blue":
        preview_color = COLOR_BLUE
    elif preview_label == "gray":
        preview_color = COLOR_GRAY
    elif preview_label == "large_candidates":
        preview_color = (220, 200, 120)
    draw.rectangle(
        (pad, pad + 26, pad + 36, pad + 62),
        fill=preview_color,
        outline=COLOR_GRID,
    )
    draw.text((pad + 46, pad + 34), f"preview: {preview_label}", fill=COLOR_TEXT, font=_get_font(14))

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
            text = ""
            if tok == TOKEN_RED:
                fill = COLOR_RED
            elif tok == TOKEN_BLUE:
                fill = COLOR_BLUE
            elif tok == TOKEN_EMPTY:
                fill = COLOR_DARK
            elif tok in (TOKEN_GRAY, TOKEN_OTHER):
                fill = COLOR_GRAY
            else:
                fill = COLOR_GRAY
                text = ""
            draw.rectangle((x0, y0, x1, y1), fill=fill, outline=COLOR_GRID, width=2)
            if text:
                font = _get_font(20)
                tw, th = _text_size(draw, text, font)
                draw.text((x0 + (cell - tw) / 2, y0 + (cell - th) / 2), text, fill=COLOR_TEXT, font=font)
    return img


def build_composite(full_img: Image.Image, board_img: Image.Image, max_width: int) -> Image.Image:
    if full_img.width > max_width:
        scale = max_width / float(full_img.width)
        new_size = (max_width, int(full_img.height * scale))
        full_img = full_img.resize(new_size, Image.LANCZOS)

    pad = 16
    width = full_img.width + pad + board_img.width
    height = max(full_img.height, board_img.height)
    composite = Image.new("RGB", (width, height), COLOR_BG)
    composite.paste(full_img, (0, 0))
    composite.paste(board_img, (full_img.width + pad, 0))
    return composite


def encode_image(path: Path) -> str:
    data = path.read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def build_prompt(board_ascii: str) -> str:
    return (
        "Compare the Threes screenshot (left) to the predicted board rendering (right). "
        "The predicted board grid is also shown below. "
        "All gray tiles are shown as generic gray (no numbers). "
        "Do not count numeric value differences among gray tiles as errors. "
        "Treat any gray tile in the screenshot as matching any gray tile in the prediction. "
        "Focus on tile placement, missing tiles (empty vs tile), or red/blue swaps. "
        "Return 1-2 sentences describing why the prediction is incorrect. "
        "If you cannot confidently spot a discrepancy, say 'uncertain' and explain what is unclear.\n\n"
        f"Predicted grid (R=red, B=blue, G=any gray tile, .=empty):\n{board_ascii}"
    )


def run_analysis(
    session_dir: Path,
    model: str,
    max_width: int,
    limit: Optional[int],
    live: bool,
) -> Path:
    captures = load_captures(session_dir)
    incorrect = [c for c in captures if c.label == "incorrect"]
    if limit is not None:
        incorrect = incorrect[:limit]

    composites_dir = session_dir / "analysis_composites"
    composites_dir.mkdir(exist_ok=True)

    out_jsonl = session_dir / "analysis_unstructured.jsonl"
    out_csv = session_dir / "analysis_unstructured.csv"

    client = OpenAI() if live else None

    with out_jsonl.open("w") as fj, out_csv.open("w", newline="") as fc:
        writer = csv.writer(fc)
        writer.writerow(["id", "reason", "composite", "meta"])

        for cap in incorrect:
            full_img = Image.open(cap.full_path).convert("RGB")
            board_img = render_predicted_board(cap.board, cap.preview_label)
            composite = build_composite(full_img, board_img, max_width=max_width)
            composite_path = composites_dir / f"{cap.capture_id:06d}_composite.png"
            composite.save(composite_path)

            board_ascii = board_to_ascii(cap.board)
            prompt = build_prompt(board_ascii)

            reason = "dry-run"
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
                            "name": "misclass_reason",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "reason": {"type": "string"}
                                },
                                "required": ["reason"],
                                "additionalProperties": False,
                            },
                            "strict": True,
                        }
                    },
                    max_output_tokens=200,
                )
                try:
                    data = json.loads(response.output_text)
                    reason = data.get("reason", "uncertain")
                except json.JSONDecodeError:
                    reason = response.output_text.strip() if response.output_text else "uncertain"

            record = {
                "id": cap.capture_id,
                "reason": reason,
                "composite": composite_path.name,
                "meta": cap.meta_path.name,
            }
            fj.write(json.dumps(record) + "\n")
            writer.writerow([cap.capture_id, reason, composite_path.name, cap.meta_path.name])

    return out_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze labeled Threes captures and describe misclassification reasons."
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
        help="OpenAI model for vision analysis.",
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
        help="Limit number of incorrect captures to analyze.",
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
    out_jsonl = run_analysis(
        session_path,
        model=args.model,
        max_width=args.max_width,
        limit=args.limit,
        live=live,
    )
    print(f"Wrote analysis to {out_jsonl}")
    if not live:
        print("Dry-run: set OPENAI_RUN_LIVE=1 to call the API.")


if __name__ == "__main__":
    main()
