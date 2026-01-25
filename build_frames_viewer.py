import argparse
import json
from pathlib import Path
from typing import Dict, List


def _read_json(path: Path) -> Dict:
    return json.loads(path.read_text())


def render_html(entries: List[Dict[str, object]], title: str) -> str:
    rows = []
    for entry in entries:
        frame = entry.get("frame")
        elapsed = entry.get("elapsed_s")
        diff_prev = entry.get("diff_prev")
        diff_first = entry.get("diff_first")
        files = entry.get("files", {})
        row = (
            "<tr>"
            f"<td>{frame}</td>"
            f"<td>{elapsed}</td>"
            f"<td>{diff_prev}</td>"
            f"<td>{diff_first}</td>"
            f"<td><img src=\"{files.get('board','')}\" alt=\"board {frame}\" /></td>"
            f"<td><img src=\"{files.get('preview','')}\" alt=\"preview {frame}\" /></td>"
            f"<td><img src=\"{files.get('full','')}\" alt=\"full {frame}\" /></td>"
            "</tr>"
        )
        rows.append(row)
    rows_html = "\n    ".join(rows)
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <style>
    body {{ font-family: Helvetica, Arial, sans-serif; background: #111319; color: #e6e6f0; margin: 20px; }}
    h1 {{ margin: 0 0 12px 0; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid #2b2f3b; padding: 6px 8px; text-align: left; }}
    th {{ position: sticky; top: 0; background: #181b24; }}
    tr:nth-child(even) {{ background: #141821; }}
    img {{ display: block; border: 1px solid #2b2f3b; border-radius: 4px; background: #0f1118; }}
    td img {{ max-width: 220px; height: auto; }}
    td:nth-child(5) img {{ width: 140px; }}
    td:nth-child(6) img {{ width: 140px; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <table>
    <thead>
      <tr>
        <th>Frame</th>
        <th>Elapsed (s)</th>
        <th>Diff Prev</th>
        <th>Diff First</th>
        <th>Board</th>
        <th>Preview</th>
        <th>Full</th>
      </tr>
    </thead>
    <tbody>
    {rows_html}
    </tbody>
  </table>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an HTML viewer for post-key frame captures."
    )
    parser.add_argument(
        "frames_json",
        type=Path,
        help="Path to frames.json produced by record_post_key_frames.py.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Output HTML path (default: <frames dir>/frames_viewer.html).",
    )
    args = parser.parse_args()

    data = _read_json(args.frames_json)
    entries = data.get("entries", [])
    title = f"Frame Viewer: {args.frames_json.parent.name}"
    html = render_html(entries, title)
    out_path = args.out or (args.frames_json.parent / "frames_viewer.html")
    out_path.write_text(html)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
