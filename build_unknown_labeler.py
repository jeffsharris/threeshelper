import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def find_latest_session(base_dir: Path) -> Path:
    sessions = sorted([p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith("session_")])
    if not sessions:
        raise FileNotFoundError(f"No sessions found in {base_dir}")
    return sessions[-1]


def load_index(index_path: Path) -> List[Dict[str, str]]:
    entries: List[Dict[str, str]] = []
    for line in index_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)
        entries.append(entry)
    return entries


def load_labels(labels_path: Path) -> Dict[str, str]:
    if not labels_path.exists():
        return {}
    labels: Dict[str, str] = {}
    with labels_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tile_id = (row.get("tile_id") or "").strip()
            label = (row.get("label") or "").strip()
            if tile_id:
                labels[tile_id] = label
    return labels


def render_html(tiles: List[Dict[str, str]], labels: Dict[str, str]) -> str:
    allowed = ["3", "6", "12", "24", "48", "96", "192", "384", "768", "1536"]
    tiles_data = [
        {
            "tile_id": t["tile_id"],
            "image": t["image"],
        }
        for t in tiles
    ]
    options = "".join(f'<option value="{x}"></option>' for x in allowed)
    rows_html = []
    for t in tiles_data:
        tile_id = t["tile_id"]
        image = t["image"]
        label = labels.get(tile_id, "")
        row = (
            '<div class="row">'
            f'<img src="../unknown_tiles/{image}" alt="{tile_id}" />'
            f'<div class="tile-id">{tile_id}</div>'
            f'<input type="text" list="numbers" value="{label}" />'
            "</div>"
        )
        rows_html.append(row)
    rows_block = "\n    ".join(rows_html)
    template = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Unknown Tile Labeler</title>
  <style>
    body { font-family: Helvetica, Arial, sans-serif; background: #12121a; color: #e6e6f0; margin: 20px; }
    h1 { margin: 0 0 10px 0; }
    .controls { display: flex; gap: 12px; align-items: center; margin-bottom: 12px; flex-wrap: wrap; }
    .row { display: flex; align-items: center; gap: 14px; margin: 8px 0; padding: 6px; background: #1c1c26; border-radius: 6px; }
    .tile-id { font-family: ui-monospace, Menlo, monospace; width: 130px; color: #c7c7d6; }
    img { width: 72px; height: 72px; border: 1px solid #34344a; border-radius: 4px; background: #202030; }
    input { width: 80px; padding: 6px; font-size: 14px; background: #10101a; color: #e6e6f0; border: 1px solid #3a3a52; border-radius: 4px; }
    button { padding: 8px 12px; background: #2e7d32; color: #fff; border: 0; border-radius: 6px; cursor: pointer; }
    button.secondary { background: #3949ab; }
    .status { color: #a8a8bf; }
  </style>
</head>
<body>
  <h1>Unknown Tile Labeler</h1>
  <div class="controls">
    <button id="save">Save CSV</button>
    <button id="saveJson" class="secondary">Save JSON</button>
    <button id="fillCsv" class="secondary">Fill CSV</button>
    <button id="copyCsv" class="secondary">Copy CSV</button>
    <span id="status" class="status"></span>
  </div>
  <textarea id="csvOutput" rows="6" style="width: 100%; margin-bottom: 12px; background: #10101a; color: #e6e6f0; border: 1px solid #3a3a52; border-radius: 6px; padding: 8px;"></textarea>
  <datalist id="numbers">
    __OPTIONS__
  </datalist>
  <div id="list">
    __ROWS__
  </div>

  <script>
    const list = document.getElementById('list');
    const status = document.getElementById('status');

    function updateStatus() {
      const inputs = list.querySelectorAll('input');
      let filled = 0;
      inputs.forEach(i => { if (i.value.trim()) filled += 1; });
      status.textContent = `Labeled ${filled} / ${inputs.length}`;
    }

    list.querySelectorAll('input').forEach(input => {
      input.addEventListener('input', updateStatus);
    });
    updateStatus();

    function gatherRows() {
      const rows = list.querySelectorAll('.row');
      const out = [];
      rows.forEach(row => {
        const tileId = row.querySelector('.tile-id').textContent.trim();
        const label = row.querySelector('input').value.trim();
        out.push({ tile_id: tileId, label });
      });
      return out;
    }

    function buildCsv() {
      const rows = gatherRows();
      const lines = ['tile_id,label'];
      rows.forEach(r => { lines.push(`${r.tile_id},${r.label}`); });
      return lines.join('\\n');
    }

    function download(filename, text) {
      const blob = new Blob([text], { type: 'text/plain' });
      const link = document.createElement('a');
      link.href = URL.createObjectURL(blob);
      link.download = filename;
      link.style.display = 'none';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(link.href);
    }

    document.getElementById('save').addEventListener('click', () => {
      download('unknown_labels.csv', buildCsv());
    });

    document.getElementById('saveJson').addEventListener('click', () => {
      const rows = gatherRows();
      download('unknown_labels.json', JSON.stringify(rows, null, 2));
    });

    document.getElementById('fillCsv').addEventListener('click', () => {
      document.getElementById('csvOutput').value = buildCsv();
    });

    document.getElementById('copyCsv').addEventListener('click', () => {
      const text = buildCsv();
      const textarea = document.getElementById('csvOutput');
      textarea.value = text;
      textarea.select();
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text);
      } else {
        document.execCommand('copy');
      }
    });
  </script>
</body>
</html>"""
    html = template.replace("__OPTIONS__", options)
    html = html.replace("__ROWS__", rows_block)
    return html


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a labeler for unknown tiles in a session.")
    parser.add_argument(
        "--session",
        type=Path,
        help="Path to datasets/session_YYYYMMDD_HHMMSS. Defaults to latest.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Output HTML path (default: <session>/unknown_labeler/index.html).",
    )
    args = parser.parse_args()

    base_dir = Path("datasets")
    session_dir = args.session or find_latest_session(base_dir)
    if not session_dir.exists():
        raise FileNotFoundError(f"Session not found: {session_dir}")

    index_path = session_dir / "unknown_tiles.jsonl"
    labels_path = session_dir / "unknown_labels.csv"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing unknown tiles index: {index_path}")

    tiles = load_index(index_path)
    labels = load_labels(labels_path)
    html = render_html(tiles, labels)

    if args.out:
        out_path = args.out
    else:
        out_path = session_dir / "unknown_labeler" / "index.html"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    print(f"Wrote labeler to {out_path}")


if __name__ == "__main__":
    main()
