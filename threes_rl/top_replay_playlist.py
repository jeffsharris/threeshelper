"""Create a stable top-three replay playlist from normal-start records."""

from __future__ import annotations

import argparse
import json
import time
from html import escape
from pathlib import Path
from typing import Any

from threes_rl.dashboard import GLOBAL_TOP_REPLAY_LIMIT, RUNS_ROOT, collect_global_top_replays
from threes_rl.record_replay import write_html as write_replay_html
from threes_rl.run_artifacts import safe_name, write_json


DEFAULT_OUT_DIR = RUNS_ROOT / "replays" / "top3"


def _resolve_run_path(root: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    raw = Path(value)
    candidates = [raw]
    if not raw.is_absolute():
        candidates.append(Path.cwd() / raw)
        parts = raw.parts
        if "runs" in parts:
            runs_index = parts.index("runs")
            candidates.append(root / Path(*parts[runs_index + 1 :]))
        else:
            candidates.append(root / raw)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def _relative(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def _signature_for_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "score": item.get("score"),
        "seed": item.get("seed"),
        "starter_tile": item.get("starter_tile"),
        "moves": item.get("moves"),
        "json": item.get("json"),
    }


def _signature_for_replays(replays: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    return [_signature_for_item(item) for item in replays[: max(0, int(limit))]]


def _read_manifest(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def playlist_is_current(
    *,
    out_dir: Path = DEFAULT_OUT_DIR,
    top_replays: list[dict[str, Any]],
    limit: int = GLOBAL_TOP_REPLAY_LIMIT,
) -> bool:
    payload = _read_manifest(Path(out_dir) / "manifest.json")
    if payload is None:
        return False
    expected_signature = _signature_for_replays(top_replays, limit=limit)
    if payload.get("signature") != expected_signature:
        return False
    replays = payload.get("replays")
    if not isinstance(replays, list) or len(replays) != len(expected_signature):
        return False
    for item in replays:
        if not isinstance(item, dict) or not item.get("copied"):
            return False
        stable_json = item.get("stable_json")
        stable_html = item.get("stable_html")
        if not stable_json or not stable_html:
            return False
        if not Path(str(stable_json)).exists() or not Path(str(stable_html)).exists():
            return False
    return True


def _playlist_html(payload: dict[str, Any], out_dir: Path) -> str:
    replays = payload.get("replays", [])
    if not isinstance(replays, list):
        replays = []
    first_html = next((item.get("stable_html") for item in replays if isinstance(item, dict) and item.get("stable_html")), "")
    rows = []
    buttons = []
    for item in replays:
        if not isinstance(item, dict):
            continue
        stable_html = str(item.get("stable_html") or "")
        href = _relative(Path(stable_html), out_dir) if stable_html else ""
        rank = int(item.get("rank") or 0)
        label = f"#{rank} {int(item.get('score') or 0):,}"
        buttons.append(
            f'<button type="button" data-src="{escape(href)}">{escape(label)}</button>'
            if href
            else f'<button type="button" disabled>{escape(label)}</button>'
        )
        rows.append(
            "<tr>"
            f"<td>{rank}</td>"
            f"<td>{escape(str(item.get('score', '')))}</td>"
            f"<td>{escape(str(item.get('score_minus_starter', '')))}</td>"
            f"<td>{escape(str(item.get('seed', '')))}</td>"
            f"<td>{escape(str(item.get('moves', '')))}</td>"
            f"<td>{escape(str(item.get('max_tile_excl_starter', item.get('max_tile', ''))))}</td>"
            f"<td>{escape(str(item.get('run', '')))}</td>"
            f"<td>{f'<a href=\"{escape(href)}\">open</a>' if href else '-'}</td>"
            "</tr>"
        )
    first_src = _relative(Path(str(first_html)), out_dir) if first_html else ""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Threes RL Top 3 Replays</title>
  <style>
    :root {{ color-scheme: dark; --bg: #101113; --panel: #191d21; --line: #364047; --ink: #f1f5f0; --muted: #a9b3ad; --gold: #e9bd4a; --blue: #7bb7e8; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--ink); font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ width: min(1280px, calc(100vw - 32px)); margin: 0 auto; padding: 22px 0 36px; }}
    header {{ display: flex; justify-content: space-between; gap: 18px; align-items: end; border-bottom: 1px solid var(--line); padding-bottom: 15px; margin-bottom: 14px; }}
    h1, p {{ margin: 0; }}
    h1 {{ font-size: 25px; }}
    .muted {{ color: var(--muted); }}
    .panel {{ border: 1px solid var(--line); background: var(--panel); border-radius: 8px; padding: 13px; margin-bottom: 14px; }}
    .buttons {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    button {{ appearance: none; border: 1px solid var(--line); background: #252b35; color: var(--ink); border-radius: 7px; padding: 8px 11px; font-weight: 750; cursor: pointer; }}
    button.active {{ border-color: var(--gold); color: var(--gold); }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; font-variant-numeric: tabular-nums; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 7px 6px; text-align: right; vertical-align: top; }}
    th:nth-child(7), td:nth-child(7) {{ text-align: left; }}
    a {{ color: var(--blue); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    iframe {{ width: 100%; height: min(900px, calc(100vh - 220px)); min-height: 620px; border: 1px solid var(--line); border-radius: 8px; background: #111318; }}
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>Top 3 Normal-Start Replays</h1>
        <p class="muted">Stable copies of the highest retained full-game starts. Generated {escape(str(payload.get('generated_at', '')))}.</p>
      </div>
      <p class="muted">{len(replays)} replay copies</p>
    </header>
    <section class="panel">
      <div class="buttons" id="buttons">{''.join(buttons)}</div>
    </section>
    <section class="panel">
      <table><thead><tr><th>Rank</th><th>Score</th><th>Minus Starter</th><th>Seed</th><th>Moves</th><th>Max Excl</th><th>Source Run</th><th>Replay</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
    </section>
    <iframe id="viewer" title="Replay viewer" src="{escape(first_src)}"></iframe>
  </main>
  <script>
    const viewer = document.getElementById("viewer");
    const buttons = Array.from(document.querySelectorAll("button[data-src]"));
    function setActive(button) {{
      buttons.forEach(item => item.classList.toggle("active", item === button));
      if (button && button.dataset.src) viewer.src = button.dataset.src;
    }}
    buttons.forEach(button => button.addEventListener("click", () => setActive(button)));
    if (buttons.length) setActive(buttons[0]);
  </script>
</body>
</html>
"""


def build_top_replay_playlist(
    *,
    root: Path = RUNS_ROOT,
    out_dir: Path = DEFAULT_OUT_DIR,
    limit: int = GLOBAL_TOP_REPLAY_LIMIT,
    top_replays: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    root = Path(root)
    out_dir = Path(out_dir)
    top_replays = list(top_replays)[: max(0, int(limit))] if top_replays is not None else collect_global_top_replays(root, limit=limit)
    rows: list[dict[str, Any]] = []
    for rank, item in enumerate(top_replays, start=1):
        source_json = _resolve_run_path(root, item.get("json"))
        source_html = _resolve_run_path(root, item.get("html"))
        score = int(item.get("score") or 0)
        seed = item.get("seed")
        game_dir = out_dir / f"rank_{rank:02d}_score_{score}_seed_{safe_name(str(seed))}"
        stable_json = game_dir / "replay.json"
        stable_html = game_dir / "replay.html"
        copied = False
        if source_json is not None and source_json.exists():
            try:
                replay = json.loads(source_json.read_text())
            except (OSError, json.JSONDecodeError):
                replay = None
            if isinstance(replay, dict):
                write_json(stable_json, replay)
                write_replay_html(stable_html, replay)
                copied = True
        rows.append(
            {
                "rank": rank,
                "run": item.get("run"),
                "run_path": item.get("run_path"),
                "score": item.get("score"),
                "score_minus_starter": item.get("score_minus_starter"),
                "seed": seed,
                "starter_tile": item.get("starter_tile"),
                "moves": item.get("moves"),
                "max_tile": item.get("max_tile"),
                "max_tile_excl_starter": item.get("max_tile_excl_starter"),
                "source_json": str(source_json) if source_json is not None else item.get("json"),
                "source_html": str(source_html) if source_html is not None else item.get("html"),
                "stable_json": str(stable_json) if copied else None,
                "stable_html": str(stable_html) if copied else None,
                "copied": bool(copied),
            }
        )
    payload = {
        "version": 1,
        "kind": "threes_top_replay_playlist",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "root": str(root),
        "out_dir": str(out_dir),
        "limit": int(limit),
        "signature": _signature_for_replays(top_replays, limit=limit),
        "replays": rows,
        "copied_count": sum(1 for row in rows if row.get("copied")),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    payload["json"] = str(out_dir / "manifest.json")
    payload["html"] = str(out_dir / "index.html")
    write_json(out_dir / "manifest.json", payload)
    (out_dir / "index.html").write_text(_playlist_html(payload, out_dir))
    write_json(out_dir / "manifest.json", payload)
    return payload


def sync_top_replay_playlist(
    *,
    root: Path = RUNS_ROOT,
    out_dir: Path = DEFAULT_OUT_DIR,
    limit: int = GLOBAL_TOP_REPLAY_LIMIT,
    top_replays: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    root = Path(root)
    out_dir = Path(out_dir)
    current_top = (
        list(top_replays)[: max(0, int(limit))]
        if top_replays is not None
        else collect_global_top_replays(root, limit=limit)
    )
    manifest_path = out_dir / "manifest.json"
    if playlist_is_current(out_dir=out_dir, top_replays=current_top, limit=limit):
        payload = _read_manifest(manifest_path)
        if payload is not None:
            payload["synced"] = False
            return payload
    payload = build_top_replay_playlist(root=root, out_dir=out_dir, limit=limit, top_replays=current_top)
    payload["synced"] = True
    write_json(manifest_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=RUNS_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--limit", type=int, default=GLOBAL_TOP_REPLAY_LIMIT)
    args = parser.parse_args()
    payload = build_top_replay_playlist(root=args.root, out_dir=args.out_dir, limit=args.limit)
    print(
        json.dumps(
            {
                "html": payload["html"],
                "json": payload["json"],
                "copied_count": payload["copied_count"],
                "scores": [item.get("score") for item in payload["replays"]],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
