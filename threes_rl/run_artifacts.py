"""Utilities for retaining top games and progress charts for experiments."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable, Protocol

from threes_rl.record_replay import record_replay_for_policy, write_html


class ScoredGame(Protocol):
    seed: int
    starter_tile: int | None
    score: int
    score_minus_starter: int
    moves: int
    max_tile: int
    max_tile_excl_starter: int


def safe_name(text: str, max_length: int = 120) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in text)
    if len(cleaned) <= max_length:
        return cleaned
    digest = hashlib.sha1(cleaned.encode("utf-8")).hexdigest()[:12]
    prefix_length = max(1, max_length - len(digest) - 1)
    return f"{cleaned[:prefix_length]}_{digest}"


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def write_progress_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_progress_chart(path: Path, rows: list[dict[str, object]], title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(rows, separators=(",", ":"))
    title_json = json.dumps(title)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title} Progress</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #101318;
      --panel: #1a2028;
      --line: #384451;
      --text: #edf2f7;
      --muted: #aab6c2;
      --high: #f2c14e;
      --mean: #7bd88f;
      --median: #77bdfb;
    }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{
      width: min(1120px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 24px 0;
    }}
    h1 {{
      margin: 0 0 4px;
      font-size: 24px;
      letter-spacing: 0;
    }}
    .muted {{ color: var(--muted); }}
    .panel {{
      margin-top: 18px;
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
      padding: 16px;
    }}
    svg {{ width: 100%; height: 420px; display: block; }}
    .legend {{
      display: flex;
      gap: 14px;
      flex-wrap: wrap;
      margin-top: 12px;
      color: var(--muted);
      font-size: 14px;
    }}
    .swatch {{ display: inline-block; width: 12px; height: 12px; border-radius: 2px; margin-right: 6px; vertical-align: -1px; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 16px;
      font-variant-numeric: tabular-nums;
      font-size: 13px;
    }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 7px 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
  </style>
</head>
<body>
  <main>
    <h1 id="title"></h1>
    <div class="muted" id="subtitle"></div>
    <section class="panel">
      <svg id="chart" viewBox="0 0 1000 420" role="img"></svg>
      <div class="legend">
        <span><span class="swatch" style="background: var(--high);"></span>High score minus starter</span>
        <span><span class="swatch" style="background: var(--mean);"></span>Mean score minus starter</span>
        <span><span class="swatch" style="background: var(--median);"></span>Median score minus starter</span>
      </div>
      <table id="table"></table>
    </section>
  </main>
  <script>
    const rows = {data};
    const title = {title_json};
    document.getElementById("title").textContent = title;
    document.getElementById("subtitle").textContent = rows.length
      ? `${{rows[rows.length - 1].games}} games / high ${{rows[rows.length - 1].high_score}} / p>=6144 ${{rows[rows.length - 1].p_max_tile_excl_starter_ge_6144}}`
      : "No progress rows yet";

    function num(value) {{
      const n = Number(value);
      return Number.isFinite(n) ? n : 0;
    }}
    function polyline(points, color) {{
      if (!points.length) return "";
      return `<polyline fill="none" stroke="${{color}}" stroke-width="3" points="${{points.map(p => p.join(",")).join(" ")}}" />`;
    }}
    function renderChart() {{
      const svg = document.getElementById("chart");
      const w = 1000, h = 420, left = 70, right = 20, top = 24, bottom = 48;
      if (!rows.length) {{
        svg.innerHTML = `<text x="50%" y="50%" text-anchor="middle" fill="#aab6c2">No data</text>`;
        return;
      }}
      const xs = rows.map(r => num(r.games));
      const series = [
        ["high_score_minus_starter", "var(--high)"],
        ["mean_score_minus_starter", "var(--mean)"],
        ["median_score_minus_starter", "var(--median)"],
      ];
      const vals = rows.flatMap(r => series.map(([key]) => num(r[key])));
      const minX = Math.min(...xs), maxX = Math.max(...xs);
      const maxY = Math.max(1, ...vals);
      const x = v => left + (maxX === minX ? 0 : (v - minX) / (maxX - minX)) * (w - left - right);
      const y = v => h - bottom - (v / maxY) * (h - top - bottom);
      const grid = [0, .25, .5, .75, 1].map(t => {{
        const yy = y(maxY * t);
        return `<line x1="${{left}}" y1="${{yy}}" x2="${{w - right}}" y2="${{yy}}" stroke="#384451" stroke-width="1" />
                <text x="${{left - 10}}" y="${{yy + 4}}" text-anchor="end" fill="#aab6c2" font-size="12">${{Math.round(maxY * t)}}</text>`;
      }}).join("");
      const lines = series.map(([key, color]) => {{
        const points = rows.map(r => [x(num(r.games)), y(num(r[key]))]);
        return polyline(points, color);
      }}).join("");
      svg.innerHTML = `${{grid}}
        <line x1="${{left}}" y1="${{h - bottom}}" x2="${{w - right}}" y2="${{h - bottom}}" stroke="#aab6c2" />
        <line x1="${{left}}" y1="${{top}}" x2="${{left}}" y2="${{h - bottom}}" stroke="#aab6c2" />
        ${{lines}}
        <text x="${{w / 2}}" y="${{h - 10}}" text-anchor="middle" fill="#aab6c2" font-size="13">Games</text>`;
    }}
    function renderTable() {{
      const table = document.getElementById("table");
      const tail = rows.slice(-25).reverse();
      const fields = ["games", "high_score", "mean_score", "median_score", "high_score_minus_starter", "mean_score_minus_starter", "median_score_minus_starter", "p_max_tile_excl_starter_ge_3072", "p_max_tile_excl_starter_ge_6144"];
      table.innerHTML = `<thead><tr>${{fields.map(f => `<th>${{f}}</th>`).join("")}}</tr></thead>` +
        `<tbody>${{tail.map(row => `<tr>${{fields.map(f => `<td>${{row[f] ?? ""}}</td>`).join("")}}</tr>`).join("")}}</tbody>`;
    }}
    renderChart();
    renderTable();
  </script>
</body>
</html>
"""
    path.write_text(html)


def write_top_replays(
    *,
    run_dir: Path,
    results: Iterable[ScoredGame],
    policy,
    policy_name: str,
    starter_tile: int | None,
    max_moves: int,
    top_n: int,
    replays_by_seed: dict[object, dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    top = sorted(results, key=lambda result: (result.score, result.moves), reverse=True)[: max(0, top_n)]
    out: list[dict[str, object]] = []
    top_dir = run_dir / "top_games"
    top_dir.mkdir(parents=True, exist_ok=True)
    for rank, result in enumerate(top, start=1):
        result_starter = getattr(result, "starter_tile", starter_tile)
        replay_key = (int(result.seed), "none" if result_starter is None else str(int(result_starter)))
        replay = None if replays_by_seed is None else replays_by_seed.get(replay_key) or replays_by_seed.get(int(result.seed))
        if replay is None:
            replay = record_replay_for_policy(policy, policy_name, int(result.seed), result_starter, max_moves)
        starter_suffix = "none" if result_starter is None else str(int(result_starter))
        game_dir = top_dir / f"rank_{rank:02d}_score_{int(result.score)}_seed_{int(result.seed)}_starter_{starter_suffix}"
        game_dir.mkdir(parents=True, exist_ok=True)
        json_path = game_dir / "replay.json"
        html_path = game_dir / "replay.html"
        write_json(json_path, replay)
        write_html(html_path, replay)
        out.append(
            {
                "rank": rank,
                "seed": int(result.seed),
                "starter_tile": result_starter,
                "score": int(result.score),
                "score_minus_starter": int(result.score_minus_starter),
                "moves": int(result.moves),
                "max_tile": int(result.max_tile),
                "max_tile_excl_starter": int(result.max_tile_excl_starter),
                "html": str(html_path),
                "json": str(json_path),
            }
        )
    write_json(top_dir / "manifest.json", out)
    return out


def write_milestone_replays(
    *,
    run_dir: Path,
    results: Iterable[ScoredGame],
    policy,
    policy_name: str,
    starter_tile: int | None,
    max_moves: int,
    threshold: int,
    max_games: int = 0,
    replays_by_seed: dict[object, dict[str, object]] | None = None,
) -> dict[str, object]:
    threshold = int(threshold)
    qualified = [result for result in results if int(result.max_tile_excl_starter) >= threshold]
    qualified.sort(key=lambda result: (int(result.seed), "" if result.starter_tile is None else str(int(result.starter_tile))))
    if max_games > 0:
        qualified = qualified[: int(max_games)]

    out: list[dict[str, object]] = []
    milestone_dir = run_dir / "milestone_games" / f"ge_{threshold}"
    milestone_dir.mkdir(parents=True, exist_ok=True)
    for index, result in enumerate(qualified, start=1):
        result_starter = getattr(result, "starter_tile", starter_tile)
        starter_suffix = "none" if result_starter is None else str(int(result_starter))
        replay_key = (int(result.seed), starter_suffix)
        replay = None if replays_by_seed is None else replays_by_seed.get(replay_key) or replays_by_seed.get(int(result.seed))
        if replay is None:
            replay = record_replay_for_policy(policy, policy_name, int(result.seed), result_starter, max_moves)
        game_dir = milestone_dir / f"seed_{int(result.seed)}_score_{int(result.score)}_starter_{starter_suffix}"
        game_dir.mkdir(parents=True, exist_ok=True)
        json_path = game_dir / "replay.json"
        html_path = game_dir / "replay.html"
        write_json(json_path, replay)
        write_html(html_path, replay)
        out.append(
            {
                "index": index,
                "seed": int(result.seed),
                "starter_tile": result_starter,
                "score": int(result.score),
                "score_minus_starter": int(result.score_minus_starter),
                "moves": int(result.moves),
                "max_tile": int(result.max_tile),
                "max_tile_excl_starter": int(result.max_tile_excl_starter),
                "html": str(html_path),
                "json": str(json_path),
            }
        )

    manifest: dict[str, object] = {
        "threshold": threshold,
        "qualified_games": len(out),
        "max_games": int(max_games),
        "replays": out,
    }
    write_json(milestone_dir / "manifest.json", manifest)
    return manifest


def write_pre_milestone_failure_replays(
    *,
    run_dir: Path,
    results: Iterable[ScoredGame],
    policy,
    policy_name: str,
    starter_tile: int | None,
    max_moves: int,
    min_tile: int,
    threshold: int,
    max_games: int,
    replays_by_seed: dict[object, dict[str, object]] | None = None,
) -> dict[str, object]:
    min_tile = int(min_tile)
    threshold = int(threshold)
    qualified = [
        result
        for result in results
        if min_tile <= int(result.max_tile_excl_starter) < threshold
    ]
    qualified.sort(key=lambda result: (int(result.score), int(result.moves)), reverse=True)
    qualified_count = len(qualified)
    if max_games > 0:
        qualified = qualified[: int(max_games)]

    out: list[dict[str, object]] = []
    failure_dir = run_dir / "diagnostic_games" / f"pre_{threshold}_min_{min_tile}"
    failure_dir.mkdir(parents=True, exist_ok=True)
    for rank, result in enumerate(qualified, start=1):
        result_starter = getattr(result, "starter_tile", starter_tile)
        starter_suffix = "none" if result_starter is None else str(int(result_starter))
        replay_key = (int(result.seed), starter_suffix)
        replay = None if replays_by_seed is None else replays_by_seed.get(replay_key) or replays_by_seed.get(int(result.seed))
        if replay is None:
            replay = record_replay_for_policy(policy, policy_name, int(result.seed), result_starter, max_moves)
        game_dir = failure_dir / f"rank_{rank:02d}_score_{int(result.score)}_seed_{int(result.seed)}_starter_{starter_suffix}"
        game_dir.mkdir(parents=True, exist_ok=True)
        json_path = game_dir / "replay.json"
        html_path = game_dir / "replay.html"
        write_json(json_path, replay)
        write_html(html_path, replay)
        out.append(
            {
                "rank": rank,
                "seed": int(result.seed),
                "starter_tile": result_starter,
                "score": int(result.score),
                "score_minus_starter": int(result.score_minus_starter),
                "moves": int(result.moves),
                "max_tile": int(result.max_tile),
                "max_tile_excl_starter": int(result.max_tile_excl_starter),
                "html": str(html_path),
                "json": str(json_path),
            }
        )

    manifest: dict[str, object] = {
        "min_tile": min_tile,
        "threshold": threshold,
        "qualified_games": qualified_count,
        "retained_games": len(out),
        "max_games": int(max_games),
        "replays": out,
    }
    write_json(failure_dir / "manifest.json", manifest)
    return manifest
