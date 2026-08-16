"""Record a deterministic policy rollout as JSON plus a browser replay."""

from __future__ import annotations

import argparse
import json
import time
from html import escape
from pathlib import Path
from typing import Any

import numpy as np

from threes_rl.replay_provenance import ORIGIN_FRESH, direct_root_fields
from threes_rl.sim import DIRECTION_NAMES, SimState, ThreesSim, score_board


def preview_payload(state: SimState) -> dict[str, Any]:
    preview = state.preview
    return {
        "kind": preview.kind,
        "label": preview.label,
        "value": preview.value,
        "candidates": list(preview.candidates),
    }


def state_payload(state: SimState, sim: ThreesSim) -> dict[str, Any]:
    return {
        "move_count": int(state.move_count),
        "board": np.asarray(state.board, dtype=int).tolist(),
        "score": int(score_board(state.board)),
        "max_tile": int(state.max_tile),
        "game_over": bool(state.game_over),
        "preview": preview_payload(state),
        "legal_actions": [DIRECTION_NAMES[action] for action in sim.legal_actions(state)],
        "legal_mask": [bool(v) for v in sim.legal_mask(state).tolist()],
        "tile_cycle": {
            "small_counts": {str(k): int(v) for k, v in state.small_counts.items()},
            "small_pos": int(state.small_pos),
            "small_seen_total": int(state.small_seen_total),
            "span_small_pos": int(state.span_small_pos),
            "large_pending": bool(state.large_pending),
            "max_tile": int(state.max_tile),
        },
    }


def record_replay_for_policy(
    policy,
    policy_name: str,
    seed: int,
    starter_tile: int | None,
    max_moves: int,
) -> dict[str, Any]:
    sim = ThreesSim(np.random.default_rng(seed), starter_tile=starter_tile)
    policy_rng = np.random.default_rng(seed + 1_000_003)
    state = sim.reset()
    frames: list[dict[str, Any]] = [
        {
            "index": 0,
            "state": state_payload(state, sim),
            "move": None,
        }
    ]

    while not state.game_over and state.move_count < max_moves:
        before = state
        action = int(policy(before, sim, policy_rng))
        state, info = sim.step(before, action)
        if not info.moved:
            legal = sim.legal_actions(before)
            if not legal:
                break
            action = int(legal[0])
            state, info = sim.step(before, action)

        frames.append(
            {
                "index": len(frames),
                "state": state_payload(state, sim),
                "move": {
                    "action": DIRECTION_NAMES[action],
                    "preview_used": preview_payload(before),
                    "inserted_value": info.inserted_value,
                    "inserted_pos": list(info.inserted_pos) if info.inserted_pos is not None else None,
                    "eligible_positions": [list(pos) for pos in info.eligible_positions],
                    "merge_score_delta": int(info.merge_score_delta),
                    "score_delta": int(info.score_delta),
                    "terminal_merge": bool(info.terminal_merge),
                    "score_before": int(score_board(before.board)),
                    "score_after": int(score_board(state.board)),
                    "max_tile_before": int(before.max_tile),
                    "max_tile_after": int(state.max_tile),
                },
            }
        )

    return {
        "policy": policy_name,
        "seed": int(seed),
        "starter_tile": starter_tile,
        "max_moves": int(max_moves),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **direct_root_fields(
            origin=ORIGIN_FRESH,
            seed=int(seed),
            policy=policy_name,
            first_score=int(frames[0]["state"]["score"]),
        ),
        "final_score": int(score_board(state.board)),
        "final_moves": int(state.move_count),
        "final_max_tile": int(state.max_tile),
        "game_over": bool(state.game_over),
        "frames": frames,
    }


def record_replay(policy_spec: str, seed: int, starter_tile: int | None, max_moves: int) -> dict[str, Any]:
    from threes_rl.eval import make_policy

    return record_replay_for_policy(make_policy(policy_spec), policy_spec, seed, starter_tile, max_moves)


def write_html(path: Path, replay: dict[str, Any]) -> None:
    data = json.dumps(replay, separators=(",", ":"))
    title = f"{replay['policy']} seed {replay['seed']}"
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Threes Replay - {escape(title)}</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #111318;
      --panel: #1b2028;
      --panel-2: #252b35;
      --line: #3a4350;
      --text: #eef2f6;
      --muted: #aeb8c4;
      --red: #ef7d8f;
      --blue: #86c8f0;
      --gray: #d7d9d4;
      --gold: #f4c45f;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{
      width: min(1180px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 24px 0;
      display: grid;
      grid-template-columns: minmax(330px, 500px) minmax(300px, 1fr);
      gap: 22px;
      align-items: start;
    }}
    header {{
      grid-column: 1 / -1;
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: end;
      border-bottom: 1px solid var(--line);
      padding-bottom: 16px;
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 24px;
      letter-spacing: 0;
    }}
    .subtle, label, .stat-label {{
      color: var(--muted);
    }}
    .summary {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }}
    .pill {{
      border: 1px solid var(--line);
      background: var(--panel);
      padding: 8px 10px;
      border-radius: 8px;
      min-width: 104px;
    }}
    .stat-label {{
      display: block;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: .08em;
      margin-bottom: 3px;
    }}
    .stat-value {{
      font-variant-numeric: tabular-nums;
      font-weight: 700;
    }}
    .board-wrap {{
      display: grid;
      gap: 16px;
    }}
    .board {{
      width: min(500px, calc(100vw - 32px));
      aspect-ratio: 1;
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 8px;
      background: #2e3641;
      border: 1px solid #4a5563;
      padding: 8px;
      border-radius: 8px;
    }}
    .cell {{
      display: grid;
      place-items: center;
      border-radius: 6px;
      font-weight: 800;
      font-size: clamp(18px, 6vw, 34px);
      font-variant-numeric: tabular-nums;
      color: #19202a;
      border: 1px solid rgba(255,255,255,.16);
    }}
    .empty {{ background: #202732; color: transparent; }}
    .v1 {{ background: var(--blue); }}
    .v2 {{ background: var(--red); }}
    .v3, .v6, .v12, .v24 {{ background: var(--gray); }}
    .v48, .v96, .v192 {{ background: #b6d98f; }}
    .v384, .v768 {{ background: var(--gold); }}
    .v1536, .v3072, .v6144, .v12288 {{ background: #f09b62; color: #141820; }}
    .controls, .panel {{
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
      padding: 14px;
    }}
    .controls {{
      display: grid;
      gap: 12px;
    }}
    .buttons {{
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
    }}
    button {{
      appearance: none;
      border: 1px solid var(--line);
      background: var(--panel-2);
      color: var(--text);
      border-radius: 7px;
      padding: 8px 11px;
      cursor: pointer;
      font-weight: 650;
    }}
    button:hover {{ border-color: var(--muted); }}
    input[type="range"] {{
      width: 100%;
      accent-color: var(--gold);
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }}
    .panel h2 {{
      margin: 0 0 12px;
      font-size: 16px;
      letter-spacing: 0;
    }}
    dl {{
      display: grid;
      grid-template-columns: minmax(120px, auto) 1fr;
      gap: 7px 12px;
      margin: 0;
      font-size: 14px;
    }}
    dt {{ color: var(--muted); }}
    dd {{
      margin: 0;
      font-variant-numeric: tabular-nums;
      overflow-wrap: anywhere;
    }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      color: #d6dde7;
      font-size: 12px;
      line-height: 1.45;
    }}
    @media (max-width: 850px) {{
      main {{ grid-template-columns: 1fr; }}
      header {{ align-items: start; flex-direction: column; }}
      .summary {{ justify-content: flex-start; }}
      .board {{ width: 100%; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>Threes Replay</h1>
        <div class="subtle" id="subtitle"></div>
      </div>
      <div class="summary">
        <div class="pill"><span class="stat-label">Score</span><span class="stat-value" id="score"></span></div>
        <div class="pill"><span class="stat-label">Move</span><span class="stat-value" id="move"></span></div>
        <div class="pill"><span class="stat-label">Max Tile</span><span class="stat-value" id="maxTile"></span></div>
      </div>
    </header>
    <section class="board-wrap">
      <div class="board" id="board"></div>
      <div class="controls">
        <input id="scrubber" type="range" min="0" value="0" step="1">
        <div class="buttons">
          <button id="first">First</button>
          <button id="prev">Prev</button>
          <button id="play">Play</button>
          <button id="next">Next</button>
          <button id="last">Last</button>
          <label>Speed <input id="speed" type="range" min="60" max="900" value="220" step="20"></label>
        </div>
      </div>
    </section>
    <section class="grid">
      <div class="panel">
        <h2>Move</h2>
        <dl id="moveDetails"></dl>
      </div>
      <div class="panel">
        <h2>State</h2>
        <dl id="stateDetails"></dl>
      </div>
      <div class="panel" style="grid-column: 1 / -1;">
        <h2>Tile Cycle</h2>
        <pre id="cycle"></pre>
      </div>
    </section>
  </main>
  <script>
    const replay = {data};
    const board = document.getElementById("board");
    const scrubber = document.getElementById("scrubber");
    const playButton = document.getElementById("play");
    const speed = document.getElementById("speed");
    let frameIndex = 0;
    let timer = null;

    scrubber.max = String(replay.frames.length - 1);
    document.getElementById("subtitle").textContent =
      `${{replay.policy}} seed ${{replay.seed}} / final ${{replay.final_score}} in ${{replay.final_moves}} moves`;

    function tileClass(value) {{
      if (!value) return "empty";
      return "v" + value;
    }}
    function previewText(preview) {{
      if (!preview) return "-";
      if (preview.kind === "bonus") return `large [${{preview.candidates.join(", ")}}]`;
      return `${{preview.label}} (${{preview.value}})`;
    }}
    function pairList(items) {{
      if (!items || !items.length) return "-";
      return items.map((p) => `(${{p[0]}},${{p[1]}})`).join(" ");
    }}
    function setDetails(node, rows) {{
      node.innerHTML = rows.map(([k, v]) => `<dt>${{k}}</dt><dd>${{v}}</dd>`).join("");
    }}
    function show(index) {{
      frameIndex = Math.max(0, Math.min(replay.frames.length - 1, index));
      const frame = replay.frames[frameIndex];
      const state = frame.state;
      scrubber.value = String(frameIndex);
      board.innerHTML = state.board.flat().map((value) =>
        `<div class="cell ${{tileClass(value)}}">${{value || ""}}</div>`
      ).join("");
      document.getElementById("score").textContent = state.score;
      document.getElementById("move").textContent = `${{state.move_count}} / ${{replay.final_moves}}`;
      document.getElementById("maxTile").textContent = state.max_tile;

      const move = frame.move;
      setDetails(document.getElementById("moveDetails"), move ? [
        ["Action", move.action],
        ["Used Preview", previewText(move.preview_used)],
        ["Inserted", move.inserted_value == null ? "-" : move.inserted_value],
        ["Inserted Pos", move.inserted_pos ? `(${{move.inserted_pos[0]}},${{move.inserted_pos[1]}})` : "-"],
        ["Eligible Slots", pairList(move.eligible_positions)],
        ["Score Delta", move.score_delta],
        ["Merge Delta", move.merge_score_delta],
        ["Terminal Merge", move.terminal_merge ? "yes" : "no"]
      ] : [["Action", "initial"], ["Used Preview", "-"], ["Inserted", "-"]]);

      setDetails(document.getElementById("stateDetails"), [
        ["Next Preview", previewText(state.preview)],
        ["Legal Actions", state.legal_actions.join(", ") || "-"],
        ["Game Over", state.game_over ? "yes" : "no"],
        ["Frame", `${{frame.index}} / ${{replay.frames.length - 1}}`]
      ]);
      document.getElementById("cycle").textContent = JSON.stringify(state.tile_cycle, null, 2);
    }}
    function stop() {{
      if (timer) clearInterval(timer);
      timer = null;
      playButton.textContent = "Play";
    }}
    function play() {{
      if (timer) return stop();
      playButton.textContent = "Pause";
      timer = setInterval(() => {{
        if (frameIndex >= replay.frames.length - 1) return stop();
        show(frameIndex + 1);
      }}, Number(speed.value));
    }}
    document.getElementById("first").onclick = () => {{ stop(); show(0); }};
    document.getElementById("prev").onclick = () => {{ stop(); show(frameIndex - 1); }};
    document.getElementById("play").onclick = play;
    document.getElementById("next").onclick = () => {{ stop(); show(frameIndex + 1); }};
    document.getElementById("last").onclick = () => {{ stop(); show(replay.frames.length - 1); }};
    scrubber.oninput = () => {{ stop(); show(Number(scrubber.value)); }};
    window.addEventListener("keydown", (event) => {{
      if (event.key === "ArrowLeft") {{ stop(); show(frameIndex - 1); }}
      if (event.key === "ArrowRight") {{ stop(); show(frameIndex + 1); }}
      if (event.key === " ") {{ event.preventDefault(); play(); }}
    }});
    show(0);
  </script>
</body>
</html>
"""
    path.write_text(html)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, help="Policy spec accepted by threes_rl.eval, e.g. expectimax2 or ppo:path.pt.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--starter", default="1536")
    parser.add_argument("--max-moves", type=int, default=5000)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    starter_tile = None if args.starter.lower() == "none" else int(args.starter)
    replay = record_replay(args.policy, args.seed, starter_tile, args.max_moves)
    safe_policy = args.policy.replace(":", "_").replace("/", "_").replace(".", "_")
    out_dir = args.out_dir or Path("threes_rl/runs/replays") / f"{safe_policy}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "replay.json"
    html_path = out_dir / "replay.html"
    json_path.write_text(json.dumps(replay, indent=2, sort_keys=True))
    write_html(html_path, replay)
    print(json.dumps({
        "html": str(html_path),
        "json": str(json_path),
        "frames": len(replay["frames"]),
        "final_score": replay["final_score"],
        "final_moves": replay["final_moves"],
        "final_max_tile": replay["final_max_tile"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
