"""Compare two deterministic policy replays and report the first divergence."""

from __future__ import annotations

import argparse
import json
import time
from html import escape
from pathlib import Path
from typing import Any

import numpy as np

from threes_rl.record_replay import record_replay_for_policy, write_html
from threes_rl.run_artifacts import safe_name, write_json
from threes_rl.sim import DIRECTION_NAMES, SimState, ThreesSim, preview_from_label


def _state_key(frame: dict[str, Any]) -> str:
    return json.dumps(frame.get("state", {}), sort_keys=True, separators=(",", ":"))


def _move_action(frame: dict[str, Any]) -> str | None:
    move = frame.get("move")
    if not isinstance(move, dict):
        return None
    action = move.get("action")
    return str(action) if action is not None else None


def _score(frame: dict[str, Any]) -> int | None:
    state = frame.get("state")
    if not isinstance(state, dict):
        return None
    score = state.get("score")
    return int(score) if isinstance(score, int) else None


def first_divergence(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_frames = list(left.get("frames", []))
    right_frames = list(right.get("frames", []))
    max_completed_moves = min(len(left_frames), len(right_frames)) - 1
    for before_idx in range(max(0, max_completed_moves)):
        left_before = left_frames[before_idx]
        right_before = right_frames[before_idx]
        if _state_key(left_before) != _state_key(right_before):
            return {
                "kind": "state_mismatch",
                "before_frame": before_idx,
                "move_number": before_idx + 1,
                "left_state": left_before.get("state"),
                "right_state": right_before.get("state"),
            }
        left_after = left_frames[before_idx + 1]
        right_after = right_frames[before_idx + 1]
        left_action = _move_action(left_after)
        right_action = _move_action(right_after)
        if left_action != right_action:
            return {
                "kind": "action",
                "before_frame": before_idx,
                "move_number": before_idx + 1,
                "left_action": left_action,
                "right_action": right_action,
                "state": left_before.get("state"),
                "left_move": left_after.get("move"),
                "right_move": right_after.get("move"),
            }

    if len(left_frames) != len(right_frames):
        return {
            "kind": "length",
            "before_frame": max_completed_moves,
            "move_number": max_completed_moves + 1,
            "left_frames": len(left_frames),
            "right_frames": len(right_frames),
            "left_state": left_frames[max_completed_moves].get("state") if left_frames else None,
            "right_state": right_frames[max_completed_moves].get("state") if right_frames else None,
        }
    return {
        "kind": "none",
        "before_frame": None,
        "move_number": None,
        "left_frames": len(left_frames),
        "right_frames": len(right_frames),
    }


def summarize_replay(replay: dict[str, Any]) -> dict[str, Any]:
    frames = list(replay.get("frames", []))
    final = frames[-1] if frames else {}
    state = final.get("state") if isinstance(final, dict) else {}
    return {
        "policy": replay.get("policy"),
        "seed": replay.get("seed"),
        "starter_tile": replay.get("starter_tile"),
        "final_score": replay.get("final_score", _score(final)),
        "final_moves": replay.get("final_moves"),
        "final_max_tile": replay.get("final_max_tile"),
        "game_over": replay.get("game_over"),
        "final_board": state.get("board") if isinstance(state, dict) else None,
        "final_preview": state.get("preview") if isinstance(state, dict) else None,
    }


def state_from_payload(payload: dict[str, Any]) -> SimState:
    preview_payload = payload.get("preview", {})
    if not isinstance(preview_payload, dict):
        raise ValueError("State payload is missing a preview")
    cycle = payload.get("tile_cycle", {})
    if not isinstance(cycle, dict):
        raise ValueError("State payload is missing tile_cycle")
    counts = cycle.get("small_counts", {})
    if not isinstance(counts, dict):
        raise ValueError("State payload is missing small_counts")
    preview = preview_from_label(
        str(preview_payload.get("label", preview_payload.get("kind", ""))),
        preview_payload.get("candidates", ()),
    )
    board = np.asarray(payload.get("board"), dtype=np.int32)
    return SimState(
        board=board,
        preview=preview,
        small_counts={str(key): int(value) for key, value in counts.items()},
        small_pos=int(cycle.get("small_pos", 0)),
        small_seen_total=int(cycle.get("small_seen_total", 0)),
        span_small_pos=int(cycle.get("span_small_pos", 0)),
        large_pending=bool(cycle.get("large_pending", False)),
        max_tile=int(payload.get("max_tile", int(board.max(initial=0)))),
        move_count=int(payload.get("move_count", 0)),
        game_over=bool(payload.get("game_over", False)),
    )


def action_values_for_policy(policy: object, state_payload: dict[str, Any], starter_tile: int | None) -> list[dict[str, Any]]:
    if not hasattr(policy, "_action_value"):
        return []
    state = state_from_payload(state_payload)
    sim = ThreesSim(np.random.default_rng(0), starter_tile=starter_tile)
    for cache_name in ("_cache", "_action_cache", "_afterstate_cache", "_post_spawn_cache", "_score_cache", "_legal_cache", "_eval_cache"):
        cache = getattr(policy, cache_name, None)
        if hasattr(cache, "clear"):
            cache.clear()
    root_depth = policy._root_depth(state) if hasattr(policy, "_root_depth") else getattr(policy, "depth", 1)
    rows = []
    for action in sim.legal_actions(state):
        rows.append(
            {
                "action": DIRECTION_NAMES[int(action)],
                "value": float(policy._action_value(state, sim, int(action), int(root_depth))),
            }
        )
    rows.sort(key=lambda row: (-float(row["value"]), str(row["action"])))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def _board_html(board: object) -> str:
    if not isinstance(board, list):
        return "<p class=\"muted\">No board.</p>"
    rows = []
    for row in board:
        if not isinstance(row, list):
            continue
        cells = "".join(f"<td>{escape(str(value))}</td>" for value in row)
        rows.append(f"<tr>{cells}</tr>")
    return "<table class=\"board\">" + "".join(rows) + "</table>"


def _kv_html(payload: dict[str, Any]) -> str:
    rows = []
    for key, value in payload.items():
        rows.append(f"<dt>{escape(str(key))}</dt><dd>{escape(str(value))}</dd>")
    return "<dl>" + "".join(rows) + "</dl>"


def _action_values_html(rows: object) -> str:
    if not isinstance(rows, list) or not rows:
        return "<p class=\"muted\">No action values.</p>"
    body = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        value = row.get("value")
        value_text = f"{float(value):.3f}" if isinstance(value, (float, int)) else str(value)
        body.append(
            "<tr>"
            f"<td>{escape(str(row.get('rank', '')))}</td>"
            f"<td>{escape(str(row.get('action', '')))}</td>"
            f"<td>{escape(value_text)}</td>"
            "</tr>"
        )
    return "<table><thead><tr><th>Rank</th><th>Action</th><th>Value</th></tr></thead><tbody>" + "".join(body) + "</tbody></table>"


def write_comparison_html(path: Path, payload: dict[str, Any]) -> None:
    divergence = payload.get("divergence", {})
    state = divergence.get("state") if isinstance(divergence, dict) else None
    if not isinstance(state, dict):
        state = {}
    left = payload.get("left", {})
    right = payload.get("right", {})
    left_replay = payload.get("left_replay_html")
    right_replay = payload.get("right_replay_html")
    left_move = divergence.get("left_move") if isinstance(divergence, dict) else {}
    right_move = divergence.get("right_move") if isinstance(divergence, dict) else {}
    left_action_values = divergence.get("left_action_values") if isinstance(divergence, dict) else []
    right_action_values = divergence.get("right_action_values") if isinstance(divergence, dict) else []
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Threes Replay Comparison</title>
  <style>
    :root {{ color-scheme: dark; --bg: #101318; --panel: #1a2028; --line: #384451; --text: #edf2f7; --muted: #aab6c2; --gold: #f2c14e; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--text); font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ width: min(1120px, calc(100vw - 32px)); margin: 0 auto; padding: 24px 0 36px; }}
    h1 {{ margin: 0 0 6px; font-size: 24px; letter-spacing: 0; }}
    h2 {{ margin: 0 0 10px; font-size: 17px; letter-spacing: 0; }}
    a {{ color: var(--gold); }}
    .muted {{ color: var(--muted); }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }}
    .panel {{ margin-top: 16px; border: 1px solid var(--line); background: var(--panel); border-radius: 8px; padding: 14px; }}
    .board {{ width: auto; border-collapse: collapse; margin-top: 8px; }}
    .board td {{ width: 62px; height: 42px; border: 1px solid var(--line); text-align: center; font-weight: 750; font-variant-numeric: tabular-nums; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 8px; font-variant-numeric: tabular-nums; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 6px 8px; text-align: right; }}
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
    dl {{ display: grid; grid-template-columns: 150px 1fr; gap: 6px 10px; margin: 0; font-size: 14px; }}
    dt {{ color: var(--muted); }}
    dd {{ margin: 0; overflow-wrap: anywhere; }}
    pre {{ white-space: pre-wrap; overflow-wrap: anywhere; margin: 0; font-size: 12px; line-height: 1.45; color: #d6dde7; }}
    @media (max-width: 760px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <main>
    <h1>Replay Comparison</h1>
    <p class="muted">Seed {escape(str(payload.get("seed")))} / starter {escape(str(payload.get("starter_tile")))} / first divergence: {escape(str(divergence.get("kind")))} at move {escape(str(divergence.get("move_number")))}</p>
    <section class="panel">
      <h2>Before Divergence</h2>
      <p class="muted">Preview: {escape(str(state.get("preview")))} / legal: {escape(str(state.get("legal_actions")))}</p>
      {_board_html(state.get("board"))}
    </section>
    <section class="grid">
      <div class="panel">
        <h2>Left</h2>
        <p><a href="{escape(str(left_replay))}">Open replay</a></p>
        {_kv_html(left if isinstance(left, dict) else {})}
        <h2>Action Values</h2>
        {_action_values_html(left_action_values)}
        <h2>Move</h2>
        <pre>{escape(json.dumps(left_move, indent=2, sort_keys=True))}</pre>
      </div>
      <div class="panel">
        <h2>Right</h2>
        <p><a href="{escape(str(right_replay))}">Open replay</a></p>
        {_kv_html(right if isinstance(right, dict) else {})}
        <h2>Action Values</h2>
        {_action_values_html(right_action_values)}
        <h2>Move</h2>
        <pre>{escape(json.dumps(right_move, indent=2, sort_keys=True))}</pre>
      </div>
    </section>
  </main>
</body>
</html>
"""
    path.write_text(html)


def compare_policies(
    *,
    left_policy_spec: str,
    right_policy_spec: str,
    seed: int,
    starter_tile: int | None,
    max_moves: int,
    out_dir: Path,
) -> dict[str, Any]:
    from threes_rl.eval import make_policy

    out_dir.mkdir(parents=True, exist_ok=True)
    left_policy = make_policy(left_policy_spec)
    right_policy = make_policy(right_policy_spec)
    left_replay = record_replay_for_policy(left_policy, left_policy_spec, seed, starter_tile, max_moves)
    right_replay = record_replay_for_policy(right_policy, right_policy_spec, seed, starter_tile, max_moves)

    left_dir = out_dir / safe_name("left_" + left_policy_spec, max_length=80)
    right_dir = out_dir / safe_name("right_" + right_policy_spec, max_length=80)
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)
    left_json = left_dir / "replay.json"
    left_html = left_dir / "replay.html"
    right_json = right_dir / "replay.json"
    right_html = right_dir / "replay.html"
    write_json(left_json, left_replay)
    write_html(left_html, left_replay)
    write_json(right_json, right_replay)
    write_html(right_html, right_replay)

    divergence = first_divergence(left_replay, right_replay)
    state_payload = divergence.get("state")
    if isinstance(state_payload, dict):
        divergence["left_action_values"] = action_values_for_policy(left_policy, state_payload, starter_tile)
        divergence["right_action_values"] = action_values_for_policy(right_policy, state_payload, starter_tile)

    payload: dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "seed": int(seed),
        "starter_tile": starter_tile,
        "max_moves": int(max_moves),
        "left": summarize_replay(left_replay),
        "right": summarize_replay(right_replay),
        "left_replay_json": str(left_json),
        "left_replay_html": str(left_html),
        "right_replay_json": str(right_json),
        "right_replay_html": str(right_html),
        "divergence": divergence,
    }
    comparison_json = out_dir / "comparison.json"
    comparison_html = out_dir / "comparison.html"
    write_json(comparison_json, payload)
    write_comparison_html(comparison_html, payload)
    payload["comparison_json"] = str(comparison_json)
    payload["comparison_html"] = str(comparison_html)
    write_json(comparison_json, payload)
    return payload


def parse_starter(text: str) -> int | None:
    value = text.strip().lower()
    return None if value == "none" else int(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-policy", required=True)
    parser.add_argument("--right-policy", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--starter", default="1536")
    parser.add_argument("--max-moves", type=int, default=5000)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    payload = compare_policies(
        left_policy_spec=args.left_policy,
        right_policy_spec=args.right_policy,
        seed=args.seed,
        starter_tile=parse_starter(args.starter),
        max_moves=args.max_moves,
        out_dir=args.out_dir,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
