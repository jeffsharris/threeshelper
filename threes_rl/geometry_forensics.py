"""Replay geometry diagnostics for high-board Threes games."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from html import escape
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import numpy as np

from threes_rl.eval import max_tile_excluding_initial_starter, starter_baseline_score
from threes_rl.run_artifacts import write_json
from threes_rl.sim import rank_for_value, score_board

ROW_SNAKE = (0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11, 15, 14, 13, 12)
COL_SNAKE = (0, 4, 8, 12, 13, 9, 5, 1, 2, 6, 10, 14, 15, 11, 7, 3)
TOP_LEFT_SNAKES = (ROW_SNAKE, COL_SNAKE)
ORTHOGONAL = ((-1, 0), (1, 0), (0, -1), (0, 1))


def board_without_free_starter(board: np.ndarray, starter_tile: int | None) -> np.ndarray:
    working = np.asarray(board, dtype=np.int32).copy()
    if starter_tile is None:
        return working
    matches = np.argwhere(working == int(starter_tile))
    if len(matches) == 0:
        return working
    match_idx = 0
    for idx, (row, col) in enumerate(matches):
        if int(row) == 0 and int(col) == 0:
            match_idx = idx
            break
    row, col = matches[match_idx]
    working[int(row), int(col)] = 0
    return working


def position_region(row: int, col: int) -> str:
    if row == 0 and col == 0:
        return "top_left"
    if row in (0, 3) and col in (0, 3):
        return "other_corner"
    if row == 0:
        return "top_row"
    if col == 0:
        return "left_col"
    if row == 3 or col == 3:
        return "edge"
    return "interior"


def _snake_inversions(board: np.ndarray, path: tuple[int, ...]) -> int:
    ranks = [rank_for_value(int(value)) for value in np.asarray(board, dtype=np.int32).reshape(-1)]
    inversions = 0
    previous = ranks[path[0]]
    for idx in path[1:]:
        current = ranks[idx]
        if current > previous:
            inversions += current - previous
        previous = current
    return int(inversions)


def _has_adjacent_value(board: np.ndarray, positions: list[tuple[int, int]], target: int) -> bool:
    if target <= 0:
        return False
    arr = np.asarray(board, dtype=np.int32)
    for row, col in positions:
        for dr, dc in ORTHOGONAL:
            nr = int(row) + dr
            nc = int(col) + dc
            if 0 <= nr < 4 and 0 <= nc < 4 and int(arr[nr, nc]) == int(target):
                return True
    return False


def _tile_counts(board: np.ndarray) -> Counter[int]:
    return Counter(int(value) for value in np.asarray(board, dtype=np.int32).reshape(-1) if int(value) > 0)


def _highest_duplicate_tile(board: np.ndarray) -> int:
    counts = _tile_counts(board)
    return max((value for value, count in counts.items() if count >= 2), default=0)


def _highest_adjacent_pair_tile(board: np.ndarray) -> int:
    arr = np.asarray(board, dtype=np.int32)
    best = 0
    for row in range(4):
        for col in range(4):
            value = int(arr[row, col])
            if value <= 0 or value <= best:
                continue
            for dr, dc in ORTHOGONAL:
                nr = row + dr
                nc = col + dc
                if 0 <= nr < 4 and 0 <= nc < 4 and int(arr[nr, nc]) == value:
                    best = value
                    break
    return int(best)


def geometry_features(board: np.ndarray, starter_tile: int | None) -> dict[str, Any]:
    arr = np.asarray(board, dtype=np.int32)
    masked = board_without_free_starter(arr, starter_tile)
    max_tile = int(masked.max(initial=0))
    positions = [tuple(int(v) for v in pos) for pos in np.argwhere(masked == max_tile)] if max_tile else []
    primary = min(positions, key=lambda pos: (pos[0] + pos[1], pos[0], pos[1])) if positions else None
    manhattan = None if primary is None else int(primary[0] + primary[1])
    region = "none" if primary is None else position_region(primary[0], primary[1])
    return {
        "score_minus_starter": int(score_board(arr) - starter_baseline_score(starter_tile)),
        "empty_count": int(np.count_nonzero(arr == 0)),
        "top_left": int(arr[0, 0]),
        "max_tile_excl_starter": max_tile,
        "max_positions": [[row, col] for row, col in positions],
        "primary_max_position": None if primary is None else [int(primary[0]), int(primary[1])],
        "primary_max_region": region,
        "manhattan_from_top_left": manhattan,
        "max_displaced_from_top_left": bool(max_tile > 0 and (0, 0) not in positions),
        "count_1536": int(np.count_nonzero(masked == 1536)),
        "count_3072": int(np.count_nonzero(masked == 3072)),
        "count_6144": int(np.count_nonzero(masked == 6144)),
        "highest_duplicate_tile": _highest_duplicate_tile(masked),
        "highest_adjacent_pair_tile": _highest_adjacent_pair_tile(masked),
        "adjacent_same_max": _has_adjacent_value(masked, positions, max_tile),
        "adjacent_half_max": _has_adjacent_value(masked, positions, max_tile // 2),
        "best_top_left_snake_inversions": min(_snake_inversions(masked, path) for path in TOP_LEFT_SNAKES),
    }


def _frame_state(frame: dict[str, Any]) -> dict[str, Any] | None:
    state = frame.get("state")
    return state if isinstance(state, dict) else None


def _frame_board(frame: dict[str, Any]) -> np.ndarray | None:
    state = _frame_state(frame)
    if state is None:
        return None
    board = state.get("board")
    if not isinstance(board, list):
        return None
    arr = np.asarray(board, dtype=np.int32)
    return arr if arr.shape == (4, 4) else None


def _feature_at_threshold(frames: list[dict[str, Any]], starter_tile: int | None, threshold: int) -> dict[str, Any] | None:
    for frame in frames:
        board = _frame_board(frame)
        if board is None:
            continue
        features = geometry_features(board, starter_tile)
        if int(features["max_tile_excl_starter"]) >= int(threshold):
            return {
                "frame_index": int(frame.get("index", 0)),
                "move_count": int((_frame_state(frame) or {}).get("move_count", frame.get("index", 0))),
                "features": features,
                "board": board.tolist(),
            }
    return None


def _feature_where(
    frames: list[dict[str, Any]],
    starter_tile: int | None,
    predicate,
) -> dict[str, Any] | None:
    for frame in frames:
        board = _frame_board(frame)
        if board is None:
            continue
        features = geometry_features(board, starter_tile)
        if predicate(features):
            return {
                "frame_index": int(frame.get("index", 0)),
                "move_count": int((_frame_state(frame) or {}).get("move_count", frame.get("index", 0))),
                "features": features,
                "board": board.tolist(),
            }
    return None


def analyze_replay(path: Path) -> dict[str, Any]:
    replay = json.loads(path.read_text())
    frames = [frame for frame in replay.get("frames", []) if isinstance(frame, dict)]
    starter_tile = replay.get("starter_tile", 1536)
    if starter_tile is not None:
        starter_tile = int(starter_tile)
    final_frame = frames[-1] if frames else {}
    final_board = _frame_board(final_frame)
    final_features = geometry_features(final_board, starter_tile) if final_board is not None else None
    post_3072_features: list[dict[str, Any]] = []
    seen_3072 = False
    for frame in frames:
        board = _frame_board(frame)
        if board is None:
            continue
        features = geometry_features(board, starter_tile)
        if int(features["max_tile_excl_starter"]) >= 3072:
            seen_3072 = True
        if seen_3072:
            post_3072_features.append(features)
    displaced_rate = (
        float(sum(1 for features in post_3072_features if features["max_displaced_from_top_left"]) / len(post_3072_features))
        if post_3072_features
        else None
    )
    adjacent_same_rate = (
        float(sum(1 for features in post_3072_features if features["adjacent_same_max"]) / len(post_3072_features))
        if post_3072_features
        else None
    )
    adjacent_half_rate = (
        float(sum(1 for features in post_3072_features if features["adjacent_half_max"]) / len(post_3072_features))
        if post_3072_features
        else None
    )
    return {
        "source_replay": str(path),
        "seed": replay.get("seed"),
        "starter_tile": starter_tile,
        "final_score": replay.get("final_score"),
        "final_score_minus_starter": None
        if replay.get("final_score") is None
        else int(replay.get("final_score")) - starter_baseline_score(starter_tile),
        "final_moves": replay.get("final_moves"),
        "first_1536": _feature_at_threshold(frames, starter_tile, 1536),
        "first_3072": _feature_at_threshold(frames, starter_tile, 3072),
        "first_6144": _feature_at_threshold(frames, starter_tile, 6144),
        "first_two_3072": _feature_where(frames, starter_tile, lambda features: int(features["count_3072"]) >= 2),
        "post_3072_frames": len(post_3072_features),
        "post_3072_displaced_rate": displaced_rate,
        "post_3072_adjacent_same_max_rate": adjacent_same_rate,
        "post_3072_adjacent_half_max_rate": adjacent_half_rate,
        "post_3072_max_count_1536": max((int(features["count_1536"]) for features in post_3072_features), default=0),
        "post_3072_max_count_3072": max((int(features["count_3072"]) for features in post_3072_features), default=0),
        "post_3072_max_highest_duplicate_tile": max(
            (int(features["highest_duplicate_tile"]) for features in post_3072_features),
            default=0,
        ),
        "post_3072_max_highest_adjacent_pair_tile": max(
            (int(features["highest_adjacent_pair_tile"]) for features in post_3072_features),
            default=0,
        ),
        "post_3072_mean_snake_inversions": None
        if not post_3072_features
        else float(mean(int(features["best_top_left_snake_inversions"]) for features in post_3072_features)),
        "final_features": final_features,
        "final_board": None if final_board is None else final_board.tolist(),
    }


def _glob_paths(patterns: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(Path().glob(pattern))
    return sorted(path for path in paths if path.is_file())


def _replay_identity(path: Path) -> str:
    try:
        replay = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return f"path:{Path(path)}"
    if not isinstance(replay, dict):
        return f"path:{Path(path)}"
    identity = {
        "policy": replay.get("policy"),
        "seed": replay.get("seed"),
        "starter_tile": replay.get("starter_tile"),
        "source_replay": replay.get("source_replay"),
        "source_frame_index": replay.get("source_frame_index"),
        "start_score": replay.get("start_score"),
        "final_score": replay.get("final_score"),
        "final_moves": replay.get("final_moves"),
    }
    return json.dumps(identity, sort_keys=True, separators=(",", ":"))


def unique_paths(paths: Iterable[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = _replay_identity(Path(path))
        if key in seen:
            continue
        seen.add(key)
        unique.append(Path(path))
    return unique


def summarize_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    first_3072_regions = Counter()
    final_regions = Counter()
    post_3072_rates: list[float] = []
    adjacent_same_rates: list[float] = []
    adjacent_half_rates: list[float] = []
    for case in cases:
        first_3072 = case.get("first_3072")
        if isinstance(first_3072, dict):
            features = first_3072.get("features") or {}
            first_3072_regions[str(features.get("primary_max_region", "unknown"))] += 1
        final_features = case.get("final_features") or {}
        if isinstance(final_features, dict):
            final_regions[str(final_features.get("primary_max_region", "unknown"))] += 1
        rate = case.get("post_3072_displaced_rate")
        if rate is not None:
            post_3072_rates.append(float(rate))
        same_rate = case.get("post_3072_adjacent_same_max_rate")
        if same_rate is not None:
            adjacent_same_rates.append(float(same_rate))
        half_rate = case.get("post_3072_adjacent_half_max_rate")
        if half_rate is not None:
            adjacent_half_rates.append(float(half_rate))
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "replays": len(cases),
        "reached_1536": sum(1 for case in cases if case.get("first_1536") is not None),
        "reached_3072": sum(1 for case in cases if case.get("first_3072") is not None),
        "reached_6144": sum(1 for case in cases if case.get("first_6144") is not None),
        "reached_two_3072": sum(1 for case in cases if case.get("first_two_3072") is not None),
        "first_3072_regions": dict(first_3072_regions),
        "final_max_regions": dict(final_regions),
        "mean_post_3072_displaced_rate": None if not post_3072_rates else float(mean(post_3072_rates)),
        "mean_post_3072_adjacent_same_max_rate": None
        if not adjacent_same_rates
        else float(mean(adjacent_same_rates)),
        "mean_post_3072_adjacent_half_max_rate": None
        if not adjacent_half_rates
        else float(mean(adjacent_half_rates)),
        "max_post_3072_count_3072": max((int(case.get("post_3072_max_count_3072") or 0) for case in cases), default=0),
        "max_post_3072_highest_duplicate_tile": max(
            (int(case.get("post_3072_max_highest_duplicate_tile") or 0) for case in cases),
            default=0,
        ),
        "max_post_3072_highest_adjacent_pair_tile": max(
            (int(case.get("post_3072_max_highest_adjacent_pair_tile") or 0) for case in cases),
            default=0,
        ),
    }


def write_html(path: Path, payload: dict[str, Any]) -> None:
    rows = []
    for case in payload.get("cases", []):
        first_3072 = case.get("first_3072") or {}
        first_two_3072 = case.get("first_two_3072") or {}
        first_features = first_3072.get("features") if isinstance(first_3072, dict) else {}
        final_features = case.get("final_features") or {}
        rows.append(
            "<tr>"
            f"<td>{escape(str(case.get('seed')))}</td>"
            f"<td>{escape(str(case.get('final_score')))}</td>"
            f"<td>{escape(str(case.get('final_score_minus_starter')))}</td>"
            f"<td>{escape(str(first_3072.get('move_count', '-')) if isinstance(first_3072, dict) else '-')}</td>"
            f"<td>{escape(str(first_two_3072.get('move_count', '-')) if isinstance(first_two_3072, dict) else '-')}</td>"
            f"<td>{escape(str(first_features.get('primary_max_position', '-')) if isinstance(first_features, dict) else '-')}</td>"
            f"<td>{escape(str(first_features.get('primary_max_region', '-')) if isinstance(first_features, dict) else '-')}</td>"
            f"<td>{escape(str(case.get('post_3072_displaced_rate', '-')))}</td>"
            f"<td>{escape(str(case.get('post_3072_max_count_3072', '-')))}</td>"
            f"<td>{escape(str(case.get('post_3072_max_highest_adjacent_pair_tile', '-')))}</td>"
            f"<td>{escape(str(final_features.get('primary_max_position', '-')) if isinstance(final_features, dict) else '-')}</td>"
            f"<td>{escape(str(final_features.get('primary_max_region', '-')) if isinstance(final_features, dict) else '-')}</td>"
            f"<td>{escape(str(final_features.get('adjacent_same_max', '-')) if isinstance(final_features, dict) else '-')}</td>"
            f"<td>{escape(str(final_features.get('adjacent_half_max', '-')) if isinstance(final_features, dict) else '-')}</td>"
            "</tr>"
        )
    summary = payload.get("summary", {})
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Threes Geometry Forensics</title>
  <style>
    :root {{ color-scheme: dark; --bg: #101214; --panel: #171c20; --line: #344049; --ink: #eef3ef; --muted: #a9b4ad; --gold: #e4bd4b; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--ink); font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ width: min(1180px, calc(100vw - 32px)); margin: 0 auto; padding: 24px 0 40px; }}
    h1 {{ margin: 0 0 6px; font-size: 24px; }}
    .muted {{ color: var(--muted); }}
    .stats {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 18px 0; }}
    .stat {{ border: 1px solid var(--line); background: var(--panel); border-radius: 8px; padding: 10px 12px; min-width: 130px; }}
    .stat b {{ display: block; color: var(--gold); font-size: 20px; }}
    table {{ width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 8px 9px; text-align: right; vertical-align: top; }}
    th:first-child, td:first-child {{ text-align: left; }}
  </style>
</head>
<body>
  <main>
    <h1>Geometry Forensics</h1>
    <p class="muted">High-board replay geometry, masking one free starter tile before measuring the built max tile.</p>
    <div class="stats">
      <div class="stat"><span>Replays</span><b>{escape(str(summary.get('replays', '-')))}</b></div>
      <div class="stat"><span>Reached 3072</span><b>{escape(str(summary.get('reached_3072', '-')))}</b></div>
      <div class="stat"><span>Reached 2x3072</span><b>{escape(str(summary.get('reached_two_3072', '-')))}</b></div>
      <div class="stat"><span>Reached 6144</span><b>{escape(str(summary.get('reached_6144', '-')))}</b></div>
      <div class="stat"><span>Mean 3072 Displaced</span><b>{escape(str(summary.get('mean_post_3072_displaced_rate', '-')))}</b></div>
      <div class="stat"><span>Max Adj Pair</span><b>{escape(str(summary.get('max_post_3072_highest_adjacent_pair_tile', '-')))}</b></div>
    </div>
    <table>
      <thead>
        <tr>
          <th>Seed</th><th>Score</th><th>Minus Starter</th><th>First 3072 Move</th>
          <th>First 2x3072 Move</th><th>First 3072 Pos</th><th>First 3072 Region</th><th>Post-3072 Displaced</th>
          <th>Max 3072 Count</th><th>Max Adj Pair</th>
          <th>Final Max Pos</th><th>Final Region</th><th>Adj Same</th><th>Adj Half</th>
        </tr>
      </thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </main>
</body>
</html>
"""
    path.write_text(html)


def run(replay_paths: list[Path], out_dir: Path) -> dict[str, Any]:
    replay_paths = unique_paths(replay_paths)
    cases = [analyze_replay(path) for path in replay_paths]
    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_replays": [str(path) for path in replay_paths],
        "summary": summarize_cases(cases),
        "cases": cases,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "geometry_forensics.json", payload)
    write_html(out_dir / "geometry_forensics.html", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-json", type=Path, nargs="+", action="append", default=[])
    parser.add_argument("--replay-glob", action="append", default=[])
    parser.add_argument("--out-dir", type=Path, default=Path("threes_rl/runs/forensics/geometry/latest"))
    args = parser.parse_args()
    replay_paths = [path for group in args.replay_json for path in group] + _glob_paths(args.replay_glob)
    if not replay_paths:
        raise SystemExit("No replay JSONs matched.")
    payload = run(replay_paths, args.out_dir)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"json={args.out_dir / 'geometry_forensics.json'}")
    print(f"html={args.out_dir / 'geometry_forensics.html'}")


if __name__ == "__main__":
    main()
