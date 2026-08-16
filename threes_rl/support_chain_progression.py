"""Replay-level support-chain progression diagnostics after first built 3072."""

from __future__ import annotations

import argparse
import glob
import json
import time
from collections import Counter
from html import escape
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable, Iterable

import numpy as np

from threes_rl.eval import starter_baseline_score
from threes_rl.geometry_forensics import board_without_free_starter, geometry_features
from threes_rl.run_artifacts import write_json
from threes_rl.sim import score_board

ORTHOGONAL = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _flatten_paths(path_groups: list[list[Path]] | None) -> list[Path]:
    if not path_groups:
        return []
    return [path for group in path_groups for path in group]


def _glob_paths(patterns: list[str] | None) -> list[Path]:
    if not patterns:
        return []
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(Path(match) for match in glob.glob(pattern, recursive=True))
    return sorted(path for path in paths if path.is_file())


def _starter_from_replay(replay: dict[str, Any], default: int | None = 1536) -> int | None:
    value = replay.get("starter_tile", default)
    return None if value is None else int(value)


def _frame_board(frame: dict[str, Any]) -> np.ndarray | None:
    state = frame.get("state") if isinstance(frame, dict) else None
    if not isinstance(state, dict):
        return None
    board = state.get("board")
    if not isinstance(board, list):
        return None
    arr = np.asarray(board, dtype=np.int32)
    return arr if arr.shape == (4, 4) else None


def _frame_move_count(frame: dict[str, Any], fallback: int) -> int:
    state = frame.get("state") if isinstance(frame, dict) else None
    if isinstance(state, dict) and state.get("move_count") is not None:
        return int(state["move_count"])
    return int(frame.get("index", fallback))


def _positions(board: np.ndarray, value: int) -> list[tuple[int, int]]:
    return [tuple(int(v) for v in pos) for pos in np.argwhere(np.asarray(board, dtype=np.int32) == int(value))]


def _has_adjacent_pair(board: np.ndarray, value: int) -> bool:
    positions = _positions(board, value)
    for idx, left in enumerate(positions):
        for right in positions[idx + 1 :]:
            if abs(left[0] - right[0]) + abs(left[1] - right[1]) == 1:
                return True
    return False


def _tile_counts(board: np.ndarray) -> Counter[int]:
    return Counter(int(value) for value in np.asarray(board, dtype=np.int32).reshape(-1) if int(value) > 0)


def _highest_duplicate_tile(board: np.ndarray) -> int:
    counts = _tile_counts(board)
    return max((value for value, count in counts.items() if count >= 2), default=0)


def _highest_adjacent_pair_tile(board: np.ndarray) -> int:
    values = sorted({int(value) for value in np.asarray(board, dtype=np.int32).reshape(-1) if int(value) > 0}, reverse=True)
    for value in values:
        if _has_adjacent_pair(board, value):
            return int(value)
    return 0


def _first_index(frames: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]) -> int | None:
    for idx, frame in enumerate(frames):
        if predicate(frame):
            return idx
    return None


def _milestone_payload(frames: list[dict[str, Any]], idx: int | None, first_3072_idx: int | None) -> dict[str, Any] | None:
    if idx is None:
        return None
    frame = frames[int(idx)]
    return {
        "frame_position": int(idx),
        "frame_index": int(frame.get("index", idx)),
        "move_count": _frame_move_count(frame, int(idx)),
        "frames_after_first_3072": None if first_3072_idx is None else int(idx) - int(first_3072_idx),
    }


def _raw_features(board: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(board, dtype=np.int32)
    return {
        "raw_count_768": int(np.count_nonzero(arr == 768)),
        "raw_count_1536": int(np.count_nonzero(arr == 1536)),
        "raw_count_3072": int(np.count_nonzero(arr == 3072)),
        "raw_count_6144": int(np.count_nonzero(arr == 6144)),
        "raw_highest_duplicate_tile": _highest_duplicate_tile(arr),
        "raw_highest_adjacent_pair_tile": _highest_adjacent_pair_tile(arr),
        "raw_has_adjacent_768": _has_adjacent_pair(arr, 768),
        "raw_has_adjacent_1536": _has_adjacent_pair(arr, 1536),
    }


def _frame_features(frame: dict[str, Any], starter_tile: int | None) -> dict[str, Any] | None:
    board = _frame_board(frame)
    if board is None:
        return None
    masked = board_without_free_starter(board, starter_tile)
    geo = geometry_features(board, starter_tile)
    return {
        **_raw_features(board),
        "masked_count_768": int(np.count_nonzero(masked == 768)),
        "masked_count_1536": int(np.count_nonzero(masked == 1536)),
        "masked_count_3072": int(np.count_nonzero(masked == 3072)),
        "masked_highest_duplicate_tile": int(geo["highest_duplicate_tile"]),
        "masked_highest_adjacent_pair_tile": int(geo["highest_adjacent_pair_tile"]),
        "top_left": int(board[0, 0]),
        "top_left_is_masked_max": bool(geo["primary_max_position"] == [0, 0]),
        "max_tile_excl_starter": int(geo["max_tile_excl_starter"]),
        "empty_count": int(np.count_nonzero(board == 0)),
        "score_minus_starter": int(score_board(board) - starter_baseline_score(starter_tile)),
    }


def analyze_replay(path: Path) -> dict[str, Any]:
    replay = json.loads(Path(path).read_text())
    if not isinstance(replay, dict):
        raise ValueError(f"{path} is not a replay object")
    starter_tile = _starter_from_replay(replay)
    frames = [frame for frame in replay.get("frames", []) if isinstance(frame, dict)]
    features_by_idx: list[dict[str, Any] | None] = [_frame_features(frame, starter_tile) for frame in frames]

    first_3072_idx = _first_index(
        frames,
        lambda frame: (_frame_features(frame, starter_tile) or {}).get("max_tile_excl_starter", 0) >= 3072,
    )
    event_idx = _first_index(
        frames,
        lambda frame: (_frame_features(frame, starter_tile) or {}).get("raw_count_3072", 0) >= 2
        or (_frame_features(frame, starter_tile) or {}).get("max_tile_excl_starter", 0) >= 6144,
    )
    first_6144_idx = _first_index(
        frames,
        lambda frame: (_frame_features(frame, starter_tile) or {}).get("max_tile_excl_starter", 0) >= 6144,
    )
    post_start = int(first_3072_idx) if first_3072_idx is not None else 0
    post_indices = range(post_start, len(frames))
    pre_event_end = int(event_idx) if event_idx is not None else len(frames)
    pre_event_indices = range(post_start, max(post_start, pre_event_end))

    def first_post(predicate: Callable[[dict[str, Any]], bool]) -> int | None:
        for idx in post_indices:
            features = features_by_idx[idx]
            if features is not None and predicate(features):
                return idx
        return None

    milestones = {
        "first_raw_three_768": first_post(lambda features: int(features["raw_count_768"]) >= 3),
        "first_raw_four_768": first_post(lambda features: int(features["raw_count_768"]) >= 4),
        "first_raw_duplicate_768": first_post(lambda features: int(features["raw_highest_duplicate_tile"]) >= 768),
        "first_raw_adjacent_pair_768": first_post(lambda features: int(features["raw_highest_adjacent_pair_tile"]) >= 768),
        "first_raw_duplicate_1536": first_post(lambda features: int(features["raw_count_1536"]) >= 2),
        "first_raw_adjacent_pair_1536": first_post(lambda features: bool(features["raw_has_adjacent_1536"])),
        "first_masked_duplicate_768": first_post(lambda features: int(features["masked_highest_duplicate_tile"]) >= 768),
        "first_masked_adjacent_pair_768": first_post(lambda features: int(features["masked_highest_adjacent_pair_tile"]) >= 768),
        "first_masked_duplicate_1536": first_post(lambda features: int(features["masked_count_1536"]) >= 2),
        "first_masked_adjacent_pair_1536": first_post(lambda features: int(features["masked_highest_adjacent_pair_tile"]) >= 1536),
    }

    post_features = [features_by_idx[idx] for idx in post_indices if features_by_idx[idx] is not None]
    pre_event_features = [features_by_idx[idx] for idx in pre_event_indices if features_by_idx[idx] is not None]
    outcome = "success" if event_idx is not None else "failure"
    final_features = next((features for features in reversed(features_by_idx) if features is not None), None)
    return {
        "source_replay": str(path),
        "seed": replay.get("seed"),
        "starter_tile": starter_tile,
        "outcome": outcome,
        "event_kind": None
        if event_idx is None
        else ("visible_two_3072" if first_6144_idx is None or event_idx < first_6144_idx else "direct_6144"),
        "final_score": replay.get("final_score"),
        "final_score_minus_starter": None
        if replay.get("final_score") is None
        else int(replay.get("final_score")) - starter_baseline_score(starter_tile),
        "final_moves": replay.get("final_moves"),
        "frames": len(frames),
        "post_3072_frames": len(post_features),
        "first_3072": _milestone_payload(frames, first_3072_idx, first_3072_idx),
        "second_3072_event": _milestone_payload(frames, event_idx, first_3072_idx),
        "first_6144": _milestone_payload(frames, first_6144_idx, first_3072_idx),
        "milestones": {
            name: _milestone_payload(frames, idx, first_3072_idx)
            for name, idx in milestones.items()
        },
        "post_3072_max_raw_duplicate_tile": max((int(features["raw_highest_duplicate_tile"]) for features in post_features), default=0),
        "post_3072_max_raw_adjacent_pair_tile": max((int(features["raw_highest_adjacent_pair_tile"]) for features in post_features), default=0),
        "post_3072_max_raw_count_768": max((int(features["raw_count_768"]) for features in post_features), default=0),
        "post_3072_max_masked_duplicate_tile": max((int(features["masked_highest_duplicate_tile"]) for features in post_features), default=0),
        "post_3072_max_masked_adjacent_pair_tile": max((int(features["masked_highest_adjacent_pair_tile"]) for features in post_features), default=0),
        "pre_event_max_raw_duplicate_tile": max((int(features["raw_highest_duplicate_tile"]) for features in pre_event_features), default=0),
        "pre_event_max_raw_adjacent_pair_tile": max((int(features["raw_highest_adjacent_pair_tile"]) for features in pre_event_features), default=0),
        "pre_event_max_raw_count_768": max((int(features["raw_count_768"]) for features in pre_event_features), default=0),
        "pre_event_max_masked_duplicate_tile": max((int(features["masked_highest_duplicate_tile"]) for features in pre_event_features), default=0),
        "pre_event_max_masked_adjacent_pair_tile": max((int(features["masked_highest_adjacent_pair_tile"]) for features in pre_event_features), default=0),
        "post_3072_max_raw_count_1536": max((int(features["raw_count_1536"]) for features in post_features), default=0),
        "post_3072_max_masked_count_1536": max((int(features["masked_count_1536"]) for features in post_features), default=0),
        "pre_event_max_raw_count_1536": max((int(features["raw_count_1536"]) for features in pre_event_features), default=0),
        "pre_event_max_masked_count_1536": max((int(features["masked_count_1536"]) for features in pre_event_features), default=0),
        "post_3072_top_left_is_max_rate": None
        if not post_features
        else sum(bool(features["top_left_is_masked_max"]) for features in post_features) / len(post_features),
        "final_features": final_features,
    }


def _value_at_path(case: dict[str, Any], path: list[str]) -> Any:
    current: Any = case
    for part in path:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _milestone_rate(cases: list[dict[str, Any]], name: str) -> dict[str, Any]:
    values = []
    present = 0
    for case in cases:
        milestone = (case.get("milestones") or {}).get(name)
        if isinstance(milestone, dict):
            present += 1
            if milestone.get("frames_after_first_3072") is not None:
                values.append(int(milestone["frames_after_first_3072"]))
    return {
        "present": int(present),
        "rate": float(present / len(cases)) if cases else 0.0,
        "median_frames_after_first_3072": float(median(values)) if values else None,
        "mean_frames_after_first_3072": float(mean(values)) if values else None,
    }


def _summarize_group(cases: list[dict[str, Any]]) -> dict[str, Any]:
    if not cases:
        return {"replays": 0}
    top_left_rates = [float(case["post_3072_top_left_is_max_rate"]) for case in cases if case.get("post_3072_top_left_is_max_rate") is not None]
    return {
        "replays": len(cases),
        "reached_second_3072": sum(1 for case in cases if case.get("second_3072_event") is not None),
        "reached_6144": sum(1 for case in cases if case.get("first_6144") is not None),
        "mean_final_score_minus_starter": float(mean(int(case.get("final_score_minus_starter") or 0) for case in cases)),
        "median_final_score_minus_starter": float(median(int(case.get("final_score_minus_starter") or 0) for case in cases)),
        "max_raw_duplicate_tile": max(int(case.get("post_3072_max_raw_duplicate_tile") or 0) for case in cases),
        "max_raw_adjacent_pair_tile": max(int(case.get("post_3072_max_raw_adjacent_pair_tile") or 0) for case in cases),
        "max_raw_count_768": max(int(case.get("post_3072_max_raw_count_768") or 0) for case in cases),
        "max_masked_duplicate_tile": max(int(case.get("post_3072_max_masked_duplicate_tile") or 0) for case in cases),
        "max_masked_adjacent_pair_tile": max(int(case.get("post_3072_max_masked_adjacent_pair_tile") or 0) for case in cases),
        "max_pre_event_raw_duplicate_tile": max(int(case.get("pre_event_max_raw_duplicate_tile") or 0) for case in cases),
        "max_pre_event_raw_adjacent_pair_tile": max(int(case.get("pre_event_max_raw_adjacent_pair_tile") or 0) for case in cases),
        "max_pre_event_raw_count_768": max(int(case.get("pre_event_max_raw_count_768") or 0) for case in cases),
        "max_pre_event_masked_duplicate_tile": max(int(case.get("pre_event_max_masked_duplicate_tile") or 0) for case in cases),
        "max_pre_event_masked_adjacent_pair_tile": max(int(case.get("pre_event_max_masked_adjacent_pair_tile") or 0) for case in cases),
        "mean_post_3072_top_left_is_max_rate": float(mean(top_left_rates)) if top_left_rates else None,
        "milestones": {
            name: _milestone_rate(cases, name)
            for name in (
                "first_raw_three_768",
                "first_raw_four_768",
                "first_raw_duplicate_768",
                "first_raw_adjacent_pair_768",
                "first_raw_duplicate_1536",
                "first_raw_adjacent_pair_1536",
                "first_masked_duplicate_1536",
                "first_masked_adjacent_pair_1536",
            )
        },
    }


def summarize_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    by_outcome = {
        outcome: _summarize_group([case for case in cases if str(case.get("outcome")) == outcome])
        for outcome in ("success", "failure")
    }
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "replays": len(cases),
        "by_outcome_count": dict(Counter(str(case.get("outcome")) for case in cases)),
        "reached_second_3072": sum(1 for case in cases if case.get("second_3072_event") is not None),
        "reached_6144": sum(1 for case in cases if case.get("first_6144") is not None),
        "by_outcome": by_outcome,
    }


def write_html(path: Path, payload: dict[str, Any]) -> None:
    summary = payload.get("summary", {})
    cases = payload.get("cases", [])

    def cell(value: object) -> str:
        return escape(str(value))

    rows = []
    for case in cases if isinstance(cases, list) else []:
        if not isinstance(case, dict):
            continue
        milestones = case.get("milestones") if isinstance(case.get("milestones"), dict) else {}
        rows.append(
            "<tr>"
            f"<td>{cell(case.get('outcome'))}</td>"
            f"<td>{cell(case.get('seed'))}</td>"
            f"<td>{cell(case.get('final_score'))}</td>"
            f"<td>{cell(_value_at_path(case, ['first_3072', 'move_count']))}</td>"
            f"<td>{cell(_value_at_path(case, ['second_3072_event', 'move_count']))}</td>"
            f"<td>{cell(_value_at_path(case, ['first_6144', 'move_count']))}</td>"
            f"<td>{cell(_value_at_path({'m': milestones.get('first_raw_three_768')}, ['m', 'frames_after_first_3072']))}</td>"
            f"<td>{cell(_value_at_path({'m': milestones.get('first_raw_four_768')}, ['m', 'frames_after_first_3072']))}</td>"
            f"<td>{cell(_value_at_path({'m': milestones.get('first_raw_duplicate_1536')}, ['m', 'frames_after_first_3072']))}</td>"
            f"<td>{cell(_value_at_path({'m': milestones.get('first_raw_adjacent_pair_1536')}, ['m', 'frames_after_first_3072']))}</td>"
            f"<td>{cell(case.get('pre_event_max_raw_count_768'))}</td>"
            f"<td>{cell(case.get('pre_event_max_raw_duplicate_tile'))}</td>"
            f"<td>{cell(case.get('pre_event_max_raw_adjacent_pair_tile'))}</td>"
            f"<td>{cell(case.get('post_3072_top_left_is_max_rate'))}</td>"
            f"<td>{cell(case.get('source_replay'))}</td>"
            "</tr>"
        )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Support-Chain Progression</title>
  <style>
    :root {{ color-scheme: dark; --bg:#101318; --panel:#171d24; --line:#34404d; --text:#edf2f7; --muted:#aab6c2; --gold:#f2c14e; }}
    body {{ margin:0; background:var(--bg); color:var(--text); font-family:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ width:min(1240px, calc(100vw - 32px)); margin:0 auto; padding:24px 0 40px; }}
    h1 {{ margin:0 0 8px; font-size:24px; }}
    .muted {{ color:var(--muted); }}
    .cards {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(150px, 1fr)); gap:10px; margin:18px 0; }}
    .card {{ border:1px solid var(--line); background:var(--panel); border-radius:8px; padding:12px; }}
    .label {{ color:var(--muted); font-size:11px; text-transform:uppercase; }}
    .value {{ margin-top:4px; color:var(--gold); font-size:22px; font-weight:800; }}
    table {{ width:100%; border-collapse:collapse; font-size:12px; font-variant-numeric:tabular-nums; }}
    th, td {{ border-bottom:1px solid var(--line); padding:7px 8px; text-align:right; vertical-align:top; }}
    th:first-child, td:first-child, th:last-child, td:last-child {{ text-align:left; }}
    td:last-child {{ max-width:360px; overflow-wrap:anywhere; color:var(--muted); }}
    pre {{ white-space:pre-wrap; overflow-wrap:anywhere; color:var(--muted); }}
  </style>
</head>
<body>
  <main>
    <h1>Support-Chain Progression</h1>
    <p class="muted">Replay-level milestones after first built 3072, using raw counts plus starter-masked geometry.</p>
    <section class="cards">
      <div class="card"><div class="label">Replays</div><div class="value">{cell(summary.get('replays', 0))}</div></div>
      <div class="card"><div class="label">2x3072</div><div class="value">{cell(summary.get('reached_second_3072', 0))}</div></div>
      <div class="card"><div class="label">6144</div><div class="value">{cell(summary.get('reached_6144', 0))}</div></div>
      <div class="card"><div class="label">Outcomes</div><div class="value">{cell(summary.get('by_outcome_count', {}))}</div></div>
    </section>
    <table><thead><tr><th>Outcome</th><th>Seed</th><th>Score</th><th>First 3072</th><th>2x3072</th><th>6144</th><th>Raw 3x768 +Frames</th><th>Raw 4x768 +Frames</th><th>Raw 1536 Dup +Frames</th><th>Raw 1536 Adj +Frames</th><th>Pre-event 768 Count</th><th>Pre-event Dup</th><th>Pre-event Adj</th><th>Top-left Max Rate</th><th>Replay</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
    <h2>Summary JSON</h2>
    <pre>{escape(json.dumps(summary, indent=2, sort_keys=True))}</pre>
  </main>
</body>
</html>
"""
    path.write_text(html)


def run(replay_paths: list[Path], out_dir: Path) -> dict[str, Any]:
    cases = []
    for path in replay_paths:
        try:
            cases.append(analyze_replay(path))
        except (OSError, json.JSONDecodeError, ValueError):
            continue
    payload = {
        "version": 1,
        "kind": "support_chain_progression",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_replays": [str(path) for path in replay_paths],
        "summary": summarize_cases(cases),
        "cases": cases,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    payload["json"] = str(out_dir / "support_chain_progression.json")
    payload["html"] = str(out_dir / "support_chain_progression.html")
    write_json(out_dir / "support_chain_progression.json", payload)
    write_html(out_dir / "support_chain_progression.html", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-json", type=Path, nargs="+", action="append", default=[])
    parser.add_argument("--replay-glob", action="append", default=[])
    parser.add_argument("--out-dir", type=Path, default=Path("threes_rl/runs/forensics/support_chain/latest"))
    args = parser.parse_args()
    replay_paths = _flatten_paths(args.replay_json) + _glob_paths(args.replay_glob)
    if not replay_paths:
        raise SystemExit("No replay JSONs matched.")
    payload = run(replay_paths, args.out_dir)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"json={payload['json']}")
    print(f"html={payload['html']}")


if __name__ == "__main__":
    main()
