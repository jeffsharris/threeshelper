"""Generate synthetic support-ladder diagnostic states from high-board records."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from html import escape
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from threes_rl.compare_replays import state_from_payload
from threes_rl.eval import max_tile_excluding_initial_starter, starter_baseline_score
from threes_rl.record_replay import state_payload
from threes_rl.run_artifacts import safe_name, write_json
from threes_rl.sim import SimState, ThreesSim, score_board
from threes_rl.support_ladder_window_reservoir import raw_ladder_features
from threes_rl.swing_label import state_features

ORTHOGONAL = ((-1, 0), (1, 0), (0, -1), (0, 1))
MODES = ("identity", "adjacent768", "three768", "four768", "one1536_adjacent768", "adjacent1536")


def _flatten_paths(path_groups: list[list[Path]] | None) -> list[Path]:
    if not path_groups:
        return []
    return [path for group in path_groups for path in group]


def parse_modes(text: str | None) -> list[str]:
    raw = text or ",".join(MODES)
    modes: list[str] = []
    seen: set[str] = set()
    for part in raw.split(","):
        mode = part.strip()
        if not mode:
            continue
        if mode not in MODES:
            raise ValueError(f"Unsupported synthetic mode: {mode}")
        if mode not in seen:
            modes.append(mode)
            seen.add(mode)
    if not modes:
        raise ValueError("at least one synthetic mode is required")
    return modes


def _load_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text())
    records = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError(f"{path} does not contain records[]")
    return [
        {**record, "_source_json": str(path), "_record_index": idx}
        for idx, record in enumerate(records)
        if isinstance(record, dict)
    ]


def load_records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in paths:
        out.extend(_load_records(Path(path)))
    return out


def _record_starter(record: dict[str, Any], default: int | None) -> int | None:
    value = record.get("starter_tile", default)
    return None if value is None else int(value)


def _record_seed(record: dict[str, Any], fallback: int) -> int:
    value = record.get("source_seed", record.get("seed"))
    return int(value) if value is not None else int(fallback)


def _source_key(record: dict[str, Any]) -> str:
    replay = record.get("source_replay", record.get("_source_json", "unknown_replay"))
    seed = record.get("source_seed", record.get("seed", "unknown_seed"))
    return f"{replay}|{seed}"


def _positions(board: np.ndarray, value: int) -> list[tuple[int, int]]:
    return [tuple(int(v) for v in pos) for pos in np.argwhere(np.asarray(board, dtype=np.int32) == int(value))]


def _min_distance_to_positions(pair: tuple[tuple[int, int], tuple[int, int]], positions: list[tuple[int, int]]) -> int:
    if not positions:
        return 99
    return min(abs(a[0] - b[0]) + abs(a[1] - b[1]) for a in pair for b in positions)


def _candidate_pairs(board: np.ndarray, *, max_existing_tile: int, prefer_near_tile: int = 3072) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    arr = np.asarray(board, dtype=np.int32)
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for row in range(4):
        for col in range(4):
            left = (row, col)
            for dr, dc in ((1, 0), (0, 1)):
                right = (row + dr, col + dc)
                if right[0] >= 4 or right[1] >= 4:
                    continue
                values = [int(arr[left]), int(arr[right])]
                if any(value >= int(max_existing_tile) for value in values):
                    continue
                pairs.append((left, right))
    anchors = _positions(arr, prefer_near_tile)
    pairs.sort(
        key=lambda pair: (
            _min_distance_to_positions(pair, anchors),
            min(pos[0] + pos[1] for pos in pair),
            sum(int(arr[pos]) for pos in pair),
        )
    )
    return pairs


def _set_pair(board: np.ndarray, pair: tuple[tuple[int, int], tuple[int, int]], value: int) -> np.ndarray:
    arr = np.asarray(board, dtype=np.int32).copy()
    for row, col in pair:
        arr[int(row), int(col)] = int(value)
    return arr


def _set_single(board: np.ndarray, value: int, *, max_existing_tile: int, prefer_near_tile: int) -> np.ndarray | None:
    arr = np.asarray(board, dtype=np.int32).copy()
    anchors = _positions(arr, prefer_near_tile)
    candidates = []
    for row in range(4):
        for col in range(4):
            existing = int(arr[row, col])
            if existing >= int(max_existing_tile):
                continue
            candidates.append(((row, col), existing))
    if not candidates:
        return None
    candidates.sort(
        key=lambda item: (
            _min_distance_to_positions((item[0], item[0]), anchors),
            item[0][0] + item[0][1],
            item[1],
        )
    )
    row, col = candidates[0][0]
    arr[row, col] = int(value)
    return arr


def _set_value_count(board: np.ndarray, value: int, *, target_count: int, prefer_near_tile: int) -> np.ndarray | None:
    arr = np.asarray(board, dtype=np.int32).copy()
    current = int(np.count_nonzero(arr == int(value)))
    if current >= int(target_count):
        return arr
    anchors = _positions(arr, value) + _positions(arr, prefer_near_tile)
    candidates = []
    for row in range(4):
        for col in range(4):
            existing = int(arr[row, col])
            if existing >= int(value):
                continue
            candidates.append(((row, col), existing))
    if len(candidates) < int(target_count) - current:
        return None
    candidates.sort(
        key=lambda item: (
            1 if item[1] == 0 else 0,
            _min_distance_to_positions((item[0], item[0]), anchors),
            -item[1],
            item[0][0] + item[0][1],
        )
    )
    for idx in range(int(target_count) - current):
        row, col = candidates[idx][0]
        arr[row, col] = int(value)
    return arr


def _copy_state_with_board(state: SimState, board: np.ndarray) -> SimState:
    arr = np.asarray(board, dtype=np.int32)
    return SimState(
        board=arr.copy(),
        preview=state.preview,
        small_counts=state.small_counts.copy(),
        small_pos=int(state.small_pos),
        small_seen_total=int(state.small_seen_total),
        span_small_pos=int(state.span_small_pos),
        large_pending=bool(state.large_pending),
        max_tile=int(arr.max(initial=0)),
        move_count=int(state.move_count),
        game_over=False,
    )


def _state_key(state: SimState) -> str:
    return json.dumps(
        {
            "board": [int(value) for value in np.asarray(state.board, dtype=np.int32).reshape(-1)],
            "preview": {
                "kind": state.preview.kind,
                "value": state.preview.value,
                "candidates": list(state.preview.candidates),
            },
            "cycle": {
                "small_counts": sorted((str(key), int(value)) for key, value in state.small_counts.items()),
                "small_pos": int(state.small_pos),
                "small_seen_total": int(state.small_seen_total),
                "span_small_pos": int(state.span_small_pos),
                "large_pending": bool(state.large_pending),
            },
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _mutated_boards(
    board: np.ndarray,
    mode: str,
    *,
    variants_per_mode: int,
    starter_tile: int | None,
) -> list[np.ndarray]:
    arr = np.asarray(board, dtype=np.int32)
    boards: list[np.ndarray] = []
    if mode == "identity":
        boards.append(arr.copy())
    elif mode == "adjacent768":
        for pair in _candidate_pairs(arr, max_existing_tile=1536):
            boards.append(_set_pair(arr, pair, 768))
            if len(boards) >= variants_per_mode:
                break
    elif mode == "three768":
        board = _set_value_count(arr, 768, target_count=3, prefer_near_tile=3072)
        if board is not None:
            boards.append(board)
    elif mode == "four768":
        board = _set_value_count(arr, 768, target_count=4, prefer_near_tile=3072)
        if board is not None:
            boards.append(board)
    elif mode == "one1536_adjacent768":
        base = arr
        starter_1536 = int(starter_tile == 1536 and int(base[0, 0]) == 1536)
        if int(np.count_nonzero(base == 1536)) <= starter_1536:
            promoted = _set_single(base, 1536, max_existing_tile=1536, prefer_near_tile=3072)
            if promoted is None:
                return []
            base = promoted
        for pair in _candidate_pairs(base, max_existing_tile=1536):
            boards.append(_set_pair(base, pair, 768))
            if len(boards) >= variants_per_mode:
                break
    elif mode == "adjacent1536":
        for pair in _candidate_pairs(arr, max_existing_tile=3072):
            boards.append(_set_pair(arr, pair, 1536))
            if len(boards) >= variants_per_mode:
                break
    else:
        raise ValueError(f"Unsupported synthetic mode: {mode}")
    return boards


def _record_id(source_record: dict[str, Any], mode: str, variant_index: int, state: SimState) -> str:
    raw = json.dumps(
        {
            "source_replay": source_record.get("source_replay"),
            "source_frame_index": source_record.get("source_frame_index", source_record.get("frame_index")),
            "mode": mode,
            "variant_index": int(variant_index),
            "state": _state_key(state),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.blake2s(raw.encode("utf-8"), digest_size=8).hexdigest()
    seed = source_record.get("source_seed", source_record.get("seed", "unknown"))
    frame = source_record.get("source_frame_index", source_record.get("frame_index", "unknown"))
    return safe_name(f"synthetic_{mode}_seed{seed}_frame{frame}_{variant_index}_{digest}", max_length=128)


def _synthetic_record(
    *,
    source_record: dict[str, Any],
    source_state: SimState,
    state: SimState,
    sim: ThreesSim,
    starter_tile: int | None,
    mode: str,
    variant_index: int,
) -> dict[str, Any]:
    features = state_features(state, sim, starter_tile)
    raw = raw_ladder_features(state.board, starter_tile)
    source_raw = raw_ladder_features(source_state.board, starter_tile)
    payload = state_payload(state, sim)
    return {
        "id": _record_id(source_record, mode, variant_index, state),
        "kind": "synthetic_support_ladder_state",
        "synthetic": True,
        "synthetic_kind": mode,
        "synthetic_variant_index": int(variant_index),
        "source_record_id": source_record.get("id"),
        "source_replay": str(source_record.get("source_replay", source_record.get("_source_json", "synthetic_source"))),
        "source_origin": source_record.get("source_origin"),
        "source_policy": source_record.get("source_policy"),
        "source_policy_family": source_record.get("source_policy_family"),
        "source_seed": source_record.get("source_seed", source_record.get("seed")),
        "seed": source_record.get("source_seed", source_record.get("seed")),
        "root_origin": source_record.get("root_origin"),
        "root_replay": source_record.get("root_replay"),
        "root_seed": source_record.get("root_seed"),
        "root_frame_index": source_record.get("root_frame_index"),
        "root_move_count": source_record.get("root_move_count"),
        "root_score": source_record.get("root_score"),
        "root_policy": source_record.get("root_policy"),
        "root_policy_family": source_record.get("root_policy_family"),
        "root_is_genuine": source_record.get("root_is_genuine"),
        "ancestry_key": source_record.get("ancestry_key"),
        "source_frame_index": int(source_record.get("source_frame_index", source_record.get("frame_index", 0))),
        "frame_position": source_record.get("frame_position"),
        "starter_tile": starter_tile,
        "move_count": int(state.move_count),
        "score": int(score_board(state.board)),
        "score_minus_starter": int(score_board(state.board) - starter_baseline_score(starter_tile)),
        "max_tile": int(state.max_tile),
        "max_tile_excl_starter": int(max_tile_excluding_initial_starter(state.board, starter_tile)),
        "phase": str(features["phase"]),
        "corner_risk": str(features["corner_risk"]),
        "stratum": str(features["stratum"]),
        "empty_count": int(features["empty_count"]),
        "legal_count": int(len(sim.legal_actions(state))),
        "preview": str(features["preview"]),
        "large_pending": bool(features["large_pending"]),
        "raw_count_768": int(raw["raw_count_768"]),
        "raw_count_1536": int(raw["raw_count_1536"]),
        "raw_highest_duplicate_tile": int(raw["raw_highest_duplicate_tile"]),
        "raw_highest_adjacent_pair_tile": int(raw["raw_highest_adjacent_pair_tile"]),
        "raw_has_adjacent_768": bool(raw["raw_has_adjacent_768"]),
        "raw_has_adjacent_1536": bool(raw["raw_has_adjacent_1536"]),
        "source_raw_ladder": source_raw,
        "features": {**features, **raw, "legal_count": int(len(sim.legal_actions(state)))},
        "state": payload,
    }


def generate_records(
    source_records: Iterable[dict[str, Any]],
    *,
    modes: Iterable[str],
    variants_per_mode: int = 2,
    default_starter_tile: int | None = 1536,
    min_tile: int = 3072,
    max_records: int = 0,
    max_per_source: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    seen_states: set[str] = set()
    mode_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    scanned = 0
    source_list = list(source_records)
    for record_idx, source_record in enumerate(source_list):
        source_key = _source_key(source_record)
        if max_per_source > 0 and source_counts[source_key] >= int(max_per_source):
            rejected["max_per_source"] += 1
            continue
        state_payload_obj = source_record.get("state")
        if not isinstance(state_payload_obj, dict):
            rejected["missing_state"] += 1
            continue
        starter_tile = _record_starter(source_record, default_starter_tile)
        seed = _record_seed(source_record, record_idx)
        sim = ThreesSim(np.random.default_rng(seed), starter_tile=starter_tile)
        try:
            source_state = state_from_payload(state_payload_obj)
        except (TypeError, ValueError):
            rejected["bad_state"] += 1
            continue
        scanned += 1
        if source_state.game_over or int(max_tile_excluding_initial_starter(source_state.board, starter_tile)) < int(min_tile):
            rejected["below_min_or_terminal"] += 1
            continue
        for mode in modes:
            for variant_index, board in enumerate(
                _mutated_boards(
                    source_state.board,
                    mode,
                    variants_per_mode=variants_per_mode,
                    starter_tile=starter_tile,
                ),
                start=1,
            ):
                state = _copy_state_with_board(source_state, board)
                if not sim.legal_actions(state):
                    rejected["no_legal_actions"] += 1
                    continue
                key = _state_key(state)
                if key in seen_states:
                    rejected["duplicate_state"] += 1
                    continue
                if max_per_source > 0 and source_counts[source_key] >= int(max_per_source):
                    rejected["max_per_source"] += 1
                    break
                seen_states.add(key)
                records.append(
                    _synthetic_record(
                        source_record=source_record,
                        source_state=source_state,
                        state=state,
                        sim=sim,
                        starter_tile=starter_tile,
                        mode=mode,
                        variant_index=variant_index,
                    )
                )
                mode_counts[mode] += 1
                source_counts[source_key] += 1
                if max_records > 0 and len(records) >= int(max_records):
                    break
            if max_records > 0 and len(records) >= int(max_records):
                break
        if max_records > 0 and len(records) >= int(max_records):
            break
    summary = summarize_records(
        records,
        source_records=len(source_list),
        scanned_records=scanned,
        modes=list(modes),
        variants_per_mode=variants_per_mode,
        min_tile=min_tile,
        max_records=max_records,
        max_per_source=max_per_source,
        mode_counts=dict(mode_counts),
        source_counts=dict(source_counts),
        rejected=dict(rejected),
    )
    return records, summary


def summarize_records(
    records: list[dict[str, Any]],
    *,
    source_records: int,
    scanned_records: int,
    modes: list[str],
    variants_per_mode: int,
    min_tile: int,
    max_records: int,
    max_per_source: int,
    mode_counts: dict[str, int],
    source_counts: dict[str, int],
    rejected: dict[str, int],
) -> dict[str, Any]:
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_records": int(source_records),
        "scanned_records": int(scanned_records),
        "records": len(records),
        "modes": list(modes),
        "mode_counts": mode_counts,
        "variants_per_mode": int(variants_per_mode),
        "min_tile": int(min_tile),
        "max_records": int(max_records),
        "max_per_source": int(max_per_source),
        "by_stratum": dict(Counter(str(record.get("stratum")) for record in records)),
        "by_synthetic_kind": dict(Counter(str(record.get("synthetic_kind")) for record in records)),
        "raw_duplicate_768": sum(int(record.get("raw_count_768", 0)) >= 2 for record in records),
        "raw_adjacent_768": sum(bool(record.get("raw_has_adjacent_768")) for record in records),
        "raw_duplicate_1536": sum(int(record.get("raw_count_1536", 0)) >= 2 for record in records),
        "raw_adjacent_1536": sum(bool(record.get("raw_has_adjacent_1536")) for record in records),
        "source_replays": len({str(record.get("source_replay")) for record in records}),
        "source_seeds": len({str(record.get("source_seed")) for record in records}),
        "source_count_min": int(min(source_counts.values(), default=0)),
        "source_count_max": int(max(source_counts.values(), default=0)),
        "rejected": rejected,
    }


def write_html(path: Path, payload: dict[str, Any]) -> None:
    summary = payload.get("summary", {})
    records = payload.get("records", [])

    def cell(value: object) -> str:
        return escape(str(value))

    rows = []
    for record in records[:300] if isinstance(records, list) else []:
        if not isinstance(record, dict):
            continue
        rows.append(
            "<tr>"
            f"<td>{cell(record.get('synthetic_kind'))}</td>"
            f"<td>{cell(record.get('source_seed'))}</td>"
            f"<td>{cell(record.get('source_frame_index'))}</td>"
            f"<td>{cell(record.get('stratum'))}</td>"
            f"<td>{cell(record.get('score_minus_starter'))}</td>"
            f"<td>{cell(record.get('raw_count_768'))}</td>"
            f"<td>{cell(record.get('raw_has_adjacent_768'))}</td>"
            f"<td>{cell(record.get('raw_count_1536'))}</td>"
            f"<td>{cell(record.get('raw_has_adjacent_1536'))}</td>"
            f"<td>{cell(record.get('source_replay'))}</td>"
            "</tr>"
        )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Synthetic Support-Ladder States</title>
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
    th:first-child, td:first-child, th:nth-child(4), td:nth-child(4), th:last-child, td:last-child {{ text-align:left; }}
    td:last-child {{ max-width:360px; overflow-wrap:anywhere; color:var(--muted); }}
    pre {{ white-space:pre-wrap; overflow-wrap:anywhere; color:var(--muted); }}
  </style>
</head>
<body>
  <main>
    <h1>Synthetic Support-Ladder States</h1>
    <p class="muted">Diagnostic generated starts; do not mix into training data without an explicit experiment.</p>
    <section class="cards">
      <div class="card"><div class="label">Records</div><div class="value">{cell(summary.get('records', 0))}</div></div>
      <div class="card"><div class="label">Source Records</div><div class="value">{cell(summary.get('source_records', 0))}</div></div>
      <div class="card"><div class="label">Modes</div><div class="value">{cell(summary.get('modes', []))}</div></div>
      <div class="card"><div class="label">1536 Adjacent</div><div class="value">{cell(summary.get('raw_adjacent_1536', 0))}</div></div>
    </section>
    <table><thead><tr><th>Mode</th><th>Seed</th><th>Frame</th><th>Stratum</th><th>Score - Starter</th><th>768 Count</th><th>Adj 768</th><th>1536 Count</th><th>Adj 1536</th><th>Source</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
    <h2>Summary JSON</h2>
    <pre>{escape(json.dumps(summary, indent=2, sort_keys=True))}</pre>
  </main>
</body>
</html>
"""
    path.write_text(html)


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    paths = _flatten_paths(args.state_json)
    source_records = load_records(paths)
    modes = parse_modes(args.modes)
    records, summary = generate_records(
        source_records,
        modes=modes,
        variants_per_mode=args.variants_per_mode,
        default_starter_tile=None if str(args.starter).strip().lower() == "none" else int(args.starter),
        min_tile=args.min_tile,
        max_records=args.max_records,
        max_per_source=args.max_per_source,
    )
    payload = {
        "version": 1,
        "kind": "synthetic_support_ladder_state_generator",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_json": [str(path) for path in paths],
        "summary": summary,
        "records": records,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload["json"] = str(args.out_dir / "synthetic_support_ladder_states.json")
    payload["records_json"] = str(args.out_dir / "records.json")
    payload["html"] = str(args.out_dir / "synthetic_support_ladder_states.html")
    write_json(args.out_dir / "synthetic_support_ladder_states.json", payload)
    write_json(args.out_dir / "records.json", records)
    write_html(args.out_dir / "synthetic_support_ladder_states.html", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-json", type=Path, nargs="+", action="append", required=True)
    parser.add_argument("--modes", default="adjacent768,one1536_adjacent768,adjacent1536")
    parser.add_argument("--variants-per-mode", type=int, default=2)
    parser.add_argument("--starter", default="1536")
    parser.add_argument("--min-tile", type=int, default=3072)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--max-per-source", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("threes_rl/runs/forensics/synthetic_support_ladder/latest"))
    args = parser.parse_args()
    payload = run_from_args(args)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"json={payload['json']}")
    print(f"records={payload['records_json']}")
    print(f"html={payload['html']}")


if __name__ == "__main__":
    main()
