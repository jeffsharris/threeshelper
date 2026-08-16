"""Search for prefixes that preserve adjacent 768 support before making 1536."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

import numpy as np

from threes_rl.rare_event_frontier import FrontierCase, load_cases, load_records, parse_root_origins, select_diverse_cases
from threes_rl.record_replay import state_payload
from threes_rl.run_artifacts import safe_name, write_json
from threes_rl.sim import DIRECTION_NAMES, SimState, ThreesSim, score_board
from threes_rl.support_ladder_window_reservoir import raw_ladder_features
from threes_rl.train_td import copy_state


@dataclass
class FrontierNode:
    case: FrontierCase
    state: SimState
    depth: int
    path: list[str]
    seed_path: list[int]
    start_score: int
    score_delta: int
    parent_id: str | None
    node_id: str


def _state_digest(state: SimState) -> str:
    raw = json.dumps(
        {
            "board": np.asarray(state.board, dtype=int).tolist(),
            "preview": state.preview.label,
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
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _raw(state: SimState, starter_tile: int | None) -> dict[str, Any]:
    raw = raw_ladder_features(state.board, starter_tile)
    return {
        "raw_count_768": int(raw["raw_count_768"]),
        "raw_count_1536": int(raw["raw_count_1536"]),
        "raw_count_3072": int(raw["raw_count_3072"]),
        "raw_has_adjacent_768": bool(raw["raw_has_adjacent_768"]),
        "raw_has_adjacent_1536": bool(raw["raw_has_adjacent_1536"]),
        "raw_highest_duplicate_tile": int(raw["raw_highest_duplicate_tile"]),
        "raw_highest_adjacent_pair_tile": int(raw["raw_highest_adjacent_pair_tile"]),
        "empty_count": int(np.count_nonzero(np.asarray(state.board, dtype=np.int32) == 0)),
    }


def _node_id(case: FrontierCase, state: SimState, *, depth: int, path: list[str], seed_path: list[int]) -> str:
    return safe_name(
        f"preserve_{case.id}_d{int(depth)}_{'_'.join(path) or 'start'}_{_state_digest(state)}",
        max_length=160,
    )


def _start_node(case: FrontierCase) -> FrontierNode:
    state = copy_state(case.state)
    return FrontierNode(
        case=case,
        state=state,
        depth=0,
        path=[],
        seed_path=[],
        start_score=int(score_board(state.board)),
        score_delta=0,
        parent_id=None,
        node_id=_node_id(case, state, depth=0, path=[], seed_path=[]),
    )


def _archive_key(node: FrontierNode) -> tuple[str, int, int, int, int]:
    raw = _raw(node.state, node.case.starter_tile)
    empty_bucket = int(raw["empty_count"]) // 2
    return (
        str(node.case.ancestry_key or node.case.id),
        int(raw["raw_count_768"]),
        int(raw["raw_highest_adjacent_pair_tile"]),
        int(empty_bucket),
        int(node.depth),
    )


def _score_node(node: FrontierNode) -> tuple[float, ...]:
    raw = _raw(node.state, node.case.starter_tile)
    sim = ThreesSim(np.random.default_rng(0), starter_tile=node.case.starter_tile)
    legal_count = len(sim.legal_actions(node.state))
    return (
        float(node.depth),
        float(raw["empty_count"]),
        float(legal_count),
        float(raw["raw_count_768"]),
        float(raw["raw_highest_adjacent_pair_tile"]) / 768.0,
        float(node.score_delta) / 200000.0,
    )


def _is_four_768_target(state: SimState, starter_tile: int | None) -> bool:
    raw = _raw(state, starter_tile)
    return bool(raw["raw_has_adjacent_768"]) and int(raw["raw_count_768"]) >= 4 and int(raw["raw_count_1536"]) == 0


def select_archive_nodes(
    candidates: list[FrontierNode],
    *,
    max_nodes: int,
    max_per_cell: int,
) -> list[FrontierNode]:
    if not candidates:
        return []
    grouped: dict[tuple[str, int, int, int, int], list[FrontierNode]] = defaultdict(list)
    seen_ids: set[str] = set()
    for node in sorted(candidates, key=_score_node, reverse=True):
        if node.node_id in seen_ids:
            continue
        seen_ids.add(node.node_id)
        key = _archive_key(node)
        if len(grouped[key]) < int(max_per_cell):
            grouped[key].append(node)
    selected = [node for nodes in grouped.values() for node in nodes]
    selected.sort(key=_score_node, reverse=True)
    return selected[: max(0, int(max_nodes))]


def _record_for_node(node: FrontierNode) -> dict[str, Any]:
    sim = ThreesSim(np.random.default_rng(node.case.source_seed or 0), starter_tile=node.case.starter_tile)
    raw = _raw(node.state, node.case.starter_tile)
    legal = sim.legal_actions(node.state)
    return {
        "id": node.node_id,
        "kind": "support_preservation_frontier_state",
        "state": state_payload(node.state, sim),
        "starter_tile": node.case.starter_tile,
        "source_replay": node.case.source_replay,
        "source_seed": node.case.source_seed,
        "source_frame_index": node.case.source_frame_index,
        "source_policy": node.case.source_policy,
        "root_origin": node.case.root_origin,
        "root_replay": node.case.root_replay,
        "root_seed": node.case.root_seed,
        "root_frame_index": node.case.root_frame_index,
        "root_policy": node.case.root_policy,
        "root_policy_family": node.case.root_policy_family,
        "ancestry_key": node.case.ancestry_key,
        "score": int(score_board(node.state.board)),
        "score_delta": int(node.score_delta),
        "move_count": int(node.state.move_count),
        "features": {
            **raw,
            "depth": int(node.depth),
            "legal_count": int(len(legal)),
            "score_delta": int(node.score_delta),
            "path_length": int(len(node.path)),
        },
        "frontier": {
            "parent_id": node.parent_id,
            "depth": int(node.depth),
            "path": list(node.path),
            "seed_path": [int(seed) for seed in node.seed_path],
            "archive_key": list(_archive_key(node)),
            "rank_tuple": list(_score_node(node)),
        },
    }


def _step_seed(base_seed: int, *, depth: int, node_index: int, action: int, repeat: int) -> int:
    return int(base_seed) + int(depth) * 1_000_003 + int(node_index) * 10_007 + int(action) * 503 + int(repeat) * 97


def expand_frontier(
    starts: list[FrontierCase],
    *,
    max_depth: int,
    repeats_per_action: int,
    max_nodes_per_depth: int,
    max_per_cell: int,
    seed: int,
    progress_every: int,
) -> tuple[list[FrontierNode], list[FrontierNode], list[dict[str, Any]], list[dict[str, Any]]]:
    frontier = [_start_node(case) for case in starts]
    archive: list[FrontierNode] = list(frontier)
    target_nodes: list[FrontierNode] = [
        node for node in frontier if _is_four_768_target(node.state, node.case.starter_tile)
    ]
    depth_rows: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    expanded = 0
    for depth in range(1, int(max_depth) + 1):
        candidates: list[FrontierNode] = []
        stats: Counter[str] = Counter()
        for node_index, node in enumerate(frontier):
            sim_for_legal = ThreesSim(np.random.default_rng(seed + depth + node_index), starter_tile=node.case.starter_tile)
            legal_actions = sim_for_legal.legal_actions(node.state)
            if not legal_actions:
                stats["no_legal"] += 1
                continue
            for action in legal_actions:
                for repeat in range(int(repeats_per_action)):
                    step_seed = _step_seed(seed, depth=depth, node_index=node_index, action=int(action), repeat=repeat)
                    sim = ThreesSim(np.random.default_rng(step_seed), starter_tile=node.case.starter_tile)
                    next_state, info = sim.step(node.state, int(action))
                    expanded += 1
                    if not info.moved:
                        stats["not_moved"] += 1
                        continue
                    raw = _raw(next_state, node.case.starter_tile)
                    outcome = "preserved"
                    if next_state.game_over:
                        outcome = "terminal"
                    elif int(raw["raw_count_1536"]) >= 1 and bool(raw["raw_has_adjacent_768"]):
                        outcome = "support_with_1536"
                    elif int(raw["raw_count_1536"]) >= 1:
                        outcome = "converted_without_support"
                    elif not bool(raw["raw_has_adjacent_768"]):
                        outcome = "lost_support"
                    stats[outcome] += 1
                    transition = {
                        "parent_id": node.node_id,
                        "case_id": node.case.id,
                        "depth": int(depth),
                        "action": DIRECTION_NAMES[int(action)],
                        "repeat_index": int(repeat),
                        "seed": int(step_seed),
                        "outcome": outcome,
                        "score_delta": int(score_board(next_state.board) - node.start_score),
                        **raw,
                    }
                    transitions.append(transition)
                    if outcome == "preserved":
                        path = node.path + [DIRECTION_NAMES[int(action)]]
                        seed_path = node.seed_path + [int(step_seed)]
                        child = FrontierNode(
                            case=node.case,
                            state=copy_state(next_state),
                            depth=int(depth),
                            path=path,
                            seed_path=seed_path,
                            start_score=node.start_score,
                            score_delta=int(score_board(next_state.board) - node.start_score),
                            parent_id=node.node_id,
                            node_id=_node_id(node.case, next_state, depth=depth, path=path, seed_path=seed_path),
                        )
                        candidates.append(child)
                        if _is_four_768_target(child.state, child.case.starter_tile):
                            target_nodes.append(child)
                    elif outcome == "support_with_1536":
                        path = node.path + [DIRECTION_NAMES[int(action)]]
                        seed_path = node.seed_path + [int(step_seed)]
                        target_nodes.append(
                            FrontierNode(
                                case=node.case,
                                state=copy_state(next_state),
                                depth=int(depth),
                                path=path,
                                seed_path=seed_path,
                                start_score=node.start_score,
                                score_delta=int(score_board(next_state.board) - node.start_score),
                                parent_id=node.node_id,
                                node_id=_node_id(node.case, next_state, depth=depth, path=path, seed_path=seed_path),
                            )
                        )
            if progress_every > 0 and expanded % int(progress_every) == 0:
                print(
                    "preserve_frontier "
                    f"depth={depth} expanded={expanded} "
                    f"candidates={len(candidates)} "
                    f"support_with_1536={stats['support_with_1536']} "
                    f"converted_without_support={stats['converted_without_support']}",
                    flush=True,
                )
        selected = select_archive_nodes(candidates, max_nodes=max_nodes_per_depth, max_per_cell=max_per_cell)
        archive.extend(selected)
        row = {
            "depth": int(depth),
            "frontier_in": len(frontier),
            "candidate_preserved": len(candidates),
            "frontier_out": len(selected),
            **{key: int(value) for key, value in sorted(stats.items())},
        }
        depth_rows.append(row)
        frontier = selected
        if not frontier:
            break
    unique_targets: dict[str, FrontierNode] = {}
    for node in sorted(target_nodes, key=_score_node, reverse=True):
        unique_targets.setdefault(node.node_id, node)
    return archive, list(unique_targets.values()), depth_rows, transitions


def summarize(
    *,
    source_records: int,
    cases_total: int,
    selected_cases: list[FrontierCase],
    rejected: dict[str, int],
    archive: list[FrontierNode],
    target_nodes: list[FrontierNode],
    depth_rows: list[dict[str, Any]],
    transitions: list[dict[str, Any]],
    max_depth: int,
    repeats_per_action: int,
    max_nodes_per_depth: int,
    max_per_cell: int,
) -> dict[str, Any]:
    deepest = max((node.depth for node in archive), default=0)
    preserved_by_depth = Counter(int(node.depth) for node in archive)
    target_by_depth = Counter(int(node.depth) for node in target_nodes)
    outcomes = Counter(str(row.get("outcome")) for row in transitions)
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_records": int(source_records),
        "cases_total": int(cases_total),
        "cases_selected": len(selected_cases),
        "rejected": rejected,
        "max_depth": int(max_depth),
        "repeats_per_action": int(repeats_per_action),
        "max_nodes_per_depth": int(max_nodes_per_depth),
        "max_per_cell": int(max_per_cell),
        "archive_records": len(archive),
        "target_records": len(target_nodes),
        "deepest_preserved_depth": int(deepest),
        "preserved_by_depth": {str(key): int(value) for key, value in sorted(preserved_by_depth.items())},
        "target_by_depth": {str(key): int(value) for key, value in sorted(target_by_depth.items())},
        "transitions": len(transitions),
        "transition_outcomes": dict(outcomes),
        "support_with_1536_transitions": int(outcomes.get("support_with_1536", 0)),
        "converted_without_support_transitions": int(outcomes.get("converted_without_support", 0)),
        "lost_support_transitions": int(outcomes.get("lost_support", 0)),
        "terminal_transitions": int(outcomes.get("terminal", 0)),
        "unique_root_seeds": len({case.root_seed for case in selected_cases if case.root_seed is not None}),
        "target_unique_root_seeds": len({node.case.root_seed for node in target_nodes if node.case.root_seed is not None}),
    }


def write_html(path: Path, payload: dict[str, Any]) -> None:
    summary = payload.get("summary", {})
    depth_rows = payload.get("depth_rows", [])
    records = payload.get("records", [])

    def cell(value: object) -> str:
        return escape(str(value))

    depth_html = "".join(
        "<tr>"
        f"<td>{cell(row.get('depth'))}</td>"
        f"<td>{cell(row.get('frontier_in'))}</td>"
        f"<td>{cell(row.get('candidate_preserved'))}</td>"
        f"<td>{cell(row.get('frontier_out'))}</td>"
        f"<td>{cell(row.get('converted_without_support', 0))}</td>"
        f"<td>{cell(row.get('lost_support', 0))}</td>"
        f"<td>{cell(row.get('terminal', 0))}</td>"
        "</tr>"
        for row in depth_rows
    )
    top_records = sorted(
        [record for record in records if isinstance(record, dict)],
        key=lambda row: tuple(row.get("frontier", {}).get("rank_tuple", [])),
        reverse=True,
    )[:50]
    record_html = "".join(
        "<tr>"
        f"<td>{cell((row.get('features') or {}).get('depth'))}</td>"
        f"<td>{cell(row.get('source_seed'))}</td>"
        f"<td>{cell(row.get('move_count'))}</td>"
        f"<td>{cell((row.get('features') or {}).get('empty_count'))}</td>"
        f"<td>{cell((row.get('features') or {}).get('raw_count_768'))}</td>"
        f"<td>{cell((row.get('features') or {}).get('raw_highest_adjacent_pair_tile'))}</td>"
        f"<td>{cell(' '.join((row.get('frontier') or {}).get('path', [])))}</td>"
        "</tr>"
        for row in top_records
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Support Preservation Frontier</title>
  <style>
    :root {{ color-scheme: dark; --bg: #101318; --panel: #171d24; --line: #34404d; --text: #edf2f7; --muted: #aab6c2; --gold: #f2c14e; }}
    body {{ margin: 0; background: var(--bg); color: var(--text); font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ width: min(1180px, calc(100vw - 32px)); margin: 0 auto; padding: 24px 0 40px; }}
    h1 {{ margin: 0 0 8px; font-size: 24px; }}
    .muted {{ color: var(--muted); }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 18px 0; }}
    .card, .panel {{ border: 1px solid var(--line); background: var(--panel); border-radius: 8px; padding: 12px; }}
    .label {{ color: var(--muted); font-size: 11px; text-transform: uppercase; }}
    .value {{ margin-top: 4px; color: var(--gold); font-size: 22px; font-weight: 800; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 12px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 6px 8px; text-align: right; vertical-align: top; }}
    th:last-child, td:last-child {{ text-align: left; overflow-wrap: anywhere; }}
    pre {{ white-space: pre-wrap; overflow-wrap: anywhere; margin: 0; color: var(--muted); }}
  </style>
</head>
<body>
  <main>
    <h1>Support Preservation Frontier</h1>
    <p class="muted">Archived prefixes that keep adjacent 768 alive while avoiding raw 1536 conversion.</p>
    <section class="cards">
      <div class="card"><div class="label">Cases</div><div class="value">{cell(summary.get('cases_selected', 0))}</div></div>
      <div class="card"><div class="label">Archive</div><div class="value">{cell(summary.get('archive_records', 0))}</div></div>
      <div class="card"><div class="label">Deepest</div><div class="value">{cell(summary.get('deepest_preserved_depth', 0))}</div></div>
      <div class="card"><div class="label">Targets</div><div class="value">{cell(summary.get('target_records', 0))}</div></div>
    </section>
    <section class="panel"><pre>{escape(json.dumps(summary, indent=2, sort_keys=True))}</pre></section>
    <section class="panel" style="margin-top: 14px;">
      <h2>Depths</h2>
      <table><thead><tr><th>Depth</th><th>In</th><th>Preserved</th><th>Out</th><th>Converted</th><th>Lost</th><th>Terminal</th></tr></thead><tbody>{depth_html}</tbody></table>
    </section>
    <section class="panel" style="margin-top: 14px;">
      <h2>Top Archived States</h2>
      <table><thead><tr><th>Depth</th><th>Seed</th><th>Move</th><th>Air</th><th>768s</th><th>Adj</th><th>Path</th></tr></thead><tbody>{record_html}</tbody></table>
    </section>
  </main>
</body>
</html>
"""
    path.write_text(html)


def run_frontier(
    *,
    records_json: list[Path],
    max_depth: int,
    repeats_per_action: int,
    max_starts: int,
    max_nodes_per_depth: int,
    max_per_cell: int,
    seed: int,
    root_origins: set[str],
    default_starter_tile: int | None,
    require_start_adjacent_768: bool,
    require_no_start_1536: bool,
    out_dir: Path,
    progress_every: int = 200,
) -> dict[str, Any]:
    records = load_records(records_json)
    cases, rejected = load_cases(records, default_starter_tile=default_starter_tile, root_origins=root_origins)
    filtered: list[FrontierCase] = []
    for case in cases:
        if require_start_adjacent_768 and not bool(case.raw.get("raw_has_adjacent_768")):
            rejected["missing_start_adjacent_768"] = rejected.get("missing_start_adjacent_768", 0) + 1
            continue
        if require_no_start_1536 and int(case.raw.get("raw_count_1536", 0)) > 0:
            rejected["start_has_1536"] = rejected.get("start_has_1536", 0) + 1
            continue
        filtered.append(case)
    selected = select_diverse_cases(filtered, max_starts=max_starts, seed=seed)
    if not selected:
        raise ValueError("No start cases matched the requested filters")
    archive, target_nodes, depth_rows, transitions = expand_frontier(
        selected,
        max_depth=max_depth,
        repeats_per_action=repeats_per_action,
        max_nodes_per_depth=max_nodes_per_depth,
        max_per_cell=max_per_cell,
        seed=seed,
        progress_every=progress_every,
    )
    archive_records = [_record_for_node(node) for node in archive]
    target_records = [_record_for_node(node) for node in target_nodes]
    summary = summarize(
        source_records=len(records),
        cases_total=len(filtered),
        selected_cases=selected,
        rejected=rejected,
        archive=archive,
        target_nodes=target_nodes,
        depth_rows=depth_rows,
        transitions=transitions,
        max_depth=max_depth,
        repeats_per_action=repeats_per_action,
        max_nodes_per_depth=max_nodes_per_depth,
        max_per_cell=max_per_cell,
    )
    payload = {
        "version": 1,
        "kind": "support_preservation_frontier",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "records_json": [str(path) for path in records_json],
        "summary": summary,
        "depth_rows": depth_rows,
        "transitions": transitions,
        "records": archive_records,
        "target_records": target_records,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    payload["json"] = str(out_dir / "support_preservation_frontier.json")
    payload["records_json"] = str(out_dir / "records.json")
    payload["summary_json"] = str(out_dir / "summary.json")
    payload["transitions_json"] = str(out_dir / "transitions.json")
    payload["target_records_json"] = str(out_dir / "target_records.json")
    payload["html"] = str(out_dir / "support_preservation_frontier.html")
    write_json(out_dir / "support_preservation_frontier.json", payload)
    write_json(out_dir / "records.json", archive_records)
    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "depth_rows.json", depth_rows)
    write_json(out_dir / "transitions.json", transitions)
    write_json(out_dir / "target_records.json", target_records)
    write_html(out_dir / "support_preservation_frontier.html", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-json", type=Path, action="append", required=True)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--repeats-per-action", type=int, default=2)
    parser.add_argument("--max-starts", type=int, default=0)
    parser.add_argument("--max-nodes-per-depth", type=int, default=128)
    parser.add_argument("--max-per-cell", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--root-origin", help="Comma-separated root origins; defaults to fresh,human.")
    parser.add_argument("--starter", default="1536")
    parser.add_argument("--require-start-adjacent-768", action="store_true")
    parser.add_argument("--allow-start-1536", action="store_true")
    parser.add_argument("--progress-every", type=int, default=200)
    parser.add_argument("--out-dir", type=Path, default=Path("threes_rl/runs/forensics/support_preservation_frontier/latest"))
    args = parser.parse_args()
    starter_text = args.starter.strip().lower()
    default_starter = None if starter_text == "none" else int(starter_text)
    payload = run_frontier(
        records_json=args.records_json,
        max_depth=args.max_depth,
        repeats_per_action=args.repeats_per_action,
        max_starts=args.max_starts,
        max_nodes_per_depth=args.max_nodes_per_depth,
        max_per_cell=args.max_per_cell,
        seed=args.seed,
        root_origins=parse_root_origins(args.root_origin),
        default_starter_tile=default_starter,
        require_start_adjacent_768=bool(args.require_start_adjacent_768),
        require_no_start_1536=not bool(args.allow_start_1536),
        out_dir=args.out_dir,
        progress_every=args.progress_every,
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"json={payload['json']}")
    print(f"records={payload['records_json']}")
    print(f"html={payload['html']}")


if __name__ == "__main__":
    main()
