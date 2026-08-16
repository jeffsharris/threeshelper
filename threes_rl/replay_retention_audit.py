"""Dry-run replay retention audit for Threes RL run artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from threes_rl.dashboard import (
    RUNS_ROOT,
    collect_global_top_replays,
    _skip_dashboard_source,
    _skip_replay_start_training,
)
from threes_rl.run_artifacts import write_json


DEFAULT_OUT = RUNS_ROOT / "dashboard" / "replay_retention_audit.json"


def _canonical(path: Path) -> str:
    return str(path.expanduser().resolve(strict=False))


def _runs_relative_path(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _resolve_artifact_path(root: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    raw = Path(value)
    candidates: list[Path] = [raw]
    if not raw.is_absolute():
        candidates.append(Path.cwd() / raw)
        parts = raw.parts
        if "runs" in parts:
            runs_index = parts.index("runs")
            suffix = Path(*parts[runs_index + 1 :])
            candidates.append(root / suffix)
        else:
            candidates.append(root / raw)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def _score(item: dict[str, Any]) -> int | None:
    try:
        return int(float(item.get("score")))
    except (TypeError, ValueError):
        return None


def _top_game_entry(root: Path, summary_path: Path, item: dict[str, Any], *, dashboard_eligible: bool) -> dict[str, Any]:
    json_path = _resolve_artifact_path(root, item.get("json"))
    html_path = _resolve_artifact_path(root, item.get("html"))
    return {
        "run": summary_path.parent.name,
        "run_path": str(summary_path.parent),
        "summary": str(summary_path),
        "dashboard_eligible": bool(dashboard_eligible),
        "score": _score(item),
        "score_minus_starter": _score({"score": item.get("score_minus_starter")}),
        "seed": item.get("seed"),
        "starter_tile": item.get("starter_tile"),
        "moves": item.get("moves"),
        "max_tile": item.get("max_tile"),
        "max_tile_excl_starter": item.get("max_tile_excl_starter"),
        "html": str(html_path) if html_path is not None else item.get("html"),
        "json": str(json_path) if json_path is not None else item.get("json"),
        "json_exists": bool(json_path and json_path.exists()),
        "html_exists": bool(html_path and html_path.exists()),
    }


def _collect_top_game_entries(root: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for summary_path in sorted(root.rglob("summary.json")):
        skip_source = _skip_dashboard_source(summary_path)
        skip_replay_start = _skip_replay_start_training(summary_path)
        try:
            payload = json.loads(summary_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        top_games = payload.get("top_games")
        if not isinstance(top_games, list):
            continue
        dashboard_eligible = not skip_source and not skip_replay_start
        for item in top_games:
            if isinstance(item, dict):
                entries.append(_top_game_entry(root, summary_path, item, dashboard_eligible=dashboard_eligible))
    entries.sort(key=lambda entry: (entry.get("score") or 0, entry.get("moves") or 0), reverse=True)
    return entries


def _protected_path_keys(root: Path, top_replays: list[dict[str, Any]]) -> set[str]:
    keys: set[str] = set()
    for replay in top_replays:
        for field in ("json", "html"):
            resolved = _resolve_artifact_path(root, replay.get(field))
            if resolved is None:
                continue
            keys.add(_canonical(resolved))
            keys.add(_canonical(resolved.parent))
    return keys


def _replay_category(path: Path) -> str:
    parts = set(path.parts)
    if "continuations" in parts:
        return "continuations"
    if "human_diagnostics" in parts:
        return "human_diagnostics"
    if "top_games" in parts:
        return "top_games"
    if "top_delta_games" in parts:
        return "top_delta_games"
    if "diagnostic_games" in parts:
        return "diagnostic_games"
    if "milestone_games" in parts:
        return "milestone_games"
    return "other"


def _count_replay_files(root: Path, protected_keys: set[str]) -> dict[str, Any]:
    json_paths = sorted(root.rglob("replay.json"))
    html_paths = sorted(root.rglob("replay.html"))
    dirs_by_category: dict[str, set[str]] = {}
    protected_dirs = 0
    for path in json_paths:
        category = _replay_category(path)
        dirs_by_category.setdefault(category, set()).add(_canonical(path.parent))
        if _canonical(path) in protected_keys or _canonical(path.parent) in protected_keys:
            protected_dirs += 1
    category_counts = {key: len(value) for key, value in sorted(dirs_by_category.items())}
    return {
        "replay_json_files": len(json_paths),
        "replay_html_files": len(html_paths),
        "replay_dirs": len({ _canonical(path.parent) for path in json_paths }),
        "protected_replay_dirs": protected_dirs,
        "replay_dirs_by_category": category_counts,
    }


def _compact_entry(root: Path, entry: dict[str, Any]) -> dict[str, Any]:
    compact = dict(entry)
    for field in ("json", "html", "summary", "run_path"):
        value = compact.get(field)
        if isinstance(value, str):
            compact[field] = _runs_relative_path(root, Path(value))
    return compact


def build_replay_retention_audit(
    root: Path = RUNS_ROOT,
    *,
    global_top_limit: int = 3,
    preview_limit: int = 25,
) -> dict[str, Any]:
    """Return a dry-run report; this function never deletes artifacts."""

    root = Path(root)
    global_top = collect_global_top_replays(root, limit=global_top_limit)
    protected_keys = _protected_path_keys(root, global_top)
    top_entries = _collect_top_game_entries(root)
    eligible_entries = [entry for entry in top_entries if entry["dashboard_eligible"]]
    excluded_entries = [entry for entry in top_entries if not entry["dashboard_eligible"]]

    protected_json_keys: set[str] = set()
    for replay in global_top:
        resolved = _resolve_artifact_path(root, replay.get("json"))
        if resolved is not None:
            protected_json_keys.add(_canonical(resolved))
    non_global_top_entries = [
        entry
        for entry in eligible_entries
        if entry.get("json") and _canonical(Path(str(entry["json"]))) not in protected_json_keys
    ]
    missing_protected = [
        replay
        for replay in global_top
        if replay.get("json") and not Path(str(_resolve_artifact_path(root, replay.get("json")))).exists()
    ]
    file_counts = _count_replay_files(root, protected_keys)

    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "dry_run",
        "root": str(root),
        "global_top_limit": int(global_top_limit),
        "protected_global_top_replays": [_compact_entry(root, replay) for replay in global_top],
        "counts": {
            "summary_files": len(list(root.rglob("summary.json"))),
            "top_game_entries_total": len(top_entries),
            "top_game_entries_dashboard_eligible": len(eligible_entries),
            "top_game_entries_excluded_from_dashboard": len(excluded_entries),
            "non_global_top_game_entries": len(non_global_top_entries),
            "missing_protected_global_top_json": len(missing_protected),
            **file_counts,
        },
        "retain_by_default": {
            "note": (
                "Diagnostic, milestone, continuation, and human-diagnostic replays are "
                "scientific evidence and are not prune candidates in this audit."
            ),
            "categories": {
                key: file_counts["replay_dirs_by_category"].get(key, 0)
                for key in ("top_delta_games", "diagnostic_games", "milestone_games", "continuations", "human_diagnostics")
            },
        },
        "potential_prune": {
            "note": (
                "Dry-run only: these are dashboard-eligible per-run top-game entries that "
                "are not in the global top replay set. Delete only after explicit review."
            ),
            "per_run_top_games_not_global_top_count": len(non_global_top_entries),
            "preview": [_compact_entry(root, entry) for entry in non_global_top_entries[: max(0, int(preview_limit))]],
        },
    }


def write_replay_retention_audit(
    out_path: Path = DEFAULT_OUT,
    *,
    root: Path = RUNS_ROOT,
    global_top_limit: int = 3,
    preview_limit: int = 25,
) -> dict[str, Any]:
    payload = build_replay_retention_audit(root, global_top_limit=global_top_limit, preview_limit=preview_limit)
    write_json(out_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=RUNS_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--global-top-limit", type=int, default=3)
    parser.add_argument("--preview-limit", type=int, default=25)
    parser.add_argument("--print-summary", action="store_true")
    args = parser.parse_args()

    payload = write_replay_retention_audit(
        args.out,
        root=args.root,
        global_top_limit=args.global_top_limit,
        preview_limit=args.preview_limit,
    )
    if args.print_summary:
        print(
            json.dumps(
                {
                    "out": str(args.out),
                    "protected_global_top_replays": len(payload["protected_global_top_replays"]),
                    "counts": payload["counts"],
                    "potential_prune_count": payload["potential_prune"]["per_run_top_games_not_global_top_count"],
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
