"""Extract replay JSONs embedded in continuation progress checkpoints."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from threes_rl.record_replay import write_html
from threes_rl.run_artifacts import safe_name, write_json


def _load_entries(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text())
    entries = payload.get("entries")
    if not isinstance(entries, dict):
        raise ValueError(f"{path} does not contain entries")
    return [entry for entry in entries.values() if isinstance(entry, dict)]


def _int_value(record: dict[str, Any], *names: str, default: int = 0) -> int:
    for name in names:
        value = record.get(name)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return int(default)


def _entry_record(entry: dict[str, Any]) -> dict[str, Any]:
    record = entry.get("record")
    return record if isinstance(record, dict) else {}


def _entry_replay(entry: dict[str, Any]) -> dict[str, Any] | None:
    replay = entry.get("replay")
    return replay if isinstance(replay, dict) else None


def _replay_with_record_metadata(replay: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    merged = dict(replay)
    metadata_fields = (
        ("seed", "seed"),
        ("start_case_id", "start_case_id"),
        ("source_replay", "source_replay"),
        ("source_seed", "source_seed"),
        ("source_frame_index", "source_frame_index"),
        ("starter_tile", "starter_tile"),
        ("start_score", "start_score"),
        ("start_max_tile_excl_starter", "start_max_tile_excl_starter"),
        ("final_score", "score"),
        ("final_score_delta", "score_delta"),
        ("final_moves_delta", "moves_delta"),
        ("final_max_tile", "max_tile"),
        ("final_max_tile_excl_starter", "max_tile_excl_starter"),
    )
    for replay_key, record_key in metadata_fields:
        if merged.get(replay_key) is not None:
            continue
        value = record.get(record_key)
        if value is not None:
            merged[replay_key] = value
    return merged


def _passes_filters(
    record: dict[str, Any],
    *,
    min_max_tile_excl_starter: int,
    max_max_tile_excl_starter: int,
) -> bool:
    max_tile = _int_value(record, "max_tile_excl_starter", "final_max_tile_excl_starter")
    if min_max_tile_excl_starter > 0 and max_tile < int(min_max_tile_excl_starter):
        return False
    if max_max_tile_excl_starter > 0 and max_tile > int(max_max_tile_excl_starter):
        return False
    return True


def _manifest_row(
    *,
    idx: int,
    record: dict[str, Any],
    replay: dict[str, Any],
    json_path: Path,
    html_path: Path,
) -> dict[str, Any]:
    return {
        "index": int(idx),
        "seed": _int_value(record, "seed", default=_int_value(replay, "seed")),
        "start_case_id": record.get("start_case_id", replay.get("start_case_id")),
        "source_seed": record.get("source_seed", replay.get("source_seed")),
        "source_frame_index": record.get("source_frame_index", replay.get("source_frame_index")),
        "score": _int_value(record, "score", "final_score"),
        "score_delta": _int_value(record, "score_delta", "final_score_delta"),
        "moves_delta": _int_value(record, "moves_delta", "final_moves_delta"),
        "max_tile": _int_value(record, "max_tile", "final_max_tile"),
        "max_tile_excl_starter": _int_value(record, "max_tile_excl_starter", "final_max_tile_excl_starter"),
        "json": str(json_path),
        "html": str(html_path),
    }


def _summary(rows: list[dict[str, Any]], *, progress_json: Path, source_entries: int, skipped: dict[str, int]) -> dict[str, Any]:
    maxes = Counter(str(row.get("max_tile_excl_starter", "unknown")) for row in rows)
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "progress_json": str(progress_json),
        "source_entries": int(source_entries),
        "replays": len(rows),
        "skipped": skipped,
        "by_max_tile_excl_starter": dict(maxes),
        "reached_6144": sum(int(row.get("max_tile_excl_starter", 0)) >= 6144 for row in rows),
        "high_score": max((_int_value(row, "score") for row in rows), default=0),
        "high_score_delta": max((_int_value(row, "score_delta") for row in rows), default=0),
    }


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    entries = _load_entries(args.progress_json)
    replay_dir = args.out_dir / "replays"
    replay_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()

    for idx, entry in enumerate(entries, start=1):
        record = _entry_record(entry)
        replay = _entry_replay(entry)
        if replay is None:
            skipped["missing_replay"] += 1
            continue
        if not _passes_filters(
            record,
            min_max_tile_excl_starter=args.min_max_tile_excl_starter,
            max_max_tile_excl_starter=args.max_max_tile_excl_starter,
        ):
            skipped["filter"] += 1
            continue
        replay = _replay_with_record_metadata(replay, record)
        name = safe_name(
            "continuation_"
            f"{idx:04d}_"
            f"score{_int_value(record, 'score', 'final_score')}_"
            f"delta{_int_value(record, 'score_delta', 'final_score_delta')}_"
            f"max{_int_value(record, 'max_tile_excl_starter', 'final_max_tile_excl_starter')}_"
            f"seed{_int_value(record, 'seed')}",
            max_length=140,
        )
        json_path = replay_dir / f"{name}.json"
        html_path = replay_dir / f"{name}.html"
        write_json(json_path, replay)
        if not args.no_html:
            write_html(html_path, replay)
        rows.append(_manifest_row(idx=idx, record=record, replay=replay, json_path=json_path, html_path=html_path))

    rows.sort(key=lambda row: (int(row.get("score_delta", 0)), int(row.get("score", 0))), reverse=True)
    summary = _summary(rows, progress_json=args.progress_json, source_entries=len(entries), skipped=dict(skipped))
    payload = {
        "version": 1,
        "kind": "extracted_continuation_replays",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "summary": summary,
        "manifest": rows,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload["json"] = str(args.out_dir / "extracted_continuation_replays.json")
    payload["manifest_json"] = str(args.out_dir / "manifest.json")
    payload["replay_dir"] = str(replay_dir)
    write_json(args.out_dir / "extracted_continuation_replays.json", payload)
    write_json(args.out_dir / "manifest.json", rows)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress-json", type=Path, required=True)
    parser.add_argument("--min-max-tile-excl-starter", type=int, default=0)
    parser.add_argument("--max-max-tile-excl-starter", type=int, default=0)
    parser.add_argument("--no-html", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("threes_rl/runs/forensics/extracted_continuations/latest"))
    args = parser.parse_args()
    payload = run_from_args(args)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"json={payload['json']}")
    print(f"manifest={payload['manifest_json']}")
    print(f"replays={payload['replay_dir']}")


if __name__ == "__main__":
    main()
