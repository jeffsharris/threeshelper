"""Read-only corpus audit for the phase-0 empirical oracle gate.

This module deliberately does not run labels, rollout continuations, search, or
training.  It inspects existing pre-milestone state-record artifacts and reports
whether the corpus is diverse enough to justify launching the phase-0 oracle
gate.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import time
from collections import Counter, defaultdict
from html import escape
from pathlib import Path
from typing import Any, Iterable

from threes_rl.replay_provenance import GENUINE_ROOT_ORIGINS, policy_family
from threes_rl.run_artifacts import write_json


DEFAULT_RECORD_GLOBS = [
    "threes_rl/runs/forensics/transition_windows/*/records.json",
    "threes_rl/runs/forensics/transition_offset_samples/*/records.json",
]

BEHAVIOR_FAMILY_RULES = [
    {
        "family": "human_observed",
        "needles": ["human_observed", "human"],
        "description": "Human/tracker imports.",
    },
    {
        "family": "phaseblend_cheap_lineage",
        "needles": ["phaseblend1b"],
        "description": "Cheap 1b approximation of the phase-blend incumbent.",
    },
    {
        "family": "phaseblend_incumbent_lineage",
        "needles": ["ntuple_phaseblend", "phaseblend", "current_incumbent"],
        "description": "Current n-tuple phase-blend incumbent and close checkpoints.",
    },
    {
        "family": "corner2_lineage",
        "needles": ["corner2"],
        "description": "Corner-aware expectimax / corner2 teacher lineage.",
    },
    {
        "family": "expectimax_baseline",
        "needles": ["expectimax"],
        "description": "Non-learned expectimax baselines not already captured above.",
    },
    {
        "family": "td_student_lineage",
        "needles": ["td_default_student", "student"],
        "description": "TD student checkpoints derived from earlier actors.",
    },
    {
        "family": "synthetic_or_frontier",
        "needles": ["synthetic", "frontier", "support_accumulation"],
        "description": "Synthetic/frontier-generated starts; not counted as fresh policy diversity.",
    },
]


def _flatten_path_groups(path_groups: list[list[Path]] | None) -> list[Path]:
    if not path_groups:
        return []
    return [path for group in path_groups for path in group]


def _glob_records(patterns: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()
    for pattern in patterns:
        for text in glob.glob(pattern):
            path = Path(text)
            key = str(path.resolve(strict=False))
            if key in seen or not path.is_file():
                continue
            seen.add(key)
            paths.append(path)
    return sorted(paths, key=lambda path: str(path))


def _load_records(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    raw = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(raw, list):
        return []
    records: list[dict[str, Any]] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        record = dict(item)
        record["_source_json"] = str(path)
        record["_record_index"] = int(idx)
        records.append(record)
    return records


def load_records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        records.extend(_load_records(Path(path)))
    return records


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def outcome_offset(record: dict[str, Any]) -> int | None:
    """Moves to the relevant event for success or failure windows."""

    offset = _int_or_none(record.get("sample_offset"))
    if offset is not None:
        return offset
    promotion = _int_or_none(record.get("moves_to_promotion"))
    if promotion is not None:
        return promotion
    return _int_or_none(record.get("moves_to_terminal"))


def ancestry_key(record: dict[str, Any]) -> str:
    existing = record.get("ancestry_key")
    if existing:
        return str(existing)
    return "root:{origin}:{replay}:{seed}:{frame}".format(
        origin=record.get("root_origin", "unknown"),
        replay=record.get("root_replay") or record.get("source_replay") or "unknown",
        seed=record.get("root_seed", record.get("source_seed", "unknown")),
        frame=record.get("root_frame_index", 0),
    )


def _policy_text(record: dict[str, Any]) -> str:
    parts = [
        record.get("root_policy"),
        record.get("root_policy_family"),
        record.get("source_policy"),
        record.get("source_policy_family"),
        record.get("source_replay"),
        record.get("_source_json"),
    ]
    return " ".join(str(part) for part in parts if part is not None).lower()


def behavior_policy_family(record: dict[str, Any]) -> str:
    """Coalesce policy identity by behavioral lineage, not checkpoint name."""

    text = _policy_text(record)
    for rule in BEHAVIOR_FAMILY_RULES:
        if any(needle in text for needle in rule["needles"]):
            return str(rule["family"])
    fallback = record.get("root_policy_family") or record.get("source_policy_family")
    return policy_family(fallback)


def is_candidate_record(
    record: dict[str, Any],
    *,
    target_tile: int,
    horizon: int,
    require_genuine: bool = True,
    allowed_root_origins: set[str] | None = None,
) -> tuple[bool, str]:
    target = _int_or_none(record.get("target_tile"))
    if target != int(target_tile):
        return False, "target_tile"
    outcome = str(record.get("outcome", "")).lower()
    if outcome not in {"success", "failure"}:
        return False, "outcome"
    offset = outcome_offset(record)
    if offset is None or offset <= 0 or offset > int(horizon):
        return False, "horizon"
    root_origin = str(record.get("root_origin", "unknown"))
    if allowed_root_origins is not None and root_origin not in allowed_root_origins:
        return False, "root_origin"
    if require_genuine and root_origin not in GENUINE_ROOT_ORIGINS:
        return False, "not_genuine"
    if not record.get("source_next_action"):
        return False, "missing_next_action"
    if not isinstance(record.get("state"), dict):
        return False, "missing_state"
    return True, "accepted"


def _selection_sort_key(record: dict[str, Any], horizon: int) -> tuple[Any, ...]:
    offset = outcome_offset(record)
    offset_distance = abs(int(horizon) - int(offset or 0))
    return (
        offset_distance,
        -int(offset or 0),
        str(record.get("outcome", "")),
        str(record.get("_source_json", "")),
        int(record.get("_record_index", 0) or 0),
    )


def root_cap_candidates(records: Iterable[dict[str, Any]], *, horizon: int) -> list[dict[str, Any]]:
    best_by_ancestry: dict[str, dict[str, Any]] = {}
    for record in records:
        key = ancestry_key(record)
        current = best_by_ancestry.get(key)
        if current is None or _selection_sort_key(record, horizon) < _selection_sort_key(current, horizon):
            row = dict(record)
            row["phase0_ancestry_key"] = key
            row["phase0_behavior_family"] = behavior_policy_family(record)
            row["phase0_outcome_offset"] = outcome_offset(record)
            best_by_ancestry[key] = row
    return sorted(
        best_by_ancestry.values(),
        key=lambda record: (
            str(record.get("phase0_behavior_family")),
            str(record.get("outcome")),
            str(record.get("phase0_ancestry_key")),
        ),
    )


def _counts(records: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(record.get(key, "unknown")) for record in records))


def _nested_counts(records: Iterable[dict[str, Any]], first: str, second: str) -> dict[str, dict[str, int]]:
    out: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        out[str(record.get(first, "unknown"))][str(record.get(second, "unknown"))] += 1
    return {key: dict(value) for key, value in sorted(out.items())}


def _offset_buckets(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        offset = _int_or_none(record.get("phase0_outcome_offset"))
        if offset is None:
            counts["unknown"] += 1
        elif offset <= 10:
            counts["h10"] += 1
        elif offset <= 20:
            counts["h20"] += 1
        else:
            counts["h40"] += 1
    return dict(counts)


def _family_share(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(record.get("phase0_behavior_family", "unknown")) for record in records)
    if not records or not counts:
        return {"family": None, "roots": 0, "share": 0.0}
    family, roots = counts.most_common(1)[0]
    return {"family": family, "roots": int(roots), "share": float(roots / len(records))}


def _margin_rationale() -> dict[str, Any]:
    return {
        "primary_lift_margin_pp": 5.0,
        "primary_lift_rationale": (
            "The prior direct gates found effects around -1.04 pp and +0.78 pp with "
            "intervals touching zero. A new improvement operator must clear a "
            "larger, practically useful bar before it earns policy-fitting compute."
        ),
        "score_noninferiority": {
            "point_delta_min": -500,
            "lower_ci_min": -1500,
            "rationale": (
                "A teacher that buys milestone hits by discarding several thousand "
                "points is not a credible normal-start improvement. The lower-CI "
                "margin is intentionally small relative to the current latest mean "
                "minus-starter score, while allowing ordinary rollout noise."
            ),
        },
        "survival_noninferiority": {
            "point_delta_pp_min": -1.0,
            "lower_ci_pp_min": -2.5,
            "rationale": (
                "The sentinel is only useful if it does not materially increase "
                "early death. A small negative point estimate is tolerated, but a "
                "multi-point survival regression kills the operator."
            ),
        },
        "power_note": (
            "This dry-run cannot estimate paired action variance because no labels "
            "are run. With 20 ancestry-clustered roots, phase 0 is a kill gate for "
            "large practical effects, not a proof of small gains."
        ),
    }


def _gate_rules(
    *,
    min_roots: int,
    min_behavior_families: int,
    max_family_share: float,
    min_roots_per_outcome: int,
) -> dict[str, Any]:
    return {
        "sentinel_endpoint": "h40 first non-starter 1536",
        "secondary_metrics": ["h10 first non-starter 1536", "h20 first non-starter 1536", "h40 score", "h40 survival"],
        "pilot_eval_separation": (
            "Pilot CRN seeds may select and freeze the oracle action. Evaluation "
            "uses independent preregistered seed blocks and never reuses pilot "
            "outcomes."
        ),
        "empirical_only_phase0": "No learned leaf, fitting, search promotion, or rollout execution in this audit.",
        "minimum_roots": int(min_roots),
        "minimum_behavior_families": int(min_behavior_families),
        "maximum_single_behavior_family_share": float(max_family_share),
        "minimum_roots_per_outcome": int(min_roots_per_outcome),
        "concentration_guard": (
            "Replace the old 30% nonnegative-root rule with leave-one-root-out and "
            "leave-one-behavior-family-out robustness: every leave-one-family-out "
            "point estimate must remain positive when at least two families are "
            "available, all leave-one-root-out point estimates must remain "
            "positive, and the largest family may not provide more than half of "
            "positive lift."
        ),
    }


def audit_records(
    records: list[dict[str, Any]],
    *,
    source_paths: list[str],
    target_tile: int = 1536,
    horizon: int = 40,
    min_roots: int = 20,
    min_behavior_families: int = 2,
    max_family_share: float = 0.5,
    min_roots_per_outcome: int = 4,
    require_genuine: bool = True,
    allowed_root_origins: set[str] | None = None,
) -> dict[str, Any]:
    rejected: Counter[str] = Counter()
    candidates: list[dict[str, Any]] = []
    for record in records:
        accepted, reason = is_candidate_record(
            record,
            target_tile=target_tile,
            horizon=horizon,
            require_genuine=require_genuine,
            allowed_root_origins=allowed_root_origins,
        )
        if accepted:
            candidates.append(record)
        else:
            rejected[reason] += 1

    selected = root_cap_candidates(candidates, horizon=horizon)
    family_share = _family_share(selected)
    behavior_families = len({str(record.get("phase0_behavior_family")) for record in selected})
    outcome_counts = _counts(selected, "outcome")
    diversity_checks = {
        "min_roots": len(selected) >= int(min_roots),
        "min_behavior_families": behavior_families >= int(min_behavior_families),
        "max_family_share": float(family_share["share"]) <= float(max_family_share) if selected else False,
        "success_controls_present": int(outcome_counts.get("success", 0)) >= int(min_roots_per_outcome),
        "failure_controls_present": int(outcome_counts.get("failure", 0)) >= int(min_roots_per_outcome),
    }
    corpus_ready = all(diversity_checks.values())
    return {
        "version": 1,
        "kind": "phase0_oracle_corpus_audit",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "mode": "dry_run_read_only",
        "source_paths": source_paths,
        "source_files": len(source_paths),
        "source_records": len(records),
        "target_tile": int(target_tile),
        "horizon": int(horizon),
        "candidate_records": len(candidates),
        "root_capped_records": len(selected),
        "unique_ancestries": len({str(record.get("phase0_ancestry_key")) for record in selected}),
        "behavior_families": behavior_families,
        "largest_behavior_family": family_share,
        "by_outcome": outcome_counts,
        "by_root_origin": _counts(selected, "root_origin"),
        "by_behavior_family": _counts(selected, "phase0_behavior_family"),
        "by_behavior_family_outcome": _nested_counts(selected, "phase0_behavior_family", "outcome"),
        "by_root_policy_family": _counts(selected, "root_policy_family"),
        "by_stratum": _counts(selected, "stratum"),
        "by_offset_bucket": _offset_buckets(selected),
        "selected_records_preview": [
            {
                "id": record.get("id"),
                "outcome": record.get("outcome"),
                "offset": record.get("phase0_outcome_offset"),
                "root_seed": record.get("root_seed"),
                "root_origin": record.get("root_origin"),
                "behavior_family": record.get("phase0_behavior_family"),
                "root_policy_family": record.get("root_policy_family"),
                "source_replay": record.get("source_replay"),
                "source_next_action": record.get("source_next_action"),
                "score_minus_starter": record.get("score_minus_starter"),
                "raw_count_768": record.get("raw_count_768"),
                "raw_count_1536": record.get("raw_count_1536"),
            }
            for record in selected[:100]
        ],
        "diversity_requirements": _gate_rules(
            min_roots=min_roots,
            min_behavior_families=min_behavior_families,
            max_family_share=max_family_share,
            min_roots_per_outcome=min_roots_per_outcome,
        ),
        "diversity_checks": diversity_checks,
        "corpus_ready_for_rollout_gate": bool(corpus_ready),
        "margin_and_power_rationale": _margin_rationale(),
        "behavior_family_rules": BEHAVIOR_FAMILY_RULES,
        "rejected": dict(rejected),
    }


def write_html(path: Path, payload: dict[str, Any]) -> None:
    def cell(value: object) -> str:
        return escape(str(value))

    checks = payload.get("diversity_checks", {})
    rows = []
    if isinstance(checks, dict):
        for key, value in checks.items():
            rows.append(f"<tr><td>{cell(key)}</td><td>{cell(value)}</td></tr>")
    preview_rows = []
    for record in payload.get("selected_records_preview", []):
        if not isinstance(record, dict):
            continue
        preview_rows.append(
            "<tr>"
            f"<td>{cell(record.get('outcome'))}</td>"
            f"<td>{cell(record.get('offset'))}</td>"
            f"<td>{cell(record.get('root_seed'))}</td>"
            f"<td>{cell(record.get('behavior_family'))}</td>"
            f"<td>{cell(record.get('score_minus_starter'))}</td>"
            f"<td>{cell(record.get('source_next_action'))}</td>"
            f"<td>{cell(record.get('source_replay'))}</td>"
            "</tr>"
        )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Phase-0 Oracle Corpus Audit</title>
  <style>
    :root {{ color-scheme: dark; --bg:#101318; --panel:#171d24; --line:#34404d; --text:#edf2f7; --muted:#aab6c2; --gold:#f2c14e; }}
    body {{ margin:0; background:var(--bg); color:var(--text); font-family:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ width:min(1180px, calc(100vw - 32px)); margin:0 auto; padding:24px 0 40px; }}
    h1 {{ margin:0 0 8px; font-size:24px; }}
    .muted {{ color:var(--muted); }}
    .cards {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(160px, 1fr)); gap:10px; margin:18px 0; }}
    .card {{ border:1px solid var(--line); background:var(--panel); border-radius:8px; padding:12px; }}
    .label {{ color:var(--muted); font-size:11px; text-transform:uppercase; }}
    .value {{ margin-top:4px; color:var(--gold); font-size:22px; font-weight:800; }}
    table {{ width:100%; border-collapse:collapse; font-size:12px; font-variant-numeric:tabular-nums; margin:12px 0 20px; }}
    th, td {{ border-bottom:1px solid var(--line); padding:7px 8px; text-align:right; vertical-align:top; }}
    th:first-child, td:first-child, th:nth-child(4), td:nth-child(4), th:last-child, td:last-child {{ text-align:left; }}
    td:last-child {{ max-width:420px; overflow-wrap:anywhere; color:var(--muted); }}
    pre {{ white-space:pre-wrap; overflow-wrap:anywhere; color:var(--muted); }}
  </style>
</head>
<body>
  <main>
    <h1>Phase-0 Oracle Corpus Audit</h1>
    <p class="muted">Dry-run only: existing records are counted, no rollout labels/search/training are executed.</p>
    <section class="cards">
      <div class="card"><div class="label">Ready</div><div class="value">{cell(payload.get('corpus_ready_for_rollout_gate'))}</div></div>
      <div class="card"><div class="label">Root-Capped Records</div><div class="value">{cell(payload.get('root_capped_records'))}</div></div>
      <div class="card"><div class="label">Behavior Families</div><div class="value">{cell(payload.get('behavior_families'))}</div></div>
      <div class="card"><div class="label">Largest Family Share</div><div class="value">{cell(round(float((payload.get('largest_behavior_family') or {}).get('share') or 0), 3))}</div></div>
    </section>
    <h2>Diversity Checks</h2>
    <table><thead><tr><th>Check</th><th>Passed</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
    <h2>Selected Root Preview</h2>
    <table><thead><tr><th>Outcome</th><th>Offset</th><th>Root</th><th>Behavior Family</th><th>Score - Starter</th><th>Recorded Action</th><th>Replay</th></tr></thead><tbody>{''.join(preview_rows)}</tbody></table>
    <h2>Summary JSON</h2>
    <pre>{escape(json.dumps(payload, indent=2, sort_keys=True))}</pre>
  </main>
</body>
</html>
"""
    path.write_text(html)


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    paths = _flatten_path_groups(args.records_json) + _glob_records(args.record_glob)
    source_records = load_records(paths)
    allowed = None
    if args.root_origin.strip().lower() != "all":
        allowed = {part.strip() for part in args.root_origin.split(",") if part.strip()}
    payload = audit_records(
        source_records,
        source_paths=[str(path) for path in paths],
        target_tile=args.target_tile,
        horizon=args.horizon,
        min_roots=args.min_roots,
        min_behavior_families=args.min_behavior_families,
        max_family_share=args.max_family_share,
        min_roots_per_outcome=args.min_roots_per_outcome,
        require_genuine=not bool(args.allow_non_genuine),
        allowed_root_origins=allowed,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload["json"] = str(args.out_dir / "phase0_oracle_corpus_audit.json")
    payload["summary_json"] = str(args.out_dir / "summary.json")
    payload["html"] = str(args.out_dir / "phase0_oracle_corpus_audit.html")
    write_json(args.out_dir / "phase0_oracle_corpus_audit.json", payload)
    write_json(args.out_dir / "summary.json", {key: value for key, value in payload.items() if key != "selected_records_preview"})
    write_html(args.out_dir / "phase0_oracle_corpus_audit.html", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-json", type=Path, nargs="+", action="append", default=[])
    parser.add_argument("--record-glob", action="append", default=list(DEFAULT_RECORD_GLOBS))
    parser.add_argument("--target-tile", type=int, default=1536)
    parser.add_argument("--horizon", type=int, default=40)
    parser.add_argument("--min-roots", type=int, default=20)
    parser.add_argument("--min-behavior-families", type=int, default=2)
    parser.add_argument("--max-family-share", type=float, default=0.5)
    parser.add_argument("--min-roots-per-outcome", type=int, default=4)
    parser.add_argument("--root-origin", default="fresh,human")
    parser.add_argument("--allow-non-genuine", action="store_true")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("threes_rl/runs/forensics/phase0_oracle_corpus_audit/latest"),
    )
    args = parser.parse_args()
    payload = run_from_args(args)
    compact = {
        "corpus_ready_for_rollout_gate": payload["corpus_ready_for_rollout_gate"],
        "root_capped_records": payload["root_capped_records"],
        "behavior_families": payload["behavior_families"],
        "largest_behavior_family": payload["largest_behavior_family"],
        "by_outcome": payload["by_outcome"],
        "json": payload["json"],
        "html": payload["html"],
    }
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
