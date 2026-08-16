"""Resample-stability audit for saved swing-label continuation labels."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

import numpy as np

from threes_rl.run_artifacts import write_json


@dataclass
class StabilityCase:
    id: str
    source_json: str
    seed: int | None
    move_count: int | None
    stratum: str
    base_action: str
    actions: list[str]
    original_stable: bool
    original_winner: str | None
    original_regret: float
    full_winners: dict[str, str]
    full_mean_diffs: dict[str, float]
    bootstrap_winner_fractions: dict[str, float]
    horizon_winner_consistent: bool
    min_bootstrap_winner_fraction: float
    robust: bool
    robust_flip: bool
    max_horizon: int


def _flatten_paths(path_groups: list[list[Path]] | None) -> list[Path]:
    if not path_groups:
        return []
    return [path for group in path_groups for path in group]


def _safe_int(value: object) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _actions(item: dict[str, Any], label: dict[str, Any]) -> list[str]:
    raw = label.get("actions")
    if not isinstance(raw, list) or len(raw) < 2:
        raw = item.get("top_two_actions")
    actions = [str(action) for action in raw or [] if action is not None]
    return actions[:2]


def _winner(first: str, second: str, diff: float) -> str:
    if diff > 0.0:
        return first
    if diff < 0.0:
        return second
    return "tie"


def _paired_diffs(by_action: dict[str, Any], first: str, second: str, horizon: int) -> list[float]:
    left = by_action.get(first, {})
    right = by_action.get(second, {})
    if not isinstance(left, dict) or not isinstance(right, dict):
        return []
    left_values = left.get(str(int(horizon)), [])
    right_values = right.get(str(int(horizon)), [])
    if not isinstance(left_values, list) or not isinstance(right_values, list):
        return []
    count = min(len(left_values), len(right_values))
    return [float(left_values[idx]) - float(right_values[idx]) for idx in range(count)]


def _bootstrap_fraction(
    diffs: list[float],
    *,
    expected_winner: str,
    first: str,
    second: str,
    resamples: int,
    rng: np.random.Generator,
) -> float:
    if not diffs or expected_winner == "tie":
        return 0.5
    arr = np.asarray(diffs, dtype=np.float64)
    kept = 0
    for _ in range(int(resamples)):
        sample = arr[rng.integers(0, len(arr), size=len(arr))]
        if _winner(first, second, float(sample.mean())) == expected_winner:
            kept += 1
    return kept / float(resamples)


def case_from_item(
    item: dict[str, Any],
    *,
    source_json: Path,
    threshold: float,
    resamples: int,
    rng: np.random.Generator,
) -> StabilityCase | None:
    label = item.get("label")
    if not isinstance(label, dict):
        return None
    actions = _actions(item, label)
    if len(actions) < 2:
        return None
    by_action = label.get("by_action")
    if not isinstance(by_action, dict):
        return None
    horizons = [int(value) for value in label.get("horizons", []) if str(value).strip()]
    if not horizons:
        return None
    first, second = actions[:2]
    full_winners: dict[str, str] = {}
    full_mean_diffs: dict[str, float] = {}
    bootstrap: dict[str, float] = {}
    for horizon in horizons:
        diffs = _paired_diffs(by_action, first, second, horizon)
        if not diffs:
            continue
        mean_diff = float(mean(diffs))
        winner = _winner(first, second, mean_diff)
        full_winners[str(int(horizon))] = winner
        full_mean_diffs[str(int(horizon))] = mean_diff
        bootstrap[str(int(horizon))] = _bootstrap_fraction(
            diffs,
            expected_winner=winner,
            first=first,
            second=second,
            resamples=resamples,
            rng=rng,
        )
    if not full_winners:
        return None
    non_tie = [winner for winner in full_winners.values() if winner != "tie"]
    horizon_consistent = len(non_tie) == len(full_winners) and len(set(non_tie)) == 1
    min_fraction = min(bootstrap.values()) if bootstrap else 0.0
    robust = bool(horizon_consistent and min_fraction >= float(threshold))
    robust_winner = non_tie[0] if robust and non_tie else None
    base_action = str(item.get("base_action") or first)
    features = item.get("features") if isinstance(item.get("features"), dict) else {}
    original_winner = label.get("stable_winner") or label.get("oracle_winner")
    original_winner_text = None if original_winner is None or str(original_winner) == "tie" else str(original_winner)
    return StabilityCase(
        id=str(item.get("id", "")),
        source_json=str(source_json),
        seed=_safe_int(item.get("seed")),
        move_count=_safe_int(item.get("move_count")),
        stratum=str(features.get("stratum", "unknown")),
        base_action=base_action,
        actions=[first, second],
        original_stable=bool(label.get("stable")),
        original_winner=original_winner_text,
        original_regret=_safe_float(label.get("oracle_regret_at_max_horizon"), 0.0),
        full_winners=full_winners,
        full_mean_diffs=full_mean_diffs,
        bootstrap_winner_fractions=bootstrap,
        horizon_winner_consistent=horizon_consistent,
        min_bootstrap_winner_fraction=float(min_fraction),
        robust=robust,
        robust_flip=bool(robust_winner is not None and robust_winner != base_action),
        max_horizon=max(int(horizon) for horizon in full_winners),
    )


def load_cases(paths: Iterable[Path], *, threshold: float, resamples: int, seed: int) -> list[StabilityCase]:
    rng = np.random.default_rng(int(seed))
    cases: list[StabilityCase] = []
    for path in paths:
        payload = json.loads(Path(path).read_text())
        labels = payload.get("labels", [])
        if not isinstance(labels, list):
            continue
        for item in labels:
            if not isinstance(item, dict):
                continue
            case = case_from_item(
                item,
                source_json=Path(path),
                threshold=threshold,
                resamples=resamples,
                rng=rng,
            )
            if case is not None:
                cases.append(case)
    return cases


def load_selected_samples(
    paths: Iterable[Path],
    *,
    threshold: float,
    resamples: int,
    seed: int,
    robust_flips_only: bool = True,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    samples: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(Path(path).read_text())
        labels = payload.get("labels", [])
        if not isinstance(labels, list):
            continue
        for item in labels:
            if not isinstance(item, dict):
                continue
            case = case_from_item(
                item,
                source_json=Path(path),
                threshold=threshold,
                resamples=resamples,
                rng=rng,
            )
            if case is None:
                continue
            if robust_flips_only and not case.robust_flip:
                continue
            sample = dict(item)
            sample["previous_label"] = sample.pop("label", None)
            sample["stability_audit"] = asdict(case)
            sample["source_label_json"] = str(path)
            samples.append(sample)
    return samples


def summarize_cases(cases: list[StabilityCase]) -> dict[str, Any]:
    robust = [case for case in cases if case.robust]
    robust_flips = [case for case in cases if case.robust_flip]
    original_stable = [case for case in cases if case.original_stable]
    original_flips = [
        case
        for case in cases
        if case.original_stable and case.original_winner is not None and case.original_winner != case.base_action
    ]
    regrets = [case.original_regret for case in robust_flips if case.original_regret > 0.0]
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "cases": len(cases),
        "original_stable": len(original_stable),
        "original_stable_rate": len(original_stable) / len(cases) if cases else 0.0,
        "original_stable_flips": len(original_flips),
        "robust": len(robust),
        "robust_rate": len(robust) / len(cases) if cases else 0.0,
        "robust_flips": len(robust_flips),
        "robust_flip_rate": len(robust_flips) / len(cases) if cases else 0.0,
        "horizon_consistent": sum(1 for case in cases if case.horizon_winner_consistent),
        "strata": dict(Counter(case.stratum for case in cases)),
        "robust_flip_regret": {
            "count": len(regrets),
            "mean": float(mean(regrets)) if regrets else 0.0,
            "median": float(median(regrets)) if regrets else 0.0,
            "max": float(max(regrets)) if regrets else 0.0,
        },
        "by_source": dict(Counter(case.source_json for case in cases)),
    }


def _sample_stratum(sample: dict[str, Any]) -> str:
    features = sample.get("features")
    if not isinstance(features, dict):
        return "unknown"
    return str(features.get("stratum", "unknown"))


def write_html(path: Path, payload: dict[str, Any]) -> None:
    summary = payload.get("summary", {})
    cases = payload.get("cases", [])

    def cell(value: object) -> str:
        return escape(str(value))

    rows = []
    for case in cases[:300] if isinstance(cases, list) else []:
        rows.append(
            "<tr>"
            f"<td>{cell(case.get('robust'))}</td>"
            f"<td>{cell(case.get('robust_flip'))}</td>"
            f"<td>{cell(case.get('id'))}</td>"
            f"<td>{cell(case.get('stratum'))}</td>"
            f"<td>{cell(case.get('base_action'))}</td>"
            f"<td>{cell(case.get('actions'))}</td>"
            f"<td>{cell(case.get('full_winners'))}</td>"
            f"<td>{float(case.get('min_bootstrap_winner_fraction') or 0.0):.3f}</td>"
            f"<td>{float(case.get('original_regret') or 0.0):.1f}</td>"
            f"<td>{cell(case.get('source_json'))}</td>"
            "</tr>"
        )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Threes Label Stability Audit</title>
  <style>
    :root {{ color-scheme: dark; --bg: #101318; --panel: #171d24; --line: #34404d; --text: #edf2f7; --muted: #aab6c2; }}
    body {{ margin: 0; background: var(--bg); color: var(--text); font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ width: min(1240px, calc(100vw - 32px)); margin: 0 auto; padding: 24px 0 40px; }}
    h1 {{ margin: 0 0 8px; font-size: 24px; }}
    .muted {{ color: var(--muted); }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 18px 0; }}
    .card {{ border: 1px solid var(--line); background: var(--panel); border-radius: 8px; padding: 12px; }}
    .label {{ color: var(--muted); font-size: 11px; text-transform: uppercase; }}
    .value {{ margin-top: 4px; font-size: 22px; font-weight: 800; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 7px 8px; text-align: right; vertical-align: top; }}
    th:nth-child(3), td:nth-child(3), th:nth-child(4), td:nth-child(4), th:nth-child(10), td:nth-child(10) {{ text-align: left; }}
    td:nth-child(10) {{ max-width: 340px; overflow-wrap: anywhere; color: var(--muted); }}
    pre {{ white-space: pre-wrap; overflow-wrap: anywhere; color: var(--muted); }}
  </style>
</head>
<body>
  <main>
    <h1>Label Stability Audit</h1>
    <p class="muted">Bootstrap resampling of saved paired continuation labels.</p>
    <section class="cards">
      <div class="card"><div class="label">Cases</div><div class="value">{cell(summary.get('cases', 0))}</div></div>
      <div class="card"><div class="label">Original Stable</div><div class="value">{cell(summary.get('original_stable', 0))}</div></div>
      <div class="card"><div class="label">Robust</div><div class="value">{cell(summary.get('robust', 0))}</div></div>
      <div class="card"><div class="label">Robust Flips</div><div class="value">{cell(summary.get('robust_flips', 0))}</div></div>
    </section>
    <table><thead><tr><th>Robust</th><th>Flip</th><th>ID</th><th>Stratum</th><th>Base</th><th>Actions</th><th>Winners</th><th>Min Boot</th><th>Regret</th><th>Source</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
    <h2>Summary JSON</h2>
    <pre>{escape(json.dumps(summary, indent=2, sort_keys=True))}</pre>
  </main>
</body>
</html>
"""
    path.write_text(html)


def run(
    paths: Iterable[Path],
    out_dir: Path,
    *,
    threshold: float = 0.70,
    resamples: int = 1000,
    seed: int = 20260707,
) -> dict[str, Any]:
    path_list = [Path(path) for path in paths]
    cases = load_cases(path_list, threshold=threshold, resamples=resamples, seed=seed)
    robust_flip_samples = load_selected_samples(
        path_list,
        threshold=threshold,
        resamples=resamples,
        seed=seed,
        robust_flips_only=True,
    )
    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "threshold": float(threshold),
        "resamples": int(resamples),
        "seed": int(seed),
        "source_json": [str(path) for path in path_list],
        "summary": summarize_cases(cases),
        "cases": [asdict(case) for case in cases],
        "robust_flip_sample_count": len(robust_flip_samples),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "label_stability_audit.json", payload)
    write_json(
        out_dir / "robust_flip_samples.json",
        {
            "created_at": payload["created_at"],
            "kind": "swing_label_sample_export",
            "selection": "robust_flip",
            "threshold": float(threshold),
            "resamples": int(resamples),
            "seed": int(seed),
            "source_json": [str(path) for path in path_list],
            "samples": robust_flip_samples,
            "summary": {
                "accepted_samples": len(robust_flip_samples),
                "strata": dict(Counter(_sample_stratum(sample) for sample in robust_flip_samples)),
                "scan_stats": {
                    "source": "label_stability_audit",
                    "sample_mode": "robust-flip-export",
                    "accepted_samples": len(robust_flip_samples),
                    "strata": dict(Counter(_sample_stratum(sample) for sample in robust_flip_samples)),
                    "rejected": {},
                },
            },
        },
    )
    write_html(out_dir / "label_stability_audit.html", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--swing-label-json", type=Path, nargs="+", action="append", required=True)
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--out-dir", type=Path, default=Path("threes_rl/runs/forensics/label_stability/latest"))
    args = parser.parse_args()
    payload = run(
        _flatten_paths(args.swing_label_json),
        args.out_dir,
        threshold=args.threshold,
        resamples=args.resamples,
        seed=args.seed,
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"json={args.out_dir / 'label_stability_audit.json'}")
    print(f"html={args.out_dir / 'label_stability_audit.html'}")


if __name__ == "__main__":
    main()
