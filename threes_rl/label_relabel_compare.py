"""Compare independent relabels against previous swing-label decisions."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path
from statistics import mean, median
from typing import Any

from threes_rl.run_artifacts import write_json


@dataclass
class RelabelComparison:
    id: str
    seed: int | None
    move_count: int | None
    stratum: str
    base_action: str
    top_two_actions: list[str]
    previous_stable: bool
    previous_winner: str | None
    previous_regret: float
    previous_flip: bool
    new_stable: bool
    new_winner: str | None
    new_regret: float
    new_flip: bool
    retained_stable_flip: bool
    changed_winner: bool
    stable_reversed_to_base: bool


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


def _label_winner(label: dict[str, Any]) -> str | None:
    winner = label.get("stable_winner") or label.get("oracle_winner")
    if winner is None or str(winner) == "tie":
        return None
    return str(winner)


def _label_stable(label: dict[str, Any]) -> bool:
    return bool(label.get("stable"))


def _stratum(item: dict[str, Any]) -> str:
    features = item.get("features")
    if not isinstance(features, dict):
        return "unknown"
    return str(features.get("stratum", "unknown"))


def comparison_from_item(item: dict[str, Any]) -> RelabelComparison | None:
    previous = item.get("previous_label")
    new = item.get("label")
    if not isinstance(previous, dict) or not isinstance(new, dict):
        return None
    base_action = str(item.get("base_action") or "")
    top_two = [str(action) for action in item.get("top_two_actions", []) if action is not None]
    previous_winner = _label_winner(previous)
    new_winner = _label_winner(new)
    previous_stable = _label_stable(previous)
    new_stable = _label_stable(new)
    previous_flip = bool(previous_stable and previous_winner is not None and previous_winner != base_action)
    new_flip = bool(new_stable and new_winner is not None and new_winner != base_action)
    return RelabelComparison(
        id=str(item.get("id", "")),
        seed=_safe_int(item.get("seed")),
        move_count=_safe_int(item.get("move_count")),
        stratum=_stratum(item),
        base_action=base_action,
        top_two_actions=top_two,
        previous_stable=previous_stable,
        previous_winner=previous_winner,
        previous_regret=_safe_float(previous.get("oracle_regret_at_max_horizon"), 0.0),
        previous_flip=previous_flip,
        new_stable=new_stable,
        new_winner=new_winner,
        new_regret=_safe_float(new.get("oracle_regret_at_max_horizon"), 0.0),
        new_flip=new_flip,
        retained_stable_flip=bool(previous_flip and new_flip and previous_winner == new_winner),
        changed_winner=bool(previous_winner != new_winner),
        stable_reversed_to_base=bool(previous_flip and new_stable and new_winner == base_action),
    )


def load_comparisons(path: Path) -> list[RelabelComparison]:
    payload = json.loads(path.read_text())
    labels = payload.get("labels", [])
    if not isinstance(labels, list):
        raise ValueError(f"{path} does not contain a labels list")
    comparisons: list[RelabelComparison] = []
    for item in labels:
        if not isinstance(item, dict):
            continue
        comparison = comparison_from_item(item)
        if comparison is not None:
            comparisons.append(comparison)
    return comparisons


def summarize(comparisons: list[RelabelComparison]) -> dict[str, Any]:
    previous_flips = [row for row in comparisons if row.previous_flip]
    new_flips = [row for row in comparisons if row.new_flip]
    retained = [row for row in comparisons if row.retained_stable_flip]
    reversed_to_base = [row for row in comparisons if row.stable_reversed_to_base]
    new_stable = [row for row in comparisons if row.new_stable]
    new_regrets = [row.new_regret for row in comparisons if row.new_regret > 0.0]
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "cases": len(comparisons),
        "previous_stable_flips": len(previous_flips),
        "new_stable_flips": len(new_flips),
        "retained_stable_flips": len(retained),
        "retained_stable_flip_rate": len(retained) / len(previous_flips) if previous_flips else 0.0,
        "stable_reversed_to_base": len(reversed_to_base),
        "new_stable_labels": len(new_stable),
        "new_stable_label_rate": len(new_stable) / len(comparisons) if comparisons else 0.0,
        "changed_winners": sum(1 for row in comparisons if row.changed_winner),
        "new_positive_regrets": len(new_regrets),
        "new_positive_regret_mean": float(mean(new_regrets)) if new_regrets else 0.0,
        "new_positive_regret_median": float(median(new_regrets)) if new_regrets else 0.0,
        "new_positive_regret_max": float(max(new_regrets)) if new_regrets else 0.0,
        "strata": dict(Counter(row.stratum for row in comparisons)),
    }


def write_html(path: Path, payload: dict[str, Any]) -> None:
    summary = payload.get("summary", {})
    rows = payload.get("comparisons", [])

    def cell(value: object) -> str:
        return escape(str(value))

    table_rows = []
    for row in rows if isinstance(rows, list) else []:
        table_rows.append(
            "<tr>"
            f"<td>{cell(row.get('retained_stable_flip'))}</td>"
            f"<td>{cell(row.get('id'))}</td>"
            f"<td>{cell(row.get('stratum'))}</td>"
            f"<td>{cell(row.get('base_action'))}</td>"
            f"<td>{cell(row.get('top_two_actions'))}</td>"
            f"<td>{cell(row.get('previous_stable'))}</td>"
            f"<td>{cell(row.get('previous_winner'))}</td>"
            f"<td>{float(row.get('previous_regret') or 0.0):.1f}</td>"
            f"<td>{cell(row.get('new_stable'))}</td>"
            f"<td>{cell(row.get('new_winner'))}</td>"
            f"<td>{float(row.get('new_regret') or 0.0):.1f}</td>"
            f"<td>{cell(row.get('stable_reversed_to_base'))}</td>"
            "</tr>"
        )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Threes Relabel Comparison</title>
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
    th:nth-child(2), td:nth-child(2), th:nth-child(3), td:nth-child(3) {{ text-align: left; }}
    td:nth-child(2) {{ max-width: 280px; overflow-wrap: anywhere; }}
    pre {{ white-space: pre-wrap; overflow-wrap: anywhere; color: var(--muted); }}
  </style>
</head>
<body>
  <main>
    <h1>Relabel Comparison</h1>
    <p class="muted">Independent continuation-label retention for previously selected samples.</p>
    <section class="cards">
      <div class="card"><div class="label">Cases</div><div class="value">{cell(summary.get('cases', 0))}</div></div>
      <div class="card"><div class="label">Previous Flips</div><div class="value">{cell(summary.get('previous_stable_flips', 0))}</div></div>
      <div class="card"><div class="label">Retained Flips</div><div class="value">{cell(summary.get('retained_stable_flips', 0))}</div></div>
      <div class="card"><div class="label">New Stable</div><div class="value">{cell(summary.get('new_stable_labels', 0))}</div></div>
    </section>
    <table><thead><tr><th>Retained</th><th>ID</th><th>Stratum</th><th>Base</th><th>Top Two</th><th>Prev Stable</th><th>Prev Winner</th><th>Prev Regret</th><th>New Stable</th><th>New Winner</th><th>New Regret</th><th>Back To Base</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table>
    <h2>Summary JSON</h2>
    <pre>{escape(json.dumps(summary, indent=2, sort_keys=True))}</pre>
  </main>
</body>
</html>
"""
    path.write_text(html)


def run(relabel_json: Path, out_dir: Path) -> dict[str, Any]:
    comparisons = load_comparisons(relabel_json)
    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "relabel_json": str(relabel_json),
        "summary": summarize(comparisons),
        "comparisons": [asdict(row) for row in comparisons],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "label_relabel_compare.json", payload)
    write_html(out_dir / "label_relabel_compare.html", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relabel-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("threes_rl/runs/forensics/label_stability/relabel_compare_latest"))
    args = parser.parse_args()
    payload = run(args.relabel_json, args.out_dir)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"json={args.out_dir / 'label_relabel_compare.json'}")
    print(f"html={args.out_dir / 'label_relabel_compare.html'}")


if __name__ == "__main__":
    main()
