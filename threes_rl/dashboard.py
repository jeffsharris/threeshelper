"""Build a local research dashboard from Threes RL run artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


RUNS_ROOT = Path("threes_rl/runs")
DEFAULT_OUT = RUNS_ROOT / "dashboard" / "index.html"
GLOBAL_TOP_REPLAY_LIMIT = 3


@dataclass
class Point:
    label: str
    path: str
    kind: str
    high_score: float
    high_score_minus_starter: float
    mean_score_minus_starter: float | None
    median_score_minus_starter: float | None
    p3072: float | None
    p6144: float | None
    games: int | None
    mtime: float
    created_at: str | None = None
    annotation: str | None = None
    replay: str | None = None
    record_eligible: bool = True
    record_eligibility_reason: str | None = None


def _float(payload: dict[str, Any], key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _top_replay(payload: dict[str, Any]) -> str | None:
    top = payload.get("top_games")
    if isinstance(top, list) and top:
        first = top[0]
        if isinstance(first, dict) and first.get("html"):
            return str(first["html"])
    return None


def _kind_for_path(path: Path) -> str:
    text = str(path)
    if "/eval_artifacts/" in text:
        return "eval"
    if path.parent.name.startswith("td_"):
        return "train"
    return "run"


def _skip_dashboard_source(path: Path) -> bool:
    return any(part in {"dashboard", "continuations"} or "_tmp" in part for part in path.parts)


def _skip_replay_start_training(path: Path) -> bool:
    config_path = path.parent / "config.json"
    if not path.parent.name.startswith("td_") or not config_path.exists():
        return False
    try:
        config = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    try:
        start_state_prob = float(config.get("start_state_prob") or 0.0)
    except (TypeError, ValueError):
        start_state_prob = 0.0
    return start_state_prob > 0.0 or bool(config.get("start_state_replays"))


def _record_eligibility(path: Path, payload: dict[str, Any]) -> tuple[bool, str | None]:
    """Return whether a run may change the normal-start record line."""
    explicit = payload.get("dashboard_record_eligible")
    if isinstance(explicit, bool):
        reason = payload.get("dashboard_record_eligibility_reason")
        return explicit, str(reason) if reason else None

    if _kind_for_path(path) != "eval":
        return False, "training or non-evaluation artifact"

    lock_path = path.parent / "confirmation_lock.json"
    if lock_path.exists():
        try:
            lock = json.loads(lock_path.read_text())
        except (OSError, json.JSONDecodeError):
            return False, "unreadable confirmation decision"
        decision = str(lock.get("decision") or "").upper()
        if decision not in {"PROMOTED", "PROMOTION_PASSED", "CONFIRMATION_PASSED_PROMOTED"}:
            return False, f"confirmation decision: {decision or 'missing'}"

    blocks = payload.get("blocks")
    if isinstance(blocks, list) and any(str(block).upper() in {"D0", "D1", "D2"} for block in blocks):
        return False, "development evaluation block"

    label = path.parent.name.lower()
    if "candidate" in label:
        return False, "unconfirmed candidate"
    if "diagnostic" in label or "failure_audit" in label:
        return False, "diagnostic evaluation"
    return True, None


def _timestamp_from_created_at(value: Any) -> float | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    candidates = [text]
    if text.endswith("Z"):
        candidates.append(f"{text[:-1]}+00:00")
    if len(text) >= 5 and text[-5] in ("+", "-") and text[-2:].isdigit():
        candidates.append(f"{text[:-5]}{text[-5:-2]}:{text[-2:]}")
    for candidate in candidates:
        try:
            return datetime.fromisoformat(candidate).timestamp()
        except ValueError:
            continue
    return None


def _point_time(path: Path, payload: dict[str, Any]) -> tuple[float, str | None]:
    created_at = payload.get("created_at")
    timestamp = _timestamp_from_created_at(created_at)
    return (
        timestamp if timestamp is not None else path.stat().st_mtime,
        created_at if isinstance(created_at, str) and created_at else None,
    )


def _annotation_for_label(label: str) -> str | None:
    rules = [
        ("corner2_1000_1200_fixed_starter", "Corrected starter and corner-aware expectimax baseline."),
        ("td_default_expected_500", "Pure n-tuple TD self-play stabilized, but no non-starter 1536 yet."),
        ("corner2_mc_50", "First corner2 Monte Carlo bootstrap."),
        ("corner2_mc_200_a001_1000_1200_full", "Lower-alpha MC tied corner2 over the full 200-seed suite."),
        ("corner2_mc_1000_a0005_1000_1050", "MC-1000 produced the first strong learned-value high-score jump."),
        ("corner2_mc_1000_a0005_1050_1100", "100-seed paired gate extension for decision hygiene."),
        ("student1_nstep_tc_50", "Student-as-actor n-step/TC improved median behavior but lost high-tail events."),
        ("student1_nstep_tc_25", "Midpoint checkpoint did not preserve the parent tail."),
        ("student2_blend_w025", "Expert-iteration student trained from blended-search trajectories with late-game replay starts."),
        ("student4_best_actor_normal", "Normal-start expert iteration from the current best actor; no replay-start reservoir."),
        ("student5_replaycal_actor", "Expert-iteration student trained from the replay-calibrated current-best actor plus high-board replay starts."),
        ("parent_mc1000_student2_w025", "Blended parent evaluator with the student2 value sidecar."),
        ("multiblend_parent_mc1000_student1_w020_student2_w010", "Three-way evaluator blend: parent plus student1 high-tail signal plus student2 median-shape signal."),
        ("maxblend_parent_mc1000_student1_student2", "Max-ensemble leaf value over parent, student1, and student2 tables."),
        ("tiebreak_parent_mc1000_student50_w025_m5", "Near-tie up/left action prior on the current best blended evaluator."),
        ("parent_mc1000_student4_w010", "Low-weight sidecar blend using the normal-start student4 value."),
        ("parent_mc1000_student50_w010", "Lower sidecar weight sweep for the student1 blended evaluator."),
        ("parent_mc1000_student50_w040", "Higher sidecar weight sweep for the student1 blended evaluator."),
        ("td_phase4_late_reservoir", "Late-start replay reservoir training; training episode scores are not directly comparable to full-game starts."),
        ("td_phase4_protected_late_endgame", "Protected late/endgame-only phase4 training with replay starts; training scores are not directly comparable to full-game starts."),
        ("replay_cal_phase4_late_midlate", "Offline replay-return calibration from retained high-score trajectories; not a played-game score."),
        ("replay_cal_big6_midlate_top24", "Big6 replay-return calibration from top trajectories; useful scale probe, but the sidecar screen was not promoted."),
        ("ntuple_expectimax2_phase4_late_reservoir", "Full-game eval of the late-start-trained phase4 checkpoint."),
        ("ntuple_expectimax2_phase4_protected_late_endgame", "Full-game eval of the protected late/endgame phase4 continuation."),
        ("ntuple_phaseblend_parent_student1_w025_replaycal", "Incumbent plus a small mid-game replay-calibrated sidecar."),
        ("ntuple_phaseblend_labelcorr_w010_endgame", "Endgame action-label correction sidecar; sparse 100-seed tail lift with unchanged median/P(3072), but no full-game 6144 yet."),
        ("anchor_guard_current_best", "Hard top-left anchor guard: prevented dislodging 1536+, but killed key incumbent tails and was not promoted."),
        ("anchor_penalty5000_current_best", "Soft top-left anchor penalty: rescued one 3072 seed, but still lost key incumbent tails and was not promoted."),
        ("action_prior_medium_prob075", "Action-conditioned medium-risk prior: lifted selected 3072 continuations but tied/slightly lost on the 50-seed full-game screen."),
        ("action_prior_mid384_highrisk", "Mid-game high-risk action prior: improved sampled mid-state continuations but lost on the full-game screen."),
        ("ntuple_additive_big6lazy_labelcorr", "Big6 lazy label-correction sidecar: median ticked up on the first screen, but mean/high-tail/P(3072) lost to the incumbent."),
        ("phase4corner3_parentinit_currentbest_reservoir", "Phase/corner-risk expert-iteration value fit: strong replay-start training signal, but weak normal-start transfer and killed."),
        ("ntuple_phaseblend_replaycal_w005_student5_w005", "Screen of the current best plus a tiny student5 sidecar; changed only a few seeds and was not promoted."),
        ("ntuple_phaseblend_replaycal_w005_big6cal_w002", "Big6 replay-calibrated sidecar screen; damaged key tails and was killed."),
        ("ntuple_phaseblend_parent_student1_w025_phase4late", "Phase-gated late-reservoir sidecar on top of the incumbent evaluator."),
        ("current_best_from_3072", "Continuation benchmark from sampled 3072 replay states; no 6144 conversions."),
        ("corner2_from_3072", "Corner2 continuation benchmark from sampled 3072 replay states; no 6144 conversions."),
        ("expectimax2a", "Adaptive exact depth was too slow for broad sweeps."),
        ("expectimax2b", "Budgeted adaptive search was faster but weaker than depth-2."),
        (
            "ntuple_phaseblend_incumbent_tailhunt_1550_1600_keep20",
            "Fresh-root tail-hunt replay collection: three 3072 games, no 6144; useful data, not a promoted policy.",
        ),
        (
            "ntuple_phaseblend_incumbent_tailhunt_1600_1620_keep3_parallel",
            "Parallel full-incumbent replay collection using --jobs 4; no non-starter 1536/3072 in this slice.",
        ),
        (
            "ntuple_phaseblend_incumbent_tailhunt_1620_1720_keep3_parallel",
            "Parallel 100-seed tail-hunt: two 3072 games and one recoverable pre-3072 duplicate-1536 failure pocket; diagnostic data, not a promotion.",
        ),
        (
            "ntuple_phaseblend_incumbent_tailhunt_1720_1820_keep3_parallel",
            "Parallel 100-seed tail-hunt: three 3072 games, but no new non-near duplicate-1536 failure pocket.",
        ),
        (
            "ntuple_phaseblend_incumbent_tailhunt_1820_1920_keep3_diagfail3_parallel",
            "Parallel 100-seed tail-hunt with diagnostic pre-3072 failure retention: four 3072 games, no pre-3072 1536 failures, and no 6144.",
        ),
        (
            "ntuple_phaseblend_incumbent_tailhunt_1920_2020_keep3_diagfail768x5_parallel",
            "Parallel replay collection with widened 768-band failure retention: three 3072 games plus five first-1536 failure-control replays.",
        ),
        (
            "ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120",
            "Fresh-root near-failure acquisition: seven 1536/3072 games plus thirty 768-band controls; diagnostic data only, no 6144 or policy promotion.",
        ),
        (
            "ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220",
            "Second fresh-root near-failure acquisition: five 1536/3072 games plus thirty 768-band controls; expanded frontier replication stayed root-concentrated.",
        ),
        (
            "ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320",
            "Third fresh-root near-failure acquisition: three 1536/3072 games plus thirty controls; 90-root frontier still failed root-diverse replication.",
        ),
        (
            "ntuple_phaseblend1b_incumbent_tailhunt_1600_1620_keep10",
            "Cheap 1b replay collection: one 3072 root, but no selector-ready non-near duplicate-1536 states.",
        ),
    ]
    for needle, text in rules:
        if needle in label:
            return text
    return None


def _point_from_summary(path: Path) -> Point | None:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    high = _float(payload, "high_score")
    if high is None:
        return None
    minus = _float(payload, "high_score_minus_starter")
    label = path.parent.name
    point_time, created_at = _point_time(path, payload)
    record_eligible, record_eligibility_reason = _record_eligibility(path, payload)
    return Point(
        label=label,
        path=str(path.parent),
        kind=_kind_for_path(path),
        high_score=high,
        high_score_minus_starter=minus if minus is not None else high,
        mean_score_minus_starter=_float(payload, "mean_score_minus_starter"),
        median_score_minus_starter=_float(payload, "median_score_minus_starter"),
        p3072=_float(payload, "p_max_tile_excl_starter_ge_3072"),
        p6144=_float(payload, "p_max_tile_excl_starter_ge_6144"),
        games=int(payload["games"]) if payload.get("games") is not None else None,
        mtime=point_time,
        created_at=created_at,
        annotation=_annotation_for_label(label),
        replay=_top_replay(payload),
        record_eligible=record_eligible,
        record_eligibility_reason=record_eligibility_reason,
    )


def _points_from_progress(path: Path) -> list[Point]:
    points: list[Point] = []
    try:
        with path.open() as fh:
            rows = list(csv.DictReader(fh))
    except OSError:
        return points
    if not rows:
        return points
    label = path.parent.name
    path_mtime = path.stat().st_mtime
    final_elapsed = _float(rows[-1], "elapsed_s") or 0.0
    for row in rows[-12:]:
        high = _float(row, "high_score")
        if high is None:
            continue
        games = int(float(row["games"])) if row.get("games") else None
        minus = _float(row, "high_score_minus_starter")
        created_at = row.get("created_at") or None
        timestamp = _timestamp_from_created_at(created_at)
        if timestamp is None:
            elapsed = _float(row, "elapsed_s")
            timestamp = path_mtime - final_elapsed + elapsed if elapsed is not None else path_mtime + (games or 0) / 1_000_000
        points.append(
            Point(
                label=f"{label} @ {games}g" if games is not None else label,
                path=str(path.parent),
                kind="progress",
                high_score=high,
                high_score_minus_starter=minus if minus is not None else high,
                mean_score_minus_starter=_float(row, "mean_score_minus_starter"),
                median_score_minus_starter=_float(row, "median_score_minus_starter"),
                p3072=_float(row, "p_max_tile_excl_starter_ge_3072"),
                p6144=_float(row, "p_max_tile_excl_starter_ge_6144"),
                games=games,
                mtime=timestamp,
                created_at=created_at if isinstance(created_at, str) else None,
                annotation=_annotation_for_label(label),
                record_eligible=False,
                record_eligibility_reason="training progress",
            )
        )
    return points


def collect_points(root: Path = RUNS_ROOT) -> list[Point]:
    points: list[Point] = []
    for path in sorted(root.rglob("summary.json")):
        if _skip_dashboard_source(path):
            continue
        if _skip_replay_start_training(path):
            continue
        point = _point_from_summary(path)
        if point is not None:
            points.append(point)
    for path in sorted(root.rglob("progress.csv")):
        if _skip_dashboard_source(path):
            continue
        if _skip_replay_start_training(path):
            continue
        if "student1" in str(path) or path.stat().st_mtime > time.time() - 24 * 3600:
            points.extend(_points_from_progress(path))

    points.sort(key=lambda point: (point.mtime, point.label))
    best = 0.0
    for point in points:
        if point.record_eligible and point.high_score > best:
            if point.annotation:
                point.annotation = f"{point.annotation} New high-score record."
            else:
                point.annotation = "New high-score record."
            best = point.high_score
    return points


def collect_global_top_replays(root: Path = RUNS_ROOT, *, limit: int = 3) -> list[dict[str, Any]]:
    replays: list[dict[str, Any]] = []
    for path in sorted(root.rglob("summary.json")):
        if _skip_dashboard_source(path):
            continue
        if _skip_replay_start_training(path):
            continue
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        record_eligible, _ = _record_eligibility(path, payload)
        if not record_eligible:
            continue
        top_games = payload.get("top_games")
        if not isinstance(top_games, list):
            continue
        for item in top_games:
            if not isinstance(item, dict):
                continue
            score = _float(item, "score")
            if score is None:
                continue
            replays.append(
                {
                    "run": path.parent.name,
                    "run_path": str(path.parent),
                    "score": int(score),
                    "score_minus_starter": int(_float(item, "score_minus_starter") or score),
                    "seed": item.get("seed"),
                    "starter_tile": item.get("starter_tile"),
                    "moves": item.get("moves"),
                    "max_tile": item.get("max_tile"),
                    "max_tile_excl_starter": item.get("max_tile_excl_starter"),
                    "html": item.get("html"),
                    "json": item.get("json"),
                }
            )
    replays.sort(
        key=lambda item: (
            float(item.get("score") or 0),
            float(item.get("score_minus_starter") or 0),
            float(item.get("moves") or 0),
        ),
        reverse=True,
    )
    unique: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for replay in replays:
        key = (
            replay.get("seed"),
            replay.get("starter_tile"),
            replay.get("score"),
            replay.get("moves"),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(replay)
        if len(unique) >= int(limit):
            break
    return unique


def _latest_json(root: Path, pattern: str) -> tuple[Path, dict[str, Any]] | None:
    candidates = list(root.glob(pattern))
    if not candidates:
        return None
    path = max(candidates, key=lambda candidate: candidate.stat().st_mtime)
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return path, payload if isinstance(payload, dict) else {}


def _frontier_transition(
    payload: dict[str, Any],
    *,
    label: str,
    context: str,
) -> dict[str, Any] | None:
    hits = _float(payload, "target_hits")
    if hits is None:
        hits = _float(payload, "hits")
    rollouts = _float(payload, "valid_rollouts")
    if rollouts is None:
        rollouts = _float(payload, "rollouts")
    rate = _float(payload, "target_rate")
    if rate is None:
        rate = _float(payload, "rate")
    if rate is None and hits is not None and rollouts:
        rate = hits / rollouts
    if hits is None or rollouts is None or rate is None:
        return None
    return {
        "label": label,
        "context": context,
        "hits": int(hits),
        "rollouts": int(rollouts),
        "rate": rate,
        "cases": int(payload["cases_selected"])
        if payload.get("cases_selected") is not None
        else int(payload["cases"])
        if payload.get("cases") is not None
        else None,
        "positive_cases": int(payload["cases_with_any_hit"])
        if payload.get("cases_with_any_hit") is not None
        else int(payload["positive_cases"])
        if payload.get("positive_cases") is not None
        else None,
        "roots": int(payload["ancestries_selected"])
        if payload.get("ancestries_selected") is not None
        else None,
        "horizon": int(payload["horizon"]) if payload.get("horizon") is not None else None,
    }


def collect_frontier_progress(root: Path = RUNS_ROOT) -> dict[str, Any] | None:
    """Collect curated continuation diagnostics without mixing them into score records."""
    chain_artifact = _latest_json(
        root,
        "forensics/frontier_compare/local_chain_to_6144_*/summary.json",
    )
    if chain_artifact is None:
        return None

    chain_path, chain = chain_artifact
    rungs = chain.get("rungs") if isinstance(chain.get("rungs"), dict) else {}
    transitions: list[dict[str, Any]] = []
    sources = [str(chain_path)]
    updated_at: str | None = None

    barrier_artifact = _latest_json(
        root,
        "forensics/frontier_compare/root*_nearadj1536_chain_*/summary.json",
    )
    bottleneck = "Duplicate 1536 to adjacent 1536 remains the unresolved local bottleneck."
    if barrier_artifact is not None:
        barrier_path, barrier = barrier_artifact
        sources.append(str(barrier_path))
        if isinstance(barrier.get("created_at"), str):
            updated_at = barrier["created_at"]
        steps = barrier.get("steps") if isinstance(barrier.get("steps"), list) else []
        for step in steps[:2]:
            if not isinstance(step, dict):
                continue
            transition = _frontier_transition(
                step,
                label=str(step.get("label") or step.get("target") or "Frontier transition"),
                context="hard-root continuation",
            )
            if transition is not None:
                transitions.append(transition)
        if isinstance(barrier.get("interpretation"), str):
            bottleneck = barrier["interpretation"]

    downstream = [
        (
            "adjacent_1536_to_second_3072",
            "Adjacent 1536 -> second 3072",
            "pooled frontier continuation",
        ),
        (
            "second_3072_to_6144",
            "Second 3072 -> 6144",
            "pooled frontier continuation",
        ),
    ]
    highest_milestone: int | None = None
    for key, label, context in downstream:
        rung = rungs.get(key) if isinstance(rungs, dict) else None
        summary = rung.get("summary") if isinstance(rung, dict) else None
        if not isinstance(summary, dict):
            continue
        transition = _frontier_transition(summary, label=label, context=context)
        if transition is not None:
            transitions.append(transition)
            if key == "second_3072_to_6144" and transition["hits"] > 0:
                highest_milestone = 6144
        created_at = summary.get("created_at")
        if updated_at is None and isinstance(created_at, str):
            updated_at = created_at

    return {
        "status": "diagnostic_only",
        "highest_milestone": highest_milestone,
        "updated_at": updated_at,
        "transitions": transitions,
        "bottleneck": bottleneck,
        "sources": sources,
    }


def collect_human_inbox_status(root: Path = RUNS_ROOT) -> dict[str, Any] | None:
    path = root / "human_diagnostics" / "human_diagnostics_batch.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    totals = payload.get("totals")
    if not isinstance(totals, dict):
        totals = {}
    target_intake = payload.get("target_intake")
    if not isinstance(target_intake, dict):
        target_intake = {}
    return {
        "status": payload.get("status") or "unknown",
        "mode": payload.get("mode") or "unknown",
        "sessions": int(totals.get("sessions") or 0),
        "pending_sessions": int(totals.get("pending_sessions") or 0),
        "current_sessions": int(totals.get("current_sessions") or 0),
        "processed_sessions": int(totals.get("processed_sessions") or 0),
        "games_imported": int(totals.get("games_imported") or 0),
        "high_score": totals.get("high_score"),
        "highest_max_tile_excl_starter": totals.get("highest_max_tile_excl_starter"),
        "games_reaching_nonstarter_1536": int(totals.get("games_reaching_nonstarter_1536") or 0),
        "games_reaching_3072": int(totals.get("games_reaching_3072") or 0),
        "games_reaching_6144": int(totals.get("games_reaching_6144") or 0),
        "target_nonstarter_1536": int(target_intake.get("independent_games_reaching_nonstarter_1536") or 5),
        "target_3072": int(target_intake.get("independent_games_reaching_3072") or 1),
        "ready_for_human_root_labeling": bool(target_intake.get("ready_for_human_root_labeling")),
        "html": payload.get("html"),
        "json": str(path),
        "next_command": payload.get("next_command"),
    }


def collect_replay_retention_status(root: Path = RUNS_ROOT) -> dict[str, Any] | None:
    path = root / "dashboard" / "replay_retention_audit.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    counts = payload.get("counts")
    if not isinstance(counts, dict):
        counts = {}
    protected = payload.get("protected_global_top_replays")
    if not isinstance(protected, list):
        protected = []
    scores = []
    for item in protected:
        if not isinstance(item, dict):
            continue
        try:
            scores.append(int(float(item.get("score"))))
        except (TypeError, ValueError):
            continue
    return {
        "status": "ok" if not int(counts.get("missing_protected_global_top_json") or 0) else "missing_artifact",
        "mode": payload.get("mode") or "dry_run",
        "global_top_limit": int(payload.get("global_top_limit") or GLOBAL_TOP_REPLAY_LIMIT),
        "protected_count": len(protected),
        "protected_scores": scores,
        "missing_protected_global_top_json": int(counts.get("missing_protected_global_top_json") or 0),
        "potential_prune_count": int(counts.get("non_global_top_game_entries") or 0),
        "replay_dirs": int(counts.get("replay_dirs") or 0),
        "json": str(path),
    }


def collect_top_replay_playlist_status(root: Path = RUNS_ROOT) -> dict[str, Any] | None:
    path = root / "replays" / "top3" / "manifest.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    replays = payload.get("replays")
    if not isinstance(replays, list):
        replays = []
    scores: list[int] = []
    for item in replays:
        if not isinstance(item, dict):
            continue
        score = _float(item, "score")
        if score is not None:
            scores.append(int(score))
    return {
        "html": payload.get("html") or str(root / "replays" / "top3" / "index.html"),
        "json": str(path),
        "copied_count": int(payload.get("copied_count") or 0),
        "scores": scores,
        "generated_at": payload.get("generated_at"),
    }


def _replay_scores(replays: Any) -> list[int]:
    if not isinstance(replays, list):
        return []
    scores: list[int] = []
    for item in replays:
        if not isinstance(item, dict):
            continue
        score = _float(item, "score")
        if score is not None:
            scores.append(int(score))
    return scores


def _normalize_top_replay_playlist(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    normalized = dict(payload)
    if "scores" not in normalized:
        normalized["scores"] = _replay_scores(normalized.get("replays"))
    return normalized


def dashboard_payload(
    points: list[Point],
    top_replays: list[dict[str, Any]] | None = None,
    frontier_progress: dict[str, Any] | None = None,
    human_inbox: dict[str, Any] | None = None,
    replay_retention: dict[str, Any] | None = None,
    top_replay_playlist: dict[str, Any] | None = None,
) -> dict[str, Any]:
    capped_top_replays = list(top_replays or [])[:GLOBAL_TOP_REPLAY_LIMIT]
    top_replay_playlist = _normalize_top_replay_playlist(top_replay_playlist)
    global_top_scores = _replay_scores(capped_top_replays)
    top_playlist_scores = (
        list(top_replay_playlist.get("scores") or [])
        if isinstance(top_replay_playlist, dict)
        else []
    )
    if not points:
        return {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "best_high_score": None,
            "best_high_score_minus_starter": None,
            "latest_high_score": None,
            "latest_mean_score_minus_starter": None,
            "latest_median_score_minus_starter": None,
            "latest_p3072": None,
            "latest_p6144": None,
            "global_top_scores": global_top_scores,
            "top_replay_playlist_scores": top_playlist_scores,
            "points": [],
            "global_top_replays": capped_top_replays,
            "global_top_replay_limit": GLOBAL_TOP_REPLAY_LIMIT,
            "frontier_progress": frontier_progress,
            "human_inbox": human_inbox,
            "replay_retention": replay_retention,
            "top_replay_playlist": top_replay_playlist,
        }
    eligible_points = [point for point in points if point.record_eligible]
    best = max(eligible_points, key=lambda point: point.high_score) if eligible_points else None
    latest = max(points, key=lambda point: point.mtime)
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "best_high_score": best.high_score if best else None,
        "best_high_score_minus_starter": best.high_score_minus_starter if best else None,
        "latest_high_score": latest.high_score,
        "latest_mean_score_minus_starter": latest.mean_score_minus_starter,
        "latest_median_score_minus_starter": latest.median_score_minus_starter,
        "latest_p3072": latest.p3072,
        "latest_p6144": latest.p6144,
        "global_top_scores": global_top_scores,
        "top_replay_playlist_scores": top_playlist_scores,
        "best": asdict(best) if best else None,
        "latest": asdict(latest),
        "points": [asdict(point) for point in points],
        "global_top_replays": capped_top_replays,
        "global_top_replay_limit": GLOBAL_TOP_REPLAY_LIMIT,
        "frontier_progress": frontier_progress,
        "human_inbox": human_inbox,
        "replay_retention": replay_retention,
        "top_replay_playlist": top_replay_playlist,
    }


def score_trends_payload(points: list[Point]) -> dict[str, Any]:
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "points": [
            {
                "label": point.label,
                "path": point.path,
                "kind": point.kind,
                "high_score": point.high_score,
                "high_score_minus_starter": point.high_score_minus_starter,
                "mean_score_minus_starter": point.mean_score_minus_starter,
                "median_score_minus_starter": point.median_score_minus_starter,
                "games": point.games,
                "mtime": point.mtime,
                "created_at": point.created_at,
                "record_eligible": point.record_eligible,
                "record_eligibility_reason": point.record_eligibility_reason,
            }
            for point in points
        ],
    }


def write_score_trends_html(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, separators=(",", ":"))
    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Threes RL Score Metrics</title>
  <style>
    :root { color-scheme: dark; --bg: #101113; --panel: #191d21; --line: #364047; --ink: #f1f5f0; --muted: #a9b3ad; --gold: #e9bd4a; --green: #71c79a; --blue: #7bb7e8; }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--ink); font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    main { width: min(1220px, calc(100vw - 32px)); margin: 0 auto; padding: 22px 0 34px; }
    header { display: flex; justify-content: space-between; gap: 18px; align-items: end; border-bottom: 1px solid var(--line); padding-bottom: 16px; margin-bottom: 18px; }
    h1, h2, p { margin: 0; }
    h1 { font-size: 25px; }
    h2 { font-size: 15px; margin-bottom: 10px; }
    .muted { color: var(--muted); }
    .panel { border: 1px solid var(--line); background: var(--panel); border-radius: 8px; padding: 13px; margin-bottom: 14px; }
    svg { width: 100%; height: 430px; display: block; background: #13171a; border: 1px solid var(--line); border-radius: 8px; }
    .legend { display: flex; gap: 14px; flex-wrap: wrap; margin-top: 10px; color: var(--muted); font-size: 13px; }
    .swatch { display: inline-block; width: 11px; height: 11px; border-radius: 2px; margin-right: 6px; vertical-align: -1px; }
    table { width: 100%; border-collapse: collapse; font-size: 12px; font-variant-numeric: tabular-nums; }
    th, td { border-bottom: 1px solid var(--line); padding: 7px 6px; text-align: right; vertical-align: top; }
    th:first-child, td:first-child { text-align: left; }
    a { color: var(--blue); text-decoration: none; }
    a:hover { text-decoration: underline; }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>Score Metrics</h1>
        <p class="muted">Global full-game trend view for high, mean, and median score minus starter.</p>
      </div>
      <p class="muted"><a href="index.html">main dashboard</a></p>
    </header>
    <section class="panel">
      <h2>High / Mean / Median Trend</h2>
      <svg id="metricsChart" viewBox="0 0 1000 430" role="img" aria-label="High, mean, and median score trend"></svg>
      <div class="legend">
        <span><span class="swatch" style="background: var(--gold);"></span>High minus starter</span>
        <span><span class="swatch" style="background: var(--green);"></span>Mean minus starter</span>
        <span><span class="swatch" style="background: var(--blue);"></span>Median minus starter</span>
      </div>
    </section>
    <section class="panel">
      <h2>Recent Metrics</h2>
      <table id="metricsTable"></table>
    </section>
  </main>
  <script>
    const payload = __DATA__;
    const points = payload.points || [];
    const fmt = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });
    const timeFmt = new Intl.DateTimeFormat([], { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
    const esc = text => String(text ?? "").replace(/[&<>"']/g, ch => ({"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"}[ch]));
    function value(point, key) {
      const raw = Number(point[key]);
      return Number.isFinite(raw) ? raw : null;
    }
    function renderChart() {
      const svg = document.getElementById("metricsChart");
      if (!points.length) {
        svg.innerHTML = '<text x="500" y="215" text-anchor="middle" fill="#a9b3ad">No score metrics found yet</text>';
        return;
      }
      const w = 1000, h = 430, left = 74, right = 30, top = 26, bottom = 58;
      const times = points.map(p => Number(p.mtime || 0) * 1000);
      const minT = Math.min(...times);
      const maxT = Math.max(...times);
      const series = [
        ["high_score_minus_starter", "var(--gold)"],
        ["mean_score_minus_starter", "var(--green)"],
        ["median_score_minus_starter", "var(--blue)"],
      ];
      const values = points.flatMap(p => series.map(([key]) => value(p, key)).filter(v => v != null));
      const maxY = Math.max(1, ...values);
      const x = t => left + (maxT === minT ? 0 : (t - minT) / (maxT - minT)) * (w - left - right);
      const y = v => h - bottom - (Number(v || 0) / maxY) * (h - top - bottom);
      const grid = [0, .25, .5, .75, 1].map(t => {
        const yy = y(maxY * t);
        return `<line x1="${left}" y1="${yy}" x2="${w - right}" y2="${yy}" stroke="#364047" />
                <text x="${left - 10}" y="${yy + 4}" text-anchor="end" fill="#a9b3ad" font-size="12">${fmt.format(maxY * t)}</text>`;
      }).join("");
      const ticks = Array.from({ length: Math.min(5, points.length) }, (_, i) => minT + (maxT - minT) * (i / Math.max(1, Math.min(5, points.length) - 1)));
      const xTicks = ticks.map(t => {
        const xx = x(t);
        return `<line x1="${xx}" y1="${h - bottom}" x2="${xx}" y2="${h - bottom + 5}" stroke="#a9b3ad" />
                <text x="${xx}" y="${h - 22}" text-anchor="middle" fill="#a9b3ad" font-size="11">${esc(timeFmt.format(new Date(t)))}</text>`;
      }).join("");
      const lines = series.map(([key, color]) => {
        const coords = points.map((p, i) => {
          const v = value(p, key);
          return v == null ? null : `${x(times[i])},${y(v)}`;
        }).filter(Boolean).join(" ");
        return coords ? `<polyline fill="none" stroke="${color}" stroke-width="3" points="${coords}" />` : "";
      }).join("");
      svg.innerHTML = `${grid}
        <line x1="${left}" y1="${h - bottom}" x2="${w - right}" y2="${h - bottom}" stroke="#a9b3ad" />
        <line x1="${left}" y1="${top}" x2="${left}" y2="${h - bottom}" stroke="#a9b3ad" />
        ${xTicks}
        ${lines}
        <text x="${w / 2}" y="${h - 6}" text-anchor="middle" fill="#a9b3ad" font-size="12">Run time</text>`;
    }
    function renderTable() {
      const rows = points.slice(-28).reverse();
      const table = document.getElementById("metricsTable");
      table.innerHTML = `<thead><tr><th>Point</th><th>Kind</th><th>High</th><th>High Minus</th><th>Mean Minus</th><th>Median Minus</th><th>Games</th><th>Time</th></tr></thead><tbody>${rows.map(p => `
        <tr>
          <td title="${esc(p.path)}">${esc(p.label)}</td>
          <td>${esc(p.kind)}</td>
          <td>${fmt.format(value(p, "high_score") || 0)}</td>
          <td>${fmt.format(value(p, "high_score_minus_starter") || 0)}</td>
          <td>${value(p, "mean_score_minus_starter") == null ? "-" : fmt.format(value(p, "mean_score_minus_starter"))}</td>
          <td>${value(p, "median_score_minus_starter") == null ? "-" : fmt.format(value(p, "median_score_minus_starter"))}</td>
          <td>${p.games ?? "-"}</td>
          <td>${timeFmt.format(new Date(Number(p.mtime || 0) * 1000))}</td>
        </tr>`).join("")}</tbody>`;
    }
    renderChart();
    renderTable();
  </script>
</body>
</html>
""".replace("__DATA__", data)
    path.write_text(html)


def write_score_trends(out_dir: Path, points: list[Point]) -> dict[str, Any]:
    payload = score_trends_payload(points)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "score_trends.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    write_score_trends_html(out_dir / "score_trends.html", payload)
    return payload


def write_html(path: Path, payload: dict[str, Any], refresh_seconds: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, separators=(",", ":"))
    refresh = f'<meta http-equiv="refresh" content="{int(refresh_seconds)}">' if refresh_seconds > 0 else ""
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  {refresh}
  <title>Threes RL Research Dashboard</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #101113;
      --panel: #191d21;
      --ink: #f1f5f0;
      --muted: #a9b3ad;
      --line: #364047;
      --gold: #e9bd4a;
      --blue: #7bb7e8;
      --red: #ee8077;
      --green: #71c79a;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{
      width: min(1220px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 22px 0 34px;
    }}
    header {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 18px;
      align-items: end;
      border-bottom: 1px solid var(--line);
      padding-bottom: 16px;
      margin-bottom: 18px;
    }}
    h1, h2, p {{ margin: 0; }}
    h1 {{ font-size: 25px; letter-spacing: 0; }}
    h2 {{ font-size: 15px; letter-spacing: 0; margin-bottom: 10px; }}
    .muted {{ color: var(--muted); }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }}
    .card, .panel {{
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
      padding: 13px;
    }}
    .label {{
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: .08em;
      margin-bottom: 4px;
    }}
    .value {{
      font-size: 23px;
      font-weight: 800;
      font-variant-numeric: tabular-nums;
    }}
    .section-heading {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: start;
      margin-bottom: 12px;
    }}
    .section-heading h2 {{ margin-bottom: 4px; }}
    .badge {{
      flex: 0 0 auto;
      border: 1px solid var(--gold);
      color: var(--gold);
      border-radius: 4px;
      padding: 3px 6px;
      font-size: 10px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: .08em;
    }}
    .frontier-metrics {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      border-top: 1px solid var(--line);
      border-bottom: 1px solid var(--line);
    }}
    .frontier-metric {{
      min-width: 0;
      padding: 12px 12px 12px 0;
    }}
    .frontier-metric + .frontier-metric {{
      border-left: 1px solid var(--line);
      padding-left: 12px;
    }}
    .frontier-rate {{
      color: var(--green);
      font-size: 20px;
      font-weight: 800;
      font-variant-numeric: tabular-nums;
    }}
    .frontier-label {{
      min-height: 34px;
      margin-top: 3px;
      font-size: 12px;
      line-height: 1.35;
    }}
    .frontier-meta, .frontier-note {{
      color: var(--muted);
      font-size: 11px;
      line-height: 1.4;
    }}
    .frontier-note {{ margin-top: 10px; }}
    svg {{
      width: 100%;
      height: 430px;
      display: block;
      background: #13171a;
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    .legend {{
      display: flex;
      gap: 14px;
      flex-wrap: wrap;
      margin-top: 10px;
      color: var(--muted);
      font-size: 13px;
    }}
    .swatch {{
      display: inline-block;
      width: 11px;
      height: 11px;
      border-radius: 2px;
      margin-right: 6px;
      vertical-align: -1px;
    }}
    .chart-wrap {{
      position: relative;
    }}
    .tooltip {{
      position: absolute;
      max-width: min(360px, calc(100vw - 72px));
      pointer-events: none;
      opacity: 0;
      transform: translate(10px, -10px);
      transition: opacity .12s ease;
      border: 1px solid var(--line);
      background: #0f1317;
      color: var(--ink);
      border-radius: 8px;
      padding: 9px 10px;
      font-size: 12px;
      line-height: 1.35;
      box-shadow: 0 10px 28px rgba(0, 0, 0, .35);
      z-index: 2;
    }}
    .tooltip.visible {{ opacity: 1; }}
    .tooltip strong {{
      display: block;
      font-size: 13px;
      margin-bottom: 4px;
    }}
    .tooltip .meta {{
      color: var(--muted);
      margin-top: 4px;
    }}
    .point {{
      cursor: help;
    }}
    .point .hit {{
      fill: transparent;
      stroke: transparent;
      pointer-events: all;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
      font-variant-numeric: tabular-nums;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 7px 6px;
      text-align: right;
      vertical-align: top;
    }}
    th:first-child, td:first-child {{ text-align: left; }}
    a {{ color: var(--blue); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .status {{
      justify-self: end;
      text-align: right;
      font-size: 13px;
      color: var(--muted);
    }}
    @media (max-width: 900px) {{
      header {{ grid-template-columns: 1fr; }}
      .cards {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .frontier-metrics {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .frontier-metric:nth-child(3) {{ border-left: 0; }}
      .frontier-metric:nth-child(n + 3) {{ border-top: 1px solid var(--line); }}
      .status {{ justify-self: start; text-align: left; }}
    }}
    @media (max-width: 560px) {{
      .frontier-metrics {{ grid-template-columns: 1fr; }}
      .frontier-metric + .frontier-metric {{ border-left: 0; border-top: 1px solid var(--line); }}
      .frontier-label {{ min-height: 0; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>Threes RL Research Dashboard</h1>
        <p class="muted">Normal-start full-game records, with frontier continuation research tracked separately. <a href="score_trends.html">score metrics</a></p>
      </div>
      <div class="status" id="status"></div>
    </header>
    <section class="cards">
      <div class="card"><div class="label">Normal-Start Record</div><div class="value" id="bestScore">-</div></div>
      <div class="card"><div class="label">Best Minus Starter</div><div class="value" id="bestMinus">-</div></div>
      <div class="card"><div class="label">Latest Full-Game High</div><div class="value" id="latestScore">-</div></div>
      <div class="card"><div class="label">Full-Game Points</div><div class="value" id="pointCount">-</div></div>
    </section>
    <section class="panel" id="opsPanel" style="margin-bottom: 14px;" hidden>
      <div class="section-heading">
        <div>
          <h2>Research Inputs & Retention</h2>
          <p class="muted">Human-data intake and protected top-replay state.</p>
        </div>
      </div>
      <table id="opsTable"></table>
    </section>
    <section class="panel" id="frontierPanel" style="margin-bottom: 14px;" hidden>
      <div class="section-heading">
        <div>
          <h2>Frontier Transition Research</h2>
          <p class="muted" id="frontierSummary"></p>
        </div>
        <span class="badge">Diagnostic only</span>
      </div>
      <div class="frontier-metrics" id="frontierMetrics"></div>
      <p class="frontier-note" id="frontierNote"></p>
    </section>
    <section class="panel">
      <div class="section-heading">
        <div>
          <h2>High Score Timeline</h2>
          <p class="muted" id="chartFreshness"></p>
        </div>
        <span class="badge" id="chartEligibility"></span>
      </div>
      <div class="chart-wrap">
        <svg id="chart" viewBox="0 0 1000 430" role="img" aria-label="Best high score timeline"></svg>
        <div class="tooltip" id="tooltip"></div>
      </div>
      <div class="legend">
        <span><span class="swatch" style="background: var(--gold);"></span>Best high score</span>
        <span><span class="swatch" style="background: var(--red);"></span>Annotated change</span>
      </div>
    </section>
    <section class="panel" style="margin-top: 14px;">
      <div class="section-heading">
        <div>
          <h2>Global Top 3 Normal-Start Replays</h2>
          <p class="muted">Highest retained full-game starts only; continuation diagnostics stay out of this list.</p>
        </div>
        <p class="muted" id="topPlaylistLink"></p>
      </div>
      <table id="topReplayTable"></table>
    </section>
    <section class="panel" style="margin-top: 14px;">
      <h2>Recent Points</h2>
      <table id="table"></table>
    </section>
  </main>
  <script>
    const dashboard = {data};
    const points = dashboard.points || [];
    const topReplays = dashboard.global_top_replays || [];
    const topReplayLimit = Number(dashboard.global_top_replay_limit || 3);
    const frontier = dashboard.frontier_progress || null;
    const humanInbox = dashboard.human_inbox || null;
    const replayRetention = dashboard.replay_retention || null;
    const topReplayPlaylist = dashboard.top_replay_playlist || null;
    const fmt = new Intl.NumberFormat("en-US", {{ maximumFractionDigits: 0 }});
    const timeFmt = new Intl.DateTimeFormat([], {{ month: "short", day: "numeric", hour: "numeric", minute: "2-digit" }});
    const pct = value => value == null ? "-" : `${{(Number(value) * 100).toFixed(1)}}%`;
    const esc = text => String(text ?? "").replace(/[&<>"']/g, ch => ({{"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"}}[ch]));
    const runsHref = path => `../${{String(path || "").replace(/^threes_rl\\/runs\\//, "")}}`;
    const best = dashboard.best || null;
    const latest = dashboard.latest || null;
    document.getElementById("status").textContent = `Generated ${{dashboard.generated_at || "-"}}${{location.protocol === "file:" ? " / file auto-refresh" : ""}}`;
    document.getElementById("bestScore").textContent = best ? fmt.format(best.high_score) : "-";
    document.getElementById("bestMinus").textContent = best ? fmt.format(best.high_score_minus_starter) : "-";
    document.getElementById("latestScore").textContent = latest ? fmt.format(latest.high_score) : "-";
    document.getElementById("pointCount").textContent = fmt.format(points.length);
    if (latest) {{
      const latestDate = new Date(Number(latest.mtime || 0) * 1000);
      const latestWhen = Number.isNaN(latestDate.getTime()) ? "unknown time" : timeFmt.format(latestDate);
      const reason = latest.record_eligibility_reason ? ` ${{latest.record_eligibility_reason}}.` : "";
      document.getElementById("chartFreshness").textContent = `Latest research result ${{latestWhen}}.${{latest.record_eligible === false ? reason : ""}}`;
      document.getElementById("chartEligibility").textContent = latest.record_eligible === false ? "Research only" : "Record eligible";
    }} else {{
      document.getElementById("chartFreshness").textContent = "No research results yet.";
      document.getElementById("chartEligibility").textContent = "No data";
    }}

    function renderOpsStatus() {{
      const rows = [];
      if (humanInbox) {{
        const pending = Number(humanInbox.pending_sessions || 0);
        const target1536 = Number(humanInbox.target_nonstarter_1536 || 5);
        const target3072 = Number(humanInbox.target_3072 || 1);
        const detail = `${{fmt.format(humanInbox.sessions || 0)}} sessions / ${{fmt.format(humanInbox.games_imported || 0)}} games imported${{pending ? ` / ${{fmt.format(pending)}} pending` : ""}}. Intake ${{fmt.format(humanInbox.games_reaching_nonstarter_1536 || 0)}}/${{fmt.format(target1536)}} non-starter 1536 and ${{fmt.format(humanInbox.games_reaching_3072 || 0)}}/${{fmt.format(target3072)}} reaching 3072.`;
        rows.push({{
          source: "Human data inbox",
          state: humanInbox.status || "-",
          detail,
          link: humanInbox.html ? `<a href="${{esc(runsHref(humanInbox.html))}}">open</a>` : (humanInbox.json ? `<a href="${{esc(runsHref(humanInbox.json))}}">json</a>` : "-"),
        }});
      }}
      if (replayRetention) {{
        const scores = (replayRetention.protected_scores || []).map(value => fmt.format(value)).join(", ");
        const detail = `Top ${{fmt.format(replayRetention.protected_count || 0)}} / ${{fmt.format(replayRetention.global_top_limit || topReplayLimit)}} protected${{scores ? `: ${{scores}}` : ""}}. ${{fmt.format(replayRetention.potential_prune_count || 0)}} older per-run top entries outside the global set.`;
        rows.push({{
          source: "Replay retention",
          state: replayRetention.status || "-",
          detail,
          link: replayRetention.json ? `<a href="${{esc(runsHref(replayRetention.json))}}">audit</a>` : "-",
        }});
      }}
      if (!rows.length) return;
      document.getElementById("opsPanel").hidden = false;
      document.getElementById("opsTable").innerHTML = `<thead><tr><th>Source</th><th>Status</th><th>Details</th><th>Link</th></tr></thead><tbody>${{rows.map(row => `
        <tr>
          <td>${{esc(row.source)}}</td>
          <td>${{esc(row.state)}}</td>
          <td>${{esc(row.detail)}}</td>
          <td>${{row.link}}</td>
        </tr>`).join("")}}</tbody>`;
    }}

    function renderFrontier() {{
      if (!frontier || !(frontier.transitions || []).length) return;
      const panel = document.getElementById("frontierPanel");
      panel.hidden = false;
      const milestone = frontier.highest_milestone ? `Local chain validated through ${{fmt.format(frontier.highest_milestone)}}.` : "Frontier chain in progress.";
      const updatedDate = frontier.updated_at ? new Date(String(frontier.updated_at).replace(/([+-]\\d{{2}})(\\d{{2}})$/, "$1:$2")) : null;
      const updated = updatedDate && !Number.isNaN(updatedDate.getTime()) ? ` Updated ${{timeFmt.format(updatedDate)}}.` : "";
      document.getElementById("frontierSummary").innerHTML = `${{milestone}} Continuation starts; not a normal-start policy result.${{updated}}`;
      document.getElementById("frontierMetrics").innerHTML = frontier.transitions.slice(0, 4).map(item => `
        <div class="frontier-metric">
          <div class="frontier-rate">${{pct(item.rate)}}</div>
          <div class="frontier-label">${{esc(item.label)}}</div>
          <div class="frontier-meta">${{fmt.format(item.hits)}} / ${{fmt.format(item.rollouts)}} rollouts${{item.horizon ? ` / h${{item.horizon}}` : ""}}${{item.roots ? ` / ${{item.roots}} roots` : ""}}</div>
        </div>`).join("");
      document.getElementById("frontierNote").textContent = `Current bottleneck: ${{frontier.bottleneck || "not yet identified"}}`;
    }}

    function renderChart() {{
      const svg = document.getElementById("chart");
      if (!points.length) {{
        svg.innerHTML = `<text x="500" y="215" text-anchor="middle" fill="#a9b3ad">No run summaries found yet</text>`;
        return;
      }}
      const w = 1000, h = 430, left = 74, right = 30, top = 26, bottom = 58;
      const recordScores = [];
      let record = 0;
      for (const point of points) {{
        if (point.record_eligible !== false) {{
          record = Math.max(record, Number(point.high_score || 0));
        }}
        recordScores.push(record);
      }}
      const maxY = Math.max(1, ...recordScores);
      const times = points.map(p => Number(p.mtime || 0) * 1000);
      const minT = Math.min(...times);
      const maxT = Math.max(...times);
      const x = t => left + (maxT === minT ? 0 : (t - minT) / (maxT - minT)) * (w - left - right);
      const y = v => h - bottom - (Number(v || 0) / maxY) * (h - top - bottom);
      const grid = [0, .25, .5, .75, 1].map(t => {{
        const yy = y(maxY * t);
        return `<line x1="${{left}}" y1="${{yy}}" x2="${{w - right}}" y2="${{yy}}" stroke="#364047" />
                <text x="${{left - 10}}" y="${{yy + 4}}" text-anchor="end" fill="#a9b3ad" font-size="12">${{fmt.format(maxY * t)}}</text>`;
      }}).join("");
      const ticks = Array.from({{ length: Math.min(5, points.length) }}, (_, i) => minT + (maxT - minT) * (i / Math.max(1, Math.min(5, points.length) - 1)));
      const xTicks = ticks.map(t => {{
        const xx = x(t);
        return `<line x1="${{xx}}" y1="${{h - bottom}}" x2="${{xx}}" y2="${{h - bottom + 5}}" stroke="#a9b3ad" />
                <text x="${{xx}}" y="${{h - 22}}" text-anchor="middle" fill="#a9b3ad" font-size="11">${{esc(timeFmt.format(new Date(t)))}}</text>`;
      }}).join("");
      const line = `<polyline fill="none" stroke="var(--gold)" stroke-width="3" points="${{points.map((_p, i) => `${{x(times[i])}},${{y(recordScores[i])}}`).join(" ")}}" />`;
      const dots = points.map((p, i) => {{
        const annotated = Boolean(p.annotation && p.kind !== "progress");
        if (!annotated) return "";
        const cx = x(times[i]), cy = y(recordScores[i]);
        const aria = `${{p.label}}${{p.annotation ? `: ${{p.annotation}}` : ""}}`;
        return `<g class="point" data-index="${{i}}" tabindex="0" role="button" aria-label="${{esc(aria)}}"><title>${{esc(aria)}}</title><circle class="hit" cx="${{cx}}" cy="${{cy}}" r="14"></circle><circle cx="${{cx}}" cy="${{cy}}" r="5.5" fill="var(--red)" stroke="#13171a" stroke-width="1.5"></circle></g>`;
      }}).join("");
      svg.innerHTML = `${{grid}}
        <line x1="${{left}}" y1="${{h - bottom}}" x2="${{w - right}}" y2="${{h - bottom}}" stroke="#a9b3ad" />
        <line x1="${{left}}" y1="${{top}}" x2="${{left}}" y2="${{h - bottom}}" stroke="#a9b3ad" />
        ${{xTicks}}
        ${{line}}
        ${{dots}}
        <text x="${{w / 2}}" y="${{h - 6}}" text-anchor="middle" fill="#a9b3ad" font-size="12">Run time</text>`;
      bindTooltips();
    }}

    function bindTooltips() {{
      const tooltip = document.getElementById("tooltip");
      const chart = document.querySelector(".chart-wrap");
      const svg = document.getElementById("chart");
      let activePoint = null;
      const placeTooltip = event => {{
        const rect = chart.getBoundingClientRect();
        const pad = 10;
        const width = tooltip.offsetWidth || 280;
        const height = tooltip.offsetHeight || 80;
        let left = event.clientX - rect.left + 12;
        let top = event.clientY - rect.top - 8;
        if (left + width + pad > rect.width) {{
          left = event.clientX - rect.left - width - 12;
        }}
        top = Math.max(pad, Math.min(top, rect.height - height - pad));
        tooltip.style.left = `${{Math.max(pad, left)}}px`;
        tooltip.style.top = `${{top}}px`;
      }};
      const pointFromEvent = event => {{
        const node = event.target && event.target.closest ? event.target.closest(".point") : null;
        return node && svg.contains(node) ? node : null;
      }};
      const pointCenterEvent = node => {{
        const rect = node.getBoundingClientRect();
        return {{ clientX: rect.left + rect.width / 2, clientY: rect.top + rect.height / 2 }};
      }};
      const showPoint = (node, event) => {{
        const point = points[Number(node.dataset.index)];
        tooltip.innerHTML = `<strong>${{esc(point.label)}}</strong>
          ${{point.annotation ? `<div>${{esc(point.annotation)}}</div>` : ""}}
          <div class="meta">High ${{fmt.format(point.high_score)}} / ${{timeFmt.format(new Date(Number(point.mtime || 0) * 1000))}}</div>`;
        tooltip.classList.add("visible");
        activePoint = node;
        placeTooltip(event);
      }};
      const hidePoint = () => {{
        activePoint = null;
        tooltip.classList.remove("visible");
      }};
      for (const eventName of ["mouseover", "pointerover"]) {{
        svg.addEventListener(eventName, event => {{
          const node = pointFromEvent(event);
          if (node) showPoint(node, event);
        }});
      }}
      for (const eventName of ["mousemove", "pointermove"]) {{
        svg.addEventListener(eventName, event => {{
          const node = pointFromEvent(event);
          if (node && node === activePoint) {{
            placeTooltip(event);
          }}
        }});
      }}
      for (const eventName of ["mouseout", "pointerout"]) {{
        svg.addEventListener(eventName, event => {{
          const node = pointFromEvent(event);
          if (!node || (event.relatedTarget && node.contains(event.relatedTarget))) return;
          hidePoint();
        }});
      }}
      svg.addEventListener("click", event => {{
        const node = pointFromEvent(event);
        if (node) showPoint(node, event);
      }});
      svg.addEventListener("focusin", event => {{
        const node = pointFromEvent(event);
        if (node) showPoint(node, pointCenterEvent(node));
      }});
      svg.addEventListener("focusout", hidePoint);
      document.querySelectorAll(".point").forEach(node => {{
        const show = event => showPoint(node, event);
        const move = event => {{
          placeTooltip(event);
        }};
        const hide = () => {{
          if (activePoint === node) hidePoint();
        }};
        node.addEventListener("mouseenter", show);
        node.addEventListener("mousemove", move);
        node.addEventListener("mouseleave", hide);
      }});
    }}

    function renderTable() {{
      const table = document.getElementById("table");
      const rows = points.slice(-24).reverse();
      table.innerHTML = `<thead><tr>
        <th>Point</th><th>Kind</th><th>Games</th><th>High</th><th>Minus Starter</th><th>Mean Minus</th><th>Median Minus</th><th>P >= 3072</th><th>Replay</th>
      </tr></thead><tbody>${{rows.map(p => `
        <tr>
          <td title="${{esc(p.path)}}">${{esc(p.label)}}</td>
          <td>${{esc(p.kind)}}</td>
          <td>${{p.games ?? "-"}}</td>
          <td>${{fmt.format(p.high_score)}}</td>
          <td>${{fmt.format(p.high_score_minus_starter)}}</td>
          <td>${{p.mean_score_minus_starter == null ? "-" : fmt.format(p.mean_score_minus_starter)}}</td>
          <td>${{p.median_score_minus_starter == null ? "-" : fmt.format(p.median_score_minus_starter)}}</td>
          <td>${{pct(p.p3072)}}</td>
          <td>${{p.replay ? `<a href="${{esc(runsHref(p.replay))}}">open</a>` : "-"}}</td>
        </tr>`).join("")}}</tbody>`;
    }}

    function renderTopReplays() {{
      if (topReplayPlaylist && topReplayPlaylist.html) {{
        const generated = topReplayPlaylist.generated_at ? ` / generated ${{esc(topReplayPlaylist.generated_at)}}` : "";
        document.getElementById("topPlaylistLink").innerHTML = `<a href="${{esc(runsHref(topReplayPlaylist.html))}}">open playlist</a>${{generated}}`;
      }}
      const table = document.getElementById("topReplayTable");
      if (!topReplays.length) {{
        table.innerHTML = `<tbody><tr><td>No retained normal-start top replays yet</td></tr></tbody>`;
        return;
      }}
      table.innerHTML = `<thead><tr>
        <th>Rank</th><th>Run</th><th>Seed</th><th>Score</th><th>Minus Starter</th><th>Max Tile</th><th>Replay</th>
      </tr></thead><tbody>${{topReplays.slice(0, topReplayLimit).map((p, i) => `
        <tr>
          <td>${{i + 1}}</td>
          <td title="${{esc(p.run_path)}}">${{esc(p.run)}}</td>
          <td>${{p.seed ?? "-"}}</td>
          <td>${{fmt.format(p.score)}}</td>
          <td>${{fmt.format(p.score_minus_starter)}}</td>
          <td>${{p.max_tile_excl_starter ?? p.max_tile ?? "-"}}</td>
          <td>${{p.html ? `<a href="${{esc(runsHref(p.html))}}">open</a>` : "-"}}</td>
        </tr>`).join("")}}</tbody>`;
    }}

    renderOpsStatus();
    renderFrontier();
    renderChart();
    renderTopReplays();
    renderTable();
  </script>
</body>
</html>
"""
    path.write_text(html)


def build_dashboard(out: Path, refresh_seconds: int = 20) -> dict[str, Any]:
    points = collect_points()
    top_replays = collect_global_top_replays()
    from threes_rl.top_replay_playlist import sync_top_replay_playlist

    top_replay_playlist = sync_top_replay_playlist(top_replays=top_replays)
    payload = dashboard_payload(
        points,
        top_replays,
        collect_frontier_progress(),
        collect_human_inbox_status(),
        collect_replay_retention_status(),
        top_replay_playlist,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    (out.parent / "dashboard.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    write_score_trends(out.parent, points)
    write_html(out, payload, refresh_seconds)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--refresh-seconds", type=int, default=20)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=10.0)
    args = parser.parse_args()

    while True:
        payload = build_dashboard(args.out, args.refresh_seconds)
        print(
            json.dumps(
                {
                    "html": str(args.out),
                    "points": len(payload.get("points", [])),
                    "best_high_score": payload.get("best", {}).get("high_score"),
                    "generated_at": payload.get("generated_at"),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if not args.watch:
            break
        time.sleep(max(1.0, float(args.interval)))


if __name__ == "__main__":
    main()
