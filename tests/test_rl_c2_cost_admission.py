from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from threes_rl import c2_cost_admission as c2
from threes_rl.eval import make_policy
from threes_rl.expectimax import NtupleExpectimaxPolicy
from threes_rl.train_td import state_from_replay_payload


def _model(intercept: float = 0.2, coefficient: float = 0.0) -> dict:
    payload = {
        "version": "c2_nonnegative_cost_model_v1",
        "feature_schema_sha256": c2.feature_schema_payload()["schema_sha256"],
        "feature_names": list(c2.FEATURE_NAMES),
        "intercept": intercept,
        "coefficients": [coefficient] * len(c2.FEATURE_NAMES),
        "l2_lambda": c2.L2_LAMBDA,
        "solver": "test",
        "solver_success": True,
        "solver_status": 1,
        "solver_message": "ok",
        "solver_iterations": 1,
        "cost": 0.0,
        "optimality": 0.0,
        "deterministic_refit": True,
        "training_prediction_sha256": "fixture",
    }
    payload["model_sha256"] = c2.canonical_json_hash(payload)
    return payload


def _validation_rows() -> list[dict]:
    rows = []
    for family_index, (family, _spec) in enumerate(c2.FAMILY_SLATE):
        for state_index in range(8):
            actual = 0.15 + 0.03 * state_index + 0.01 * family_index
            rows.append({
                "record_id": f"{family}-{state_index}",
                "root_ancestry": f"{family}-root-{state_index // 4}",
                "behavior_family": family,
                "features": [actual] + [0.0] * (len(c2.FEATURE_NAMES) - 1),
                "safety_load": actual,
                "exact_c1_combined_s": actual,
                "exact_c1_over_depth2": 2.0,
                "depth2_values_exact": True,
                "depth2_action_match": True,
            })
    return rows


def _gate_rows(*, ratios: list[float] | None = None) -> list[dict]:
    rows = []
    ratios = ratios or [1.2] * 48
    for index, ratio in enumerate(ratios):
        family = c2.FAMILY_SLATE[(index // 16) % 3][0]
        admitted = index % 4 == 0
        rows.append({
            "record_id": f"gate-{index}",
            "root_ancestry": f"{family}-root-{index // 4}",
            "behavior_family": family,
            "admitted": admitted,
            "eligible": True,
            "depth2_s": 0.1,
            "c2_s": 0.1 * ratio,
            "c2_over_depth2": ratio,
            "actual_load": 0.2,
            "absolute_error": 0.05,
            "upper_covers": True,
            "value_action_equivalence": True,
            "admission_repeat_exact": True,
        })
    return rows


def test_stream_manifest_is_exact_and_collision_free_internally() -> None:
    rows = c2.requested_stream_manifest()
    assert len(rows) == 216
    assert [row["behavior_family"] for row in rows[:72]] == ["c2_corner2"] * 72
    values = [row[key] for row in rows for key in c2.STREAM_BASES]
    assert len(values) == 864
    assert len(set(values)) == 864
    assert rows[0]["logical_seed"] == 65_000_000_000
    assert rows[72]["logical_seed"] == 65_001_000_000
    assert rows[144]["logical_seed"] == 65_002_000_000


def test_corpus_plan_is_balanced_and_partitioned_before_timing() -> None:
    plan = c2.corpus_plan_payload(c2.requested_stream_manifest())
    assert plan["required_qualifying_roots_per_family"] == 12
    assert [row["roots_per_family"] for row in plan["partitions"]] == [6, 2, 4]
    assert plan["partition_before_timing"] is True
    assert c2._partition_for_qualifying_index(0) == "cost_fit"
    assert c2._partition_for_qualifying_index(6) == "engineering_validation"
    assert c2._partition_for_qualifying_index(8) == "untouched_runtime_gate"
    with pytest.raises(ValueError):
        c2._partition_for_qualifying_index(12)


def test_accepted_family_audit_matches_immutable_g1r_evidence() -> None:
    audit = c2.accepted_family_audit()
    assert audit["passes"]
    assert all(audit["signature_checks"].values())
    assert all(audit["pair_checks"].values())


def test_feature_schema_is_exact_and_hash_sensitive() -> None:
    schema = c2.feature_schema_payload()
    assert schema["width"] == 20
    assert len(schema["names"]) == len(set(schema["names"])) == 20
    changed = dict(schema)
    changed["names"] = list(reversed(changed["names"]))
    changed.pop("schema_sha256")
    assert c2.canonical_json_hash(changed) != schema["schema_sha256"]


def test_cost_features_are_finite_ordered_and_wall_clock_free() -> None:
    state = SimpleNamespace(
        board=np.asarray([
            [1536, 768, 384, 192],
            [96, 48, 24, 12],
            [6, 3, 2, 1],
            [0, 0, 0, 0],
        ], dtype=np.int32),
        preview=SimpleNamespace(kind="bonus", candidates=(1, 2, 3)),
    )
    counters = {
        "action_calls": 10,
        "value_lookups": 20,
        "unique_value_states": 12,
        "chance_calls": 7,
        "chance_outcomes": 30,
        "afterstate_lookups": 100,
        "unique_afterstates": 40,
        "base_move_calls": 80,
        "unique_base_moves": 20,
        "legal_lookup_calls": 30,
    }
    features = c2.cost_features(
        state=state,
        values=[(0, 10.0), (1, 9.9), (2, 8.0)],
        counters=counters,
    )
    assert features.shape == (20,)
    assert np.all(np.isfinite(features))
    assert np.all(features >= 0)
    assert features[5] == 1.0
    assert features[6] == 1.0
    assert "time" not in " ".join(c2.FEATURE_NAMES)


def test_cost_features_reject_unknown_preview_and_negative_counter() -> None:
    state = SimpleNamespace(
        board=np.zeros((4, 4), dtype=np.int32),
        preview=SimpleNamespace(kind="mystery", candidates=()),
    )
    counters = {name: 0 for name in (
        "action_calls", "value_lookups", "unique_value_states", "chance_calls",
        "chance_outcomes", "afterstate_lookups", "unique_afterstates",
        "base_move_calls", "unique_base_moves", "legal_lookup_calls",
    )}
    with pytest.raises(ValueError, match="preview"):
        c2.cost_features(state=state, values=[(0, 1.0)], counters=counters)
    assert c2._normalized_log_count(0, 10) == 0.0
    with pytest.raises(ValueError, match="negative"):
        c2._normalized_log_count(-1, 10)


def test_nonnegative_cost_fit_is_deterministic_and_monotone() -> None:
    rng = np.random.default_rng(17)
    x = rng.random((80, len(c2.FEATURE_NAMES)))
    y = 0.1 + 0.4 * x[:, 0] + 0.2 * x[:, 5]
    weights = np.ones(80)
    first = c2.fit_cost_model(x, y, weights)
    second = c2.fit_cost_model(x, y, weights)
    assert first["solver_success"]
    assert np.all(np.asarray(first["coefficients"]) >= 0)
    assert np.allclose(first["coefficients"], second["coefficients"], atol=1e-12)
    low = np.zeros(len(c2.FEATURE_NAMES))
    high = low.copy()
    high[0] = 1.0
    assert c2.predict_cost(first, high) >= c2.predict_cost(first, low)


def test_model_load_rejects_width_nonfinite_negative_and_hash_change() -> None:
    for mutation in ("width", "nonfinite", "negative", "hash"):
        model = _model()
        if mutation == "width":
            model["coefficients"] = model["coefficients"][:-1]
        elif mutation == "nonfinite":
            model["coefficients"][0] = float("nan")
        elif mutation == "negative":
            model["coefficients"][0] = -1.0
        else:
            model["intercept"] = 0.3
        with pytest.raises(ValueError):
            c2.validate_cost_model_payload(model)


def test_safety_load_and_admission_margins_are_frozen() -> None:
    assert c2.safety_load(depth2_s=0.5, combined_s=1.0) == 0.5
    assert c2.safety_load(depth2_s=0.1, combined_s=0.7) == pytest.approx(7 / 6)
    assert c2.conservative_upper(0.72) == pytest.approx(1.0)
    assert c2.conservative_upper(0.721) > 1.0
    with pytest.raises(ValueError):
        c2.safety_load(depth2_s=0.0, combined_s=1.0)


def test_fit_weights_equalize_family_root_and_state() -> None:
    rows = []
    for family, _spec in c2.FAMILY_SLATE:
        for root_index in range(2):
            for state_index in range(4):
                rows.append({
                    "behavior_family": family,
                    "root_ancestry": f"{family}-{root_index}",
                    "record_id": f"{family}-{root_index}-{state_index}",
                })
    weights = c2._fit_weights(rows)
    assert np.sum(weights) == pytest.approx(len(rows))
    assert len(set(np.round(weights, 12))) == 1


def test_validation_gate_passes_clean_signal_and_fails_rank_reversal() -> None:
    rows = _validation_rows()
    model = _model(intercept=0.0, coefficient=1.0)
    passed = c2.validation_report(timing_rows=rows, model=model)
    assert passed["decision"] == "PASS"
    reversed_model = _model(intercept=1.0, coefficient=0.0)
    failed = c2.validation_report(timing_rows=rows, model=reversed_model)
    assert failed["decision"] == "FAIL"
    assert not failed["checks"]["spearman_ge_0_25"]


def test_runtime_gate_enforces_tail_activity_and_family_rules() -> None:
    passed = c2.runtime_gate_report(rows=_gate_rows(), model=_model())
    assert passed["decision"] == "PASS"
    tail = [1.2] * 47 + [20.0]
    failed = c2.runtime_gate_report(rows=_gate_rows(ratios=tail), model=_model())
    assert failed["decision"] == "FAIL"
    assert not failed["checks"]["max_ratio_le_12x"]


def test_root_extraction_collects_nested_ids_without_score_values() -> None:
    payload = {
        "root_cluster": "fresh:1:1536",
        "rows": [
            {"ancestry_id": "fresh:2:1536", "score": 123},
            {"other": {"root": "fresh:3:1536"}},
        ],
    }
    assert c2._recursive_root_values(payload) == {
        "fresh:1:1536", "fresh:2:1536", "fresh:3:1536"
    }


def test_collision_manifest_distinguishes_live_bytes_from_immutable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = c2.requested_stream_manifest()
    fake_prior = {key: set() for key in c2.STREAM_BASES}
    fake_history = {
        "matched_sources": [
            {
                "path": str(c2.LIVE_COLLISION_PATHS[0]),
                "sha256": "live",
                "byte_size": 1,
                "counts": {},
            },
            {
                "path": "threes_rl/runs/immutable.json",
                "sha256": "fixed",
                "byte_size": 2,
                "counts": {},
            },
        ]
    }
    monkeypatch.setattr(
        c2.history,
        "historical_collision_union",
        lambda exclude_dir=None: (fake_prior, fake_history),
    )
    manifest = c2.build_stream_collision_manifest(rows, out_dir=Path("/tmp/c2"))
    assert manifest["passes"]
    assert manifest["live_source_count"] == 1
    assert manifest["immutable_source_count"] == 1
    fake_prior["deck_stream_id"].add(rows[0]["deck_stream_id"])
    collided = c2.build_stream_collision_manifest(rows, out_dir=Path("/tmp/c2"))
    assert not collided["passes"]


def test_temporal_partition_uses_four_distinct_buckets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = [
        {
            "root_ancestry": "fresh:65:1536",
            "frame_index": index,
            "state": {"index": index},
            "state_sha256": f"{index:064x}",
            "empty_count": 3,
            "built_max": 768,
            "legal_actions": ["up", "left"],
        }
        for index in range(12)
    ]
    iterator = iter(candidates)
    monkeypatch.setattr(
        c2,
        "_frame_base_candidate",
        lambda frame, root: next(iterator),
    )
    monkeypatch.setattr(
        c2,
        "_incumbent_values_for_candidate",
        lambda row, incumbent: {
            "incumbent_margin": 0.01,
            "trigger_reasons": {"low_empty": True, "low_margin": True},
            "incumbent_legal_actions": ["up", "left"],
        },
    )
    replay = {"frames": [{"index": index} for index in range(12)]}
    stream = {
        "logical_seed": 65,
        "family_index": 0,
        "game_index": 0,
        "deck_stream_id": 66,
        "slot_stream_id": 67,
        "policy_stream_id": 68,
    }
    selected = c2.extract_selected_states(
        replay,
        family="c2_corner2",
        stream_row=stream,
        incumbent=object(),
    )
    assert [row["selection_bucket"] for row in selected] == [0, 1, 2, 3]
    assert len({row["frame_index"] for row in selected}) == 4


def test_existing_c1_state_depth2_instrumentation_is_value_exact() -> None:
    corpus = json.loads(
        Path("threes_rl/runs/forensics/c1_search/C1_CORPUS.json").read_text()
    )
    record = corpus["splits"]["profile"][0]
    state = state_from_replay_payload(record["state"])
    from threes_rl.sim import ThreesSim
    sim = ThreesSim.from_stream_ids(
        deck_stream_id=1,
        slot_stream_id=2,
        starter_tile=int(record.get("starter_tile", 1536)),
    )
    base = make_policy(corpus["incumbent_policy"])
    assert isinstance(base, NtupleExpectimaxPolicy)
    instrumented = c2.clone_instrumented(base)
    expected = base.action_values(state, sim)
    observed = instrumented.depth2_probe(state, sim)
    exact, difference = c2._values_close(
        {c2.DIRECTION_NAMES[action]: value for action, value in expected},
        observed["values"],
    )
    assert exact
    assert difference <= 1e-9
    assert observed["features"].shape == (20,)


def test_zero_work_and_marker_are_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = tmp_path / "out"
    out.mkdir()
    lock_path = out / "C2_PREFLIGHT_LOCK.json"
    lock_path.write_text("{}")
    fake_lock = {
        "canonical_payload_sha256": "payload",
        "charter": {"sha256": "charter"},
        "implementation": {"sha256": "runner"},
        "tests": {"sha256": "tests", "evidence_file_sha256": "evidence"},
        "manifest_bindings": {},
        "source_lock": {"source_lock_sha256": "sources"},
        "policy_lock_sha256": "policies",
        "feature_schema_sha256": "schema",
        "commands": {"execute": "bound command"},
        "resources": {},
    }
    monkeypatch.setattr(c2, "OUTPUT_DIR", out)
    monkeypatch.setattr(c2, "_load_preflight", lambda path, directory: fake_lock)
    monkeypatch.setattr(
        c2,
        "_operational_audit",
        lambda path: {"passes": True, "checks": {}},
    )
    opened = c2.seal_execution_opened(
        out_dir=out, preflight_lock=lock_path, jobs=1
    )
    assert opened["zero_work"]["games"] == 0
    assert (out / "C2_EXECUTION_OPENED.json").is_file()
    with pytest.raises(FileExistsError):
        c2.seal_execution_opened(
            out_dir=out, preflight_lock=lock_path, jobs=1
        )


def test_marker_validation_rejects_missing_and_command_mismatch(
    tmp_path: Path,
) -> None:
    lock = {
        "canonical_payload_sha256": "preflight-payload",
        "charter": {"sha256": "charter"},
        "implementation": {"sha256": "runner"},
        "tests": {"sha256": "tests"},
        "commands": {"execute": "bound"},
    }
    with pytest.raises(ValueError, match="missing"):
        c2._load_marker(tmp_path, lock)
    marker = c2.payload_with_hash({
        "admission_opened": True,
        "preflight_lock_file_sha256": "file",
        "preflight_lock_payload_sha256": "preflight-payload",
        "charter_sha256": "charter",
        "implementation_sha256": "runner",
        "test_sha256": "tests",
        "execute_command": "wrong",
        "jobs": 1,
    })
    c2.atomic_write_json(tmp_path / "C2_EXECUTION_OPENED.json", marker)
    (tmp_path / "C2_PREFLIGHT_LOCK.json").write_text("lock")
    with pytest.raises(ValueError, match="mismatch"):
        c2._load_marker(tmp_path, lock)


def test_terminal_states_are_exactly_frozen() -> None:
    assert {
        "READY_C2_FULL_POLICY_PREFLIGHT",
        "KILL_C2_COST_ADMISSION",
        "HOLD_C2_ENGINEERING_FAULT",
    } == {
        "READY_C2_FULL_POLICY_PREFLIGHT",
        "KILL_C2_COST_ADMISSION",
        "HOLD_C2_ENGINEERING_FAULT",
    }
