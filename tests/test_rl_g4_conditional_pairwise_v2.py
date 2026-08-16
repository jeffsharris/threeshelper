from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from threes_rl import g4_conditional_pairwise as v1
from threes_rl import g4_conditional_pairwise_v2 as v2


def _roots(count: int = 352) -> list[dict[str, object]]:
    rows = []
    families = (
        "phaseblend_incumbent_lineage",
        "corner2_lineage",
        "legacy_learned_lineage",
        "expectimax_baseline",
        "phaseblend_cheap_lineage",
    )
    for index in range(count):
        scales = (
            "pre1536+pre768"
            if index % 11 == 0
            else ("pre768" if index % 3 else "pre1536")
        )
        rows.append(
            {
                "root_cluster": f"root-{index:04d}",
                "behavior_family": families[index % len(families)],
                "partition": "train" if index % 5 else "development",
                "scale_signature": scales,
                "record_count": 1 + (index % 2),
            }
        )
    return rows


def _pair_rows(
    *,
    roots: int = 40,
    families: int = 4,
) -> list[dict[str, object]]:
    rows = []
    for root_index in range(roots):
        for unit_index in range(1 + root_index % 3):
            label = (root_index + unit_index) % 2
            delta = np.zeros(v1.FEATURE_WIDTH, dtype=np.float64)
            delta[7] = 1.0 if label else -1.0
            rows.append(
                {
                    "partition": (
                        "train" if root_index % 5 else "development"
                    ),
                    "scale": (
                        "pre768" if root_index % 2 == 0 else "pre1536"
                    ),
                    "behavior_family": f"family-{root_index % families}",
                    "root_cluster": f"root-{root_index}",
                    "record_id": f"record-{root_index}",
                    "horizon": ("h10", "h20", "h40")[unit_index % 3],
                    "replicate": unit_index % 2,
                    "action_pair": "up:down",
                    "action_a_id": 0,
                    "action_b_id": 1,
                    "label": label,
                    "delta": delta,
                    "unit_key": (
                        f"{('h10', 'h20', 'h40')[unit_index % 3]}:"
                        f"r{unit_index % 2}"
                    ),
                    "fold": root_index % 5,
                    "logit": 1.0 if label else -1.0,
                    "concordance": 1.0,
                }
            )
    return rows


def _support() -> dict[str, object]:
    family_counts = {
        "phaseblend_incumbent_lineage": 154,
        "corner2_lineage": 14,
        "legacy_learned_lineage": 12,
        "expectimax_baseline": 3,
        "phaseblend_cheap_lineage": 2,
    }
    return {
        "pair_dataset_sha256": v2.V1_PAIR_DATASET_SHA256,
        "total_pairs": 727,
        "total_informative_roots": 185,
        "informative_roots_by_scale": {
            "pre768": 162,
            "pre1536": 45,
        },
        "informative_roots_by_family": family_counts,
        "support_eligible_families": {
            key: value
            for key, value in family_counts.items()
            if value >= 8
        },
        "conservative_max_raw_root_share": 13 / 727,
        "power": {
            "overall": {
                "mde_true_root_direction_rate": 0.6058598391551271
            }
        },
    }


def test_fold_assignments_are_deterministic_and_balanced() -> None:
    first, first_audit = v2.build_fold_assignments(_roots())
    second, second_audit = v2.build_fold_assignments(_roots())
    assert first == second
    assert first_audit == second_audit
    assert first_audit["passes"]
    assert len({row["root_cluster"] for row in first}) == 352
    assert all(
        row["max_minus_min"] <= 1
        for row in first_audit["stratum_balance"]
    )


def test_fold_assignment_changes_only_with_root_metadata() -> None:
    roots = _roots()
    baseline, _audit = v2.build_fold_assignments(roots)
    mutated = [dict(row) for row in roots]
    mutated[0]["unused_outcome"] = 1
    repeated, _audit = v2.build_fold_assignments(mutated)
    assert baseline == repeated
    mutated[0]["behavior_family"] = "different_family"
    changed, _audit = v2.build_fold_assignments(mutated)
    assert baseline != changed


def test_exact_mde_is_frozen_at_available_root_count() -> None:
    assert v2._minimum_detectable_rate(185) == pytest.approx(
        0.6058598391551271
    )
    assert v2._minimum_detectable_rate(45) == pytest.approx(
        0.7118975492370878
    )


def test_prefit_support_ready_and_underpowered() -> None:
    support = _support()
    checks = v2._prefit_support_checks(
        support,
        root_metadata_audit={"passes": True},
    )
    assert all(checks.values())
    support["informative_roots_by_scale"]["pre1536"] = 31
    checks = v2._prefit_support_checks(
        support,
        root_metadata_audit={"passes": True},
    )
    assert checks["each_scale_at_least_32"] is False


def test_root_balanced_summary_gives_every_root_equal_total() -> None:
    rows = _pair_rows(roots=20)
    summary = v2.root_balanced_summary(rows)
    assert summary["roots"] == 20
    assert summary["root_direction_rate"] == 1.0
    assert summary["continuous_root_concordance"] == 1.0
    weights = v2._root_local_weights(rows)
    by_root: dict[str, float] = {}
    for row in weights:
        by_root[row["root_cluster"]] = (
            by_root.get(row["root_cluster"], 0.0)
            + row["evaluation_weight"]
        )
    assert set(round(value, 12) for value in by_root.values()) == {
        round(1 / 20, 12)
    }


def test_root_direction_uses_balanced_within_root_majority() -> None:
    rows = _pair_rows(roots=2)
    for row in rows:
        if row["root_cluster"] == "root-0":
            row["concordance"] = 0.0
    values = v2._root_values(rows)
    assert values["root-0"]["direction"] == 0.0
    assert values["root-1"]["direction"] == 1.0
    assert v2.root_balanced_summary(rows)["root_direction_rate"] == 0.5


def test_root_bootstrap_is_deterministic() -> None:
    rows = _pair_rows(roots=30)
    first = v2.root_direction_bootstrap(rows, seed=11, repeats=100)
    second = v2.root_direction_bootstrap(rows, seed=11, repeats=100)
    assert first == second
    assert first["lower_95"] == 1.0


def test_terminal_pass_requires_material_floor_and_both_scales() -> None:
    primary = {"root_direction_rate": 0.65}
    bootstrap = {"lower_95": 0.55}
    by_scale = {
        "pre768": {"root_direction_rate": 0.60},
        "pre1536": {"root_direction_rate": 0.55},
    }
    eligible = {"family-a": 100, "family-b": 20, "family-c": 10}
    by_family = {
        family: {"root_direction_rate": 0.55} for family in eligible
    }
    decision, checks = v2._terminal_decision(
        primary=primary,
        bootstrap=bootstrap,
        by_scale=by_scale,
        by_family=by_family,
        eligible_families=eligible,
        material_floor=0.605,
        model_checks={"models": True},
    )
    assert decision == "READY_G4_FRESH_ROOT_ACQUISITION_PREFLIGHT"
    assert all(
        value
        for key, value in checks.items()
        if isinstance(value, bool)
    )
    by_scale["pre1536"]["root_direction_rate"] = 0.5
    decision, _checks = v2._terminal_decision(
        primary=primary,
        bootstrap=bootstrap,
        by_scale=by_scale,
        by_family=by_family,
        eligible_families=eligible,
        material_floor=0.605,
        model_checks={"models": True},
    )
    assert decision == "KILL_G4_PAIRWISE_MECHANISM"


def test_fold_models_score_antisymmetrically() -> None:
    rows = _pair_rows(roots=40)
    model = v1.fit_pairwise_model(
        rows,
        source_hashes={"source": "a" * 64},
        pair_dataset_sha256="b" * 64,
    )
    heldout = rows[:10]
    scored = v2._score_heldout_rows(model, heldout, fold=0)
    assert all(row["concordance"] == 1.0 for row in scored)
    matrix = np.stack([row["delta"] for row in heldout])
    assert np.allclose(
        model.logits(-matrix),
        -model.logits(matrix),
        rtol=0.0,
        atol=1e-12,
    )


def test_prediction_payload_contains_no_score_or_source_outcome() -> None:
    payload = v2._prediction_payload_rows(_pair_rows(roots=2))
    forbidden = {"score", "final_score", "source_replay", "event_move"}
    assert payload
    assert all(forbidden.isdisjoint(row) for row in payload)


def test_prefit_does_not_open_pair_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = tmp_path / "g4v2"
    evidence_path = tmp_path / "tests.json"
    evidence_path.write_text("{}")
    monkeypatch.setattr(v2, "TEST_EVIDENCE_PATH", evidence_path)
    monkeypatch.setattr(
        v2,
        "_source_audit",
        lambda: {"passes": True, "g3_transfer_access": 0},
    )
    monkeypatch.setattr(v2, "_ordinary_root_metadata", lambda: _roots())
    monkeypatch.setattr(v2, "_v1_support_summary", _support)
    monkeypatch.setattr(
        v2,
        "_load_test_evidence",
        lambda: {
            "canonical_payload_sha256": "evidence",
            "focused_failed": 0,
            "applicable_regressions_failed": 0,
        },
    )
    monkeypatch.setattr(
        v1,
        "_operational_audit",
        lambda: {"passes": True},
    )
    monkeypatch.setattr(
        v1,
        "_build_dataset",
        lambda: (_ for _ in ()).throw(
            AssertionError("pair outcomes opened during prefit")
        ),
    )
    lock = v2.run_prefit(out_dir)
    assert lock["decision"] == "READY_G4_V2_CROSSFIT_EXECUTION"
    assert lock["forbidden_work"]["pair_outcomes_reopened"] == 0
    assert not (out_dir / v2.OPEN_MARKER_NAME).exists()


def test_open_is_marker_only_and_one_shot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = tmp_path / "g4v2"
    out_dir.mkdir()
    lock_path = out_dir / v2.PREFIT_LOCK_NAME
    lock_path.write_text("{}")
    lock = {
        "canonical_payload_sha256": "lock-payload",
        "fold_manifest_file_sha256": "fold",
        "execute_command": "bound-command",
    }
    monkeypatch.setattr(
        v2,
        "_validate_prefit_lock",
        lambda *_args, **_kwargs: lock,
    )
    monkeypatch.setattr(
        v1,
        "_operational_audit",
        lambda: {"passes": True},
    )
    marker = v2.open_execution(out_dir, lock_path)
    assert marker["zero_work_before_marker"]["models"] == 0
    assert (out_dir / v2.OPEN_MARKER_NAME).is_file()
    assert not (out_dir / v2.MODELS_DIR_NAME).exists()
    assert not (out_dir / v2.PREDICTION_NAME).exists()
    with pytest.raises(FileExistsError, match="one-shot"):
        v2.open_execution(out_dir, lock_path)


def test_marker_validation_rejects_command_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = tmp_path / "g4v2"
    out_dir.mkdir()
    lock_path = out_dir / v2.PREFIT_LOCK_NAME
    lock_path.write_text("{}")
    lock = {
        "canonical_payload_sha256": "payload",
        "fold_manifest_file_sha256": "fold",
        "execute_command": "expected",
    }
    monkeypatch.setattr(
        v2,
        "_validate_prefit_lock",
        lambda *_args, **_kwargs: lock,
    )
    marker = {
        "version": f"{v2.VERSION}_execution_opened",
        "prefit_lock_file_sha256": v2.sha256_path(lock_path),
        "prefit_lock_payload_sha256": "payload",
        "fold_manifest_file_sha256": "fold",
        "pair_dataset_sha256": v2.V1_PAIR_DATASET_SHA256,
        "implementation_sha256": v2.sha256_path(Path(v2.__file__)),
        "test_sha256": v2.sha256_path(v2.TEST_PATH),
        "execute_command": "wrong",
    }
    marker["canonical_payload_sha256"] = v2.canonical_sha256(marker)
    (out_dir / v2.OPEN_MARKER_NAME).write_text(json.dumps(marker))
    with pytest.raises(ValueError, match="command"):
        v2._validate_marker(out_dir, lock_path)


def test_allowed_terminal_states_are_exhaustive() -> None:
    text = v2.AMENDMENT_PATH.read_text()
    for state in (
        "HOLD_G4_V2_CROSSFIT_UNDERPOWERED",
        "KILL_G4_PAIRWISE_MECHANISM",
        "READY_G4_FRESH_ROOT_ACQUISITION_PREFLIGHT",
    ):
        assert state in text
    assert "PROMOTE=false" in text
