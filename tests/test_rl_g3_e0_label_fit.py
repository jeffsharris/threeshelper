from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from threes_rl import g3_e0_label_fit as e0
from threes_rl import g3_e0_preflight as preflight


def _feature(index: int, horizon: int) -> np.ndarray:
    values = np.zeros(e0.FEATURE_WIDTH, dtype=np.float64)
    values[{10: 0, 20: 1, 40: 2}[horizon]] = 1.0
    values[3 + (index % 4)] = 1.0
    values[7] = (index % 5) / 4.0
    values[8] = ((index * 3) % 7) / 6.0
    return values


def _grouped_rows(
    *,
    roots: int = 30,
    families: int = 3,
    scale: str | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for root_index in range(roots):
        local_scale = scale or ("pre768" if root_index % 2 == 0 else "pre1536")
        for action_index, action in enumerate(e0.CANONICAL_ACTIONS[:2]):
            for horizon in e0.HORIZONS:
                signal = (
                    1
                    if action_index == root_index % 2
                    and horizon == 40
                    else 0
                )
                rows.append(
                    {
                        "record_id": f"record-{root_index}",
                        "root_cluster": f"root-{root_index}",
                        "behavior_family": f"family-{root_index % families}",
                        "scale": local_scale,
                        "action": action,
                        "horizon": f"h{horizon}",
                        "features": _feature(action_index, horizon),
                        "events": signal,
                        "trials": 2,
                        "event_fraction": float(signal),
                    }
                )
    e0.assign_group_weights(rows, family_balanced=True)
    return rows


@pytest.mark.parametrize(
    ("event", "terminal", "completed", "expected"),
    [
        (
            7,
            None,
            7,
            [("h10", True, 1, None)],
        ),
        (
            20,
            None,
            20,
            [("h10", True, 0, None), ("h20", True, 1, None)],
        ),
        (
            None,
            7,
            7,
            [("h10", False, None, 7)],
        ),
        (
            None,
            10,
            10,
            [("h10", True, 0, None)],
        ),
        (
            None,
            25,
            25,
            [
                ("h10", True, 0, None),
                ("h20", True, 0, None),
                ("h40", False, None, 25),
            ],
        ),
        (
            None,
            None,
            40,
            [
                ("h10", True, 0, None),
                ("h20", True, 0, None),
                ("h40", True, 0, None),
            ],
        ),
    ],
)
def test_event_censor_rows(
    event: int | None,
    terminal: int | None,
    completed: int,
    expected: list[tuple[str, bool, int | None, int | None]],
) -> None:
    rows = e0.event_censor_rows(
        event_move=event,
        terminal_move=terminal,
        completed_moves=completed,
    )
    assert [
        (
            row["horizon"],
            row["observed"],
            row["event"],
            row["censor_move"],
        )
        for row in rows
    ] == expected


def test_event_wins_when_terminal_coincides() -> None:
    assert e0.event_censor_rows(
        event_move=10,
        terminal_move=10,
        completed_moves=10,
    ) == [
        {
            "horizon": "h10",
            "start_move": 0,
            "end_move": 10,
            "observed": True,
            "event": 1,
            "censor_move": None,
        }
    ]


def test_event_censor_rejects_incomplete_path() -> None:
    with pytest.raises(ValueError, match="Incomplete path"):
        e0.event_censor_rows(
            event_move=None,
            terminal_move=None,
            completed_moves=9,
        )


def test_build_tasks_filters_e0_and_shares_action_tapes() -> None:
    records = [
        {
            "record_id": "r",
            "partition": "train",
            "root_cluster": "root",
            "behavior_family": "family",
            "scale": "pre768",
            "target": 768,
            "state_sha1": "state",
            "legal_action_ids": [0, 2],
            "legal_actions": ["up", "left"],
        }
    ]
    streams = []
    for action_id, action in ((0, "up"), (2, "left")):
        for replicate in range(8):
            streams.append(
                {
                    "record_ordinal": 0,
                    "partition": "train",
                    "record_id": "r",
                    "root_cluster": "root",
                    "behavior_family": "family",
                    "scale": "pre768",
                    "target": 768,
                    "state_sha1": "state",
                    "action_id": action_id,
                    "action": action,
                    "replicate": replicate,
                    "logical_seed": 57_000_000_000 + replicate,
                    "deck_stream_id": 58_000_000_000 + replicate,
                    "slot_stream_id": 59_000_000_000 + replicate,
                    "policy_stream_id": 60_000_000_000 + replicate,
                }
            )
    tasks = e0.build_e0_tasks(records, streams)
    assert len(tasks) == 4
    assert [task["replicate"] for task in tasks] == [0, 0, 1, 1]
    assert e0.task_coupling_audit(tasks)["passes"]


def test_family_balanced_weights_are_root_equal_within_family() -> None:
    rows = _grouped_rows(roots=12, families=3)
    audit = e0.weight_audit(rows, family_balanced=True)
    assert audit["passes"]
    assert audit["total_weight"] == pytest.approx(1.0)
    assert set(round(value, 12) for value in audit["family_weights"].values()) == {
        round(1 / 3, 12)
    }
    roots = audit["root_weights"]
    assert roots["root-0"] == pytest.approx(1 / 12)


def test_root_equal_weights_ignore_family_size() -> None:
    rows = _grouped_rows(roots=11, families=2)
    e0.assign_group_weights(rows, family_balanced=False)
    audit = e0.weight_audit(rows, family_balanced=False)
    assert audit["passes"]
    assert set(round(value, 12) for value in audit["root_weights"].values()) == {
        round(1 / 11, 12)
    }


def test_label_store_resume_and_conflict(tmp_path: Path) -> None:
    path = tmp_path / "labels.sqlite3"
    identity = {"manifest": "a"}
    row = {"task_key": "task", "event_move": 10}
    with e0.LabelStore(path, identity=identity) as store:
        store.insert_chunk([row])
        store.insert_chunk([row])
        assert store.count() == 1
    with e0.LabelStore(path, identity=identity) as store:
        assert store.completed_keys() == {"task"}
        with pytest.raises(ValueError, match="Conflicting"):
            store.insert_chunk([{"task_key": "task", "event_move": 20}])
    with pytest.raises(ValueError, match="identity"):
        e0.LabelStore(path, identity={"manifest": "b"})


def test_transfer_store_requires_both_seals(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="checkpoint and prediction"):
        e0.LabelStore(
            tmp_path / "transfer.sqlite3",
            identity={"a": 1},
            transfer=True,
        )
    checkpoint = tmp_path / "checkpoint.json"
    prediction = tmp_path / "prediction.json"
    checkpoint.write_text("{}\n")
    prediction.write_text("{}\n")
    with e0.LabelStore(
        tmp_path / "transfer.sqlite3",
        identity={"a": 1},
        transfer=True,
        checkpoint_seal=checkpoint,
        prediction_seal=prediction,
    ) as store:
        assert store.count() == 0


def test_scientific_partitions_do_not_cross_store_boundary(
    tmp_path: Path,
) -> None:
    ordinary = tmp_path / "ordinary.sqlite3"
    transfer = tmp_path / "transfer.sqlite3"
    checkpoint = tmp_path / "checkpoint.json"
    prediction = tmp_path / "prediction.json"
    checkpoint.write_text("{}\n")
    prediction.write_text("{}\n")
    with e0.LabelStore(ordinary, identity={"partition": "ordinary"}) as store:
        store.insert_chunk([{"task_key": "ordinary", "partition": "train"}])
    with e0.LabelStore(
        transfer,
        identity={"partition": "transfer"},
        transfer=True,
        checkpoint_seal=checkpoint,
        prediction_seal=prediction,
    ) as store:
        store.insert_chunk(
            [{"task_key": "transfer", "partition": "transfer_diagnostic"}]
        )
    with e0.LabelStore(ordinary, identity={"partition": "ordinary"}) as store:
        assert [row["partition"] for row in store.rows()] == ["train"]


def test_deterministic_fit_and_serialization(tmp_path: Path) -> None:
    train = _grouped_rows(roots=36, families=3)
    development = _grouped_rows(roots=18, families=3)
    e0.assign_group_weights(development, family_balanced=False)
    first = e0.fit_hazard_model(
        train, development, source_hashes={"fixture": "one"}
    )
    second = e0.fit_hazard_model(
        train, development, source_hashes={"fixture": "one"}
    )
    assert np.array_equal(first.coefficients, second.coefficients)
    assert first.intercept == second.intercept
    assert first.calibration_intercept == second.calibration_intercept
    model_dir = tmp_path / "model"
    first.save(model_dir)
    loaded = e0.G3HazardModel.load(
        model_dir, expected_source_hashes={"fixture": "one"}
    )
    sample = np.stack([_feature(0, 10), _feature(1, 40)])
    assert np.array_equal(first.predict(sample), loaded.predict(sample))
    with pytest.raises(ValueError, match="source hash"):
        e0.G3HazardModel.load(
            model_dir, expected_source_hashes={"fixture": "other"}
        )


def test_model_loader_rejects_changed_width(tmp_path: Path) -> None:
    train = _grouped_rows(roots=12, families=3)
    development = _grouped_rows(roots=12, families=3)
    e0.assign_group_weights(development, family_balanced=False)
    model = e0.fit_hazard_model(train, development, source_hashes={"a": "b"})
    model_dir = tmp_path / "model"
    model.save(model_dir)
    meta_path = model_dir / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["feature_width"] = 63
    meta["canonical_payload_sha256"] = e0.canonical_sha256(
        {
            key: value
            for key, value in meta.items()
            if key != "canonical_payload_sha256"
        }
    )
    meta_path.write_text(json.dumps(meta, sort_keys=True))
    with pytest.raises(ValueError, match="width"):
        e0.G3HazardModel.load(model_dir)


def test_action_tie_rule_and_rank() -> None:
    values = {"right": 0.8, "left": 0.8, "up": 0.1}
    assert e0.choose_action(values) == "left"
    assert e0.choose_action(
        {"up": 0.8 - 0.5e-12, "left": 0.8}
    ) == "up"
    assert e0.spearman_average_ties([0, 1, 2], [0, 1, 2]) == pytest.approx(
        1.0
    )
    assert e0.spearman_average_ties([1, 1], [0, 1]) is None


def test_bootstrap_is_deterministic() -> None:
    train = _grouped_rows(roots=18, families=3)
    development = _grouped_rows(roots=12, families=3)
    e0.assign_group_weights(development, family_balanced=False)
    model = e0.fit_hazard_model(train, development, source_hashes={"a": "b"})
    first = e0.bootstrap_metric_improvement(
        development, model, seed=91, repeats=20
    )
    second = e0.bootstrap_metric_improvement(
        development, model, seed=91, repeats=20
    )
    assert first == second


def _ordinary_report(*, passing: bool) -> dict[str, object]:
    value = 0.1 if passing else -0.1
    return {
        "overall": {
            "log_loss_improvement": value,
            "brier_improvement": value,
        },
        "scales": {
            scale: {
                "log_loss_improvement": value,
                "brier_improvement": value,
            }
            for scale in ("pre768", "pre1536")
        },
        "bootstrap": {
            "log_loss_improvement_ci95": [value, value + 0.1],
            "brier_improvement_ci95": [value, value + 0.1],
        },
        "rank": {
            "overall": value,
            "by_scale": {"pre768": value, "pre1536": value},
            "informative_records": 20,
            "informative_by_scale": {"pre768": 5, "pre1536": 5},
        },
        "families": {
            "a": {
                "roots": 10,
                "log_loss_improvement": value,
                "brier_improvement": value,
            }
        },
    }


def test_ordinary_gate_ready_and_kill() -> None:
    assert e0.ordinary_gate_decision(
        _ordinary_report(passing=True),
        integrity_passes=True,
        model_stable=True,
    ) == "READY_G3_E0_ORDINARY_PREDICTIVE"
    assert e0.ordinary_gate_decision(
        _ordinary_report(passing=False),
        integrity_passes=True,
        model_stable=True,
    ) == "KILL_G3_BOOTSTRAP_PREDICTIVE"
    assert e0.ordinary_gate_decision(
        _ordinary_report(passing=True),
        integrity_passes=False,
        model_stable=True,
    ) == "KILL_G3_BOOTSTRAP_PREDICTIVE"


def _transfer_report(value: float) -> dict[str, object]:
    families = {
        family: {
            "log_loss_improvement": value,
            "brier_improvement": value,
        }
        for family in (
            "g2_transfer_corner2",
            "g2_transfer_phaseblend_incumbent",
        )
    }
    return {
        "pooled": {
            "log_loss_improvement": value,
            "brier_improvement": value,
        },
        "families": families,
        "rank": {
            "overall": value,
            "by_family": {
                family: value for family in families
            },
        },
    }


def test_transfer_gate_distinguishes_ready_from_underpowered() -> None:
    activity = {"roots": 6, "corner2": 1, "incumbent": 1}
    assert e0.transfer_gate_decision(
        _transfer_report(0.1),
        activity=activity,
        integrity_passes=True,
    ) == "READY_G3_E1_COMPLETION"
    assert e0.transfer_gate_decision(
        _transfer_report(-0.1),
        activity=activity,
        integrity_passes=True,
    ) == "HOLD_G3_E0_UNDERPOWERED_TRANSFER"
    assert e0.transfer_gate_decision(
        _transfer_report(0.1),
        activity={"roots": 5, "corner2": 1, "incumbent": 1},
        integrity_passes=True,
    ) == "HOLD_G3_E0_UNDERPOWERED_TRANSFER"


def test_execution_marker_is_one_shot(tmp_path: Path) -> None:
    identity = {"preflight": "hash"}
    marker = e0.seal_execution_opened(
        tmp_path, identity=identity, command=["frozen"]
    )
    assert e0.verify_payload_hash(marker)
    with pytest.raises(FileExistsError, match="already opened"):
        e0.seal_execution_opened(
            tmp_path, identity=identity, command=["frozen"]
        )


def test_execution_rejects_jobs_before_opening(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="jobs=1"):
        e0.execute(
            out_dir=tmp_path,
            preflight_lock=tmp_path / "missing.json",
            jobs=2,
            command=["bad"],
        )
    assert not (tmp_path / e0.OPEN_MARKER_NAME).exists()


def test_collision_audit_finds_external_and_excludes_exact_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = tmp_path / "runs"
    excluded = runs / "excluded"
    external = runs / "external"
    excluded.mkdir(parents=True)
    external.mkdir(parents=True)
    payload = {
        "logical_seed": 57_000_000_000,
        "deck_stream_id": 58_000_000_000,
        "slot_stream_id": 59_000_000_000,
        "policy_stream_id": 60_000_000_000,
    }
    (excluded / "lock.json").write_text(json.dumps(payload))
    (external / "lock.json").write_text(json.dumps(payload))
    monkeypatch.setattr(preflight, "RUNS_ROOT", runs)
    tasks = [payload]
    audit = preflight.historical_collision_audit(
        tasks, excluded_exact_directories=(excluded,)
    )
    assert not audit["passes"]
    (external / "lock.json").unlink()
    audit = preflight.historical_collision_audit(
        tasks, excluded_exact_directories=(excluded,)
    )
    assert audit["passes"]


def test_preflight_atomic_promotion_with_no_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_dir = tmp_path / "preflight"
    simple = (
        {
            "version": preflight.VERSION,
            "decision": "READY_G3_E0_LABEL_FIT_EXECUTION",
            "out_dir_resolved": str(out_dir.resolve()),
            "zero_forbidden_work": {
                "label_paths_generated": 0,
                "scientific_models_fit": 0,
            },
        },
        e0.payload_with_hash({"records": []}),
        e0.payload_with_hash({"rows": []}),
        e0.payload_with_hash({"tasks": []}),
    )
    monkeypatch.setattr(
        preflight, "build_preflight_payload", lambda out_dir: simple
    )
    lock = preflight.run_preflight(out_dir)
    assert lock["decision"] == "READY_G3_E0_LABEL_FIT_EXECUTION"
    assert (out_dir / "preflight_lock.json").is_file()
    assert not out_dir.with_name(out_dir.name + ".staging").exists()
    assert not (out_dir / e0.ORDINARY_DB_NAME).exists()
    assert not (out_dir / e0.MODEL_DIR_NAME).exists()


def test_preflight_failure_seals_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_dir = tmp_path / "preflight"

    def fail(*, out_dir: Path) -> object:
        raise RuntimeError(f"failed {out_dir.name}")

    monkeypatch.setattr(preflight, "build_preflight_payload", fail)
    with pytest.raises(RuntimeError, match="failed preflight"):
        preflight.run_preflight(out_dir)
    staging = out_dir.with_name(out_dir.name + ".staging")
    failure = json.loads((staging / "PREFLIGHT_FAILURE.json").read_text())
    assert failure["decision"] == "KILL_G3_E0_PREFLIGHT_INTEGRITY"
    assert failure["zero_forbidden_work"]["label_paths_generated"] == 0


def test_real_manifests_produce_exact_e0_counts_without_outcomes() -> None:
    record_manifest = e0.json_object(preflight.V1_RECORD_MANIFEST_PATH)
    stream_manifest = e0.json_object(preflight.V1_STREAM_MANIFEST_PATH)
    tasks = e0.build_e0_tasks(
        record_manifest["records"], stream_manifest["rows"]
    )
    assert len(tasks) == preflight.EXPECTED_E0_PATHS
    assert Counter(task["partition"] for task in tasks) == Counter(
        preflight.EXPECTED_E0_BY_PARTITION
    )
    assert e0.task_coupling_audit(tasks)["passes"]


def test_one_real_record_feature_digest_matches_frozen_manifest() -> None:
    manifest = e0.json_object(preflight.V1_RECORD_MANIFEST_PATH)
    record = manifest["records"][0]
    _rows, digest = e0.feature_rows_for_record(record)
    assert digest == record["feature_rows_sha256"]


def test_upstream_seals_and_schema_are_exact() -> None:
    audit = preflight.upstream_input_audit()
    assert audit["passes"]
