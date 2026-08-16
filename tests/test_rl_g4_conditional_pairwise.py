from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from threes_rl import g4_conditional_pairwise as g4


def _features(record_id: str, actions: list[int]) -> dict[tuple[str, int, str], np.ndarray]:
    result = {}
    for action in actions:
        for horizon_index, horizon in enumerate(g4.HORIZON_NAMES):
            values = np.zeros(g4.FEATURE_WIDTH, dtype=np.float64)
            values[horizon_index] = 1.0
            values[3 + action] = 1.0
            values[7] = action / 3.0
            values[8] = (3 - action) / 3.0
            result[(record_id, action, horizon)] = values
    return result


def _record(
    *,
    record_id: str = "record",
    root: str = "root",
    family: str = "family",
    partition: str = "train",
    scale: str = "pre768",
    actions: list[int] | None = None,
) -> dict[str, object]:
    action_ids = actions or [0, 1, 2]
    return {
        "record_id": record_id,
        "root_cluster": root,
        "behavior_family": family,
        "partition": partition,
        "scale": scale,
        "target": 768 if scale == "pre768" else 1536,
        "state_sha1": f"state-{record_id}",
        "legal_action_ids": action_ids,
        "legal_actions": [g4.CANONICAL_ACTIONS[action] for action in action_ids],
    }


def _path(
    *,
    record: dict[str, object],
    action: int,
    replicate: int,
    intervals: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "task_key": f"{record['record_id']}:{action}:{replicate}",
        "partition": record["partition"],
        "record_id": record["record_id"],
        "root_cluster": record["root_cluster"],
        "behavior_family": record["behavior_family"],
        "scale": record["scale"],
        "target": record["target"],
        "state_sha1": record["state_sha1"],
        "action": g4.CANONICAL_ACTIONS[action],
        "action_id": action,
        "replicate": replicate,
        "logical_seed": 100 + replicate,
        "deck_stream_id": 200 + replicate,
        "slot_stream_id": 300 + replicate,
        "policy_stream_id": 400 + replicate,
        "interval_rows": intervals,
    }


def _observed(horizon: str, event: int) -> dict[str, object]:
    index = g4.HORIZON_NAMES.index(horizon)
    starts = (0, 10, 20)
    ends = (10, 20, 40)
    return {
        "horizon": horizon,
        "start_move": starts[index],
        "end_move": ends[index],
        "observed": True,
        "event": event,
        "censor_move": None,
    }


def _censored(horizon: str) -> dict[str, object]:
    index = g4.HORIZON_NAMES.index(horizon)
    starts = (0, 10, 20)
    ends = (10, 20, 40)
    return {
        "horizon": horizon,
        "start_move": starts[index],
        "end_move": ends[index],
        "observed": False,
        "event": None,
        "censor_move": starts[index] + 3,
    }


def _pair_rows(
    *,
    roots: int = 40,
    families: int = 4,
    partition: str = "train",
) -> list[dict[str, object]]:
    rows = []
    for root_index in range(roots):
        label = root_index % 2
        delta = np.zeros(g4.FEATURE_WIDTH, dtype=np.float64)
        delta[7] = 1.0 if label else -1.0
        delta[8] = 0.25 * ((root_index % 3) - 1)
        rows.append(
            {
                "partition": partition,
                "scale": "pre768" if root_index % 2 == 0 else "pre1536",
                "behavior_family": f"family-{root_index % families}",
                "root_cluster": f"root-{root_index}",
                "record_id": f"record-{root_index}",
                "horizon": "h40",
                "replicate": 0,
                "action_pair": "up:down",
                "action_a_id": 0,
                "action_b_id": 1,
                "label": label,
                "delta": delta,
                "unit_key": "h40:r0",
            }
        )
    return rows


def test_pair_builder_keeps_only_comparable_discordance() -> None:
    record = _record(actions=[0, 1, 2])
    paths = []
    for replicate in (0, 1):
        paths.extend(
            [
                _path(
                    record=record,
                    action=0,
                    replicate=replicate,
                    intervals=[
                        _observed("h10", 1),
                    ],
                ),
                _path(
                    record=record,
                    action=1,
                    replicate=replicate,
                    intervals=[
                        _observed("h10", 0),
                        _censored("h20"),
                    ],
                ),
                _path(
                    record=record,
                    action=2,
                    replicate=replicate,
                    intervals=[
                        _observed("h10", 0),
                        _observed("h20", 0),
                        _observed("h40", 0),
                    ],
                ),
            ]
        )
    rows, audit = g4.build_pair_dataset(
        [record],
        paths,
        _features("record", [0, 1, 2]),
    )
    assert len(rows) == 4
    assert audit["status_counts"]["discordant"] == 4
    assert audit["status_counts"]["concordant_no_event"] == 2
    assert audit["status_counts"]["noncomparable"] > 0
    assert all(np.array_equal(row["delta"][:3], np.zeros(3)) for row in rows)
    assert audit["passes"]


def test_pair_builder_rejects_crn_mismatch() -> None:
    record = _record(actions=[0, 1])
    paths = [
        _path(
            record=record,
            action=action,
            replicate=replicate,
            intervals=[_observed("h10", action)],
        )
        for replicate in (0, 1)
        for action in (0, 1)
    ]
    paths[1]["deck_stream_id"] = 999
    with pytest.raises(ValueError, match="CRN"):
        g4.build_pair_dataset(
            [record],
            paths,
            _features("record", [0, 1]),
        )


def test_pair_delta_reversal_and_canonical_label() -> None:
    record = _record(actions=[0, 1])
    paths = [
        _path(
            record=record,
            action=action,
            replicate=replicate,
            intervals=[_observed("h10", 1 if action == 0 else 0)],
        )
        for replicate in (0, 1)
        for action in (0, 1)
    ]
    feature_map = _features("record", [0, 1])
    rows, _audit = g4.build_pair_dataset([record], paths, feature_map)
    expected = feature_map[("record", 0, "h10")] - feature_map[
        ("record", 1, "h10")
    ]
    assert all(row["label"] == 1 for row in rows)
    assert all(np.array_equal(row["delta"], expected) for row in rows)
    assert np.array_equal(-expected, feature_map[("record", 1, "h10")] - feature_map[("record", 0, "h10")])


def test_hierarchical_weights_balance_family_root_record_unit_pair() -> None:
    rows = _pair_rows(roots=12, families=3)
    extra = dict(rows[0])
    extra["record_id"] = "extra-record"
    extra["replicate"] = 1
    extra["unit_key"] = "h40:r1"
    rows.append(extra)
    weighted = g4.assign_pair_weights(rows)
    assert sum(row["weight"] for row in weighted) == pytest.approx(1.0)
    by_family: dict[str, float] = {}
    by_root_local: dict[str, float] = {}
    for row in weighted:
        by_family[row["behavior_family"]] = (
            by_family.get(row["behavior_family"], 0.0) + row["weight"]
        )
        by_root_local[row["root_cluster"]] = (
            by_root_local.get(row["root_cluster"], 0.0)
            + row["root_local_weight"]
        )
    assert set(round(value, 12) for value in by_family.values()) == {
        round(1 / 3, 12)
    }
    assert set(round(value, 12) for value in by_root_local.values()) == {1.0}


def test_pairwise_model_has_no_intercept_and_round_trips(tmp_path: Path) -> None:
    rows = _pair_rows(roots=40, families=4)
    source_hashes = {"source": "a" * 64}
    model = g4.fit_pairwise_model(
        rows,
        source_hashes=source_hashes,
        pair_dataset_sha256="b" * 64,
    )
    assert model.coefficients[7] > 0
    directory = tmp_path / "model"
    metadata = model.save(directory)
    assert metadata["has_intercept"] is False
    assert metadata["parameter_count"] == 64
    loaded = g4.PairwiseModel.load(
        directory,
        expected_source_hashes=source_hashes,
    )
    matrix = np.stack([row["delta"] for row in rows])
    assert np.array_equal(model.logits(matrix), loaded.logits(matrix))
    assert np.array_equal(model.logits(-matrix), -model.logits(matrix))


def test_pairwise_model_load_rejects_nonfinite_and_schema_mismatch(
    tmp_path: Path,
) -> None:
    rows = _pair_rows(roots=20, families=4)
    model = g4.fit_pairwise_model(
        rows,
        source_hashes={"source": "a" * 64},
        pair_dataset_sha256="b" * 64,
    )
    directory = tmp_path / "model"
    model.save(directory)
    metadata_path = directory / "meta.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["parameter_count"] = 65
    metadata["canonical_payload_sha256"] = g4.canonical_sha256(
        {
            key: value
            for key, value in metadata.items()
            if key != "canonical_payload_sha256"
        }
    )
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="schema"):
        g4.PairwiseModel.load(directory)


def test_root_bootstrap_is_deterministic() -> None:
    rows = _pair_rows(roots=24, families=4)
    model = g4.fit_pairwise_model(
        rows,
        source_hashes={"source": "a" * 64},
        pair_dataset_sha256="b" * 64,
    )
    scored, _summary = g4._metric_rows(model, rows)
    first = g4.root_bootstrap_metrics(scored, seed=7, repeats=100)
    second = g4.root_bootstrap_metrics(scored, seed=7, repeats=100)
    assert first == second


def test_preflight_gate_ready_hold_and_kill() -> None:
    train = {
        "roots": 100,
        "families": 4,
        "roots_by_scale": {"pre768": 50, "pre1536": 50},
        "pairs": 400,
        "max_raw_pair_share_by_root": 0.01,
    }
    development = {
        "roots": 40,
        "families": 3,
        "roots_by_scale": {"pre768": 20, "pre1536": 20},
        "pairs": 160,
        "max_raw_pair_share_by_root": 0.025,
    }
    decision, checks = g4._preflight_decision(
        integrity_checks={"integrity": True},
        train=train,
        development=development,
    )
    assert decision == "READY_G4_SPENT_DIAGNOSTIC"
    assert all(checks.values())
    development["pairs"] = 127
    assert g4._preflight_decision(
        integrity_checks={"integrity": True},
        train=train,
        development=development,
    )[0] == "HOLD_G4_PAIRWISE_UNDERPOWERED"
    assert g4._preflight_decision(
        integrity_checks={"integrity": False},
        train=train,
        development=development,
    )[0] == "KILL_G4_PAIRWISE_INFEASIBLE"


def test_read_only_path_reader_rejects_transfer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "labels.sqlite3"
    connection = sqlite3.connect(database)
    connection.execute(
        "CREATE TABLE paths("
        "task_key TEXT PRIMARY KEY,payload_json TEXT,payload_sha256 TEXT)"
    )
    payload = {
        "task_key": "transfer",
        "partition": "transfer_diagnostic",
    }
    payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    connection.execute(
        "INSERT INTO paths VALUES(?,?,?)",
        (
            "transfer",
            payload_json,
            g4.hashlib.sha256(payload_json.encode("ascii")).hexdigest(),
        ),
    )
    connection.commit()
    connection.close()
    monkeypatch.setattr(g4, "ORDINARY_DB_PATH", database)
    monkeypatch.setattr(g4, "EXPECTED_ORDINARY_PATHS", 1)
    with pytest.raises(ValueError, match="Transfer"):
        g4._read_ordinary_paths()
    assert database.exists()


def test_future_stream_manifest_is_unique_and_unconsumed() -> None:
    rows = g4._future_stream_rows()
    assert len(rows) == 512
    values = [
        row[key]
        for row in rows
        for key in g4.FUTURE_STREAM_BASES
    ]
    assert len(values) == len(set(values))
    assert rows[0]["logical_seed"] == 61_000_000_000
    assert rows[-1]["policy_stream_id"] == 64_000_000_511


def test_exact_power_table_has_monotone_useful_signal() -> None:
    table = g4._power_table()
    powers = [
        row["power_true_concordance_0.65"] for row in table["rows"]
    ]
    assert powers == sorted(powers)
    assert table["smallest_listed_n_with_80pct_power_at_0_65"] in {
        96,
        128,
        192,
        256,
    }


def test_diagnostic_decision_uses_rank_not_calibration() -> None:
    primary = {"log_loss_improvement": 0.1, "concordance": 0.6}
    bootstrap = {
        "metrics": {
            "log_loss_improvement": {
                "lower_95": 0.01,
                "upper_95": 0.2,
            },
            "concordance": {"lower_95": 0.51, "upper_95": 0.7},
        }
    }
    by_scale = {
        "pre768": {
            "pairs": 20,
            "log_loss_improvement": 0.1,
            "concordance": 0.6,
        },
        "pre1536": {
            "pairs": 20,
            "log_loss_improvement": 0.1,
            "concordance": 0.6,
        },
    }
    by_family = {
        "family": {
            "roots": 8,
            "log_loss_improvement": 0.01,
            "concordance": 0.51,
        }
    }
    decision, checks = g4._diagnostic_decision(
        optimizer={"success": True, "gradient_infinity_norm": 1e-6},
        primary=primary,
        bootstrap=bootstrap,
        by_scale=by_scale,
        by_family=by_family,
    )
    assert decision == "SUPPORT_G4_PAIRWISE_MECHANISM_SPENT"
    assert all(checks.values())
    bootstrap["metrics"]["concordance"]["lower_95"] = 0.49
    decision, _checks = g4._diagnostic_decision(
        optimizer={"success": True, "gradient_infinity_norm": 1e-6},
        primary=primary,
        bootstrap=bootstrap,
        by_scale=by_scale,
        by_family=by_family,
    )
    assert decision == "HOLD_G4_PAIRWISE_MECHANISM_AMBIGUOUS"
