"""G3 E0 breadth-first hazard labels and the frozen logistic fit.

Scientific execution is gated by a separately sealed preflight lock. Importing
this module, running its tests, and running the preflight do not generate label
paths or fit a scientific model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.stats import rankdata

from threes_rl.eval import make_policy, max_tile_excluding_initial_starter
from threes_rl.g2_scale_relational_hazard import (
    feature_vector,
    schema_manifest,
    schema_sha256,
)
from threes_rl.s3_power_preflight import sha256_path
from threes_rl.sim import DIRECTION_NAMES, ThreesSim
from threes_rl.train_td import state_from_replay_payload


VERSION = "g3_e0_label_fit_v1"
CHARTER_PATH = Path("threes_rl/G3_E0_LABEL_FIT_EXECUTION_CHARTER.md")
CHARTER_SHA256 = (
    "78c7a83601f71de46e0ea53db98023eef12fe16d2f024362d33fd710c82d0591"
)
SCHEMA_SHA256 = (
    "6af0cd515e5886b5fd8bc4d9f52cc9202bd3ed1f149d0ae146829681aea8340e"
)
FEATURE_WIDTH = 64
HORIZONS = (10, 20, 40)
INTERVALS = ((0, 10), (10, 20), (20, 40))
HORIZON_NAMES = ("h10", "h20", "h40")
CANONICAL_ACTIONS = tuple(DIRECTION_NAMES)
E0_REPLICATES = (0, 1)
BOOTSTRAP_REPEATS = 10_000
DEV_BOOTSTRAP_SEED = 2_026_072_601
TRANSFER_BOOTSTRAP_SEED = 2_026_072_602
MODEL_TIE_TOLERANCE = 1e-12
L2_LAMBDA = 1.0
MAX_OPTIMIZER_ITERATIONS = 500
OPTIMIZER_GTOL = 1e-8
OUTPUT_DIR = Path("threes_rl/runs/forensics/g3_e0_label_fit_v1")
OPEN_MARKER_NAME = "G3_E0_EXECUTION_OPENED.json"
CHECKPOINT_SEAL_NAME = "G3_E0_CHECKPOINT_SEALED.json"
PREDICTION_SEAL_NAME = "G3_E0_TRANSFER_PREDICTIONS_SEALED.json"
TERMINAL_RESULT_NAME = "G3_E0_TERMINAL_RESULT.json"
ORDINARY_DB_NAME = "ordinary_labels.sqlite3"
TRANSFER_DB_NAME = "transfer_labels.sqlite3"
MODEL_DIR_NAME = "checkpoint"
MIN_NICE = 10
MAX_CHUNK_SIZE = 8
MIN_FREE_GIB = 100.0
MAX_OUTPUT_BYTES = 4 * 1024**3


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def payload_with_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["canonical_payload_sha256"] = canonical_sha256(result)
    return result


def verify_payload_hash(
    payload: Mapping[str, Any],
    *,
    field: str = "canonical_payload_sha256",
) -> bool:
    expected = payload.get(field)
    if not isinstance(expected, str):
        return False
    return canonical_sha256(
        {key: value for key, value in payload.items() if key != field}
    ) == expected


def event_censor_rows(
    *,
    event_move: int | None,
    terminal_move: int | None,
    completed_moves: int,
) -> list[dict[str, Any]]:
    """Apply the frozen interval event/censor arithmetic."""

    event = None if event_move is None else int(event_move)
    terminal = None if terminal_move is None else int(terminal_move)
    completed = int(completed_moves)
    if event is not None and not 1 <= event <= 40:
        raise ValueError("event_move must be in 1..40")
    if terminal is not None and not 1 <= terminal <= 40:
        raise ValueError("terminal_move must be in 1..40")
    if not 0 <= completed <= 40:
        raise ValueError("completed_moves must be in 0..40")
    if event is not None and event > completed:
        raise ValueError("event_move exceeds completed moves")
    if terminal is not None and terminal > completed:
        raise ValueError("terminal_move exceeds completed moves")

    rows: list[dict[str, Any]] = []
    for name, (start, end) in zip(HORIZON_NAMES, INTERVALS):
        prior_event = event is not None and event <= start
        prior_terminal = terminal is not None and terminal <= start
        if prior_event or prior_terminal:
            break
        if event is not None and start < event <= end:
            rows.append(
                {
                    "horizon": name,
                    "start_move": start,
                    "end_move": end,
                    "observed": True,
                    "event": 1,
                    "censor_move": None,
                }
            )
            break
        if terminal is not None and terminal < end:
            rows.append(
                {
                    "horizon": name,
                    "start_move": start,
                    "end_move": end,
                    "observed": False,
                    "event": None,
                    "censor_move": terminal,
                }
            )
            break
        if completed >= end:
            rows.append(
                {
                    "horizon": name,
                    "start_move": start,
                    "end_move": end,
                    "observed": True,
                    "event": 0,
                    "censor_move": None,
                }
            )
            if terminal == end:
                break
            continue
        raise ValueError("Incomplete path without event or terminal censoring")
    return rows


def _find_replay_frame(
    replay: Mapping[str, Any], frame_index: int
) -> dict[str, Any]:
    matches = []
    for fallback, frame in enumerate(replay.get("frames", [])):
        if (
            isinstance(frame, dict)
            and int(frame.get("index", fallback)) == int(frame_index)
            and isinstance(frame.get("state"), dict)
        ):
            matches.append(frame["state"])
    if len(matches) != 1:
        raise ValueError(
            f"Expected one source frame {frame_index}, found {len(matches)}"
        )
    return dict(matches[0])


def load_record_state(record: Mapping[str, Any]) -> dict[str, Any]:
    if record.get("source_state"):
        state_artifact = json_object(Path(str(record["source_state"])))
        payload = state_artifact.get("state")
        if not isinstance(payload, dict):
            raise ValueError("Transfer source state has no state payload")
        return dict(payload)
    replay = json_object(Path(str(record["source_replay"])))
    return _find_replay_frame(replay, int(record["source_frame_index"]))


def _state_fingerprint(state: Any) -> str:
    payload = {
        "board": np.asarray(state.board, dtype=np.int32).tolist(),
        "preview": {
            "kind": state.preview.kind,
            "value": state.preview.value,
            "candidates": list(state.preview.candidates),
        },
        "small_counts": dict(sorted(state.small_counts.items())),
        "small_pos": int(state.small_pos),
        "small_seen_total": int(state.small_seen_total),
        "span_small_pos": int(state.span_small_pos),
        "large_pending": bool(state.large_pending),
        "max_tile": int(state.max_tile),
        "move_count": int(state.move_count),
        "game_over": bool(state.game_over),
    }
    return canonical_sha256(payload)


def feature_rows_for_record(
    record: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], str]:
    payload = load_record_state(record)
    state = state_from_replay_payload(payload)
    before = _state_fingerprint(state)
    sim = ThreesSim.from_stream_ids(
        deck_stream_id=2_026_072_541,
        slot_stream_id=2_026_072_542,
        starter_tile=int(record["starter_tile"]),
    )
    deck_before = canonical_sha256(sim.deck_rng.bit_generator.state)
    slot_before = canonical_sha256(sim.slot_rng.bit_generator.state)
    rows: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    for action_id, action_name in zip(
        record["legal_action_ids"], record["legal_actions"]
    ):
        for horizon in HORIZONS:
            vector = feature_vector(
                state,
                sim,
                int(action_id),
                target=int(record["target"]),
                horizon=horizon,
                starter_tile=int(record["starter_tile"]),
            ).astype(np.float64, copy=False)
            if vector.shape != (FEATURE_WIDTH,) or not np.all(
                np.isfinite(vector)
            ):
                raise ValueError("Feature vector is not finite width 64")
            digest.update(vector.tobytes())
            rows.append(
                {
                    "record_id": str(record["record_id"]),
                    "action": str(action_name),
                    "action_id": int(action_id),
                    "horizon": int(horizon),
                    "features": vector,
                }
            )
    if _state_fingerprint(state) != before:
        raise ValueError("Feature extraction mutated source state")
    if canonical_sha256(sim.deck_rng.bit_generator.state) != deck_before:
        raise ValueError("Feature extraction consumed deck RNG")
    if canonical_sha256(sim.slot_rng.bit_generator.state) != slot_before:
        raise ValueError("Feature extraction consumed slot RNG")
    return rows, digest.hexdigest()


def build_e0_tasks(
    records: Sequence[Mapping[str, Any]],
    stream_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_record = {str(row["record_id"]): dict(row) for row in records}
    if len(by_record) != len(records):
        raise ValueError("Duplicate record ID")
    tasks = []
    for stream in stream_rows:
        if int(stream["replicate"]) not in E0_REPLICATES:
            continue
        record_id = str(stream["record_id"])
        record = by_record.get(record_id)
        if record is None:
            raise ValueError(f"Stream row references unknown record {record_id}")
        task = {
            "task_key": (
                f"{record_id}:{stream['action']}:{int(stream['replicate'])}"
            ),
            "record": record,
            "record_ordinal": int(stream["record_ordinal"]),
            "partition": str(stream["partition"]),
            "record_id": record_id,
            "root_cluster": str(stream["root_cluster"]),
            "behavior_family": str(stream["behavior_family"]),
            "scale": str(stream["scale"]),
            "target": int(stream["target"]),
            "state_sha1": str(stream["state_sha1"]),
            "action_id": int(stream["action_id"]),
            "action": str(stream["action"]),
            "replicate": int(stream["replicate"]),
            "logical_seed": int(stream["logical_seed"]),
            "deck_stream_id": int(stream["deck_stream_id"]),
            "slot_stream_id": int(stream["slot_stream_id"]),
            "policy_stream_id": int(stream["policy_stream_id"]),
        }
        tasks.append(task)
    tasks.sort(
        key=lambda row: (
            int(row["replicate"]),
            int(row["record_ordinal"]),
            CANONICAL_ACTIONS.index(str(row["action"])),
        )
    )
    if len({row["task_key"] for row in tasks}) != len(tasks):
        raise ValueError("Duplicate E0 task key")
    return tasks


def task_coupling_audit(tasks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_unit: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for task in tasks:
        by_unit[(str(task["record_id"]), int(task["replicate"]))].append(task)
    stream_keys = (
        "logical_seed",
        "deck_stream_id",
        "slot_stream_id",
        "policy_stream_id",
    )
    checks = {
        "replicates_exact": {
            int(task["replicate"]) for task in tasks
        } == set(E0_REPLICATES),
        "action_arm_streams_shared": all(
            all(len({int(row[key]) for row in rows}) == 1 for key in stream_keys)
            for rows in by_unit.values()
        ),
        "unit_streams_unique": all(
            len(
                {
                    int(rows[0][key])
                    for rows in by_unit.values()
                }
            )
            == len(by_unit)
            for key in stream_keys
        ),
        "task_keys_unique":
            len({str(task["task_key"]) for task in tasks}) == len(tasks),
    }
    return {
        "record_replicate_units": len(by_unit),
        "tasks": len(tasks),
        "checks": checks,
        "passes": all(checks.values()),
    }


def rollout_label(task: Mapping[str, Any], policy: Any) -> dict[str, Any]:
    record = task["record"]
    payload = load_record_state(record)
    state = state_from_replay_payload(payload)
    starter = int(record["starter_tile"])
    target = int(task["target"])
    action_id = int(task["action_id"])
    if action_id not in ThreesSim.from_stream_ids(
        deck_stream_id=2_026_072_543,
        slot_stream_id=2_026_072_544,
        starter_tile=starter,
    ).legal_actions(state):
        raise ValueError("Forced action is not legal in restored state")
    if max_tile_excluding_initial_starter(state.board, starter) >= target:
        raise ValueError("Source state already attained target")

    sim = ThreesSim.from_stream_ids(
        deck_stream_id=int(task["deck_stream_id"]),
        slot_stream_id=int(task["slot_stream_id"]),
        starter_tile=starter,
    )
    policy_rng = np.random.default_rng(int(task["policy_stream_id"]))
    completed = 0
    event_move: int | None = None
    terminal_move: int | None = None
    while completed < 40:
        if state.game_over:
            terminal_move = completed
            break
        action = (
            action_id
            if completed == 0
            else int(policy(state, sim, policy_rng))
        )
        state, info = sim.step(state, action)
        if not info.moved:
            raise RuntimeError(
                f"Illegal action {action} at task {task['task_key']} move "
                f"{completed}"
            )
        completed += 1
        attained = (
            max_tile_excluding_initial_starter(state.board, starter) >= target
        )
        if attained:
            event_move = completed
            if state.game_over:
                terminal_move = completed
            break
        if state.game_over:
            terminal_move = completed
            break

    interval_rows = event_censor_rows(
        event_move=event_move,
        terminal_move=terminal_move,
        completed_moves=completed,
    )
    return {
        "version": VERSION,
        "task_key": str(task["task_key"]),
        "partition": str(task["partition"]),
        "record_id": str(task["record_id"]),
        "root_cluster": str(task["root_cluster"]),
        "behavior_family": str(task["behavior_family"]),
        "scale": str(task["scale"]),
        "target": target,
        "state_sha1": str(task["state_sha1"]),
        "action": str(task["action"]),
        "action_id": action_id,
        "replicate": int(task["replicate"]),
        "logical_seed": int(task["logical_seed"]),
        "deck_stream_id": int(task["deck_stream_id"]),
        "slot_stream_id": int(task["slot_stream_id"]),
        "policy_stream_id": int(task["policy_stream_id"]),
        "event_move": event_move,
        "terminal_move": terminal_move,
        "completed_moves": completed,
        "interval_rows": interval_rows,
    }


class LabelStore:
    """Transactional compact sufficient-statistic store."""

    def __init__(
        self,
        path: Path,
        *,
        identity: Mapping[str, Any],
        transfer: bool = False,
        checkpoint_seal: Path | None = None,
        prediction_seal: Path | None = None,
    ):
        if transfer and (
            checkpoint_seal is None
            or prediction_seal is None
            or not checkpoint_seal.is_file()
            or not prediction_seal.is_file()
        ):
            raise PermissionError(
                "Transfer labels require checkpoint and prediction seals"
            )
        self.path = path
        self.identity = dict(identity)
        self.identity_sha256 = canonical_sha256(self.identity)
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS metadata "
            "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS paths ("
            "task_key TEXT PRIMARY KEY, "
            "payload_json TEXT NOT NULL, "
            "payload_sha256 TEXT NOT NULL)"
        )
        row = self.connection.execute(
            "SELECT value FROM metadata WHERE key='identity_sha256'"
        ).fetchone()
        if row is None:
            self.connection.execute(
                "INSERT INTO metadata(key,value) VALUES(?,?)",
                ("identity_sha256", self.identity_sha256),
            )
            self.connection.execute(
                "INSERT INTO metadata(key,value) VALUES(?,?)",
                ("identity_json", canonical_bytes(self.identity).decode("ascii")),
            )
            self.connection.commit()
        elif row[0] != self.identity_sha256:
            self.connection.close()
            raise ValueError("Label-store resume identity mismatch")

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "LabelStore":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def completed_keys(self) -> set[str]:
        return {
            str(row[0])
            for row in self.connection.execute("SELECT task_key FROM paths")
        }

    def insert_chunk(self, rows: Sequence[Mapping[str, Any]]) -> None:
        if not 1 <= len(rows) <= MAX_CHUNK_SIZE:
            raise ValueError("Label chunks must contain 1..8 paths")
        with self.connection:
            for row in rows:
                task_key = str(row["task_key"])
                payload_json = canonical_bytes(dict(row)).decode("ascii")
                payload_sha = hashlib.sha256(payload_json.encode("ascii")).hexdigest()
                existing = self.connection.execute(
                    "SELECT payload_sha256 FROM paths WHERE task_key=?",
                    (task_key,),
                ).fetchone()
                if existing is not None:
                    if existing[0] != payload_sha:
                        raise ValueError(
                            f"Conflicting resume payload for {task_key}"
                        )
                    continue
                self.connection.execute(
                    "INSERT INTO paths(task_key,payload_json,payload_sha256) "
                    "VALUES(?,?,?)",
                    (task_key, payload_json, payload_sha),
                )

    def rows(self) -> list[dict[str, Any]]:
        return [
            json.loads(row[0])
            for row in self.connection.execute(
                "SELECT payload_json FROM paths ORDER BY task_key"
            )
        ]

    def count(self) -> int:
        row = self.connection.execute("SELECT COUNT(*) FROM paths").fetchone()
        return int(row[0])


def aggregate_grouped_rows(
    records: Sequence[Mapping[str, Any]],
    paths: Sequence[Mapping[str, Any]],
    *,
    family_balanced: bool,
) -> list[dict[str, Any]]:
    record_map = {str(record["record_id"]): record for record in records}
    observed: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for path in paths:
        for interval in path["interval_rows"]:
            if interval["observed"]:
                observed[
                    (
                        str(path["record_id"]),
                        str(path["action"]),
                        str(interval["horizon"]),
                    )
                ].append(int(interval["event"]))

    feature_cache: dict[str, dict[tuple[str, str], np.ndarray]] = {}
    grouped: list[dict[str, Any]] = []
    for record_id, record in record_map.items():
        feature_rows, _digest = feature_rows_for_record(record)
        feature_cache[record_id] = {
            (str(row["action"]), f"h{int(row['horizon'])}"): row["features"]
            for row in feature_rows
        }
        for (candidate_id, action, horizon), events in observed.items():
            if candidate_id != record_id or not events:
                continue
            grouped.append(
                {
                    "record_id": record_id,
                    "root_cluster": str(record["root_cluster"]),
                    "behavior_family": str(record["behavior_family"]),
                    "scale": str(record["scale"]),
                    "action": action,
                    "horizon": horizon,
                    "features": feature_cache[record_id][
                        (action, horizon)
                    ].copy(),
                    "events": int(sum(events)),
                    "trials": len(events),
                    "event_fraction": float(sum(events) / len(events)),
                }
            )
    assign_group_weights(grouped, family_balanced=family_balanced)
    return grouped


def assign_group_weights(
    rows: list[dict[str, Any]], *, family_balanced: bool
) -> None:
    by_root: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_root[str(row["root_cluster"])].append(row)
    roots_by_family: dict[str, set[str]] = defaultdict(set)
    for root, root_rows in by_root.items():
        roots_by_family[str(root_rows[0]["behavior_family"])].add(root)
    families = sorted(roots_by_family)
    root_count = len(by_root)
    for root, root_rows in by_root.items():
        family = str(root_rows[0]["behavior_family"])
        root_weight = (
            1.0 / (len(families) * len(roots_by_family[family]))
            if family_balanced
            else 1.0 / root_count
        )
        records = sorted({str(row["record_id"]) for row in root_rows})
        for record_id in records:
            record_rows = [
                row for row in root_rows if str(row["record_id"]) == record_id
            ]
            actions = sorted({str(row["action"]) for row in record_rows})
            for action in actions:
                action_rows = [
                    row
                    for row in record_rows
                    if str(row["action"]) == action
                ]
                weight = (
                    root_weight
                    / len(records)
                    / len(actions)
                    / len(action_rows)
                )
                for row in action_rows:
                    row["weight"] = float(weight)


def weight_audit(
    rows: Sequence[Mapping[str, Any]], *, family_balanced: bool
) -> dict[str, Any]:
    by_root: dict[str, float] = defaultdict(float)
    by_family: dict[str, float] = defaultdict(float)
    for row in rows:
        weight = float(row["weight"])
        by_root[str(row["root_cluster"])] += weight
        by_family[str(row["behavior_family"])] += weight
    family_values = list(by_family.values())
    checks = {
        "weights_finite_positive": all(
            math.isfinite(float(row["weight"])) and float(row["weight"]) > 0
            for row in rows
        ),
        "total_at_most_one": sum(by_root.values()) <= 1.0 + 1e-12,
        "families_equal_when_requested": (
            not family_balanced
            or not family_values
            or max(family_values) - min(family_values) <= 1e-12
        ),
    }
    return {
        "total_weight": sum(by_root.values()),
        "root_weights": dict(sorted(by_root.items())),
        "family_weights": dict(sorted(by_family.items())),
        "checks": checks,
        "passes": all(checks.values()),
    }


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))


@dataclass
class G3HazardModel:
    feature_names: tuple[str, ...]
    standardize_mask: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coefficients: np.ndarray
    intercept: float
    calibration_intercept: float
    calibration_log_slope: float
    constant_hazards: np.ndarray
    optimizer_summary: dict[str, Any]
    calibration_summary: dict[str, Any]
    source_hashes: dict[str, str]

    @property
    def calibration_slope(self) -> float:
        return math.exp(float(self.calibration_log_slope))

    def normalized(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.shape[-1] != FEATURE_WIDTH:
            raise ValueError("Expected 64 features")
        return (values - self.feature_mean) / self.feature_scale

    def base_logits(self, features: np.ndarray) -> np.ndarray:
        values = self.normalized(features)
        return values @ self.coefficients + float(self.intercept)

    def predict(self, features: np.ndarray, *, calibrated: bool = True) -> np.ndarray:
        logits = self.base_logits(features)
        if calibrated:
            logits = (
                float(self.calibration_intercept)
                + self.calibration_slope * logits
            )
        return _sigmoid(np.asarray(logits, dtype=np.float64))

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=False)
        arrays_path = directory / "arrays.npz"
        np.savez_compressed(
            arrays_path,
            standardize_mask=self.standardize_mask.astype(np.bool_),
            feature_mean=self.feature_mean.astype(np.float64),
            feature_scale=self.feature_scale.astype(np.float64),
            coefficients=self.coefficients.astype(np.float64),
            constant_hazards=self.constant_hazards.astype(np.float64),
        )
        metadata = payload_with_hash(
            {
                "version": VERSION,
                "schema_sha256": SCHEMA_SHA256,
                "feature_width": FEATURE_WIDTH,
                "parameter_count": FEATURE_WIDTH + 1,
                "feature_names": list(self.feature_names),
                "intercept": float(self.intercept),
                "calibration_intercept": float(
                    self.calibration_intercept
                ),
                "calibration_log_slope": float(
                    self.calibration_log_slope
                ),
                "optimizer_summary": self.optimizer_summary,
                "calibration_summary": self.calibration_summary,
                "source_hashes": self.source_hashes,
                "arrays_file_sha256": sha256_path(arrays_path),
            }
        )
        write_immutable_json(directory / "meta.json", metadata)

    @classmethod
    def load(
        cls,
        directory: Path,
        *,
        expected_source_hashes: Mapping[str, str] | None = None,
    ) -> "G3HazardModel":
        metadata = json_object(directory / "meta.json")
        if not verify_payload_hash(metadata):
            raise ValueError("Model metadata payload hash mismatch")
        if metadata.get("version") != VERSION:
            raise ValueError("Incompatible G3 E0 model version")
        if metadata.get("schema_sha256") != SCHEMA_SHA256:
            raise ValueError("Incompatible G3 feature schema")
        if int(metadata.get("feature_width", -1)) != FEATURE_WIDTH:
            raise ValueError("Invalid model width")
        if int(metadata.get("parameter_count", -1)) != FEATURE_WIDTH + 1:
            raise ValueError("Invalid model parameter count")
        arrays_path = directory / "arrays.npz"
        if sha256_path(arrays_path) != metadata["arrays_file_sha256"]:
            raise ValueError("Model array hash mismatch")
        arrays = np.load(arrays_path, allow_pickle=False)
        model = cls(
            feature_names=tuple(str(value) for value in metadata["feature_names"]),
            standardize_mask=np.asarray(
                arrays["standardize_mask"], dtype=np.bool_
            ),
            feature_mean=np.asarray(arrays["feature_mean"], dtype=np.float64),
            feature_scale=np.asarray(arrays["feature_scale"], dtype=np.float64),
            coefficients=np.asarray(arrays["coefficients"], dtype=np.float64),
            intercept=float(metadata["intercept"]),
            calibration_intercept=float(
                metadata["calibration_intercept"]
            ),
            calibration_log_slope=float(
                metadata["calibration_log_slope"]
            ),
            constant_hazards=np.asarray(
                arrays["constant_hazards"], dtype=np.float64
            ),
            optimizer_summary=dict(metadata["optimizer_summary"]),
            calibration_summary=dict(metadata["calibration_summary"]),
            source_hashes={
                str(key): str(value)
                for key, value in metadata["source_hashes"].items()
            },
        )
        expected_names = tuple(
            str(column["name"]) for column in schema_manifest()["columns"]
        )
        arrays_to_check = (
            model.feature_mean,
            model.feature_scale,
            model.coefficients,
            model.constant_hazards,
        )
        if (
            model.feature_names != expected_names
            or model.standardize_mask.shape != (FEATURE_WIDTH,)
            or model.feature_mean.shape != (FEATURE_WIDTH,)
            or model.feature_scale.shape != (FEATURE_WIDTH,)
            or model.coefficients.shape != (FEATURE_WIDTH,)
            or model.constant_hazards.shape != (3,)
            or not all(np.all(np.isfinite(values)) for values in arrays_to_check)
            or np.any(model.feature_scale <= 0.0)
            or not math.isfinite(model.intercept)
            or not math.isfinite(model.calibration_intercept)
            or not math.isfinite(model.calibration_log_slope)
            or model.calibration_slope <= 0.0
        ):
            raise ValueError("Invalid model payload")
        if (
            expected_source_hashes is not None
            and model.source_hashes != dict(expected_source_hashes)
        ):
            raise ValueError("Model source hash mismatch")
        return model


def _matrix(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.stack(
        [np.asarray(row["features"], dtype=np.float64) for row in rows]
    )
    labels = np.asarray(
        [float(row["event_fraction"]) for row in rows], dtype=np.float64
    )
    weights = np.asarray(
        [float(row["weight"]) for row in rows], dtype=np.float64
    )
    return features, labels, weights


def _normalization(
    features: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    columns = schema_manifest()["columns"]
    mask = np.asarray(
        [bool(column["train_standardize"]) for column in columns],
        dtype=np.bool_,
    )
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("Training weights sum to zero")
    mean = np.zeros(FEATURE_WIDTH, dtype=np.float64)
    scale = np.ones(FEATURE_WIDTH, dtype=np.float64)
    mean[mask] = np.sum(features[:, mask] * weights[:, None], axis=0) / total
    centered = features[:, mask] - mean[mask]
    variance = np.sum(centered * centered * weights[:, None], axis=0) / total
    scale_values = np.sqrt(np.maximum(variance, 0.0))
    scale[mask] = np.where(scale_values < 1e-12, 1.0, scale_values)
    normalized = (features - mean) / scale
    return mask, mean, scale


def _weighted_log_loss(
    probabilities: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
) -> float:
    p = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    total = weights.sum()
    return float(
        -np.sum(weights * (labels * np.log(p) + (1.0 - labels) * np.log1p(-p)))
        / total
    )


def _weighted_brier(
    probabilities: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
) -> float:
    return float(np.sum(weights * (probabilities - labels) ** 2) / weights.sum())


def _ece(
    probabilities: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
) -> float:
    total = weights.sum()
    result = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        upper = lower + 0.1
        mask = (
            (probabilities >= lower)
            & (
                probabilities < upper
                if upper < 1.0
                else probabilities <= upper
            )
        )
        if not np.any(mask):
            continue
        local_weight = weights[mask].sum()
        result += (
            local_weight
            / total
            * abs(
                float(np.sum(weights[mask] * probabilities[mask]) / local_weight)
                - float(np.sum(weights[mask] * labels[mask]) / local_weight)
            )
        )
    return float(result)


def fit_hazard_model(
    train_rows: Sequence[Mapping[str, Any]],
    development_rows: Sequence[Mapping[str, Any]],
    *,
    source_hashes: Mapping[str, str],
) -> G3HazardModel:
    if not train_rows or not development_rows:
        raise ValueError("Train and development rows are required")
    train_x, train_y, train_w = _matrix(train_rows)
    dev_x, dev_y, dev_w = _matrix(development_rows)
    mask, mean, scale = _normalization(train_x, train_w)
    train_z = (train_x - mean) / scale

    penalty_mask = np.ones(FEATURE_WIDTH, dtype=np.float64)
    penalty_mask[:3] = 0.0

    def objective(params: np.ndarray) -> tuple[float, np.ndarray]:
        intercept = params[0]
        coefficients = params[1:]
        logits = train_z @ coefficients + intercept
        probabilities = _sigmoid(logits)
        loss = _weighted_log_loss(probabilities, train_y, train_w)
        loss += 0.5 * L2_LAMBDA * float(
            np.sum((coefficients * penalty_mask) ** 2)
        )
        residual = probabilities - train_y
        gradient = np.empty_like(params)
        gradient[0] = float(np.sum(train_w * residual) / train_w.sum())
        gradient[1:] = (
            train_z.T @ (train_w * residual) / train_w.sum()
            + L2_LAMBDA * coefficients * penalty_mask
        )
        return float(loss), gradient

    base = minimize(
        objective,
        np.zeros(FEATURE_WIDTH + 1, dtype=np.float64),
        jac=True,
        method="L-BFGS-B",
        options={
            "maxiter": MAX_OPTIMIZER_ITERATIONS,
            "gtol": OPTIMIZER_GTOL,
        },
    )
    base_gradient = np.asarray(base.jac, dtype=np.float64)
    base_logits = (dev_x - mean) / scale @ base.x[1:] + base.x[0]

    def calibration_objective(params: np.ndarray) -> tuple[float, np.ndarray]:
        intercept, log_slope = float(params[0]), float(params[1])
        slope = math.exp(log_slope)
        logits = intercept + slope * base_logits
        probabilities = _sigmoid(logits)
        loss = _weighted_log_loss(probabilities, dev_y, dev_w)
        residual = probabilities - dev_y
        gradient = np.asarray(
            [
                np.sum(dev_w * residual) / dev_w.sum(),
                np.sum(dev_w * residual * slope * base_logits) / dev_w.sum(),
            ],
            dtype=np.float64,
        )
        return float(loss), gradient

    calibration = minimize(
        calibration_objective,
        np.zeros(2, dtype=np.float64),
        jac=True,
        method="L-BFGS-B",
        options={
            "maxiter": MAX_OPTIMIZER_ITERATIONS,
            "gtol": OPTIMIZER_GTOL,
        },
    )
    calibrated = _sigmoid(
        calibration.x[0] + math.exp(float(calibration.x[1])) * base_logits
    )

    constant_hazards = np.empty(3, dtype=np.float64)
    for index in range(3):
        horizon_mask = train_x[:, index] == 1.0
        weighted_events = float(
            np.sum(train_w[horizon_mask] * train_y[horizon_mask])
        )
        weighted_trials = float(train_w[horizon_mask].sum())
        constant_hazards[index] = np.clip(
            (weighted_events + 0.5e-12) / (weighted_trials + 1e-12),
            1e-9,
            1.0 - 1e-9,
        )

    columns = schema_manifest()["columns"]
    return G3HazardModel(
        feature_names=tuple(str(column["name"]) for column in columns),
        standardize_mask=mask,
        feature_mean=mean,
        feature_scale=scale,
        coefficients=np.asarray(base.x[1:], dtype=np.float64),
        intercept=float(base.x[0]),
        calibration_intercept=float(calibration.x[0]),
        calibration_log_slope=float(calibration.x[1]),
        constant_hazards=constant_hazards,
        optimizer_summary={
            "success": bool(base.success),
            "status": int(base.status),
            "message": str(base.message),
            "iterations": int(base.nit),
            "gradient_infinity_norm": float(
                np.max(np.abs(base_gradient), initial=0.0)
            ),
            "objective": float(base.fun),
        },
        calibration_summary={
            "success": bool(calibration.success),
            "status": int(calibration.status),
            "message": str(calibration.message),
            "iterations": int(calibration.nit),
            "objective": float(calibration.fun),
            "slope": math.exp(float(calibration.x[1])),
            "intercept": float(calibration.x[0]),
            "development_ece": _ece(calibrated, dev_y, dev_w),
        },
        source_hashes=dict(source_hashes),
    )


def model_stability_audit(model: G3HazardModel) -> dict[str, Any]:
    checks = {
        "base_optimizer_success":
            bool(model.optimizer_summary.get("success")),
        "base_gradient_at_most_1e_4":
            float(model.optimizer_summary.get("gradient_infinity_norm", math.inf))
            <= 1e-4,
        "calibration_optimizer_success":
            bool(model.calibration_summary.get("success")),
        "calibration_slope_in_bounds":
            0.05 <= model.calibration_slope <= 20.0,
        "calibration_intercept_in_bounds":
            abs(float(model.calibration_intercept)) <= 3.0,
        "development_ece_at_most_0_25":
            float(model.calibration_summary.get("development_ece", math.inf))
            <= 0.25,
        "arrays_finite": all(
            np.all(np.isfinite(values))
            for values in (
                model.feature_mean,
                model.feature_scale,
                model.coefficients,
                model.constant_hazards,
            )
        ),
    }
    return {"checks": checks, "passes": all(checks.values())}


def cumulative_h40(interval_probabilities: Sequence[float]) -> float:
    values = np.asarray(interval_probabilities, dtype=np.float64)
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise ValueError("Expected three finite interval probabilities")
    return float(1.0 - np.prod(1.0 - values))


def choose_action(
    values: Mapping[str, float],
    *,
    tolerance: float = MODEL_TIE_TOLERANCE,
) -> str:
    if not values:
        raise ValueError("No legal action values")
    if any(
        action not in CANONICAL_ACTIONS or not math.isfinite(float(value))
        for action, value in values.items()
    ):
        raise ValueError("Invalid action values")
    best = max(float(value) for value in values.values())
    return next(
        action
        for action in CANONICAL_ACTIONS
        if action in values and best - float(values[action]) <= tolerance
    )


def spearman_average_ties(
    left: Sequence[float], right: Sequence[float]
) -> float | None:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.size < 2 or y.size != x.size:
        return None
    x_rank = rankdata(x, method="average")
    y_rank = rankdata(y, method="average")
    if np.ptp(x_rank) == 0.0 or np.ptp(y_rank) == 0.0:
        return None
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def metric_summary(
    rows: Sequence[Mapping[str, Any]],
    model: G3HazardModel,
) -> dict[str, Any]:
    features, labels, weights = _matrix(rows)
    probabilities = model.predict(features)
    horizon_index = np.argmax(features[:, :3], axis=1)
    constants = model.constant_hazards[horizon_index]
    return {
        "records": len({str(row["record_id"]) for row in rows}),
        "roots": len({str(row["root_cluster"]) for row in rows}),
        "log_loss": _weighted_log_loss(probabilities, labels, weights),
        "constant_log_loss": _weighted_log_loss(constants, labels, weights),
        "log_loss_improvement": (
            _weighted_log_loss(constants, labels, weights)
            - _weighted_log_loss(probabilities, labels, weights)
        ),
        "brier": _weighted_brier(probabilities, labels, weights),
        "constant_brier": _weighted_brier(constants, labels, weights),
        "brier_improvement": (
            _weighted_brier(constants, labels, weights)
            - _weighted_brier(probabilities, labels, weights)
        ),
        "ece": _ece(probabilities, labels, weights),
    }


def _root_equal_copy(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    copied = [dict(row) for row in rows]
    assign_group_weights(copied, family_balanced=False)
    return copied


def action_rank_report(
    records: Sequence[Mapping[str, Any]],
    paths: Sequence[Mapping[str, Any]],
    model: G3HazardModel,
) -> dict[str, Any]:
    paths_by_record_action: dict[
        tuple[str, str], list[Mapping[str, Any]]
    ] = defaultdict(list)
    for path in paths:
        paths_by_record_action[
            (str(path["record_id"]), str(path["action"]))
        ].append(path)

    record_rows = []
    for record in records:
        predicted: list[float] = []
        empirical: list[float] = []
        actions: list[str] = []
        features, _digest = feature_rows_for_record(record)
        by_action_horizon = {
            (str(row["action"]), int(row["horizon"])): row["features"]
            for row in features
        }
        for action in record["legal_actions"]:
            action_paths = paths_by_record_action.get(
                (str(record["record_id"]), str(action)), []
            )
            if not action_paths:
                continue
            interval_probabilities = [
                float(
                    model.predict(
                        np.asarray(
                            by_action_horizon[(str(action), horizon)]
                        )[None, :]
                    )[0]
                )
                for horizon in HORIZONS
            ]
            predicted.append(cumulative_h40(interval_probabilities))
            empirical.append(
                float(
                    np.mean(
                        [
                            path.get("event_move") is not None
                            and int(path["event_move"]) <= 40
                            for path in action_paths
                        ]
                    )
                )
            )
            actions.append(str(action))
        correlation = spearman_average_ties(predicted, empirical)
        record_rows.append(
            {
                "record_id": str(record["record_id"]),
                "root_cluster": str(record["root_cluster"]),
                "behavior_family": str(record["behavior_family"]),
                "scale": str(record["scale"]),
                "actions": actions,
                "predicted": predicted,
                "empirical": empirical,
                "correlation": correlation,
                "informative": correlation is not None,
            }
        )

    def summarize(rows: Sequence[Mapping[str, Any]]) -> tuple[float, int]:
        by_root: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            if row["correlation"] is not None:
                by_root[str(row["root_cluster"])].append(
                    float(row["correlation"])
                )
        root_values = [
            float(np.mean(values)) for values in by_root.values() if values
        ]
        return (
            float(np.mean(root_values)) if root_values else math.nan,
            sum(bool(row["informative"]) for row in rows),
        )

    overall, count = summarize(record_rows)
    by_scale = {}
    counts_by_scale = {}
    for scale in sorted({str(row["scale"]) for row in record_rows}):
        value, local_count = summarize(
            [row for row in record_rows if row["scale"] == scale]
        )
        by_scale[scale] = value
        counts_by_scale[scale] = local_count
    by_family = {}
    for family in sorted(
        {str(row["behavior_family"]) for row in record_rows}
    ):
        value, _local_count = summarize(
            [row for row in record_rows if row["behavior_family"] == family]
        )
        by_family[family] = value
    return {
        "overall": overall,
        "by_scale": by_scale,
        "by_family": by_family,
        "informative_records": count,
        "informative_by_scale": counts_by_scale,
        "records": record_rows,
    }


def bootstrap_metric_improvement(
    rows: Sequence[Mapping[str, Any]],
    model: G3HazardModel,
    *,
    seed: int,
    repeats: int = BOOTSTRAP_REPEATS,
    records: Sequence[Mapping[str, Any]] | None = None,
    paths: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    by_root: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_root[str(row["root_cluster"])].append(row)
    roots = sorted(by_root)
    if not roots:
        raise ValueError("Bootstrap requires roots")
    rng = np.random.default_rng(seed)
    log_values = np.empty(repeats, dtype=np.float64)
    brier_values = np.empty(repeats, dtype=np.float64)
    rank_values: list[float] = []
    records_by_root: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    paths_by_root: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    if records is not None and paths is not None:
        for record in records:
            records_by_root[str(record["root_cluster"])].append(record)
        for path in paths:
            paths_by_root[str(path["root_cluster"])].append(path)
    for repeat in range(repeats):
        selected = rng.integers(0, len(roots), size=len(roots))
        sampled: list[dict[str, Any]] = []
        sampled_records = []
        sampled_paths = []
        for draw, index in enumerate(selected):
            source_root = roots[int(index)]
            synthetic_root = f"draw-{draw}:{source_root}"
            sampled.extend(
                {**dict(row), "root_cluster": synthetic_root}
                for row in by_root[source_root]
            )
            sampled_records.extend(
                {**dict(record), "root_cluster": synthetic_root}
                for record in records_by_root.get(source_root, [])
            )
            sampled_paths.extend(
                {**dict(path), "root_cluster": synthetic_root}
                for path in paths_by_root.get(source_root, [])
            )
        assign_group_weights(sampled, family_balanced=False)
        metrics = metric_summary(sampled, model)
        log_values[repeat] = metrics["log_loss_improvement"]
        brier_values[repeat] = metrics["brier_improvement"]
        if sampled_records and sampled_paths:
            rank = action_rank_report(
                sampled_records, sampled_paths, model
            )["overall"]
            if math.isfinite(float(rank)):
                rank_values.append(float(rank))
    result = {
        "repeats": repeats,
        "seed": seed,
        "log_loss_improvement_ci95": [
            float(np.percentile(log_values, 2.5)),
            float(np.percentile(log_values, 97.5)),
        ],
        "brier_improvement_ci95": [
            float(np.percentile(brier_values, 2.5)),
            float(np.percentile(brier_values, 97.5)),
        ],
    }
    if rank_values:
        result["rank_ci95"] = [
            float(np.percentile(rank_values, 2.5)),
            float(np.percentile(rank_values, 97.5)),
        ]
        result["rank_finite_draws"] = len(rank_values)
    return result


def predictive_report(
    records: Sequence[Mapping[str, Any]],
    paths: Sequence[Mapping[str, Any]],
    model: G3HazardModel,
    *,
    bootstrap_seed: int,
    bootstrap_repeats: int = BOOTSTRAP_REPEATS,
) -> dict[str, Any]:
    grouped = aggregate_grouped_rows(
        records, paths, family_balanced=False
    )
    overall = metric_summary(grouped, model)
    scales = {}
    for scale in sorted({str(row["scale"]) for row in grouped}):
        local = _root_equal_copy(
            [row for row in grouped if row["scale"] == scale]
        )
        scales[scale] = metric_summary(local, model)
    families = {}
    for family in sorted(
        {str(row["behavior_family"]) for row in grouped}
    ):
        local = _root_equal_copy(
            [row for row in grouped if row["behavior_family"] == family]
        )
        families[family] = metric_summary(local, model)
    rank = action_rank_report(records, paths, model)
    bootstrap = bootstrap_metric_improvement(
        grouped,
        model,
        seed=bootstrap_seed,
        repeats=bootstrap_repeats,
    )
    rank_by_root: dict[str, list[float]] = defaultdict(list)
    for row in rank["records"]:
        if row["correlation"] is not None:
            rank_by_root[str(row["root_cluster"])].append(
                float(row["correlation"])
            )
    if rank_by_root:
        roots = sorted(rank_by_root)
        root_rank = np.asarray(
            [np.mean(rank_by_root[root]) for root in roots],
            dtype=np.float64,
        )
        rng = np.random.default_rng(bootstrap_seed)
        draws = np.empty(bootstrap_repeats, dtype=np.float64)
        for repeat in range(bootstrap_repeats):
            selected = rng.integers(0, len(roots), size=len(roots))
            draws[repeat] = float(np.mean(root_rank[selected]))
        bootstrap["rank_ci95"] = [
            float(np.percentile(draws, 2.5)),
            float(np.percentile(draws, 97.5)),
        ]
        bootstrap["rank_finite_draws"] = bootstrap_repeats
    return {
        "overall": overall,
        "pooled": overall,
        "scales": scales,
        "families": families,
        "rank": rank,
        "bootstrap": bootstrap,
    }


def predict_transfer_actions(
    records: Sequence[Mapping[str, Any]],
    tasks: Sequence[Mapping[str, Any]],
    model: G3HazardModel,
    incumbent: Any,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    task_by_record = {}
    for task in tasks:
        if int(task["replicate"]) == 0:
            task_by_record.setdefault(str(task["record_id"]), task)
    predictions = []
    activity = {"roots": 0, "corner2": 0, "incumbent": 0}
    for record in records:
        record_id = str(record["record_id"])
        feature_rows, _digest = feature_rows_for_record(record)
        by_action: dict[str, list[float]] = defaultdict(list)
        for row in feature_rows:
            probability = float(
                model.predict(np.asarray(row["features"])[None, :])[0]
            )
            by_action[str(row["action"])].append(probability)
        values = {
            action: cumulative_h40(probabilities)
            for action, probabilities in by_action.items()
        }
        model_action = choose_action(values)
        task = task_by_record[record_id]
        payload = load_record_state(record)
        state = state_from_replay_payload(payload)
        sim = ThreesSim.from_stream_ids(
            deck_stream_id=int(task["deck_stream_id"]),
            slot_stream_id=int(task["slot_stream_id"]),
            starter_tile=int(record["starter_tile"]),
        )
        policy_rng = np.random.default_rng(int(task["policy_stream_id"]))
        incumbent_action_id = int(incumbent(state, sim, policy_rng))
        incumbent_action = DIRECTION_NAMES[incumbent_action_id]
        changed = model_action != incumbent_action
        family = str(record["behavior_family"])
        if changed:
            activity["roots"] += 1
            if family == "g2_transfer_corner2":
                activity["corner2"] += 1
            if family == "g2_transfer_phaseblend_incumbent":
                activity["incumbent"] += 1
        predictions.append(
            {
                "record_id": record_id,
                "root_cluster": str(record["root_cluster"]),
                "behavior_family": family,
                "model_action": model_action,
                "incumbent_action": incumbent_action,
                "changed": changed,
                "action_values": values,
            }
        )
    return predictions, activity


def ordinary_gate_decision(
    report: Mapping[str, Any],
    *,
    integrity_passes: bool,
    model_stable: bool,
) -> str:
    if not integrity_passes or not model_stable:
        return "KILL_G3_BOOTSTRAP_PREDICTIVE"
    overall = report["overall"]
    scales = report["scales"]
    bootstrap = report["bootstrap"]
    checks = {
        "overall_point_metrics_positive":
            float(overall["log_loss_improvement"]) > 0
            and float(overall["brier_improvement"]) > 0,
        "one_primary_ci_excludes_zero":
            float(bootstrap["log_loss_improvement_ci95"][0]) > 0
            or float(bootstrap["brier_improvement_ci95"][0]) > 0,
        "both_scale_point_metrics_positive": all(
            float(scales[scale]["log_loss_improvement"]) > 0
            and float(scales[scale]["brier_improvement"]) > 0
            for scale in ("pre768", "pre1536")
        ),
        "rank_direction":
            float(report["rank"]["overall"]) > 0
            and all(
                float(report["rank"]["by_scale"][scale]) >= 0
                for scale in ("pre768", "pre1536")
            ),
        "rank_counts":
            int(report["rank"]["informative_records"]) >= 20
            and all(
                int(report["rank"]["informative_by_scale"][scale]) >= 5
                for scale in ("pre768", "pre1536")
            ),
        "family_robustness": not any(
            int(row["roots"]) >= 5
            and float(row["log_loss_improvement"]) < 0
            and float(row["brier_improvement"]) < 0
            for row in report["families"].values()
        ),
    }
    return (
        "READY_G3_E0_ORDINARY_PREDICTIVE"
        if all(checks.values())
        else "KILL_G3_BOOTSTRAP_PREDICTIVE"
    )


def transfer_gate_decision(
    report: Mapping[str, Any],
    *,
    activity: Mapping[str, int],
    integrity_passes: bool,
) -> str:
    if not integrity_passes:
        return "HOLD_G3_E0_UNDERPOWERED_TRANSFER"
    pooled = report["pooled"]
    families = report["families"]
    checks = {
        "pooled_direction":
            float(pooled["log_loss_improvement"]) >= 0
            and float(pooled["brier_improvement"]) >= 0
            and float(report["rank"]["overall"]) > 0,
        "large_families_nonnegative": all(
            float(families[family]["log_loss_improvement"]) >= 0
            and float(families[family]["brier_improvement"]) >= 0
            and float(report["rank"]["by_family"][family]) >= 0
            for family in (
                "g2_transfer_corner2",
                "g2_transfer_phaseblend_incumbent",
            )
        ),
        "activity_floor":
            int(activity["roots"]) >= 6
            and int(activity["corner2"]) >= 1
            and int(activity["incumbent"]) >= 1,
    }
    return (
        "READY_G3_E1_COMPLETION"
        if all(checks.values())
        else "HOLD_G3_E0_UNDERPOWERED_TRANSFER"
    )


def execution_identity(lock: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "version": VERSION,
        "preflight_lock_payload_sha256": lock["canonical_payload_sha256"],
        "task_manifest_file_sha256": lock["task_manifest_file_sha256"],
        "stream_manifest_file_sha256": lock["stream_manifest_file_sha256"],
        "charter_sha256": CHARTER_SHA256,
        "implementation_sha256": sha256_path(Path(__file__)),
    }


def seal_execution_opened(
    out_dir: Path,
    *,
    identity: Mapping[str, Any],
    command: Sequence[str],
) -> dict[str, Any]:
    marker = out_dir / OPEN_MARKER_NAME
    if marker.exists() or (out_dir / TERMINAL_RESULT_NAME).exists():
        raise FileExistsError("E0 execution already opened or completed")
    payload = payload_with_hash(
        {
            "version": VERSION,
            "admission": "G3 E0 execution",
            "opened_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "identity": dict(identity),
            "command": list(command),
            "labels_before_open": 0,
            "models_before_open": 0,
            "transfer_outcomes_before_open": 0,
            "promotable": False,
        }
    )
    write_immutable_json(marker, payload)
    return payload


def seal_terminal_error(
    out_dir: Path, *, stage: str, error: BaseException
) -> dict[str, Any]:
    path = out_dir / TERMINAL_RESULT_NAME
    payload = payload_with_hash(
        {
            "version": VERSION,
            "decision": "KILL_G3_BOOTSTRAP_PREDICTIVE",
            "stage": stage,
            "error": f"{type(error).__name__}: {error}",
            "promotable": False,
            "policy_evaluation_authorized": False,
            "dashboard_eligible": False,
        }
    )
    write_immutable_json(path, payload)
    return payload


def seal_terminal_decision(
    out_dir: Path,
    *,
    decision: str,
    stage: str,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    if decision not in {
        "READY_G3_E1_COMPLETION",
        "HOLD_G3_E0_UNDERPOWERED_TRANSFER",
        "KILL_G3_BOOTSTRAP_PREDICTIVE",
    }:
        raise ValueError(f"Unsupported E0 terminal decision: {decision}")
    payload = payload_with_hash(
        {
            "version": VERSION,
            "decision": decision,
            "stage": stage,
            "evidence": dict(evidence),
            "e0_non_promotable": True,
            "e1_authorized": decision == "READY_G3_E1_COMPLETION",
            "policy_evaluation_authorized": False,
            "promotion_authorized": False,
            "dashboard_eligible": False,
        }
    )
    write_immutable_json(out_dir / TERMINAL_RESULT_NAME, payload)
    return payload


def _directory_size(path: Path) -> int:
    return sum(
        item.stat().st_size
        for item in path.rglob("*")
        if item.is_file()
    )


def execute(
    *,
    out_dir: Path,
    preflight_lock: Path,
    jobs: int,
    command: Sequence[str],
) -> dict[str, Any]:
    """Future separately authorized E0 execution entry point."""

    if jobs != 1:
        raise ValueError("Frozen E0 execution requires jobs=1")
    lock = json_object(preflight_lock)
    if not verify_payload_hash(lock):
        raise ValueError("Preflight lock payload hash mismatch")
    if lock.get("decision") != "READY_G3_E0_LABEL_FIT_EXECUTION":
        raise PermissionError("Preflight did not authorize E0 execution")
    if Path(str(lock["out_dir_resolved"])).resolve() != out_dir.resolve():
        raise ValueError("Preflight output directory mismatch")
    identity = execution_identity(lock)
    seal_execution_opened(out_dir, identity=identity, command=command)
    stage = "ordinary_labels"
    try:
        records_manifest = json_object(
            out_dir / str(lock["record_manifest_name"])
        )
        task_manifest = json_object(
            out_dir / str(lock["task_manifest_name"])
        )
        ordinary_tasks = [
            task
            for task in task_manifest["tasks"]
            if task["partition"] != "transfer_diagnostic"
        ]
        policy = make_policy(str(lock["incumbent_policy_spec"]))
        store_identity = {
            **identity,
            "partition": "ordinary",
            "tasks_sha256": canonical_sha256(ordinary_tasks),
        }
        with LabelStore(
            out_dir / ORDINARY_DB_NAME,
            identity=store_identity,
        ) as store:
            completed = store.completed_keys()
            pending = [
                task
                for task in ordinary_tasks
                if task["task_key"] not in completed
            ]
            for offset in range(0, len(pending), MAX_CHUNK_SIZE):
                chunk = [
                    rollout_label(task, policy)
                    for task in pending[offset : offset + MAX_CHUNK_SIZE]
                ]
                store.insert_chunk(chunk)
                if _directory_size(out_dir) >= MAX_OUTPUT_BYTES:
                    raise RuntimeError("E0 output exceeded 4 GiB")
                stat = os.statvfs(out_dir)
                if stat.f_bavail * stat.f_frsize / 1024**3 < MIN_FREE_GIB:
                    raise RuntimeError("Free disk fell below 100 GiB")
            ordinary_paths = store.rows()
        if len(ordinary_paths) != int(lock["ordinary_path_count"]):
            raise RuntimeError("Ordinary label completion mismatch")

        stage = "ordinary_fit"
        records = records_manifest["records"]
        train_records = [
            row for row in records if row["partition"] == "train"
        ]
        dev_records = [
            row for row in records if row["partition"] == "development"
        ]
        train_paths = [
            row for row in ordinary_paths if row["partition"] == "train"
        ]
        dev_paths = [
            row for row in ordinary_paths
            if row["partition"] == "development"
        ]
        train_rows = aggregate_grouped_rows(
            train_records, train_paths, family_balanced=True
        )
        dev_rows = aggregate_grouped_rows(
            dev_records, dev_paths, family_balanced=False
        )
        model = fit_hazard_model(
            train_rows,
            dev_rows,
            source_hashes={
                "preflight_lock": sha256_path(preflight_lock),
                "ordinary_labels": sha256_path(out_dir / ORDINARY_DB_NAME),
            },
        )
        model.save(out_dir / MODEL_DIR_NAME)
        stability = model_stability_audit(model)
        stage = "ordinary_evaluation"
        ordinary_report = predictive_report(
            dev_records,
            dev_paths,
            model,
            bootstrap_seed=DEV_BOOTSTRAP_SEED,
        )
        ordinary_decision = ordinary_gate_decision(
            ordinary_report,
            integrity_passes=True,
            model_stable=stability["passes"],
        )
        checkpoint = payload_with_hash(
            {
                "version": VERSION,
                "model_meta_sha256": sha256_path(
                    out_dir / MODEL_DIR_NAME / "meta.json"
                ),
                "model_arrays_sha256": sha256_path(
                    out_dir / MODEL_DIR_NAME / "arrays.npz"
                ),
                "ordinary_labels_sha256": sha256_path(
                    out_dir / ORDINARY_DB_NAME
                ),
                "model_stability": stability,
                "ordinary_report": ordinary_report,
                "ordinary_decision": ordinary_decision,
                "promotable": False,
            }
        )
        write_immutable_json(out_dir / CHECKPOINT_SEAL_NAME, checkpoint)
        if ordinary_decision != "READY_G3_E0_ORDINARY_PREDICTIVE":
            return seal_terminal_decision(
                out_dir,
                decision="KILL_G3_BOOTSTRAP_PREDICTIVE",
                stage="ordinary_evaluation",
                evidence={
                    "checkpoint_file_sha256": sha256_path(
                        out_dir / CHECKPOINT_SEAL_NAME
                    ),
                    "transfer_predictions_opened": False,
                    "transfer_labels_opened": False,
                },
            )

        stage = "transfer_predictions"
        transfer_records = [
            row
            for row in records
            if row["partition"] == "transfer_diagnostic"
        ]
        transfer_tasks = [
            task
            for task in task_manifest["tasks"]
            if task["partition"] == "transfer_diagnostic"
        ]
        predictions, activity = predict_transfer_actions(
            transfer_records,
            transfer_tasks,
            model,
            policy,
        )
        prediction_seal = payload_with_hash(
            {
                "version": VERSION,
                "checkpoint_file_sha256": sha256_path(
                    out_dir / CHECKPOINT_SEAL_NAME
                ),
                "predictions": predictions,
                "predictions_sha256": canonical_sha256(predictions),
                "activity": activity,
                "transfer_label_values_opened": False,
                "promotable": False,
            }
        )
        write_immutable_json(
            out_dir / PREDICTION_SEAL_NAME, prediction_seal
        )
        activity_passes = (
            activity["roots"] >= 6
            and activity["corner2"] >= 1
            and activity["incumbent"] >= 1
        )
        if not activity_passes:
            return seal_terminal_decision(
                out_dir,
                decision="HOLD_G3_E0_UNDERPOWERED_TRANSFER",
                stage="transfer_activity",
                evidence={
                    "checkpoint_file_sha256": sha256_path(
                        out_dir / CHECKPOINT_SEAL_NAME
                    ),
                    "prediction_file_sha256": sha256_path(
                        out_dir / PREDICTION_SEAL_NAME
                    ),
                    "activity": activity,
                    "transfer_labels_opened": False,
                },
            )

        stage = "transfer_labels"
        transfer_identity = {
            **identity,
            "partition": "transfer_diagnostic",
            "tasks_sha256": canonical_sha256(transfer_tasks),
            "checkpoint_sha256": sha256_path(
                out_dir / CHECKPOINT_SEAL_NAME
            ),
            "prediction_sha256": sha256_path(
                out_dir / PREDICTION_SEAL_NAME
            ),
        }
        with LabelStore(
            out_dir / TRANSFER_DB_NAME,
            identity=transfer_identity,
            transfer=True,
            checkpoint_seal=out_dir / CHECKPOINT_SEAL_NAME,
            prediction_seal=out_dir / PREDICTION_SEAL_NAME,
        ) as store:
            completed = store.completed_keys()
            pending = [
                task
                for task in transfer_tasks
                if task["task_key"] not in completed
            ]
            for offset in range(0, len(pending), MAX_CHUNK_SIZE):
                chunk = [
                    rollout_label(task, policy)
                    for task in pending[offset : offset + MAX_CHUNK_SIZE]
                ]
                store.insert_chunk(chunk)
            transfer_paths = store.rows()
        if len(transfer_paths) != int(lock["transfer_path_count"]):
            raise RuntimeError("Transfer label completion mismatch")

        stage = "transfer_evaluation"
        transfer_report = predictive_report(
            transfer_records,
            transfer_paths,
            model,
            bootstrap_seed=TRANSFER_BOOTSTRAP_SEED,
        )
        decision = transfer_gate_decision(
            transfer_report,
            activity=activity,
            integrity_passes=True,
        )
        return seal_terminal_decision(
            out_dir,
            decision=decision,
            stage="transfer_evaluation",
            evidence={
                "checkpoint_file_sha256": sha256_path(
                    out_dir / CHECKPOINT_SEAL_NAME
                ),
                "prediction_file_sha256": sha256_path(
                    out_dir / PREDICTION_SEAL_NAME
                ),
                "transfer_labels_sha256": sha256_path(
                    out_dir / TRANSFER_DB_NAME
                ),
                "activity": activity,
                "transfer_report": transfer_report,
                "n32_mde_or": 4.0,
            },
        )
    except Exception as error:
        if not (out_dir / TERMINAL_RESULT_NAME).exists():
            seal_terminal_error(out_dir, stage=stage, error=error)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    execute_parser = subparsers.add_parser("execute")
    execute_parser.add_argument("--out-dir", type=Path, required=True)
    execute_parser.add_argument("--preflight-lock", type=Path, required=True)
    execute_parser.add_argument("--jobs", type=int, required=True)
    args = parser.parse_args()
    if os.nice(0) < MIN_NICE:
        os.nice(MIN_NICE - os.nice(0))
    if args.command == "execute":
        command = [
            "python",
            "-m",
            "threes_rl.g3_e0_label_fit",
            "execute",
            "--out-dir",
            str(args.out_dir),
            "--preflight-lock",
            str(args.preflight_lock),
            "--jobs",
            str(args.jobs),
        ]
        result = execute(
            out_dir=args.out_dir,
            preflight_lock=args.preflight_lock,
            jobs=args.jobs,
            command=command,
        )
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
