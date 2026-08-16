"""Local browser interface for playing and recording exact simulator games."""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np

from threes_rl.eval import make_policy
from threes_rl.record_replay import preview_payload, state_payload, write_html
from threes_rl.replay_provenance import ORIGIN_CONTINUATION, ORIGIN_HUMAN, direct_root_fields
from threes_rl.sim import DIRECTION_NAMES, ThreesSim, preview_from_label, score_board, score_tile


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8770
DEFAULT_DATA_ROOT = Path("datasets/human_play")
DEFAULT_HTML = Path(__file__).with_name("human_play.html")
DEFAULT_POLICY_FILE = Path("threes_rl/current_incumbent_policy.txt")
POLICY_NAME = "human_web"
RESTART_POLICY_NAME = "human_restart_web"
QUALITY_ANNOTATIONS = {"good", "mistakes", "calibration-discard"}


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(path)


def _policy_spec(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    spec = next(
        (line.strip() for line in raw.decode().splitlines() if line.strip() and not line.lstrip().startswith("#")),
        "",
    )
    if not spec:
        raise ValueError(f"Policy file contains no policy: {path}")
    return spec, hashlib.sha256(raw).hexdigest()


class PolicyAdvisor:
    """Thread-safe, hot-reloading read-only policy recommendations."""

    def __init__(self, policy_file: Path) -> None:
        self.policy_file = Path(policy_file)
        self.lock = threading.RLock()
        self.policy: Any | None = None
        self.spec: str | None = None
        self.policy_file_sha256: str | None = None
        self.error: str | None = None
        self._refresh(force=True)

    def _refresh(self, *, force: bool = False) -> None:
        try:
            spec, file_hash = _policy_spec(self.policy_file)
            if force or file_hash != self.policy_file_sha256:
                self.policy = make_policy(spec)
                self.spec = spec
                self.policy_file_sha256 = file_hash
            self.error = None
        except Exception as exc:  # Keep human play available if the advisor cannot load.
            self.policy = None
            self.error = f"{type(exc).__name__}: {exc}"

    def metadata(self) -> dict[str, Any]:
        with self.lock:
            return {
                "mode": "incumbent_recommendation_visible",
                "policy_file": str(self.policy_file),
                "policy_file_sha256": self.policy_file_sha256,
                "policy_spec": self.spec,
                "status": "ready" if self.policy is not None else "unavailable",
                "error": self.error,
            }

    def recommend(self, state, sim: ThreesSim, *, decision_seed: int) -> dict[str, Any]:
        started = time.perf_counter()
        with self.lock:
            self._refresh()
            if self.policy is None:
                return {
                    "status": "unavailable",
                    "error": self.error,
                    "policy_file_sha256": self.policy_file_sha256,
                }
            legal = tuple(int(action) for action in sim.legal_actions(state))
            if not legal:
                return {
                    "status": "terminal",
                    "action": None,
                    "action_index": None,
                    "action_values": [],
                    "policy_file_sha256": self.policy_file_sha256,
                }
            rng = np.random.default_rng(int(decision_seed))
            action_values_method = getattr(self.policy, "action_values", None)
            if callable(action_values_method):
                raw_values = [(int(action), float(value)) for action, value in action_values_method(state, sim)]
                selector = getattr(self.policy, "_select_action", None)
                if callable(selector):
                    recommendation = int(selector(raw_values, rng))
                else:
                    recommendation = max(raw_values, key=lambda item: item[1])[0]
            else:
                recommendation = int(self.policy(state, sim, rng))
                raw_values = []
            ordered = sorted(raw_values, key=lambda item: item[1], reverse=True)
            ranks = {action: rank for rank, (action, _value) in enumerate(ordered, start=1)}
            values = [
                {
                    "action": DIRECTION_NAMES[action],
                    "action_index": action,
                    "value": value,
                    "rank": ranks[action],
                }
                for action, value in raw_values
            ]
            margin = None
            normalized_margin = None
            if len(ordered) >= 2:
                margin = float(ordered[0][1] - ordered[1][1])
                normalized_margin = margin / max(1.0, abs(float(ordered[0][1])))
            return {
                "status": "ok",
                "action": DIRECTION_NAMES[recommendation],
                "action_index": recommendation,
                "action_values": values,
                "top_two_margin": margin,
                "normalized_top_two_margin": normalized_margin,
                "legal_actions": [DIRECTION_NAMES[action] for action in legal],
                "policy_file": str(self.policy_file),
                "policy_file_sha256": self.policy_file_sha256,
                "policy_spec": self.spec,
                "latency_ms": (time.perf_counter() - started) * 1000.0,
                "computed_at": _timestamp(),
            }


def _state_from_payload(payload: dict[str, Any], sim: ThreesSim):
    preview_data = payload["preview"]
    preview = preview_from_label(
        "large_candidates" if preview_data["kind"] == "bonus" else str(preview_data["kind"]),
        preview_data.get("candidates", ()),
    )
    cycle = payload["tile_cycle"]
    return sim.state_from_snapshot(
        payload["board"],
        preview,
        (
            {str(key): int(value) for key, value in cycle["small_counts"].items()},
            int(cycle["small_pos"]),
            int(cycle["small_seen_total"]),
            int(cycle["span_small_pos"]),
            bool(cycle["large_pending"]),
            int(cycle["max_tile"]),
        ),
        move_count=int(payload["move_count"]),
    )


def load_restart_catalog(manifest_path: Path | None) -> dict[str, dict[str, Any]]:
    if manifest_path is None:
        return {}
    payload = json.loads(manifest_path.read_text())
    roots = payload.get("roots")
    if not isinstance(roots, list):
        raise ValueError("Restart manifest must contain a roots list")
    catalog: dict[str, dict[str, Any]] = {}
    for root in roots:
        root_id = str(root["root_id"])
        source_path = Path(root["source_replay"])
        source_bytes = source_path.read_bytes()
        expected_hash = root.get("source_replay_sha256")
        actual_hash = hashlib.sha256(source_bytes).hexdigest()
        if expected_hash and actual_hash != expected_hash:
            raise ValueError(f"Restart source hash mismatch: {root_id}")
        source = json.loads(source_bytes)
        frame_index = int(root["source_frame_index"])
        source_state = source["frames"][frame_index]["state"]
        if source_state != root["state"]:
            raise ValueError(f"Restart state differs from source frame: {root_id}")
        if root_id in catalog:
            raise ValueError(f"Duplicate restart root: {root_id}")
        catalog[root_id] = {
            **root,
            "restart_manifest": str(manifest_path),
            "source_stream_metadata": source.get("stream_metadata"),
            "source_seed": source.get("seed"),
            "source_policy": source.get("policy"),
            "source_root_origin": source.get("root_origin", ORIGIN_HUMAN),
            "source_root_replay": source.get("root_replay", str(source_path)),
            "source_root_seed": source.get("root_seed", source.get("seed")),
            "source_root_frame_index": source.get("root_frame_index", 0),
        }
    return catalog


def _step_payload(
    action: int,
    before,
    after,
    info,
    decision_time_ms: float | None,
    recommendation: dict[str, Any] | None,
) -> dict[str, Any]:
    recommended_action = None if recommendation is None else recommendation.get("action")
    return {
        "action": DIRECTION_NAMES[int(action)],
        "action_index": int(action),
        "preview_used": preview_payload(before),
        "inserted_value": info.inserted_value,
        "inserted_pos": list(info.inserted_pos) if info.inserted_pos is not None else None,
        "eligible_positions": [list(pos) for pos in info.eligible_positions],
        "merge_score_delta": int(info.merge_score_delta),
        "score_delta": int(info.score_delta),
        "terminal_merge": bool(info.terminal_merge),
        "score_before": int(score_board(before.board)),
        "score_after": int(score_board(after.board)),
        "max_tile_before": int(before.max_tile),
        "max_tile_after": int(after.max_tile),
        "decision_time_ms": None if decision_time_ms is None else float(decision_time_ms),
        "model_recommendation": recommendation,
        "recommended_action": recommended_action,
        "human_model_agreement": (
            None if recommended_action is None else DIRECTION_NAMES[int(action)] == recommended_action
        ),
        "recorded_at": _timestamp(),
    }


class HumanGameSession:
    """One exact human-controlled simulator game with durable per-move writes."""

    def __init__(
        self,
        data_root: Path,
        *,
        starter_tile: int | None = 1536,
        player_id: str = "local_player",
        logical_seed: int | None = None,
        deck_stream_id: int | None = None,
        slot_stream_id: int | None = None,
        restart_record: dict[str, Any] | None = None,
        advisor: PolicyAdvisor | None = None,
    ) -> None:
        self.lock = threading.RLock()
        self.data_root = Path(data_root)
        self.logical_seed = int(logical_seed if logical_seed is not None else secrets.randbits(63))
        self.deck_stream_id = int(deck_stream_id if deck_stream_id is not None else secrets.randbits(63))
        self.slot_stream_id = int(slot_stream_id if slot_stream_id is not None else secrets.randbits(63))
        self.starter_tile = None if starter_tile is None else int(starter_tile)
        self.player_id = str(player_id or "local_player")[:80]
        self.restart_record = restart_record
        self.advisor = advisor
        self.quality_annotation: str | None = None
        self.created_at = _timestamp()
        safe_stamp = time.strftime("%Y%m%d_%H%M%S")
        prefix = "human_restart" if restart_record is not None else "human"
        self.session_id = f"{prefix}_{safe_stamp}_{self.logical_seed:016x}"
        self.out_dir = self.data_root / self.session_id
        self.replay_path = self.out_dir / "replay.json"
        self.html_path = self.out_dir / "replay.html"
        self.status = "active"
        self.sim = ThreesSim.from_stream_ids(
            deck_stream_id=self.deck_stream_id,
            slot_stream_id=self.slot_stream_id,
            starter_tile=self.starter_tile,
        )
        self.state = self.sim.reset() if restart_record is None else _state_from_payload(restart_record["state"], self.sim)
        self.initial_score = int(score_board(self.state.board))
        self.frames: list[dict[str, Any]] = [{
            "index": 0,
            "state": state_payload(self.state, self.sim),
            "recommendation": self._recommendation(),
            "move": None,
        }]
        self._persist()

    def _recommendation(self) -> dict[str, Any] | None:
        if self.advisor is None:
            return None
        seed_material = f"{self.logical_seed}:{self.state.move_count}:{self.advisor.policy_file_sha256 or ''}"
        decision_seed = int.from_bytes(hashlib.sha256(seed_material.encode()).digest()[:8], "little")
        return self.advisor.recommend(self.state, self.sim, decision_seed=decision_seed)

    @property
    def score_minus_starter(self) -> int:
        starter_score = 0 if self.starter_tile is None else score_tile(self.starter_tile)
        return int(score_board(self.state.board) - starter_score)

    def replay_payload(self) -> dict[str, Any]:
        initial_score = int(self.frames[0]["state"]["score"])
        policy = RESTART_POLICY_NAME if self.restart_record is not None else POLICY_NAME
        provenance = direct_root_fields(
            origin=ORIGIN_HUMAN,
            seed=self.logical_seed,
            policy=policy,
            replay_path=self.replay_path,
            first_score=initial_score,
        )
        restart_metadata = None
        if self.restart_record is not None:
            root = self.restart_record
            provenance = {
                "replay_origin": ORIGIN_CONTINUATION,
                "source_origin": ORIGIN_HUMAN,
                "source_replay": root["source_replay"],
                "source_seed": root.get("source_seed"),
                "source_frame_index": int(root["source_frame_index"]),
                "source_policy": root.get("source_policy"),
                "root_origin": root.get("source_root_origin", ORIGIN_HUMAN),
                "root_replay": root.get("source_root_replay", root["source_replay"]),
                "root_seed": root.get("source_root_seed"),
                "root_frame_index": root.get("source_root_frame_index", 0),
                "root_move_count": 0,
                "root_score": None,
                "root_policy": root.get("source_policy"),
                "root_policy_family": "human_player",
            }
            restart_metadata = {
                "root_id": root["root_id"],
                "manifest": root["restart_manifest"],
                "source_replay_sha256": root.get("source_replay_sha256"),
                "source_frame_index": int(root["source_frame_index"]),
                "source_move_count": int(root["source_move_count"]),
                "source_stream_metadata": root.get("source_stream_metadata"),
                "continuation_stream_metadata": self.sim.stream_metadata(),
            }
        return {
            "kind": "human_restart_replay_v1" if self.restart_record is not None else "human_simulator_replay_v1",
            "policy": policy,
            "policy_family": "human_restart" if self.restart_record is not None else "human_player",
            "player_id": self.player_id,
            "session_id": self.session_id,
            "seed": self.logical_seed,
            "starter_tile": self.starter_tile,
            "created_at": self.created_at,
            "updated_at": _timestamp(),
            "status": self.status,
            "quality_annotation": self.quality_annotation,
            "dashboard_eligible": False,
            "dashboard_record_eligible": False,
            "dashboard_record_eligibility_reason": (
                "human continuation" if self.restart_record is not None else "human development game"
            ),
            "human_input": {
                "interface": "local_web_v1",
                "player_id": self.player_id,
                "exact_simulator": True,
                "model_assistance_visible": self.advisor is not None,
            },
            "model_assistance": None if self.advisor is None else self.advisor.metadata(),
            "stream_metadata": self.sim.stream_metadata(),
            **provenance,
            "restart_metadata": restart_metadata,
            "final_score": int(score_board(self.state.board)),
            "final_score_minus_starter": self.score_minus_starter,
            "final_score_delta_from_restart": int(score_board(self.state.board) - self.initial_score),
            "final_moves": int(self.state.move_count),
            "final_max_tile": int(self.state.max_tile),
            "game_over": bool(self.state.game_over),
            "frames": self.frames,
        }

    def public_payload(self, *, moved: bool | None = None) -> dict[str, Any]:
        payload = {
            "session_id": self.session_id,
            "status": self.status,
            "state": state_payload(self.state, self.sim),
            "score_minus_starter": self.score_minus_starter,
            "recorded_frames": len(self.frames),
            "replay_json": str(self.replay_path),
            "replay_html": str(self.html_path) if self.html_path.exists() else None,
            "replay_url": f"/api/games/{self.session_id}/replay",
            "quality_annotation": self.quality_annotation,
            "restart_root_id": None if self.restart_record is None else self.restart_record["root_id"],
            "dashboard_eligible": False,
            "recommendation": self.frames[-1].get("recommendation"),
        }
        if moved is not None:
            payload["moved"] = bool(moved)
        return payload

    def move(self, action: int | str, *, decision_time_ms: float | None = None) -> dict[str, Any]:
        with self.lock:
            if self.status != "active" or self.state.game_over:
                return self.public_payload(moved=False)
            action_index = DIRECTION_NAMES.index(action) if isinstance(action, str) else int(action)
            if action_index < 0 or action_index >= len(DIRECTION_NAMES):
                raise ValueError(f"Unsupported action: {action!r}")
            before = self.state
            recommendation = self.frames[-1].get("recommendation")
            after, info = self.sim.step(before, action_index)
            if not info.moved:
                return self.public_payload(moved=False)
            self.state = after
            self.frames.append(
                {
                    "index": len(self.frames),
                    "state": state_payload(after, self.sim),
                    "recommendation": self._recommendation(),
                    "move": _step_payload(
                        action_index,
                        before,
                        after,
                        info,
                        decision_time_ms,
                        recommendation,
                    ),
                }
            )
            if after.game_over:
                self.status = "game_over"
            self._persist()
            return self.public_payload(moved=True)

    def finish(self) -> dict[str, Any]:
        with self.lock:
            if self.status == "active":
                self.status = "ended_by_player"
            self._persist(write_replay_html=True)
            return self.public_payload()

    def annotate_quality(self, annotation: str) -> dict[str, Any]:
        with self.lock:
            if annotation not in QUALITY_ANNOTATIONS:
                raise ValueError(f"Unsupported quality annotation: {annotation!r}")
            self.quality_annotation = annotation
            self._persist(write_replay_html=self.state.game_over)
            return self.public_payload()

    def ensure_replay_html(self) -> Path:
        with self.lock:
            write_html(self.html_path, self.replay_payload())
            return self.html_path

    def _persist(self, *, write_replay_html: bool = False) -> None:
        replay = self.replay_payload()
        _atomic_json(self.replay_path, replay)
        _atomic_json(
            self.out_dir / "session.json",
            {
                "kind": "human_simulator_session_v1",
                "session_id": self.session_id,
                "status": self.status,
                "quality_annotation": self.quality_annotation,
                "dashboard_eligible": False,
                "player_id": self.player_id,
                "seed": self.logical_seed,
                "starter_tile": self.starter_tile,
                "stream_metadata": self.sim.stream_metadata(),
                "moves": int(self.state.move_count),
                "score": int(score_board(self.state.board)),
                "score_minus_starter": self.score_minus_starter,
                "max_tile": int(self.state.max_tile),
                "game_over": bool(self.state.game_over),
                "updated_at": _timestamp(),
                "replay_json": str(self.replay_path),
                "replay_html": str(self.html_path) if self.html_path.exists() else None,
            },
        )
        if write_replay_html or self.state.game_over:
            write_html(self.html_path, replay)


class HumanPlayHTTPServer(ThreadingHTTPServer):
    def __init__(
        self,
        address: tuple[str, int],
        handler,
        *,
        data_root: Path,
        html_path: Path,
        restart_manifest: Path | None = None,
        policy_file: Path = DEFAULT_POLICY_FILE,
    ):
        super().__init__(address, handler)
        self.data_root = Path(data_root)
        self.html_path = Path(html_path)
        self.sessions: dict[str, HumanGameSession] = {}
        self.sessions_lock = threading.RLock()
        self.restart_catalog = load_restart_catalog(restart_manifest)
        self.advisor = PolicyAdvisor(policy_file)

    def new_session(self, payload: dict[str, Any]) -> HumanGameSession:
        restart_root_id = payload.get("restart_root_id")
        restart_record = None
        if restart_root_id is not None:
            restart_record = self.restart_catalog.get(str(restart_root_id))
            if restart_record is None:
                raise ValueError(f"Unknown restart root: {restart_root_id}")
        starter = restart_record.get("starter_tile", 1536) if restart_record else payload.get("starter_tile", 1536)
        starter_tile = None if starter is None else int(starter)
        session = HumanGameSession(
            self.data_root,
            starter_tile=starter_tile,
            player_id=str(payload.get("player_id") or "local_player"),
            restart_record=restart_record,
            advisor=self.advisor,
        )
        with self.sessions_lock:
            self.sessions[session.session_id] = session
        return session


class HumanPlayHandler(BaseHTTPRequestHandler):
    server: HumanPlayHTTPServer

    def log_message(self, format: str, *args: object) -> None:
        print(f"[{self.log_date_time_string()}] {format % args}", flush=True)

    def _send_json(self, payload: dict[str, Any], status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        payload = json.loads(self.rfile.read(length))
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object")
        return payload

    def _session(self, session_id: str) -> HumanGameSession:
        with self.server.sessions_lock:
            session = self.server.sessions.get(session_id)
        if session is None:
            raise KeyError(session_id)
        return session

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", "/human-play"):
            body = self.server.html_path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/api/health":
            self._send_json({
                "status": "ok",
                "active_sessions": len(self.server.sessions),
                "restart_roots": len(self.server.restart_catalog),
                "advisor": self.server.advisor.metadata(),
            })
            return
        if path == "/api/restart-roots":
            self._send_json({
                "roots": [
                    {
                        "root_id": root["root_id"],
                        "ancestry_cluster": root.get("ancestry_cluster"),
                        "role": root.get("role"),
                        "source_frame_index": root.get("source_frame_index"),
                        "source_move_count": root.get("source_move_count"),
                    }
                    for root in self.server.restart_catalog.values()
                ]
            })
            return
        parts = [part for part in path.split("/") if part]
        if len(parts) == 3 and parts[:2] == ["api", "games"]:
            try:
                self._send_json(self._session(parts[2]).public_payload())
            except KeyError:
                self._send_json({"error": "session_not_found"}, HTTPStatus.NOT_FOUND)
            return
        if len(parts) == 4 and parts[:2] == ["api", "games"] and parts[3] == "replay":
            try:
                replay_path = self._session(parts[2]).ensure_replay_html()
                body = replay_path.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            except KeyError:
                self._send_json({"error": "session_not_found"}, HTTPStatus.NOT_FOUND)
            return
        self._send_json({"error": "not_found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            payload = self._read_json()
            if path == "/api/games":
                session = self.server.new_session(payload)
                self._send_json(session.public_payload(), HTTPStatus.CREATED)
                return
            parts = [part for part in path.split("/") if part]
            if len(parts) == 4 and parts[:2] == ["api", "games"]:
                session = self._session(parts[2])
                if parts[3] == "moves":
                    result = session.move(
                        payload.get("action"),
                        decision_time_ms=payload.get("decision_time_ms"),
                    )
                    self._send_json(result)
                    return
                if parts[3] == "finish":
                    self._send_json(session.finish())
                    return
                if parts[3] == "quality":
                    self._send_json(session.annotate_quality(str(payload.get("annotation") or "")))
                    return
            self._send_json({"error": "not_found"}, HTTPStatus.NOT_FOUND)
        except KeyError:
            self._send_json({"error": "session_not_found"}, HTTPStatus.NOT_FOUND)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            self._send_json({"error": "bad_request", "detail": str(exc)}, HTTPStatus.BAD_REQUEST)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--html", type=Path, default=DEFAULT_HTML)
    parser.add_argument("--restart-manifest", type=Path)
    parser.add_argument("--policy-file", type=Path, default=DEFAULT_POLICY_FILE)
    args = parser.parse_args()

    server = HumanPlayHTTPServer(
        (str(args.host), int(args.port)),
        HumanPlayHandler,
        data_root=args.data_root,
        html_path=args.html,
        restart_manifest=args.restart_manifest,
        policy_file=args.policy_file,
    )
    print(
        json.dumps(
            {
                "url": f"http://{args.host}:{args.port}/",
                "data_root": str(args.data_root),
                "restart_roots": len(server.restart_catalog),
                "advisor": server.advisor.metadata(),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
