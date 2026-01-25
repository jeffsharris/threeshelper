import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import window_stream as ws


ARROW_KEYS = {"left", "right", "up", "down"}


def _iso_ts(ts: Optional[float] = None) -> str:
    stamp = ts if ts is not None else time.time()
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(stamp))


def _pick_window(window_id: Optional[int], prefix: str) -> Tuple[int, str, str]:
    if window_id is not None:
        return window_id, "(provided id)", ""
    auto_pick = ws.find_window_by_prefix(prefix or "")
    if auto_pick:
        return auto_pick
    return ws.choose_window_interactive()


def _capture_frame(window_id: int) -> Tuple[np.ndarray, Dict[str, object]]:
    img = ws.capture_window(window_id)
    arr = np.array(img)
    board_roi, board_box = ws.find_board_roi(arr)
    preview_roi, preview_box = ws.find_preview_roi(arr)
    sig, sig_box = ws.board_signature(arr)
    meta = {
        "board_box": list(board_box),
        "preview_box": list(preview_box),
        "signature_box": list(sig_box),
    }
    return arr, {"board_roi": board_roi, "preview_roi": preview_roi, "sig": sig, "meta": meta}


def _record_event(
    window_id: int,
    event_dir: Path,
    key: str,
    ts_event: float,
    frames: int,
    interval: float,
) -> List[Dict[str, object]]:
    event_dir.mkdir(parents=True, exist_ok=False)
    start = time.time()
    prev_sig = None
    first_sig = None
    entries: List[Dict[str, object]] = []
    for idx in range(frames):
        frame_start = time.time()
        arr, data = _capture_frame(window_id)
        board_roi = data["board_roi"]
        preview_roi = data["preview_roi"]
        sig = data["sig"]
        meta = data["meta"]

        if first_sig is None:
            first_sig = sig
        diff_prev = None if prev_sig is None else ws.board_signature_diff(prev_sig, sig)
        diff_first = None if first_sig is None else ws.board_signature_diff(first_sig, sig)
        prev_sig = sig

        full_name = f"full_{idx:03d}.png"
        board_name = f"board_{idx:03d}.png"
        preview_name = f"preview_{idx:03d}.png"
        Image.fromarray(arr).save(event_dir / full_name)
        Image.fromarray(board_roi).save(event_dir / board_name)
        Image.fromarray(preview_roi).save(event_dir / preview_name)

        entry = {
            "frame": idx,
            "ts_event": _iso_ts(ts_event),
            "ts_capture": _iso_ts(frame_start),
            "elapsed_s": round(frame_start - start, 4),
            "diff_prev": diff_prev,
            "diff_first": diff_first,
            "files": {
                "full": full_name,
                "board": board_name,
                "preview": preview_name,
            },
        }
        entry.update(meta)
        entries.append(entry)

        next_t = start + (idx + 1) * interval
        sleep_for = next_t - time.time()
        if sleep_for > 0:
            time.sleep(sleep_for)
    summary = {
        "key": key,
        "ts_event": _iso_ts(ts_event),
        "frames": frames,
        "interval_s": interval,
        "entries": entries,
    }
    (event_dir / "frames.json").write_text(
        json.dumps(summary, indent=2, default=ws._json_safe)
    )
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture a burst of frames after an arrow key to calibrate delays."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets"),
        help="Base output directory (default: datasets).",
    )
    parser.add_argument(
        "--window-id",
        type=int,
        help="Window ID to monitor (otherwise an iPhone Mirroring window is auto-picked).",
    )
    parser.add_argument(
        "--auto-window-prefix",
        default="iPhone Mirroring",
        help="Auto-select the first window whose title starts with this prefix.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=1.0,
        help="Seconds to capture after the key press (default: 1.0).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.05,
        help="Seconds between frames (default: 0.05).",
    )
    parser.add_argument(
        "--events",
        type=int,
        default=1,
        help="Number of arrow key events to record before exiting (default: 1).",
    )
    args = parser.parse_args()

    window_id, app_name, win_name = _pick_window(args.window_id, args.auto_window_prefix)
    print(f"Target window {window_id} ({app_name} – {win_name})")

    session = f"capture_{time.strftime('%Y%m%d_%H%M%S', time.localtime())}"
    session_dir = args.output_dir / session
    session_dir.mkdir(parents=True, exist_ok=False)
    (session_dir / "session.json").write_text(
        json.dumps(
            {
                "created": _iso_ts(),
                "window": {"id": window_id, "app": app_name, "title": win_name},
                "duration_s": args.duration,
                "interval_s": args.interval,
                "events": args.events,
            },
            indent=2,
        )
    )

    events = ws.start_key_listener()
    print("Press an arrow key to record frames.")
    event_idx = 0
    frames = max(1, int(args.duration / args.interval) + 1)
    while event_idx < args.events:
        ts_event, key = events.get()
        if key not in ARROW_KEYS:
            continue
        event_idx += 1
        event_dir = session_dir / f"event_{event_idx:03d}_{key}"
        print(f"[{_iso_ts(ts_event)}] recording {frames} frames after {key}...")
        _record_event(
            window_id,
            event_dir,
            key,
            ts_event,
            frames,
            args.interval,
        )
        print(f"Wrote {event_dir}/frames.json")


if __name__ == "__main__":
    main()
