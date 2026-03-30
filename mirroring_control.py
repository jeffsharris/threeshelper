import argparse
import random
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

import Quartz

import window_stream as ws


APP_BUNDLE_ID = "com.apple.ScreenContinuity"
DIRECTIONS = ("up", "down", "left", "right")
SCENE_GAME = "game"
SCENE_TITLE = "title"
SCENE_GAME_OVER = "game_over"
SCENE_POSTGAME = "postgame"
SCENE_MENU = "menu"
SCENE_END_CONFIRM = "end_confirm"
SCENE_PHONE_IN_USE = "phone_in_use"
SCENE_UNKNOWN = "unknown"
SCENE_SCREEN_OFF = "screen_off"
TITLE_PLAY_TAP = (0.50, 0.78)
POSTGAME_RETRY_TAP = (0.16, 0.145)
INGAME_MENU_TAP = (0.16, 0.15)
MENU_MAIN_MENU_TAP = (0.50, 0.75)
END_CONFIRM_END_GAME_TAP = (0.50, 0.56)
ENDGAME_SUMMARY_SWIPE_DIRECTION = "left"
ENDGAME_SUMMARY_SWIPE_SPAN_RATIO = 0.26
ENDGAME_SUMMARY_SWIPE_DURATION = 0.08
ENDGAME_SUMMARY_SWIPE_STEPS = 8
ENDGAME_SUMMARY_START_REL_X = 0.50
ENDGAME_SUMMARY_START_REL_Y = 0.55
SCENE_REF_DIR = Path(__file__).with_name("scene_refs")
SCENE_MATCH_THRESHOLD = {
    SCENE_TITLE: 0.01,
    SCENE_GAME_OVER: 0.0012,
    SCENE_POSTGAME: 0.005,
    SCENE_MENU: 0.02,
    SCENE_END_CONFIRM: 0.02,
    SCENE_PHONE_IN_USE: 0.03,
}
DARK_FRAME_MEAN_THRESHOLD = 3.0
DARK_FRAME_MAX_THRESHOLD = 12.0
KNOWN_PREVIEW_LABELS = {"red", "blue", "gray", "large_candidates"}
_SPECIAL_REF_CACHE: Dict[Tuple[str, Tuple[int, int]], np.ndarray] = {}


def _ref_signature(values: Sequence[int]) -> np.ndarray:
    return np.array(values, dtype=np.float32).reshape((6, 6, 3)) / 255.0


SCENE_FINGERPRINTS = {
    SCENE_TITLE: {
        "play": {
            "box": (0.10, 0.70, 0.90, 0.84),
            "ref": _ref_signature(
                (
                    40, 39, 50, 40, 39, 50, 40, 39, 50, 40, 39, 50, 40, 39, 50, 40,
                    39, 50, 40, 39, 50, 40, 39, 50, 40, 39, 50, 40, 39, 50, 40, 39,
                    50, 40, 39, 50, 43, 42, 55, 43, 42, 55, 43, 42, 55, 43, 42, 55,
                    43, 42, 55, 43, 42, 55, 78, 77, 99, 81, 81, 102, 81, 80, 102, 88,
                    88, 108, 89, 88, 109, 76, 76, 98, 116, 116, 135, 140, 140, 156,
                    114, 114, 133, 152, 152, 166, 149, 149, 164, 114, 114, 133, 69,
                    68, 91, 77, 77, 98, 68, 68, 90, 76, 76, 97, 81, 81, 101, 73, 73,
                    94,
                )
            ),
        },
        "hero": {
            "box": (0.14, 0.26, 0.86, 0.63),
            "ref": _ref_signature(
                (
                    42, 41, 52, 41, 40, 51, 42, 41, 52, 41, 41, 52, 41, 40, 51, 42,
                    41, 52, 108, 111, 119, 86, 89, 102, 89, 94, 107, 87, 90, 104, 82,
                    85, 98, 88, 93, 106, 127, 134, 142, 98, 105, 118, 101, 109, 124,
                    96, 104, 119, 88, 97, 111, 88, 101, 117, 100, 108, 123, 90, 96,
                    110, 95, 103, 118, 86, 99, 115, 79, 95, 111, 80, 100, 117, 95,
                    106, 122, 85, 94, 109, 88, 98, 113, 96, 108, 124, 84, 100, 116,
                    84, 104, 122, 96, 107, 122, 85, 100, 117, 95, 104, 118, 99, 106,
                    120, 89, 108, 126, 75, 99, 118,
                )
            ),
        },
    },
    SCENE_POSTGAME: {
        "retry": {
            "box": (0.04, 0.08, 0.25, 0.20),
            "ref": _ref_signature(
                (
                    39, 38, 48, 40, 39, 49, 56, 55, 64, 82, 81, 90, 87, 86, 95, 87,
                    86, 94, 40, 39, 49, 40, 39, 49, 41, 40, 50, 42, 41, 51, 42, 41,
                    51, 42, 41, 51, 40, 39, 49, 40, 39, 49, 40, 39, 49, 40, 39, 49,
                    40, 39, 49, 40, 39, 49, 40, 39, 49, 40, 39, 49, 40, 39, 49, 40,
                    39, 49, 40, 39, 49, 40, 39, 49, 40, 39, 49, 42, 41, 52, 45, 44,
                    57, 50, 50, 62, 47, 46, 60, 45, 44, 57, 40, 39, 49, 48, 48, 63,
                    64, 64, 88, 130, 130, 146, 120, 119, 137, 65, 64, 89,
                )
            ),
        },
        "summary": {
            "box": (0.17, 0.23, 0.83, 0.66),
            "ref": _ref_signature(
                (
                    41, 40, 50, 41, 40, 50, 46, 45, 56, 45, 45, 55, 41, 40, 50, 42,
                    41, 51, 54, 54, 65, 72, 74, 84, 70, 72, 84, 70, 72, 84, 70, 72,
                    84, 54, 54, 65, 87, 88, 97, 140, 143, 149, 122, 128, 144, 122,
                    128, 144, 123, 129, 145, 83, 86, 99, 79, 82, 95, 122, 125, 141,
                    132, 124, 140, 132, 123, 140, 133, 125, 141, 87, 84, 97, 79, 82,
                    95, 133, 119, 136, 192, 108, 128, 197, 105, 123, 197, 105, 123,
                    113, 75, 89, 81, 84, 97, 121, 134, 155, 122, 155, 192, 135, 125,
                    144, 140, 123, 140, 90, 83, 97,
                )
            ),
        },
    },
    SCENE_MENU: {
        "main_menu": {
            "box": (0.08, 0.68, 0.92, 0.82),
            "ref": _ref_signature(
                (
                    42, 39, 51, 42, 40, 51, 42, 40, 51, 42, 40, 51, 42, 40, 51, 42,
                    39, 51, 159, 73, 91, 170, 79, 97, 170, 79, 97, 170, 80, 98, 170,
                    80, 98, 161, 73, 91, 236, 114, 134, 253, 163, 177, 253, 155, 170,
                    253, 163, 177, 253, 161, 176, 239, 109, 130, 231, 99, 122, 248,
                    127, 147, 247, 122, 143, 248, 124, 145, 248, 128, 148, 234, 100,
                    123, 148, 60, 92, 158, 62, 96, 158, 62, 96, 158, 62, 96, 158, 62,
                    96, 150, 61, 93, 46, 40, 53, 47, 40, 53, 47, 40, 53, 47, 40, 53,
                    47, 40, 53, 46, 40, 53,
                )
            ),
        },
        "options": {
            "box": (0.05, 0.28, 0.95, 0.60),
            "ref": _ref_signature(
                (
                    44, 43, 55, 45, 44, 56, 44, 44, 55, 40, 39, 50, 40, 39, 50, 40,
                    39, 50, 50, 49, 61, 50, 50, 62, 47, 47, 58, 39, 38, 49, 43, 42,
                    54, 48, 47, 60, 49, 48, 60, 44, 43, 54, 36, 35, 45, 36, 35, 45,
                    48, 47, 60, 61, 60, 76, 47, 46, 58, 43, 42, 53, 41, 41, 51, 36,
                    35, 45, 49, 49, 61, 66, 65, 81, 54, 45, 58, 44, 42, 53, 41, 40,
                    51, 37, 36, 46, 64, 47, 59, 94, 61, 75, 77, 52, 64, 50, 46, 57,
                    42, 41, 51, 36, 35, 45, 92, 53, 67, 150, 73, 93,
                )
            ),
        },
    },
    SCENE_END_CONFIRM: {
        "end_game": {
            "box": (0.08, 0.48, 0.92, 0.62),
            "ref": _ref_signature(
                (
                    40, 39, 50, 40, 39, 50, 40, 39, 50, 40, 39, 50, 40, 39, 50, 40,
                    39, 50, 41, 39, 50, 41, 39, 50, 41, 39, 50, 41, 39, 50, 41, 39,
                    50, 41, 39, 50, 151, 70, 88, 161, 76, 94, 161, 75, 93, 161, 76,
                    94, 161, 76, 94, 153, 70, 88, 235, 100, 121, 253, 152, 168, 253,
                    147, 163, 253, 157, 172, 253, 154, 169, 238, 98, 121, 232, 95,
                    118, 249, 127, 147, 249, 125, 146, 249, 133, 152, 249, 126, 146,
                    235, 95, 119, 171, 65, 101, 182, 68, 105, 182, 68, 105, 182, 68,
                    105, 182, 68, 105, 173, 66, 101,
                )
            ),
        },
        "caution": {
            "box": (0.10, 0.30, 0.85, 0.46),
            "ref": _ref_signature(
                (
                    38, 36, 47, 37, 36, 47, 37, 36, 47, 37, 36, 47, 37, 36, 47, 37,
                    36, 47, 35, 33, 43, 35, 34, 43, 36, 35, 45, 37, 36, 46, 35, 34,
                    43, 35, 34, 43, 36, 35, 44, 37, 37, 46, 48, 48, 58, 54, 55, 65,
                    39, 38, 47, 36, 36, 45, 43, 42, 52, 54, 54, 65, 55, 56, 67, 56,
                    57, 67, 56, 57, 68, 49, 49, 59, 35, 34, 43, 46, 46, 56, 53, 53,
                    63, 55, 55, 66, 53, 53, 64, 38, 37, 47, 36, 35, 44, 37, 36, 45,
                    37, 36, 45, 37, 36, 45, 37, 36, 45, 36, 35, 44,
                )
            ),
        },
    },
    SCENE_PHONE_IN_USE: {
        "text": {
            "box": (0.18, 0.45, 0.82, 0.66),
            "ref": _ref_signature(
                (
                    234, 236, 237, 233, 235, 236, 233, 235, 236, 233, 235, 236,
                    234, 236, 237, 234, 236, 237, 234, 236, 237, 215, 217, 218,
                    203, 204, 205, 207, 209, 210, 215, 217, 217, 234, 236, 237,
                    222, 224, 225, 219, 221, 221, 216, 217, 218, 217, 219, 219,
                    219, 220, 221, 224, 226, 227, 231, 233, 234, 229, 231, 232,
                    217, 219, 220, 218, 220, 221, 229, 231, 232, 231, 233, 234,
                    225, 227, 228, 221, 222, 223, 219, 221, 222, 221, 223, 224,
                    220, 222, 223, 225, 227, 228, 233, 235, 235, 226, 231, 236,
                    190, 209, 238, 188, 208, 238, 224, 229, 236, 233, 235, 236,
                )
            ),
        },
        "button": {
            "box": (0.33, 0.63, 0.67, 0.76),
            "ref": _ref_signature(
                (
                    231, 235, 238, 203, 218, 240, 192, 211, 241, 192, 211, 241,
                    199, 216, 241, 229, 234, 239, 187, 208, 241, 69, 138, 248,
                    65, 136, 250, 65, 135, 249, 62, 134, 249, 162, 193, 243,
                    182, 205, 241, 64, 135, 249, 69, 138, 250, 72, 140, 250,
                    60, 132, 250, 156, 190, 243, 230, 234, 238, 197, 214, 240,
                    186, 208, 241, 186, 208, 241, 194, 212, 241, 227, 232, 239,
                    235, 237, 238, 235, 237, 238, 235, 237, 238, 235, 237, 238,
                    236, 238, 239, 236, 238, 239, 235, 237, 238, 235, 237, 238,
                    235, 237, 238, 235, 237, 238, 236, 238, 239, 236, 238, 239,
                )
            ),
        },
    },
}

SPECIAL_SCENE_MATCHERS = {
    SCENE_GAME_OVER: (
        {
            "box": (0.14, 0.82, 0.86, 0.90),
            "size": (64, 20),
            "refs": (
                SCENE_REF_DIR / "game_over_prompt_see_score.png",
                SCENE_REF_DIR / "game_over_prompt_save_score.png",
            ),
        },
    ),
    SCENE_POSTGAME: (
        {
            "box": (0.08, 0.12, 0.92, 0.30),
            "size": (64, 28),
            "refs": (
                SCENE_REF_DIR / "postgame_summary_top.png",
            ),
        },
    ),
    SCENE_PHONE_IN_USE: (
        {
            "box": (0.22, 0.30, 0.78, 0.78),
            "size": (64, 64),
            "refs": (
                SCENE_REF_DIR / "phone_in_use_center.png",
                SCENE_REF_DIR / "phone_in_use_center_20260330.png",
            ),
        },
    ),
}


@dataclass(frozen=True)
class WindowBounds:
    x: float
    y: float
    width: float
    height: float

    def point(self, rel_x: float, rel_y: float) -> Tuple[float, float]:
        return (self.x + self.width * rel_x, self.y + self.height * rel_y)


@dataclass
class FrameState:
    arr: np.ndarray
    board: List[List[str]]
    preview_label: str
    preview_debug: Dict[str, object]
    board_sig: np.ndarray


@dataclass
class ScreenState:
    arr: np.ndarray
    scene: str
    scene_score: float
    scene_scores: Dict[str, float]
    frame: Optional[FrameState] = None
    candidate_frame: Optional[FrameState] = None


@dataclass
class MoveResult:
    direction: str
    changed: bool
    scene: str
    board_delta: Optional[float]
    game_over: bool = False


def activate_mirroring() -> None:
    subprocess.run(
        ["osascript", "-e", f'tell application id "{APP_BUNDLE_ID}" to activate'],
        check=True,
    )


def get_window_bounds(window_id: int) -> WindowBounds:
    info = Quartz.CGWindowListCopyWindowInfo(
        Quartz.kCGWindowListOptionIncludingWindow,
        window_id,
    )
    if not info:
        raise RuntimeError(f"Could not find window {window_id}")
    bounds = info[0].get("kCGWindowBounds")
    if not bounds:
        raise RuntimeError(f"Window {window_id} has no bounds")
    return WindowBounds(
        x=float(bounds["X"]),
        y=float(bounds["Y"]),
        width=float(bounds["Width"]),
        height=float(bounds["Height"]),
    )


def capture_window_image(window_id: int, backend: str) -> Image.Image:
    if backend == "quartz":
        image = Quartz.CGWindowListCreateImage(
            Quartz.CGRectNull,
            Quartz.kCGWindowListOptionIncludingWindow,
            window_id,
            Quartz.kCGWindowImageBoundsIgnoreFraming,
        )
        if image is None:
            raise RuntimeError("CGWindowListCreateImage returned None")
        width = Quartz.CGImageGetWidth(image)
        height = Quartz.CGImageGetHeight(image)
        bytes_per_row = Quartz.CGImageGetBytesPerRow(image)
        data = Quartz.CGDataProviderCopyData(Quartz.CGImageGetDataProvider(image))
        buf = bytes(data)
        return Image.frombuffer(
            "RGBA",
            (width, height),
            buf,
            "raw",
            "BGRA",
            bytes_per_row,
            1,
        ).convert("RGB")

    fd, path = tempfile.mkstemp(suffix=".png")
    try:
        subprocess.run(["screencapture", "-x", "-l", str(window_id), path], check=True)
        return Image.open(path).convert("RGB")
    finally:
        try:
            Path(path).unlink()
        except OSError:
            pass


def _crop_from_rel_box(arr: np.ndarray, box: Tuple[float, float, float, float]) -> Image.Image:
    h, w = arr.shape[:2]
    x0 = max(0, int(w * box[0]))
    y0 = max(0, int(h * box[1]))
    x1 = min(w, int(w * box[2]))
    y1 = min(h, int(h * box[3]))
    return Image.fromarray(arr[y0:y1, x0:x1]).convert("RGB")


def _crop_signature(arr: np.ndarray, box: Tuple[float, float, float, float]) -> np.ndarray:
    crop = _crop_from_rel_box(arr, box)
    small = crop.resize((6, 6), Image.Resampling.BILINEAR)
    return np.array(small, dtype=np.float32) / 255.0


def _crop_gray_signature(
    arr: np.ndarray,
    box: Tuple[float, float, float, float],
    size: Tuple[int, int],
) -> np.ndarray:
    crop = _crop_from_rel_box(arr, box).convert("L")
    small = crop.resize(size, Image.Resampling.BILINEAR)
    return np.array(small, dtype=np.float32) / 255.0


def _load_special_ref(path: Path, size: Tuple[int, int]) -> np.ndarray:
    key = (str(path), size)
    cached = _SPECIAL_REF_CACHE.get(key)
    if cached is not None:
        return cached
    ref = Image.open(path).convert("L").resize(size, Image.Resampling.BILINEAR)
    arr = np.array(ref, dtype=np.float32) / 255.0
    _SPECIAL_REF_CACHE[key] = arr
    return arr


def _scene_match_scores(arr: np.ndarray) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for scene, refs in SCENE_FINGERPRINTS.items():
        if scene == SCENE_POSTGAME:
            continue
        total = 0.0
        for ref in refs.values():
            sig = _crop_signature(arr, ref["box"])
            total += float(((sig - ref["ref"]) ** 2).mean())
        scores[scene] = total
    return scores


def _special_scene_scores(arr: np.ndarray) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for scene, matchers in SPECIAL_SCENE_MATCHERS.items():
        total = 0.0
        for matcher in matchers:
            sig = _crop_gray_signature(arr, matcher["box"], matcher["size"])
            best = min(
                float(((sig - _load_special_ref(Path(ref), matcher["size"])) ** 2).mean())
                for ref in matcher["refs"]
            )
            total += best
        scores[scene] = total
    return scores


def detect_scene(arr: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
    rgb = arr[:, :, :3].astype(np.float32, copy=False)
    if float(rgb.mean()) <= DARK_FRAME_MEAN_THRESHOLD and float(rgb.max()) <= DARK_FRAME_MAX_THRESHOLD:
        return SCENE_SCREEN_OFF, 0.0, {}
    scores = _scene_match_scores(arr)
    scores.update(_special_scene_scores(arr))
    best_scene, best_score = min(scores.items(), key=lambda item: item[1])
    threshold = SCENE_MATCH_THRESHOLD.get(best_scene, 0.0)
    if best_score <= threshold:
        return best_scene, best_score, scores
    return SCENE_UNKNOWN, best_score, scores


def _board_roi_stats(arr: np.ndarray) -> Tuple[float, float, float]:
    roi, _box = ws.find_board_roi(arr)
    gray = roi[:, :, :3].astype(np.float32).mean(axis=2)
    return (
        float(gray.mean()),
        float((gray < 80.0).mean()),
        float((gray > 180.0).mean()),
    )


def _is_plausible_game_frame(
    arr: np.ndarray,
    board: Sequence[Sequence[str]],
    preview_label: str,
) -> bool:
    roi_mean, dark_fraction, bright_fraction = _board_roi_stats(arr)
    if roi_mean > 170.0:
        return False
    if dark_fraction < 0.15:
        return False
    if bright_fraction > 0.55:
        return False
    if preview_label in KNOWN_PREVIEW_LABELS:
        return True

    nonempty = 0
    unknown = 0
    for row in board:
        for cell in row:
            if cell == ws.TOKEN_OTHER:
                unknown += 1
            elif cell != ws.TOKEN_EMPTY:
                nonempty += 1
    if nonempty >= 4 and unknown <= 4:
        return True
    if nonempty >= 2 and unknown == 0:
        return True
    return False


def _raise_for_unready_scene(scene: str) -> None:
    if scene == SCENE_SCREEN_OFF:
        raise RuntimeError("Mirrored device screen appears off. Wake the phone and try again.")
    if scene == SCENE_PHONE_IN_USE:
        raise RuntimeError("iPhone Mirroring is showing the reconnect prompt. Lock the iPhone and reconnect before continuing.")
    if scene == SCENE_UNKNOWN:
        raise RuntimeError("Current mirrored device screen is not a recognized Threes scene.")


def capture_frame(window_id: int, backend: str) -> FrameState:
    arr = np.array(capture_window_image(window_id, backend))
    board = ws.classify_board(arr)
    preview_label, preview_debug = ws.classify_array(arr)
    board_sig, _ = ws.board_signature(arr)
    return FrameState(
        arr=arr,
        board=board,
        preview_label=preview_label,
        preview_debug=preview_debug,
        board_sig=board_sig,
    )


def capture_screen_state(window_id: int, backend: str) -> ScreenState:
    arr = np.array(capture_window_image(window_id, backend))
    scene, score, scores = detect_scene(arr)
    frame = None
    candidate_frame = None
    if scene == SCENE_UNKNOWN:
        board = ws.classify_board(arr)
        preview_label, preview_debug = ws.classify_array(arr)
        board_sig, _ = ws.board_signature(arr)
        candidate_frame = FrameState(
            arr=arr,
            board=board,
            preview_label=preview_label,
            preview_debug=preview_debug,
            board_sig=board_sig,
        )
        if _is_plausible_game_frame(arr, board, preview_label):
            scene = SCENE_GAME
            frame = candidate_frame
    return ScreenState(
        arr=arr,
        scene=scene,
        scene_score=score,
        scene_scores=scores,
        frame=frame,
        candidate_frame=candidate_frame,
    )


def wait_for_scene(
    window_id: int,
    backend: str,
    expected_scenes: Sequence[str],
    timeout: float,
    poll: float,
) -> ScreenState:
    deadline = time.time() + timeout
    last_state = capture_screen_state(window_id, backend)
    while time.time() < deadline:
        if last_state.scene in expected_scenes:
            return last_state
        time.sleep(poll)
        last_state = capture_screen_state(window_id, backend)
    raise RuntimeError(
        f"Timed out waiting for scene {list(expected_scenes)!r}; last scene was {last_state.scene!r}"
    )


def wait_for_initial_game_state(
    window_id: int,
    backend: str,
    timeout: float,
    poll: float,
) -> ScreenState:
    deadline = time.time() + timeout
    last_state = capture_screen_state(window_id, backend)
    last_error: Optional[str] = None
    while time.time() < deadline:
        if last_state.scene == SCENE_GAME and last_state.frame is not None:
            last_error = ws._initial_state_error(last_state.frame.board, last_state.frame.preview_label)
            if last_error is None:
                return last_state
        time.sleep(poll)
        last_state = capture_screen_state(window_id, backend)
    if last_state.scene != SCENE_GAME:
        raise RuntimeError(
            f"Timed out waiting for initial game board; last scene was {last_state.scene!r}"
        )
    raise RuntimeError(
        f"Timed out waiting for initial game board to settle: {last_error or 'unknown error'}"
    )


def _post_mouse(event_type: int, point: Tuple[float, float]) -> None:
    event = Quartz.CGEventCreateMouseEvent(
        None,
        event_type,
        point,
        Quartz.kCGMouseButtonLeft,
    )
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)


def tap_window(
    window_id: int,
    rel_x: float,
    rel_y: float,
    focus_delay: float,
) -> None:
    activate_mirroring()
    time.sleep(focus_delay)
    bounds = get_window_bounds(window_id)
    point = bounds.point(rel_x, rel_y)
    _post_mouse(Quartz.kCGEventMouseMoved, point)
    time.sleep(0.01)
    _post_mouse(Quartz.kCGEventLeftMouseDown, point)
    time.sleep(0.02)
    _post_mouse(Quartz.kCGEventLeftMouseUp, point)


def start_game_from_title(window_id: int, focus_delay: float) -> None:
    tap_window(window_id, TITLE_PLAY_TAP[0], TITLE_PLAY_TAP[1], focus_delay)


def retry_from_postgame(window_id: int, focus_delay: float) -> None:
    tap_window(window_id, POSTGAME_RETRY_TAP[0], POSTGAME_RETRY_TAP[1], focus_delay)


def swipe_to_postgame_summary(window_id: int, focus_delay: float) -> None:
    drag_window(
        window_id,
        ENDGAME_SUMMARY_SWIPE_DIRECTION,
        span_ratio=ENDGAME_SUMMARY_SWIPE_SPAN_RATIO,
        duration=ENDGAME_SUMMARY_SWIPE_DURATION,
        steps=ENDGAME_SUMMARY_SWIPE_STEPS,
        start_rel_x=ENDGAME_SUMMARY_START_REL_X,
        start_rel_y=ENDGAME_SUMMARY_START_REL_Y,
        focus_delay=focus_delay,
    )


def advance_postgame_sequence(
    window_id: int,
    backend: str,
    focus_delay: float,
    transition_delay: float,
    timeout: float,
    poll: float,
) -> ScreenState:
    state = capture_screen_state(window_id, backend)
    if state.scene == SCENE_POSTGAME:
        return state
    _raise_for_unready_scene(state.scene)
    if state.scene != SCENE_GAME_OVER:
        raise RuntimeError(f"Expected game-over screen before advance_postgame; got {state.scene!r}")

    deadline = time.time() + timeout
    last_state = state
    while time.time() < deadline:
        swipe_to_postgame_summary(window_id, focus_delay)
        settle_deadline = time.time() + transition_delay
        while time.time() < settle_deadline:
            time.sleep(poll)
            last_state = capture_screen_state(window_id, backend)
            if last_state.scene == SCENE_POSTGAME:
                return last_state
        time.sleep(poll)
        last_state = capture_screen_state(window_id, backend)
        if last_state.scene == SCENE_POSTGAME:
            return last_state
        if last_state.scene == SCENE_GAME_OVER:
            continue
    raise RuntimeError(f"Timed out waiting for post-game summary; last scene was {last_state.scene!r}")


def open_menu_from_game(window_id: int, focus_delay: float) -> None:
    tap_window(window_id, INGAME_MENU_TAP[0], INGAME_MENU_TAP[1], focus_delay)


def tap_main_menu(window_id: int, focus_delay: float) -> None:
    tap_window(window_id, MENU_MAIN_MENU_TAP[0], MENU_MAIN_MENU_TAP[1], focus_delay)


def confirm_end_game(window_id: int, focus_delay: float) -> None:
    tap_window(window_id, END_CONFIRM_END_GAME_TAP[0], END_CONFIRM_END_GAME_TAP[1], focus_delay)


def start_game_sequence(
    window_id: int,
    backend: str,
    focus_delay: float,
    transition_delay: float,
    timeout: float,
    poll: float,
) -> ScreenState:
    state = capture_screen_state(window_id, backend)
    if state.scene == SCENE_GAME:
        return state
    _raise_for_unready_scene(state.scene)
    if state.scene != SCENE_TITLE:
        raise RuntimeError(f"Expected title screen before start_game; got {state.scene!r}")
    start_game_from_title(window_id, focus_delay)
    time.sleep(transition_delay)
    return wait_for_initial_game_state(window_id, backend, timeout=timeout, poll=poll)


def retry_game_sequence(
    window_id: int,
    backend: str,
    focus_delay: float,
    transition_delay: float,
    timeout: float,
    poll: float,
) -> ScreenState:
    state = capture_screen_state(window_id, backend)
    if state.scene == SCENE_GAME:
        return state
    _raise_for_unready_scene(state.scene)
    if state.scene == SCENE_GAME_OVER:
        state = advance_postgame_sequence(
            window_id,
            backend,
            focus_delay=focus_delay,
            transition_delay=transition_delay,
            timeout=timeout,
            poll=poll,
        )
    if state.scene != SCENE_POSTGAME:
        raise RuntimeError(f"Expected post-game screen before retry_game; got {state.scene!r}")
    retry_from_postgame(window_id, focus_delay)
    time.sleep(transition_delay)
    return wait_for_initial_game_state(window_id, backend, timeout=timeout, poll=poll)


def exit_current_game_sequence(
    window_id: int,
    backend: str,
    focus_delay: float,
    transition_delay: float,
    timeout: float,
    poll: float,
) -> ScreenState:
    state = capture_screen_state(window_id, backend)
    if state.scene == SCENE_TITLE:
        return state
    _raise_for_unready_scene(state.scene)
    if state.scene == SCENE_POSTGAME:
        raise RuntimeError("exit_current_game_sequence expects an active game/menu/confirm screen")
    if state.scene == SCENE_GAME:
        open_menu_from_game(window_id, focus_delay)
        time.sleep(transition_delay)
        state = wait_for_scene(window_id, backend, [SCENE_MENU], timeout=timeout, poll=poll)
    if state.scene == SCENE_MENU:
        tap_main_menu(window_id, focus_delay)
        time.sleep(transition_delay)
        state = wait_for_scene(window_id, backend, [SCENE_END_CONFIRM], timeout=timeout, poll=poll)
    if state.scene == SCENE_END_CONFIRM:
        confirm_end_game(window_id, focus_delay)
        time.sleep(transition_delay)
        state = wait_for_scene(window_id, backend, [SCENE_TITLE], timeout=timeout, poll=poll)
    if state.scene != SCENE_TITLE:
        raise RuntimeError(f"Could not exit to title; ended on scene {state.scene!r}")
    return state


def ensure_title_scene(
    window_id: int,
    backend: str,
    focus_delay: float,
    transition_delay: float,
    timeout: float,
    poll: float,
) -> ScreenState:
    state = capture_screen_state(window_id, backend)
    if state.scene == SCENE_TITLE:
        return state
    _raise_for_unready_scene(state.scene)
    if state.scene in (SCENE_GAME, SCENE_MENU, SCENE_END_CONFIRM):
        return exit_current_game_sequence(
            window_id,
            backend,
            focus_delay=focus_delay,
            transition_delay=transition_delay,
            timeout=timeout,
            poll=poll,
        )
    if state.scene in (SCENE_GAME_OVER, SCENE_POSTGAME):
        retry_game_sequence(
            window_id,
            backend,
            focus_delay=focus_delay,
            transition_delay=transition_delay,
            timeout=timeout,
            poll=poll,
        )
        return exit_current_game_sequence(
            window_id,
            backend,
            focus_delay=focus_delay,
            transition_delay=transition_delay,
            timeout=timeout,
            poll=poll,
        )
    raise RuntimeError(f"Cannot ensure title from scene {state.scene!r}")


def ensure_game_scene(
    window_id: int,
    backend: str,
    focus_delay: float,
    transition_delay: float,
    timeout: float,
    poll: float,
) -> ScreenState:
    state = capture_screen_state(window_id, backend)
    if state.scene == SCENE_GAME:
        return state
    _raise_for_unready_scene(state.scene)
    if state.scene == SCENE_TITLE:
        return start_game_sequence(
            window_id,
            backend,
            focus_delay=focus_delay,
            transition_delay=transition_delay,
            timeout=timeout,
            poll=poll,
        )
    if state.scene in (SCENE_GAME_OVER, SCENE_POSTGAME):
        return retry_game_sequence(
            window_id,
            backend,
            focus_delay=focus_delay,
            transition_delay=transition_delay,
            timeout=timeout,
            poll=poll,
        )
    if state.scene in (SCENE_MENU, SCENE_END_CONFIRM):
        ensure_title_scene(
            window_id,
            backend,
            focus_delay=focus_delay,
            transition_delay=transition_delay,
            timeout=timeout,
            poll=poll,
        )
        return start_game_sequence(
            window_id,
            backend,
            focus_delay=focus_delay,
            transition_delay=transition_delay,
            timeout=timeout,
            poll=poll,
        )
    raise RuntimeError(f"Cannot ensure game from scene {state.scene!r}")


def drag_window(
    window_id: int,
    direction: str,
    span_ratio: float,
    duration: float,
    steps: int,
    start_rel_x: float,
    start_rel_y: float,
    focus_delay: float,
) -> None:
    if direction not in DIRECTIONS:
        raise ValueError(f"Unsupported direction: {direction}")
    activate_mirroring()
    time.sleep(focus_delay)
    bounds = get_window_bounds(window_id)
    start = bounds.point(start_rel_x, start_rel_y)

    rel_dx = 0.0
    rel_dy = 0.0
    if direction == "left":
        rel_dx = -span_ratio
    elif direction == "right":
        rel_dx = span_ratio
    elif direction == "up":
        rel_dy = -span_ratio
    elif direction == "down":
        rel_dy = span_ratio

    end = bounds.point(start_rel_x + rel_dx, start_rel_y + rel_dy)

    _post_mouse(Quartz.kCGEventMouseMoved, start)
    time.sleep(0.01)
    _post_mouse(Quartz.kCGEventLeftMouseDown, start)
    drag_delay = max(duration / max(1, steps), 0.001)
    for idx in range(1, max(steps, 1) + 1):
        t = idx / max(steps, 1)
        point = (
            start[0] + (end[0] - start[0]) * t,
            start[1] + (end[1] - start[1]) * t,
        )
        _post_mouse(Quartz.kCGEventLeftMouseDragged, point)
        time.sleep(drag_delay)
    _post_mouse(Quartz.kCGEventLeftMouseUp, end)


class AutoPlayer:
    def __init__(
        self,
        window_id: int,
        backend: str,
        settle_frames: int,
        settle_poll: float,
        settle_threshold: float,
        settle_timeout: float,
        board_delta_threshold: float,
        swipe_span_ratio: float,
        swipe_duration: float,
        swipe_steps: int,
        start_rel_x: float,
        start_rel_y: float,
        focus_delay: float,
        recorder: Optional[ws.DatasetRecorder] = None,
    ) -> None:
        self.window_id = window_id
        self.backend = backend
        self.settle_frames = settle_frames
        self.settle_poll = settle_poll
        self.settle_threshold = settle_threshold
        self.settle_timeout = settle_timeout
        self.board_delta_threshold = board_delta_threshold
        self.swipe_span_ratio = swipe_span_ratio
        self.swipe_duration = swipe_duration
        self.swipe_steps = swipe_steps
        self.start_rel_x = start_rel_x
        self.start_rel_y = start_rel_y
        self.focus_delay = focus_delay
        self.recorder = recorder

        self.tile_cycle: Optional[ws.TileCycle] = None
        self.current_state: Optional[ScreenState] = None

    def _render_state(self, frame: FrameState) -> str:
        if self.tile_cycle is None:
            return ws.format_board_with_preview(frame.board, frame.preview_label)
        return ws.render_move_table(frame.board, frame.preview_label, self.tile_cycle)

    def initialize(self) -> FrameState:
        state = capture_screen_state(self.window_id, self.backend)
        if state.scene != SCENE_GAME or state.frame is None:
            raise RuntimeError(f"Expected in-game screen at initialize; got {state.scene!r}")
        frame = state.frame
        err = ws._initial_state_error(frame.board, frame.preview_label)
        if err is None:
            self.tile_cycle = ws.TileCycle()
            ws.seed_tile_cycle_from_initial_state(self.tile_cycle, frame.board, frame.preview_label)
            self.tile_cycle.set_max_tile(ws.board_max_tile(frame.board))
            ok, reason = ws.preview_possible(self.tile_cycle, frame.preview_label)
            if not ok:
                ws.print_error(f"preview '{frame.preview_label}' not possible at init: {reason}")
            self.tile_cycle.update(frame.preview_label)
        else:
            self.tile_cycle = None
            print(f"tracking disabled: {err}")
        self.current_state = state
        print(self._render_state(frame))
        print()
        return frame

    def _capture_after_settle(self) -> ScreenState:
        state = capture_screen_state(self.window_id, self.backend)
        if self.settle_frames <= 0:
            return state

        if state.scene != SCENE_GAME or state.frame is None:
            stable_scene = state.scene
            stable = 1
            last_state = state
            start = time.time()
            while time.time() - start < self.settle_timeout:
                time.sleep(self.settle_poll)
                current = capture_screen_state(self.window_id, self.backend)
                last_state = current
                if current.scene == stable_scene:
                    stable += 1
                else:
                    stable_scene = current.scene
                    stable = 1
                if stable >= self.settle_frames:
                    return current
            return last_state

        prev_sig = state.frame.board_sig
        prev_board = [row[:] for row in state.frame.board]
        prev_preview = state.frame.preview_label
        last_state = state
        stable = 0
        start = time.time()
        while time.time() - start < self.settle_timeout:
            time.sleep(self.settle_poll)
            current = capture_screen_state(self.window_id, self.backend)
            last_state = current
            if current.scene != SCENE_GAME or current.frame is None:
                return current
            diff = ws.board_signature_diff(prev_sig, current.frame.board_sig)
            same_board = current.frame.board == prev_board
            same_preview = current.frame.preview_label == prev_preview
            prev_sig = current.frame.board_sig
            prev_board = [row[:] for row in current.frame.board]
            prev_preview = current.frame.preview_label
            if diff < self.settle_threshold and same_board and same_preview:
                stable += 1
            else:
                stable = 0
            if stable >= self.settle_frames:
                return current
        return last_state

    def _record(self, frame: FrameState, ts_event: float) -> None:
        if not self.recorder:
            return
        self.recorder.record_capture(
            frame.arr,
            frame.board,
            frame.preview_label,
            frame.preview_debug,
            self.window_id,
            ts_event,
        )

    def attempt_move(self, direction: str) -> MoveResult:
        if self.current_state is None or self.current_state.scene != SCENE_GAME or self.current_state.frame is None:
            raise RuntimeError("AutoPlayer must be initialized before moves")

        before = self.current_state.frame
        drag_window(
            self.window_id,
            direction,
            span_ratio=self.swipe_span_ratio,
            duration=self.swipe_duration,
            steps=self.swipe_steps,
            start_rel_x=self.start_rel_x,
            start_rel_y=self.start_rel_y,
            focus_delay=self.focus_delay,
        )
        after_state = self._capture_after_settle()
        ts_event = time.time()
        ts = time.strftime("%H:%M:%S", time.localtime(ts_event))

        if after_state.scene in (SCENE_GAME_OVER, SCENE_POSTGAME):
            self.current_state = after_state
            label = after_state.scene.replace("_", " ")
            print(f"[{ts}] swipe {direction}: reached {label} screen")
            print()
            return MoveResult(
                direction=direction,
                changed=True,
                scene=after_state.scene,
                board_delta=None,
                game_over=True,
            )

        if after_state.scene != SCENE_GAME or after_state.frame is None:
            self.current_state = after_state
            ws.print_error(f"unexpected scene after swipe: {after_state.scene}")
            return MoveResult(
                direction=direction,
                changed=False,
                scene=after_state.scene,
                board_delta=None,
            )

        after = after_state.frame
        diff = ws.board_signature_diff(before.board_sig, after.board_sig)
        semantic_changed = (
            after.board != before.board
            or after.preview_label != before.preview_label
        )
        if not semantic_changed and diff <= self.board_delta_threshold:
            print(f"[{ts}] swipe {direction}: no board change (boardΔ={diff:.3f})")
            return MoveResult(
                direction=direction,
                changed=False,
                scene=SCENE_GAME,
                board_delta=diff,
            )

        self.current_state = after_state
        if ws._board_has_unknowns(after.board) or after.preview_label == "unknown":
            ws.print_error("state contains unknown cells or preview label")
        if self.tile_cycle is not None:
            self.tile_cycle.set_max_tile(ws.board_max_tile(after.board))
            ok, reason = ws.preview_possible(self.tile_cycle, after.preview_label)
            if not ok:
                ws.print_error(f"preview '{after.preview_label}' not possible: {reason}")
            self.tile_cycle.update(after.preview_label)
        self._record(after, ts_event)
        print(f"[{ts}] swipe {direction} (boardΔ={diff:.3f})")
        print(self._render_state(after))
        print()
        return MoveResult(
            direction=direction,
            changed=True,
            scene=SCENE_GAME,
            board_delta=diff,
        )

    def autoplay(
        self,
        max_moves: int,
        rng: random.Random,
        order_mode: str,
    ) -> int:
        moves_made = 0
        rotation = 0
        while moves_made < max_moves:
            directions = list(DIRECTIONS)
            if order_mode == "random":
                rng.shuffle(directions)
            elif order_mode == "cycle":
                directions = directions[rotation:] + directions[:rotation]
                rotation = (rotation + 1) % len(directions)

            moved = False
            for direction in directions:
                result = self.attempt_move(direction)
                if result.game_over:
                    if result.changed:
                        moves_made += 1
                    print("Detected post-game screen. Stopping autoplay.")
                    return moves_made
                if result.changed:
                    moves_made += 1
                    moved = True
                    break

            if not moved:
                print("No swipe direction changed the board. Stopping autoplay.")
                break
        return moves_made


def resolve_window(window_id: Optional[int], auto_window_prefix: str) -> Tuple[int, Dict[str, object]]:
    if window_id is not None:
        return window_id, {"id": window_id, "app": "(provided id)", "title": ""}

    auto_pick = ws.find_window_by_prefix(auto_window_prefix or "")
    if auto_pick:
        wid, app_name, win_name = auto_pick
        return wid, {"id": wid, "app": app_name, "title": win_name}

    wid, app_name, win_name = ws.choose_window_interactive()
    return wid, {"id": wid, "app": app_name, "title": win_name}


def print_scene_state(state: ScreenState) -> None:
    print(f"scene={state.scene} best={state.scene_score:.4f}")
    for scene, score in sorted(state.scene_scores.items(), key=lambda item: item[1]):
        print(f"  {scene}={score:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Drive the iPhone Mirroring window with swipes and optional autoplay."
    )
    parser.add_argument("--window-id", type=int, help="Window ID to target directly.")
    parser.add_argument(
        "--auto-window-prefix",
        default="iPhone Mirroring",
        help="Automatically select the first window whose title starts with this prefix.",
    )
    parser.add_argument(
        "--capture-backend",
        choices=("quartz", "screencapture"),
        default="quartz",
        help="Capture backend used for board/preview parsing.",
    )
    parser.add_argument(
        "--scene",
        action="store_true",
        help="Print the currently detected UI scene and exit unless more actions are requested.",
    )
    parser.add_argument(
        "--ensure-title",
        action="store_true",
        help="Drive the UI to the title screen from any known game/menu state.",
    )
    parser.add_argument(
        "--ensure-game",
        action="store_true",
        help="Drive the UI to an active game from title/post-game/menu states.",
    )
    parser.add_argument(
        "--exit-game",
        action="store_true",
        help="Abort the current in-progress game back to the title screen.",
    )
    parser.add_argument(
        "--start-game",
        action="store_true",
        help="Tap the title-screen PLAY THREES button before continuing.",
    )
    parser.add_argument(
        "--retry-game",
        action="store_true",
        help="Tap the post-game Retry button before continuing.",
    )
    parser.add_argument(
        "--tap-rel",
        nargs=2,
        type=float,
        metavar=("X", "Y"),
        help="Send one tap at relative window coordinates before exiting.",
    )
    parser.add_argument(
        "--swipe",
        choices=DIRECTIONS,
        help="Send one swipe gesture and print the resulting state.",
    )
    parser.add_argument(
        "--autoplay",
        action="store_true",
        help="Keep swiping until no move changes the board or --max-moves is reached.",
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=200,
        help="Maximum successful moves to make during autoplay.",
    )
    parser.add_argument(
        "--move-order",
        choices=("random", "cycle"),
        default="random",
        help="Direction ordering strategy during autoplay.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used when --move-order random is selected.",
    )
    parser.add_argument(
        "--board-delta-threshold",
        type=float,
        default=0.3,
        help="Minimum board signature delta required to treat a swipe as a real move.",
    )
    parser.add_argument(
        "--settle-frames",
        type=int,
        default=2,
        help="Number of stable board samples required after a swipe.",
    )
    parser.add_argument(
        "--settle-poll",
        type=float,
        default=0.1,
        help="Polling interval while waiting for the board to settle.",
    )
    parser.add_argument(
        "--settle-threshold",
        type=float,
        default=0.15,
        help="Board signature delta threshold used to count a frame as stable.",
    )
    parser.add_argument(
        "--settle-timeout",
        type=float,
        default=1.5,
        help="Maximum seconds to wait for the board to settle after a swipe.",
    )
    parser.add_argument(
        "--swipe-span-ratio",
        type=float,
        default=0.22,
        help="Swipe distance as a fraction of the window width/height.",
    )
    parser.add_argument(
        "--swipe-duration",
        type=float,
        default=0.12,
        help="Total time to spend dragging during a swipe gesture.",
    )
    parser.add_argument(
        "--swipe-steps",
        type=int,
        default=12,
        help="Number of drag interpolation steps per swipe.",
    )
    parser.add_argument(
        "--start-rel-x",
        type=float,
        default=0.5,
        help="Relative x position inside the mirrored window where swipes begin.",
    )
    parser.add_argument(
        "--start-rel-y",
        type=float,
        default=0.55,
        help="Relative y position inside the mirrored window where swipes begin.",
    )
    parser.add_argument(
        "--focus-delay",
        type=float,
        default=0.2,
        help="Delay after activating iPhone Mirroring before sending the gesture.",
    )
    parser.add_argument(
        "--transition-delay",
        type=float,
        default=0.8,
        help="Delay after start/retry taps before capturing the new state.",
    )
    parser.add_argument(
        "--record-dataset",
        type=str,
        help="Optional dataset directory for saving successful move captures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    window_id, window_info = resolve_window(args.window_id, args.auto_window_prefix)
    print(
        f"Using window {window_id} ({window_info.get('app', '')} - {window_info.get('title', '')})"
    )

    recorder = None
    if args.record_dataset:
        recorder = ws.DatasetRecorder(Path(args.record_dataset), window_info=window_info)
        print(f"Recording successful moves to {recorder.session_dir}")

    player = AutoPlayer(
        window_id=window_id,
        backend=args.capture_backend,
        settle_frames=args.settle_frames,
        settle_poll=args.settle_poll,
        settle_threshold=args.settle_threshold,
        settle_timeout=args.settle_timeout,
        board_delta_threshold=args.board_delta_threshold,
        swipe_span_ratio=args.swipe_span_ratio,
        swipe_duration=args.swipe_duration,
        swipe_steps=args.swipe_steps,
        start_rel_x=args.start_rel_x,
        start_rel_y=args.start_rel_y,
        focus_delay=args.focus_delay,
        recorder=recorder,
    )

    if args.tap_rel:
        tap_window(window_id, args.tap_rel[0], args.tap_rel[1], args.focus_delay)
        return

    scene_timeout = max(3.0, args.transition_delay * 4, args.settle_timeout)
    state = capture_screen_state(window_id, args.capture_backend)
    if args.scene:
        print_scene_state(state)
        if not any(
            (
                args.ensure_title,
                args.ensure_game,
                args.exit_game,
                args.start_game,
                args.retry_game,
                args.swipe,
                args.autoplay,
            )
        ):
            return

    if args.exit_game or args.ensure_title:
        state = ensure_title_scene(
            window_id,
            args.capture_backend,
            focus_delay=args.focus_delay,
            transition_delay=args.transition_delay,
            timeout=scene_timeout,
            poll=args.settle_poll,
        )
        if not any((args.start_game, args.retry_game, args.ensure_game, args.swipe, args.autoplay)):
            print_scene_state(state)
            return

    if args.start_game:
        state = start_game_sequence(
            window_id,
            args.capture_backend,
            focus_delay=args.focus_delay,
            transition_delay=args.transition_delay,
            timeout=scene_timeout,
            poll=args.settle_poll,
        )

    if args.retry_game:
        state = retry_game_sequence(
            window_id,
            args.capture_backend,
            focus_delay=args.focus_delay,
            transition_delay=args.transition_delay,
            timeout=scene_timeout,
            poll=args.settle_poll,
        )

    if args.ensure_game:
        state = ensure_game_scene(
            window_id,
            args.capture_backend,
            focus_delay=args.focus_delay,
            transition_delay=args.transition_delay,
            timeout=scene_timeout,
            poll=args.settle_poll,
        )

    if not any((args.swipe, args.autoplay, args.start_game, args.retry_game, args.ensure_game)):
        return

    player.initialize()
    ws.print_legend()

    if args.swipe:
        player.attempt_move(args.swipe)
        return

    if args.autoplay:
        rng = random.Random(args.seed)
        moves = player.autoplay(args.max_moves, rng=rng, order_mode=args.move_order)
        print(f"Autoplay finished after {moves} successful moves.")


if __name__ == "__main__":
    main()
