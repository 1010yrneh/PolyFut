"""Stage 3: flag long low-motion stretches as deadtime (halftime, breaks).

Cheap motion heuristic — not SoccerNet action spotting (Level 2+ upgrade path).
"""

from __future__ import annotations

from typing import Iterator

import cv2
import numpy as np

from polyfut_video.config import PipelineConfig
from polyfut_video.pipeline.decode import iter_frames


def _shot_motion_mean(
    video_path: str,
    start_sec: float,
    end_sec: float,
    target_width: int,
) -> float:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0.0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, start_sec * 1000.0))
    prev = None
    diffs: list[float] = []
    t = start_sec
    while t <= end_sec:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        if w > target_width:
            scale = target_width / w
            frame = cv2.resize(frame, (target_width, max(1, int(h * scale))))
        if prev is not None:
            ga = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
            gb = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diffs.append(float(np.mean(cv2.absdiff(ga, gb))))
        prev = frame
        t += 1.0 / fps if fps > 0 else 0.04
    cap.release()
    return float(np.mean(diffs)) if diffs else 0.0


def filter_deadtime(
    shots: list[dict],
    frame_iter: Iterator[tuple[int, float, np.ndarray]] | None,
    motion_threshold: float,
    min_duration_sec: float = 60.0,
    *,
    video_path: str | None = None,
    target_width: int = 640,
) -> tuple[list[dict], list[dict]]:
    """
    Returns (live_play_shots, removed_segments).

    removed_segments catalog entries removed in Stages 2–3 for match-clock remap:
    {"start_sec", "end_sec", "reason": "discard"|"deadtime"}
    """
    removed: list[dict] = []
    live: list[dict] = []

    # Optional fast motion from provided iterator (grouped by shot)
    motion_by_shot: dict[int, float] = {}
    if frame_iter is not None:
        shot_idx = 0
        buf: list[np.ndarray] = []
        prev = None
        for _fi, t_sec, frame in frame_iter:
            while shot_idx < len(shots) and t_sec > shots[shot_idx]["end_sec"]:
                if buf:
                    motion_by_shot[shot_idx] = _frames_motion(buf)
                buf = []
                prev = None
                shot_idx += 1
            if shot_idx >= len(shots):
                break
            if shots[shot_idx]["start_sec"] <= t_sec <= shots[shot_idx]["end_sec"]:
                if prev is not None:
                    ga = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
                    gb = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    buf.append(float(np.mean(cv2.absdiff(ga, gb))))
                prev = frame
        if buf and shot_idx < len(shots):
            motion_by_shot[shot_idx] = float(np.mean(buf)) if buf else 0.0

    for i, shot in enumerate(shots):
        dur = max(0.0, shot["end_sec"] - shot["start_sec"])
        if shot.get("label") == "discard":
            removed.append({
                "start_sec": shot["start_sec"],
                "end_sec": shot["end_sec"],
                "reason": "discard",
            })
            continue

        motion = motion_by_shot.get(i)
        if motion is None and video_path:
            motion = _shot_motion_mean(
                video_path, shot["start_sec"], shot["end_sec"], target_width,
            )
        motion = motion or 0.0

        if dur >= min_duration_sec and motion < motion_threshold:
            removed.append({
                "start_sec": shot["start_sec"],
                "end_sec": shot["end_sec"],
                "reason": "deadtime",
            })
            continue

        live.append({**shot, "label": "live_play"})

    return live, removed


def _frames_motion(diffs: list) -> float:
    if not diffs:
        return 0.0
    return float(np.mean(diffs))
