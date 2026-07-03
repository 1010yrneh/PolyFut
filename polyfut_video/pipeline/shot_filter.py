"""Stage 2: heuristic shot segmentation and main_camera vs discard classification.

Level 1 deliberately uses cheap heuristics, not SoccerNet-trained classifiers.
Level 2+ can swap in a heavier semantic shot classifier if recall is insufficient.
"""

from __future__ import annotations

from typing import Iterator

import cv2
import numpy as np

from polyfut_video.config import PipelineConfig


def _frame_diff(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    return float(np.mean(cv2.absdiff(ga, gb)) / 255.0)


def _hist_diff(a: np.ndarray, b: np.ndarray) -> float:
    ha = cv2.calcHist([a], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hb = cv2.calcHist([b], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(ha, ha)
    cv2.normalize(hb, hb)
    return float(cv2.compareHist(ha, hb, cv2.HISTCMP_BHATTACHARYYA))


def _green_ratio(frame: np.ndarray) -> float:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Broad grass band
    mask = cv2.inRange(hsv, (25, 25, 25), (95, 255, 255))
    return float(np.count_nonzero(mask)) / max(mask.size, 1)


def _motion_magnitude(prev: np.ndarray | None, cur: np.ndarray) -> float:
    if prev is None:
        return 0.0
    return _frame_diff(prev, cur) * 100.0


def _graphic_overlay_score(frame: np.ndarray) -> float:
    """Large uniform-color blocks often indicate scoreboard graphics."""
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Top/bottom bands
    bands = [gray[: h // 8, :], gray[-h // 8 :, :]]
    scores = []
    for band in bands:
        std = float(np.std(band))
        if std < 8.0:
            scores.append(1.0 - std / 8.0)
    return max(scores) if scores else 0.0


PIPELINE_VERSION = "1.2.0"


def _analysis_frame(frame: np.ndarray, analysis_width: int = 160) -> np.ndarray:
    """Small copy for shot heuristics — keeps only one lightweight frame in memory."""
    h, w = frame.shape[:2]
    if w <= analysis_width:
        return frame
    scale = analysis_width / float(w)
    return cv2.resize(
        frame,
        (analysis_width, max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def _new_shot_stats() -> dict[str, float]:
    return {"green_sum": 0.0, "graphic_sum": 0.0, "motion_sum": 0.0, "n": 0.0}


def _accumulate_shot_stats(stats: dict[str, float], frame: np.ndarray, motion: float) -> None:
    stats["green_sum"] += _green_ratio(frame)
    stats["graphic_sum"] += _graphic_overlay_score(frame)
    stats["motion_sum"] += motion
    stats["n"] += 1.0


def _label_from_stats(stats: dict[str, float], cfg: PipelineConfig) -> str:
    if stats["n"] <= 0:
        return "discard"
    n = stats["n"]
    green = stats["green_sum"] / n
    motion = stats["motion_sum"] / n
    graphic = stats["graphic_sum"] / n

    if green < cfg.green_ratio_min:
        return "discard"
    if motion < cfg.motion_smooth_max and green < cfg.green_ratio_min + 0.08:
        return "discard"
    if graphic > cfg.graphic_uniform_ratio:
        return "discard"
    return "main_camera"


def _classify_shot(frames: list[np.ndarray], motions: list[float], cfg: PipelineConfig) -> str:
    if not frames:
        return "discard"
    stats = _new_shot_stats()
    for frame, motion in zip(frames, motions):
        _accumulate_shot_stats(stats, frame, motion)
    return _label_from_stats(stats, cfg)


def segment_and_classify_shots(
    frame_iter: Iterator[tuple[int, float, np.ndarray]],
    cfg: PipelineConfig | None = None,
) -> list[dict]:
    """
    Returns shot records:
    {"start_frame", "end_frame", "start_sec", "end_sec", "label": "main_camera"|"discard"}

    Uses running per-shot statistics only (no frame buffering) so full-match
    single-camera broadcasts do not exhaust RAM.
    """
    cfg = cfg or PipelineConfig()
    shots: list[dict] = []
    prev: np.ndarray | None = None
    cur_stats = _new_shot_stats()
    cur_start_frame = 0
    cur_start_sec = 0.0
    last_frame_idx = 0
    last_sec = 0.0

    def flush_shot() -> None:
        nonlocal cur_stats, cur_start_frame, cur_start_sec
        if cur_stats["n"] <= 0:
            return
        label = _label_from_stats(cur_stats, cfg)
        shots.append({
            "start_frame": cur_start_frame,
            "end_frame": last_frame_idx,
            "start_sec": cur_start_sec,
            "end_sec": last_sec,
            "label": label,
        })
        cur_stats = _new_shot_stats()

    for frame_idx, t_sec, frame in frame_iter:
        thumb = _analysis_frame(frame)
        hist_d = _hist_diff(prev, thumb) if prev is not None else 0.0
        mot = _motion_magnitude(prev, thumb)
        if prev is not None and hist_d > cfg.cut_hist_threshold:
            flush_shot()
            cur_start_frame = frame_idx
            cur_start_sec = t_sec
        if cur_stats["n"] <= 0:
            cur_start_frame = frame_idx
            cur_start_sec = t_sec
        _accumulate_shot_stats(cur_stats, thumb, mot)
        last_frame_idx = frame_idx
        last_sec = t_sec
        prev = thumb

    flush_shot()
    return shots
