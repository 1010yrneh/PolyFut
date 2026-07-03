"""Stage 1: optimized frame reading with grab/retrieve and decode-time resize."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


def _resize_frame(frame: np.ndarray, target_width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= target_width:
        return frame
    scale = target_width / float(w)
    nh = max(1, int(round(h * scale)))
    return cv2.resize(frame, (target_width, nh), interpolation=cv2.INTER_AREA)


def iter_frames(
    video_path: str,
    target_width: int = 640,
    sample_every_n: int = 1,
) -> Iterator[tuple[int, float, np.ndarray]]:
    """
    Yields (frame_index, timestamp_sec, frame) tuples.

    Uses grab() to advance the stream and only calls retrieve() on frames
    that will actually be processed (every Nth frame per sample_every_n).
    Downsamples to target_width, preserving aspect ratio, before returning.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    frame_idx = 0
    yielded = 0

    try:
        while True:
            if not cap.grab():
                break
            if frame_idx % max(1, sample_every_n) != 0:
                frame_idx += 1
                continue
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                frame_idx += 1
                continue
            t_sec = frame_idx / fps if fps > 0 else 0.0
            yield frame_idx, t_sec, _resize_frame(frame, target_width)
            yielded += 1
            frame_idx += 1
    finally:
        cap.release()


def probe_video(video_path: str) -> dict:
    """Return fps, frame_count, duration_sec, width, height via OpenCV."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    dur = n / fps if fps > 0 else 0.0
    return {"fps": fps, "frame_count": n, "duration_sec": dur, "width": w, "height": h}


def ffprobe_frame_count(video_path: str) -> int | None:
    """Optional cross-check against ffprobe when available."""
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-count_packets", "-show_entries", "stream=nb_read_packets",
                "-of", "csv=p=0", str(video_path),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        return int(out.stdout.strip())
    except Exception:
        return None


def benchmark_decode(video_path: str, target_width: int = 640, max_frames: int = 300) -> dict:
    """Compare grab/retrieve iterator vs naive read() loop."""
    # Naive
    t0 = time.perf_counter()
    cap = cv2.VideoCapture(str(video_path))
    n_naive = 0
    while n_naive < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        _ = _resize_frame(frame, target_width)
        n_naive += 1
    cap.release()
    naive_sec = time.perf_counter() - t0

    # Optimized
    t1 = time.perf_counter()
    n_opt = 0
    for _fi, _t, _fr in iter_frames(video_path, target_width=target_width):
        n_opt += 1
        if n_opt >= max_frames:
            break
    opt_sec = time.perf_counter() - t1

    return {
        "frames": min(n_naive, n_opt),
        "naive_sec": round(naive_sec, 4),
        "optimized_sec": round(opt_sec, 4),
        "speedup": round(naive_sec / max(opt_sec, 1e-6), 3),
    }
