"""Stage 4: adaptive motion-based full vs cheap frame routing."""

from __future__ import annotations

from typing import Iterator

import cv2
import numpy as np


def route_frames(
    frame_iter: Iterator[tuple[int, float, np.ndarray]],
    motion_threshold: float,
    *,
    downscale: int = 4,
) -> Iterator[tuple[int, np.ndarray, str, float]]:
    """
    Yields (frame_index, frame, route, timestamp_sec).
    route is "full" or "cheap".

    Motion is computed on a downscaled grayscale pass for speed; routing
    decisions are identical in spirit but ~16× cheaper than full-res diff.
    """
    step = max(1, downscale)
    prev_small: np.ndarray | None = None

    for frame_idx, t_sec, frame in frame_iter:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = gray[::step, ::step]
        if prev_small is None:
            motion = motion_threshold + 1.0
        else:
            if small.shape != prev_small.shape:
                prev_small = cv2.resize(prev_small, (small.shape[1], small.shape[0]))
            motion = float(np.mean(cv2.absdiff(small, prev_small)))
        prev_small = small

        if motion >= motion_threshold:
            yield frame_idx, frame, "full", t_sec
        else:
            yield frame_idx, frame, "cheap", t_sec
