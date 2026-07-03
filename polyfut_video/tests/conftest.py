"""Shared test helpers — synthetic broadcast-style clips."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def make_synthetic_clip(
    path: Path,
    *,
    duration_sec: float = 5.0,
    fps: int = 25,
    width: int = 640,
    height: int = 360,
    include_replay: bool = True,
) -> Path:
    """
    Write a short MP4 with green pitch + moving blobs (players/ball).
    Optional replay segment (non-green) for shot-filter tests.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    n_frames = int(duration_sec * fps)
    replay_start = int(n_frames * 0.6) if include_replay else n_frames + 1

    for i in range(n_frames):
        if i >= replay_start:
            # Crowd/replay shot — brown, low green
            frame = np.full((height, width, 3), (40, 60, 120), dtype=np.uint8)
        else:
            frame = np.full((height, width, 3), (35, 140, 35), dtype=np.uint8)
            # Players
            for j, ox in enumerate([120, 280, 400]):
                x = int(ox + 30 * np.sin(i / 10.0 + j))
                y = 200 + j * 5
                cv2.rectangle(frame, (x, y), (x + 30, y + 70), (0, 0, 220) if j % 2 == 0 else (220, 220, 220), -1)
            # Ball
            bx = int(250 + 50 * np.sin(i / 8.0))
            cv2.circle(frame, (bx, 250), 6, (0, 0, 0), -1)
        writer.write(frame)
    writer.release()
    return path
