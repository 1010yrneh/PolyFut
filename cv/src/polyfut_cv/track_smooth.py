"""Lightweight ball smoothing across frames (NOT player identity tracking).

YOLO ball detections flicker badly at small sizes. We keep a single "active ball"
estimate: when a detection is missing for a few analysed frames we hold the last
known position briefly (capped) so possession does not blink off during a one- or
two-frame miss. This is intentionally simpler than ByteTrack and needs no extra
dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from polyfut_cv.geometry import box_center, dist


@dataclass
class BallSmoothConfig:
    # Max number of consecutive analysed frames to hold a missing ball.
    # At ~2 analysed fps this bridges ~4 s of missed ball detections, which is
    # essential on wide footage where the small ball flickers in and out.
    max_hold_frames: int = 10
    # Reject a new detection that jumps further than this (px) from the held
    # position within the hold window (likely a different object).
    max_jump_px: float = 600.0


class BallSmoother:
    def __init__(self, cfg: BallSmoothConfig | None = None):
        self.cfg = cfg or BallSmoothConfig()
        self._last_xyxy: np.ndarray | None = None
        self._last_conf: float = 0.0
        self._held: int = 0

    def update(self, ball_xyxy: np.ndarray | None, ball_conf: float) -> tuple[np.ndarray | None, float, bool]:
        """Return (xyxy, conf, is_interpolated).

        is_interpolated=True means we are reusing a held position (no fresh detection).
        """
        if ball_xyxy is not None:
            if (
                self._last_xyxy is not None
                and self._held > 0
                and dist(box_center(ball_xyxy), box_center(self._last_xyxy)) > self.cfg.max_jump_px
            ):
                # Implausible jump while holding: keep holding the previous one.
                self._held += 1
                if self._held <= self.cfg.max_hold_frames:
                    return self._last_xyxy, self._last_conf * 0.6, True
                self._reset()
                return None, 0.0, False
            self._last_xyxy = ball_xyxy
            self._last_conf = ball_conf
            self._held = 0
            return ball_xyxy, ball_conf, False

        # No detection this frame.
        if self._last_xyxy is not None and self._held < self.cfg.max_hold_frames:
            self._held += 1
            return self._last_xyxy, self._last_conf * 0.6, True

        self._reset()
        return None, 0.0, False

    def _reset(self) -> None:
        self._last_xyxy = None
        self._last_conf = 0.0
        self._held = 0

    def reset(self) -> None:
        """Call on a hard cut so a held ball does not bleed across angles."""
        self._reset()
