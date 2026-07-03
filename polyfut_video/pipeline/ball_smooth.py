"""Hold last ball position across brief YOLO misses (wide / distant footage)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from polyfut_video.pipeline.geometry import box_center, dist


@dataclass
class BallSmoothConfig:
    max_hold_frames: int = 8
    max_jump_px: float = 500.0


class BallSmoother:
    def __init__(self, cfg: BallSmoothConfig | None = None):
        self.cfg = cfg or BallSmoothConfig()
        self._last_xyxy: np.ndarray | None = None
        self._last_conf: float = 0.0
        self._held: int = 0

    def update(
        self,
        ball_xyxy: np.ndarray | list[float] | None,
        ball_conf: float,
    ) -> tuple[list[float] | None, float, bool]:
        """Return (bbox, conf, is_interpolated)."""
        xyxy = np.asarray(ball_xyxy, dtype=np.float32) if ball_xyxy is not None else None

        if xyxy is not None:
            if (
                self._last_xyxy is not None
                and self._held > 0
                and dist(box_center(xyxy), box_center(self._last_xyxy)) > self.cfg.max_jump_px
            ):
                self._held += 1
                if self._held <= self.cfg.max_hold_frames:
                    return [float(x) for x in self._last_xyxy], self._last_conf * 0.6, True
                self._reset()
                return None, 0.0, False
            self._last_xyxy = xyxy
            self._last_conf = ball_conf
            self._held = 0
            return [float(x) for x in xyxy], ball_conf, False

        if self._last_xyxy is not None and self._held < self.cfg.max_hold_frames:
            self._held += 1
            return [float(x) for x in self._last_xyxy], self._last_conf * 0.6, True

        self._reset()
        return None, 0.0, False

    def _reset(self) -> None:
        self._last_xyxy = None
        self._last_conf = 0.0
        self._held = 0

    def reset(self) -> None:
        self._reset()
