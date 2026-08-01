"""Hold last ball position across brief YOLO misses (wide / distant footage)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from polyfut_video.pipeline.geometry import box_center, dist


@dataclass
class BallSmoothConfig:
    max_hold_frames: int = 8
    max_jump_px: float = 500.0
    # Confidence-aware teleport gate. A real ball barely moves between adjacent
    # samples (real-footage measurement at 640x360: median 5.6px, p75 20px), so a
    # detection that leaps a long way is usually a false positive — a white sock,
    # line marking or shirt picked up elsewhere on the pitch. Those fabricate
    # huge velocity swings, which Stage 4 reads as a "kick / direction change"
    # and turns into a phantom touch at a spot the ball never occupied.
    # Measured: 75% of >120px jumps landed on a sub-0.30-confidence detection.
    suspect_jump_px: float = 120.0
    suspect_jump_conf: float = 0.30
    # Two-frame confirmation for far jumps. Confidence alone is weak evidence:
    # the remaining 25% of far jumps were confident false positives, and a
    # single frame can't distinguish "the ball was cleared" from "a sock lit up
    # across the pitch". So a jump beyond ``suspect_jump_px`` is parked as
    # *pending* and only committed when the NEXT detection agrees with it —
    # either landing near the same place, or continuing along the same heading
    # at a comparable speed (which is what a genuinely fast ball does). A lone
    # flicker never gets that agreement; a real clearance costs one frame of
    # latency and is then followed normally.
    require_jump_confirmation: bool = True
    confirm_dist_px: float = 120.0     # "landed near where the jump pointed"
    confirm_angle_deg: float = 45.0    # "kept going the same way"
    confirm_speed_ratio: float = 1.6   # and not implausibly faster than the jump


class BallSmoother:
    def __init__(self, cfg: BallSmoothConfig | None = None):
        self.cfg = cfg or BallSmoothConfig()
        self._last_xyxy: np.ndarray | None = None
        self._last_conf: float = 0.0
        self._held: int = 0
        # An unconfirmed far jump, kept only until the next detection either
        # corroborates it or contradicts it.
        self._pending: np.ndarray | None = None

    def update(
        self,
        ball_xyxy: np.ndarray | list[float] | None,
        ball_conf: float,
    ) -> tuple[list[float] | None, float, bool]:
        """Return (bbox, conf, is_interpolated)."""
        xyxy = np.asarray(ball_xyxy, dtype=np.float32) if ball_xyxy is not None else None

        if xyxy is not None:
            if self._last_xyxy is None:
                return self._accept(xyxy, ball_conf)

            d = dist(box_center(xyxy), box_center(self._last_xyxy))

            # Spatial continuity — the overwhelmingly common case. Trust it
            # immediately regardless of confidence: the ball is often detected
            # weakly while being tracked correctly.
            if d <= self.cfg.suspect_jump_px:
                self._pending = None
                return self._accept(xyxy, ball_conf)

            # Beyond anything physical: never commit, whatever the confidence.
            # The hold budget still expires, which is how a genuinely lost ball
            # gets re-acquired (see below).
            if d > self.cfg.max_jump_px:
                self._pending = None
                return self._hold()

            if not self.cfg.require_jump_confirmation:
                # Legacy single-frame gate: reject only far AND low-confidence.
                if ball_conf >= self.cfg.suspect_jump_conf:
                    return self._accept(xyxy, ball_conf)
                return self._hold()

            # NOTE: the jump gate deliberately applies even when ``_held == 0``
            # (i.e. straight after a good detection). It used to be skipped in
            # that state, which is exactly when a false positive does the most
            # damage: the tracker would accept a teleport off a cleanly-tracked
            # ball and corrupt the velocities on both sides of it.
            if self._pending is not None and self._confirms(xyxy):
                self._pending = None
                return self._accept(xyxy, ball_conf)
            self._pending = xyxy
            return self._hold()

        if self._last_xyxy is not None:
            return self._hold()

        self._reset()
        return None, 0.0, False

    def _confirms(self, xyxy: np.ndarray) -> bool:
        """True when this detection corroborates the pending far jump."""
        if self._pending is None or self._last_xyxy is None:
            return False
        last_c = box_center(self._last_xyxy)
        pend_c = box_center(self._pending)
        new_c = box_center(xyxy)

        # The ball really is over there (it stayed put between the two frames).
        if dist(new_c, pend_c) <= self.cfg.confirm_dist_px:
            return True

        # Or it is genuinely travelling: same heading, comparable speed. This is
        # what keeps a fast clearance trackable — consecutive far detections
        # that march in one direction confirm each other, while random flicker
        # (which changes direction every frame) never does.
        v1 = (float(pend_c[0] - last_c[0]), float(pend_c[1] - last_c[1]))
        v2 = (float(new_c[0] - pend_c[0]), float(new_c[1] - pend_c[1]))
        n1 = math.hypot(*v1)
        n2 = math.hypot(*v2)
        if n1 <= 0.0 or n2 <= 0.0:
            return False
        cos = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
        angle = math.degrees(math.acos(max(-1.0, min(1.0, cos))))
        return angle <= self.cfg.confirm_angle_deg and n2 <= self.cfg.confirm_speed_ratio * n1

    def _accept(self, xyxy: np.ndarray, conf: float) -> tuple[list[float], float, bool]:
        self._last_xyxy = xyxy
        self._last_conf = conf
        self._held = 0
        return [float(x) for x in xyxy], conf, False

    def _hold(self) -> tuple[list[float] | None, float, bool]:
        """Carry the last trusted position, or release once the budget runs out."""
        self._held += 1
        if self._last_xyxy is not None and self._held <= self.cfg.max_hold_frames:
            return [float(x) for x in self._last_xyxy], self._last_conf * 0.6, True
        # Held too long — the ball really is gone; allow re-acquisition.
        self._reset()
        return None, 0.0, False

    def _reset(self) -> None:
        self._last_xyxy = None
        self._last_conf = 0.0
        self._held = 0
        self._pending = None

    def reset(self) -> None:
        self._reset()
