"""Optional green-pitch HSV mask to discard out-of-bounds detections.

Players/ball detected over the stands, benches, or sidelines are usually false
positives for "on-pitch possession". We estimate the grass colour from the centre
of an early frame, build a mask, and reject detections whose foot point is not on
grass. Optional because shadows / artificial turf / brown patches can break it.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PitchMaskConfig:
    h_tol: float = 25.0
    s_min: float = 40.0
    v_min: float = 40.0
    # Fraction of a small neighbourhood that must be grass for a point to count.
    point_grass_frac: float = 0.25
    sample_box_frac: float = 0.25  # central region used to estimate grass hue


class PitchMask:
    def __init__(self, frame_bgr: np.ndarray, cfg: PitchMaskConfig | None = None):
        self.cfg = cfg or PitchMaskConfig()
        self._grass_h = self._estimate_grass_hue(frame_bgr)
        self.mask = self._build_mask(frame_bgr)

    def _estimate_grass_hue(self, frame_bgr: np.ndarray) -> float:
        h, w = frame_bgr.shape[:2]
        f = self.cfg.sample_box_frac
        x1, x2 = int(w * (0.5 - f / 2)), int(w * (0.5 + f / 2))
        y1, y2 = int(h * (0.5 - f / 2)), int(h * (0.5 + f / 2))
        hsv = cv2.cvtColor(frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
        # Median hue of reasonably saturated/bright pixels.
        sel = hsv[(hsv[:, :, 1] > self.cfg.s_min) & (hsv[:, :, 2] > self.cfg.v_min)]
        if sel.size == 0:
            return 60.0  # default greenish hue
        return float(np.median(sel[:, 0]))

    def _build_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        h = hsv[:, :, 0].astype(np.int16)
        dh = np.abs(h - int(self._grass_h))
        dh = np.minimum(dh, 180 - dh)
        mask = (
            (dh <= self.cfg.h_tol)
            & (hsv[:, :, 1] >= self.cfg.s_min)
            & (hsv[:, :, 2] >= self.cfg.v_min)
        ).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        return mask

    def point_on_pitch(self, x: float, y: float, rad: int = 6) -> bool:
        h, w = self.mask.shape[:2]
        xi, yi = int(round(x)), int(round(y))
        if xi < 0 or yi < 0 or xi >= w or yi >= h:
            return False
        x1, x2 = max(0, xi - rad), min(w, xi + rad)
        y1, y2 = max(0, yi - rad), min(h, yi + rad)
        region = self.mask[y1:y2, x1:x2]
        if region.size == 0:
            return False
        return float((region > 0).mean()) >= self.cfg.point_grass_frac
