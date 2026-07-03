"""Hard camera-cut detection via colour-histogram difference between frames.

A large, sudden histogram change means the camera angle jumped (common in
amateur multi-angle footage). Possession scoring should ignore cut frames so a
ball that "teleports" across a cut doesn't create a false possession event.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CutConfig:
    # Correlation below this between consecutive histograms => treat as a cut.
    corr_thresh: float = 0.55
    downscale_w: int = 160


def _hist(frame_bgr: np.ndarray, w: int) -> np.ndarray:
    h = max(1, int(frame_bgr.shape[0] * w / max(1, frame_bgr.shape[1])))
    small = cv2.resize(frame_bgr, (w, h), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


class CutDetector:
    """Stateful: feed frames in order; reports True when a cut is detected."""

    def __init__(self, cfg: CutConfig | None = None):
        self.cfg = cfg or CutConfig()
        self._prev: np.ndarray | None = None

    def is_cut(self, frame_bgr: np.ndarray) -> bool:
        hist = _hist(frame_bgr, self.cfg.downscale_w)
        if self._prev is None:
            self._prev = hist
            return False
        corr = cv2.compareHist(self._prev, hist, cv2.HISTCMP_CORREL)
        self._prev = hist
        return corr < self.cfg.corr_thresh
