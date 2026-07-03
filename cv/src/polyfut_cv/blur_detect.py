"""Motion-blur gate via variance of the Laplacian.

Low Laplacian variance == blurry frame. Detections on heavily blurred frames are
unreliable (especially the tiny ball), so we skip them for possession scoring.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class BlurConfig:
    # Frames with Laplacian variance below this are considered too blurry.
    # Tune per camera in Session 5; 60 is a reasonable 640x360 default.
    var_thresh: float = 60.0
    downscale_w: int = 320


def laplacian_variance(frame_bgr: np.ndarray, w: int = 320) -> float:
    h = max(1, int(frame_bgr.shape[0] * w / max(1, frame_bgr.shape[1])))
    small = cv2.resize(frame_bgr, (w, h), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def is_blurry(frame_bgr: np.ndarray, cfg: BlurConfig | None = None) -> bool:
    cfg = cfg or BlurConfig()
    return laplacian_variance(frame_bgr, cfg.downscale_w) < cfg.var_thresh
