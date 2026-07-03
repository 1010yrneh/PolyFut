"""Binary team-colour classification: 'my team' vs 'not', by torso HSV.

We only model OUR colour. "Is this player mine - yes/no" is far more robust at
low resolution than trying to classify both teams. Sampling the median of a torso
region (not one pixel) avoids grabbing background grass on tiny boxes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np


@dataclass
class ColorRef:
    h: float  # OpenCV hue 0-179
    s: float  # 0-255
    v: float  # 0-255

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @staticmethod
    def from_json(path: str | Path) -> "ColorRef":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return ColorRef(h=float(d["h"]), s=float(d["s"]), v=float(d["v"]))


@dataclass
class ColorConfig:
    # Hue is circular (0-179 in OpenCV). Tolerances are generous by default
    # because amateur footage lighting varies a lot.
    h_tol: float = 18.0
    s_tol: float = 90.0
    v_tol: float = 110.0
    # Treat the reference as "dark/black" when value or saturation is low;
    # then match mostly on darkness rather than hue (the hard case).
    dark_v_thresh: float = 70.0
    dark_s_thresh: float = 60.0


def torso_crop(frame_bgr: np.ndarray, xyxy: np.ndarray) -> np.ndarray | None:
    """Center torso region of a person box: middle 50% width, upper-mid height."""
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    bw = x2 - x1
    bh = y2 - y1
    if bw < 2 or bh < 4:
        return None
    cx1 = int(round(x1 + 0.25 * bw))
    cx2 = int(round(x1 + 0.75 * bw))
    cy1 = int(round(y1 + 0.20 * bh))
    cy2 = int(round(y1 + 0.55 * bh))
    cx1, cx2 = max(0, cx1), min(w, cx2)
    cy1, cy2 = max(0, cy1), min(h, cy2)
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    return frame_bgr[cy1:cy2, cx1:cx2]


def median_hsv(frame_bgr: np.ndarray, xyxy: np.ndarray) -> tuple[float, float, float] | None:
    crop = torso_crop(frame_bgr, xyxy)
    if crop is None or crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    med = np.median(hsv.reshape(-1, 3), axis=0)
    return float(med[0]), float(med[1]), float(med[2])


def sample_color_ref(frame_bgr: np.ndarray, click_xy: tuple[float, float], box: int = 12) -> ColorRef:
    """Median HSV of a small patch around a click point."""
    h, w = frame_bgr.shape[:2]
    x, y = int(round(click_xy[0])), int(round(click_xy[1]))
    x1, x2 = max(0, x - box), min(w, x + box)
    y1, y2 = max(0, y - box), min(h, y + box)
    patch = frame_bgr[y1:y2, x1:x2]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    med = np.median(hsv.reshape(-1, 3), axis=0)
    return ColorRef(h=float(med[0]), s=float(med[1]), v=float(med[2]))


def hsv_to_lab(h: float, s: float, v: float) -> np.ndarray:
    """Convert a single OpenCV-HSV colour to CIELab (float32 L,a,b)."""
    px = np.uint8([[[int(h) % 180, int(max(0, min(255, s))), int(max(0, min(255, v)))]]])
    bgr = cv2.cvtColor(px, cv2.COLOR_HSV2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    return lab[0, 0].astype(np.float32)


def median_lab(frame_bgr: np.ndarray, xyxy: np.ndarray) -> np.ndarray | None:
    """Median CIELab of a player's torso region (robust to small boxes)."""
    crop = torso_crop(frame_bgr, xyxy)
    if crop is None or crop.size == 0:
        return None
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    med = np.median(lab.reshape(-1, 3), axis=0)
    return med.astype(np.float32)


def classify_is_mine(
    frame_bgr: np.ndarray,
    xyxy: np.ndarray,
    my_lab: np.ndarray,
    other_lab: np.ndarray,
) -> bool:
    """Two-colour classifier: True if the torso is closer to MY kit than the other.

    More robust than single-colour tolerance because it only has to decide which
    of the two known kits a player is nearer to.
    """
    lab = median_lab(frame_bgr, xyxy)
    if lab is None:
        return False
    dm = float(np.linalg.norm(lab - my_lab))
    do = float(np.linalg.norm(lab - other_lab))
    return dm < do


def _hue_diff(a: float, b: float) -> float:
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def is_my_team(
    frame_bgr: np.ndarray,
    xyxy: np.ndarray,
    ref: ColorRef,
    cfg: ColorConfig | None = None,
) -> bool:
    cfg = cfg or ColorConfig()
    hsv = median_hsv(frame_bgr, xyxy)
    if hsv is None:
        return False
    h, s, v = hsv

    ref_is_dark = ref.v <= cfg.dark_v_thresh or ref.s <= cfg.dark_s_thresh
    if ref_is_dark:
        # Match on darkness: low value, low-ish saturation. Hue is unreliable here.
        return v <= (ref.v + cfg.v_tol) and s <= max(cfg.dark_s_thresh + cfg.s_tol, ref.s + cfg.s_tol)

    return (
        _hue_diff(h, ref.h) <= cfg.h_tol
        and abs(s - ref.s) <= cfg.s_tol
        and abs(v - ref.v) <= cfg.v_tol
    )
