"""Jersey-colour primitives shared by Stage 0 (seed) and Stage 6 (team filter).

These are the robust part of v1's old team classifier — the torso crop + median
HSV — reimplemented here (they are a few lines) so v2 does not import v1's
``team_classify`` and drag in its ``sklearn`` DBSCAN dependency, which v2
explicitly drops in favour of a cheap per-contact colour check.
"""

from __future__ import annotations

import math

import cv2
import numpy as np


def torso_crop(frame: np.ndarray, bbox: list[float]) -> np.ndarray | None:
    """Central torso region of a player box (excludes head/legs/side background).

    Matches v1's crop geometry (15-55% of height, inset 20% each side) so kit
    colours are consistent with the shipped pipeline.
    """
    x1, y1, x2, y2 = bbox
    h = max(y2 - y1, 1)
    w = max(x2 - x1, 1)
    ty1, ty2 = int(y1 + 0.15 * h), int(y1 + 0.55 * h)
    tx1, tx2 = int(x1 + 0.2 * w), int(x2 - 0.2 * w)
    if ty2 <= ty1 or tx2 <= tx1:
        return None
    crop = frame[ty1:ty2, tx1:tx2]
    return crop if crop.size > 0 else None


def hsv_feature(crop: np.ndarray) -> np.ndarray:
    """Median HSV (OpenCV ranges: H 0-179, S/V 0-255) over a crop."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    return np.median(hsv.reshape(-1, 3), axis=0).astype(np.float32)


def torso_hsv(
    frame: np.ndarray, bbox: list[float], *, min_area: int = 0
) -> np.ndarray | None:
    """Median torso HSV of a player, or None if the torso region is degenerate.

    ``min_area`` guards against tiny, unreliable crops: on wide/distant footage a
    player may be only a handful of pixels, so the "torso" is mostly grass and
    its colour is meaningless. Below ``min_area`` px we return None (colour
    unmeasurable) so Stage 6 leaves the contact *undecided* rather than
    confidently mislabelling it — failing safe toward recall.
    """
    crop = torso_crop(frame, bbox)
    if crop is None or crop.size == 0:
        return None
    if min_area > 0 and (crop.shape[0] * crop.shape[1]) < min_area:
        return None
    return hsv_feature(crop)


# Pitch grass is a fairly tight green band in HSV. On distant footage a player's
# torso crop is mostly grass, which swamps the jersey colour, so team-colour
# reads (yellow vs black) collapse toward green. Excluding grass first recovers
# the actual kit.
_GRASS_H_LO, _GRASS_H_HI = 32, 95   # OpenCV hue (0-179): yellow ~20-31, grass ~35-85
_GRASS_S_MIN = 40
_GRASS_V_MIN = 40


def jersey_hsv(
    frame: np.ndarray,
    bbox: list[float],
    *,
    min_area: int = 0,
    min_keep_frac: float = 0.12,
) -> np.ndarray | None:
    """Median torso HSV with pitch-grass pixels removed, so a distant player's
    kit colour isn't drowned out by the grass in the crop. Returns None if the
    crop is too small or almost entirely grass (colour unmeasurable)."""
    crop = torso_crop(frame, bbox)
    if crop is None or crop.size == 0:
        return None
    if min_area > 0 and (crop.shape[0] * crop.shape[1]) < min_area:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    grass = ((h >= _GRASS_H_LO) & (h <= _GRASS_H_HI)
             & (s >= _GRASS_S_MIN) & (v >= _GRASS_V_MIN))
    keep = ~grass
    n = int(keep.sum())
    if n < max(8, int(min_keep_frac * keep.size)):
        return None                       # mostly grass → don't trust it
    px = hsv.reshape(-1, 3)[keep.reshape(-1)]
    return np.median(px, axis=0).astype(np.float32)


def median_hsv(features: list[np.ndarray]) -> np.ndarray | None:
    """Robust median across several HSV samples (drops per-frame noise)."""
    if not features:
        return None
    return np.median(np.stack(features, axis=0), axis=0).astype(np.float32)


def hsv_distance(
    a: np.ndarray | None,
    b: np.ndarray | None,
    *,
    hue_w: float = 2.0,
    sat_w: float = 1.0,
    val_w: float = 0.5,
) -> float | None:
    """Hue-weighted HSV distance with circular hue.

    Hue dominates (kit identity); brightness (V) is down-weighted because it
    swings most with lighting/shadow. Returns None if either input is None.
    """
    if a is None or b is None:
        return None
    dh = abs(float(a[0]) - float(b[0]))
    dh = min(dh, 180.0 - dh)  # OpenCV hue is 0..179, circular
    ds = abs(float(a[1]) - float(b[1]))
    dv = abs(float(a[2]) - float(b[2]))
    return math.sqrt((hue_w * dh) ** 2 + (sat_w * ds) ** 2 + (val_w * dv) ** 2)
