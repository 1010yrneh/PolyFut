"""Tests for the /api/teams kit-colour preview and its grass-masking fix.

Bug this covers: on wide/low-res broadcast footage a player is only ~15-20px
tall, so the fixed torso sub-crop often includes real pitch grass. Reading a
plain median colour over the whole crop lets that grass pull both kit swatches
toward a muddy olive/green — exactly what a real 640x360 Community Shield test
clip showed (khaki + sage swatches instead of red + sky blue). Grass-masking
(already shipped in polyfut_v2.pipeline.color) is mirrored here for the
legacy team_preview/team_classify module that /api/teams still uses.
"""

import cv2
import numpy as np

from polyfut_video.pipeline.team_classify import (
    _jersey_bgr_median, _jersey_hsv_feature,
)
from polyfut_video.pipeline.team_preview import _crops_to_hex, _kmeans_two_kits

RED_BGR = (0, 0, 200)
BLUE_BGR = (200, 0, 0)
GRASS_BGR = (40, 130, 40)   # inside the grass hue band


def _hsv_solid(h, s, v, hh=80, ww=60):
    hsv = np.full((hh, ww, 3), (h, s, v), np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _solid(color, h=40, w=30):
    return np.full((h, w, 3), color, dtype=np.uint8)


def _mostly_grass_crop(kit_bgr, kit_frac=0.2, h=40, w=30):
    """A torso crop that's mostly pitch grass with a kit-coloured strip
    covering exactly ``kit_frac`` of the crop's area, like a real
    tiny/distant player detection where grass dominates the torso box."""
    crop = _hsv_solid(55, 180, 150, h, w)          # grass fill
    kh = max(1, int(round(h * kit_frac)))          # full-width strip: area == kit_frac
    crop[0:kh, :] = kit_bgr
    return crop


def test_jersey_hsv_feature_ignores_grass():
    crop = _mostly_grass_crop(RED_BGR)
    feat = _jersey_hsv_feature(crop)
    assert feat is not None
    assert feat[0] < 10 or feat[0] > 170   # reads red, not grass green (~55)


def test_jersey_hsv_feature_none_when_all_grass():
    crop = _hsv_solid(55, 180, 150)
    assert _jersey_hsv_feature(crop) is None


def test_jersey_bgr_median_recovers_kit_colour_not_grass():
    crop = _mostly_grass_crop(BLUE_BGR)
    med = _jersey_bgr_median(crop)
    assert med is not None
    b, g, r = med
    assert b > g and b > r   # blue kit, not green grass


def test_crops_to_hex_ignores_grass_bleed():
    """A pool of red-kit crops, each with a grass-contaminated corner, should
    still swatch to red — not the muddy olive a plain whole-crop median gives."""
    crops = [_mostly_grass_crop(RED_BGR, kit_frac=0.6) for _ in range(6)]
    hexval = _crops_to_hex(crops)
    assert hexval is not None
    r = int(hexval[1:3], 16); g = int(hexval[3:5], 16); b = int(hexval[5:7], 16)
    assert r > g and r > b, f"expected reddish swatch, got #{hexval}"


def test_crops_to_hex_all_grass_returns_none():
    crops = [_hsv_solid(55, 180, 150) for _ in range(4)]
    assert _crops_to_hex(crops) is None


def test_kmeans_two_kits_separates_red_and_blue_despite_grass():
    red_crops = [_mostly_grass_crop(RED_BGR, kit_frac=0.5) for _ in range(6)]
    blue_crops = [_mostly_grass_crop(BLUE_BGR, kit_frac=0.5) for _ in range(6)]
    grouped = _kmeans_two_kits(red_crops + blue_crops)
    assert grouped is not None
    g0, g1 = grouped
    hex0, hex1 = _crops_to_hex(g0), _crops_to_hex(g1)
    assert hex0 is not None and hex1 is not None
    hexes = {hex0, hex1}
    reddish = any(int(h[1:3], 16) > int(h[3:5], 16) and int(h[1:3], 16) > int(h[5:7], 16) for h in hexes)
    bluish = any(int(h[5:7], 16) > int(h[1:3], 16) and int(h[5:7], 16) > int(h[3:5], 16) for h in hexes)
    assert reddish and bluish, f"expected one red-ish and one blue-ish swatch, got {hexes}"


# --------------------------------------------------------------------------- #
# Multi-coloured kits: keep the colours a kit contains, don't average them.
# A red/blue halved shirt averaged to purple gives the user a swatch their kit
# does not contain, and nothing downstream then matches either half.
# --------------------------------------------------------------------------- #

def test_crops_to_hexes_keeps_both_colours_of_a_halved_kit():
    from polyfut_video.pipeline.team_preview import _crops_to_hexes

    crops = [_solid(RED_BGR) for _ in range(6)] + [_solid(BLUE_BGR) for _ in range(5)]
    hexes = _crops_to_hexes(crops)
    assert len(hexes) == 2
    # Dominant (red) first; both are saturated primaries, neither is purple.
    r0, g0, b0 = (int(hexes[0][i:i + 2], 16) for i in (1, 3, 5))
    r1, g1, b1 = (int(hexes[1][i:i + 2], 16) for i in (1, 3, 5))
    assert r0 > 150 and b0 < 80          # red
    assert b1 > 150 and r1 < 80          # blue


def test_crops_to_hexes_collapses_a_flat_kit():
    from polyfut_video.pipeline.team_preview import _crops_to_hexes

    assert len(_crops_to_hexes([_solid(RED_BGR) for _ in range(8)])) == 1


def test_crops_to_hexes_ignores_a_lone_odd_crop():
    from polyfut_video.pipeline.team_preview import _crops_to_hexes

    crops = [_solid(RED_BGR) for _ in range(14)] + [_solid(BLUE_BGR)]
    assert len(_crops_to_hexes(crops)) == 1


def test_crops_to_hex_still_returns_the_dominant_single_colour():
    """The single-colour helper is unchanged for existing callers."""
    hx = _crops_to_hex([_solid(RED_BGR) for _ in range(6)])
    assert hx is not None
    r, g, b = (int(hx[i:i + 2], 16) for i in (1, 3, 5))
    assert r > 150 and b < 80
