"""Measuring the pitch instead of assuming a hue band.

The fixed band (hue 32-95) is a guess about turf, and it is wrong often enough
to break everything downstream. Measured across the uploads on disk one venue's
pitch sits at hue 28 while others sit at 42-43; on the first, the band masks
1.2% of the ground, so `jersey_hsv` returned the GROUND for every player (all 14
tracked players on f1fcfbb84a0d read hue 17-26, spread 8.5, against a turf hue
of 28) and every colour decision downstream ran on turf brightness.

Two ideas replace it, and these tests pin both:

* **local** — the surface is sampled from a ring around the player's own box, so
  it is the ground they are standing on. No global constant, any venue.
* **chroma, not brightness** — illumination changes how bright a surface is,
  material changes its colour. Matching on hue+saturation while ignoring value
  covers sunlit and shaded turf from one sample, and keeps navy (different hue)
  and white (near-zero saturation) out of the mask.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from polyfut_v2.pipeline.color import (
    _surface_mask,
    jersey_hsv,
    jersey_hsv_from_crop,
    local_surface_hsv,
)

# OpenCV HSV. The bleached pitch that broke the band, and a normal green one.
OLIVE = (28, 163, 80)
GREEN = (42, 150, 120)
NAVY = (112, 190, 70)
WHITE = (20, 18, 235)
YELLOW = (25, 220, 210)


def _hsv_bgr(hsv):
    return cv2.cvtColor(np.uint8([[list(hsv)]]), cv2.COLOR_HSV2BGR)[0, 0]


def _scene(pitch, kit, *, box=(60, 40, 90, 110), size=(200, 300), fill=0.55):
    """A frame of pitch with one player box whose torso carries ``kit``."""
    frame = np.zeros((size[0], size[1], 3), np.uint8)
    frame[:] = _hsv_bgr(pitch)
    x1, y1, x2, y2 = box
    h, w = y2 - y1, x2 - x1
    ty1, ty2 = int(y1 + 0.15 * h), int(y1 + 0.55 * h)
    tx1, tx2 = int(x1 + 0.2 * w), int(x2 - 0.2 * w)
    # only part of the torso is kit; the rest stays pitch, as in a real crop
    kw = max(1, int((tx2 - tx1) * fill))
    frame[ty1:ty2, tx1:tx1 + kw] = _hsv_bgr(kit)
    return frame, [float(v) for v in box]


# --------------------------------------------------------- measuring the ring
@pytest.mark.parametrize("pitch", [OLIVE, GREEN, (12, 140, 90), (60, 120, 100)])
def test_the_ring_measures_whatever_the_ground_actually_is(pitch):
    """No hue assumption at all — the reference is the ground that is there."""
    frame, box = _scene(pitch, NAVY)
    surf = local_surface_hsv(frame, box)
    assert surf is not None
    assert abs(float(surf[0]) - pitch[0]) <= 3.0, surf


def test_the_players_own_pixels_do_not_define_the_ground():
    """A big navy player must not drag the surface toward navy."""
    frame, box = _scene(OLIVE, NAVY, box=(60, 40, 120, 160), fill=1.0)
    surf = local_surface_hsv(frame, box)
    assert surf is not None
    assert abs(float(surf[0]) - OLIVE[0]) <= 3.0


def test_a_cluttered_surround_refuses_rather_than_guessing():
    """Against a crowd or hoarding the ring is not one surface; the caller then
    keeps the old band rather than subtracting something that is not ground."""
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (200, 300, 3), dtype=np.uint8)
    assert local_surface_hsv(frame, [60.0, 40.0, 90.0, 110.0]) is None


def test_a_box_with_no_room_around_it_refuses():
    frame, _box = _scene(OLIVE, NAVY)
    assert local_surface_hsv(frame, [0.0, 0.0, 2.0, 2.0]) is None


# ------------------------------------------------- chroma, not brightness
def test_the_saturation_tolerance_is_measured_not_fixed():
    """How much a pitch varies is a property of that pitch.

    Measured p90 deviation of turf saturation from its own local median across
    the uploads on disk: 41, 39, 58, 48, 50, 79. A single constant would either
    leak worn turf through as a kit or erase a yellow shirt on an olive pitch.
    """
    frame, box = _scene(OLIVE, NAVY)
    surf = local_surface_hsv(frame, box)
    assert surf is not None and surf.shape[0] == 4
    assert 25.0 <= float(surf[3]) <= 70.0


def test_a_yellow_kit_survives_a_bleached_olive_pitch():
    """The case that broke a fixed tolerance: olive turf is a dark yellow, so
    hue alone cannot separate them and the saturation gap is all there is."""
    frame, box = _scene(OLIVE, YELLOW)
    got = jersey_hsv(frame, box)
    assert got is not None, "the yellow kit was erased as ground"
    assert float(got[1]) > float(OLIVE[1]) + 20.0, got


def test_sun_and_shade_on_one_surface_are_both_masked():
    """The whole point: a single sample covers the entire lighting range.

    A band wide enough to do this in value would swallow kits, which is why the
    per-clip band attempt was reverted.
    """
    surface = np.array([28.0, 163.0, 80.0], np.float32)
    sunlit = np.zeros((4, 4, 3), np.float32)
    sunlit[..., :] = (28.0, 155.0, 200.0)      # same turf, bright sun
    shaded = np.zeros((4, 4, 3), np.float32)
    shaded[..., :] = (28.0, 170.0, 30.0)       # same turf, deep shade
    assert _surface_mask(sunlit, surface).all()
    assert _surface_mask(shaded, surface).all()


def test_a_navy_shirt_is_not_mistaken_for_turf():
    surface = np.array([28.0, 163.0, 80.0], np.float32)
    navy = np.zeros((4, 4, 3), np.float32)
    navy[..., :] = NAVY
    assert not _surface_mask(navy, surface).any()


def test_a_white_shirt_is_not_mistaken_for_turf():
    """White is dim in hue terms but its saturation is nothing like turf."""
    surface = np.array([28.0, 163.0, 80.0], np.float32)
    white = np.zeros((4, 4, 3), np.float32)
    white[..., :] = WHITE
    assert not _surface_mask(white, surface).any()


# ------------------------------------------------------- the read end to end
@pytest.mark.parametrize("pitch", [OLIVE, GREEN])
@pytest.mark.parametrize("kit,expect_h", [(NAVY, 112), (YELLOW, 25)])
def test_the_kit_is_recovered_on_any_pitch(pitch, kit, expect_h):
    """Given pixels to work with, the read is the shirt, not the ground."""
    frame, box = _scene(pitch, kit)
    got = jersey_hsv(frame, box)
    assert got is not None
    dh = abs(float(got[0]) - expect_h)
    assert min(dh, 180 - dh) <= 8.0, got


def test_the_band_alone_fails_on_the_bleached_pitch():
    """Pins why this work exists: without a measured surface, an olive pitch is
    invisible to the band and the read comes back as the ground."""
    frame, box = _scene(OLIVE, NAVY)
    from polyfut_v2.pipeline.color import torso_crop
    crop = torso_crop(frame, box)
    band_read = jersey_hsv_from_crop(crop)                     # no surface
    surf_read = jersey_hsv_from_crop(crop,
                                     surface=local_surface_hsv(frame, box))
    assert band_read is not None and surf_read is not None
    # the band lets the olive ground win; the measured surface does not
    assert abs(float(band_read[0]) - OLIVE[0]) <= 6.0, band_read
    assert abs(float(surf_read[0]) - NAVY[0]) <= 8.0, surf_read


def test_no_surface_still_returns_the_old_behaviour():
    """Fallback must be exactly the previous code path, not a new one."""
    frame, box = _scene(GREEN, NAVY)
    from polyfut_v2.pipeline.color import torso_crop
    crop = torso_crop(frame, box)
    assert jersey_hsv_from_crop(crop, surface=None) is not None
