"""Team-picker swatches: real kit colours, not the average of everything.

Two failures this covers, both seen on real uploads:

* A yellow-and-blue kit averaged to olive. A median across two colours is a
  colour the kit does not contain, and the old code fell back to exactly that
  median whenever the multi-colour split was rejected.
* Both teams at one venue came out brown, because that pitch's turf reads hue
  31 and the fixed grass band started at 32 — so no grass was masked out of the
  torso crops and every "kit" colour was really pitch.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from polyfut_video.pipeline.team_classify import (
    _grass_mask,
    _jersey_bgr_pixels,
    dominant_kit_lab,
    looks_like_pitch,
    pitch_palette,
    pitch_reference_lab,
    standout_colours,
    standout_kit_lab,
)
from polyfut_video.pipeline.team_preview import (
    _bgr_centers_are_distinct_kits, _crops_to_hexes,
)


def _hsv_solid(h, s, v, hh=80, ww=60):
    return cv2.cvtColor(np.full((hh, ww, 3), (h, s, v), np.uint8), cv2.COLOR_HSV2BGR)


def _hue_of(hexstr: str) -> int:
    r, g, b = (int(hexstr[i:i + 2], 16) for i in (1, 3, 5))
    return int(cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0, 0, 0])


def _sat_of(hexstr: str) -> int:
    r, g, b = (int(hexstr[i:i + 2], 16) for i in (1, 3, 5))
    return int(cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0, 0, 1])


def _frame_with_turf(hue: int, w=640, h=360):
    """A frame whose middle band is turf of a given hue, sky above."""
    f = np.zeros((h, w, 3), np.uint8)
    f[: int(h * 0.3)] = _hsv_solid(105, 40, 200, 1, 1)[0, 0]      # pale sky
    f[int(h * 0.3):] = _hsv_solid(hue, 150, 120, 1, 1)[0, 0]      # turf
    return f


# ------------------------------------------------ measuring the pitch, not
# ------------------------------------------------ assuming what it looks like
@pytest.mark.parametrize("hue", [28, 35, 43])
def test_pitch_is_measured_on_any_turf_hue(hue):
    """The fixed grass band only covers *some* pitches; measuring covers all.

    Measured across the uploads on disk, one venue's turf sits at hue 28 and the
    band (which starts at 32) masked 1.2% of it, while other venues at hue 42-43
    masked ~95%. A palette read off the frame has no hue assumption, so it must
    land on the turf either way.
    """
    frames = [_frame_with_turf(hue) for _ in range(4)]
    pal = pitch_palette(frames)
    assert pal is not None and len(pal)
    turf = _hsv_solid(hue, 150, 120, 1, 1)
    turf_lab = cv2.cvtColor(turf, cv2.COLOR_BGR2LAB).reshape(3).astype(np.float32)
    # one of the measured centres must actually be the turf
    assert float(np.linalg.norm(pal - turf_lab[None, :], axis=1).min()) < 20.0


def test_the_fixed_band_really_does_miss_a_bleached_pitch():
    """Pins the bug the palette exists to fix, so it cannot quietly come back."""
    hsv = cv2.cvtColor(_frame_with_turf(28)[150:], cv2.COLOR_BGR2HSV)
    assert _grass_mask(hsv).mean() < 0.05          # band sees essentially none
    pal = pitch_palette([_frame_with_turf(28)] * 4)
    assert pal is not None


def test_a_kit_survives_masking_against_its_own_pitch():
    """Removing the pitch must not also remove the shirt."""
    frames = [_frame_with_turf(28) for _ in range(4)]
    pal = pitch_palette(frames)
    crop = np.zeros((40, 30, 3), np.uint8)
    crop[:] = _hsv_solid(28, 150, 120, 1, 1)[0, 0]          # turf background
    crop[10:30, 8:22] = _hsv_solid(25, 240, 230, 1, 1)[0, 0]  # yellow shirt
    got = standout_colours(crop, None, max_colours=1, palette=pal)
    assert got, "the shirt was masked away with the pitch"
    assert _sat_of("#%02x%02x%02x" % (int(got[0][2]), int(got[0][1]),
                                      int(got[0][0]))) > 120


# ------------------------------------------------------ the second-colour gate
def test_white_and_navy_count_as_two_kit_colours():
    """The old gate demanded S>=70 and V>=70 of both centres.

    White is the least saturated thing on the field and navy fails on value, so
    the two commonest kit colours in this footage could never pass it — and every
    failure fell back to a single averaged swatch.
    """
    white = np.array([235.0, 238.0, 240.0], np.float32)
    navy = np.array([70.0, 40.0, 25.0], np.float32)
    assert _bgr_centers_are_distinct_kits([white, navy])


def test_one_shirt_lit_two_ways_is_still_one_kit_colour():
    sun = np.array([190.0, 70.0, 60.0], np.float32)
    shade = np.array([95.0, 35.0, 30.0], np.float32)
    assert not _bgr_centers_are_distinct_kits([sun, shade])


# ------------------------------------------- dropping crops that are just pitch
def test_recognises_a_crop_that_is_only_worn_pitch():
    """A torso box on a 15px player often contains no player at all.

    A crop of pure turf is already dropped upstream (nothing survives the grass
    mask). The ones that got through — and then dominated the clustering with
    the colour of the ground — are *worn* pitch: dirt and dry grass just outside
    the grass hue band, which the mask keeps and which looks like a kit to
    everything downstream.
    """
    frames = [_frame_with_turf(40)] * 3
    pitch = pitch_reference_lab(frames)
    assert pitch is not None
    worn = _hsv_solid(40, 150, 120)                        # turf, masked away
    worn[:, :20] = _hsv_solid(29, 120, 118, 1, 1)[0, 0]    # dry patch, kept
    lab = dominant_kit_lab(worn)
    assert lab is not None, "worn pitch survives the grass mask — that's the problem"
    assert looks_like_pitch(lab, pitch)


def test_a_real_kit_is_not_mistaken_for_pitch():
    frames = [_frame_with_turf(40)] * 3
    pitch = pitch_reference_lab(frames)
    for hue in (0, 25, 115, 160):        # red, yellow, blue, magenta
        kit = _hsv_solid(hue, 220, 200)
        assert not looks_like_pitch(dominant_kit_lab(kit), pitch), hue


def test_pitch_reference_is_none_without_pitch():
    """No turf in view -> no reference, and the filter simply does not fire."""
    sky = np.full((360, 640, 3), _hsv_solid(115, 200, 200, 1, 1)[0, 0], np.uint8)
    assert pitch_reference_lab([sky] * 3) is None
    assert not looks_like_pitch(dominant_kit_lab(_hsv_solid(25, 220, 200)), None)


# ------------------------------------------- a colour the kit actually has
def test_two_colour_kit_never_reports_the_blend():
    """Yellow + blue must not average to olive — the reported bug."""
    crop = np.zeros((80, 60, 3), np.uint8)
    crop[:40] = _hsv_solid(25, 240, 230, 1, 1)[0, 0]    # yellow
    crop[40:] = _hsv_solid(110, 240, 200, 1, 1)[0, 0]   # blue
    got = _crops_to_hexes([crop] * 4)
    assert got
    hues = [_hue_of(h) for h in got]
    # every reported colour is one the kit contains, none is the olive between
    for hue in hues:
        assert min(abs(hue - 25), abs(hue - 110)) < 18, (got, hues)


def test_single_colour_kit_is_unchanged():
    crop = _hsv_solid(115, 220, 210)
    got = _crops_to_hexes([crop] * 4)
    assert got
    assert abs(_hue_of(got[0]) - 115) < 10, got


def test_shading_does_not_manufacture_a_second_colour():
    """Sunlit and shaded halves of one kit are one colour, not two."""
    crop = np.zeros((80, 60, 3), np.uint8)
    crop[:40] = _hsv_solid(115, 220, 230, 1, 1)[0, 0]
    crop[40:] = _hsv_solid(115, 220, 110, 1, 1)[0, 0]   # same hue, darker
    got = _crops_to_hexes([crop] * 4)
    assert len(got) == 1, got
    assert abs(_hue_of(got[0]) - 115) < 12, got


def test_dominant_colour_wins_when_the_split_is_rejected():
    """A mostly-red kit with a small contrast trim reports red, not pink-brown."""
    crop = np.zeros((80, 60, 3), np.uint8)
    crop[:70] = _hsv_solid(175, 230, 200, 1, 1)[0, 0]   # red (near the 0/180 seam)
    crop[70:] = _hsv_solid(95, 230, 200, 1, 1)[0, 0]    # small cyan trim
    got = _crops_to_hexes([crop] * 4)
    assert got
    hue = _hue_of(got[0])
    assert min(hue, 180 - hue) < 20, got     # red, not the average of red+cyan


def test_eyedropper_picks_the_shirt_not_the_biggest_region():
    """The most COMMON colour in a torso crop is often not the shirt.

    Here two thirds of the crop is dull shadow and one third is a red shirt.
    Picking by size returns the shadow; picking by distinctiveness returns red.
    """
    pitch = pitch_reference_lab([_frame_with_turf(40)] * 3)
    crop = np.zeros((90, 60, 3), np.uint8)
    crop[:60] = _hsv_solid(20, 30, 70, 1, 1)[0, 0]      # drab shadow, 2/3
    crop[60:] = _hsv_solid(175, 235, 205, 1, 1)[0, 0]   # red shirt, 1/3
    got = standout_colours(crop, pitch, max_colours=1)
    assert got, "a readable crop must yield a colour"
    hexed = "#%02x%02x%02x" % (int(got[0][2]), int(got[0][1]), int(got[0][0]))
    hue = _hue_of(hexed)
    assert min(hue, 180 - hue) < 22, hexed          # red, not the drab majority
    assert _sat_of(hexed) > 120, hexed


def test_eyedropper_finds_a_white_kit_which_saturation_cannot():
    """White is the least saturated thing on the pitch but the furthest from
    green turf, so distinctiveness must be measured against the pitch."""
    pitch = pitch_reference_lab([_frame_with_turf(40)] * 3)
    crop = np.zeros((90, 60, 3), np.uint8)
    crop[:55] = _hsv_solid(40, 120, 110, 1, 1)[0, 0]    # turf-ish majority
    crop[55:] = _hsv_solid(0, 6, 240, 1, 1)[0, 0]       # white shirt
    got = standout_colours(crop, pitch, max_colours=1)
    assert got
    assert min(got[0]) > 170, got[0]                    # near-white in BGR


def test_eyedropper_reports_both_colours_of_a_halved_kit():
    pitch = pitch_reference_lab([_frame_with_turf(40)] * 3)
    crop = np.zeros((80, 60, 3), np.uint8)
    crop[:40] = _hsv_solid(25, 240, 230, 1, 1)[0, 0]    # yellow
    crop[40:] = _hsv_solid(110, 240, 200, 1, 1)[0, 0]   # blue
    got = standout_colours(crop, pitch, max_colours=2)
    assert len(got) == 2, got
    hues = sorted(_hue_of("#%02x%02x%02x" % (int(c[2]), int(c[1]), int(c[0])))
                  for c in got)
    assert abs(hues[0] - 25) < 18 and abs(hues[1] - 110) < 18, hues


@pytest.mark.parametrize("hue", [115, 175, 25])
def test_eyedropper_does_not_split_one_shirt_lit_two_ways(hue):
    """Sun and shade on one shirt is not a second kit colour."""
    pitch = pitch_reference_lab([_frame_with_turf(40)] * 3)
    crop = np.zeros((80, 60, 3), np.uint8)
    crop[:40] = _hsv_solid(hue, 220, 230, 1, 1)[0, 0]
    crop[40:] = _hsv_solid(hue, 220, 115, 1, 1)[0, 0]   # same hue, shaded
    got = standout_colours(crop, pitch, max_colours=2)
    assert len(got) == 1, got


def test_eyedropper_still_splits_a_white_and_black_kit():
    """Whose entire difference is lightness — chroma cannot see it at all."""
    pitch = pitch_reference_lab([_frame_with_turf(40)] * 3)
    crop = np.zeros((80, 60, 3), np.uint8)
    crop[:40] = _hsv_solid(0, 6, 245, 1, 1)[0, 0]      # white half
    crop[40:] = _hsv_solid(0, 10, 25, 1, 1)[0, 0]      # black half
    got = standout_colours(crop, pitch, max_colours=2)
    assert len(got) == 2, got
    lightnesses = sorted(int(max(c)) for c in got)
    assert lightnesses[0] < 60 and lightnesses[1] > 200, got


def test_white_and_navy_are_separable_by_the_clustering_feature():
    """Hue cannot tell white from navy — an unsaturated pixel's hue is noise —
    which is why the two teams were never separated. Lab must."""
    white = dominant_kit_lab(_hsv_solid(0, 8, 230))
    navy = dominant_kit_lab(_hsv_solid(115, 200, 60))
    yellow = dominant_kit_lab(_hsv_solid(25, 230, 230))
    blue = dominant_kit_lab(_hsv_solid(115, 230, 200))
    assert np.linalg.norm(white - navy) > 60      # differ mostly on lightness
    assert np.linalg.norm(yellow - blue) > 60     # differ mostly on colour


def test_swatches_still_work_with_no_pitch_reference():
    """Uncalibrated / no-turf footage keeps the previous behaviour."""
    crop = _hsv_solid(115, 220, 210)
    assert _jersey_bgr_pixels(crop) is not None
    assert _crops_to_hexes([crop] * 3)
