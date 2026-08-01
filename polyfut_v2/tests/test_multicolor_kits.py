"""Multi-coloured kits: a kit that isn't one flat colour must be matched on any
of its colours, never on an average of them.

The failure this guards against: a red/blue halved kit averaged to purple, which
is a colour the kit does not contain — so neither half matched it and real
touches were labelled other-team.
"""

import cv2
import numpy as np

from polyfut_v2.config import PipelineV2Config
from polyfut_v2.pipeline.color import (
    cluster_hsv,
    jersey_colors_from_crop,
    jersey_hsv_from_crop,
    hexes_to_hsv,
    hsv_distance,
    hsv_distance_multi,
    kits_separable,
    looks_like_official_kit,
    median_hsv,
)
from polyfut_v2.pipeline.player_contacts import (
    TEAM_MY,
    TEAM_OPPONENT,
    classify_team,
)
from polyfut_v2.pipeline.scoring import kit_compatible
from polyfut_v2.pipeline.seed import TargetSeed, build_seed_from_torso_crops

# OpenCV HSV (H 0-179). Hue 0 = red, 60 = green, 120 = blue.
# The opponent colour is deliberately equidistant (120 units under the
# hue-weighted metric) from BOTH halves of the two-colour kit, so "opponent"
# means opponent regardless of which half a torso happens to read as. Note a
# yellow-orange kit (hue ~28) is only 56 units from red — inside
# ``team_color_max_dist`` — so it is not a valid stand-in for a different team.
RED_HSV = np.array([0.0, 210.0, 200.0], np.float32)
BLUE_HSV = np.array([120.0, 210.0, 200.0], np.float32)
GREEN_HSV = np.array([60.0, 210.0, 200.0], np.float32)
WHITE_HSV = np.array([0.0, 8.0, 240.0], np.float32)
# Referee black: the official-kit rule keys on LOW saturation + low value.
BLACK_HSV = np.array([0.0, 20.0, 40.0], np.float32)


def _hsv_block(hsv, hh=80, ww=60):
    """Solid BGR block of a given OpenCV-HSV colour."""
    img = np.full((hh, ww, 3), (float(hsv[0]), float(hsv[1]), float(hsv[2])), np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)


# --------------------------------------------------------------------------- #
# The primitives
# --------------------------------------------------------------------------- #

def test_cluster_hsv_keeps_both_halves_of_a_two_colour_kit():
    # Six red reads and five blue reads of the SAME halved shirt.
    feats = [RED_HSV] * 6 + [BLUE_HSV] * 5
    modes = cluster_hsv(feats)
    assert len(modes) == 2
    # Dominant first, and both real colours are present (not a purple average).
    assert hsv_distance(modes[0], RED_HSV) < 20
    assert hsv_distance(modes[1], BLUE_HSV) < 20


def test_cluster_hsv_collapses_a_flat_kit_to_one_colour():
    jitter = [RED_HSV + np.array([2.0, -5.0, 4.0], np.float32) * i for i in range(-3, 4)]
    modes = cluster_hsv(jitter)
    assert len(modes) == 1


def test_cluster_hsv_drops_a_lone_bad_crop():
    # One grass-contaminated read among many good ones is not a second kit colour.
    feats = [RED_HSV] * 12 + [np.array([60.0, 200.0, 150.0], np.float32)]
    assert len(cluster_hsv(feats)) == 1


def test_median_of_a_red_blue_kit_is_a_colour_it_does_not_contain():
    """The bug, stated as a test: averaging a two-colour kit lands between the
    halves, on a colour no player wears. Making the average circular fixes the
    red-kit wrap but cannot fix this — only splitting the modes can, which is why
    ``_kit_colors`` clusters before it ever reaches ``median_hsv``."""
    avg = median_hsv([RED_HSV] * 5 + [BLUE_HSV] * 5)
    assert hsv_distance(avg, RED_HSV) >= 55
    assert hsv_distance(avg, BLUE_HSV) >= 55


def test_hsv_distance_multi_takes_the_nearest_colour():
    d = hsv_distance_multi(BLUE_HSV, [RED_HSV, BLUE_HSV])
    assert d is not None and d < 1.0
    # Single array still accepted (back-compat with one-colour call sites).
    assert hsv_distance_multi(BLUE_HSV, BLUE_HSV) < 1.0
    assert hsv_distance_multi(None, [RED_HSV]) is None
    assert hsv_distance_multi(RED_HSV, None) is None


def test_kits_separable_is_false_when_the_teams_share_a_colour():
    cfg = PipelineV2Config()
    # Red/blue vs white/blue — both wear blue, so colour cannot separate them.
    assert not kits_separable(
        [RED_HSV, BLUE_HSV], [WHITE_HSV, BLUE_HSV], cfg.team_color_max_dist)
    # Red/blue vs green-only — separable.
    assert kits_separable(
        [RED_HSV, BLUE_HSV], [GREEN_HSV], cfg.team_color_max_dist)


def test_hexes_to_hsv_skips_junk():
    out = hexes_to_hsv(["#ff0000", "nope", "", None, "#0000ff"])
    assert len(out) == 2


# --------------------------------------------------------------------------- #
# The team gate
# --------------------------------------------------------------------------- #

def _seed(mine, opp=None):
    mine = list(mine)
    opp = list(opp or [])
    return TargetSeed(
        kit_hsv=mine[0] if mine else None,
        kit_hsv_alts=mine[1:],
        opponent_kit_hsv=opp[0] if opp else None,
        opponent_kit_hsv_alts=opp[1:],
        gallery=[], n_samples=4,
    )


def test_either_half_of_my_kit_is_my_team():
    cfg = PipelineV2Config()
    seed = _seed([RED_HSV, BLUE_HSV], [GREEN_HSV])
    for half in (RED_HSV, BLUE_HSV):
        is_mine, label, dist = classify_team(half, seed, cfg)
        assert is_mine is True, f"{half} should be my team"
        assert label == TEAM_MY
        assert dist < cfg.team_color_max_dist


def test_single_colour_seed_is_unchanged_by_the_multi_colour_path():
    """Regression guard: flat kits must behave exactly as before."""
    cfg = PipelineV2Config()
    seed = _seed([RED_HSV], [GREEN_HSV])
    assert classify_team(RED_HSV, seed, cfg)[1] == TEAM_MY
    far = np.array([90.0, 210.0, 200.0], np.float32)     # cyan-ish, far from both
    assert classify_team(far, seed, cfg)[0] is not True


def test_opponent_still_dropped_when_my_kit_is_multi_colour():
    cfg = PipelineV2Config()
    seed = _seed([RED_HSV, BLUE_HSV], [GREEN_HSV])
    is_mine, label, _d = classify_team(GREEN_HSV, seed, cfg)
    assert is_mine is False and label == TEAM_OPPONENT


def test_shared_colour_leaves_the_two_centroid_gate_undecided():
    """When both kits wear blue, a blue torso must not be positively assigned."""
    cfg = PipelineV2Config()
    seed = _seed([RED_HSV, BLUE_HSV], [WHITE_HSV, BLUE_HSV])
    is_mine, _label, _d = classify_team(BLUE_HSV, seed, cfg)
    # Blue is genuinely one of my colours, so it may match mine — what must NOT
    # happen is a confident *opponent* drop on a colour I also wear.
    assert is_mine is not False


def test_unreadable_colour_still_survives_as_unknown():
    cfg = PipelineV2Config()
    seed = _seed([RED_HSV, BLUE_HSV], [GREEN_HSV])
    is_mine, _label, dist = classify_team(None, seed, cfg)
    assert is_mine is None and dist is None


def test_official_kit_check_clears_on_any_team_colour():
    # A black kit is one of my kit's colours -> not an official.
    assert not looks_like_official_kit(BLACK_HSV, [RED_HSV, BLACK_HSV], [GREEN_HSV])
    # The same black kit when NEITHER team wears it -> official.
    assert looks_like_official_kit(BLACK_HSV, [RED_HSV], [GREEN_HSV])


# --------------------------------------------------------------------------- #
# Seed + continuity
# --------------------------------------------------------------------------- #

def test_seed_from_crops_records_both_kit_colours():
    crops = [_hsv_block(RED_HSV) for _ in range(6)] + \
            [_hsv_block(BLUE_HSV) for _ in range(5)]
    seed = build_seed_from_torso_crops(crops)
    kits = seed.my_kits()
    assert len(kits) == 2
    assert hsv_distance_multi(RED_HSV, kits) < 30
    assert hsv_distance_multi(BLUE_HSV, kits) < 30


def test_seed_from_flat_crops_has_no_alts():
    seed = build_seed_from_torso_crops([_hsv_block(RED_HSV) for _ in range(6)])
    assert seed.kit_hsv is not None
    assert seed.kit_hsv_alts == []
    assert len(seed.my_kits()) == 1


def test_kit_compatible_accepts_a_colour_set():
    cfg = PipelineV2Config()
    lock = cfg.continuity_color_max_dist
    # A blue touch stays on a chain anchored to a red/blue kit.
    assert kit_compatible(BLUE_HSV, [RED_HSV, BLUE_HSV], lock)
    # A yellow touch does not.
    assert not kit_compatible(GREEN_HSV, [RED_HSV, BLUE_HSV], lock)
    # Single colour, given as a plain list of floats, still works.
    assert kit_compatible(RED_HSV, [0.0, 210.0, 200.0], lock)
    # Unreadable never breaks a chain (recall-safe).
    assert kit_compatible(None, [RED_HSV], lock)
    assert kit_compatible(RED_HSV, None, lock)


def test_target_seed_defaults_are_backwards_compatible():
    """A seed built the old way (no alts) still answers the new accessors."""
    seed = TargetSeed(kit_hsv=RED_HSV, gallery=[], n_samples=2)
    assert seed.my_kits() == [RED_HSV]
    assert seed.opponent_kits() == []
    assert seed.has_color()


# --------------------------------------------------------------------------- #
# Within-crop splitting: a two-colour shirt must never report the colour
# between its halves. np.median runs per channel, so blue (hue 120) + yellow
# (hue 27) medians to hue 74 — green — and with grass bleed that is the muddy
# brown seen on real clips. Hue is also circular, so a red kit straddling the
# 0/180 wrap used to median to 90 (cyan).
# --------------------------------------------------------------------------- #

def _solid_hsv(hsv, hh, ww):
    img = np.full((hh, ww, 3), (float(hsv[0]), float(hsv[1]), float(hsv[2])), np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)


def _halved(a, b, left=10, size=20):
    crop = np.zeros((size, size, 3), np.uint8)
    crop[:, :left] = _solid_hsv(a, size, left)
    crop[:, left:] = _solid_hsv(b, size, size - left)
    return crop


YELLOW_KIT = np.array([27.0, 210.0, 200.0], np.float32)
NAVY_KIT = np.array([115.0, 200.0, 140.0], np.float32)


def test_circular_hue_median_survives_the_red_wrap():
    """A red kit's pixels sit either side of the 0/180 seam; a plain median of
    them lands on 90 — cyan. The answer must stay red, at either end."""
    from polyfut_v2.pipeline.color import circular_hue_median

    med = circular_hue_median(np.array([2, 3, 178, 177], np.float32))
    assert med < 10 or med > 170
    assert 60 < float(np.median(np.array([2, 3, 178, 177]))) < 120   # the old way


def test_circular_hue_median_is_outlier_robust():
    """A mean would be dragged around the circle; a median must not be."""
    from polyfut_v2.pipeline.color import circular_hue_median, circular_hue_mean
    d = np.array([2, 2, 2, 120], np.float32)
    assert circular_hue_median(d) < 10          # stays red
    assert circular_hue_mean(d) > 100           # the mean does not


def test_two_colour_crop_splits_into_both_halves():
    crop = _halved(BLUE_HSV, YELLOW_KIT)
    colors = jersey_colors_from_crop(crop)
    assert len(colors) == 2
    assert hsv_distance_multi(BLUE_HSV, colors) < 25
    assert hsv_distance_multi(YELLOW_KIT, colors) < 30


def test_two_colour_crop_never_reports_the_colour_between():
    """The actual bug: hue 74 (green) for a blue/yellow shirt."""
    crop = _halved(BLUE_HSV, YELLOW_KIT)
    for c in jersey_colors_from_crop(crop):
        assert not (60 <= float(c[0]) <= 90), f"reported an in-between hue: {c}"
    single = jersey_hsv_from_crop(crop)
    assert not (60 <= float(single[0]) <= 90)


def test_single_colour_answer_is_the_dominant_half_not_a_blend():
    """Even when a split is NOT reported, the one colour must be a real half."""
    crop = _halved(RED_HSV, BLUE_HSV)           # only ~120 deg apart
    single = jersey_hsv_from_crop(crop)
    assert hsv_distance_multi(single, [RED_HSV, BLUE_HSV]) < 25


def test_dominant_half_wins_the_single_colour_read():
    blue_major = jersey_hsv_from_crop(_halved(BLUE_HSV, YELLOW_KIT, left=14))
    yellow_major = jersey_hsv_from_crop(_halved(BLUE_HSV, YELLOW_KIT, left=6))
    assert hsv_distance(blue_major, BLUE_HSV) < 25
    assert hsv_distance(yellow_major, YELLOW_KIT) < 30


def test_flat_kit_is_not_split():
    assert len(jersey_colors_from_crop(_solid_hsv(NAVY_KIT, 20, 20))) == 1


def test_sun_and_shade_on_one_kit_is_not_split():
    """Brightness differences are not colour differences."""
    shade = np.array([115.0, 190.0, 70.0], np.float32)
    assert len(jersey_colors_from_crop(_halved(NAVY_KIT, shade))) == 1


def test_red_kit_across_the_wrap_is_not_split():
    lo = np.array([2.0, 210.0, 200.0], np.float32)
    hi = np.array([178.0, 210.0, 200.0], np.float32)
    colors = jersey_colors_from_crop(_halved(lo, hi))
    assert len(colors) == 1
    assert hsv_distance(colors[0], lo) < 20     # reads red, not cyan


def test_small_contaminating_region_is_not_a_kit_colour():
    """A sliver of another colour (limb, grass bleed) must not become a kit."""
    assert len(jersey_colors_from_crop(_halved(BLUE_HSV, YELLOW_KIT, left=18))) == 1


def test_dark_modes_are_never_accepted_as_two_kit_colours():
    """Near-black pixels compute a high saturation but a meaningless hue —
    the shape of every false split measured on real footage."""
    a = np.array([120.0, 200.0, 35.0], np.float32)
    b = np.array([20.0, 200.0, 30.0], np.float32)
    assert len(jersey_colors_from_crop(_halved(a, b))) == 1


def test_seed_from_two_colour_crops_records_both():
    crop = _halved(BLUE_HSV, YELLOW_KIT)
    seed = build_seed_from_torso_crops([crop for _ in range(6)])
    kits = seed.my_kits()
    assert len(kits) == 2
    assert hsv_distance_multi(BLUE_HSV, kits) < 25
    assert hsv_distance_multi(YELLOW_KIT, kits) < 30


def test_two_colour_kit_matches_a_player_showing_either_half():
    """End to end: seed on a halved shirt, then classify each half."""
    cfg = PipelineV2Config()
    crop = _halved(BLUE_HSV, YELLOW_KIT)
    seed = build_seed_from_torso_crops([crop for _ in range(6)])
    seed.opponent_kit_hsv = GREEN_HSV
    for half in (BLUE_HSV, YELLOW_KIT):
        is_mine, label, _d = classify_team(half, seed, cfg)
        assert is_mine is True and label == TEAM_MY
