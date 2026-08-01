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


def grass_fraction(frame: np.ndarray) -> float:
    """Fraction of a whole frame's pixels that read as pitch grass.

    A normal wide broadcast angle is dominated by pitch; a crowd shot, bench/
    dugout close-up, or scoreboard graphic mostly isn't. Cheap scene-type
    signal — reuses the same grass band as the jersey-colour masking above.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    grass = ((h >= _GRASS_H_LO) & (h <= _GRASS_H_HI)
             & (s >= _GRASS_S_MIN) & (v >= _GRASS_V_MIN))
    return float(grass.mean())


# Any plausible pitch hue. Deliberately much wider than the ``_GRASS_*`` band
# above: turf reads anywhere from bleached yellow-green (hue ~25 in bright sun)
# to deep green (hue ~45), and this band only has to say "could be pitch", not
# "is definitely grass". Running tracks (saturated blue ~100 or red ~5/175) and
# painted surfaces fall well outside it.
_PITCH_HUE_LO, _PITCH_HUE_HI = 15, 60
# A pixel below this saturation carries no usable hue (white lines, concrete,
# shadow, the ball itself) — those get no vote either way.
_SURFACE_S_MIN = 60


def is_bbox_on_foreign_surface(
    frame: np.ndarray,
    bbox: list[float],
    *,
    check_half_px: float = 18.0,
    min_foreign_frac: float = 0.70,
    min_colored_frac: float = 0.25,
) -> bool:
    """True only when a detection is *confidently* on a non-pitch surface.

    Replaces the old "grass fraction ≥ threshold" test, which asked the wrong
    question: it required positive proof of grass, so it rejected real on-pitch
    balls on any turf the narrow grass band didn't match (measured: 49% of
    on-pitch positions on bright sun-bleached pitches). This asks instead whether
    the surroundings are positively something else — a blue or red running
    track, painted concrete — and rejects only then.

    The ball's own box is excluded from the sample, so a white ball can't make
    its own surroundings look unreadable. Only saturated pixels vote: greys,
    white lines and shadow are genuinely ambiguous and abstain. Too few voters,
    or a sample that isn't clearly foreign, means **keep** — recall-safe, and
    with no dependence on the exact hue of this pitch's turf.
    """
    if frame is None or frame.size == 0 or len(bbox) != 4:
        return False
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half = max(1.0, float(check_half_px))
    px1, py1 = int(max(0, cx - half)), int(max(0, cy - half))
    px2, py2 = int(min(w, cx + half)), int(min(h, cy + half))
    if px2 - px1 < 4 or py2 - py1 < 4:
        return False
    region = frame[py1:py2, px1:px2]
    if region.size == 0:
        return False

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    hh, ss = hsv[..., 0], hsv[..., 1]

    # Mask out the ball's own pixels — it is the one thing we know isn't surface.
    keep = np.ones(hh.shape, dtype=bool)
    bx1, by1 = int(max(px1, x1)) - px1, int(max(py1, y1)) - py1
    bx2, by2 = int(min(px2, x2)) - px1, int(min(py2, y2)) - py1
    if bx2 > bx1 and by2 > by1:
        keep[by1:by2, bx1:bx2] = False
    if keep.sum() < 16:
        return False

    colored = keep & (ss >= _SURFACE_S_MIN)
    n_colored = int(colored.sum())
    if n_colored < 12 or (n_colored / float(keep.sum())) < min_colored_frac:
        return False                      # not enough opinion to reject on
    hv = hh[colored]
    foreign = ((hv < _PITCH_HUE_LO) | (hv > _PITCH_HUE_HI))
    return float(foreign.mean()) >= min_foreign_frac


def feet_grass_fraction(frame: np.ndarray, bbox: list[float]) -> float | None:
    """Grass fraction in a strip under the player's feet.

    Coaches / bench staff / sideline officials stand on dirt, track, or
    concrete — their feet sample reads mostly non-grass. On-pitch players
    stand on grass. Returns None when the region is too small to trust
    (recall-safe: caller must not drop on None).
    """
    if frame is None or frame.size == 0 or len(bbox) != 4:
        return None
    fh, fw = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    h = max(1.0, y2 - y1)
    w = max(1.0, x2 - x1)
    # Lower ~15% of the body box, plus a few px of ground below the box.
    fy1 = int(max(0, y1 + 0.85 * h))
    fy2 = int(min(fh, y2 + max(2.0, 0.08 * h)))
    fx1 = int(max(0, x1 + 0.2 * w))
    fx2 = int(min(fw, x2 - 0.2 * w))
    if fy2 - fy1 < 2 or fx2 - fx1 < 2:
        return None
    region = frame[fy1:fy2, fx1:fx2]
    if region.size < 16:
        return None
    return grass_fraction(region)


def is_off_pitch(
    frame: np.ndarray,
    bbox: list[float],
    *,
    min_feet_grass: float = 0.18,
    min_scene_grass: float = 0.12,
) -> bool:
    """True only when feet are *confidently* not on pitch grass.

    Requires the frame itself to look like a pitch (enough grass somewhere);
    otherwise a solid-colour crop or indoor shot would falsely flag every body
    as sideline. Unmeasurable feet samples return False (keep — recall-safe).
    """
    if frame is None or frame.size == 0:
        return False
    if grass_fraction(frame) < min_scene_grass:
        return False
    frac = feet_grass_fraction(frame, bbox)
    if frac is None:
        return False
    return frac < min_feet_grass


def looks_like_official_kit(
    kit: np.ndarray | None,
    seed_kit: np.ndarray | list[np.ndarray] | None,
    opponent_kit: np.ndarray | list[np.ndarray] | None = None,
    *,
    team_max_dist: float = 60.0,
) -> bool:
    """Soft referee-kit detector for when the model labels a ref as ``player``.

    Matches common official kits (dark/black, or neon yellow/green) that are
    clearly not either team's kit. Returns False when the colour is unreadable
    or matches a team centroid — never trades a real black-kit player touch.
    """
    if kit is None:
        return False
    h, s, v = float(kit[0]), float(kit[1]), float(kit[2])
    dark = s < 55.0 and v < 100.0
    # Neon yellow / lime refs (high sat + bright, hue near yellow-green).
    neon = s > 110.0 and v > 130.0 and 20.0 <= h <= 55.0
    if not (dark or neon):
        return False
    # Either argument may be one colour or a multi-coloured kit's colour set;
    # matching ANY of a team's colours clears the official label.
    if seed_kit is not None:
        d = hsv_distance_multi(kit, seed_kit)
        if d is not None and d <= team_max_dist:
            return False
    if opponent_kit is not None:
        d = hsv_distance_multi(kit, opponent_kit)
        if d is not None and d <= team_max_dist:
            return False
    return True


# --- measuring the surface instead of assuming it -------------------------- #
#
# The band above is a guess about what turf looks like, and it is wrong often
# enough to break everything downstream. Measured across the uploads on disk, one
# venue's pitch sits at hue 28 while others sit at 42-43; the band starts at 32,
# so it masked 1.2% of the first pitch. Every torso crop there stayed full of
# grass and `jersey_hsv` returned the GROUND for every player — measured on
# f1fcfbb84a0d, all 14 tracked players read hue 17-26 with a spread of 8.5.
# Every colour-driven decision downstream was then operating on turf brightness.
#
# The fix is not a wider band, and not a per-clip band either (that was tried and
# reverted: widening one interval to cover sun and shade makes it swallow kits).
# It is to stop assuming and start measuring, with two ideas doing the work:
#
#   1. LOCAL. The surface is sampled from a ring around this player's own box,
#      so it is the ground they are actually standing on. That adapts to shadow,
#      wear and floodlight across a single frame, which no global value can.
#   2. CHROMA, NOT BRIGHTNESS. Illumination changes how bright a surface is;
#      material changes its colour. Sunlit and shaded turf differ hugely in
#      value but barely in hue, so matching on hue+saturation and ignoring value
#      covers the whole lighting range from one sample — and simultaneously
#      keeps a navy shirt (different hue) and a white shirt (near-zero
#      saturation, where turf is strongly saturated) out of the mask.
#
# Nothing here is tuned to a particular pitch: the reference is whatever the
# ground under this player happens to look like.

_SURFACE_H_TOL = 9.0      # OpenCV hue units (0-179); turf spans a narrow hue
_SURFACE_MIN_PX = 24      # too small a ring to characterise → don't trust it
_SURFACE_MAX_SPREAD = 22.0  # ring this varied isn't one clean surface
# Saturation slack is MEASURED per player, not fixed, because how much a pitch
# varies is itself a property of the pitch. Deviation of turf saturation from
# its own local median, p90, across the uploads on disk: 41, 39, 58, 48, 50, 79
# — a single constant is the same mistake as the hue band, one level down. Too
# tight and worn or shaded turf leaks through as a kit; too loose and a yellow
# shirt on a bleached olive pitch (hue 25 against hue 28 — olive *is* a dark
# yellow) is erased as ground, which is exactly the kit in this footage.
# The clamp keeps a freak ring from disabling the mask or swallowing everything.
_SURFACE_S_TOL_MIN = 25.0
_SURFACE_S_TOL_MAX = 70.0


def local_surface_hsv(
    frame: np.ndarray, bbox: list[float], *, pad: float = 1.6,
) -> np.ndarray | None:
    """Median HSV of the ground immediately around a player box.

    Sampled from a ring: the padded box minus the box itself, so the player's
    own pixels never define the surface they are standing on. Returns None when
    the ring is too small, or too varied to be a single surface (a player against
    a crowd, a hoarding, or the sideline) — in which case the caller keeps the
    old fixed-band behaviour rather than subtracting something that isn't ground.
    """
    if frame is None or frame.size == 0 or len(bbox) != 4:
        return None
    fh, fw = frame.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in bbox)
    bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    px1 = int(max(0, cx - bw * pad / 2.0))
    px2 = int(min(fw, cx + bw * pad / 2.0))
    py1 = int(max(0, cy - bh * pad / 2.0))
    py2 = int(min(fh, cy + bh * pad / 2.0))
    if px2 - px1 < 3 or py2 - py1 < 3:
        return None
    region = frame[py1:py2, px1:px2]
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    # mask out the player's own box within the region
    mask = np.ones(region.shape[:2], dtype=bool)
    bx1, by1 = int(max(px1, x1)) - px1, int(max(py1, y1)) - py1
    bx2, by2 = int(min(px2, x2)) - px1, int(min(py2, y2)) - py1
    if bx2 > bx1 and by2 > by1:
        mask[by1:by2, bx1:bx2] = False
    ring = hsv[mask.reshape(-1)]
    if ring.shape[0] < _SURFACE_MIN_PX:
        return None
    # Circular median for hue; plain median for S and V.
    ang = ring[:, 0] * 2.0 * np.pi / 180.0
    mean_ang = math.atan2(float(np.sin(ang).mean()), float(np.cos(ang).mean()))
    centre = (math.degrees(mean_ang) / 2.0) % 180.0
    unwrapped = (ring[:, 0] - centre + 90.0) % 180.0
    hue = float((np.median(unwrapped) + centre - 90.0) % 180.0)
    # A clean surface is chromatically tight. Spread is measured on hue only —
    # brightness variation across sun and shade is expected and is exactly what
    # this design tolerates.
    if float(np.percentile(np.abs(unwrapped - np.median(unwrapped)), 75)) > _SURFACE_MAX_SPREAD:
        return None
    # How much this particular ground varies in saturation becomes the tolerance
    # used against it — measured only over the ring pixels that share its hue, so
    # a stray shirt or line marking in the ring cannot inflate it.
    dh_ring = np.abs(ring[:, 0] - hue)
    dh_ring = np.minimum(dh_ring, 180.0 - dh_ring)
    same_hue = ring[dh_ring <= _SURFACE_H_TOL]
    s_med = float(np.median(ring[:, 1]))
    if same_hue.shape[0] >= 12:
        s_tol = float(np.percentile(np.abs(same_hue[:, 1] - s_med), 90))
    else:
        s_tol = _SURFACE_S_TOL_MIN
    s_tol = float(np.clip(s_tol, _SURFACE_S_TOL_MIN, _SURFACE_S_TOL_MAX))
    return np.array([hue, s_med, float(np.median(ring[:, 2])), s_tol],
                    dtype=np.float32)


def _surface_mask(hsv: np.ndarray, surface: np.ndarray) -> np.ndarray:
    """Which pixels of ``hsv`` are the same material as ``surface``.

    Value is deliberately not compared: it is what illumination changes, so
    including it would split one surface into "sunlit" and "shaded" and let half
    of it through as if it were a kit.

    ``surface`` may carry a measured saturation tolerance as a 4th element (see
    :func:`local_surface_hsv`); a plain 3-vector falls back to the conservative
    floor.
    """
    h, s = hsv[..., 0], hsv[..., 1]
    dh = np.abs(h - float(surface[0]))
    dh = np.minimum(dh, 180.0 - dh)
    ds = np.abs(s - float(surface[1]))
    s_tol = (float(surface[3]) if np.asarray(surface).shape[0] >= 4
             else _SURFACE_S_TOL_MIN)
    return (dh <= _SURFACE_H_TOL) & (ds <= s_tol)


def _non_grass_pixels(
    crop: np.ndarray | None, *, min_keep_frac: float = 0.12,
    surface: np.ndarray | None = None,
) -> np.ndarray | None:
    """The (N,3) HSV pixels of a torso crop with the pitch removed, or None when
    the crop is empty or too nearly all pitch to trust.

    ``surface`` is the measured local ground colour (see
    :func:`local_surface_hsv`) and is strongly preferred. Without it the fixed
    hue band is used, which works on a typical green pitch and does almost
    nothing on a bleached one.
    """
    if crop is None or crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    grass = ((h >= _GRASS_H_LO) & (h <= _GRASS_H_HI)
             & (s >= _GRASS_S_MIN) & (v >= _GRASS_V_MIN))
    if surface is not None:
        # UNION, not replacement. The band spans 63 hue units and the measured
        # surface only ~18, so on ordinary green turf the band removes strictly
        # more — swapping one for the other regressed two green-pitch clips
        # (reads landing on the pitch hue went 0%->10% and 5%->14%). The band is
        # a decent prior for typical turf; it is simply blind to turf outside
        # it. Keeping both means this can never mask less ground than before,
        # and adds cover exactly where the band fails.
        grass = grass | _surface_mask(hsv, surface)
    keep = ~grass
    if int(keep.sum()) < max(8, int(min_keep_frac * keep.size)):
        return None                       # mostly grass → don't trust it
    return hsv.reshape(-1, 3)[keep.reshape(-1)].astype(np.float32)


def jersey_hsv_from_crop(
    crop: np.ndarray | None, *, min_keep_frac: float = 0.12,
    surface: np.ndarray | None = None,
) -> np.ndarray | None:
    """One representative HSV for a torso crop, with pitch-grass pixels removed.

    Hue is averaged as an angle, not medianed as a number: a red kit straddles
    the 0/180 wrap, and a plain per-channel median put it at hue 90 — cyan. This
    stays a *single* colour by design; the per-contact team read wants one value,
    and a 6x12px contact crop is too small to split a second colour out of
    without inventing one. Use :func:`jersey_colors_from_crop` where several
    crops back the read (the seed and the team picker) and a real second kit
    colour can be confirmed.

    When the crop *is* multi-coloured the **dominant** mode is returned, not an
    average of the modes: averaging two opposed hues is what produced the muddy
    colour in the first place, and circular-averaging them is no better (it lands
    90 degrees away from both). A two-colour player therefore reads as whichever
    of their colours the camera saw most of, which the multi-colour kit sets on
    the other side of the comparison are built to match.
    """
    px = _non_grass_pixels(crop, min_keep_frac=min_keep_frac, surface=surface)
    if px is None and surface is not None:
        # Subtracting the measured ground left nothing — either this player is
        # genuinely the same colour as the pitch, or the ring was contaminated
        # by them. Either way the measurement taught us nothing, so fall back to
        # what the band alone says rather than turning a previously-readable
        # contact into an unreadable one. This keeps the change strictly
        # additive: never a worse answer than before, only sometimes a better.
        px = _non_grass_pixels(crop, min_keep_frac=min_keep_frac)
    if px is None:
        return None
    colors = split_hsv_pixels(px)
    return colors[0] if colors else None


def jersey_hsv(
    frame: np.ndarray,
    bbox: list[float],
    *,
    min_area: int = 0,
    min_keep_frac: float = 0.12,
) -> np.ndarray | None:
    """Median torso HSV with the pitch removed, so a distant player's kit colour
    isn't drowned out by the ground in the crop. Returns None if the crop is too
    small or almost entirely pitch (colour unmeasurable).

    The pitch is measured from a ring around this player rather than assumed
    from a hue band, so it works on turf of any colour. Falls back to the band
    when the surroundings can't be characterised.
    """
    crop = torso_crop(frame, bbox)
    if crop is None or crop.size == 0:
        return None
    if min_area > 0 and (crop.shape[0] * crop.shape[1]) < min_area:
        return None
    return jersey_hsv_from_crop(crop, min_keep_frac=min_keep_frac,
                                surface=local_surface_hsv(frame, bbox))


# --------------------------------------------------------------------------- #
# Multi-coloured shirts: splitting a torso read instead of averaging it
#
# ``np.median`` runs per channel and independently, so a torso that is half one
# colour and half another lands *between* the two modes — a colour the shirt
# does not contain. Measured on synthetic halves: blue (hue 120) + yellow (hue
# 27) reads hue 74 (green); red + blue reads hue 60 (green). With grass bleed and
# the lower saturation of real footage those become the olive/brown swatches seen
# on real clips.
#
# Hue is also *circular* (OpenCV packs 0-360 degrees into 0-179), which breaks
# both the average and any plain spread measure near red: a pure red kit with
# pixels at hue 2 and 178 has a plain median of 90 (cyan) and a plain standard
# deviation of 88 — it would look maximally multi-coloured while being one
# colour. Everything below therefore treats hue as an angle.
# --------------------------------------------------------------------------- #

# A pixel's hue is only meaningful when it is both saturated enough and bright
# enough. Near-white and near-grey pixels have no hue; so do near-black ones, and
# those are the sneakier case — a very dark pixel can compute a *high* saturation
# while its hue is pure quantisation noise (#0a0815 reads S=170, V=21). Both
# floors are needed or shadow votes on what colour a shirt is.
_HUE_VOTE_S_MIN = 40
_HUE_VOTE_V_MIN = 50
# Circular hue spread, in OpenCV hue units, above which a crop is *considered*
# for splitting. Measured over 924 real torso crops from 7 clips, single-colour
# kits run p50=6, p90=20, p95=30, p99=42 — the spread of one dyed shirt at 360p
# is much wider than it looks, because of grass bleed, shadow and skin at the
# neck. Set clear of that noise floor.
#
# The consequence, stated plainly: only *strongly opposed* colour pairs split
# automatically. Blue+yellow are ~186 degrees apart and read ~70, so they split.
# A red+blue kit is only ~120 degrees apart and reads ~34 — inside the
# single-colour noise — so it does not, and falls back to the dominant colour.
# That is the conservative direction (it is exactly today's behaviour) and the
# team screen lets the second colour be added by hand. Lowering this to catch
# those costs false splits on ordinary kits, which is worse: an invented kit
# colour makes the team gate match players who aren't on your team.
_SPLIT_HUE_STD = 45.0
# A *lower* bar for distrusting the plain median. Between this and the split
# threshold we are not confident enough to report two kit colours, but we are
# confident the crop is not one tight colour — so the single answer becomes the
# dominant mode rather than the median of everything. Without this a red/blue
# crop (spread ~34, below the split bar) reported magenta: the exact
# between-the-halves colour this whole module exists to stop.
_MODE_TRUST_HUE_STD = 20.0
# The smaller mode must hold this share of the voting pixels...
_SPLIT_MIN_SHARE = 0.25
# ...and this many pixels outright. A 6x12px contact-scale torso has only a few
# dozen saturated pixels; splitting those is reading noise.
_SPLIT_MIN_VOTERS = 48
# Both modes must be this saturated to count as kit colours. This is the gate
# that does the real work: on real footage the false splits were uniformly pairs
# of dark, washed-out colours (shadow against grass bleed, e.g. #2f3249 vs
# #412c31) whose hues differ only because near-grey pixels have no reliable hue.
# Dye is saturated; contamination is not.
_SPLIT_MODE_S_MIN = 70.0
_SPLIT_MODE_V_MIN = 70.0
# And they must differ in HUE specifically, not merely in brightness — otherwise
# the sunlit and shaded halves of one shirt read as two colours.
_SPLIT_MIN_HUE_SEP = 25.0
_SPLIT_MIN_SEP = 55.0


def _hue_voters(px: np.ndarray) -> np.ndarray:
    """Pixels whose hue is trustworthy enough to vote on the kit colour."""
    px = np.asarray(px, dtype=np.float32).reshape(-1, 3)
    return px[(px[:, 1] >= _HUE_VOTE_S_MIN) & (px[:, 2] >= _HUE_VOTE_V_MIN)]


def _hue_angles(h: np.ndarray) -> np.ndarray:
    """OpenCV hue (0-179) → radians on the full circle."""
    return h.astype(np.float32) * (2.0 * math.pi / 180.0)


def circular_hue_mean(h: np.ndarray) -> float:
    """Mean hue as an angle, so red pixels either side of the 0/180 wrap average
    to red rather than to cyan. Not outlier-robust — see
    :func:`circular_hue_median`, which is what the kit reads use."""
    ang = _hue_angles(np.asarray(h))
    c, s = float(np.cos(ang).mean()), float(np.sin(ang).mean())
    if abs(c) < 1e-9 and abs(s) < 1e-9:
        return float(np.median(h))
    a = math.atan2(s, c)
    if a < 0:
        a += 2.0 * math.pi
    return a * (180.0 / (2.0 * math.pi))


def circular_hue_median(h: np.ndarray) -> float:
    """Median hue as an angle: correct across the 0/180 wrap *and* robust to
    outliers, which the circular mean is not.

    Rotates the samples so the circular mean sits at zero, unwraps the deviations
    onto a line, takes an ordinary median there, and rotates back. A handful of
    wrong-coloured pixels (a limb, a sliver of another player, a bad frame) then
    moves the answer as little as it would in a linear median — the property the
    kit reads have always depended on.
    """
    h = np.asarray(h, dtype=np.float32).reshape(-1)
    if h.size == 0:
        return 0.0
    centre = circular_hue_mean(h) * (2.0 * math.pi / 180.0)
    dev = _hue_angles(h) - centre
    dev = (dev + math.pi) % (2.0 * math.pi) - math.pi     # wrap to (-pi, pi]
    med = centre + float(np.median(dev))
    med %= 2.0 * math.pi
    return med * (180.0 / (2.0 * math.pi))


def circular_hue_std(h: np.ndarray) -> float:
    """Circular standard deviation of hue, in OpenCV hue units.

    Uses the resultant-length form ``sqrt(-2 ln R)``: R near 1 means the pixels
    point the same way (one colour, small spread), R near 0 means they cancel out
    (opposed colours, large spread). Unlike a plain std this is correct across
    the 0/180 wrap.
    """
    ang = _hue_angles(np.asarray(h))
    if ang.size == 0:
        return 0.0
    c, s = float(np.cos(ang).mean()), float(np.sin(ang).mean())
    r = math.hypot(c, s)
    if r >= 1.0:
        return 0.0
    if r <= 1e-6:
        return 180.0
    return math.sqrt(-2.0 * math.log(r)) * (180.0 / (2.0 * math.pi))


def robust_hsv(px: np.ndarray) -> np.ndarray | None:
    """One representative HSV for a pixel set, with hue averaged circularly."""
    px = np.asarray(px, dtype=np.float32).reshape(-1, 3)
    if px.shape[0] == 0:
        return None
    voters = _hue_voters(px)
    hue_src = voters if voters.shape[0] >= 4 else px
    return np.array(
        [circular_hue_median(hue_src[:, 0]),
         float(np.median(px[:, 1])),
         float(np.median(px[:, 2]))],
        dtype=np.float32,
    )


def hue_spread(px: np.ndarray) -> float:
    """Circular hue spread of a pixel set, counting only saturated pixels."""
    px = np.asarray(px, dtype=np.float32).reshape(-1, 3)
    voters = _hue_voters(px)
    if voters.shape[0] < 8:
        return 0.0          # too few opinions to call it multi-coloured
    return circular_hue_std(voters[:, 0])


def split_hsv_pixels(
    px: np.ndarray,
    *,
    max_colors: int = 2,
    hue_std_limit: float = _SPLIT_HUE_STD,
    min_share: float = _SPLIT_MIN_SHARE,
    min_sep: float = _SPLIT_MIN_SEP,
) -> list[np.ndarray]:
    """Split a set of torso pixels into the colours the shirt actually contains.

    Returns one colour for a flat kit and the separated modes for a genuinely
    multi-coloured one — largest first. This is the fix for a two-colour shirt
    averaging to a colour it does not contain.

    There are two bars, not one. Above ``_MODE_TRUST_HUE_STD`` the pixels are no
    longer one tight colour, so the single answer stops being the median of
    everything and becomes the **dominant mode** — otherwise a rejected split
    still reports the between-the-halves colour we are trying to eliminate.
    Above ``hue_std_limit``, and only if the modes then look like real dye, both
    colours are reported.

    The spread test is just the trigger; the split itself is a small circular
    k-means, because spread says "not one colour" without saying which colours.
    """
    px = np.asarray(px, dtype=np.float32).reshape(-1, 3)
    if px.shape[0] == 0:
        return []
    single = robust_hsv(px)
    if single is None:
        return []
    spread = hue_spread(px)
    if max_colors < 2 or spread <= _MODE_TRUST_HUE_STD:
        return [single]
    report_both = spread > hue_std_limit

    voters = _hue_voters(px)
    if voters.shape[0] < _SPLIT_MIN_VOTERS:
        return [single]

    # Circular k-means on hue, seeded at the two most-opposed directions present.
    ang = _hue_angles(voters[:, 0])
    unit = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    mean_dir = unit.mean(axis=0)
    n = np.linalg.norm(mean_dir)
    # Seed away from the resultant: with two opposed modes the resultant sits
    # between them, so its perpendicular separates them cleanly.
    axis = (np.array([-mean_dir[1], mean_dir[0]]) / n if n > 1e-6
            else np.array([1.0, 0.0]))
    proj = unit @ axis
    cent = [unit[proj <= 0].mean(axis=0) if (proj <= 0).any() else unit[0],
            unit[proj > 0].mean(axis=0) if (proj > 0).any() else unit[-1]]
    labels = np.zeros(unit.shape[0], dtype=np.int32)
    for _ in range(12):
        d0 = unit @ cent[0]
        d1 = unit @ cent[1]
        new = (d1 > d0).astype(np.int32)
        if np.array_equal(new, labels):
            break
        labels = new
        for gi in (0, 1):
            m = labels == gi
            if m.any():
                v = unit[m].mean(axis=0)
                if np.linalg.norm(v) > 1e-6:
                    cent[gi] = v

    groups = [voters[labels == gi] for gi in (0, 1)]
    groups = [g for g in groups if g.shape[0] > 0]
    if len(groups) < 2:
        return [single]
    groups.sort(key=lambda g: -g.shape[0])
    colors = [robust_hsv(g) for g in groups]
    colors = [c for c in colors if c is not None]
    if len(colors) < 2:
        return [single]

    # From here the crop is known not to be one tight colour, so the fallback
    # is the dominant mode — never ``single``, which is the median across both
    # modes and therefore a colour the shirt does not contain.
    dominant = [colors[0]]
    if not report_both:
        return dominant
    if groups[1].shape[0] / float(voters.shape[0]) < min_share:
        return dominant
    # Both modes must look like dye, not like contamination: saturated enough
    # to have a hue at all, and bright enough for that hue to be real.
    if min(float(c[1]) for c in colors) < _SPLIT_MODE_S_MIN:
        return dominant
    if min(float(c[2]) for c in colors) < _SPLIT_MODE_V_MIN:
        return dominant
    # ...and differ in hue, not just in how much light fell on them.
    dh = abs(float(colors[0][0]) - float(colors[1][0]))
    dh = min(dh, 180.0 - dh)
    if dh < _SPLIT_MIN_HUE_SEP:
        return dominant
    d = hsv_distance(colors[0], colors[1])
    if d is None or d < min_sep:
        return dominant
    return colors[:max_colors]


def jersey_colors_from_crop(
    crop: np.ndarray | None,
    *,
    min_keep_frac: float = 0.12,
    max_colors: int = 2,
) -> list[np.ndarray]:
    """Grass-masked kit colours of a torso crop — one entry for a flat kit, more
    for a genuinely multi-coloured one. Empty when the crop is unreadable."""
    px = _non_grass_pixels(crop, min_keep_frac=min_keep_frac)
    if px is None:
        return []
    return split_hsv_pixels(px, max_colors=max_colors)


def jersey_colors(
    frame: np.ndarray,
    bbox: list[float],
    *,
    min_area: int = 0,
    min_keep_frac: float = 0.12,
    max_colors: int = 2,
) -> list[np.ndarray]:
    """:func:`jersey_colors_from_crop` for a player box on a frame."""
    crop = torso_crop(frame, bbox)
    if crop is None or crop.size == 0:
        return []
    if min_area > 0 and (crop.shape[0] * crop.shape[1]) < min_area:
        return []
    return jersey_colors_from_crop(
        crop, min_keep_frac=min_keep_frac, max_colors=max_colors,
    )


def hex_to_hsv(hex_color: str | None) -> np.ndarray | None:
    """Convert a '#rrggbb' colour string (as picked in the team-colour UI) to
    OpenCV HSV (H 0-179, S/V 0-255). None for anything unparsable."""
    if not hex_color:
        return None
    s = hex_color.lstrip("#")
    if len(s) != 6:
        return None
    try:
        r, g, b = int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
    except ValueError:
        return None
    bgr = np.uint8([[[b, g, r]]])
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0, 0]
    return hsv.astype(np.float32)


def median_hsv(features: list[np.ndarray]) -> np.ndarray | None:
    """Robust average across several HSV samples (drops per-frame noise).

    Hue is combined circularly for the same reason as :func:`robust_hsv` — a red
    kit sampled either side of the 0/180 wrap must average to red, not cyan.
    """
    if not features:
        return None
    return robust_hsv(np.stack(features, axis=0))


def cluster_hsv(
    features: list[np.ndarray],
    *,
    max_colors: int = 3,
    min_sep: float = 55.0,
    min_share: float = 0.18,
) -> list[np.ndarray]:
    """Split HSV samples into the distinct colours a *single kit* is made of.

    A red/blue halved shirt, hoops, or a strongly contrasting sleeve gives two
    genuine colour modes; ``median_hsv`` collapses them into a colour the kit
    does not contain (the classic red+blue → purple). This keeps the modes
    instead, ordered by how many samples back them.

    Greedy nearest-mode assignment with running means (same shape as
    ``grouping._greedy_cluster``, kept local so ``color`` has no upward
    dependency). ``min_sep`` is the hue-weighted distance below which two modes
    are the same colour; ``min_share`` drops modes backed by too few samples to
    be a real second kit colour rather than a bad crop. Returns ``[]`` for no
    input, and always at least one colour otherwise.
    """
    feats = [np.asarray(f, dtype=np.float32) for f in features if f is not None]
    if not feats:
        return []

    centroids: list[np.ndarray] = []
    members: list[list[np.ndarray]] = []
    for f in feats:
        best, best_d = -1, float("inf")
        for i, c in enumerate(centroids):
            d = hsv_distance(f, c)
            if d is not None and d < best_d:
                best_d, best = d, i
        if best >= 0 and best_d <= min_sep:
            members[best].append(f)
            centroids[best] = np.mean(np.stack(members[best]), axis=0).astype(np.float32)
        elif len(centroids) < max_colors:
            centroids.append(f.copy())
            members.append([f])
        else:
            # At capacity: fold into the nearest mode rather than discard.
            if best >= 0:
                members[best].append(f)
                centroids[best] = np.mean(np.stack(members[best]), axis=0).astype(np.float32)

    order = sorted(range(len(centroids)), key=lambda i: -len(members[i]))
    total = float(len(feats))
    out = [centroids[order[0]]]                      # primary always survives
    for i in order[1:]:
        if len(members[i]) / total >= min_share:
            out.append(centroids[i])
    return out


def hsv_distance_multi(
    kit: np.ndarray | None,
    refs: list[np.ndarray] | np.ndarray | None,
    **kw,
) -> float | None:
    """Distance from ``kit`` to the *nearest* of a team's colours.

    A multi-coloured kit matches on any one of its colours, so the team a torso
    belongs to is decided by its closest colour, never by an average of them.
    Accepts a single HSV array for ``refs`` so existing single-colour call sites
    keep working. None when nothing is measurable.
    """
    if kit is None or refs is None:
        return None
    if isinstance(refs, np.ndarray) and refs.ndim == 1:
        refs = [refs]
    best: float | None = None
    for r in refs:
        d = hsv_distance(kit, r, **kw)
        if d is not None and (best is None or d < best):
            best = d
    return best


def kits_separable(
    a: list[np.ndarray] | np.ndarray | None,
    b: list[np.ndarray] | np.ndarray | None,
    max_dist: float,
) -> bool:
    """True when two kits' colour sets are far enough apart to tell apart.

    Uses the *closest* cross pair: if either team wears a colour the other also
    wears, colour cannot separate them and the two-centroid logic must not
    guess — the same rule as the single-colour version, applied to sets.
    """
    if a is None or b is None:
        return False
    if isinstance(a, np.ndarray) and a.ndim == 1:
        a = [a]
    if isinstance(b, np.ndarray) and b.ndim == 1:
        b = [b]
    closest: float | None = None
    for x in a:
        d = hsv_distance_multi(x, list(b))
        if d is not None and (closest is None or d < closest):
            closest = d
    return closest is not None and closest > max_dist


def hexes_to_hsv(hexes) -> list[np.ndarray]:
    """Parse a list of '#rrggbb' strings to OpenCV HSV, dropping unparsable ones."""
    out: list[np.ndarray] = []
    for h in (hexes or []):
        v = hex_to_hsv(h)
        if v is not None:
            out.append(v)
    return out


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
