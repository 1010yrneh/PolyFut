"""Quick kit-colour preview for the team picker (before full pipeline run)."""

from __future__ import annotations

import cv2
import numpy as np

from polyfut_video.pipeline.decode import iter_frames, probe_video
from polyfut_video.pipeline.detection import DetectConfig, Detector
from polyfut_video.pipeline.team_classify import (
    _hsv_feature, _is_second_kit_colour, _jersey_bgr_median, _jersey_bgr_pixels,
    _jersey_hsv_feature, _torso_crop, looks_like_pitch, pitch_palette,
    pitch_reference_lab, standout_colours, standout_kit_lab,
)


def _resize_frame(frame: np.ndarray, target_width: int = 960) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= target_width:
        return frame
    scale = target_width / float(w)
    return cv2.resize(frame, (target_width, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)


def _bgr_to_hex(bgr) -> str:
    b, g, r = (int(np.clip(x, 0, 255)) for x in bgr)
    return f"#{r:02x}{g:02x}{b:02x}"


def _crops_to_hex(crops: list[np.ndarray], pitch_lab=None, palette=None) -> str | None:
    """Swatch colour from a group of torso crops, excluding pitch-grass pixels
    so a partly-grass crop doesn't wash the swatch toward green (a player is
    only ~15-20px tall on wide/low-res broadcast footage, so the torso crop
    often includes some grass bleed even after the mostly-grass ones are
    filtered out upstream)."""
    hexes = _crops_to_hexes(crops, max_colors=1, pitch_lab=pitch_lab,
                            palette=palette)
    return hexes[0] if hexes else None


_MAX_KIT_PIXELS = 20000     # subsample cap — k-means cost, not accuracy, driver
# Gates a candidate cluster must clear to count as a *separate kit colour*
# rather than the same kit under different light. Mirrors the checks in
# polyfut_v2.pipeline.color.split_hsv_pixels (kept local — polyfut_video must not
# depend on polyfut_v2, which is built on top of it).
_KIT_MODE_S_MIN = 70        # dye is saturated; grass bleed and shadow are not
_KIT_MODE_V_MIN = 70        # near-black pixels compute a high S but a junk hue
_KIT_MIN_HUE_SEP = 25       # OpenCV hue units — a real colour difference


def _bgr_centers_are_distinct_kits(centers) -> bool:
    """Do these cluster centres differ in COLOUR, not merely in brightness?

    Euclidean distance in BGR happily separates the sunlit half of a shirt from
    its shaded half — measured on real clips, an unguarded split returned
    ['#4c3d27', '#9f8650'], which is one olive kit at two exposures.

    The test is :func:`_is_second_kit_colour`, whose thresholds were measured
    against real sun/shade pairs and real two-colour kits. It replaced a
    saturation-and-hue gate that demanded S>=70 and V>=70 of *both* centres and
    so could never pass a white shirt (white is the least saturated thing on the
    field) nor a navy one (navy fails on value) — the two commonest kit colours
    in this footage. Whenever it failed, the caller fell back to a single
    averaged swatch, which is part of how kits came out brown.
    """
    arr = np.asarray(centers, np.float32).reshape(-1, 3)
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if not _is_second_kit_colour(arr[i], arr[j]):
                return False
    return True


def _dominant_reading(px: np.ndarray, criteria) -> np.ndarray:
    """Median of the largest group of eye-dropper readings.

    A plain median over readings from a two-coloured kit lands between its two
    colours — a colour nobody wears. Splitting first and taking the bigger side
    keeps the swatch real.
    """
    if px.shape[0] < 8:
        return np.median(px, axis=0)
    try:
        _c, labels, _cen = cv2.kmeans(px, 2, None, criteria, 3,
                                      cv2.KMEANS_PP_CENTERS)
    except cv2.error:
        return np.median(px, axis=0)
    labels = labels.flatten()
    counts = np.bincount(labels, minlength=2)
    sel = px[labels == int(np.argmax(counts))]
    return np.median(sel if sel.shape[0] >= 4 else px, axis=0)


def _crops_to_hexes(
    crops: list[np.ndarray],
    *,
    max_colors: int = 3,
    min_sep_bgr: float = 60.0,
    min_share: float = 0.18,
    pitch_lab=None,
    palette=None,
) -> list[str]:
    """Swatch colours for one kit, keeping the colours it actually contains.

    Built from **eye-dropper readings**: one or two standout colours sampled per
    crop (see :func:`standout_colours`), then clustered. Nothing here ever
    averages raw pixels. Every torso crop carries shorts, skin, shadow and
    whatever the box overshot onto, so any statistic over all of it — mean,
    median, biggest cluster — drifts toward a colour nobody is wearing, which is
    how both teams' swatches came out brown.

    Because each reading is already a real colour off a real shirt, the
    single-colour answer can be their median without inventing anything.

    Clustering is in BGR, where — unlike hue — there is no 0/180 wrap to trip
    over, so a red kit clusters as red rather than splitting across the seam.
    ``k`` grows only while each new cluster is at least ``min_sep_bgr`` from the
    others and holds at least ``min_share`` of the readings, so shading and noise
    cannot manufacture a second kit colour. Returned dominant-first.
    """
    if not crops:
        return []
    # One eye-dropper reading per crop (two if the kit really has two colours),
    # never a pool of raw pixels. Pooling mixes in shorts, skin, shadow and
    # background from every crop, and any average over that lands on a colour
    # nobody is wearing — which is how both teams came out brown.
    readings: list[np.ndarray] = []
    for c in crops:
        readings.extend(standout_colours(
            c, pitch_lab, max_colours=(1 if max_colors <= 1 else 2),
            palette=palette))
    if not readings:
        return []
    px = np.stack(readings, axis=0).astype(np.float32)
    if px.shape[0] < 3:
        return [_bgr_to_hex(np.median(px, axis=0))]

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    # The single-colour answer is the median of the BIGGEST GROUP of readings,
    # not of all of them. Each reading is already a real colour off a real shirt,
    # but a median taken across a genuinely two-coloured kit still lands between
    # the two — the exact "average of everything" failure this function exists to
    # avoid, reintroduced at the last step. Taking the dominant group keeps the
    # answer a colour somebody is actually wearing.
    best: list[str] = [_bgr_to_hex(_dominant_reading(px, criteria))]
    if max_colors <= 1 or px.shape[0] < 8:
        return best

    for k in range(2, max_colors + 1):
        if px.shape[0] < k * 4:
            break
        _c, labels, centers = cv2.kmeans(
            px, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        labels = labels.flatten()
        counts = np.bincount(labels, minlength=k).astype(float)
        shares = counts / counts.sum()
        if shares.min() < min_share:
            break
        pairs = [float(np.linalg.norm(centers[i] - centers[j]))
                 for i in range(k) for j in range(i + 1, k)]
        if min(pairs) < min_sep_bgr:
            break
        if not _bgr_centers_are_distinct_kits(centers):
            break
        order = np.argsort(-counts)
        best = [_bgr_to_hex(np.median(px[labels == i], axis=0)) for i in order]
    return best


def _is_referee_crop(crop: np.ndarray) -> bool:
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = (int(x) for x in np.median(hsv.reshape(-1, 3), axis=0))
    return (h <= 12 or h >= 168) and s >= 70 and v >= 50


# A crop with no colour in it is only uninformative if it is also mid-toned.
# The previous test was `s < 30 or (v > 210 and s < 50)`, which threw away every
# unsaturated crop — and white is the least saturated thing on a pitch, so it
# discarded an entire white-kitted team, plus black kits (also near-zero
# saturation) and pale blues. Two of the commonest kit colours in football.
#
# It barely showed on the current footage because turf contamination keeps
# measured saturation high, which made it a latent bug that would get WORSE as
# footage improved: the better the exposure, the more cleanly a white shirt
# reads as white, and the more reliably it was dropped.
#
# White is bright, black is dark, and neither is grey. What genuinely carries no
# kit signal is the middle: worn concrete, deep shadow, a washed-out blur.
_NEUTRAL_S_MAX = 30     # below this a crop has no usable hue
_NEUTRAL_V_LO = 55      # darker than this is a black/navy kit, not "neutral"
_NEUTRAL_V_HI = 195     # brighter than this is a white kit, not "neutral"


def _is_neutral_crop(crop: np.ndarray) -> bool:
    """True when a crop carries no kit signal at all — not merely no colour."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    _h, s, v = (int(x) for x in np.median(hsv.reshape(-1, 3), axis=0))
    if s >= _NEUTRAL_S_MAX:
        return False                      # it has a colour; that is a kit
    # Colourless. Keep the extremes (white / black kits), drop the middle.
    # A light-grey kit does fall in the dropped band, but grey is genuinely
    # indistinguishable from worn concrete at this resolution and it is a far
    # rarer kit than white or black.
    return _NEUTRAL_V_LO < v < _NEUTRAL_V_HI


def _kmeans_two_kits(
    crops: list[np.ndarray],
    pitch_lab=None,
    palette=None,
) -> tuple[list[np.ndarray], list[np.ndarray]] | None:
    """Force k=2 kit clusters; do not drop the smaller team (preview-only)."""
    if len(crops) < 4:
        return None

    # Each crop is reduced to its STANDOUT colour in Lab — an eye-dropper
    # reading — not a hue-weighted median. Two things were wrong with the median
    # feature, and together they meant the two kits were never separated at all:
    # measured on real footage, k-means put 200 of 222 crops into one cluster
    # holding white, navy AND yellow shirts, so both swatches averaged to brown.
    #   * a median over a torso crop mixes shirt, shorts, skin, shadow and
    #     background into a colour nobody wears (see standout_colours);
    #   * hue cannot separate white from navy — an unsaturated pixel's hue is
    #     noise — yet hue carried double weight, so white shirts scattered.
    # Lab handles both axes: white vs navy differ on L, yellow vs blue on b*.
    feats = []
    usable = []
    for c in crops:
        f = standout_kit_lab(c, pitch_lab, palette)
        if f is None:
            continue
        feats.append(f)
        usable.append(c)
    if len(usable) < 4:
        return None
    scaled = np.stack(feats, axis=0).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _compactness, labels, _centers = cv2.kmeans(
        scaled, 2, None, criteria, 5, cv2.KMEANS_PP_CENTERS,
    )
    labels = labels.flatten()
    g0 = [usable[i] for i in range(len(usable)) if labels[i] == 0]
    g1 = [usable[i] for i in range(len(usable)) if labels[i] == 1]
    if not g0 or not g1:
        return None
    # Larger cluster first (cosmetic only)
    if len(g0) >= len(g1):
        return g0, g1
    return g1, g0


def detect_team_kits(
    video_path: str,
    *,
    weights: str = "yolov8n.pt",
    device: str = "cpu",
    n_samples: int = 20,
    # 640, not 960: kit colours only need to cluster players by shirt hue, and
    # detection at 960 is ~4x slower on CPU for no colour benefit on 360p footage
    # (real-frame check: 960 -> 5s vs 21s, near-identical swatches). At 960 the
    # detection could run past its subprocess timeout under CPU contention and
    # fall back to default red/white colours; 640 keeps it fast and robust.
    target_width: int = 640,
    imgsz: int = 640,
    sample_window_minutes: float = 8.0,
    sample_every_seconds: float = 3.0,
) -> list[dict] | None:
    """
    Sample frames, detect players, k-means torso colours → two kit swatches.
    Returns [{"id","label","hex"}, ...] or None if detection fails.

    Frames are sampled *sequentially* (grab/retrieve — no per-frame seeks) from a
    bounded early window. A cap.set() per frame re-decodes a whole GOP each on
    sparse-keyframe phone/broadcast video, which can stall for minutes; kit
    colours are constant, so an early window is sufficient and far faster.
    """
    info = probe_video(video_path)
    n_frames = int(info.get("frame_count") or 0)
    fps = float(info.get("fps") or 25.0)
    if n_frames < 1:
        return None

    stride = max(1, int(round(fps * sample_every_seconds)))
    max_frame = min(n_frames - 1, int(sample_window_minutes * 60 * fps))

    det = Detector(DetectConfig(
        weights=weights,
        device=device,
        conf_threshold=0.20,
        imgsz=imgsz,
    ))

    crops: list[np.ndarray] = []
    seen_frames: list[np.ndarray] = []
    sampled = 0
    pitch_lab = None
    palette = None
    for fidx, _t, frame in iter_frames(
        video_path, target_width=target_width, sample_every_n=stride,
    ):
        if fidx > max_frame or sampled >= n_samples:
            break
        # Measure this pitch's own colours before using them to mask anything.
        # The fixed grass hue band starts at 32; measured across the uploads on
        # disk, one venue's pitch sits at hue 28 and the band masks 1.2% of it,
        # so every crop stayed full of turf and both kits read as the ground.
        # Four frames, not more: the loop below can break out after six samples
        # once it has enough crops, so anything that waits for a bigger backlog
        # never runs at all — which silently disabled the pitch filter entirely.
        if len(seen_frames) < 4:
            seen_frames.append(frame)
            if len(seen_frames) == 4:
                palette = pitch_palette(seen_frames)
                pitch_lab = pitch_reference_lab(seen_frames)
        dets = det.detect_frame(frame)
        for d in dets:
            if d.get("class") != "player":
                continue
            crop = _torso_crop(frame, d["bbox"])
            if crop is None or _is_referee_crop(crop) or _is_neutral_crop(crop):
                continue
            # A player is only ~15-20px tall on wide/low-res broadcast footage,
            # so some "player" detections land almost entirely on grass (a
            # sliver of pitch mis-detected, or a distant player whose torso
            # crop is mostly background). Drop those here rather than let
            # their grass-green colour dilute the kit clusters below.
            if palette is None and _jersey_hsv_feature(crop) is None:
                continue
            # ...and drop the ones that are simply the pitch again. A box on a
            # 15px player often contains no player at all; those crops passed
            # the grass-fraction filter (dirt and shadow aren't grass-hued) and
            # then dominated the clustering with the colour of the ground.
            ref = palette if palette is not None else pitch_lab
            if ref is not None and looks_like_pitch(
                standout_kit_lab(crop, pitch_lab, palette), ref
            ):
                continue
            crops.append(crop)
        sampled += 1
        # Enough signal already — stop early (kit colours don't change).
        if sampled >= 6 and len(crops) >= 40:
            break

    if seen_frames and palette is None:
        palette = pitch_palette(seen_frames)
    if seen_frames and pitch_lab is None:
        pitch_lab = pitch_reference_lab(seen_frames)
    # Late-derived pitch reference still gets applied — crops collected before
    # it was known are filtered here rather than silently kept.
    ref = palette if palette is not None else pitch_lab
    if ref is not None:
        crops = [c for c in crops
                 if not looks_like_pitch(
                     standout_kit_lab(c, pitch_lab, palette), ref)]
    if len(crops) < 4:
        return None

    grouped = _kmeans_two_kits(crops, pitch_lab, palette)
    if grouped is None:
        return None

    team_a_crops, team_b_crops = grouped
    hexes_a = _crops_to_hexes(team_a_crops, pitch_lab=pitch_lab, palette=palette)
    hexes_b = _crops_to_hexes(team_b_crops, pitch_lab=pitch_lab, palette=palette)

    if not hexes_a or not hexes_b:
        return None
    # A kit colour claimed by both clusters says nothing about which team a
    # player is on, so drop the overlap from the smaller cluster rather than
    # offering the user two swatch sets that share a colour.
    hexes_b = [h for h in hexes_b if h not in hexes_a] or hexes_b
    if hexes_a[0] == hexes_b[0]:
        return None

    # ``hex`` stays the dominant colour so older clients / saved sessions keep
    # working; ``hexes`` carries the full set for multi-coloured kits.
    return [
        {"id": "team_a", "label": "Team A", "hex": hexes_a[0], "hexes": hexes_a,
         "sample_count": len(team_a_crops)},
        {"id": "team_b", "label": "Team B", "hex": hexes_b[0], "hexes": hexes_b,
         "sample_count": len(team_b_crops)},
    ]
