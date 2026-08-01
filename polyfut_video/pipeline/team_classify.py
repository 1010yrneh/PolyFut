"""Stage 7: DBSCAN jersey colour clustering per shot."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN

import cv2


def _torso_crop(frame: np.ndarray, bbox: list[float]) -> np.ndarray | None:
    x1, y1, x2, y2 = bbox
    h = max(y2 - y1, 1)
    w = max(x2 - x1, 1)
    ty1 = int(y1 + 0.15 * h)
    ty2 = int(y1 + 0.55 * h)
    tx1 = int(x1 + 0.2 * w)
    tx2 = int(x2 - 0.2 * w)
    if ty2 <= ty1 or tx2 <= tx1:
        return None
    crop = frame[ty1:ty2, tx1:tx2]
    return crop if crop.size > 0 else None


def _hsv_feature(crop: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    return np.median(hsv.reshape(-1, 3), axis=0).astype(np.float32)


# Pitch grass is a fairly tight green band in HSV. On distant/low-res footage a
# player's torso crop is mostly grass, which swamps the jersey colour and pulls
# team-colour reads toward green. Excluding grass first recovers the actual
# kit. Mirrors polyfut_v2.pipeline.color's (already-shipped, tested) constants
# and algorithm; kept local rather than imported so this legacy module doesn't
# gain a new cross-package dependency for a few lines of logic.
_GRASS_H_LO, _GRASS_H_HI = 32, 95   # OpenCV hue (0-179): yellow ~20-31, grass ~35-85
_GRASS_S_MIN = 40
_GRASS_V_MIN = 40


def _grass_mask(hsv: np.ndarray) -> np.ndarray:
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return ((h >= _GRASS_H_LO) & (h <= _GRASS_H_HI)
            & (s >= _GRASS_S_MIN) & (v >= _GRASS_V_MIN))


# --------------------------------------------------------------- pitch palette
#
# The band above is a GUESS about what turf looks like, and it is wrong often
# enough to matter. Measured across the uploads on disk, the pitch's median hue
# is 28 at one venue and 42-43 at the others; the band starts at 32, so it masks
# 1.2% of the first pitch and ~95% of the rest. At 1.2% the mask does nothing,
# every torso crop stays ~90% turf, and both kits read as the ground the players
# are standing on — which is exactly the "both swatches are brown" report.
#
# So measure the pitch instead of assuming it. In a wide match shot the pitch is
# by far the largest thing in the lower half of the frame, so its colours are
# simply the biggest colour clusters there. Several centres rather than one
# interval matters: sunlit turf and shaded turf are far apart, and a single band
# stretched to cover both is wide enough to swallow kits (a per-clip hue band was
# tried for this and reverted for exactly that reason). Separate tight centres
# cover both without ever widening.
#
# This is the same correction `polyfut_v2.pipeline.color.is_bbox_on_foreign_surface`
# already made for ball surfaces, where requiring positive proof of grass was
# measured rejecting 49% of on-pitch positions on sun-bleached turf.

_PITCH_K = 3               # sunlit turf, shaded turf, worn dirt
_PITCH_MIN_SHARE = 0.08    # a real surface, not a stray object
_PITCH_RADIUS = 26.0       # Lab distance still counted as "this is the pitch"
_PITCH_SAMPLE_PX = 4000    # per frame; we want the modes, not every pixel


def _lab_rows(px_bgr: np.ndarray) -> np.ndarray:
    """(N,3) BGR -> (N,3) Lab, float32."""
    a = np.clip(np.asarray(px_bgr, np.float32), 0, 255).astype(np.uint8)
    return cv2.cvtColor(a.reshape(1, -1, 3), cv2.COLOR_BGR2LAB)[0].astype(np.float32)


def pitch_palette(frames, *, k: int = _PITCH_K,
                  min_share: float = _PITCH_MIN_SHARE) -> np.ndarray | None:
    """The colours the pitch is made of, as Lab centres, dominant first.

    Sampled from the lower-middle of each frame, which in a wide match shot is
    overwhelmingly playing surface. Makes no assumption about hue, so a bleached
    olive pitch and a deep green one are handled identically.

    Returns (M,3) float32, or None when there is nothing to measure.
    """
    rng = np.random.default_rng(0)
    pool = []
    for f in frames or []:
        if f is None or f.size == 0:
            continue
        h, w = f.shape[:2]
        mid = f[int(h * 0.45):int(h * 0.95), int(w * 0.1):int(w * 0.9)]
        if mid.size == 0:
            continue
        px = mid.reshape(-1, 3)
        n = min(_PITCH_SAMPLE_PX, px.shape[0])
        pool.append(px[rng.choice(px.shape[0], size=n, replace=False)])
    if not pool:
        return None
    lab = _lab_rows(np.concatenate(pool, 0))
    kk = int(min(k, max(1, lab.shape[0] // 50)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 1.0)
    try:
        _c, labels, centres = cv2.kmeans(
            lab, kk, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    except cv2.error:
        return None
    labels = labels.flatten()
    counts = np.bincount(labels, minlength=kk).astype(float)
    shares = counts / max(counts.sum(), 1.0)
    order = np.argsort(-counts)
    keep = [centres[i] for i in order if shares[i] >= min_share]
    return np.stack(keep, 0).astype(np.float32) if keep else None


def _not_pitch_mask(crop: np.ndarray, palette: np.ndarray,
                    *, radius: float = _PITCH_RADIUS) -> np.ndarray:
    """Flat boolean mask of the crop's pixels that are NOT the measured pitch."""
    lab = _lab_rows(crop.reshape(-1, 3))
    d = np.linalg.norm(lab[:, None, :] - np.asarray(palette, np.float32)[None, :, :],
                       axis=2).min(axis=1)
    return d > float(radius)


def _jersey_hsv_feature(crop: np.ndarray, *,
                        min_keep_frac: float = 0.12) -> np.ndarray | None:
    """Dominant HSV colour of a torso crop, pitch grass excluded.

    The *dominant* colour, not the median. A torso crop at this resolution is
    never pure kit — it carries shorts, skin, shadow, background and whatever
    the box overshot onto — and a median across that mixture lands on a colour
    nobody is wearing. Measured on real footage, medians left white, navy and
    yellow shirts so close together that k-means put 200 of 222 crops in a
    single cluster and the two teams were never separated at all; both swatches
    then came out brown, because brown is the honest average of everything.

    Taking the biggest colour cluster inside the crop keeps a white shirt white
    and a navy shirt navy, which is what makes the two kits separable.

    None when the crop is empty or almost entirely grass (colour unmeasurable).
    """
    if crop is None or crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    keep = ~_grass_mask(hsv)
    if int(keep.sum()) < max(8, int(min_keep_frac * keep.size)):
        return None
    px = hsv.reshape(-1, 3)[keep.reshape(-1)].astype(np.float32)
    return _dominant_hsv(px)


def pitch_reference_lab(frames):
    """The pitch's own colour, in Lab — what a crop looks like with no player.

    A torso box on a 15px player frequently contains no player at all (a sliver
    of turf, a mis-detection, an overshooting box). Those crops survive a
    "fraction of non-grass pixels" filter because worn dirt, shadow and line
    paint are not inside the grass hue band, and they then dominate the kit
    clustering with what is really the pitch. Knowing the pitch's colour lets
    them be recognised and dropped.
    """
    labs = []
    for f in frames or []:
        if f is None or f.size == 0:
            continue
        h = f.shape[0]
        mid = f[int(h * 0.4):int(h * 0.9)]
        if mid.size == 0:
            continue
        hsv = cv2.cvtColor(mid, cv2.COLOR_BGR2HSV)
        m = _grass_mask(hsv)
        if m.sum() < 100:
            continue
        px = mid.reshape(-1, 3)[m.reshape(-1)]
        labs.append(np.median(px, axis=0))
    if not labs:
        return None
    bgr = np.median(np.stack(labs, 0), axis=0)
    lab = cv2.cvtColor(np.clip(bgr, 0, 255).reshape(1, 1, 3).astype(np.uint8),
                       cv2.COLOR_BGR2LAB)
    return lab.reshape(3).astype(np.float32)


def looks_like_pitch(crop_lab, pitch_lab, *, tol: float = 26.0) -> bool:
    """Is this crop's dominant colour just the pitch again?

    ``pitch_lab`` may be a single Lab colour or a (M,3) palette of them, in which
    case the nearest is used — sunlit and shaded turf are far apart, and a crop
    matching either one is still just the pitch.

    ``tol`` is a Lab distance. Deliberately modest: a real kit — even a drab one
    — sits further from turf than shading moves turf itself, and being too eager
    here would throw away genuinely olive-kitted players.
    """
    if crop_lab is None or pitch_lab is None:
        return False
    ref = np.asarray(pitch_lab, np.float32).reshape(-1, 3)
    if not len(ref):
        return False
    d = np.linalg.norm(np.asarray(crop_lab, np.float32).reshape(1, 3) - ref, axis=1)
    return float(d.min()) < tol


def dominant_kit_lab(crop: np.ndarray, *,
                     min_keep_frac: float = 0.12) -> np.ndarray | None:
    """Dominant kit colour of a torso crop, in CIE Lab. None if unreadable.

    Lab rather than hue-weighted HSV, because the two kits in a match may differ
    in *either* hue or lightness and one metric has to cover both. Hue cannot
    separate white from navy at all — an unsaturated pixel's hue is essentially
    noise — so hue-weighted clustering scattered white shirts randomly and left
    them mixed in with everything else. Lab puts white and navy far apart on L
    while still putting yellow and blue far apart on b*.
    """
    px = _jersey_bgr_pixels(crop, min_keep_frac=min_keep_frac)
    if px is None:
        return None
    dom = _dominant_bgr(px)
    lab = cv2.cvtColor(
        np.clip(dom, 0, 255).reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB)
    return lab.reshape(3).astype(np.float32)


def _lab_of(bgr) -> np.ndarray:
    return cv2.cvtColor(
        np.clip(np.asarray(bgr, np.float32), 0, 255).reshape(1, 1, 3).astype(np.uint8),
        cv2.COLOR_BGR2LAB).reshape(3).astype(np.float32)


def standout_colours(crop: np.ndarray, pitch_lab=None, *,
                     max_colours: int = 2, min_share: float = 0.12,
                     min_keep_frac: float = 0.12,
                     palette: np.ndarray | None = None) -> list[np.ndarray]:
    """Eye-dropper: the colours that STAND OUT in a torso crop, dominant-first.

    The most *common* colour in a torso crop is very often not the shirt — it is
    shadow, shorts, skin, or whatever the box overshot onto — so picking by size
    lands on something drab, and averaging lands on something nobody wears. What
    identifies a kit is being *distinctive*: far from the pitch the player is
    standing on.

    Distance from the pitch rather than saturation, because saturation cannot
    see a white kit at all (white is the least saturated thing on the field) yet
    white is about as far from green turf as a colour can get. Support is
    weighted in as a square root so a big drab region cannot be beaten by a
    handful of bright pixels on a line marking, while a genuine minority colour
    (a sleeve, a hoop) still competes.

    Returns [] when the crop is unreadable.
    """
    px = _jersey_bgr_pixels(crop, min_keep_frac=min_keep_frac, palette=palette)
    if px is None or px.shape[0] < 12:
        return []
    # Distance is measured to the NEAREST pitch colour, not to a single average
    # of them: sunlit and shaded turf are far apart, and averaging them puts the
    # reference in between, where neither actually is — which makes real turf
    # look distinctive and score as a kit.
    if palette is not None and len(palette):
        ref = np.asarray(palette, np.float32)
    elif pitch_lab is not None:
        ref = np.asarray(pitch_lab, np.float32).reshape(1, 3)
    else:
        ref = _lab_of(np.median(px, axis=0)).reshape(1, 3)

    k = 3 if px.shape[0] >= 60 else 2
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    try:
        _c, labels, centers = cv2.kmeans(
            px.astype(np.float32), k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    except cv2.error:
        return [np.median(px, axis=0).astype(np.float32)]
    labels = labels.flatten()
    counts = np.bincount(labels, minlength=k).astype(float)
    shares = counts / max(counts.sum(), 1.0)

    scored = []
    for i in range(k):
        if shares[i] < min_share:
            continue
        sel = px[labels == i]
        if sel.shape[0] < 8:
            continue
        med = np.median(sel, axis=0).astype(np.float32)
        dist = float(np.linalg.norm(_lab_of(med)[None, :] - ref, axis=1).min())
        scored.append((dist * float(np.sqrt(shares[i])), med))
    if not scored:
        return [np.median(px, axis=0).astype(np.float32)]
    scored.sort(key=lambda t: -t[0])
    out = [scored[0][1]]
    for _s, c in scored[1:]:
        if len(out) >= max_colours:
            break
        if all(_is_second_kit_colour(c, o) for o in out):
            out.append(c)
    return out


# Measured on synthetic kits, Lab deltas between the two readings:
#
#   blue shirt, sun vs shade      dL  49   chroma  41
#   red shirt,  sun vs shade      dL  63   chroma  33
#   yellow vs blue (real pair)    dL 109   chroma 152
#   white vs navy  (real pair)    dL 207   chroma  29
#
# So neither axis alone works. Chroma separates yellow/blue from lighting
# comfortably but is blind to white-vs-black, whose whole difference is
# lightness; lightness alone cannot tell a white-and-black kit from a shirt half
# in shadow except by sheer size. Requiring either a clear chroma change or an
# extreme lightness change covers both without letting shading through.
_SECOND_COLOUR_CHROMA = 60.0
_SECOND_COLOUR_LIGHTNESS = 120.0


def _is_second_kit_colour(a, b) -> bool:
    """Are these two readings different KIT COLOURS, not one shirt lit two ways?"""
    la, lb = _lab_of(a), _lab_of(b)
    chroma = float(np.hypot(la[1] - lb[1], la[2] - lb[2]))
    lightness = abs(float(la[0] - lb[0]))
    return chroma > _SECOND_COLOUR_CHROMA or lightness > _SECOND_COLOUR_LIGHTNESS


def standout_kit_lab(crop: np.ndarray, pitch_lab=None,
                     palette: np.ndarray | None = None):
    """Lab of the crop's single most distinctive colour, or None."""
    got = standout_colours(crop, pitch_lab, max_colours=1, palette=palette)
    return _lab_of(got[0]) if got else None


def dominant_cluster_pixels(px: np.ndarray) -> np.ndarray:
    """The pixels of the largest colour cluster in ``px`` (BGR rows).

    Keeping the cluster rather than collapsing it to one value lets callers pool
    several crops and still find a second kit colour, while leaving behind the
    skin, shorts, shadow and background that share every torso crop.
    """
    if px.shape[0] < 16:
        return px
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0)
    try:
        _c, labels, _centers = cv2.kmeans(
            px.astype(np.float32), 2, None, criteria, 2, cv2.KMEANS_PP_CENTERS)
    except cv2.error:
        return px
    labels = labels.flatten()
    counts = np.bincount(labels, minlength=2)
    sel = px[labels == int(np.argmax(counts))]
    return sel if sel.shape[0] >= 8 else px


def _dominant_bgr(px: np.ndarray) -> np.ndarray:
    """Median of the largest colour cluster among these BGR pixels."""
    sel = dominant_cluster_pixels(px)
    return np.median(sel, axis=0).astype(np.float32)


def _dominant_hsv(px: np.ndarray) -> np.ndarray:
    """Median of the largest colour cluster in ``px`` (HSV rows).

    Clustering is done in BGR-like Cartesian space on hue *unwrapped* around its
    circular mean, so a red kit straddling the 0/180 seam does not split into
    two clusters at opposite ends of the scale.
    """
    if px.shape[0] < 16:
        return np.median(px, axis=0).astype(np.float32)
    h = px[:, 0] * 2.0 * np.pi / 180.0
    mean_ang = np.arctan2(np.sin(h).mean(), np.cos(h).mean())
    unwrapped = ((px[:, 0] - np.degrees(mean_ang) / 2.0 + 90.0) % 180.0)
    feat = np.stack([unwrapped * 2.0, px[:, 1], px[:, 2]], axis=1).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0)
    try:
        _c, labels, _centers = cv2.kmeans(
            feat, 2, None, criteria, 2, cv2.KMEANS_PP_CENTERS)
    except cv2.error:
        return np.median(px, axis=0).astype(np.float32)
    labels = labels.flatten()
    counts = np.bincount(labels, minlength=2)
    sel = px[labels == int(np.argmax(counts))]
    if sel.shape[0] < 8:
        return np.median(px, axis=0).astype(np.float32)
    # circular median for hue, plain median for S and V
    hh = sel[:, 0] * 2.0 * np.pi / 180.0
    ang = np.arctan2(np.sin(hh).mean(), np.cos(hh).mean())
    centre = (np.degrees(ang) / 2.0) % 180.0
    unw = ((sel[:, 0] - centre + 90.0) % 180.0)
    hue = float((np.median(unw) + centre - 90.0) % 180.0)
    return np.array([hue, float(np.median(sel[:, 1])),
                     float(np.median(sel[:, 2]))], dtype=np.float32)


def _jersey_bgr_median(crop: np.ndarray, *, min_keep_frac: float = 0.12) -> np.ndarray | None:
    """Median BGR over the same non-grass pixel mask as ``_jersey_hsv_feature``,
    for producing a display-ready kit-colour swatch rather than a clustering
    feature."""
    px = _jersey_bgr_pixels(crop, min_keep_frac=min_keep_frac)
    if px is None:
        return None
    return np.median(px, axis=0).astype(np.float32)


def _jersey_bgr_pixels(crop: np.ndarray, *,
                       min_keep_frac: float = 0.12,
                       palette: np.ndarray | None = None) -> np.ndarray | None:
    """The (N,3) non-pitch BGR pixels of a torso crop, or None when the crop is
    empty or too nearly all pitch to trust.

    ``palette`` is this clip's measured pitch colours (see :func:`pitch_palette`)
    and is strongly preferred. Without one the fixed grass hue band is used, which
    works on a typical green pitch but masks almost nothing on a bleached olive
    one — the caller should supply a palette wherever it can.

    Returning the pixels rather than their median lets a caller pool several
    crops and find the colours a kit actually *contains*. A median cannot: a
    shirt that is half one colour and half another medians to a colour it does
    not have (blue + yellow reads as a muddy olive), and no amount of averaging
    across crops recovers the two halves once each crop has been flattened.

    """
    if crop is None or crop.size == 0:
        return None
    if palette is not None and len(palette):
        keep = _not_pitch_mask(crop, palette)
    else:
        keep = (~_grass_mask(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV))).reshape(-1)
    if int(keep.sum()) < max(8, int(min_keep_frac * keep.size)):
        return None
    return crop.reshape(-1, 3)[keep].astype(np.float32)


def classify_teams(
    player_crops: list[np.ndarray],
    *,
    eps: float = 18.0,
    min_samples: int = 3,
    min_cluster_size: int = 4,
) -> list[int]:
    """
    Returns cluster label per player crop.
    Two largest clusters map to 0 (team_a) and 1 (team_b).
    Outliers (-1) and small clusters are -1 (unassigned).
    """
    if not player_crops:
        return []

    feats = np.stack([_hsv_feature(c) for c in player_crops], axis=0)
    # Weight hue more for clustering
    scaled = feats.copy()
    scaled[:, 0] *= 2.0

    if len(player_crops) < min_samples:
        return [-1] * len(player_crops)

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(scaled)

    # Count cluster sizes (exclude noise -1)
    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    if len(unique) == 0:
        return [-1] * len(player_crops)

    order = np.argsort(-counts)
    top_clusters = unique[order[:2]]
    cluster_to_team = {int(top_clusters[0]): 0}
    if len(top_clusters) > 1:
        cluster_to_team[int(top_clusters[1])] = 1

    out: list[int] = []
    for lab in labels:
        if lab < 0:
            out.append(-1)
            continue
        cnt = int(counts[list(unique).index(lab)]) if lab in unique else 0
        if cnt < min_cluster_size:
            out.append(-1)
        elif lab in cluster_to_team:
            out.append(cluster_to_team[int(lab)])
        else:
            out.append(-1)
    return out


def assign_teams_to_tracked_frame(
    frame: np.ndarray,
    tracked_dets: list[dict],
    team_labels: dict[int, int],
) -> list[dict]:
    """Attach team_id (0=team_a, 1=team_b, -1=unassigned) to player detections."""
    out = []
    for det in tracked_dets:
        if det.get("class") != "player":
            out.append(det)
            continue
        tid = det.get("track_id", -1)
        team_id = team_labels.get(tid, -1)
        out.append({**det, "team_id": team_id})
    return out


def build_team_labels_for_shot(
    frames: list[np.ndarray],
    tracked_per_frame: list[list[dict]],
    *,
    eps: float = 18.0,
    min_samples: int = 3,
    min_cluster_size: int = 4,
) -> dict[int, int]:
    """
    Cluster all player torso crops in a shot; return track_id -> team_id (0/1/-1).
    """
    crops: list[np.ndarray] = []
    track_ids: list[int] = []
    seen: set[int] = set()

    for frame, dets in zip(frames, tracked_per_frame):
        for det in dets:
            if det.get("class") != "player":
                continue
            tid = int(det.get("track_id", -1))
            if tid < 0 or tid in seen:
                continue
            crop = _torso_crop(frame, det["bbox"])
            if crop is None:
                continue
            crops.append(crop)
            track_ids.append(tid)
            seen.add(tid)

    if not crops:
        return {}

    cluster_labels = classify_teams(
        crops, eps=eps, min_samples=min_samples, min_cluster_size=min_cluster_size,
    )
    return {tid: lab for tid, lab in zip(track_ids, cluster_labels)}


class TeamCropAccumulator:
    """Collect player torso crops across chunks within one broadcast shot."""

    def __init__(self, max_crops_per_track: int = 3):
        self._max = max(1, max_crops_per_track)
        self._crops: dict[int, list[np.ndarray]] = {}

    def reset(self) -> None:
        self._crops.clear()

    def observe(self, frame: np.ndarray, tracked_dets: list[dict]) -> None:
        for det in tracked_dets:
            if det.get("class") != "player":
                continue
            tid = int(det.get("track_id", -1))
            if tid < 0:
                continue
            bucket = self._crops.setdefault(tid, [])
            if len(bucket) >= self._max:
                continue
            crop = _torso_crop(frame, det["bbox"])
            if crop is not None:
                bucket.append(crop)

    def team_labels(
        self,
        *,
        eps: float = 18.0,
        min_samples: int = 3,
        min_cluster_size: int = 4,
    ) -> dict[int, int]:
        """Cluster using the latest crop per track (stable IDs across chunks)."""
        if not self._crops:
            return {}
        crops: list[np.ndarray] = []
        track_ids: list[int] = []
        for tid, clist in self._crops.items():
            if not clist:
                continue
            crops.append(clist[-1])
            track_ids.append(tid)
        cluster_labels = classify_teams(
            crops,
            eps=eps,
            min_samples=min_samples,
            min_cluster_size=min_cluster_size,
        )
        return {tid: lab for tid, lab in zip(track_ids, cluster_labels)}
