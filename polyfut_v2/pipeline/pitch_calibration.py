"""Stage 5b: where the pitch is, so "who touched it" can be judged in metres.

The attribution gate (`contact_max_player_dist_px`) is a pixel distance, but a
pixel is not a fixed distance on the ground: near the camera one pixel is ~11cm,
at the far touchline ~62cm. So the same 80px rule allows 7.9m of pitch near the
camera and 49.5m far away, and a player 50m from the ball can win "nearest
player". Measured on a real export, 36% of attributed touches had the player more
than 5m from the ball and 10% more than 10m. See ``docs/detection-issues.md``
Issue 14.

Fixing that needs a map from image pixels to pitch metres. This module fits one
from landmarks the user clicks.

**The anchors win.** The transform actually used is fitted straight to the
landmarks the user clicked (``direct_H``), so the pitch lands where they marked
it. A rigid camera model is fitted alongside, but only as a plausibility signal.

That ordering was learned the hard way. The camera model assumes a regulation
100x64 pitch with regulation box sizes, no lens distortion, and the lens dead
centre — a school or community ground owes us none of that. When those
assumptions could not reach the clicked points, the leftover error got reported
as "a landmark is on the wrong thing", which blamed the one part of the process
that is actually reliable: a person marking where two painted lines cross. Now a
mismatch is reported as what it is — a statement about the regulation model, not
about the clicking.

The camera model still earns its place as a *check*: 8 free matrix parameters
can describe surfaces no camera could see (pitches folded through the image
plane, corners behind the camera, self-intersecting rectangles), so a large
disagreement is worth surfacing. It just no longer overrides the anchors.

Two hard-won rules encoded here:

* **Fit each click in its own frame's pixels.** Mapping clicks into a shared
  reference frame first runs them through the camera-motion chain, whose
  similarity approximation is not exact for a rotating camera. Measured, that
  alone pushed the fit from 3.8px to 16.9px and put the recovered camera height
  out by 43%.
* **Residuals cannot validate a calibration.** A fit with 5 clicks has almost no
  slack, so a low error is close to guaranteed and proves nothing. Real
  validation is visual (draw the pitch back on the frame) and happens in the UI;
  what this module can do is report the error *and* the spare degrees of
  freedom, so a meaningless number is labelled as one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# Landmark positions on a pitch of length L and width W, in metres. x runs along
# the length (0 = left goal line), y across the width (0 = the touchline nearest
# the camera). Box and circle sizes are fixed by the laws and do not scale with
# the pitch: goal 7.32 wide, goal area 5.5 deep, penalty area 16.5 deep and
# 40.32 wide, penalty spot 11 out, circle radius 9.15.
LANDMARKS: dict[str, tuple[str, str]] = {
    "corner_near_left":          ("0", "0"),
    "corner_far_left":           ("0", "W"),
    "corner_near_right":         ("L", "0"),
    "corner_far_right":          ("L", "W"),
    "penarea_L_goalline_near":   ("0", "W/2-20.16"),
    "penarea_L_goalline_far":    ("0", "W/2+20.16"),
    "penarea_L_outer_near":      ("16.5", "W/2-20.16"),
    "penarea_L_outer_far":       ("16.5", "W/2+20.16"),
    "goalarea_L_goalline_near":  ("0", "W/2-9.16"),
    "goalarea_L_goalline_far":   ("0", "W/2+9.16"),
    "goalarea_L_outer_near":     ("5.5", "W/2-9.16"),
    "goalarea_L_outer_far":      ("5.5", "W/2+9.16"),
    "penspot_L":                 ("11", "W/2"),
    "post_L_near":               ("0", "W/2-3.66"),
    "post_L_far":                ("0", "W/2+3.66"),
    "halfway_near":              ("L/2", "0"),
    "halfway_far":               ("L/2", "W"),
    "centre_spot":               ("L/2", "W/2"),
    "circle_near":               ("L/2", "W/2-9.15"),
    "circle_far":                ("L/2", "W/2+9.15"),
    "circle_left":               ("L/2-9.15", "W/2"),
    "circle_right":              ("L/2+9.15", "W/2"),
    "penarea_R_goalline_near":   ("L", "W/2-20.16"),
    "penarea_R_goalline_far":    ("L", "W/2+20.16"),
    "penarea_R_outer_near":      ("L-16.5", "W/2-20.16"),
    "penarea_R_outer_far":       ("L-16.5", "W/2+20.16"),
    "penspot_R":                 ("L-11", "W/2"),
}

# Parameter order: Xc, Yc, h, f, roll, then (pan, tilt) per clicked frame.
_N_SHARED = 5


def landmark_xy(key: str, L: float, W: float) -> tuple[float, float] | None:
    spec = LANDMARKS.get(key)
    if spec is None:
        return None
    env = {"L": float(L), "W": float(W)}
    try:
        return (float(eval(spec[0], {"__builtins__": {}}, env)),
                float(eval(spec[1], {"__builtins__": {}}, env)))
    except Exception:
        return None


def rotation(pan: float, tilt: float, roll: float = 0.0) -> np.ndarray | None:
    """World->camera rotation. Rows are the camera axes expressed in world.

    Camera axes are x=right, y=down, z=forward, so y = z x x. Pinned by
    ``assert_convention``: at pan=0 (looking along +X with world up = +Z) right
    must be -Y and down must be -Z. Getting these signs wrong flips the image
    vertically, and a self-test that both generates and fits with the same
    convention cannot detect it.
    """
    fwd = np.array([math.cos(tilt) * math.cos(pan),
                    math.cos(tilt) * math.sin(pan),
                    -math.sin(tilt)])
    right = np.array([math.sin(pan), -math.cos(pan), 0.0])
    down = np.cross(fwd, right)
    n = float(np.linalg.norm(down))
    if n < 1e-12:
        return None
    down = down / n
    if roll:
        c, s = math.cos(roll), math.sin(roll)
        right, down = c * right + s * down, -s * right + c * down
    return np.vstack([right, down, fwd])


def pose_to_H(params, cx: float, cy: float) -> np.ndarray | None:
    """Pitch metres (z=0) -> image pixels, for one camera pose.

    ``params`` = (Xc, Yc, h, pan, tilt, f, roll).
    """
    Xc, Yc, h, pan, tilt, f, roll = params
    R = rotation(pan, tilt, roll)
    if R is None:
        return None
    t = -R @ np.array([Xc, Yc, h])
    K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])
    return K @ np.column_stack([R[:, 0], R[:, 1], t])


def project(params, pitch_pts, cx: float, cy: float) -> np.ndarray:
    """Project pitch points to pixels; NaN wherever the point is behind."""
    pitch_pts = np.asarray(pitch_pts, dtype=np.float64).reshape(-1, 2)
    out = np.full((len(pitch_pts), 2), np.nan)
    H = pose_to_H(params, cx, cy)
    if H is None:
        return out
    Q = (H @ np.column_stack([pitch_pts, np.ones(len(pitch_pts))]).T).T
    w = Q[:, 2]
    ok = w > 1e-9
    out[ok] = Q[ok, :2] / w[ok, None]
    return out


def assert_convention() -> None:
    """Pin the axis convention against answers known a priori."""
    R = rotation(0.0, 0.0)
    assert np.allclose(R[0], [0, -1, 0]), R[0]
    assert np.allclose(R[1], [0, 0, -1]), R[1]
    assert np.allclose(R[2], [1, 0, 0]), R[2]
    R = rotation(math.pi / 2, 0.0)
    assert np.allclose(R[0], [1, 0, 0]), R[0]
    assert np.allclose(R[1], [0, 0, -1]), R[1]
    for pan in np.linspace(-math.pi, math.pi, 9):
        for tilt in np.radians([1.0, 15.0, 45.0, 80.0]):
            R = rotation(float(pan), float(tilt))
            assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
            assert abs(np.linalg.det(R) - 1.0) < 1e-9
            assert R[1, 2] < 0.0, "image-down must point downward in the world"


@dataclass
class PitchCalibration:
    """A fitted camera plus everything needed to reuse it on other frames."""

    params: list[float]              # Xc, Yc, h, f, roll, (pan, tilt) per frame
    frames: list[int]                # frame indices, aligned with the pan/tilt pairs
    anchor_frame: int                # the frame the stored homography belongs to
    anchor_sec: float
    pitch_length_m: float
    pitch_width_m: float
    principal: tuple[float, float]
    rms_px: float
    median_px: float
    dof: int
    n_landmarks: int
    frame_width: int                 # width the clicks were made at
    warnings: list[str] = field(default_factory=list)
    # Pitch -> anchor-frame pixels, fitted directly to the anchors. Takes
    # precedence over the camera model (see anchor_H).
    direct_H: list | None = None
    direct_median_px: float | None = None

    @property
    def camera_height_m(self) -> float:
        return float(self.params[2])

    @property
    def focal_px(self) -> float:
        return float(self.params[3])

    def pose_at(self, k: int) -> list[float]:
        Xc, Yc, h, f, roll = self.params[:_N_SHARED]
        pan = self.params[_N_SHARED + 2 * k]
        tilt = self.params[_N_SHARED + 2 * k + 1]
        return [Xc, Yc, h, pan, tilt, f, roll]

    def anchor_H(self) -> np.ndarray | None:
        """Pitch metres -> pixels of the anchor frame.

        Prefers a transform fitted straight to the user's anchors when one was
        supplied. The camera model assumes a regulation 100x64 pitch with
        regulation box sizes, no lens distortion and the lens dead centre; a
        community pitch owes us none of that, and forcing it puts the pitch
        somewhere the user did not click. Their marks are the measurement — the
        camera model is the assumption — so the marks win, and the camera fit
        stays only as a plausibility signal.
        """
        if self.direct_H is not None:
            H = np.asarray(self.direct_H, dtype=np.float64).reshape(3, 3)
            if np.isfinite(H).all() and abs(float(np.linalg.det(H))) > 1e-12:
                return H
        k = self.frames.index(self.anchor_frame)
        return pose_to_H(self.pose_at(k), *self.principal)

    def describe(self) -> str:
        Xc, Yc, h, f, roll = self.params[:_N_SHARED]
        pan = math.degrees(self.params[_N_SHARED])
        tilt = math.degrees(self.params[_N_SHARED + 1])
        across = abs(abs(pan) - 90.0) < 25.0
        return (f"{h:.1f} m high, {abs(Yc):.0f} m back from the touchline, "
                f"pointed {'across the pitch' if across else f'{pan:.0f} deg'}, "
                f"tilted {tilt:.0f} deg down")

    def to_dict(self) -> dict:
        return {
            "params": [float(v) for v in self.params],
            "frames": [int(v) for v in self.frames],
            "anchor_frame": int(self.anchor_frame),
            "anchor_sec": float(self.anchor_sec),
            "pitch_length_m": float(self.pitch_length_m),
            "pitch_width_m": float(self.pitch_width_m),
            "principal": [float(v) for v in self.principal],
            "rms_px": round(float(self.rms_px), 3),
            "median_px": round(float(self.median_px), 3),
            "dof": int(self.dof),
            "n_landmarks": int(self.n_landmarks),
            "frame_width": int(self.frame_width),
            "camera": self.describe(),
            "warnings": list(self.warnings),
            "direct_H": (None if self.direct_H is None
                         else [float(v) for v in np.asarray(self.direct_H).reshape(-1)]),
            "direct_median_px": (None if self.direct_median_px is None
                                 else round(float(self.direct_median_px), 3)),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PitchCalibration":
        return cls(
            params=[float(v) for v in d["params"]],
            frames=[int(v) for v in d["frames"]],
            anchor_frame=int(d["anchor_frame"]),
            anchor_sec=float(d.get("anchor_sec", 0.0)),
            pitch_length_m=float(d.get("pitch_length_m", 100.0)),
            pitch_width_m=float(d.get("pitch_width_m", 64.0)),
            principal=tuple(float(v) for v in d.get("principal", (320.0, 180.0))),
            rms_px=float(d.get("rms_px", 0.0)),
            median_px=float(d.get("median_px", 0.0)),
            dof=int(d.get("dof", 0)),
            n_landmarks=int(d.get("n_landmarks", 0)),
            frame_width=int(d.get("frame_width", 640)),
            warnings=list(d.get("warnings", [])),
            direct_H=d.get("direct_H"),
            direct_median_px=d.get("direct_median_px"),
        )


def fit(
    clicks: list[tuple[int, tuple[float, float], str]],
    *,
    principal: tuple[float, float],
    pitch_length_m: float = 100.0,
    pitch_width_m: float = 64.0,
    free_pitch_size: bool = False,
    seed_params=None,
) -> PitchCalibration | None:
    """Fit one shared camera with per-frame pan/tilt.

    ``clicks`` are ``(frame_index, (u, v), landmark_key)`` in that frame's OWN
    pixels. Position, height, focal length and roll are shared across frames
    because a tripod only rotates; only pan and tilt vary.

    ``seed_params`` is an optional starting point (the browser's own fit of the
    same clicks). It turns a ~12s blind search into a sub-second polish, and is
    verified rather than trusted: a seed that refines poorly is discarded and the
    blind search runs anyway.

    ``free_pitch_size`` is off by default. It was built and tested against a
    venue whose mapping was failing, and it did *not* fix it — the error fell 25x
    while the overlay stayed just as wrong, which is what extra parameters do to
    an under-determined fit. Useful only where the ground is known to be
    non-standard.
    """
    from scipy.optimize import least_squares

    usable = [(fi, xy, k) for fi, xy, k in clicks if k in LANDMARKS]
    if len(usable) < 4:
        return None
    order = sorted({fi for fi, _, _ in usable})
    idx = {fi: k for k, fi in enumerate(order)}
    who = np.array([idx[fi] for fi, _, _ in usable], dtype=int)
    img = np.array([xy for _, xy, _ in usable], dtype=np.float64)
    keys = [k for _, _, k in usable]
    nf = len(order)
    cx, cy = principal

    lo = [-40.0, -60.0, 1.2, 150.0, math.radians(-12.0)]
    hi = [max(160.0, pitch_length_m + 40.0), -0.5, 45.0, 6000.0, math.radians(12.0)]
    for _ in range(nf):
        lo += [-math.pi, math.radians(1.0)]
        hi += [math.pi, math.radians(85.0)]
    if free_pitch_size:
        lo += [60.0, 40.0]
        hi += [125.0, 90.0]

    # Landmark positions are parsed once, not per residual evaluation: the
    # optimiser calls resid thousands of times and re-running eval() on every
    # landmark each time dominated the runtime (17s of a 27s fit).
    _fixed_pitch = (None if free_pitch_size else
                    np.array([landmark_xy(k, pitch_length_m, pitch_width_m)
                              for k in keys], dtype=np.float64))
    # rows belonging to each frame, precomputed
    groups = [np.flatnonzero(who == k) for k in range(nf)]

    def pitch_of(v):
        if _fixed_pitch is not None:
            return _fixed_pitch
        return np.array([landmark_xy(k, v[-2], v[-1]) for k in keys],
                        dtype=np.float64)

    def resid(v):
        pit = pitch_of(v)
        out = np.empty_like(img)
        for k, rows in enumerate(groups):
            if rows.size == 0:
                continue
            pose = (v[0], v[1], v[2], v[_N_SHARED + 2 * k],
                    v[_N_SHARED + 2 * k + 1], v[3], v[4])
            H = pose_to_H(pose, cx, cy)
            if H is None:
                out[rows] = 1e4
                continue
            p = pit[rows]
            Q = (H @ np.column_stack([p, np.ones(len(p))]).T).T
            w = Q[:, 2]
            ok = w > 1e-9
            # A point behind the camera is not infinitely wrong — a large finite
            # penalty lets the optimiser climb back out.
            sub = np.full((len(p), 2), 1e4)
            if ok.any():
                sub[ok] = Q[ok, :2] / w[ok, None]
            out[rows] = sub
        r = (out - img).ravel()
        r[~np.isfinite(r)] = 5e3
        return r

    def _solve(seed, budget):
        try:
            res = least_squares(
                resid, seed, bounds=(lo, hi), method="trf",
                # robust loss: one mislabelled landmark must not drag the whole
                # camera, and plain least squares has no defence against that
                loss="soft_l1", f_scale=6.0,
                max_nfev=budget, xtol=1e-10, ftol=1e-10,
            )
        except Exception:
            return None
        d = np.linalg.norm(resid(res.x).reshape(-1, 2), axis=1)
        if not np.isfinite(d).all():
            return None
        return float(np.median(d)), float(np.sqrt(np.mean(d ** 2))), res.x

    # The UI has already fitted these same clicks (it draws the pitch live), so
    # when it passes its answer through we only need a local polish. The full
    # sweep below costs ~12s, which is too long to sit behind a button.
    best = None
    if seed_params is not None:
        try:
            s0 = np.clip(np.asarray(seed_params, dtype=np.float64), lo, hi)
            if s0.shape == (len(lo),):
                best = _solve(s0, 800)
        except Exception:
            best = None

    # Fall back to a blind search when there is no seed, or the seed refined to
    # something poor (a stale or mismatched hint must not silently win).
    if best is None or best[0] > 12.0:
        # Coarse-to-fine. Pan must be swept because the optimiser cannot rotate
        # the camera halfway round the pitch on its own; tilt and focal length
        # are far less multi-modal, so one seed each is enough to find the basin.
        seeds = []
        for pan_deg in (-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180):
            for tilt_deg in (10.0, 28.0):
                s = [pitch_length_m / 2, -12.0, 8.0, 800.0, 0.0]
                s += [math.radians(pan_deg), math.radians(tilt_deg)] * nf
                if free_pitch_size:
                    s += [pitch_length_m, pitch_width_m]
                seeds.append(np.clip(np.array(s, dtype=np.float64), lo, hi))
        coarse = [r for r in (_solve(s, 40) for s in seeds) if r is not None]
        coarse.sort(key=lambda r: r[0])
        for _med, _rms, x in coarse[:3]:
            r = _solve(x, 800)
            if r is not None and (best is None or r[0] < best[0]):
                best = r
    if best is None:
        return None

    med, rms, v = best
    L = float(v[-2]) if free_pitch_size else float(pitch_length_m)
    W = float(v[-1]) if free_pitch_size else float(pitch_width_m)
    n_par = _N_SHARED + 2 * nf + (2 if free_pitch_size else 0)
    n_distinct = len(set(keys))
    dof = 2 * n_distinct - n_par

    warnings: list[str] = []
    if dof < 4:
        warnings.append(
            f"only {dof} spare degrees of freedom ({n_par} unknowns vs "
            f"{2 * n_distinct} constraints) — a low error proves little here")
    if rms > max(2.5 * med, med + 4.0):
        warnings.append(
            f"one landmark is much worse than the rest (worst {rms:.1f}px vs "
            f"typical {med:.1f}px) — likely a single bad click")
    # the anchor is the frame carrying the most clicks: fewest composed steps
    counts = {fi: int((who == idx[fi]).sum()) for fi in order}
    anchor = max(order, key=lambda fi: counts[fi])

    direct_H, direct_med = _direct_homography(
        [(fi, xy, k) for fi, xy, k in usable if fi == anchor], L, W)
    if direct_H is not None:
        # The drawn/used pitch now passes through the anchors, so a leftover
        # camera-model residual is a statement about our assumptions, not about
        # the user's clicking. Say so rather than implying a bad click.
        warnings = [w for w in warnings if "bad click" not in w]
        if med > 8.0:
            warnings.append(
                f"a regulation-sized pitch is {med:.0f}px off these anchors — "
                f"expected if this pitch is not standard size; the anchors are "
                f"used, not the regulation model")

    return PitchCalibration(
        direct_H=(None if direct_H is None else direct_H.reshape(-1).tolist()),
        direct_median_px=direct_med,
        params=[float(x) for x in v[:_N_SHARED + 2 * nf]],
        frames=list(order), anchor_frame=int(anchor), anchor_sec=0.0,
        pitch_length_m=L, pitch_width_m=W,
        principal=(float(cx), float(cy)),
        rms_px=rms, median_px=med, dof=int(dof),
        n_landmarks=n_distinct, frame_width=int(round(2 * cx)),
        warnings=warnings,
    )


def _direct_homography(clicks, L: float, W: float):
    """Pitch metres -> image pixels, fitted straight to one frame's anchors.

    A person marking where two painted lines cross is about as reliable as this
    system gets; the camera model's assumptions (regulation pitch, regulation
    boxes, no lens distortion, centred principal point) are not. So the transform
    that gets used is the one that honours the anchors, and any disagreement with
    a regulation pitch is reported as exactly that.

    Returns (3x3, median residual px) or (None, None).
    """
    if len(clicks) < 4:
        return None, None
    pit, img, keys = [], [], []
    for _fi, xy, key in clicks:
        p = landmark_xy(key, L, W)
        if p is None:
            continue
        pit.append(p)
        img.append(xy)
        keys.append(key)
    if len(set(keys)) < 4:
        return None, None      # four DISTINCT spots, or the solve is degenerate
    src = np.asarray(pit, dtype=np.float64)
    dst = np.asarray(img, dtype=np.float64)
    try:
        import cv2
        H, _mask = cv2.findHomography(src, dst, 0)
    except Exception:
        return None, None
    if H is None or not np.isfinite(H).all():
        return None, None
    if abs(float(np.linalg.det(H))) < 1e-12:
        return None, None
    Q = (H @ np.column_stack([src, np.ones(len(src))]).T).T
    w = Q[:, 2]
    if not np.all(np.abs(w) > 1e-9):
        return None, None
    resid = np.linalg.norm(Q[:, :2] / w[:, None] - dst, axis=1)
    return H, float(np.median(resid))


class PitchMapper:
    """Turns a calibration into "where is this pixel, in metres" on any frame.

    The calibration is anchored to one frame. For any other moment the camera
    has panned, so the anchor homography is carried across by the Stage 3d
    camera track. Two things make that safe rather than silent:

    * the track refuses to compose across a shot boundary, so a cut yields
      ``None`` instead of an invented transform;
    * every failure returns ``None``, and callers fall back to the pixel gate.

    Distances are measured on the ground plane between the ball's projection and
    the player's FEET. Feet, because a homography maps the ground and only a
    ground contact point is valid. The ball may be airborne, which pushes its
    ground projection further from the camera than the ball really is — that can
    only ever make a distance look bigger, i.e. cause a rejection, so the caller
    keeps a small-pixel-distance override to protect those touches.
    """

    def __init__(self, calib: PitchCalibration, camera_track, frame_width: int):
        self.calib = calib
        self.track = camera_track
        # Clicks may have been made at a different resolution than the pipeline
        # runs at; everything below works in analysed-frame pixels.
        self.click_scale = (float(frame_width) / float(calib.frame_width)
                            if calib.frame_width else 1.0)
        self._anchor_H: np.ndarray | None = None
        H = calib.anchor_H()
        if H is not None and np.isfinite(H).all():
            if abs(self.click_scale - 1.0) > 1e-6:
                s = self.click_scale
                S = np.array([[s, 0.0, 0.0], [0.0, s, 0.0], [0.0, 0.0, 1.0]])
                H = S @ H
            self._anchor_H = H
        self._cache: dict[int, np.ndarray | None] = {}
        self.n_ok = 0
        self.n_fallback = 0

    def image_to_pitch(self, frame_index: int) -> np.ndarray | None:
        """Pixels -> pitch metres on this FRAME, or None if not trustworthy.

        Addressed by frame index rather than seconds on purpose: the camera
        track's ``processed_sec`` is a compacted timeline with dead time removed,
        so the video time the UI knows does not address it. Frame index is the
        only unambiguous key shared by the UI, the trajectory and the track.
        """
        if self._anchor_H is None:
            return None
        key = int(frame_index)
        if key in self._cache:
            return self._cache[key]
        H = None
        rel = (self.track.relative_by_frame(self.calib.anchor_frame, key)
               if self.track is not None else None)
        if rel is not None and np.isfinite(rel).all():
            try:
                H = np.linalg.inv(rel @ self._anchor_H)
            except np.linalg.LinAlgError:
                H = None
        if H is not None and not np.isfinite(H).all():
            H = None
        self._cache[key] = H
        return H

    def to_pitch(self, frame_index: int, xy) -> tuple[float, float] | None:
        H = self.image_to_pitch(frame_index)
        if H is None:
            return None
        v = H @ np.array([float(xy[0]), float(xy[1]), 1.0])
        if abs(v[2]) < 1e-9 or not np.isfinite(v).all():
            return None
        # a point at or above the horizon projects behind the camera
        if v[2] <= 0:
            return None
        return (float(v[0] / v[2]), float(v[1] / v[2]))

    def ground_distance_m(
        self, frame_index: int, ball_xy, feet_xy,
    ) -> float | None:
        a = self.to_pitch(frame_index, ball_xy)
        b = self.to_pitch(frame_index, feet_xy)
        if a is None or b is None:
            return None
        return float(math.hypot(a[0] - b[0], a[1] - b[1]))

    def note(self, used: bool) -> None:
        if used:
            self.n_ok += 1
        else:
            self.n_fallback += 1

    def stats(self) -> dict:
        total = self.n_ok + self.n_fallback
        return {
            "contacts_metric": self.n_ok,
            "contacts_pixel_fallback": self.n_fallback,
            "metric_coverage": round(self.n_ok / total, 4) if total else 0.0,
        }
