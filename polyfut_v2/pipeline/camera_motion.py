"""Stage 3d: camera-motion (pan) compensation for the Stage 7 orbital prior.

The orbital prior asks "could the same player have got from the last sighting to
here in this much time?" — a question about motion *on the pitch*. It is
evaluated in raw frame pixels, so a camera pan moves every player at once and
the prior reads a stationary player as having teleported. On panning footage the
strongest same-kit identity signal is therefore mostly noise.

This module estimates the frame-to-frame background shift with a cheap
Lucas-Kanade median flow (the same pattern as ``review_track.py``, no extra
model) and accumulates it into a camera track. Contact positions mapped through
that track live in a pan-stabilised space, so orbital distances mean what they
are supposed to mean.

Recall-safety rules, in order of importance:

* **Never invent motion.** A step with too few tracked points, or an
  implausibly large shift, contributes *zero* shift — the estimate degrades to
  identity rather than to a guess.
* **No continuity across cuts.** ``reset()`` at a shot boundary drops the
  reference frame, so the first frame of a new shot contributes no shift.
* Only *relative* offset between two contacts matters (both go through the same
  transform), so the absolute accumulated value never has to be correct.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

_LK = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
)

Transform = Callable[[tuple[float, float], float], tuple[float, float]]


@dataclass
class CameraSample:
    """Cumulative camera offset (in analysed-frame px) at one moment.

    ``dx``/``dy`` are the translation part, kept as the primary fields because
    the Stage 7 orbital only needs a shift. ``matrix`` additionally carries the
    full cumulative similarity (translation + rotation + scale) mapping this
    frame's pixels into the current shot's reference frame — needed by pitch
    calibration, which cares about rotation and zoom as well as pan.

    ``shot`` increments at every shot boundary. The jump across a cut is
    unmeasurable, so anything derived by composing across two different shots is
    meaningless; consumers must check that two samples share a ``shot``.
    """

    processed_sec: float
    dx: float
    dy: float
    confident: bool
    matrix: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    shot: int = 0
    # Source frame index. ``processed_sec`` is a COMPACTED timeline (dead time
    # and inter-shot gaps removed), so it cannot be derived from video time and
    # anything user-supplied must key off the frame index instead.
    frame_index: int = -1

    def to_dict(self) -> dict:
        return {
            "processed_sec": round(self.processed_sec, 4),
            "dx": round(self.dx, 2),
            "dy": round(self.dy, 2),
            "confident": bool(self.confident),
            "matrix": [round(float(v), 6) for v in self.matrix],
            "shot": int(self.shot),
            "frame_index": int(self.frame_index),
        }


def _identity(xy: tuple[float, float], t: float) -> tuple[float, float]:
    return xy


@dataclass
class CameraTrack:
    """Time series of cumulative camera offsets, with lookup + transform."""

    samples: list[CameraSample] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.samples)

    def confident_ratio(self) -> float:
        if not self.samples:
            return 0.0
        return sum(1 for s in self.samples if s.confident) / len(self.samples)

    def offset_at(self, t_sec: float) -> tuple[float, float]:
        """Cumulative (dx, dy) at ``t_sec``, linearly interpolated and clamped."""
        n = len(self.samples)
        if n == 0:
            return (0.0, 0.0)
        if t_sec <= self.samples[0].processed_sec:
            s = self.samples[0]
            return (s.dx, s.dy)
        if t_sec >= self.samples[-1].processed_sec:
            s = self.samples[-1]
            return (s.dx, s.dy)
        lo, hi = 0, n - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self.samples[mid].processed_sec <= t_sec:
                lo = mid
            else:
                hi = mid
        a, b = self.samples[lo], self.samples[hi]
        span = b.processed_sec - a.processed_sec
        if span <= 0:
            return (a.dx, a.dy)
        w = (t_sec - a.processed_sec) / span
        return (a.dx + w * (b.dx - a.dx), a.dy + w * (b.dy - a.dy))

    def _nearest(self, t_sec: float) -> CameraSample | None:
        if not self.samples:
            return None
        lo, hi = 0, len(self.samples) - 1
        if t_sec <= self.samples[lo].processed_sec:
            return self.samples[lo]
        if t_sec >= self.samples[hi].processed_sec:
            return self.samples[hi]
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self.samples[mid].processed_sec <= t_sec:
                lo = mid
            else:
                hi = mid
        a, b = self.samples[lo], self.samples[hi]
        return a if (t_sec - a.processed_sec) <= (b.processed_sec - t_sec) else b

    def _nearest_frame(self, frame_index: int) -> CameraSample | None:
        best = None
        best_gap = None
        for s in self.samples:
            if s.frame_index < 0:
                continue
            gap = abs(int(s.frame_index) - int(frame_index))
            if best_gap is None or gap < best_gap:
                best, best_gap = s, gap
        return best

    def relative_by_frame(self, frame_from: int, frame_to: int):
        """Image transform from one frame's pixels to another's, by INDEX.

        Frame index, not seconds: ``processed_sec`` is compacted, so a caller
        holding video time (the UI) cannot address this track by time at all.
        None when either frame is unknown or the two sit in different shots.
        """
        a = self._nearest_frame(frame_from)
        b = self._nearest_frame(frame_to)
        if a is None or b is None or int(a.shot) != int(b.shot):
            return None
        Ma = np.asarray(a.matrix, dtype=np.float64).reshape(3, 3)
        Mb = np.asarray(b.matrix, dtype=np.float64).reshape(3, 3)
        try:
            return Mb @ np.linalg.inv(Ma)
        except np.linalg.LinAlgError:
            return None

    def shot_at(self, t_sec: float) -> int | None:
        s = self._nearest(t_sec)
        return None if s is None else int(s.shot)

    def matrix_at(self, t_sec: float) -> np.ndarray | None:
        """Cumulative frame->shot-reference similarity nearest ``t_sec``.

        Nearest sample rather than interpolated: interpolating rotation and
        scale entrywise is not a valid transform, and the sampling interval is
        a fraction of a second.
        """
        s = self._nearest(t_sec)
        if s is None:
            return None
        return np.asarray(s.matrix, dtype=np.float64).reshape(3, 3)

    def relative(self, t_from: float, t_to: float) -> np.ndarray | None:
        """Image transform mapping frame-at-``t_from`` pixels to frame-at-``t_to``.

        None when either moment is unknown or the two sit in different shots —
        the camera jump across a cut is never measured, so composing through one
        would silently invent a transform.
        """
        a, b = self._nearest(t_from), self._nearest(t_to)
        if a is None or b is None or int(a.shot) != int(b.shot):
            return None
        Ma = np.asarray(a.matrix, dtype=np.float64).reshape(3, 3)
        Mb = np.asarray(b.matrix, dtype=np.float64).reshape(3, 3)
        try:
            return Mb @ np.linalg.inv(Ma)
        except np.linalg.LinAlgError:
            return None

    def transform(self) -> Transform:
        """Map a frame-space point at ``t`` into pan-stabilised space."""
        if not self.samples:
            return _identity

        def _t(xy: tuple[float, float], t: float) -> tuple[float, float]:
            dx, dy = self.offset_at(t)
            return (xy[0] - dx, xy[1] - dy)

        return _t

    def to_dict(self) -> dict:
        return {
            "count": len(self.samples),
            "confident_ratio": round(self.confident_ratio(), 4),
            "samples": [s.to_dict() for s in self.samples],
        }


def _tracked_points(
    prev_gray: np.ndarray,
    gray: np.ndarray,
    exclude: tuple[float, float, float] | None,
    min_points: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Forward-backward validated LK correspondences, or None."""
    if prev_gray is None or gray is None or prev_gray.shape != gray.shape:
        return None
    h, w = prev_gray.shape[:2]
    mask = None
    if exclude is not None:
        cx, cy, half = exclude
        mask = np.full((h, w), 255, dtype=np.uint8)
        x1, y1 = int(max(0, cx - half)), int(max(0, cy - half))
        x2, y2 = int(min(w, cx + half)), int(min(h, cy + half))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 0
        if int(mask.sum()) == 0:
            mask = None
    pts = cv2.goodFeaturesToTrack(
        prev_gray, maxCorners=200, qualityLevel=0.01,
        minDistance=8, blockSize=5, mask=mask,
    )
    if pts is None or len(pts) < min_points:
        return None
    p0 = pts.reshape(-1, 1, 2).astype(np.float32)
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **_LK)
    if p1 is None or st is None:
        return None
    p0r, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, p1, None, **_LK)
    if p0r is None or st2 is None:
        return None
    good = (st.reshape(-1) == 1) & (st2.reshape(-1) == 1)
    good &= np.abs(p0r - p0).reshape(-1, 2).max(axis=1) < 2.0
    if int(good.sum()) < min_points:
        return None
    return p0.reshape(-1, 2)[good], p1.reshape(-1, 2)[good]


def estimate_similarity(
    prev_gray: np.ndarray,
    gray: np.ndarray,
    *,
    exclude: tuple[float, float, float] | None = None,
    min_points: int = 8,
    max_shift_px: float = 200.0,
) -> np.ndarray | None:
    """Per-step similarity (translation + rotation + scale) ``prev -> cur``.

    A 4-DOF similarity, not the 8-DOF homography that is exactly correct for a
    rotating camera: measured on real footage, composing per-step homographies
    *diverges* (a 25s pan produced a canvas 8.7x the frame width instead of the
    correct 2.2x, with 5.17px drift), because each fit absorbs a little noise as
    perspective and the errors compound. The same chain fitted as a similarity
    drifted 0.72px. Rotation and scale still matter — plain translation cannot
    represent a tripod's roll or any zoom.

    RANSAC also handles screen-fixed graphics for free: a scoreboard does not
    move when the camera pans, so its points disagree with every other point and
    fall outside the consensus.

    Returns a 3x3 matrix, or None when not confident.
    """
    tracked = _tracked_points(prev_gray, gray, exclude, min_points)
    if tracked is None:
        return None
    a, b = tracked
    M, inliers = cv2.estimateAffinePartial2D(
        a, b, method=cv2.RANSAC, ransacReprojThreshold=2.0,
        maxIters=2000, confidence=0.995,
    )
    if M is None or inliers is None or int(inliers.sum()) < min_points:
        return None
    H = np.vstack([M, [0.0, 0.0, 1.0]]).astype(np.float64)
    if not np.isfinite(H).all():
        return None
    if abs(H[0, 2]) > max_shift_px or abs(H[1, 2]) > max_shift_px:
        return None  # implausible for one step — treat as unmeasured
    scale = float(np.hypot(H[0, 0], H[1, 0]))
    if not (0.5 < scale < 2.0):
        return None  # a single step cannot halve or double the zoom
    return H


def estimate_shift(
    prev_gray: np.ndarray,
    gray: np.ndarray,
    *,
    exclude: tuple[float, float, float] | None = None,
    min_points: int = 8,
    max_shift_px: float = 200.0,
) -> tuple[float, float] | None:
    """Median background shift ``prev_gray → gray``, or None if not confident.

    ``exclude`` is an optional ``(cx, cy, half)`` region (in ``prev_gray``
    coords) masked out of feature selection — the ball and the players around it
    move independently of the camera and would bias the median.
    """
    if prev_gray is None or gray is None:
        return None
    if prev_gray.shape != gray.shape:
        return None

    h, w = prev_gray.shape[:2]
    mask = None
    if exclude is not None:
        cx, cy, half = exclude
        mask = np.full((h, w), 255, dtype=np.uint8)
        x1 = int(max(0, cx - half))
        y1 = int(max(0, cy - half))
        x2 = int(min(w, cx + half))
        y2 = int(min(h, cy + half))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 0
        if int(mask.sum()) == 0:
            mask = None

    pts = cv2.goodFeaturesToTrack(
        prev_gray, maxCorners=120, qualityLevel=0.01,
        minDistance=8, blockSize=5, mask=mask,
    )
    if pts is None or len(pts) < min_points:
        return None

    p0 = pts.reshape(-1, 1, 2).astype(np.float32)
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **_LK)
    if p1 is None or st is None:
        return None
    # Forward-backward consistency: drop points that don't track both ways.
    p0r, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, p1, None, **_LK)
    if p0r is None or st2 is None:
        return None
    good = (st.reshape(-1) == 1) & (st2.reshape(-1) == 1)
    fb = np.abs(p0r - p0).reshape(-1, 2).max(axis=1)
    good &= fb < 2.0
    if int(good.sum()) < min_points:
        return None

    d = (p1 - p0).reshape(-1, 2)[good]
    dx = float(np.median(d[:, 0]))
    dy = float(np.median(d[:, 1]))
    if not np.isfinite(dx) or not np.isfinite(dy):
        return None
    if abs(dx) > max_shift_px or abs(dy) > max_shift_px:
        return None  # implausible for one step — treat as unmeasured
    return dx, dy


class CameraMotionEstimator:
    """Accumulates per-step background shifts into a :class:`CameraTrack`.

    Fed from the Stage 3 ball pass, which already has every analysed frame
    decoded — so this costs a downscaled grayscale LK pass, not a second decode.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self._prev_small: np.ndarray | None = None
        self._cum_dx = 0.0
        self._cum_dy = 0.0
        self._calls = 0
        self._scale = 1.0
        self._cum = np.eye(3, dtype=np.float64)
        self._shot = 0
        self.samples: list[CameraSample] = []

    def reset(self) -> None:
        """Shot boundary: forget the reference frame, keep the accumulator.

        Keeping the accumulated translation (rather than zeroing it) means the
        unknown inter-shot camera jump is treated as zero shift — the recall-safe
        choice for the Stage 7 orbital, since a fabricated jump would wrongly
        down-weight real touches.

        Pitch calibration cannot make that trade: a wrong transform there moves
        players to wrong places on the pitch. So the similarity accumulator
        restarts at identity and the shot counter advances, which is how
        ``CameraTrack.relative`` knows to refuse to compose across the cut.
        """
        self._prev_small = None
        self._cum = np.eye(3, dtype=np.float64)
        self._shot += 1

    def _downscale(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        target = int(getattr(self.cfg, "camera_motion_width", 320) or 0)
        if target > 0 and w > target:
            self._scale = w / float(target)
            frame = cv2.resize(frame, (target, max(1, int(round(h / self._scale)))))
        else:
            self._scale = 1.0
        if frame.ndim == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    def update(
        self,
        frame: np.ndarray,
        processed_sec: float,
        *,
        ball_xy: tuple[float, float] | None = None,
        frame_index: int = -1,
    ) -> CameraSample | None:
        """Fold one analysed frame in. Returns the new sample, or None if this
        frame was skipped by ``camera_motion_every_n``."""
        every = max(1, int(getattr(self.cfg, "camera_motion_every_n", 1)))
        self._calls += 1
        if (self._calls - 1) % every != 0:
            return None

        small = self._downscale(frame)
        confident = False
        if self._prev_small is not None:
            exclude = None
            half = float(getattr(self.cfg, "camera_motion_exclude_half_px", 0.0) or 0.0)
            if ball_xy is not None and half > 0:
                exclude = (ball_xy[0] / self._scale, ball_xy[1] / self._scale,
                           half / self._scale)
            step = estimate_similarity(
                self._prev_small, small,
                exclude=exclude,
                min_points=int(getattr(self.cfg, "camera_motion_min_points", 8)),
                max_shift_px=float(
                    getattr(self.cfg, "camera_motion_max_shift_px", 200.0)
                ) / max(self._scale, 1e-6),
            )
            if step is not None:
                # The step was measured on the downscaled frame; conjugate it
                # back into analysed-frame (target_width) pixels so translation
                # scales while rotation and zoom do not.
                s = self._scale
                S = np.array([[s, 0.0, 0.0], [0.0, s, 0.0], [0.0, 0.0, 1.0]])
                full = S @ step @ np.linalg.inv(S)
                self._cum = full @ self._cum
                self._cum_dx += float(step[0, 2]) * s
                self._cum_dy += float(step[1, 2]) * s
                confident = True
        self._prev_small = small

        sample = CameraSample(
            processed_sec=float(processed_sec),
            dx=self._cum_dx,
            dy=self._cum_dy,
            confident=confident,
            matrix=tuple(float(v) for v in self._cum.reshape(-1)),
            shot=int(self._shot),
            frame_index=int(frame_index),
        )
        self.samples.append(sample)
        return sample

    def track(self) -> CameraTrack:
        return CameraTrack(samples=list(self.samples))
