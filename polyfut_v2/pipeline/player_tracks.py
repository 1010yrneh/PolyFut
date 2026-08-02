"""Issue 15, step 2: turn per-frame player boxes into tracklets.

Step 1 keeps the players the full-frame ball scan already found. On its own that
is just a pile of boxes; this links them across frames so the pipeline can know
where a body *was*, not only where some body *is*.

Three things make this different from a textbook tracker, and each is forced by
something measured on this footage:

* **The gate is depth-aware.** A pixel is not a fixed distance — measured on a
  calibrated clip, one pixel at the far touchline is worth 5.7x one near the
  camera. A fixed pixel gate would be far too tight near the camera and far too
  loose at the far side, which is the same defect Issue 14 found in attribution.
  The allowance is therefore set in metres and converted per box using its own
  height (a player is ~1.75 m), so it means the same thing everywhere in frame.

* **It can associate in camera-compensated space, but does not by default.**
  Measured on `b48758eb195e`: between consecutive harvested frames the camera
  translates a median of **2.79 px** (p90 21.3), against a gate that already
  allows ~39 px — so compensation is almost never *needed* at this cadence,
  while its estimation error perturbs every prediction. Switching it on made
  things worse, not better: 109 tracks at a 12.9s mean life against 89 at 16.0s
  with raw pixels. It stays available (``track_use_camera``) because a longer
  cadence or a faster pan would change that, and because the orbital prior
  (Issue 5) reasons over multi-second gaps where pans *do* accumulate — that is
  a different regime from 0.27s association, and both findings can hold.

* **It expects gaps.** Players arrive only on frames that ran a full-frame scan
  (measured: 54% of analysed frames), so the allowance grows with the gap
  rather than assuming consecutive frames.

Deliberately NOT here: kit colour. The colour lock that would stop two bodies
swapping is unreliable at the bleached venue, where every kit reads as turf
(Issue 11) — wiring it in would look like a safeguard while doing nothing.
Position and size only, and an honest note that same-kit swaps are possible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# A human is about this tall; a box's height in pixels is therefore a usable
# scale factor for converting a metre allowance into pixels at that depth.
_PLAYER_HEIGHT_M = 1.75
# Sprinting is ~9 m/s; a little headroom absorbs detection jitter on the box.
_MAX_SPEED_M_S = 11.0
# Even at dt=0 two detections of one player differ by a bit of box noise.
_BASE_M = 1.2


@dataclass
class TrackPoint:
    frame_index: int
    t_sec: float
    bbox: list[float]
    conf: float
    class_id: int

    @property
    def centre(self) -> tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2.0,
                (self.bbox[1] + self.bbox[3]) / 2.0)

    @property
    def height(self) -> float:
        return max(1.0, self.bbox[3] - self.bbox[1])


@dataclass
class PlayerTrack:
    id: int
    points: list[TrackPoint] = field(default_factory=list)

    @property
    def last(self) -> TrackPoint:
        return self.points[-1]

    @property
    def duration_sec(self) -> float:
        return self.points[-1].t_sec - self.points[0].t_sec if self.points else 0.0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "n_points": len(self.points),
            "start_sec": round(self.points[0].t_sec, 3) if self.points else None,
            "end_sec": round(self.points[-1].t_sec, 3) if self.points else None,
            "duration_sec": round(self.duration_sec, 3),
        }


def _allowance_px(height_px: float, dt: float) -> float:
    """How far a real player could have moved, in pixels at this box's depth."""
    px_per_m = max(height_px, 1.0) / _PLAYER_HEIGHT_M
    return (_BASE_M + _MAX_SPEED_M_S * max(0.0, dt)) * px_per_m


class PlayerTracker:
    """Greedy nearest-neighbour association over sparse player detections.

    Greedy rather than Hungarian on purpose: matches are taken in ascending
    distance so the most certain pairing is committed first, it is deterministic,
    and at ~20 boxes a frame the optimal assignment is not worth the dependency.
    """

    def __init__(self, cfg=None, *, camera=None):
        self.cfg = cfg
        # Off unless asked for: measured to fragment tracks at this cadence
        # (see module docstring). Passing a camera explicitly still enables it,
        # so callers and tests can exercise the path.
        use_cam = bool(getattr(cfg, "track_use_camera", False)) if cfg else True
        self.camera = camera if (camera is not None and use_cam) else None
        self.tracks: list[PlayerTrack] = []
        self._live: list[PlayerTrack] = []
        self._next_id = 0
        self.max_gap_sec = float(getattr(cfg, "track_max_gap_sec", 2.0)) if cfg else 2.0

    # -- internals ---------------------------------------------------------
    def _predict(self, track: PlayerTrack, frame_index: int) -> tuple[float, float] | None:
        """Where this track's last point sits in THIS frame's pixels.

        None when the camera track refuses to relate the two frames — different
        shots, or an unmeasured stretch. Refusing is the point: a match made
        across a cut is a match between two different people.
        """
        cx, cy = track.last.centre
        if self.camera is None:
            return cx, cy
        rel = self.camera.relative_by_frame(track.last.frame_index, frame_index)
        if rel is None:
            return None
        v = np.asarray(rel, dtype=np.float64).reshape(3, 3) @ np.array([cx, cy, 1.0])
        if not np.isfinite(v).all() or abs(v[2]) < 1e-9:
            return None
        return float(v[0] / v[2]), float(v[1] / v[2])

    # -- api ---------------------------------------------------------------
    def update(self, frame_index: int, t_sec: float, players) -> list[int]:
        """Fold one frame's detections in. Returns the track id per detection.

        ``players`` is ``(xyxy, conf, cls)`` — the shape the ball scan harvests
        and the player detector returns. An empty or None frame is a no-op, not
        a break: a missing detection is not evidence the player left.
        """
        if players is None:
            return []
        xyxy, conf, cls = players
        n = len(xyxy)
        if n == 0:
            return []

        pts = [TrackPoint(frame_index, t_sec, [float(v) for v in xyxy[i]],
                          float(conf[i]), int(cls[i])) for i in range(n)]

        # close tracks that have been silent too long
        self._live = [t for t in self._live
                      if (t_sec - t.last.t_sec) <= self.max_gap_sec]

        pairs = []
        for ti, tr in enumerate(self._live):
            dt = t_sec - tr.last.t_sec
            if dt < 0:
                continue
            pred = self._predict(tr, frame_index)
            if pred is None:
                continue          # cut / unmeasured — do not link across it
            for pi, p in enumerate(pts):
                cx, cy = p.centre
                d = math.hypot(cx - pred[0], cy - pred[1])
                # scale by the CURRENT box: the track's old height belongs to
                # an older depth, and players walk toward the camera
                if d <= _allowance_px(p.height, dt):
                    pairs.append((d, ti, pi))

        pairs.sort(key=lambda x: x[0])
        used_t: set[int] = set()
        used_p: set[int] = set()
        assigned: dict[int, int] = {}
        for _d, ti, pi in pairs:
            if ti in used_t or pi in used_p:
                continue
            used_t.add(ti)
            used_p.add(pi)
            self._live[ti].points.append(pts[pi])
            assigned[pi] = self._live[ti].id

        out = []
        for pi, p in enumerate(pts):
            if pi in assigned:
                out.append(assigned[pi])
                continue
            tr = PlayerTrack(id=self._next_id, points=[p])
            self._next_id += 1
            self.tracks.append(tr)
            self._live.append(tr)
            out.append(tr.id)
        return out

    def finished(self, *, min_points: int = 2,
                 min_duration_sec: float = 0.0) -> list[PlayerTrack]:
        """Tracks worth keeping. A one-sighting 'track' is a detection."""
        return [t for t in self.tracks
                if len(t.points) >= min_points
                and t.duration_sec >= min_duration_sec]
