"""Stage 3: continuous ball tracking → dense trajectory.

Runs the (single) ball detector on every analysed frame that falls inside a live
shot, threads detections through the v1 :class:`BallSmoother` for temporal hold /
interpolation across misses, and emits a :class:`BallTrajectory`.

Two extra passes ride along on the frames this stage already decodes:

* **Hold interpolation** (Issue 6) — the smoother is an online component, so
  across a miss it can only *freeze* the last position. Once a segment is
  finished, any frozen run bracketed by two real detections is replaced with a
  straight-line interpolation. The samples stay flagged ``interpolated`` so
  Stage 4 still refuses them as velocity nodes.
* **Camera motion** (Issue 5) — a cheap background-flow estimate, accumulated so
  Stage 7 can compare positions in pan-stabilised space.

No player detection happens here — that is the whole point of v2. Player work is
deferred to the sparse contact frames (Step 3).
"""

from __future__ import annotations

from typing import Callable, Iterator

import numpy as np

from polyfut_video.pipeline.ball_smooth import BallSmoother, BallSmoothConfig

from polyfut_v2.config import PipelineV2Config
from polyfut_v2.pipeline.ball_detector import BallDetectorProtocol
from polyfut_v2.pipeline.camera_motion import CameraMotionEstimator
from polyfut_v2.pipeline.trajectory import BallSample, BallTrajectory

ProgressCb = Callable[[float, str], None]


def _center(bbox: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _lerp_run(
    samples: list[BallSample], start: int, stop: int,
    prev_det: BallSample, next_det: BallSample,
) -> int:
    """Straight-line fill of ``samples[start:stop]`` between two detections."""
    t0 = prev_det.processed_sec
    t1 = next_det.processed_sec
    span = t1 - t0
    if span <= 0:
        return 0
    n = 0
    for k in range(start, stop):
        s = samples[k]
        w = (s.processed_sec - t0) / span
        if not (0.0 <= w <= 1.0):
            continue
        s.x = float(prev_det.x) + w * (float(next_det.x) - float(prev_det.x))
        s.y = float(prev_det.y) + w * (float(next_det.y) - float(prev_det.y))
        if prev_det.bbox is not None and next_det.bbox is not None:
            s.bbox = [
                float(a) + w * (float(b) - float(a))
                for a, b in zip(prev_det.bbox, next_det.bbox)
            ]
        n += 1
    return n


def interpolate_held_samples(samples: list[BallSample]) -> int:
    """Rewrite frozen hold positions as interpolations between real detections.

    Only runs of held samples bracketed by a detection on BOTH sides are
    rewritten — a trailing hold has no forward anchor, so its frozen position is
    the only honest answer and is left alone. Returns the number of samples
    rewritten. Flags are untouched: these stay ``interpolated`` and not
    ``detected``, so Stage 4 kinematics still ignore them as velocity nodes.
    """
    n = len(samples)
    rewritten = 0
    i = 0
    while i < n:
        s = samples[i]
        if s.interpolated and not s.detected and s.has_position():
            j = i
            while (
                j < n and samples[j].interpolated
                and not samples[j].detected and samples[j].has_position()
            ):
                j += 1
            prev_det = samples[i - 1] if i > 0 else None
            next_det = samples[j] if j < n else None
            if (
                prev_det is not None and prev_det.detected and prev_det.has_position()
                and next_det is not None and next_det.detected and next_det.has_position()
            ):
                rewritten += _lerp_run(samples, i, j, prev_det, next_det)
            i = j
        else:
            i += 1
    return rewritten


class _LiveShotCursor:
    """Maps a monotonically-increasing timestamp to its live shot (or None for
    frames that fall in a removed gap between shots)."""

    def __init__(self, live_shots: list[dict]):
        self._shots = sorted(live_shots, key=lambda s: float(s["start_sec"]))
        self._i = 0

    def shot_for(self, t_sec: float) -> dict | None:
        while self._i < len(self._shots) and t_sec > float(self._shots[self._i]["end_sec"]):
            self._i += 1
        if self._i >= len(self._shots):
            return None
        shot = self._shots[self._i]
        if t_sec < float(shot["start_sec"]):
            return None  # in a removed gap
        return shot


def track_ball(
    frame_iter: Iterator[tuple[int, float, np.ndarray]],
    live_shots: list[dict],
    detector: BallDetectorProtocol,
    cfg: PipelineV2Config,
    *,
    progress: ProgressCb | None = None,
    should_cancel: Callable[[], bool] | None = None,
    total_live_sec: float | None = None,
) -> BallTrajectory:
    """Consume a decoded frame stream and return the ball trajectory.

    ``frame_iter`` must be time-ordered (as produced by ``iter_frames``).
    ``total_live_sec`` is only used to scale the progress fraction.
    """
    cancel = should_cancel or (lambda: False)
    cursor = _LiveShotCursor(live_shots)
    smoother = BallSmoother(BallSmoothConfig(
        max_hold_frames=cfg.ball_hold_frames,
        max_jump_px=cfg.ball_max_jump_px,
        suspect_jump_px=cfg.ball_suspect_jump_px,
        suspect_jump_conf=cfg.ball_suspect_jump_conf,
        require_jump_confirmation=getattr(cfg, "ball_confirm_jumps", True),
        confirm_dist_px=getattr(cfg, "ball_confirm_dist_px", 120.0),
        confirm_angle_deg=getattr(cfg, "ball_confirm_angle_deg", 45.0),
        confirm_speed_ratio=getattr(cfg, "ball_confirm_speed_ratio", 1.6),
    ))
    camera = (
        CameraMotionEstimator(cfg)
        if getattr(cfg, "camera_motion_enabled", False) else None
    )
    interpolate = getattr(cfg, "ball_interpolate_holds", True)

    traj = BallTrajectory()
    cur_shot: dict | None = None
    shot_first_t: float | None = None
    shot_last_t: float | None = None
    processed_base = 0.0
    last_center: tuple[float, float] | None = None
    seg_start = 0  # index in traj.samples where the current shot's run begins

    for frame_index, t_sec, frame in frame_iter:
        if cancel():
            raise RuntimeError("cancelled")

        shot = cursor.shot_for(t_sec)
        if shot is None:
            continue  # frame is in a removed (dead/discard) segment

        # Shot boundary → reset temporal state (no continuity across cuts).
        if shot is not cur_shot:
            if cur_shot is not None and shot_first_t is not None and shot_last_t is not None:
                processed_base += max(0.0, shot_last_t - shot_first_t)
            # Interpolate the finished segment before the new one starts, so a
            # fill can never span a cut.
            if interpolate and len(traj.samples) > seg_start:
                interpolate_held_samples(traj.samples[seg_start:])
            seg_start = len(traj.samples)
            cur_shot = shot
            shot_first_t = t_sec
            smoother.reset()
            if camera is not None:
                camera.reset()
            # Fresh scene → clear any miss-storm backoff so the detector scans
            # the very first frame of the new shot.
            if hasattr(detector, "reset"):
                detector.reset()
            last_center = None

        # shot_first_t is guaranteed set above (the boundary block runs on the
        # first live frame). Do NOT use `shot_first_t or t_sec`: a first shot
        # starting at t=0 makes shot_first_t == 0.0, which is falsy, collapsing
        # every processed_sec to 0 and breaking Stage 4 merge / Stage 7 orbital.
        assert shot_first_t is not None
        processed_sec = processed_base + max(0.0, t_sec - shot_first_t)

        det = detector.detect(frame, last_center)
        raw_bbox = det.bbox if det is not None else None
        raw_conf = det.conf if det is not None else 0.0

        bbox, conf, interpolated = smoother.update(raw_bbox, raw_conf)

        if camera is not None:
            ball_xy = _center(bbox) if bbox is not None else last_center
            camera.update(frame, processed_sec, ball_xy=ball_xy,
                          frame_index=frame_index)

        if bbox is not None:
            cx, cy = _center(bbox)
            last_center = (cx, cy)
            traj.add(BallSample(
                frame_index=frame_index,
                t_sec=t_sec,
                processed_sec=processed_sec,
                x=cx, y=cy,
                bbox=[float(v) for v in bbox],
                conf=float(conf),
                detected=(det is not None and not interpolated),
                interpolated=bool(interpolated),
            ))
        else:
            # Hold exhausted: no known position. Force a full-frame re-acquire
            # next frame by dropping the ROI anchor.
            last_center = None
            traj.add(BallSample(
                frame_index=frame_index,
                t_sec=t_sec,
                processed_sec=processed_sec,
                x=None, y=None, bbox=None,
                conf=0.0,
                detected=False,
                interpolated=False,
            ))

        shot_last_t = t_sec

        if progress is not None and total_live_sec:
            frac = min(1.0, processed_sec / total_live_sec)
            if len(traj) % 200 == 0:
                progress(frac, f"Stage 3: ball trajectory ({len(traj)} frames)…")

    if interpolate and len(traj.samples) > seg_start:
        interpolate_held_samples(traj.samples[seg_start:])
    if camera is not None:
        traj.camera = camera.track()

    return traj
