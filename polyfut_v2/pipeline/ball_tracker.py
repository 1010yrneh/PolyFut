"""Stage 3: continuous ball tracking → dense trajectory.

Runs the (single) ball detector on every analysed frame that falls inside a live
shot, threads detections through the v1 :class:`BallSmoother` for temporal hold /
interpolation across misses, and emits a :class:`BallTrajectory`.

No player detection happens here — that is the whole point of v2. Player work is
deferred to the sparse contact frames (Step 3).
"""

from __future__ import annotations

from typing import Callable, Iterator

import numpy as np

from polyfut_video.pipeline.ball_smooth import BallSmoother, BallSmoothConfig

from polyfut_v2.config import PipelineV2Config
from polyfut_v2.pipeline.ball_detector import BallDetectorProtocol
from polyfut_v2.pipeline.trajectory import BallSample, BallTrajectory

ProgressCb = Callable[[float, str], None]


def _center(bbox: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


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
    ))

    traj = BallTrajectory()
    cur_shot: dict | None = None
    shot_first_t: float | None = None
    shot_last_t: float | None = None
    processed_base = 0.0
    last_center: tuple[float, float] | None = None

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
            cur_shot = shot
            shot_first_t = t_sec
            smoother.reset()
            last_center = None

        processed_sec = processed_base + max(0.0, t_sec - (shot_first_t or t_sec))

        det = detector.detect(frame, last_center)
        raw_bbox = det.bbox if det is not None else None
        raw_conf = det.conf if det is not None else 0.0

        bbox, conf, interpolated = smoother.update(raw_bbox, raw_conf)

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

    return traj
