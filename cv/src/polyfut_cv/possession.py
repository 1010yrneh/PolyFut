"""Team possession: is the ball within contact range of ANY my-team player?

Geometry ported in spirit from the old single-player pipeline, but applied to the
nearest my-team player rather than one tracked ID. Uses foot-point distance scaled
by player height and ball size, plus asymmetric temporal hysteresis (fast to open,
slow to close) so brief misses don't fragment a possession.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from polyfut_cv.geometry import ball_diagonal, box_center, dist, foot_point


@dataclass
class PossessionConfig:
    conf_ball_min: float = 0.08
    on_frames: int = 1          # open on the first frame the ball is near my team
    off_frames: int = 8         # hold through brief misses (~4 s at 2 analysed fps)
    # Base contact distance (px) from a player's foot point to the ball centre,
    # scaled by player height / a reference, and by ball size. Generous because
    # the ball is small/flickery on wide footage; nearest-player attribution
    # (not pixel-perfect contact) is what we want for "team had the ball".
    base_thresh_px: float = 95.0
    ref_person_h: float = 120.0
    ref_ball_diag: float = 22.0


def _scaled_threshold(cfg: PossessionConfig, person_h: float, ball_d: float) -> float:
    size_f = 1.0 if ball_d <= 1 else cfg.ref_ball_diag / max(ball_d, 1.0)
    # Tiny balls (small ball_d) get a LARGER allowance, so distant play still counts.
    size_f = min(1.8, max(0.5, size_f))
    return cfg.base_thresh_px * (person_h / cfg.ref_person_h) * size_f


def nearest_my_team_contact(
    my_team_persons: list[np.ndarray],
    ball_xyxy: np.ndarray | None,
    ball_conf: float,
    cfg: PossessionConfig,
) -> tuple[bool, int]:
    """Return (contact, index_of_player). index is -1 when no contact."""
    if ball_xyxy is None or ball_conf < cfg.conf_ball_min or not my_team_persons:
        return False, -1
    bc = box_center(ball_xyxy)
    bd = ball_diagonal(ball_xyxy)
    best_i = -1
    best_d = 1e18
    for i, p in enumerate(my_team_persons):
        fp = foot_point(p)
        d = dist(fp, bc)
        if d < best_d:
            best_d = d
            best_i = i
    if best_i < 0:
        return False, -1
    p = my_team_persons[best_i]
    person_h = max(float(p[3] - p[1]), 1.0)
    thr = _scaled_threshold(cfg, person_h, bd)
    return (best_d <= thr), best_i


def extract_intervals(
    times_sec: list[float],
    flags: list[bool],
    cfg: PossessionConfig | None = None,
) -> list[tuple[float, float]]:
    """Asymmetric hysteresis -> closed [t0, t1] intervals in seconds."""
    cfg = cfg or PossessionConfig()
    if not times_sec or len(times_sec) != len(flags):
        return []

    on_streak = 0
    off_streak = 0
    active = False
    t_open: float | None = None
    last_ok_t: float | None = None
    out: list[tuple[float, float]] = []

    for t, ok in zip(times_sec, flags):
        if ok:
            last_ok_t = t
            off_streak = 0
            if not active:
                on_streak += 1
                if on_streak >= cfg.on_frames:
                    active = True
                    t_open = t
                    on_streak = 0
            continue
        on_streak = 0
        if active:
            off_streak += 1
            if off_streak >= cfg.off_frames:
                t_close = last_ok_t if last_ok_t is not None else t
                if t_open is not None:
                    out.append((t_open, max(t_close, t_open)))
                active = False
                off_streak = 0
                t_open = None
                last_ok_t = None

    if active and t_open is not None:
        out.append((t_open, max(times_sec[-1], t_open)))
    return out
