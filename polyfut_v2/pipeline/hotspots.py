"""Stage 9: assemble confirmed 'me' touches into hotspot windows.

For each confirmed target touch, take a ±pad window, merge windows that fall
within the gap threshold (so a dribble / quick sequence becomes one hotspot),
enforce a minimum zone length, and clamp to the video. Reuses v1's hotspot
config semantics (pad / gap-merge / min-zone), fed cleaner single-player events.

**Possession extension.** A fixed pad after the last *detected* touch cuts the
clip off mid-move: measured on job ``fdc541cf493e``, the last confirmed touch was
at 288.8s and the hotspot ended at 290.8s — exactly ``pad_after`` later — while
the player still had the ball. Touches are discrete events, but possession is
continuous, and with the ball detected on only 40% of frames the *last* touch of
a sequence is often simply the last one that was seen, not the last one that
happened. So when a ball track is supplied the window is extended until the ball
demonstrably leaves, which is what actually ends the moment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from polyfut_v2.config import PipelineV2Config


@dataclass
class Hotspot:
    start_sec: float
    end_sec: float
    contact_times: list[float]  # source-video times folded into this hotspot

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec

    def to_dict(self) -> dict:
        return {
            "start_sec": round(self.start_sec, 3),
            "end_sec": round(self.end_sec, 3),
            "duration_sec": round(self.duration_sec, 3),
            "n_contacts": len(self.contact_times),
            "contact_times": [round(t, 3) for t in self.contact_times],
        }


BallTrack = list[tuple[float, float, float]]   # (source t_sec, x, y), time-sorted


def ball_track_from_trajectory(traj) -> BallTrack:
    """(t_sec, x, y) for every sample the ball was actually located in.

    Source ``t_sec``, not ``processed_sec``: hotspots are cut against the real
    video, and ``processed_sec`` is a compacted timeline with dead time removed.
    """
    out: BallTrack = []
    for s in (getattr(traj, "positions", None) or (lambda: []))():
        try:
            out.append((float(s.t_sec), float(s.x), float(s.y)))
        except (TypeError, ValueError, AttributeError):
            continue
    out.sort(key=lambda r: r[0])
    return out


def ball_departure_sec(
    ball_track: BallTrack,
    from_sec: float,
    *,
    leave_px: float,
    max_extend_sec: float,
) -> float | None:
    """When, after ``from_sec``, does the ball clearly leave where it was?

    Returns the moment the ball is first further than ``leave_px`` from its
    position at ``from_sec`` — the departure that actually ends a possession
    (the pass, the clearance, the tackle) — or None when that cannot be
    established: no track, no samples in the window, or the ball still hasn't
    left by ``max_extend_sec``.

    None means "don't extend", so a blind stretch of trajectory leaves today's
    fixed pad untouched rather than stretching a clip on a guess. The cap
    matters for the same reason: a ball that is never seen to leave is usually a
    tracking failure, not a 30-second dribble.
    """
    if not ball_track or leave_px <= 0 or max_extend_sec <= 0:
        return None
    origin = None
    for t, x, y in ball_track:
        if t < from_sec:
            continue
        if t > from_sec + max_extend_sec:
            break
        if origin is None:
            origin = (x, y)
            continue
        if math.hypot(x - origin[0], y - origin[1]) > leave_px:
            return float(t)
    return None


def assemble_hotspots(
    me_times: list[float],
    cfg: PipelineV2Config | None = None,
    *,
    duration_sec: float | None = None,
    ball_track: BallTrack | None = None,
) -> list[Hotspot]:
    """Turn confirmed touch times into merged, padded hotspot windows.

    ``ball_track`` is (source ``t_sec``, x, y) in analysed-frame pixels. When
    given, each window runs on until the ball leaves rather than stopping a
    fixed pad after the last touch that happened to be detected.
    """
    cfg = cfg or PipelineV2Config()
    times = sorted(t for t in me_times if t is not None)
    if not times:
        return []

    hotspots: list[Hotspot] = []
    cur_start = times[0] - cfg.hotspot_pad_before_sec
    cur_end = times[0] + cfg.hotspot_pad_after_sec
    cur_times = [times[0]]

    for t in times[1:]:
        w_start = t - cfg.hotspot_pad_before_sec
        w_end = t + cfg.hotspot_pad_after_sec
        # Merge if this window starts within gap_merge of the current end.
        if w_start - cur_end <= cfg.hotspot_gap_merge_sec:
            cur_end = max(cur_end, w_end)
            cur_times.append(t)
        else:
            hotspots.append(Hotspot(cur_start, cur_end, cur_times))
            cur_start, cur_end, cur_times = w_start, w_end, [t]
    hotspots.append(Hotspot(cur_start, cur_end, cur_times))

    # Run each window on until the ball actually leaves, then re-pad. Done
    # before the min-zone/clamp pass so an extended window is still bounded by
    # the video.
    if ball_track and getattr(cfg, "hotspot_possession_enabled", False):
        leave_px = float(getattr(cfg, "hotspot_possession_leave_px", 120.0))
        max_ext = float(getattr(cfg, "hotspot_possession_max_extend_sec", 6.0))
        for h in hotspots:
            last = h.contact_times[-1]
            gone = ball_departure_sec(ball_track, last, leave_px=leave_px,
                                      max_extend_sec=max_ext)
            if gone is not None:
                # Include the departure itself — the final pass is the part the
                # old fixed pad was cutting off — then the normal trailing pad.
                h.end_sec = max(h.end_sec, gone + cfg.hotspot_pad_after_sec)

    # Enforce min zone length (grow symmetrically), then clamp to the video.
    out: list[Hotspot] = []
    for h in hotspots:
        start, end = h.start_sec, h.end_sec
        if (end - start) < cfg.hotspot_min_zone_sec:
            deficit = cfg.hotspot_min_zone_sec - (end - start)
            start -= deficit / 2.0
            end += deficit / 2.0
        start = max(0.0, start)
        if duration_sec is not None:
            end = min(end, duration_sec)
        out.append(Hotspot(start, end, h.contact_times))
    return out
