"""Stage 9: assemble confirmed 'me' touches into hotspot windows.

For each confirmed target touch, take a ±pad window, merge windows that fall
within the gap threshold (so a dribble / quick sequence becomes one hotspot),
enforce a minimum zone length, and clamp to the video. Reuses v1's hotspot
config semantics (pad / gap-merge / min-zone), fed cleaner single-player events.
"""

from __future__ import annotations

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


def assemble_hotspots(
    me_times: list[float],
    cfg: PipelineV2Config | None = None,
    *,
    duration_sec: float | None = None,
) -> list[Hotspot]:
    """Turn confirmed touch times into merged, padded hotspot windows."""
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
