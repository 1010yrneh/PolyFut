"""Merge possession intervals into play buckets, pad, and write clip_segments.json.

Play buckets: quick successive touches within `gap_merge_sec` collapse into one
clickable range (default 7s) so the user reviews ~20-35 plays, not 40-70 blips.
Optional `action_triggers` per segment keep the raw touch timestamps for UI
micro-ticks without breaking the simple {start,end} contract.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ExportConfig:
    gap_merge_sec: float = 9.0   # play-bucket merge gap (group bursts of touches)
    pad_before_sec: float = 6.0  # recall-first: generous padding before possession
    pad_after_sec: float = 10.0


def merge_nearby(intervals: list[tuple[float, float]], gap_merge_sec: float) -> list[tuple[float, float]]:
    if not intervals:
        return []
    iv = sorted(intervals, key=lambda x: x[0])
    out = [list(iv[0])]
    for s, e in iv[1:]:
        if s <= out[-1][1] + gap_merge_sec:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(a, b) for a, b in out]


def pad_and_merge(
    intervals: list[tuple[float, float]],
    duration_sec: float,
    cfg: ExportConfig,
) -> list[tuple[float, float]]:
    padded: list[tuple[float, float]] = []
    for s, e in intervals:
        ns = max(0.0, s - cfg.pad_before_sec)
        ne = min(duration_sec, e + cfg.pad_after_sec) if duration_sec > 0 else e + cfg.pad_after_sec
        if ne > ns:
            padded.append((ns, ne))
    # Merge again after padding so overlapping padded windows fuse.
    return merge_nearby(padded, cfg.gap_merge_sec)


def build_segments(
    raw_intervals: list[tuple[float, float]],
    duration_sec: float,
    cfg: ExportConfig | None = None,
    *,
    touch_times: list[float] | None = None,
) -> list[dict[str, Any]]:
    cfg = cfg or ExportConfig()
    merged = merge_nearby(raw_intervals, cfg.gap_merge_sec)
    windows = pad_and_merge(merged, duration_sec, cfg)
    segments: list[dict[str, Any]] = []
    for s, e in windows:
        seg: dict[str, Any] = {"start": round(s, 3), "end": round(e, 3)}
        if touch_times:
            triggers = [round(t, 3) for t in touch_times if s <= t <= e]
            if triggers:
                seg["action_triggers"] = triggers
        segments.append(seg)
    return segments


def write_clip_segments_json(
    out_path: str | Path,
    segments: list[dict[str, Any]],
    *,
    source_video: str | None = None,
    partial: bool = False,
) -> None:
    payload: dict[str, Any] = {"version": 1, "segments": segments}
    if source_video:
        payload["source_video"] = str(source_video)
    if partial:
        payload["partial"] = True
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
