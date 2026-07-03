"""Stage 9: interval generation and match-clock remapping."""

from __future__ import annotations

from typing import Any


def _sec_to_hms(seconds: float) -> str:
    s = max(0.0, float(seconds))
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = int(s % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"


def _hms_to_sec(hms: str) -> float:
    parts = hms.strip().split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return float(parts[0])


def _build_removed_timeline(removed_segments: list[dict]) -> list[tuple[float, float, str]]:
    """Sorted (start, end, reason) on source video timeline."""
    segs = [
        (float(s["start_sec"]), float(s["end_sec"]), s.get("reason", "discard"))
        for s in removed_segments
    ]
    return sorted(segs, key=lambda x: x[0])


def processed_to_match_clock(
    processed_sec: float,
    removed_segments: list[dict],
) -> float:
    """
    Map processed-stream time (live play only, sequential) back to original video seconds.

    Walks the source timeline: live regions advance both clocks 1:1; removed regions
    advance only match clock.
    """
    removed = _build_removed_timeline(removed_segments)
    if not removed:
        return processed_sec

    proc_cursor = 0.0
    match_cursor = 0.0
    target = max(0.0, processed_sec)

    for start, end, _reason in removed:
        live_len = max(0.0, start - match_cursor)
        if proc_cursor + live_len >= target:
            return match_cursor + (target - proc_cursor)
        proc_cursor += live_len
        match_cursor = end  # skip removed [start, end)

    return match_cursor + (target - proc_cursor)


def generate_timestamps(
    possession_frames: list[dict],
    removed_segments: list[dict],
) -> list[dict]:
    """
    Collapse per-frame labels into intervals in match-clock time.

    Returns [{"start": "HH:MM:SS", "end": "HH:MM:SS", "team": "team_a"|"team_b"|"contested"}]
    """
    if not possession_frames:
        return []

    intervals: list[dict] = []
    cur_team = possession_frames[0].get("possession", "unknown")
    cur_proc_start = possession_frames[0].get("processed_sec", possession_frames[0].get("timestamp_sec", 0.0))
    cur_proc_end = cur_proc_start

    for rec in possession_frames[1:]:
        team = rec.get("possession", "unknown")
        proc_t = rec.get("processed_sec", rec.get("timestamp_sec", 0.0))
        if team != cur_team:
            if cur_team in ("team_a", "team_b", "contested"):
                intervals.append({
                    "processed_start": cur_proc_start,
                    "processed_end": cur_proc_end,
                    "team": cur_team,
                })
            cur_team = team
            cur_proc_start = proc_t
        cur_proc_end = proc_t

    if cur_team in ("team_a", "team_b", "contested"):
        intervals.append({
            "processed_start": cur_proc_start,
            "processed_end": cur_proc_end,
            "team": cur_team,
        })

    out: list[dict] = []
    for iv in intervals:
        m_start = processed_to_match_clock(iv["processed_start"], removed_segments)
        m_end = processed_to_match_clock(iv["processed_end"], removed_segments)
        if m_end <= m_start:
            m_end = m_start + 0.5
        out.append({
            "start": _sec_to_hms(m_start),
            "end": _sec_to_hms(m_end),
            "start_sec": round(m_start, 3),
            "end_sec": round(m_end, 3),
            "team": iv["team"],
        })
    return out


def _extend_zone(
    core_start: float,
    core_end: float,
    *,
    pad_before: float,
    pad_after: float,
    min_zone_sec: float,
    duration_sec: float | None,
) -> tuple[float, float]:
    """Pad possession core and enforce a minimum hotspot zone width."""
    ns = max(0.0, core_start - pad_before)
    ne = core_end + pad_after
    if ne - ns < min_zone_sec:
        mid = (core_start + core_end) / 2.0
        half = min_zone_sec / 2.0
        ns = max(0.0, mid - half)
        ne = mid + half
    if duration_sec is not None and duration_sec > 0:
        ne = min(duration_sec, ne)
    return ns, ne


def timeline_to_clip_segments(
    timeline: list[dict],
    my_team: str = "team_a",
    *,
    pad_before: float = 8.0,
    pad_after: float = 12.0,
    gap_merge: float = 10.0,
    min_zone_sec: float = 5.0,
    duration_sec: float | None = None,
) -> list[dict[str, Any]]:
    """
    Build touch-hotspot zones for the PolyFut player UI.

    Raw possession blips are merged when close together, padded with safety
    margins, and exported as highlighted seek-bar zones. Each zone keeps
    action_triggers (touch centroids) for gold ticks on the play line.
    """
    raw: list[tuple[float, float]] = []
    for iv in timeline:
        if iv.get("team") != my_team:
            continue
        s = float(iv.get("start_sec", _hms_to_sec(iv.get("start", "0:00"))))
        e = float(iv.get("end_sec", _hms_to_sec(iv.get("end", "0:00"))))
        if e <= s:
            e = s + 0.25
        raw.append((s, e))

    if not raw:
        return []

    raw.sort(key=lambda x: x[0])
    buckets: list[dict[str, Any]] = [{
        "core_start": raw[0][0],
        "core_end": raw[0][1],
        "cores": [raw[0]],
    }]
    for s, e in raw[1:]:
        if s <= buckets[-1]["core_end"] + gap_merge:
            buckets[-1]["core_end"] = max(buckets[-1]["core_end"], e)
            buckets[-1]["cores"].append((s, e))
        else:
            buckets.append({"core_start": s, "core_end": e, "cores": [(s, e)]})

    out: list[dict] = []
    for bucket in buckets:
        core_start = float(bucket["core_start"])
        core_end = float(bucket["core_end"])
        ns, ne = _extend_zone(
            core_start,
            core_end,
            pad_before=pad_before,
            pad_after=pad_after,
            min_zone_sec=min_zone_sec,
            duration_sec=duration_sec,
        )
        if ne <= ns:
            continue
        triggers = [round((s + e) / 2.0, 2) for s, e in bucket["cores"]]
        seg: dict[str, Any] = {
            "type": "hotspot",
            "start": round(ns, 2),
            "end": round(ne, 2),
            "core_start": round(core_start, 2),
            "core_end": round(core_end, 2),
            "action_triggers": triggers,
        }
        out.append(seg)
    return out
