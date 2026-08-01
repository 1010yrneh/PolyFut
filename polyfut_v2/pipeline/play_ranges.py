"""Playing-time windows: the on-pitch ranges the user declares up front.

Everything downstream — seed-clip moments, the shuffle, decode bounds, live
shots, the candidate cap — is confined to these ranges. The motivating failure:
on a 111-minute video where the user subbed on at 63', seed clips landed at
11/39/66/94 min and the shuffle roamed to 12 and 51 min, so the appearance
gallery and the identity anchor were seeded from *another player*. Every
auto-accepted touch was then wrong.

A range is ``(start_sec, end_sec)`` in **video time**, and timestamps stay in
video time everywhere — nothing here re-bases the clock. The empty list means
"unrestricted" (whole match), which is the default and preserves the old
behaviour exactly; :func:`resolve` turns that into a single full-duration range
so the rest of the module never has to special-case it.

Kept pure (no video, no model) so the mapping and intersection rules can be
unit-tested directly, per the project's verification norms.
"""

from __future__ import annotations

import hashlib

Range = tuple[float, float]

# Two ranges closer than this are merged rather than left as a hairline gap —
# handles the user dragging two periods flush against each other.
_MERGE_EPS = 0.5


def normalize_ranges(raw, duration_sec: float | None = None) -> list[Range]:
    """Clean whatever the client sent into sorted, merged, in-bounds ranges.

    Accepts ``[[start, end], ...]`` or ``[{"start": s, "end": e}, ...]``.
    Unparseable entries, zero/negative-length ranges, and ranges entirely
    outside the video are dropped rather than raising — a malformed window must
    degrade to "whole match", never to a failed analysis.
    """
    out: list[Range] = []
    for item in raw or []:
        try:
            if isinstance(item, dict):
                start = float(item.get("start", item.get("start_sec")))
                end = float(item.get("end", item.get("end_sec")))
            else:
                start, end = float(item[0]), float(item[1])
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        if start != start or end != end:      # NaN
            continue
        if end < start:
            start, end = end, start
        start = max(0.0, start)
        if duration_sec and duration_sec > 0:
            start = min(start, float(duration_sec))
            end = min(end, float(duration_sec))
        if end - start <= 0:
            continue
        out.append((round(start, 3), round(end, 3)))

    out.sort()
    merged: list[Range] = []
    for start, end in out:
        if merged and start <= merged[-1][1] + _MERGE_EPS:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def resolve(ranges: list[Range] | None, duration_sec: float) -> list[Range]:
    """The ranges to actually use: the whole video when none were declared.

    Every other function in this module expects a *resolved* list, so callers
    convert once at the boundary and the "no window set" case disappears.
    """
    if ranges:
        return list(ranges)
    return [(0.0, float(duration_sec))] if duration_sec > 0 else []


def is_whole_match(ranges: list[Range] | None, duration_sec: float,
                   tol: float = 1.0) -> bool:
    """True when the window covers (essentially) the entire video — i.e. the
    user skipped the step or dragged the handles to the ends. Used to keep the
    warning banner and the cache key quiet in the no-op case."""
    if not ranges:
        return True
    if len(ranges) != 1 or duration_sec <= 0:
        return False
    start, end = ranges[0]
    return start <= tol and end >= duration_sec - tol


def total_seconds(ranges: list[Range] | None) -> float:
    """On-pitch seconds across all ranges (0.0 for an unresolved empty list)."""
    return sum(max(0.0, end - start) for start, end in (ranges or []))


def contains(ranges: list[Range] | None, t_sec: float) -> bool:
    """Is ``t_sec`` inside any range? An empty list is unrestricted → True."""
    if not ranges:
        return True
    return any(start <= t_sec <= end for start, end in ranges)


def envelope(ranges: list[Range] | None) -> tuple[float, float] | None:
    """(first start, last end) — the span a single sequential decode must cover.

    Decoding can only seek once and read forward, so multi-range windows still
    stream through the interior gaps; the live-shot intersection is what stops
    those gap frames from costing detection work (the tracker's cursor skips
    non-live frames with a cheap ``grab()``).
    """
    if not ranges:
        return None
    return (min(s for s, _ in ranges), max(e for _, e in ranges))


def pad_ranges(ranges: list[Range] | None, pad_sec: float,
               duration_sec: float | None = None) -> list[Range]:
    """Widen each range by ``pad_sec`` on both sides, then re-merge.

    Recall-safety: the user drags handles by eye against a scrubbing preview,
    so the declared edges are approximate. Padding means a touch a few seconds
    either side of a guessed substitution time still gets analysed. Padding is
    for the *pipeline* only — seed moments stay strictly inside the declared
    ranges, because a seed clip from a moment the user wasn't on the pitch is
    exactly the contamination this feature exists to prevent.
    """
    if not ranges or pad_sec <= 0:
        return list(ranges or [])
    widened = [
        [max(0.0, start - pad_sec),
         (end + pad_sec if not duration_sec else min(float(duration_sec), end + pad_sec))]
        for start, end in ranges
    ]
    return normalize_ranges(widened, duration_sec)


def intersect_shots(shots: list[dict], ranges: list[Range] | None) -> list[dict]:
    """Clip shot segments to the ranges: fully-inside shots pass through, an
    overlapping shot is trimmed, a shot straddling a gap is split in two, and a
    shot entirely outside is dropped.

    Every other key on the shot dict (``label`` and friends) is carried onto
    each produced piece, so this stays transparent to the shot filter's
    downstream consumers.
    """
    if not ranges:
        return list(shots or [])
    out: list[dict] = []
    for shot in shots or []:
        try:
            s0 = float(shot["start_sec"])
            e0 = float(shot["end_sec"])
        except (KeyError, TypeError, ValueError):
            continue
        for r_start, r_end in ranges:
            start, end = max(s0, r_start), min(e0, r_end)
            if end > start:
                piece = dict(shot)
                piece["start_sec"] = round(start, 3)
                piece["end_sec"] = round(end, 3)
                out.append(piece)
    out.sort(key=lambda s: s["start_sec"])
    return out


def time_at_fraction(ranges: list[Range] | None, frac: float) -> float:
    """Map ``frac`` in [0, 1] onto the *union* of on-pitch time.

    This is what keeps the four seed slots and the golden-ratio shuffle inside
    the window. The mapping is cumulative, not per-range: fraction 0.5 is the
    halfway point of total playing time, which may sit in the second range. So
    the existing spread logic (fixed fractions, then a low-discrepancy hop)
    carries over unchanged and still spreads evenly across everything the user
    actually played.
    """
    if not ranges:
        return 0.0
    total = total_seconds(ranges)
    if total <= 0:
        return float(ranges[0][0])
    target = max(0.0, min(1.0, float(frac))) * total
    for start, end in ranges:
        span = end - start
        if target <= span:
            return round(start + target, 3)
        target -= span
    return round(float(ranges[-1][1]), 3)   # frac == 1.0 lands on the final edge


def clamp_to_ranges(ranges: list[Range] | None, t_sec: float) -> float:
    """Nearest in-range timestamp to ``t_sec`` — used as a last resort when a
    widening search finds nothing acceptable inside the window."""
    if not ranges:
        return t_sec
    if contains(ranges, t_sec):
        return t_sec
    best, best_d = t_sec, float("inf")
    for start, end in ranges:
        cand = min(max(t_sec, start), end)
        d = abs(cand - t_sec)
        if d < best_d:
            best_d, best = d, cand
    return round(best, 3)


def ranges_hash(ranges: list[Range] | None) -> str:
    """Short stable digest of a window, for cache keys.

    Seed clips are cached per (reroll, index); without a range component the
    whole-match prefetch and a later windowed clip collide on the same filename
    and the user gets served a clip from a moment they weren't playing. Whole
    match hashes to ``"all"`` so existing caches keep their meaning.
    """
    if not ranges:
        return "all"
    key = ";".join(f"{start:.1f}-{end:.1f}" for start, end in ranges)
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]
