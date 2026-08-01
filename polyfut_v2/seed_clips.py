"""Stage 0 seed clips: enhanced 3-second clips with tracked player 'nodes'.

Instead of tapping a static frame, the seed step shows a short enhanced clip and
overlays a clickable node on every player that follows them as the clip plays.
The user clicks their node; the selected player's track across the clip becomes a
multi-crop appearance gallery (far stronger than a single tap).

This module produces, for one moment in the match:
  * an enhanced (upscaled+sharpened) 3s clip written to disk for playback, and
  * player tracklets in NORMALIZED coordinates (resolution-independent, so they
    map onto the original video for seed building) with per-frame timing.

The enhancer is pluggable (``enhance_frame``); today it is a fast classical
upscale (real SR models don't load in this OpenCV build).
"""

from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from polyfut_video.pipeline.decode import probe_video
from polyfut_v2.config import PipelineV2Config
from polyfut_v2.pipeline.color import (
    grass_fraction, hex_to_hsv, hsv_distance, jersey_hsv, median_hsv,
)
from polyfut_v2.pipeline import play_ranges as pr
from polyfut_v2.pipeline.player_contacts import _looks_like_ball, looks_like_person

CLIP_LEN_SEC = 3.0
ENHANCE_SCALE = 2

# Ball-shape rejection thresholds (see player_contacts._looks_like_ball) — the
# height floor is calibrated against native-resolution frames, so it's scaled
# up by ENHANCE_SCALE here since seed clips detect on the 2x-upscaled frame.
_BALL_SHAPE_CFG = PipelineV2Config()


def enhance_frame(bgr: np.ndarray, scale: int = ENHANCE_SCALE) -> np.ndarray:
    """Upscale + sharpen a frame to make small players clearer. Pluggable — a
    real super-resolution model can replace this later."""
    up = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
    blur = cv2.GaussianBlur(up, (0, 0), 1.0)
    return cv2.addWeighted(up, 1.6, blur, -0.6, 0)  # unsharp mask


def _ffmpeg_exe() -> str | None:
    return shutil.which("ffmpeg")


def write_browser_clip(frames: list[np.ndarray], out_path: str, fps: float) -> bool:
    """Write frames to a browser-playable MP4.

    OpenCV on this platform can only emit ``mp4v`` (MPEG-4 Part 2), which
    Chromium/pywebview can't decode — the <video> renders black. So encode
    H.264 (yuv420p, +faststart) via ffmpeg when available, piping raw frames in.
    Falls back to cv2 mp4v (unplayable in-browser but keeps offline tools working).
    Returns True if an H.264 clip was written.
    """
    if not frames:
        return False
    h, w = frames[0].shape[:2]
    w -= w % 2                      # H.264 yuv420p needs even dimensions
    h -= h % 2
    ff = _ffmpeg_exe()
    if ff:
        cmd = [
            ff, "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{w}x{h}", "-r", f"{fps:.4f}", "-i", "pipe:0",
            "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path,
        ]
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            for f in frames:
                proc.stdin.write(np.ascontiguousarray(f[:h, :w]).tobytes())
            proc.stdin.close()
            proc.wait(timeout=120)
            if proc.returncode == 0 and Path(out_path).exists():
                return True
        except Exception:  # noqa: BLE001 — fall back to cv2 below
            pass
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(f[:h, :w])
    vw.release()
    return False


_BASE_FRACTIONS = [0.10, 0.35, 0.60, 0.85]
_GOLDEN = 0.6180339887498949   # golden-ratio conjugate — low-discrepancy roam step


def moment_for_index(duration_sec: float, index: int, reroll: int = 0,
                     play_ranges: list[tuple[float, float]] | None = None) -> float:
    """Base timestamp (sec) for one seed-clip slot.

    ``reroll == 0`` is the initial pick: the four slots sit at fixed fractions
    spread across the match (early / mid / late). Each shuffle (``reroll`` 1, 2,
    3 …) **roams the whole match** via a golden-ratio hop, so successive
    shuffles land far apart across the full timeline instead of cycling through
    a couple of nearby offsets. Combined with the per-slot history passed in
    ``avoid_sec`` (see ``pick_best_moment_near``), every shuffle explores a
    genuinely new region rather than flip-flopping between the same clips.

    With ``play_ranges`` set, the same fractions are mapped onto the *union* of
    the user's on-pitch time instead of raw duration, so both the initial spread
    and every shuffle stay inside a window they were actually playing in.
    Without it the behaviour is byte-for-byte what it was."""
    if duration_sec <= 0:
        return 0.0
    index = max(0, min(index, len(_BASE_FRACTIONS) - 1))
    if reroll <= 0:
        frac = min(0.95, max(0.03, _BASE_FRACTIONS[index]))
    else:
        # Map the golden-ratio sequence into the valid [0.03, 0.95] band instead
        # of clamping to its edges — clamping would bunch every shuffle that
        # overshoots the ends onto the same 0.95/0.03 moment. Remapping keeps the
        # low-discrepancy spread intact across the whole usable timeline.
        seq = (_BASE_FRACTIONS[index] + reroll * _GOLDEN) % 1.0
        frac = 0.03 + seq * (0.95 - 0.03)
    if play_ranges:
        return round(pr.time_at_fraction(play_ranges, frac), 2)
    return round(frac * duration_sec, 2)


def default_moments(duration_sec: float, reroll: int = 0, n: int = 4,
                    play_ranges: list[tuple[float, float]] | None = None) -> list[float]:
    """Timestamps (sec) to sample the seed clips from. reroll shifts the whole
    set so the user can get a fresh selection if the defaults are unhelpful."""
    if duration_sec <= 0:
        return [0.0]
    return [moment_for_index(duration_sec, i, reroll, play_ranges)
            for i in range(min(n, len(_BASE_FRACTIONS)))]


# --- Moment quality: avoid landing a seed clip on a bad camera angle -------
# default_moments() above picks pure fixed fractions of match duration with no
# idea what's on screen there — a stoppage, a tight replay close-up, a corner
# flag, anything. This does a quick (single-frame, no enhancement) local
# search around each anchor for a nearby moment that actually has plenty of
# players, isn't a tight zoom/replay, and (when the user's kit colour is
# known) has that kit visible — so the four clips shown are ones a person can
# actually tag themselves in.

_ZOOM_MAX_HEIGHT_FRAC = 0.35     # a player taller than this fraction of frame height -> too zoomed in
_KIT_MATCH_MAX_DIST = 60.0       # same scale as the other kit-colour distance gates
_GOOD_ENOUGH_SCORE = 8.0         # stop searching once a candidate is clearly fine
_MIN_GRASS_FRACTION = 0.12       # below this, treat as crowd/bench/graphic, not on-pitch
_STATIC_DISPLACEMENT_PX = 6.0    # median player displacement below this across the clip window -> stationary (huddle/stoppage)
_CUT_HIST_DIST = 0.4             # Bhattacharyya distance above this between window start/end -> camera cut
# A usable tagging clip needs a real group of players, not a lone player or a
# handful — a single-player shot (or an empty stretch) is never chosen, even if
# it's the "least bad" nearby moment. pick_best_moment_near widens its search
# outward until a moment clears this bar.
_MIN_PLAYERS_FOR_CLIP = 6


def _frame_at(video_path: str, t_sec: float, target_width: int = 640) -> np.ndarray | None:
    """Decode a single resized frame at ``t_sec``, or None if unreadable."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(t_sec * fps)))
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        return None
    h0, w0 = frame.shape[:2]
    if w0 != target_width:
        scale = target_width / float(w0)
        frame = cv2.resize(frame, (target_width, max(1, int(round(h0 * scale)))))
    return frame


def _score_moment(
    frame: np.ndarray, player_detector, *, my_kit_hsv: np.ndarray | None,
) -> dict:
    """Quick-and-cheap quality score for one candidate seed-clip frame."""
    h = frame.shape[0]
    players = [
        p for p in player_detector.detect(frame, None)
        if looks_like_person(
            p.bbox,
            min_height_px=_BALL_SHAPE_CFG.player_min_height_px,
            min_aspect=_BALL_SHAPE_CFG.player_human_min_aspect,
            max_aspect=_BALL_SHAPE_CFG.player_human_max_aspect,
        ) and not _looks_like_ball(
            p.bbox,
            min_height_px=_BALL_SHAPE_CFG.player_min_height_px,
            min_aspect=_BALL_SHAPE_CFG.player_min_aspect,
        )
    ]
    n = len(players)
    if n == 0:
        return {"n_players": 0, "is_zoom": True, "kit_visible": False, "score": -1e9}

    heights = sorted(p.bbox[3] - p.bbox[1] for p in players)
    median_h_frac = heights[len(heights) // 2] / h
    is_zoom = median_h_frac > _ZOOM_MAX_HEIGHT_FRAC

    kit_visible = my_kit_hsv is None   # unknown kit -> don't gate on it
    if my_kit_hsv is not None:
        for p in players:
            hsv = jersey_hsv(frame, p.bbox, min_area=100)
            d = hsv_distance(hsv, my_kit_hsv)
            if d is not None and d <= _KIT_MATCH_MAX_DIST:
                kit_visible = True
                break

    score = float(n)
    if is_zoom:
        score -= 6.0      # heavy penalty, not a hard ban — still beats an empty frame
    if not kit_visible:
        score -= 3.0
    return {"n_players": n, "is_zoom": is_zoom, "kit_visible": kit_visible, "score": score}


def _median_player_displacement(
    frame_a: np.ndarray, frame_b: np.ndarray, player_detector,
) -> float | None:
    """Median nearest-neighbour displacement (px) between player detections in
    two frames a short time apart — a cheap proxy for "is anything actually
    happening". A stationary huddle/stoppage has near-zero displacement even
    though people are clearly detected; active play or a panning camera does
    not. None if either frame has no (non-ball-shaped) players."""
    def _centers(frame):
        return [
            p.center for p in player_detector.detect(frame, None)
            if looks_like_person(
                p.bbox,
                min_height_px=_BALL_SHAPE_CFG.player_min_height_px,
                min_aspect=_BALL_SHAPE_CFG.player_human_min_aspect,
                max_aspect=_BALL_SHAPE_CFG.player_human_max_aspect,
            ) and not _looks_like_ball(
                p.bbox,
                min_height_px=_BALL_SHAPE_CFG.player_min_height_px,
                min_aspect=_BALL_SHAPE_CFG.player_min_aspect,
            )
        ]
    a, b = _centers(frame_a), _centers(frame_b)
    if not a or not b:
        return None
    dists = sorted(min(math.hypot(ax - bx, ay - by) for bx, by in b) for ax, ay in a)
    return dists[len(dists) // 2]


def _hist_diff(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """0 (identical) .. 1 (totally different) colour-histogram dissimilarity —
    a cheap camera-cut proxy. Two frames from the same continuous broadcast
    shot (even mid-pan) look statistically similar; a cut to a replay, a
    different angle, or a crowd shot generally does not."""
    ha = cv2.calcHist([frame_a], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hb = cv2.calcHist([frame_b], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(ha, ha)
    cv2.normalize(hb, hb)
    return float(cv2.compareHist(ha, hb, cv2.HISTCMP_BHATTACHARYYA))


def _score_candidate(
    video_path: str,
    t_center: float,
    player_detector,
    *,
    my_kit_hsv: np.ndarray | None,
    clip_half_sec: float,
    target_width: int = 640,
    min_players_for_full: int = 1,
) -> dict | None:
    """Full quality score for one candidate seed-clip moment: player count,
    zoom, kit visibility (all on the center frame), plus whether the frame is
    dominated by pitch (not crowd/bench), whether players are actually moving
    across the window (not a stationary huddle/stoppage), and whether the
    camera cuts mid-window (replay/angle switch). None if the center frame
    can't be read at all.

    Candidates with fewer than ``min_players_for_full`` players skip the extra
    motion/cut/crowd frame decodes — they can't clear the player-count bar
    anyway, so scoring them cheaply keeps the widening search fast."""
    frame = _frame_at(video_path, t_center, target_width)
    if frame is None:
        return None
    result = _score_moment(frame, player_detector, my_kit_hsv=my_kit_hsv)
    result.update(is_crowd=result["n_players"] == 0, is_static=result["n_players"] == 0, is_cut=False)
    if result["n_players"] < max(1, min_players_for_full):
        return result

    is_crowd = grass_fraction(frame) < _MIN_GRASS_FRACTION

    # One decoded pair (the window's start/end frames) serves both the motion
    # and the cut check. A separate ±0.5 s pair used to feed the displacement
    # check, but each _frame_at is a fresh seek that re-decodes a whole GOP on
    # sparse-keyframe match video (~2 s each on a 90-min file), so two extra
    # decodes per candidate added seconds to every seed-clip search. Judging
    # "is anyone moving" across the full window is conservative: motion
    # accumulates over the longer span, so active play clears the same
    # threshold even more clearly, while a stationary huddle stays under it.
    frame_start = _frame_at(video_path, max(0.0, t_center - clip_half_sec), target_width)
    frame_end = _frame_at(video_path, t_center + clip_half_sec, target_width)
    if frame_end is None:
        # Window end past EOF — compare start vs. the (already decoded) center
        # frame so the motion/cut checks still run on a shorter span.
        frame_end = frame
    is_static = False
    if frame_start is not None and frame_end is not None:
        disp = _median_player_displacement(frame_start, frame_end, player_detector)
        is_static = disp is not None and disp < _STATIC_DISPLACEMENT_PX

    is_cut = (frame_start is not None and frame_end is not None
              and _hist_diff(frame_start, frame_end) > _CUT_HIST_DIST)

    score = result["score"]
    if is_crowd:
        score -= 8.0    # heaviest penalty — a crowd/bench shot has essentially no value
    if is_static:
        score -= 5.0
    if is_cut:
        score -= 5.0
    result.update(is_crowd=is_crowd, is_static=is_static, is_cut=is_cut, score=score)
    return result


def _search_offsets(radius_sec: float, step_sec: float = 5.0) -> list[float]:
    """Offsets from an anchor, ordered by increasing distance: 0, +5, -5, +10,
    -10, ... out to ``radius_sec``. Searching nearest-first keeps a chosen
    moment close to its intended spread position across the match."""
    out = [0.0]
    d = step_sec
    while d <= radius_sec + 1e-6:
        out.append(d)
        out.append(-d)
        d += step_sec
    return out


def _qualifies(result: dict, min_players: int) -> bool:
    """A moment is good enough to tag in only if it has a real group of players
    and none of the bad-scene flags fired."""
    return (result["n_players"] >= min_players
            and not result.get("is_zoom", False)
            and not result.get("is_crowd", False)
            and not result.get("is_static", False)
            and not result.get("is_cut", False))


def pick_best_moment_near(
    video_path: str,
    t_center: float,
    player_detector,
    duration_sec: float,
    *,
    my_kit_hex: str | None = None,
    target_width: int = 640,
    clip_half_sec: float = CLIP_LEN_SEC / 2,
    min_players: int = _MIN_PLAYERS_FOR_CLIP,
    search_radius_sec: float | None = None,
    avoid_sec: tuple[float, ...] = (),
    min_separation_sec: float = 8.0,
    play_ranges: list[tuple[float, float]] | None = None,
) -> float:
    """Nudge ``t_center`` to the nearest timestamp that's actually good for seed
    tagging — a real group of players (``min_players``+), on the pitch, moving,
    with no mid-clip cut. Searches outward from the anchor and stops at the
    first qualifying moment, widening progressively when the local area is bad
    (a stoppage, replays, lone-player shots) until one is found.

    ``avoid_sec`` are moments already chosen for the other seed clips; a
    candidate within ``min_separation_sec`` of any of them is skipped so two
    clips can't end up on the same passage of play (which produced identical
    'clip 3 == clip 4' content). Only if *nothing* qualifies within the whole
    search radius does it fall back to the best-scoring moment seen (preferring
    one that doesn't collide), so it never returns nothing.

    ``play_ranges`` confines the widening search to the user's on-pitch time:
    out-of-window offsets are skipped outright (the search simply keeps
    widening past them), and even the never-qualified fallback is clamped back
    inside. Unlike the pipeline, seed moments are **not** padded — a clip from a
    moment the user wasn't on the pitch is what seeds the wrong player's
    appearance gallery, the exact failure this window exists to stop."""
    my_kit_hsv = hex_to_hsv(my_kit_hex)
    if search_radius_sec is None:
        # Cover a wide neighbourhood, but bounded so a pathological all-bad
        # stretch can't search forever. Scaled by on-pitch time, not raw
        # duration, so a short window doesn't get a radius that's almost
        # entirely skipped as out-of-range.
        span = pr.total_seconds(play_ranges) if play_ranges else duration_sec
        search_radius_sec = min(180.0, max(20.0, span / 2.0))

    def _far_from_others(t: float) -> bool:
        return all(abs(t - a) >= min_separation_sec for a in avoid_sec)

    best_t, best_score = pr.clamp_to_ranges(play_ranges, t_center), -1e9
    for off in _search_offsets(search_radius_sec):
        t = t_center + off
        if t < 0 or t > duration_sec:
            continue
        if not pr.contains(play_ranges, t):
            continue
        result = _score_candidate(
            video_path, t, player_detector,
            my_kit_hsv=my_kit_hsv, clip_half_sec=clip_half_sec,
            target_width=target_width, min_players_for_full=min_players,
        )
        if result is None:
            continue
        far = _far_from_others(t)
        if far and _qualifies(result, min_players):
            return round(t, 2)          # first clean, well-populated, distinct moment wins
        # Fallback ranking: a collision is a last resort, so a distinct moment
        # always outranks a colliding one regardless of raw score.
        eff = result["score"] - (0.0 if far else 1e6)
        if eff > best_score:
            best_score, best_t = eff, t
    return round(best_t, 2)             # nothing qualified anywhere → least-bad, distinct if possible


def _merge_broken_tracks(
    tracks: list[dict], w: int, h: int,
    max_dist_frac: float = 0.08, color_max_dist: float = 60.0,
    max_gap_sec: float = 0.5,
) -> list[dict]:
    """Reconnect track fragments that are almost certainly the same on-screen
    player after a brief break, rather than a new person.

    The frame-to-frame matching in ``_track`` has no memory once it fails to
    re-associate a detection for one sampled frame — a single occlusion, a
    colour read that briefly crosses ``color_max_dist``, or one jump past
    ``max_dist`` all silently end that track and start an entirely new one the
    moment the same player is re-detected nearby. Over a whole clip this
    fragments one real player into several tags clustered in the same spot
    ("detection is all over the place"). This pass merges a track into an
    earlier one that ended shortly before it started, at a nearby position
    (more drift allowed the longer the gap), with a compatible kit colour —
    the same colour-lock-then-nearest rule ``_track`` already uses, just
    applied across the gap between fragments instead of between frames.
    """
    if len(tracks) < 2:
        return tracks

    ordered = sorted(tracks, key=lambda t: t["points"][0]["t"])
    chains: list[dict] = []
    for tr in ordered:
        t0 = tr["points"][0]["t"]
        x0 = tr["points"][0]["nx"] * w
        y0 = tr["points"][0]["ny"] * h
        col = tr.get("_color")

        best, best_d = None, 1e9
        for ch in chains:
            gap = t0 - ch["points"][-1]["t"]
            if gap <= 0 or gap > max_gap_sec:
                continue
            ccol = ch.get("_color")
            if col is not None and ccol is not None:
                cd = hsv_distance(col, ccol)
                if cd is not None and cd > color_max_dist:
                    continue
            d = math.hypot(x0 - ch["_cx"], y0 - ch["_cy"])
            allowed = max_dist_frac * w * (1.0 + gap / max_gap_sec)
            if d > allowed:
                continue
            if d < best_d:
                best_d, best = d, ch

        if best is not None:
            best["points"].extend(tr["points"])
            best["_cx"], best["_cy"] = tr["_cx"], tr["_cy"]
            best["_hsv"].extend(tr["_hsv"])
            if best["_hsv"]:
                best["_color"] = median_hsv(best["_hsv"])
        else:
            chains.append(tr)
    return chains


def _track(dets_per_frame: list[tuple[int, list]], w: int, h: int, fps: float,
           max_dist_frac: float = 0.08, color_max_dist: float = 60.0) -> list[dict]:
    """Colour-locked nearest-centroid tracking over sampled frames → tracklets in
    normalized coords, each tagged with a median kit colour.

    A tag is hard-locked to one kit colour: matching a track to a detection first
    rejects any detection whose (known) colour is clearly different from the
    track's established colour, THEN takes the nearest survivor. This stops a tag
    from jumping between, say, a yellow and a black player when they cross —
    same-colour swaps can still happen (unavoidable without re-ID), but a
    cross-colour switch cannot. Detections with an unreadable colour fall back to
    position only, so a momentary bad frame doesn't drop the tag. Dets are
    (bbox, cx, cy, hsv)."""
    tracks: list[dict] = []
    max_dist = max_dist_frac * w
    for fi, dets in dets_per_frame:
        used: set[int] = set()
        for tr in tracks:
            tcol = tr.get("_color")
            best, bd = None, 1e9
            for j, (bbox, cx, cy, hsv) in enumerate(dets):
                if j in used:
                    continue
                # Colour lock FIRST: skip a detection of a clearly different kit.
                if tcol is not None and hsv is not None:
                    cd = hsv_distance(hsv, tcol)
                    if cd is not None and cd > color_max_dist:
                        continue
                d = math.hypot(cx - tr["_cx"], cy - tr["_cy"])
                if d < bd:
                    bd, best = d, j
            if best is not None and bd <= max_dist:
                bbox, cx, cy, hsv = dets[best]
                used.add(best)
                tr["points"].append(_pt(fi, fps, bbox, cx, cy, w, h, hsv))
                tr["_cx"], tr["_cy"] = cx, cy
                if hsv is not None:
                    tr["_hsv"].append(hsv)
                    tr["_color"] = median_hsv(tr["_hsv"])
        for j, (bbox, cx, cy, hsv) in enumerate(dets):
            if j in used:
                continue
            hs = [hsv] if hsv is not None else []
            tracks.append({"_cx": cx, "_cy": cy,
                           "points": [_pt(fi, fps, bbox, cx, cy, w, h, hsv)],
                           "_hsv": hs, "_color": (median_hsv(hs) if hs else None)})
    tracks = _merge_broken_tracks(tracks, w, h, max_dist_frac, color_max_dist)
    # Then collapse duplicates that ran side-by-side the whole time (the
    # detector emitting several boxes for one player) — the sequential merge
    # above cannot see those.
    tracks = _merge_concurrent_tracks(tracks, w, h, color_max_dist=color_max_dist)
    out = []
    for i, tr in enumerate(t for t in tracks if len(t["points"]) >= 2):
        kit = median_hsv(tr["_hsv"]) if tr["_hsv"] else None
        out.append({
            "id": i, "points": tr["points"],
            "kit_hsv": None if kit is None else [round(float(v), 1) for v in kit],
        })
    return out


def _merge_concurrent_tracks(
    tracks: list[dict], w: int, h: int,
    max_dist_frac: float = 0.035, color_max_dist: float = 60.0,
    min_shared: int = 3,
) -> list[dict]:
    """Collapse tracks that run *at the same time* on the same player.

    ``_merge_broken_tracks`` only rejoins fragments that are sequential (one
    ends, the next begins). It cannot touch the other duplicate mode: the
    detector emitting two overlapping boxes for one player on the *same*
    frames, which seeds two parallel tracks that coexist for the whole clip.
    Those show up as two tappable markers a few pixels apart — measured at 12
    of 13 markers on real footage before this pass existed.

    Two tracks are the same player when they share enough sampled frames, stay
    close on the *median* of those frames (a median, not a minimum, so two
    different players merely crossing paths are not merged), and their kit
    colours are compatible. The shorter track is absorbed into the longer one,
    contributing only the timestamps the longer one is missing.
    """
    def _key(t: float) -> float:
        return round(t, 3)

    changed = True
    while changed and len(tracks) > 1:
        changed = False
        for i in range(len(tracks)):
            for j in range(i + 1, len(tracks)):
                pa = {_key(p["t"]): p for p in tracks[i]["points"]}
                pb = {_key(p["t"]): p for p in tracks[j]["points"]}
                shared = set(pa) & set(pb)
                if len(shared) < min_shared:
                    continue
                dists = sorted(
                    math.hypot((pa[t]["nx"] - pb[t]["nx"]) * w,
                               (pa[t]["ny"] - pb[t]["ny"]) * h)
                    for t in shared
                )
                if dists[len(dists) // 2] > max_dist_frac * w:
                    continue
                ca, cb = tracks[i].get("_color"), tracks[j].get("_color")
                if ca is not None and cb is not None:
                    cd = hsv_distance(ca, cb)
                    if cd is not None and cd > color_max_dist:
                        continue   # different kits -> genuinely different people

                keep, drop = (i, j) if len(tracks[i]["points"]) >= len(tracks[j]["points"]) else (j, i)
                a, b = tracks[keep], tracks[drop]
                have = {_key(p["t"]) for p in a["points"]}
                a["points"].extend(p for p in b["points"] if _key(p["t"]) not in have)
                a["points"].sort(key=lambda p: p["t"])
                a["_hsv"].extend(b.get("_hsv", []))
                if a["_hsv"]:
                    a["_color"] = median_hsv(a["_hsv"])
                tracks.pop(drop)
                changed = True
                break
            if changed:
                break
    return tracks


def _filter_short_tracklets(
    tracklets: list[dict], *, min_duration_sec: float = 0.5, min_points: int = 4,
) -> list[dict]:
    """Drop tracklets too brief to be a trustworthy tag before they ever reach
    the seed UI as a tappable marker.

    A tracklet that only exists for a fraction of a second (a couple of
    detections, then gone) is disproportionately likely to be a false-positive
    detection or an unrecoverable tracking blip rather than a real person the
    user could meaningfully identify — and on a busy broadcast frame these
    short-lived tags are most of what makes the screen feel cluttered ("all
    over the place"), on top of the real players. Both a minimum wall-clock
    span AND a minimum point count are required: a couple of points spanning a
    long gap (e.g. one detection, then nothing for a second, then one more) is
    just as unreliable as a tight cluster of points spanning almost no time.
    """
    out = []
    for tr in tracklets:
        pts = tr["points"]
        duration = pts[-1]["t"] - pts[0]["t"]
        if duration >= min_duration_sec and len(pts) >= min_points:
            out.append(tr)
    for i, tr in enumerate(out):
        tr["id"] = i
    return out


def _pt(fi: int, fps: float, bbox, cx: float, cy: float, w: int, h: int,
        hsv=None) -> dict:
    return {
        "t": round(fi / fps, 3),
        "nx": round(cx / w, 4), "ny": round(cy / h, 4),
        "nw": round((bbox[2] - bbox[0]) / w, 4),
        "nh": round((bbox[3] - bbox[1]) / h, 4),
        # Per-point kit colour (or None) so the UI can spot a mid-clip colour flip
        # (a likely tag switch) and warn / drop the clip.
        "c": None if hsv is None else [round(float(v), 1) for v in hsv],
    }


def build_seed_clip(
    video_path: str,
    t_center_sec: float,
    player_detector,
    out_path: str | None = None,
    *,
    clip_len: float = CLIP_LEN_SEC,
    # Detection cadence for the seed tracker. Sparser sampling means the
    # client interpolates the marker across a bigger time gap between real
    # detections; during a fast sprint or a fast camera pan the true path
    # over that gap isn't a straight line, so the marker visibly cuts corners
    # and over/undershoots ("speed lines" trailing the real player). 2 halves
    # that gap vs. the old default of 3, trading some extra detector calls
    # per seed clip for a noticeably tighter follow during fast motion.
    track_every: int = 2,
) -> dict | None:
    """Extract and track players in a 3s clip around ``t_center_sec``.

    Plays from the source upload in the browser — no upscale/sharpen pass and no
  separate encoded clip unless ``out_path`` is given (tests only). Returns
    metadata + tracklets, or None if the clip can't be read.
    """
    info = probe_video(video_path)
    fps = float(info.get("fps") or 25.0)
    start_sec = max(0.0, t_center_sec - clip_len / 2.0)
    start_frame = int(start_sec * fps)
    n_frames = max(1, int(clip_len * fps))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames: list[np.ndarray] = []
    for _ in range(n_frames):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        return None

    hs, ws = frames[0].shape[:2]
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        write_browser_clip(frames, out_path, fps)

    sample_idx = list(range(0, len(frames), max(1, track_every)))
    # One batched model call for all sampled frames when the detector supports
    # it — per-frame predict() calls pay ~38 rounds of Python/preprocess
    # overhead per clip, a measurable slice of the "Loading clip" wait.
    if hasattr(player_detector, "detect_many"):
        all_players = player_detector.detect_many([(frames[i], None) for i in sample_idx])
    else:
        all_players = [player_detector.detect(frames[i], None) for i in sample_idx]

    dets_per_frame: list[tuple[int, list]] = []
    for i, players in zip(sample_idx, all_players):
        dets = []
        for pl in players:
            if not looks_like_person(
                pl.bbox,
                min_height_px=_BALL_SHAPE_CFG.player_min_height_px,
                min_aspect=_BALL_SHAPE_CFG.player_human_min_aspect,
                max_aspect=_BALL_SHAPE_CFG.player_human_max_aspect,
            ) or _looks_like_ball(
                pl.bbox,
                min_height_px=_BALL_SHAPE_CFG.player_min_height_px,
                min_aspect=_BALL_SHAPE_CFG.player_min_aspect,
            ):
                continue
            x1, y1, x2, y2 = pl.bbox
            hsv = jersey_hsv(frames[i], pl.bbox, min_area=100)
            dets.append((pl.bbox, (x1 + x2) / 2.0, (y1 + y2) / 2.0, hsv))
        dets_per_frame.append((i, dets))

    tracklets = _filter_short_tracklets(_track(dets_per_frame, ws, hs, fps))
    return {
        "fps": fps,
        "duration_sec": len(frames) / fps,
        "start_sec": round(start_sec, 3),
        "width": ws, "height": hs,
        "tracklets": tracklets,
    }
