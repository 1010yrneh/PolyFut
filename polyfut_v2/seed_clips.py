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
from pathlib import Path

import cv2
import numpy as np

from polyfut_video.pipeline.decode import probe_video

CLIP_LEN_SEC = 3.0
ENHANCE_SCALE = 2


def enhance_frame(bgr: np.ndarray, scale: int = ENHANCE_SCALE) -> np.ndarray:
    """Upscale + sharpen a frame to make small players clearer. Pluggable — a
    real super-resolution model can replace this later."""
    up = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
    blur = cv2.GaussianBlur(up, (0, 0), 1.0)
    return cv2.addWeighted(up, 1.6, blur, -0.6, 0)  # unsharp mask


def default_moments(duration_sec: float, reroll: int = 0, n: int = 4) -> list[float]:
    """Timestamps (sec) to sample the seed clips from. reroll shifts the set so
    the user can get a fresh selection if the defaults are unhelpful."""
    if duration_sec <= 0:
        return [0.0]
    base = [0.10, 0.35, 0.60, 0.85][:n]
    # Each reroll nudges the fractions by a rotating offset within a safe band.
    off = ((reroll % 5) * 0.07)
    out = []
    for f in base:
        frac = min(0.95, max(0.03, f + off if f + off < 0.95 else f - off))
        out.append(round(frac * duration_sec, 2))
    return out


def _track(dets_per_frame: list[tuple[int, list]], w: int, h: int, fps: float,
           max_dist_frac: float = 0.08) -> list[dict]:
    """Nearest-centroid tracking over sampled frames → tracklets in normalized
    coords. Good enough for a 3s clip; no re-ID needed."""
    tracks: list[dict] = []
    max_dist = max_dist_frac * w
    for fi, dets in dets_per_frame:
        used: set[int] = set()
        for tr in tracks:
            best, bd = None, 1e9
            for j, (bbox, cx, cy) in enumerate(dets):
                if j in used:
                    continue
                d = math.hypot(cx - tr["_cx"], cy - tr["_cy"])
                if d < bd:
                    bd, best = d, j
            if best is not None and bd <= max_dist:
                bbox, cx, cy = dets[best]
                used.add(best)
                tr["points"].append(_pt(fi, fps, bbox, cx, cy, w, h))
                tr["_cx"], tr["_cy"] = cx, cy
        for j, (bbox, cx, cy) in enumerate(dets):
            if j in used:
                continue
            tracks.append({"_cx": cx, "_cy": cy,
                           "points": [_pt(fi, fps, bbox, cx, cy, w, h)]})
    out = []
    for i, tr in enumerate(t for t in tracks if len(t["points"]) >= 2):
        out.append({"id": i, "points": tr["points"]})
    return out


def _pt(fi: int, fps: float, bbox, cx: float, cy: float, w: int, h: int) -> dict:
    return {
        "t": round(fi / fps, 3),
        "nx": round(cx / w, 4), "ny": round(cy / h, 4),
        "nw": round((bbox[2] - bbox[0]) / w, 4),
        "nh": round((bbox[3] - bbox[1]) / h, 4),
    }


def build_seed_clip(
    video_path: str,
    t_center_sec: float,
    player_detector,
    out_path: str,
    *,
    clip_len: float = CLIP_LEN_SEC,
    track_every: int = 3,
) -> dict | None:
    """Extract, enhance, and track players in a 3s clip around ``t_center_sec``.

    Writes the enhanced clip to ``out_path`` and returns metadata + tracklets.
    Returns None if the clip can't be read.
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
    enhanced: list[np.ndarray] = []
    for _ in range(n_frames):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        enhanced.append(enhance_frame(frame))
    cap.release()
    if not enhanced:
        return None

    hs, ws = enhanced[0].shape[:2]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (ws, hs))
    for e in enhanced:
        vw.write(e)
    vw.release()

    dets_per_frame: list[tuple[int, list]] = []
    for i in range(0, len(enhanced), max(1, track_every)):
        players = player_detector.detect(enhanced[i], None)
        dets = []
        for pl in players:
            x1, y1, x2, y2 = pl.bbox
            dets.append((pl.bbox, (x1 + x2) / 2.0, (y1 + y2) / 2.0))
        dets_per_frame.append((i, dets))

    tracklets = _track(dets_per_frame, ws, hs, fps)
    return {
        "fps": fps,
        "duration_sec": len(enhanced) / fps,
        "start_sec": round(start_sec, 3),   # original-video time of clip frame 0
        "width": ws, "height": hs,
        "tracklets": tracklets,
    }
