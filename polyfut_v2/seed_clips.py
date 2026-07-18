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
from polyfut_v2.pipeline.color import hsv_distance, median_hsv, torso_hsv

CLIP_LEN_SEC = 3.0
ENHANCE_SCALE = 2


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
                tr["points"].append(_pt(fi, fps, bbox, cx, cy, w, h))
                tr["_cx"], tr["_cy"] = cx, cy
                if hsv is not None:
                    tr["_hsv"].append(hsv)
                    tr["_color"] = median_hsv(tr["_hsv"])
        for j, (bbox, cx, cy, hsv) in enumerate(dets):
            if j in used:
                continue
            hs = [hsv] if hsv is not None else []
            tracks.append({"_cx": cx, "_cy": cy,
                           "points": [_pt(fi, fps, bbox, cx, cy, w, h)],
                           "_hsv": hs, "_color": (median_hsv(hs) if hs else None)})
    out = []
    for i, tr in enumerate(t for t in tracks if len(t["points"]) >= 2):
        kit = median_hsv(tr["_hsv"]) if tr["_hsv"] else None
        out.append({
            "id": i, "points": tr["points"],
            "kit_hsv": None if kit is None else [round(float(v), 1) for v in kit],
        })
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
    write_browser_clip(enhanced, out_path, fps)

    dets_per_frame: list[tuple[int, list]] = []
    for i in range(0, len(enhanced), max(1, track_every)):
        players = player_detector.detect(enhanced[i], None)
        dets = []
        for pl in players:
            x1, y1, x2, y2 = pl.bbox
            # Kit colour of this player (None if the crop is too small/unreliable
            # → treated as "unknown" by the UI, which keeps it shown).
            hsv = torso_hsv(enhanced[i], pl.bbox, min_area=100)
            dets.append((pl.bbox, (x1 + x2) / 2.0, (y1 + y2) / 2.0, hsv))
        dets_per_frame.append((i, dets))

    tracklets = _track(dets_per_frame, ws, hs, fps)
    return {
        "fps": fps,
        "duration_sec": len(enhanced) / fps,
        "start_sec": round(start_sec, 3),   # original-video time of clip frame 0
        "width": ws, "height": hs,
        "tracklets": tracklets,
    }
