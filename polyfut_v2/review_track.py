"""Track the reviewed player across a montage clip so the "was this you?" ring
follows them instead of sitting at a fixed spot.

At each contact we know the player's box at the touch instant (the montage
``crop``). To follow them through the short review window we run a cheap
Lucas-Kanade median-flow tracker outward from the touch frame in both
directions — no extra model, no contrib build, fast enough to compute on demand
during review. Output is a tracklet in normalized coords (like the seed nodes),
so the frontend can move the ring by interpolating on the video's currentTime.
"""

from __future__ import annotations

import cv2
import numpy as np

from polyfut_video.pipeline.decode import probe_video

_LK = dict(winSize=(21, 21), maxLevel=3,
           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))


def _center_size(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0, max(4.0, x2 - x1), max(4.0, y2 - y1)


def _shift_one(prev_gray, gray, cx, cy, w, h):
    """Median optical-flow shift of the box (cx,cy,w,h) from prev→gray.
    Returns (cx, cy) or None if tracking is lost."""
    x1 = int(max(0, cx - w / 2)); y1 = int(max(0, cy - h / 2))
    x2 = int(min(prev_gray.shape[1], cx + w / 2)); y2 = int(min(prev_gray.shape[0], cy + h / 2))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    roi = prev_gray[y1:y2, x1:x2]
    pts = cv2.goodFeaturesToTrack(roi, maxCorners=40, qualityLevel=0.01,
                                  minDistance=4, blockSize=5)
    if pts is None or len(pts) < 4:
        return None
    p0 = pts.reshape(-1, 2) + np.array([x1, y1], dtype=np.float32)
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **_LK)
    # forward-backward consistency check to drop unreliable points
    p0r, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, p1, None, **_LK)
    good = (st.reshape(-1) == 1) & (st2.reshape(-1) == 1)
    fb = np.abs(p0r - p0).reshape(-1, 2).max(axis=1)
    good &= fb < 2.0
    if good.sum() < 3:
        return None
    dxy = (p1 - p0)[good]
    dx = float(np.median(dxy[:, 0])); dy = float(np.median(dxy[:, 1]))
    return cx + dx, cy + dy


def build_review_track(
    video_path: str,
    crop: list[float],
    t_sec: float,
    clip_start_sec: float,
    clip_end_sec: float,
    target_width: int = 640,
    sample_every: int = 1,
) -> dict | None:
    """Track the player from ``crop`` (at ``t_sec``) across the review window.

    ``crop`` is [x1,y1,x2,y2] in ``target_width``-resized space (same space the
    montage stores). Returns a tracklet of absolute-time normalized points, or
    None if the window can't be read.
    """
    try:
        info = probe_video(video_path)
    except Exception:  # noqa: BLE001 — unreadable/missing → caller falls back
        return None
    fps = float(info.get("fps") or 25.0)
    start_f = int(max(0.0, clip_start_sec) * fps)
    end_f = int(clip_end_sec * fps)
    anchor_f = int(round(t_sec * fps))
    if end_f <= start_f:
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    grays: list[np.ndarray] = []
    base_w = base_h = 0
    n = end_f - start_f + 1
    for _ in range(n):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        h0, w0 = frame.shape[:2]
        if w0 != target_width:
            scale = target_width / float(w0)
            frame = cv2.resize(frame, (target_width, int(round(h0 * scale))))
        base_w, base_h = frame.shape[1], frame.shape[0]
        grays.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()
    if len(grays) < 2:
        return None

    anchor = min(max(0, anchor_f - start_f), len(grays) - 1)
    cx, cy, w, h = _center_size(crop)
    centers: dict[int, tuple[float, float]] = {anchor: (cx, cy)}

    # walk forward, then backward, from the anchor frame
    for rng in (range(anchor + 1, len(grays)), range(anchor - 1, -1, -1)):
        ccx, ccy = cx, cy
        prev = anchor
        for i in rng:
            res = _shift_one(grays[prev], grays[i], ccx, ccy, w, h)
            if res is None:
                break
            ccx, ccy = res
            ccx = min(max(0.0, ccx), base_w); ccy = min(max(0.0, ccy), base_h)
            centers[i] = (ccx, ccy)
            prev = i

    nw = round(w / base_w, 4); nh = round(h / base_h, 4)
    points = []
    for i in sorted(centers):
        px, py = centers[i]
        points.append({
            "t_sec": round((start_f + i) / fps, 3),
            "nx": round(px / base_w, 4), "ny": round(py / base_h, 4),
            "nw": nw, "nh": nh,
        })
    return {"fps": fps, "base_w": base_w, "base_h": base_h, "points": points}
