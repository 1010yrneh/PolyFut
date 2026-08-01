"""Track the reviewed player across a montage clip so the "was this you?" ring
follows them instead of sitting at a fixed spot.

At each contact we know the player's box at the touch instant (the montage
``player_bbox``). To follow them through the short review window we run a cheap
Lucas-Kanade median-flow tracker outward from the touch frame in both
directions — no extra model, no contrib build, fast enough to compute on demand
during review. Output is a tracklet in normalized coords (like the seed nodes),
so the frontend can move the ring by interpolating on the video's currentTime.

When ``player_bbox`` is missing we do *not* invent a faraway body to box — that
confused review by outlining someone who never touched the ball. Re-detect only
inside the pipeline's contact distance gate; otherwise return an empty track so
the UI hides the box until identity is certain again.
"""

from __future__ import annotations

import cv2
import numpy as np

from polyfut_video.pipeline.decode import probe_video
from polyfut_v2.config import PipelineV2Config
from polyfut_v2.pipeline.player_contacts import nearest_player

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



# Fallback body box when no real detection is available at the touch instant
# (colour/detector missed them that frame). A fraction of frame height, in a
# roughly person-shaped aspect ratio — deliberately NOT the montage `crop`
# (that's a fixed 240x240 zoom window centered on the *ball*, many times a
# real player's size, and was the previous, incorrect fallback).
def build_review_track(
    video_path: str,
    crop: list[float],
    t_sec: float,
    clip_start_sec: float,
    clip_end_sec: float,
    target_width: int = 640,
    sample_every: int = 1,
    player_bbox: list[float] | None = None,
    detector=None,
    cfg: PipelineV2Config | None = None,
    frame_index: int | None = None,
) -> dict | None:
    """Track the player across the review window, anchored on their body box at
    the touch instant.

    ``crop`` is [x1,y1,x2,y2] in ``target_width``-resized space (same space the
    montage stores) and is used only as a fallback anchor point — it's a fixed
    zoom window centered on the *ball*, not the player.

    Anchor selection, best → worst:
      1. ``player_bbox`` — the real detection box attached to the contact.
      2. If that's missing and a ``detector`` is supplied, re-detect *only*
         within the same contact distance gate the pipeline used. Never reach
         farther: inventing a random nearby body was boxing the wrong person.
      3. If neither yields a player — return an empty track (``visible=False``).
         The review UI hides the box rather than painting a fake body on the ball.

    Prefer ``frame_index`` (the source frame the contact was read from) over
    ``round(t_sec * fps)`` so the box lands on the same frame as ``player_bbox``.

    When Lucas-Kanade loses lock, that direction stops emitting points — the UI
    hides the box until (if ever) points resume. No hold of the last position
    off-screen.
    """
    cfg = cfg or PipelineV2Config()
    try:
        info = probe_video(video_path)
    except Exception:  # noqa: BLE001 — unreadable/missing → caller falls back
        return None
    fps = float(info.get("fps") or 25.0)
    start_f = int(max(0.0, clip_start_sec) * fps)
    end_f = int(clip_end_sec * fps)
    if frame_index is not None:
        anchor_f = int(frame_index)
    else:
        anchor_f = int(round(t_sec * fps))
    if end_f <= start_f:
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    grays: list[np.ndarray] = []
    color_anchor: np.ndarray | None = None      # BGR anchor frame, for the detector fallback
    target_anchor_off = anchor_f - start_f
    base_w = base_h = 0
    n = end_f - start_f + 1
    for i in range(n):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        h0, w0 = frame.shape[:2]
        if w0 != target_width:
            scale = target_width / float(w0)
            frame = cv2.resize(frame, (target_width, int(round(h0 * scale))))
        base_w, base_h = frame.shape[1], frame.shape[0]
        grays.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if i == target_anchor_off:
            color_anchor = frame
    cap.release()
    if len(grays) < 2:
        return None

    anchor = min(max(0, anchor_f - start_f), len(grays) - 1)
    anchor_box = None
    if player_bbox and len(player_bbox) == 4 and \
            player_bbox[2] > player_bbox[0] and player_bbox[3] > player_bbox[1]:
        anchor_box = player_bbox                              # (1) the real contact box
    elif detector is not None and color_anchor is not None:
        # (2) No player was attached to this contact. Only accept a re-detect
        # inside the same gate the pipeline uses — a looser radius used to box
        # a random sideline body and call it "the toucher".
        bx, by, _, _ = _center_size(crop)
        max_dist = float(getattr(cfg, "contact_max_player_dist_px", 80.0) or 80.0)
        players = detector.detect(color_anchor, (bx, by))
        pl, _d = nearest_player(
            players, (bx, by), max_dist,
            min_height_px=cfg.player_min_height_px, min_aspect=cfg.player_min_aspect,
            human_min_aspect=getattr(cfg, "player_human_min_aspect", 1.30),
            human_max_aspect=getattr(cfg, "player_human_max_aspect", 5.50),
        )
        if pl is not None:
            anchor_box = pl.bbox

    if anchor_box is not None:
        cx, cy, w, h = _center_size(anchor_box)
    else:
        # No attributed / re-detected player — do not invent a body box on the
        # ball (that outlined the wrong person). Caller + UI hide the ring.
        return {
            "fps": fps, "base_w": base_w, "base_h": base_h,
            "points": [], "visible": False,
        }
    centers: dict[int, tuple[float, float]] = {anchor: (cx, cy)}

    # walk forward, then backward, from the anchor frame. On LK failure stop
    # that direction — do not hold the last box off-screen / onto another body.
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
    return {
        "fps": fps, "base_w": base_w, "base_h": base_h,
        "points": points, "visible": bool(points),
    }
