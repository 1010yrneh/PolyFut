"""Auto-detect the two team kit colours from a match video."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from polyfut_cv.color_classify import median_lab
from polyfut_cv.detect import DetectConfig, Detector, video_info

# region agent log
_DEBUG_LOG = Path(os.environ.get(
    "POLYFUT_DEBUG_LOG",
    str(Path.home() / ".cursor" / "debug-logs" / "debug-16e722.log"),
))


def _dbg_log(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    try:
        import time as _time
        rec = {
            "sessionId": "16e722",
            "runId": "team-colors",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(_time.time() * 1000),
        }
        with open(_DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass
# endregion


def _lab_to_bgr(lab: np.ndarray) -> tuple[int, int, int]:
    px = np.uint8([[[int(lab[0]), int(lab[1]), int(lab[2])]]])
    bgr = cv2.cvtColor(px, cv2.COLOR_LAB2BGR)
    b, g, r = (int(x) for x in bgr[0, 0])
    return b, g, r


def _lab_to_hsv(lab: np.ndarray) -> tuple[int, int, int]:
    px = np.uint8([[[int(lab[0]), int(lab[1]), int(lab[2])]]])
    bgr = cv2.cvtColor(px, cv2.COLOR_LAB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = (int(x) for x in hsv[0, 0])
    return h, s, v


def _is_referee_torso(h: int, s: int, v: int) -> bool:
    """Bright red/orange kit (referee) — not typical player yellow/black."""
    return (h <= 12 or h >= 168) and s >= 70 and v >= 50


def _is_neutral_kit(h: int, s: int, v: int) -> bool:
    """Pitch lines, gray kits, washed-out background on torso crop."""
    return s < 35 or (v > 200 and s < 55)


def _filter_player_torso_samples(
    labs: list[np.ndarray],
) -> tuple[list[np.ndarray], int]:
    """Drop referee-red and neutral samples before clustering team kits."""
    kept: list[np.ndarray] = []
    dropped = 0
    for lab in labs:
        h, s, v = _lab_to_hsv(lab)
        if _is_referee_torso(h, s, v) or _is_neutral_kit(h, s, v):
            dropped += 1
            continue
        kept.append(lab)
    return kept, dropped


def detect_team_colors(
    video_path: str,
    weights: str = "yolov8n.pt",
    *,
    n_samples: int = 16,
    imgsz: int = 640,
    conf: float = 0.25,
    max_players_per_frame: int = 12,
    proxy_path: str | None = None,
) -> list[dict]:
    """Return up to 2 team colours, most common first."""
    source = proxy_path or video_path
    fps, n_frames, w, h = video_info(source)
    if n_frames and n_frames > 0:
        idxs = [int(n_frames * (i + 0.5) / n_samples) for i in range(n_samples)]
    else:
        idxs = list(range(0, n_samples * 15, 15))

    det = Detector(DetectConfig(weights=weights, imgsz=imgsz, conf=conf, classes=(0,)))

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source}")

    frames: list[np.ndarray] = []
    try:
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok:
                frames.append(frame)
    finally:
        cap.release()

    labs: list[np.ndarray] = []
    batch_size = 8
    for i in range(0, len(frames), batch_size):
        chunk = frames[i : i + batch_size]
        for frame, (persons, _balls) in zip(chunk, det.predict_batch(chunk, imgsz=imgsz)):
            persons = sorted(persons, key=lambda d: d.conf, reverse=True)[:max_players_per_frame]
            for p in persons:
                lab = median_lab(frame, p.xyxy)
                if lab is not None:
                    labs.append(lab)

    raw_n = len(labs)
    filtered, dropped = _filter_player_torso_samples(labs)
    if len(filtered) >= 6:
        labs = filtered
    # else keep all samples — filter may remove too much on sparse footage

    if len(labs) < 2:
        # region agent log
        _dbg_log("H3", "team_colors.py:detect", "insufficient samples", {"raw_n": raw_n, "dropped": dropped})
        # endregion
        return [{"h": 0, "s": 0, "v": 200, "hex": "#c8c8c8",
                 "rgb": [200, 200, 200], "lab": [200.0, 128.0, 128.0], "count": len(labs)}]

    data = np.float32(np.vstack(labs))
    k = 3 if len(labs) >= 12 else 2
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _compactness, labels, centers = cv2.kmeans(
        data, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS
    )
    labels = labels.flatten()

    out: list[dict] = []
    for ci in range(k):
        c = centers[ci]
        count = int((labels == ci).sum())
        b, g, r = _lab_to_bgr(c)
        hh, ss, vv = _lab_to_hsv(c)
        out.append({
            "h": hh, "s": ss, "v": vv,
            "hex": f"#{r:02x}{g:02x}{b:02x}",
            "rgb": [r, g, b],
            "lab": [float(c[0]), float(c[1]), float(c[2])],
            "count": count,
        })
    # Drop referee-like cluster, then keep two largest kit clusters.
    out = [c for c in out if not _is_referee_torso(c["h"], c["s"], c["v"])]
    out.sort(key=lambda d: d["count"], reverse=True)
    out = out[:2]
    # region agent log
    _dbg_log(
        "H3",
        "team_colors.py:detect",
        "clusters",
        {
            "raw_n": raw_n,
            "filtered_n": len(labs),
            "dropped_ref_neutral": dropped,
            "teams": [{"hex": t["hex"], "h": t["h"], "s": t["s"], "count": t["count"]} for t in out],
        },
    )
    # endregion
    return out
