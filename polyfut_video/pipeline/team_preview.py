"""Quick kit-colour preview for the team picker (before full pipeline run)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

from polyfut_video.pipeline.decode import probe_video
from polyfut_video.pipeline.detection import DetectConfig, Detector
from polyfut_video.pipeline.team_classify import _hsv_feature, _torso_crop

# region agent log
_DEBUG_LOG = Path(os.environ.get(
    "POLYFUT_DEBUG_LOG",
    str(Path(__file__).resolve().parents[2] / ".cursor" / "debug-9e74f8.log"),
))


def _dbg_log(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    try:
        rec = {
            "sessionId": "9e74f8",
            "runId": "team-color-debug",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        _DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(_DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass
# endregion


def _resize_frame(frame: np.ndarray, target_width: int = 960) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= target_width:
        return frame
    scale = target_width / float(w)
    return cv2.resize(frame, (target_width, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)


def _crops_to_hex(crops: list[np.ndarray]) -> str | None:
    if not crops:
        return None
    pixels = np.vstack([c.reshape(-1, 3) for c in crops])
    med = np.median(pixels, axis=0)
    b, g, r = (int(np.clip(x, 0, 255)) for x in med)
    return f"#{r:02x}{g:02x}{b:02x}"


def _is_referee_crop(crop: np.ndarray) -> bool:
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = (int(x) for x in np.median(hsv.reshape(-1, 3), axis=0))
    return (h <= 12 or h >= 168) and s >= 70 and v >= 50


def _is_neutral_crop(crop: np.ndarray) -> bool:
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = (int(x) for x in np.median(hsv.reshape(-1, 3), axis=0))
    return s < 30 or (v > 210 and s < 50)


def _kmeans_two_kits(crops: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]] | None:
    """Force k=2 kit clusters; do not drop the smaller team (preview-only)."""
    if len(crops) < 4:
        return None

    feats = np.stack([_hsv_feature(c) for c in crops], axis=0).astype(np.float32)
    scaled = feats.copy()
    scaled[:, 0] *= 2.0  # hue weight

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _compactness, labels, _centers = cv2.kmeans(
        scaled, 2, None, criteria, 5, cv2.KMEANS_PP_CENTERS,
    )
    labels = labels.flatten()
    g0 = [crops[i] for i in range(len(crops)) if labels[i] == 0]
    g1 = [crops[i] for i in range(len(crops)) if labels[i] == 1]
    if not g0 or not g1:
        return None
    # Larger cluster first (cosmetic only)
    if len(g0) >= len(g1):
        return g0, g1
    return g1, g0


def detect_team_kits(
    video_path: str,
    *,
    weights: str = "yolov8n.pt",
    device: str = "cpu",
    n_samples: int = 24,
    target_width: int = 960,
    max_analyze_minutes: float = 75.0,
) -> list[dict] | None:
    """
    Sample frames, detect players, k-means torso colours → two kit swatches.
    Returns [{"id","label","hex"}, ...] or None if detection fails.
    """
    info = probe_video(video_path)
    n_frames = int(info.get("frame_count") or 0)
    fps = float(info.get("fps") or 25.0)
    if n_frames < 1:
        _dbg_log("H6", "team_preview", "no frames", {"video": video_path})
        return None

    max_frame = min(n_frames - 1, int(max_analyze_minutes * 60 * fps))
    idxs = [min(max_frame, int(max_frame * (i + 0.5) / n_samples)) for i in range(n_samples)]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        _dbg_log("H6", "team_preview", "cannot open video", {"video": video_path})
        return None

    det = Detector(DetectConfig(
        weights=weights,
        device=device,
        conf_threshold=0.20,
        imgsz=1280,
    ))

    crops: list[np.ndarray] = []
    players_per_frame: list[int] = []
    try:
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            frame = _resize_frame(frame, target_width)
            dets = det.detect_frame(frame)
            n_players = 0
            for d in dets:
                if d.get("class") != "player":
                    continue
                crop = _torso_crop(frame, d["bbox"])
                if crop is None or _is_referee_crop(crop) or _is_neutral_crop(crop):
                    continue
                crops.append(crop)
                n_players += 1
            players_per_frame.append(n_players)
    finally:
        cap.release()

    _dbg_log("H6", "team_preview", "crops collected", {
        "n_crops": len(crops),
        "n_samples": len(idxs),
        "players_per_frame": players_per_frame,
        "imgsz": 1280,
        "conf": 0.20,
    })

    if len(crops) < 4:
        _dbg_log("H7", "team_preview", "too few crops", {"n_crops": len(crops)})
        return None

    grouped = _kmeans_two_kits(crops)
    if grouped is None:
        _dbg_log("H9", "team_preview", "kmeans failed", {"n_crops": len(crops)})
        return None

    team_a_crops, team_b_crops = grouped
    hex_a = _crops_to_hex(team_a_crops)
    hex_b = _crops_to_hex(team_b_crops)

    if not hex_a or not hex_b:
        _dbg_log("H9", "team_preview", "hex failed", {"hex_a": hex_a, "hex_b": hex_b})
        return None
    if hex_a == hex_b:
        _dbg_log("H9", "team_preview", "identical hex", {"hex": hex_a})
        return None

    result = [
        {"id": "team_a", "label": "Team A", "hex": hex_a, "sample_count": len(team_a_crops)},
        {"id": "team_b", "label": "Team B", "hex": hex_b, "sample_count": len(team_b_crops)},
    ]
    _dbg_log("H6", "team_preview", "kits ok", {
        "hex_a": hex_a, "hex_b": hex_b,
        "count_a": len(team_a_crops), "count_b": len(team_b_crops),
    })
    return result
