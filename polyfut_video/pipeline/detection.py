"""Stage 5: YOLOv8 player + ball detection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# COCO class ids
CLASS_PERSON = 0
CLASS_BALL = 32

_MODEL_CACHE: dict[str, Any] = {}


@dataclass
class DetectConfig:
    weights: str = "yolov8s.pt"
    conf_threshold: float = 0.25
    ball_conf_min: float = 0.08
    device: str = "cpu"
    imgsz: int = 640
    ball_imgsz: int = 416


def _get_model(weights: str, device: str):
    key = f"{weights}|{device}"
    if key not in _MODEL_CACHE:
        from ultralytics import YOLO
        model = YOLO(weights)
        if not Path(weights).is_dir():
            try:
                model.to(device)
            except Exception:
                pass
        _MODEL_CACHE[key] = model
    return _MODEL_CACHE[key]


def _parse_boxes(
    results,
    *,
    conf_threshold: float,
    classes: set[int] | None = None,
) -> list[dict]:
    out: list[dict] = []
    if not results:
        return out
    res = results[0]
    if res.boxes is None or len(res.boxes) == 0:
        return out

    xyxy = res.boxes.xyxy.cpu().numpy()
    conf = res.boxes.conf.cpu().numpy()
    cls = res.boxes.cls.cpu().numpy().astype(int)

    for box, c, k in zip(xyxy, conf, cls):
        kid = int(k)
        if classes is not None and kid not in classes:
            continue
        label = "player" if kid == CLASS_PERSON else "ball"
        min_conf = conf_threshold * 0.35 if label == "ball" else conf_threshold
        if float(c) < min_conf:
            continue
        out.append({
            "bbox": [float(x) for x in box],
            "class": label,
            "conf": float(c),
        })
    return out


def detect(
    frame: np.ndarray,
    model,
    conf_threshold: float = 0.3,
    imgsz: int = 640,
    *,
    classes: list[int] | None = None,
) -> list[dict]:
    """
    Returns list of {"bbox": [x1,y1,x2,y2], "class": "player"|"ball", "conf": float}.
    """
    results = model.predict(
        frame,
        imgsz=imgsz,
        conf=conf_threshold * 0.35 if classes == [CLASS_BALL] else conf_threshold,
        verbose=False,
        classes=classes,
    )
    allowed = set(classes) if classes is not None else None
    return _parse_boxes(results, conf_threshold=conf_threshold, classes=allowed)


def _filter_balls(balls: list[dict], players: list[dict]) -> list[dict]:
    if not balls:
        return []
    if not players:
        return balls
    ph = max(p["bbox"][3] - p["bbox"][1] for p in players)
    return [
        b for b in balls
        if (b["bbox"][2] - b["bbox"][0]) <= ph * 0.45
    ]


class Detector:
    """Stateful wrapper for batch-capable inference."""

    def __init__(self, cfg: DetectConfig | None = None):
        self.cfg = cfg or DetectConfig()
        self.model = _get_model(self.cfg.weights, self.cfg.device)

    def detect_frame(self, frame: np.ndarray) -> list[dict]:
        dets = detect(
            frame,
            self.model,
            conf_threshold=self.cfg.conf_threshold,
            imgsz=self.cfg.imgsz,
        )
        players = [d for d in dets if d["class"] == "player"]
        balls = [d for d in dets if d["class"] == "ball" and d["conf"] >= self.cfg.ball_conf_min]
        balls = _filter_balls(balls, players)
        return players + balls

    def detect_ball_only(self, frame: np.ndarray) -> list[dict]:
        """Lightweight ball pass for cheap-routed frames (players reused)."""
        dets = detect(
            frame,
            self.model,
            conf_threshold=self.cfg.conf_threshold,
            imgsz=self.cfg.ball_imgsz,
            classes=[CLASS_BALL],
        )
        return [d for d in dets if d["conf"] >= self.cfg.ball_conf_min]

    def detect_balls_batch(self, frames: list[np.ndarray]) -> list[list[dict]]:
        """Batch ball-only inference (smaller imgsz for speed)."""
        if not frames:
            return []
        if len(frames) == 1:
            return [self.detect_ball_only(frames[0])]

        results = self.model.predict(
            frames,
            imgsz=self.cfg.ball_imgsz,
            conf=self.cfg.conf_threshold * 0.35,
            verbose=False,
            classes=[CLASS_BALL],
        )
        out: list[list[dict]] = []
        for res in results:
            dets = _parse_boxes(
                [res],
                conf_threshold=self.cfg.conf_threshold,
                classes={CLASS_BALL},
            )
            out.append([d for d in dets if d["conf"] >= self.cfg.ball_conf_min])
        return out

    def merge_players_and_ball(
        self,
        players: list[dict],
        frame: np.ndarray,
    ) -> list[dict]:
        """Reuse player boxes; refresh ball on cheap-motion frames."""
        balls = self.detect_ball_only(frame)
        balls = _filter_balls(balls, players)
        return [dict(p) for p in players] + balls

    def detect_frames_batch(self, frames: list[np.ndarray]) -> list[list[dict]]:
        """Batch full detection for consecutive frames (much faster on CPU)."""
        if not frames:
            return []
        if len(frames) == 1:
            return [self.detect_frame(frames[0])]

        results = self.model.predict(
            frames,
            imgsz=self.cfg.imgsz,
            conf=self.cfg.conf_threshold,
            verbose=False,
        )
        out: list[list[dict]] = []
        for res in results:
            dets = _parse_boxes([res], conf_threshold=self.cfg.conf_threshold)
            players = [d for d in dets if d["class"] == "player"]
            balls = [
                d for d in dets if d["class"] == "ball" and d["conf"] >= self.cfg.ball_conf_min
            ]
            balls = _filter_balls(balls, players)
            out.append(players + balls)
        return out
