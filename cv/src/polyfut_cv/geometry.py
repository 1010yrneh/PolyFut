"""Small box-geometry helpers shared across modules (no heavy deps)."""

from __future__ import annotations

import numpy as np

CLASS_PERSON = 0
CLASS_BALL = 32  # COCO "sports ball"


def foot_point(xyxy: np.ndarray) -> np.ndarray:
    x1, _, x2, y2 = xyxy
    return np.array([(x1 + x2) / 2.0, y2], dtype=np.float32)


def box_center(xyxy: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = xyxy
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)


def ball_diagonal(xyxy: np.ndarray) -> float:
    x1, y1, x2, y2 = xyxy
    w = max(x2 - x1, 0.0)
    h = max(y2 - y1, 0.0)
    return float((w * w + h * h) ** 0.5)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return float(inter / (aa + bb - inter + 1e-6))


def dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)))
