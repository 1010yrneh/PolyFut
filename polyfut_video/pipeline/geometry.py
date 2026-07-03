"""Small box-geometry helpers (no heavy deps)."""

from __future__ import annotations

import numpy as np


def foot_point(xyxy: np.ndarray | list[float]) -> np.ndarray:
    x1, _, x2, y2 = xyxy
    return np.array([(x1 + x2) / 2.0, y2], dtype=np.float32)


def box_center(xyxy: np.ndarray | list[float]) -> np.ndarray:
    x1, y1, x2, y2 = xyxy
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)


def ball_diagonal(xyxy: np.ndarray | list[float]) -> float:
    x1, y1, x2, y2 = xyxy
    w = max(float(x2) - float(x1), 0.0)
    h = max(float(y2) - float(y1), 0.0)
    return float((w * w + h * h) ** 0.5)


def dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)))
