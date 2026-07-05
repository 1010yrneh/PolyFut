"""Stage 5: pluggable player detector (sparse — only at contact candidates).

The big v2 speed win: full-frame player detection no longer runs every frame,
only in a crop around the ball at the few hundred contact moments. Same
pluggable pattern as the ball detector — swap ``player_weights`` /
``player_class_id`` for a soccer player/keeper/ref model and it drops in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from polyfut_v2.pipeline.ball_detector import _get_model, map_bbox_to_full, roi_crop


@dataclass
class PlayerDetection:
    bbox: list[float]  # [x1, y1, x2, y2] full-frame coords
    conf: float

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class PlayerDetectorProtocol(Protocol):
    def detect(
        self,
        frame: np.ndarray,
        near: tuple[float, float] | None = None,
    ) -> list[PlayerDetection]: ...


def _parse_players(
    xyxy: np.ndarray,
    conf: np.ndarray,
    cls: np.ndarray,
    *,
    player_class_id: int,
    conf_min: float,
) -> list[PlayerDetection]:
    out: list[PlayerDetection] = []
    for box, c, k in zip(xyxy, conf, cls):
        if int(k) != player_class_id:
            continue
        if float(c) < conf_min:
            continue
        out.append(PlayerDetection(bbox=[float(v) for v in box], conf=float(c)))
    return out


class YoloPlayerDetector:
    """Default player detector: ROI crop around the ball when a point is given,
    else full-frame. ``model`` may be injected for tests / custom weights."""

    def __init__(self, cfg, model: Any | None = None):
        self.cfg = cfg
        self._model = model

    @property
    def model(self):
        if self._model is None:
            self._model = _get_model(self.cfg.player_weights, self.cfg.device)
        return self._model

    def _infer(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        results = self.model.predict(
            image,
            imgsz=self.cfg.player_imgsz,
            conf=self.cfg.player_conf_min,
            classes=[self.cfg.player_class_id],
            verbose=False,
        )
        if not results:
            return None
        res = results[0]
        if res.boxes is None or len(res.boxes) == 0:
            return None
        return (
            res.boxes.xyxy.cpu().numpy(),
            res.boxes.conf.cpu().numpy(),
            res.boxes.cls.cpu().numpy(),
        )

    def detect(
        self,
        frame: np.ndarray,
        near: tuple[float, float] | None = None,
    ) -> list[PlayerDetection]:
        offset = (0, 0)
        image = frame
        if near is not None and self.cfg.player_roi_half_px > 0:
            image, offset = roi_crop(frame, near, self.cfg.player_roi_half_px)
            if image.size == 0:
                image, offset = frame, (0, 0)

        parsed = self._infer(image)
        if parsed is None:
            return []
        xyxy, conf, cls = parsed
        dets = _parse_players(
            xyxy, conf, cls,
            player_class_id=self.cfg.player_class_id,
            conf_min=self.cfg.player_conf_min,
        )
        if offset != (0, 0):
            for d in dets:
                d.bbox = map_bbox_to_full(d.bbox, offset)
        return dets
