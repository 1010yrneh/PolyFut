"""Stage 5: pluggable player detector (sparse — only at contact candidates).

The big v2 speed win: full-frame player detection no longer runs every frame,
only in a crop around the ball at the few hundred contact moments. Same
pluggable pattern as the ball detector — swap ``player_weights`` /
``player_class_id`` for a soccer player/keeper/ref model and it drops in.

When ``goalkeeper_class_id`` / ``referee_class_id`` are set (soccer model), the
detector requests all three classes; keepers are valid contact candidates and
referees are returned with their class id so callers can exclude them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from polyfut_v2.pipeline import fast_infer
from polyfut_v2.pipeline.ball_detector import _get_model, map_bbox_to_full, roi_crop


@dataclass
class PlayerDetection:
    bbox: list[float]  # [x1, y1, x2, y2] full-frame coords
    conf: float
    class_id: int = 0

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


def player_request_classes(cfg) -> list[int]:
    """Class ids passed to the model ``classes=`` filter."""
    ids = [int(cfg.player_class_id)]
    gk = getattr(cfg, "goalkeeper_class_id", None)
    ref = getattr(cfg, "referee_class_id", None)
    if gk is not None:
        ids.append(int(gk))
    if ref is not None:
        ids.append(int(ref))
    # Preserve order, drop dupes.
    out: list[int] = []
    for i in ids:
        if i not in out:
            out.append(i)
    return out


def player_contact_class_ids(cfg) -> set[int]:
    """Class ids eligible as the contacting player (outfield + keeper)."""
    ids = {int(cfg.player_class_id)}
    gk = getattr(cfg, "goalkeeper_class_id", None)
    if gk is not None:
        ids.add(int(gk))
    return ids


def _parse_players(
    xyxy: np.ndarray,
    conf: np.ndarray,
    cls: np.ndarray,
    *,
    allowed_class_ids: set[int],
    conf_min: float,
) -> list[PlayerDetection]:
    out: list[PlayerDetection] = []
    for box, c, k in zip(xyxy, conf, cls):
        kid = int(k)
        if kid not in allowed_class_ids:
            continue
        if float(c) < conf_min:
            continue
        out.append(PlayerDetection(
            bbox=[float(v) for v in box], conf=float(c), class_id=kid,
        ))
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
        # Straight to the compiled model where possible — see fast_infer. The
        # NMS threshold must be the same one the predict() call below uses, or
        # the two paths would disagree about overlapping boxes.
        fast = fast_infer.try_detect(
            self.model, image,
            imgsz=self.cfg.player_imgsz,
            conf=self.cfg.player_conf_min,
            iou=float(getattr(self.cfg, "player_nms_iou", 0.45)),
            classes=player_request_classes(self.cfg),
            enabled=bool(getattr(self.cfg, "fast_infer_enabled", True)),
        )
        if fast is not None:
            return None if fast[0].shape[0] == 0 else fast

        with fast_infer.model_lock(self.model):
            results = self.model.predict(
                image,
                imgsz=self.cfg.player_imgsz,
                conf=self.cfg.player_conf_min,
                classes=player_request_classes(self.cfg),
                # Tighter than Ultralytics' 0.7 default: at 0.7 a single player
                # routinely survives NMS as several overlapping boxes, which become
                # duplicate tracks / duplicate markers downstream.
                iou=getattr(self.cfg, "player_nms_iou", 0.45),
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
        # Keep every requested class (incl. referee) so callers can exclude by id.
        allowed = set(player_request_classes(self.cfg))
        dets = _parse_players(
            xyxy, conf, cls,
            allowed_class_ids=allowed,
            conf_min=self.cfg.player_conf_min,
        )
        if offset != (0, 0):
            for d in dets:
                d.bbox = map_bbox_to_full(d.bbox, offset)
        return dets

    def detect_many(
        self,
        items: list[tuple[np.ndarray, tuple[float, float] | None]],
    ) -> list[list[PlayerDetection]]:
        """Batched detect for several (frame, near) pairs in one model call.

        Falls back to per-item ``detect`` when the batch is empty/size-1 or the
        model cannot take a list. ROI crops may differ in size; Ultralytics
        letterboxes them. Time order of ``items`` is preserved in the result.
        """
        if not items:
            return []
        if len(items) == 1:
            frame, near = items[0]
            return [self.detect(frame, near)]

        images: list[np.ndarray] = []
        offsets: list[tuple[int, int]] = []
        for frame, near in items:
            offset = (0, 0)
            image = frame
            if near is not None and self.cfg.player_roi_half_px > 0:
                image, offset = roi_crop(frame, near, self.cfg.player_roi_half_px)
                if image.size == 0:
                    image, offset = frame, (0, 0)
            images.append(image)
            offsets.append(offset)

        try:
            with fast_infer.model_lock(self.model):
                results = self.model.predict(
                    images,
                    imgsz=self.cfg.player_imgsz,
                    conf=self.cfg.player_conf_min,
                    classes=player_request_classes(self.cfg),
                    iou=getattr(self.cfg, "player_nms_iou", 0.45),
                    verbose=False,
                )
        except Exception:
            # Injected fakes / older backends may not accept a list — degrade.
            return [self.detect(frame, near) for frame, near in items]

        if not results or len(results) != len(items):
            return [self.detect(frame, near) for frame, near in items]

        allowed = set(player_request_classes(self.cfg))
        out: list[list[PlayerDetection]] = []
        for res, offset in zip(results, offsets):
            if res.boxes is None or len(res.boxes) == 0:
                out.append([])
                continue
            dets = _parse_players(
                res.boxes.xyxy.cpu().numpy(),
                res.boxes.conf.cpu().numpy(),
                res.boxes.cls.cpu().numpy(),
                allowed_class_ids=allowed,
                conf_min=self.cfg.player_conf_min,
            )
            if offset != (0, 0):
                for d in dets:
                    d.bbox = map_bbox_to_full(d.bbox, offset)
            out.append(dets)
        return out
