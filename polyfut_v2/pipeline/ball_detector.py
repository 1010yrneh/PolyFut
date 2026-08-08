"""Stage 3b: pluggable ball detector with ROI search.

The ball is the anchor of the whole v2 pipeline, so this is the only detector
that runs on every analysed frame. It is deliberately pluggable: any object with
a ``detect(frame, last_center=None) -> BallDetection | None`` method works, so a
soccer-specific ball model drops in behind the same interface. The default
implementation wraps an Ultralytics YOLO model (COCO "sports ball" by default).

The pure geometry helpers (``roi_crop``, ``map_bbox_to_full``, ``parse_best_ball``)
are split out so they can be unit-tested without loading a model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from polyfut_v2.pipeline import fast_infer
from polyfut_v2.pipeline.color import is_bbox_on_foreign_surface

_MODEL_CACHE: dict[str, Any] = {}


@dataclass
class BallDetection:
    """A single ball detection in full-frame (resized) coordinates."""

    bbox: list[float]  # [x1, y1, x2, y2]
    conf: float

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class BallDetectorProtocol(Protocol):
    def detect(
        self,
        frame: np.ndarray,
        last_center: tuple[float, float] | None = None,
    ) -> BallDetection | None: ...


# --------------------------------------------------------------------------- #
# Pure geometry helpers (no model needed)
# --------------------------------------------------------------------------- #

def roi_crop(
    frame: np.ndarray,
    center: tuple[float, float],
    half: float,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Crop a square ROI around ``center``, clamped to frame bounds.

    Returns ``(crop, (offset_x, offset_y))`` where the offset maps crop-local
    coordinates back to the full frame.
    """
    h, w = frame.shape[:2]
    cx, cy = center
    x1 = int(max(0, min(w - 1, round(cx - half))))
    y1 = int(max(0, min(h - 1, round(cy - half))))
    x2 = int(max(x1 + 1, min(w, round(cx + half))))
    y2 = int(max(y1 + 1, min(h, round(cy + half))))
    return frame[y1:y2, x1:x2], (x1, y1)


def map_bbox_to_full(bbox: list[float], offset: tuple[int, int]) -> list[float]:
    """Shift a crop-local bbox back into full-frame coordinates."""
    ox, oy = offset
    x1, y1, x2, y2 = bbox
    return [x1 + ox, y1 + oy, x2 + ox, y2 + oy]


def parse_best_ball(
    xyxy: np.ndarray,
    conf: np.ndarray,
    cls: np.ndarray,
    *,
    ball_class_id: int,
    conf_min: float,
) -> BallDetection | None:
    """Pick the highest-confidence ball box above ``conf_min``.

    Arrays are the raw ``(N,4)`` / ``(N,)`` / ``(N,)`` model outputs.
    """
    best: BallDetection | None = None
    for box, c, k in zip(xyxy, conf, cls):
        if int(k) != ball_class_id:
            continue
        cf = float(c)
        if cf < conf_min:
            continue
        if best is None or cf > best.conf:
            best = BallDetection(bbox=[float(v) for v in box], conf=cf)
    return best


# --------------------------------------------------------------------------- #
# Default YOLO-backed detector
# --------------------------------------------------------------------------- #

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


class YoloBallDetector:
    """Default ball detector: ROI search first, full-frame re-acquire on cold
    start or when the ROI is unavailable.

    ``model`` may be injected (tests / custom weights); otherwise it is loaded
    lazily from ``cfg.ball_weights``.

    Miss-storm backoff: on low-recall footage the dominant runtime cost is the
    full-frame re-acquire firing on *every* sampled frame of a long miss run
    (often 2 inferences/frame: ROI miss + full fallback). After
    ``ball_miss_backoff_after`` consecutive misses the full-frame scan runs only
    every ``ball_miss_backoff_stride``-th call; any detection (or ``reset()`` at
    a shot boundary) restores per-frame scanning. During a long miss the model
    demonstrably can't see the ball, so skipped scans cost little recall —
    worst case, re-acquire is delayed by (stride−1) sampled frames.
    """

    def __init__(self, cfg, model: Any | None = None):
        self.cfg = cfg
        self._model = model
        self._roi_model = None
        self._consec_misses = 0
        # Players taken off the most recent full-frame scan, as
        # (xyxy, conf, cls) — the same shape the player detector returns — or
        # None when this frame had no full-frame scan. Read it right after
        # detect(); it is overwritten on the next call.
        self.last_players: tuple | None = None
        self.stats = {
            "roi_hits": 0, "roi_misses": 0,
            "full_scans": 0, "skipped_full_scans": 0,
            "harvested_frames": 0, "harvested_players": 0,
        }

    def reset(self) -> None:
        """Shot boundary: fresh scene, scan immediately."""
        self._consec_misses = 0

    @property
    def model(self):
        if self._model is None:
            self._model = _get_model(self.cfg.ball_weights, self.cfg.device)
        return self._model

    @property
    def roi_model(self):
        """The model used for ROI crops - a smaller export when one exists.

        Falls back to the main model, so a config without ``ball_roi_weights``
        behaves exactly as before. ``_get_model`` caches per weights path, so
        the two live side by side rather than reloading.
        """
        w = getattr(self.cfg, "ball_roi_weights", "") or ""
        if not w:
            return self.model
        if self._roi_model is None:
            self._roi_model = _get_model(w, self.cfg.device)
        return self._roi_model

    @property
    def roi_imgsz(self) -> int:
        return int(getattr(self.cfg, "ball_roi_imgsz", 0) or self.cfg.ball_imgsz)

    def _should_full_scan(self) -> bool:
        after = int(getattr(self.cfg, "ball_miss_backoff_after", 0) or 0)
        if after <= 0 or self._consec_misses < after:
            return True
        stride = max(1, int(getattr(self.cfg, "ball_miss_backoff_stride", 3) or 1))
        return (self._consec_misses - after) % stride == 0

    def _harvest_classes(self) -> list[int] | None:
        """Ball + every body class, or None when there is nothing extra to take.

        The model emits all four classes from one forward pass — the class list
        is a post-NMS filter, not a cheaper inference — so a full-frame ball scan
        already computed the players and then discarded them.
        """
        if not getattr(self.cfg, "harvest_players_from_ball_pass", False):
            return None
        extra = [getattr(self.cfg, name, None) for name in
                 ("player_class_id", "goalkeeper_class_id", "referee_class_id")]
        extra = [int(c) for c in extra if c is not None
                 and int(c) != int(self.cfg.ball_class_id)]
        if not extra:
            return None                    # single-class model: nothing to gain
        return [int(self.cfg.ball_class_id)] + extra

    def _raw(self, image: np.ndarray, imgsz: int, classes: list[int],
             conf: float):
        """(xyxy, conf, cls) for one inference, or None."""
        fast = fast_infer.try_detect(
            self.model, image,
            imgsz=imgsz, conf=conf, iou=0.7, classes=classes,
            enabled=bool(getattr(self.cfg, "fast_infer_enabled", True)),
        )
        if fast is not None:
            return fast
        results = self.model.predict(
            image, imgsz=imgsz, conf=conf, classes=classes, verbose=False,
        )
        if not results:
            return None
        res = results[0]
        if res.boxes is None or len(res.boxes) == 0:
            return None
        return (res.boxes.xyxy.cpu().numpy(),
                res.boxes.conf.cpu().numpy(),
                res.boxes.cls.cpu().numpy())

    def _infer_and_harvest(self, image: np.ndarray, imgsz: int):
        """Full-frame ball scan that also keeps the players it already found.

        The ball half is deliberately unchanged: candidates are filtered on the
        max class score *before* the class list is applied, and NMS offsets
        boxes by class, so widening the request cannot alter which ball box
        wins. Verified detection-for-detection against the ball-only path.
        """
        classes = self._harvest_classes()
        if classes is None:
            return self._infer(image, imgsz), None
        got = self._raw(image, imgsz, classes, self.cfg.ball_conf_min)
        if got is None:
            return None, None
        xyxy, conf, cls = got
        is_ball = cls == int(self.cfg.ball_class_id)
        ball = parse_best_ball(
            xyxy[is_ball], conf[is_ball], cls[is_ball],
            ball_class_id=self.cfg.ball_class_id,
            conf_min=self.cfg.ball_conf_min,
        ) if bool(is_ball.any()) else None
        # Players were requested at the ball's (lower) threshold so the ball
        # candidate set is untouched; hold them to their own threshold here.
        body = (~is_ball) & (conf >= float(self.cfg.player_conf_min))
        players = ((xyxy[body], conf[body], cls[body])
                   if bool(body.any()) else None)
        return ball, players

    def _infer(self, image: np.ndarray, imgsz: int,
               model=None) -> BallDetection | None:
        # Ball tracking is ~91% of a run and 43% of every predict() call is
        # Ultralytics plumbing, so go straight to the compiled model when we
        # can. Verified to return identical detections; falls back silently on
        # the first call (the predictor is built lazily) and on any backend
        # without a compiled OpenVINO model.
        model = self.model if model is None else model
        fast = fast_infer.try_detect(
            model, image,
            imgsz=imgsz,
            conf=self.cfg.ball_conf_min,
            # Ultralytics' default when predict() is called without `iou`,
            # which is what the fallback below does.
            iou=0.7,
            classes=[self.cfg.ball_class_id],
            enabled=bool(getattr(self.cfg, "fast_infer_enabled", True)),
        )
        if fast is not None:
            xyxy, conf, cls = fast
            if xyxy.shape[0] == 0:
                return None
            return parse_best_ball(
                xyxy, conf, cls,
                ball_class_id=self.cfg.ball_class_id,
                conf_min=self.cfg.ball_conf_min,
            )

        results = model.predict(
            image,
            imgsz=imgsz,
            conf=self.cfg.ball_conf_min,
            classes=[self.cfg.ball_class_id],
            verbose=False,
        )
        if not results:
            return None
        res = results[0]
        if res.boxes is None or len(res.boxes) == 0:
            return None
        xyxy = res.boxes.xyxy.cpu().numpy()
        conf = res.boxes.conf.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy()
        return parse_best_ball(
            xyxy, conf, cls,
            ball_class_id=self.cfg.ball_class_id,
            conf_min=self.cfg.ball_conf_min,
        )

    def detect(
        self,
        frame: np.ndarray,
        last_center: tuple[float, float] | None = None,
    ) -> BallDetection | None:
        # Warm path: search a small ROI around the last known position.
        det: BallDetection | None = None
        roi_ran = False

        if self.cfg.roi_enabled and last_center is not None:
            roi_ran = True
            crop, offset = roi_crop(frame, last_center, self.cfg.roi_half_px)
            if crop.size > 0:
                det = self._infer(crop, self.roi_imgsz, self.roi_model)
                if det is not None:
                    det = BallDetection(
                        bbox=map_bbox_to_full(det.bbox, offset),
                        conf=det.conf,
                    )
            self.stats["roi_hits" if det is not None else "roi_misses"] += 1

        # Cold path (or ROI-miss fallback): full-frame re-acquire, throttled
        # during a miss storm (see class docstring).
        #
        # ``roi_fallback_full`` decides only what happens after an ROI MISS:
        # re-scan the whole frame now, or coast on the stale position. On a
        # small pitch or through a zoom the ball can leave the ROI between
        # samples, and without the re-scan recall collapses on fast-ball
        # footage — hence the default of True.
        #
        # It used to be applied as an unconditional ``det = None`` here, which
        # did the opposite of what it says on both counts: it discarded ROI
        # *hits*, and by emptying ``det`` it then triggered the very full scan
        # it was meant to suppress. Dormant at the default, but it silently
        # made "ROI only" benchmarks slower than ROI-plus-fallback.
        allow_full = (not roi_ran) or self.cfg.roi_fallback_full
        self.last_players = None
        if det is None and allow_full:
            if self._should_full_scan():
                # Take the players out of this pass too — it computed them
                # anyway. Only the FULL-FRAME scan is harvested: the ROI crop is
                # a 240px window, so bodies in it are cut off and their boxes
                # would be wrong.
                det, self.last_players = self._infer_and_harvest(
                    frame, self.cfg.ball_full_imgsz)
                self.stats["full_scans"] += 1
                if self.last_players is not None:
                    self.stats["harvested_frames"] += 1
                    self.stats["harvested_players"] += len(self.last_players[0])
            else:
                self.stats["skipped_full_scans"] += 1

        # Spare balls on the sideline can be detected as the ball and then
        # corrupt Stage 4 contact candidates. Reject only when the surroundings
        # are positively a non-pitch surface (running track, painted concrete).
        # The old test demanded positive proof of *grass*, which threw away real
        # on-pitch balls on any turf outside a narrow hue band — measured at 49%
        # of on-pitch positions on bright sun-bleached footage.
        if det is not None and getattr(self.cfg, "ball_pitch_gate_enabled", False):
            if is_bbox_on_foreign_surface(
                frame,
                det.bbox,
                check_half_px=getattr(self.cfg, "ball_surface_check_half_px", 18.0),
                min_foreign_frac=getattr(self.cfg, "ball_foreign_surface_frac", 0.70),
                min_colored_frac=getattr(self.cfg, "ball_surface_min_colored_frac", 0.25),
            ):
                det = None

        self._consec_misses = 0 if det is not None else self._consec_misses + 1
        return det
