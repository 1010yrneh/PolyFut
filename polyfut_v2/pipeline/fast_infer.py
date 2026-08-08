"""Call the OpenVINO model directly, skipping Ultralytics' predictor.

Measured on this project's soccer model: a full ``model.predict()`` call costs
69.3 ms, of which only 39.6 ms is the network — **43% is framework overhead**.
Ball tracking is 91% of a run's wall clock, so that overhead is the single
largest cost in PolyFut that buys nothing.

It is not the predictor's fault; it is generality. Every call rebuilds a
``LetterBox``, stacks a batch of one, round-trips through a torch tensor on the
way in, and constructs ``Results`` objects (with ``Boxes`` views, ``orig_img``
copies and name maps) on the way out, none of which survives past
``parse_best_ball``. This module does the same arithmetic in numpy and returns
plain arrays.

**This must not change a single detection.** Every step below is transcribed
from the installed Ultralytics, not reimplemented from memory:

  * ``LetterBox`` geometry — ``data/augment.py`` (scaleup, center, value 114,
    INTER_LINEAR), with ``auto=False`` because that is what
    ``predictor.pre_transform`` computes for a non-``pt``, non-dynamic backend.
  * tensor layout — ``BasePredictor.preprocess``: BGR→RGB, BHWC→BCHW, /255.
  * NMS — ``utils/nms.py::non_max_suppression`` with ``multi_label=False``,
    best-class-only, class filter, boxes offset by ``cls * 7680``, then
    ``[:max_det]``.
  * box mapping — ``utils/ops.py::scale_boxes`` (which recomputes gain and pad
    from the shapes, so it must agree with the letterbox above) and
    ``clip_boxes``.

``verify_matches_ultralytics`` in the tests holds it to that: identical box
counts and coordinates within a pixel across real frames.
"""

from __future__ import annotations

import threading
from typing import Any

import cv2
import numpy as np

# Ultralytics constants, quoted rather than guessed.
_PAD_VALUE = 114
_MAX_WH = 7680          # nms.py: class offset multiplier
_MAX_NMS = 30000        # nms.py: cap before the IoU pass


def compiled_model(model: Any):
    """The raw OpenVINO ``CompiledModel`` behind an Ultralytics YOLO, or None.

    Returns None for any backend that isn't a compiled OpenVINO model (a .pt on
    torch, say) so the caller keeps using ``predict``.
    """
    for holder in (getattr(model, "predictor", None), model):
        inner = getattr(holder, "model", None)
        ov = getattr(inner, "ov_compiled_model", None)
        if ov is not None:
            return ov
    return None


def letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, float, float, float]:
    """Ultralytics ``LetterBox(auto=False, center=True, scaleup=True)``.

    Returns (padded image, gain, pad_left, pad_top).
    """
    h0, w0 = img.shape[:2]
    r = min(size / h0, size / w0)
    new_unpad = (round(w0 * r), round(h0 * r))
    dw = (size - new_unpad[0]) / 2.0
    dh = (size - new_unpad[1]) / 2.0
    if (w0, h0) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    if top or bottom or left or right:
        img = cv2.copyMakeBorder(img, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT,
                                 value=(_PAD_VALUE,) * 3)
    return img, r, float(left), float(top)


def to_tensor(padded: np.ndarray) -> np.ndarray:
    """BGR HWC uint8 -> RGB BCHW float32 in [0,1] (BasePredictor.preprocess)."""
    x = padded[..., ::-1].transpose(2, 0, 1)          # BGR->RGB, HWC->CHW
    x = np.ascontiguousarray(x, dtype=np.float32)
    x /= 255.0
    return x[None]                                     # add batch


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> list[int]:
    """Greedy IoU suppression, score-descending — torchvision.ops.nms semantics."""
    order = scores.argsort()[::-1]
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)
        order = rest[iou <= iou_thres]
    return keep


def detect(
    ov_model: Any,
    image: np.ndarray,
    *,
    imgsz: int,
    conf: float,
    iou: float = 0.7,
    classes: list[int] | None = None,
    max_det: int = 300,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Detections for one image, in that image's own pixel coordinates.

    Returns (xyxy[N,4], conf[N], cls[N]) as float32/float32/float32 — the same
    three arrays the callers already pull off ``res.boxes``. None means the
    fast path could not run and the caller should fall back.
    """
    if ov_model is None or image is None or image.size == 0:
        return None
    h0, w0 = image.shape[:2]
    padded, gain, _pl, _pt = letterbox(image, imgsz)
    if padded.shape[0] != imgsz or padded.shape[1] != imgsz:
        return None                       # not the fixed shape the export wants
    try:
        raw = ov_model(to_tensor(padded))
    except Exception:
        return None
    pred = raw[0] if not hasattr(raw, "values") else next(iter(raw.values()))
    pred = np.asarray(pred)
    if pred.ndim != 3 or pred.shape[0] != 1:
        return None
    p = pred[0]                            # (4 + nc, anchors)
    nc = p.shape[0] - 4
    if nc < 1:
        return None
    p = p.T                                # (anchors, 4 + nc)
    cls_scores = p[:, 4:4 + nc]
    best = cls_scores.max(axis=1)
    m = best > conf                        # nms.py: candidates by max class score
    if not m.any():
        return _empty()
    sel = p[m]
    score = best[m]
    kls = cls_scores[m].argmax(axis=1)
    if classes is not None:
        keep_cls = np.isin(kls, np.asarray(classes))
        if not keep_cls.any():
            return _empty()
        sel, score, kls = sel[keep_cls], score[keep_cls], kls[keep_cls]
    if sel.shape[0] > _MAX_NMS:
        top = score.argsort()[::-1][:_MAX_NMS]
        sel, score, kls = sel[top], score[top], kls[top]

    # xywh -> xyxy (nms.py applies this before suppression)
    cx, cy, w, h = sel[:, 0], sel[:, 1], sel[:, 2], sel[:, 3]
    xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)

    offset = kls.astype(np.float32)[:, None] * float(_MAX_WH)
    keep = _nms(xyxy + offset, score, iou)[:max_det]
    if not keep:
        return _empty()
    xyxy = xyxy[keep]
    score = score[keep]
    kls = kls[keep]

    # ops.scale_boxes: recomputes gain/pad from the shapes, so this has to agree
    # with the letterbox above rather than reuse its numbers.
    g = min(imgsz / h0, imgsz / w0)
    pad_x = round((imgsz - round(w0 * g)) / 2 - 0.1)
    pad_y = round((imgsz - round(h0 * g)) / 2 - 0.1)
    xyxy[:, [0, 2]] -= pad_x
    xyxy[:, [1, 3]] -= pad_y
    xyxy /= g
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, w0)
    xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, h0)
    return (xyxy.astype(np.float32), score.astype(np.float32),
            kls.astype(np.float32))


def _empty():
    return (np.zeros((0, 4), np.float32), np.zeros((0,), np.float32),
            np.zeros((0,), np.float32))


# Resolving the compiled model walks a few attributes; cache it per YOLO object.
_OV_CACHE: dict[int, Any] = {}

# --------------------------------------------------------------------------- #
# One inference at a time, per model
# --------------------------------------------------------------------------- #
# Calling a CompiledModel (``ov_model(x)`` below) uses its ONE default infer
# request, and Ultralytics' predict() reuses that same request. Models are
# cached globally and handed to every caller, while the server runs inference on
# several threads at once - the analysis job, the seed prefetch, the kit preview
# and the model warm-up. Two of them landing on one model raises
#
#     RuntimeError: Infer Request is busy
#
# which killed a real 20-minute run at the 9-minute mark. It is a race, so it
# hides: it needs two threads inside the same model at the same moment, which is
# why runs that reached the same code minutes apart were fine.
#
# A lock rather than a request pool because the contending threads all want the
# same CPU anyway - serialising costs nothing real, and a pool would have to be
# sized and drained. RLock so a caller already holding it can take the fast path
# without deadlocking itself.
_LOCKS: dict[int, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


def model_lock(model: Any) -> threading.RLock:
    """The inference lock for ``model``. Every call site must hold it.

    Keyed on id() to match _OV_CACHE, so a model and its lock live and die
    together.
    """
    key = id(model)
    lk = _LOCKS.get(key)
    if lk is None:
        with _LOCKS_GUARD:
            lk = _LOCKS.get(key)
            if lk is None:
                lk = threading.RLock()
                _LOCKS[key] = lk
    return lk


def try_detect(
    model: Any,
    image: np.ndarray,
    *,
    imgsz: int,
    conf: float,
    iou: float,
    classes: list[int] | None = None,
    max_det: int = 300,
    enabled: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Fast path if this model exposes a compiled OpenVINO backend, else None.

    Ultralytics builds ``model.predictor`` lazily on the first ``predict()``, so
    the compiled model cannot be reached until one call has gone through the
    normal path. Returning None then is correct and self-healing: the caller
    falls back, that call initialises the predictor, and every subsequent call
    takes the fast path.
    """
    if not enabled or model is None:
        return None
    key = id(model)
    ov = _OV_CACHE.get(key)
    if ov is None:
        ov = compiled_model(model)
        if ov is None:
            return None
        _OV_CACHE[key] = ov
    with model_lock(model):
        return detect(ov, image, imgsz=imgsz, conf=conf, iou=iou,
                      classes=classes, max_det=max_det)
