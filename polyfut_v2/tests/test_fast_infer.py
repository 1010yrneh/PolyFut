"""Calling the compiled model directly must not change a single detection.

Ball tracking is ~91% of a run and 43% of every Ultralytics ``predict()`` call
is framework overhead, so the bypass is worth having — but only while it is
provably equivalent. These tests pin the arithmetic that has to match
(``LetterBox`` geometry, tensor layout, NMS semantics, ``scale_boxes`` mapping)
against a stub backend, so they run without the real model and fail loudly if
any of it drifts.

The end-to-end check lives outside the suite because it needs the model and a
real clip: on a 60s slice of b48758eb195e both paths produced 451 samples, an
identical detected_ratio of 0.5743, and a worst position delta of 0.0000 px, at
1.35x the speed.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from polyfut_v2.pipeline import fast_infer as FI


# ------------------------------------------------------------- letterbox
@pytest.mark.parametrize("shape,exp_gain,exp_left,exp_top", [
    ((360, 640), 1.0, 0, 140),        # a full analysed frame: pad top/bottom
    ((240, 240), 640 / 240, 0, 0),    # a square ROI crop: pure upscale
    ((100, 400), 1.6, 0, 240),
])
def test_letterbox_geometry(shape, exp_gain, exp_left, exp_top):
    img = np.zeros((shape[0], shape[1], 3), np.uint8)
    out, gain, left, top = FI.letterbox(img, 640)
    assert out.shape[:2] == (640, 640)
    assert gain == pytest.approx(exp_gain)
    assert left == pytest.approx(exp_left)
    assert top == pytest.approx(exp_top)


def test_letterbox_pads_with_114_not_black():
    """Ultralytics pads with 114; black padding would shift the network's
    input statistics and quietly change borderline detections."""
    img = np.full((360, 640, 3), 200, np.uint8)
    out, _g, _l, top = FI.letterbox(img, 640)
    assert int(out[0, 0, 0]) == 114
    assert int(out[int(top) + 5, 320, 0]) == 200


# ---------------------------------------------------------------- tensor
def test_tensor_is_rgb_chw_normalised():
    img = np.zeros((640, 640, 3), np.uint8)
    img[..., 0] = 255           # blue channel in BGR
    t = FI.to_tensor(img)
    assert t.shape == (1, 3, 640, 640)
    assert t.dtype == np.float32
    # BGR->RGB means the blue channel must land in plane 2, not plane 0
    assert t[0, 2].max() == pytest.approx(1.0)
    assert t[0, 0].max() == pytest.approx(0.0)


# ------------------------------------------------------------------- NMS
def test_nms_suppresses_overlap_and_keeps_the_best():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]],
                     np.float32)
    scores = np.array([0.9, 0.8, 0.7], np.float32)
    keep = FI._nms(boxes, scores, 0.5)
    assert keep == [0, 2]


def test_nms_keeps_overlap_when_the_threshold_allows_it():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11]], np.float32)
    scores = np.array([0.9, 0.8], np.float32)
    assert FI._nms(boxes, scores, 0.95) == [0, 1]


# ------------------------------------------------- the whole postprocess
class _StubBackend:
    """Returns one detection at a chosen model-space box, YOLOv8 layout."""

    def __init__(self, cx, cy, w, h, cls=0, conf=0.9, nc=4):
        self.args = (cx, cy, w, h, cls, conf, nc)

    def __call__(self, _tensor):
        cx, cy, w, h, cls, conf, nc = self.args
        out = np.zeros((1, 4 + nc, 2), np.float32)
        out[0, :4, 0] = (cx, cy, w, h)
        out[0, 4 + cls, 0] = conf
        # a second anchor well below threshold, so the conf filter is exercised
        out[0, :4, 1] = (10, 10, 4, 4)
        out[0, 4, 1] = 0.001
        return [out]


def test_boxes_come_back_in_the_original_images_coordinates():
    """The mapping that undoes the letterbox — a sign error here would move
    every detection by the padding, which is 140px on a 640x360 frame."""
    img = np.zeros((360, 640, 3), np.uint8)
    # model space: centred horizontally, 140px of padding above the frame
    ov = _StubBackend(cx=320, cy=320, w=40, h=40)
    xyxy, conf, cls = FI.detect(ov, img, imgsz=640, conf=0.5, iou=0.7,
                                classes=[0])
    assert xyxy.shape == (1, 4)
    assert xyxy[0] == pytest.approx([300.0, 160.0, 340.0, 200.0])
    assert conf[0] == pytest.approx(0.9)
    assert cls[0] == 0


def test_boxes_are_clipped_to_the_frame():
    img = np.zeros((360, 640, 3), np.uint8)
    ov = _StubBackend(cx=10, cy=150, w=80, h=80)      # runs off the left edge
    xyxy, _c, _k = FI.detect(ov, img, imgsz=640, conf=0.5, iou=0.7, classes=[0])
    assert xyxy[0][0] >= 0.0
    assert xyxy[0][1] >= 0.0


def test_a_class_that_was_not_requested_is_dropped():
    img = np.zeros((360, 640, 3), np.uint8)
    ov = _StubBackend(cx=320, cy=320, w=40, h=40, cls=2)
    xyxy, _c, _k = FI.detect(ov, img, imgsz=640, conf=0.5, iou=0.7, classes=[0])
    assert xyxy.shape[0] == 0


def test_everything_below_the_confidence_threshold_yields_nothing():
    img = np.zeros((360, 640, 3), np.uint8)
    ov = _StubBackend(cx=320, cy=320, w=40, h=40, conf=0.2)
    xyxy, _c, _k = FI.detect(ov, img, imgsz=640, conf=0.5, iou=0.7, classes=[0])
    assert xyxy.shape[0] == 0


# --------------------------------------------------------------- fallback
def test_disabled_falls_back():
    assert FI.try_detect(object(), np.zeros((360, 640, 3), np.uint8),
                         imgsz=640, conf=0.5, iou=0.7, enabled=False) is None


def test_a_model_without_a_compiled_backend_falls_back():
    """Also the first call on a real model: Ultralytics builds `predictor`
    lazily, so the compiled model is unreachable until one predict() has run.
    Returning None keeps that call on the normal path instead of failing."""
    class _Bare:
        pass
    assert FI.try_detect(_Bare(), np.zeros((360, 640, 3), np.uint8),
                         imgsz=640, conf=0.5, iou=0.7) is None


def test_a_backend_that_raises_falls_back_rather_than_killing_the_run():
    class _Boom:
        def __call__(self, _t):
            raise RuntimeError("inference failed")
    assert FI.detect(_Boom(), np.zeros((360, 640, 3), np.uint8),
                     imgsz=640, conf=0.5, iou=0.7, classes=[0]) is None
