"""Tests for Stage 5 player detection: parsing + ROI offset mapping."""

import numpy as np

from polyfut_v2.config import PipelineV2Config
from polyfut_v2.pipeline.player_detector import (
    YoloPlayerDetector,
    _parse_players,
)


def test_parse_players_filters_class_and_conf():
    xyxy = np.array([[0, 0, 10, 20], [5, 5, 15, 25], [1, 1, 2, 2]], dtype=float)
    conf = np.array([0.9, 0.10, 0.8], dtype=float)
    cls = np.array([0, 0, 32], dtype=float)  # last is a ball, not a person
    out = _parse_players(xyxy, conf, cls, player_class_id=0, conf_min=0.25)
    assert len(out) == 1  # 0.10 dropped by conf, class-32 dropped by class
    assert out[0].bbox == [0.0, 0.0, 10.0, 20.0]
    assert out[0].conf == 0.9


# --- Minimal ultralytics-shaped fake so detect() can be exercised offline ---

class _Arr:
    def __init__(self, a): self._a = a
    def cpu(self): return self
    def numpy(self): return self._a


class _Boxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy, self.conf, self.cls = _Arr(xyxy), _Arr(conf), _Arr(cls)
    def __len__(self): return len(self.xyxy._a)


class _Res:
    def __init__(self, xyxy, conf, cls):
        self.boxes = _Boxes(xyxy, conf, cls)


class FakeModel:
    """Returns one person box in crop-local coords, regardless of the image."""
    def __init__(self, box): self.box = box
    def predict(self, image, **kw):
        return [_Res(np.array([self.box], float), np.array([0.9], float),
                     np.array([0.0], float))]


def test_detect_maps_roi_box_back_to_full_frame():
    cfg = PipelineV2Config(player_roi_half_px=50.0)
    # Fake finds a box at (10,10,30,40) within the ROI crop.
    det = YoloPlayerDetector(cfg, model=FakeModel([10, 10, 30, 40]))
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    # ROI centered at (320,180), half 50 → offset (270,130).
    out = det.detect(frame, near=(320, 180))
    assert len(out) == 1
    assert out[0].bbox == [280.0, 140.0, 300.0, 170.0]  # box + offset


def test_detect_full_frame_when_no_point():
    cfg = PipelineV2Config()
    det = YoloPlayerDetector(cfg, model=FakeModel([10, 10, 30, 40]))
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    out = det.detect(frame, near=None)
    assert out[0].bbox == [10.0, 10.0, 30.0, 40.0]  # no offset
