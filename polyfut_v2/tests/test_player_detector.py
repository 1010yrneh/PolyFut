"""Tests for Stage 5 player detection: parsing + ROI offset mapping + classes."""

import numpy as np

from polyfut_v2.config import PipelineV2Config
from polyfut_v2.pipeline.player_detector import (
    YoloPlayerDetector,
    _parse_players,
    player_contact_class_ids,
    player_request_classes,
)


def test_parse_players_filters_class_and_conf():
    xyxy = np.array([[0, 0, 10, 20], [5, 5, 15, 25], [1, 1, 2, 2]], dtype=float)
    conf = np.array([0.9, 0.10, 0.8], dtype=float)
    cls = np.array([0, 0, 32], dtype=float)  # last is a ball, not a person
    out = _parse_players(
        xyxy, conf, cls, allowed_class_ids={0}, conf_min=0.25,
    )
    assert len(out) == 1  # 0.10 dropped by conf, class-32 dropped by class
    assert out[0].bbox == [0.0, 0.0, 10.0, 20.0]
    assert out[0].conf == 0.9
    assert out[0].class_id == 0


def test_parse_players_keeps_keeper_and_player_and_ref():
    xyxy = np.array(
        [[0, 0, 10, 20], [5, 5, 15, 25], [20, 20, 30, 40]], dtype=float,
    )
    conf = np.array([0.9, 0.8, 0.7], dtype=float)
    cls = np.array([1, 2, 3], dtype=float)  # keeper, player, ref
    out = _parse_players(
        xyxy, conf, cls, allowed_class_ids={1, 2, 3}, conf_min=0.25,
    )
    assert [d.class_id for d in out] == [1, 2, 3]


def test_soccer_request_and_contact_class_sets():
    cfg = PipelineV2Config(
        player_class_id=2, goalkeeper_class_id=1, referee_class_id=3,
    )
    assert player_request_classes(cfg) == [2, 1, 3]
    assert player_contact_class_ids(cfg) == {1, 2}


def test_coco_request_is_single_class():
    cfg = PipelineV2Config()  # defaults: person only
    assert player_request_classes(cfg) == [0]
    assert player_contact_class_ids(cfg) == {0}


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
    def __init__(self, box, class_id=0.0):
        self.box = box
        self.class_id = class_id
        self.last_classes = None

    def predict(self, image, **kw):
        self.last_classes = kw.get("classes")
        return [_Res(
            np.array([self.box], float),
            np.array([0.9], float),
            np.array([self.class_id], float),
        )]


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


def test_detect_requests_soccer_classes():
    cfg = PipelineV2Config(
        player_class_id=2, goalkeeper_class_id=1, referee_class_id=3,
        player_roi_half_px=0,
    )
    model = FakeModel([10, 10, 30, 40], class_id=1.0)
    det = YoloPlayerDetector(cfg, model=model)
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    out = det.detect(frame, near=None)
    assert model.last_classes == [2, 1, 3]
    assert out[0].class_id == 1
