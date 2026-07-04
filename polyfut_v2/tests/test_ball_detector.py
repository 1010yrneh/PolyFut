"""Tests for the pure ball-detector geometry helpers (no model needed)."""

import numpy as np

from polyfut_v2.pipeline.ball_detector import (
    BallDetection,
    map_bbox_to_full,
    parse_best_ball,
    roi_crop,
)


def test_roi_crop_centered():
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    crop, (ox, oy) = roi_crop(frame, (320, 180), 50)
    assert crop.shape[0] == 100 and crop.shape[1] == 100
    assert (ox, oy) == (270, 130)


def test_roi_crop_clamps_at_edge():
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    crop, (ox, oy) = roi_crop(frame, (5, 5), 50)
    assert (ox, oy) == (0, 0)
    assert crop.shape[0] > 0 and crop.shape[1] > 0
    # Bottom-right corner clamps to frame bounds.
    crop2, (ox2, oy2) = roi_crop(frame, (639, 359), 50)
    assert ox2 == 589 and oy2 == 309


def test_map_bbox_to_full():
    assert map_bbox_to_full([10, 20, 30, 40], (100, 200)) == [110, 220, 130, 240]


def test_parse_best_ball_picks_highest_conf():
    xyxy = np.array([[0, 0, 10, 10], [5, 5, 15, 15], [1, 1, 2, 2]], dtype=float)
    conf = np.array([0.30, 0.80, 0.90], dtype=float)
    cls = np.array([32, 32, 0], dtype=float)  # last is a person, ignored
    det = parse_best_ball(xyxy, conf, cls, ball_class_id=32, conf_min=0.07)
    assert isinstance(det, BallDetection)
    assert det.conf == 0.80
    assert det.bbox == [5.0, 5.0, 15.0, 15.0]
    assert det.center == (10.0, 10.0)


def test_parse_best_ball_respects_conf_min_and_class():
    xyxy = np.array([[0, 0, 10, 10]], dtype=float)
    # Below conf_min → dropped.
    assert parse_best_ball(xyxy, np.array([0.05]), np.array([32.0]),
                           ball_class_id=32, conf_min=0.07) is None
    # Wrong class → dropped.
    assert parse_best_ball(xyxy, np.array([0.9]), np.array([0.0]),
                           ball_class_id=32, conf_min=0.07) is None
