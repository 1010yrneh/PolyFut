"""Tests for the pure ball-detector geometry helpers (no model needed)."""

import cv2
import numpy as np

from polyfut_v2.pipeline.ball_detector import (
    BallDetection,
    map_bbox_to_full,
    parse_best_ball,
    roi_crop,
)
from polyfut_v2.pipeline.color import is_bbox_on_foreign_surface


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


# --------------------------------------------------------------------------- #
# Stage 3e pitch gate.
#
# The gate asks "is this positively a NON-pitch surface?" rather than "is there
# enough grass?". The old grass-fraction form needed the turf to fall inside a
# narrow hue band, so on bright sun-bleached pitches (measured hue ~28 against a
# band starting at 32) it rejected ~49% of real on-pitch ball positions. These
# tests pin both directions: a saturated foreign surface is rejected, and turf of
# ANY plausible hue — plus anything unreadable — is kept.
# --------------------------------------------------------------------------- #

def _hsv_fill(h, s, v, shape=(360, 640)):
    img = np.full((shape[0], shape[1], 3), (h, s, v), np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)


def test_ball_gate_rejects_blue_running_track():
    frame = _hsv_fill(55, 180, 150)                       # green pitch
    frame[0:80, :] = _hsv_fill(105, 200, 170, (80, 640))  # blue track strip
    assert is_bbox_on_foreign_surface(frame, [300, 20, 310, 30]) is True


def test_ball_gate_rejects_red_running_track():
    frame = _hsv_fill(55, 180, 150)
    frame[0:80, :] = _hsv_fill(3, 200, 170, (80, 640))    # red/terracotta track
    assert is_bbox_on_foreign_surface(frame, [300, 20, 310, 30]) is True


def test_ball_gate_keeps_ball_on_deep_green_turf():
    frame = _hsv_fill(45, 180, 150)
    assert is_bbox_on_foreign_surface(frame, [300, 200, 310, 210]) is False


def test_ball_gate_keeps_ball_on_sunbleached_yellow_green_turf():
    """The regression that mattered: hue 28 turf must not read as foreign."""
    frame = _hsv_fill(28, 170, 190)
    assert is_bbox_on_foreign_surface(frame, [300, 200, 310, 210]) is False


def test_ball_gate_keeps_ball_on_a_white_line():
    """A washed-out probe has no hue opinion, so it must abstain (keep)."""
    frame = _hsv_fill(55, 180, 150)
    frame[190:230, :] = _hsv_fill(0, 8, 245, (40, 640))   # painted line
    assert is_bbox_on_foreign_surface(frame, [300, 200, 310, 210]) is False


def test_ball_gate_ignores_the_balls_own_pixels():
    """A bright white ball fills much of the probe; it must not decide the gate."""
    frame = _hsv_fill(28, 170, 190)
    frame[198:212, 298:312] = _hsv_fill(0, 5, 250, (14, 14))   # the ball itself
    assert is_bbox_on_foreign_surface(frame, [298, 198, 312, 212]) is False


def test_ball_gate_keeps_on_a_mixed_touchline_sample():
    """Grass plus some foreign colour must survive — only a clearly foreign
    sample is rejected, so a ball on the touchline isn't thrown away."""
    frame = _hsv_fill(30, 175, 180)
    frame[:, 0:310] = _hsv_fill(105, 200, 170, (360, 310))     # ~half blue
    assert is_bbox_on_foreign_surface(frame, [300, 200, 310, 210]) is False


def test_ball_gate_keeps_when_the_probe_is_degenerate():
    frame = _hsv_fill(55, 180, 150)
    assert is_bbox_on_foreign_surface(frame, [1, 1, 3, 3], check_half_px=1.0) is False
    assert is_bbox_on_foreign_surface(None, [0, 0, 10, 10]) is False
    assert is_bbox_on_foreign_surface(frame, [0, 0, 10]) is False
