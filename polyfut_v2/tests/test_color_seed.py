"""Tests for Stage 0 seed + jersey-colour primitives."""

import numpy as np

from polyfut_v2.pipeline.color import (
    hsv_distance, jersey_hsv, median_hsv, torso_hsv,
)
from polyfut_v2.pipeline.player_detector import PlayerDetection
from polyfut_v2.pipeline.seed import build_seed_from_taps, build_seed_from_torso_crops

RED = (0, 0, 200)    # BGR
BLUE = (200, 0, 0)   # BGR


def _solid(color, h=80, w=60):
    return np.full((h, w, 3), color, dtype=np.uint8)


def test_torso_hsv_on_solid_player():
    frame = _solid(RED, 100, 100)
    hsv = torso_hsv(frame, [10, 10, 90, 90])
    assert hsv is not None
    assert 0 <= hsv[0] <= 179
    # Red sits near hue 0.
    assert hsv[0] < 10 or hsv[0] > 170


def _hsv_solid(h, s, v, hh=80, ww=60):
    """A solid block of a given HSV colour, as a BGR image."""
    import cv2
    hsv = np.full((hh, ww, 3), (h, s, v), np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def test_jersey_hsv_ignores_grass_and_recovers_kit():
    import cv2
    grass = _hsv_solid(55, 180, 150)                 # green pitch
    # A distant black player: torso is mostly grass with a dark jersey patch.
    frame = grass.copy()
    frame[30:55, 20:45] = _hsv_solid(0, 20, 30, 25, 25)   # dark jersey pixels
    kit = jersey_hsv(frame, [0, 0, 60, 80])
    assert kit is not None
    assert kit[2] < 90                               # reads DARK, not bright green
    # A yellow player likewise: grass removed → yellow survives.
    frame2 = grass.copy()
    frame2[25:60, 18:48] = _hsv_solid(27, 200, 200, 35, 30)
    kit2 = jersey_hsv(frame2, [0, 0, 60, 80])
    assert kit2 is not None
    assert kit2[0] < 34 and kit2[2] > 120            # reads yellow, bright
    # The two kits are now clearly separable.
    assert hsv_distance(kit, kit2) > 60


def test_jersey_hsv_all_grass_returns_none():
    grass = _hsv_solid(55, 180, 150)
    assert jersey_hsv(grass, [0, 0, 60, 80]) is None  # nothing but grass → unknown


def test_hsv_distance_same_is_zero_and_circular():
    red = torso_hsv(_solid(RED, 100, 100), [10, 10, 90, 90])
    assert hsv_distance(red, red) == 0.0
    # Circular hue: 1 vs 179 is close, not 178 apart.
    a = np.array([1.0, 200.0, 200.0], dtype=np.float32)
    b = np.array([179.0, 200.0, 200.0], dtype=np.float32)
    assert hsv_distance(a, b) < hsv_distance(a, np.array([90.0, 200.0, 200.0], np.float32))


def test_hsv_distance_none_inputs():
    assert hsv_distance(None, np.zeros(3, np.float32)) is None
    assert hsv_distance(np.zeros(3, np.float32), None) is None


def test_red_vs_blue_is_far():
    red = torso_hsv(_solid(RED, 100, 100), [10, 10, 90, 90])
    blue = torso_hsv(_solid(BLUE, 100, 100), [10, 10, 90, 90])
    assert hsv_distance(red, blue) > 60.0


def test_median_hsv_reduces_outlier():
    reds = [np.array([2, 200, 200], np.float32) for _ in range(3)]
    reds.append(np.array([120, 200, 200], np.float32))  # one blue outlier
    med = median_hsv(reds)
    assert med[0] < 60  # median stays near red despite the outlier


def test_build_seed_from_crops():
    crops = [_solid(RED) for _ in range(3)]
    seed = build_seed_from_torso_crops(crops)
    assert seed.n_samples == 3
    assert seed.has_color()
    assert len(seed.gallery) == 3
    assert not seed.is_weak()


def test_single_sample_seed_is_weak():
    seed = build_seed_from_torso_crops([_solid(RED)])
    assert seed.is_weak()


class _FakeDetector:
    """Returns one player at ``bbox`` regardless of frame/tap."""
    def __init__(self, bbox, found=True):
        self.bbox, self.found = bbox, found
    def detect(self, frame, near=None):
        return [PlayerDetection(bbox=list(self.bbox), conf=0.9)] if self.found else []


def test_build_seed_from_taps_finds_player():
    frames = [(_solid(RED, 100, 100), (50, 50)) for _ in range(3)]
    seed = build_seed_from_taps(frames, _FakeDetector([10, 10, 90, 90]),
                                max_tap_dist_px=80)
    assert seed.n_samples == 3
    assert seed.has_color() and len(seed.gallery) == 3
    assert seed.kit_hsv[0] < 10 or seed.kit_hsv[0] > 170  # red


def test_build_seed_from_taps_tap_misses_player():
    # Player far from the tap → not selected; sample still counts.
    frames = [(_solid(RED, 200, 200), (10, 10))]
    seed = build_seed_from_taps(frames, _FakeDetector([150, 150, 190, 190]),
                                max_tap_dist_px=40)
    assert seed.n_samples == 1
    assert not seed.has_color() and seed.gallery == []


def test_build_seed_from_taps_grass_crop_has_no_colour():
    # A player crop that is essentially all pitch grass → no reliable kit colour
    # (grass-masked away), but the gallery crop is still kept for appearance.
    grass = (40, 130, 40)   # BGR, inside the grass hue band
    frames = [(_solid(grass, 100, 100), (50, 50)) for _ in range(2)]
    seed = build_seed_from_taps(frames, _FakeDetector([30, 30, 70, 70]),
                                max_tap_dist_px=40)
    assert seed.n_samples == 2
    assert not seed.has_color()      # all grass → no reliable colour
    assert len(seed.gallery) == 2    # crops still gathered for appearance
