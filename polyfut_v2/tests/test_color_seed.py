"""Tests for Stage 0 seed + jersey-colour primitives."""

import numpy as np

from polyfut_v2.pipeline.color import hsv_distance, median_hsv, torso_hsv
from polyfut_v2.pipeline.seed import build_seed_from_torso_crops

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
