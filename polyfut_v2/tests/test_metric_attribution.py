"""The metric attribution gate (Issue 14).

The point of these tests is the recall-safety rules, not the arithmetic. A gate
that drops a real touch is worse than one that admits a wrong player, because the
review montage can fix the second and nothing can recover the first.
"""

from __future__ import annotations

import pytest

from polyfut_v2.pipeline.player_contacts import (
    MetricGate,
    count_contesting_players,
    nearest_player,
)
from polyfut_v2.pipeline.player_detector import PlayerDetection


def _player(x, y, h=30.0, w=12.0, cls=2, conf=0.9):
    """Box whose feet sit at (x, y)."""
    return PlayerDetection(bbox=[x - w / 2, y - h, x + w / 2, y], conf=conf,
                           class_id=cls)


def _flat_mapper(scale_m_per_px):
    """Pretend map where 1px = ``scale_m_per_px`` metres in both axes."""
    return lambda pt: (pt[0] * scale_m_per_px, pt[1] * scale_m_per_px)


BALL = (300.0, 300.0)


def test_far_field_player_is_rejected_in_metres_but_passes_in_pixels():
    """The core bug: 60px is ~4m near the camera and ~37m at the far touchline."""
    near_side = _player(360.0, 300.0)          # 60px from the ball
    coarse = MetricGate(_flat_mapper(0.62), max_dist_m=8.0, override_px=25.0)
    fine = MetricGate(_flat_mapper(0.11), max_dist_m=8.0, override_px=25.0)

    # identical pixels, identical pixel gate — only the ground scale differs
    got_px, _ = nearest_player([near_side], BALL, 80.0)
    assert got_px is not None

    got_far, _ = nearest_player([near_side], BALL, 80.0, metric=coarse)
    assert got_far is None, "37m away must not be attributed the touch"

    got_near, _ = nearest_player([near_side], BALL, 80.0, metric=fine)
    assert got_near is not None, "4m away is a plausible toucher"


def test_image_adjacency_overrides_the_metric_gate():
    """Protects headers: the map cannot see ball height, only ground position."""
    on_the_ball = _player(315.0, 300.0)        # 15px away in the image
    gate = MetricGate(_flat_mapper(0.62), max_dist_m=8.0, override_px=25.0)
    assert gate.ok(BALL, on_the_ball.bbox, 15.0)
    got, _ = nearest_player([on_the_ball], BALL, 80.0, metric=gate)
    assert got is not None


def test_unmeasurable_geometry_keeps_the_candidate():
    """No camera track / past a cut / above the horizon -> pixel gate alone."""
    p = _player(360.0, 300.0)
    gate = MetricGate(lambda pt: None, max_dist_m=1.0, override_px=0.0)
    assert gate.distance_m(BALL, p.bbox) is None
    assert gate.ok(BALL, p.bbox, 60.0)
    got, _ = nearest_player([p], BALL, 80.0, metric=gate)
    assert got is not None


def test_distance_is_measured_to_the_feet_not_the_box_centre():
    """A tall box's centre is metres above the ground; only feet project right."""
    tall = _player(300.0, 300.0, h=120.0)
    gate = MetricGate(_flat_mapper(0.1), max_dist_m=8.0)
    d = gate.distance_m(BALL, tall.bbox)
    assert d == pytest.approx(0.0, abs=1e-6)   # feet are exactly on the ball


def test_metric_gate_never_widens_the_pixel_cap():
    """Metres decide within the pixel cap; they cannot admit something beyond it."""
    way_out = _player(600.0, 300.0)            # 294px away
    gate = MetricGate(_flat_mapper(0.001), max_dist_m=8.0)   # ~0.3m in metres
    got, _ = nearest_player([way_out], BALL, 80.0, metric=gate)
    assert got is None


def test_crowd_count_uses_metres_so_far_side_play_is_not_a_fake_scramble():
    players = [_player(300.0 + 20 * i, 300.0) for i in range(5)]
    coarse = MetricGate(_flat_mapper(0.62), max_dist_m=9.0, override_px=25.0)
    fine = MetricGate(_flat_mapper(0.11), max_dist_m=9.0, override_px=25.0)

    raw = count_contesting_players(players, BALL, 90.0)
    far = count_contesting_players(players, BALL, 90.0, metric=coarse)
    near = count_contesting_players(players, BALL, 90.0, metric=fine)

    assert raw == 5
    assert far < raw, "at the far touchline 90px spans ~56m — not a scramble"
    assert near == 5, "near the camera 90px really is ~9m — a real scramble"


def test_gate_is_inert_when_not_supplied():
    """Uncalibrated videos must behave exactly as before."""
    players = [_player(360.0, 300.0), _player(310.0, 300.0)]
    a, da = nearest_player(players, BALL, 80.0)
    b, db = nearest_player(players, BALL, 80.0, metric=None)
    assert a is b and da == db
    assert count_contesting_players(players, BALL, 90.0) == \
        count_contesting_players(players, BALL, 90.0, metric=None)
