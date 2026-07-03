"""Tests for ball smoothing."""

from polyfut_video.pipeline.ball_smooth import BallSmoother, BallSmoothConfig


def test_ball_hold_across_misses():
    s = BallSmoother(BallSmoothConfig(max_hold_frames=3, max_jump_px=9999))
    bbox = [100.0, 100.0, 110.0, 110.0]
    out1, c1, held1 = s.update(bbox, 0.9)
    assert out1 == bbox
    assert held1 is False

    out2, c2, held2 = s.update(None, 0.0)
    assert out2 == bbox
    assert held2 is True
    assert c2 < c1


def test_ball_reset_after_hold_expires():
    s = BallSmoother(BallSmoothConfig(max_hold_frames=2))
    bbox = [50.0, 50.0, 60.0, 60.0]
    s.update(bbox, 0.8)
    s.update(None, 0.0)
    s.update(None, 0.0)
    out, _, held = s.update(None, 0.0)
    assert out is None
    assert held is False
