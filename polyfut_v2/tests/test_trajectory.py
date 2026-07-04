"""Tests for the ball-trajectory data structures."""

from polyfut_v2.pipeline.trajectory import BallSample, BallTrajectory


def _detected(idx, x, y):
    return BallSample(
        frame_index=idx, t_sec=idx * 0.1, processed_sec=idx * 0.1,
        x=x, y=y, bbox=[x - 5, y - 5, x + 5, y + 5],
        conf=0.9, detected=True, interpolated=False,
    )


def _missing(idx):
    return BallSample(
        frame_index=idx, t_sec=idx * 0.1, processed_sec=idx * 0.1,
        x=None, y=None, bbox=None, conf=0.0, detected=False, interpolated=False,
    )


def test_sample_roundtrip():
    s = _detected(3, 100.0, 120.0)
    s2 = BallSample.from_dict(s.to_dict())
    assert s2.frame_index == 3
    assert s2.x == 100.0 and s2.y == 120.0
    assert s2.detected is True
    assert s2.has_position()


def test_missing_sample_has_no_position():
    s = _missing(1)
    assert not s.has_position()
    assert s.to_dict()["x"] is None


def test_trajectory_summary_counts():
    traj = BallTrajectory()
    traj.add(_detected(0, 10, 10))
    traj.add(BallSample(1, 0.1, 0.1, 12, 10, [7, 5, 17, 15], 0.5, False, True))  # held
    traj.add(_missing(2))
    d = traj.to_dict()
    assert d["count"] == 3
    assert d["detected"] == 1
    assert d["interpolated"] == 1
    assert d["missing"] == 1
    assert len(traj.positioned()) == 2  # detected + held
    assert abs(traj.detected_ratio() - 1 / 3) < 1e-6


def test_trajectory_dict_roundtrip():
    traj = BallTrajectory()
    traj.add(_detected(0, 10, 10))
    traj.add(_missing(1))
    traj2 = BallTrajectory.from_dict(traj.to_dict())
    assert len(traj2) == 2
    assert traj2.samples[0].detected
    assert not traj2.samples[1].has_position()
