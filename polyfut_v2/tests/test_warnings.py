"""Tests for the reliability guardrail (low ball-recall warning)."""

from polyfut_v2.config import PipelineV2Config
from polyfut_v2.main import trajectory_warnings
from polyfut_v2.pipeline.trajectory import BallSample, BallTrajectory


def _sample(idx, detected):
    x = 100.0 if detected else None
    return BallSample(
        frame_index=idx, t_sec=idx * 0.1, processed_sec=idx * 0.1,
        x=x, y=x, bbox=None if x is None else [x, x, x + 1, x + 1],
        conf=0.9 if detected else 0.0, detected=detected, interpolated=False,
    )


def _traj(n_detected, n_missing):
    t = BallTrajectory()
    for i in range(n_detected):
        t.add(_sample(i, True))
    for i in range(n_missing):
        t.add(_sample(n_detected + i, False))
    return t


def test_empty_trajectory_warns():
    ws = trajectory_warnings(BallTrajectory(), PipelineV2Config())
    assert len(ws) == 1
    assert "empty trajectory" in ws[0]


def test_zero_recall_warns():
    # 0 / 100 detected — the real-footage COCO case.
    ws = trajectory_warnings(_traj(0, 100), PipelineV2Config())
    assert len(ws) == 1
    assert "low ball-detection recall" in ws[0]
    assert "detected_ratio=0.000" in ws[0]


def test_healthy_trajectory_no_warning():
    # 80% recall, well above the default 0.2 threshold.
    ws = trajectory_warnings(_traj(80, 20), PipelineV2Config())
    assert ws == []


def test_threshold_is_configurable():
    traj = _traj(30, 70)  # 0.30 detected_ratio
    assert trajectory_warnings(traj, PipelineV2Config(min_detected_ratio_warn=0.2)) == []
    assert len(trajectory_warnings(traj, PipelineV2Config(min_detected_ratio_warn=0.5))) == 1
