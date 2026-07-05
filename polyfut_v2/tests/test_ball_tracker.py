"""Tests for the continuous ball tracker (Stage 3) with a fake detector."""

import numpy as np
import pytest

from polyfut_v2.config import PipelineV2Config
from polyfut_v2.pipeline.ball_detector import BallDetection
from polyfut_v2.pipeline.ball_tracker import track_ball


class FakeDetector:
    """Returns scripted detections keyed by frame_index; None = a miss."""

    def __init__(self, script: dict[int, BallDetection | None]):
        self.script = script
        self.calls: list[tuple[int, tuple | None]] = []
        self._idx = 0

    def detect(self, frame, last_center=None):
        idx = int(frame[0, 0, 0])  # we stash frame_index in the pixel
        self.calls.append((idx, last_center))
        return self.script.get(idx)


def _frame(idx):
    f = np.zeros((32, 32, 3), dtype=np.uint8)
    f[0, 0, 0] = idx
    return f


def _det(x, y):
    return BallDetection(bbox=[x - 5, y - 5, x + 5, y + 5], conf=0.9)


def _stream(frames):
    for idx, t in frames:
        yield idx, t, _frame(idx)


def test_tracker_basic_flow_and_gap_skip():
    # shot1: t in [0,1], shot2: t in [2,3]; frames at 1.4/1.8 fall in the gap.
    frames = [
        (0, 0.0), (1, 0.2), (2, 0.4), (3, 0.6), (4, 0.8), (5, 1.0),
        (6, 1.4), (7, 1.8),                      # removed gap
        (8, 2.0), (9, 2.2), (10, 2.4),
    ]
    script = {
        0: _det(100, 100), 1: _det(105, 100),
        2: None, 3: None,                        # two misses → held (hold=2)
        4: _det(120, 100), 5: _det(125, 100),
        8: _det(200, 100), 9: _det(205, 100), 10: _det(210, 100),
    }
    cfg = PipelineV2Config(ball_hold_frames=2)
    det = FakeDetector(script)
    live_shots = [
        {"start_sec": 0.0, "end_sec": 1.0},
        {"start_sec": 2.0, "end_sec": 3.0},
    ]

    traj = track_ball(_stream(frames), live_shots, det, cfg)

    # Gap frames (6, 7) are skipped.
    assert len(traj) == 9
    assert 6 not in {s.frame_index for s in traj.samples}

    by_idx = {s.frame_index: s for s in traj.samples}
    assert by_idx[0].detected and by_idx[0].x == 100
    # Misses are held from the last detection (idx 1, x=105).
    assert by_idx[2].interpolated and not by_idx[2].detected
    assert by_idx[2].x == 105 and by_idx[2].has_position()
    assert by_idx[3].interpolated and by_idx[3].x == 105
    assert by_idx[4].detected and by_idx[4].x == 120


def test_processed_sec_collapses_gap_and_is_monotonic():
    frames = [
        (0, 0.0), (1, 0.5), (2, 1.0),
        (3, 2.0), (4, 2.5),
    ]
    script = {i: _det(100 + i, 100) for i in range(5)}
    cfg = PipelineV2Config(ball_hold_frames=2)
    live_shots = [
        {"start_sec": 0.0, "end_sec": 1.0},
        {"start_sec": 2.0, "end_sec": 3.0},
    ]

    traj = track_ball(_stream(frames), live_shots, FakeDetector(script), cfg)
    pts = traj.samples
    ps = [s.processed_sec for s in pts]

    assert ps == sorted(ps)  # monotonic non-decreasing
    # processed_sec must ADVANCE within the first shot even though it starts at
    # t=0 (regression: a `shot_first_t or t_sec` falsy-zero bug collapsed every
    # processed_sec to 0, which still passed a monotonicity-only check).
    assert pts[0].processed_sec == pytest.approx(0.0)
    assert pts[1].processed_sec == pytest.approx(0.5)
    assert pts[2].processed_sec == pytest.approx(1.0)
    # First shot spans 1.0s of play; shot-2 first frame continues from there,
    # collapsing the 1.0s removed gap.
    assert pts[3].frame_index == 3
    assert pts[3].processed_sec == pytest.approx(1.0)
    assert pts[4].processed_sec == pytest.approx(1.5)


def test_shot_boundary_resets_roi_anchor():
    # First frame of shot 2 must query the detector with last_center=None
    # (cold re-acquire), not a stale center carried across the cut.
    frames = [(0, 0.0), (1, 1.0), (2, 2.0)]
    script = {0: _det(100, 100), 1: _det(110, 100), 2: _det(300, 100)}
    cfg = PipelineV2Config(ball_hold_frames=2)
    live_shots = [
        {"start_sec": 0.0, "end_sec": 1.0},
        {"start_sec": 2.0, "end_sec": 3.0},
    ]
    det = FakeDetector(script)
    track_ball(_stream(frames), live_shots, det, cfg)

    last_center_by_idx = {idx: lc for idx, lc in det.calls}
    assert last_center_by_idx[0] is None          # cold start
    assert last_center_by_idx[1] is not None       # warm within shot 1
    assert last_center_by_idx[2] is None           # reset across shot boundary
