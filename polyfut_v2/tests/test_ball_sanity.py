"""Stage 3f: out-and-back excursion rejection.

Guards the failure measured on real footage: once the ROI search locks onto a
false positive it sustains the lock, and the trajectory becomes an alternation
between two points — 45% of consecutive links reversing 170-180 degrees at a
median 456 px/s, against 22.6% of real links turning <=30 degrees at 64 px/s.
"""

import math

from polyfut_v2.config import PipelineV2Config
from polyfut_v2.pipeline.ball_sanity import reject_pingpong
from polyfut_v2.pipeline.trajectory import BallSample, BallTrajectory

DT = 0.1333   # one analysed frame at sample_every_n=4 on 30fps


def _s(i, x, y, *, detected=True, t=None):
    tt = i * DT if t is None else t
    return BallSample(
        frame_index=i * 4, t_sec=tt, processed_sec=tt,
        x=x, y=y,
        bbox=None if x is None else [x - 4, y - 4, x + 4, y + 4],
        conf=0.4 if detected else 0.0,
        detected=detected and x is not None,
        interpolated=False,
    )


def _traj(points, **kw):
    return BallTrajectory(samples=[_s(i, p[0], p[1], **kw) for i, p in enumerate(points)])


def _detected(traj):
    return [(round(s.x, 1), round(s.y, 1)) for s in traj.samples if s.detected]


# --------------------------------------------------------------------------- #
# Rejects the measured failure pattern
# --------------------------------------------------------------------------- #

def test_rejects_a_sustained_alternation():
    """The measured failure: an A-B-A-B lock between two points 115px apart.

    The whole run is blanked — in an alternation there is no way to say which
    endpoint is the ball, and a velocity across it is meaningless either way.
    """
    pts = []
    for _ in range(4):
        pts += [(100, 100), (215, 100)]
    traj = _traj(pts)
    out, stats = reject_pingpong(traj)
    assert stats.n_rejected == 8
    assert _detected(out) == []
    # Demoted, not deleted: the samples remain as position-less misses.
    assert len(out.samples) == 8
    assert all(s.detected is False and s.has_position() is False and s.conf == 0.0
               for s in out.samples)


def test_a_single_out_and_back_is_left_alone():
    """Deliberate: one out-and-back is indistinguishable from a real return pass,
    so recall-safety says keep it."""
    traj = _traj([(100, 100), (217, 100), (100, 100), (104, 102)])
    out, stats = reject_pingpong(traj)
    assert stats.n_rejected == 0
    assert len(_detected(out)) == 4


def test_alternation_is_blanked_but_the_smooth_path_after_it_survives():
    pts = [(100, 100), (215, 100), (100, 100), (215, 100), (100, 100)]
    pts += [(300, 200), (330, 208), (360, 216)]      # play resumes, smoothly
    out, stats = reject_pingpong(_traj(pts))
    assert stats.n_rejected == 5
    assert _detected(out) == [(300.0, 200.0), (330.0, 208.0), (360.0, 216.0)]


def test_stats_report_what_happened():
    pts = []
    for _ in range(3):
        pts += [(100, 100), (215, 100)]
    _out, stats = reject_pingpong(_traj(pts))
    d = stats.to_dict()
    assert d["detected_before"] == 6
    assert d["rejected_pingpong"] == 6
    assert d["links_checked"] == 4


# --------------------------------------------------------------------------- #
# Recall-safety: real ball motion must survive
# --------------------------------------------------------------------------- #

def test_keeps_a_smooth_path():
    traj = _traj([(50, 100), (80, 105), (110, 111), (140, 118), (170, 126)])
    out, stats = reject_pingpong(traj)
    assert stats.n_rejected == 0
    assert len(_detected(out)) == 5


def test_keeps_a_real_touch_that_reverses_and_keeps_going():
    """A genuine deflection reverses direction but does NOT return to the start,
    so it must never be mistaken for an excursion."""
    traj = _traj([(100, 100), (200, 100), (110, 100), (20, 100)])
    out, stats = reject_pingpong(traj)
    assert stats.n_rejected == 0
    assert len(_detected(out)) == 4


def test_keeps_small_jitter_even_when_it_alternates():
    """Sub-threshold wobble on a near-stationary ball is not a false lock: the
    steps are below ``ball_pingpong_min_step_px``, so nothing qualifies."""
    pts = []
    for _ in range(4):
        pts += [(100, 100), (112, 100)]
    out, stats = reject_pingpong(_traj(pts))
    assert stats.n_rejected == 0
    assert len(_detected(out)) == 8


def test_keeps_a_wide_angle_that_is_not_a_full_reversal():
    # ~90 degree turn: a real pass being controlled and played square.
    traj = _traj([(100, 100), (200, 100), (200, 200), (200, 300)])
    out, stats = reject_pingpong(traj)
    assert stats.n_rejected == 0


def test_never_links_across_a_long_blind_stretch():
    """Two unrelated sightings either side of a gap must not form an excursion."""
    samples = [
        _s(0, 100, 100, t=0.0),
        _s(1, 217, 100, t=5.0),      # 5s later — a different phase of play
        _s(2, 100, 100, t=10.0),
    ]
    out, stats = reject_pingpong(BallTrajectory(samples=samples))
    assert stats.n_rejected == 0
    assert len(_detected(out)) == 3


# --------------------------------------------------------------------------- #
# Plumbing
# --------------------------------------------------------------------------- #

def test_misses_between_detections_are_skipped_not_treated_as_neighbours():
    """Neighbours are adjacent *detections*; interleaved misses must not hide an
    alternation from the test."""
    samples = []
    i = 0
    for x in (100, 215, 100, 215, 100):
        samples.append(_s(i, x, 100)); i += 1
        samples.append(_s(i, None, None, detected=False)); i += 1
    out, stats = reject_pingpong(BallTrajectory(samples=samples))
    assert stats.n_rejected == 5
    assert _detected(out) == []


def test_disabled_by_config_is_a_no_op():
    cfg = PipelineV2Config(ball_pingpong_reject_enabled=False)
    pts = []
    for _ in range(4):
        pts += [(100, 100), (215, 100)]
    out, stats = reject_pingpong(_traj(pts), cfg)
    assert stats.n_rejected == 0
    assert len(_detected(out)) == 8


def test_empty_and_tiny_trajectories_are_safe():
    for pts in ([], [(1, 1)], [(1, 1), (2, 2)]):
        out, stats = reject_pingpong(_traj(pts))
        assert stats.n_rejected == 0
        assert len(out.samples) == len(pts)


def test_camera_track_survives_the_pass():
    marker = object()
    pts = []
    for _ in range(4):
        pts += [(100, 100), (215, 100)]
    traj = _traj(pts)
    traj.camera = marker
    out, stats = reject_pingpong(traj)
    assert stats.n_rejected > 0            # the pass did rebuild the trajectory
    assert out.camera is marker


def test_thresholds_are_configurable():
    pts = []
    for _ in range(4):
        pts += [(100, 100), (215, 100)]
    # Requiring a longer step than the alternation disables the rejection.
    _out, stats = reject_pingpong(
        _traj(pts), PipelineV2Config(ball_pingpong_min_step_px=500.0))
    assert stats.n_rejected == 0
    # Demanding a longer run than is present also disables it.
    _out2, stats2 = reject_pingpong(
        _traj(pts), PipelineV2Config(ball_pingpong_min_alternations=99))
    assert stats2.n_rejected == 0
