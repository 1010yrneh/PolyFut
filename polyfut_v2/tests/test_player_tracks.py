"""Issue 15, step 2: linking per-frame player boxes into tracklets.

The properties that matter are the ones forced by this footage rather than by
textbook tracking: a gate that means the same thing at both ends of the pitch,
association in camera-compensated space, tolerance of the gaps that come from
players arriving on only ~54% of frames, and a hard refusal to link across a cut.
"""

from __future__ import annotations

import numpy as np
import pytest

from polyfut_v2.pipeline.player_tracks import (
    PlayerTracker,
    _allowance_px,
)


def _dets(*boxes, conf=0.9, cls=2):
    xyxy = np.array([list(b) for b in boxes], np.float32).reshape(-1, 4)
    return (xyxy, np.full(len(xyxy), conf, np.float32),
            np.full(len(xyxy), cls, np.float32))


def _box(cx, cy, h=26.0, w=11.0):
    return (cx - w/2, cy - h/2, cx + w/2, cy + h/2)


# ------------------------------------------------------------ the gate
def test_the_gate_is_measured_in_metres_not_pixels():
    """One pixel is worth 5.7x more distance at the far touchline than near the
    camera, so a fixed pixel gate is two different rules."""
    near = _allowance_px(40.0, 0.5)      # a close player
    far = _allowance_px(12.0, 0.5)       # the same player, far away
    assert near > far
    # both correspond to the same real distance
    assert near / 40.0 == pytest.approx(far / 12.0, rel=1e-6)


def test_the_gate_grows_with_the_gap():
    """Players arrive on ~54% of frames, so gaps are the normal case."""
    assert _allowance_px(26.0, 1.0) > _allowance_px(26.0, 0.1)


def test_the_gate_refuses_an_impossible_sprint():
    """~11 m/s of headroom; 40 m in a quarter second is not a player."""
    t = PlayerTracker()
    t.update(0, 0.0, _dets(_box(100, 200)))
    t.update(1, 0.25, _dets(_box(500, 200)))
    assert len(t.tracks) == 2, "a 400px jump was linked as one player"


# ------------------------------------------------------- basic linking
def test_a_walking_player_stays_one_track():
    t = PlayerTracker()
    for k in range(10):
        t.update(k, k * 0.13, _dets(_box(100 + k * 3, 200)))
    assert len(t.tracks) == 1
    assert len(t.tracks[0].points) == 10


def test_two_separated_players_stay_separate():
    t = PlayerTracker()
    for k in range(6):
        t.update(k, k * 0.13, _dets(_box(100 + k*2, 200), _box(400 - k*2, 210)))
    assert len(t.tracks) == 2
    assert all(len(tr.points) == 6 for tr in t.tracks)


def test_a_missed_frame_does_not_break_the_track():
    """A detector miss is not evidence the player left the pitch."""
    t = PlayerTracker()
    t.update(0, 0.0, _dets(_box(100, 200)))
    t.update(1, 0.13, None)                       # nothing found this frame
    t.update(2, 0.26, _dets(_box(106, 200)))
    assert len(t.tracks) == 1
    assert len(t.tracks[0].points) == 2


def test_a_long_silence_ends_the_track():
    t = PlayerTracker()
    t.update(0, 0.0, _dets(_box(100, 200)))
    t.update(60, 8.0, _dets(_box(104, 200)))      # 8s later, past max_gap
    assert len(t.tracks) == 2


def test_the_nearest_pairing_is_committed_first():
    """Greedy in ascending distance: the confident match must not be stolen by
    an earlier-indexed track that is further away."""
    t = PlayerTracker()
    t.update(0, 0.0, _dets(_box(100, 200), _box(140, 200)))
    ids = t.update(1, 0.13, _dets(_box(141, 200), _box(101, 200)))
    # detection 0 is nearest the SECOND track, detection 1 nearest the first
    assert ids[0] != ids[1]
    assert len(t.tracks) == 2


# ------------------------------------------------- camera compensation
class _Camera:
    """Pans by a fixed shift per frame; refuses across a shot boundary."""

    def __init__(self, dx=0.0, cut_after=None):
        self.dx = dx
        self.cut_after = cut_after

    def relative_by_frame(self, a, b):
        if self.cut_after is not None and (a <= self.cut_after < b):
            return None                     # a cut: never compose across it
        return np.array([[1.0, 0.0, self.dx * (b - a)],
                         [0.0, 1.0, 0.0],
                         [0.0, 0.0, 1.0]])


# The pan has to exceed what the gate would allow anyway, or the pair of tests
# below proves nothing: at h=26px and dt=0.13s the allowance is already ~39px,
# so a 30px pan is indistinguishable from a player jogging and BOTH variants
# would link it. 60px/frame is unambiguously camera motion.
_PAN = 60.0


def test_a_pan_does_not_split_a_stationary_player():
    """In raw pixels a fast pan moves everyone; compensated, nobody moves."""
    cam = _Camera(dx=_PAN)
    t = PlayerTracker(camera=cam)
    for k in range(6):
        t.update(k, k * 0.13, _dets(_box(100 + _PAN * k, 200)))
    assert len(t.tracks) == 1, "the pan was read as player motion"


def test_without_compensation_the_same_pan_would_split_it():
    """The control: shows the test above is actually exercising the camera
    path rather than passing because the gate is loose."""
    t = PlayerTracker(camera=None)
    for k in range(6):
        t.update(k, k * 0.13, _dets(_box(100 + _PAN * k, 200)))
    assert len(t.tracks) > 1


def test_no_track_survives_a_cut():
    """The camera jump across a cut is never measured, so a link across one is
    a link between two different people."""
    cam = _Camera(dx=0.0, cut_after=2)
    t = PlayerTracker(camera=cam)
    for k in range(6):
        t.update(k, k * 0.13, _dets(_box(100, 200)))
    assert len(t.tracks) == 2


# ------------------------------------------------------------- output
def test_single_sightings_are_not_reported_as_tracks():
    t = PlayerTracker()
    t.update(0, 0.0, _dets(_box(100, 200)))
    t.update(1, 0.13, _dets(_box(103, 200), _box(500, 300)))
    kept = t.finished(min_points=2)
    assert len(kept) == 1
    assert len(t.tracks) == 2


def test_camera_compensation_is_off_by_default_in_config():
    """Measured to fragment tracks at this cadence: the camera moves a median
    2.79px between harvested frames against a ~39px gate, so it adds noise
    without adding information. Kept available, not enabled."""
    from polyfut_v2.config import PipelineV2Config

    cfg = PipelineV2Config()
    assert cfg.track_use_camera is False
    t = PlayerTracker(cfg, camera=_Camera(dx=_PAN))
    assert t.camera is None, "config said off but the camera was used anyway"


def test_the_config_flag_can_turn_it_back_on():
    from polyfut_v2.config import PipelineV2Config

    cfg = PipelineV2Config()
    cfg.track_use_camera = True
    t = PlayerTracker(cfg, camera=_Camera(dx=_PAN))
    assert t.camera is not None


def test_a_track_reports_its_span():
    t = PlayerTracker()
    for k in range(5):
        t.update(k, k * 0.5, _dets(_box(100 + k, 200)))
    d = t.tracks[0].to_dict()
    assert d["n_points"] == 5
    assert d["duration_sec"] == pytest.approx(2.0)
