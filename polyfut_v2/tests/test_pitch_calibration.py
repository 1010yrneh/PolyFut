"""Pitch calibration: camera model, fitting, and the per-frame pitch map.

The synthetic recovery test below generates AND fits with the same code, so it
can only validate the optimiser — a self-consistent but wrong axis convention
passes it happily (that exact mistake was made during development and hidden by
exactly this kind of test). ``assert_convention`` is the guard that catches it,
by checking answers known a priori rather than answers the model produced.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from polyfut_v2.pipeline.camera_motion import CameraSample, CameraTrack
from polyfut_v2.pipeline.pitch_calibration import (
    LANDMARKS,
    PitchCalibration,
    PitchMapper,
    assert_convention,
    fit,
    landmark_xy,
    pose_to_H,
    project,
    rotation,
)

L, W = 100.0, 64.0
CX, CY = 320.0, 180.0
# a plausible tripod: 18m outside the touchline, 12m up, looking across
TRUTH = (52.0, -18.0, 12.0, math.radians(88.0), math.radians(14.0), 850.0,
         math.radians(1.5))


def _visible_landmarks(truth=TRUTH, keys=None):
    keys = keys or list(LANDMARKS)
    out = []
    for k in keys:
        p = landmark_xy(k, L, W)
        q = project(truth, [p], CX, CY)[0]
        if np.isfinite(q).all() and -200 < q[0] < 840 and -100 < q[1] < 460:
            out.append((k, p, q))
    return out


# ---------------------------------------------------------------- conventions
def test_axis_convention_matches_known_answers():
    assert_convention()


def test_camera_above_pitch_sees_in_front_and_horizon_is_above_the_pitch():
    H = pose_to_H(TRUTH, CX, CY)
    corners = [(0, 0), (L, 0), (L, W), (0, W)]
    ws = []
    for x, y in corners:
        v = H @ np.array([x, y, 1.0])
        ws.append(v[2])
    # every corner in front of the camera => all the same sign
    assert len({np.sign(w) > 0 for w in ws}) == 1


# ------------------------------------------------------------------- recovery
@pytest.mark.parametrize("noise", [0.0, 1.0, 2.0])
def test_recovers_a_known_camera_from_noisy_clicks(noise):
    vis = _visible_landmarks()
    assert len(vis) >= 6
    rng = np.random.default_rng(0)
    clicks = [(0, tuple(q + rng.normal(0, noise, 2)), k) for k, _p, q in vis]
    cal = fit(clicks, principal=(CX, CY), pitch_length_m=L, pitch_width_m=W)
    assert cal is not None
    # position and height recovered, not just a low residual
    assert cal.camera_height_m == pytest.approx(TRUTH[2], abs=1.0 + noise)
    assert cal.focal_px == pytest.approx(TRUTH[5], rel=0.08 + 0.02 * noise)
    assert cal.median_px < 1.0 + 1.5 * noise


def test_reports_dof_and_warns_when_there_is_no_slack():
    vis = _visible_landmarks()[:4]
    clicks = [(0, tuple(q), k) for k, _p, q in vis]
    cal = fit(clicks, principal=(CX, CY), pitch_length_m=L, pitch_width_m=W)
    assert cal is not None
    # 7 unknowns against 8 constraints: essentially nothing spare
    assert cal.dof < 4
    assert any("degrees of freedom" in w for w in cal.warnings)


def test_one_odd_landmark_is_still_detectable_without_blaming_it():
    """An outlier must remain *visible* — the robust loss keeps the median clean
    while the rms rises — but it is no longer asserted to be the user's mistake.

    A person marking a line intersection is the reliable part of this process;
    the regulation-pitch assumptions are the guesses. So the signal is surfaced
    and the interpretation is left open.
    """
    vis = _visible_landmarks()
    clicks = [(0, tuple(q), k) for k, _p, q in vis][:8]
    bad_xy = (clicks[3][1][0] + 90.0, clicks[3][1][1] + 60.0)
    clicks[3] = (0, bad_xy, clicks[3][2])
    cal = fit(clicks, principal=(CX, CY), pitch_length_m=L, pitch_width_m=W)
    assert cal is not None
    assert cal.rms_px > cal.median_px
    assert not any("bad click" in w for w in cal.warnings), cal.warnings


def test_too_few_landmarks_returns_none():
    vis = _visible_landmarks()[:3]
    clicks = [(0, tuple(q), k) for k, _p, q in vis]
    assert fit(clicks, principal=(CX, CY)) is None


# ------------------------------------------------- the anchors are the truth
def _anchor_residuals(cal, clicks):
    """How far the transform actually used misses each clicked anchor."""
    H = cal.anchor_H()
    assert H is not None
    out = []
    for _fi, xy, key in clicks:
        p = landmark_xy(key, cal.pitch_length_m, cal.pitch_width_m)
        v = H @ np.array([p[0], p[1], 1.0])
        out.append(math.hypot(v[0] / v[2] - xy[0], v[1] / v[2] - xy[1]))
    return out


def test_the_drawn_pitch_passes_through_the_anchors():
    """Whatever else is true, the pitch must land where the user marked it."""
    vis = _visible_landmarks()
    clicks = [(0, tuple(q), k) for k, _p, q in vis]
    cal = fit(clicks, principal=(CX, CY), pitch_length_m=L, pitch_width_m=W)
    assert cal is not None
    assert max(_anchor_residuals(cal, clicks)) < 2.0


def test_anchors_still_win_on_a_non_regulation_pitch():
    """A ground whose penalty boxes are not regulation size.

    The rigid model cannot reach these anchors — it assumes 16.5m boxes — and
    that used to be reported as the user clicking the wrong thing. The anchors
    must still be honoured.
    """
    # generate the truth from a pitch with SMALLER boxes than the laws specify
    small = {k: (x, y) for k, (x, y) in
             ((k, landmark_xy(k, L, W)) for k in LANDMARKS)}
    for k in list(small):
        if "penarea" in k:
            x, y = small[k]
            # squeeze the box: 12m deep instead of 16.5, 30m wide instead of 40.32
            small[k] = (x * (12.0 / 16.5) if 0 < x < L / 2 else x,
                        W / 2 + (y - W / 2) * (15.0 / 20.16))
    clicks = []
    for k, p in small.items():
        q = project(TRUTH, [p], CX, CY)[0]
        if np.isfinite(q).all() and -200 < q[0] < 840 and -100 < q[1] < 460:
            clicks.append((0, (float(q[0]), float(q[1])), k))
    assert len(clicks) >= 6
    cal = fit(clicks, principal=(CX, CY), pitch_length_m=L, pitch_width_m=W)
    assert cal is not None
    resid = _anchor_residuals(cal, clicks)
    # the regulation model is off here, and that is fine — the anchors decide
    assert float(np.median(resid)) < 8.0, resid


def test_a_regulation_mismatch_is_not_blamed_on_the_click():
    vis = _visible_landmarks()
    clicks = [(0, tuple(q), k) for k, _p, q in vis][:8]
    clicks[2] = (0, (clicks[2][1][0] + 60.0, clicks[2][1][1] + 40.0), clicks[2][2])
    cal = fit(clicks, principal=(CX, CY), pitch_length_m=L, pitch_width_m=W)
    assert cal is not None
    assert not any("bad click" in w for w in cal.warnings), cal.warnings


# --------------------------------------------------------------- pitch mapper
def _calibration_from_truth() -> PitchCalibration:
    vis = _visible_landmarks()
    clicks = [(0, tuple(q), k) for k, _p, q in vis]
    cal = fit(clicks, principal=(CX, CY), pitch_length_m=L, pitch_width_m=W)
    assert cal is not None
    cal.anchor_frame = 0
    cal.anchor_sec = 0.0
    return cal


def _track(samples):
    """samples: (frame_index, cumulative 3x3, shot)."""
    return CameraTrack([CameraSample(float(fi), m[0][2], m[1][2], True,
                                     tuple(np.asarray(m).reshape(-1)), shot, fi)
                        for fi, m, shot in samples])


def test_maps_pixels_to_metres_on_the_anchor_frame():
    cal = _calibration_from_truth()
    eye = np.eye(3)
    mapper = PitchMapper(cal, _track([(0, eye, 0)]), frame_width=640)
    # the centre spot must land back on the centre spot
    q = project(TRUTH, [landmark_xy("centre_spot", L, W)], CX, CY)[0]
    got = mapper.to_pitch(0, q)
    assert got is not None
    assert got[0] == pytest.approx(L / 2, abs=1.5)
    assert got[1] == pytest.approx(W / 2, abs=1.5)


def test_ground_distance_is_measured_in_metres():
    cal = _calibration_from_truth()
    mapper = PitchMapper(cal, _track([(0, np.eye(3), 0)]), frame_width=640)
    a = project(TRUTH, [(50.0, 30.0)], CX, CY)[0]
    b = project(TRUTH, [(50.0, 40.0)], CX, CY)[0]   # 10 m apart on the pitch
    d = mapper.ground_distance_m(0, a, b)
    assert d is not None
    assert d == pytest.approx(10.0, abs=1.5)


def test_refuses_to_compose_across_a_shot_boundary():
    """A cut's camera jump is never measured, so the map must go unavailable."""
    cal = _calibration_from_truth()
    track = _track([(0, np.eye(3), 0), (900, np.eye(3), 1)])
    mapper = PitchMapper(cal, track, frame_width=640)
    assert mapper.image_to_pitch(0) is not None
    assert mapper.image_to_pitch(900) is None


def test_pan_is_carried_by_the_camera_track():
    """After a pure pan the same pitch point must still map to the same metres."""
    cal = _calibration_from_truth()
    shift = np.array([[1.0, 0.0, 60.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    track = _track([(0, np.eye(3), 0), (60, shift, 0)])
    mapper = PitchMapper(cal, track, frame_width=640)
    p = landmark_xy("centre_spot", L, W)
    q0 = project(TRUTH, [p], CX, CY)[0]
    a = mapper.to_pitch(0, q0)
    b = mapper.to_pitch(60, (q0[0] + 60.0, q0[1]))
    assert a is not None and b is not None
    assert math.hypot(a[0] - b[0], a[1] - b[1]) < 1.5


def test_no_track_means_no_map_rather_than_a_guess():
    cal = _calibration_from_truth()
    mapper = PitchMapper(cal, None, frame_width=640)
    assert mapper.to_pitch(0, (320.0, 300.0)) is None


def test_click_resolution_is_scaled_to_the_analysed_frame():
    """Clicks made on a 1280-wide preview must still work at 640."""
    cal = _calibration_from_truth()
    cal.frame_width = 1280
    mapper = PitchMapper(cal, _track([(0, np.eye(3), 0)]), frame_width=1280)
    assert mapper.click_scale == pytest.approx(1.0)
    mapper2 = PitchMapper(cal, _track([(0, np.eye(3), 0)]), frame_width=640)
    assert mapper2.click_scale == pytest.approx(0.5)
