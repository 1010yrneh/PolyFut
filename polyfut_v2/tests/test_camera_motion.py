"""Issue 5 — camera-motion (pan) compensation for the orbital prior.

The orbital prior asks a question about motion on the pitch but measures it in
frame pixels, so a pan makes a stationary player look like they teleported.
These tests pin the compensation contract:

  * a real pan is measured and cancelled out;
  * an unmeasurable step contributes ZERO shift (never a guess);
  * shot cuts do not accumulate a fabricated jump;
  * with compensation wired in, a panned-but-stationary player keeps a full
    orbital prior, while a genuinely moving player is still penalised.
"""

from __future__ import annotations

import numpy as np
import pytest

from polyfut_v2.config import PipelineV2Config
from polyfut_v2.pipeline.camera_motion import (
    CameraMotionEstimator,
    CameraSample,
    CameraTrack,
    estimate_shift,
)
from polyfut_v2.pipeline.contacts import ContactCandidate
from polyfut_v2.pipeline.player_contacts import PlayerContact
from polyfut_v2.pipeline.scoring import score_contacts
from polyfut_v2.pipeline.seed import TargetSeed

RNG = np.random.default_rng(7)


def _textured(w=640, h=360):
    """A richly-textured frame so goodFeaturesToTrack has corners to lock onto."""
    base = RNG.integers(0, 255, size=(h, w), dtype=np.uint8)
    img = np.dstack([base, base, base])
    # Add hard rectangles for strong, unambiguous corners.
    for i in range(12):
        x = 20 + i * 50
        img[60:120, x:x + 25] = 255
        img[200:260, x:x + 25] = 0
    return img


def _shifted(img, dx, dy):
    """Translate the frame by (dx, dy) — a synthetic camera pan."""
    out = np.zeros_like(img)
    h, w = img.shape[:2]
    xs0, xs1 = max(0, dx), min(w, w + dx)
    ys0, ys1 = max(0, dy), min(h, h + dy)
    out[ys0:ys1, xs0:xs1] = img[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
    return out


def _cfg(**kw):
    base = dict(camera_motion_enabled=True, camera_motion_every_n=1,
                camera_motion_width=320, camera_motion_min_points=8,
                camera_motion_exclude_half_px=0.0)
    base.update(kw)
    return PipelineV2Config(**base)


# ---------------------------------------------------------------------------
# estimate_shift
# ---------------------------------------------------------------------------

def test_estimate_shift_measures_a_known_translation():
    import cv2

    img = _textured()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    moved = cv2.cvtColor(_shifted(img, 12, -5), cv2.COLOR_BGR2GRAY)
    shift = estimate_shift(gray, moved, min_points=6)
    assert shift is not None
    dx, dy = shift
    assert dx == pytest.approx(12.0, abs=1.5)
    assert dy == pytest.approx(-5.0, abs=1.5)


def test_estimate_shift_returns_none_on_featureless_frames():
    flat = np.zeros((180, 320), dtype=np.uint8)
    assert estimate_shift(flat, flat.copy(), min_points=8) is None


def test_estimate_shift_rejects_implausible_shift():
    import cv2

    img = _textured()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    moved = cv2.cvtColor(_shifted(img, 40, 0), cv2.COLOR_BGR2GRAY)
    # A 40px step is real, but a 5px cap makes it "implausible" → unmeasured.
    assert estimate_shift(gray, moved, min_points=6, max_shift_px=5.0) is None


# ---------------------------------------------------------------------------
# CameraTrack lookup / transform
# ---------------------------------------------------------------------------

def test_track_interpolates_and_clamps_offsets():
    track = CameraTrack([
        CameraSample(0.0, 0.0, 0.0, True),
        CameraSample(2.0, 100.0, -20.0, True),
    ])
    assert track.offset_at(-5.0) == (0.0, 0.0)          # clamped low
    assert track.offset_at(1.0) == pytest.approx((50.0, -10.0))
    assert track.offset_at(9.0) == pytest.approx((100.0, -20.0))  # clamped high


def test_empty_track_transform_is_identity():
    t = CameraTrack().transform()
    assert t((100.0, 50.0), 3.0) == (100.0, 50.0)


def test_transform_cancels_a_constant_pan():
    track = CameraTrack([
        CameraSample(0.0, 0.0, 0.0, True),
        CameraSample(1.0, 300.0, 0.0, True),
    ])
    t = track.transform()
    # A player standing still on the pitch appears to move +300px with the pan;
    # in stabilised space both sightings land on the same point.
    assert t((100.0, 100.0), 0.0)[0] == pytest.approx(t((400.0, 100.0), 1.0)[0])


# ---------------------------------------------------------------------------
# Estimator accumulation / resets
# ---------------------------------------------------------------------------

def test_estimator_accumulates_a_steady_pan():
    img = _textured()
    est = CameraMotionEstimator(_cfg())
    est.update(img, 0.0)
    total = 0
    for i in range(1, 4):
        total += 10
        est.update(_shifted(img, total, 0), float(i) * 0.2)
    track = est.track()
    assert len(track) == 4
    assert track.samples[0].confident is False   # no reference frame yet
    assert track.confident_ratio() > 0.5
    assert track.samples[-1].dx == pytest.approx(30.0, abs=4.0)


def test_unmeasurable_step_contributes_zero_shift():
    """Recall-safety: a step we cannot measure must not invent motion."""
    img = _textured()
    flat = np.zeros_like(img)
    est = CameraMotionEstimator(_cfg())
    est.update(img, 0.0)
    est.update(flat, 0.2)          # featureless → no estimate
    s = est.samples[-1]
    assert s.confident is False
    assert (s.dx, s.dy) == (0.0, 0.0)


def test_shot_reset_does_not_fabricate_a_jump():
    img = _textured()
    est = CameraMotionEstimator(_cfg())
    est.update(img, 0.0)
    est.update(_shifted(img, 10, 0), 0.2)
    before = est.samples[-1].dx
    est.reset()                                   # cut: forget the reference
    est.update(_textured(), 0.4)                  # unrelated new scene
    after = est.samples[-1]
    assert after.confident is False
    assert after.dx == pytest.approx(before)      # carried, not jumped


def test_every_n_skips_frames_without_losing_the_reference():
    img = _textured()
    est = CameraMotionEstimator(_cfg(camera_motion_every_n=2))
    assert est.update(img, 0.0) is not None       # 1st call runs
    assert est.update(_shifted(img, 5, 0), 0.1) is None   # skipped
    assert est.update(_shifted(img, 10, 0), 0.2) is not None
    assert len(est.samples) == 2


# ---------------------------------------------------------------------------
# End-to-end: the prior a pan used to destroy
# ---------------------------------------------------------------------------

def _contact(t, x, y):
    cand = ContactCandidate(frame_index=int(t * 10), t_sec=t, processed_sec=t,
                            x=x, y=y, kinds=["kick"], strength=0.8)
    return PlayerContact(candidate=cand, player_bbox=[x - 5, y - 5, x + 5, y + 5],
                         player_dist_px=0.0, jersey_hsv=[0.0, 200.0, 200.0],
                         color_dist=5.0, is_my_team=True, n_color_samples=3)


def _seed():
    red = np.full((40, 30, 3), (0, 0, 200), np.uint8)
    return TargetSeed(kit_hsv=np.asarray([0.0, 200.0, 200.0], np.float32),
                      gallery=[red, red.copy()])


def test_pan_no_longer_reads_as_a_teleport():
    """The Issue 5 payoff: during a 400px/s pan a stationary player's second
    touch is 400px away in raw pixels (heavily down-weighted) but sits on the
    anchor once compensated."""
    cfg = PipelineV2Config()
    red = np.full((40, 30, 3), (0, 0, 200), np.uint8)
    contacts = [_contact(0.0, 100, 100), _contact(1.0, 500, 100)]
    crops = [red, red.copy()]

    raw = score_contacts(contacts, crops, _seed(), cfg)
    assert raw[1].orbital_prior < 1.0             # looks like a teleport

    track = CameraTrack([
        CameraSample(0.0, 0.0, 0.0, True),
        CameraSample(1.0, 400.0, 0.0, True),
    ])
    fixed = score_contacts(contacts, crops, _seed(), cfg, transform=track.transform())
    assert fixed[1].orbital_prior == 1.0          # stationary once compensated
    assert fixed[1].confidence >= raw[1].confidence


def test_compensation_still_penalises_genuine_movement():
    """Compensation must not flatten everything — a player who really did cross
    the pitch during the pan is still outside the orbital."""
    cfg = PipelineV2Config()
    red = np.full((40, 30, 3), (0, 0, 200), np.uint8)
    # Pan is 400px; this contact moved a further 600px on the pitch.
    contacts = [_contact(0.0, 100, 100), _contact(1.0, 1100, 100)]
    track = CameraTrack([
        CameraSample(0.0, 0.0, 0.0, True),
        CameraSample(1.0, 400.0, 0.0, True),
    ])
    scored = score_contacts(contacts, [red, red.copy()], _seed(), cfg,
                            transform=track.transform())
    assert scored[1].orbital_prior < 1.0
    assert scored[1].orbital_prior >= cfg.orbital_floor   # never a hard reject
