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


def _cfg(**kw):
    base = dict(max_hold_frames=8, suspect_jump_px=120.0, suspect_jump_conf=0.30)
    base.update(kw)
    return BallSmoothConfig(**base)


def test_low_confidence_teleport_is_rejected_and_last_position_held():
    """The phantom-touch bug: a low-confidence detection leaping far from a
    cleanly-tracked ball is a false positive (white sock / line marking), not a
    fast ball. Following it fabricates huge velocity swings that Stage 4 reads
    as a kick, creating a touch where the ball never was."""
    s = BallSmoother(_cfg())
    good = [100.0, 100.0, 110.0, 110.0]
    out, _, held = s.update(good, 0.67)
    assert out == good and held is False
    # 160px away, confidence 0.11 -> reject, keep holding the trusted position.
    out2, _, held2 = s.update([260.0, 100.0, 270.0, 110.0], 0.11)
    assert out2 == good and held2 is True


def test_confident_long_jump_needs_one_confirming_frame():
    """A genuine clearance/shot moves the ball a long way and must still be
    followed — but confidence alone is weak evidence (a confident false positive
    looks identical on one frame). The jump is parked for exactly one frame and
    committed as soon as a second detection agrees with it."""
    s = BallSmoother(_cfg())
    good = [100.0, 100.0, 110.0, 110.0]
    s.update(good, 0.67)
    far = [300.0, 100.0, 310.0, 110.0]          # 200px away, confident
    out1, _, held1 = s.update(far, 0.80)
    assert out1 == good and held1 is True       # pending, not yet committed
    out2, _, held2 = s.update(far, 0.80)        # same place again → confirmed
    assert out2 == far and held2 is False


def test_fast_ball_confirms_by_continuing_in_the_same_direction():
    """A ball travelling fast never lands twice in the same spot, so proximity
    alone would never confirm it and tracking would collapse during clearances.
    Consecutive far detections marching in one direction confirm each other."""
    s = BallSmoother(_cfg())
    s.update([100.0, 100.0, 110.0, 110.0], 0.7)
    # 150px per sample, straight line: pending, then confirmed by continuation.
    out1, _, held1 = s.update([250.0, 100.0, 260.0, 110.0], 0.7)
    assert held1 is True                         # first far sample parked
    far2 = [400.0, 100.0, 410.0, 110.0]
    out2, _, held2 = s.update(far2, 0.7)
    assert out2 == far2 and held2 is False       # same heading, same speed


def test_random_flicker_never_confirms_and_is_never_followed():
    """The phantom-touch pattern: far detections that jump to a different place
    every frame (socks, line markings). None of them agree, so none commit."""
    s = BallSmoother(_cfg(max_hold_frames=8))
    good = [100.0, 100.0, 110.0, 110.0]
    s.update(good, 0.9)
    for box in (
        [300.0, 100.0, 310.0, 110.0],
        [100.0, 300.0, 110.0, 310.0],
        [320.0, 320.0, 330.0, 330.0],
    ):
        out, _, held = s.update(box, 0.9)
        assert out == good and held is True


def test_small_low_confidence_move_is_accepted():
    """Low confidence alone must not reject — the ball is often detected weakly
    while being tracked correctly. Only far AND weak is suspicious."""
    s = BallSmoother(_cfg())
    s.update([100.0, 100.0, 110.0, 110.0], 0.67)
    near = [112.0, 100.0, 122.0, 110.0]         # 12px, low conf
    out, _, held = s.update(near, 0.10)
    assert out == near and held is False


def test_gate_applies_immediately_after_a_good_detection():
    """Regression: the gate used to be skipped whenever the tracker wasn't
    already holding (``_held == 0``) — i.e. right after a good detection, which
    is exactly when a teleport does the most damage."""
    s = BallSmoother(_cfg())
    good = [100.0, 100.0, 110.0, 110.0]
    s.update(good, 0.9)                          # _held == 0 here
    out, _, held = s.update([400.0, 300.0, 410.0, 310.0], 0.12)
    assert out == good and held is True          # rejected, not followed


def test_persistent_teleport_is_confirmed_rather_than_waiting_out_the_budget():
    """If the 'teleport' keeps showing up in the same place, the ball genuinely
    moved (or was lost and found elsewhere). Corroboration is stronger evidence
    than confidence, so we commit on the second sighting instead of stalling for
    the whole hold budget — even at low confidence."""
    s = BallSmoother(_cfg(max_hold_frames=8))
    good = [100.0, 100.0, 110.0, 110.0]
    s.update(good, 0.9)
    far = [400.0, 300.0, 410.0, 310.0]
    out1, _, held1 = s.update(far, 0.12)
    assert out1 == good and held1 is True
    out2, _, held2 = s.update(far, 0.12)
    assert out2 == far and held2 is False


def test_hold_budget_expires_when_nothing_ever_confirms():
    """Unconfirmable far detections must not pin the tracker to a stale position
    forever — once the hold budget runs out we let go and re-acquire cold."""
    s = BallSmoother(_cfg(max_hold_frames=2))
    s.update([100.0, 100.0, 110.0, 110.0], 0.9)
    spots = [
        [400.0, 300.0, 410.0, 310.0],
        [100.0, 320.0, 110.0, 330.0],
        [330.0, 100.0, 340.0, 110.0],
    ]
    s.update(spots[0], 0.12)   # held 1
    s.update(spots[1], 0.12)   # held 2
    out, _, held = s.update(spots[2], 0.12)   # budget exhausted -> release
    assert out is None and held is False
    out2, _, held2 = s.update(spots[2], 0.12)  # no anchor left -> re-acquire
    assert out2 == spots[2] and held2 is False


def test_legacy_single_frame_gate_still_available():
    """``require_jump_confirmation=False`` restores the old confidence-only gate
    for callers that need the previous behaviour."""
    s = BallSmoother(_cfg(require_jump_confirmation=False))
    s.update([100.0, 100.0, 110.0, 110.0], 0.67)
    far = [300.0, 100.0, 310.0, 110.0]
    out, _, held = s.update(far, 0.80)
    assert out == far and held is False


def test_ball_reset_after_hold_expires():
    s = BallSmoother(BallSmoothConfig(max_hold_frames=2))
    bbox = [50.0, 50.0, 60.0, 60.0]
    s.update(bbox, 0.8)
    s.update(None, 0.0)
    s.update(None, 0.0)
    out, _, held = s.update(None, 0.0)
    assert out is None
    assert held is False
