"""A hotspot should end when the ball leaves, not a fixed pad after the last
touch that happened to be detected.

Measured on job ``fdc541cf493e``: the last confirmed touch was at 288.8s and the
final hotspot ended at 290.8s — exactly ``hotspot_pad_after_sec`` later — while
the player still had the ball and went on to make a pass. With the ball detected
on only 40% of frames, the last *seen* touch is routinely not the last touch.
"""

from __future__ import annotations

from polyfut_v2.config import PipelineV2Config
from polyfut_v2.pipeline.hotspots import assemble_hotspots, ball_departure_sec

CFG = PipelineV2Config()
OFF = PipelineV2Config(hotspot_possession_enabled=False)


def _ball_holds_then_leaves(last_touch=10.0, hold_sec=3.0, hz=10.0):
    """Ball sits at the player's feet, then is passed sharply away."""
    track = []
    t = last_touch
    while t < last_touch + hold_sec:
        track.append((t, 100.0, 100.0))          # at the feet, barely moving
        t += 1.0 / hz
    while t < last_touch + hold_sec + 2.0:
        track.append((t, 100.0 + (t - last_touch - hold_sec) * 400.0, 100.0))
        t += 1.0 / hz
    return track


# --------------------------------------------------------------- the primitive
def test_departure_is_found_when_the_ball_is_played_away():
    track = _ball_holds_then_leaves()
    gone = ball_departure_sec(track, 10.0, leave_px=120.0, max_extend_sec=6.0)
    assert gone is not None
    assert 13.0 < gone < 13.6           # just after the pass starts at 13.0s


def test_no_track_means_no_extension():
    assert ball_departure_sec([], 10.0, leave_px=120.0, max_extend_sec=6.0) is None


def test_a_ball_that_never_leaves_does_not_extend_forever():
    """A ball never seen to leave is a tracking failure, not a long dribble."""
    still = [(10.0 + i * 0.1, 100.0, 100.0) for i in range(200)]
    assert ball_departure_sec(still, 10.0, leave_px=120.0,
                              max_extend_sec=6.0) is None


def test_a_blind_stretch_does_not_extend_on_a_guess():
    """Samples exist, but none inside the window — must not invent a departure."""
    elsewhere = [(50.0, 100.0, 100.0), (51.0, 900.0, 900.0)]
    assert ball_departure_sec(elsewhere, 10.0, leave_px=120.0,
                              max_extend_sec=6.0) is None


# ------------------------------------------------------------- the assembly
def test_the_window_now_covers_the_final_pass():
    track = _ball_holds_then_leaves()
    before = assemble_hotspots([10.0], OFF, duration_sec=60.0)[0]
    after = assemble_hotspots([10.0], CFG, duration_sec=60.0,
                              ball_track=track)[0]
    assert before.end_sec == 12.0                # 10.0 + pad_after, cuts the pass
    assert after.end_sec > 15.0                  # departure (~13.1) + pad_after
    assert after.start_sec == before.start_sec   # the start must not move


def test_without_a_ball_track_nothing_changes():
    a = assemble_hotspots([10.0, 12.0], CFG, duration_sec=60.0)
    b = assemble_hotspots([10.0, 12.0], OFF, duration_sec=60.0)
    assert [(h.start_sec, h.end_sec) for h in a] == \
           [(h.start_sec, h.end_sec) for h in b]


def test_extension_is_still_clamped_to_the_video():
    track = _ball_holds_then_leaves(last_touch=57.0)
    hs = assemble_hotspots([57.0], CFG, duration_sec=60.0, ball_track=track)
    assert hs[0].end_sec == 60.0


def test_the_extension_can_be_switched_off():
    track = _ball_holds_then_leaves()
    hs = assemble_hotspots([10.0], OFF, duration_sec=60.0, ball_track=track)
    assert hs[0].end_sec == 12.0


def test_a_quick_pass_barely_extends():
    """The ball leaving immediately should look like today's behaviour."""
    track = [(10.0 + i * 0.1, 100.0 + i * 100.0, 100.0) for i in range(40)]
    hs = assemble_hotspots([10.0], CFG, duration_sec=60.0, ball_track=track)
    assert 12.0 <= hs[0].end_sec <= 13.0
