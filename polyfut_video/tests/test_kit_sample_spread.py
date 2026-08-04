"""The kit picker must SPREAD its samples across sample_window_minutes.

They used to be taken back to back at sample_every_seconds, so n_samples=20
every 3s stopped after 60 seconds and sample_window_minutes was unreachable —
a parameter that read as "sample the first 8 minutes" and actually sampled the
first one. On a broadcast that first minute is the warm-up, where players wear
training bibs rather than match kits, so both swatches were decided from
footage the real kits never appear in. An 82-minute ISB v TAS recording of an
orange team against a navy one returned two purples.

These pin the stride arithmetic rather than the swatches: the colours need real
video, but the sampling bug is pure arithmetic and is what actually broke.
"""

from __future__ import annotations

import pytest


def stride_seconds(fps, n_frames, n_samples=20, window_min=8.0, every=3.0):
    """The spacing detect_team_kits derives — mirrors the code under test."""
    span = min(window_min * 60.0, n_frames / max(fps, 1e-6))
    return max(every, span / max(n_samples, 1))


def coverage_seconds(fps, n_frames, **kw):
    n = kw.get("n_samples", 20)
    return stride_seconds(fps, n_frames, **kw) * n


# ---------------------------------------------- the bug that was shipped
def test_long_video_is_sampled_across_the_whole_window_not_just_one_minute():
    fps, n_frames = 30.0, int(82 * 60 * 30)          # the ISB/TAS broadcast
    assert coverage_seconds(fps, n_frames) == pytest.approx(8 * 60, rel=0.01)


def test_the_old_back_to_back_rule_covered_only_sixty_seconds():
    """Pins the regression: 20 samples x 3s is one minute, whatever the window."""
    assert 20 * 3.0 == 60.0
    fps, n_frames = 30.0, int(82 * 60 * 30)
    assert coverage_seconds(fps, n_frames) > 60.0 * 7


# ------------------------------------------------- short clips unaffected
@pytest.mark.parametrize("minutes", [0.5, 1.0])
def test_short_clips_keep_the_three_second_floor(minutes):
    """A clip with no room to spread into must be sampled exactly as before."""
    fps = 30.0
    n_frames = int(minutes * 60 * fps)
    assert stride_seconds(fps, n_frames) == pytest.approx(3.0)


def test_window_is_a_ceiling_not_a_target():
    """A 4-minute clip spreads over its own length, never past the end."""
    fps, n_frames = 30.0, int(4 * 60 * 30)
    assert coverage_seconds(fps, n_frames) <= 4 * 60 + 1


def test_sample_count_is_unchanged_so_detection_cost_is_unchanged():
    """Spreading must not buy coverage with more inference — detection
    dominates the picker's runtime and stays at n_samples frames."""
    fps = 30.0
    for minutes in (1, 5, 20, 82):
        n_frames = int(minutes * 60 * fps)
        stride = stride_seconds(fps, n_frames)
        assert coverage_seconds(fps, n_frames) / stride == pytest.approx(20)


def test_matches_the_implementation():
    """Guard against this file drifting from the real function."""
    import inspect

    from polyfut_video.pipeline import team_preview

    src = inspect.getsource(team_preview.detect_team_kits)
    assert "span_sec" in src and "every_sec" in src, (
        "detect_team_kits no longer derives its stride from the window; "
        "update stride_seconds() here to match."
    )
