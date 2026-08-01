"""Tests for Stage 1 decode."""

from pathlib import Path

import pytest

from polyfut_video.pipeline.decode import benchmark_decode, iter_frames, probe_video
from polyfut_video.tests.conftest import make_synthetic_clip


@pytest.fixture
def sample_clip(tmp_path: Path) -> Path:
    p = tmp_path / "live_play.mp4"
    make_synthetic_clip(p, duration_sec=3.0, include_replay=True)
    return p


def test_iter_frames_yields_source_frame_index(sample_clip: Path):
    """First tuple element must be the source video frame index, not a yield counter."""
    info = probe_video(str(sample_clip))
    sample_n = 3
    frames = list(iter_frames(str(sample_clip), target_width=320, sample_every_n=sample_n))
    assert len(frames) >= 2
    assert frames[0][0] == 0
    assert frames[1][0] == sample_n
    assert frames[-1][0] < info["frame_count"]


def test_iter_frames_timestamps_match_index(sample_clip: Path):
    info = probe_video(str(sample_clip))
    fps = info["fps"]
    for frame_idx, t_sec, _ in iter_frames(str(sample_clip), target_width=320, sample_every_n=2):
        expected = frame_idx / fps
        assert abs(t_sec - expected) < 0.02


@pytest.fixture
def long_clip(tmp_path: Path) -> Path:
    p = tmp_path / "long_play.mp4"
    make_synthetic_clip(p, duration_sec=10.0, include_replay=False)
    return p


def test_iter_frames_bounded_by_start_and_end(long_clip: Path):
    """The playing-time window bounds decode. Indices and timestamps must stay
    ABSOLUTE — the window changes what is read, never the clock — so shots and
    touches remain comparable with an unbounded run."""
    info = probe_video(str(long_clip))
    fps = info["fps"]
    frames = list(iter_frames(str(long_clip), target_width=320, sample_every_n=5,
                              start_sec=4.0, end_sec=6.0))
    assert frames, "bounded pass yielded nothing"
    times = [t for _fi, t, _f in frames]
    assert min(times) >= 4.0 - 1.0 / fps
    assert max(times) <= 6.0 + 1.0 / fps
    # Absolute, not re-based to the window.
    assert all(abs(t - fi / fps) < 0.02 for fi, t, _f in frames)
    assert frames[0][0] >= int(4.0 * fps) - 1


def test_iter_frames_bounded_matches_the_unbounded_pass_in_that_span(long_clip: Path):
    """Sampling phase is absolute, so a bounded pass yields exactly the frames an
    unbounded pass would have yielded in the same span — no drift in which
    frames get analysed just because a window was set."""
    kw = dict(target_width=320, sample_every_n=5)
    full = [fi for fi, t, _ in iter_frames(str(long_clip), **kw) if 4.0 <= t <= 6.0]
    windowed = [fi for fi, _t, _f in
                iter_frames(str(long_clip), start_sec=4.0, end_sec=6.0, **kw)]
    assert windowed == full


def test_iter_frames_unbounded_by_default(long_clip: Path):
    info = probe_video(str(long_clip))
    frames = list(iter_frames(str(long_clip), target_width=320, sample_every_n=10))
    assert frames[0][0] == 0
    assert frames[-1][0] > info["frame_count"] * 0.8


def test_iter_frames_start_past_end_of_video_yields_nothing(long_clip: Path):
    assert list(iter_frames(str(long_clip), target_width=320, start_sec=999.0)) == []


def test_benchmark_decode(sample_clip: Path):
    bench = benchmark_decode(str(sample_clip), target_width=320, max_frames=50)
    assert bench["frames"] > 0
    assert bench["optimized_sec"] >= 0
