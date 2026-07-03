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


def test_benchmark_decode(sample_clip: Path):
    bench = benchmark_decode(str(sample_clip), target_width=320, max_frames=50)
    assert bench["frames"] > 0
    assert bench["optimized_sec"] >= 0
