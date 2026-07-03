"""Tests for Stage 4 frame router."""

from pathlib import Path

import numpy as np

from polyfut_video.pipeline.decode import iter_frames
from polyfut_video.pipeline.frame_router import route_frames
from polyfut_video.tests.conftest import make_synthetic_clip


def test_route_frames_mix(tmp_path: Path):
    p = tmp_path / "r.mp4"
    make_synthetic_clip(p, duration_sec=2.0, include_replay=False)
    routes = [r[2] for r in route_frames(iter_frames(str(p), target_width=320), motion_threshold=2.0)]
    assert len(routes) > 0
    assert "full" in routes or "cheap" in routes
