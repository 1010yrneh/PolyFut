"""Tests for Stage 3 deadtime filter."""

from pathlib import Path

import pytest

from polyfut_video.pipeline.deadtime_filter import filter_deadtime
from polyfut_video.tests.conftest import make_synthetic_clip


def test_filter_deadtime_keeps_live_play(tmp_path: Path):
    p = tmp_path / "short.mp4"
    make_synthetic_clip(p, duration_sec=2.0, include_replay=False)
    shots = [{
        "start_frame": 0,
        "end_frame": 49,
        "start_sec": 0.0,
        "end_sec": 2.0,
        "label": "main_camera",
    }]
    live, removed = filter_deadtime(
        shots, None, motion_threshold=0.5, min_duration_sec=60.0,
        video_path=str(p), target_width=320,
    )
    assert len(live) == 1
    assert live[0]["label"] == "live_play"


def test_filter_deadtime_discards_labeled(tmp_path: Path):
    shots = [{
        "start_frame": 0, "end_frame": 10,
        "start_sec": 0.0, "end_sec": 5.0, "label": "discard",
    }]
    live, removed = filter_deadtime(shots, None, motion_threshold=1.0)
    assert len(live) == 0
    assert len(removed) == 1
    assert removed[0]["reason"] == "discard"
