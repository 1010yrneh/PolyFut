"""Tests for Stage 2 shot filter."""

from pathlib import Path

import pytest

from polyfut_video.config import PipelineConfig
from polyfut_video.pipeline.decode import iter_frames, probe_video
from polyfut_video.pipeline.shot_filter import segment_and_classify_shots
from polyfut_video.tests.conftest import make_synthetic_clip


@pytest.fixture
def clip_with_replay(tmp_path: Path) -> Path:
    p = tmp_path / "replay.mp4"
    make_synthetic_clip(p, duration_sec=4.0, include_replay=True)
    return p


def test_segment_and_classify(clip_with_replay: Path):
    cfg = PipelineConfig()
    it = iter_frames(str(clip_with_replay), target_width=320)
    shots = segment_and_classify_shots(it, cfg)
    assert len(shots) >= 1
    labels = {s["label"] for s in shots}
    assert "main_camera" in labels or "discard" in labels


def test_shot_sec_span_covers_clip(clip_with_replay: Path):
    info = probe_video(str(clip_with_replay))
    cfg = PipelineConfig()
    shots = segment_and_classify_shots(
        iter_frames(str(clip_with_replay), target_width=320), cfg
    )
    assert shots
    assert shots[-1]["end_sec"] >= info["duration_sec"] * 0.85
    assert shots[0]["start_sec"] <= 0.5
