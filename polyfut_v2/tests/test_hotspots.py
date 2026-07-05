"""Tests for Stage 9 hotspot assembly."""

from polyfut_v2.config import PipelineV2Config
from polyfut_v2.pipeline.hotspots import assemble_hotspots

CFG = PipelineV2Config()  # pad ±2, gap_merge 5, min_zone 3


def test_empty():
    assert assemble_hotspots([], CFG) == []


def test_single_touch_window_and_min_zone():
    hs = assemble_hotspots([10.0], CFG)
    assert len(hs) == 1
    assert hs[0].start_sec == 8.0 and hs[0].end_sec == 12.0  # ±2, already ≥ min_zone
    assert hs[0].contact_times == [10.0]


def test_close_touches_merge():
    hs = assemble_hotspots([10.0, 12.0], CFG)
    assert len(hs) == 1
    assert hs[0].start_sec == 8.0 and hs[0].end_sec == 14.0
    assert hs[0].contact_times == [10.0, 12.0]


def test_far_touches_split():
    hs = assemble_hotspots([10.0, 30.0], CFG)
    assert len(hs) == 2
    assert [round(h.start_sec) for h in hs] == [8, 28]


def test_min_zone_extends_short_window():
    cfg = PipelineV2Config(hotspot_pad_before_sec=0.2, hotspot_pad_after_sec=0.2,
                           hotspot_min_zone_sec=3.0)
    hs = assemble_hotspots([10.0], cfg)
    assert abs(hs[0].duration_sec - 3.0) < 1e-6
    assert abs((hs[0].start_sec + hs[0].end_sec) / 2 - 10.0) < 1e-6  # centered


def test_clamped_to_video_bounds():
    hs = assemble_hotspots([1.0, 99.5], CFG, duration_sec=100.0)
    assert hs[0].start_sec == 0.0        # clamped at start
    assert hs[-1].end_sec == 100.0       # clamped at end
