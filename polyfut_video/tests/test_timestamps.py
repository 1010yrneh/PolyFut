"""Tests for Stage 9 timestamps."""

from polyfut_video.pipeline.timestamps import (
    generate_timestamps,
    processed_to_match_clock,
    timeline_to_clip_segments,
)


def test_match_clock_remap_with_gap():
    removed = [{"start_sec": 100.0, "end_sec": 160.0, "reason": "deadtime"}]
    # 50s processed maps to 50s match (before removed gap)
    assert abs(processed_to_match_clock(50.0, removed) - 50.0) < 0.01
    # End of first live block
    assert abs(processed_to_match_clock(99.9, removed) - 99.9) < 0.02
    # Just after gap: second live block starts at match 160
    assert abs(processed_to_match_clock(100.1, removed) - 160.1) < 0.02
    assert abs(processed_to_match_clock(110.0, removed) - 170.0) < 0.02


def test_generate_timestamps():
    frames = [
        {"processed_sec": 0.0, "timestamp_sec": 0.0, "possession": "team_a"},
        {"processed_sec": 1.0, "timestamp_sec": 1.0, "possession": "team_a"},
        {"processed_sec": 2.0, "timestamp_sec": 2.0, "possession": "team_b"},
    ]
    ivs = generate_timestamps(frames, [])
    assert len(ivs) >= 1
    assert "start" in ivs[0]


def test_timeline_to_clip_segments():
    timeline = [
        {"team": "team_a", "start_sec": 10.0, "end_sec": 20.0},
        {"team": "team_b", "start_sec": 30.0, "end_sec": 40.0},
        {"team": "team_a", "start_sec": 50.0, "end_sec": 55.0},
    ]
    segs = timeline_to_clip_segments(
        timeline,
        my_team="team_a",
        pad_before=3.0,
        pad_after=3.0,
        gap_merge=5.0,
        min_zone_sec=3.0,
    )
    assert len(segs) == 2
    assert all(s.get("type") == "hotspot" for s in segs)
    assert all("start" in s and "end" in s for s in segs)
    assert all("action_triggers" in s for s in segs)
    assert segs[0]["start"] == 7.0
    assert segs[0]["end"] == 23.0
    assert segs[0]["core_start"] == 10.0
    assert segs[0]["core_end"] == 20.0


def test_hotspot_merge_within_five_seconds():
    timeline = [
        {"team": "team_a", "start_sec": 10.0, "end_sec": 12.0},
        {"team": "team_a", "start_sec": 16.0, "end_sec": 18.0},
    ]
    segs = timeline_to_clip_segments(
        timeline,
        my_team="team_a",
        pad_before=3.0,
        pad_after=3.0,
        gap_merge=5.0,
        min_zone_sec=3.0,
    )
    assert len(segs) == 1
    assert segs[0]["core_start"] == 10.0
    assert segs[0]["core_end"] == 18.0
    assert segs[0]["start"] == 7.0
    assert segs[0]["end"] == 21.0


def test_hotspot_min_zone_width():
    timeline = [{"team": "team_a", "start_sec": 100.0, "end_sec": 100.5}]
    segs = timeline_to_clip_segments(
        timeline,
        my_team="team_a",
        pad_before=3.0,
        pad_after=3.0,
        min_zone_sec=3.0,
    )
    assert len(segs) == 1
    assert segs[0]["end"] - segs[0]["start"] >= 3.0
    assert segs[0]["start"] == 97.0
    assert segs[0]["end"] == 103.5
