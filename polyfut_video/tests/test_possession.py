"""Tests for Stage 8 possession."""

from polyfut_video.pipeline.possession import (
    PossessionConfig,
    _scaled_contact_threshold,
    compute_possession,
)


def test_scaled_threshold_wider_for_tiny_ball():
    cfg = PossessionConfig()
    near = _scaled_contact_threshold(100.0, 20.0, cfg)
    far = _scaled_contact_threshold(50.0, 8.0, cfg)
    assert far >= near * 0.8
    assert far >= cfg.min_thresh_px


def test_hysteresis_holds_through_miss():
    frames = []
    for i in range(12):
        dets = []
        if i in (0, 1, 2, 8, 9):
            dets = [
                {"class": "player", "team_id": 0, "bbox": [100, 100, 130, 180]},
                {"class": "ball", "bbox": [115, 175, 120, 180], "conf": 0.7},
            ]
        frames.append({
            "frame_index": i,
            "timestamp_sec": i / 7.5,
            "detections": dets,
        })
    cfg = PossessionConfig(on_frames=1, off_frames=4, window_size_sec=0.2)
    out = compute_possession(frames, window_size_sec=0.2, fps=7.5, cfg=cfg)
    team_a = [r for r in out if r["possession"] == "team_a"]
    assert len(team_a) >= 6


def test_contested_when_mixed():
    frames = []
    for i in range(20):
        team = "team_a" if i % 2 == 0 else "team_b"
        tid = 0 if team == "team_a" else 1
        frames.append({
            "frame_index": i,
            "timestamp_sec": i / 25.0,
            "detections": [
                {"class": "player", "team_id": tid, "bbox": [100, 100, 130, 180]},
                {"class": "ball", "bbox": [115, 175, 125, 185], "conf": 0.8},
            ],
        })
    out = compute_possession(frames, window_size_sec=0.5, fps=25.0, contested_margin=0.2)
    contested = sum(1 for r in out if r["possession"] == "contested")
    assert contested >= 1
