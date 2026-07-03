"""Tests for Stage 6 tracking."""

from polyfut_video.pipeline.tracking import track_shot


def test_track_shot_assigns_ids():
    dets_per_frame = [
        [{"bbox": [10, 10, 40, 80], "class": "player", "conf": 0.9}],
        [{"bbox": [12, 12, 42, 82], "class": "player", "conf": 0.9}],
        [{"bbox": [14, 14, 44, 84], "class": "player", "conf": 0.9}],
    ]
    tracked, _ = track_shot(dets_per_frame)
    assert len(tracked) == 3
    ids = [tracked[i][0].get("track_id") for i in range(3)]
    assert all(tid is not None for tid in ids)


def test_track_resets_per_shot_call():
    shot_a, _ = track_shot([[{"bbox": [0, 0, 10, 10], "class": "player", "conf": 0.9}]])
    shot_b, _ = track_shot([[{"bbox": [0, 0, 10, 10], "class": "player", "conf": 0.9}]])
    assert shot_a[0][0]["track_id"] == shot_b[0][0]["track_id"]
