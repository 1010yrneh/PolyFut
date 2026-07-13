"""Tests for the review-clip player tracker (ring-follow in 'was this you?')."""

import cv2
import numpy as np

from polyfut_v2 import review_track as rt
from polyfut_v2.app_service import build_review_track_for_item


def _moving_box_clip(path, n=50, fps=25):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (640, 360))
    for i in range(n):
        f = np.zeros((360, 640, 3), np.uint8)
        # textured box (so optical flow has features) drifting right
        x = 80 + i * 6
        cv2.rectangle(f, (x, 150), (x + 40, 240), (200, 200, 200), -1)
        cv2.rectangle(f, (x + 8, 160), (x + 18, 175), (40, 40, 40), -1)
        cv2.rectangle(f, (x + 24, 195), (x + 34, 215), (60, 90, 160), -1)
        vw.write(f)
    vw.release()
    return fps, n


def test_review_track_follows_moving_box(tmp_path):
    vp = tmp_path / "clip.mp4"
    fps, n = _moving_box_clip(vp)
    # anchor at the middle frame; box centre there ≈ (80 + 25*6 + 20, 195)
    anchor_t = (n // 2) / fps
    cx = 80 + (n // 2) * 6 + 20
    crop = [cx - 20, 155, cx + 20, 235]
    tr = rt.build_review_track(str(vp), crop, anchor_t, 0.0, (n - 1) / fps)
    assert tr is not None
    pts = tr["points"]
    assert len(pts) >= n - 2
    assert tr["base_w"] == 640 and tr["base_h"] == 360
    # nx should increase over time (box moves right) and stay on the box row
    xs = [p["nx"] for p in pts]
    assert xs[0] < xs[-1]
    assert max(p["ny"] for p in pts) - min(p["ny"] for p in pts) < 0.15


def test_build_review_track_for_item_shapes_and_guards(tmp_path):
    vp = tmp_path / "clip.mp4"
    fps, n = _moving_box_clip(vp)
    item = {"crop": [200, 155, 240, 235], "t_sec": (n // 2) / fps,
            "clip_start_sec": 0.0, "clip_end_sec": (n - 1) / fps}
    tr = build_review_track_for_item(str(vp), item)
    assert tr and tr["points"]
    # missing/short crop → None (caller falls back to the fixed ring)
    assert build_review_track_for_item(str(vp), {"crop": None}) is None
    assert build_review_track_for_item(str(vp), {"crop": [1, 2, 3]}) is None


def test_review_track_unreadable_returns_none(tmp_path):
    missing = tmp_path / "nope.mp4"
    assert rt.build_review_track(str(missing), [10, 10, 30, 40], 1.0, 0.0, 2.0) is None


def test_build_enhanced_review_clip(tmp_path):
    from polyfut_v2.app_service import build_enhanced_review_clip
    vp = tmp_path / "clip.mp4"
    fps, n = _moving_box_clip(vp)
    item = {"clip_start_sec": 0.2, "clip_end_sec": (n - 1) / fps}
    out = tmp_path / "enh.mp4"
    meta = build_enhanced_review_clip(str(vp), item, str(out))
    assert meta is not None
    assert out.exists() and out.stat().st_size > 0
    assert meta["start_sec"] == 0.2 and meta["duration"] > 0
    # unreadable → None (frontend keeps the original)
    assert build_enhanced_review_clip(str(tmp_path / "no.mp4"), item, str(out)) is None
