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
    # player_bbox = the real box (crop here just plays the ball-zoom fallback
    # anchor); this is the box the tracker actually follows.
    tr = rt.build_review_track(str(vp), crop, anchor_t, 0.0, (n - 1) / fps,
                                player_bbox=crop)
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
    cx = 80 + (n // 2) * 6 + 20
    item = {
        "crop": [cx - 20, 155, cx + 20, 235],
        "player_bbox": [cx - 8, 170, cx + 8, 210],
        "t_sec": (n // 2) / fps,
        "clip_start_sec": 0.0, "clip_end_sec": (n - 1) / fps,
    }
    tr = build_review_track_for_item(str(vp), item)
    assert tr and tr["points"]
    # missing/short crop → None (caller falls back to the fixed ring)
    assert build_review_track_for_item(str(vp), {"crop": None}) is None
    assert build_review_track_for_item(str(vp), {"crop": [1, 2, 3]}) is None


def test_review_track_without_player_bbox_hides_box(tmp_path):
    """No player_bbox and no detector hit → empty track (UI hides the box).
    Never invent a body on the ball."""
    vp = tmp_path / "clip.mp4"
    fps, n = _moving_box_clip(vp)
    anchor_t = (n // 2) / fps
    cx = 80 + (n // 2) * 6 + 20
    crop = [cx - 20, 155, cx + 20, 235]
    tr = rt.build_review_track(str(vp), crop, anchor_t, 0.0, (n - 1) / fps)
    assert tr is not None
    assert tr["points"] == []
    assert tr.get("visible") is False


def test_build_review_track_for_item_prefers_player_bbox_over_crop(tmp_path):
    """When the montage item carries a real player_bbox, use its size — not
    the much larger ball-centered zoom crop."""
    vp = tmp_path / "clip.mp4"
    fps, n = _moving_box_clip(vp)
    anchor_t = (n // 2) / fps
    cx = 80 + (n // 2) * 6 + 20
    crop = [cx - 20, 155, cx + 20, 235]        # 40x80 zoom window
    player_bbox = [cx - 8, 170, cx + 8, 210]   # 16x40 real player box
    item = {"crop": crop, "player_bbox": player_bbox, "t_sec": anchor_t,
            "clip_start_sec": 0.0, "clip_end_sec": (n - 1) / fps}
    tr = build_review_track_for_item(str(vp), item)
    assert tr is not None
    p0 = tr["points"][0]
    assert abs(p0["nw"] * tr["base_w"] - 16) < 1.0
    assert abs(p0["nh"] * tr["base_h"] - 40) < 1.0


class _FakeDetector:
    """Returns a fixed player box regardless of frame (or nothing if empty)."""
    def __init__(self, bbox=None):
        self.bbox = bbox
    def detect(self, frame, near=None):
        from polyfut_v2.pipeline.player_detector import PlayerDetection
        return [PlayerDetection(list(self.bbox), 0.9)] if self.bbox else []


def test_review_track_fallback_anchors_on_nearest_detected_player(tmp_path):
    """No player_bbox, but a detector finds someone inside the contact gate:
    anchor on them rather than an empty-grass body box on the ball."""
    vp = tmp_path / "clip.mp4"
    fps, n = _moving_box_clip(vp)
    anchor_t = (n // 2) / fps
    # ball crop sits ~50px left of the textured box — still inside the 80px gate.
    crop = [170, 175, 210, 215]                 # centre ≈ (190, 195)
    player_box = [240, 160, 260, 220]           # 20x60 person-shaped box, on the texture
    det = _FakeDetector(player_box)
    tr = rt.build_review_track(str(vp), crop, anchor_t, 0.0, (n - 1) / fps,
                                player_bbox=None, detector=det)
    assert tr is not None
    p0 = tr["points"][0]
    assert abs(p0["nh"] * tr["base_h"] - 60) < 5
    assert p0["nx"] * tr["base_w"] > 150


def test_review_track_fallback_ignores_players_outside_contact_gate(tmp_path):
    """A body well beyond contact_max_player_dist must NOT become the review
    box — that was the 'boxed the wrong person' bug (old 220px reach)."""
    vp = tmp_path / "clip.mp4"
    fps, n = _moving_box_clip(vp)
    anchor_t = (n // 2) / fps
    crop = [100, 175, 140, 215]                 # ball centre ≈ (120, 195)
    far_player = [300, 160, 320, 220]           # ~180px away → outside the 80px gate
    tr = rt.build_review_track(str(vp), crop, anchor_t, 0.0, (n - 1) / fps,
                                player_bbox=None, detector=_FakeDetector(far_player))
    assert tr is not None
    assert tr["points"] == []
    assert tr.get("visible") is False


def test_review_track_uses_frame_index_over_t_sec(tmp_path):
    """Anchor on the contact's source frame_index when provided."""
    vp = tmp_path / "clip.mp4"
    fps, n = _moving_box_clip(vp)
    # Put the box at frame 10; claim t_sec of a different frame so a t_sec-only
    # anchor would miss. frame_index must win.
    fi = 10
    cx = 80 + fi * 6 + 20
    player_bbox = [cx - 8, 170, cx + 8, 210]
    crop = [cx - 20, 155, cx + 20, 235]
    wrong_t = (n // 2) / fps
    tr = rt.build_review_track(
        str(vp), crop, wrong_t, 0.0, (n - 1) / fps,
        player_bbox=player_bbox, frame_index=fi,
    )
    assert tr is not None
    # First point should be at the frame_index timestamp, not wrong_t.
    assert abs(tr["points"][0]["t_sec"] - fi / fps) < 1.0 / fps + 0.05
    assert abs(tr["points"][0]["nx"] * 640 - cx) < 3


def test_review_track_empty_when_detector_finds_nobody(tmp_path):
    """Detector supplied but finds no player nearby → hide the box."""
    vp = tmp_path / "clip.mp4"
    fps, n = _moving_box_clip(vp)
    anchor_t = (n // 2) / fps
    crop = [100, 155, 180, 235]
    tr = rt.build_review_track(str(vp), crop, anchor_t, 0.0, (n - 1) / fps,
                                player_bbox=None, detector=_FakeDetector(None))
    assert tr is not None
    assert tr["points"] == []
    assert tr.get("visible") is False


def test_review_track_fallback_ignores_ball_shaped_detection(tmp_path):
    """A small, square 'player' box (the ball misclassified) must NOT be chosen
    as the fallback anchor — hide rather than paint a fake body."""
    vp = tmp_path / "clip.mp4"
    fps, n = _moving_box_clip(vp)
    anchor_t = (n // 2) / fps
    crop = [170, 175, 210, 215]
    ball_mimic = [188, 190, 197, 201]           # ~9x11, aspect ~1.2 → ball-shaped
    tr = rt.build_review_track(str(vp), crop, anchor_t, 0.0, (n - 1) / fps,
                                player_bbox=None, detector=_FakeDetector(ball_mimic))
    assert tr is not None
    assert tr["points"] == []


def test_review_track_unreadable_returns_none(tmp_path):
    missing = tmp_path / "nope.mp4"
    assert rt.build_review_track(str(missing), [10, 10, 30, 40], 1.0, 0.0, 2.0) is None
