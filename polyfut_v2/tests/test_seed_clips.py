"""Tests for the enhanced seed-clip + tracked-node feature (Stage 0 UX)."""

import cv2
import numpy as np
import pytest

from polyfut_v2 import seed_clips as sc
from polyfut_v2.app_service import taps_from_tracklet


def test_enhance_frame_upscales_2x():
    src = np.zeros((40, 60, 3), dtype=np.uint8)
    out = sc.enhance_frame(src)
    assert out.shape[:2] == (80, 120)  # 2x


def test_default_moments_reroll_shifts_and_bounds():
    m0 = sc.default_moments(100.0, 0)
    m1 = sc.default_moments(100.0, 1)
    assert len(m0) == 4 and len(m1) == 4
    assert m0 != m1                       # reroll changes the set
    assert all(0 <= t <= 100 for t in m0 + m1)
    assert sc.default_moments(0.0) == [0.0]   # degenerate duration


def test_track_follows_a_moving_player():
    # One detection per sampled frame, drifting right → a single tracklet.
    w, h = 1000, 500
    dets_per_frame = []
    for fi in range(0, 30, 3):
        cx = 100 + fi * 5
        bbox = [cx - 20, 200, cx + 20, 300]
        dets_per_frame.append((fi, [(bbox, float(cx), 250.0, None)]))
    tracks = sc._track(dets_per_frame, w, h, fps=25.0)
    assert len(tracks) == 1
    assert "kit_hsv" in tracks[0]      # colour tag present (None here)
    pts = tracks[0]["points"]
    assert len(pts) == 10
    assert pts[0]["nx"] < pts[-1]["nx"]   # node followed the player rightward
    assert pts[0]["t"] < pts[-1]["t"]


def test_track_two_players_stay_separate():
    w, h = 1000, 500
    dets_per_frame = []
    for fi in range(0, 15, 3):
        a = [100, 200, 140, 300]
        b = [800, 200, 840, 300]
        dets_per_frame.append((fi, [(a, 120.0, 250.0, None), (b, 820.0, 250.0, None)]))
    tracks = sc._track(dets_per_frame, w, h, fps=25.0)
    assert len(tracks) == 2


def test_track_colour_lock_no_cross_team_swap():
    """A yellow and a black player cross paths. With colour-locked tracking each
    tag must stay on its own colour — never jump teams at the crossover frame."""
    import numpy as np
    w, h = 1000, 500
    yellow = np.array([27, 200, 200], np.float32)
    black = np.array([0, 30, 40], np.float32)
    dets_per_frame = []
    for k, fi in enumerate(range(0, 21, 3)):        # 8 sampled frames
        # yellow drifts right, black drifts left; they swap sides at the middle,
        # passing within a few px of each other (would confuse position-only).
        yx = 200 + k * 80
        bx = 760 - k * 80
        y = ([yx - 20, 200, yx + 20, 300], float(yx), 250.0, yellow)
        b = ([bx - 20, 200, bx + 20, 300], float(bx), 250.0, black)
        dets_per_frame.append((fi, [y, b]))
    tracks = sc._track(dets_per_frame, w, h, fps=25.0)
    # Each surviving tag must be a single colour end-to-end.
    for tr in tracks:
        kit = tr["kit_hsv"]
        assert kit is not None
        # kit is close to exactly one of the two team colours (not an average).
        from polyfut_v2.pipeline.color import hsv_distance
        d_yellow = hsv_distance(np.array(kit, np.float32), yellow)
        d_black = hsv_distance(np.array(kit, np.float32), black)
        assert min(d_yellow, d_black) < 30   # firmly one colour, not blended


def test_track_emits_median_kit_colour():
    import numpy as np
    w, h = 1000, 500
    red = np.array([120, 200, 200], np.float32)   # HSV-ish tag
    dets_per_frame = []
    for fi in range(0, 15, 3):
        bbox = [100, 200, 140, 300]
        dets_per_frame.append((fi, [(bbox, 120.0, 250.0, red)]))
    tracks = sc._track(dets_per_frame, w, h, fps=25.0)
    assert tracks[0]["kit_hsv"] == [120.0, 200.0, 200.0]


def test_taps_from_tracklet_subsamples_and_offsets():
    pts = [{"t": i * 0.1, "nx": 0.5, "ny": 0.5, "nw": 0.1, "nh": 0.2}
           for i in range(20)]
    taps = taps_from_tracklet({"points": pts}, start_sec=10.0, max_taps=8)
    assert len(taps) == 8
    assert taps[0]["t_sec"] == pytest.approx(10.0)
    assert all(set(t) == {"t_sec", "nx", "ny"} for t in taps)
    assert taps[-1]["t_sec"] > taps[0]["t_sec"]


def test_build_seed_clip_writes_clip_and_tracklets(tmp_path):
    # Synthetic clip: a bright box drifting right on a dark field.
    vpath = tmp_path / "clip.mp4"
    vw = cv2.VideoWriter(str(vpath), cv2.VideoWriter_fourcc(*"mp4v"), 25, (320, 240))
    for i in range(60):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        x = 40 + i * 3
        cv2.rectangle(frame, (x, 100), (x + 30, 180), (255, 255, 255), -1)
        vw.write(frame)
    vw.release()

    class _FakeDet:
        """Returns the bright box as the only 'player'."""
        def detect(self, frame, near=None):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ys, xs = np.where(gray > 128)
            if len(xs) == 0:
                return []
            b = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
            return [type("D", (), {"bbox": b})()]

    out = tmp_path / "seed.mp4"
    res = sc.build_seed_clip(str(vpath), 1.2, _FakeDet(), str(out))
    assert res is not None
    assert out.exists() and out.stat().st_size > 0
    assert res["width"] == 640 and res["height"] == 480   # 2x enhanced
    assert len(res["tracklets"]) >= 1
    pts = res["tracklets"][0]["points"]
    assert pts[0]["nx"] < pts[-1]["nx"]                   # node followed the box
