"""Tests for the enhanced seed-clip + tracked-node feature (Stage 0 UX)."""

import cv2
import numpy as np
import pytest

from polyfut_v2 import seed_clips as sc
from polyfut_v2.app_service import taps_from_tracklet
from polyfut_v2.pipeline.color import hex_to_hsv
from polyfut_v2.pipeline.player_detector import PlayerDetection


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


def test_reroll_base_roams_the_whole_match_without_a_short_cycle():
    """Successive shuffles of one slot must roam across the full match, not
    cycle back to a couple of nearby offsets (the flip-flop bug). The first
    12 rerolls should be well spread and contain no near-duplicates."""
    dur = 1000.0
    bases = [sc.moment_for_index(dur, index=1, reroll=r) for r in range(1, 13)]
    # No two shuffles land within ~4.5% of the match of each other.
    for i in range(len(bases)):
        for j in range(i + 1, len(bases)):
            assert abs(bases[i] - bases[j]) > 0.045 * dur, (bases[i], bases[j])
    # And they actually cover the timeline (not clustered in one region).
    assert min(bases) < 0.25 * dur and max(bases) > 0.75 * dur


def test_reroll_zero_is_the_spread_anchor():
    dur = 1000.0
    assert [sc.moment_for_index(dur, i, 0) for i in range(4)] == [100.0, 350.0, 600.0, 850.0]


class _FakeCountDetector:
    """Returns a fixed set of tall, well-spaced player boxes on a bright
    frame, none on a dark one — lets a test steer "how many players" per
    candidate moment without a real model."""
    def __init__(self, n=8, box_h=80, box_w=20, frame_mean_thresh=100):
        self.n, self.box_h, self.box_w, self.thresh = n, box_h, box_w, frame_mean_thresh

    def detect(self, frame, near=None):
        if frame.mean() < self.thresh:
            return []
        return [PlayerDetection([20 + i * 30, 50, 20 + i * 30 + self.box_w, 50 + self.box_h], 0.8)
                for i in range(self.n)]


def test_score_moment_zero_players_scores_lowest():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = sc._score_moment(frame, _FakeCountDetector(), my_kit_hsv=None)
    assert result["n_players"] == 0 and result["is_zoom"] is True
    assert result["score"] < 0


def test_score_moment_many_players_scores_well():
    frame = np.full((480, 640, 3), 200, dtype=np.uint8)
    result = sc._score_moment(frame, _FakeCountDetector(n=8), my_kit_hsv=None)
    assert result["n_players"] == 8
    assert result["is_zoom"] is False
    assert result["score"] == 8.0


def test_score_moment_flags_tight_zoom():
    """A couple of huge boxes (each well over a third of the frame's height)
    reads as a tight zoom/replay, not the normal wide broadcast angle."""
    frame = np.full((480, 640, 3), 200, dtype=np.uint8)
    # Wide enough to clear the human-proportion max-aspect gate (220/50 = 4.4).
    det = _FakeCountDetector(n=2, box_h=220, box_w=50)
    result = sc._score_moment(frame, det, my_kit_hsv=None)
    assert result["n_players"] == 2
    assert result["is_zoom"] is True
    assert result["score"] < result["n_players"]   # zoom penalty applied


def test_score_moment_kit_visibility():
    red_hsv = hex_to_hsv("#e23b3b")
    frame = np.full((480, 640, 3), 200, dtype=np.uint8)
    # Paint one player's box red (matches), leave the rest the background colour.
    frame[50:130, 20:40] = (59, 59, 226)   # BGR red-ish, matches #e23b3b roughly
    det = _FakeCountDetector(n=3)
    with_kit = sc._score_moment(frame, det, my_kit_hsv=red_hsv)
    assert with_kit["kit_visible"] is True

    blue_hsv = hex_to_hsv("#3b5de2")
    without_kit = sc._score_moment(frame, det, my_kit_hsv=blue_hsv)
    assert without_kit["kit_visible"] is False
    assert without_kit["score"] < with_kit["score"]


def _grass_frame(w=640, h=480):
    """A solid pitch-grass-coloured frame (BGR), for scene-type tests."""
    hsv = np.full((h, w, 3), (55, 180, 150), np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


class _FakeMovingDetector:
    """Finds bright rectangles on the (grass-coloured) background — unlike
    _FakeCountDetector, positions come from the actual frame content, so a
    video with genuinely shifting rectangles produces genuine measured motion
    for the motion/cut checks, and a plain grass frame correctly finds none."""

    def detect(self, frame, near=None):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = (gray > 180).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w < 3 or h < 3:
                continue
            out.append(PlayerDetection([float(x), float(y), float(x + w), float(y + h)], 0.8))
        return out


def _write_populated_moving_video(path, n_frames=20, fps=2, n_players=8, populated_from=6):
    """Grass background throughout; frames >= ``populated_from`` show n_players
    moving white rectangles (well-spaced, tall, genuinely shifting position each
    frame) — a stand-in for a normal, active, on-pitch broadcast shot. The
    populated region starts early enough that a candidate's full ±1.5s
    scoring window can fall entirely inside it."""
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (640, 480))
    for i in range(n_frames):
        frame = _grass_frame()
        if i >= populated_from:
            shift = (i - populated_from) * 15
            for k in range(n_players):
                x = 20 + k * 70 + shift
                cv2.rectangle(frame, (x, 50), (x + 20, 130), (255, 255, 255), -1)
        vw.write(frame)
    vw.release()


def test_pick_best_moment_near_prefers_a_populated_nearby_moment(tmp_path):
    """The base anchor lands on an empty stretch of pitch; a nearby moment (a
    few seconds away) is a normal, populated, actively-moving shot. The search
    should move to the good one instead of building a seed clip from an empty
    frame."""
    vpath = tmp_path / "clip.mp4"
    _write_populated_moving_video(vpath, populated_from=6)   # empty until t=3, populated after

    t = sc.pick_best_moment_near(
        str(vpath), t_center=1.0, player_detector=_FakeMovingDetector(),
        duration_sec=9.5,
    )
    assert t >= 3.0   # moved into the populated segment, not stuck at the empty anchor


def test_pick_best_moment_near_rejects_lone_player_and_widens_to_a_group(tmp_path):
    """The exact bug from the field: the anchor's whole local area has at most
    a single player, so the old logic picked that lone-player moment. Now the
    search must widen past it to a moment with a real group of players."""
    vpath = tmp_path / "clip.mp4"
    fps = 2
    vw = cv2.VideoWriter(str(vpath), cv2.VideoWriter_fourcc(*"mp4v"), fps, (640, 480))
    for i in range(40):        # 20s
        frame = _grass_frame()
        if i < 24:             # first 12s: at most ONE lone player, drifting
            cv2.rectangle(frame, (100 + i * 4, 50), (120 + i * 4, 130), (255, 255, 255), -1)
        else:                  # after 12s: a full group of 8 moving players
            for k in range(8):
                x = 20 + k * 70 + (i - 24) * 12
                cv2.rectangle(frame, (x, 50), (x + 20, 130), (255, 255, 255), -1)
        vw.write(frame)
    vw.release()

    t = sc.pick_best_moment_near(
        str(vpath), t_center=2.0, player_detector=_FakeMovingDetector(),
        duration_sec=19.5, min_players=6,
    )
    assert t >= 12.0   # skipped the lone-player region (t<12) entirely, into the group


def _write_busy_video(path, n_frames=40, fps=2, n_players=8):
    """A grass pitch with n_players moving (oscillating, always on-screen) every
    frame throughout — a long stretch of active play with several distinct,
    taggable moments."""
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (640, 480))
    for i in range(n_frames):
        frame = _grass_frame()
        for k in range(n_players):
            x = 40 + k * 60 + (i % 3) * 10   # shifts every frame -> genuine motion
            cv2.rectangle(frame, (x, 50), (x + 18, 130), (255, 255, 255), -1)
        vw.write(frame)
    vw.release()


def test_pick_best_moment_near_avoids_an_already_chosen_moment(tmp_path):
    """Two clips anchored at the same busy spot must NOT land on the same
    moment — the exact 'clip 3 == clip 4' failure. The second pick, told to
    avoid the first, must be well separated from it."""
    vpath = tmp_path / "clip.mp4"
    _write_busy_video(vpath, n_frames=40)   # 20s of continuous active play
    det = _FakeMovingDetector()

    first = sc.pick_best_moment_near(str(vpath), 6.0, det, duration_sec=19.5)
    second = sc.pick_best_moment_near(
        str(vpath), 6.0, det, duration_sec=19.5, avoid_sec=(first,))
    assert abs(second - first) >= 8.0     # distinct passage of play, not the same clip


def test_pick_best_moment_near_falls_back_when_nothing_qualifies(tmp_path):
    """If nothing in the whole search radius clears the player-count bar, don't
    return nothing — fall back to the original anchor."""
    vpath = tmp_path / "clip.mp4"
    vw = cv2.VideoWriter(str(vpath), cv2.VideoWriter_fourcc(*"mp4v"), 2, (640, 480))
    for _ in range(20):
        vw.write(_grass_frame())   # grass throughout, but never any players
    vw.release()

    t = sc.pick_best_moment_near(
        str(vpath), t_center=2.0, player_detector=_FakeMovingDetector(),
        duration_sec=9.5,
    )
    assert t == 2.0


def test_moment_for_index_matches_default_moments():
    dur = 200.0
    for reroll in (0, 1, 3):
        assert [sc.moment_for_index(dur, i, reroll) for i in range(4)] == \
            sc.default_moments(dur, reroll)


def test_moment_for_index_reroll_shifts_only_that_slot():
    dur = 200.0
    base = sc.moment_for_index(dur, 2, reroll=0)
    shifted = sc.moment_for_index(dur, 2, reroll=1)
    assert base != shifted                         # this slot moved
    # other slots are computed independently of slot 2's reroll
    assert sc.moment_for_index(dur, 0, reroll=0) == sc.moment_for_index(dur, 0, reroll=0)


def test_median_player_displacement_detects_motion():
    frame_a = _grass_frame()
    frame_b = _grass_frame()
    cv2.rectangle(frame_a, (100, 50), (120, 130), (255, 255, 255), -1)
    cv2.rectangle(frame_b, (160, 50), (180, 130), (255, 255, 255), -1)   # shifted 60px
    disp = sc._median_player_displacement(frame_a, frame_b, _FakeMovingDetector())
    assert disp is not None and disp > sc._STATIC_DISPLACEMENT_PX


def test_median_player_displacement_detects_stillness():
    frame_a = _grass_frame()
    frame_b = _grass_frame()
    for f in (frame_a, frame_b):
        cv2.rectangle(f, (100, 50), (120, 130), (255, 255, 255), -1)   # identical position
    disp = sc._median_player_displacement(frame_a, frame_b, _FakeMovingDetector())
    assert disp is not None and disp < sc._STATIC_DISPLACEMENT_PX


def test_median_player_displacement_none_without_players():
    assert sc._median_player_displacement(
        _grass_frame(), _grass_frame(), _FakeMovingDetector()) is None


def test_hist_diff_similar_frames_is_low():
    frame_a = _grass_frame()
    frame_b = _grass_frame()
    cv2.rectangle(frame_b, (100, 50), (120, 130), (255, 255, 255), -1)   # minor difference
    assert sc._hist_diff(frame_a, frame_b) < sc._CUT_HIST_DIST


def test_hist_diff_different_frames_is_high():
    frame_a = _grass_frame()
    frame_b = np.full((480, 640, 3), (10, 10, 10), dtype=np.uint8)   # a totally different shot
    assert sc._hist_diff(frame_a, frame_b) > sc._CUT_HIST_DIST


def test_score_candidate_flags_crowd_shot(tmp_path):
    """A frame with plenty of "players" but no pitch grass visible (a crowd
    stand, a bench close-up, a graphic) should be penalised as a crowd shot."""
    vpath = tmp_path / "clip.mp4"
    vw = cv2.VideoWriter(str(vpath), cv2.VideoWriter_fourcc(*"mp4v"), 2, (640, 480))
    frame = np.full((480, 640, 3), (60, 60, 60), dtype=np.uint8)   # no grass at all
    for k in range(8):
        cv2.rectangle(frame, (20 + k * 70, 50), (20 + k * 70 + 20, 130), (255, 255, 255), -1)
    for _ in range(4):
        vw.write(frame)
    vw.release()

    result = sc._score_candidate(
        str(vpath), 1.0, _FakeMovingDetector(), my_kit_hsv=None, clip_half_sec=1.5)
    assert result is not None
    assert result["n_players"] > 0
    assert result["is_crowd"] is True
    assert result["score"] < result["n_players"]


def test_score_candidate_flags_stationary_huddle(tmp_path):
    """Plenty of players, on the pitch, but nobody moves between frames a
    moment apart — a stoppage/substitution/team-huddle, not active play."""
    vpath = tmp_path / "clip.mp4"
    vw = cv2.VideoWriter(str(vpath), cv2.VideoWriter_fourcc(*"mp4v"), 2, (640, 480))
    frame = _grass_frame()
    for k in range(8):
        cv2.rectangle(frame, (20 + k * 70, 50), (20 + k * 70 + 20, 130), (255, 255, 255), -1)
    for _ in range(6):   # identical frame every sample -> zero motion
        vw.write(frame)
    vw.release()

    result = sc._score_candidate(
        str(vpath), 1.5, _FakeMovingDetector(), my_kit_hsv=None, clip_half_sec=1.5)
    assert result is not None
    assert result["n_players"] > 0
    assert result["is_static"] is True
    assert result["score"] < result["n_players"]


def test_score_candidate_flags_camera_cut(tmp_path):
    """The window starts as a normal pitch shot and ends as a completely
    different-looking shot — a mid-clip cut (replay/angle switch)."""
    vpath = tmp_path / "clip.mp4"
    vw = cv2.VideoWriter(str(vpath), cv2.VideoWriter_fourcc(*"mp4v"), 2, (640, 480))
    n = 8
    for i in range(n):
        if i < n // 2:
            frame = _grass_frame()
            for k in range(8):
                cv2.rectangle(frame, (20 + k * 70, 50), (20 + k * 70 + 20, 130), (255, 255, 255), -1)
        else:
            frame = np.full((480, 640, 3), (10, 10, 200), dtype=np.uint8)   # a totally different shot
        vw.write(frame)
    vw.release()

    result = sc._score_candidate(
        str(vpath), 1.0, _FakeMovingDetector(), my_kit_hsv=None, clip_half_sec=1.5)
    assert result is not None
    assert result["is_cut"] is True
    assert result["score"] < result["n_players"]


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


def test_track_reconnects_across_a_missed_detection():
    """A player is briefly missed (occlusion / one bad frame), then reappears
    nearby with the same kit colour. Without merging this looks like two
    unrelated tags in the same spot ("detection is all over the place");
    with it, it must be a single continuous tracklet."""
    import numpy as np
    w, h = 1000, 500
    kit = np.array([27, 200, 200], np.float32)
    dets_per_frame = []
    for k, fi in enumerate(range(0, 12, 3)):     # frames 0,3,6,9 — present
        cx = 200 + k * 20
        bbox = [cx - 20, 200, cx + 20, 300]
        dets_per_frame.append((fi, [(bbox, float(cx), 250.0, kit)]))
    # missed at fi=12 (occlusion) — no detection this sampled frame at all
    dets_per_frame.append((12, []))
    # reappears at fi=15, close to where it was last seen, same colour
    for k, fi in enumerate(range(15, 24, 3)):
        cx = 260 + k * 20
        bbox = [cx - 20, 200, cx + 20, 300]
        dets_per_frame.append((fi, [(bbox, float(cx), 250.0, kit)]))
    tracks = sc._track(dets_per_frame, w, h, fps=25.0)
    assert len(tracks) == 1
    pts = tracks[0]["points"]
    assert pts[0]["t"] < pts[-1]["t"]
    assert pts[0]["nx"] < pts[-1]["nx"]


def test_track_does_not_merge_across_a_long_gap():
    """A long silence between fragments (well beyond a momentary miss) should
    NOT be bridged, even at a position/colour a short gap would happily
    allow — that gap is long enough it could plausibly be a different player.

    The position drift here (>max_dist=80px) also makes ``_track``'s own
    frame-to-frame loop treat these as two tracks from the start (it has no
    gap awareness at all — see the merge-across-a-miss test above), so this
    isolates the merge pass's explicit gap cutoff rather than accidentally
    relying on ``_track``'s unrelated static distance threshold.
    """
    import numpy as np
    w, h = 1000, 500
    kit = np.array([27, 200, 200], np.float32)
    dets_per_frame = [
        (0, [([180, 200, 220, 300], 200.0, 250.0, kit)]),
        (3, [([180, 200, 220, 300], 200.0, 250.0, kit)]),
        # >0.5s gap (25fps) before the next detection reappears, drifted well
        # past max_dist so it's a fresh track either way, not a frame-loop match.
        (40, [([390, 200, 430, 300], 410.0, 250.0, kit)]),
        (43, [([390, 200, 430, 300], 410.0, 250.0, kit)]),
    ]
    tracks = sc._track(dets_per_frame, w, h, fps=25.0)
    assert len(tracks) == 2


def test_track_does_not_merge_a_colour_mismatch_across_a_gap():
    """A same-position reappearance with a clearly different kit colour is a
    different player, not a reconnection — colour-lock must still hold across
    the gap, not just frame-to-frame."""
    import numpy as np
    w, h = 1000, 500
    yellow = np.array([27, 200, 200], np.float32)
    black = np.array([0, 30, 40], np.float32)
    dets_per_frame = [
        (0, [([180, 200, 220, 300], 200.0, 250.0, yellow)]),
        (3, [([180, 200, 220, 300], 200.0, 250.0, yellow)]),
        (9, [([190, 200, 230, 300], 210.0, 250.0, black)]),
        (12, [([190, 200, 230, 300], 210.0, 250.0, black)]),
    ]
    tracks = sc._track(dets_per_frame, w, h, fps=25.0)
    assert len(tracks) == 2


def test_track_collapses_duplicate_boxes_on_one_player():
    """The detector emitting two overlapping boxes for ONE player creates two
    parallel tracks that coexist for the whole clip — two tappable markers a
    few pixels apart. The sequential merge can't touch those (it only rejoins
    fragments), so the concurrent pass must collapse them."""
    import numpy as np
    w, h = 1000, 500
    kit = np.array([27, 200, 200], np.float32)
    dets_per_frame = []
    for fi in range(0, 24, 3):
        cx = 200 + fi * 4
        a = ([cx - 20, 200, cx + 20, 300], float(cx), 250.0, kit)
        b = ([cx - 14, 206, cx + 26, 306], float(cx + 6), 256.0, kit)  # ~8px away
        dets_per_frame.append((fi, [a, b]))
    tracks = sc._track(dets_per_frame, w, h, fps=25.0)
    assert len(tracks) == 1          # collapsed to a single marker


def test_track_keeps_two_genuinely_separate_players():
    """Regression guard: the dedup must not swallow real, distinct players."""
    import numpy as np
    w, h = 1000, 500
    kit = np.array([27, 200, 200], np.float32)
    dets_per_frame = []
    for fi in range(0, 24, 3):
        a = ([100, 200, 140, 300], 120.0, 250.0, kit)
        b = ([800, 200, 840, 300], 820.0, 250.0, kit)   # far apart all clip
        dets_per_frame.append((fi, [a, b]))
    tracks = sc._track(dets_per_frame, w, h, fps=25.0)
    assert len(tracks) == 2


def test_merge_concurrent_keeps_players_who_only_cross_briefly():
    """Two players passing close for a moment share a few frames at small
    distance — but the MEDIAN distance over shared frames stays large, so they
    must not be merged."""
    import numpy as np
    kit = np.array([27, 200, 200], np.float32)
    w, h = 1000, 500
    def mk(xs):
        return {"_cx": xs[-1], "_cy": 250.0, "_hsv": [kit], "_color": kit,
                "points": [{"t": i * 0.12, "nx": x / w, "ny": 250.0 / h,
                            "nw": 0.04, "nh": 0.2, "c": None}
                           for i, x in enumerate(xs)]}
    # cross at the middle sample only
    left  = mk([100, 300, 500, 700, 900])
    right = mk([900, 700, 500, 300, 100])
    out = sc._merge_concurrent_tracks([left, right], w, h)
    assert len(out) == 2


def test_merge_concurrent_respects_kit_colour():
    """Same position, different kit -> different people (e.g. a tight duel);
    colour must veto the merge."""
    import numpy as np
    w, h = 1000, 500
    yellow = np.array([27, 200, 200], np.float32)
    black = np.array([0, 30, 40], np.float32)
    def mk(col):
        return {"_cx": 200.0, "_cy": 250.0, "_hsv": [col], "_color": col,
                "points": [{"t": i * 0.12, "nx": 0.2, "ny": 0.5,
                            "nw": 0.04, "nh": 0.2, "c": None} for i in range(5)]}
    out = sc._merge_concurrent_tracks([mk(yellow), mk(black)], w, h)
    assert len(out) == 2


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


def _mk_tracklet(id_, t_list):
    return {"id": id_, "points": [{"t": t, "nx": 0.5, "ny": 0.5, "nw": 0.05, "nh": 0.1}
                                   for t in t_list], "kit_hsv": None}


def test_filter_short_tracklets_drops_brief_blips():
    tracklets = [
        _mk_tracklet(0, [0.0, 0.08, 0.16]),          # 0.16s span, 3 pts — too brief
        _mk_tracklet(1, [0.0, 0.5, 1.0, 1.5, 2.0]),  # 2.0s span, 5 pts — keep
    ]
    out = sc._filter_short_tracklets(tracklets)
    assert len(out) == 1
    assert out[0]["points"][0]["t"] == 0.0 and out[0]["points"][-1]["t"] == 2.0


def test_filter_short_tracklets_drops_sparse_long_span():
    """Two points spanning a long duration (one detection, a big gap, one more)
    is just as unreliable as a tight short cluster — duration alone isn't
    enough evidence, so the point-count floor must also apply."""
    tracklets = [_mk_tracklet(0, [0.0, 1.8])]   # 1.8s span but only 2 points
    assert sc._filter_short_tracklets(tracklets) == []


def test_filter_short_tracklets_renumbers_ids():
    tracklets = [
        _mk_tracklet(0, [0.0, 0.08]),                 # dropped
        _mk_tracklet(1, [0.0, 0.5, 1.0, 1.5, 2.0]),    # kept -> becomes id 0
        _mk_tracklet(2, [0.0, 0.5, 1.0, 1.5, 2.0]),    # kept -> becomes id 1
    ]
    out = sc._filter_short_tracklets(tracklets)
    assert [tr["id"] for tr in out] == [0, 1]


def test_taps_from_tracklet_subsamples_and_offsets():
    pts = [{"t": i * 0.1, "nx": 0.5, "ny": 0.5, "nw": 0.1, "nh": 0.2}
           for i in range(20)]
    taps = taps_from_tracklet({"points": pts}, start_sec=10.0, max_taps=8)
    assert len(taps) == 8
    assert taps[0]["t_sec"] == pytest.approx(10.0)
    assert all(set(t) == {"t_sec", "nx", "ny"} for t in taps)
    assert taps[-1]["t_sec"] > taps[0]["t_sec"]


def test_build_seed_clip_filters_ball_shaped_detection(tmp_path):
    """A small, roughly-square "player" detection (the classic ball
    misclassification) must not produce its own tappable node on the seed
    screen — it should be filtered before ever reaching the tracker."""
    vpath = tmp_path / "clip.mp4"
    vw = cv2.VideoWriter(str(vpath), cv2.VideoWriter_fourcc(*"mp4v"), 25, (320, 240))
    for i in range(60):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        x = 40 + i * 2
        # A tiny bright square (~6x7px in this 320x240 frame -> ~12x14px once
        # the seed clip 2x-enhances it) — ball-sized, not player-sized.
        cv2.rectangle(frame, (x, 100), (x + 6, 107), (255, 255, 255), -1)
        vw.write(frame)
    vw.release()

    class _FakeBallShapedDet:
        def detect(self, frame, near=None):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ys, xs = np.where(gray > 128)
            if len(xs) == 0:
                return []
            b = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
            return [type("D", (), {"bbox": b})()]

    out = tmp_path / "seed.mp4"
    res = sc.build_seed_clip(str(vpath), 1.2, _FakeBallShapedDet(), str(out))
    assert res is not None
    assert res["tracklets"] == []


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
    assert res["width"] == 320 and res["height"] == 240   # native resolution
    assert len(res["tracklets"]) >= 1
    pts = res["tracklets"][0]["points"]
    assert pts[0]["nx"] < pts[-1]["nx"]                   # node followed the box
