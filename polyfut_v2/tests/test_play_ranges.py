"""Tests for the playing-time window (v3).

The failure this feature exists to stop (job e7efd5ac4bc3): on a 111-minute
video where the user came on at 63', seed clips landed at 11/39/66/94 min and
the shuffle roamed to 12 and 51 min — moments the user wasn't on the pitch — so
the appearance gallery was seeded from a different player and every
auto-accepted touch was wrong. These tests pin the rules that prevent it.
"""

import numpy as np
import pytest

from polyfut_v2 import seed_clips as sc
from polyfut_v2.config import PipelineV2Config
from polyfut_v2.pipeline import play_ranges as pr
from polyfut_v2.pipeline.player_detector import PlayerDetection

# The real case: 111-min video, subbed on at 63'.
SUB_ON = [(63 * 60.0, 111 * 60.0)]
FULL = 111 * 60.0


# --- normalize / merge -------------------------------------------------------

def test_normalize_sorts_merges_and_clamps():
    out = pr.normalize_ranges([[100, 60], [500, 700], [650, 800]], duration_sec=750)
    # reversed pair is fixed, overlapping pair merged, tail clamped to duration
    assert out == [(60.0, 100.0), (500.0, 750.0)]


def test_normalize_accepts_dicts_and_drops_junk():
    out = pr.normalize_ranges(
        [{"start": 10, "end": 20}, {"start": 5, "end": 5}, "nope", [None, 3], [30, 40]],
        duration_sec=100,
    )
    assert out == [(10.0, 20.0), (30.0, 40.0)]   # zero-length + unparseable dropped


def test_normalize_never_raises_on_malformed_input():
    """A bad window must degrade to 'whole match', never to a failed run."""
    assert pr.normalize_ranges(None) == []
    assert pr.normalize_ranges([[float("nan"), 10]]) == []
    assert pr.normalize_ranges([[-50, -10]], duration_sec=100) == []


def test_touching_ranges_merge():
    assert pr.normalize_ranges([[0, 100], [100.2, 200]]) == [(0.0, 200.0)]


def test_is_whole_match():
    assert pr.is_whole_match([], FULL)
    assert pr.is_whole_match([(0.0, FULL)], FULL)
    assert pr.is_whole_match([(0.5, FULL - 0.5)], FULL)      # within tolerance
    assert not pr.is_whole_match(SUB_ON, FULL)
    assert not pr.is_whole_match([(0, 100), (200, FULL)], FULL)


# --- empty list means unrestricted ------------------------------------------

def test_empty_ranges_are_unrestricted_everywhere():
    assert pr.contains([], 12345.0)
    assert pr.envelope([]) is None
    assert pr.resolve([], 90.0) == [(0.0, 90.0)]
    shots = [{"start_sec": 0, "end_sec": 10, "label": "main_camera"}]
    assert pr.intersect_shots(shots, []) == shots


# --- cumulative fraction mapping --------------------------------------------

def test_time_at_fraction_maps_onto_the_union_not_the_video():
    ranges = [(0.0, 100.0), (900.0, 1000.0)]   # 200s of play across a big gap
    assert pr.time_at_fraction(ranges, 0.0) == 0.0
    assert pr.time_at_fraction(ranges, 0.25) == 50.0     # quarter of PLAY time
    assert pr.time_at_fraction(ranges, 0.75) == 950.0    # lands in the 2nd range
    assert pr.time_at_fraction(ranges, 1.0) == 1000.0


def test_time_at_fraction_never_lands_in_a_gap():
    ranges = [(0.0, 60.0), (600.0, 660.0), (1200.0, 1260.0)]
    for i in range(101):
        t = pr.time_at_fraction(ranges, i / 100.0)
        assert pr.contains(ranges, t), (i, t)


def test_time_at_fraction_tiny_range():
    assert pr.contains([(50.0, 51.0)], pr.time_at_fraction([(50.0, 51.0)], 0.5))


def test_time_at_fraction_range_at_video_end():
    ranges = [(FULL - 30.0, FULL)]
    for frac in (0.0, 0.5, 1.0):
        t = pr.time_at_fraction(ranges, frac)
        assert FULL - 30.0 <= t <= FULL


# --- padding -----------------------------------------------------------------

def test_pad_widens_both_sides_and_clamps_to_the_video():
    assert pr.pad_ranges([(100.0, 200.0)], 45.0, 1000.0) == [(55.0, 245.0)]
    # can't pad before 0 or past the end
    assert pr.pad_ranges([(10.0, 990.0)], 45.0, 1000.0) == [(0.0, 1000.0)]


def test_pad_merges_ranges_that_grow_into_each_other():
    assert pr.pad_ranges([(0.0, 100.0), (150.0, 250.0)], 45.0, 1000.0) == [(0.0, 295.0)]


def test_pad_is_a_noop_without_ranges_or_padding():
    assert pr.pad_ranges([], 45.0, 100.0) == []
    assert pr.pad_ranges([(1.0, 2.0)], 0.0, 100.0) == [(1.0, 2.0)]


# --- live-shot intersection --------------------------------------------------

def _shot(a, b, label="main_camera"):
    return {"start_sec": a, "end_sec": b, "label": label}


def test_shots_are_clipped_split_and_dropped():
    ranges = [(100.0, 200.0), (300.0, 400.0)]
    shots = [
        _shot(0, 50),        # entirely before  -> dropped
        _shot(90, 120),      # overlaps start   -> clipped
        _shot(130, 150),     # fully inside     -> untouched
        _shot(180, 320),     # straddles a gap  -> split in two
        _shot(220, 260),     # inside the gap   -> dropped
        _shot(500, 600),     # entirely after   -> dropped
    ]
    out = pr.intersect_shots(shots, ranges)
    assert [(s["start_sec"], s["end_sec"]) for s in out] == [
        (100.0, 120.0), (130.0, 150.0), (180.0, 200.0), (300.0, 320.0),
    ]


def test_intersect_preserves_other_shot_keys_on_every_piece():
    out = pr.intersect_shots(
        [{"start_sec": 0, "end_sec": 1000, "label": "main_camera", "idx": 7}],
        [(100.0, 200.0), (300.0, 400.0)],
    )
    assert len(out) == 2
    assert all(s["label"] == "main_camera" and s["idx"] == 7 for s in out)


def test_intersect_skips_malformed_shots():
    out = pr.intersect_shots([{"label": "main_camera"}, _shot(10, 20)], [(0.0, 100.0)])
    assert len(out) == 1


# --- hashing (cache keys) ----------------------------------------------------

def test_ranges_hash_is_stable_distinct_and_all_for_whole_match():
    assert pr.ranges_hash([]) == "all"
    assert pr.ranges_hash(SUB_ON) == pr.ranges_hash(list(SUB_ON))
    assert pr.ranges_hash(SUB_ON) != pr.ranges_hash([(0.0, FULL)])
    assert len(pr.ranges_hash(SUB_ON)) == 8


def test_clamp_to_ranges_pulls_a_gap_time_to_the_nearest_edge():
    ranges = [(100.0, 200.0), (300.0, 400.0)]
    assert pr.clamp_to_ranges(ranges, 150.0) == 150.0    # already inside
    assert pr.clamp_to_ranges(ranges, 210.0) == 200.0    # nearer the 1st range
    assert pr.clamp_to_ranges(ranges, 290.0) == 300.0    # nearer the 2nd
    assert pr.clamp_to_ranges(ranges, 50.0) == 100.0
    assert pr.clamp_to_ranges([], 50.0) == 50.0          # unrestricted


# --- seed moments confined ---------------------------------------------------

def test_seed_moments_land_inside_a_single_range():
    """The regression: default slots at 11/39/66/94 min on a 111-min video."""
    unbounded = sc.default_moments(FULL, 0)
    assert any(t < 63 * 60 for t in unbounded)          # the old behaviour was broken

    bounded = sc.default_moments(FULL, 0, play_ranges=SUB_ON)
    assert len(bounded) == 4
    assert all(pr.contains(SUB_ON, t) for t in bounded), bounded
    assert len(set(bounded)) == 4                        # still spread apart


def test_every_shuffle_stays_inside_the_window():
    """The other half of the regression: shuffle roamed to 12 and 51 min."""
    for reroll in range(1, 40):
        for index in range(4):
            t = sc.moment_for_index(FULL, index, reroll, play_ranges=SUB_ON)
            assert pr.contains(SUB_ON, t), (index, reroll, t)


def test_moments_stay_inside_multiple_ranges():
    ranges = [(10 * 60.0, 25 * 60.0), (70 * 60.0, 85 * 60.0)]
    for reroll in range(0, 20):
        for t in sc.default_moments(FULL, reroll, play_ranges=ranges):
            assert pr.contains(ranges, t), (reroll, t)


def test_moments_unchanged_without_a_window():
    """Whole-match default must be byte-for-byte the old behaviour."""
    for reroll in (0, 1, 5):
        assert (sc.default_moments(FULL, reroll)
                == sc.default_moments(FULL, reroll, play_ranges=None))


def test_moments_spread_across_both_ranges_not_bunched_in_one():
    ranges = [(0.0, 600.0), (3000.0, 3600.0)]
    moments = sc.default_moments(FULL, 0, play_ranges=ranges)
    assert any(t <= 600.0 for t in moments)
    assert any(t >= 3000.0 for t in moments)


# --- moment search confined --------------------------------------------------

class _FlatDetector:
    """Every frame looks equally good, so the search returns the first
    candidate it considers acceptable — which makes 'which offsets were even
    tried' directly observable."""

    def detect(self, frame, _last=None):
        return [PlayerDetection(bbox=[x, 100, x + 12, 140], conf=0.9)
                for x in range(0, 240, 20)]


@pytest.fixture()
def flat_video(tmp_path):
    import cv2
    path = tmp_path / "flat.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (320, 180))
    rng = np.random.default_rng(0)
    for _ in range(25 * 40):          # 40s of noise so frames read fine anywhere
        writer.write(rng.integers(0, 255, (180, 320, 3), dtype=np.uint8))
    writer.release()
    return str(path)


def test_pick_best_moment_never_leaves_the_window(flat_video):
    ranges = [(20.0, 30.0)]
    for anchor in (0.0, 10.0, 25.0, 35.0, 39.0):
        t = sc.pick_best_moment_near(
            flat_video, anchor, _FlatDetector(), 40.0,
            min_players=1, play_ranges=ranges,
        )
        assert pr.contains(ranges, t), (anchor, t)


def test_pick_best_moment_falls_back_inside_the_window(flat_video):
    """Even when nothing qualifies, the returned moment is in-window — the
    fallback is clamped rather than left at an out-of-range anchor."""
    ranges = [(20.0, 30.0)]
    t = sc.pick_best_moment_near(
        flat_video, 0.0, _FlatDetector(), 40.0,
        min_players=999,                    # nothing can ever qualify
        play_ranges=ranges,
    )
    assert pr.contains(ranges, t)


# --- the pipeline actually runs inside the window ---------------------------

class _NoBall:
    """Ball detector that never fires — Stages 1-3 still segment shots, which is
    what these tests inspect."""

    def detect(self, frame, _last=None):
        return None


@pytest.fixture()
def match_clip(tmp_path):
    """A clip the shot filter actually labels ``main_camera``.

    conftest's plain-fill pitch is scored as a graphic overlay (its top/bottom
    bands have near-zero variance once downscaled) and comes back ``discard``,
    which would make every live-shot assertion below vacuously true. Mown-grass
    stripes and pitch lines give those bands real variance while keeping the
    frame overwhelmingly green.
    """
    import cv2
    path = tmp_path / "match.mp4"
    h, w, fps = 360, 640, 25
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(int(12.0 * fps)):
        frame = np.full((h, w, 3), (35, 140, 35), dtype=np.uint8)
        for x in range(0, w, 40):
            frame[:, x:x + 20] = (28, 115, 28)
        cv2.line(frame, (0, 60), (w, 60), (230, 230, 230), 3)
        cv2.line(frame, (0, h - 40), (w, h - 40), (230, 230, 230), 3)
        for j, ox in enumerate([120, 280, 400, 520]):
            x = int(ox + 45 * np.sin(i / 9.0 + j))
            colour = (0, 0, 220) if j % 2 == 0 else (235, 235, 235)
            cv2.rectangle(frame, (x, 190 + j * 6), (x + 26, 260 + j * 6), colour, -1)
        cv2.circle(frame, (int(250 + 70 * np.sin(i / 7.0)), 250), 6, (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    return str(path)


def test_match_clip_fixture_is_actually_live_play(match_clip):
    """Guard: if the fixture ever stops classifying as main_camera, the
    live-shot assertions below would silently pass on an empty list."""
    from polyfut_video.pipeline.decode import iter_frames
    from polyfut_video.pipeline.shot_filter import segment_and_classify_shots

    cfg = PipelineV2Config()
    shots = segment_and_classify_shots(
        iter_frames(match_clip, target_width=cfg.target_width,
                    sample_every_n=cfg.shot_filter_sample_every_n),
        cfg.v1_config(),
    )
    assert any(s["label"] == "main_camera" for s in shots), shots


def test_compute_trajectory_confines_live_shots_to_the_padded_window(match_clip):
    from polyfut_v2.main import compute_trajectory

    cfg = PipelineV2Config(playing_time_pad_sec=1.0)
    res = compute_trajectory(match_clip, cfg, detector=_NoBall(),
                             play_ranges=[(4.0, 7.0)])
    padded = [tuple(r) for r in res["play_ranges_padded"]]
    assert padded == [(3.0, 8.0)]
    assert res["play_ranges"] == [[4.0, 7.0]]
    assert res["live_shots"], "window analysis produced no live shots at all"
    for shot in res["live_shots"]:
        assert shot["start_sec"] >= 3.0 - 0.1
        assert shot["end_sec"] <= 8.0 + 0.1


def test_compute_trajectory_unconfined_without_a_window(match_clip):
    """No window == the old behaviour: the whole video is analysed."""
    from polyfut_v2.main import compute_trajectory

    cfg = PipelineV2Config()
    res = compute_trajectory(match_clip, cfg, detector=_NoBall())
    assert res["play_ranges"] == [] and res["play_ranges_padded"] == []
    span = max(s["end_sec"] for s in res["live_shots"]) - \
        min(s["start_sec"] for s in res["live_shots"])
    assert span > 8.0        # covers far more than any 3s window would


def test_multi_period_window_leaves_the_interior_gap_unanalysed(match_clip):
    from polyfut_v2.main import compute_trajectory

    cfg = PipelineV2Config(playing_time_pad_sec=0.5)
    res = compute_trajectory(match_clip, cfg, detector=_NoBall(),
                             play_ranges=[(1.0, 3.0), (9.0, 11.0)])
    for shot in res["live_shots"]:
        mid = (shot["start_sec"] + shot["end_sec"]) / 2
        assert not (3.6 < mid < 8.4), f"shot in the gap between periods: {shot}"


# --- config ------------------------------------------------------------------

def test_padding_config_default_is_present_and_sane():
    cfg = PipelineV2Config()
    assert 0 < cfg.playing_time_pad_sec <= 120
