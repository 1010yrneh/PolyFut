"""Tests for Stage 7 orbital prior + sequential confidence scoring."""

import numpy as np

from polyfut_v2.config import PipelineV2Config
from polyfut_v2.pipeline.contacts import ContactCandidate
from polyfut_v2.pipeline.player_contacts import PlayerContact
from polyfut_v2.pipeline.scoring import orbital_prior, score_contacts
from polyfut_v2.pipeline.seed import build_seed_from_torso_crops

RED = (0, 0, 200)
CFG = PipelineV2Config()


def _crop(color=RED, h=40, w=30):
    return np.full((h, w, 3), color, dtype=np.uint8)


def _contact(t, x, y, is_my_team=True):
    cand = ContactCandidate(frame_index=int(t * 10), t_sec=t, processed_sec=t,
                            x=x, y=y, kinds=["kick"], strength=0.8)
    return PlayerContact(candidate=cand, player_bbox=[x - 5, y - 5, x + 5, y + 5],
                         player_dist_px=0.0, jersey_hsv=[0, 200, 200],
                         color_dist=5.0, is_my_team=is_my_team, n_color_samples=3)


def _seed():
    return build_seed_from_torso_crops([_crop(RED) for _ in range(3)])


# --- orbital_prior ---

def test_orbital_inside_is_full():
    assert orbital_prior(10.0, 0.0, CFG) == 1.0


def test_orbital_far_outside_hits_floor_never_zero():
    p = orbital_prior(5000.0, 0.0, CFG)
    assert p == CFG.orbital_floor
    assert p > 0.0  # never a hard reject


def test_orbital_long_gap_is_neutral():
    assert orbital_prior(5000.0, CFG.orbital_max_gap_sec + 1, CFG) == 1.0


def test_orbital_radius_grows_with_time():
    # A distance outside the base radius becomes "inside" after enough time.
    d = CFG.orbital_base_px + 100.0
    assert orbital_prior(d, 0.0, CFG) < 1.0
    assert orbital_prior(d, 5.0, CFG) == 1.0


# --- score_contacts ---

def test_orbital_prior_always_within_bounds():
    seed = _seed()
    contacts = [_contact(0.0, 100, 100), _contact(1.0, 140, 100), _contact(2.0, 500, 100)]
    scored = score_contacts(contacts, [_crop()] * 3, seed, CFG)
    for s in scored:
        assert CFG.orbital_floor <= s.orbital_prior <= 1.0
        assert 0.0 <= s.confidence <= 1.0


def test_strong_match_anchors_and_orbital_breaks_tie():
    seed = _seed()
    # Anchor at (100,100); two undecided red contacts 1s later — near vs teleport.
    contacts = [
        _contact(0.0, 100, 100, is_my_team=True),
        _contact(1.0, 140, 100, is_my_team=None),   # near the orbital
        _contact(1.0, 600, 100, is_my_team=None),   # would need a teleport
    ]
    scored = score_contacts(contacts, [_crop()] * 3, seed, CFG)
    by_x = {round(s.contact.candidate.x): s for s in scored}
    assert by_x[100].anchored is True           # strong appearance + my_team anchors
    assert by_x[140].confidence > by_x[600].confidence  # orbital breaks the tie
    assert by_x[600].orbital_prior >= CFG.orbital_floor  # far one down-weighted, not killed


def test_anchor_requires_my_team_and_strong_appearance():
    seed = _seed()
    contacts = [
        _contact(0.0, 100, 100, is_my_team=False),  # right kit but not confirmed team
        _contact(1.0, 120, 100, is_my_team=True),   # confirmed → anchors
    ]
    scored = score_contacts(contacts, [_crop(), _crop()], seed, CFG)
    assert scored[0].anchored is False
    assert scored[1].anchored is True


def test_missing_crop_uses_neutral_appearance():
    seed = _seed()
    contacts = [_contact(0.0, 100, 100)]
    scored = score_contacts(contacts, [None], seed, CFG)
    assert scored[0].appearance_score is None
    # No anchor yet → prior 1.0; confidence falls back to the neutral default.
    assert abs(scored[0].confidence - CFG.appearance_default) < 1e-6
    assert scored[0].anchored is False  # can't anchor without a measured match


def test_tracklets_split_on_large_time_gap():
    seed = _seed()
    contacts = [_contact(0.0, 100, 100), _contact(0.5, 110, 100),
                _contact(10.0, 300, 100)]  # big gap → new tracklet
    scored = score_contacts(contacts, [_crop()] * 3, seed, CFG)
    tids = [s.tracklet_id for s in scored]
    assert tids[0] == tids[1]
    assert tids[2] != tids[1]


def test_camera_transform_hook_applied():
    seed = _seed()
    contacts = [_contact(0.0, 100, 100, is_my_team=True), _contact(1.0, 100, 100, is_my_team=None)]
    # Pan of +500px/s: without compensation the 2nd contact sits on the anchor;
    # a transform that removes the pan should keep it inside the orbital.
    def undo_pan(xy, t):
        return (xy[0] - 500.0 * t, xy[1])
    scored = score_contacts(contacts, [_crop()] * 2, seed, CFG, transform=undo_pan)
    # 2nd contact compensated to x=100-500 vs anchor at x=100 → far → down-weighted.
    assert scored[1].orbital_prior < 1.0
