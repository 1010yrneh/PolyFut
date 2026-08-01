"""Issue 4 — motion continuity as the same-kit identity backbone.

Same-kit teammates are near-indistinguishable by appearance (HSV histogram) on
~15px players. These tests pin the motion-continuity contract so regressions
cannot silently restore "appearance leads, motion nudges":

  * a strong motion chain carries identity across gaps the old 3s tracklet cut
    would snap;
  * weak / identical appearance is floored by motion when inside the orbital;
  * a teleporting same-kit teammate cannot outrank a motion-carried chain;
  * colour-lock blocks cross-kit chain hijacks;
  * past continuity / orbital horizon the chain hard-breaks and degrades safely;
  * confidence never hard-rejects (floor > 0).

Uses a controllable FakeAppearance so identical-kit crops do not accidentally
depend on histogram noise — the chokepoint under test is motion, not colour.
"""

from __future__ import annotations

import numpy as np
import pytest

from polyfut_v2.config import PipelineV2Config
from polyfut_v2.pipeline.contacts import ContactCandidate
from polyfut_v2.pipeline.player_contacts import PlayerContact
from polyfut_v2.pipeline.scoring import (
    combine_confidence,
    kit_compatible,
    orbital_prior,
    score_contacts,
)
from polyfut_v2.pipeline.seed import TargetSeed

RED_KIT = [0.0, 200.0, 200.0]
BLUE_KIT = [120.0, 200.0, 200.0]


class FakeAppearance:
    """Appearance model that returns a scripted score per crop id.

    ``scores`` maps ``id(crop)`` → float. Unknown / None crops → None
    (unmeasurable). Gallery descriptors are ignored for the score itself —
    ``descriptor`` encodes the scripted value so ``similarity`` can recover it.
    """

    def __init__(self, scores: dict[int, float | None] | None = None, default: float | None = 0.55):
        self.scores = scores or {}
        self.default = default

    def descriptor(self, crop):
        if crop is None:
            return None
        if id(crop) in self.scores:
            s = self.scores[id(crop)]
            return None if s is None else np.array([float(s)], np.float32)
        if self.default is None:
            return None
        return np.array([float(self.default)], np.float32)

    def similarity(self, a, b):
        # Contact descriptor carries the scripted score in a[0].
        return float(a[0]) if a is not None and len(a) else 0.0

    def gallery_descriptors(self, crops):
        # Gallery entries just need to exist; similarity ignores their values.
        return [np.array([1.0], np.float32) for _ in crops if _ is not None]

    def gallery_score(self, crop, gallery):
        d = self.descriptor(crop)
        if d is None or not gallery:
            return None
        return max(self.similarity(d, g) for g in gallery)


def _crop(tag: str = "x"):
    """Unique ndarray identity so FakeAppearance can key on id(crop)."""
    img = np.zeros((20, 12, 3), np.uint8)
    # Embed a unique tag byte so arrays aren't accidentally shared.
    img[0, 0, 0] = hash(tag) % 256
    return img


def _contact(
    t: float,
    x: float,
    y: float,
    *,
    is_my_team=True,
    jersey_hsv=None,
):
    cand = ContactCandidate(
        frame_index=int(t * 10), t_sec=t, processed_sec=t,
        x=x, y=y, kinds=["kick"], strength=0.8,
    )
    return PlayerContact(
        candidate=cand,
        player_bbox=[x - 5, y - 5, x + 5, y + 5],
        player_dist_px=0.0,
        jersey_hsv=list(jersey_hsv) if jersey_hsv is not None else list(RED_KIT),
        color_dist=5.0,
        is_my_team=is_my_team,
        n_color_samples=3,
    )


def _seed():
    return TargetSeed(kit_hsv=np.asarray(RED_KIT, np.float32), gallery=[_crop("g0"), _crop("g1")])


def _cfg(**kwargs) -> PipelineV2Config:
    base = dict(
        tracklet_max_gap_sec=3.0,
        continuity_max_gap_sec=8.0,
        continuity_color_max_dist=60.0,
        motion_carry_floor=0.70,
        orbital_base_px=80.0,
        orbital_growth_px_s=60.0,
        orbital_max_gap_sec=8.0,
        orbital_floor=0.5,
        orbital_anchor_min=0.6,
        appearance_default=0.5,
    )
    base.update(kwargs)
    return PipelineV2Config(**base)


# ---------------------------------------------------------------------------
# Unit: helpers
# ---------------------------------------------------------------------------

def test_kit_compatible_allows_unreadable_either_side():
    assert kit_compatible(None, RED_KIT, 60.0) is True
    assert kit_compatible(RED_KIT, None, 60.0) is True
    assert kit_compatible(None, None, 60.0) is True


def test_kit_compatible_rejects_clearly_different_kits():
    assert kit_compatible(RED_KIT, RED_KIT, 60.0) is True
    assert kit_compatible(RED_KIT, BLUE_KIT, 60.0) is False


def test_combine_confidence_motion_carries_weak_appearance_inside_orbital():
    cfg = _cfg(motion_carry_floor=0.70)
    conf, carried = combine_confidence(0.40, 1.0, cfg, in_chain=True)
    assert carried is True
    assert conf == pytest.approx(0.70)
    # Product alone would have been 0.40.
    assert conf > 0.40


def test_combine_confidence_no_carry_outside_orbital_or_off_chain():
    cfg = _cfg(motion_carry_floor=0.70)
    conf, carried = combine_confidence(0.40, 0.5, cfg, in_chain=True)
    assert carried is False
    assert conf == pytest.approx(0.20)  # 0.4 * 0.5
    conf2, carried2 = combine_confidence(0.40, 1.0, cfg, in_chain=False)
    assert carried2 is False
    assert conf2 == pytest.approx(0.40)


def test_combine_confidence_strong_appearance_not_capped_down():
    """Motion carry is a floor, never a ceiling — strong app still wins."""
    cfg = _cfg(motion_carry_floor=0.70)
    conf, carried = combine_confidence(0.95, 1.0, cfg, in_chain=True)
    assert carried is False
    assert conf == pytest.approx(0.95)


def test_combine_confidence_never_hard_rejects():
    cfg = _cfg(motion_carry_floor=0.70, appearance_default=0.5)
    conf, _ = combine_confidence(None, cfg.orbital_floor, cfg, in_chain=False)
    assert conf > 0.0
    assert conf == pytest.approx(0.5 * cfg.orbital_floor)


# ---------------------------------------------------------------------------
# Tracklet continuity across the old 3s cut
# ---------------------------------------------------------------------------

def test_soft_rejoin_keeps_tracklet_across_gap_inside_orbital():
    """Gap of 5s (> tracklet_max=3) but still inside growing orbital → same id."""
    cfg = _cfg()
    # radius at 5s = 80 + 60*5 = 380px; displace only 40px → inside.
    c0, crop0 = _contact(0.0, 100, 100), _crop("a0")
    c1, crop1 = _contact(5.0, 140, 100), _crop("a1")
    app = FakeAppearance({id(crop0): 0.9, id(crop1): 0.45})  # weak 2nd read
    scored = score_contacts([c0, c1], [crop0, crop1], _seed(), cfg, appearance=app)
    assert scored[0].tracklet_id == scored[1].tracklet_id
    assert scored[0].anchored is True
    assert scored[1].motion_carried is True
    assert scored[1].confidence >= cfg.motion_carry_floor


def test_hard_break_past_continuity_cap_starts_new_tracklet():
    cfg = _cfg(continuity_max_gap_sec=8.0)
    c0, crop0 = _contact(0.0, 100, 100), _crop("b0")
    c1, crop1 = _contact(10.0, 120, 100), _crop("b1")  # 10s > 8s cap
    app = FakeAppearance({id(crop0): 0.9, id(crop1): 0.9})
    scored = score_contacts([c0, c1], [crop0, crop1], _seed(), cfg, appearance=app)
    assert scored[1].tracklet_id != scored[0].tracklet_id
    assert scored[1].motion_carried is False


def test_teleport_outside_orbital_breaks_even_within_continuity_cap():
    """5s gap but 600px away — outside orbital → hard break, not soft rejoin."""
    cfg = _cfg()
    c0, crop0 = _contact(0.0, 100, 100), _crop("c0")
    c1, crop1 = _contact(5.0, 700, 100), _crop("c1")
    app = FakeAppearance({id(crop0): 0.9, id(crop1): 0.9})
    scored = score_contacts([c0, c1], [crop0, crop1], _seed(), cfg, appearance=app)
    assert scored[1].tracklet_id != scored[0].tracklet_id
    assert scored[1].orbital_prior < 1.0 or scored[1].motion_carried is False


def test_colour_lock_blocks_cross_kit_soft_rejoin():
    """Same position/time gap that would soft-rejoin, but blue kit → new chain."""
    cfg = _cfg()
    c0, crop0 = _contact(0.0, 100, 100, jersey_hsv=RED_KIT), _crop("d0")
    c1, crop1 = _contact(5.0, 140, 100, jersey_hsv=BLUE_KIT), _crop("d1")
    app = FakeAppearance({id(crop0): 0.9, id(crop1): 0.9})
    scored = score_contacts([c0, c1], [crop0, crop1], _seed(), cfg, appearance=app)
    assert scored[1].tracklet_id != scored[0].tracklet_id
    assert scored[1].motion_carried is False


def test_unreadable_colour_does_not_break_soft_rejoin():
    """Missing jersey HSV must not snap the chain (recall-safe colour lock)."""
    cfg = _cfg()
    c0 = _contact(0.0, 100, 100, jersey_hsv=RED_KIT)
    c1 = _contact(5.0, 140, 100, jersey_hsv=None)
    c1.jersey_hsv = None
    crop0, crop1 = _crop("e0"), _crop("e1")
    app = FakeAppearance({id(crop0): 0.9, id(crop1): 0.40})
    scored = score_contacts([c0, c1], [crop0, crop1], _seed(), cfg, appearance=app)
    assert scored[0].tracklet_id == scored[1].tracklet_id
    assert scored[1].motion_carried is True


# ---------------------------------------------------------------------------
# Same-kit chokepoint: motion must beat appearance when appearance is useless
# ---------------------------------------------------------------------------

def test_same_kit_motion_chain_outranks_teleporting_teammate():
    """THE chokepoint: identical appearance scores for you + teammate.

    Anchor on you at t=0. At t=1: you stay near (motion-carried) vs a teammate
    who "touches" 500px away with the same kit / same appearance. Motion must
    rank the near contact higher — appearance alone cannot.
    """
    cfg = _cfg()
    you0, crop0 = _contact(0.0, 100, 100), _crop("you0")
    you1, crop_you = _contact(1.0, 130, 100), _crop("you1")
    mate, crop_mate = _contact(1.0, 600, 100), _crop("mate")
    # Identical weak-ish appearance for both later contacts — colour can't tell.
    app = FakeAppearance({
        id(crop0): 0.92,
        id(crop_you): 0.55,
        id(crop_mate): 0.55,
    })
    scored = score_contacts(
        [you0, you1, mate], [crop0, crop_you, crop_mate], _seed(), cfg,
        appearance=app,
    )
    by_x = {round(s.contact.candidate.x): s for s in scored}
    assert by_x[100].anchored is True
    assert by_x[130].confidence > by_x[600].confidence
    assert by_x[130].motion_carried is True
    assert by_x[600].motion_carried is False
    assert by_x[600].orbital_prior < 1.0


def test_same_kit_identical_gallery_scores_still_separated_by_orbital():
    """Even with perfect identical appearance (1.0), orbital still tie-breaks."""
    cfg = _cfg()
    a0, c0 = _contact(0.0, 100, 100), _crop("s0")
    near, cn = _contact(1.0, 120, 100), _crop("sn")
    far, cf = _contact(1.0, 500, 100), _crop("sf")
    app = FakeAppearance({id(c0): 1.0, id(cn): 1.0, id(cf): 1.0})
    scored = score_contacts([a0, near, far], [c0, cn, cf], _seed(), cfg, appearance=app)
    by_x = {round(s.contact.candidate.x): s for s in scored}
    assert by_x[120].confidence > by_x[500].confidence
    assert by_x[500].orbital_prior >= cfg.orbital_floor


def test_motion_carry_across_multi_touch_possession_chain():
    """Several weak-appearance touches in a row along a path stay one chain and
    stay motion-carried — the 3s cut must not reset mid-possession."""
    cfg = _cfg()
    times = [0.0, 1.5, 3.5, 5.5]  # gaps 1.5, 2.0, 2.0 — last pair crosses 3s from start
    xs = [100.0, 130.0, 160.0, 190.0]
    contacts, crops = [], []
    scores_map: dict[int, float | None] = {}
    for i, (t, x) in enumerate(zip(times, xs)):
        contacts.append(_contact(t, x, 100))
        crop = _crop(f"m{i}")
        crops.append(crop)
        scores_map[id(crop)] = 0.95 if i == 0 else 0.42  # only first is strong
    app = FakeAppearance(scores_map)
    scored = score_contacts(contacts, crops, _seed(), cfg, appearance=app)
    tids = {s.tracklet_id for s in scored}
    assert len(tids) == 1
    assert scored[0].anchored is True
    assert all(s.motion_carried for s in scored[1:])
    assert all(s.confidence >= cfg.motion_carry_floor for s in scored[1:])


def test_teammate_cannot_steal_anchor_mid_chain_without_being_inside_orbital():
    """Strong appearance on a far teammate mid-chain must NOT re-anchor."""
    cfg = _cfg()
    you, cy = _contact(0.0, 100, 100), _crop("t0")
    mate, cm = _contact(1.0, 600, 100), _crop("t1")
    app = FakeAppearance({id(cy): 0.9, id(cm): 0.99})  # teammate "looks" stronger
    scored = score_contacts([you, mate], [cy, cm], _seed(), cfg, appearance=app)
    assert scored[0].anchored is True
    assert scored[1].anchored is False  # prior < 1 → blocked
    assert scored[1].orbital_prior < 1.0


def test_cross_kit_strong_appearance_cannot_steal_anchor_mid_chain():
    cfg = _cfg()
    you, cy = _contact(0.0, 100, 100, jersey_hsv=RED_KIT), _crop("k0")
    # Near enough to be inside orbital, but blue kit.
    other, co = _contact(1.0, 130, 100, jersey_hsv=BLUE_KIT), _crop("k1")
    app = FakeAppearance({id(cy): 0.9, id(co): 0.99})
    scored = score_contacts([you, other], [cy, co], _seed(), cfg, appearance=app)
    assert scored[0].anchored is True
    assert scored[1].anchored is False  # colour-lock blocks


# ---------------------------------------------------------------------------
# Degradation / safety
# ---------------------------------------------------------------------------

def test_past_orbital_max_gap_prior_is_neutral_no_carry():
    cfg = _cfg(orbital_max_gap_sec=8.0, continuity_max_gap_sec=8.0)
    # Exactly at the edge of continuity: 8.1s hard-breaks; use 7.5s near-edge
    # with a huge teleport so prior hits neutral via dt > max? Actually prior
    # returns 1.0 (neutral) when dt > orbital_max_gap. Force that with dt=9
    # and continuity_cap=12 so tracklet logic isn't what clears the anchor.
    cfg = _cfg(orbital_max_gap_sec=8.0, continuity_max_gap_sec=12.0)
    c0, crop0 = _contact(0.0, 100, 100), _crop("n0")
    c1, crop1 = _contact(9.0, 110, 100), _crop("n1")
    app = FakeAppearance({id(crop0): 0.9, id(crop1): 0.4})
    scored = score_contacts([c0, c1], [crop0, crop1], _seed(), cfg, appearance=app)
    # dt > orbital_max → prior neutral 1.0, but hard_break because
    # gap > tracklet_max and not (inside_orbital): inside_orbital requires
    # dt_anchor <= orbital_max_gap, so hard_break True → no carry.
    assert scored[1].motion_carried is False
    assert scored[1].orbital_prior == 1.0  # neutral, not floored


def test_confidence_always_in_unit_interval_across_adversarial_layout():
    cfg = _cfg()
    contacts, crops, smap = [], [], {}
    layout = [
        (0.0, 100, 100, 0.95),
        (0.5, 700, 50, 0.2),
        (2.0, 120, 110, None),
        (6.0, 150, 100, 0.41),
        (15.0, 400, 200, 0.99),
    ]
    for i, (t, x, y, sc) in enumerate(layout):
        contacts.append(_contact(t, x, y))
        crop = _crop(f"adv{i}")
        crops.append(crop)
        smap[id(crop)] = sc
    app = FakeAppearance(smap)
    scored = score_contacts(contacts, crops, _seed(), cfg, appearance=app)
    for s in scored:
        assert 0.0 <= s.confidence <= 1.0
        assert cfg.orbital_floor <= s.orbital_prior <= 1.0 or s.orbital_prior == 1.0


def test_orbital_radius_growth_allows_rejoin_at_long_gap_near_player():
    """At dt=6s radius=80+360=440; a 200px move must still soft-rejoin."""
    cfg = _cfg()
    assert orbital_prior(200.0, 6.0, cfg) == 1.0
    c0, crop0 = _contact(0.0, 100, 100), _crop("r0")
    c1, crop1 = _contact(6.0, 300, 100), _crop("r1")
    app = FakeAppearance({id(crop0): 0.9, id(crop1): 0.35})
    scored = score_contacts([c0, c1], [crop0, crop1], _seed(), cfg, appearance=app)
    assert scored[0].tracklet_id == scored[1].tracklet_id
    assert scored[1].motion_carried is True


def test_default_histogram_path_still_runs_end_to_end():
    """Sanity: real HistogramAppearance path doesn't crash with continuity on."""
    from polyfut_v2.pipeline.seed import build_seed_from_torso_crops

    red = np.full((40, 30, 3), (0, 0, 200), np.uint8)
    seed = build_seed_from_torso_crops([red, red.copy(), red.copy()])
    contacts = [_contact(0.0, 100, 100), _contact(1.0, 130, 100), _contact(5.0, 160, 100)]
    crops = [red.copy(), red.copy(), red.copy()]
    scored = score_contacts(contacts, crops, seed, _cfg())
    assert len(scored) == 3
    assert scored[0].tracklet_id == scored[2].tracklet_id


def test_seed_sighting_links_near_contact_not_far_teammate():
    """Orbital bootstraps from seed taps: near-you links, far same-kit does not."""
    from polyfut_v2.pipeline.seed import TargetSeed

    seed = TargetSeed(
        kit_hsv=np.array([0.0, 200.0, 200.0], np.float32),
        gallery=[],
        n_samples=4,
        sightings=[(100.0, 100.0, 0.0)],
    )
    near = _contact(1.0, 120, 100)   # inside orbital of seed
    far = _contact(1.5, 500, 100)    # teleport — not you
    crop_n, crop_f = _crop("near"), _crop("far")
    app = FakeAppearance({id(crop_n): 0.95, id(crop_f): 0.95})
    scored = score_contacts(
        [near, far], [crop_n, crop_f], seed, _cfg(), appearance=app,
    )
    by_x = {s.contact.candidate.x: s for s in scored}
    assert by_x[120].identity_linked is True
    assert by_x[500].identity_linked is False


def test_cold_start_strong_appearance_is_not_identity_linked():
    """Without a seed sighting / prior chain, high appearance alone is not 'you'."""
    c0, crop0 = _contact(0.0, 100, 100), _crop("solo")
    app = FakeAppearance({id(crop0): 0.95})
    scored = score_contacts([c0], [crop0], _seed(), _cfg(), appearance=app)
    assert scored[0].anchored is True
    assert scored[0].identity_linked is False
