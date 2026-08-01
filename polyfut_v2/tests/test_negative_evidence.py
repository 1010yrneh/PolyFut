"""Saying "not you" needs evidence, not just a low number.

Measured on a real 5-minute run (job ``fdc541cf493e``): 73 of 88 contacts went
to review and **zero** were ever auto-hidden. That is arithmetic, not bad luck —
confidence is ``appearance x orbital``, both are floored at 0.5
(``appearance_default``, ``orbital_floor``), so the minimum reachable confidence
is 0.25 while ``autohide_conf`` is 0.15. The observed minimum across all 88
contacts was exactly 0.250.

So the pipeline could say "definitely you" or "I don't know", never "definitely
not you", and every uncertain touch drained to the human.

These tests pin the fix and, more importantly, its limit: a contact is hidden
only on something we *measured*, never on something we failed to measure.
"""

from __future__ import annotations

import numpy as np

from polyfut_v2.config import PipelineV2Config
from polyfut_v2.pipeline.contacts import ContactCandidate
from polyfut_v2.pipeline.montage import AUTO_HIDE, REVIEW, _status
from polyfut_v2.pipeline.player_contacts import PlayerContact
from polyfut_v2.pipeline.scoring import ScoredContact, negative_evidence_for

CFG = PipelineV2Config()


def _cand():
    return ContactCandidate(frame_index=30, t_sec=3.0, processed_sec=3.0,
                            x=150.0, y=150.0, kinds=["kick"], strength=0.8)


def _scored(**kw):
    base = dict(
        contact=PlayerContact(
            candidate=_cand(), player_bbox=[140.0, 130.0, 160.0, 175.0],
            player_dist_px=5.0, jersey_hsv=None, color_dist=None,
            is_my_team=None, n_color_samples=0),
        appearance_score=None, orbital_prior=1.0, confidence=0.5,
        tracklet_id=0, anchored=False,
    )
    base.update(kw)
    return ScoredContact(**base)


# ------------------------------------------------------- the arithmetic itself
def test_the_autohide_threshold_really_is_unreachable():
    """Pins the bug: the floors cannot produce a number under the threshold."""
    floor = CFG.appearance_default * CFG.orbital_floor
    assert floor == 0.25
    assert floor > CFG.autohide_conf, (
        "if this ever fails the arithmetic changed and the evidence path "
        "may no longer be needed")


# ------------------------------------------------------------------- evidence
def test_unmeasurable_appearance_is_never_evidence():
    """The whole recall-safety rule in one test.

    A crop too small or too grass-contaminated to read gives ``app=None``. That
    is ignorance, not a mismatch, and must never hide a touch.
    """
    assert negative_evidence_for(
        None, 1.0, had_anchor=True, dt_anchor=0.5, is_my_team=None, cfg=CFG,
    ) == ()


def test_a_measured_mismatch_is_evidence():
    ev = negative_evidence_for(
        0.10, 1.0, had_anchor=False, dt_anchor=0.0, is_my_team=None, cfg=CFG)
    assert "appearance_mismatch" in ev


def test_a_merely_unconvincing_appearance_is_not_evidence():
    """Between reject_max and anchor_min is "not clearly right" — still review."""
    ev = negative_evidence_for(
        0.5, 1.0, had_anchor=False, dt_anchor=0.0, is_my_team=None, cfg=CFG)
    assert "appearance_mismatch" not in ev
    assert CFG.appearance_reject_max < 0.5 < CFG.orbital_anchor_min


def test_being_demonstrably_elsewhere_is_evidence():
    """A live anchor plus a floored prior means the seed said you were away."""
    ev = negative_evidence_for(
        None, CFG.orbital_floor, had_anchor=True, dt_anchor=1.0,
        is_my_team=None, cfg=CFG)
    assert "outside_orbital" in ev


def test_a_stale_anchor_is_not_evidence():
    """Past ``orbital_max_gap_sec`` the orbital covers the pitch and means
    nothing — it must not be read as proof you were elsewhere."""
    ev = negative_evidence_for(
        None, CFG.orbital_floor, had_anchor=True,
        dt_anchor=CFG.orbital_max_gap_sec + 1.0, is_my_team=None, cfg=CFG)
    assert "outside_orbital" not in ev


def test_no_anchor_at_all_is_not_evidence():
    ev = negative_evidence_for(
        None, CFG.orbital_floor, had_anchor=False, dt_anchor=0.0,
        is_my_team=None, cfg=CFG)
    assert ev == ()


# ---------------------------------------------------------------- the verdict
def test_evidence_hides_where_the_threshold_could_not():
    s = _scored(confidence=0.5, negative_evidence=("appearance_mismatch",))
    assert _status(s, CFG) == AUTO_HIDE


def test_no_evidence_still_goes_to_a_human():
    s = _scored(confidence=0.5, negative_evidence=())
    assert _status(s, CFG) == REVIEW


def test_the_experiment_can_be_switched_off_completely():
    cfg = PipelineV2Config(autohide_on_evidence=False)
    s = _scored(confidence=0.5, negative_evidence=("appearance_mismatch",))
    assert _status(s, cfg) == REVIEW


def test_evidence_never_overrides_a_confident_linked_accept():
    s = _scored(confidence=0.95, identity_linked=True,
                negative_evidence=("appearance_mismatch",))
    assert _status(s, CFG) != AUTO_HIDE


def test_the_reason_is_recorded_for_anything_hidden():
    """A vanished touch must be explainable after the fact."""
    s = _scored(confidence=0.5, negative_evidence=("outside_orbital",))
    assert "outside_orbital" in s.to_dict()["negative_evidence"]
