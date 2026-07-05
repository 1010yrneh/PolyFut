"""Tests for Stage 8 review montage."""

import pytest

from polyfut_v2.config import PipelineV2Config
from polyfut_v2.pipeline.contacts import ContactCandidate
from polyfut_v2.pipeline.montage import (
    apply_decisions,
    build_montage,
    confirmed_me_times,
    review_queue,
)
from polyfut_v2.pipeline.player_contacts import PlayerContact
from polyfut_v2.pipeline.scoring import ScoredContact

CFG = PipelineV2Config()  # autoaccept 0.85, autohide 0.15


def _scored(t, conf, x=100, y=100):
    cand = ContactCandidate(int(t * 10), t, t, x, y, ["kick"], 0.8)
    pc = PlayerContact(cand, [x - 5, y - 5, x + 5, y + 5], 0.0, None, None, None, 0)
    return ScoredContact(pc, appearance_score=conf, orbital_prior=1.0,
                         confidence=conf, tracklet_id=0, anchored=False)


def test_ranked_by_confidence_desc():
    m = build_montage([_scored(1, 0.3), _scored(2, 0.9), _scored(3, 0.6)], CFG)
    assert [it.confidence for it in m] == [0.9, 0.6, 0.3]
    assert [it.rank for it in m] == [0, 1, 2]


def test_status_thresholds_and_default_decisions():
    m = build_montage([_scored(1, 0.95), _scored(2, 0.5), _scored(3, 0.05)], CFG)
    by_conf = {it.confidence: it for it in m}
    assert (by_conf[0.95].status, by_conf[0.95].decision) == ("auto_accept", "me")
    assert (by_conf[0.5].status, by_conf[0.5].decision) == ("review", None)
    assert (by_conf[0.05].status, by_conf[0.05].decision) == ("auto_hide", "not_me")


def test_clip_window_and_crop():
    m = build_montage([_scored(10.0, 0.5, x=200, y=150)], CFG, duration_sec=100.0)
    it = m[0]
    assert it.clip_start_sec == 10.0 - CFG.montage_clip_pad_sec
    assert it.clip_end_sec == 10.0 + CFG.montage_clip_pad_sec
    h = CFG.montage_crop_half_px
    assert it.crop == [200 - h, 150 - h, 200 + h, 150 + h]


def test_clip_window_clamped():
    m = build_montage([_scored(0.5, 0.5)], CFG, duration_sec=100.0)
    assert m[0].clip_start_sec == 0.0  # can't go negative


def test_review_queue_only_middle():
    m = build_montage([_scored(1, 0.95), _scored(2, 0.5), _scored(3, 0.05)], CFG)
    rq = review_queue(m)
    assert len(rq) == 1 and rq[0].confidence == 0.5


def test_apply_decisions_and_confirmed_times():
    m = build_montage([_scored(5.0, 0.95), _scored(9.0, 0.5), _scored(3.0, 0.05)], CFG)
    # Auto-accepted 5.0 is already 'me'; user marks the review item (9.0) as 'me'.
    review = review_queue(m)[0]
    apply_decisions(m, {review.rank: "me"})
    assert confirmed_me_times(m) == [5.0, 9.0]


def test_apply_decisions_rejects_bad_value():
    m = build_montage([_scored(1, 0.5)], CFG)
    with pytest.raises(ValueError):
        apply_decisions(m, {0: "maybe"})
