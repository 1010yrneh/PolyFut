"""Crowded contacts (corner kicks / goalmouth scrambles).

When several bodies are packed around the ball, "nearest player" attribution
and the kit colour read off it are both unreliable. These tests pin the
contract: such a touch is flagged, never dropped by the team gate, never
auto-hidden, and always reaches the human review queue — while ordinary 1v1
touches keep their existing behaviour.
"""

import numpy as np

from polyfut_v2.config import PipelineV2Config
from polyfut_v2.pipeline.contacts import ContactCandidate
from polyfut_v2.pipeline.montage import build_montage, review_queue
from polyfut_v2.pipeline.player_contacts import (
    TEAM_OPPONENT,
    TEAM_UNKNOWN,
    count_contesting_players,
    enrich_contacts,
    filter_my_team,
)
from polyfut_v2.pipeline.player_detector import PlayerDetection
from polyfut_v2.pipeline.scoring import ScoredContact
from polyfut_v2.pipeline.seed import build_seed_from_torso_crops

CFG = PipelineV2Config()
RED = (0, 0, 200)
BLUE = (200, 0, 0)


def _frame(color=RED):
    return np.full((300, 300, 3), color, dtype=np.uint8)


def _cand(x=150, y=150):
    return ContactCandidate(frame_index=30, t_sec=3.0, processed_sec=3.0, x=x, y=y,
                            kinds=["kick"], strength=0.8)


def _player(cx, cy, *, h=40, w=16, conf=0.9, class_id=0):
    return PlayerDetection(
        bbox=[cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], conf=conf,
        class_id=class_id,
    )


class _Provider:
    def __init__(self, color=RED):
        self.color = color

    def window(self, center_index, radius, step):
        step = max(1, step)
        return [(i, _frame(self.color))
                for i in range(center_index - radius, center_index + radius + 1, step)]


class _Detector:
    """Returns a fixed set of detections on every frame."""

    def __init__(self, dets):
        self.dets = dets

    def detect(self, frame, near=None):
        return list(self.dets)


def _seed(color=RED):
    return build_seed_from_torso_crops([_frame(color) for _ in range(3)])


# --- counting ---------------------------------------------------------------

def test_count_contesting_players_only_counts_bodies_inside_the_radius():
    players = [_player(150, 150), _player(170, 150), _player(400, 150)]
    assert count_contesting_players(players, (150, 150), 90.0) == 2


def test_count_contesting_players_skips_ineligible_classes():
    players = [_player(150, 150, class_id=2), _player(160, 150, class_id=3)]
    n = count_contesting_players(
        players, (150, 150), 90.0, contact_class_ids={2},
    )
    assert n == 1


def test_count_contesting_players_skips_the_ball_misread_as_a_player():
    # A short, square box sitting exactly on the ball is the ball, not a body.
    players = [_player(150, 150), _player(150, 150, h=10, w=10)]
    n = count_contesting_players(
        players, (150, 150), 90.0,
        min_height_px=CFG.player_min_height_px, min_aspect=CFG.player_min_aspect,
        ball_bbox=[142, 142, 158, 158],
    )
    assert n == 1


# --- enrichment flags -------------------------------------------------------

def _enrich(players, cfg=CFG, seed_color=RED, frame_color=RED):
    return enrich_contacts(
        [_cand()], _Provider(frame_color), _Detector(players),
        _seed(seed_color), cfg,
    )[0]


def test_ordinary_duel_is_not_flagged_crowded():
    c = _enrich([_player(150, 150), _player(180, 150)])
    assert c.n_nearby_players == 2
    assert c.crowded is False


def test_corner_kick_pack_is_flagged_crowded():
    pack = [_player(150 + dx, 150 + dy)
            for dx, dy in [(0, 0), (20, 5), (-18, 8), (30, -10), (-30, -6)]]
    c = _enrich(pack)
    assert c.n_nearby_players == 5
    assert c.crowded is True


def test_crowd_detection_can_be_disabled():
    pack = [_player(150 + 20 * i, 150) for i in range(5)]
    c = _enrich(pack, cfg=PipelineV2Config(crowd_detect_enabled=False))
    assert c.crowded is False and c.n_nearby_players == 0


def test_crowded_flag_survives_serialization():
    pack = [_player(150 + 15 * i, 150) for i in range(4)]
    d = _enrich(pack).to_dict()
    assert d["crowded"] is True and d["n_nearby_players"] == 4


# --- the team gate must not drop a crowded touch ----------------------------

def _contact(*, crowded, team_label=TEAM_OPPONENT, is_my_team=False):
    from polyfut_v2.pipeline.player_contacts import PlayerContact

    return PlayerContact(
        candidate=_cand(), player_bbox=[0, 0, 10, 40], player_dist_px=0.0,
        jersey_hsv=None, color_dist=None, is_my_team=is_my_team,
        n_color_samples=0, team_label=team_label, crowded=crowded,
    )


def test_filter_my_team_keeps_a_crowded_opponent_read():
    kept = filter_my_team(
        [_contact(crowded=True), _contact(crowded=False)], enabled=True,
    )
    assert len(kept) == 1 and kept[0].crowded is True


def test_filter_my_team_still_drops_a_clean_opponent_touch():
    kept = filter_my_team([_contact(crowded=False)], enabled=True)
    assert kept == []


def test_orchestrator_gate_keeps_crowded_opponents_but_drops_clean_ones():
    from polyfut_v2.orchestrator import assemble_touches  # noqa: F401  (import sanity)

    # Mirror the orchestrator's pre-filter predicate on a mixed batch.
    contacts = [_contact(crowded=True), _contact(crowded=False),
                _contact(crowded=False, is_my_team=None, team_label=TEAM_UNKNOWN)]
    keep_crowded = CFG.crowd_keep_other_team
    kept = [c for c in contacts
            if c.is_my_team is not False or (keep_crowded and c.crowded)]
    assert len(kept) == 2


# --- montage: crowded touches always reach the human ------------------------

def _scored(conf, *, crowded, t=1.0, identity_linked=None):
    from polyfut_v2.pipeline.player_contacts import PlayerContact

    cand = ContactCandidate(int(t * 10), t, t, 100, 100, ["kick"], 0.8)
    pc = PlayerContact(cand, [95, 95, 105, 105], 0.0, None, None, None, 0,
                       crowded=crowded, n_nearby_players=5 if crowded else 1)
    if identity_linked is None:
        identity_linked = conf >= CFG.autoaccept_conf
    return ScoredContact(pc, appearance_score=conf, orbital_prior=1.0,
                         confidence=conf, tracklet_id=0, anchored=False,
                         identity_linked=identity_linked)


def test_low_confidence_crowded_touch_is_reviewed_not_auto_hidden():
    m = build_montage([_scored(0.05, crowded=True), _scored(0.05, crowded=False)], CFG)
    by_crowd = {it.to_dict()["crowded"]: it for it in m}
    assert by_crowd[True].status == "review"
    assert by_crowd[False].status == "auto_hide"


def test_high_confidence_crowded_touch_is_reviewed_awaits_user():
    it = build_montage([_scored(0.95, crowded=True, identity_linked=False)], CFG)[0]
    assert it.status == "review"
    # Unlinked crowded touches do not pre-fill "me" — that used to create false
    # hotspots. The user must tap.
    assert it.decision is None


def test_crowded_touches_are_exempt_from_the_review_cap():
    cfg = PipelineV2Config(max_review=2, crowd_max_review=40)
    scored = ([_scored(0.5 - i * 0.001, crowded=False, t=i) for i in range(10)]
              + [_scored(0.4 - i * 0.001, crowded=True, t=100 + i) for i in range(5)])
    m = build_montage(scored, cfg)
    queue = review_queue(m)
    n_crowded = sum(1 for it in queue if it.to_dict()["crowded"])
    n_plain = len(queue) - n_crowded
    assert n_plain == 2      # ordinary queue still capped
    assert n_crowded == 5    # every crowded touch survives despite lower conf


def test_crowded_review_has_its_own_budget():
    cfg = PipelineV2Config(max_review=0, crowd_max_review=3)
    m = build_montage([_scored(0.5, crowded=True, t=i) for i in range(6)], cfg)
    assert len(review_queue(m)) == 3


def test_over_budget_crowded_touch_falls_back_to_its_confidence_verdict():
    cfg = PipelineV2Config(crowd_max_review=1)
    m = build_montage([_scored(0.95, crowded=True, t=1, identity_linked=True),
                       _scored(0.9, crowded=True, t=2, identity_linked=True)], cfg)
    assert m[0].status == "review"          # inside the budget
    assert m[1].status == "auto_accept"     # over it, linked + high conf
    assert m[1].decision == "me"


def test_ordinary_montage_behaviour_is_unchanged_without_crowding():
    m = build_montage([_scored(0.95, crowded=False), _scored(0.5, crowded=False),
                       _scored(0.05, crowded=False)], CFG)
    assert [it.status for it in m] == ["auto_accept", "review", "auto_hide"]
