"""Tests for Stages 5+6 wiring: contacting player + team-colour decision."""

import numpy as np

from polyfut_v2.config import PipelineV2Config
from polyfut_v2.pipeline.contacts import ContactCandidate
from polyfut_v2.pipeline.player_contacts import (
    enrich_contacts,
    filter_my_team,
    nearest_player,
    point_to_bbox_dist,
)
from polyfut_v2.pipeline.player_detector import PlayerDetection
from polyfut_v2.pipeline.seed import build_seed_from_torso_crops

RED = (0, 0, 200)
BLUE = (200, 0, 0)
BOX = [10, 10, 90, 90]


def _frame(color):
    return np.full((100, 100, 3), color, dtype=np.uint8)


def _cand(x=50, y=80):
    return ContactCandidate(frame_index=30, t_sec=3.0, processed_sec=3.0, x=x, y=y,
                            kinds=["kick"], strength=0.8)


class FakeProvider:
    def __init__(self, color, n=3):
        self.color = color
        self.n = n
    def window(self, center_index, radius, step):
        return [(center_index + i, _frame(self.color)) for i in range(self.n)]


class FakePlayerDetector:
    """Returns a single player at ``bbox`` (default BOX), or nothing if empty."""
    def __init__(self, empty=False, bbox=BOX):
        self.empty = empty
        self.bbox = bbox
    def detect(self, frame, near=None):
        return [] if self.empty else [PlayerDetection(bbox=list(self.bbox), conf=0.9)]


def _seed(color):
    return build_seed_from_torso_crops([_frame(color) for _ in range(3)])


def test_point_to_bbox_dist_inside_is_zero():
    assert point_to_bbox_dist((50, 50), [10, 10, 90, 90]) == 0.0
    assert point_to_bbox_dist((100, 50), [10, 10, 90, 90]) == 10.0


def test_nearest_player_respects_max_dist():
    players = [PlayerDetection([0, 0, 20, 20], 0.9), PlayerDetection([200, 200, 220, 220], 0.8)]
    pl, d = nearest_player(players, (10, 10), max_dist=50)
    assert pl is players[0] and d == 0.0
    pl2, _ = nearest_player(players, (500, 500), max_dist=50)
    assert pl2 is None


def test_same_kit_contact_is_my_team():
    cfg = PipelineV2Config()
    contacts = enrich_contacts([_cand()], FakeProvider(RED), FakePlayerDetector(),
                               _seed(RED), cfg)
    c = contacts[0]
    assert c.is_my_team is True
    assert c.player_bbox == BOX
    assert c.n_color_samples == 3
    assert c.color_dist is not None and c.color_dist <= cfg.team_color_max_dist


def test_opponent_kit_contact_dropped():
    cfg = PipelineV2Config()
    contacts = enrich_contacts([_cand()], FakeProvider(BLUE), FakePlayerDetector(),
                               _seed(RED), cfg)
    assert contacts[0].is_my_team is False
    kept = filter_my_team(contacts)
    assert kept == []


def test_no_player_is_undecided_and_kept():
    cfg = PipelineV2Config()
    contacts = enrich_contacts([_cand()], FakeProvider(RED), FakePlayerDetector(empty=True),
                               _seed(RED), cfg)
    c = contacts[0]
    assert c.is_my_team is None       # no colour measured → undecided
    assert c.player_bbox is None
    assert c.n_color_samples == 0
    # Undecided kept by default (recall), droppable when strict.
    assert len(filter_my_team(contacts, keep_undecided=True)) == 1
    assert len(filter_my_team(contacts, keep_undecided=False)) == 0


def test_filter_keeps_only_my_team():
    cfg = PipelineV2Config()
    mine = enrich_contacts([_cand()], FakeProvider(RED), FakePlayerDetector(), _seed(RED), cfg)
    opp = enrich_contacts([_cand()], FakeProvider(BLUE), FakePlayerDetector(), _seed(RED), cfg)
    kept = filter_my_team(mine + opp)
    assert len(kept) == 1
    assert kept[0].is_my_team is True


def test_tiny_torso_crop_is_undecided_not_dropped():
    # A tiny player box (wide-footage grass contamination) → colour unmeasurable,
    # so the contact stays undecided and is kept, never confidently dropped.
    cfg = PipelineV2Config()
    tiny = FakePlayerDetector(bbox=[10, 10, 20, 25])  # torso ~6x6 = 36px < 50
    contacts = enrich_contacts([_cand(x=15, y=20)], FakeProvider(BLUE), tiny, _seed(RED), cfg)
    c = contacts[0]
    assert c.player_bbox == [10, 10, 20, 25]  # player still detected (Stage 5)
    assert c.jersey_hsv is None               # colour rejected as too small (Stage 6)
    assert c.is_my_team is None               # undecided, not False
    assert len(filter_my_team(contacts)) == 1  # kept


def test_filter_disabled_keeps_everything():
    cfg = PipelineV2Config()
    opp = enrich_contacts([_cand()], FakeProvider(BLUE), FakePlayerDetector(), _seed(RED), cfg)
    assert opp[0].is_my_team is False
    # Filter on → opponent dropped; filter off → kept (footage where colour fails).
    assert filter_my_team(opp, enabled=True) == []
    assert len(filter_my_team(opp, enabled=False)) == 1


def test_disabled_filter_leaves_team_undecided_but_records_color():
    # With the filter off, even an opponent-coloured contact is left undecided
    # (so it can't be dropped or de-anchored), but color_dist is still measured.
    cfg = PipelineV2Config(team_filter_enabled=False)
    opp = enrich_contacts([_cand()], FakeProvider(BLUE), FakePlayerDetector(), _seed(RED), cfg)
    assert opp[0].is_my_team is None
    assert opp[0].color_dist is not None


def test_to_dict_shape():
    cfg = PipelineV2Config()
    c = enrich_contacts([_cand()], FakeProvider(RED), FakePlayerDetector(), _seed(RED), cfg)[0]
    d = c.to_dict()
    # Carries both the candidate fields and the enrichment.
    assert d["kinds"] == ["kick"] and d["is_my_team"] is True
    assert "player_bbox" in d and "color_dist" in d
