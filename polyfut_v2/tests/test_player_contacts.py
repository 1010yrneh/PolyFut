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
GRASS = (40, 130, 40)   # BGR → falls in the pitch-grass hue band, gets masked out
BOX = [10, 10, 90, 90]

# The colour filter is OFF by default (amateur footage); tests of the colour
# decision opt in explicitly.
FILTER_ON = PipelineV2Config(team_filter_enabled=True)


def _frame(color):
    return np.full((100, 100, 3), color, dtype=np.uint8)


def _cand(x=50, y=80):
    return ContactCandidate(frame_index=30, t_sec=3.0, processed_sec=3.0, x=x, y=y,
                            kinds=["kick"], strength=0.8)


class FakeProvider:
    """Respects radius/step like the real VideoFrameProvider."""
    def __init__(self, color):
        self.color = color
    def window(self, center_index, radius, step):
        step = max(1, step)
        return [(i, _frame(self.color))
                for i in range(center_index - radius, center_index + radius + 1, step)]


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
    contacts = enrich_contacts([_cand()], FakeProvider(RED), FakePlayerDetector(),
                               _seed(RED), FILTER_ON)
    c = contacts[0]
    assert c.is_my_team is True
    assert c.player_bbox == BOX
    assert c.n_color_samples == 3
    assert c.color_dist is not None and c.color_dist <= FILTER_ON.team_color_max_dist
    assert c.torso_crop is not None


def test_opponent_kit_contact_dropped():
    contacts = enrich_contacts([_cand()], FakeProvider(BLUE), FakePlayerDetector(),
                               _seed(RED), FILTER_ON)
    assert contacts[0].is_my_team is False
    assert filter_my_team(contacts) == []


def test_no_player_is_undecided_and_kept():
    contacts = enrich_contacts([_cand()], FakeProvider(RED), FakePlayerDetector(empty=True),
                               _seed(RED), FILTER_ON)
    c = contacts[0]
    assert c.is_my_team is None       # no colour measured → undecided
    assert c.player_bbox is None
    assert c.n_color_samples == 0
    assert len(filter_my_team(contacts, keep_undecided=True)) == 1
    assert len(filter_my_team(contacts, keep_undecided=False)) == 0


def test_filter_keeps_only_my_team():
    mine = enrich_contacts([_cand()], FakeProvider(RED), FakePlayerDetector(), _seed(RED), FILTER_ON)
    opp = enrich_contacts([_cand()], FakeProvider(BLUE), FakePlayerDetector(), _seed(RED), FILTER_ON)
    kept = filter_my_team(mine + opp)
    assert len(kept) == 1
    assert kept[0].is_my_team is True


class CountingDetector:
    """Counts detect() calls to measure Stage 5-6 per-candidate cost."""
    def __init__(self, bbox=BOX):
        self.calls = 0
        self.bbox = bbox
    def detect(self, frame, near=None):
        self.calls += 1
        return [PlayerDetection(bbox=list(self.bbox), conf=0.9)]


def test_filter_on_detects_each_window_frame_once():
    det = CountingDetector()
    contacts = enrich_contacts([_cand()], FakeProvider(RED), det, _seed(RED), FILTER_ON)
    assert det.calls == 3            # window frames, each once (no redundant centre detect)
    assert contacts[0].n_color_samples == 3


def test_filter_off_detects_only_contact_frame_and_captures_crop():
    cfg = PipelineV2Config()          # default: filter OFF
    det = CountingDetector()
    contacts = enrich_contacts([_cand()], FakeProvider(RED), det, _seed(RED), cfg)
    assert det.calls == 1             # only the contact frame — 3x cheaper
    assert contacts[0].torso_crop is not None   # crop captured for Stage 7 (no 2nd pass)
    # Same kit as the seed → clearly your team, even with the aggressive filter off.
    assert contacts[0].is_my_team is True


def test_grass_dominated_crop_is_undecided_not_dropped():
    # A distant player whose torso crop is essentially all pitch grass → colour
    # unmeasurable (grass-masked away) → undecided → kept, never confidently
    # dropped. This is the recall safety net now that the area floor is gone.
    contacts = enrich_contacts([_cand()], FakeProvider(GRASS), FakePlayerDetector(),
                               _seed(RED), FILTER_ON)
    c = contacts[0]
    assert c.player_bbox == BOX               # player still detected (Stage 5)
    assert c.jersey_hsv is None               # all grass → colour unmeasurable
    assert c.is_my_team is None               # undecided, not dropped
    assert len(filter_my_team(contacts)) == 1  # kept


def test_filter_disabled_keeps_everything():
    opp = enrich_contacts([_cand()], FakeProvider(BLUE), FakePlayerDetector(), _seed(RED), FILTER_ON)
    assert opp[0].is_my_team is False
    assert filter_my_team(opp, enabled=True) == []           # on → opponent dropped
    assert len(filter_my_team(opp, enabled=False)) == 1      # off → kept


def test_disabled_filter_still_flags_clearly_other_team():
    # Even with the aggressive filter off, a kit clearly different from the seed
    # (grass-masked colour) is labelled opponent so the orchestrator can drop it —
    # this is what removes "wrong team touched the ball" clips.
    cfg = PipelineV2Config(team_filter_enabled=False)
    opp = enrich_contacts([_cand()], FakeProvider(BLUE), FakePlayerDetector(), _seed(RED), cfg)
    assert opp[0].is_my_team is False
    assert opp[0].color_dist is not None and opp[0].color_dist > cfg.contact_other_team_dist


def test_to_dict_shape():
    c = enrich_contacts([_cand()], FakeProvider(RED), FakePlayerDetector(), _seed(RED), FILTER_ON)[0]
    d = c.to_dict()
    assert d["kinds"] == ["kick"] and d["is_my_team"] is True
    assert "player_bbox" in d and "color_dist" in d
    assert "torso_crop" not in d   # crop is not serialized
