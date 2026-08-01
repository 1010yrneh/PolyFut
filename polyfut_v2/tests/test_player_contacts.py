"""Tests for Stages 5+6 wiring: contacting player + team-colour decision."""

import numpy as np

from polyfut_v2.config import PipelineV2Config
from polyfut_v2.pipeline.contacts import ContactCandidate
from polyfut_v2.pipeline.player_contacts import (
    TEAM_MY,
    TEAM_OFFICIAL,
    TEAM_OPPONENT,
    TEAM_UNKNOWN,
    _closer_to_my_team,
    _closer_to_opponent,
    _looks_like_ball,
    classify_team,
    enrich_contacts,
    filter_my_team,
    nearest_player,
    point_to_bbox_dist,
)
from polyfut_v2.pipeline.player_detector import PlayerDetection
from polyfut_v2.pipeline.seed import TargetSeed, build_seed_from_torso_crops

RED = (0, 0, 200)
BLUE = (200, 0, 0)
GRASS = (40, 130, 40)   # BGR → falls in the pitch-grass hue band, gets masked out
BOX = [30, 10, 70, 90]  # 40×80 person-shaped (aspect 2.0); not a square camera/ball

# The colour filter is OFF by default (amateur footage); tests of the colour
# decision opt in explicitly.
FILTER_ON = PipelineV2Config(team_filter_enabled=True)
SOCCER_CFG = PipelineV2Config(
    team_filter_enabled=True,
    player_class_id=2,
    goalkeeper_class_id=1,
    referee_class_id=3,
)


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
    def __init__(self, empty=False, bbox=BOX, class_id=0, detections=None):
        self.empty = empty
        self.bbox = bbox
        self.class_id = class_id
        self.detections = detections  # optional full list override
    def detect(self, frame, near=None):
        if self.detections is not None:
            return list(self.detections)
        return [] if self.empty else [
            PlayerDetection(bbox=list(self.bbox), conf=0.9, class_id=self.class_id)
        ]


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


def test_looks_like_ball_flags_small_square_box():
    # A 9x11 box (real ball-misclassification size from real broadcast footage).
    assert _looks_like_ball([100, 100, 109, 111], min_height_px=16.0, min_aspect=1.8) is True


def test_looks_like_ball_spares_tall_players():
    # Short but clearly person-shaped (aspect well above 1.8) → not a ball.
    assert _looks_like_ball([100, 100, 106, 130], min_height_px=16.0, min_aspect=1.8) is False


def test_looks_like_ball_spares_tall_short_players():
    # Tall enough on its own (>=16px), even though roughly square → not a ball
    # (a real player can be this compact at a distance; only small AND square
    # should be treated as suspicious).
    assert _looks_like_ball([100, 100, 118, 120], min_height_px=16.0, min_aspect=1.8) is False


def test_looks_like_person_rejects_camera_and_spare_ball():
    from polyfut_v2.pipeline.player_contacts import looks_like_person

    # Near-square spare ball / camera (~18x18) — clears old height floor, fails aspect.
    assert looks_like_person([100, 100, 118, 118], min_height_px=16.0, min_aspect=1.30) is False
    # Wide camera housing.
    assert looks_like_person([100, 100, 140, 118], min_height_px=16.0, min_aspect=1.30) is False
    # Tiny fragment.
    assert looks_like_person([100, 100, 108, 112], min_height_px=16.0, min_aspect=1.30) is False
    # Ultra-thin pole.
    assert looks_like_person(
        [100, 50, 106, 200], min_height_px=16.0, min_aspect=1.30, max_aspect=5.5,
    ) is False


def test_looks_like_person_keeps_standing_player():
    from polyfut_v2.pipeline.player_contacts import looks_like_person

    # Typical 640-space player ~15x30 (aspect 2.0).
    assert looks_like_person([200, 100, 215, 130], min_height_px=16.0, min_aspect=1.30) is True
    # Slightly stocky but still taller-than-wide.
    assert looks_like_person([200, 100, 220, 128], min_height_px=16.0, min_aspect=1.30) is True


def test_nearest_player_skips_camera_shaped_box():
    camera = PlayerDetection([50, 60, 70, 78], 0.99)   # 20x18, aspect ~0.9
    player = PlayerDetection([80, 40, 96, 90], 0.8)     # 16x50, aspect ~3.1
    pl, _ = nearest_player(
        [camera, player], (60, 70), max_dist=80,
        min_height_px=16.0, min_aspect=1.8,
        human_min_aspect=1.30, human_max_aspect=5.5,
    )
    assert pl is player
    pl2, _ = nearest_player(
        [camera], (60, 70), max_dist=80,
        min_height_px=16.0, min_aspect=1.8,
        human_min_aspect=1.30, human_max_aspect=5.5,
    )
    assert pl2 is None


def test_nearest_player_skips_ball_shaped_box():
    """The classic failure this guards against: the soccer model classifies the
    ball itself as "player" — a small square box sitting almost exactly on the
    search point — while the real contacting player stands a little further
    away. Without the ball-shape guard the ball-mimic wins on raw distance."""
    ball_mimic = PlayerDetection([100, 100, 109, 111], 0.5)   # ~9x11, aspect~1.2
    real_player = PlayerDetection([90, 70, 115, 140], 0.7)    # 25x70, aspect~2.8

    # Old behaviour (no shape guard): the ball-mimic wins purely on distance.
    pl_unguarded, _ = nearest_player([ball_mimic, real_player], (104.5, 105.5), max_dist=80)
    assert pl_unguarded is ball_mimic

    # With the guard enabled, the real player is correctly picked instead.
    pl_guarded, _ = nearest_player(
        [ball_mimic, real_player], (104.5, 105.5), max_dist=80,
        min_height_px=16.0, min_aspect=1.8,
    )
    assert pl_guarded is real_player


def test_nearest_player_returns_none_if_only_ball_shaped_box_available():
    ball_mimic = PlayerDetection([100, 100, 109, 111], 0.5)
    pl, _ = nearest_player(
        [ball_mimic], (104.5, 105.5), max_dist=80,
        min_height_px=16.0, min_aspect=1.8,
    )
    assert pl is None


def test_nearest_player_skips_referee_class():
    ref = PlayerDetection([40, 40, 70, 90], 0.9, class_id=3)
    player = PlayerDetection([10, 10, 40, 80], 0.8, class_id=2)
    pl, _ = nearest_player(
        [ref, player], (55, 65), max_dist=80, contact_class_ids={1, 2},
    )
    assert pl is player
    pl_ref_only, _ = nearest_player(
        [ref], (55, 65), max_dist=80, contact_class_ids={1, 2},
    )
    assert pl_ref_only is None


def test_nearest_player_accepts_goalkeeper():
    keeper = PlayerDetection([40, 40, 70, 90], 0.9, class_id=1)
    pl, _ = nearest_player(
        [keeper], (55, 65), max_dist=80, contact_class_ids={1, 2},
    )
    assert pl is keeper


def test_nearest_player_prefers_feet_near_ball_over_large_sideline_box():
    """A huge sideline/graphic box can contain the ball (dist 0) while the real
    contesting player's feet are nearer the ball. Ranking must prefer feet."""
    ball = (100.0, 100.0)
    # Tall sideline blob: ball sits near the top of the box, feet are far below.
    sideline = PlayerDetection([40, 20, 160, 220], 0.95, class_id=2)  # 120x200
    # Compact contesting player: feet near the ball.
    contesting = PlayerDetection([88, 70, 112, 130], 0.80, class_id=2)  # 24x60

    # Both contain / reach the ball within max_dist; unguarded bbox-dist is 0
    # for the sideline box, so a pure nearest-box pick would prefer it.
    assert point_to_bbox_dist(ball, sideline.bbox) == 0.0

    pl, _ = nearest_player(
        [sideline, contesting], ball, max_dist=80,
        min_height_px=16.0, min_aspect=1.8,
    )
    assert pl is contesting


def test_nearest_player_tiebreak_prefers_smaller_higher_conf_when_ball_inside():
    """When several boxes contain the ball, prefer compact + confident."""
    ball = (50.0, 50.0)
    big = PlayerDetection([0, 0, 120, 120], 0.5, class_id=2)
    small = PlayerDetection([40, 30, 60, 80], 0.9, class_id=2)
    pl, _ = nearest_player([big, small], ball, max_dist=80)
    assert pl is small


def test_soft_ball_overlap_rejects_short_or_squat_box():
    """With ball-bbox overlap, failing EITHER height OR aspect is enough to
    reject — the classic AND guard alone would keep an elongated tiny ball."""
    from polyfut_v2.pipeline.player_contacts import ball_proxy_bbox

    ball_pt = (104.5, 105.5)
    ball_box = ball_proxy_bbox(ball_pt, 8.0)
    # Short but elongated (aspect ~2.5) — survives the strict AND, fails soft OR.
    elongated_ball = PlayerDetection([100, 100, 106, 115], 0.6, class_id=2)  # 6x15
    real = PlayerDetection([90, 70, 115, 140], 0.7, class_id=2)

    pl_strict, _ = nearest_player(
        [elongated_ball, real], ball_pt, max_dist=80,
        min_height_px=16.0, min_aspect=1.8,
        # no ball_bbox → strict AND → elongated_ball kept, nearer → wins
    )
    assert pl_strict is elongated_ball

    pl_soft, _ = nearest_player(
        [elongated_ball, real], ball_pt, max_dist=80,
        min_height_px=16.0, min_aspect=1.8,
        ball_bbox=ball_box, ball_iou_soft=0.25,
    )
    assert pl_soft is real


def _two_team_seed(my_hsv, opp_hsv):
    return TargetSeed(kit_hsv=np.asarray(my_hsv, np.float32),
                      opponent_kit_hsv=np.asarray(opp_hsv, np.float32))


# HSV distance is ~hue-dominated: hue 0 (my kit) vs 60 (opponent) are well
# separated (~120), so the two teams are distinguishable by colour.
MY_KIT = [0, 200, 200]
OPP_KIT = [60, 200, 200]


def test_closer_to_opponent_drops_clear_opponent_in_undecided_band():
    cfg = PipelineV2Config()
    seed = _two_team_seed(MY_KIT, OPP_KIT)
    # A contact at hue 40: ~80 from my kit (in the 60-115 undecided band) but
    # only ~40 from the opponent → clearly the other team.
    kit = np.array([40, 200, 200], np.float32)
    assert _closer_to_opponent(kit, my_dist=80.0, seed=seed, cfg=cfg) is True


def test_closer_to_my_team_confirms_mine_in_undecided_band():
    cfg = PipelineV2Config()
    seed = _two_team_seed(MY_KIT, OPP_KIT)
    # Hue 20: ~40 from my kit, ~80 from opponent → nearer mine by margin.
    kit = np.array([20, 200, 200], np.float32)
    my_dist = 40.0
    assert _closer_to_my_team(kit, my_dist=my_dist, seed=seed, cfg=cfg) is True


def test_closer_to_opponent_keeps_near_ties_undecided():
    cfg = PipelineV2Config()
    seed = _two_team_seed(MY_KIT, OPP_KIT)
    # Hue 15: ~90 from my kit, ~90 from opponent → a near tie (within margin),
    # so it must stay undecided (kept for review), not be dropped.
    kit = np.array([15, 200, 200], np.float32)
    assert _closer_to_opponent(kit, my_dist=100.0, seed=seed, cfg=cfg) is False


def test_closer_to_opponent_needs_a_known_opponent_kit():
    cfg = PipelineV2Config()
    seed = TargetSeed(kit_hsv=np.asarray(MY_KIT, np.float32))   # opponent unknown
    kit = np.array([40, 200, 200], np.float32)
    assert _closer_to_opponent(kit, my_dist=80.0, seed=seed, cfg=cfg) is False


def test_closer_to_opponent_disabled_when_kits_too_similar():
    """If the two teams' kits are nearly the same colour, colour can't separate
    them at all — the tie-break must not fire and risk a wrong drop."""
    cfg = PipelineV2Config()
    seed = _two_team_seed([0, 200, 200], [18, 200, 200])   # centroids ~36 apart (< 60)
    kit = np.array([12, 200, 200], np.float32)
    assert _closer_to_opponent(kit, my_dist=80.0, seed=seed, cfg=cfg) is False


def test_classify_team_two_centroid_labels():
    cfg = PipelineV2Config()
    seed = _two_team_seed(MY_KIT, OPP_KIT)
    is_mine, label, dist = classify_team(
        np.array([40, 200, 200], np.float32), seed, cfg,
    )
    assert is_mine is False and label == TEAM_OPPONENT and dist is not None
    is_mine, label, _ = classify_team(
        np.array([10, 200, 200], np.float32), seed, cfg,
    )
    assert is_mine is True and label == TEAM_MY


def test_classify_team_referee_is_official():
    cfg = PipelineV2Config(referee_class_id=3)
    seed = _two_team_seed(MY_KIT, OPP_KIT)
    is_mine, label, dist = classify_team(
        np.array([0, 200, 200], np.float32), seed, cfg, player_class_id=3,
    )
    assert is_mine is False and label == TEAM_OFFICIAL and dist is None


def test_opponent_gate_off_when_no_opponent_kit_matches_old_behaviour():
    """A seed built the old way (no opponent kit) must classify exactly as
    before — the opponent gate is purely additive."""
    mine = enrich_contacts([_cand()], FakeProvider(RED), FakePlayerDetector(), _seed(RED), FILTER_ON)
    assert mine[0].is_my_team is True
    assert mine[0].team_label == TEAM_MY
    grass = enrich_contacts([_cand()], FakeProvider(GRASS), FakePlayerDetector(), _seed(RED), FILTER_ON)
    assert grass[0].is_my_team is None   # unmeasurable → undecided, unchanged
    assert grass[0].team_label == TEAM_UNKNOWN


def test_same_kit_contact_is_my_team():
    contacts = enrich_contacts([_cand()], FakeProvider(RED), FakePlayerDetector(),
                               _seed(RED), FILTER_ON)
    c = contacts[0]
    assert c.is_my_team is True
    assert c.team_label == TEAM_MY
    assert c.player_bbox == BOX
    assert c.n_color_samples == 3
    assert c.color_dist is not None and c.color_dist <= FILTER_ON.team_color_max_dist
    assert c.torso_crop is not None


def test_opponent_kit_contact_dropped():
    contacts = enrich_contacts([_cand()], FakeProvider(BLUE), FakePlayerDetector(),
                               _seed(RED), FILTER_ON)
    assert contacts[0].is_my_team is False
    assert contacts[0].team_label == TEAM_OPPONENT
    assert filter_my_team(contacts) == []


def test_no_player_is_undecided_and_kept():
    contacts = enrich_contacts([_cand()], FakeProvider(RED), FakePlayerDetector(empty=True),
                               _seed(RED), FILTER_ON)
    c = contacts[0]
    assert c.is_my_team is None       # no colour measured → undecided
    assert c.team_label == TEAM_UNKNOWN
    assert c.player_bbox is None
    assert c.n_color_samples == 0
    assert len(filter_my_team(contacts, keep_undecided=True)) == 1
    assert len(filter_my_team(contacts, keep_undecided=False)) == 0


def test_referee_only_contact_is_official_and_dropped():
    ref = PlayerDetection(bbox=list(BOX), conf=0.9, class_id=3)
    contacts = enrich_contacts(
        [_cand()], FakeProvider(RED),
        FakePlayerDetector(detections=[ref]),
        _seed(RED), SOCCER_CFG,
    )
    c = contacts[0]
    assert c.is_my_team is False
    assert c.team_label == TEAM_OFFICIAL
    assert c.player_class_id == 3
    assert filter_my_team(contacts) == []


def test_goalkeeper_contact_is_eligible():
    keeper = PlayerDetection(bbox=list(BOX), conf=0.9, class_id=1)
    contacts = enrich_contacts(
        [_cand()], FakeProvider(RED),
        FakePlayerDetector(detections=[keeper]),
        _seed(RED), SOCCER_CFG,
    )
    c = contacts[0]
    assert c.player_class_id == 1
    assert c.is_my_team is True
    assert c.team_label == TEAM_MY
    assert c.player_bbox == BOX


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
    assert c.team_label == TEAM_UNKNOWN
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
    assert opp[0].team_label == TEAM_OPPONENT
    assert opp[0].color_dist is not None and opp[0].color_dist > cfg.contact_other_team_dist


def test_to_dict_shape():
    c = enrich_contacts([_cand()], FakeProvider(RED), FakePlayerDetector(), _seed(RED), FILTER_ON)[0]
    d = c.to_dict()
    assert d["kinds"] == ["kick"] and d["is_my_team"] is True
    assert d["team_label"] == TEAM_MY
    assert "player_bbox" in d and "color_dist" in d and "player_class_id" in d
    assert "torso_crop" not in d   # crop is not serialized


# --- sideline / official differentiation ------------------------------------

def test_is_off_pitch_dirt_vs_grass():
    from polyfut_v2.pipeline.color import is_off_pitch

    grass = np.full((200, 200, 3), (40, 160, 40), dtype=np.uint8)  # BGR green
    dirt = np.full((200, 200, 3), (40, 80, 120), dtype=np.uint8)   # brown track
    box = [80, 40, 120, 160]
    assert is_off_pitch(grass, box) is False
    # Pure dirt frame has no scene grass → refuse to reject (recall-safe).
    assert is_off_pitch(dirt, box) is False
    # Pitch with a dirt strip under the feet → reject.
    mixed = grass.copy()
    mixed[130:, :] = (40, 80, 120)
    assert is_off_pitch(mixed, box) is True


def test_looks_like_official_kit_black_and_neon():
    from polyfut_v2.pipeline.color import looks_like_official_kit

    red = np.array([0, 200, 200], np.float32)
    blue = np.array([110, 200, 200], np.float32)
    black = np.array([0, 20, 40], np.float32)
    neon = np.array([30, 200, 220], np.float32)
    assert looks_like_official_kit(black, red, blue) is True
    assert looks_like_official_kit(neon, red, blue) is True
    # A black kit that IS the user's team must not be flagged.
    assert looks_like_official_kit(black, black, blue) is False
    # Ordinary red kit is not official.
    assert looks_like_official_kit(red, blue, None) is False


def test_classify_team_soft_official_kit_when_model_misses_ref_class():
    """Black kit matching neither team → official even with class_id=player."""
    cfg = PipelineV2Config(
        player_class_id=2, referee_class_id=3, official_kit_reject_enabled=True,
    )
    seed = _two_team_seed(MY_KIT, OPP_KIT)
    black = np.array([0, 25, 45], np.float32)
    is_mine, label, _ = classify_team(black, seed, cfg, player_class_id=2)
    assert is_mine is False and label == TEAM_OFFICIAL


def test_classify_team_does_not_flag_black_when_it_is_my_kit():
    cfg = PipelineV2Config(official_kit_reject_enabled=True)
    black = np.array([0, 25, 45], np.float32)
    seed = TargetSeed(kit_hsv=black, gallery=[], n_samples=1,
                      opponent_kit_hsv=np.asarray(OPP_KIT, np.float32))
    is_mine, label, _ = classify_team(black, seed, cfg, player_class_id=2)
    assert is_mine is True and label == TEAM_MY


def _pitch_with_track():
    """300x300 frame: grass above, dirt track along the bottom (sideline)."""
    f = np.full((300, 300, 3), (40, 160, 40), dtype=np.uint8)
    f[240:, :] = (30, 70, 110)
    return f


class _PitchProvider:
    def window(self, center_index, radius, step):
        step = max(1, step)
        return [(i, _pitch_with_track())
                for i in range(center_index - radius, center_index + radius + 1, step)]


def test_enrich_prefers_on_pitch_player_over_sideline_coach():
    """Coach on the dirt track must not beat a real player on the grass."""
    from polyfut_v2.pipeline.player_contacts import TEAM_SIDELINE  # noqa: F401

    player = PlayerDetection([140, 100, 170, 190], 0.9, class_id=2)   # feet ~y=175 grass
    coach = PlayerDetection([145, 220, 175, 290], 0.95, class_id=2)   # feet ~y=280 dirt
    cand = _cand(x=155, y=200)
    det = FakePlayerDetector(detections=[coach, player])
    cfg = PipelineV2Config(team_filter_enabled=False, player_class_id=2)
    c = enrich_contacts([cand], _PitchProvider(), det, _seed(RED), cfg)[0]
    assert c.player_bbox == player.bbox
    assert c.team_label != TEAM_SIDELINE


def test_enrich_labels_coach_only_contact_as_sideline_and_drops():
    """When the only body near the ball is off-pitch, label sideline + drop."""
    from polyfut_v2.pipeline.player_contacts import TEAM_SIDELINE

    coach = PlayerDetection([140, 220, 170, 290], 0.9, class_id=2)
    cand = _cand(x=155, y=250)
    det = FakePlayerDetector(detections=[coach])
    cfg = PipelineV2Config(team_filter_enabled=False, player_class_id=2)
    c = enrich_contacts([cand], _PitchProvider(), det, _seed(RED), cfg)[0]
    assert c.team_label == TEAM_SIDELINE
    assert c.is_my_team is False
    assert filter_my_team([c], enabled=True) == []
