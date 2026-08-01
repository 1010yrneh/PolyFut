"""A pack of bodies is not the same thing as a contested touch.

Measured on a real 5-minute run (job ``fdc541cf493e``): 31 of 73 review clips
were crowd-forced, and 12 of those sat exactly on ``crowd_min_players=4``. The
box heights in that run (median 26px, ~1.75m of person) put the 90px crowd
radius at roughly 4-9m depending on depth — so the count was also measuring
different things in different parts of the frame.

The rule these tests pin: what makes attribution untrustworthy is a *close
runner-up*, not a crowd. Eight players can surround the ball while one is
plainly on it; two players can be a genuine coin-flip.
"""

from __future__ import annotations

from polyfut_v2.config import PipelineV2Config
from polyfut_v2.pipeline.player_contacts import (
    attribution_is_ambiguous,
    count_contesting_players,
)
from polyfut_v2.pipeline.player_detector import PlayerDetection

CFG = PipelineV2Config()
BALL = (150.0, 150.0)


def _player(cx, cy, *, h=26, w=11, conf=0.9, class_id=0):
    """A player box at this clip's real scale: ~26px tall, ~1.75m of person."""
    return PlayerDetection(
        bbox=[cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
        conf=conf, class_id=class_id,
    )


def _ambiguous(players, **kw):
    return attribution_is_ambiguous(
        players, BALL, CFG.crowd_radius_px,
        gap_heights=CFG.crowd_ambiguous_gap_heights,
        min_height_px=CFG.player_min_height_px,
        min_aspect=CFG.player_min_aspect,
        human_min_aspect=CFG.player_human_min_aspect,
        human_max_aspect=CFG.player_human_max_aspect,
        **kw,
    )


def test_a_pack_with_a_clear_winner_is_not_a_contest():
    """Five bodies inside the radius, one unmistakably on the ball."""
    players = [_player(150, 152)] + [
        _player(150 + dx, 150 + dy) for dx, dy in
        ((60, 10), (-55, 20), (35, -50), (-40, -45))
    ]
    assert count_contesting_players(
        players, BALL, CFG.crowd_radius_px) >= CFG.crowd_min_players
    assert not _ambiguous(players), "a clear winner should not read as contested"


def test_two_players_on_the_ball_is_a_contest():
    """Below the crowd count, but genuinely 50/50."""
    players = [_player(147, 150), _player(154, 151)]
    assert count_contesting_players(
        players, BALL, CFG.crowd_radius_px) < CFG.crowd_min_players
    assert _ambiguous(players)


def test_a_lone_player_is_never_a_contest():
    assert not _ambiguous([_player(150, 151)])


def test_nobody_found_is_treated_as_contested():
    """Recall-safety: this result is used to *relax* a review flag, so an
    absence of candidates must not relax it."""
    assert _ambiguous([])


def test_the_gap_scales_with_distance_from_the_camera():
    """The same real-world separation must read the same near and far.

    A fixed pixel gap cannot do this: the near pair below is separated by more
    pixels than the far pair while representing the same metres of pitch.
    """
    # Heights stay above player_min_height_px (16) so both pairs are actually
    # evaluated — a rejected box would make this pass for the wrong reason.
    near = [_player(150, 150, h=40, w=17), _player(150 + 26, 150, h=40, w=17)]
    far = [_player(150, 150, h=18, w=8), _player(150 + 12, 150, h=18, w=8)]
    # separations are well under one box-height in both cases -> both contested
    assert _ambiguous(near)
    assert _ambiguous(far)


def test_a_clearly_separated_pair_is_not_contested_at_either_scale():
    near = [_player(150, 150, h=40, w=17), _player(150 + 75, 150, h=40, w=17)]
    far = [_player(150, 150, h=18, w=8), _player(150 + 40, 150, h=18, w=8)]
    assert not _ambiguous(near)
    assert not _ambiguous(far)


def test_the_change_can_be_switched_off():
    cfg = PipelineV2Config(crowd_require_ambiguity=False)
    assert cfg.crowd_require_ambiguity is False
    assert CFG.crowd_require_ambiguity is True
