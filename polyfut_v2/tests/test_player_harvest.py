"""Issue 15, step 1: keep the players the ball scan already found.

The soccer model emits ball + keeper + player + referee from ONE forward pass —
the class list is a post-NMS filter, not a cheaper inference — and on a measured
run the full-frame re-acquire fires on 242 of 451 analysed frames. So roughly
half the video's player positions are already paid for and thrown away.

The contract these tests hold: taking them must not move the ball. Verified end
to end on b48758eb195e too (0 frames differ, 0.0000 px, twice), but that needs
the model and a clip, so the invariants live here against a stub.
"""

from __future__ import annotations

import numpy as np
import pytest

from polyfut_v2.config import PipelineV2Config
from polyfut_v2.pipeline.ball_detector import YoloBallDetector

BALL, KEEPER, PLAYER, REF = 0, 1, 2, 3


def _cfg(**over):
    cfg = PipelineV2Config()
    cfg.ball_class_id = BALL
    cfg.player_class_id = PLAYER
    cfg.goalkeeper_class_id = KEEPER
    cfg.referee_class_id = REF
    cfg.ball_conf_min = 0.07
    cfg.player_conf_min = 0.25
    cfg.roi_enabled = False          # exercise the full-frame path directly
    cfg.ball_pitch_gate_enabled = False
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


class _StubModel:
    """Records the class list it was asked for and returns fixed detections."""

    def __init__(self, rows):
        self.rows = rows            # (x1,y1,x2,y2,conf,cls)
        self.asked = []

    def predict(self, _img, imgsz=640, conf=0.0, classes=None, verbose=False,
                **kw):
        self.asked.append(list(classes) if classes else None)
        keep = [r for r in self.rows
                if r[4] >= conf and (classes is None or int(r[5]) in classes)]

        class _B:
            def __init__(self, rows):
                self._r = rows

            def __len__(self):
                return len(self._r)

            @property
            def xyxy(self):
                return _T(np.array([r[:4] for r in self._r], np.float32)
                          .reshape(-1, 4))

            @property
            def conf(self):
                return _T(np.array([r[4] for r in self._r], np.float32))

            @property
            def cls(self):
                return _T(np.array([r[5] for r in self._r], np.float32))

        class _R:
            boxes = _B(keep)

        return [_R()]


class _T:
    def __init__(self, a):
        self._a = a

    def cpu(self):
        return self

    def numpy(self):
        return self._a


ROWS = [
    (300.0, 160.0, 312.0, 172.0, 0.55, BALL),     # the ball
    (100.0, 100.0, 116.0, 140.0, 0.90, PLAYER),   # solid players
    (200.0, 110.0, 214.0, 148.0, 0.70, PLAYER),
    (400.0, 120.0, 412.0, 156.0, 0.30, KEEPER),
    (500.0, 130.0, 510.0, 160.0, 0.12, PLAYER),   # under player_conf_min
    (520.0, 130.0, 530.0, 160.0, 0.80, REF),      # a referee
]


def _detect(cfg):
    m = _StubModel(ROWS)
    det = YoloBallDetector(cfg, model=m)
    ball = det.detect(np.zeros((360, 640, 3), np.uint8), None)
    return det, ball, m


# ------------------------------------------------------------- the ball
def test_the_ball_is_unchanged_by_harvesting():
    """The whole point: this is meant to be free, including free of side effects."""
    _d_off, ball_off, _m = _detect(_cfg(harvest_players_from_ball_pass=False))
    _d_on, ball_on, _m2 = _detect(_cfg(harvest_players_from_ball_pass=True))
    assert ball_off is not None and ball_on is not None
    assert ball_off.bbox == ball_on.bbox
    assert ball_off.conf == ball_on.conf


def test_the_ball_still_wins_against_much_more_confident_players():
    """Players outscore the ball here 0.90 to 0.55; class filtering, not score,
    decides which box is the ball."""
    _d, ball, _m = _detect(_cfg(harvest_players_from_ball_pass=True))
    assert ball.bbox == [300.0, 160.0, 312.0, 172.0]


# ---------------------------------------------------------- the players
def test_players_come_back_from_the_same_pass():
    det, _ball, model = _detect(_cfg(harvest_players_from_ball_pass=True))
    assert det.last_players is not None
    xyxy, conf, cls = det.last_players
    # two players (0.90, 0.70), the keeper (0.30) and the referee (0.80);
    # the 0.12 player is under player_conf_min
    assert len(xyxy) == 4
    assert model.asked[-1] is not None
    assert set(model.asked[-1]) == {BALL, KEEPER, PLAYER, REF}


def test_players_are_held_to_their_own_confidence_threshold():
    """The pass runs at the ball's lower threshold so the ball candidate set is
    untouched; the 0.12 player must not survive that."""
    det, _b, _m = _detect(_cfg(harvest_players_from_ball_pass=True))
    _xyxy, conf, _cls = det.last_players
    assert conf.min() >= 0.25
    assert 0.12 not in set(np.round(conf, 2))


def test_no_ball_class_among_the_harvested_players():
    det, _b, _m = _detect(_cfg(harvest_players_from_ball_pass=True))
    _xyxy, _conf, cls = det.last_players
    assert BALL not in set(cls.astype(int))


def test_one_inference_serves_both():
    """If this ever needs a second predict() the saving is gone."""
    _det, _ball, model = _detect(_cfg(harvest_players_from_ball_pass=True))
    assert len(model.asked) == 1


# --------------------------------------------------------------- off path
def test_disabled_asks_only_for_the_ball_and_harvests_nothing():
    det, _ball, model = _detect(_cfg(harvest_players_from_ball_pass=False))
    assert det.last_players is None
    assert model.asked[-1] == [BALL]


def test_a_single_class_model_has_nothing_to_harvest():
    """COCO mode: one class, so there is no free lunch and no wider request."""
    cfg = _cfg(harvest_players_from_ball_pass=True)
    cfg.goalkeeper_class_id = None
    cfg.referee_class_id = None
    cfg.player_class_id = cfg.ball_class_id
    det = YoloBallDetector(cfg, model=_StubModel(ROWS))
    assert det._harvest_classes() is None


def test_the_roi_pass_is_never_harvested():
    """A 240px crop cuts bodies in half; their boxes would be wrong.

    Driven through an ROI *hit*, which is the only way to reach the ROI path
    without a full scan behind it — note that `roi_fallback_full=False` does
    NOT do that today, it forces a full scan on every frame and throws ROI hits
    away (a separate, dormant bug in `detect`).
    """
    cfg = _cfg(harvest_players_from_ball_pass=True)
    cfg.roi_enabled = True
    cfg.roi_fallback_full = True
    m = _StubModel(ROWS)
    det = YoloBallDetector(cfg, model=m)
    ball = det.detect(np.zeros((360, 640, 3), np.uint8), (306.0, 166.0))
    assert ball is not None                      # the ROI hit
    assert det.stats["roi_hits"] == 1
    assert det.stats["full_scans"] == 0          # so no full pass happened
    assert det.last_players is None


def test_stats_record_what_was_harvested():
    det, _b, _m = _detect(_cfg(harvest_players_from_ball_pass=True))
    assert det.stats["harvested_frames"] == 1
    assert det.stats["harvested_players"] == 4
