"""``roi_fallback_full`` governs what happens after an ROI MISS, nothing else.

It was applied as an unconditional ``det = None``, which did the opposite of
what it says on both counts: it threw away ROI *hits*, and by emptying ``det``
it triggered the very full-frame scan it was supposed to suppress. Dormant at
the default of True, but it made an "ROI only" benchmark come out slower than
ROI-plus-fallback, which is how it surfaced.

The truth table these tests pin:

| roi ran | ROI result | roi_fallback_full | full scan? | detection      |
|---------|------------|-------------------|------------|----------------|
| no      | -          | either            | yes        | from full scan |
| yes     | hit        | either            | **no**     | **the ROI hit**|
| yes     | miss       | True              | yes        | from full scan |
| yes     | miss       | False             | **no**     | none (coast)   |
"""

from __future__ import annotations

import numpy as np

from polyfut_v2.config import PipelineV2Config
from polyfut_v2.pipeline.ball_detector import YoloBallDetector

BALL = 0
FRAME = np.zeros((360, 640, 3), np.uint8)


def _cfg(**over):
    cfg = PipelineV2Config()
    cfg.ball_class_id = BALL
    cfg.ball_conf_min = 0.07
    cfg.roi_enabled = True
    cfg.roi_fallback_full = True
    cfg.ball_pitch_gate_enabled = False
    cfg.harvest_players_from_ball_pass = False
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


class _Stub:
    """Finds the ball only on the full frame, or everywhere, as configured."""

    def __init__(self, roi_finds: bool):
        self.roi_finds = roi_finds
        self.calls = []          # (width, height) of each image handed over

    def predict(self, img, imgsz=640, conf=0.0, classes=None, verbose=False,
                **kw):
        h, w = img.shape[:2]
        self.calls.append((w, h))
        is_roi = w < 640
        found = (not is_roi) or self.roi_finds

        class _B:
            def __len__(self_inner):
                return 1 if found else 0

            @property
            def xyxy(self_inner):
                return _T(np.array([[10.0, 10.0, 20.0, 20.0]], np.float32))

            @property
            def conf(self_inner):
                return _T(np.array([0.9], np.float32))

            @property
            def cls(self_inner):
                return _T(np.array([float(BALL)], np.float32))

        class _R:
            boxes = _B()

        return [_R()]


class _T:
    def __init__(self, a):
        self._a = a

    def cpu(self):
        return self

    def numpy(self):
        return self._a


def _run(cfg, roi_finds, last_center=(320.0, 180.0)):
    m = _Stub(roi_finds)
    d = YoloBallDetector(cfg, model=m)
    det = d.detect(FRAME, last_center)
    return d, det, m


def test_an_roi_hit_is_kept_and_no_full_scan_follows():
    for flag in (True, False):
        d, det, m = _run(_cfg(roi_fallback_full=flag), roi_finds=True)
        assert det is not None, f"ROI hit discarded with flag={flag}"
        assert d.stats["roi_hits"] == 1
        assert d.stats["full_scans"] == 0, f"needless full scan with flag={flag}"
        assert len(m.calls) == 1


def test_an_roi_miss_falls_back_when_enabled():
    d, det, m = _run(_cfg(roi_fallback_full=True), roi_finds=False)
    assert d.stats["roi_misses"] == 1
    assert d.stats["full_scans"] == 1
    assert det is not None                      # the full scan found it
    assert len(m.calls) == 2


def test_an_roi_miss_coasts_when_the_fallback_is_off():
    """The point of the flag: no second inference, no detection this step."""
    d, det, m = _run(_cfg(roi_fallback_full=False), roi_finds=False)
    assert d.stats["roi_misses"] == 1
    assert d.stats["full_scans"] == 0
    assert det is None
    assert len(m.calls) == 1


def test_a_cold_start_still_scans_the_full_frame():
    """No last position means no ROI ran, so the flag must not block the only
    pass that can find the ball at all."""
    for flag in (True, False):
        d, det, m = _run(_cfg(roi_fallback_full=flag), roi_finds=False,
                         last_center=None)
        assert d.stats["full_scans"] == 1, f"cold start blocked with flag={flag}"
        assert det is not None
        assert m.calls == [(640, 360)]


def test_roi_disabled_scans_the_full_frame():
    for flag in (True, False):
        d, det, _m = _run(_cfg(roi_enabled=False, roi_fallback_full=flag),
                          roi_finds=False)
        assert d.stats["full_scans"] == 1
        assert det is not None


def test_the_default_path_is_untouched():
    """Everything above must leave the shipped configuration exactly as it was:
    hit -> one call, miss -> two."""
    cfg = _cfg()
    assert cfg.roi_fallback_full is True
    d_hit, _det, m_hit = _run(cfg, roi_finds=True)
    d_miss, _det2, m_miss = _run(_cfg(), roi_finds=False)
    assert (len(m_hit.calls), d_hit.stats["full_scans"]) == (1, 0)
    assert (len(m_miss.calls), d_miss.stats["full_scans"]) == (2, 1)
