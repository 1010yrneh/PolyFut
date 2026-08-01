"""Tests for the v2 speed budget: ball-detector miss-storm backoff and the
duration-scaled, time-fair Stage 4 candidate cap."""

import numpy as np

from polyfut_v2.config import PipelineV2Config
from polyfut_v2.pipeline.ball_detector import YoloBallDetector
from polyfut_v2.pipeline.contacts import ContactCandidate, cap_candidates


# --- fakes for the YOLO model -------------------------------------------------

class _Arr:
    def __init__(self, a):
        self._a = np.asarray(a, dtype=float)

    def cpu(self):
        return self

    def numpy(self):
        return self._a

    def __len__(self):
        return len(self._a)


class _Boxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = _Arr(xyxy)
        self.conf = _Arr(conf)
        self.cls = _Arr(cls)

    def __len__(self):
        return len(self.xyxy)


class _Res:
    def __init__(self, boxes):
        self.boxes = boxes


class _CountingModel:
    """Fake YOLO model: counts predict() calls; hits on scripted call numbers."""

    def __init__(self, hit_on=()):
        self.calls = 0
        self._hit_on = set(hit_on)

    def predict(self, image, **kw):
        self.calls += 1
        if self.calls in self._hit_on:
            return [_Res(_Boxes([[10, 10, 20, 20]], [0.9], [32.0]))]
        return []


def _cfg(**over):
    cfg = PipelineV2Config()
    cfg.ball_pitch_gate_enabled = False
    cfg.roi_enabled = False  # cold path only — one inference site
    cfg.ball_miss_backoff_after = 4
    cfg.ball_miss_backoff_stride = 3
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


FRAME = np.zeros((360, 640, 3), dtype=np.uint8)


def test_miss_storm_backoff_throttles_full_scans():
    model = _CountingModel()  # never detects
    det = YoloBallDetector(_cfg(), model=model)
    for _ in range(12):
        det.detect(FRAME, None)
    # Calls 1-4 always scan (below threshold); from consec_misses=4 onward only
    # every 3rd call scans: calls 5, 8, 11. Total 7 scans, 5 skipped.
    assert model.calls == 7
    assert det.stats["full_scans"] == 7
    assert det.stats["skipped_full_scans"] == 5


def test_detection_resets_backoff():
    # Miss 6 times (enters backoff), then the next scan hits.
    model = _CountingModel(hit_on={6})
    det = YoloBallDetector(_cfg(), model=model)
    results = [det.detect(FRAME, None) for _ in range(9)]
    # Calls 1-5 scan; 6-7 skipped... trace: scans happen on detect-calls
    # 1,2,3,4,5 then skip,skip, scan(=model call 6, a HIT) — after the hit the
    # miss counter resets so every following call scans again.
    hit_idx = next(i for i, r in enumerate(results) if r is not None)
    assert results[hit_idx].conf == 0.9
    post_hit = results[hit_idx + 1:]
    assert det.stats["skipped_full_scans"] == 2
    # All post-hit calls ran a real scan (misses again, but below threshold).
    assert model.calls == 6 + len(post_hit)


def test_shot_boundary_reset_restores_scanning():
    model = _CountingModel()
    det = YoloBallDetector(_cfg(), model=model)
    for _ in range(8):
        det.detect(FRAME, None)
    in_backoff_calls = model.calls
    det.reset()
    det.detect(FRAME, None)
    assert model.calls == in_backoff_calls + 1  # scanned immediately after reset


def test_backoff_disabled_when_after_is_zero():
    model = _CountingModel()
    det = YoloBallDetector(_cfg(ball_miss_backoff_after=0), model=model)
    for _ in range(10):
        det.detect(FRAME, None)
    assert model.calls == 10
    assert det.stats["skipped_full_scans"] == 0


# --- Stage 4 candidate cap ----------------------------------------------------

def _cand(t, strength):
    return ContactCandidate(
        frame_index=int(t * 30), t_sec=t, processed_sec=t,
        x=100.0, y=100.0, kinds=["kick"], strength=strength,
    )


def test_cap_scales_with_duration():
    cfg = PipelineV2Config()
    cfg.max_candidates = 600
    cfg.max_candidates_per_min = 40
    cands = [_cand(i * 0.5, 0.5) for i in range(400)]  # 400 over ~200 s
    out = cap_candidates(cands, cfg, duration_sec=180.0)  # 3 min → cap 120
    assert len(out) == 120
    # Time order preserved.
    assert all(a.processed_sec <= b.processed_sec for a, b in zip(out, out[1:]))


def test_cap_floor_protects_short_clips():
    cfg = PipelineV2Config()
    cfg.max_candidates = 600
    cfg.max_candidates_per_min = 40
    cands = [_cand(i * 0.2, 0.5) for i in range(100)]  # 100 in a 30 s clip
    out = cap_candidates(cands, cfg, duration_sec=30.0)  # 40*0.5=20 → floor 60
    assert len(out) == 60


def test_fair_selection_spreads_across_windows():
    cfg = PipelineV2Config()
    cfg.max_candidates = 20
    cfg.max_candidates_per_min = 0  # exercise the flat cap
    cfg.candidate_fair_window_sec = 30.0
    # A noisy early window with 100 strong candidates, and a quiet later window
    # with 10 weak ones. Global top-K would silence the later window entirely.
    noisy = [_cand(0.1 + i * 0.25, 0.9) for i in range(100)]   # 0-25 s
    quiet = [_cand(61.0 + i * 1.0, 0.1) for i in range(10)]    # 61-70 s
    out = cap_candidates(noisy + quiet, cfg, duration_sec=90.0)
    assert len(out) == 20
    late = [c for c in out if c.processed_sec > 60.0]
    assert len(late) == 10  # every real-window candidate survived


def test_no_cap_when_under_budget():
    cfg = PipelineV2Config()
    cands = [_cand(i * 1.0, 0.5) for i in range(30)]
    out = cap_candidates(cands, cfg, duration_sec=180.0)
    assert out == cands
