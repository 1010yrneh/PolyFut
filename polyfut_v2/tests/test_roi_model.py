"""The ROI pass must use its own smaller model when one exists.

Background: the OpenVINO export has a static [1,3,640,640] input, so
``_apply_soccer_model`` pinned ball_imgsz = ball_full_imgsz = 640. That sent the
240px ROI crop through a 640 graph — upscaling it — and measured on an idle
i7-1255U the "fast" warm path cost MORE than the full scan it exists to avoid:

    full frame 640x360 -> 640 export   134 ms
    ROI crop   240x240 -> 640 export   192 ms
    ROI crop   240x240 -> 320 export    64 ms

End to end on 400 real frames it was also finding fewer balls, because
upscaling moved the ball away from the scale the model was trained at:
101.5 -> 73.7 ms/frame and 98 -> 107 balls once the ROI ran at 320.

What these tests protect is the *wiring*, which is where it can silently
regress: if ball_roi_weights stops reaching the ROI call, everything still
works and just quietly gets slow again — the exact failure that shipped.
"""

from __future__ import annotations

import numpy as np
import pytest

from polyfut_v2.config import PipelineV2Config
from polyfut_v2.pipeline.ball_detector import YoloBallDetector


class _SpyModel:
    """Records the imgsz of every call, and finds nothing."""

    def __init__(self, name):
        self.name = name
        self.calls: list[int] = []

    def predict(self, image, imgsz=None, **kw):
        self.calls.append(int(imgsz))
        return []


def _cfg(**kw):
    cfg = PipelineV2Config()
    cfg.fast_infer_enabled = False       # force the predict() path we can spy on
    cfg.harvest_players_from_ball_pass = False
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def _frame():
    return np.zeros((360, 640, 3), dtype=np.uint8)


# ------------------------------------------------------------- defaults hold
def test_without_a_roi_model_everything_uses_the_main_model():
    """No ball_roi_weights: identical to the old behaviour, one model only."""
    cfg = _cfg()
    d = YoloBallDetector(cfg, model=_SpyModel("main"))
    assert d.roi_model is d.model
    assert d.roi_imgsz == cfg.ball_imgsz


def test_roi_imgsz_falls_back_when_unset():
    cfg = _cfg(ball_imgsz=416, ball_roi_imgsz=0)
    d = YoloBallDetector(cfg, model=_SpyModel("main"))
    assert d.roi_imgsz == 416


# ------------------------------------------------- the ROI actually uses it
def test_roi_pass_runs_at_the_roi_size(monkeypatch):
    """The whole point: a warm frame must infer at 320, not 640."""
    cfg = _cfg(ball_imgsz=640, ball_full_imgsz=640,
               ball_roi_imgsz=320, ball_roi_weights="roi.xml")
    main, roi = _SpyModel("main"), _SpyModel("roi")
    d = YoloBallDetector(cfg, model=main)
    d._roi_model = roi

    d.detect(_frame(), last_center=(320.0, 180.0))

    assert roi.calls == [320], f"ROI should infer once at 320, got {roi.calls}"
    # the ROI found nothing, so a full scan follows - on the MAIN model at 640
    assert main.calls == [640], f"full scan should be 640 on main, got {main.calls}"


def test_cold_start_never_touches_the_roi_model():
    """No last_center means no ROI: a full scan on the main model only."""
    cfg = _cfg(ball_roi_imgsz=320, ball_roi_weights="roi.xml")
    main, roi = _SpyModel("main"), _SpyModel("roi")
    d = YoloBallDetector(cfg, model=main)
    d._roi_model = roi

    d.detect(_frame(), last_center=None)

    assert roi.calls == []
    assert main.calls == [cfg.ball_full_imgsz]


def test_roi_disabled_bypasses_the_roi_model():
    cfg = _cfg(roi_enabled=False, ball_roi_imgsz=320, ball_roi_weights="roi.xml")
    main, roi = _SpyModel("main"), _SpyModel("roi")
    d = YoloBallDetector(cfg, model=main)
    d._roi_model = roi

    d.detect(_frame(), last_center=(320.0, 180.0))

    assert roi.calls == []


# --------------------------------------------------- the export must be safe
def test_roi_export_never_writes_over_the_640_model(tmp_path, monkeypatch):
    """Ultralytics exports to <stem>_openvino_model beside the .pt, which IS the
    640 model's directory. An in-place 320 export deletes the model the pipeline
    depends on - that happened during development, so it is pinned here."""
    from polyfut_v2 import ball_model as bm

    pt = tmp_path / "soccer_uisikdag.pt"
    pt.write_bytes(b"x" * 2_000_000)
    ov640 = tmp_path / "soccer_uisikdag_openvino_model"
    ov640.mkdir()
    (ov640 / "soccer_uisikdag.xml").write_text("<net/>", encoding="utf-8")
    roi_dir = tmp_path / "soccer_uisikdag_roi320_openvino_model"

    monkeypatch.setattr(bm, "SOCCER_MODEL_PATH", pt)
    monkeypatch.setattr(bm, "SOCCER_MODEL_OV_DIR", ov640)
    monkeypatch.setattr(bm, "SOCCER_MODEL_OV_ROI_DIR", roi_dir)
    monkeypatch.setattr(bm, "ensure_soccer_model", lambda download=True: pt)

    class _Boom:
        def __init__(self, *a, **k): pass
        def export(self, **kw):
            raise RuntimeError("export blew up halfway")

    import sys, types
    fake = types.ModuleType("ultralytics")
    fake.YOLO = _Boom
    monkeypatch.setitem(sys.modules, "ultralytics", fake)

    assert bm.ensure_soccer_model_openvino_roi() is None      # degrades, no raise
    # and the 640 model is untouched, which is the thing that matters
    assert (ov640 / "soccer_uisikdag.xml").exists()
    assert not (tmp_path / "_roi_export_tmp").exists()        # scratch cleaned up


def test_roi_export_is_optional(tmp_path, monkeypatch):
    """A missing ROI export must cost speed, never correctness."""
    cfg = _cfg(ball_roi_weights="", ball_roi_imgsz=0)
    main = _SpyModel("main")
    d = YoloBallDetector(cfg, model=main)
    d.detect(_frame(), last_center=(320.0, 180.0))
    # ROI ran on the main model at the main size - old behaviour exactly
    assert main.calls and main.calls[0] == cfg.ball_imgsz
