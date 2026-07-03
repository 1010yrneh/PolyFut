"""Auto-tune pipeline settings and resolve the fastest available weights."""

from __future__ import annotations

import os
from pathlib import Path

from polyfut_cv.pipeline import PipelineConfig
from polyfut_cv.preprocess import have_ffmpeg


def resolve_weights(default_pt: str | Path) -> str:
    """Prefer OpenVINO export if present (3-5x faster on Intel CPU)."""
    p = Path(default_pt).resolve()
    env_ov = os.environ.get("POLYFUT_OPENVINO_WEIGHTS", "")
    candidates = [
        p.parent / "yolov8n_openvino_model",
        p.parent.parent / "yolov8n_openvino_model",
        Path(env_ov) if env_ov else None,
        p.parent / "yolov8n_openvino_int8_model",
        p.parent.parent / "yolov8n_openvino_int8_model",
    ]
    for c in candidates:
        if c and c.is_dir() and any(c.glob("*.xml")):
            return str(c)
    return str(p)


def using_openvino(weights: str | Path) -> bool:
    wp = Path(weights)
    return wp.is_dir() and any(wp.glob("*.xml"))


def tune_for_duration(duration_min: float, *, have_ff: bool | None = None) -> PipelineConfig:
    """Pick stride/imgsz/proxy/two-pass based on match length.

    Targets:
      - short clip (<15 min): accurate, still fast
      - medium (15-45 min): balanced
      - full match (>45 min): aggressive speed, two-pass + proxy
    """
    have_ff = have_ffmpeg() if have_ff is None else have_ff
    cfg = PipelineConfig()
    cfg.batch_size = 8
    cfg.two_pass = True
    cfg.coarse_stride_mult = 8
    cfg.coarse_imgsz = 640
    cfg.conf = 0.15
    cfg.auto_proxy = have_ff and duration_min > 20.0
    cfg.proxy_threshold_min = 20.0
    cfg.proxy_fps = 12.0
    cfg.proxy_width = 1280       # preserve ball detail through the proxy

    if duration_min <= 15:
        # Short clip: SAHI tiles + single high-res pass (ball recall priority).
        cfg.stride = 5
        cfg.imgsz = 1280
        cfg.two_pass = False
        cfg.auto_proxy = False
        cfg.use_tiled_dense = True
    elif duration_min <= 45:
        cfg.stride = 6
        cfg.imgsz = 1280
        cfg.use_tiled_dense = False
    else:
        # Full match: dense @ 1280 without 4× tiled infer; fewer frames via stride 8.
        # OpenVINO strongly recommended (3–5×). Export: cv/scripts/export_openvino.py
        cfg.stride = 8
        cfg.imgsz = 1280
        cfg.coarse_stride_mult = 10
        cfg.use_tiled_dense = False
        cfg.batch_size = 16

    return cfg
