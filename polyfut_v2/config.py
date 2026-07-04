"""Central configuration for the v2 (ball-anchored) pipeline.

Only the knobs needed by the stages built so far are defined here; later steps
append their own sections. Stages 1-3 reuse the v1 shot-filter / deadtime code,
so ``v1_config`` projects the shared knobs onto a v1 ``PipelineConfig``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PipelineV2Config:
    # --- Stage 1: decode ---
    target_width: int = 640
    # Ball is analysed on every sampled frame (Stage 3, the continuous cost).
    # 3 → ~10 analysed fps at 30 fps source: enough temporal resolution for the
    # Stage 4 kinematics without paying full frame rate.
    ball_sample_every_n: int = 3
    # Coarse stride for the cheap shot-filter pass.
    shot_filter_sample_every_n: int = 10

    # --- Stage 2: shot filter (heuristics, mirrors v1 defaults) ---
    cut_hist_threshold: float = 0.45
    green_ratio_min: float = 0.22
    motion_smooth_max: float = 2.5
    graphic_uniform_ratio: float = 0.35

    # --- Stage 3a: deadtime ---
    deadtime_motion_threshold: float = 1.2
    deadtime_min_duration_sec: float = 60.0

    # --- Stage 3b: ball detection (pluggable; default COCO YOLO) ---
    # Swap ``ball_weights`` for a soccer-specific ball model (and set
    # ``ball_class_id`` to that model's ball class) to drop it in.
    ball_weights: str = "yolov8s.pt"
    ball_class_id: int = 32  # COCO "sports ball"
    device: str = "cpu"
    conf_threshold: float = 0.25
    ball_conf_min: float = 0.07
    # Inference size for the small ROI crop vs. the full-frame re-acquire.
    ball_imgsz: int = 416
    ball_full_imgsz: int = 640

    # --- Stage 3b: ROI search around last known ball position ---
    # A region-of-interest around the last ball gives higher effective
    # resolution on a tiny object and is far cheaper than full-frame every step.
    roi_enabled: bool = True
    roi_half_px: float = 120.0  # half-width of the square ROI, in resized px

    # --- Stage 3c: temporal hold / interpolation across YOLO misses ---
    # Reuses polyfut_video.pipeline.ball_smooth (BallSmoother).
    ball_hold_frames: int = 10
    ball_max_jump_px: float = 500.0

    # --- Reliability guardrail ---
    # A trajectory with almost no real detections means the ball model can't
    # see the ball (e.g. COCO yolov8 on a tiny/distant soccer ball). Per the
    # design doc, ball recall is the reliability ceiling and a degenerate
    # trajectory silently starves every downstream stage — so warn loudly.
    min_detected_ratio_warn: float = 0.2

    # --- Output ---
    output_dir: Path = Path("output_v2")

    def v1_config(self):
        """Project shared knobs onto a v1 ``PipelineConfig`` for the reused
        Stage 2 / Stage 3a code paths."""
        from polyfut_video.config import PipelineConfig

        return PipelineConfig(
            target_width=self.target_width,
            shot_filter_sample_every_n=self.shot_filter_sample_every_n,
            cut_hist_threshold=self.cut_hist_threshold,
            green_ratio_min=self.green_ratio_min,
            motion_smooth_max=self.motion_smooth_max,
            graphic_uniform_ratio=self.graphic_uniform_ratio,
            deadtime_motion_threshold=self.deadtime_motion_threshold,
            deadtime_min_duration_sec=self.deadtime_min_duration_sec,
        )


DEFAULT_CONFIG = PipelineV2Config()
