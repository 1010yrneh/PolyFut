"""Central configuration for the Level 1 video pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PipelineConfig:
    # Stage 1 — decode
    target_width: int = 640
    sample_every_n: int = 1
    shot_filter_sample_every_n: int = 10

    # Stage 2 — shot filter
    cut_hist_threshold: float = 0.45
    green_ratio_min: float = 0.22
    motion_smooth_max: float = 2.5
    graphic_uniform_ratio: float = 0.35

    # Stage 3 — deadtime
    deadtime_motion_threshold: float = 1.2
    deadtime_min_duration_sec: float = 60.0

    # Stages 4–7 — tuned for ~94-min match in ≤3 h on CPU (yolov8s)
    # Larger chunks → fewer DBSCAN/tracker boundaries; same total frames.
    max_chunk_sec: float = 45.0
    # ~7.5 analysed fps at 30 fps source — better touch timing than stride 5.
    infer_sample_every_n: int = 4
    # Cheap-route YOLO schedule (static broadcast spends most time on cheap).
    cheap_ball_refresh_every_n: int = 5
    cheap_player_refresh_every_n: int = 15

    # Stage 4 — frame router
    router_motion_threshold: float = 2.5
    router_downscale: int = 4

    # Stage 5 — detection (yolov8s: better small-ball recall than nano)
    yolo_weights: str = "yolov8s.pt"
    conf_threshold: float = 0.25
    ball_conf_min: float = 0.07
    device: str = "cpu"
    imgsz: int = 640
    ball_imgsz: int = 416

    # Stage 5b — ball temporal hold across YOLO misses
    ball_hold_frames: int = 10
    ball_max_jump_px: float = 500.0

    # Stage 6 — tracking (boxmot ByteTrack)
    track_thresh: float = 0.25
    match_thresh: float = 0.8

    # Stage 7 — team classify (accumulate crops across chunks within a shot)
    dbscan_eps: float = 18.0
    dbscan_min_samples: int = 3
    min_cluster_size: int = 4
    team_crops_per_track: int = 3

    # Stage 8 — possession (distance-aware contact + hysteresis)
    possession_window_sec: float = 0.8
    possession_on_frames: int = 1
    possession_off_frames: int = 6
    possession_base_thresh_px: float = 85.0
    possession_ref_person_h: float = 100.0
    possession_ref_ball_diag: float = 18.0
    possession_min_thresh_px: float = 55.0
    possession_max_thresh_px: float = 140.0
    contested_margin: float = 0.15

    # Stage 9 — touch hotspot zones (UI)
    # Merge same-team touches ≤5s apart; add 3s context before/after raw clip.
    hotspot_pad_before_sec: float = 3.0
    hotspot_pad_after_sec: float = 3.0
    hotspot_gap_merge_sec: float = 5.0
    hotspot_min_zone_sec: float = 3.0

    # Stage 9 — export
    output_dir: Path = field(default_factory=lambda: Path("output"))

    # Hardware hints — larger batches amortize CPU inference overhead
    batch_size: int = 12

    def estimated_infer_chunks(self, duration_sec: float) -> int:
        """Approximate chunk count for progress / runtime budgeting."""
        import math
        live = max(0.0, duration_sec)
        return max(1, math.ceil(live / max(self.max_chunk_sec, 1.0)))


DEFAULT_CONFIG = PipelineConfig()
