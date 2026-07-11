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
    # Ball is analysed on every sampled frame (Stage 3, the continuous cost — the
    # single biggest runtime term). 4 → ~7.5 analysed fps at 30 fps source: still
    # fine for Stage 4 kinematics, and keeps a full 94-min match to ~2.7h with the
    # OpenVINO soccer model on a CPU (vs ~18h at every_n=3 on PyTorch).
    ball_sample_every_n: int = 4
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
    # On an ROI miss, re-scan the full frame this same step (vs coasting on the
    # stale position). Essential for fast-ball / small-pitch / zoom-changing
    # footage; near-free when the ROI usually hits (slow-ball wide shots).
    roi_fallback_full: bool = True

    # --- Stage 3c: temporal hold / interpolation across YOLO misses ---
    # Reuses polyfut_video.pipeline.ball_smooth (BallSmoother).
    ball_hold_frames: int = 10
    ball_max_jump_px: float = 500.0

    # --- Stage 4: contact candidates (kinematics on the trajectory) ---
    # Speeds are in px/s on the resized (target_width) frame; thresholds are
    # deliberately generous (high recall) and will be tuned on real trajectories
    # once a soccer ball model is plugged in.
    contact_min_speed_px_s: float = 30.0   # noise floor — ignore near-stationary jitter
    contact_dir_change_deg: float = 50.0   # heading swing that flags a deflection
    contact_stop_ratio: float = 0.4        # speed_after/speed_before below this = a trap
    contact_speed_spike_ratio: float = 2.0  # speed_after/speed_before above this = a kick
    contact_merge_sec: float = 0.3         # collapse events within this window into one
    contact_gap_sec: float = 0.5           # velocity interval longer than this = unreliable
    # Compute budget: a single player touches the ball ~30-80x/match, over-
    # generated to a few hundred, so 600 is already generous. If Stage 4 yields
    # more (noisy trajectory / demo ball), keep only the strongest — each
    # survivor costs a sparse player-detection pass (the dominant runtime term),
    # and no human reviews hundreds of clips anyway.
    max_candidates: int = 600
    # Centered position smoothing (samples). Real detector output jitters, and a
    # synthetic sweep showed window=3 lifts noisy-trajectory recall ~0.56→0.74 at
    # equal precision (5 over-smooths). Provisional — retune on real footage.
    contact_smooth_window: int = 3

    # --- Stage 0: multi-sample seed ---
    seed_sample_count: int = 4   # UI target (up to 4, min 2; 1 warns → weak gallery)
    seed_min_samples: int = 2

    # --- Stage 5: sparse player detection (pluggable; default COCO person) ---
    player_weights: str = "yolov8s.pt"
    player_class_id: int = 0     # COCO "person"
    player_conf_min: float = 0.25
    player_imgsz: int = 640
    # ROI around the ball for the sparse player pass (0 → full frame).
    player_roi_half_px: float = 160.0
    # Max ball-to-player distance (px) to count a player as the one in contact.
    contact_max_player_dist_px: float = 80.0

    # --- Stage 6: per-contact team colour filter ---
    # 1 → 3 frames per contact (centre ±1). Each frame is a player-detection call,
    # so this directly scales Stage 5-6 cost; the jersey median over 3 frames is
    # robust enough.
    contact_color_window: int = 1   # frames each side of the contact to sample jersey
    contact_color_step: int = 1     # window step, in analysed-frame units
    team_color_max_dist: float = 60.0  # hue-weighted HSV distance ≤ this ⇒ your team
    # Min torso crop area (px) to trust a jersey colour. On wide footage tiny
    # crops are grass-contaminated; below this the contact is left undecided
    # (kept) rather than mislabelled. Real-frame debug: 10-30px-tall players give
    # ~6x12px torsos, so ~50px is a sane floor. Retune per footage.
    color_min_torso_px: int = 50
    # Master switch for the colour team-filter. Default OFF: PolyFut's footage is
    # typically wide amateur/phone video where torso colour is grass-contaminated
    # and can't separate teams (real-frame debug: even the target fails to match
    # its own seed), so filtering risks silently dropping real touches — and
    # sampling jersey colour across frames triples the Stage 5-6 detection cost.
    # Enable only for clean, broadcast-quality footage with distinct kits.
    team_filter_enabled: bool = False

    # --- Stage 7: target scoring (appearance x orbital, sequential) ---
    # Appearance
    appearance_default: float = 0.5    # neutral score when appearance is unmeasurable
    orbital_anchor_min: float = 0.6    # appearance sim needed to (re)anchor the orbital
    # Orbital motion prior (pixel space; camera-comp/homography plug in via transform)
    orbital_base_px: float = 80.0      # radius at zero time gap
    orbital_growth_px_s: float = 60.0  # radius growth per second of gap
    orbital_max_gap_sec: float = 8.0   # beyond this the orbital covers the pitch → neutral
    orbital_floor: float = 0.5         # min prior — boost/tie-break only, never reject
    orbital_falloff: float = 0.5       # penalty slope outside the orbital
    # Tracklets
    tracklet_max_gap_sec: float = 3.0  # contacts within this window can share a tracklet
    # Auto accept/hide thresholds (consumed by the Step 5 montage)
    autoaccept_conf: float = 0.85
    autohide_conf: float = 0.15
    # Hard cap on the review queue: only the top-N highest-confidence "review"
    # clips are shown; the rest are auto-hidden. Bounds the human review burden
    # even when appearance can't discriminate (low-res footage → most touches
    # land in the ambiguous middle). The montage is confidence-ranked, so these
    # are the most-likely-you clips.
    max_review: int = 80

    # --- Stage 8: review montage ---
    montage_clip_pad_sec: float = 1.0    # ±1s around the contact → ~2s review clips
    montage_crop_half_px: float = 120.0  # zoom half-size (resized px) around the contact

    # --- Adaptive review grouping (Stage 8 UX) ---
    # Tag each review touch with a kit-colour group + (within your team) an
    # appearance group, so "Not me" on the other team clears them all, and same-
    # kit decisions propagate softly. Cheap: reuses the contact torso crops.
    # Kit distance from your seed beyond this ⇒ confidently the *other team*
    # (hard-removable). Deliberately looser than team_color_max_dist (60) so we
    # only ever hard-remove clearly-different kits, never risk your own.
    grouping_other_team_dist: float = 105.0
    grouping_kit_cluster_dist: float = 55.0    # colour-clustering radius
    grouping_appearance_min_sim: float = 0.62  # same-kit appearance grouping (soft)

    # --- Stage 9: hotspot assembly (v1 semantics; pad widened to ±2s per doc) ---
    hotspot_pad_before_sec: float = 2.0
    hotspot_pad_after_sec: float = 2.0
    hotspot_gap_merge_sec: float = 5.0
    hotspot_min_zone_sec: float = 3.0

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
