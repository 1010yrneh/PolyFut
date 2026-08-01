"""Analysing at the source's resolution instead of always downscaling to 640.

``target_width`` was a hard 640, so every upload was downscaled to it on decode
and uploading 720p or 1080p changed nothing but decode cost — while the
low-resolution warning told the user that higher resolution "gives dramatically
better results". The advice was not deliverable by the code that printed it.

The danger in fixing it is that ~20 thresholds are expressed in pixels *of a
640-wide frame*. An 80px "nearest player" gate covers half as much pitch at 1280
as at 640, so moving the width without moving them silently redefines the whole
pipeline. These tests pin that they move together, and that 640-wide footage
(every clip currently on disk) is completely unaffected.
"""

from __future__ import annotations

from dataclasses import fields

from polyfut_v2.config import PipelineV2Config

CFG = PipelineV2Config()


def test_the_common_case_is_a_strict_no_op():
    """640-wide source is every upload on disk today; it must not change."""
    assert CFG.for_frame_width(640) is CFG
    assert CFG.for_frame_width(None) is CFG
    assert CFG.for_frame_width(0) is CFG


def test_smaller_sources_are_never_upscaled():
    """Decode never upscales, so pretending to analyse at 640 would be a lie."""
    out = CFG.for_frame_width(480)
    assert out.target_width == 480


def test_a_higher_resolution_source_is_actually_used():
    out = CFG.for_frame_width(1920)
    assert out.target_width == CFG.analysis_max_width == 1280


def test_pixel_thresholds_move_with_the_width():
    out = CFG.for_frame_width(1280)
    s = 1280 / 640
    assert out.contact_max_player_dist_px == CFG.contact_max_player_dist_px * s
    assert out.crowd_radius_px == CFG.crowd_radius_px * s
    assert out.orbital_base_px == CFG.orbital_base_px * s
    assert out.player_roi_half_px == CFG.player_roi_half_px * s
    assert out.hotspot_possession_leave_px == CFG.hotspot_possession_leave_px * s


def test_the_calibration_residual_gate_does_not_move():
    """It is measured on native-resolution calibration frames, so the analysis
    width tells us nothing about it — scaling it would silently loosen or
    tighten the quality screen for no reason."""
    out = CFG.for_frame_width(1280)
    assert out.calibration_max_median_px == CFG.calibration_max_median_px


def test_an_area_threshold_scales_with_the_square():
    """``color_min_torso_px`` is an area, so doubling the width quadruples it."""
    out = CFG.for_frame_width(1280)
    assert out.color_min_torso_px == CFG.color_min_torso_px * 4


def test_ratios_and_times_are_left_alone():
    """A similarity, an aspect ratio, an angle and a second are resolution-free."""
    out = CFG.for_frame_width(1280)
    for f in ("orbital_anchor_min", "player_human_min_aspect",
              "contact_dir_change_deg", "orbital_max_gap_sec",
              "team_color_max_dist", "autoaccept_conf", "crowd_min_players",
              "crowd_ambiguous_gap_heights", "appearance_reject_max"):
        assert getattr(out, f) == getattr(CFG, f), f


def test_every_px_field_is_either_scaled_or_deliberately_not():
    """Guard against a new ``_px`` threshold being added and silently missed."""
    known = (set(PipelineV2Config._PX_SCALED_FIELDS)
             | set(PipelineV2Config._PX_AREA_FIELDS)
             | set(PipelineV2Config._PX_NOT_FRAME_RELATIVE))
    # Not frame-relative: a processing width, and pixel-independent maxima.
    exempt = {"target_width", "analysis_max_width", "camera_motion_width",
              "ball_imgsz", "ball_full_imgsz", "player_imgsz"}
    px_fields = {f.name for f in fields(PipelineV2Config)
                 if f.name.endswith("_px") or f.name.endswith("_px_s")}
    missed = px_fields - known - exempt
    assert not missed, f"new pixel threshold(s) not handled by for_frame_width: {missed}"


def test_it_can_be_switched_off():
    cfg = PipelineV2Config(analysis_follow_source=False)
    assert cfg.for_frame_width(1920) is cfg
