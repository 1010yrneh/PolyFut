"""Why did the pitch calibration not get used?

A real run (job ``fdc541cf493e``) recorded ``metric_coverage: 0.0`` with all 179
contacts on the pixel gate, and nothing anywhere said which of the five exits in
``_build_pitch_mapper`` had fired. A skipped calibration screen, an unreadable
payload, a fit rejected by the quality gate and a missing camera track all
produced byte-identical output, so "attribution still looks wrong" had no
answerable first question.

Falling back quietly is still the right behaviour — a wrong pitch map is worse
than none. Being unable to say *why* afterwards is not.
"""

from __future__ import annotations

from polyfut_v2.config import PipelineV2Config
from polyfut_v2.orchestrator import (
    CALIBRATION_FALLBACK_MESSAGES,
    _build_pitch_mapper,
)

CFG = PipelineV2Config()


class _Traj:
    def __init__(self, camera=None):
        self.camera = camera


def _reason(calibration, traj=None, cfg=CFG):
    _mapper, reason = _build_pitch_mapper(traj or _Traj(), cfg, calibration)
    return reason


def test_no_calibration_reports_that_it_was_never_supplied():
    assert _reason(None) == "not_supplied"


def test_a_skipped_screen_is_not_reported_as_a_fault():
    """The user choosing Skip is a decision, not an error — no warning text."""
    assert "not_supplied" not in CALIBRATION_FALLBACK_MESSAGES


def test_an_unreadable_calibration_says_so():
    assert _reason({"nonsense": True}) == "unreadable"


def test_a_calibration_the_quality_gate_rejects_says_so():
    """Above ``calibration_max_median_px`` the fit is not trusted."""
    from polyfut_v2.pipeline.pitch_calibration import PitchCalibration

    bad = PitchCalibration(
        params=[0.0] * 9, frames=[0], anchor_frame=0, anchor_sec=0.0,
        principal=(320.0, 180.0), pitch_length_m=100.0, pitch_width_m=64.0,
        median_px=CFG.calibration_max_median_px + 10.0, rms_px=20.0, dof=5,
        n_landmarks=6, frame_width=640,
    )
    assert _reason(bad) == "quality_rejected"


def test_a_good_calibration_with_no_camera_track_says_so():
    from polyfut_v2.pipeline.pitch_calibration import PitchCalibration

    ok = PitchCalibration(
        params=[0.0] * 9, frames=[0], anchor_frame=0, anchor_sec=0.0,
        principal=(320.0, 180.0), pitch_length_m=100.0, pitch_width_m=64.0,
        median_px=1.0, rms_px=1.5, dof=5, n_landmarks=6, frame_width=640,
    )
    assert _reason(ok, traj=_Traj(camera=None)) == "no_camera_track"


def test_every_failure_reason_has_something_to_show_the_user():
    """A reason nobody can read is the bug this fixes, reintroduced."""
    for reason in ("unreadable", "quality_rejected", "no_camera_track",
                   "mapper_failed"):
        msg = CALIBRATION_FALLBACK_MESSAGES[reason]
        assert msg and msg.format(limit=6.0)
