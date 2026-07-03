"""Regression: ball smoother must persist across 30s chunk boundaries within one shot."""

from polyfut_video.config import PipelineConfig
from polyfut_video.pipeline.ball_smooth import BallSmoother, BallSmoothConfig


def _run_with_reset_per_chunk() -> int:
    cfg = PipelineConfig()
    total = 0
    for chunk in range(2):
        s = BallSmoother(BallSmoothConfig(max_hold_frames=cfg.ball_hold_frames))
        for i in range(5):
            has_det = not (chunk == 1 and i == 0)
            bbox, _, _ = s.update(
                [100.0, 100.0, 108.0, 108.0] if has_det else None,
                0.9 if has_det else 0.0,
            )
            if bbox is not None:
                total += 1
    return total


def _run_continuous() -> int:
    cfg = PipelineConfig()
    s = BallSmoother(BallSmoothConfig(max_hold_frames=cfg.ball_hold_frames))
    total = 0
    for chunk in range(2):
        for i in range(5):
            has_det = not (chunk == 1 and i == 0)
            bbox, _, _ = s.update(
                [100.0, 100.0, 108.0, 108.0] if has_det else None,
                0.9 if has_det else 0.0,
            )
            if bbox is not None:
                total += 1
    return total


def test_chunk_boundary_ball_hold():
    assert _run_continuous() >= 9
    assert _run_continuous() > _run_with_reset_per_chunk()
