"""Tests for batched / cheap-skip inference wiring."""

from __future__ import annotations

import numpy as np

from polyfut_video.config import PipelineConfig
from polyfut_video.main import _detections_for_routed
from polyfut_video.pipeline.ball_smooth import BallSmoother, BallSmoothConfig


class _CountingDetector:
    def __init__(self):
        self.calls = 0

    def detect_frame(self, frame):
        self.calls += 1
        return [{"bbox": [0, 0, 40, 80], "class": "player", "conf": 0.9}]

    def detect_frames_batch(self, frames):
        self.calls += len(frames)
        return [
            [{"bbox": [0, 0, 40, 80], "class": "player", "conf": 0.9}]
            for _ in frames
        ]

    def merge_players_and_ball(self, players, frame):
        self.calls += 1
        return list(players)


def test_cheap_frames_skip_most_yolo_calls():
    cfg = PipelineConfig(
        cheap_ball_refresh_every_n=10,
        cheap_player_refresh_every_n=18,
        batch_size=4,
    )
    smoother = BallSmoother(BallSmoothConfig())
    det = _CountingDetector()
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    routed = [(i, frame, "cheap", float(i)) for i in range(30)]

    dets, _players = _detections_for_routed(routed, det, smoother, [], cfg)
    assert len(dets) == 30
    # initial full + ball refresh at streak 10 and 20
    assert det.calls == 3
    assert det.calls < len(routed) // 2


def test_player_refresh_fires_on_long_cheap_run():
    cfg = PipelineConfig(
        cheap_ball_refresh_every_n=6,
        cheap_player_refresh_every_n=18,
        batch_size=4,
    )
    smoother = BallSmoother(BallSmoothConfig())
    det = _CountingDetector()
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    routed = [(i, frame, "cheap", float(i)) for i in range(36)]

    _detections_for_routed(routed, det, smoother, [], cfg)
    assert det.calls >= 5  # initial + periodic ball + at least one player refresh
    assert det.calls < 36


def test_full_frames_use_batch():
    cfg = PipelineConfig(batch_size=4)
    smoother = BallSmoother(BallSmoothConfig())
    det = _CountingDetector()
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    routed = [(i, frame, "full", float(i)) for i in range(8)]

    dets, _ = _detections_for_routed(routed, det, smoother, [], cfg)
    assert len(dets) == 8
    assert det.calls == 8
