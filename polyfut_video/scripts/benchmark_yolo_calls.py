"""Synthetic benchmark: YOLO call budget for stages 4–7."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from polyfut_video.config import PipelineConfig
from polyfut_video.main import _detections_for_routed
from polyfut_video.pipeline.ball_smooth import BallSmoother, BallSmoothConfig
from polyfut_video.scripts.estimate_runtime import estimate_match_hours


class _CountingDetector:
    def __init__(self):
        self.full = 0
        self.ball = 0

    @property
    def calls(self):
        return self.full + self.ball

    def detect_frame(self, frame):
        self.full += 1
        return [{"bbox": [0, 0, 40, 80], "class": "player", "conf": 0.9}]

    def detect_frames_batch(self, frames):
        self.full += len(frames)
        return [
            [{"bbox": [0, 0, 40, 80], "class": "player", "conf": 0.9}]
            for _ in frames
        ]

    def merge_players_and_ball(self, players, frame):
        self.ball += 1
        return list(players)


def main():
    cfg = PipelineConfig()
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    fps = 30.0
    frames_per_chunk = int(cfg.max_chunk_sec * fps / cfg.infer_sample_every_n)

    routed = []
    for i in range(frames_per_chunk):
        # Broadcast-like: brief motion bursts then long static stretches
        route = "full" if i < 8 or (i % 120 < 4) else "cheap"
        routed.append((i, frame, route, i / (fps / cfg.infer_sample_every_n)))

    det = _CountingDetector()
    smoother = BallSmoother(BallSmoothConfig())
    _detections_for_routed(routed, det, smoother, [], cfg)

    chunks = cfg.estimated_infer_chunks(94 * 60)
    budget = estimate_match_hours(94 * 60)

    print(f"Config: chunk={cfg.max_chunk_sec}s stride={cfg.infer_sample_every_n} "
          f"ball/{cfg.cheap_ball_refresh_every_n} player/{cfg.cheap_player_refresh_every_n}")
    print(f"Frames per chunk (sampled): {frames_per_chunk}")
    print(f"YOLO calls this chunk: full={det.full} ball-only={det.ball} total={det.calls}")
    print(f"Match chunks (~94 min): {chunks}")
    print(f"Projected total YOLO calls: ~{det.calls * chunks}")
    print(f"Estimated total runtime: {budget['total_hours_est']} h "
          f"(within 3h: {budget['within_3h']})")


if __name__ == "__main__":
    main()
