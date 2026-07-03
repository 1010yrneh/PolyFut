"""Estimate stages 4–7 runtime for a match (CPU budget check)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from polyfut_video.config import PipelineConfig
from polyfut_video.pipeline.decode import probe_video


def estimate_yolo_calls_per_chunk(frames: int, full_pct: float, cfg: PipelineConfig) -> dict:
    """Rough YOLO call count for one chunk (full batch + cheap schedule)."""
    n_full = int(frames * full_pct)
    n_cheap = frames - n_full
    ball_every = cfg.cheap_ball_refresh_every_n
    player_every = cfg.cheap_player_refresh_every_n

    full_calls = math.ceil(n_full / max(1, cfg.batch_size)) * min(cfg.batch_size, n_full)
    # Conservative: count each full-route frame as one batch slot
    full_calls = max(n_full, math.ceil(n_full / cfg.batch_size) * (cfg.batch_size // 2))

    ball_calls = 0
    player_calls = 1  # initial cheap if no cache
    streak = 0
    for _ in range(n_cheap):
        streak += 1
        if streak >= player_every:
            player_calls += 1
            streak = 0
        elif streak >= ball_every:
            ball_calls += 1
            streak = 0

    return {
        "frames": frames,
        "full_frames": n_full,
        "cheap_frames": n_cheap,
        "full_yolo_equiv": n_full,
        "ball_only_calls": ball_calls,
        "full_on_cheap_calls": player_calls,
        "total_yolo_equiv": n_full + ball_calls + player_calls,
    }


def estimate_match_hours(
    duration_sec: float,
    fps: float = 30.0,
    full_pct: float = 0.22,
    *,
    cfg: PipelineConfig | None = None,
    sec_per_full_yolo: float = 0.18,
    sec_per_ball_yolo: float = 0.07,
    sec_per_chunk_overhead: float = 2.5,
) -> dict:
    """
    Project total pipeline time. Tune sec_per_* from profile_inference on your machine.
    Default constants assume yolov8s on a modern laptop CPU.
    """
    cfg = cfg or PipelineConfig()
    stride = max(1, cfg.infer_sample_every_n)
    frames_per_chunk = int(cfg.max_chunk_sec * fps / stride)
    chunks = cfg.estimated_infer_chunks(duration_sec)
    per = estimate_yolo_calls_per_chunk(frames_per_chunk, full_pct, cfg)

    infer_sec = chunks * (
        per["full_yolo_equiv"] * sec_per_full_yolo
        + per["ball_only_calls"] * sec_per_ball_yolo
        + per["full_on_cheap_calls"] * sec_per_full_yolo
        + sec_per_chunk_overhead
    )
    stages_12_sec = duration_sec * 0.015  # ~1.5% of realtime for decode/filter
    stages_89_sec = 120.0

    total_sec = infer_sec + stages_12_sec + stages_89_sec
    return {
        "duration_min": round(duration_sec / 60, 1),
        "chunks": chunks,
        "frames_per_chunk": frames_per_chunk,
        "yolo_calls_per_chunk": per["total_yolo_equiv"],
        "total_yolo_calls_est": per["total_yolo_equiv"] * chunks,
        "infer_hours": round(infer_sec / 3600, 2),
        "total_hours_est": round(total_sec / 3600, 2),
        "within_3h": total_sec <= 3 * 3600,
        "config": {
            "max_chunk_sec": cfg.max_chunk_sec,
            "infer_stride": stride,
            "ball_refresh": cfg.cheap_ball_refresh_every_n,
            "player_refresh": cfg.cheap_player_refresh_every_n,
            "batch_size": cfg.batch_size,
        },
    }


if __name__ == "__main__":
    import argparse
    import json

    p = argparse.ArgumentParser(description="Estimate PolyFut inference runtime")
    p.add_argument("--video", help="Probe duration from video file")
    p.add_argument("--minutes", type=float, default=94.0, help="Match length if no video")
    args = p.parse_args()

    if args.video:
        info = probe_video(args.video)
        dur = info["duration_sec"]
    else:
        dur = args.minutes * 60.0

    print(json.dumps(estimate_match_hours(dur), indent=2))
