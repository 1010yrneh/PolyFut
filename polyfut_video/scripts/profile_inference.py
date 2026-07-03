"""Profile inference stage (routing ratio + projected runtime with cheap-skip)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from polyfut_video.config import PipelineConfig
from polyfut_video.main import _count_chunks, _detections_for_routed, _iter_live_chunks
from polyfut_video.pipeline.ball_smooth import BallSmoother, BallSmoothConfig
from polyfut_video.pipeline.deadtime_filter import filter_deadtime
from polyfut_video.pipeline.decode import iter_frames, probe_video
from polyfut_video.pipeline.detection import DetectConfig, Detector
from polyfut_video.pipeline.frame_router import route_frames
from polyfut_video.pipeline.shot_filter import segment_and_classify_shots


def profile(video_path: str, max_chunks: int = 3) -> dict:
    cfg = PipelineConfig()
    info = probe_video(video_path)
    frame_iter = iter_frames(
        str(video_path),
        target_width=cfg.target_width,
        sample_every_n=max(1, cfg.shot_filter_sample_every_n),
    )
    shots = segment_and_classify_shots(frame_iter, cfg)
    live_shots, _ = filter_deadtime(
        shots,
        iter_frames(
            str(video_path),
            target_width=cfg.target_width,
            sample_every_n=max(1, cfg.sample_every_n * 3),
        ),
        motion_threshold=cfg.deadtime_motion_threshold,
        min_duration_sec=cfg.deadtime_min_duration_sec,
        video_path=str(video_path),
        target_width=cfg.target_width,
    )
    chunk_count = _count_chunks(live_shots, cfg.max_chunk_sec)
    decode_stream = iter_frames(
        str(video_path),
        target_width=cfg.target_width,
        sample_every_n=max(1, cfg.infer_sample_every_n),
    )
    detector = Detector(DetectConfig(device=cfg.device, weights=cfg.yolo_weights, imgsz=cfg.imgsz))
    smoother = BallSmoother(BallSmoothConfig())

    n_full = n_cheap = 0
    new_sec = 0.0
    chunks = 0
    last_players: list = []

    for _shot, shot_frames in _iter_live_chunks(decode_stream, live_shots, cfg.max_chunk_sec):
        if chunks >= max_chunks:
            break
        routed = list(route_frames(iter(shot_frames), cfg.router_motion_threshold))
        n_full += sum(1 for r in routed if r[2] == "full")
        n_cheap += sum(1 for r in routed if r[2] == "cheap")

        t0 = time.perf_counter()
        _dets, last_players = _detections_for_routed(
            routed, detector, smoother, last_players, cfg,
        )
        new_sec += time.perf_counter() - t0
        chunks += 1

    total_frames = n_full + n_cheap
    avg_chunk = new_sec / max(chunks, 1)
    return {
        "duration_min": round(info["duration_sec"] / 60, 1),
        "total_chunks": chunk_count,
        "chunks_sampled": chunks,
        "frames_sampled": total_frames,
        "full_pct": round(100 * n_full / max(total_frames, 1), 1),
        "cheap_pct": round(100 * n_cheap / max(total_frames, 1), 1),
        "avg_chunk_sec": round(avg_chunk, 2),
        "projected_hours": round(avg_chunk * chunk_count / 3600, 2),
        "infer_stride": cfg.infer_sample_every_n,
        "cheap_refresh_every": cfg.cheap_ball_refresh_every_n,
        "batch_size": cfg.batch_size,
    }


if __name__ == "__main__":
    import argparse
    import json

    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--max-chunks", type=int, default=3)
    args = p.parse_args()
    print(json.dumps(profile(args.video, args.max_chunks), indent=2))
