"""End-to-end Level 1 possession pipeline orchestration."""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from polyfut_video.config import PipelineConfig
from polyfut_video.pipeline.ball_smooth import BallSmoother, BallSmoothConfig
from polyfut_video.pipeline.deadtime_filter import filter_deadtime
from polyfut_video.pipeline.decode import iter_frames, probe_video
from polyfut_video.pipeline.detection import DetectConfig, Detector
from polyfut_video.pipeline.frame_router import route_frames
from polyfut_video.pipeline.possession import PossessionConfig, compute_possession
from polyfut_video.pipeline.shot_filter import PIPELINE_VERSION, segment_and_classify_shots
from polyfut_video.pipeline.team_classify import (
    TeamCropAccumulator,
    assign_teams_to_tracked_frame,
)
from polyfut_video.pipeline.timestamps import generate_timestamps, timeline_to_clip_segments
from polyfut_video.pipeline.tracking import track_shot

ProgressCb = Callable[[float, str], None]


def _noop_progress(frac: float, msg: str) -> None:
    pass


def _count_chunks(shots: list[dict], max_chunk_sec: float) -> int:
    """Exact number of bounded chunks without decoding (for progress)."""
    import math
    if max_chunk_sec <= 0:
        return max(1, len(shots))
    total = 0
    for shot in shots:
        dur = max(0.0, float(shot["end_sec"]) - float(shot["start_sec"]))
        total += max(1, math.ceil(dur / max_chunk_sec))
    return max(1, total)


def _iter_live_chunks(
    frame_stream: Iterator[tuple[int, float, np.ndarray]],
    live_shots: list[dict],
    max_chunk_sec: float,
) -> Iterator[tuple[dict, list[tuple[int, float, "np.ndarray"]]]]:
    """Single decode pass → bounded (shot, frames) chunks.

    Decodes the video exactly once. Frames are grouped into the live shot
    that contains them and flushed whenever a chunk reaches max_chunk_sec or
    a shot boundary is crossed. This avoids the previous O(n²) behaviour of
    re-opening and skipping the video from the start for every chunk.
    """
    shots = sorted(live_shots, key=lambda s: float(s["start_sec"]))
    idx = 0
    buf: list[tuple[int, float, np.ndarray]] = []
    cur_shot: dict | None = None
    chunk_start_t: float | None = None

    def _flush():
        nonlocal buf, cur_shot, chunk_start_t
        if buf and cur_shot is not None:
            out = (cur_shot, buf)
            buf = []
            chunk_start_t = None
            return out
        return None

    for fi, t_sec, frame in frame_stream:
        while idx < len(shots) and t_sec > float(shots[idx]["end_sec"]):
            flushed = _flush()
            if flushed:
                yield flushed
            idx += 1
        if idx >= len(shots):
            break
        shot = shots[idx]
        if t_sec < float(shot["start_sec"]):
            continue  # in a removed gap between live shots
        if cur_shot is not None and shot is not cur_shot:
            flushed = _flush()
            if flushed:
                yield flushed
        cur_shot = shot
        if chunk_start_t is None:
            chunk_start_t = t_sec
        buf.append((fi, t_sec, frame))
        if max_chunk_sec > 0 and (t_sec - chunk_start_t) >= max_chunk_sec:
            flushed = _flush()
            if flushed:
                yield flushed

    flushed = _flush()
    if flushed:
        yield flushed


def _apply_ball_smooth(
    dets: list[dict],
    smoother: BallSmoother,
) -> list[dict]:
    players = [dict(d) for d in dets if d.get("class") == "player"]
    balls = [d for d in dets if d.get("class") == "ball"]
    best = max(balls, key=lambda b: b.get("conf", 0)) if balls else None
    bbox, conf, _held = smoother.update(
        best["bbox"] if best else None,
        float(best["conf"]) if best else 0.0,
    )
    out = list(players)
    if bbox is not None:
        out.append({"class": "ball", "bbox": bbox, "conf": conf})
    return out


def _detections_for_routed(
    routed: list[tuple[int, np.ndarray, str, float]],
    detector: Detector,
    smoother: BallSmoother,
    last_players: list[dict],
    cfg: PipelineConfig,
) -> tuple[list[list[dict]], list[dict]]:
    """Batched full detect + scheduled cheap refresh for players/ball."""
    detections_per_frame: list[list[dict]] = []
    cheap_streak = 0
    batch_size = max(1, cfg.batch_size)
    ball_every = max(1, cfg.cheap_ball_refresh_every_n)
    player_every = max(ball_every, cfg.cheap_player_refresh_every_n)
    i = 0

    while i < len(routed):
        _fi, frame, route, _t = routed[i]
        if route == "full":
            cheap_streak = 0
            batch_frames: list[np.ndarray] = []
            while i < len(routed) and routed[i][2] == "full":
                batch_frames.append(routed[i][1])
                i += 1
            for start in range(0, len(batch_frames), batch_size):
                chunk = batch_frames[start : start + batch_size]
                for dets in detector.detect_frames_batch(chunk):
                    last_players = [dict(d) for d in dets if d.get("class") == "player"]
                    detections_per_frame.append(_apply_ball_smooth(dets, smoother))
            continue

        cheap_streak += 1
        i += 1
        if not last_players or cheap_streak >= player_every:
            dets = detector.detect_frame(frame)
            last_players = [dict(d) for d in dets if d.get("class") == "player"]
            cheap_streak = 0
        elif cheap_streak >= ball_every:
            cheap_streak = 0
            dets = detector.merge_players_and_ball(last_players, frame)
        else:
            dets = [dict(p) for p in last_players]
        detections_per_frame.append(_apply_ball_smooth(dets, smoother))

    return detections_per_frame, last_players


def run_pipeline(
    video_path: str | Path,
    out_dir: str | Path | None = None,
    *,
    cfg: PipelineConfig | None = None,
    my_team: str = "team_a",
    progress: ProgressCb | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict:
    """
    Run the full 9-stage pipeline sequentially.

    Writes output/possession_timeline.json and clip_segments.json (filtered to my_team).
    Returns metadata dict with paths and per-stage timings.
    """
    cfg = cfg or PipelineConfig()
    video_path = Path(video_path)
    out_dir = Path(out_dir or cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress = progress or _noop_progress
    cancel = should_cancel or (lambda: False)

    timings: dict[str, float] = {}
    t_total = time.perf_counter()

    info = probe_video(str(video_path))
    fps = info["fps"]

    # --- Stage 1 + 2: decode and shot filter ---
    progress(0.02, f"Stage 1–2: decode + shot filter (v{PIPELINE_VERSION})…")
    t0 = time.perf_counter()
    frame_iter = iter_frames(
        str(video_path),
        target_width=cfg.target_width,
        sample_every_n=max(1, cfg.shot_filter_sample_every_n),
    )
    shots = segment_and_classify_shots(frame_iter, cfg)
    n_main = sum(1 for s in shots if s.get("label") == "main_camera")
    timings["shot_filter"] = time.perf_counter() - t0
    progress(0.06, f"Stage 1–2: {len(shots)} shot(s), {n_main} main_camera…")
    if cancel():
        raise RuntimeError("cancelled")

    # --- Stage 3: deadtime ---
    progress(0.08, "Stage 3: deadtime filter…")
    t0 = time.perf_counter()
    frame_iter2 = iter_frames(
        str(video_path),
        target_width=cfg.target_width,
        sample_every_n=max(1, cfg.sample_every_n * 8),
    )
    live_shots, removed = filter_deadtime(
        shots,
        frame_iter2,
        motion_threshold=cfg.deadtime_motion_threshold,
        min_duration_sec=cfg.deadtime_min_duration_sec,
        video_path=str(video_path),
        target_width=cfg.target_width,
    )
    timings["deadtime"] = time.perf_counter() - t0
    if cancel():
        raise RuntimeError("cancelled")

    # --- Stages 4–7 per live shot ---
    progress(0.12, f"Stages 4–7: {len(live_shots)} live shot(s)…")
    detector = Detector(DetectConfig(
        weights=cfg.yolo_weights,
        conf_threshold=cfg.conf_threshold,
        ball_conf_min=cfg.ball_conf_min,
        device=cfg.device,
        imgsz=cfg.imgsz,
        ball_imgsz=cfg.ball_imgsz,
    ))

    all_tracked_records: list[dict] = []
    processed_sec_cursor = 0.0
    chunk_count = _count_chunks(live_shots, cfg.max_chunk_sec)
    infer_step = max(1, cfg.infer_sample_every_n)

    # Single decode pass over the whole video, grouped into bounded chunks.
    decode_stream = iter_frames(
        str(video_path),
        target_width=cfg.target_width,
        sample_every_n=infer_step,
    )

    t_infer = time.perf_counter()
    si = 0
    ball_smoother = BallSmoother(BallSmoothConfig(
        max_hold_frames=cfg.ball_hold_frames,
        max_jump_px=cfg.ball_max_jump_px,
    ))
    last_shot_start: float | None = None
    last_players: list[dict] = []
    active_tracker: object | None = None
    team_acc = TeamCropAccumulator(max_crops_per_track=cfg.team_crops_per_track)

    for shot, shot_frames in _iter_live_chunks(decode_stream, live_shots, cfg.max_chunk_sec):
        if cancel():
            raise RuntimeError("cancelled")
        progress(
            0.12 + 0.70 * (si / chunk_count),
            f"Chunk {si + 1}/{chunk_count}: routing + detect + track…",
        )
        si += 1

        shot_start = float(shot["start_sec"])
        if last_shot_start is None or shot_start != last_shot_start:
            ball_smoother.reset()
            last_players = []
            active_tracker = None
            team_acc.reset()
            last_shot_start = shot_start

        routed = list(route_frames(
            iter(shot_frames),
            cfg.router_motion_threshold,
            downscale=cfg.router_downscale,
        ))
        if not routed:
            continue
        chunk_start_t = routed[0][3]

        detections_per_frame, last_players = _detections_for_routed(
            routed, detector, ball_smoother, last_players, cfg,
        )

        tracked_per_frame, active_tracker = track_shot(
            detections_per_frame,
            track_thresh=cfg.track_thresh,
            match_thresh=cfg.match_thresh,
            tracker=active_tracker,
        )

        for frame, dets in zip([r[1] for r in routed], tracked_per_frame):
            team_acc.observe(frame, dets)
        team_labels = team_acc.team_labels(
            eps=cfg.dbscan_eps,
            min_samples=cfg.dbscan_min_samples,
            min_cluster_size=cfg.min_cluster_size,
        )

        for (frame_idx, frame, route, t_sec), dets in zip(routed, tracked_per_frame):
            classified = assign_teams_to_tracked_frame(frame, dets, team_labels)
            proc_t = processed_sec_cursor + max(0.0, t_sec - chunk_start_t)
            all_tracked_records.append({
                "frame_index": frame_idx,
                "timestamp_sec": t_sec,
                "processed_sec": proc_t,
                "detections": classified,
                "route": route,
            })
        processed_sec_cursor += max(0.0, routed[-1][3] - chunk_start_t)

        del routed, detections_per_frame, tracked_per_frame, shot_frames
        if si % 25 == 0:
            gc.collect()

    timings["inference"] = time.perf_counter() - t_infer
    if cancel():
        raise RuntimeError("cancelled")

    # --- Stage 8: possession ---
    progress(0.85, "Stage 8: possession smoothing…")
    t0 = time.perf_counter()
    possession_cfg = PossessionConfig(
        base_thresh_px=cfg.possession_base_thresh_px,
        ref_person_h=cfg.possession_ref_person_h,
        ref_ball_diag=cfg.possession_ref_ball_diag,
        min_thresh_px=cfg.possession_min_thresh_px,
        max_thresh_px=cfg.possession_max_thresh_px,
        on_frames=cfg.possession_on_frames,
        off_frames=cfg.possession_off_frames,
        contested_margin=cfg.contested_margin,
        window_size_sec=cfg.possession_window_sec,
    )
    analysed_fps = fps / max(1, infer_step)
    possession_frames = compute_possession(
        all_tracked_records,
        window_size_sec=cfg.possession_window_sec,
        contested_margin=cfg.contested_margin,
        fps=analysed_fps,
        cfg=possession_cfg,
    )
    timings["possession"] = time.perf_counter() - t0

    # --- Stage 9: timestamps ---
    progress(0.92, "Stage 9: building touch hotspots…")
    t0 = time.perf_counter()
    timeline = generate_timestamps(possession_frames, removed)
    timings["timestamps"] = time.perf_counter() - t0

    timeline_path = out_dir / "possession_timeline.json"
    timeline_doc = {
        "version": 1,
        "source_video": str(video_path.resolve()),
        "my_team_default": my_team,
        "removed_segments": removed,
        "intervals": timeline,
        "stage_timings_sec": {k: round(v, 2) for k, v in timings.items()},
    }
    timeline_path.write_text(json.dumps(timeline_doc, indent=2), encoding="utf-8")

    segments = timeline_to_clip_segments(
        timeline,
        my_team=my_team,
        pad_before=cfg.hotspot_pad_before_sec,
        pad_after=cfg.hotspot_pad_after_sec,
        gap_merge=cfg.hotspot_gap_merge_sec,
        min_zone_sec=cfg.hotspot_min_zone_sec,
        duration_sec=info.get("duration_sec"),
    )
    seg_path = out_dir / "clip_segments.json"
    seg_path.write_text(json.dumps({
        "version": 3,
        "kind": "touch_hotspots",
        "source_video": str(video_path.resolve()),
        "my_team": my_team,
        "possession_timeline": str(timeline_path.name),
        "segments": segments,
    }, indent=2), encoding="utf-8")

    timings["total"] = time.perf_counter() - t_total
    progress(1.0, "Done")

    return {
        "source_video": str(video_path.resolve()),
        "possession_timeline_path": str(timeline_path),
        "clip_segments_path": str(seg_path),
        "n_intervals": len(timeline),
        "n_segments": len(segments),
        "live_shots": len(live_shots),
        "removed_segments": len(removed),
        "timings_sec": timings,
    }


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="PolyFut Level 1 possession pipeline")
    p.add_argument("--video", required=True, help="Input match video")
    p.add_argument("--out", default="output", help="Output directory")
    p.add_argument("--my-team", default="team_a", choices=["team_a", "team_b"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--weights", default="yolov8s.pt")
    args = p.parse_args()

    cfg = PipelineConfig(yolo_weights=args.weights, device=args.device)

    def _prog(frac: float, msg: str) -> None:
        print(f"[{frac * 100:5.1f}%] {msg}")

    meta = run_pipeline(args.video, args.out, cfg=cfg, my_team=args.my_team, progress=_prog)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
