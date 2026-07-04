"""v2 orchestrator — Step 1: continuous ball trajectory (Stages 1-3).

Later steps extend this: Stage 4 consumes the trajectory to flag contact
candidates, and so on. For now the deliverable is a dense, reviewable
``ball_trajectory.json``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

from polyfut_video.pipeline.deadtime_filter import filter_deadtime
from polyfut_video.pipeline.decode import iter_frames, probe_video
from polyfut_video.pipeline.shot_filter import PIPELINE_VERSION, segment_and_classify_shots

from polyfut_v2.config import PipelineV2Config
from polyfut_v2.pipeline.ball_detector import YoloBallDetector
from polyfut_v2.pipeline.ball_tracker import track_ball
from polyfut_v2.pipeline.trajectory import BallTrajectory

ProgressCb = Callable[[float, str], None]


def _noop_progress(frac: float, msg: str) -> None:
    pass


def _live_seconds(live_shots: list[dict]) -> float:
    return sum(max(0.0, float(s["end_sec"]) - float(s["start_sec"])) for s in live_shots)


def trajectory_warnings(traj: BallTrajectory, cfg: PipelineV2Config) -> list[str]:
    """Flag reliability problems in a finished trajectory.

    Kept pure (no video / model) so it can be unit-tested and reused by the
    server later. The key one is near-zero ball recall: the design doc's only
    *silent* failure mode, so it must be surfaced, not buried in JSON.
    """
    warnings: list[str] = []
    if len(traj) == 0:
        warnings.append(
            "empty trajectory: no live frames were analysed — check the shot "
            "filter / deadtime stages (nothing classified as main_camera?)."
        )
        return warnings
    dr = traj.detected_ratio()
    if dr < cfg.min_detected_ratio_warn:
        warnings.append(
            f"low ball-detection recall: detected_ratio={dr:.3f} "
            f"(< {cfg.min_detected_ratio_warn}). The default COCO ball model has "
            f"near-zero recall on small/distant balls; plug in a soccer-specific "
            f"ball model via cfg.ball_weights + cfg.ball_class_id. Every "
            f"downstream v2 stage is starved until this is fixed."
        )
    return warnings


def run_ball_trajectory(
    video_path: str | Path,
    out_dir: str | Path | None = None,
    *,
    cfg: PipelineV2Config | None = None,
    detector=None,
    progress: ProgressCb | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict:
    """Run Stages 1-3 and write ``ball_trajectory.json``.

    ``detector`` may be injected (e.g. a soccer-specific model or a test double);
    it defaults to :class:`YoloBallDetector` on ``cfg.ball_weights``.
    """
    cfg = cfg or PipelineV2Config()
    video_path = Path(video_path)
    out_dir = Path(out_dir or cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress = progress or _noop_progress
    cancel = should_cancel or (lambda: False)

    timings: dict[str, float] = {}
    t_total = time.perf_counter()

    info = probe_video(str(video_path))
    v1cfg = cfg.v1_config()

    # --- Stages 1-2: decode + shot filter ---
    progress(0.02, f"Stage 1-2: decode + shot filter (v{PIPELINE_VERSION})…")
    t0 = time.perf_counter()
    frame_iter = iter_frames(
        str(video_path),
        target_width=cfg.target_width,
        sample_every_n=max(1, cfg.shot_filter_sample_every_n),
    )
    shots = segment_and_classify_shots(frame_iter, v1cfg)
    n_main = sum(1 for s in shots if s.get("label") == "main_camera")
    timings["shot_filter"] = time.perf_counter() - t0
    progress(0.06, f"Stage 1-2: {len(shots)} shot(s), {n_main} main_camera…")
    if cancel():
        raise RuntimeError("cancelled")

    # --- Stage 3a: deadtime ---
    progress(0.08, "Stage 3a: deadtime filter…")
    t0 = time.perf_counter()
    frame_iter2 = iter_frames(
        str(video_path),
        target_width=cfg.target_width,
        sample_every_n=max(1, cfg.shot_filter_sample_every_n * 8),
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

    # --- Stage 3b-c: continuous ball trajectory ---
    total_live = _live_seconds(live_shots)
    progress(0.12, f"Stage 3: ball trajectory over {len(live_shots)} live shot(s)…")
    t0 = time.perf_counter()
    detector = detector or YoloBallDetector(cfg)
    decode_stream = iter_frames(
        str(video_path),
        target_width=cfg.target_width,
        sample_every_n=max(1, cfg.ball_sample_every_n),
    )
    traj: BallTrajectory = track_ball(
        decode_stream,
        live_shots,
        detector,
        cfg,
        progress=progress,
        should_cancel=should_cancel,
        total_live_sec=total_live,
    )
    timings["ball_tracking"] = time.perf_counter() - t0
    timings["total"] = time.perf_counter() - t_total

    warnings = trajectory_warnings(traj, cfg)
    for w in warnings:
        progress(1.0, f"WARNING: {w}")

    # --- Write trajectory ---
    traj_doc = {
        "version": 1,
        "step": 1,
        "source_video": str(video_path.resolve()),
        "fps": info["fps"],
        "duration_sec": info.get("duration_sec"),
        "ball_sample_every_n": cfg.ball_sample_every_n,
        "removed_segments": removed,
        "live_shots": [
            {"start_sec": float(s["start_sec"]), "end_sec": float(s["end_sec"])}
            for s in live_shots
        ],
        "trajectory": traj.to_dict(),
        "warnings": warnings,
        "stage_timings_sec": {k: round(v, 2) for k, v in timings.items()},
    }
    traj_path = out_dir / "ball_trajectory.json"
    traj_path.write_text(json.dumps(traj_doc, indent=2), encoding="utf-8")

    progress(1.0, "Done")

    return {
        "source_video": str(video_path.resolve()),
        "ball_trajectory_path": str(traj_path),
        "n_samples": len(traj),
        "detected_ratio": round(traj.detected_ratio(), 4),
        "live_shots": len(live_shots),
        "removed_segments": len(removed),
        "warnings": warnings,
        "timings_sec": timings,
    }


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="PolyFut v2 — Step 1: ball trajectory")
    p.add_argument("--video", required=True, help="Input match video")
    p.add_argument("--out", default="output_v2", help="Output directory")
    p.add_argument("--device", default="cpu")
    p.add_argument("--weights", default="yolov8s.pt", help="Ball detector weights")
    args = p.parse_args()

    cfg = PipelineV2Config(ball_weights=args.weights, device=args.device)

    def _prog(frac: float, msg: str) -> None:
        print(f"[{frac * 100:5.1f}%] {msg}")

    meta = run_ball_trajectory(args.video, args.out, cfg=cfg, progress=_prog)
    print(json.dumps(meta, indent=2))
    for w in meta.get("warnings", []):
        print(f"\n[!] {w}")


if __name__ == "__main__":
    main()
