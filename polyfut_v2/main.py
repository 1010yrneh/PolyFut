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
from polyfut_v2.pipeline import play_ranges as pr
from polyfut_v2.pipeline.ball_detector import YoloBallDetector
from polyfut_v2.pipeline.ball_sanity import reject_pingpong
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
            f"(< {cfg.min_detected_ratio_warn}). The ball is missed on most frames "
            f"— usually low-resolution footage, or a small/fast ball. Fewer touches "
            f"will be found; higher-resolution video helps most."
        )
    return warnings


def compute_trajectory(
    video_path: Path,
    cfg: PipelineV2Config,
    *,
    detector=None,
    progress: ProgressCb | None = None,
    should_cancel: Callable[[], bool] | None = None,
    play_ranges: list[tuple[float, float]] | None = None,
) -> dict:
    """Stages 1-3: decode → shot filter → deadtime → continuous ball trajectory.

    Shared by the Step-1 CLI and the full orchestrator. Returns a dict with
    ``info``, ``live_shots``, ``removed``, ``trajectory`` and ``timings``.
    ``detector`` defaults to :class:`YoloBallDetector` on ``cfg.ball_weights``.

    ``play_ranges`` confines all three decode passes to the user's declared
    on-pitch time (padded by ``cfg.playing_time_pad_sec``): decoding is bounded
    to the ranges' envelope, and shots are clipped to the padded ranges so the
    interior gaps between periods cost only cheap ``grab()`` calls rather than
    detection work. Timestamps stay in video time throughout. Passing None (the
    default) analyses the whole video exactly as before.
    """
    progress = progress or _noop_progress
    cancel = should_cancel or (lambda: False)
    timings: dict[str, float] = {}

    info = probe_video(str(video_path))
    v1cfg = cfg.v1_config()

    # Pad the declared window before it bounds anything: the handles are dragged
    # by eye, so a touch just outside a guessed substitution time must still be
    # analysed (recall-safety). ``padded`` empty == unrestricted.
    duration = float(info.get("duration_sec") or 0.0)
    padded = pr.pad_ranges(play_ranges, cfg.playing_time_pad_sec, duration) if play_ranges else []
    env = pr.envelope(padded)
    start_sec, end_sec = (env if env else (None, None))
    if padded:
        progress(0.01, f"Playing-time window: {pr.total_seconds(padded) / 60:.1f} min "
                       f"of {duration / 60:.1f} min analysed…")

    # --- Stages 1-2: decode + shot filter ---
    progress(0.02, f"Stage 1-2: decode + shot filter (v{PIPELINE_VERSION})…")
    t0 = time.perf_counter()
    frame_iter = iter_frames(
        str(video_path),
        target_width=cfg.target_width,
        sample_every_n=max(1, cfg.shot_filter_sample_every_n),
        start_sec=start_sec,
        end_sec=end_sec,
    )
    shots = segment_and_classify_shots(frame_iter, v1cfg)
    # Trim to the padded ranges: the envelope bound above already excluded
    # everything before/after, this splits out the interior gaps between
    # periods. Done here rather than after deadtime so the deadtime pass — which
    # falls back to its own seek-and-read when the iterator gives it no motion
    # for a shot — never touches out-of-window video either.
    if padded:
        shots = pr.intersect_shots(shots, padded)
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
        start_sec=start_sec,
        end_sec=end_sec,
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
    if padded:
        live_shots = pr.intersect_shots(live_shots, padded)
    total_live = _live_seconds(live_shots)
    progress(0.12, f"Stage 3: ball trajectory over {len(live_shots)} live shot(s)…")
    t0 = time.perf_counter()
    detector = detector or YoloBallDetector(cfg)
    decode_stream = iter_frames(
        str(video_path),
        target_width=cfg.target_width,
        sample_every_n=max(1, cfg.ball_sample_every_n),
        start_sec=start_sec,
        end_sec=end_sec,
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

    # --- Stage 3f: drop physically impossible out-and-back excursions ---
    # Needs the whole trajectory (the test looks at the sample *after* the
    # suspect one), so it cannot live in the online smoother.
    t0 = time.perf_counter()
    traj, sanity = reject_pingpong(traj, cfg)
    timings["ball_sanity"] = time.perf_counter() - t0

    return {
        "info": info,
        "live_shots": live_shots,
        "removed": removed,
        "trajectory": traj,
        "ball_sanity": sanity.to_dict(),
        "timings": timings,
        # Audit trail: the declared window and the padded one actually analysed.
        "play_ranges": [list(r) for r in (play_ranges or [])],
        "play_ranges_padded": [list(r) for r in padded],
        # ROI hit/miss + full-scan/skip counters (None for detectors without
        # stats) — makes "where did ball tracking time go" answerable from
        # job_state.json instead of a profiler.
        "ball_detector_stats": getattr(detector, "stats", None),
    }


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

    t_total = time.perf_counter()
    result = compute_trajectory(
        video_path, cfg, detector=detector, progress=progress, should_cancel=should_cancel,
    )
    info = result["info"]
    live_shots = result["live_shots"]
    removed = result["removed"]
    traj = result["trajectory"]
    timings = result["timings"]
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
