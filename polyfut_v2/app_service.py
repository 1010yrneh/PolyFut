"""Server-facing glue for the v2 pipeline.

Keeps the Flask layer thin: the server calls ``run_to_montage`` in a job thread
and ``hotspots_from_decisions`` when the user submits me/not-me taps. Everything
here is plain data (JSON-serializable dicts) so results survive across requests
and can be written to / rebuilt from disk.

Uses the same ball model as v1 (COCO ``yolov8s.pt``, class 32) via the default
detectors — there is no separate ball model to copy; it is the identical file.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Callable

from polyfut_video.pipeline.decode import probe_video

from polyfut_v2.config import PipelineV2Config
from polyfut_v2.main import compute_trajectory, trajectory_warnings
from polyfut_v2.orchestrator import assemble_touches
from polyfut_v2.pipeline.frame_provider import VideoFrameProvider
from polyfut_v2.pipeline.hotspots import assemble_hotspots
from polyfut_v2.pipeline.player_detector import YoloPlayerDetector
from polyfut_v2.pipeline.seed import TargetSeed, build_seed_from_taps

ProgressCb = Callable[[float, str], None]


def _noop(frac: float, msg: str) -> None:
    pass


class _SyntheticBall:
    """Deterministic stand-in ball (patrols with periodic sharp turns → contacts).

    Env-gated demo aid: the default COCO detector can't see a small soccer ball,
    so with POLYFUT_V2_SYNTH_BALL set the pipeline uses this instead — letting the
    full review UI be exercised on real frames before a soccer ball model exists.
    """

    def __init__(self) -> None:
        self.i = 0
        self.x, self.y, self.heading = 320.0, 180.0, 0.3

    def detect(self, frame, last_center=None):
        from polyfut_v2.pipeline.ball_detector import BallDetection
        # Turn ~every 18 analysed frames so a full match doesn't fabricate
        # thousands of contacts (demo aid, not a realistic ball).
        if self.i % 18 == 0 and self.i > 0:
            self.heading += 1.9
        sp = 30.0
        self.x += sp * math.cos(self.heading)
        self.y += sp * math.sin(self.heading)
        if not (30 < self.x < 610):
            self.heading = math.pi - self.heading
            self.x = min(610.0, max(30.0, self.x))
        if not (30 < self.y < 330):
            self.heading = -self.heading
            self.y = min(330.0, max(30.0, self.y))
        self.i += 1
        return BallDetection(bbox=[self.x - 4, self.y - 4, self.x + 4, self.y + 4], conf=0.9)


def _synthetic_ball_enabled() -> bool:
    return os.environ.get("POLYFUT_V2_SYNTH_BALL", "") not in ("", "0", "false", "False")


def _footage_warnings(info: dict, cfg: PipelineV2Config) -> list[str]:
    """Warn about footage that will give poor results, with concrete numbers."""
    w = int(info.get("width") or 0)
    h = int(info.get("height") or 0)
    dur = float(info.get("duration_sec") or 0.0)
    out: list[str] = []
    if h and h < 540:
        out.append(
            f"Low-resolution footage ({w}x{h}). At this size a soccer ball is only "
            f"a few pixels and each player ~15px, so ball-detection recall is low "
            f"(typically 15-50%) and the appearance filter usually can't tell you "
            f"apart from teammates, so most touches end up in review. 720p "
            f"(1280x720) or higher gives dramatically better results.")
    if dur >= 30 * 60:
        mins = dur / 60.0
        touches = int(mins * 18)  # ball is touched ~18x/min across all 22 players
        out.append(
            f"Long footage (~{mins:.0f} min). The ball is touched ~{touches} times "
            f"across all players here; v2 analyses the whole thing but keeps only "
            f"the {cfg.max_candidates} strongest candidates and shows at most "
            f"{cfg.max_review} clips for review. Processing time scales with "
            f"length (roughly {mins * 1.7 / 60:.1f}h for this clip on a CPU).")
    return out


def _apply_soccer_model(cfg: PipelineV2Config) -> str | None:
    """Point the config at the soccer-specific ball+player model, preferring the
    OpenVINO export (≈6x faster on Intel CPUs). Downloads/exports on first use.
    Returns a warning string if it degrades to PyTorch or COCO."""
    from polyfut_v2 import ball_model as bm

    ov = bm.ensure_soccer_model_openvino()
    model, warning = (ov, None) if ov is not None else (
        bm.ensure_soccer_model(),
        "soccer model running on PyTorch (OpenVINO unavailable) — several times "
        "slower on this CPU.",
    )
    if model is not None:
        cfg.ball_weights = str(model)
        cfg.ball_class_id = bm.SOCCER_BALL_CLASS
        cfg.player_weights = str(model)
        cfg.player_class_id = bm.SOCCER_PLAYER_CLASS
        # OpenVINO export is fixed at this size; keep all passes consistent.
        cfg.ball_imgsz = cfg.ball_full_imgsz = bm.SOCCER_MODEL_IMGSZ
        cfg.player_imgsz = bm.SOCCER_MODEL_IMGSZ
        return warning
    return ("soccer ball model unavailable (offline?) — using the general COCO "
            "model, which barely detects a soccer ball. Real touches will be sparse.")


def build_seed_from_tap_specs(
    video_path: str,
    taps: list[dict] | None,
    cfg: PipelineV2Config,
    player_detector,
) -> TargetSeed:
    """Build a TargetSeed from UI taps.

    ``taps`` is a list of ``{"t_sec": float, "nx": float, "ny": float}`` where
    nx/ny are normalized coordinates in [0, 1] (fraction of frame width/height),
    so the caller needn't know the backend's resize. Returns an empty seed if no
    usable taps are given.
    """
    if not taps:
        return TargetSeed(kit_hsv=None, gallery=[], n_samples=0)
    info = probe_video(video_path)
    fps = float(info.get("fps") or 25.0)
    samples: list = []
    with VideoFrameProvider(video_path, target_width=cfg.target_width) as prov:
        for tap in taps:
            try:
                t = float(tap["t_sec"])
                nx = float(tap["nx"])
                ny = float(tap["ny"])
            except (KeyError, TypeError, ValueError):
                continue
            win = prov.window(int(round(t * fps)), 0, 1)
            if not win:
                continue
            frame = win[0][1]
            h, w = frame.shape[:2]
            samples.append((frame, (nx * w, ny * h)))
    if not samples:
        return TargetSeed(kit_hsv=None, gallery=[], n_samples=0)
    return build_seed_from_taps(
        samples, player_detector,
        max_tap_dist_px=cfg.contact_max_player_dist_px,
        min_torso_px=cfg.color_min_torso_px,
    )


def run_to_montage(
    video_path: str,
    *,
    seed_taps: list[dict] | None = None,
    cfg: PipelineV2Config | None = None,
    ball_detector=None,
    progress: ProgressCb | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict:
    """Run Stages 0-8 and return a JSON-serializable montage document.

    Hotspots here come only from auto-accepted contacts; the user refines them
    via me/not-me review, applied later with ``hotspots_from_decisions``.
    """
    cfg = cfg or PipelineV2Config()
    progress = progress or _noop

    # Upgrade ball + player detection to the soccer-specific model (COCO can't
    # see the ball). Downloads on first use; falls back to COCO if offline.
    progress(0.01, "Loading soccer detection model…")
    model_warning = _apply_soccer_model(cfg)
    player_detector = YoloPlayerDetector(cfg)

    if ball_detector is None and _synthetic_ball_enabled():
        ball_detector = _SyntheticBall()

    res = compute_trajectory(
        Path(video_path), cfg, detector=ball_detector,
        progress=progress, should_cancel=should_cancel,
    )
    traj = res["trajectory"]
    duration = res["info"].get("duration_sec")
    warnings = _footage_warnings(res["info"], cfg)
    warnings.extend(trajectory_warnings(traj, cfg))
    if model_warning:
        warnings.insert(0, model_warning)

    seed = build_seed_from_tap_specs(video_path, seed_taps, cfg, player_detector)
    if seed.n_samples == 0:
        warnings.append("no seed taps — appearance scoring is neutral")
    elif seed.is_weak():
        warnings.append("weak seed (≤1 tap) — expect a larger review montage")

    with VideoFrameProvider(video_path, target_width=cfg.target_width) as provider:
        out = assemble_touches(
            traj, duration, cfg, seed, player_detector, provider, progress=progress,
        )

    n_raw = out.get("n_candidates_raw", len(out["candidates"]))
    if n_raw > len(out["candidates"]):
        warnings.append(
            f"Stage 4 produced {n_raw} contact candidates; kept the "
            f"{len(out['candidates'])} strongest to bound processing time. A "
            f"soccer-specific ball model yields far fewer, cleaner candidates."
        )

    items = [it.to_dict() for it in out["montage"]]
    return {
        "pipeline_version": "v2",
        "duration_sec": duration,
        "n_samples": len(traj),
        "detected_ratio": round(traj.detected_ratio(), 4),
        "n_candidates": len(out["candidates"]),
        "n_your_team": len(out["kept"]),
        "n_review": sum(1 for it in items if it["status"] == "review"),
        "seed": {"n_samples": seed.n_samples, "has_color": seed.has_color(),
                 "gallery": len(seed.gallery)},
        "warnings": warnings,
        "montage": items,
        "hotspots": [h.to_dict() for h in out["hotspots"]],
    }


def hotspots_from_decisions(
    montage_items: list[dict],
    decisions: dict,
    cfg: PipelineV2Config | None = None,
    *,
    duration_sec: float | None = None,
) -> tuple[list[dict], list[dict]]:
    """Apply me/not-me decisions to montage items (by rank) and assemble
    hotspots from every 'me' touch. Returns (hotspot_dicts, updated_items)."""
    cfg = cfg or PipelineV2Config()
    # Normalize decision keys to int rank.
    norm = {}
    for k, v in (decisions or {}).items():
        try:
            norm[int(k)] = v
        except (TypeError, ValueError):
            continue

    for it in montage_items:
        if it["rank"] in norm and norm[it["rank"]] in ("me", "not_me"):
            it["decision"] = norm[it["rank"]]

    me_times = sorted(it["t_sec"] for it in montage_items if it.get("decision") == "me")
    hotspots = assemble_hotspots(me_times, cfg, duration_sec=duration_sec)
    return [h.to_dict() for h in hotspots], montage_items
