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
        if self.i % 7 == 0 and self.i > 0:
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
    player_detector = YoloPlayerDetector(cfg)

    if ball_detector is None and _synthetic_ball_enabled():
        ball_detector = _SyntheticBall()

    res = compute_trajectory(
        Path(video_path), cfg, detector=ball_detector,
        progress=progress, should_cancel=should_cancel,
    )
    traj = res["trajectory"]
    duration = res["info"].get("duration_sec")
    warnings = trajectory_warnings(traj, cfg)

    seed = build_seed_from_tap_specs(video_path, seed_taps, cfg, player_detector)
    if seed.n_samples == 0:
        warnings.append("no seed taps — appearance scoring is neutral")
    elif seed.is_weak():
        warnings.append("weak seed (≤1 tap) — expect a larger review montage")

    with VideoFrameProvider(video_path, target_width=cfg.target_width) as provider:
        out = assemble_touches(
            traj, duration, cfg, seed, player_detector, provider, progress=progress,
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
