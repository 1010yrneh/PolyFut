"""Full v2 orchestrator — wires Stages 0-9 end to end.

    Stage 1-3  ball trajectory              (continuous)
    Stage 4    kinematic contact candidates
    Stage 0    target seed (taps → gallery + kit colour)
    Stage 5-6  sparse player detect + team colour filter   (per candidate)
    Stage 7    appearance × orbital confidence scoring
    Stage 8    ranked review montage
    Stage 9    hotspot assembly

Writes ``montage.json`` (for the review UI) and ``hotspots.json``. With no human
decisions supplied, only auto-accepted contacts become hotspots and the rest are
left in the montage's review queue — the honest automated output.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

from polyfut_v2.config import PipelineV2Config
from polyfut_v2.main import compute_trajectory, trajectory_warnings
from polyfut_v2.pipeline.contacts import cap_candidates, contacts_doc, detect_contacts
from polyfut_v2.pipeline.frame_provider import VideoFrameProvider
from polyfut_v2.pipeline.hotspots import assemble_hotspots
from polyfut_v2.pipeline.montage import (
    apply_decisions,
    build_montage,
    confirmed_me_times,
    review_queue,
)
from polyfut_v2.pipeline.player_contacts import (
    TEAM_OFFICIAL,
    TEAM_SIDELINE,
    contact_crops,
    enrich_contacts,
    filter_my_team,
)
from polyfut_v2.pipeline.player_detector import YoloPlayerDetector
from polyfut_v2.pipeline.scoring import score_contacts
from polyfut_v2.pipeline.seed import TargetSeed, build_seed_from_taps

ProgressCb = Callable[[float, str], None]


def _noop(frac: float, msg: str) -> None:
    pass


def assemble_touches(
    traj,
    duration_sec,
    cfg: PipelineV2Config,
    seed: TargetSeed,
    player_detector,
    provider,
    *,
    decisions: dict[int, str] | None = None,
    progress: ProgressCb | None = None,
    budget_sec: float | None = None,
    calibration=None,
) -> dict:
    """Stages 4-9 over a finished trajectory: candidates → player+colour →
    scoring → montage → hotspots. Provider/detector are injected so this is
    testable without a real video or model. Returns the intermediate objects
    plus per-stage ``timings`` (seconds).

    ``budget_sec`` scales the candidate cap and defaults to ``duration_sec``.
    With a playing-time window it is the *on-pitch* seconds instead: the cap is
    a "how many touches could plausibly be in this footage" budget, so 20
    minutes of declared playing time inside a 111-minute video should get a
    20-minute budget. ``duration_sec`` itself stays full video duration —
    montage clip ends and hotspot ends clamp against it and timestamps never
    leave video time."""
    progress = progress or _noop
    timings: dict[str, float] = {}

    progress(0.84, "Stage 4: contact candidates…")
    t0 = time.perf_counter()
    candidates = detect_contacts(traj, cfg)
    n_candidates_raw = len(candidates)
    # Duration-scaled, time-fair budget: each survivor costs a player pass.
    candidates = cap_candidates(
        candidates, cfg,
        duration_sec=(budget_sec if budget_sec is not None else duration_sec),
    )
    timings["contacts"] = time.perf_counter() - t0

    # Stage 5b: when the user calibrated the pitch, judge "who touched it" in
    # metres rather than pixels. Silently absent otherwise — see Issue 14.
    mapper, calibration_status = _build_pitch_mapper(traj, cfg, calibration)

    progress(0.88, f"Stages 5-6: player + colour over {len(candidates)} candidate(s)…")
    t0 = time.perf_counter()
    enriched = enrich_contacts(candidates, provider, player_detector, seed, cfg,
                               mapper=mapper)
    # Always drop contacts whose kit is CLEARLY the other team, or who were
    # attributed to an official / sideline coach — removes "wrong team / ref /
    # coach touched the ball" clips without the aggressive same-team filter.
    # Colour-undecided (unknown) contacts survive for recall.
    # A crowded contact (corner kick / scramble) is exempt: with several bodies
    # on the ball the kit we read may not be the toucher's, so dropping it would
    # risk losing a real touch. Officials and sideline staff still go.
    keep_crowded = bool(getattr(cfg, "crowd_keep_other_team", False))
    n_before = len(enriched)
    enriched = [
        c for c in enriched
        if c.is_my_team is not False
        or (keep_crowded and c.crowded
            and c.team_label not in (TEAM_OFFICIAL, TEAM_SIDELINE))
    ]
    n_other_dropped = n_before - len(enriched)
    kept = filter_my_team(
        enriched, keep_crowded=keep_crowded, enabled=cfg.team_filter_enabled,
    )
    n_crowded = sum(1 for c in kept if c.crowded)
    timings["player_enrich"] = time.perf_counter() - t0

    progress(0.93, "Stage 7: appearance × orbital scoring…")
    t0 = time.perf_counter()
    crops = contact_crops(kept)  # captured during enrichment — no extra decode pass
    # Compare positions in pan-stabilised space when Stage 3 measured the camera
    # (Issue 5); without it the orbital prior reads a pan as a teleport.
    camera = getattr(traj, "camera", None)
    transform = camera.transform() if camera is not None else None
    scored = score_contacts(kept, crops, seed, cfg, transform=transform)
    timings["scoring"] = time.perf_counter() - t0

    progress(0.97, "Stage 8: building review montage…")
    t0 = time.perf_counter()
    montage = build_montage(scored, cfg, duration_sec=duration_sec)
    if decisions:
        apply_decisions(montage, decisions)

    me_times = confirmed_me_times(montage)
    hotspots = assemble_hotspots(me_times, cfg, duration_sec=duration_sec)
    timings["montage"] = time.perf_counter() - t0

    return {
        "candidates": candidates,
        "n_candidates_raw": n_candidates_raw,
        "attribution": (mapper.stats() if mapper is not None
                        else {"contacts_metric": 0,
                              "contacts_pixel_fallback": len(candidates),
                              "metric_coverage": 0.0}),
        # Which of _build_pitch_mapper's exits fired. "ok" means the map was
        # built; anything else explains a metric_coverage of 0.
        "calibration_status": calibration_status,
        "n_other_team_dropped": n_other_dropped,
        "n_crowded": n_crowded,
        "kept": kept,
        "scored": scored,
        "montage": montage,
        "hotspots": hotspots,
        "timings": timings,
    }


def run_v2(
    video_path: str | Path,
    out_dir: str | Path | None = None,
    *,
    cfg: PipelineV2Config | None = None,
    seed: TargetSeed | None = None,
    seed_taps: list | None = None,
    ball_detector=None,
    player_detector=None,
    decisions: dict[int, str] | None = None,
    progress: ProgressCb | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict:
    """Run the whole pipeline. ``seed`` (or ``seed_taps`` = [(frame, (x,y)), …])
    supplies the target; detectors default to the pluggable YOLO wrappers."""
    cfg = cfg or PipelineV2Config()
    video_path = Path(video_path)
    out_dir = Path(out_dir or cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress = progress or _noop
    t_total = time.perf_counter()

    player_detector = player_detector or YoloPlayerDetector(cfg)

    # --- Stages 1-3: ball trajectory ---
    traj_res = compute_trajectory(
        video_path, cfg, detector=ball_detector,
        progress=progress, should_cancel=should_cancel,
    )
    traj = traj_res["trajectory"]
    info = traj_res["info"]
    duration = info.get("duration_sec")
    warnings = trajectory_warnings(traj, cfg)

    # --- Stage 0: seed ---
    t0 = time.perf_counter()
    if seed is None:
        if seed_taps:
            seed = build_seed_from_taps(
                seed_taps, player_detector,
                max_tap_dist_px=cfg.contact_max_player_dist_px,
                min_torso_px=cfg.color_min_torso_px,
            )
        else:
            seed = TargetSeed(kit_hsv=None, gallery=[], n_samples=0)
            warnings.append("no seed provided — appearance scoring degrades to neutral")
    if seed.is_weak():
        warnings.append("weak seed (≤1 sample) — expect a larger review montage")
    seed_sec = time.perf_counter() - t0

    # --- Stages 4-9 (need frame access) ---
    with VideoFrameProvider(str(video_path), target_width=cfg.target_width) as provider:
        res = assemble_touches(
            traj, duration, cfg, seed, player_detector, provider,
            decisions=decisions, progress=progress,
        )
    candidates, kept, montage, hotspots = (
        res["candidates"], res["kept"], res["montage"], res["hotspots"],
    )
    me_times = confirmed_me_times(montage)

    timings = dict(traj_res["timings"])
    timings["seed"] = seed_sec
    timings.update(res.get("timings") or {})
    timings["total"] = time.perf_counter() - t_total
    stage_timings = {k: round(v, 2) for k, v in timings.items()}

    # --- Write outputs ---
    montage_doc = {
        "version": 1, "step": 5,
        "source_video": str(video_path.resolve()),
        "duration_sec": duration,
        "n_candidates": len(candidates),
        "n_your_team": len(kept),
        "n_review": len(review_queue(montage)),
        "n_crowded": res.get("n_crowded", 0),
        "seed": {"n_samples": seed.n_samples, "has_color": seed.has_color(),
                 "gallery": len(seed.gallery)},
        "warnings": warnings,
        "stage_timings_sec": stage_timings,
        "items": [it.to_dict() for it in montage],
    }
    (out_dir / "montage.json").write_text(json.dumps(montage_doc, indent=2), encoding="utf-8")

    hotspots_doc = {
        "version": 1, "step": 5,
        "source_video": str(video_path.resolve()),
        "n_hotspots": len(hotspots),
        "n_confirmed_me": len(me_times),
        "pending_review": len(review_queue(montage)),
        "hotspots": [h.to_dict() for h in hotspots],
    }
    (out_dir / "hotspots.json").write_text(json.dumps(hotspots_doc, indent=2), encoding="utf-8")

    (out_dir / "contacts.json").write_text(
        json.dumps(contacts_doc(candidates, source={"source_video": str(video_path.resolve())}),
                   indent=2), encoding="utf-8")

    progress(1.0, "Done")
    return {
        "source_video": str(video_path.resolve()),
        "n_samples": len(traj),
        "detected_ratio": round(traj.detected_ratio(), 4),
        "n_candidates": len(candidates),
        "n_candidates_raw": res.get("n_candidates_raw", len(candidates)),
        "n_your_team": len(kept),
        "n_review": len(review_queue(montage)),
        "n_crowded": res.get("n_crowded", 0),
        "n_hotspots": len(hotspots),
        "ball_detector_stats": traj_res.get("ball_detector_stats"),
        "ball_sanity": traj_res.get("ball_sanity"),
        "warnings": warnings,
        "timings_sec": timings,
        "stage_timings_sec": stage_timings,
        "out_dir": str(out_dir),
    }


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="PolyFut v2 — full pipeline (Stages 0-9)")
    p.add_argument("--video", required=True)
    p.add_argument("--out", default="output_v2")
    p.add_argument("--device", default="cpu")
    p.add_argument("--ball-weights", default="yolov8s.pt")
    p.add_argument("--player-weights", default="yolov8s.pt")
    args = p.parse_args()

    cfg = PipelineV2Config(
        ball_weights=args.ball_weights, player_weights=args.player_weights, device=args.device,
    )

    def _prog(frac: float, msg: str) -> None:
        print(f"[{frac * 100:5.1f}%] {msg}")

    meta = run_v2(args.video, args.out, cfg=cfg, progress=_prog)
    print(json.dumps(meta, indent=2))
    for w in meta.get("warnings", []):
        print(f"\n[!] {w}")


if __name__ == "__main__":
    main()


def _build_pitch_mapper(traj, cfg, calibration):
    """PitchMapper for this run, or None to keep today's pixel behaviour.

    Refuses rather than guesses at every step: no calibration, a calibration the
    quality screen rejects, or no camera track all yield None. A wrong pitch map
    silently moves players to the wrong place, which is worse than not having
    one, so every uncertainty resolves to "fall back".

    Returns ``(mapper, reason)``. The reason exists because the fallback used to
    be *entirely* silent: a real run was measured at ``metric_coverage: 0.0``
    with every one of its 179 contacts on the pixel gate, and nothing recorded
    which of the five exits below had fired — a skipped screen, a rejected fit
    and a broken camera track all produced byte-identical output. Falling back
    quietly is still right; being unable to say why afterwards is not.
    """
    if calibration is None:
        return None, "not_supplied"
    try:
        from polyfut_v2.pipeline.pitch_calibration import (
            PitchCalibration,
            PitchMapper,
        )
        calib = (calibration if isinstance(calibration, PitchCalibration)
                 else PitchCalibration.from_dict(calibration))
    except Exception:
        return None, "unreadable"
    limit = float(getattr(cfg, "calibration_max_median_px", 6.0))
    if limit > 0 and calib.median_px > limit:
        return None, "quality_rejected"
    camera = getattr(traj, "camera", None)
    if camera is None:
        return None, "no_camera_track"
    try:
        return PitchMapper(calib, camera, frame_width=int(cfg.target_width)), "ok"
    except Exception:
        return None, "mapper_failed"


# Human-readable form of the reasons above, for the run warnings. Only the
# genuine failures appear here: "not_supplied" is a user choice, not a fault.
CALIBRATION_FALLBACK_MESSAGES = {
    "unreadable": ("The pitch calibration could not be read, so touches were "
                   "attributed in pixels instead of metres."),
    "quality_rejected": ("The pitch calibration was rejected as too imprecise "
                         "(typical landmark error above "
                         "{limit:.0f}px), so touches were attributed in pixels "
                         "instead of metres."),
    "no_camera_track": ("No camera-motion track was available, so the pitch "
                        "calibration could not be carried across the video and "
                        "touches were attributed in pixels."),
    "mapper_failed": ("The pitch map could not be built from the calibration, "
                      "so touches were attributed in pixels."),
}
