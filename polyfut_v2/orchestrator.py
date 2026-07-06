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
from polyfut_v2.pipeline.contacts import contacts_doc, detect_contacts
from polyfut_v2.pipeline.frame_provider import VideoFrameProvider
from polyfut_v2.pipeline.hotspots import assemble_hotspots
from polyfut_v2.pipeline.montage import (
    apply_decisions,
    build_montage,
    confirmed_me_times,
    review_queue,
)
from polyfut_v2.pipeline.player_contacts import (
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
) -> dict:
    """Stages 4-9 over a finished trajectory: candidates → player+colour →
    scoring → montage → hotspots. Provider/detector are injected so this is
    testable without a real video or model. Returns the intermediate objects."""
    progress = progress or _noop

    progress(0.84, "Stage 4: contact candidates…")
    candidates = detect_contacts(traj, cfg)
    n_candidates_raw = len(candidates)
    if cfg.max_candidates and n_candidates_raw > cfg.max_candidates:
        # Keep the strongest kinematic signals; each survivor costs a player pass.
        candidates = sorted(candidates, key=lambda c: c.strength, reverse=True)[:cfg.max_candidates]
        candidates.sort(key=lambda c: c.processed_sec)  # restore time order

    progress(0.88, f"Stages 5-6: player + colour over {len(candidates)} candidate(s)…")
    enriched = enrich_contacts(candidates, provider, player_detector, seed, cfg)
    kept = filter_my_team(enriched, enabled=cfg.team_filter_enabled)

    progress(0.93, "Stage 7: appearance × orbital scoring…")
    crops = contact_crops(kept)  # captured during enrichment — no extra decode pass
    scored = score_contacts(kept, crops, seed, cfg)

    progress(0.97, "Stage 8: building review montage…")
    montage = build_montage(scored, cfg, duration_sec=duration_sec)
    if decisions:
        apply_decisions(montage, decisions)

    me_times = confirmed_me_times(montage)
    hotspots = assemble_hotspots(me_times, cfg, duration_sec=duration_sec)

    return {
        "candidates": candidates,
        "n_candidates_raw": n_candidates_raw,
        "kept": kept,
        "scored": scored,
        "montage": montage,
        "hotspots": hotspots,
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
    timings["total"] = time.perf_counter() - t_total

    # --- Write outputs ---
    montage_doc = {
        "version": 1, "step": 5,
        "source_video": str(video_path.resolve()),
        "duration_sec": duration,
        "n_candidates": len(candidates),
        "n_your_team": len(kept),
        "n_review": len(review_queue(montage)),
        "seed": {"n_samples": seed.n_samples, "has_color": seed.has_color(),
                 "gallery": len(seed.gallery)},
        "warnings": warnings,
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
        "n_your_team": len(kept),
        "n_review": len(review_queue(montage)),
        "n_hotspots": len(hotspots),
        "warnings": warnings,
        "timings_sec": timings,
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
