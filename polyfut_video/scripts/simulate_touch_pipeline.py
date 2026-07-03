"""
End-to-end simulation of stages 4–9 wiring (no YOLO).

Verifies: ball smooth → track passthrough → team assign → possession → hotspots.
Run: python polyfut_video/scripts/simulate_touch_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from polyfut_video.config import PipelineConfig
from polyfut_video.pipeline.ball_smooth import BallSmoother, BallSmoothConfig
from polyfut_video.pipeline.possession import PossessionConfig, compute_possession
from polyfut_video.pipeline.timestamps import generate_timestamps, timeline_to_clip_segments


def _mock_chunk_frames(cfg: PipelineConfig) -> list[dict]:
    """Simulate two 30s chunks with ball flicker and distant small ball."""
    smoother = BallSmoother(BallSmoothConfig(
        max_hold_frames=cfg.ball_hold_frames,
        max_jump_px=cfg.ball_max_jump_px,
    ))
    records: list[dict] = []
    proc_t = 0.0
    analysed_fps = 30.0 / max(1, cfg.infer_sample_every_n)

    for chunk in range(2):
        if chunk == 1:
            smoother.reset()  # should NOT happen in production across same shot
        for i in range(20):
            t = chunk * 30.0 + i / analysed_fps
            # Distant player (small box) + intermittent ball
            players = [{"class": "player", "team_id": 0, "track_id": 1,
                        "bbox": [200, 150, 220, 200]}]
            ball_det = None
            if i % 4 != 3:  # miss every 4th frame
                ball_det = {"class": "ball", "bbox": [208, 196, 212, 200], "conf": 0.15}

            dets_in = list(players)
            if ball_det:
                dets_in.append(ball_det)
            balls = [d for d in dets_in if d["class"] == "ball"]
            best = balls[0] if balls else None
            bbox, conf, held = smoother.update(
                best["bbox"] if best else None,
                float(best["conf"]) if best else 0.0,
            )
            dets_out = list(players)
            if bbox:
                dets_out.append({"class": "ball", "bbox": bbox, "conf": conf})

            records.append({
                "frame_index": chunk * 20 + i,
                "timestamp_sec": t,
                "processed_sec": proc_t + i / analysed_fps,
                "detections": dets_out,
            })
        proc_t += 20 / analysed_fps

    return records


def _mock_chunk_frames_continuous_smoother(cfg: PipelineConfig) -> list[dict]:
    """Same as above but smoother persists across chunk boundary (correct wiring)."""
    smoother = BallSmoother(BallSmoothConfig(
        max_hold_frames=cfg.ball_hold_frames,
        max_jump_px=cfg.ball_max_jump_px,
    ))
    records: list[dict] = []
    proc_t = 0.0
    analysed_fps = 30.0 / max(1, cfg.infer_sample_every_n)

    for chunk in range(2):
        for i in range(20):
            t = chunk * 30.0 + i / analysed_fps
            players = [{"class": "player", "team_id": 0, "track_id": 1,
                        "bbox": [200, 150, 220, 200]}]
            ball_det = None
            if i % 4 != 3:
                ball_det = {"class": "ball", "bbox": [208, 196, 212, 200], "conf": 0.15}

            dets_in = list(players)
            if ball_det:
                dets_in.append(ball_det)
            balls = [d for d in dets_in if d["class"] == "ball"]
            best = balls[0] if balls else None
            bbox, conf, _ = smoother.update(
                best["bbox"] if best else None,
                float(best["conf"]) if best else 0.0,
            )
            dets_out = list(players)
            if bbox:
                dets_out.append({"class": "ball", "bbox": bbox, "conf": conf})

            records.append({
                "frame_index": chunk * 20 + i,
                "timestamp_sec": t,
                "processed_sec": proc_t + i / analysed_fps,
                "detections": dets_out,
            })
        proc_t += 20 / analysed_fps

    return records


def run_simulation() -> dict:
    cfg = PipelineConfig()
    issues: list[str] = []

    # Config ↔ possession wiring
    pcfg = PossessionConfig(
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
    analysed_fps = 30.0 / max(1, cfg.infer_sample_every_n)

    # Compare chunk-reset vs continuous smoother
    reset_records = _mock_chunk_frames(cfg)
    cont_records = _mock_chunk_frames_continuous_smoother(cfg)

    def _ball_coverage(recs: list[dict]) -> int:
        return sum(1 for r in recs if any(d.get("class") == "ball" for d in r["detections"]))

    reset_balls = _ball_coverage(reset_records)
    cont_balls = _ball_coverage(cont_records)
    if cont_balls > reset_balls:
        issues.append(
            f"chunk-boundary smoother reset loses ball hold "
            f"(reset={reset_balls}/40 frames with ball, continuous={cont_balls}/40)"
        )

    records = cont_records
    poss = compute_possession(
        records,
        window_size_sec=cfg.possession_window_sec,
        contested_margin=cfg.contested_margin,
        fps=analysed_fps,
        cfg=pcfg,
    )
    team_a = sum(1 for p in poss if p["possession"] == "team_a")
    unknown = sum(1 for p in poss if p["possession"] == "unknown")

    if team_a < 10:
        issues.append(f"expected team_a possession on distant touch sim, got {team_a}/40 frames")

    timeline = generate_timestamps(poss, removed_segments=[])
    segments = timeline_to_clip_segments(
        timeline,
        my_team="team_a",
        pad_before=cfg.hotspot_pad_before_sec,
        pad_after=cfg.hotspot_pad_after_sec,
        gap_merge=cfg.hotspot_gap_merge_sec,
        min_zone_sec=cfg.hotspot_min_zone_sec,
        duration_sec=120.0,
    )

    if not segments:
        issues.append("no hotspot segments produced from simulated possession")
    else:
        seg = segments[0]
        for key in ("type", "start", "end", "core_start", "core_end", "action_triggers"):
            if key not in seg:
                issues.append(f"segment missing key: {key}")
        if seg.get("type") != "hotspot":
            issues.append(f"segment type should be hotspot, got {seg.get('type')}")

    # Server weights default
    weights_path = ROOT / "server.py"
    if weights_path.is_file():
        text = weights_path.read_text(encoding="utf-8")
        if "yolov8n.pt" in text and "yolov8s.pt" not in text.split("WEIGHTS")[1][:80]:
            issues.append("server.py WEIGHTS default may still point to yolov8n")

    ok = len(issues) == 0
    print("=== Touch pipeline simulation ===")
    print(f"Config: weights={cfg.yolo_weights}, infer_stride={cfg.infer_sample_every_n}, "
          f"analysed_fps~{analysed_fps:.1f}")
    print(f"Ball coverage: per-chunk reset={reset_balls}/40, continuous={cont_balls}/40")
    print(f"Possession: team_a={team_a}, unknown={unknown}")
    print(f"Hotspots: {len(segments)} segment(s)")
    if segments:
        print(f"  first zone {segments[0]['start']}–{segments[0]['end']}s "
              f"({len(segments[0]['action_triggers'])} triggers)")
    print(f"Result: {'PASS' if ok else 'FAIL'}")
    for issue in issues:
        print(f"  - {issue}")
    return {"ok": ok, "issues": issues, "segments": len(segments)}


if __name__ == "__main__":
    result = run_simulation()
    sys.exit(0 if result["ok"] else 1)
