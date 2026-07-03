"""Stage 8: proximity possession with distance-aware contact + hysteresis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from polyfut_video.pipeline.geometry import ball_diagonal, box_center, dist, foot_point


@dataclass
class PossessionConfig:
    """Tuned for 640px-wide broadcast footage."""
    base_thresh_px: float = 85.0
    ref_person_h: float = 100.0
    ref_ball_diag: float = 18.0
    min_thresh_px: float = 55.0
    max_thresh_px: float = 140.0
    on_frames: int = 1
    off_frames: int = 6
    contested_margin: float = 0.15
    window_size_sec: float = 0.8


def _scaled_contact_threshold(
    person_h: float,
    ball_d: float,
    cfg: PossessionConfig,
) -> float:
    """Tiny distant balls get a larger px allowance."""
    size_f = 1.0 if ball_d <= 1 else cfg.ref_ball_diag / max(ball_d, 1.0)
    size_f = min(1.8, max(0.55, size_f))
    thr = cfg.base_thresh_px * (person_h / cfg.ref_person_h) * size_f
    return float(min(cfg.max_thresh_px, max(cfg.min_thresh_px, thr)))


def _nearest_team_contact(
    players: list[dict],
    ball: dict | None,
    cfg: PossessionConfig,
) -> str | None:
    if ball is None or not players:
        return None

    bc = box_center(ball["bbox"])
    bd = ball_diagonal(ball["bbox"])
    best_team: str | None = None
    best_d = 1e18

    for p in players:
        if p.get("class") != "player":
            continue
        tid = p.get("team_id", -1)
        if tid not in (0, 1):
            continue
        fp = foot_point(p["bbox"])
        person_h = max(float(p["bbox"][3] - p["bbox"][1]), 1.0)
        d = dist(fp, bc)
        thr = _scaled_contact_threshold(person_h, bd, cfg)
        if d <= thr and d < best_d:
            best_d = d
            best_team = "team_a" if tid == 0 else "team_b"

    return best_team


def _hysteresis_labels(raw_teams: list[str | None], cfg: PossessionConfig) -> list[str]:
    """Fast open / slow close so brief ball misses don't drop possession."""
    out: list[str] = []
    active: str | None = None
    on_streak = 0
    off_streak = 0

    for team in raw_teams:
        if team in ("team_a", "team_b"):
            off_streak = 0
            if active is None:
                on_streak += 1
                if on_streak >= cfg.on_frames:
                    active = team
                    on_streak = 0
                out.append(active if active else "unknown")
            elif team != active:
                out.append("contested")
            else:
                out.append(active)
            continue

        on_streak = 0
        if active is not None:
            off_streak += 1
            if off_streak >= cfg.off_frames:
                active = None
                off_streak = 0
                out.append("unknown")
            else:
                out.append(active)
        else:
            out.append("unknown")

    return out


def _window_smooth(labels: list[str], window_frames: int, contested_margin: float) -> list[str]:
    if window_frames <= 1:
        return labels
    half = max(0, window_frames // 2)
    out: list[str] = []
    for i in range(len(labels)):
        if labels[i] == "contested":
            out.append("contested")
            continue
        start = max(0, i - half)
        end = min(len(labels), i + half + 1)
        window = [l for l in labels[start:end] if l in ("team_a", "team_b")]
        if not window:
            out.append(labels[i])
            continue
        a_count = window.count("team_a")
        b_count = window.count("team_b")
        total = len(window)
        if abs(a_count / total - b_count / total) < contested_margin:
            out.append("contested")
        elif a_count > b_count:
            out.append("team_a")
        else:
            out.append("team_b")
    return out


def compute_possession(
    tracked_frames: list[dict],
    window_size_sec: float = 0.8,
    *,
    contested_margin: float = 0.15,
    fps: float = 25.0,
    cfg: PossessionConfig | None = None,
) -> list[dict]:
    """
    tracked_frames: list of {"frame_index", "timestamp_sec", "detections": [...]}

    Returns per-frame records with possession: team_a | team_b | contested | unknown.
    ``fps`` should be the effective analysed frame rate (source_fps / infer_stride).
    """
    if not tracked_frames:
        return []

    pcfg = cfg or PossessionConfig()
    if contested_margin != pcfg.contested_margin:
        pcfg = PossessionConfig(
            base_thresh_px=pcfg.base_thresh_px,
            ref_person_h=pcfg.ref_person_h,
            ref_ball_diag=pcfg.ref_ball_diag,
            min_thresh_px=pcfg.min_thresh_px,
            max_thresh_px=pcfg.max_thresh_px,
            on_frames=pcfg.on_frames,
            off_frames=pcfg.off_frames,
            contested_margin=contested_margin,
            window_size_sec=window_size_sec,
        )

    raw_teams: list[str | None] = []
    for rec in tracked_frames:
        dets = rec.get("detections", [])
        players = [d for d in dets if d.get("class") == "player"]
        balls = [d for d in dets if d.get("class") == "ball"]
        ball = max(balls, key=lambda b: b.get("conf", 0)) if balls else None
        raw_teams.append(_nearest_team_contact(players, ball, pcfg))

    labels = _hysteresis_labels(raw_teams, pcfg)
    window_frames = max(1, int(round(window_size_sec * fps)))
    labels = _window_smooth(labels, window_frames, contested_margin)

    out: list[dict] = []
    for rec, label in zip(tracked_frames, labels):
        out.append({
            "frame_index": rec.get("frame_index"),
            "timestamp_sec": rec.get("timestamp_sec"),
            "processed_sec": rec.get("processed_sec", rec.get("timestamp_sec")),
            "possession": label,
        })
    return out
