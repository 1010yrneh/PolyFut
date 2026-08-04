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
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np

from polyfut_video.pipeline.decode import probe_video

from polyfut_v2.config import PipelineV2Config
from polyfut_v2.main import compute_trajectory, trajectory_warnings
from polyfut_v2.orchestrator import assemble_touches
from polyfut_v2.pipeline import play_ranges as pr
from polyfut_v2.pipeline.color import hex_to_hsv, hexes_to_hsv, hsv_distance_multi
from polyfut_v2.pipeline.frame_provider import VideoFrameProvider
from polyfut_v2.pipeline.hotspots import (
    assemble_hotspots,
    ball_track_from_trajectory,
)
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
            f"(1280x720) or higher gives dramatically better results — this is "
            f"the source file's own resolution, so re-exporting the same footage "
            f"at a higher setting is what changes it.")
    elif w and cfg.analysis_follow_source and w > cfg.analysis_max_width:
        out.append(
            f"Footage is {w}x{h}; analysed at {cfg.analysis_max_width}px wide to "
            f"bound processing time. Raise analysis_max_width to use all of it.")
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
        cfg.goalkeeper_class_id = bm.SOCCER_GOALKEEPER_CLASS
        cfg.referee_class_id = bm.SOCCER_REFEREE_CLASS
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
    sample_times: list[float] = []
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
            sample_times.append(t)
    if not samples:
        return TargetSeed(kit_hsv=None, gallery=[], n_samples=0)
    return build_seed_from_taps(
        samples, player_detector,
        max_tap_dist_px=cfg.contact_max_player_dist_px,
        min_torso_px=cfg.color_min_torso_px,
        sample_times_sec=sample_times,
    )


_SEED_DETECTOR_CACHE: dict[str, object] = {}
_SEED_DETECTOR_LOCK = threading.Lock()


def _seed_player_detector(cfg: PipelineV2Config):
    """A player detector for the seed step, cached per-weights so the OpenVINO
    model is compiled once and reused across seed-clip requests.

    Creation is serialized (the async /api/v2/warm call and the server's seed
    prefetch can race to be first) and finishes with a dummy inference: the
    first OpenVINO inference after compile costs tens of seconds, and paying it
    here — in the background warm path — keeps it off the first user-visible
    clip build."""
    key = f"{cfg.player_weights}|{cfg.player_imgsz}|{cfg.device}"
    with _SEED_DETECTOR_LOCK:
        det = _SEED_DETECTOR_CACHE.get(key)
        if det is None:
            det = YoloPlayerDetector(cfg)
            _ = det.model  # force load/compile now
            try:
                det.detect(np.zeros((360, 640, 3), dtype=np.uint8), None)
            except Exception:  # noqa: BLE001 — warmup is best-effort
                pass
            _SEED_DETECTOR_CACHE[key] = det
    return det


def build_review_track_for_item(
    video_path: str, item: dict, cfg: PipelineV2Config | None = None
) -> dict | None:
    """Track the reviewed player across a montage item's clip window so the
    review ring can follow them. Returns a normalized tracklet, or None if it
    can't be built (caller should fall back to the fixed crop)."""
    from polyfut_v2 import review_track as rt

    cfg = cfg or PipelineV2Config()
    crop = item.get("crop")
    if not crop or len(crop) != 4:
        return None
    pb = item.get("player_bbox")
    player_bbox = [float(v) for v in pb] if pb and len(pb) == 4 else None
    # Only when there's no real contact box do we need a detector, so the tracker
    # can re-check the *same* contact distance gate. Never invent a far-away
    # body — that was boxing the wrong person on the review screen.
    detector = None
    if player_bbox is None:
        _apply_soccer_model(cfg)
        detector = _seed_player_detector(cfg)
    frame_index = item.get("frame_index")
    try:
        fi = int(frame_index) if frame_index is not None else None
    except (TypeError, ValueError):
        fi = None
    try:
        return rt.build_review_track(
            video_path, [float(v) for v in crop],
            float(item["t_sec"]), float(item["clip_start_sec"]),
            float(item["clip_end_sec"]), target_width=cfg.target_width,
            player_bbox=player_bbox, detector=detector, cfg=cfg,
            frame_index=fi,
        )
    except (KeyError, TypeError, ValueError):
        return None


def warm_seed_detector(cfg: PipelineV2Config | None = None) -> None:
    """Compile/load the soccer player model ahead of time (≈150s one-time on this
    CPU) so the first seed clip doesn't pay for it. Safe to call repeatedly."""
    cfg = cfg or PipelineV2Config()
    _apply_soccer_model(cfg)
    _seed_player_detector(cfg)


def build_seed_clips_index(video_path: str, reroll: int = 0,
                           play_ranges: list[tuple[float, float]] | None = None) -> dict:
    """Return just the moment timestamps for a reroll set (cheap — no detection),
    so the UI can list the clips before any are built."""
    from polyfut_v2 import seed_clips as sc

    info = probe_video(video_path)
    dur = float(info.get("duration_sec") or 0.0)
    return {"reroll": reroll, "duration_sec": dur,
            "play_ranges": [list(r) for r in (play_ranges or [])],
            "moments": sc.default_moments(dur, reroll, play_ranges=play_ranges)}


def build_one_seed_clip(
    video_path: str,
    index: int,
    *,
    reroll: int = 0,
    cfg: PipelineV2Config | None = None,
    my_kit_hex: str | None = None,
    avoid_sec: tuple[float, ...] = (),
    play_ranges: list[tuple[float, float]] | None = None,
) -> dict | None:
    """Build tracked player nodes for one seed moment (no separate video encode).

    ``index`` selects which moment (0-3) of the ``reroll`` set. The browser
    plays the uploaded video directly at ``start_sec``; this only computes
    tracklets for the tappable markers.
    """
    from polyfut_v2 import seed_clips as sc

    cfg = cfg or PipelineV2Config()
    _apply_soccer_model(cfg)  # so nodes use the real soccer player model
    detector = _seed_player_detector(cfg)

    idx_info = build_seed_clips_index(video_path, 0, play_ranges)
    moments = idx_info["moments"]
    if not moments:
        return None
    dur = idx_info["duration_sec"]
    n_moments = len(moments)
    index = max(0, min(index, n_moments - 1))
    # ``reroll`` is per-slot: it nudges only THIS clip's base moment, so a single
    # clip can be reshuffled without disturbing the others. The base is then
    # refined outward to the nearest well-populated, taggable moment — never
    # outside the declared playing-time window.
    base_t = sc.moment_for_index(dur, index, reroll, play_ranges)
    t_center = sc.pick_best_moment_near(
        video_path, base_t, detector, dur, my_kit_hex=my_kit_hex,
        avoid_sec=tuple(avoid_sec or ()), play_ranges=play_ranges,
    )

    clip = sc.build_seed_clip(video_path, t_center, detector)
    if clip is None:
        return None
    # Precompute the seed taps for each node so the UI can build a gallery from a
    # click without re-deriving them (coords map back to the original video).
    start = clip["start_sec"]
    for tr in clip["tracklets"]:
        tr["taps"] = taps_from_tracklet(tr, start)
    clip.update(index=index, reroll=reroll, t_center=t_center,
                moments=moments, n_moments=n_moments,
                play_ranges=[list(r) for r in (play_ranges or [])])
    return clip


def taps_from_tracklet(tracklet: dict, start_sec: float, max_taps: int = 8) -> list[dict]:
    """Convert a selected player's tracklet into seed tap specs spread across the
    clip → a multi-crop appearance gallery. Normalized coords map back to the
    original video, so ``t_sec`` is the clip start plus the point's local time."""
    pts = tracklet.get("points") or []
    if not pts:
        return []
    if len(pts) > max_taps:
        step = len(pts) / float(max_taps)
        pts = [pts[int(i * step)] for i in range(max_taps)]
    return [{"t_sec": round(start_sec + p["t"], 3), "nx": p["nx"], "ny": p["ny"]}
            for p in pts]


def run_to_montage(
    video_path: str,
    *,
    seed_taps: list[dict] | None = None,
    cfg: PipelineV2Config | None = None,
    ball_detector=None,
    progress: ProgressCb | None = None,
    should_cancel: Callable[[], bool] | None = None,
    opponent_hex: str | None = None,
    opponent_hexes: list[str] | None = None,
    my_team_hexes: list[str] | None = None,
    play_ranges: list[tuple[float, float]] | None = None,
    calibration: dict | None = None,
) -> dict:
    """Run Stages 0-8 and return a JSON-serializable montage document.

    Hotspots here come only from auto-accepted contacts; the user refines them
    via me/not-me review, applied later with ``hotspots_from_decisions``.

    ``calibration`` is the optional pitch calibration produced by the UI. With
    it, "who touched the ball" is judged in metres rather than pixels (Issue 14);
    without it, or if it fails the quality screen, behaviour is unchanged.

    ``play_ranges`` is the user's declared on-pitch window (video-time seconds).
    Everything downstream is confined to it: the trajectory only exists inside
    the padded ranges, so candidates, montage and hotspots follow automatically
    — Stages 4-9 need no range awareness of their own.
    """
    cfg = cfg or PipelineV2Config()
    progress = progress or _noop
    t_total = time.perf_counter()

    # Analyse at the source's own resolution (capped) rather than always
    # downscaling to 640. Must happen before anything decodes a frame or reads a
    # px threshold, since ``for_frame_width`` rescales both together — they are
    # meaningless apart. A 640-wide source returns the identical config.
    try:
        cfg = cfg.for_frame_width(int(probe_video(video_path).get("width") or 0))
    except Exception:  # noqa: BLE001 — a probe failure must not fail the run
        pass

    # Upgrade ball + player detection to the soccer-specific model (COCO can't
    # see the ball). Downloads on first use; falls back to COCO if offline.
    progress(0.01, "Loading soccer detection model…")
    t0 = time.perf_counter()
    model_warning = _apply_soccer_model(cfg)
    player_detector = YoloPlayerDetector(cfg)
    model_load_sec = time.perf_counter() - t0

    if ball_detector is None and _synthetic_ball_enabled():
        ball_detector = _SyntheticBall()

    res = compute_trajectory(
        Path(video_path), cfg, detector=ball_detector,
        progress=progress, should_cancel=should_cancel,
        play_ranges=play_ranges,
    )
    traj = res["trajectory"]
    duration = res["info"].get("duration_sec")
    warnings = _footage_warnings(res["info"], cfg)
    warnings.extend(trajectory_warnings(traj, cfg))
    if model_warning:
        warnings.insert(0, model_warning)

    # On-pitch seconds budget the candidate cap; full duration still bounds clip
    # and hotspot ends, because timestamps never leave video time.
    on_pitch_sec = pr.total_seconds(res.get("play_ranges_padded") or []) or None
    if play_ranges and duration:
        warnings.insert(0, (
            f"Analysis limited to your playing time: "
            f"{pr.total_seconds(play_ranges) / 60:.0f} of {float(duration) / 60:.0f} min. "
            f"Touches outside the periods you marked were not looked for. Each "
            f"period is padded by {cfg.playing_time_pad_sec:.0f}s on both sides "
            f"so a slightly-off handle doesn't lose a real touch."
        ))

    t0 = time.perf_counter()
    seed = build_seed_from_tap_specs(video_path, seed_taps, cfg, player_detector)
    # The other team's kit colours (from the team picker) let the per-contact
    # team gate use both centroids to drop obvious wrong-team touches. A kit that
    # isn't one flat colour contributes all of its colours; a touch matching any
    # of them is that team's.
    opp_all = hexes_to_hsv(opponent_hexes) if opponent_hexes else []
    if not opp_all:
        primary = hex_to_hsv(opponent_hex)
        opp_all = [primary] if primary is not None else []
    seed.opponent_kit_hsv = opp_all[0] if opp_all else None
    seed.opponent_kit_hsv_alts = opp_all[1:]
    # Colours the user confirmed on the team screen are added to the tap-derived
    # kit rather than replacing it: the taps come from an actual player so they
    # stay dominant, while an edited swatch can only ever add a colour that will
    # be *matched*. Recall-safe — more colours can never drop a real touch.
    for extra in hexes_to_hsv(my_team_hexes):
        if seed.kit_hsv is None:
            seed.kit_hsv = extra
            continue
        d = hsv_distance_multi(extra, seed.my_kits())
        if d is None or d > cfg.team_color_max_dist:
            seed.kit_hsv_alts.append(extra)
    seed_sec = time.perf_counter() - t0
    if seed.n_samples == 0:
        warnings.append("no seed taps — appearance scoring is neutral")
    elif seed.is_weak():
        warnings.append("weak seed (≤1 tap) — expect a larger review montage")

    with VideoFrameProvider(video_path, target_width=cfg.target_width) as provider:
        out = assemble_touches(
            traj, duration, cfg, seed, player_detector, provider, progress=progress,
            budget_sec=on_pitch_sec, calibration=calibration,
        )

    n_raw = out.get("n_candidates_raw", len(out["candidates"]))
    if n_raw > len(out["candidates"]):
        warnings.append(
            f"Stage 4 produced {n_raw} contact candidates; kept the strongest "
            f"{len(out['candidates'])} (spread across the clip) to bound "
            f"processing time. A soccer-specific ball model yields far fewer, "
            f"cleaner candidates."
        )

    # Stage 5b: say out loud when the pitch calibration did not end up being
    # used. This used to be entirely silent — a run with a broken calibration
    # looked exactly like one with none — so "attribution still looks wrong"
    # had no answerable first question.
    calib_status = out.get("calibration_status")
    if calib_status and calib_status != "ok":
        from polyfut_v2.orchestrator import CALIBRATION_FALLBACK_MESSAGES
        msg = CALIBRATION_FALLBACK_MESSAGES.get(calib_status)
        if msg:
            warnings.append(msg.format(
                limit=float(getattr(cfg, "calibration_max_median_px", 6.0))))
    elif calib_status == "ok":
        cov = float((out.get("attribution") or {}).get("metric_coverage") or 0.0)
        if cov < 0.5:
            warnings.append(
                f"The pitch calibration was only usable for {cov:.0%} of "
                f"touches — the rest fell back to pixel distances. Usually a "
                f"camera cut or a stretch where the camera track broke.")

    items = [it.to_dict() for it in out["montage"]]
    # Adaptive review grouping: tag kit colour / other-team / same-kit look-alike
    # so the review UI can clear a whole team on one "Not me". Cheap (reuses the
    # torso crops / Stage 7 descriptors already captured); failures degrade to
    # per-clip decisions.
    t0 = time.perf_counter()
    try:
        from polyfut_v2.pipeline.grouping import assign_player_groups
        groups = assign_player_groups(out["montage"], seed, cfg)
        for d in items:
            g = groups.get(d["rank"])
            if g:
                d.update(g)
    except Exception:  # noqa: BLE001 — grouping is a UX aid, never fatal
        pass
    grouping_sec = time.perf_counter() - t0

    timings = dict(res.get("timings") or {})
    timings["model_load"] = model_load_sec
    timings["seed"] = seed_sec
    timings.update(out.get("timings") or {})
    timings["grouping"] = grouping_sec
    timings["total"] = time.perf_counter() - t_total
    stage_timings = {k: round(v, 2) for k, v in timings.items()}

    return {
        "pipeline_version": "v2",
        "duration_sec": duration,
        # Audit trail: what the user declared, and what was actually analysed
        # after padding — so a later "why is there nothing at 20 min?" is
        # answerable from job_state.json alone.
        "play_ranges": res.get("play_ranges") or [],
        "play_ranges_padded": res.get("play_ranges_padded") or [],
        "on_pitch_sec": round(on_pitch_sec, 2) if on_pitch_sec else None,
        "n_samples": len(traj),
        "detected_ratio": round(traj.detected_ratio(), 4),
        "ball_detector_stats": res.get("ball_detector_stats"),
        # Stage 3f: how much of the trajectory was blanked as a false lock. Part
        # of the audit trail — a run with a high rejected_pingpong count is one
        # where the tracker was oscillating, which explains sparse candidates.
        "ball_sanity": res.get("ball_sanity"),
        # Stage 5b coverage: how many contacts were judged in metres vs fell
        # back to pixels. A low ratio means the calibration was unusable over
        # most of the video (a cut, or a camera track that broke), which is the
        # first thing to check if attribution still looks wrong.
        "attribution": out.get("attribution"),
        # Which exit _build_pitch_mapper took — the answerable form of
        # "why was metric_coverage zero?".
        "calibration_status": out.get("calibration_status"),
        # Where the ball was, in source time. Persisted because hotspots are
        # rebuilt from scratch after every review decision, and without this the
        # possession extension would silently revert to a fixed pad the moment
        # the user confirmed a single touch.
        "ball_track": [[round(t, 3), round(x, 1), round(y, 1)]
                       for t, x, y in ball_track_from_trajectory(traj)],
        "calibration": calibration if isinstance(calibration, dict) else None,
        "n_candidates_raw": n_raw,
        "n_candidates": len(out["candidates"]),
        "n_your_team": len(out["kept"]),
        "n_review": sum(1 for it in items if it["status"] == "review"),
        "n_crowded": sum(1 for it in items if it.get("crowded")),
        # The kit hexes decide the team gate outright: with the right pair
        # 98% of your touches survive it and 100% of the opponent's are removed;
        # with a wrong pair that collapses to 16-26% kept, measured on the
        # ISB/TAS footage. So a run that dropped your own team is only
        # explicable if the record says which colours it was given.
        # has_opponent matters on its own — without it classify_team falls back
        # to the one-sided band, which loses 26% of your team by itself.
        "seed": {"n_samples": seed.n_samples, "has_color": seed.has_color(),
                 "gallery": len(seed.gallery),
                 "my_team_hexes": list(my_team_hexes or []),
                 "opponent_hex": opponent_hex,
                 "opponent_hexes": list(opponent_hexes or []),
                 "has_opponent": bool(seed.opponent_kits())},
        "warnings": warnings,
        "montage": items,
        "hotspots": [h.to_dict() for h in out["hotspots"]],
        "stage_timings_sec": stage_timings,
        "timings_sec": timings,
    }


def hotspots_from_decisions(
    montage_items: list[dict],
    decisions: dict,
    cfg: PipelineV2Config | None = None,
    *,
    duration_sec: float | None = None,
    ball_track: list | None = None,
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
    track = None
    if ball_track:
        try:
            track = sorted((float(t), float(x), float(y))
                           for t, x, y in ball_track)
        except (TypeError, ValueError):
            track = None
    hotspots = assemble_hotspots(me_times, cfg, duration_sec=duration_sec,
                                 ball_track=track)
    return [h.to_dict() for h in hotspots], montage_items


def calibration_from_clicks(
    clicks: list[dict],
    *,
    frame_width: int,
    frame_height: int,
    pitch_length_m: float = 100.0,
    pitch_width_m: float = 64.0,
    seed_params: list | None = None,
    free_pitch_size: bool = False,
) -> dict | None:
    """Fit a pitch calibration server-side from the landmarks the user clicked.

    The browser already fits these same clicks (it draws the pitch live while
    you click), but the pipeline must run on a fit produced by the code that
    will *use* it — otherwise a browser/Python disagreement would show up as
    mysteriously wrong attribution. The browser's answer comes along as
    ``seed_params``: it turns a ~9s blind search into a sub-second polish, and
    it is verified rather than trusted (a seed that refines poorly is discarded
    and the blind search runs anyway).

    Returns a JSON-serialisable dict, or None if the clicks can't be fitted.
    """
    from polyfut_v2.pipeline.pitch_calibration import fit as fit_calibration

    parsed: list[tuple[int, tuple[float, float], str]] = []
    for c in clicks or []:
        try:
            fi = int(c["frame_index"])
            xy = (float(c["x"]), float(c["y"]))
            key = str(c["landmark"])
        except (KeyError, TypeError, ValueError):
            continue
        parsed.append((fi, xy, key))
    if len(parsed) < 4:
        return None

    calib = fit_calibration(
        parsed,
        principal=(float(frame_width) / 2.0, float(frame_height) / 2.0),
        pitch_length_m=float(pitch_length_m),
        pitch_width_m=float(pitch_width_m),
        free_pitch_size=bool(free_pitch_size),
        seed_params=seed_params,
    )
    if calib is None:
        return None
    calib.frame_width = int(frame_width)
    d = calib.to_dict()
    d["clicks"] = [
        {"frame_index": fi, "x": xy[0], "y": xy[1], "landmark": k}
        for fi, xy, k in parsed
    ]
    return d
