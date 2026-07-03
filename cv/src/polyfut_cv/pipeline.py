"""End-to-end: video + team colours -> clip_segments.json.

Optimisation stack (auto-enabled for long matches via optimize.tune_for_duration):
  1. ffmpeg proxy (12 fps, 1280px) for matches >20 min ? reuse cached proxy when provided
  2. OpenVINO weights when yolov8n_openvino_model/ exists
  3. grab()-skip frame iteration (no decode of skipped frames)
  4. batch YOLO inference (default batch_size=8)
  5. two-pass: permissive coarse sweep -> dense scan only in padded windows
  6. cut + blur gates before YOLO
  7. optional 2x2 tiled inference on dense pass (SAHI-style ball recall)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from polyfut_cv.blur_detect import BlurConfig, is_blurry
from polyfut_cv.clip_export import ExportConfig, build_segments, write_clip_segments_json
from polyfut_cv.color_classify import ColorRef, classify_is_mine, hsv_to_lab, is_my_team
from polyfut_cv.cut_detect import CutConfig, CutDetector
from polyfut_cv.detect import DetectConfig, Detector, iter_frames, video_info
from polyfut_cv.geometry import box_center, dist, foot_point
from polyfut_cv.pitch_mask import PitchMask
from polyfut_cv.possession import PossessionConfig, extract_intervals, nearest_my_team_contact
from polyfut_cv.preprocess import build_proxy, have_ffmpeg
from polyfut_cv.track_smooth import BallSmoothConfig, BallSmoother

ProgressCb = Callable[[float, str], None]


def _split_frame_ranges(
    ranges: list[tuple[int, int]],
    *,
    max_span_frames: int,
) -> list[tuple[int, int]]:
    """Split oversized dense windows for steadier progress (same frames, no skips)."""
    if max_span_frames <= 0:
        return ranges
    out: list[tuple[int, int]] = []
    for sf, ef in ranges:
        t = sf
        while t < ef:
            chunk_end = min(ef, t + max_span_frames)
            if chunk_end > t:
                out.append((t, chunk_end))
            t = chunk_end
    return out


# Coarse pass flags "live play" windows (recall-first ? do not tighten for speed).
def _coarse_is_active(
    n_persons: int,
    my_team: list[np.ndarray],
    ball_xyxy: np.ndarray | None,
    ball_conf: float,
    *,
    min_players: int = 3,
    min_ball_conf: float = 0.08,
) -> bool:
    if n_persons >= min_players:
        return True
    if ball_xyxy is None or ball_conf < min_ball_conf:
        return False
    bc = box_center(ball_xyxy)
    for p in my_team:
        person_h = max(float(p[3] - p[1]), 1.0)
        if dist(foot_point(p), bc) <= person_h * 2.5:
            return True
    return False


def _merge_times_to_windows(
    times: list[float],
    *,
    gap_sec: float,
    pad_sec: float,
    duration_sec: float,
) -> list[tuple[float, float]]:
    if not times:
        return []
    times = sorted(times)
    raw: list[tuple[float, float]] = []
    t0 = t_last = times[0]
    for t in times[1:]:
        if t - t_last <= gap_sec:
            t_last = t
        else:
            raw.append((t0, t_last))
            t0 = t_last = t
    raw.append((t0, t_last))
    padded: list[tuple[float, float]] = []
    for s, e in raw:
        ns = max(0.0, s - pad_sec)
        ne = min(duration_sec, e + pad_sec) if duration_sec > 0 else e + pad_sec
        if ne > ns:
            padded.append((ns, ne))
    if not padded:
        return []
    out = [list(padded[0])]
    for s, e in padded[1:]:
        if s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(a, b) for a, b in out]


@dataclass
class PipelineConfig:
    stride: int = 6
    imgsz: int = 1280
    coarse_imgsz: int = 640
    coarse_stride_mult: int = 8
    batch_size: int = 8
    conf: float = 0.15
    weights: str = "yolov8n.pt"
    device: str = "cpu"
    two_pass: bool = True
    auto_proxy: bool = True
    proxy_threshold_min: float = 20.0
    proxy_fps: float = 12.0
    proxy_width: int = 1280
    coarse_window_pad_sec: float = 12.0
    coarse_candidate_gap_sec: float = 4.0
    min_players_for_active: int = 3
    use_cut_gate: bool = True
    use_blur_gate: bool = True
    use_pitch_mask: bool = False
    use_tiled_dense: bool = False
    dense_chunk_minutes: float = 12.0
    max_minutes: float = 90.0
    chunk_minutes: float = 15.0
    export: ExportConfig = field(default_factory=ExportConfig)
    possession: PossessionConfig = field(default_factory=PossessionConfig)
    ball_smooth: BallSmoothConfig = field(default_factory=BallSmoothConfig)
    cut: CutConfig = field(default_factory=CutConfig)
    blur: BlurConfig = field(default_factory=BlurConfig)


class _FrameScanner:
    def __init__(
        self,
        detector: Detector,
        color_ref: ColorRef,
        other_ref: ColorRef | None,
        cfg: PipelineConfig,
    ):
        self.detector = detector
        self.color_ref = color_ref
        self.other_ref = other_ref
        self.cfg = cfg
        self.cut_det = CutDetector(cfg.cut) if cfg.use_cut_gate else None
        self.ball_sm = BallSmoother(cfg.ball_smooth)
        self.pitch = None
        self._my_lab = hsv_to_lab(color_ref.h, color_ref.s, color_ref.v) if other_ref else None
        self._other_lab = hsv_to_lab(other_ref.h, other_ref.s, other_ref.v) if other_ref else None
        self.diag = {
            "frames": 0,
            "gated": 0,
            "ball_seen": 0,
            "myteam_seen": 0,
            "active": 0,
            "contact": 0,
        }

    def _classify_mine(self, frame: np.ndarray, xyxy: np.ndarray) -> bool:
        if self._my_lab is not None and self._other_lab is not None:
            return classify_is_mine(frame, xyxy, self._my_lab, self._other_lab)
        return is_my_team(frame, xyxy, self.color_ref)

    def _filter_pitch(self, frame: np.ndarray, persons: list, balls: list):
        if self.pitch is None:
            return persons, balls
        fp = []
        for p in persons:
            if self.pitch.point_on_pitch(float(foot_point(p.xyxy)[0]), float(foot_point(p.xyxy)[1])):
                fp.append(p)
        fb = []
        for b in balls:
            c = box_center(b.xyxy)
            if self.pitch.point_on_pitch(float(c[0]), float(c[1])):
                fb.append(b)
        return fp, fb

    def process_batch(
        self,
        frames: list[np.ndarray],
        times: list[float],
        *,
        imgsz: int | None = None,
        tiled: bool | None = None,
        coarse_mode: bool = False,
    ) -> list[tuple[float, bool]]:
        if not frames:
            return []
        if self.pitch is None and self.cfg.use_pitch_mask:
            self.pitch = PitchMask(frames[0])

        n = len(frames)
        slots: list[tuple[float, bool] | None] = [None] * n
        infer_frames: list[np.ndarray] = []
        infer_times: list[float] = []
        infer_slots: list[int] = []

        for i, (frame, t_sec) in enumerate(zip(frames, times)):
            self.diag["frames"] += 1
            is_cut = self.cut_det.is_cut(frame) if self.cut_det else False
            if is_cut:
                self.ball_sm.reset()
            if is_cut or (self.cfg.use_blur_gate and is_blurry(frame, self.cfg.blur)):
                self.diag["gated"] += 1
                slots[i] = (t_sec, False)
                continue
            infer_frames.append(frame)
            infer_times.append(t_sec)
            infer_slots.append(i)

        if infer_frames:
            use_tiled = tiled if tiled is not None else (False if coarse_mode else self.cfg.use_tiled_dense)
            batch_out = self.detector.predict_batch(
                infer_frames, imgsz=imgsz, tiled=use_tiled
            )
            for j, (slot_i, t_sec) in enumerate(zip(infer_slots, infer_times)):
                frame = infer_frames[j]
                persons, balls = batch_out[j]
                persons, balls = self._filter_pitch(frame, persons, balls)
                my_team = [p.xyxy for p in persons if self._classify_mine(frame, p.xyxy)]
                if my_team:
                    self.diag["myteam_seen"] += 1
                ball_xyxy = balls[0].xyxy if balls else None
                ball_conf = balls[0].conf if balls else 0.0
                if ball_xyxy is not None:
                    self.diag["ball_seen"] += 1
                ball_xyxy, ball_conf, _ = self.ball_sm.update(ball_xyxy, ball_conf)

                if coarse_mode:
                    active = _coarse_is_active(
                        len(persons), my_team, ball_xyxy, ball_conf,
                        min_players=self.cfg.min_players_for_active,
                        min_ball_conf=self.cfg.possession.conf_ball_min,
                    )
                    if active:
                        self.diag["active"] += 1
                    slots[slot_i] = (t_sec, active)
                    continue

                contact, _ = nearest_my_team_contact(my_team, ball_xyxy, ball_conf, self.cfg.possession)
                if contact:
                    self.diag["contact"] += 1
                slots[slot_i] = (t_sec, contact)

        out: list[tuple[float, bool]] = []
        for slot in slots:
            if slot is not None:
                out.append(slot)
        return out


def _scan_range_batched(
    scanner: _FrameScanner,
    video_path: Path,
    *,
    stride: int,
    start_frame: int,
    end_frame: int,
    imgsz: int | None,
    tiled: bool | None,
    coarse_mode: bool,
    batch_size: int,
    progress_cb: Callable[[int, int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> list[tuple[float, bool]]:
    total = max(1, (end_frame - start_frame) // max(stride, 1))
    done = 0
    buf_f: list[np.ndarray] = []
    buf_t: list[float] = []
    results: list[tuple[float, bool]] = []

    for _idx, t_sec, frame in iter_frames(
        video_path, stride=stride, start_frame=start_frame, end_frame=end_frame
    ):
        if should_cancel and should_cancel():
            raise RuntimeError("cancelled")
        buf_f.append(frame)
        buf_t.append(t_sec)
        if len(buf_f) < batch_size:
            continue
        chunk = scanner.process_batch(buf_f, buf_t, imgsz=imgsz, tiled=tiled, coarse_mode=coarse_mode)
        results.extend(chunk)
        done += len(buf_f)
        buf_f, buf_t = [], []
        if progress_cb:
            progress_cb(min(done, total), total)

    if buf_f:
        chunk = scanner.process_batch(buf_f, buf_t, imgsz=imgsz, tiled=tiled, coarse_mode=coarse_mode)
        results.extend(chunk)
        done += len(buf_f)
        if progress_cb:
            progress_cb(min(done, total), total)
    return results


def run_pipeline(
    video_path: str | Path,
    color_ref: ColorRef,
    out_dir: str | Path,
    cfg: PipelineConfig | None = None,
    *,
    other_ref: ColorRef | None = None,
    proxy_path: str | Path | None = None,
    progress: ProgressCb | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict:
    cfg = cfg or PipelineConfig()
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if proxy_path is not None:
        proxy_path = Path(proxy_path)

    def emit(frac: float, msg: str) -> None:
        if progress:
            progress(max(0.0, min(1.0, frac)), msg)

    emit(0.01, "Reading video...")
    src_fps, src_nframes, src_w, _src_h = video_info(video_path)
    src_duration = src_nframes / src_fps if src_fps > 0 else 0.0
    src_dur_min = src_duration / 60.0
    max_frames = int(cfg.max_minutes * 60.0 * src_fps) if src_fps > 0 else src_nframes
    end_frame = min(src_nframes - 1, max_frames) if max_frames else src_nframes - 1
    duration_sec = end_frame / src_fps if src_fps > 0 else src_duration

    analysis_path = video_path
    used_proxy = False

    if proxy_path and proxy_path.exists():
        analysis_path = proxy_path
        used_proxy = True
        emit(0.03, "Using cached proxy...")
    elif cfg.auto_proxy and src_dur_min > cfg.proxy_threshold_min and have_ffmpeg():
        proxy_out = out_dir / f"{video_path.stem}_proxy.mp4"
        if not proxy_out.exists():
            emit(0.02, "Building proxy (ffmpeg)...")

            def _proxy_prog(p: float) -> None:
                emit(0.02 + 0.02 * p, f"Building proxy... {int(p * 100)}%")

            build_proxy(
                video_path, proxy_out,
                max_width=cfg.proxy_width,
                fps=cfg.proxy_fps,
                duration_sec=duration_sec,
                progress_cb=_proxy_prog,
            )
        analysis_path = proxy_out
        used_proxy = True
        emit(0.04, "Proxy ready")

    ana_fps, ana_nframes, _, _ = video_info(analysis_path)
    if ana_fps <= 0:
        ana_fps = src_fps or 25.0
    ana_end = min(ana_nframes - 1, int(duration_sec * ana_fps))

    det_cfg = DetectConfig(
        weights=cfg.weights,
        imgsz=cfg.imgsz,
        conf=cfg.conf,
        stride=cfg.stride,
        device=cfg.device,
        batch_size=cfg.batch_size,
        use_tiled_ball=cfg.use_tiled_dense,
    )
    emit(0.05, "Loading detector...")
    detector = Detector(det_cfg)
    scanner = _FrameScanner(detector, color_ref, other_ref, cfg)

    coarse_stride = cfg.stride * cfg.coarse_stride_mult
    windows: list[tuple[float, float]] | None = None
    coarse_diag: dict = {}
    dense_diag: dict = {}

    if cfg.two_pass:
        emit(0.08, f"Coarse scan (stride {coarse_stride})...")
        coarse_scanner = _FrameScanner(detector, color_ref, other_ref, cfg)
        coarse_times: list[float] = []

        def _coarse_prog(done: int, total: int) -> None:
            emit(0.08 + 0.22 * (done / max(total, 1)), f"Coarse {done}/{total}")

        coarse_results = _scan_range_batched(
            coarse_scanner, analysis_path,
            stride=coarse_stride,
            start_frame=0,
            end_frame=ana_end,
            imgsz=cfg.coarse_imgsz,
            tiled=False,
            coarse_mode=True,
            batch_size=cfg.batch_size,
            progress_cb=_coarse_prog,
            should_cancel=should_cancel,
        )
        coarse_diag = dict(coarse_scanner.diag)
        for t_sec, active in coarse_results:
            if active:
                coarse_times.append(t_sec)
        windows = _merge_times_to_windows(
            coarse_times,
            gap_sec=cfg.coarse_candidate_gap_sec,
            pad_sec=cfg.coarse_window_pad_sec,
            duration_sec=duration_sec,
        )
        if not windows:
            emit(0.30, "No coarse windows ? scanning full video (dense)...")
            cfg.two_pass = False
        else:
            covered = sum(e - s for s, e in windows)
            emit(0.30, f"Coarse done ? {len(windows)} windows ({covered:.0f}s / {duration_sec:.0f}s)")

    all_times: list[float] = []
    all_flags: list[bool] = []
    touch_times: list[float] = []

    def _dense_prog(display_done: int, total: int) -> None:
        capped = min(max(0, display_done), total)
        frac = 0.35 + 0.55 * (capped / max(total, 1))
        # region agent log
        if capped % 200 < 16 or capped >= total:
            try:
                import json as _json
                import time as _time
                rec = {
                    "sessionId": "16e722",
                    "runId": "post-fix",
                    "hypothesisId": "H4",
                    "location": "pipeline.py:_dense_prog",
                    "message": "dense progress",
                    "data": {
                        "display_done": display_done,
                        "capped": capped,
                        "total": total,
                        "frac": round(frac, 4),
                    },
                    "timestamp": int(_time.time() * 1000),
                }
                _logp = Path(os.environ.get(
                    "POLYFUT_DEBUG_LOG",
                    str(Path.home() / ".cursor" / "debug-logs" / "debug-16e722.log"),
                ))
                with open(_logp, "a", encoding="utf-8") as _f:
                    _f.write(_json.dumps(rec) + "\n")
            except Exception:
                pass
        # endregion
        emit(min(0.90, frac), f"Dense scan... {capped}/{total}")

    dense_ranges: list[tuple[int, int]] = []
    if cfg.two_pass and windows:
        for ws, we in windows:
            sf = max(0, int(ws * ana_fps))
            ef = min(ana_end, int(we * ana_fps))
            if ef > sf:
                dense_ranges.append((sf, ef))
    else:
        dense_ranges = [(0, ana_end)]

    chunk_frames = int(cfg.dense_chunk_minutes * 60.0 * ana_fps) if cfg.dense_chunk_minutes > 0 else 0
    dense_ranges = _split_frame_ranges(dense_ranges, max_span_frames=chunk_frames)

    dense_total = sum(max(1, (ef - sf) // max(cfg.stride, 1)) for sf, ef in dense_ranges)
    tiled_label = "on" if cfg.use_tiled_dense else "off"
    emit(
        0.32,
        f"Dense: {dense_total} frames, stride {cfg.stride}, imgsz {cfg.imgsz}, tiles {tiled_label}",
    )
    dense_done = 0

    for wi, (sf, ef) in enumerate(dense_ranges):
        if should_cancel and should_cancel():
            raise RuntimeError("cancelled")
        win_total = max(1, (ef - sf) // max(cfg.stride, 1))
        win_base = dense_done
        label = f"Dense chunk {wi + 1}/{len(dense_ranges)}"
        emit(0.35 + 0.55 * (win_base / max(dense_total, 1)), label)

        def _win_prog(done: int, _total: int, _base=win_base) -> None:
            _dense_prog(min(_base + done, dense_total), dense_total)

        chunk = _scan_range_batched(
            scanner, analysis_path,
            stride=cfg.stride,
            start_frame=sf,
            end_frame=ef,
            imgsz=cfg.imgsz,
            tiled=cfg.use_tiled_dense,
            coarse_mode=False,
            batch_size=cfg.batch_size,
            progress_cb=_win_prog,
            should_cancel=should_cancel,
        )
        for t_sec, ok in chunk:
            all_times.append(t_sec)
            all_flags.append(ok)
            if ok:
                touch_times.append(t_sec)
        dense_done = min(dense_total, win_base + win_total)

    dense_diag = dict(scanner.diag)
    emit(0.92, "Merging possession intervals...")
    raw_intervals = extract_intervals(all_times, all_flags, cfg.possession)
    segments = build_segments(raw_intervals, duration_sec, cfg.export, touch_times=touch_times)

    seg_path = out_dir / "clip_segments.json"
    write_clip_segments_json(seg_path, segments, source_video=str(video_path.resolve()))

    total_diag = {
        "frames": coarse_diag.get("frames", 0) + dense_diag.get("frames", 0),
        "gated": coarse_diag.get("gated", 0) + dense_diag.get("gated", 0),
        "ball_seen": coarse_diag.get("ball_seen", 0) + dense_diag.get("ball_seen", 0),
        "myteam_seen": coarse_diag.get("myteam_seen", 0) + dense_diag.get("myteam_seen", 0),
        "active": coarse_diag.get("active", 0),
        "contact": dense_diag.get("contact", 0),
    }
    diag = {
        "coarse": coarse_diag,
        "dense": dense_diag,
        "total": total_diag,
        "n_raw_intervals": len(raw_intervals),
        "n_segments": len(segments),
        "n_windows": len(windows) if windows else 0,
        "used_proxy": used_proxy,
        "imgsz": cfg.imgsz,
        "coarse_imgsz": cfg.coarse_imgsz,
        "stride": cfg.stride,
        "two_pass": cfg.two_pass,
        "conf": cfg.conf,
        "use_tiled_dense": cfg.use_tiled_dense,
        "stride_dense": cfg.stride,
        "batch_size": cfg.batch_size,
    }
    (out_dir / "diag.json").write_text(json.dumps(diag, indent=2), encoding="utf-8")

    meta = {
        "source_video": str(video_path.resolve()),
        "analysis_video": str(analysis_path.resolve()),
        "used_proxy": used_proxy,
        "clip_segments_path": str(seg_path.resolve()),
        "n_segments": len(segments),
        "n_raw_intervals": len(raw_intervals),
        "diag": diag,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    emit(1.0, "Done")
    return meta
