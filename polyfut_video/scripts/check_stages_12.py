"""Specialized diagnostic for pipeline stages 1–2 (decode + shot filter)."""

from __future__ import annotations

import gc
import sys
import time
import tracemalloc
from pathlib import Path

# Allow running as script from repo root
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from polyfut_video.config import PipelineConfig
from polyfut_video.pipeline.decode import iter_frames, probe_video
from polyfut_video.pipeline.shot_filter import PIPELINE_VERSION, segment_and_classify_shots


def check_stages_12(video_path: str) -> dict:
    cfg = PipelineConfig()
    info = probe_video(video_path)
    print(f"=== Stage 1–2 check (pipeline v{PIPELINE_VERSION}) ===")
    print(f"Video: {video_path}")
    print(f"Probe: fps={info['fps']:.2f} frames={info['frame_count']} "
          f"dur={info['duration_sec']:.1f}s size={info['width']}x{info['height']}")

    # --- Stage 1: decode iterator sanity ---
    t0 = time.perf_counter()
    sample_n = max(1, cfg.shot_filter_sample_every_n)
    frames = list(iter_frames(video_path, target_width=cfg.target_width, sample_every_n=sample_n))
    decode_sec = time.perf_counter() - t0
    print(f"\nStage 1 decode: {len(frames)} sampled frames in {decode_sec:.1f}s "
          f"(every {sample_n} frames)")

    if not frames:
        return {"ok": False, "error": "no frames decoded"}

    # Timestamp / index monotonicity
    idx_ok = all(frames[i][0] <= frames[i + 1][0] for i in range(len(frames) - 1))
    ts_ok = all(frames[i][1] <= frames[i + 1][1] for i in range(len(frames) - 1))
    first_idx, first_ts, first_fr = frames[0]
    last_idx, last_ts, _ = frames[-1]
    print(f"  index monotonic: {idx_ok}  (first={first_idx}, last={last_idx})")
    print(f"  timestamp monotonic: {ts_ok}  (first={first_ts:.2f}s, last={last_ts:.2f}s)")
    print(f"  last_ts vs duration: {last_ts:.1f}s / {info['duration_sec']:.1f}s "
          f"({100 * last_ts / max(info['duration_sec'], 1):.1f}%)")
    print(f"  frame shape: {first_fr.shape}")

    # Expected sampled count
    expected = (info["frame_count"] + sample_n - 1) // sample_n if info["frame_count"] else len(frames)
    count_delta = abs(len(frames) - expected)
    print(f"  expected ~{expected} samples, got {len(frames)} (delta={count_delta})")

    # --- Stage 2: shot filter ---
    gc.collect()
    tracemalloc.start()
    t1 = time.perf_counter()
    frame_iter = iter_frames(video_path, target_width=cfg.target_width, sample_every_n=sample_n)
    shots = segment_and_classify_shots(frame_iter, cfg)
    shot_sec = time.perf_counter() - t1
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    gc.collect()

    n_main = sum(1 for s in shots if s.get("label") == "main_camera")
    n_disc = len(shots) - n_main
    print(f"\nStage 2 shot filter: {len(shots)} shot(s) in {shot_sec:.1f}s "
          f"(peak RAM ~{peak_bytes / 1024 / 1024:.1f} MB)")
    print(f"  main_camera={n_main}  discard={n_disc}")

    for i, s in enumerate(shots[:8]):
        print(f"  [{i}] {s['label']:12s} {s['start_sec']:7.1f}–{s['end_sec']:7.1f}s "
              f"frames {s['start_frame']}–{s['end_frame']}")
    if len(shots) > 8:
        print(f"  … +{len(shots) - 8} more")

    # Coverage checks
    if shots:
        span_start = min(s["start_sec"] for s in shots)
        span_end = max(s["end_sec"] for s in shots)
        main_shots = [s for s in shots if s["label"] == "main_camera"]
        main_dur = sum(s["end_sec"] - s["start_sec"] for s in main_shots)
        print(f"\nCoverage: shots span {span_start:.1f}–{span_end:.1f}s "
              f"(video {info['duration_sec']:.1f}s)")
        print(f"  main_camera total duration: {main_dur / 60:.1f} min")

    issues: list[str] = []
    if not idx_ok:
        issues.append("frame indices not monotonic")
    if not ts_ok:
        issues.append("timestamps not monotonic")
    if last_ts < info["duration_sec"] * 0.95:
        issues.append(f"last timestamp only {100*last_ts/info['duration_sec']:.0f}% of duration")
    if count_delta > sample_n:
        issues.append(f"sample count off by {count_delta}")
    if peak_bytes > 500 * 1024 * 1024:
        issues.append(f"peak RAM {peak_bytes/1024/1024:.0f} MB exceeds 500 MB budget")
    if not main_shots:
        issues.append("no main_camera shots detected")

    ok = len(issues) == 0
    print(f"\nResult: {'PASS' if ok else 'FAIL'}")
    for issue in issues:
        print(f"  - {issue}")

    return {
        "ok": ok,
        "issues": issues,
        "decode_sec": decode_sec,
        "shot_filter_sec": shot_sec,
        "peak_mb": peak_bytes / 1024 / 1024,
        "n_shots": len(shots),
        "n_main": n_main,
        "duration_sec": info["duration_sec"],
        "last_ts": last_ts,
    }


if __name__ == "__main__":
    default = ROOT / "uploads" / "4dd3c8580671.mp4"
    path = sys.argv[1] if len(sys.argv) > 1 else str(default)
    if not Path(path).is_file():
        print(f"Video not found: {path}")
        sys.exit(2)
    result = check_stages_12(path)
    sys.exit(0 if result.get("ok") else 1)
