"""ffmpeg proxy transcode — smaller + lower fps for long-match analysis."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable


def have_ffmpeg() -> bool:
    ff = os.environ.get("POLYFUT_FFMPEG")
    if ff and Path(ff).exists():
        return True
    return shutil.which("ffmpeg") is not None


def _ffmpeg_bin() -> str:
    ff = os.environ.get("POLYFUT_FFMPEG")
    if ff and Path(ff).exists():
        return ff
    return "ffmpeg"


def build_proxy(
    source_video: Path,
    out_proxy: Path,
    *,
    max_width: int = 960,
    fps: float = 12.0,
    crf: int = 28,
    duration_sec: float | None = None,
    progress_cb: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    """CFR proxy; wall-clock duration matches source."""
    if not have_ffmpeg():
        raise RuntimeError("ffmpeg not found on PATH. Install ffmpeg and retry.")

    out_proxy.parent.mkdir(parents=True, exist_ok=True)
    vf = f"scale='min({max_width},iw)':-2,fps={fps}"

    cmd = [
        _ffmpeg_bin(), "-y", "-i", str(source_video),
        "-an", "-vf", vf,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-progress", "pipe:1", "-nostats",
        str(out_proxy),
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    cur: dict[str, str] = {}
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.strip()
        if "=" in line:
            k, _, v = line.partition("=")
            cur[k] = v
        if line in ("progress=continue", "progress=end") and progress_cb and duration_sec:
            out_us = cur.get("out_time_us") or cur.get("out_time_ms") or "0"
            try:
                out_sec = int(out_us) / 1_000_000.0
                progress_cb(min(1.0, max(0.0, out_sec / max(duration_sec, 1.0))))
            except ValueError:
                pass
            cur = {}
    rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)

    meta = {
        "source_path": str(source_video.resolve()),
        "proxy_path": str(out_proxy.resolve()),
        "proxy_fps": fps,
        "scale_width": max_width,
    }
    meta_path = out_proxy.with_suffix(out_proxy.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
