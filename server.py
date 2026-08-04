"""PolyFut local server — Level 1 polyfut_video pipeline."""

from __future__ import annotations

import gc
import json
import os
import sys
import tempfile
import threading
import time
import traceback
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from keep_awake import during_analysis

os.environ.setdefault("OMP_NUM_THREADS", str(max(1, (os.cpu_count() or 4) // 2)))
os.environ.setdefault("MKL_NUM_THREADS", os.environ["OMP_NUM_THREADS"])

ROOT = Path(__file__).parent
DATA_ROOT = Path(os.environ.get("POLYFUT_DATA_DIR", str(ROOT)))
UPLOADS = DATA_ROOT / "uploads"
EXPORTS = DATA_ROOT / "exports"
FAKE_CV = os.environ.get("POLYFUT_FAKE_CV", "") not in ("", "0", "false", "False")
WEIGHTS = os.environ.get("POLYFUT_WEIGHTS", str(ROOT / "yolov8s.pt"))
DEVICE = os.environ.get("POLYFUT_DEVICE", "cpu")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    # Shared video utilities used by the v2 pipeline and the /api/teams kit
    # preview. probe_video imports fine at top level; team kit detection runs
    # in-process on a timed worker thread (see _detect_team_kits_isolated).
    from polyfut_video.pipeline.decode import probe_video
    PIPELINE_OK = True
    PIPELINE_IMPORT_ERR = ""
except Exception as exc:
    PIPELINE_OK = False
    PIPELINE_IMPORT_ERR = f"{type(exc).__name__}: {exc}"
    probe_video = None  # type: ignore

# v2 (ball-anchored, single-player) pipeline — experimental, alongside v1.
try:
    from polyfut_v2.config import PipelineV2Config
    from polyfut_v2.app_service import (
        calibration_from_clicks,
        build_one_seed_clip,
        build_review_track_for_item,
        build_seed_clips_index,
        hotspots_from_decisions,
        run_to_montage,
        taps_from_tracklet,
        warm_seed_detector,
    )
    from polyfut_v2.pipeline import play_ranges as pr
    PIPELINE_V2_OK = True
    PIPELINE_V2_ERR = ""
except Exception as exc:  # pragma: no cover
    PIPELINE_V2_OK = False
    PIPELINE_V2_ERR = f"{type(exc).__name__}: {exc}"
    run_to_montage = None  # type: ignore
    hotspots_from_decisions = None  # type: ignore
    pr = None  # type: ignore

UPLOADS.mkdir(parents=True, exist_ok=True)
EXPORTS.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, static_folder=str(ROOT), static_url_path="")


@app.after_request
def _cors_headers(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


@app.route("/api/<path:_any>", methods=["OPTIONS"])
@app.route("/api/health", methods=["OPTIONS"])
def _api_options(_any=None):
    return "", 204


app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024 * 1024
# None = no werkzeug multipart buffer cap (required for large video uploads).
app.config["MAX_FORM_MEMORY_SIZE"] = None

UPLOADS_TMP = UPLOADS / "_tmp"
UPLOADS_TMP.mkdir(parents=True, exist_ok=True)
tempfile.tempdir = str(UPLOADS_TMP)

JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()
TOKEN_META: dict[str, dict] = {}
JOB_START: dict[str, float] = {}

# Level 1 team picker slots (DBSCAN assigns kits during analysis)
TEAM_SLOTS = [
    {"id": "team_a", "label": "Team A", "hex": "#e23b3b"},
    {"id": "team_b", "label": "Team B", "hex": "#e6efe6"},
]

# region agent log
_DEBUG_LOG = Path(os.environ.get(
    "POLYFUT_DEBUG_LOG",
    str(ROOT / ".cursor" / "debug-9e74f8.log"),
))


def _dbg_log(hypothesis_id: str, location: str, message: str, data: dict, run_id: str = "team-color-debug") -> None:
    try:
        rec = {
            "sessionId": "9e74f8",
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        _DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(_DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass
# endregion


def _parse_progress_counts(status: str) -> tuple[int | None, int | None, str]:
    import re
    s = status or ""
    m = re.search(r"shot\s*(\d+)\s*/\s*(\d+)", s, re.I)
    if m:
        return int(m.group(1)), int(m.group(2)), "shots"
    m = re.search(r"(\d+)\s*/\s*(\d+)", s)
    if m:
        return int(m.group(1)), int(m.group(2)), "steps"
    m = re.search(r"(\d+)\s+live\s+shot", s, re.I)
    if m:
        return 0, int(m.group(1)), "shots"
    return None, None, ""


def _parse_stage(status: str) -> str:
    import re
    s = (status or "").lower()
    if "stage 1" in s or "stage 2" in s or "stage 1–2" in s or "stage 1-2" in s:
        return "shot_filter"
    if "stage 3" in s or "deadtime" in s:
        return "deadtime"
    if "stage 4" in s or "stage 5" in s or "stage 6" in s or "stage 7" in s or re.search(r"(shot|chunk)\s+\d+/\d+", s):
        return "inference"
    if "stage 8" in s or "possession" in s:
        return "possession"
    if "stage 9" in s or "timestamp" in s:
        return "timestamps"
    if "done" in s:
        return "done"
    return "running"


def _set_job(job_id: str, **kw) -> None:
    with JOBS_LOCK:
        JOBS.setdefault(job_id, {}).update(kw)
    _persist_job(job_id)


def _job_state_path(job_id: str) -> Path:
    return EXPORTS / job_id / "job_state.json"


def _persist_job(job_id: str) -> None:
    """Write job progress to disk so status survives tab close (server keeps running)."""
    try:
        with JOBS_LOCK:
            j = JOBS.get(job_id)
            if not j:
                return
            payload = dict(j)
        payload["job_id"] = job_id
        out_dir = EXPORTS / job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        _job_state_path(job_id).write_text(
            json.dumps(payload, indent=2, default=str),
            encoding="utf-8",
        )
    except Exception:
        pass


def _load_job_from_disk(job_id: str) -> dict | None:
    path = _job_state_path(job_id)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("state") == "running":
            data["state"] = "interrupted"
            data["status"] = "Analysis was interrupted (server restarted)"
            data["stage"] = "error"
            data["error"] = (
                "The analysis stopped when the server restarted. "
                "Start a new run — your uploaded video is still saved if you use the same session token."
            )
        with JOBS_LOCK:
            JOBS[job_id] = data
        if data.get("started_at"):
            JOB_START[job_id] = float(data["started_at"])
        return dict(data)
    except Exception:
        return None


def _hydrate_jobs_from_disk() -> None:
    """Load persisted jobs into memory on server boot."""
    try:
        for d in EXPORTS.iterdir():
            if not d.is_dir():
                continue
            job_id = d.name
            if job_id in JOBS:
                continue
            _load_job_from_disk(job_id)
    except Exception:
        pass


def _find_running_job_for_token(token: str) -> str | None:
    _hydrate_jobs_from_disk()
    with JOBS_LOCK:
        for jid, j in JOBS.items():
            if j.get("token") == token and j.get("state") == "running":
                return jid
    return None


def _get_job(job_id: str) -> dict | None:
    with JOBS_LOCK:
        j = JOBS.get(job_id)
        if j:
            return dict(j)
    loaded = _load_job_from_disk(job_id)
    return loaded


def _match_metadata_from_form() -> dict:
    """Optional setup fields sent with /api/process."""
    out: dict = {}
    for key in ("opponent", "match_date", "position"):
        val = (request.form.get(key) or "").strip()
        if val:
            out[key] = val
    for key in ("score_us", "score_them"):
        raw = request.form.get(key)
        if raw is not None and str(raw).strip() != "":
            try:
                out[key] = int(raw)
            except ValueError:
                pass
    return out


def _match_metadata_from_json() -> dict:
    data = request.get_json(silent=True) or {}
    out: dict = {}
    for key in ("opponent", "match_date", "position"):
        if key in data and data[key] is not None:
            out[key] = str(data[key]).strip()
    for key in ("score_us", "score_them"):
        if key in data and data[key] is not None and str(data[key]).strip() != "":
            try:
                out[key] = int(data[key])
            except (TypeError, ValueError):
                pass
    return out


def _segment_count_for_job(job_id: str, j: dict) -> int:
    segs = j.get("segments")
    if isinstance(segs, list):
        return len(segs)
    seg_path = EXPORTS / job_id / "clip_segments.json"
    if seg_path.is_file():
        try:
            data = json.loads(seg_path.read_text(encoding="utf-8"))
            return len(data.get("segments") or [])
        except Exception:
            pass
    return 0


def _session_path(job_id: str) -> Path:
    return EXPORTS / job_id / "session_data.json"


def _session_summary(job_id: str) -> dict:
    path = _session_path(job_id)
    if not path.is_file():
        return {"has_session": False, "n_actions": 0}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        sess = data.get("session") or data
        stats = sess.get("matchStats") or []
        benches = sess.get("benchBlocks") or []
        hybrid = sess.get("hybridResults")
        has = bool(stats) or bool(benches) or hybrid is not None
        return {"has_session": has, "n_actions": len(stats)}
    except Exception:
        return {"has_session": False, "n_actions": 0}


def _catalogue_entry(job_id: str, j: dict | None = None) -> dict | None:
    j = j or _get_job(job_id)
    if not j or j.get("state") != "done":
        return None
    token = j.get("token")
    video_ok = bool(token and (UPLOADS / f"{token}.mp4").is_file())
    sess = _session_summary(job_id)
    return {
        "job_id": job_id,
        "token": token,
        "my_team": j.get("my_team", "team_a"),
        "opponent": j.get("opponent") or "",
        "match_date": j.get("match_date") or "",
        "score_us": j.get("score_us"),
        "score_them": j.get("score_them"),
        "position": j.get("position") or "",
        "n_hotspots": _segment_count_for_job(job_id, j),
        "n_actions": j.get("n_actions", sess.get("n_actions", 0)),
        "has_session": bool(j.get("has_session", sess.get("has_session"))),
        "analysed_at": j.get("finished_at") or j.get("started_at"),
        "video_available": video_ok,
        "note": j.get("note") or "",
    }


_hydrate_jobs_from_disk()


@app.route("/api/video/<token>")
def serve_video(token: str):
    import re
    if not re.fullmatch(r"[a-f0-9]{12}", token or ""):
        return jsonify({"error": "invalid token"}), 400
    path = UPLOADS / f"{token}.mp4"
    if not path.is_file():
        return jsonify({"error": "video not found"}), 404
    return send_from_directory(str(UPLOADS), f"{token}.mp4", mimetype="video/mp4")


@app.route("/")
def index():
    return send_from_directory(str(ROOT), "index.html")


@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "pipeline": "polyfut_video",
        "pipeline_ready": PIPELINE_OK,
        "pipeline_error": PIPELINE_IMPORT_ERR,
        "weights": str(WEIGHTS),
        "device": DEVICE,
        "fake_cv": FAKE_CV,
        "data_dir": str(DATA_ROOT),
    })


# --- AI scout report: where the client should send report requests ---------- #
# The Groq key used to be pasted in by each user and kept in localStorage. It
# now lives in a Modal secret behind a proxy endpoint (see ai_backend/), so the
# client needs to be told the URL + shared app token. Serving them from here —
# rather than baking them into script.js — means the endpoint can be moved or
# the token rotated by editing one file, without rebuilding and reshipping the
# installer to every user.
#
# Resolution order: env vars (POLYFUT_AI_PROXY_URL / POLYFUT_AI_APP_TOKEN) win,
# then ai_config.json beside the app, then next to the user's data dir.

def _ai_config_paths() -> list[Path]:
    return [ROOT / "ai_config.json", DATA_ROOT / "ai_config.json"]


def _load_ai_config() -> dict:
    url = os.environ.get("POLYFUT_AI_PROXY_URL", "").strip()
    token = os.environ.get("POLYFUT_AI_APP_TOKEN", "").strip()
    if not (url and token):
        for path in _ai_config_paths():
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            url = url or str(doc.get("proxy_url", "")).strip()
            token = token or str(doc.get("app_token", "")).strip()
            if url and token:
                break
    return {"proxy_url": url, "app_token": token}


@app.route("/api/ai_config")
def ai_config():
    """Tell the client how to reach the AI proxy.

    ``enabled`` false means no proxy is configured on this install — the client
    falls back to the user's own pasted Groq key, which is exactly the old
    behaviour, so an unconfigured build keeps working instead of losing the
    feature outright.
    """
    cfg = _load_ai_config()
    enabled = bool(cfg["proxy_url"] and cfg["app_token"])
    return jsonify({
        "enabled": enabled,
        "proxy_url": cfg["proxy_url"] if enabled else "",
        # Shipped to the client by design: the client must present it to the
        # proxy. It is a traffic filter, not a secret — the Groq key is what
        # stays on the server and is never sent here.
        "app_token": cfg["app_token"] if enabled else "",
    })


def _fake_segments(duration_sec: float) -> list[dict]:
    import random
    random.seed(7)
    segs = []
    t = 20.0
    horizon = min(duration_sec or 600.0, 600.0)
    while t < horizon - 20:
        start = t + random.uniform(0, 25)
        end = start + random.uniform(8, 20)
        segs.append({
            "type": "hotspot",
            "start": round(start, 1),
            "end": round(end, 1),
            "core_start": round(start + 2, 1),
            "core_end": round(end - 2, 1),
            "action_triggers": [round((start + end) / 2, 1)],
        })
        t = end + random.uniform(15, 45)
    return segs


def _probe_duration(video_path: Path) -> float:
    try:
        if probe_video:
            return float(probe_video(str(video_path))["duration_sec"])
    except Exception:
        pass
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        return n / fps if fps > 0 else 0.0
    except Exception:
        return 0.0


def _warm_kit_detector() -> None:
    """Pre-build the kit-preview model at server boot, off the request path.

    Measured cause of job fbe036aab37d timing out on a 3-minute, 10MB clip (see
    .cursor/debug-9e74f8.log): the request landed 271s after boot — the FIRST
    /api/teams call in this process, so `_MODEL_CACHE` (detection.py) was empty
    — and the worker thread spent the entire 300s budget on cold torch/
    ultralytics/cv2 imports plus building the YOLO model, never reaching actual
    detection (which took 4.4s once warm, measured separately). This machine
    has Windows Defender real-time protection on, which is exactly the
    "AV-heavy machine" case the timeout-vs-subprocess history already called
    out in _detect_team_kits_isolated's docstring.

    Building the model once here means the model cache is already populated by
    the time a real upload arrives — the request then only ever pays for actual
    detection, not the import. Runs in a daemon thread so it can't block or
    delay server startup; any failure here just means the first real request
    pays the cost instead, same as before this existed.
    """
    try:
        import numpy as np
        from polyfut_video.pipeline.detection import Detector, DetectConfig

        det = Detector(DetectConfig(weights=WEIGHTS, device=DEVICE, imgsz=640))
        # One dummy inference too — covers any further lazy first-call init
        # inside ultralytics itself (e.g. backend/graph setup), not just import.
        det.detect_frame(np.zeros((640, 640, 3), dtype=np.uint8))
    except Exception:
        traceback.print_exc()


def _detect_team_kits_isolated(video_path: str, timeout_sec: float = 480.0) -> tuple[list[dict] | None, str | None]:
    """Run kit preview in-process on a worker thread with a timeout.

    This used to spawn a fresh Python subprocess "so YOLO RAM is released".
    Two problems: (1) the v2 pipeline loads YOLO into this process anyway, so
    isolation bought no lasting RAM benefit; (2) on managed/AV-heavy machines a
    fresh interpreter spawn plus the torch/ultralytics import costs 1-4
    *minutes* before any detection runs (measured 194s import vs 57s of actual
    work on a 90-min match), which blew the (then 300s) cap and silently fell
    back to default kit colours. In-process, the import is paid once per server
    lifetime — `_warm_kit_detector()` now pays it at boot instead of on the
    first request, which is what actually blew this budget on job
    fbe036aab37d (271s after boot, then the full old 300s cap with nothing to
    show for it — see that function's docstring). timeout_sec is generous
    (480s) as a backstop for the case a request arrives before boot-time
    warming has finished; once warm, real detection takes single-digit
    seconds. The worker thread keeps the old contract either way: if detection
    hangs, the caller still gets defaults instead of a stuck request (the
    orphaned thread is daemon and dies with the server).
    """
    result: dict = {}

    def _work() -> None:
        try:
            from polyfut_video.pipeline.team_preview import detect_team_kits
            result["kits"] = detect_team_kits(
                video_path, weights=str(WEIGHTS), device=str(DEVICE),
            )
        except Exception as exc:  # noqa: BLE001 — reported to caller
            result["err"] = f"{type(exc).__name__}: {exc}"
        finally:
            gc.collect()  # drop the detector's model weights promptly

    th = threading.Thread(target=_work, daemon=True, name="kit-preview")
    th.start()
    th.join(timeout_sec)
    if th.is_alive():
        return None, "kit detection timed out"
    if "err" in result:
        return None, result["err"]
    return result.get("kits"), None



def _validate_video(video_path: Path) -> str | None:
    """Return a user-facing error if the video is empty/unreadable, else None.

    Catches failed downloads/uploads (0-byte or truncated files) up front so the
    app shows a clear message instead of silently falling back to default kit
    colours and running a doomed analysis.
    """
    try:
        size = video_path.stat().st_size
    except OSError:
        return "The uploaded file is missing. Please try uploading again."
    if size < 10_000:
        return (
            f"This video file is empty or incomplete ({size} bytes). The download "
            f"or upload didn't finish — please re-download the clip and try again."
        )
    if probe_video is None:
        return None  # no decoder available to validate; let it proceed
    try:
        info = probe_video(str(video_path))
    except Exception:
        return (
            "This file couldn't be opened as a video — it may be corrupt or an "
            "unsupported format. Please re-export or re-download it and try again."
        )
    if int(info.get("frame_count") or 0) < 1 or float(info.get("duration_sec") or 0) <= 0:
        return (
            "This video has no readable frames — it may be corrupt. Please "
            "re-export or re-download it and try again."
        )
    return None


@app.route("/api/teams", methods=["POST"])
def teams():
    """Upload video and return token + Team A/B slots for user selection."""
    if "video" not in request.files:
        return jsonify({"error": "No video file in request."}), 400
    token = uuid.uuid4().hex[:12]
    video_path = UPLOADS / f"{token}.mp4"
    # region agent log
    _dbg_log("M1", "server.py:teams:pre_save", "upload incoming", {
        "token": token,
        "content_length": request.content_length,
        "content_mb": round((request.content_length or 0) / 1e6, 2),
    }, run_id="memory-fix-v1")
    # endregion
    try:
        request.files["video"].save(str(video_path))
    except MemoryError:
        # region agent log
        _dbg_log("M1", "server.py:teams:save", "MemoryError on save", {
            "token": token,
            "content_mb": round((request.content_length or 0) / 1e6, 2),
        }, run_id="memory-fix-v1")
        # endregion
        return jsonify({
            "error": "Server ran out of memory while receiving the video. "
            "Restart the server and try again. If this persists, close other apps.",
        }), 507
    except Exception as exc:
        # region agent log
        _dbg_log("M1", "server.py:teams:save", "upload save failed", {
            "token": token,
            "error": str(exc)[:300],
            "content_mb": round((request.content_length or 0) / 1e6, 2),
        }, run_id="memory-fix-v1")
        # endregion
        return jsonify({"error": f"Upload failed: {exc}"}), 500
    # region agent log
    _dbg_log("M1", "server.py:teams:post_save", "upload saved", {
        "token": token,
        "video_mb": round(video_path.stat().st_size / 1e6, 2),
    }, run_id="memory-fix-v1")
    # endregion
    TOKEN_META[token] = {"video": str(video_path)}

    verr = _validate_video(video_path)
    if verr:
        _dbg_log("H1", "server.py:teams:invalid_video", "video validation failed", {
            "token": token, "size": video_path.stat().st_size if video_path.exists() else 0,
            "error": verr,
        })
        return jsonify({"error": verr, "invalid_video": True}), 400

    # region agent log
    _dbg_log("H1", "server.py:teams:entry", "teams upload", {
        "token": token,
        "video_path": str(video_path),
        "video_mb": round(video_path.stat().st_size / 1e6, 2),
        "pipeline_ok": PIPELINE_OK,
        "fake_cv": FAKE_CV,
        "team_slots_hex": [t["hex"] for t in TEAM_SLOTS],
    })
    # endregion

    if FAKE_CV or not PIPELINE_OK:
        reason = "POLYFUT_FAKE_CV is enabled" if FAKE_CV else f"CV unavailable: {PIPELINE_IMPORT_ERR}"
        resp = {
            "token": token,
            "demo": True,
            "mode": "demo",
            "warning": (
                f"Demo mode — not analysing your video. {reason}. "
                "Run: pip install -r requirements.txt (from PolyFut-Clean) "
                "and start server without POLYFUT_FAKE_CV."
            ),
            "pipeline_ok": PIPELINE_OK,
            "pipeline_error": PIPELINE_IMPORT_ERR,
            "teams": TEAM_SLOTS,
        }
        # region agent log
        _dbg_log("H3", "server.py:teams:demo", "demo path teams", {
            "reason": reason, "teams_hex": [t["hex"] for t in resp["teams"]],
        })
        # endregion
        return jsonify(resp)

    teams_out = list(TEAM_SLOTS)
    ran_detection = False
    detect_error = None
    try:
        detected, detect_error = _detect_team_kits_isolated(str(video_path))
        if detected and len(detected) >= 2:
            teams_out = detected
            ran_detection = True
    except Exception as exc:
        detect_error = str(exc)
        traceback.print_exc()
    gc.collect()

    # NOTE: seed prefetch is deliberately NOT started here any more. It used to
    # fire as soon as kit detection finished, but the moments it picks depend on
    # the playing-time window, which the user declares on the step before this
    # one. Building four whole-match clips here would spend 4-8 CPU-minutes on
    # moments the user may not have been on the pitch for, and they'd be thrown
    # away the moment a window arrived. /api/v2/playing_time starts it instead —
    # including for "whole match", which the client posts when the step is
    # skipped, so the head start is preserved.

    resp = {
        "token": token,
        "demo": False,
        "mode": "live",
        "pipeline_ok": True,
        "teams": teams_out,
        "kits_detected": ran_detection,
        # Why detection fell back to defaults (null on success) — kept for the
        # browser console; `warning` below is the same information surfaced to
        # the user, since a silent fallback to red/white gave no indication
        # anything had gone wrong (this used to only ever reach the console).
        "kit_detect_error": detect_error,
        "warning": (
            None if ran_detection else
            "Couldn't detect kit colours automatically ({}) — using default "
            "colours. Pick your team below; you can still tag the match "
            "normally.".format(detect_error or "not enough visible players in the sampled frames")
        ),
        "note": "Pick which side you played for.",
    }
    # region agent log
    _dbg_log("H1", "server.py:teams:live", "teams response", {
        "teams_hex": [t.get("hex") for t in teams_out],
        "ran_color_detection": ran_detection,
        "detect_error": detect_error,
        "is_default_red_white": (
            len(teams_out) == 2
            and teams_out[0].get("hex") == "#e23b3b"
            and teams_out[1].get("hex") == "#e6efe6"
        ),
    })
    # endregion
    return jsonify(resp)




@app.route("/api/process/active")
def active_jobs():
    """Running jobs on this server (for resume after tab close without localStorage)."""
    with JOBS_LOCK:
        runs = [
            {
                "job_id": jid,
                "token": j.get("token"),
                "my_team": j.get("my_team"),
                "progress": j.get("progress", 0.0),
                "status": j.get("status", ""),
                "state": j.get("state"),
                "progress_current": j.get("progress_current"),
                "progress_total": j.get("progress_total"),
            }
            for jid, j in JOBS.items()
            if j.get("state") == "running"
        ]
    return jsonify({"runs": runs})


@app.route("/api/process/status/<job_id>")
def status(job_id: str):
    j = _get_job(job_id)
    if not j:
        return jsonify({"error": "unknown job", "state": "unknown"}), 404
    return jsonify({
        "job_id": job_id,
        "token": j.get("token"),
        "my_team": j.get("my_team"),
        "progress": j.get("progress", 0.0),
        "status": j.get("status", ""),
        "state": j.get("state", "running"),
        "stage": j.get("stage", "running"),
        "stage_progress": j.get("stage_progress", j.get("progress", 0.0)),
        "elapsed_sec": j.get("elapsed_sec", 0.0),
        "progress_current": j.get("progress_current"),
        "progress_total": j.get("progress_total"),
        "progress_unit": j.get("progress_unit", ""),
        "status_updated_at": j.get("status_updated_at"),
        "segments": j.get("segments"),
        "segments_partial": j.get("segments_partial"),
        "possession_timeline": j.get("possession_timeline"),
        "timings": j.get("timings"),
        "error": j.get("error"),
        "note": j.get("note"),
        # v2-only fields (None/absent for v1 jobs)
        "pipeline_version": j.get("pipeline_version", "v1"),
        "montage": j.get("montage"),
        "hotspots": j.get("hotspots"),
        "warnings": j.get("warnings"),
        "n_review": j.get("n_review"),
        "n_candidates": j.get("n_candidates"),
        "detected_ratio": j.get("detected_ratio"),
        "play_ranges": j.get("play_ranges"),
        "on_pitch_sec": j.get("on_pitch_sec"),
    })


@app.route("/api/process/<job_id>", methods=["DELETE"])
def cancel(job_id: str):
    j = _get_job(job_id)
    if not j:
        return jsonify({"error": "unknown job"}), 404
    _set_job(job_id, cancel=True, state="cancelled", status="Cancelled", stage="cancelled")
    return jsonify({"ok": True, "discarded": True})


# --------------------------------------------------------------------------- #
# v2 pipeline (experimental, alongside v1)
# --------------------------------------------------------------------------- #

def _run_v2_job(job_id: str, video_path: Path, seed_taps: list, out_dir: Path,
                opponent_hex: str | None = None, play_ranges: list | None = None,
                opponent_hexes: list | None = None,
                my_team_hexes: list | None = None,
                calibration: dict | None = None) -> None:
    def progress(frac: float, msg: str) -> None:
        elapsed = time.time() - JOB_START.get(job_id, time.time())
        cur, tot, unit = _parse_progress_counts(msg)
        _set_job(
            job_id, progress=frac, status=msg, stage=_parse_stage(msg),
            stage_progress=frac, elapsed_sec=round(elapsed, 1),
            progress_current=cur, progress_total=tot, progress_unit=unit,
            status_updated_at=time.time(),
        )

    def should_cancel() -> bool:
        j = _get_job(job_id)
        return bool(j and j.get("cancel"))

    with during_analysis():
        try:
            cfg = PipelineV2Config(ball_weights=WEIGHTS, player_weights=WEIGHTS, device=DEVICE)
            progress(0.02, f"v2 pipeline: weights={WEIGHTS}, device={DEVICE}")
            result = run_to_montage(
                str(video_path), seed_taps=seed_taps, cfg=cfg,
                progress=progress, should_cancel=should_cancel,
                opponent_hex=opponent_hex, play_ranges=play_ranges or None,
                opponent_hexes=opponent_hexes or None,
                my_team_hexes=my_team_hexes or None,
                calibration=calibration,
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "montage.json").write_text(
                json.dumps(result, indent=2), encoding="utf-8")
            # "review" state: montage is ready, awaiting me/not-me decisions.
            _set_job(
                job_id, progress=1.0, status="Review your touches",
                state="review", stage="review",
                pipeline_version="v2",
                montage=result["montage"],
                hotspots=result["hotspots"],
                warnings=result["warnings"],
                n_review=result["n_review"],
                n_candidates=result["n_candidates"],
                detected_ratio=result["detected_ratio"],
                duration_sec=result.get("duration_sec"),
                play_ranges=result.get("play_ranges") or [],
                play_ranges_padded=result.get("play_ranges_padded") or [],
                on_pitch_sec=result.get("on_pitch_sec"),
                attribution=result.get("attribution"),
                calibration_status=result.get("calibration_status"),
                # Ball tracking is ~76% of a run, and how long it takes depends
                # on how many network calls each frame costs — an ROI hit is one,
                # an ROI miss plus a full re-acquire is two. app_service has
                # always returned these counters and this whitelist dropped them,
                # so "why did this run take three hours" could only be answered
                # by re-running it. Persist them: with n_analysed_frames they
                # make ms-per-frame and calls-per-frame readable straight off
                # job_state.json.
                ball_detector_stats=result.get("ball_detector_stats"),
                ball_sanity=result.get("ball_sanity"),
                n_analysed_frames=result.get("n_samples"),
                # Which kit colours the run was actually given. The team gate
                # keeps 98% of your touches with the right pair and 16-26% with
                # a wrong one, so "why did it miss my player" is unanswerable
                # without them — as it was for the ISB/TAS report.
                seed=result.get("seed"),
                # Needed again at decision time: hotspots are rebuilt from
                # scratch there, and the possession extension needs the ball.
                ball_track=result.get("ball_track"),
                timings=result.get("stage_timings_sec") or result.get("timings_sec"),
                finished_at=time.time(),
            )
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            _set_job(job_id, state="error", status="Error", stage="error",
                     error=f"{type(exc).__name__}: {exc}", finished_at=time.time())


@app.route("/api/v2/process", methods=["POST"])
def v2_process():
    if not PIPELINE_V2_OK:
        return jsonify({"error": f"v2 pipeline unavailable: {PIPELINE_V2_ERR}"}), 503
    token = request.form.get("token")
    if not token:
        return jsonify({"error": "token required"}), 400
    video_path = UPLOADS / f"{token}.mp4"
    if not video_path.exists():
        return jsonify({"error": "unknown or expired token"}), 400

    verr = _validate_video(video_path)
    if verr:
        return jsonify({"error": verr, "invalid_video": True}), 400

    seed_taps = []
    raw = request.form.get("seed_taps")
    if raw:
        try:
            seed_taps = json.loads(raw)
        except (ValueError, TypeError):
            return jsonify({"error": "seed_taps must be JSON"}), 400

    # Optional pitch calibration from the "mark out the pitch" screen. Bad or
    # missing JSON is ignored rather than failing the run: calibration is an
    # enhancement, and losing a whole analysis over it would be a poor trade.
    calibration = None
    raw_cal = request.form.get("calibration")
    if raw_cal:
        try:
            parsed = json.loads(raw_cal)
            if isinstance(parsed, dict) and parsed.get("clicks"):
                # Re-fit here so the pipeline runs on a calibration produced by
                # the same code that consumes it; the browser's fit rides along
                # as a starting point and is verified, not trusted.
                calibration = calibration_from_clicks(
                    parsed.get("clicks") or [],
                    frame_width=int(parsed.get("frame_width") or 0) or 640,
                    frame_height=int(parsed.get("frame_height") or 0) or 360,
                    pitch_length_m=float(parsed.get("pitch_length_m") or 100.0),
                    pitch_width_m=float(parsed.get("pitch_width_m") or 64.0),
                    seed_params=parsed.get("params"),
                    free_pitch_size=bool(parsed.get("free_pitch_size")),
                )
            elif isinstance(parsed, dict) and parsed.get("params"):
                calibration = parsed
        except (ValueError, TypeError):
            calibration = None

    opponent_hex = request.form.get("opponent_hex") or None
    # Full colour sets for multi-coloured kits (red/blue halves, hoops). Optional:
    # older clients send only the single `*_hex` fields and the pipeline falls
    # back to those. Unparsable JSON is ignored rather than failing the run — a
    # bad swatch list must not cost the user a whole analysis.
    def _hex_list(field: str) -> list:
        raw_l = request.form.get(field)
        if not raw_l:
            return []
        try:
            vals = json.loads(raw_l)
        except (ValueError, TypeError):
            return []
        if not isinstance(vals, list):
            return []
        return [v for v in vals if isinstance(v, str) and v.startswith("#") and len(v) == 7]

    opponent_hexes = _hex_list("opponent_hexes")
    my_team_hexes = _hex_list("my_team_hexes")

    # The stored window wins over the form copy — see _effective_playing_time.
    client_ranges = None
    raw_ranges = request.form.get("play_ranges")
    if raw_ranges:
        try:
            client_ranges = json.loads(raw_ranges)
        except (ValueError, TypeError):
            return jsonify({"error": "play_ranges must be JSON"}), 400
    play_ranges = _effective_playing_time(token, client_ranges)

    existing = _find_running_job_for_token(token)
    if existing:
        return jsonify({"job_id": existing, "resumed": True, "token": token})

    job_id = uuid.uuid4().hex[:12]
    job_dir = EXPORTS / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    JOB_START[job_id] = started
    _set_job(
        job_id, progress=0.0, status="Queued", state="running", stage="init",
        cancel=False, error=None, token=token, pipeline_version="v2",
        started_at=started, video_path=str(video_path),
        play_ranges=[list(r) for r in play_ranges],
        **_match_metadata_from_form(),
    )
    threading.Thread(
        target=_run_v2_job,
        args=(job_id, video_path, seed_taps, job_dir, opponent_hex, play_ranges,
              opponent_hexes, my_team_hexes, calibration),
        daemon=False,
    ).start()
    return jsonify({"job_id": job_id, "token": token, "pipeline_version": "v2",
                    "play_ranges": [list(r) for r in play_ranges]})


# --- pitch calibration: frames for the "mark out the pitch" screen ---------- #

@app.route("/api/v2/calibration_frames", methods=["POST"])
def v2_calibration_frames():
    """Frames the user can click pitch landmarks on, spread across the video.

    Returned at the video's native resolution so faint markings stay as visible
    as they can be; the pipeline runs at ``target_width`` and the calibration
    records which width the clicks were made at, so the two are reconciled later.
    """
    if not PIPELINE_V2_OK:
        return jsonify({"error": f"v2 pipeline unavailable: {PIPELINE_V2_ERR}"}), 503
    token = (request.form.get("token") or "").strip()
    video_path = UPLOADS / f"{token}.mp4"
    if not token or not video_path.exists():
        return jsonify({"error": "unknown or expired token"}), 400

    try:
        count = max(2, min(12, int(request.form.get("count") or 6)))
    except (TypeError, ValueError):
        count = 6

    # Prefer moments the user says they were playing — those are the frames whose
    # pitch actually matters, and a calibration anchored there needs the camera
    # track to carry it the shortest distance.
    ranges = []
    raw_r = request.form.get("play_ranges")
    if raw_r:
        try:
            parsed = json.loads(raw_r)
            if isinstance(parsed, list):
                ranges = [(float(a), float(b)) for a, b in parsed if b > a]
        except (ValueError, TypeError):
            ranges = []

    import base64
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = (total / fps) if fps else 0.0
    if duration <= 0:
        cap.release()
        return jsonify({"error": "could not read video"}), 400

    span = ranges[0] if ranges else (0.0, duration)
    lo = max(0.0, min(span[0], duration - 1.0))
    hi = max(lo + 1.0, min(span[1], duration))
    # skip the very edges: kickoff and whistles are often graphics or crowd
    times = [lo + (hi - lo) * (i + 1) / (count + 1) for i in range(count)]

    frames = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
        ok, frame = cap.read()
        if not ok:
            continue
        ok, buf = cv2.imencode(".jpg", frame,
                               [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok:
            continue
        h, w = frame.shape[:2]
        frames.append({
            "frame_index": int(round(t * fps)),
            "t_sec": round(float(t), 3),
            "width": int(w),
            "height": int(h),
            "jpeg_b64": base64.b64encode(buf.tobytes()).decode("ascii"),
        })
    cap.release()
    if not frames:
        return jsonify({"error": "could not extract frames"}), 400
    return jsonify({
        "frames": frames,
        "duration_sec": round(duration, 2),
        "fps": round(float(fps), 4),
    })


# --- seed clips: enhanced 3s clips with tracked, clickable player nodes ------ #

_SEED_CLIP_LOCKS: dict[str, threading.Lock] = {}
_SEED_CLIP_LOCKS_GUARD = threading.Lock()


def _seed_clip_lock(key: str) -> threading.Lock:
    with _SEED_CLIP_LOCKS_GUARD:
        lk = _SEED_CLIP_LOCKS.get(key)
        if lk is None:
            lk = _SEED_CLIP_LOCKS[key] = threading.Lock()
        return lk


_SEED_PREFETCH_TOKENS: set[str] = set()
_SEED_PREFETCH_GUARD = threading.Lock()


# --- playing-time window: the ranges the user says they were on the pitch ---- #
# Stored per upload token, next to that token's seed clips, so it survives a
# server restart and both the seed step and the analysis run read the same
# source of truth. The client also sends its copy on each call; the file wins,
# because the prefetch worker running in the background has no other way to
# learn the user narrowed the window after it started.

def _playing_time_path(token: str) -> Path:
    return EXPORTS / "seed" / token / "playing_time.json"


def _load_playing_time(token: str) -> list[tuple[float, float]]:
    """The token's stored ranges, or [] for "whole match"."""
    if pr is None:
        return []
    path = _playing_time_path(token)
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return pr.normalize_ranges(doc.get("ranges"), doc.get("duration_sec"))


def _save_playing_time(token: str, ranges: list, duration_sec: float) -> None:
    path = _playing_time_path(token)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "ranges": [list(r) for r in ranges],
        "duration_sec": duration_sec,
        "saved_at": time.time(),
    }), encoding="utf-8")


def _effective_playing_time(token: str, client_ranges) -> list[tuple[float, float]]:
    """Ranges to use for a request: the stored file if present, else whatever
    the client sent. The file wins so a stale client copy (an old tab, a resumed
    session) can never widen the window behind the user's back."""
    stored = _load_playing_time(token)
    if stored:
        return stored
    if pr is None:
        return []
    return pr.normalize_ranges(client_ranges)


@app.route("/api/v2/playing_time", methods=["POST"])
def v2_playing_time():
    """Persist the user's on-pitch ranges and start the seed prefetch for them.

    This is also the trigger for the prefetch: building the four seed clips
    costs 1-2 CPU-minutes each, and before the window is known there is no way
    to know which moments to build. Starting here means no clip is ever built
    for a moment the user wasn't playing.
    """
    if not PIPELINE_V2_OK:
        return jsonify({"error": f"v2 pipeline unavailable: {PIPELINE_V2_ERR}"}), 503
    payload = request.get_json(silent=True) or {}
    token = payload.get("token", "")
    video_path, err = _seed_token_video(token)
    if err:
        return err

    duration = _probe_duration(video_path)
    ranges = pr.normalize_ranges(payload.get("ranges"), duration)
    # An explicit whole-match choice is stored as [] — same as never asking —
    # so cache keys, warnings and the pipeline all stay on the old path.
    if pr.is_whole_match(ranges, duration):
        ranges = []
    _save_playing_time(token, ranges, duration)

    my_kit_hex = payload.get("my_team_hex") or None
    _start_seed_prefetch(token, str(video_path), my_kit_hex, ranges)

    return jsonify({
        "ok": True,
        "token": token,
        "duration_sec": round(duration, 3),
        "ranges": [list(r) for r in ranges],
        "on_pitch_sec": round(pr.total_seconds(ranges) or duration, 3),
        "whole_match": not ranges,
    })


def _start_seed_prefetch(token: str, video_path: str, my_kit_hex: str | None,
                         play_ranges: list | None = None) -> None:
    """Build the four reroll-0 seed clips into the disk cache in the background.

    Each clip costs ~1-2 minutes of moment search + player tracking on CPU;
    built on demand, the user stared at "Loading clip" for that long on every
    clip. Building them ahead (kicked off as soon as the team screen returns,
    while the user is still picking a side) means by the time they reach the
    tap-yourself step most clips are already on disk and load instantly.

    Idempotent per (token, window). Uses the same per-clip locks as the request
    path, so a live request for a clip mid-build simply waits for that one clip
    and duplicated work is impossible. The avoid-list chains exactly like the
    client's own sequential requests, so prefetched moments match what the
    on-demand path would have chosen. ``my_kit_hex`` may be None when the user
    hasn't picked a side yet — the moment search then skips its (soft)
    kit-visibility preference, an acceptable trade for the head start.

    Before each clip the worker re-reads the token's stored window and stops if
    it changed: if the user goes back and edits their periods, the clips this
    chain is building are for moments they may not have played, and two chains
    grinding at once would just contend for the same CPU.
    """
    if not PIPELINE_V2_OK:
        return
    ranges = list(play_ranges or [])
    key = f"{token}:{pr.ranges_hash(ranges)}"
    with _SEED_PREFETCH_GUARD:
        if key in _SEED_PREFETCH_TOKENS:
            return
        _SEED_PREFETCH_TOKENS.add(key)

    def _work() -> None:
        try:
            cfg = PipelineV2Config(ball_weights=WEIGHTS, player_weights=WEIGHTS, device=DEVICE)
            seed_dir = EXPORTS / "seed" / token
            seed_dir.mkdir(parents=True, exist_ok=True)
            rhash = pr.ranges_hash(ranges)
            avoid: list[float] = []
            for index in range(4):
                if pr.ranges_hash(_load_playing_time(token)) != rhash:
                    return          # window changed under us — this chain is stale
                meta = None
                name = f"clip_{rhash}_0_{index}"
                meta_path = seed_dir / f"{name}.json"
                with _seed_clip_lock(f"{token}:{name}"):
                    if meta_path.exists():
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    else:
                        meta = build_one_seed_clip(
                            str(video_path), index, reroll=0, cfg=cfg,
                            my_kit_hex=my_kit_hex, avoid_sec=tuple(avoid),
                            play_ranges=ranges or None)
                        if meta is not None:
                            meta_path.write_text(json.dumps(meta), encoding="utf-8")
                if meta and meta.get("t_center") is not None:
                    avoid.append(float(meta["t_center"]))
        except Exception:  # noqa: BLE001 — prefetch must never break the app
            traceback.print_exc()

    threading.Thread(target=_work, daemon=True, name=f"seed-prefetch-{token[:6]}").start()


def _seed_token_video(token: str) -> tuple[Path | None, object | None]:
    import re
    if not re.fullmatch(r"[a-f0-9]{12}", token or ""):
        return None, (jsonify({"error": "invalid token"}), 400)
    video_path = UPLOADS / f"{token}.mp4"
    if not video_path.exists():
        return None, (jsonify({"error": "unknown or expired token"}), 400)
    return video_path, None


_SEED_WARM_STARTED = False
_SEED_WARM_LOCK = threading.Lock()


@app.route("/api/v2/warm", methods=["POST"])
def v2_warm():
    """Kick off soccer-model compile in the background so the first seed clip is
    fast. Fire-and-forget; returns immediately. Idempotent per process."""
    global _SEED_WARM_STARTED
    if not PIPELINE_V2_OK:
        return jsonify({"ok": False, "reason": "v2 unavailable"}), 200
    with _SEED_WARM_LOCK:
        if _SEED_WARM_STARTED:
            return jsonify({"ok": True, "already": True})
        _SEED_WARM_STARTED = True

    def _warm():
        try:
            warm_seed_detector(PipelineV2Config(
                ball_weights=WEIGHTS, player_weights=WEIGHTS, device=DEVICE))
        except Exception:  # noqa: BLE001
            traceback.print_exc()

    threading.Thread(target=_warm, daemon=True).start()
    return jsonify({"ok": True, "warming": True})


@app.route("/api/v2/seed_clips_index", methods=["POST"])
def v2_seed_clips_index():
    """Cheap: the 4 moment timestamps for a reroll set (no clip building)."""
    if not PIPELINE_V2_OK:
        return jsonify({"error": f"v2 pipeline unavailable: {PIPELINE_V2_ERR}"}), 503
    payload = request.get_json(silent=True) or {}
    video_path, err = _seed_token_video(payload.get("token", ""))
    if err:
        return err
    reroll = int(payload.get("reroll", 0) or 0)
    ranges = _effective_playing_time(payload.get("token", ""), payload.get("play_ranges"))
    return jsonify({"ok": True, "token": payload.get("token"),
                    **build_seed_clips_index(str(video_path), reroll, ranges or None)})


@app.route("/api/v2/seed_clip", methods=["POST"])
def v2_seed_clip():
    """Build (or return cached) tracked player nodes for one seed moment.

    The browser plays the uploaded video directly — no separate encoded clip.
    """
    if not PIPELINE_V2_OK:
        return jsonify({"error": f"v2 pipeline unavailable: {PIPELINE_V2_ERR}"}), 503
    payload = request.get_json(silent=True) or {}
    token = payload.get("token", "")
    video_path, err = _seed_token_video(token)
    if err:
        return err
    verr = _validate_video(video_path)
    if verr:
        return jsonify({"error": verr, "invalid_video": True}), 400

    index = int(payload.get("index", 0) or 0)
    reroll = int(payload.get("reroll", 0) or 0)
    my_kit_hex = payload.get("my_team_hex") or None
    # Moments to avoid (client-supplied): the OTHER clips' current moments (so
    # two clips don't share a passage of play) plus this slot's own shuffle
    # history (so shuffling gives a genuinely new clip, not a repeat).
    try:
        avoid_sec = tuple(float(x) for x in (payload.get("avoid") or []))[:24]
    except (TypeError, ValueError):
        avoid_sec = ()
    ranges = _effective_playing_time(token, payload.get("play_ranges"))
    # Safety net if the playing-time prefetch never ran (server restart, direct
    # deep-link) — idempotent per window, so normally a no-op.
    _start_seed_prefetch(token, str(video_path), my_kit_hex, ranges)

    seed_dir = EXPORTS / "seed" / token
    seed_dir.mkdir(parents=True, exist_ok=True)
    # The window is part of the cache key: without it a whole-match prefetch and
    # a later windowed clip collide on the same filename, and the user gets
    # served a clip from a moment they weren't playing.
    name = f"clip_{pr.ranges_hash(ranges)}_{reroll}_{index}"
    meta_path = seed_dir / f"{name}.json"

    with _seed_clip_lock(f"{token}:{name}"):
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        else:
            cfg = PipelineV2Config(ball_weights=WEIGHTS, player_weights=WEIGHTS, device=DEVICE)
            try:
                meta = build_one_seed_clip(
                    str(video_path), index, reroll=reroll, cfg=cfg,
                    my_kit_hex=my_kit_hex, avoid_sec=avoid_sec,
                    play_ranges=ranges or None)
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500
            if meta is None:
                return jsonify({"error": "could not read that moment of the video"}), 422
            meta_path.write_text(json.dumps(meta), encoding="utf-8")

    return jsonify({
        "ok": True, "token": token,
        "video_url": f"/api/video/{token}",
        "clip_url": f"/api/video/{token}",
        **meta,
    })


@app.route("/api/v2/seed_clip_file/<token>/<name>", methods=["GET"])
def v2_seed_clip_file(token: str, name: str):
    import re
    if not re.fullmatch(r"[a-f0-9]{12}", token or "") or \
       not re.fullmatch(r"clip_(?:all|[a-f0-9]{8})_\d+_\d+\.mp4", name or ""):
        return jsonify({"error": "invalid path"}), 400
    seed_dir = EXPORTS / "seed" / token
    if not (seed_dir / name).is_file():
        return jsonify({"error": "clip not found"}), 404
    return send_from_directory(str(seed_dir), name, mimetype="video/mp4")


@app.route("/api/v2/review_track/<job_id>/<int:rank>", methods=["GET"])
def v2_review_track(job_id: str, rank: int):
    """Tracklet of the reviewed player across one montage clip, so the review
    ring follows them. Cached per (job, rank); built on demand."""
    if not PIPELINE_V2_OK:
        return jsonify({"error": f"v2 pipeline unavailable: {PIPELINE_V2_ERR}"}), 503
    j = _get_job(job_id)
    if not j or not j.get("montage"):
        return jsonify({"error": "unknown job or no montage"}), 404
    item = next((it for it in j["montage"] if int(it.get("rank", -1)) == rank), None)
    if item is None:
        return jsonify({"error": "unknown rank"}), 404

    tdir = EXPORTS / job_id / "tracks"
    tdir.mkdir(parents=True, exist_ok=True)
    cache = tdir / f"rank_{rank}.json"
    if cache.exists():
        return jsonify(json.loads(cache.read_text(encoding="utf-8")))

    token = j.get("token")
    video_path = UPLOADS / f"{token}.mp4"
    if not token or not video_path.exists():
        return jsonify({"error": "video unavailable"}), 400

    with _seed_clip_lock(f"track:{job_id}:{rank}"):
        if cache.exists():
            return jsonify(json.loads(cache.read_text(encoding="utf-8")))
        try:
            tr = build_review_track_for_item(str(video_path), item, PipelineV2Config())
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500
        # Graceful: no track → empty points, frontend keeps the fixed ring.
        out = {"ok": True, "rank": rank, "points": (tr or {}).get("points", []),
               "base_w": (tr or {}).get("base_w"), "base_h": (tr or {}).get("base_h")}
        cache.write_text(json.dumps(out), encoding="utf-8")
    return jsonify(out)


@app.route("/api/v2/enhance_clip/<job_id>/<int:rank>", methods=["GET"])
def v2_enhance_clip(job_id: str, rank: int):
    """Review-clip enhance was removed: it routinely blocked the UI for >10s
    (and sometimes >1 min) depending on environment / CPU / Defender. Zoom +
    the on-canvas SVG sharpen remain for a closer look."""
    return jsonify({
        "error": "enhance removed — use zoom for a closer look",
        "removed": True,
    }), 410


@app.route("/api/v2/enhance_clip_file/<job_id>/<name>", methods=["GET"])
def v2_enhance_clip_file(job_id: str, name: str):
    return jsonify({"error": "enhance removed"}), 410


@app.route("/api/v2/decisions/<job_id>", methods=["POST"])
def v2_decisions(job_id: str):
    if not PIPELINE_V2_OK:
        return jsonify({"error": f"v2 pipeline unavailable: {PIPELINE_V2_ERR}"}), 503
    j = _get_job(job_id)
    if not j or not j.get("montage"):
        return jsonify({"error": "unknown job or no montage"}), 404
    payload = request.get_json(silent=True) or {}
    decisions = payload.get("decisions", {})
    cfg = PipelineV2Config()
    hotspots, items = hotspots_from_decisions(
        j["montage"], decisions, cfg, duration_sec=j.get("duration_sec"),
        ball_track=j.get("ball_track"),
    )
    state = "done" if payload.get("finalize") else "review"
    _set_job(job_id, montage=items, hotspots=hotspots, state=state,
             stage=state, status="Done" if state == "done" else "Review your touches")
    out_dir = EXPORTS / job_id
    if out_dir.exists():
        (out_dir / "hotspots.json").write_text(
            json.dumps({"hotspots": hotspots}, indent=2), encoding="utf-8")
    return jsonify({"ok": True, "hotspots": hotspots, "n_hotspots": len(hotspots)})


@app.route("/api/catalogue", methods=["GET"])
def list_catalogue():
    """Completed analyses saved on this machine (newest first)."""
    _hydrate_jobs_from_disk()
    entries: list[dict] = []
    with JOBS_LOCK:
        job_ids = list(JOBS.keys())
    for job_id in job_ids:
        entry = _catalogue_entry(job_id)
        if entry:
            entries.append(entry)
    entries.sort(key=lambda e: float(e.get("analysed_at") or 0), reverse=True)
    return jsonify({"matches": entries})


@app.route("/api/catalogue/<job_id>/metadata", methods=["POST", "PATCH"])
def update_catalogue_metadata(job_id: str):
    j = _get_job(job_id)
    if not j:
        return jsonify({"error": "unknown job"}), 404
    meta = _match_metadata_from_json()
    if meta:
        _set_job(job_id, **meta)
    entry = _catalogue_entry(job_id)
    return jsonify({"ok": True, "match": entry})


@app.route("/api/catalogue/<job_id>", methods=["DELETE"])
def remove_catalogue_entry(job_id: str):
    """Remove a finished analysis from the catalogue (video file is kept)."""
    j = _get_job(job_id)
    if not j:
        return jsonify({"error": "unknown job"}), 404
    _set_job(job_id, state="archived", status="Removed from catalogue", stage="archived")
    sess_path = _session_path(job_id)
    if sess_path.is_file():
        try:
            sess_path.unlink()
        except Exception:
            pass
    return jsonify({"ok": True, "removed": True})


@app.route("/api/catalogue/<job_id>/session", methods=["GET"])
def get_match_session(job_id: str):
    j = _get_job(job_id)
    if not j or j.get("state") not in ("done", "archived"):
        if not j:
            return jsonify({"error": "unknown job"}), 404
    path = _session_path(job_id)
    if not path.is_file():
        return jsonify({"session": None})
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return jsonify({"session": data.get("session")})
    except Exception:
        return jsonify({"session": None})


@app.route("/api/catalogue/<job_id>/session", methods=["PUT"])
def save_match_session(job_id: str):
    j = _get_job(job_id)
    if not j:
        return jsonify({"error": "unknown job"}), 404
    body = request.get_json(silent=True) or {}
    session = body.get("session")
    if session is None:
        return jsonify({"error": "session required"}), 400
    out_dir = EXPORTS / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    _session_path(job_id).write_text(
        json.dumps({"session": session, "updated_at": time.time()}, indent=2),
        encoding="utf-8",
    )
    n_actions = len(session.get("matchStats") or [])
    benches = session.get("benchBlocks") or []
    hybrid = session.get("hybridResults")
    has_session = n_actions > 0 or bool(benches) or hybrid is not None
    _set_job(job_id, has_session=has_session, n_actions=n_actions)
    return jsonify({"ok": True, "n_actions": n_actions})


if __name__ == "__main__":
    port = int(os.environ.get("POLYFUT_PORT", "5000"))
    print("=" * 55)
    print(f"  PolyFut (Level 1)  ->  http://127.0.0.1:{port}")
    print(f"  Data dir: {DATA_ROOT}")
    print("=" * 55)
    if FAKE_CV:
        print("  *** POLYFUT_FAKE_CV is ON — demo clips only")
    elif not PIPELINE_OK:
        print(f"  *** PIPELINE DISABLED: {PIPELINE_IMPORT_ERR}")
        print("  *** Fix: pip install -r requirements.txt")
    else:
        print(f"  Module: polyfut_video")
        print(f"  Weights: {WEIGHTS}")
        print(f"  Device: {DEVICE}")
        # Off the request path on purpose — see _warm_kit_detector's docstring
        # for the incident this fixes (first /api/teams call after boot paying
        # the full cold-import cost inside its own timeout window).
        threading.Thread(target=_warm_kit_detector, daemon=True, name="kit-model-warmup").start()
    print()
    use_waitress = os.environ.get("POLYFUT_WSGI", "waitress").lower() not in (
        "flask", "werkzeug", "dev",
    )
    # region agent log
    _dbg_log("M1", "server.py:boot", "server boot", {
        "wsgi": "waitress" if use_waitress else "flask-dev",
        "max_content_gb": round(app.config.get("MAX_CONTENT_LENGTH", 0) / 1e9, 1),
        "upload_tmp": str(UPLOADS_TMP),
    }, run_id="memory-fix-v1")
    # endregion
    if use_waitress:
        try:
            from waitress import serve
            print("  Server: waitress (streaming uploads — large videos OK)")
            serve(
                app,
                host="127.0.0.1",
                port=port,
                threads=4,
                channel_timeout=7200,
            )
        except ImportError:
            print("  ERROR: waitress required for large video uploads.")
            print("  Run: pip install waitress")
            sys.exit(1)
    else:
        print("  Server: Flask dev (POLYFUT_WSGI=flask)")
        app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
