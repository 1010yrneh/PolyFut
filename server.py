"""PolyFut local server — Level 1 polyfut_video pipeline."""

from __future__ import annotations

import gc
import json
import os
import subprocess
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
    from polyfut_video.config import PipelineConfig
    from polyfut_video.main import run_pipeline
    from polyfut_video.pipeline.decode import probe_video
    PIPELINE_OK = True
    PIPELINE_IMPORT_ERR = ""
except Exception as exc:
    PIPELINE_OK = False
    PIPELINE_IMPORT_ERR = f"{type(exc).__name__}: {exc}"
    run_pipeline = None  # type: ignore
    probe_video = None  # type: ignore

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


def _detect_team_kits_isolated(video_path: str) -> tuple[list[dict] | None, str | None]:
    """Run kit preview in a child process so YOLO RAM is released after detection."""
    script = (
        "import json, sys\n"
        "from polyfut_video.pipeline.team_preview import detect_team_kits\n"
        "r = detect_team_kits(sys.argv[1], weights=sys.argv[2], device=sys.argv[3])\n"
        "print(json.dumps(r))\n"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script, video_path, str(WEIGHTS), str(DEVICE)],
            capture_output=True,
            text=True,
            timeout=900,
            cwd=str(ROOT),
        )
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "kit subprocess failed").strip()[:500]
            return None, err
        raw = (proc.stdout or "").strip()
        if not raw:
            return None, "empty kit subprocess output"
        return json.loads(raw), None
    except subprocess.TimeoutExpired:
        return None, "kit detection timed out"
    except Exception as exc:
        return None, str(exc)


def _run_job(job_id: str, video_path: Path, my_team: str, out_dir: Path) -> None:
    seg_path = out_dir / "clip_segments.json"
    timeline_path = out_dir / "possession_timeline.json"

    def progress(frac: float, msg: str) -> None:
        partial = None
        if seg_path.exists():
            try:
                partial = json.loads(seg_path.read_text(encoding="utf-8")).get("segments")
            except Exception:
                pass
        elapsed = time.time() - JOB_START.get(job_id, time.time())
        cur, tot, unit = _parse_progress_counts(msg)
        _set_job(
            job_id,
            progress=frac,
            status=msg,
            stage=_parse_stage(msg),
            stage_progress=frac,
            elapsed_sec=round(elapsed, 1),
            segments_partial=partial,
            progress_current=cur,
            progress_total=tot,
            progress_unit=unit,
            status_updated_at=time.time(),
        )
        if cur is not None and tot is not None:
            # region agent log
            _dbg_log("B3", "server.py:progress", "progress counts", {
                "job_id": job_id,
                "current": cur,
                "total": tot,
                "unit": unit,
                "frac": round(frac, 3),
                "status": msg[:120],
            }, run_id="general-audit-v1")
            # endregion

    def should_cancel() -> bool:
        j = _get_job(job_id)
        return bool(j and j.get("cancel"))

    with during_analysis():
        try:
            if FAKE_CV or not PIPELINE_OK:
                dur = _probe_duration(video_path)
                for i in range(30):
                    if should_cancel():
                        raise RuntimeError("cancelled")
                    progress(0.05 + 0.9 * i / 30, f"Analyzing (demo)... {i+1}/30")
                    time.sleep(0.1)
                segments = _fake_segments(dur)
                out_dir.mkdir(parents=True, exist_ok=True)
                seg_path.write_text(
                    json.dumps({"version": 2, "segments": segments, "my_team": my_team}, indent=2),
                    encoding="utf-8",
                )
                note = "demo" if FAKE_CV else f"CV unavailable ({PIPELINE_IMPORT_ERR})"
                _set_job(
                    job_id, progress=1.0, status="Done", state="done", stage="done",
                    segments=segments, note=note, finished_at=time.time(),
                )
                return

            cfg = PipelineConfig(
                yolo_weights=WEIGHTS,
                device=DEVICE,
                output_dir=out_dir,
            )
            progress(0.02, f"Level 1 pipeline: weights={WEIGHTS}, device={DEVICE}")
            meta = run_pipeline(
                video_path,
                out_dir,
                cfg=cfg,
                my_team=my_team,
                progress=progress,
                should_cancel=should_cancel,
            )
            seg_data = json.loads(Path(meta["clip_segments_path"]).read_text(encoding="utf-8"))
            timeline = None
            if timeline_path.exists():
                timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
            _set_job(
                job_id,
                progress=1.0,
                status="Done",
                state="done",
                stage="done",
                segments=seg_data.get("segments", []),
                possession_timeline=timeline,
                timings=meta.get("timings_sec"),
                finished_at=time.time(),
            )
            # region agent log
            _dbg_log("B5", "server.py:_run_job", "job done", {
                "job_id": job_id,
                "n_segments": len(seg_data.get("segments", [])),
                "live_shots": meta.get("live_shots"),
            }, run_id="general-audit-v1")
            # endregion
        except MemoryError:
            traceback.print_exc()
            _dbg_log("B5", "server.py:_run_job", "job MemoryError", {
                "job_id": job_id,
            }, run_id="general-audit-v1")
            _set_job(
                job_id,
                state="error",
                status="Out of memory during analysis",
                stage="error",
                error=(
                    "MemoryError in pipeline (usually stages 1–2 on long videos). "
                    "Stop the server, run python server.py again, hard-refresh the browser, "
                    "and re-run. Stage 1–2 progress should mention v1.1.0."
                ),
            )
        except Exception as exc:
            if str(exc) == "cancelled":
                _set_job(job_id, state="cancelled", status="Cancelled", stage="cancelled")
            else:
                traceback.print_exc()
                # region agent log
                _dbg_log("B5", "server.py:_run_job", "job error", {
                    "job_id": job_id,
                    "error": str(exc),
                }, run_id="general-audit-v1")
                # endregion
                _set_job(job_id, state="error", status="Error", stage="error", error=str(exc))


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

    resp = {
        "token": token,
        "demo": False,
        "mode": "live",
        "pipeline_ok": True,
        "teams": teams_out,
        "kits_detected": ran_detection,
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


@app.route("/api/process", methods=["POST"])
def process():
    token = request.form.get("token")
    if token:
        video_path = UPLOADS / f"{token}.mp4"
        if not video_path.exists():
            return jsonify({"error": "unknown or expired token"}), 400
    elif "video" in request.files:
        token = uuid.uuid4().hex[:12]
        video_path = UPLOADS / f"{token}.mp4"
        request.files["video"].save(str(video_path))
        TOKEN_META[token] = {"video": str(video_path)}
    else:
        return jsonify({"error": "No video or token in request."}), 400

    my_team = request.form.get("my_team", "team_a")
    if my_team not in ("team_a", "team_b"):
        my_team = "team_a"

    existing = _find_running_job_for_token(token)
    if existing:
        return jsonify({"job_id": existing, "resumed": True, "token": token})

    job_id = uuid.uuid4().hex[:12]
    job_dir = EXPORTS / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    JOB_START[job_id] = started
    _set_job(
        job_id,
        progress=0.0,
        status="Queued",
        state="running",
        stage="init",
        cancel=False,
        segments=None,
        segments_partial=None,
        error=None,
        token=token,
        my_team=my_team,
        started_at=started,
        video_path=str(video_path),
        **_match_metadata_from_form(),
    )
    t = threading.Thread(
        target=_run_job,
        args=(job_id, video_path, my_team, job_dir),
        daemon=False,
    )
    t.start()
    return jsonify({"job_id": job_id, "token": token, "resumed": False})


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
    })


@app.route("/api/process/<job_id>", methods=["DELETE"])
def cancel(job_id: str):
    j = _get_job(job_id)
    if not j:
        return jsonify({"error": "unknown job"}), 404
    _set_job(job_id, cancel=True, state="cancelled", status="Cancelled", stage="cancelled")
    return jsonify({"ok": True, "discarded": True})


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
