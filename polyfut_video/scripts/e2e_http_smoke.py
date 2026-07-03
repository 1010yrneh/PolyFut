"""HTTP E2E smoke test for PolyFut server + API contract."""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = os.environ.get("POLYFUT_E2E_BASE", "http://127.0.0.1:5001")
VIDEO = ROOT / "uploads" / "e83a5c379ce7.mp4"
if not VIDEO.is_file():
    VIDEO = ROOT / "uploads" / "_test_dummy.mp4"


def _get(path: str) -> dict:
    with urllib.request.urlopen(f"{BASE}{path}", timeout=30) as r:
        return json.loads(r.read().decode())


def _multipart_process(token: str, my_team: str = "team_a") -> dict:
    import uuid
    boundary = f"----PolyFut{uuid.uuid4().hex}"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="token"\r\n\r\n'
        f"{token}\r\n"
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="my_team"\r\n\r\n'
        f"{my_team}\r\n"
        f"--{boundary}--\r\n"
    ).encode()
    req = urllib.request.Request(
        f"{BASE}/api/process",
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def _upload_teams(video: Path) -> dict:
    import uuid
    boundary = f"----PolyFut{uuid.uuid4().hex}"
    data = video.read_bytes()
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="video"; filename="{video.name}"\r\n'
        f"Content-Type: video/mp4\r\n\r\n"
    ).encode() + data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        f"{BASE}/api/teams",
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())


def run() -> int:
    issues: list[str] = []
    print("=== PolyFut HTTP E2E ===")

    # 1. Index + health
    try:
        with urllib.request.urlopen(f"{BASE}/", timeout=10) as r:
            html = r.read().decode(errors="replace")
        if "FIND TOUCH HOTSPOTS" not in html and "FIND KEY MOMENTS" not in html:
            issues.append("index.html missing start button text")
        if "TOUCH HOTSPOTS" not in html and "KEY MOMENTS" not in html:
            issues.append("index.html missing hotspot panel label")
        if "seek-zones" not in html:
            issues.append("index.html missing seek-zones element")
        print(f"GET / -> {len(html)} bytes OK")
    except Exception as e:
        issues.append(f"GET / failed: {e}")
        print("FAIL: server not reachable at", BASE)
        for i in issues:
            print(" ", i)
        return 1

    try:
        health = _get("/api/health")
        print(f"Health: pipeline_ready={health.get('pipeline_ready')} fake_cv={health.get('fake_cv')}")
        if not health.get("pipeline_ready"):
            issues.append(f"pipeline not ready: {health.get('pipeline_error')}")
    except Exception as e:
        issues.append(f"/api/health failed: {e}")

    if not VIDEO.is_file():
        issues.append(f"no test video at {VIDEO}")
        print("SKIP teams/process (no video)")
    else:
        # 2. Teams upload
        try:
            teams = _upload_teams(VIDEO)
            token = teams.get("token")
            print(f"Teams: token={token} teams={len(teams.get('teams') or [])} demo={teams.get('demo')}")
            if not token:
                issues.append("teams response missing token")
            if not teams.get("teams"):
                issues.append("teams response missing teams array")
        except Exception as e:
            issues.append(f"/api/teams failed: {e}")
            token = None

        # 3. Process job (fake CV finishes fast)
        if token:
            try:
                job = _multipart_process(token)
                job_id = job.get("job_id")
                print(f"Process: job_id={job_id}")
                if not job_id:
                    issues.append("process response missing job_id")
                else:
                    deadline = time.time() + 120
                    state = "running"
                    segments = []
                    while time.time() < deadline:
                        st = _get(f"/api/process/status/{job_id}")
                        state = st.get("state")
                        prog = st.get("progress", 0)
                        status = st.get("status", "")
                        stage = st.get("stage", "")
                        if int(prog * 10) % 3 == 0:
                            print(f"  poll: {prog:.0%} stage={stage} {status[:60]}")
                        if state in ("done", "error", "cancelled"):
                            segments = st.get("segments") or []
                            break
                        time.sleep(0.8)
                    print(f"Job finished: state={state} segments={len(segments)}")
                    if state != "done":
                        issues.append(f"job ended in state={state} err={st.get('error')}")
                    elif not segments:
                        issues.append("job done but no segments")
                    else:
                        seg = segments[0]
                        for key in ("start", "end"):
                            if key not in seg:
                                issues.append(f"segment missing {key}")
                        if seg.get("type") and seg.get("type") != "hotspot":
                            issues.append(f"unexpected segment type {seg.get('type')}")
            except Exception as e:
                issues.append(f"/api/process failed: {e}")

    ok = not issues
    print(f"Result: {'PASS' if ok else 'FAIL'}")
    for i in issues:
        print(f"  - {i}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(run())
